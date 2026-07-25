//! Write pipeline: upsert, batch delete+append, delete, compaction.

use std::sync::Arc;

use arrow_array::RecordBatchIterator;

use crate::kernel::error::StoreError;

use super::{BranchKey, ChunkRow, LanceStore};
use super::index::io_compact_data;

impl LanceStore {
    /// Upsert chunk rows for a document on `branch` (layout A). First deletes all
    /// existing rows for the doc_id, then appends the new rows — the
    /// delete-then-append that implements real `Changed` (replace full chunk set,
    /// no stale chunks; RISK-13).
    ///
    /// Writes target `branch`'s head via a branch-bound handle so sibling
    /// branches are untouched. Ensures the domain dataset exists first.
    pub async fn io_upsert_chunks(
        &self,
        domain: &str,
        branch: &str,
        doc_id: &str,
        rows: &[ChunkRow],
    ) -> Result<u64, StoreError> {
        if rows.is_empty() {
            return Ok(0);
        }

        // Serialise dataset creation against DELETE /domain (BLOCKER-2 / #6):
        // hold the per-domain guard across the ensure-exists so a concurrent
        // delete can't observe a half-created dataset.
        let _domain_guard = self.acquire_domain_guard(domain).await;

        // Ensure the domain dataset exists (creates the main branch on first use).
        self.io_open_dataset(domain, branch).await?;

        // Open a fresh branch-bound handle so the write targets `branch`'s head.
        let mut ds = self.io_open_branch_for_write(domain, branch).await?;

        // Delete existing rows for this doc_id (replace semantics).
        let filter = format!("doc_id = '{}'", doc_id.replace('\'', "''"));
        ds.delete(&filter)
            .await
            .map_err(|e| StoreError::Internal(format!("delete failed: {}", e)))?;

        // Append new rows.
        let batch = self.rows_to_batch(rows)?;
        let schema = self.chunk_schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        ds.append(reader, None)
            .await
            .map_err(|e| StoreError::Internal(format!("append failed: {}", e)))?;

        // Keep the cache consistent with the branch we just advanced (main).
        // For non-main branches the cached main handle is unaffected; refreshing
        // is harmless (re-opens at default branch head).
        self.io_refresh_cached_dataset(domain, branch).await?;

        Ok(ds.version().version)
    }

    /// Batched write for one commit's operations (Phase 6A Step 1).
    ///
    /// Performs at most TWO Lance version advances in a fixed, crash-safe order:
    ///   1. (Optional) Delete: `ds.delete(doc_id IN (...))` for all docs that are
    ///      being replaced (Changed) or removed (Deleted). Creates deletion vectors
    ///      only — no new data fragment. Handles shrinking chunk counts correctly
    ///      (all old chunks for a doc_id are removed regardless of count change).
    ///   2. (Optional) Append: `ds.append(batch)` with ALL new rows for Insert +
    ///      Changed docs. Exactly ONE new data fragment for all rows.
    ///
    /// Crash safety (two-version delete-then-append, Option B):
    ///   This is NOT an atomic single-version merge-insert. It performs two separate
    ///   Lance version advances (delete, then append). Crash-safety comes from the
    ///   untagged-commit→invisible→re-pushable property:
    ///   - Crash before (1): no change. Commit remains untagged → invisible → re-pushable.
    ///   - Crash after (1) but before (2): Changed/Deleted docs' old chunks removed
    ///     (correct for Deleted; for Changed, commit untagged → invisible → re-pushable,
    ///     the re-push will re-insert the replacement rows).
    ///   - Crash after (2): both writes committed. Tag not yet written → re-push is
    ///     idempotent (delete of already-deleted rows is no-op; append of same rows
    ///     would duplicate, but since commit is untagged it was never served →
    ///     re-push via the pipeline lock clears and retries correctly).
    ///
    /// Caller MUST hold the pipeline lock. Dataset existence is ensured internally.
    ///
    /// Returns the final Lance version after all writes.
    pub async fn io_batch_delete_append(
        &self,
        domain: &str,
        branch: &str,
        delete_ids: &[String],
        rows: &[ChunkRow],
    ) -> Result<u64, StoreError> {
        // Serialise dataset creation against DELETE /domain (BLOCKER-2 / #6):
        let _domain_guard = self.acquire_domain_guard(domain).await;

        // Ensure the domain dataset exists (creates the main branch on first use).
        self.io_open_dataset(domain, branch).await?;

        // Use a SINGLE mutable handle for both delete and append. This ensures
        // the append version is built ON TOP of the post-delete version (the append
        // inherits the deletion vectors). Using separate handles would risk the
        // second Dataset::open returning a stale version if the filesystem hasn't
        // flushed the latest_version_hint before the second open reads it.
        let mut ds = self.io_open_branch_for_write(domain, branch).await?;

        // --- Phase 1: delete old chunks for Changed + Deleted doc_ids ---
        // Creates deletion vectors only (no new data fragment). Handles shrinking
        // chunk counts: ALL rows for a doc_id are removed regardless of old/new count.
        if !delete_ids.is_empty() {
            let filter = build_doc_id_in_filter(delete_ids);
            ds.delete(&filter)
                .await
                .map_err(|e| StoreError::Internal(format!("batch delete failed: {}", e)))?;
        }

        // --- Phase 2: append all new rows (one fragment for all Insert + Changed) ---
        // Since Phase 1 already deleted old rows for Changed docs, this is a pure
        // append — no merge semantics needed. Produces exactly ONE new data fragment.
        // The append builds on the post-delete version (same handle), so the final
        // manifest correctly includes the deletion vectors.
        if !rows.is_empty() {
            let batch = self.rows_to_batch(rows)?;
            let schema = self.chunk_schema();
            let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
            ds.append(reader, None)
                .await
                .map_err(|e| StoreError::Internal(format!("batch append failed: {}", e)))?;
        }

        // Capture the version from the write handle BEFORE refreshing the cache.
        // This is the authoritative post-write version (delete + append).
        let version = ds.version().version;

        // Refresh the cached handle so subsequent reads see the new data.
        self.io_refresh_cached_dataset(domain, branch).await?;

        Ok(version)
    }

    /// Probabilistic background compaction trigger (BUG-FD24).
    ///
    /// Called after a fragment-creating write (io_upsert_chunks). Rolls a 5%
    /// chance; on a hit, spawns a BACKGROUND compaction task if one is not
    /// already in-flight for this (domain, branch). The spawned task:
    ///   1. Opens the dataset uncached (no write-lock contention with reads)
    ///   2. Calls io_compact_data (no-op if fragments <= threshold)
    ///   3. Retags all stale commits to the compacted version
    ///   4. Refreshes the cached handle
    ///   5. Clears the in-progress guard
    ///
    /// Fail-loud: compaction errors are logged to stderr (visible via container
    /// logs). The guard is always cleared so subsequent rolls can retry.
    ///
    /// Does NOT block the caller — returns immediately after the roll.
    pub fn maybe_trigger_background_compaction(
        store: Arc<Self>,
        domain: String,
        branch: String,
    ) {
        use rand::Rng;

        /// Probability of triggering compaction on each fragment write (5%).
        const COMPACTION_PROBABILITY: f64 = 0.05;

        // Roll the dice (cheap thread-local RNG, no allocation).
        let roll: f64 = rand::rng().random();
        if roll >= COMPACTION_PROBABILITY {
            return;
        }

        // Check (and set) the in-progress guard. If a compaction is already
        // running for this (domain, branch), skip (no-op).
        //
        // WHY: blocking_write() panics when called from within a tokio runtime
        // (this fn is invoked from a tokio::spawn context).
        // INVARIANT: compaction is probabilistic (5% roll); a skipped roll is
        // retried on the next successful write. Steady-state compaction rate is
        // negligibly affected by rare contention skips.
        // CONSEQUENCE: if contention persists (pathological), fragments accumulate
        // until contention clears and a roll succeeds. This is bounded by the
        // 5% trigger rate and is self-healing.
        let key: BranchKey = (domain.clone(), branch.clone());
        {
            let mut guard = match store.compaction_in_progress.try_write() {
                Ok(g) => g,
                Err(_) => return,
            };
            if guard.contains(&key) {
                return; // Already in-flight — no-op.
            }
            guard.insert(key.clone());
        }

        // Spawn the background compaction task.
        let store_bg = Arc::clone(&store);
        tokio::spawn(async move {
            let result = io_background_compact(&store_bg, &domain, &branch).await;
            // Always clear the guard (success or failure).
            {
                let mut guard = store_bg.compaction_in_progress.write().await;
                guard.remove(&key);
            }
            if let Err(e) = result {
                // FAIL LOUD: log to stderr so container logs surface the error.
                eprintln!(
                    "[compaction] ERROR: background compaction for {}/{} failed: {}",
                    domain, branch, e
                );
            }
        });
    }

    /// Delete all chunks for a doc_id on `branch` (`Deleted` op; RISK-13).
    /// Writes target `branch`'s head via a branch-bound handle.
    pub async fn io_delete_doc(
        &self,
        domain: &str,
        branch: &str,
        doc_id: &str,
    ) -> Result<u64, StoreError> {
        // A delete against a domain that does not exist is a no-op — it must NOT
        // create the dataset (BLOCKER-2 resurrection guard). Only a genuine
        // insert/change creates a domain.
        if self.io_open_dataset_readonly(domain).await?.is_none() {
            return Ok(0);
        }

        let mut ds = self.io_open_branch_for_write(domain, branch).await?;

        let filter = format!("doc_id = '{}'", doc_id.replace('\'', "''"));
        ds.delete(&filter)
            .await
            .map_err(|e| StoreError::Internal(format!("delete failed: {}", e)))?;

        self.io_refresh_cached_dataset(domain, branch).await?;

        Ok(ds.version().version)
    }
}

/// Background compaction: open dataset uncached, compact, retag stale commits,
/// refresh cache. Called by the 5%-probability trigger from the write path.
/// Runs on an uncached handle so it does NOT block concurrent reads.
async fn io_background_compact(
    store: &LanceStore,
    domain: &str,
    branch: &str,
) -> Result<(), String> {
    let ds = store
        .io_open_dataset_uncached(domain, branch)
        .await
        .map_err(|e| format!("uncached open for compaction failed: {}", e))?;

    let mut ds = match ds {
        Some(d) => d,
        None => return Ok(()), // Dataset gone (deleted concurrently) — no-op.
    };

    // Compact (threshold-gated: no-op if fragments <= 16).
    io_compact_data(&mut ds)
        .await
        .map_err(|e| format!("compaction failed: {}", e))?;

    let compacted_version = ds.version().version;

    // Retag stale commits to the compacted version so tag-resolved snapshots
    // see the fewer-fragment layout.
    let all_tags = ds
        .tags()
        .list()
        .await
        .map_err(|e| format!("tag list for retag failed: {}", e))?;

    for (tag_name, tag_contents) in &all_tags {
        if tag_contents.version != compacted_version {
            ds.tags()
                .update(tag_name, compacted_version)
                .await
                .map_err(|e| format!("retag '{}' to v{} failed: {}", tag_name, compacted_version, e))?;
        }
    }

    // Refresh the cached handle so subsequent reads use compacted fragments.
    store
        .io_refresh_cached_dataset(domain, branch)
        .await
        .map_err(|e| format!("cache refresh after compaction failed: {}", e))?;

    Ok(())
}

/// Build a Lance SQL filter expression `doc_id IN ('id1', 'id2', ...)` for batch
/// deletion of documents by their doc_id. SQL-escapes single quotes in doc_ids.
///
/// Precondition: `ids` is non-empty (caller must check). A filter on an empty set
/// is a logic error (Lance will reject the empty IN-list).
fn build_doc_id_in_filter(ids: &[String]) -> String {
    let values: Vec<String> = ids
        .iter()
        .map(|id| format!("'{}'", id.replace('\'', "''")))
        .collect();
    format!("doc_id IN ({})", values.join(", "))
}
