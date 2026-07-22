// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 DFRNT AB

//! Write pipeline: upsert, batch delete+append, delete, compaction.

use std::sync::Arc;

use arrow_array::RecordBatchIterator;
use lance::dataset::Dataset;
use lance::dataset::write::DeleteBuilder;
use lance::deps::datafusion::logical_expr::{col, lit, in_list};

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
        let expr = col("doc_id").eq(lit(doc_id));
        let result = DeleteBuilder::from_expr(Arc::new(ds.clone()), expr)
            .execute()
            .await
            .map_err(|e| StoreError::Internal(format!("delete failed: {}", e)))?;
        ds = result.new_dataset.as_ref().clone();

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
            let values: Vec<_> = delete_ids.iter().map(|id| lit(id.as_str())).collect();
            let expr = in_list(col("doc_id"), values, false);
            let result = DeleteBuilder::from_expr(Arc::new(ds.clone()), expr)
                .execute()
                .await
                .map_err(|e| StoreError::Internal(format!("batch delete failed: {}", e)))?;
            ds = result.new_dataset.as_ref().clone();
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

    /// Append a micro-batch of rows to an already-open write handle.
    /// Used by the streaming pipeline to write incrementally without
    /// accumulating all rows in memory. Each call produces one Lance fragment.
    /// The caller MUST hold the pipeline lock and the domain guard.
    pub async fn io_microbatch_append(
        &self,
        ds: &mut Dataset,
        rows: &[ChunkRow],
    ) -> Result<(), StoreError> {
        if rows.is_empty() {
            return Ok(());
        }
        let batch = self.rows_to_batch(rows)?;
        let schema = self.chunk_schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        ds.append(reader, None)
            .await
            .map_err(|e| StoreError::Internal(format!("microbatch append failed: {}", e)))?;
        Ok(())
    }

    /// Delete old chunks for a set of doc_ids on an already-open write handle.
    /// Used by the streaming pipeline to do the delete phase once at the start
    /// before micro-batch appends begin.
    pub async fn io_microbatch_delete(
        &self,
        ds: &mut Dataset,
        delete_ids: &[String],
    ) -> Result<(), StoreError> {
        if delete_ids.is_empty() {
            return Ok(());
        }
        let values: Vec<_> = delete_ids.iter().map(|id| lit(id.as_str())).collect();
        let expr = in_list(col("doc_id"), values, false);
        let result = DeleteBuilder::from_expr(Arc::new(ds.clone()), expr)
            .execute()
            .await
            .map_err(|e| StoreError::Internal(format!("microbatch delete failed: {}", e)))?;
        *ds = result.new_dataset.as_ref().clone();
        Ok(())
    }

    /// Probabilistic background compaction trigger (BUG-FD24).
    ///
    /// Called after a fragment-creating write (io_upsert_chunks). Rolls a 5%
    /// chance; on a hit, spawns a BACKGROUND compaction task if one is not
    /// already in-flight for this (domain, branch). The spawned task:
    ///   1. Opens the dataset uncached (no write-lock contention with reads)
    ///   2. Calls io_compact_data (no-op if fragments <= threshold)
    ///   3. Refreshes the cached handle
    ///   4. Clears the in-progress guard
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

        // Early-exit when auto-compaction is disabled (deterministic test mode).
        if !store.auto_compaction_enabled.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }

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

        let expr = col("doc_id").eq(lit(doc_id));
        let result = DeleteBuilder::from_expr(Arc::new(ds.clone()), expr)
            .execute()
            .await
            .map_err(|e| StoreError::Internal(format!("delete failed: {}", e)))?;
        ds = result.new_dataset.as_ref().clone();

        self.io_refresh_cached_dataset(domain, branch).await?;

        Ok(ds.version().version)
    }
}

/// Background compaction: open dataset uncached, compact, refresh cache.
/// Called by the 5%-probability trigger from the write path.
/// Runs on an uncached handle so it does NOT block concurrent reads.
///
/// Delta-fork retags intermediate commits to new indexed versions on a temp
/// branch (transactional within the compaction flow). This unpins old versions
/// so aggressive cleanup can delete their orphaned index files while preserving
/// snapshot isolation at every commit.
///
/// Background auto-compaction: compacts data fragments, delta-fork retags
/// intermediate commits, and runs aggressive cleanup to delete old index files.
/// Triggered probabilistically after each push.
pub(crate) async fn io_background_compact(
    store: &LanceStore,
    domain: &str,
    branch: &str,
) -> Result<(), String> {
    use lance::index::DatasetIndexExt;
    use crate::store::vector_index::{io_ensure_vector_index, VECTOR_INDEX_NAME};
    use crate::store::lance::io_ensure_fts_index_on_dataset;
    use crate::store::lance::io_incremental_cascade;

    // Acquire the per-branch pipeline lock. This serializes compaction
    // with pipeline pushes. Compaction runs after a push completes, so
    // the lock should be free. If a new push arrives during compaction,
    // it will be rejected with 503 Service Unavailable by push_stream
    // (which uses try_acquire_pipeline_lock).
    let _pipeline_guard = store.acquire_pipeline_lock(domain, branch).await;

    let ds = store
        .io_open_dataset_uncached(domain, branch)
        .await
        .map_err(|e| format!("uncached open for compaction failed: {}", e))?;

    let mut ds = match ds {
        Some(d) => d,
        None => return Ok(()), // Dataset gone (deleted concurrently) — no-op.
    };

    let fragments_before = ds.get_fragments().len();
    let version_before = ds.version().version;

    // Step 1: Drop existing vector + FTS indices BEFORE compacting data.
    // LanceDB's DefaultCompactionPlanner groups fragments by index set. If
    // fragments have different index UUIDs (e.g. after incremental pushes
    // that each created a delta index), they go into separate bins and
    // cannot be merged. Dropping indices first ensures all fragments share
    // the same (empty) index set, allowing full consolidation.
    let indices = ds
        .load_indices()
        .await
        .map_err(|e| format!("load indices for compaction failed: {}", e))?;
    let has_vector_idx = indices.iter().any(|idx| idx.name == VECTOR_INDEX_NAME);
    let has_fts_idx = indices.iter().any(|idx| idx.name == "content_fts");

    if has_vector_idx {
        ds.drop_index(VECTOR_INDEX_NAME)
            .await
            .map_err(|e| format!("drop vector index failed: {}", e))?;
        eprintln!("[compaction] domain={} branch={} dropped vector index before compaction", domain, branch);
    }
    if has_fts_idx {
        ds.drop_index("content_fts")
            .await
            .map_err(|e| format!("drop FTS index failed: {}", e))?;
        eprintln!("[compaction] domain={} branch={} dropped FTS index before compaction", domain, branch);
    }
    if has_vector_idx || has_fts_idx {
        if let Err(e) = store.io_cleanup_aggressive(domain, branch).await {
            eprintln!("[compaction] cleanup after index drop failed (soft): {}", e);
        }
    }

    // Step 2: Compact data fragments (threshold-gated: no-op if fragments <= 16).
    // Now all fragments share the same (empty) index set, so the compaction
    // planner can merge them into fewer large fragments.
    io_compact_data(&mut ds, false)
        .await
        .map_err(|e| format!("compaction failed: {}", e))?;

    let fragments_after = ds.get_fragments().len();
    let version_after = ds.version().version;

    eprintln!(
        "[compaction] domain={} branch={} version {}→{} fragments {}→{}",
        domain, branch, version_before, version_after, fragments_before, fragments_after
    );

    // Step 3: Aggressive cleanup of old untagged versions.
    if let Err(e) = store.io_cleanup_aggressive(domain, branch).await {
        eprintln!("[compaction] aggressive cleanup failed (soft): {}", e);
    }

    // Step 4: Recreate fresh vector + FTS indices covering the compacted
    // fragments. Without this, the old indices are stale (reference deleted
    // fragment IDs) and search degrades to flat-scan over all rows.
    // force=true skips the 256-row IVF training minimum since data already exists.
    let vector_config = store.vector_index_config().clone();
    io_ensure_vector_index(&mut ds, &vector_config, true)
        .await
        .map_err(|e| format!("vector index recreation failed: {}", e))?;
    io_ensure_fts_index_on_dataset(&mut ds)
        .await
        .map_err(|e| format!("FTS index recreation failed: {}", e))?;
    eprintln!(
        "[compaction] domain={} branch={} recreated vector + FTS indices after compaction",
        domain, branch
    );

    // Reset delta count — compaction dropped all deltas and recreated fresh indices.
    store.io_reset_delta_count(domain, branch).await;

    if let Err(e) = store.io_cleanup_aggressive(domain, branch).await {
        eprintln!("[compaction] cleanup after index recreate failed (soft): {}", e);
    }

    // Step 5: Run incremental cascade to consolidate any remaining index deltas.
    let delta_count = store.io_get_delta_count(domain, branch).await;
    if let Err(e) = io_incremental_cascade(&mut ds, delta_count).await {
        eprintln!("[compaction] incremental cascade failed (soft): {}", e);
    }
    let merge_version = ds.version().version;

    // Step 6: Delta-fork retag intermediate commits to new indexed versions
    // on a temp branch. This unpins old versions so their index files can be
    // deleted while preserving snapshot isolation at every commit.
    if merge_version != version_before {
        match store.io_retag_with_delta_forks(domain, branch, merge_version).await {
            Ok((total, retagged)) => {
                eprintln!(
                    "[compaction] delta-fork retagged {}/{} tags",
                    retagged, total
                );
            }
            Err(e) => {
                eprintln!("[compaction] delta-fork retagging failed (soft): {}", e);
            }
        }
    }

    // Step 7: Final aggressive cleanup after retagging.
    if let Err(e) = store.io_cleanup_aggressive(domain, branch).await {
        eprintln!("[compaction] final cleanup failed (soft): {}", e);
    }

    // Step 8: Clean up orphaned tags and stale rebuild branches.
    if let Err(e) = store.io_cleanup_orphaned_tags_and_stale_branches(domain).await {
        eprintln!("[compaction] orphaned tag/stale branch cleanup failed (soft): {}", e);
    }

    // Step 9: Prune stale index directories (not just empty ones).
    if let Err(e) = store.io_prune_stale_index_dirs(domain, branch).await {
        eprintln!("[compaction] prune stale dirs failed (soft): {}", e);
    }

    // Step 10: Prune empty index directories left by LanceDB cleanup.
    if let Err(e) = store.io_prune_empty_index_dirs(domain) {
        eprintln!("[compaction] prune empty dirs failed (soft): {}", e);
    }

    // Refresh the cached handle so subsequent reads use compacted fragments.
    store
        .io_refresh_cached_dataset(domain, branch)
        .await
        .map_err(|e| format!("cache refresh after compaction failed: {}", e))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the same Expr used by io_upsert_chunks and io_delete_doc
    /// for individual doc_id deletion — demonstrates injection-safe construction.
    fn build_single_expr(doc_id: &str) -> lance::deps::datafusion::logical_expr::Expr {
        col("doc_id").eq(lit(doc_id))
    }

    #[test]
    fn filter_normal_iri_is_safe() {
        // Normal IRIs produce a simple equality Expr — no SQL string involved.
        let expr = build_single_expr("terminusdb:///db/People/123");
        let rendered = format!("{}", expr);
        assert!(rendered.contains("terminusdb:///db/People/123"));
    }

    #[test]
    fn filter_backslash_quote_is_safe_via_expr() {
        // Previously, a doc_id like x\' OR 1=1 could break out of a SQL string
        // literal via backslash escaping. With Expr-based construction, the
        // entire doc_id is a literal value — no SQL parsing occurs.
        let malicious = "x\\' OR 1=1";
        let expr = build_single_expr(malicious);
        let rendered = format!("{}", expr);
        // The malicious content is contained within the literal — no SQL injection.
        assert!(rendered.contains("OR 1=1"));
        // It's part of the literal value, not injected SQL.
        assert!(rendered.starts_with("doc_id = ") || rendered.contains("Utf8"));
    }

    #[test]
    fn filter_newline_is_safe_via_expr() {
        // Newlines in doc_ids are also safe — they're literal values, not SQL.
        let malicious = "doc1\n-- OR 1=1";
        let expr = build_single_expr(malicious);
        let rendered = format!("{}", expr);
        // The newline is part of the literal, not a SQL comment injection.
        assert!(rendered.contains("OR 1=1"));
    }

    #[test]
    fn filter_in_list_is_safe_via_expr() {
        // The IN-list path (build_doc_id_in_filter replacement) also uses Expr.
        let ids = ["evil\\' OR doc_id = 'admin".to_string()];
        let values: Vec<_> = ids.iter().map(|id| lit(id.as_str())).collect();
        let expr = lance::deps::datafusion::logical_expr::in_list(
            col("doc_id"),
            values,
            false,
        );
        let rendered = format!("{}", expr);
        // The entire malicious string is a single literal — no SQL injection.
        assert!(rendered.contains("admin"));
    }
}
