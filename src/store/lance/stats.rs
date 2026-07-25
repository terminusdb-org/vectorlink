//! Statistics and domain deletion.

use arrow_array::{Array, RecordBatch, StringArray};
use futures::TryStreamExt;

use crate::kernel::error::StoreError;
use crate::kernel::model::Statistics;

use super::{BranchKey, LanceStore};

impl LanceStore {
    /// Count total rows across all datasets (for statistics).
    pub async fn statistics(&self) -> Statistics {
        let datasets = self.datasets.read().await;
        let indexes = self.branch_indexes.read().await;

        // Layout A: each cache entry is one domain dataset. Branch count comes
        // from the per-(domain, branch) index map, not the dataset count.
        let mut domains = std::collections::HashSet::new();
        let mut distinct_branches: std::collections::HashSet<BranchKey> =
            std::collections::HashSet::new();
        let mut indexed_commits = 0u64;
        let mut chunks = 0u64;
        let mut distinct_docs: std::collections::HashSet<String> = std::collections::HashSet::new();

        for domain in datasets.keys() {
            domains.insert(domain.clone());
        }

        for (key, bi) in indexes.iter() {
            distinct_branches.insert(key.clone());
            domains.insert(key.0.clone());
            if bi.commit.is_some() {
                indexed_commits += 1;
            }
        }
        let branches = distinct_branches.len() as u64;

        // Count rows and distinct doc_ids from datasets (best-effort).
        for (_key, ds_arc) in datasets.iter() {
            let ds = ds_arc.read().await;
            if let Ok(count) = ds.count_rows(None).await {
                chunks += count as u64;
            }
            // Scan doc_id column for distinct count.
            let mut scanner = ds.scan();
            if scanner.project(&["doc_id"]).is_ok() {
                if let Ok(stream) = scanner.try_into_stream().await {
                    if let Ok(batches) = stream.try_collect::<Vec<RecordBatch>>().await {
                        for batch in &batches {
                            if let Some(ids) = batch
                                .column_by_name("doc_id")
                                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                            {
                                for i in 0..ids.len() {
                                    distinct_docs.insert(ids.value(i).to_owned());
                                }
                            }
                        }
                    }
                }
            }
        }

        // Count pending index fragments across all datasets.
        let mut pending_index_fragments: u64 = 0;
        for (_key, ds_arc) in datasets.iter() {
            let ds = ds_arc.read().await;
            if let Ok(pending) = crate::store::vector_index::count_unindexed_fragments(&ds).await {
                pending_index_fragments += pending;
            }
        }

        Statistics {
            domains: domains.len() as u64,
            branches,
            indexed_commits,
            documents: distinct_docs.len() as u64,
            chunks,
            pending_index_fragments,
        }
    }

    /// Domain-scoped statistics — aggregates ONLY the named domain's footprint.
    /// The `domain` parameter is the normalised org/db key (e.g. "admin/abt_buy_e2e").
    /// Uses cached handles (no fresh opens) — FD-safe.
    pub async fn statistics_for_domain(&self, domain: &str) -> Statistics {
        let datasets = self.datasets.read().await;
        let indexes = self.branch_indexes.read().await;

        let mut domains_count: u64 = 0;
        let mut distinct_branches: std::collections::HashSet<BranchKey> =
            std::collections::HashSet::new();
        let mut indexed_commits = 0u64;
        let mut chunks = 0u64;
        let mut distinct_docs: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Check if this domain exists in datasets.
        if datasets.contains_key(domain) {
            domains_count = 1;
        }

        // Filter branch_indexes to only this domain.
        for (key, bi) in indexes.iter() {
            if key.0 == domain {
                distinct_branches.insert(key.clone());
                if domains_count == 0 {
                    domains_count = 1;
                }
                if bi.commit.is_some() {
                    indexed_commits += 1;
                }
            }
        }
        let branches = distinct_branches.len() as u64;

        // Count rows and distinct doc_ids from this domain's dataset only.
        if let Some(ds_arc) = datasets.get(domain) {
            let ds = ds_arc.read().await;
            if let Ok(count) = ds.count_rows(None).await {
                chunks += count as u64;
            }
            // Scan doc_id column for distinct count.
            let mut scanner = ds.scan();
            if scanner.project(&["doc_id"]).is_ok() {
                if let Ok(stream) = scanner.try_into_stream().await {
                    if let Ok(batches) = stream.try_collect::<Vec<RecordBatch>>().await {
                        for batch in &batches {
                            if let Some(ids) = batch
                                .column_by_name("doc_id")
                                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                            {
                                for i in 0..ids.len() {
                                    distinct_docs.insert(ids.value(i).to_owned());
                                }
                            }
                        }
                    }
                }
            }
        }

        // Count pending index fragments for this domain only.
        let mut pending_index_fragments: u64 = 0;
        if let Some(ds_arc) = datasets.get(domain) {
            let ds = ds_arc.read().await;
            if let Ok(pending) = crate::store::vector_index::count_unindexed_fragments(&ds).await {
                pending_index_fragments += pending;
            }
        }

        Statistics {
            domains: domains_count,
            branches,
            indexed_commits,
            documents: distinct_docs.len() as u64,
            chunks,
            pending_index_fragments,
        }
    }

    /// Delete a domain's ENTIRE store footprint (layout A): drop the single
    /// `{domain}.lance` dataset (all branches/versions/tags) AND purge the
    /// domain's in-memory state (dataset cache entry, every `(domain, *)` entry
    /// in `branch_indexes` and `pipeline_locks`).
    ///
    /// IDEMPOTENT: deleting an unknown/already-removed domain succeeds (returns
    /// `Ok(())`) — TerminusDB may retry. We fail loud only on a genuine I/O error
    /// (the directory exists but cannot be removed).
    ///
    /// FAIL-LOUD partial-delete guard: the on-disk dataset is removed FIRST; only
    /// if that succeeds (or there was nothing on disk) do we purge in-memory
    /// state. If the disk removal errors, we surface it and leave in-memory state
    /// intact — never a half-deleted footprint (searchable map entry with no
    /// dataset, or a dataset with no map entry).
    pub async fn io_delete_domain(&self, domain: &str) -> Result<(), StoreError> {
        // Hold the per-domain guard across the WHOLE remove-then-purge so a
        // concurrent first-write that creates the dataset cannot interleave
        // (BLOCKER-2 / #6): either the create completes before we remove (and we
        // remove it), or it waits until after we have removed+purged (and then
        // legitimately re-creates as a fresh index). Never a half-deleted state.
        let _guard = self.acquire_domain_guard(domain).await;

        // 1. Drop the on-disk dataset directory first.
        let path = self.dataset_path(domain);
        if path.exists() {
            tokio::fs::remove_dir_all(&path).await.map_err(|e| {
                StoreError::Internal(format!(
                    "failed to remove dataset directory {}: {}",
                    path.display(),
                    e
                ))
            })?;
        }

        // 2. Disk removal succeeded (or nothing was there) — purge in-memory
        //    state. Each map is purged under its own lock.
        {
            let mut datasets = self.datasets.write().await;
            datasets.remove(domain);
        }
        {
            let mut indexes = self.branch_indexes.write().await;
            indexes.retain(|(d, _b), _| d != domain);
        }
        {
            let mut locks = self.pipeline_locks.write().await;
            locks.retain(|(d, _b), _| d != domain);
        }
        {
            // Purge any in-flight reservations for this domain so a delete +
            // re-push of the same commit is not wrongly 409'd by a stale
            // reservation left over from a push that the delete superseded.
            let mut inflight = self.inflight_commits.write().await;
            inflight.retain(|(d, _b, _c)| d != domain);
        }

        Ok(())
    }
}
