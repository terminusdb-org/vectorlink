// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 DFRNT AB

//! Statistics and domain deletion.

use crate::kernel::error::StoreError;
use crate::kernel::model::{InternalStats, Statistics};

use arrow_array::{Array, RecordBatch, StringArray};
use futures::TryStreamExt as _;
use lance::dataset::Dataset;

use super::{BranchKey, LanceStore, MAIN_BRANCH};

impl LanceStore {
    /// Discover on-disk datasets by scanning `base_dir` for `*.lance` directories.
    /// Returns a list of decoded domain strings (e.g. "admin/product_assortment").
    pub(crate) fn discover_on_disk_datasets(&self) -> Vec<String> {
        let mut found = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.base_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.ends_with(".lance") && path.is_dir() {
                        let stem = &name[..name.len() - ".lance".len()];
                        if let Some(domain) = super::decode_domain_path(stem) {
                            found.push(domain);
                        }
                    }
                }
            }
        }
        found
    }

    /// Count live chunks from a single dataset using manifest metadata only.
    /// live = sum(physical_rows) - sum(fast_num_deletions).
    /// No data file I/O — reads fragment metadata from the manifest.
    async fn count_dataset_rows(&self, uri: &str) -> u64 {
        let ds = match self.io_open_fresh(uri).await {
            Ok(ds) => ds,
            Err(_) => return 0,
        };

        let mut physical = 0u64;
        let mut deleted = 0u64;
        for frag in ds.get_fragments() {
            if let Some(rows) = frag.metadata().physical_rows {
                physical += rows as u64;
            }
            if let Ok(n) = frag.fast_num_deletions() {
                deleted += n as u64;
            }
        }
        physical.saturating_sub(deleted)
    }

    /// Count distinct doc_id values in a dataset by scanning only the doc_id
    /// column (no embedding data read). Used for the per-domain statistics
    /// endpoint so the UI shows an accurate "Searchable Documents" count.
    async fn count_distinct_doc_ids(&self, ds: &Dataset) -> u64 {
        let mut scanner = ds.scan();
        if scanner.project(&["doc_id"]).is_err() {
            return 0;
        }
        let stream = match scanner.try_into_stream().await {
            Ok(s) => s,
            Err(_) => return 0,
        };
        let batches: Vec<RecordBatch> = match stream.try_collect().await {
            Ok(b) => b,
            Err(_) => return 0,
        };
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for batch in &batches {
            if let Some(ids) = batch
                .column_by_name("doc_id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            {
                for i in 0..ids.len() {
                    seen.insert(ids.value(i).to_owned());
                }
            }
        }
        seen.len() as u64
    }

    /// Count total rows across all datasets (for statistics).
    /// Discovers on-disk datasets in addition to the in-memory cache, so
    /// statistics remain accurate after a server restart.
    pub async fn statistics(&self) -> Statistics {
        let datasets = self.datasets.read().await;
        let indexes = self.branch_indexes.read().await;

        let mut domains = std::collections::HashSet::new();
        let mut distinct_branches: std::collections::HashSet<BranchKey> =
            std::collections::HashSet::new();
        let mut indexed_commits = 0u64;
        let mut chunks = 0u64;

        // Collect domains from in-memory caches.
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

        // Count live chunks from cached datasets using manifest metadata only.
        // live = sum(physical_rows) - sum(fast_num_deletions). No data I/O.
        let mut deleted_total = 0u64;
        let mut physical_total = 0u64;
        for ds_arc in datasets.values() {
            let ds = ds_arc.read().await;
            for frag in ds.get_fragments() {
                if let Some(rows) = frag.metadata().physical_rows {
                    physical_total += rows as u64;
                }
                if let Ok(n) = frag.fast_num_deletions() {
                    deleted_total += n as u64;
                }
            }
        }

        // Count pending index fragments and documents across all cached datasets.
        let mut pending_index_fragments: u64 = 0;
        let mut pending_index_documents: u64 = 0;
        for ds_arc in datasets.values() {
            let ds = ds_arc.read().await;
            if let Ok((frags, rows)) = crate::store::vector_index::count_unindexed_rows(&ds).await {
                pending_index_fragments += frags;
                pending_index_documents += rows;
            }
        }

        // Release locks before doing fresh I/O for on-disk discovery.
        drop(datasets);
        drop(indexes);

        // Discover on-disk datasets not in the cache (e.g. after restart).
        let on_disk = self.discover_on_disk_datasets();
        for domain in &on_disk {
            domains.insert(domain.clone());
        }

        // Track which domains were already counted from the cache to avoid
        // double-counting when iterating on-disk datasets.
        let cached_domains: std::collections::HashSet<String> = {
            let datasets = self.datasets.read().await;
            datasets.keys().cloned().collect()
        };

        // For on-disk datasets not in the cache, open and cache the handle.
        for domain in &on_disk {
            if cached_domains.contains(domain) {
                continue;
            }
            if let Ok(Some(ds_arc)) = self.io_open_dataset_readonly(domain).await {
                let ds = ds_arc.read().await;
                for frag in ds.get_fragments() {
                    if let Some(rows) = frag.metadata().physical_rows {
                        physical_total += rows as u64;
                    }
                    if let Ok(n) = frag.fast_num_deletions() {
                        deleted_total += n as u64;
                    }
                }
                if let Ok(tags) = ds.tags().list().await {
                    indexed_commits += tags.len() as u64;
                }
            } else {
                let path = self.dataset_path(domain);
                let uri = path.to_string_lossy().to_string();
                chunks += self.count_dataset_rows(&uri).await;
            }
        }

        // Live chunks = physical - soft-deleted.
        chunks += physical_total.saturating_sub(deleted_total);

        // Document count is not available in the global statistics path
        // (would require scanning all datasets' doc_id columns). The
        // per-domain endpoint provides accurate counts.
        let documents = 0;

        Statistics {
            domains: domains.len() as u64,
            branches: distinct_branches.len() as u64,
            indexed_commits,
            documents,
            chunks,
            pending_index_fragments,
            pending_index_documents,
            store_clustering: None,
        }
    }

    /// Domain-scoped statistics — aggregates ONLY the named domain's footprint.
    /// The `domain` parameter is the normalised org/db key (e.g. "admin/abt_buy_e2e").
    /// Uses cached handles when available, falls back to on-disk discovery.
    pub async fn statistics_for_domain(&self, domain: &str) -> Statistics {
        let datasets = self.datasets.read().await;
        let indexes = self.branch_indexes.read().await;

        let mut domains_count: u64 = 0;
        let mut distinct_branches: std::collections::HashSet<BranchKey> =
            std::collections::HashSet::new();
        let mut indexed_commits = 0u64;
        let mut chunks = 0u64;

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

        // Count live chunks from this domain's cached dataset using manifest metadata only.
        let mut deleted_total = 0u64;
        let mut physical_total = 0u64;
        if let Some(ds_arc) = datasets.get(domain) {
            let ds = ds_arc.read().await;
            for frag in ds.get_fragments() {
                if let Some(rows) = frag.metadata().physical_rows {
                    physical_total += rows as u64;
                }
                if let Ok(n) = frag.fast_num_deletions() {
                    deleted_total += n as u64;
                }
            }
        }

        // Count pending index fragments and documents for this domain only.
        let mut pending_index_fragments: u64 = 0;
        let mut pending_index_documents: u64 = 0;
        if let Some(ds_arc) = datasets.get(domain) {
            let ds = ds_arc.read().await;
            if let Ok((frags, rows)) = crate::store::vector_index::count_unindexed_rows(&ds).await {
                pending_index_fragments += frags;
                pending_index_documents += rows;
            }
        }

        // Release locks before doing fresh I/O.
        drop(datasets);
        drop(indexes);

        // If the domain wasn't in the cache, check on disk (e.g. after restart).
        let path = self.dataset_path(domain);
        if path.exists() && domains_count == 0 {
            domains_count = 1;
        }

        // If we didn't get data from the cache, open from disk and cache it.
        if physical_total == 0 && path.exists() {
            if let Ok(Some(ds_arc)) = self.io_open_dataset_readonly(domain).await {
                let ds = ds_arc.read().await;
                for frag in ds.get_fragments() {
                    if let Some(rows) = frag.metadata().physical_rows {
                        physical_total += rows as u64;
                    }
                    if let Ok(n) = frag.fast_num_deletions() {
                        deleted_total += n as u64;
                    }
                }
                if let Ok(tags) = ds.tags().list().await {
                    indexed_commits = tags.len() as u64;
                }
            } else {
                let uri = path.to_string_lossy().to_string();
                chunks = self.count_dataset_rows(&uri).await;

                // Count tagged commits from the on-disk dataset.
                if let Ok(ds) = self.io_open_fresh(&uri).await {
                    if let Ok(tags) = ds.tags().list().await {
                        indexed_commits = tags.len() as u64;
                    }
                }
            }
        }

        // Live chunks = physical - soft-deleted (add to any fallback count).
        chunks += physical_total.saturating_sub(deleted_total);

        // Count distinct documents by scanning the doc_id column.
        // This is accurate even after a server restart (unlike the pipeline
        // counter). The scan reads only the doc_id column — no embedding data.
        let documents = if chunks > 0 {
            // Try cached dataset first, then fall back to a fresh open.
            let cached = self.datasets.read().await.get(domain).cloned();
            if let Some(ds_arc) = cached {
                let ds = ds_arc.read().await;
                self.count_distinct_doc_ids(&ds).await
            } else {
                let uri = path.to_string_lossy().to_string();
                if let Ok(ds) = self.io_open_fresh(&uri).await {
                    self.count_distinct_doc_ids(&ds).await
                } else {
                    0
                }
            }
        } else {
            0
        };

        // Surface clustering flag for this domain.
        let store_clustering = self.domain_settings.store_clustering(domain).await;

        Statistics {
            domains: domains_count,
            branches,
            indexed_commits,
            documents,
            chunks,
            pending_index_fragments,
            pending_index_documents,
            store_clustering: Some(store_clustering),
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
            assert!(
                path.extension().is_some_and(|ext| ext == "lance"),
                "refusing to delete non-.lance path: {}",
                path.display()
            );
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
            // Cancel and purge any in-flight pipeline tasks for this domain.
            let mut cancel_tokens = self.pipeline_cancel_tokens.write().await;
            let keys_to_remove: Vec<_> = cancel_tokens.keys().filter(|(d, _b)| d == domain).cloned().collect();
            for key in keys_to_remove {
                if let Some(token) = cancel_tokens.remove(&key) {
                    token.cancel();
                }
            }
        }
        {
            // Purge any in-flight reservations for this domain so a delete +
            // re-push of the same commit is not wrongly 409'd by a stale
            // reservation left over from a push that the delete superseded.
            let mut inflight = self.inflight_commits.write().await;
            inflight.retain(|(d, _b, _c)| d != domain);
        }
        {
            // Purge the domain guard entry. The guard itself was held during
            // this deletion (acquired above), but the map entry persists —
            // remove it so domains created/deleted many times don't leak.
            let mut guards = self.domain_guards.write().await;
            guards.remove(domain);
        }
        {
            // Purge index delta counts for all branches of this domain.
            let mut deltas = self.index_delta_counts.write().await;
            deltas.retain(|(d, _b), _| d != domain);
        }

        Ok(())
    }

    /// Delete all Lance tags belonging to a specific (domain, branch) pair,
    /// and purge the in-memory caches for that branch only. Other branches
    /// sharing the same domain dataset are left intact.
    ///
    /// IDEMPOTENT: deleting an unknown/already-removed branch succeeds.
    /// Does NOT touch the on-disk dataset directory or sled.
    pub async fn io_delete_branch_index(
        &self,
        domain: &str,
        branch: &str,
    ) -> Result<(), StoreError> {
        let key: BranchKey = (domain.to_owned(), branch.to_owned());

        // 1. Cancel any in-flight pipeline task for this branch.
        {
            let mut cancel_tokens = self.pipeline_cancel_tokens.write().await;
            if let Some(token) = cancel_tokens.remove(&key) {
                token.cancel();
            }
        }

        // 2. Acquire the pipeline lock — this waits for any in-flight push
        //    to finish (the cancel token above ensures it stops ASAP, but
        //    it only checks between sub-batches, so we must wait).
        //    This prevents a race where we delete tags while the pipeline
        //    is still writing new ones.
        let _pipeline_guard = self.acquire_pipeline_lock(domain, branch).await;

        // 3. Delete all Lance tags belonging to this branch.
        if let Some(ds_arc) = self.io_open_dataset_readonly(domain).await? {
            let ds = ds_arc.read().await;
            let tags = ds
                .tags()
                .list()
                .await
                .map_err(|e| StoreError::Internal(format!("tag list failed: {}", e)))?;

            let want_branch: Option<&str> =
                if branch == MAIN_BRANCH { None } else { Some(branch) };

            let tags_to_delete: Vec<String> = tags
                .into_iter()
                .filter(|(_, contents)| contents.branch.as_deref() == want_branch)
                .map(|(tag_name, _)| tag_name)
                .collect();

            drop(ds);

            for tag_name in &tags_to_delete {
                ds_arc.read()
                    .await
                    .tags()
                    .delete(tag_name)
                    .await
                    .map_err(|e| {
                        StoreError::Internal(format!("tag delete failed: {}", e))
                    })?;
            }
        }

        // 4. Purge in-memory state for this (domain, branch) only.
        //    Do NOT remove the pipeline_lock entry — the lock itself is
        //    released when _pipeline_guard drops, and removing the map
        //    entry could race with a concurrent acquire_pipeline_lock.
        {
            let mut indexes = self.branch_indexes.write().await;
            indexes.remove(&key);
        }
        {
            let mut inflight = self.inflight_commits.write().await;
            inflight.retain(|(d, b, _c)| !(*d == domain && *b == branch));
        }
        {
            // Purge index delta count for this (domain, branch).
            let mut deltas = self.index_delta_counts.write().await;
            deltas.remove(&key);
        }

        Ok(())
    }

    /// Internal instrumentation: sizes of all in-memory data structures.
    /// Used by the global /statistics endpoint for leak monitoring.
    pub async fn internal_stats(&self) -> InternalStats {
        let cached_datasets = self.datasets.read().await.len();
        let tasks = self.tasks.read().await.len();
        let branch_indexes = self.branch_indexes.read().await.len();
        let pipeline_locks = self.pipeline_locks.read().await.len();
        let domain_guards = self.domain_guards.read().await.len();
        let inflight_commits = self.inflight_commits.read().await.len();

        InternalStats {
            cached_datasets,
            tasks,
            branch_indexes,
            pipeline_locks,
            domain_guards,
            inflight_commits,
            pipeline_pending_chunks: self.pipeline_pending_chunks.load(std::sync::atomic::Ordering::Relaxed),
            pipeline_embedded_chunks: self.pipeline_embedded_chunks.load(std::sync::atomic::Ordering::Relaxed),
            pipeline_written_chunks: self.pipeline_written_chunks.load(std::sync::atomic::Ordering::Relaxed),
            pipeline_active_tasks: self.pipeline_active_tasks.load(std::sync::atomic::Ordering::Relaxed),
            fresh_open_count: self.fresh_open_count.load(std::sync::atomic::Ordering::Relaxed),
            embed_cache_entries: 0,
            embed_cache_size_bytes: 0,
            lance_index_cache_capacity_bytes: self.lance_index_cache_capacity,
            lance_metadata_cache_capacity_bytes: self.lance_metadata_cache_capacity,
        }
    }
}
