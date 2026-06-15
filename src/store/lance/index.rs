//! Index maintenance: FTS index creation/optimization and data compaction.

use lance::dataset::Dataset;
use lance::index::DatasetIndexExt;
use lance_index::IndexType;
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::InvertedIndexParams;

use crate::kernel::error::StoreError;

use super::LanceStore;

impl LanceStore {
    /// Ensure the FTS inverted index EXISTS (create if not present, no-op otherwise).
    /// Used inline during push to guarantee FTS is available for search immediately.
    /// Does NOT call optimize_indices — that's deferred to the background worker.
    ///
    /// Unlike vector search (which flat-scans unindexed fragments), FTS REQUIRES
    /// the inverted index to exist. Calling this inline on each push is O(1) after
    /// the first creation because it's a metadata check only.
    pub async fn io_ensure_fts_index_created(
        &self,
        domain: &str,
        branch: &str,
    ) -> Result<u64, StoreError> {
        let ds_arc = self.io_open_dataset(domain, branch).await?;
        let mut ds = ds_arc.write().await;

        let indices = ds.load_indices().await
            .map_err(|e| StoreError::Internal(format!("load indices failed: {}", e)))?;
        let has_fts = indices.iter().any(|idx| idx.name == "content_fts");

        if !has_fts {
            // First time: create the inverted index covering all current fragments.
            let params = InvertedIndexParams::default();
            ds.create_index(
                &["content"],
                IndexType::Inverted,
                Some("content_fts".to_owned()),
                &params,
                false,
            )
            .await
            .map_err(|e| StoreError::Internal(format!("FTS index creation failed: {}", e)))?;
        }

        Ok(ds.version().version)
    }

    /// Ensure the FTS inverted index is fully up-to-date: create if missing, then
    /// incrementally index any new (unindexed) fragments via optimize_indices(append).
    /// Called by the background index worker — NOT on the push hot path.
    pub async fn io_ensure_fts_index(
        &self,
        domain: &str,
        branch: &str,
    ) -> Result<u64, StoreError> {
        let ds_arc = self.io_open_dataset(domain, branch).await?;
        let mut ds = ds_arc.write().await;

        // Check if the FTS index already exists.
        let indices = ds.load_indices().await
            .map_err(|e| StoreError::Internal(format!("load indices failed: {}", e)))?;
        let has_fts = indices.iter().any(|idx| idx.name == "content_fts");

        if !has_fts {
            // First time: create the inverted index.
            let params = InvertedIndexParams::default();
            ds.create_index(
                &["content"],
                IndexType::Inverted,
                Some("content_fts".to_owned()),
                &params,
                false,
            )
            .await
            .map_err(|e| StoreError::Internal(format!("FTS index creation failed: {}", e)))?;
        } else {
            // Index exists: incrementally index only new (unindexed) fragments.
            // OptimizeOptions::append() sets num_indices_to_merge=0, meaning:
            // create a new delta index from unindexed fragments only (O(delta)).
            ds.optimize_indices(&OptimizeOptions::append())
                .await
                .map_err(|e| StoreError::Internal(format!("FTS index optimize failed: {}", e)))?;
        }

        Ok(ds.version().version)
    }
}

/// Ensure the FTS inverted index is up-to-date on an already-open Dataset handle.
/// Used by the background index worker to operate on an uncached handle (no RwLock
/// contention with search reads).
///
/// Creates the index if missing; if it already exists, runs optimize_indices(append())
/// to index only new (unindexed) fragments — O(delta).
pub async fn io_ensure_fts_index_on_dataset(ds: &mut Dataset) -> Result<u64, StoreError> {
    let indices = ds
        .load_indices()
        .await
        .map_err(|e| StoreError::Internal(format!("load indices failed: {}", e)))?;
    let has_fts = indices.iter().any(|idx| idx.name == "content_fts");

    if !has_fts {
        let params = InvertedIndexParams::default();
        ds.create_index(
            &["content"],
            IndexType::Inverted,
            Some("content_fts".to_owned()),
            &params,
            false,
        )
        .await
        .map_err(|e| StoreError::Internal(format!("FTS index creation failed: {}", e)))?;
    } else {
        ds.optimize_indices(&OptimizeOptions::append())
            .await
            .map_err(|e| StoreError::Internal(format!("FTS index optimize failed: {}", e)))?;
    }

    Ok(ds.version().version)
}

/// Compact data fragments in a dataset, merging many small fragments into fewer
/// large ones. This is critical for FD safety (BUG-FD24): each push creates one
/// fragment, so a dataset with N pushes has N fragments. A full-table scan opens
/// ALL fragments simultaneously — with 2000+ pushes this exceeds nofile=1024.
///
/// After compaction, the fragment count is reduced to O(1) (typically 1-2 large
/// fragments), so full-table scans use only a few FDs.
///
/// Called by the background index worker AFTER index optimization. Idempotent:
/// if the dataset already has a reasonable number of fragments (<= target), this
/// is effectively a no-op (returns quickly without rewriting data).
pub async fn io_compact_data(ds: &mut Dataset) -> Result<(), StoreError> {
    use lance::dataset::optimize::{compact_files, CompactionOptions};

    // Only compact if fragment count is above the threshold — avoids needless
    // I/O when the dataset is already well-compacted.
    const COMPACT_FRAGMENT_THRESHOLD: usize = 16;

    let fragment_count = ds.get_fragments().len();
    if fragment_count <= COMPACT_FRAGMENT_THRESHOLD {
        return Ok(());
    }

    let options = CompactionOptions::default();
    compact_files(ds, options, None)
        .await
        .map_err(|e| StoreError::Internal(format!("data compaction failed: {}", e)))?;

    Ok(())
}
