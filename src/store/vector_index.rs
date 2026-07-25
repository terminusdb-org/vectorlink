// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 DFRNT AB

#![forbid(unsafe_code)]

//! Vector ANN index management for LanceDB datasets.
//!
//! Creates and maintains a vector index (IVF_HNSW_SQ) on the `embedding` column.
//! The index uses cosine distance (vectors are L2-normalised before insert,
//! so Lance's DistanceType::Cosine gives the correct metric).
//!
//! Index type: IVF_HNSW_SQ (Inverted File + HNSW graph + Scalar Quantisation).
//! - IVF partitions the space for coarse search (scalability).
//! - HNSW graph provides high-recall traversal within each partition.
//! - Scalar Quantisation (SQ) preserves per-dimension precision (8-bit per dim),
//!   far less lossy than the previous PQ (Product Quantisation) which compressed
//!   sub-vector groups into codebook centroids.
//!
//! Index lifecycle:
//! - Created once when first needed (`io_ensure_vector_index`).
//! - Incrementally updated via `optimize_indices(OptimizeOptions::append())`
//!   after each push (indexes only new/unindexed fragments).
//! - Searched via `nearest()` with `nprobes` and `refine_factor` tuning params.
//!
//! Lance flat-searches unindexed fragments alongside the indexed ones, so
//! correctness is guaranteed even before optimize runs — only latency is affected.

use lance::dataset::Dataset;
use lance::index::DatasetIndexExt;
use lance_index::IndexType;
use lance_index::optimize::OptimizeOptions;
use lance_linalg::distance::DistanceType;

use crate::kernel::error::StoreError;
use crate::store::lance::VectorIndexConfig;

/// Index name for the vector ANN index on `embedding`.
pub const VECTOR_INDEX_NAME: &str = "embedding_ann";

/// Ensure a vector ANN index exists on the `embedding` column.
///
/// On first call: creates the IVF_HNSW_SQ index with the configured parameters.
/// On subsequent calls: incrementally indexes only new (unindexed) fragments
/// via `optimize_indices(OptimizeOptions::append())` — O(new_data), not O(corpus).
///
/// INVARIANT: if the index SHOULD exist (i.e., this function has been called
/// before without error) but is missing, this is a system integrity failure
/// and must fail loud. We never silently degrade to flat scan after initial creation.
///
/// Returns the dataset version after index creation/optimization.
pub async fn io_ensure_vector_index(
    ds: &mut Dataset,
    config: &VectorIndexConfig,
    force: bool,
) -> Result<u64, StoreError> {
    let indices = ds
        .load_indices()
        .await
        .map_err(|e| StoreError::Internal(format!("load indices failed: {}", e)))?;

    let has_vector_idx = indices.iter().any(|idx| idx.name == VECTOR_INDEX_NAME);

    if !has_vector_idx {
        // IVF_HNSW_SQ minimum training requirement:
        // - IVF KMeans needs at least `num_partitions` vectors (one per centroid).
        // - SQ (Scalar Quantisation) uses sample_rate * 256 training vectors
        //   (default sample_rate=256 → 65536 samples, but it can subsample from
        //   fewer rows). The binding constraint is IVF: num_partitions centroids.
        // - HNSW graph construction has no minimum; it works with any count > 0.
        // Without enough data for IVF KMeans, `create_index` fails with a training error.
        // Lance flat-searches unindexed fragments correctly, so search stays correct
        // during this window. The index will be created on a future call once enough
        // data accumulates.
        let row_count = ds
            .count_rows(None)
            .await
            .map_err(|e| StoreError::Internal(format!("count rows failed: {}", e)))?;

        // Compute adaptive partition count: max(config floor, sqrt(row_count)).
        let num_partitions = config.recommended_num_partitions(row_count);

        // IVF needs num_partitions centroids. SQ's sample_rate is adaptive
        // (works with fewer rows by subsampling). The safe floor is the IVF
        // partitions requirement, with a practical minimum of 256 for stable
        // centroid training (KMeans needs reasonable cluster diversity).
        // When `force` is true (manual compaction), the 256 floor is skipped
        // and only the adaptive num_partitions minimum is used.
        let ivf_min_training_vectors: usize = 256;
        let min_rows_for_index = if force {
            num_partitions
        } else {
            num_partitions.max(ivf_min_training_vectors)
        };
        if row_count < min_rows_for_index {
            // Not enough data to train the index yet — return current version.
            // Search uses flat-scan (correct, higher latency on small corpus is acceptable).
            return Ok(ds.version().version);
        }

        // First time: create the vector index with adaptive partition count.
        let mut adaptive_config = config.clone();
        adaptive_config.num_partitions = num_partitions;
        eprintln!(
            "[io_ensure_vector_index] creating index: {} partitions for {} rows",
            num_partitions, row_count
        );
        let params = build_vector_index_params(&adaptive_config);
        ds.create_index(
            &["embedding"],
            IndexType::Vector,
            Some(VECTOR_INDEX_NAME.to_owned()),
            &params,
            false,
        )
        .await
        .map_err(|e| StoreError::Internal(format!("vector index creation failed: {}", e)))?;
    } else {
        // Index exists: incrementally index new fragments only.
        // Per-push uses append() — the incremental merge(3) cascade
        // (io_incremental_cascade) consolidates deltas after each push.
        ds.optimize_indices(&OptimizeOptions::append())
            .await
            .map_err(|e| StoreError::Internal(format!("vector index optimize failed: {}", e)))?;
    }

    Ok(ds.version().version)
}

/// Index name for the vector ANN index on `clustering_embedding`.
pub const CLUSTERING_INDEX_NAME: &str = "clustering_embedding_ann";

/// Ensure a vector ANN index exists on the `clustering_embedding` column.
///
/// Only called when `store_clustering` is enabled for the domain. Uses the same
/// IVF_HNSW_SQ index type and cosine distance as the document embedding index.
/// Returns the dataset version after index creation/optimization.
pub async fn io_ensure_clustering_vector_index(
    ds: &mut Dataset,
    config: &VectorIndexConfig,
    force: bool,
) -> Result<u64, StoreError> {
    let indices = ds
        .load_indices()
        .await
        .map_err(|e| StoreError::Internal(format!("load indices failed: {}", e)))?;

    let has_clustering_idx = indices.iter().any(|idx| idx.name == CLUSTERING_INDEX_NAME);

    if !has_clustering_idx {
        let row_count = ds
            .count_rows(None)
            .await
            .map_err(|e| StoreError::Internal(format!("count rows failed: {}", e)))?;

        let num_partitions = config.recommended_num_partitions(row_count);
        let ivf_min_training_vectors: usize = 256;
        let min_rows_for_index = if force {
            num_partitions
        } else {
            num_partitions.max(ivf_min_training_vectors)
        };
        if row_count < min_rows_for_index {
            return Ok(ds.version().version);
        }

        let mut adaptive_config = config.clone();
        adaptive_config.num_partitions = num_partitions;
        let params = build_vector_index_params(&adaptive_config);
        ds.create_index(
            &["clustering_embedding"],
            IndexType::Vector,
            Some(CLUSTERING_INDEX_NAME.to_owned()),
            &params,
            false,
        )
        .await
        .map_err(|e| StoreError::Internal(format!("clustering vector index creation failed: {}", e)))?;
    } else {
        ds.optimize_indices(&OptimizeOptions::append())
            .await
            .map_err(|e| StoreError::Internal(format!("clustering vector index optimize failed: {}", e)))?;
    }

    Ok(ds.version().version)
}

/// Count the number of unindexed fragments for the vector index.
/// Returns 0 if no vector index exists (pre-creation state).
///
/// This powers the `pending_index_fragments` field in `/statistics`,
/// giving operators observable depth into the indexing backlog.
///
/// The vector index may consist of multiple segments (after incremental
/// optimize), each covering a different set of fragments. We union all
/// segment coverage bitmaps and subtract from the dataset's current
/// fragment set to find unindexed fragments.
pub async fn count_unindexed_fragments(ds: &Dataset) -> Result<u64, StoreError> {
    let indices = ds
        .load_indices()
        .await
        .map_err(|e| StoreError::Internal(format!("load indices failed: {}", e)))?;

    // Collect all segments belonging to the vector index.
    let vector_segments: Vec<_> = indices
        .iter()
        .filter(|idx| idx.name == VECTOR_INDEX_NAME)
        .collect();

    if vector_segments.is_empty() {
        // No vector index exists yet — report 0 (pre-creation state).
        return Ok(0);
    }

    // Union all fragment bitmaps across segments.
    use roaring::RoaringBitmap;
    let mut covered = RoaringBitmap::new();
    for segment in &vector_segments {
        if let Some(bitmap) = &segment.fragment_bitmap {
            covered |= bitmap;
        }
    }

    // Get current fragment IDs from the dataset.
    let mut all_fragment_ids = RoaringBitmap::new();
    for frag in ds.get_fragments() {
        // frag.id() is usize (from u64). Lance uses these in its own RoaringBitmap (u32),
        // so truncation is safe (Lance's fragment IDs fit in u32 by design).
        let id_u32: u32 = frag.id().try_into().map_err(|_| {
            StoreError::Internal(format!("fragment id {} exceeds u32", frag.id()))
        })?;
        all_fragment_ids.insert(id_u32);
    }

    // Unindexed = fragments in dataset that are NOT in any index segment.
    let unindexed = &all_fragment_ids - &covered;
    Ok(unindexed.len())
}

/// Count total physical rows in unindexed fragments for the vector index.
/// Sibling to `count_unindexed_fragments` — measures flat-scan cost in rows,
/// not just fragment count. Returns (fragment_count, total_rows).
pub async fn count_unindexed_rows(ds: &Dataset) -> Result<(u64, u64), StoreError> {
    let indices = ds
        .load_indices()
        .await
        .map_err(|e| StoreError::Internal(format!("load indices failed: {}", e)))?;

    let vector_segments: Vec<_> = indices
        .iter()
        .filter(|idx| idx.name == VECTOR_INDEX_NAME)
        .collect();

    if vector_segments.is_empty() {
        return Ok((0, 0));
    }

    use roaring::RoaringBitmap;
    let mut covered = RoaringBitmap::new();
    for segment in &vector_segments {
        if let Some(bitmap) = &segment.fragment_bitmap {
            covered |= bitmap;
        }
    }

    let mut unindexed_count = 0u64;
    let mut unindexed_rows = 0u64;
    for frag in ds.get_fragments() {
        let id_u32: u32 = frag.id().try_into().map_err(|_| {
            StoreError::Internal(format!("fragment id {} exceeds u32", frag.id()))
        })?;
        if !covered.contains(id_u32) {
            unindexed_count += 1;
            if let Some(rows) = frag.metadata().physical_rows {
                unindexed_rows += rows as u64;
            }
        }
    }

    Ok((unindexed_count, unindexed_rows))
}

/// Build VectorIndexParams from config.
/// Uses IVF_HNSW_SQ with cosine distance metric.
///
/// IVF_HNSW_SQ combines:
/// - IVF (num_partitions) for coarse partitioning
/// - HNSW (M, ef_construction) for graph-based traversal within partitions
/// - SQ (8-bit scalar quantisation) for per-dimension precision preservation
fn build_vector_index_params(
    config: &VectorIndexConfig,
) -> lance::index::vector::VectorIndexParams {
    use lance::index::vector::VectorIndexParams;
    use lance_index::vector::hnsw::builder::HnswBuildParams;
    use lance_index::vector::ivf::IvfBuildParams;
    use lance_index::vector::sq::builder::SQBuildParams;

    let ivf = IvfBuildParams::new(config.num_partitions);
    let hnsw = HnswBuildParams::default()
        .num_edges(config.m)
        .ef_construction(config.ef_construction);
    let sq = SQBuildParams::default(); // 8-bit scalar quantisation, sample_rate=256

    VectorIndexParams::with_ivf_hnsw_sq_params(DistanceType::Cosine, ivf, hnsw, sq)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::store::lance::LanceStore;
    use crate::store::lance::ChunkRow;

    /// Create a seeded, L2-normalised random embedding (deterministic).
    /// Adapted from the reference bench: `terminusdb-semantic-indexer/benches/distance.rs`.
    pub(crate) fn seeded_normalized_embedding(dim: usize, seed: u64) -> Vec<f32> {
        // Simple seeded PRNG (xorshift64) — deterministic, no rand dependency needed.
        let mut state = seed;
        let raw: Vec<f32> = (0..dim)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // Map to [-1, 1] range.
                (state as f32 / u64::MAX as f32) * 2.0 - 1.0
            })
            .collect();
        // L2-normalise.
        let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            raw
        } else {
            raw.into_iter().map(|x| x / norm).collect()
        }
    }

    pub(crate) fn make_test_store(dim: usize) -> (LanceStore, tempfile::TempDir) {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let store = LanceStore::new(tmp.path(), dim, 256 * 1024 * 1024, 128 * 1024 * 1024);
        (store, tmp)
    }

    pub(crate) fn make_test_config(_dim: usize) -> VectorIndexConfig {
        VectorIndexConfig {
            num_partitions: 4,
            m: 16,
            ef_construction: 100,
            nprobes: 20,
            refine_factor: Some(10),
        }
    }

    pub(crate) fn make_chunk_row(doc_id: &str, dim: usize, seed: u64) -> ChunkRow {
        ChunkRow {
            doc_id: doc_id.to_owned(),
            doc_type: "TestDoc".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 100,
            embedding: seeded_normalized_embedding(dim, seed),
            clustering_embedding: seeded_normalized_embedding(dim, seed + 10000),
            content: format!("content for {}", doc_id),
        }
    }

    /// P2.5-1: Vector index is created by io_ensure_vector_index.
    #[tokio::test]
    async fn vector_index_created_on_first_call() {
        let (store, _tmp) = make_test_store(128);
        let config = make_test_config(128);

        // Insert enough rows for IVF_PQ to work (needs >= num_partitions rows).
        let rows: Vec<ChunkRow> = (0..300)
            .map(|i| make_chunk_row(&format!("doc/{}", i), 128, i as u64 + 1))
            .collect();

        // Upsert all at once (batch).
        for row in &rows {
            store
                .io_upsert_chunks("admin/test", "main", &row.doc_id, std::slice::from_ref(row))
                .await
                .expect("upsert");
        }

        let ds_arc = store.io_open_dataset("admin/test", "main").await.unwrap();
        let mut ds = ds_arc.write().await;

        // Before: no vector index.
        let indices_before = ds.load_indices().await.unwrap();
        assert!(
            !indices_before.iter().any(|i| i.name == VECTOR_INDEX_NAME),
            "vector index should not exist before io_ensure_vector_index"
        );

        // Create.
        let version = io_ensure_vector_index(&mut ds, &config, false).await.unwrap();
        assert!(version > 0);

        // After: vector index exists.
        let indices_after = ds.load_indices().await.unwrap();
        assert!(
            indices_after.iter().any(|i| i.name == VECTOR_INDEX_NAME),
            "vector index should exist after io_ensure_vector_index"
        );
    }

    /// P2.5-2: Subsequent calls to io_ensure_vector_index are incremental (append).
    #[tokio::test]
    async fn vector_index_incremental_on_subsequent_calls() {
        let (store, _tmp) = make_test_store(128);
        let config = make_test_config(128);

        // Insert initial batch.
        let rows: Vec<ChunkRow> = (0..300)
            .map(|i| make_chunk_row(&format!("doc/{}", i), 128, i as u64 + 1))
            .collect();
        for row in &rows {
            store
                .io_upsert_chunks("admin/test", "main", &row.doc_id, std::slice::from_ref(row))
                .await
                .expect("upsert");
        }

        let ds_arc = store.io_open_dataset("admin/test", "main").await.unwrap();
        let mut ds = ds_arc.write().await;

        // First call: creates.
        let v1 = io_ensure_vector_index(&mut ds, &config, false).await.unwrap();

        // Insert more data.
        drop(ds);
        for i in 300..350 {
            let row = make_chunk_row(&format!("doc/{}", i), 128, i as u64 + 1);
            store
                .io_upsert_chunks("admin/test", "main", &row.doc_id, std::slice::from_ref(&row))
                .await
                .expect("upsert new");
        }

        let ds_arc = store.io_open_dataset("admin/test", "main").await.unwrap();
        let mut ds = ds_arc.write().await;

        // Second call: incremental (append), produces a new version.
        let v2 = io_ensure_vector_index(&mut ds, &config, false).await.unwrap();
        assert!(v2 > v1, "optimize should produce a new version");
    }

    /// P2.5-3: ANN recall guard — ANN results match flat-KNN within tolerance.
    /// This is the key correctness test: approximate results must be close to exact.
    ///
    /// Uses nprobes = num_partitions (scan all partitions) + refine_factor to
    /// maximise recall. With SQ (less lossy than PQ), recall should be higher;
    /// 90% recall@10 is a conservative lower bound for correctness.
    #[tokio::test]
    async fn ann_recall_within_tolerance() {
        let dim = 128;
        let corpus_size = 500;
        let k = 10;
        let recall_threshold = 0.9; // 90% recall with full-partition scan + refine.

        let (mut store, _tmp) = make_test_store(dim);
        // For the recall test, we want to scan ALL partitions (no IVF approximation).
        // With adaptive partitions, sqrt(500)/4 ≈ 5, floored to 16 partitions.
        let corpus_rows = corpus_size;
        let num_partitions = VectorIndexConfig::default_for_dim(dim)
            .recommended_num_partitions(corpus_rows);
        let config = VectorIndexConfig {
            num_partitions, // Adaptive: sqrt(corpus_size) for the test corpus.
            m: 30,           // High connectivity for recall.
            ef_construction: 200,
            nprobes: num_partitions, // Scan ALL partitions — no approximation in partition selection.
            refine_factor: Some(20), // Re-rank top 20x candidates with full vectors.
        };
        // Set the store's search params to match.
        store.set_vector_index_config(config.clone());

        // Build corpus with seeded embeddings.
        let rows: Vec<ChunkRow> = (0..corpus_size)
            .map(|i| make_chunk_row(&format!("doc/{}", i), dim, i as u64 + 100))
            .collect();
        for row in &rows {
            store
                .io_upsert_chunks("admin/test", "main", &row.doc_id, std::slice::from_ref(row))
                .await
                .expect("upsert");
        }

        // Create vector index.
        let ds_arc = store.io_open_dataset("admin/test", "main").await.unwrap();
        let mut ds = ds_arc.write().await;
        io_ensure_vector_index(&mut ds, &config, false).await.unwrap();
        drop(ds);

        // Tag a commit.
        let ds_arc = store.io_open_dataset("admin/test", "main").await.unwrap();
        let ds = ds_arc.read().await;
        let current_version = ds.version().version;
        drop(ds);
        store
            .io_tag_commit("admin/test", "main", "recall_test_commit", current_version)
            .await
            .unwrap();

        // Query embedding (seeded, deterministic).
        let query_emb = seeded_normalized_embedding(dim, 9999);

        // ANN search via the store's io_search (which uses the vector index).
        let ann_query = crate::store::lance::SearchQuery {
            query_embedding: query_emb.clone(),
            query_text: String::new(),
            mode: crate::kernel::model::SearchMode::Vector,
            start: 0,
            count: k,
            doc_type_filter: Vec::new(),
            doc_id_filter: Vec::new(),
            snippet: false,
        };
        let ann_hits = store
            .io_search("admin/test", "main", "recall_test_commit", &ann_query)
            .await
            .expect("ANN search");

        // Flat-KNN baseline: compute cosine distance to all vectors, sort, take top-k.
        let flat_top_k = flat_knn_baseline(&rows, &query_emb, k);

        // Compute recall: fraction of flat-KNN top-k that appear in ANN results.
        let ann_doc_ids: std::collections::HashSet<&str> =
            ann_hits.iter().map(|h| h.doc_id.as_str()).collect();
        let recall_count = flat_top_k
            .iter()
            .filter(|doc_id| ann_doc_ids.contains(doc_id.as_str()))
            .count();
        let recall = recall_count as f64 / k as f64;

        assert!(
            recall >= recall_threshold,
            "ANN recall@{} = {:.2} is below threshold {:.2}. \
             ANN returned: {:?}, flat baseline: {:?}",
            k,
            recall,
            recall_threshold,
            ann_hits.iter().map(|h| &h.doc_id).collect::<Vec<_>>(),
            flat_top_k,
        );
    }

    /// P2.5-4: Search during indexing lag — unindexed fragments are still searched.
    #[tokio::test]
    async fn search_during_indexing_lag_returns_correct_results() {
        let dim = 128;
        let (store, _tmp) = make_test_store(dim);
        let config = make_test_config(dim);

        // Insert initial corpus and create index.
        let initial_rows: Vec<ChunkRow> = (0..300)
            .map(|i| make_chunk_row(&format!("doc/{}", i), dim, i as u64 + 1))
            .collect();
        for row in &initial_rows {
            store
                .io_upsert_chunks("admin/test", "main", &row.doc_id, std::slice::from_ref(row))
                .await
                .expect("upsert");
        }

        let ds_arc = store.io_open_dataset("admin/test", "main").await.unwrap();
        {
            let mut ds = ds_arc.write().await;
            io_ensure_vector_index(&mut ds, &config, false).await.unwrap();
        }

        // Insert NEW data AFTER index creation (these are unindexed fragments).
        let new_doc_seed: u64 = 42424242;
        let new_row = ChunkRow {
            doc_id: "doc/new_unindexed".to_owned(),
            doc_type: "TestDoc".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 50,
            embedding: seeded_normalized_embedding(dim, new_doc_seed),
            clustering_embedding: seeded_normalized_embedding(dim, new_doc_seed + 10000),
            content: "brand new unindexed document".to_owned(),
        };
        store
            .io_upsert_chunks("admin/test", "main", "doc/new_unindexed", std::slice::from_ref(&new_row))
            .await
            .expect("upsert new");

        // Tag commit to the CURRENT version (which has unindexed fragments).
        let ds_arc = store.io_open_dataset("admin/test", "main").await.unwrap();
        let ds = ds_arc.read().await;
        let v = ds.version().version;
        drop(ds);
        store
            .io_tag_commit("admin/test", "main", "lag_commit", v)
            .await
            .unwrap();

        // Search using the new doc's own embedding as query (should find itself).
        let query = crate::store::lance::SearchQuery {
            query_embedding: seeded_normalized_embedding(dim, new_doc_seed),
            query_text: String::new(),
            mode: crate::kernel::model::SearchMode::Vector,
            start: 0,
            count: 10,
            doc_type_filter: Vec::new(),
            doc_id_filter: Vec::new(),
            snippet: false,
        };
        let hits = store
            .io_search("admin/test", "main", "lag_commit", &query)
            .await
            .expect("search during lag");

        let found_new = hits.iter().any(|h| h.doc_id == "doc/new_unindexed");
        assert!(
            found_new,
            "unindexed fragment must still be searchable (Lance flat-searches unindexed). \
             Got hits: {:?}",
            hits.iter().map(|h| &h.doc_id).collect::<Vec<_>>()
        );
    }

    /// P2.5-5: pending_index_fragments in Statistics reflects unindexed data.
    #[tokio::test]
    async fn statistics_pending_index_fragments() {
        let dim = 128;
        let (store, _tmp) = make_test_store(dim);
        let config = make_test_config(dim);

        // Insert data.
        let rows: Vec<ChunkRow> = (0..300)
            .map(|i| make_chunk_row(&format!("doc/{}", i), dim, i as u64 + 1))
            .collect();
        for row in &rows {
            store
                .io_upsert_chunks("admin/test", "main", &row.doc_id, std::slice::from_ref(row))
                .await
                .expect("upsert");
        }

        // Create index (should cover all current fragments).
        let ds_arc = store.io_open_dataset("admin/test", "main").await.unwrap();
        {
            let mut ds = ds_arc.write().await;
            io_ensure_vector_index(&mut ds, &config, false).await.unwrap();
        }

        // After index creation: pending should be 0 (or very low).
        // Note: We don't assert == 0 here because the index creation itself may
        // produce a new version with metadata changes. We verify the full cycle below.
        drop(ds_arc);

        // Insert more data (creates unindexed fragments).
        for i in 300..310 {
            let row = make_chunk_row(&format!("doc/{}", i), dim, i as u64 + 1);
            store
                .io_upsert_chunks("admin/test", "main", &row.doc_id, std::slice::from_ref(&row))
                .await
                .expect("upsert new");
        }

        let ds_arc = store.io_open_dataset("admin/test", "main").await.unwrap();
        let ds = ds_arc.read().await;
        let pending_with_new = count_unindexed_fragments(&ds).await.unwrap();
        drop(ds);

        assert!(
            pending_with_new > 0,
            "should have pending fragments after new inserts without optimize"
        );

        // After optimize: pending should return to 0.
        let ds_arc = store.io_open_dataset("admin/test", "main").await.unwrap();
        {
            let mut ds = ds_arc.write().await;
            io_ensure_vector_index(&mut ds, &config, false).await.unwrap();
        }
        let ds_arc = store.io_open_dataset("admin/test", "main").await.unwrap();
        let ds = ds_arc.read().await;
        let pending_after_optimize = count_unindexed_fragments(&ds).await.unwrap();
        assert_eq!(
            pending_after_optimize, 0,
            "pending should be 0 after optimize"
        );
    }

    /// Sibling to statistics_pending_index_fragments — verifies count_unindexed_rows
    /// returns both fragment count and total rows, and both return to 0 after optimize.
    #[tokio::test]
    async fn statistics_pending_index_documents() {
        let dim = 128;
        let (store, _tmp) = make_test_store(dim);
        let config = make_test_config(dim);

        let rows: Vec<ChunkRow> = (0..300)
            .map(|i| make_chunk_row(&format!("doc/{}", i), dim, i as u64 + 1))
            .collect();
        for row in &rows {
            store
                .io_upsert_chunks("admin/test", "main", &row.doc_id, std::slice::from_ref(row))
                .await
                .expect("upsert");
        }

        let ds_arc = store.io_open_dataset("admin/test", "main").await.unwrap();
        {
            let mut ds = ds_arc.write().await;
            io_ensure_vector_index(&mut ds, &config, false).await.unwrap();
        }
        drop(ds_arc);

        for i in 300..310 {
            let row = make_chunk_row(&format!("doc/{}", i), dim, i as u64 + 1);
            store
                .io_upsert_chunks("admin/test", "main", &row.doc_id, std::slice::from_ref(&row))
                .await
                .expect("upsert new");
        }

        let ds_arc = store.io_open_dataset("admin/test", "main").await.unwrap();
        let ds = ds_arc.read().await;
        let (pending_frags, pending_rows) = count_unindexed_rows(&ds).await.unwrap();
        drop(ds);

        assert!(
            pending_frags > 0,
            "should have pending fragments after new inserts without optimize"
        );
        assert!(
            pending_rows > 0,
            "should have pending rows after new inserts without optimize"
        );

        let ds_arc = store.io_open_dataset("admin/test", "main").await.unwrap();
        {
            let mut ds = ds_arc.write().await;
            io_ensure_vector_index(&mut ds, &config, false).await.unwrap();
        }
        let ds_arc = store.io_open_dataset("admin/test", "main").await.unwrap();
        let ds = ds_arc.read().await;
        let (pending_frags_after, pending_rows_after) = count_unindexed_rows(&ds).await.unwrap();
        assert_eq!(pending_frags_after, 0, "pending fragments should be 0 after optimize");
        assert_eq!(pending_rows_after, 0, "pending rows should be 0 after optimize");
    }

    /// P2.5-6: Vector index is available at the tagged commit version.
    /// (Confirms index travels with the version manifest.)
    #[tokio::test]
    async fn vector_index_available_at_tagged_version() {
        let dim = 128;
        let (store, _tmp) = make_test_store(dim);
        let config = make_test_config(dim);

        // Insert data + create index.
        let rows: Vec<ChunkRow> = (0..300)
            .map(|i| make_chunk_row(&format!("doc/{}", i), dim, i as u64 + 1))
            .collect();
        for row in &rows {
            store
                .io_upsert_chunks("admin/test", "main", &row.doc_id, std::slice::from_ref(row))
                .await
                .expect("upsert");
        }
        let ds_arc = store.io_open_dataset("admin/test", "main").await.unwrap();
        let version_with_index;
        {
            let mut ds = ds_arc.write().await;
            version_with_index = io_ensure_vector_index(&mut ds, &config, false).await.unwrap();
        }

        // Tag commit to the version that includes the index.
        store
            .io_tag_commit("admin/test", "main", "commit_with_idx", version_with_index)
            .await
            .unwrap();

        // Checkout at that version and verify the index is there.
        let ds_arc = store.io_open_dataset("admin/test", "main").await.unwrap();
        let ds = ds_arc.read().await;
        let snapshot = ds
            .checkout_version(version_with_index)
            .await
            .expect("checkout");
        let indices = snapshot.load_indices().await.unwrap();
        assert!(
            indices.iter().any(|i| i.name == VECTOR_INDEX_NAME),
            "vector index must be present at the tagged version"
        );
    }

    /// Flat KNN baseline for recall comparison (test-only, O(n)).
    /// Returns the top-k doc_ids sorted by cosine distance (ascending).
    fn flat_knn_baseline(rows: &[ChunkRow], query: &[f32], k: usize) -> Vec<String> {
        let mut distances: Vec<(f32, &str)> = rows
            .iter()
            .map(|row| {
                let dist = cosine_distance(query, &row.embedding);
                (dist, row.doc_id.as_str())
            })
            .collect();
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        distances.into_iter().take(k).map(|(_, id)| id.to_owned()).collect()
    }

    /// Pure cosine distance between two L2-normalised vectors.
    /// = 1 - dot_product (since both are unit vectors).
    fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        1.0 - dot
    }
}
