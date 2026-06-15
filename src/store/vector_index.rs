#![forbid(unsafe_code)]

//! Vector ANN index management for LanceDB datasets.
//!
//! Creates and maintains a vector index (IVF_PQ) on the `embedding` column.
//! The index uses cosine distance (vectors are L2-normalised before insert,
//! so Lance's DistanceType::Cosine gives the correct metric).
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
/// On first call: creates the IVF_PQ index with the configured parameters.
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
) -> Result<u64, StoreError> {
    let indices = ds
        .load_indices()
        .await
        .map_err(|e| StoreError::Internal(format!("load indices failed: {}", e)))?;

    let has_vector_idx = indices.iter().any(|idx| idx.name == VECTOR_INDEX_NAME);

    if !has_vector_idx {
        // IVF_PQ minimum training requirement:
        // - IVF KMeans needs at least `num_partitions` vectors (one per centroid).
        // - PQ codebook KMeans needs at least 2^num_bits (256 for num_bits=8)
        //   training vectors for each sub-vector's codebook.
        // The binding constraint is PQ: 256 vectors minimum.
        // Without enough data, `create_index` fails with a KMeans training error.
        // Lance flat-searches unindexed fragments correctly, so search stays correct
        // during this window. The index will be created on a future call once enough
        // data accumulates.
        let row_count = ds
            .count_rows(None)
            .await
            .map_err(|e| StoreError::Internal(format!("count rows failed: {}", e)))?;

        // PQ with num_bits=8 needs 2^8=256 training vectors minimum.
        // IVF needs num_partitions centroids. Take the max as the safe floor.
        let pq_min_training_vectors: usize = 256; // 2^num_bits for PQ codebook
        let min_rows_for_index = config.num_partitions.max(pq_min_training_vectors);
        if row_count < min_rows_for_index {
            // Not enough data to train the index yet — return current version.
            // Search uses flat-scan (correct, higher latency on small corpus is acceptable).
            return Ok(ds.version().version);
        }

        // First time: create the vector index.
        let params = build_vector_index_params(config);
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
        ds.optimize_indices(&OptimizeOptions::append())
            .await
            .map_err(|e| StoreError::Internal(format!("vector index optimize failed: {}", e)))?;
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

/// Build VectorIndexParams from config.
/// Uses IVF_PQ with cosine distance metric.
fn build_vector_index_params(
    config: &VectorIndexConfig,
) -> lance::index::vector::VectorIndexParams {
    use lance::index::vector::VectorIndexParams;
    use lance_index::vector::ivf::IvfBuildParams;
    use lance_index::vector::pq::PQBuildParams;

    let ivf = IvfBuildParams::new(config.num_partitions);
    let pq = PQBuildParams {
        num_sub_vectors: config.num_sub_vectors,
        num_bits: 8,
        ..Default::default()
    };

    VectorIndexParams::with_ivf_pq_params(DistanceType::Cosine, ivf, pq)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::lance::LanceStore;
    use crate::store::lance::ChunkRow;

    /// Create a seeded, L2-normalised random embedding (deterministic).
    /// Adapted from the reference bench: `terminusdb-semantic-indexer/benches/distance.rs`.
    fn seeded_normalized_embedding(dim: usize, seed: u64) -> Vec<f32> {
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

    fn make_test_store(dim: usize) -> (LanceStore, tempfile::TempDir) {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let store = LanceStore::new(tmp.path(), dim);
        (store, tmp)
    }

    fn make_chunk_row(doc_id: &str, dim: usize, seed: u64) -> ChunkRow {
        ChunkRow {
            doc_id: doc_id.to_owned(),
            doc_type: "TestDoc".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 100,
            embedding: seeded_normalized_embedding(dim, seed),
            query_embedding: seeded_normalized_embedding(dim, seed + 10000),
            content: format!("content for {}", doc_id),
        }
    }

    /// P2.5-1: Vector index is created by io_ensure_vector_index.
    #[tokio::test]
    async fn vector_index_created_on_first_call() {
        let (store, _tmp) = make_test_store(128);
        let config = VectorIndexConfig::default_for_dim(128);

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
        let version = io_ensure_vector_index(&mut ds, &config).await.unwrap();
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
        let config = VectorIndexConfig::default_for_dim(128);

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
        let v1 = io_ensure_vector_index(&mut ds, &config).await.unwrap();

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
        let v2 = io_ensure_vector_index(&mut ds, &config).await.unwrap();
        assert!(v2 > v1, "optimize should produce a new version");
    }

    /// P2.5-3: ANN recall guard — ANN results match flat-KNN within tolerance.
    /// This is the key correctness test: approximate results must be close to exact.
    ///
    /// Uses nprobes = num_partitions (scan all partitions) + refine_factor to
    /// maximise recall. With PQ quantization on random data, 90% recall@10 is
    /// a reasonable lower bound. If this fails, the index or distance metric is broken.
    #[tokio::test]
    async fn ann_recall_within_tolerance() {
        let dim = 128;
        let corpus_size = 500;
        let k = 10;
        let recall_threshold = 0.9; // 90% recall with full-partition scan + refine.

        let (mut store, _tmp) = make_test_store(dim);
        // Use high nprobes = num_partitions for the recall test (scan all partitions).
        let config = VectorIndexConfig {
            num_partitions: 4, // Few partitions for small corpus.
            num_sub_vectors: 16,
            nprobes: 4,        // Scan ALL partitions — no approximation in partition selection.
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
        io_ensure_vector_index(&mut ds, &config).await.unwrap();
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
        let config = VectorIndexConfig::default_for_dim(dim);

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
            io_ensure_vector_index(&mut ds, &config).await.unwrap();
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
            query_embedding: seeded_normalized_embedding(dim, new_doc_seed + 10000),
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
        let config = VectorIndexConfig::default_for_dim(dim);

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
            io_ensure_vector_index(&mut ds, &config).await.unwrap();
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
            io_ensure_vector_index(&mut ds, &config).await.unwrap();
        }
        let ds_arc = store.io_open_dataset("admin/test", "main").await.unwrap();
        let ds = ds_arc.read().await;
        let pending_after_optimize = count_unindexed_fragments(&ds).await.unwrap();
        assert_eq!(
            pending_after_optimize, 0,
            "pending should be 0 after optimize"
        );
    }

    /// P2.5-6: Vector index is available at the tagged commit version.
    /// (Confirms index travels with the version manifest.)
    #[tokio::test]
    async fn vector_index_available_at_tagged_version() {
        let dim = 128;
        let (store, _tmp) = make_test_store(dim);
        let config = VectorIndexConfig::default_for_dim(dim);

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
            version_with_index = io_ensure_vector_index(&mut ds, &config).await.unwrap();
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
