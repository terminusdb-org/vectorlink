//! Search: vector, FTS, hybrid, snapshot resolution, doc-chunk lookup.

use arrow_array::{
    Array, FixedSizeListArray, Float32Array, Int32Array, RecordBatch, StringArray,
};
use futures::TryStreamExt;
use lance::dataset::Dataset;
use lance_index::scalar::FullTextSearchQuery;
use lance_linalg::distance::DistanceType;

use crate::kernel::error::StoreError;
use crate::kernel::model::SearchMode;

use super::{ChunkHit, DistanceKind, LanceStore, SearchQuery, MAIN_BRANCH};

impl LanceStore {
    /// Vector/FTS/hybrid search over chunk rows at a given version.
    /// Returns chunk-level hits (caller dedups to documents).
    /// INVARIANT: searches the snapshot at the commit's tagged version, NOT the latest.
    pub async fn io_search(
        &self,
        domain: &str,
        branch: &str,
        commit: &str,
        query: &SearchQuery,
    ) -> Result<Vec<ChunkHit>, StoreError> {
        // Resolve `commit` to its versioned read-only snapshot off the CACHED
        // domain handle (shared object_store + session — see
        // `io_snapshot_from_cache`). Reusing the cache instead of `Dataset::open`
        // per search bounds the open file-descriptor count under load (BUG-FD24);
        // writers refresh the cache on mutation so freshly-tagged commits are still
        // visible (the 409 / search-at-just-indexed-commit invariant).
        let snapshot = self.io_snapshot_from_cache(domain, branch, commit).await?;

        // Build the search based on mode — always against the versioned snapshot.
        let hits = match query.mode {
            SearchMode::Vector => self.vector_search(&snapshot, query).await?,
            SearchMode::Fts => self.fts_search(&snapshot, query).await?,
            SearchMode::Hybrid => self.hybrid_search(&snapshot, query).await?,
        };

        Ok(hits)
    }

    /// Vector search using Lance's `nearest()` with `DistanceType::Cosine`.
    ///
    /// When a vector ANN index exists on the `embedding` column (created by
    /// `io_ensure_vector_index`), Lance routes the query through the index for
    /// sub-linear performance. Unindexed fragments are flat-searched alongside
    /// the index (correct results, higher latency on the unindexed portion).
    ///
    /// Without an index (legacy path or pre-index versions), this degrades to
    /// O(n) flat KNN — functionally correct but not suitable for large corpora.
    ///
    /// Distance: embeddings are L2-normalised before insert, so cosine distance
    /// is in [0, 2]. The `DistanceType::Cosine` metric is set explicitly on the
    /// scanner to ensure correct distance computation regardless of index type.
    async fn vector_search(
        &self,
        ds: &Dataset,
        query: &SearchQuery,
    ) -> Result<Vec<ChunkHit>, StoreError> {
        // Over-fetch to allow for dedup (multiple chunks per doc).
        let k = (query.start + query.count) * 3;

        let mut scanner = ds.scan();
        scanner
            .nearest("embedding", &Float32Array::from(query.query_embedding.clone()), k)
            .map_err(|e| StoreError::Internal(format!("vector search setup failed: {}", e)))?;
        scanner.distance_metric(DistanceType::Cosine);
        // ANN tuning: nprobes controls how many IVF partitions are scanned;
        // refine re-ranks the top candidates with full-precision vectors.
        scanner.nprobes(self.vector_index_config.nprobes);
        if let Some(rf) = self.vector_index_config.refine_factor {
            scanner.refine(rf);
        }

        // Apply filters if present.
        if !query.doc_type_filter.is_empty() || !query.doc_id_filter.is_empty() {
            let filter = build_filter_expression(&query.doc_type_filter, &query.doc_id_filter);
            if !filter.is_empty() {
                scanner.filter(&filter)
                    .map_err(|e| StoreError::Internal(format!("filter failed: {}", e)))?;
            }
        }

        let batches: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .map_err(|e| StoreError::Internal(format!("scan stream failed: {}", e)))?
            .try_collect()
            .await
            .map_err(|e| StoreError::Internal(format!("batch collect failed: {}", e)))?;

        // KNN search via nearest() — a `_distance` column is REQUIRED; fail loud
        // if absent rather than corrupt ranking with 0.0 (#E).
        batches_to_vector_hits(&batches, true)
    }

    /// Full-text search.
    /// Returns empty results if no FTS (INVERTED) index exists on the dataset.
    async fn fts_search(
        &self,
        ds: &Dataset,
        query: &SearchQuery,
    ) -> Result<Vec<ChunkHit>, StoreError> {
        let k = (query.start + query.count) * 3;

        let mut scanner = ds.scan();
        // Lance FTS via full_text_search on the "content" column.
        scanner
            .full_text_search(FullTextSearchQuery::new(query.query_text.clone()))
            .map_err(|e| StoreError::Internal(format!("FTS search setup failed: {}", e)))?;
        scanner
            .limit(Some(k as i64), None)
            .map_err(|e| StoreError::Internal(format!("limit failed: {}", e)))?;

        // Apply filters if present.
        if !query.doc_type_filter.is_empty() || !query.doc_id_filter.is_empty() {
            let filter = build_filter_expression(&query.doc_type_filter, &query.doc_id_filter);
            if !filter.is_empty() {
                scanner
                    .filter(&filter)
                    .map_err(|e| StoreError::Internal(format!("filter failed: {}", e)))?;
            }
        }

        let stream = match scanner.try_into_stream().await {
            Ok(s) => s,
            Err(e) => {
                let msg = e.to_string();
                // WHY: FTS is called from hybrid_search, which fuses vector + FTS.
                // At historical versions (checkout_version), the INVERTED index may
                // not yet exist (created after that version was tagged). Returning
                // empty degrades hybrid to vector-only — still useful and correct.
                // INVARIANT: the vector side of hybrid always runs (vector_search is
                // called independently); hybrid is never empty solely because FTS is.
                // CONSEQUENCE: search returns vector-only results at versions that
                // predate FTS index creation. No data loss; ranking is less rich.
                if msg.contains("INVERTED index") || msg.contains("full text search") {
                    return Ok(Vec::new());
                }
                return Err(StoreError::Internal(format!("FTS stream failed: {}", msg)));
            }
        };

        match stream.try_collect::<Vec<RecordBatch>>().await {
            Ok(batches) => Ok(batches_to_fts_hits(&batches)),
            Err(e) => {
                let msg = e.to_string();
                // WHY/INVARIANT/CONSEQUENCE: same as above — historical snapshot
                // may lack the INVERTED index; hybrid degrades to vector-only.
                if msg.contains("INVERTED index") || msg.contains("full text search") {
                    return Ok(Vec::new());
                }
                Err(StoreError::Internal(format!("FTS collect failed: {}", msg)))
            }
        }
    }

    /// Hybrid search: Reciprocal Rank Fusion (RRF) over vector + FTS ranked lists.
    /// Deterministic given deterministic inputs.
    /// RRF score = sum over lists of 1/(k + rank_in_list) where k=60 (standard).
    async fn hybrid_search(
        &self,
        ds: &Dataset,
        query: &SearchQuery,
    ) -> Result<Vec<ChunkHit>, StoreError> {
        let vector_hits = self.vector_search(ds, query).await?;
        let fts_hits = self.fts_search(ds, query).await?;
        Ok(rrf_merge(vector_hits, fts_hits))
    }

    /// Look up a document's chunks by doc_id at the head of `branch` (indexed
    /// lookup for /similar).
    /// INVARIANT: uses a filter on the doc_id column (indexed), NOT a full scan.
    /// Layout A: reads the BRANCH head (checked out), not the cached main handle —
    /// so a lookup on a feature branch sees the branch's data, not main's.
    ///
    /// FD (BUG-FD24): BOTH the main and the non-main paths read off the CACHED
    /// domain handle. The non-main path clones the cached `Dataset` (shares its
    /// object_store + session via `Arc`) and `checkout_branch` off the clone —
    /// `checkout_by_ref` REUSES that object_store + session, so it opens NO fresh
    /// file descriptors. It must NOT `Dataset::open` per call (which would spin up
    /// a new object_store + session and leak the index reader FDs under repeated
    /// `/similar`/lookup load on a feature branch — the same leak class as the
    /// `io_search` non-main path, which uses this exact pattern).
    pub async fn io_lookup_doc_chunks(
        &self,
        domain: &str,
        branch: &str,
        doc_id: &str,
    ) -> Result<Vec<ChunkHit>, StoreError> {
        // READ-ONLY (BLOCKER-2): never resurrect a deleted domain on lookup. An
        // absent dataset has no chunks for any doc.
        let cached = match self.io_open_dataset_readonly(domain).await? {
            Some(ds) => ds,
            None => return Ok(Vec::new()),
        };

        if branch == MAIN_BRANCH {
            // Main = the cached default-branch handle; scan under the read guard.
            let guard = cached.read().await;
            return Self::scan_doc_chunks(&guard, doc_id).await;
        }

        // Non-main: clone the cached handle (shares object_store + session via Arc)
        // and check out the branch off the clone — NO fresh `Dataset::open`, so no
        // FD pressure (mirrors `io_snapshot_from_cache`'s non-main path).
        let base = {
            let guard = cached.read().await;
            guard.clone()
        };
        let branch_ds = base.checkout_branch(branch).await.map_err(|e| {
            StoreError::Internal(format!("lookup checkout '{}' failed: {}", branch, e))
        })?;

        Self::scan_doc_chunks(&branch_ds, doc_id).await
    }

    /// Scan a dataset handle for all chunks of `doc_id` (filter on indexed column).
    async fn scan_doc_chunks(ds: &Dataset, doc_id: &str) -> Result<Vec<ChunkHit>, StoreError> {
        let filter = format!("doc_id = '{}'", doc_id.replace('\'', "''"));
        let mut scanner = ds.scan();
        scanner
            .filter(&filter)
            .map_err(|e| StoreError::Internal(format!("lookup filter failed: {}", e)))?;

        let batches: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .map_err(|e| StoreError::Internal(format!("lookup stream failed: {}", e)))?
            .try_collect()
            .await
            .map_err(|e| StoreError::Internal(format!("lookup collect failed: {}", e)))?;

        // Plain filter scan (no KNN) — no `_distance` column by design; the
        // caller (/similar) re-embeds and ranks separately.
        batches_to_vector_hits(&batches, false)
    }

    /// Resolve `commit` on `branch` to its versioned, read-only snapshot off the
    /// CACHED domain handle — the single shared resolution path for `/search`,
    /// `/similar` and `/duplicates` (so commit→snapshot resolution and the
    /// not-indexed behaviour are CONSISTENT across all three).
    ///
    /// CACHED, NOT FRESH-PER-CALL (BUG-FD24): we clone the cached domain `Dataset`
    /// (cheap — `Dataset::clone` shares the object_store + session via `Arc`) and
    /// `checkout_branch`/`checkout_version` off that clone. `checkout_by_ref`
    /// REUSES the handle's object_store and session (it does not allocate a new
    /// one), so resolving a snapshot from the cache opens NO new file descriptors —
    /// unlike `Dataset::open`, which spins up a fresh object_store + session
    /// holding its own file readers per call (the FD-exhaustion source under load).
    ///
    /// TAG VISIBILITY (the reason fresh-open previously existed): the cached handle
    /// is refreshed by every mutation that changes tags/versions (`io_tag_commit`,
    /// `io_upsert_chunks`, `io_delete_doc`, optimize, assign) via
    /// `io_refresh_cached_dataset`, so a commit tagged by the worker is visible to
    /// a subsequent search/resolve. This preserves the 409 / search-at-just-
    /// indexed-commit invariant WITHOUT opening fresh every query.
    ///
    /// READ-ONLY (BLOCKER-2): an absent dataset is "not indexed", never resurrected
    /// (`io_open_dataset_readonly` does not auto-create).
    pub(super) async fn io_snapshot_from_cache(
        &self,
        domain: &str,
        branch: &str,
        commit: &str,
    ) -> Result<Dataset, StoreError> {
        let cached = match self.io_open_dataset_readonly(domain).await? {
            Some(ds) => ds,
            None => {
                return Err(StoreError::Internal(format!(
                    "commit not indexed: domain '{}' has no dataset",
                    domain
                )));
            }
        };

        // Clone the cached Dataset (shares object_store + session via Arc) so the
        // checkouts below run without holding the cache lock for the whole search.
        let base = {
            let guard = cached.read().await;
            guard.clone()
        };

        let owned_branch_ds;
        let base: &Dataset = if branch == MAIN_BRANCH {
            &base
        } else {
            owned_branch_ds = base.checkout_branch(branch).await.map_err(|e| {
                StoreError::Internal(format!("checkout '{}' for snapshot failed: {}", branch, e))
            })?;
            &owned_branch_ds
        };

        let tag = crate::layeridx::encode_commit_tag(commit);
        let version = base
            .tags()
            .get_version(&tag)
            .await
            .map_err(|e| StoreError::Internal(format!("commit not indexed: {}", e)))?;

        base.checkout_version(version).await.map_err(|e| {
            StoreError::Internal(format!("checkout version {} failed: {}", version, e))
        })
    }

    /// Resolve `commit` on `branch` to its versioned, read-only snapshot.
    /// Thin alias over `io_snapshot_from_cache` retained for `/duplicates` and
    /// `/similar` call sites.
    pub(super) async fn io_open_snapshot(
        &self,
        domain: &str,
        branch: &str,
        commit: &str,
    ) -> Result<Dataset, StoreError> {
        self.io_snapshot_from_cache(domain, branch, commit).await
    }
}

/// Build a Lance SQL filter combining doc_type and doc_id IN-lists.
/// Used by search (vector/fts filters) and resolve (set/target population filters).
pub(super) fn build_filter_expression(doc_types: &[String], doc_ids: &[String]) -> String {
    let mut parts = Vec::new();

    if !doc_types.is_empty() {
        let values: Vec<String> = doc_types
            .iter()
            .map(|t| format!("'{}'", t.replace('\'', "''")))
            .collect();
        parts.push(format!("doc_type IN ({})", values.join(", ")));
    }

    if !doc_ids.is_empty() {
        let values: Vec<String> = doc_ids
            .iter()
            .map(|id| format!("'{}'", id.replace('\'', "''")))
            .collect();
        parts.push(format!("doc_id IN ({})", values.join(", ")));
    }

    // AND across the two sets (OR within each via IN).
    parts.join(" AND ")
}

/// Extract one row's embedding from a `FixedSizeListArray` of Float32 values.
///
/// FAIL-LOUD: a null embedding or a non-Float32 inner array is a real
/// schema/corruption error (every stored vector is a non-null Float32 list at
/// insert time), never silently coerced to an empty/zero vector — that would
/// corrupt downstream ANN ranking. `doc_id` is included for diagnosability.
pub(super) fn extract_embedding_row(
    embeddings: &FixedSizeListArray,
    row: usize,
    doc_id: &str,
) -> Result<Vec<f32>, StoreError> {
    if embeddings.is_null(row) {
        return Err(StoreError::Internal(format!(
            "embedding extraction: null embedding for doc {}",
            doc_id
        )));
    }
    let values = embeddings.value(row);
    let floats = values
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| {
            StoreError::Internal(format!(
                "embedding extraction: embedding is not Float32 for doc {}",
                doc_id
            ))
        })?;
    Ok(floats.values().to_vec())
}

/// Extract ChunkHit records from RecordBatches.
///
/// `require_distance` distinguishes the two read shapes:
///  - A vector `nearest()` search MUST carry a `_distance` column; a missing one
///    means the scanner did not run as a KNN query — defaulting every hit to 0.0
///    would silently corrupt ranking (all-equal distances → arbitrary order),
///    so we FAIL LOUD with `StoreError::Internal` (#E).
///  - A plain filter scan (e.g. doc-chunk lookup for `/similar`) has NO
///    `_distance` column by design and does NOT rank — `require_distance=false`
///    yields a neutral 0.0 there, which is correct (the caller re-embeds and
///    ranks separately).
pub(super) fn batches_to_vector_hits(
    batches: &[RecordBatch],
    require_distance: bool,
) -> Result<Vec<ChunkHit>, StoreError> {
    let mut hits = Vec::new();

    for batch in batches {
        let n = batch.num_rows();
        let doc_ids = batch
            .column_by_name("doc_id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let chunk_indexes = batch
            .column_by_name("chunk_index")
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
        let chunk_counts = batch
            .column_by_name("chunk_count")
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
        let chunk_token_starts = batch
            .column_by_name("chunk_token_start")
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
        let doc_token_lens = batch
            .column_by_name("doc_token_len")
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
        let contents = batch
            .column_by_name("content")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let distances = batch
            .column_by_name("_distance")
            .and_then(|c| c.as_any().downcast_ref::<Float32Array>());
        // The stored embedding is present only on the plain doc-chunk lookup scan
        // (which selects all columns); ranked vector search projects it away. When
        // present, `/similar` reuses it directly instead of re-embedding.
        let embeddings = batch
            .column_by_name("embedding")
            .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>());

        // FAIL LOUD (#E): a ranked vector search with no usable `_distance`
        // column is a real error — never default to 0.0 and corrupt ranking.
        if require_distance && distances.is_none() && n > 0 {
            return Err(StoreError::Internal(
                "vector search result is missing the `_distance` column — \
                 refusing to default distances to 0.0 (would corrupt ranking)"
                    .to_owned(),
            ));
        }

        if let (Some(ids), Some(ci), Some(cc), Some(cts), Some(dtl), Some(cnt)) =
            (doc_ids, chunk_indexes, chunk_counts, chunk_token_starts, doc_token_lens, contents)
        {
            for i in 0..n {
                // For a ranked search `distances` is guaranteed Some by the guard
                // above; for a plain scan a neutral 0.0 is correct (no ranking).
                let distance = distances.map(|d| d.value(i)).unwrap_or(0.0);
                let embedding = match embeddings {
                    Some(arr) => extract_embedding_row(arr, i, ids.value(i))?,
                    None => Vec::new(),
                };
                hits.push(ChunkHit {
                    doc_id: ids.value(i).to_owned(),
                    distance,
                    distance_kind: DistanceKind::RawCosine,
                    chunk_index: ci.value(i),
                    chunk_count: cc.value(i),
                    chunk_token_start: cts.value(i),
                    doc_token_len: dtl.value(i),
                    content: cnt.value(i).to_owned(),
                    embedding,
                });
            }
        }
    }

    Ok(hits)
}

/// Extract ChunkHit records from RecordBatches (FTS search — reads `_score`).
/// BM25 scores are converted to distances: `distance = 1/(1+score)` so 0=best.
/// Results are preserved in stream order (BM25 rank order from Lance).
fn batches_to_fts_hits(batches: &[RecordBatch]) -> Vec<ChunkHit> {
    let mut hits = Vec::new();

    for batch in batches {
        let n = batch.num_rows();
        let doc_ids = batch
            .column_by_name("doc_id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let chunk_indexes = batch
            .column_by_name("chunk_index")
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
        let chunk_counts = batch
            .column_by_name("chunk_count")
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
        let chunk_token_starts = batch
            .column_by_name("chunk_token_start")
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
        let doc_token_lens = batch
            .column_by_name("doc_token_len")
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
        let contents = batch
            .column_by_name("content")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let scores = batch
            .column_by_name("_score")
            .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

        if let (Some(ids), Some(ci), Some(cc), Some(cts), Some(dtl), Some(cnt)) =
            (doc_ids, chunk_indexes, chunk_counts, chunk_token_starts, doc_token_lens, contents)
        {
            for i in 0..n {
                // BM25 score → distance: higher score = more relevant = lower distance.
                let score = scores.map(|s| s.value(i)).unwrap_or(0.0);
                let distance = 1.0 / (1.0 + score);
                hits.push(ChunkHit {
                    doc_id: ids.value(i).to_owned(),
                    distance,
                    distance_kind: DistanceKind::Normalised,
                    chunk_index: ci.value(i),
                    chunk_count: cc.value(i),
                    chunk_token_start: cts.value(i),
                    doc_token_len: dtl.value(i),
                    content: cnt.value(i).to_owned(),
                    // FTS path ranks by `_score`; the raw vector is not projected.
                    embedding: Vec::new(),
                });
            }
        }
    }

    hits
}

/// Reciprocal Rank Fusion: merge two ranked lists into a single list.
/// RRF score = sum of 1/(k + rank) across all lists where the item appears.
/// k=60 is the standard constant. Items are keyed by (doc_id, chunk_index).
/// The output is sorted by descending RRF score (highest relevance first).
/// Distance is set to 1.0 - normalized_rrf_score (so smaller = more relevant).
pub(super) fn rrf_merge(vector_hits: Vec<ChunkHit>, fts_hits: Vec<ChunkHit>) -> Vec<ChunkHit> {
    use std::collections::HashMap;
    const K: f32 = 60.0;

    // Key: (doc_id, chunk_index) → (rrf_score, best ChunkHit)
    let mut scores: HashMap<(String, i32), (f32, ChunkHit)> = HashMap::new();

    // Score vector results by rank (already sorted by distance — best first).
    for (rank, hit) in vector_hits.into_iter().enumerate() {
        let key = (hit.doc_id.clone(), hit.chunk_index);
        let rrf_score = 1.0 / (K + rank as f32 + 1.0);
        let entry = scores.entry(key).or_insert_with(|| (0.0, hit.clone()));
        entry.0 += rrf_score;
        // Keep the hit with the better (smaller) original distance.
        if hit.distance < entry.1.distance {
            entry.1 = hit;
        }
    }

    // Score FTS results by rank (already sorted by relevance — best first).
    for (rank, hit) in fts_hits.into_iter().enumerate() {
        let key = (hit.doc_id.clone(), hit.chunk_index);
        let rrf_score = 1.0 / (K + rank as f32 + 1.0);
        let entry = scores.entry(key).or_insert_with(|| (0.0, hit.clone()));
        entry.0 += rrf_score;
        if hit.distance < entry.1.distance {
            entry.1 = hit;
        }
    }

    // Sort by RRF score descending (highest relevance first).
    let mut ranked: Vec<(f32, ChunkHit)> = scores.into_values().collect();
    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Convert RRF score into a synthetic distance (lower = better).
    // Normalise: max possible single-list score is 1/(K+1) ≈ 0.0164.
    // Max combined (rank 1 in both) is 2/(K+1) ≈ 0.0328.
    // We map to [0, 1] via: distance = 1.0 - (score / max_score).
    let max_score = 2.0 / (K + 1.0);
    ranked
        .into_iter()
        .map(|(score, mut hit)| {
            hit.distance = (1.0 - score / max_score).clamp(0.0, 1.0);
            hit.distance_kind = DistanceKind::Normalised;
            hit
        })
        .collect()
}
