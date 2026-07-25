// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 DFRNT AB

//! Search: vector, FTS, hybrid, snapshot resolution, doc-chunk lookup.

use arrow_array::{
    Array, FixedSizeListArray, Float32Array, Int32Array, RecordBatch, StringArray,
};
use futures::StreamExt as _;
use futures::TryStreamExt as _;
use lance::dataset::Dataset;
use lance::deps::datafusion::logical_expr::{col, in_list, lit, Expr};
use lance_index::scalar::FullTextSearchQuery;
use lance_linalg::distance::DistanceType;

use crate::kernel::error::StoreError;
use crate::kernel::model::SearchMode;

use super::{ChunkHit, DistanceKind, EmbeddingRecord, LanceStore, SearchQuery, SuggestQuery, SuggestResult, SuggestHit, MAIN_BRANCH};

impl LanceStore {
    /// Vector/FTS/hybrid search over chunk rows.
    /// Returns chunk-level hits (caller dedups to documents).
    /// Searches the current head of the cached dataset for responsiveness.
    /// The head is a superset of all tagged commits, so results are correct
    /// (may include in-flight rows from active indexing).
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

    /// Suggest (typeahead): FTS-only search for partial-query UI assistance.
    /// No embedding call — uses the existing FTS inverted index directly.
    /// Returns approximate total match count, completion suggestions extracted
    /// from matched content, and the first `count` document IDs.
    pub async fn io_suggest(
        &self,
        domain: &str,
        branch: &str,
        commit: &str,
        query: &SuggestQuery,
    ) -> Result<SuggestResult, StoreError> {
        let snapshot = self.io_snapshot_from_cache(domain, branch, commit).await?;

        // Over-fetch for dedup (multiple chunks per doc) + completion extraction.
        // Cap at a reasonable ceiling to bound latency for typeahead.
        let scan_limit = ((query.count + 1) * 5).min(200) as i64;

        let mut scanner = snapshot.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new(query.query_text.clone()))
            .map_err(|e| StoreError::Internal(format!("suggest FTS setup failed: {}", e)))?;
        scanner
            .limit(Some(scan_limit), None)
            .map_err(|e| StoreError::Internal(format!("suggest limit failed: {}", e)))?;

        if !query.doc_type_filter.is_empty() || !query.doc_id_filter.is_empty() {
            if let Some(expr) = build_filter_expr(&query.doc_type_filter, &query.doc_id_filter) {
                scanner.filter_expr(expr);
            }
        }

        let stream = match scanner.try_into_stream().await {
            Ok(s) => s,
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("INVERTED index") || msg.contains("full text search") {
                    // FTS index missing — fall through to substring fallback.
                    let fallback_hits = self.substring_suggest(&snapshot, query).await.unwrap_or_else(|e2| {
                        eprintln!("[suggest] substring fallback (stream err) failed: {e2}");
                        Vec::new()
                    });
                    return Ok(self.build_suggest_result(fallback_hits, query));
                }
                return Err(StoreError::Internal(format!("suggest stream failed: {}", msg)));
            }
        };

        let batches = match stream.try_collect::<Vec<RecordBatch>>().await {
            Ok(b) => b,
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("INVERTED index") || msg.contains("full text search") {
                    // FTS index missing — fall through to substring fallback.
                    let fallback_hits = self.substring_suggest(&snapshot, query).await.unwrap_or_else(|e2| {
                        eprintln!("[suggest] substring fallback (collect err) failed: {e2}");
                        Vec::new()
                    });
                    return Ok(self.build_suggest_result(fallback_hits, query));
                }
                return Err(StoreError::Internal(format!("suggest collect failed: {}", msg)));
            }
        };

        let chunk_hits = batches_to_fts_hits(&batches);

        // If FTS returned nothing (exact token match failed for a partial query),
        // fall back to a substring scan. This catches partial-word matches
        // (e.g. "elect" → "electrical") that the inverted index misses.
        let chunk_hits = if chunk_hits.is_empty() && !query.query_text.is_empty() {
            self.substring_suggest(&snapshot, query).await.unwrap_or_else(|e| {
                eprintln!("[suggest] substring fallback failed: {e}");
                Vec::new()
            })
        } else {
            chunk_hits
        };

        Ok(self.build_suggest_result(chunk_hits, query))
    }

    /// Build the final SuggestResult from chunk hits: dedup to document level,
    /// sort by distance, extract completions from top snippets, and populate
    /// each hit with snippet content, match position, and next-words for
    /// smart compose.
    fn build_suggest_result(&self, chunk_hits: Vec<ChunkHit>, query: &SuggestQuery) -> SuggestResult {
        // Dedup to document level (best chunk per doc_id).
        let mut best_per_doc: std::collections::HashMap<String, ChunkHit> =
            std::collections::HashMap::new();
        for hit in chunk_hits {
            let entry = best_per_doc
                .entry(hit.doc_id.clone())
                .or_insert_with(|| hit.clone());
            if hit.distance < entry.distance {
                *entry = hit;
            }
        }

        let total_approx = best_per_doc.len();

        // Sort by distance (best first) and take count hits.
        let mut sorted_hits: Vec<(String, f32, String)> = best_per_doc
            .into_iter()
            .map(|(id, hit)| (id, hit.distance, hit.content))
            .collect();
        sorted_hits.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Build hits with snippet, match position, and next words.
        let query_lower = query.query_text.to_lowercase();
        let hits: Vec<SuggestHit> = sorted_hits
            .iter()
            .take(query.count)
            .map(|(id, _dist, content)| {
                let (match_start, match_end, next_words) =
                    find_match_and_next_words(content, &query_lower);
                SuggestHit {
                    id: id.clone(),
                    snippet: Some(content.clone()),
                    match_start,
                    match_end,
                    next_words,
                }
            })
            .collect();

        // Extract completions from the top matched content snippets.
        let completions = extract_completions(
            &query.query_text,
            sorted_hits.iter().take(5).map(|(_, _, content)| content.as_str()).collect(),
        );

        SuggestResult {
            total_approx,
            completions,
            hits,
        }
    }

    /// Substring fallback for suggest: scans the content column with a
    /// `LIKE '%query%'` filter when the FTS inverted index returns no results.
    /// This catches partial-word matches (e.g. "elect" → "electrical") that
    /// the inverted index misses. Uses a linear scan but is bounded by
    /// `scan_limit` and only triggered when FTS finds nothing.
    async fn substring_suggest(
        &self,
        snapshot: &Dataset,
        query: &SuggestQuery,
    ) -> Result<Vec<ChunkHit>, StoreError> {
        let scan_limit = ((query.count + 1) * 5).min(200) as i64;

        // Scan all rows and filter in Rust — Lance's SQL filter may not
        // reliably support LIKE for substring matching. This is a fallback
        // path only triggered when FTS returns nothing, so the linear scan
        // cost is acceptable.
        let query_lower = query.query_text.to_lowercase();

        let mut scanner = snapshot.scan();
        scanner
            .project(&["doc_id", "chunk_index", "chunk_count", "chunk_token_start", "doc_token_len", "content"])
            .map_err(|e| StoreError::Internal(format!("substring_suggest project failed: {}", e)))?;
        scanner
            .limit(Some(scan_limit), None)
            .map_err(|e| StoreError::Internal(format!("substring_suggest limit failed: {}", e)))?;

        let stream = scanner
            .try_into_stream()
            .await
            .map_err(|e| StoreError::Internal(format!("substring_suggest stream failed: {}", e)))?;

        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .map_err(|e| StoreError::Internal(format!("substring_suggest collect failed: {}", e)))?;

        // Extract hits from non-FTS batches, filtering by substring in Rust.
        let mut hits = Vec::new();
        for (batch_idx, batch) in batches.iter().enumerate() {
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

            if let (Some(ids), Some(ci), Some(cc), Some(cts), Some(dtl), Some(cnt)) =
                (doc_ids, chunk_indexes, chunk_counts, chunk_token_starts, doc_token_lens, contents)
            {
                for i in 0..n {
                    let content = cnt.value(i);
                    if !content.to_lowercase().contains(&query_lower) {
                        continue;
                    }
                    // Positive text match — query substring found in content.
                    // No embedding or BM25 score available; distance 0.0 reflects
                    // "this is a match" on the [0, 0.5, 1] scale.
                    let distance: f32 = 0.0;
                    hits.push(ChunkHit {
                        doc_id: ids.value(i).to_owned(),
                        distance,
                        distance_kind: DistanceKind::Normalised,
                        chunk_index: ci.value(i),
                        chunk_count: cc.value(i),
                        chunk_token_start: cts.value(i),
                        doc_token_len: dtl.value(i),
                        content: content.to_owned(),
                        embedding: Vec::new(),
                        clustering_embedding: Vec::new(),
                    });
                }
            }
        }

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
        // Checked arithmetic: prevents silent overflow if start+count is large.
        let k = query.start.saturating_add(query.count).saturating_mul(3);

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

        // Project only the columns needed for search results — skip the large
        // `embedding` and `clustering_embedding` columns (768 floats each = 3KB/row).
        // Those are only needed for /similar lookups, not ranked search.
        scanner.project(&["doc_id", "chunk_index", "chunk_count", "chunk_token_start", "doc_token_len", "content"])
            .map_err(|e| StoreError::Internal(format!("project failed: {}", e)))?;

        // Apply filters if present.
        if !query.doc_type_filter.is_empty() || !query.doc_id_filter.is_empty() {
            if let Some(expr) = build_filter_expr(&query.doc_type_filter, &query.doc_id_filter) {
                scanner.filter_expr(expr);
            }
        }

        let stream = scanner
            .try_into_stream()
            .await
            .map_err(|e| StoreError::Internal(format!("scan stream failed: {}", e)))?;

        let batches: Vec<RecordBatch> = stream
            .try_collect::<Vec<RecordBatch>>()
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
        let k = query.start.saturating_add(query.count).saturating_mul(3);

        let mut scanner = ds.scan();
        // Lance FTS via full_text_search on the "content" column.
        scanner
            .full_text_search(FullTextSearchQuery::new(query.query_text.clone()))
            .map_err(|e| StoreError::Internal(format!("FTS search setup failed: {}", e)))?;
        scanner
            .limit(Some(k as i64), None)
            .map_err(|e| StoreError::Internal(format!("limit failed: {}", e)))?;

        // Project only needed columns — skip embedding columns (3KB/row).
        scanner.project(&["doc_id", "chunk_index", "chunk_count", "chunk_token_start", "doc_token_len", "content"])
            .map_err(|e| StoreError::Internal(format!("project failed: {}", e)))?;

        // Apply filters if present.
        if !query.doc_type_filter.is_empty() || !query.doc_id_filter.is_empty() {
            if let Some(expr) = build_filter_expr(&query.doc_type_filter, &query.doc_id_filter) {
                scanner.filter_expr(expr);
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

        let batches: Vec<RecordBatch> = match stream.try_collect::<Vec<RecordBatch>>().await {
            Ok(b) => b,
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
                return Err(StoreError::Internal(format!("FTS collect failed: {}", msg)));
            }
        };

        Ok(batches_to_fts_hits(&batches))
    }

    /// Hybrid search: Reciprocal Rank Fusion (RRF) over vector + FTS ranked lists.
    /// Deterministic given deterministic inputs.
    /// RRF score = sum over lists of 1/(k + rank_in_list) where k=60 (standard).
    ///
    /// Vector and FTS run concurrently via `tokio::join!` to minimise latency.
    async fn hybrid_search(
        &self,
        ds: &Dataset,
        query: &SearchQuery,
    ) -> Result<Vec<ChunkHit>, StoreError> {
        let (vector_hits, fts_hits) = tokio::join!(
            self.vector_search(ds, query),
            self.fts_search(ds, query),
        );
        Ok(rrf_merge(vector_hits?, fts_hits?))
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
        let mut scanner = ds.scan();
        scanner.filter_expr(doc_id_eq_expr(doc_id));

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
        _branch: &str,
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

        let tag = crate::layeridx::encode_commit_tag(commit);

        // Use tags().list() to get full TagContents (including branch field).
        // Version numbers are branch-scoped, so we must check out the owning
        // branch before checkout_version when the tag lives on a rebuild branch.
        let tags = base
            .tags()
            .list()
            .await
            .map_err(|e| StoreError::Internal(format!("tag list failed: {}", e)))?;

        let tag_contents = tags
            .get(&tag)
            .ok_or_else(|| StoreError::Internal(format!("commit not indexed: tag '{}' not found", tag)))?;

        let version = tag_contents.version;

        match &tag_contents.branch {
            None => {
                // Tag on main: checkout_version from the base (main) handle.
                base.checkout_version(version).await.map_err(|e| {
                    StoreError::Internal(format!("checkout version {} failed: {}", version, e))
                })
            }
            Some(branch_name) => {
                // Tag on a non-main branch (e.g. a rebuild branch).
                // Must checkout that branch first, then the version within it.
                let ds_branch = base.checkout_branch(branch_name).await.map_err(|e| {
                    StoreError::Internal(format!(
                        "checkout branch '{}' for snapshot failed: {}", branch_name, e
                    ))
                })?;
                ds_branch.checkout_version(version).await.map_err(|e| {
                    StoreError::Internal(format!(
                        "checkout version {} on branch '{}' failed: {}", version, branch_name, e
                    ))
                })
            }
        }
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

impl LanceStore {
    /// Fetch the best (chunk_index=0 preferred) stored embedding for each doc_id
    /// in a result set. Used by the same-role exact-duplicate distance override.
    ///
    /// `column` is the embedding column name: `"embedding"` (doc-role) or
    /// `"clustering_embedding"` (clustering-role).
    ///
    /// Returns a map from doc_id → Vec<f32>. Documents not found in the snapshot
    /// (deleted between search and this lookup — edge case) are silently absent
    /// from the map (the override simply does not fire for them).
    ///
    /// FAIL-LOUD: a schema corruption (null embedding, non-Float32 values) is a
    /// real error — never silently skipped.
    pub async fn io_fetch_result_embeddings(
        &self,
        domain: &str,
        branch: &str,
        commit: &str,
        doc_ids: &[String],
        doc_types: &[String],
        column: &str,
    ) -> Result<std::collections::HashMap<String, Vec<f32>>, StoreError> {
        let snapshot = self.io_snapshot_from_cache(domain, branch, commit).await?;

        // Build a combined filter for doc_types and doc_ids.
        // When both are empty, no filter is applied — all rows are returned.
        let filter = build_filter_expr(doc_types, doc_ids);

        let mut scanner = snapshot.scan();
        if let Some(expr) = filter {
            scanner.filter_expr(expr);
        }
        scanner
            .project(&["doc_id", "chunk_index", column])
            .map_err(|e| StoreError::Internal(format!(
                "result embedding projection failed: {}", e
            )))?;

        let batches: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .map_err(|e| StoreError::Internal(format!(
                "result embedding stream failed: {}", e
            )))?
            .try_collect()
            .await
            .map_err(|e| StoreError::Internal(format!(
                "result embedding collect failed: {}", e
            )))?;

        // For each doc_id, keep only the chunk_index=0 embedding (best representative).
        // If chunk_index=0 is not present (edge case: partial data), keep the lowest.
        let mut best_per_doc: std::collections::HashMap<String, (i32, Vec<f32>)> =
            std::collections::HashMap::new();

        for batch in &batches {
            if batch.num_rows() == 0 {
                continue;
            }
            let ids = batch
                .column_by_name("doc_id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| StoreError::Internal(
                    "result embedding scan: missing `doc_id` column".to_owned(),
                ))?;
            let chunk_indexes = batch
                .column_by_name("chunk_index")
                .and_then(|c| c.as_any().downcast_ref::<Int32Array>())
                .ok_or_else(|| StoreError::Internal(
                    "result embedding scan: missing `chunk_index` column".to_owned(),
                ))?;
            let embeddings = batch
                .column_by_name(column)
                .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>())
                .ok_or_else(|| StoreError::Internal(format!(
                    "result embedding scan: missing `{}` column", column
                )))?;

            for i in 0..batch.num_rows() {
                let doc_id = ids.value(i);
                let ci = chunk_indexes.value(i);
                let embedding = extract_embedding_row(embeddings, i, doc_id)?;

                let should_update = match best_per_doc.get(doc_id) {
                    None => true,
                    Some((existing_ci, _)) => ci < *existing_ci,
                };
                if should_update {
                    best_per_doc.insert(doc_id.to_owned(), (ci, embedding));
                }
            }
        }

        Ok(best_per_doc
            .into_iter()
            .map(|(id, (_, emb))| (id, emb))
            .collect())
    }

    /// Stream embeddings for a set of doc IDs and/or doc types via a tokio mpsc
    /// channel. Each `EmbeddingRecord` yields a doc_id, the document embedding,
    /// and optionally the clustering embedding (when `store_clustering` is true).
    ///
    /// Deduplicates by doc_id keeping chunk_index=0 (same logic as
    /// `io_fetch_result_embeddings`). The count of unique doc_ids is returned
    /// alongside the receiver for progress reporting.
    ///
    /// When `store_clustering` is false, only the `embedding` column is scanned
    /// and `clustering_embedding` is `None` in every record.
    pub async fn io_stream_embeddings(
        &self,
        domain: &str,
        branch: &str,
        commit: &str,
        doc_ids: &[String],
        doc_types: &[String],
        store_clustering: bool,
    ) -> Result<(u64, tokio::sync::mpsc::Receiver<EmbeddingRecord>), StoreError> {
        let snapshot = self.io_snapshot_from_cache(domain, branch, commit).await?;

        let filter = build_filter_expr(doc_types, doc_ids);

        // Count unique doc_ids by scanning the doc_id column.
        let mut count_scanner = snapshot.scan();
        count_scanner
            .project(&["doc_id", "chunk_index"])
            .map_err(|e| StoreError::Internal(format!("count projection failed: {}", e)))?;
        if let Some(expr) = &filter {
            count_scanner.filter_expr(expr.clone());
        }
        let count_batches: Vec<RecordBatch> = count_scanner
            .try_into_stream()
            .await
            .map_err(|e| StoreError::Internal(format!("count stream failed: {}", e)))?
            .try_collect()
            .await
            .map_err(|e| StoreError::Internal(format!("count collect failed: {}", e)))?;

        let mut unique_doc_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        for batch in &count_batches {
            if batch.num_rows() == 0 {
                continue;
            }
            if let Some(ids) = batch
                .column_by_name("doc_id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            {
                for i in 0..batch.num_rows() {
                    unique_doc_ids.insert(ids.value(i).to_owned());
                }
            }
        }
        let total_count = unique_doc_ids.len() as u64;
        drop(unique_doc_ids);

        // Now stream the actual embeddings.
        let columns: Vec<&str> = if store_clustering {
            vec!["doc_id", "chunk_index", "embedding", "clustering_embedding"]
        } else {
            vec!["doc_id", "chunk_index", "embedding"]
        };

        let mut scanner = snapshot.scan();
        scanner
            .project(&columns)
            .map_err(|e| StoreError::Internal(format!("stream projection failed: {}", e)))?;
        if let Some(expr) = filter {
            scanner.filter_expr(expr);
        }

        let stream = scanner
            .try_into_stream()
            .await
            .map_err(|e| StoreError::Internal(format!("stream embeddings failed: {}", e)))?;

        let (tx, rx) = tokio::sync::mpsc::channel::<EmbeddingRecord>(128);

        // We need to track best per doc_id for dedup (chunk_index=0 preferred).
        // Since we're streaming, we collect all rows, dedup, then send.
        // This is a simplification — for true streaming dedup we'd need a state
        // machine, but the current approach still avoids holding all embeddings
        // in memory at once (we only hold the best per doc_id).
        let store_clustering_clone = store_clustering;
        tokio::spawn(async move {
            #[allow(clippy::type_complexity)]
            let mut best_per_doc: std::collections::HashMap<String, (i32, Vec<f32>, Option<Vec<f32>>)> =
                std::collections::HashMap::new();

            let mut stream = stream;
            while let Some(batch_result) = stream.next().await {
                let batch = match batch_result {
                    Ok(b) => b,
                    Err(e) => {
                        let _ = tx
                            .send(EmbeddingRecord {
                                doc_id: format!("__error__: {}", e),
                                embedding: vec![],
                                clustering_embedding: None,
                            })
                            .await;
                        return;
                    }
                };

                if batch.num_rows() == 0 {
                    continue;
                }

                let ids = match batch
                    .column_by_name("doc_id")
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                {
                    Some(ids) => ids,
                    None => continue,
                };
                let chunk_indexes = match batch
                    .column_by_name("chunk_index")
                    .and_then(|c| c.as_any().downcast_ref::<Int32Array>())
                {
                    Some(ci) => ci,
                    None => continue,
                };
                let embeddings = match batch
                    .column_by_name("embedding")
                    .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>())
                {
                    Some(emb) => emb,
                    None => continue,
                };

                let clustering_embeddings = if store_clustering_clone {
                    batch
                        .column_by_name("clustering_embedding")
                        .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>())
                } else {
                    None
                };

                for i in 0..batch.num_rows() {
                    let doc_id = ids.value(i);
                    let ci = chunk_indexes.value(i);

                    let embedding = match extract_embedding_row(embeddings, i, doc_id) {
                        Ok(emb) => emb,
                        Err(_) => continue,
                    };

                    let clustering_emb = if let Some(ce) = clustering_embeddings {
                        if ce.is_null(i) {
                            None
                        } else {
                            extract_embedding_row(ce, i, doc_id).ok()
                        }
                    } else {
                        None
                    };

                    let should_update = match best_per_doc.get(doc_id) {
                        None => true,
                        Some((existing_ci, _, _)) => ci < *existing_ci,
                    };
                    if should_update {
                        best_per_doc.insert(
                            doc_id.to_owned(),
                            (ci, embedding, clustering_emb),
                        );
                    }
                }
            }

            // Send all deduplicated records.
            for (doc_id, (_, emb, clustering_emb)) in best_per_doc {
                let _ = tx
                    .send(EmbeddingRecord {
                        doc_id,
                        embedding: emb,
                        clustering_embedding: clustering_emb,
                    })
                    .await;
            }
        });

        Ok((total_count, rx))
    }

    /// Fetch concatenated chunk content for a set of document IDs.
    /// Returns a map from doc_id to the full text (all chunks concatenated in order).
    /// Used by /candidates when include=content is requested.
    pub async fn io_fetch_doc_contents(
        &self,
        domain: &str,
        branch: &str,
        commit: &str,
        doc_ids: &[String],
    ) -> Result<std::collections::HashMap<String, String>, StoreError> {
        if doc_ids.is_empty() {
            return Ok(std::collections::HashMap::new());
        }

        let snapshot = self.io_snapshot_from_cache(domain, branch, commit).await?;

        let filter = build_filter_expr(&[], doc_ids);

        let mut scanner = snapshot.scan();
        if let Some(expr) = filter {
            scanner.filter_expr(expr);
        }
        scanner
            .project(&["doc_id", "chunk_index", "content"])
            .map_err(|e| StoreError::Internal(format!(
                "doc content projection failed: {}", e
            )))?;

        let batches: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .map_err(|e| StoreError::Internal(format!(
                "doc content stream failed: {}", e
            )))?
            .try_collect()
            .await
            .map_err(|e| StoreError::Internal(format!(
                "doc content collect failed: {}", e
            )))?;

        let mut per_doc: std::collections::HashMap<String, Vec<(i32, String)>> =
            std::collections::HashMap::new();

        for batch in &batches {
            if batch.num_rows() == 0 {
                continue;
            }
            let ids = batch
                .column_by_name("doc_id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| StoreError::Internal(
                    "doc content scan: missing `doc_id` column".to_owned(),
                ))?;
            let chunk_indexes = batch
                .column_by_name("chunk_index")
                .and_then(|c| c.as_any().downcast_ref::<Int32Array>())
                .ok_or_else(|| StoreError::Internal(
                    "doc content scan: missing `chunk_index` column".to_owned(),
                ))?;
            let contents = batch
                .column_by_name("content")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| StoreError::Internal(
                    "doc content scan: missing `content` column".to_owned(),
                ))?;

            for i in 0..batch.num_rows() {
                let doc_id = ids.value(i);
                let ci = chunk_indexes.value(i);
                let content = contents.value(i);
                per_doc
                    .entry(doc_id.to_owned())
                    .or_default()
                    .push((ci, content.to_owned()));
            }
        }

        Ok(per_doc
            .into_iter()
            .map(|(id, mut chunks)| {
                chunks.sort_by_key(|(ci, _)| *ci);
                let text = chunks
                    .iter()
                    .map(|(_, c)| c.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
                (id, text)
            })
            .collect())
    }
}

/// Build a DataFusion Expr filter combining doc_type and doc_id IN-lists.
/// Used by search (vector/fts filters) and resolve (set/target population filters).
/// Returns None when both lists are empty (no filter needed).
///
/// SECURITY: Uses pre-parsed DataFusion Expr values (lit/in_list) instead of
/// SQL string interpolation. This eliminates SQL injection via backslash-quote
/// escaping or control characters in doc_ids/doc_types.
pub(super) fn build_filter_expr(
    doc_types: &[String],
    doc_ids: &[String],
) -> Option<Expr> {
    let mut parts: Vec<Expr> = Vec::new();

    if !doc_types.is_empty() {
        let values: Vec<Expr> = doc_types.iter().map(|t| lit(t.as_str())).collect();
        parts.push(in_list(col("doc_type"), values, false));
    }

    if !doc_ids.is_empty() {
        let values: Vec<Expr> = doc_ids.iter().map(|id| lit(id.as_str())).collect();
        parts.push(in_list(col("doc_id"), values, false));
    }

    match parts.len() {
        0 => None,
        1 => Some(parts.into_iter().next().unwrap()),
        _ => Some(parts.into_iter().reduce(Expr::and).unwrap()),
    }
}

/// Build a DataFusion Expr for a single doc_id equality filter.
/// SECURITY: Uses lit() instead of SQL string interpolation.
pub(super) fn doc_id_eq_expr(doc_id: &str) -> Expr {
    col("doc_id").eq(lit(doc_id))
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
        // The stored embeddings are present only on the plain doc-chunk lookup scan
        // (which selects all columns); ranked vector search projects them away. When
        // present, `/similar` reuses them directly instead of re-embedding.
        let embeddings = batch
            .column_by_name("embedding")
            .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>());
        let clustering_embeddings = batch
            .column_by_name("clustering_embedding")
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
                let clustering_embedding = match clustering_embeddings {
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
                    clustering_embedding,
                });
            }
        }
    }

    Ok(hits)
}

/// Extract ChunkHit records from RecordBatches (FTS search — reads `_score`).
/// BM25 scores are converted to distances: `distance = 1/(1+score)` so 0=best.
/// This maps strong text matches to low distances (near 0) and weak ones toward
/// 0.5+, consistent with the [0, 0.5, 1] reference scale.
/// Results are preserved in stream order (BM25 rank order from Lance).
pub(super) fn batches_to_fts_hits(batches: &[RecordBatch]) -> Vec<ChunkHit> {
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
                    // FTS path ranks by `_score`; the raw vectors are not projected.
                    embedding: Vec::new(),
                    clustering_embedding: Vec::new(),
                });
            }
        }
    }

    hits
}

/// Find the first occurrence of `query_lower` in `content`, return the byte
/// offsets of the match and the next few words after it for smart compose.
///
/// The match is extended to the full word boundary (e.g. "elect" matching inside
/// "electrical" extends `match_end` to the end of "electrical"). This ensures
/// `next_words` starts from the word AFTER the containing word, not from the
/// partial suffix.
///
/// Returns `(match_start, match_end, next_words)`:
/// - `match_start` / `match_end`: byte offsets of the query match within the
///   snippet, for UI highlighting (e.g. Lance/React can highlight this range).
/// - `next_words`: the next words after the match, ordered by proximity. The
///   first word is the most likely next word. A tab key can cycle through them.
fn find_match_and_next_words(content: &str, query_lower: &str) -> (Option<usize>, Option<usize>, Vec<String>) {
    let content_lower = content.to_lowercase();
    let pos = match content_lower.find(query_lower) {
        Some(p) => p,
        None => return (None, None, Vec::new()),
    };

    // Extend match_end to the end of the containing word (word boundary).
    let raw_end = pos + query_lower.len();
    let after_match = &content[raw_end..];
    let extra = after_match
        .find(|c: char| c.is_whitespace() || c == '.' || c == ',' || c == ';' || c == ':')
        .unwrap_or(after_match.len());
    let match_end = raw_end + extra;

    // Extract up to 5 next words after the full containing word.
    let mut next_words = Vec::new();
    let rest = &content[match_end..];
    let mut cursor = 0usize;
    for _ in 0..5 {
        // Skip whitespace AND punctuation.
        let remaining = &rest[cursor..];
        let trimmed = remaining.trim_start_matches(|c: char| c.is_whitespace() || c == '.' || c == ',' || c == ';' || c == ':');
        let skipped = remaining.len() - trimmed.len();
        let after_ws = cursor + skipped;
        if after_ws >= rest.len() {
            break;
        }
        // Find the end of the next word.
        let word_rest = &rest[after_ws..];
        let word_end_in_rest = word_rest
            .find(|c: char| c.is_whitespace() || c == '.' || c == ',' || c == ';' || c == ':')
            .unwrap_or(word_rest.len());
        if word_end_in_rest == 0 {
            break;
        }
        let word = &rest[after_ws..after_ws + word_end_in_rest];
        let word_trimmed = word.trim_matches(|c: char| c.is_whitespace() || c == '.' || c == ',' || c == ';' || c == ':');
        if !word_trimmed.is_empty() {
            next_words.push(word_trimmed.to_owned());
        }
        cursor = after_ws + word_end_in_rest;
        if cursor >= rest.len() {
            break;
        }
    }

    (Some(pos), Some(match_end), next_words)
}

/// Extract typeahead completion suggestions from matched content snippets.
///
/// Given the user's partial query and a set of matched content strings, produces
/// a ranked list of completions by finding the query text within each snippet
/// and extracting multi-word continuations (up to 3 words after the query).
/// Completions are ranked by frequency across all snippets — the most common
/// continuation appears first. Returns at most 8 completions.
pub(super) fn extract_completions(query: &str, snippets: Vec<&str>) -> Vec<String> {
    let query_lower = query.to_lowercase();
    // Map: completion (lowercase) → (count, original-cased completion)
    let mut freq: std::collections::HashMap<String, (usize, String)> =
        std::collections::HashMap::new();

    for snippet in snippets {
        let snippet_lower = snippet.to_lowercase();
        let mut search_from = 0usize;
        // Find ALL occurrences of the query in this snippet (not just the first).
        while let Some(pos) = snippet_lower[search_from..].find(&query_lower) {
            let abs_pos = search_from + pos;
            let query_end = abs_pos + query_lower.len();

            // Extend to the full word boundary (the containing word).
            let after_match = &snippet[query_end..];
            let extra = after_match
                .find(|c: char| c.is_whitespace() || c == '.' || c == ',' || c == ';' || c == ':')
                .unwrap_or(after_match.len());
            let word_end = query_end + extra;

            // Extract up to 3 words after the containing word.
            let rest = &snippet[word_end..];
            let words: Vec<&str> = rest
                .split(|c: char| c.is_whitespace() || c == '.' || c == ',' || c == ';' || c == ':')
                .filter(|w| !w.is_empty())
                .take(3)
                .collect();

            // Build completions of increasing length: containing word, +1word, +2words, +3words.
            let containing_word = &snippet[abs_pos..word_end];
            let containing_trimmed = containing_word.trim();
            if containing_trimmed.len() > query.len() {
                let completion_lower = containing_trimmed.to_lowercase();
                let entry = freq
                    .entry(completion_lower)
                    .or_insert((0, containing_trimmed.to_owned()));
                entry.0 += 1;
            }
            let mut cumulative = containing_trimmed.to_owned();
            for w in &words {
                cumulative.push(' ');
                cumulative.push_str(w);
                let completion = cumulative.trim().to_owned();
                if completion.len() > query.len() {
                    let completion_lower = completion.to_lowercase();
                    let entry = freq
                        .entry(completion_lower)
                        .or_insert((0, completion.clone()));
                    entry.0 += 1;
                    // Keep the first-seen original casing.
                    if entry.1.is_empty() {
                        entry.1 = completion;
                    }
                }
            }

            // Move past this match to find the next occurrence.
            search_from = query_end;
        }
    }

    // Sort by frequency (descending), then by completion length (descending —
    // longer completions are more useful), then alphabetically for determinism.
    let mut ranked: Vec<(usize, String)> = freq
        .into_iter()
        .map(|(_, (count, text))| (count, text))
        .collect();
    ranked.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| b.1.len().cmp(&a.1.len()))
            .then_with(|| a.1.cmp(&b.1))
    });

    ranked.into_iter().take(8).map(|(_, text)| text).collect()
}

/// Reciprocal Rank Fusion: merge two ranked lists into a single list.
/// RRF score = sum of 1/(k + rank) across all lists where the item appears.
/// k=60 is the standard constant. Items are keyed by (doc_id, chunk_index).
/// The output is sorted by descending RRF score (highest relevance first).
///
/// Distances are NOT replaced with synthetic rank-normalised values. Each hit
/// retains its original distance: vector hits keep their RawCosine distance
/// (in [0, 2], normalised to [0, 1] by dedup), FTS-only hits keep their
/// BM25-derived distance (Normalised, in [0, 1]). For hits appearing in both
/// lists, the vector hit's distance is preferred (it has a real geometric
/// distance).
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
        // Always keep the vector hit — it has a real cosine distance.
        entry.1 = hit;
    }

    // Score FTS results by rank (already sorted by relevance — best first).
    // FTS contributes to RRF ranking. For hits also in the vector list, the
    // vector distance is preferred. For FTS-only hits, the BM25-derived
    // distance is kept.
    for (rank, hit) in fts_hits.into_iter().enumerate() {
        let key = (hit.doc_id.clone(), hit.chunk_index);
        let rrf_score = 1.0 / (K + rank as f32 + 1.0);
        let entry = scores.entry(key).or_insert_with(|| (0.0, hit.clone()));
        entry.0 += rrf_score;
        // Only use the FTS hit if there is no vector hit for this key.
        if entry.1.distance_kind != DistanceKind::RawCosine {
            entry.1 = hit;
        }
    }

    // Sort by RRF score descending (highest relevance first).
    let mut ranked: Vec<(f32, ChunkHit)> = scores.into_values().collect();
    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Return hits in RRF order, preserving each hit's original distance.
    ranked
        .into_iter()
        .map(|(_, hit)| hit)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{extract_completions, find_match_and_next_words};

    #[test]
    fn test_find_match_basic() {
        let content = "The electrical engineering department";
        let (start, end, words) = find_match_and_next_words(content, "elect");
        assert_eq!(start, Some(4));
        assert_eq!(end, Some(14)); // "electrical" (10 chars) ends at byte 14
        assert_eq!(words, vec!["engineering", "department"]);
    }

    #[test]
    fn test_find_match_no_match() {
        let content = "The mechanical engineering department";
        let (start, end, words) = find_match_and_next_words(content, "elect");
        assert_eq!(start, None);
        assert_eq!(end, None);
        assert!(words.is_empty());
    }

    #[test]
    fn test_find_match_with_punctuation() {
        let content = "Electrical, engineering and computer science.";
        let (start, end, words) = find_match_and_next_words(content, "elect");
        assert_eq!(start, Some(0));
        assert_eq!(end, Some(10)); // "Electrical" (10 chars) ends at byte 10
        assert_eq!(words, vec!["engineering", "and", "computer", "science"]);
    }

    #[test]
    fn test_find_match_at_end_of_content() {
        let content = "The department of elect";
        let (start, end, words) = find_match_and_next_words(content, "elect");
        assert_eq!(start, Some(18));
        assert_eq!(end, Some(23)); // "elect" is the full word at end
        assert!(words.is_empty());
    }

    #[test]
    fn test_find_match_case_insensitive() {
        let content = "ELECTRICAL components are here";
        let (start, end, words) = find_match_and_next_words(content, "elect");
        assert_eq!(start, Some(0));
        assert_eq!(end, Some(10)); // "ELECTRICAL" (10 chars) ends at byte 10
        assert_eq!(words, vec!["components", "are", "here"]);
    }

    #[test]
    fn test_find_match_max_five_words() {
        let content = "elect one two three four five six seven";
        let (start, end, words) = find_match_and_next_words(content, "elect");
        assert_eq!(start, Some(0));
        assert_eq!(end, Some(5)); // "elect" is the full word
        assert_eq!(words.len(), 5);
        assert_eq!(words, vec!["one", "two", "three", "four", "five"]);
    }

    #[test]
    fn test_extract_completions_basic() {
        let snippets = vec!["The electrical engineering department"];
        let completions = extract_completions("elect", snippets);
        assert!(!completions.is_empty());
        // The most frequent completion should contain "electrical" at minimum.
        assert!(completions[0].to_lowercase().contains("electrical"));
    }

    #[test]
    fn test_extract_completions_multi_word() {
        let snippets = vec!["The electrical engineering department is here"];
        let completions = extract_completions("elect", snippets);
        // Should produce completions of increasing length: electrical, electrical engineering, electrical engineering department
        assert!(completions.iter().any(|c| c.to_lowercase() == "electrical"));
        assert!(completions.iter().any(|c| c.to_lowercase() == "electrical engineering"));
        assert!(completions.iter().any(|c| c.to_lowercase() == "electrical engineering department"));
    }

    #[test]
    fn test_extract_completions_frequency_ranking() {
        // "electrical engineering" appears twice, "electrical components" once.
        let snippets = vec![
            "The electrical engineering department",
            "electrical engineering is great",
            "electrical components are cheap",
        ];
        let completions = extract_completions("elect", snippets);
        // "electrical engineering" (count=2) should rank before "electrical components" (count=1).
        let eng_idx = completions.iter().position(|c| c.to_lowercase() == "electrical engineering");
        let comp_idx = completions.iter().position(|c| c.to_lowercase() == "electrical components");
        assert!(eng_idx.is_some());
        assert!(comp_idx.is_some());
        assert!(eng_idx.unwrap() < comp_idx.unwrap());
    }

    #[test]
    fn test_extract_completions_no_match() {
        let snippets = vec!["The mechanical engineering department"];
        let completions = extract_completions("elect", snippets);
        assert!(completions.is_empty());
    }

    #[test]
    fn test_extract_completions_multiple_occurrences() {
        let snippets = vec!["electrical power and electrical systems"];
        let completions = extract_completions("elect", snippets);
        // "electrical" should appear with count=2 (two occurrences in one snippet).
        assert!(completions.iter().any(|c| c.to_lowercase() == "electrical"));
        // "electrical power" and "electrical systems" should both be present.
        assert!(completions.iter().any(|c| c.to_lowercase().contains("power")));
        assert!(completions.iter().any(|c| c.to_lowercase().contains("systems")));
    }
}
