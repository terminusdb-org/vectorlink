// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 DFRNT AB

//! Resolution/duplicates: near-duplicate grouping and cross-neighbour resolution.

use std::collections::HashMap;
use std::time::Instant;

use arrow_array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch, StringArray,
};
use futures::TryStreamExt;
use lance::dataset::Dataset;
use lance::deps::datafusion::logical_expr::{col, in_list, lit, Expr};
use lance_linalg::distance::DistanceType;

use crate::kernel::distance::normalized_cosine_from_lance;
use crate::kernel::error::StoreError;
use crate::kernel::model::DuplicateGroup;

use super::{
    DuplicateScope, LanceStore, NeighbourObservation, ResolveNeighbourMaps,
    DEFAULT_DUPLICATE_MAX_PAIRS,
};
use super::dedup::pairs_from_neighbours;
use super::search::{build_filter_expr, extract_embedding_row};

// ────────────────────────────────────────────────────────────────────────────
// Performance markers — cheap Instant::now() timing for pipeline diagnosis.
// Left in production code; guarded by RESOLVE_PERF env var (zero cost when off).
// Enable: `VECTORLINK_RESOLVE_PERF=1`
// ────────────────────────────────────────────────────────────────────────────

/// Returns true when resolve performance markers are enabled.
/// Checked once per call (branch-predicted constant after first check).
fn perf_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("VECTORLINK_RESOLVE_PERF").is_ok())
}

/// Emit a performance marker to stderr (cheap, no allocation when disabled).
macro_rules! perf_mark {
    ($start:expr, $label:expr) => {
        if perf_enabled() {
            let elapsed = $start.elapsed();
            eprintln!("[resolve-perf] {}: {:.1}ms", $label, elapsed.as_secs_f64() * 1000.0);
        }
    };
    ($start:expr, $label:expr, $($arg:tt)*) => {
        if perf_enabled() {
            let elapsed = $start.elapsed();
            eprintln!("[resolve-perf] {} ({}): {:.1}ms", $label, format!($($arg)*), elapsed.as_secs_f64() * 1000.0);
        }
    };
}

impl LanceStore {
    /// Scoped near-duplicate document groups at a commit's snapshot.
    ///
    /// ALGORITHM (robust against the k=2 starvation bug): for each indexed point
    /// (chunk vector) in the SET population, issue ONE ANN `nearest()` query whose
    /// filter EXCLUDES the point's own `doc_id` (within-set) or RESTRICTS to the
    /// TARGET population (cross-set). Because the filter removes the query point's
    /// own document, every returned row is GUARANTEED a genuine cross-document
    /// neighbour — the scan can never be starved by a multi-chunk document's own
    /// sibling chunks filling the result (the defect that returned `[]` at scale
    /// with a fixed `nearest(k=2)`). This is O(n) cheap indexed queries, NOT an
    /// O(n²) all-pairs scan. Reduction to DOCUMENT-level groups happens in the pure
    /// [`pairs_from_neighbours`] (lower-id-first, best distance per pair, snippets).
    ///
    /// UNITS: `threshold` is on the reference [0, 1] cosine scale (matches
    /// `/search`); raw Lance distances are converted via
    /// `normalized_cosine_from_lance` before comparison.
    ///
    /// BOUNDED: the SET point count is checked against `max_points` BEFORE any
    /// per-point query — a larger set is REJECTED (fail-loud) rather than silently
    /// truncated, so results are never quietly partial. Group collection is bounded
    /// by `DEFAULT_DUPLICATE_MAX_PAIRS`.
    ///
    /// FD: the snapshot is opened off the CACHED domain handle
    /// (`io_open_snapshot` → `io_snapshot_from_cache`), so the scan opens NO fresh
    /// dataset (no new object_store/session, no FD pressure — BUG-FD24). Per-point
    /// `nearest()` queries scan the already-open snapshot handle.
    #[allow(clippy::too_many_arguments)]
    pub async fn io_duplicate_groups(
        &self,
        domain: &str,
        branch: &str,
        commit: &str,
        threshold: f32,
        scope: &DuplicateScope,
        snippet: bool,
        max_points: usize,
    ) -> Result<Vec<DuplicateGroup>, StoreError> {
        let snapshot = self.io_open_snapshot(domain, branch, commit).await?;

        let set_filter = build_filter_expr(&scope.set_doc_types, &scope.set_doc_ids);

        // Bound on the SET population (the points we iterate), not the whole
        // snapshot — a filtered run over a small set inside a huge corpus is still
        // safely bounded.
        let set_point_count = if let Some(expr) = &set_filter {
            let mut scanner = snapshot.scan();
            scanner.filter_expr(expr.clone());
            scanner.count_rows().await
                .map_err(|e| StoreError::Internal(format!("count_rows for duplicates failed: {}", e)))? as usize
        } else {
            snapshot.count_rows(None)
                .await
                .map_err(|e| StoreError::Internal(format!("count_rows for duplicates failed: {}", e)))?
        };

        if set_point_count > max_points {
            return Err(StoreError::Internal(format!(
                "near-duplicate scan refused: set population has {} points, exceeds the bound of \
                 {} (would not be a safely bounded run) — narrow the `set` scope or chunk the request",
                set_point_count, max_points
            )));
        }

        let points = io_scan_points(&snapshot, set_filter, snippet).await?;
        let observations = io_collect_neighbours(&snapshot, &points, scope, snippet).await?;

        Ok(pairs_from_neighbours(
            &observations,
            threshold,
            DEFAULT_DUPLICATE_MAX_PAIRS,
        ))
    }

    /// Collect reciprocal cross-NN maps for entity resolution.
    ///
    /// Returns `(set_to_target, target_to_set)` where each map is
    /// `HashMap<doc_id, Vec<Neighbour { id, distance }>>` on the reference [0,1]
    /// cosine scale. Each entry's neighbours are sorted nearest-first and capped
    /// at `k`.
    ///
    /// ALGORITHM:
    ///  1. Open one cached snapshot (FD-safe: BUG-FD24).
    ///  2. Scan set points (doc_id + embedding).
    ///  3. Scan target points (doc_id + embedding).
    ///  4. For each set point: ANN `nearest(k)` filtered to target doc_ids.
    ///  5. For each target point: ANN `nearest(k)` filtered to set doc_ids.
    ///
    /// BOUNDED: the combined set+target point count must be <= `max_points`
    /// (fail-loud). This prevents unbounded in-process loops.
    #[allow(clippy::too_many_arguments)]
    pub async fn io_resolve_cross_neighbours(
        &self,
        domain: &str,
        branch: &str,
        commit: &str,
        scope: &DuplicateScope,
        k: usize,
        threshold_set: f32,
        threshold_target: f32,
        max_points: usize,
    ) -> Result<ResolveNeighbourMaps, StoreError> {
        let t0 = Instant::now();
        let snapshot = self.io_open_snapshot(domain, branch, commit).await?;
        perf_mark!(t0, "snapshot_open");

        // Build filters for set and target populations.
        let set_filter = build_filter_expr(&scope.set_doc_types, &scope.set_doc_ids);
        let set_filter_opt = set_filter;

        let target_filter =
            build_filter_expr(&scope.target_doc_types, &scope.target_doc_ids);
        let target_filter_opt = target_filter;

        // Scan set points (embedding + doc_id, no snippets).
        let t1 = Instant::now();
        let set_points = io_scan_points(&snapshot, set_filter_opt, false).await?;
        perf_mark!(t1, "scan_set_points", "n={}", set_points.len());

        // Scan target points. If no explicit target, target = set (within-set dedup).
        let t2 = Instant::now();
        let target_points = if scope.has_target() {
            io_scan_points(&snapshot, target_filter_opt, false).await?
        } else {
            set_points.clone()
        };
        perf_mark!(t2, "scan_target_points", "n={}", target_points.len());

        // Bounded check: combined population must be <= max_points.
        let total_points = set_points.len() + target_points.len();
        if total_points > max_points {
            return Err(StoreError::Internal(format!(
                "resolve refused: combined population has {} points (set={}, target={}), \
                 exceeds the bound of {} — narrow the scope",
                total_points,
                set_points.len(),
                target_points.len(),
                max_points
            )));
        }

        // Build the doc-id sets for filtering.
        let target_doc_ids: Vec<String> = target_points
            .iter()
            .map(|p| p.doc_id.clone())
            .collect::<std::collections::HashSet<String>>()
            .into_iter()
            .collect();

        let set_doc_ids: Vec<String> = set_points
            .iter()
            .map(|p| p.doc_id.clone())
            .collect::<std::collections::HashSet<String>>()
            .into_iter()
            .collect();

        // For each set point: top-K filtered to target doc_ids.
        let t3 = Instant::now();
        let set_to_target =
            io_collect_top_k_cross(&snapshot, &set_points, &target_doc_ids, k, threshold_set).await?;
        perf_mark!(t3, "cross_set_to_target", "probes={}, candidates={}", set_points.len(), target_doc_ids.len());

        // For each target point: top-K filtered to set doc_ids.
        let t4 = Instant::now();
        let target_to_set =
            io_collect_top_k_cross(&snapshot, &target_points, &set_doc_ids, k, threshold_target).await?;
        perf_mark!(t4, "cross_target_to_set", "probes={}, candidates={}", target_points.len(), set_doc_ids.len());

        perf_mark!(t0, "resolve_total", "set={}, target={}", set_points.len(), target_points.len());

        Ok(ResolveNeighbourMaps {
            set_to_target,
            target_to_set,
        })
    }
}

/// Adaptive candidate-set size threshold for choosing between flat-KNN and ANN.
///
/// Below this threshold: materialised flat-KNN (exact, fast for small N).
/// Above this threshold: HNSW ANN + pre-parsed filter_expr (scales to large N).
///
/// VALUE: 30,000 candidates. Empirically measured (2026-06-16, resolve-crossover-
/// measurement.md): flat-KNN is faster than ANN at ALL sizes from 250 to 10,000
/// (3.7x faster even at 10k). The lines diverge — ANN never catches flat in
/// the measured range. Extrapolated crossover is 25,000-50,000 where flat's
/// linear O(N) memory pressure (exceeds L3 cache) finally gives ANN's O(log N)
/// index traversal the advantage.
///
/// At 30,000 candidates with 768-d embeddings, the materialised set is ~88MB
/// (4 * 768 * 30000 bytes). This may spill from L3 to RAM on some CPUs, but
/// flat-KNN's sequential access pattern and SIMD compute still outperform ANN's
/// random-access index navigation up to this scale. Beyond 30k, the ANN path
/// with its indexed O(log N) lookups becomes necessary.
///
/// RISK-24: brute force is BOUNDED at this threshold (max ~88MB memory, max
/// 30000 * dim float ops per probe). Above it, HNSW ANN provides O(log N) scaling.
const ADAPTIVE_ANN_CANDIDATE_THRESHOLD: usize = 30_000;

/// For each point in `points`, find its top-K nearest neighbours among the
/// `candidate_doc_ids` population. Returns a map from each point's doc_id to
/// its top-K neighbours (doc_id + normalised distance), sorted nearest-first.
/// Doc-level dedup: the best (nearest) chunk per candidate doc is kept. The
/// point's own doc_id is excluded from results.
///
/// ADAPTIVE STRATEGY: chooses the query path based on candidate-set size:
///
/// - SMALL sets (<= ADAPTIVE_ANN_CANDIDATE_THRESHOLD): materialised flat-KNN.
///   Scan candidate embeddings ONCE, compute exact cosine distances in-memory.
///   O(N*P*d) compute but zero disk I/O per probe and EXACT precision (no ANN
///   approximation). Optimal for Abt-Buy scale (~1092 candidates, ~6MB memory).
///
/// - LARGE sets (> ADAPTIVE_ANN_CANDIDATE_THRESHOLD): HNSW ANN + filter_expr.
///   Build the candidate IN-list as a pre-parsed DataFusion `Expr` ONCE, then
///   per-probe compose with self-exclusion and pass to `scanner.filter_expr()`.
///   Bypasses Lance's SQL parser entirely (eliminates O(N*P) parse cost).
///   O(P * log N) via the IVF_HNSW_SQ index. Scales to 100k+ candidates.
async fn io_collect_top_k_cross(
    snapshot: &Dataset,
    points: &[IndexedPoint],
    candidate_doc_ids: &[String],
    k: usize,
    threshold: f32,
) -> Result<std::collections::HashMap<String, Vec<crate::resolve::Neighbour>>, StoreError> {
    if candidate_doc_ids.len() <= ADAPTIVE_ANN_CANDIDATE_THRESHOLD {
        io_collect_top_k_cross_flat(snapshot, points, candidate_doc_ids, k, threshold).await
    } else {
        io_collect_top_k_cross_ann(snapshot, points, candidate_doc_ids, k, threshold).await
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Path A: Materialised flat-KNN (small candidate sets)
// ────────────────────────────────────────────────────────────────────────────

/// Materialised flat-KNN path for small candidate sets.
///
/// ALGORITHM:
///  1. SCAN the candidate population's `embedding` + `doc_id` columns ONCE from
///     the snapshot (single I/O pass with the IN-list filter).
///  2. For each probe point, compute flat cosine distance against ALL materialised
///     candidate embeddings in-memory. O(set_size * candidate_count * dim) pure
///     compute — no per-probe filter re-parse, no per-probe disk I/O.
///
/// PRECISION: flat KNN is EXACT (no ANN approximation loss). Provides strictly
/// better precision than the ANN path for the same population.
///
/// BOUNDED: candidate count <= ADAPTIVE_ANN_CANDIDATE_THRESHOLD ensures memory
/// is bounded (4 * dim * threshold bytes at maximum).
///
/// FD-SAFE: only ONE scan is issued (the materialisation scan); no per-probe
/// scanner FDs accumulate.
async fn io_collect_top_k_cross_flat(
    snapshot: &Dataset,
    points: &[IndexedPoint],
    candidate_doc_ids: &[String],
    k: usize,
    threshold: f32,
) -> Result<std::collections::HashMap<String, Vec<crate::resolve::Neighbour>>, StoreError> {
    use crate::resolve::Neighbour;

    let t_mat = Instant::now();
    let candidates = io_materialise_candidate_embeddings(snapshot, candidate_doc_ids).await?;
    perf_mark!(t_mat, "materialise_candidates", "n={}", candidates.len());

    if candidates.is_empty() {
        return Ok(std::collections::HashMap::new());
    }

    // Process in batches with yield between each batch — allows the tokio runtime
    // to service other tasks (health checks, concurrent /search) during long
    // compute runs. Without yielding, 2000+ probes * 5000 candidates * 768 dims
    // monopolises a worker thread for seconds.
    const COMPUTE_BATCH_SIZE: usize = 64;

    let t_compute = Instant::now();
    let mut result: std::collections::HashMap<String, Vec<Neighbour>> =
        std::collections::HashMap::new();

    for batch in points.chunks(COMPUTE_BATCH_SIZE) {
        for point in batch {
            let neighbours =
                flat_knn_top_k(&point.embedding, &point.doc_id, &candidates, k, threshold);

            if !neighbours.is_empty() {
                result
                    .entry(point.doc_id.clone())
                    .and_modify(|existing| {
                        merge_neighbours(existing, &neighbours);
                    })
                    .or_insert(neighbours);
            }
        }

        // Yield between batches — allows the tokio runtime to poll other futures
        // (health checks, concurrent requests) between CPU-bound compute bursts.
        tokio::task::yield_now().await;
    }
    perf_mark!(t_compute, "flat_knn_compute", "probes={}, candidates={}", points.len(), candidates.len());

    Ok(result)
}

/// A materialised candidate point: doc_id + embedding vector.
/// Held in memory for flat-KNN computation during resolve (small-set path only).
struct MaterialisedCandidate {
    doc_id: String,
    embedding: Vec<f32>,
}

/// Scan the candidate population's embeddings from the snapshot in a SINGLE I/O pass.
/// Applies the candidate doc_id IN-list filter ONCE (not per-probe).
///
/// FAIL-LOUD: missing columns or null embeddings are schema/corruption errors —
/// never silently skipped (would yield quietly-incomplete resolve results).
async fn io_materialise_candidate_embeddings(
    snapshot: &Dataset,
    candidate_doc_ids: &[String],
) -> Result<Vec<MaterialisedCandidate>, StoreError> {
    if candidate_doc_ids.is_empty() {
        return Ok(Vec::new());
    }

    // Reuse the shared filter builder (same escaping as /search, /duplicates paths).
    let t_filter = Instant::now();
    let filter = build_filter_expr(&[], candidate_doc_ids)
        .ok_or_else(|| StoreError::Internal(
            "candidate filter unexpectedly empty (non-empty doc_ids list produced no Expr)".to_owned()
        ))?;
    perf_mark!(t_filter, "materialise_build_filter", "ids={}", candidate_doc_ids.len());

    let t_scan_setup = Instant::now();
    let mut scanner = snapshot.scan();
    scanner
        .project(&["doc_id", "embedding"])
        .map_err(|e| StoreError::Internal(format!("candidate scan projection failed: {}", e)))?;
    scanner.filter_expr(filter);
    perf_mark!(t_scan_setup, "materialise_scanner_setup");

    let t_stream = Instant::now();
    let stream = scanner
        .try_into_stream()
        .await
        .map_err(|e| StoreError::Internal(format!("candidate scan stream failed: {}", e)))?;
    perf_mark!(t_stream, "materialise_try_into_stream");

    // STREAMING: process each batch as it arrives, extract embeddings, then
    // drop the batch before reading the next. This avoids holding all
    // RecordBatches AND all MaterialisedCandidates in memory simultaneously
    // (halves peak memory for the 30k-candidate path).
    let t_collect = Instant::now();
    let mut batch_count = 0usize;
    let mut candidates = Vec::new();
    tokio::pin!(stream);
    while let Some(batch_result) = futures::StreamExt::next(&mut stream).await {
        let batch = batch_result
            .map_err(|e| StoreError::Internal(format!("candidate scan stream batch failed: {}", e)))?;
        batch_count += 1;

        if batch.num_rows() == 0 {
            continue;
        }
        let doc_ids = batch
            .column_by_name("doc_id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| {
                StoreError::Internal("candidate scan: missing `doc_id` column".to_owned())
            })?;
        let embeddings = batch
            .column_by_name("embedding")
            .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>())
            .ok_or_else(|| {
                StoreError::Internal("candidate scan: missing `embedding` column".to_owned())
            })?;

        for i in 0..batch.num_rows() {
            let embedding = extract_embedding_row(embeddings, i, doc_ids.value(i))?;
            candidates.push(MaterialisedCandidate {
                doc_id: doc_ids.value(i).to_owned(),
                embedding,
            });
        }
        // batch is dropped here — Arrow buffers freed before next batch arrives.
    }
    perf_mark!(t_collect, "materialise_stream_batches", "batches={}", batch_count);
    perf_mark!(t_collect, "materialise_extract_vectors", "rows={}", candidates.len());

    Ok(candidates)
}

/// Pure flat-KNN: compute cosine distance from `probe` to all candidates,
/// exclude self-document, take top-k within threshold. Doc-level dedup: keeps
/// the best (nearest) chunk per candidate doc_id.
///
/// Returns neighbours sorted nearest-first, capped at k.
fn flat_knn_top_k(
    probe: &[f32],
    self_doc_id: &str,
    candidates: &[MaterialisedCandidate],
    k: usize,
    threshold: f32,
) -> Vec<crate::resolve::Neighbour> {
    use crate::kernel::distance::cosine_distance_normalized;
    use crate::resolve::Neighbour;

    let mut best_per_doc: HashMap<&str, f32> = HashMap::new();

    for candidate in candidates {
        if candidate.doc_id == self_doc_id {
            continue;
        }

        let dist = cosine_distance_normalized(probe, &candidate.embedding);

        if dist > threshold {
            continue;
        }

        best_per_doc
            .entry(candidate.doc_id.as_str())
            .and_modify(|best| {
                if dist < *best {
                    *best = dist;
                }
            })
            .or_insert(dist);
    }

    let mut sorted: Vec<(&str, f32)> = best_per_doc.into_iter().collect();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    sorted.truncate(k);

    sorted
        .into_iter()
        .map(|(id, distance)| Neighbour {
            id: id.to_owned(),
            distance,
        })
        .collect()
}

// ────────────────────────────────────────────────────────────────────────────
// Path B: HNSW ANN + pre-parsed filter_expr (large candidate sets)
// ────────────────────────────────────────────────────────────────────────────

/// HNSW ANN path for large candidate sets, using pre-parsed DataFusion `Expr`.
///
/// FILTER STRATEGY: builds the candidate IN-list as a DataFusion `Expr` ONCE
/// (pre-parsed, no SQL string), then per-probe composes it with the self-exclusion
/// clause via `Expr::and()` and passes to `scanner.filter_expr()`. This bypasses
/// Lance's SQL parser entirely — eliminates the O(N*P) parse cost that made the
/// old string-based `scanner.filter(&str)` approach a bottleneck for large
/// candidate populations.
///
/// BOUNDED CONCURRENCY: queries are processed in sequential batches of
/// RESOLVE_BATCH_SIZE; between batches the runtime yields, allowing completed
/// scans' IVF fragment FDs to drain. Each scan opens ~8-12 fragment files;
/// a batch of 16 peaks at ~128-192 working FDs — well under nofile=1024.
async fn io_collect_top_k_cross_ann(
    snapshot: &Dataset,
    points: &[IndexedPoint],
    candidate_doc_ids: &[String],
    k: usize,
    threshold: f32,
) -> Result<std::collections::HashMap<String, Vec<crate::resolve::Neighbour>>, StoreError> {
    use crate::resolve::Neighbour;

    /// Batch size for sequential ANN query processing. Points are processed in
    /// chunks of this size; between chunks the runtime yields, allowing completed
    /// scans' IVF fragment FDs to drain.
    const RESOLVE_BATCH_SIZE: usize = 16;

    // Over-fetch factor: ANN recall can miss, so fetch more and take the best k.
    let fetch_k = k * 2;

    // Build the candidate IN-list as a pre-parsed DataFusion Expr ONCE.
    // This eliminates the O(N*P) SQL re-parse cost: the old approach called
    // scanner.filter(&str) per probe, which parsed the large IN-list SQL string
    // on EVERY try_into_stream() call. Now we build the Expr tree once (O(N))
    // and clone it per probe (tree-copy, no parsing).
    let candidate_in_expr: Expr = in_list(
        col("doc_id"),
        candidate_doc_ids
            .iter()
            .map(|id| lit(id.as_str()))
            .collect(),
        false, // not negated — IN (positive match)
    );

    let mut result: std::collections::HashMap<String, Vec<Neighbour>> =
        std::collections::HashMap::new();

    for batch in points.chunks(RESOLVE_BATCH_SIZE) {
        for point in batch {
            // Compose the full filter: candidate IN-list AND exclude own doc.
            // Expr::clone() is a tree-clone of already-parsed AST nodes — O(N)
            // allocation but NO SQL parsing. The .and() and .not_eq() are
            // single-node constructions (O(1)).
            let expr = candidate_in_expr
                .clone()
                .and(col("doc_id").not_eq(lit(point.doc_id.as_str())));

            let mut scanner = snapshot.scan();
            scanner
                .nearest(
                    "embedding",
                    // Probe vector: the stored DOCUMENT-role embedding (doc→doc
                    // same-role probe). The asymmetric query→document probe has been
                    // removed in favour of same-role doc→doc comparison.
                    &Float32Array::from(point.embedding.clone()),
                    fetch_k,
                )
                .map_err(|e| {
                    StoreError::Internal(format!("resolve nearest setup failed: {}", e))
                })?;
            scanner.distance_metric(DistanceType::Cosine);
            scanner.project(&["doc_id"]).map_err(|e| {
                StoreError::Internal(format!("resolve nearest projection failed: {}", e))
            })?;
            // filter_expr: pre-parsed DataFusion Expr — NO SQL string parsing.
            scanner.filter_expr(expr);

            let batches: Vec<RecordBatch> = scanner
                .try_into_stream()
                .await
                .map_err(|e| {
                    StoreError::Internal(format!("resolve nearest stream failed: {}", e))
                })?
                .try_collect()
                .await
                .map_err(|e| {
                    StoreError::Internal(format!("resolve nearest collect failed: {}", e))
                })?;

            let neighbours = extract_top_k_neighbours(&batches, k, threshold)?;
            result
                .entry(point.doc_id.clone())
                .and_modify(|existing| {
                    merge_neighbours(existing, &neighbours);
                })
                .or_insert(neighbours);
        }

        // Yield between batches — allows the runtime to poll other futures and
        // lets Lance's I/O scheduler flush completed fragment readers, draining
        // FDs before the next batch opens new ones.
        tokio::task::yield_now().await;
    }

    Ok(result)
}

/// Extract doc-level top-K neighbours from ANN result batches.
/// Deduplicates by doc_id (keeps best distance), caps at k, filters by threshold.
/// Distances are normalised from Lance [0,2] to reference [0,1] scale.
///
/// FAIL-LOUD: missing `doc_id` or `_distance` columns are schema/corruption errors
/// that indicate a broken ANN result contract — never silently skipped.
fn extract_top_k_neighbours(
    batches: &[RecordBatch],
    k: usize,
    threshold: f32,
) -> Result<Vec<crate::resolve::Neighbour>, StoreError> {
    use crate::resolve::Neighbour;

    let mut best_per_doc: HashMap<String, f32> = HashMap::new();

    for batch in batches {
        if batch.num_rows() == 0 {
            continue;
        }
        let doc_ids = batch
            .column_by_name("doc_id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| {
                StoreError::Internal(
                    "ANN result batch missing `doc_id` column — Lance contract violated".to_owned(),
                )
            })?;
        let distances = batch
            .column_by_name("_distance")
            .and_then(|c| c.as_any().downcast_ref::<Float32Array>())
            .ok_or_else(|| {
                StoreError::Internal(
                    "ANN result batch missing `_distance` column — Lance nearest() contract violated".to_owned(),
                )
            })?;

        for i in 0..batch.num_rows() {
            let doc_id = doc_ids.value(i).to_owned();
            let raw_dist = distances.value(i);
            let norm_dist = normalized_cosine_from_lance(raw_dist);
            if norm_dist > threshold {
                continue;
            }
            best_per_doc
                .entry(doc_id)
                .and_modify(|best| {
                    if norm_dist < *best {
                        *best = norm_dist;
                    }
                })
                .or_insert(norm_dist);
        }
    }

    let mut sorted: Vec<(String, f32)> = best_per_doc.into_iter().collect();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    sorted.truncate(k);

    Ok(sorted
        .into_iter()
        .map(|(id, distance)| Neighbour { id, distance })
        .collect())
}

/// Merge new neighbours into an existing list, keeping the best distance per doc.
fn merge_neighbours(
    existing: &mut Vec<crate::resolve::Neighbour>,
    new: &[crate::resolve::Neighbour],
) {
    for n in new {
        if let Some(e) = existing.iter_mut().find(|e| e.id == n.id) {
            if n.distance < e.distance {
                e.distance = n.distance;
            }
        } else {
            existing.push(n.clone());
        }
    }
}

/// A single indexed point in the SET population: its owning document id, its
/// document-role embedding (for same-role doc→doc probe), and (when snippets
/// are requested) its chunk text.
#[derive(Clone)]
pub(super) struct IndexedPoint {
    pub(super) doc_id: String,
    /// Document-role embedding (search_document: prefix). Used by /resolve and
    /// /duplicates to probe set.embedding against target.embedding ANN index
    /// (same-role doc→doc comparison).
    pub(super) embedding: Vec<f32>,
    pub(super) content: Option<String>,
}

/// Scan the SET population's points (doc_id + embedding, plus content for
/// snippets) from a versioned snapshot, optionally filtered to the set scope.
/// Projects `embedding` (document-role) for same-role doc→doc probing.
async fn io_scan_points(
    snapshot: &Dataset,
    set_filter: Option<Expr>,
    snippet: bool,
) -> Result<Vec<IndexedPoint>, StoreError> {
    let t_setup = Instant::now();
    let mut scanner = snapshot.scan();
    let projection: &[&str] = if snippet {
        &["doc_id", "embedding", "content"]
    } else {
        &["doc_id", "embedding"]
    };
    scanner
        .project(projection)
        .map_err(|e| StoreError::Internal(format!("duplicates projection failed: {}", e)))?;
    if let Some(filter) = set_filter {
        scanner.filter_expr(filter);
    }
    perf_mark!(t_setup, "scan_points_setup");

    let t_stream = Instant::now();
    let stream = scanner
        .try_into_stream()
        .await
        .map_err(|e| StoreError::Internal(format!("duplicates scan stream failed: {}", e)))?;
    perf_mark!(t_stream, "scan_points_try_into_stream");

    let t_collect = Instant::now();
    let batches: Vec<RecordBatch> = stream
        .try_collect()
        .await
        .map_err(|e| StoreError::Internal(format!("duplicates scan collect failed: {}", e)))?;
    perf_mark!(t_collect, "scan_points_collect", "batches={}", batches.len());

    let t_extract = Instant::now();
    let result = batches_to_points(&batches, snippet)?;
    perf_mark!(t_extract, "scan_points_extract", "points={}", result.len());

    Ok(result)
}

/// Extract `IndexedPoint`s from projected RecordBatches (doc_id +
/// embedding, plus content when snippets requested).
/// FAIL-LOUD: a batch missing a required column, or an embedding row that is null,
/// is a real schema/corruption error — never silently skipped (which would drop
/// points and yield quietly-incomplete duplicate results).
fn batches_to_points(batches: &[RecordBatch], snippet: bool) -> Result<Vec<IndexedPoint>, StoreError> {
    batches.iter().try_fold(Vec::new(), |mut points, batch| {
        let n = batch.num_rows();
        if n == 0 {
            return Ok(points);
        }
        let doc_ids = batch
            .column_by_name("doc_id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| {
                StoreError::Internal("duplicates scan: missing `doc_id` column".to_owned())
            })?;
        let embeddings = batch
            .column_by_name("embedding")
            .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>())
            .ok_or_else(|| {
                StoreError::Internal("duplicates scan: missing `embedding` column".to_owned())
            })?;
        let contents = if snippet {
            Some(
                batch
                    .column_by_name("content")
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                    .ok_or_else(|| {
                        StoreError::Internal("duplicates scan: missing `content` column".to_owned())
                    })?,
            )
        } else {
            None
        };

        for i in 0..n {
            let embedding = extract_embedding_row(embeddings, i, doc_ids.value(i))?;
            points.push(IndexedPoint {
                doc_id: doc_ids.value(i).to_owned(),
                embedding,
                content: contents.map(|c| c.value(i).to_owned()),
            });
        }
        Ok(points)
    })
}

/// For each SET point, run ONE filtered `nearest()` ANN query against the snapshot
/// and record its nearest neighbour as a [`NeighbourObservation`] on the reference
/// [0, 1] cosine scale. O(n) indexed queries.
async fn io_collect_neighbours(
    snapshot: &Dataset,
    points: &[IndexedPoint],
    scope: &DuplicateScope,
    snippet: bool,
) -> Result<Vec<NeighbourObservation>, StoreError> {
    let mut observations = Vec::new();
    for point in points {
        if let Some(obs) = io_nearest_neighbour(snapshot, point, scope, snippet).await? {
            observations.push(obs);
        }
    }
    Ok(observations)
}

/// Run a FILTERED `nearest()` ANN query for one set point and return its nearest
/// CROSS-DOCUMENT neighbour observation, or `None` if no neighbour exists in scope.
///
/// The filter is what makes this robust (vs the old `nearest(k=2)` + skip-own-doc,
/// which was starved by a multi-chunk document's own sibling chunks at scale):
///  - within-set (no target): exclude the point's own `doc_id` so the result can
///    only contain OTHER documents — a genuine cross-doc neighbour, never the
///    point's own siblings.
///  - cross-set (target present): restrict the result to the target population
///    (its doc_type/doc_id IN-lists) so the neighbour STRADDLES set↔target.
///
/// `k` is a small fixed over-fetch to absorb ANN recall slack (nprobes), then we
/// take the first (nearest) returned row. Because the filter already guarantees
/// cross-document rows, `k` does not need to grow with documents' chunk counts.
async fn io_nearest_neighbour(
    snapshot: &Dataset,
    point: &IndexedPoint,
    scope: &DuplicateScope,
    snippet: bool,
) -> Result<Option<NeighbourObservation>, StoreError> {
    const NEIGHBOUR_K: usize = 8;

    let filter = build_neighbour_filter(&point.doc_id, scope);

    let mut scanner = snapshot.scan();
    scanner
        .nearest(
            "embedding",
            // Same-role doc→doc probe: use the document-role embedding to probe
            // the document-role ANN index.
            &Float32Array::from(point.embedding.clone()),
            NEIGHBOUR_K,
        )
        .map_err(|e| StoreError::Internal(format!("duplicates nearest setup failed: {}", e)))?;
    scanner.distance_metric(DistanceType::Cosine);
    let projection: &[&str] = if snippet {
        &["doc_id", "content"]
    } else {
        &["doc_id"]
    };
    scanner
        .project(projection)
        .map_err(|e| StoreError::Internal(format!("duplicates nearest projection failed: {}", e)))?;
    scanner.filter_expr(filter);

    let batches: Vec<RecordBatch> = scanner
        .try_into_stream()
        .await
        .map_err(|e| StoreError::Internal(format!("duplicates nearest stream failed: {}", e)))?
        .try_collect()
        .await
        .map_err(|e| StoreError::Internal(format!("duplicates nearest collect failed: {}", e)))?;

    Ok(nearest_observation(&batches, point, snippet))
}

/// Build the ANN filter for a set point's neighbour query. The query point's own
/// document is excluded in BOTH modes (a doc is never its own duplicate); in
/// cross-set mode the target population's IN-list is AND-ed on so the neighbour is
/// drawn only from the target.
///
/// SECURITY: Uses DataFusion Expr values — no SQL string interpolation.
fn build_neighbour_filter(self_doc_id: &str, scope: &DuplicateScope) -> Expr {
    let exclude_self = col("doc_id").not_eq(lit(self_doc_id));
    if scope.has_target() {
        let target = build_filter_expr(&scope.target_doc_types, &scope.target_doc_ids);
        // has_target() guaranteed at least one of the target lists is non-empty,
        // so `target` is non-empty here.
        exclude_self.and(target.expect("has_target guarantees non-empty filter"))
    } else {
        exclude_self
    }
}

/// From a filtered `nearest()` result (already cross-document by construction),
/// take the first (nearest) row and build a normalised-scale observation, with
/// snippets when requested. Returns `None` when the result is empty (no in-scope
/// neighbour for this point). Fails loud if the ranked result lacks `_distance`
/// (would otherwise corrupt the distance with a default).
fn nearest_observation(
    batches: &[RecordBatch],
    point: &IndexedPoint,
    snippet: bool,
) -> Option<NeighbourObservation> {
    for batch in batches {
        if batch.num_rows() == 0 {
            continue;
        }
        let doc_ids = batch
            .column_by_name("doc_id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let distances = batch
            .column_by_name("_distance")
            .and_then(|c| c.as_any().downcast_ref::<Float32Array>());
        let contents = if snippet {
            batch
                .column_by_name("content")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        } else {
            None
        };
        let (Some(ids), Some(dists)) = (doc_ids, distances) else {
            continue;
        };
        return Some(NeighbourObservation {
            doc_a: point.doc_id.clone(),
            doc_b: ids.value(0).to_owned(),
            normalized_distance: normalized_cosine_from_lance(dists.value(0)),
            content_a: point.content.clone(),
            content_b: contents.map(|c| c.value(0).to_owned()),
        });
    }
    None
}
