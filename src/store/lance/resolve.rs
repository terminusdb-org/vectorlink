//! Resolution/duplicates: near-duplicate grouping and cross-neighbour resolution.

use std::collections::HashMap;

use arrow_array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch, StringArray,
};
use futures::TryStreamExt;
use lance::dataset::Dataset;
use lance_linalg::distance::DistanceType;

use crate::kernel::distance::normalized_cosine_from_lance;
use crate::kernel::error::StoreError;
use crate::kernel::model::DuplicateGroup;

use super::{
    DuplicateScope, LanceStore, NeighbourObservation, ResolveNeighbourMaps,
    DEFAULT_DUPLICATE_MAX_PAIRS,
};
use super::dedup::pairs_from_neighbours;
use super::search::{build_filter_expression, extract_embedding_row};

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

        let set_filter = build_filter_expression(&scope.set_doc_types, &scope.set_doc_ids);
        let set_filter_opt = if set_filter.is_empty() {
            None
        } else {
            Some(set_filter)
        };

        // Bound on the SET population (the points we iterate), not the whole
        // snapshot — a filtered run over a small set inside a huge corpus is still
        // safely bounded.
        let set_point_count = snapshot
            .count_rows(set_filter_opt.clone())
            .await
            .map_err(|e| StoreError::Internal(format!("count_rows for duplicates failed: {}", e)))?;

        if set_point_count > max_points {
            return Err(StoreError::Internal(format!(
                "near-duplicate scan refused: set population has {} points, exceeds the bound of \
                 {} (would not be a safely bounded run) — narrow the `set` scope or chunk the request",
                set_point_count, max_points
            )));
        }

        let points = io_scan_points(&snapshot, set_filter_opt.as_deref(), snippet).await?;
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
        threshold: f32,
        max_points: usize,
    ) -> Result<ResolveNeighbourMaps, StoreError> {
        let snapshot = self.io_open_snapshot(domain, branch, commit).await?;

        // Build filters for set and target populations.
        let set_filter = build_filter_expression(&scope.set_doc_types, &scope.set_doc_ids);
        let set_filter_opt = if set_filter.is_empty() { None } else { Some(set_filter) };

        let target_filter =
            build_filter_expression(&scope.target_doc_types, &scope.target_doc_ids);
        let target_filter_opt =
            if target_filter.is_empty() { None } else { Some(target_filter) };

        // Scan set points (embeddings, no snippets).
        let set_points = io_scan_points(&snapshot, set_filter_opt.as_deref(), false).await?;

        // Scan target points. If no explicit target, target = set (within-set dedup).
        let target_points = if scope.has_target() {
            io_scan_points(&snapshot, target_filter_opt.as_deref(), false).await?
        } else {
            set_points.clone()
        };

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

        // For each set point: ANN top-K filtered to target doc_ids.
        let set_to_target =
            io_collect_top_k_cross(&snapshot, &set_points, &target_doc_ids, k, threshold).await?;

        // For each target point: ANN top-K filtered to set doc_ids.
        let target_to_set =
            io_collect_top_k_cross(&snapshot, &target_points, &set_doc_ids, k, threshold).await?;

        Ok(ResolveNeighbourMaps {
            set_to_target,
            target_to_set,
        })
    }
}

/// For each point in `points`, run a filtered top-K ANN query against `snapshot`
/// restricted to the `candidate_doc_ids` population. Returns a map from each
/// point's doc_id to its top-K neighbours (doc_id + normalised distance), sorted
/// nearest-first. Doc-level dedup: the best (nearest) chunk per candidate doc is
/// kept. The point's own doc_id is excluded from results.
///
/// This is the resolve-specific generalisation of `io_collect_neighbours` — it
/// returns ALL k neighbours per point (not just the nearest 1), which the
/// resolve algorithm needs for reciprocal top-K grounding.
///
/// BOUNDED CONCURRENCY: queries run through
/// `buffer_unordered(RESOLVE_ANN_CONCURRENCY)` so at most N scans are in-flight
/// simultaneously. Completed scans' IVF fragment FDs drain while new ones start,
/// keeping the peak FD count well under the default nofile=1024 limit even for
/// large populations (~2000+ points). Sequential processing caused ~1107 FDs to
/// accumulate before draining (exceeding default nofile); bounded concurrency
/// caps the working set to ~N × per-query FDs.
async fn io_collect_top_k_cross(
    snapshot: &Dataset,
    points: &[IndexedPoint],
    candidate_doc_ids: &[String],
    k: usize,
    threshold: f32,
) -> Result<std::collections::HashMap<String, Vec<crate::resolve::Neighbour>>, StoreError> {
    use crate::resolve::Neighbour;

    /// Batch size for sequential ANN query processing. Points are processed in
    /// chunks of this size; between chunks the runtime yields, allowing completed
    /// scans' IVF fragment FDs to drain. Each scan opens ~8-12 fragment files;
    /// a batch of 16 peaks at ~128-192 working FDs — well under nofile=1024 with
    /// baseline overhead (~14 FDs). Swapping the probe vector (dual-vector recall
    /// fix) is a one-line change at the `point.embedding.clone()` site below.
    const RESOLVE_BATCH_SIZE: usize = 16;

    // Over-fetch factor: ANN recall can miss, so fetch more and take the best k.
    let fetch_k = k * 2;

    // Build the IN-list filter for candidates ONCE (shared across all queries).
    // This avoids O(points × candidates) string allocation; only the per-point
    // `doc_id != '...'` suffix varies.
    let candidate_in_list: String = candidate_doc_ids
        .iter()
        .map(|id| format!("'{}'", id.replace('\'', "''")))
        .collect::<Vec<_>>()
        .join(", ");

    let mut result: std::collections::HashMap<String, Vec<Neighbour>> =
        std::collections::HashMap::new();

    // Process in sequential batches with yield between each batch.
    // The yield allows the tokio runtime to reclaim resources (including FDs
    // from completed I/O scheduler flushes in Lance 7.0) before the next batch
    // opens new fragment files.
    for batch in points.chunks(RESOLVE_BATCH_SIZE) {
        for point in batch {
            // Filter: restrict to candidate population AND exclude own doc.
            let filter = format!(
                "doc_id IN ({}) AND doc_id != '{}'",
                candidate_in_list,
                point.doc_id.replace('\'', "''")
            );

            let mut scanner = snapshot.scan();
            scanner
                .nearest(
                    "embedding",
                    // Probe vector: currently the stored document embedding.
                    // Dual-vector recall fix: swap `point.embedding` for the
                    // query_embedding field here (one-line change).
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
            scanner.filter(&filter).map_err(|e| {
                StoreError::Internal(format!("resolve nearest filter failed: {}", e))
            })?;

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

            let neighbours = extract_top_k_neighbours(&batches, k, threshold);
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
fn extract_top_k_neighbours(
    batches: &[RecordBatch],
    k: usize,
    threshold: f32,
) -> Vec<crate::resolve::Neighbour> {
    use crate::resolve::Neighbour;

    // Collect all (doc_id, distance) pairs, dedup by doc_id (min distance).
    let mut best_per_doc: HashMap<String, f32> = HashMap::new();

    for batch in batches {
        if batch.num_rows() == 0 {
            continue;
        }
        let doc_ids = match batch
            .column_by_name("doc_id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        {
            Some(ids) => ids,
            None => continue,
        };
        let distances = match batch
            .column_by_name("_distance")
            .and_then(|c| c.as_any().downcast_ref::<Float32Array>())
        {
            Some(d) => d,
            None => continue,
        };

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

    // Sort by distance (nearest first), take top k.
    let mut sorted: Vec<(String, f32)> = best_per_doc.into_iter().collect();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    sorted.truncate(k);

    sorted
        .into_iter()
        .map(|(id, distance)| Neighbour { id, distance })
        .collect()
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
/// embedding vector, and (when snippets are requested) its chunk text.
#[derive(Clone)]
pub(super) struct IndexedPoint {
    pub(super) doc_id: String,
    pub(super) embedding: Vec<f32>,
    pub(super) content: Option<String>,
}

/// Scan the SET population's points (doc_id + embedding, plus content for
/// snippets) from a versioned snapshot, optionally filtered to the set scope.
async fn io_scan_points(
    snapshot: &Dataset,
    set_filter: Option<&str>,
    snippet: bool,
) -> Result<Vec<IndexedPoint>, StoreError> {
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
        scanner
            .filter(filter)
            .map_err(|e| StoreError::Internal(format!("duplicates set filter failed: {}", e)))?;
    }

    let batches: Vec<RecordBatch> = scanner
        .try_into_stream()
        .await
        .map_err(|e| StoreError::Internal(format!("duplicates scan stream failed: {}", e)))?
        .try_collect()
        .await
        .map_err(|e| StoreError::Internal(format!("duplicates scan collect failed: {}", e)))?;

    batches_to_points(&batches, snippet)
}

/// Extract `IndexedPoint`s from projected RecordBatches (doc_id + embedding, plus
/// content when snippets requested).
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
        .nearest("embedding", &Float32Array::from(point.embedding.clone()), NEIGHBOUR_K)
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
    scanner
        .filter(&filter)
        .map_err(|e| StoreError::Internal(format!("duplicates nearest filter failed: {}", e)))?;

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
/// drawn only from the target. Single-quotes in ids are escaped (SQL string).
fn build_neighbour_filter(self_doc_id: &str, scope: &DuplicateScope) -> String {
    let exclude_self = format!("doc_id != '{}'", self_doc_id.replace('\'', "''"));
    if scope.has_target() {
        let target = build_filter_expression(&scope.target_doc_types, &scope.target_doc_ids);
        // has_target() guaranteed at least one of the target lists is non-empty,
        // so `target` is non-empty here.
        format!("{} AND ({})", exclude_self, target)
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
