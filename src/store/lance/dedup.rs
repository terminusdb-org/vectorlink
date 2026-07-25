//! Pure document-level dedup logic (no I/O, no LanceStore dependency).

use std::collections::HashMap;

use crate::chunk::chunk_location;
use crate::kernel::distance::normalized_cosine_from_lance;
use crate::kernel::model::{ChunkInfo, DuplicateGroup, DuplicateMember, SearchHit};

use super::{ChunkHit, DistanceKind, NeighbourObservation};

/// Reduce raw chunk-level nearest-neighbour observations to DOCUMENT-level
/// near-duplicate pairs. Pure (no I/O) so the pairing/dedup rules are unit-tested
/// in isolation from the Lance scan.
///
/// Rules (the doc-level pairing contract):
///  - UNITS: `threshold` and every `normalized_distance` are on the reference
///    [0, 1] cosine scale (0 = identical, 0.5 = orthogonal, 1 = opposite) — the
///    SAME scale `/search` reports. An observation counts as a near-duplicate iff
///    `normalized_distance <= threshold`.
///  - DOC-LEVEL, NOT CHUNK-LEVEL: a document has multiple chunk vectors. We key
///    pairs by the two distinct DOCUMENT ids, never by chunk. The distance kept
///    for a document pair is the BEST (smallest) over all chunk observations
///    between those two documents.
///  - NEVER (docX, docX): an observation whose two chunks belong to the same
///    document is discarded (a document is not its own duplicate).
///  - LOWER ID FIRST: each pair is canonicalised so the lexicographically smaller
///    document id is first, which also collapses the symmetric (a,b)/(b,a)
///    observations into one pair.
///  - BOUNDED + DETERMINISTIC: the result is sorted NEAREST-FIRST (smallest
///    distance), ties broken by id1 then id2 for stable output, and truncated to
///    `max_pairs` AFTER sorting, so the cap keeps the closest pairs.
///  - SNIPPETS: each member carries its chunk text from the BEST (kept)
///    observation when the scan collected content; absent otherwise.
pub fn pairs_from_neighbours(
    observations: &[NeighbourObservation],
    threshold: f32,
    max_pairs: usize,
) -> Vec<DuplicateGroup> {
    // Key: canonical (lower_id, higher_id) → the best (smallest-distance)
    // observation seen for that document pair (carries distance + snippets).
    let best_per_pair: HashMap<(String, String), CanonicalObservation> = observations
        .iter()
        .filter(|obs| obs.normalized_distance <= threshold)
        .filter(|obs| obs.doc_a != obs.doc_b)
        .map(canonical_observation)
        .fold(HashMap::new(), |mut acc, candidate| {
            acc.entry(candidate.pair.clone())
                .and_modify(|best| {
                    if candidate.distance < best.distance {
                        *best = candidate.clone();
                    }
                })
                .or_insert(candidate);
            acc
        });

    let mut groups: Vec<DuplicateGroup> = best_per_pair
        .into_values()
        .map(|obs| DuplicateGroup {
            group: vec![
                DuplicateMember { id: obs.pair.0, snippet: obs.snippet_lo },
                DuplicateMember { id: obs.pair.1, snippet: obs.snippet_hi },
            ],
            distance: obs.distance,
        })
        .collect();

    // Nearest-first; deterministic tie-break by the canonical member ids.
    groups.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.group[0].id.cmp(&b.group[0].id))
            .then_with(|| a.group[1].id.cmp(&b.group[1].id))
    });
    groups.truncate(max_pairs);
    groups
}

/// Dedup chunk-level hits to document-level hits (best chunk per doc_id).
/// Distance transform is mode-aware: only RawCosine hits get `normalized_cosine_from_lance`;
/// already-normalised hits (FTS, RRF) pass through unchanged.
pub fn dedup_chunks_to_documents(hits: Vec<ChunkHit>, snippet: bool) -> Vec<SearchHit> {
    use std::collections::HashMap;

    // Group by doc_id, keep the best (smallest distance) chunk per document.
    let mut best_per_doc: HashMap<String, ChunkHit> = HashMap::new();
    for hit in hits {
        let entry = best_per_doc
            .entry(hit.doc_id.clone())
            .or_insert_with(|| hit.clone());
        if hit.distance < entry.distance {
            *entry = hit;
        }
    }

    let mut results: Vec<SearchHit> = best_per_doc
        .into_values()
        .map(|hit| {
            let location = chunk_location(hit.chunk_token_start as u32, hit.doc_token_len as u32);
            let final_distance = match hit.distance_kind {
                DistanceKind::RawCosine => normalized_cosine_from_lance(hit.distance),
                DistanceKind::Normalised => hit.distance,
            };
            SearchHit {
                id: hit.doc_id,
                distance: final_distance,
                chunk: ChunkInfo {
                    index: hit.chunk_index as u32,
                    count: hit.chunk_count as u32,
                    token_start: hit.chunk_token_start as u32,
                    doc_token_len: hit.doc_token_len as u32,
                    location,
                    snippet: if snippet { Some(hit.content) } else { None },
                },
            }
        })
        .collect();

    // Sort by distance (nearest first).
    results.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

/// A neighbour observation canonicalised to (lower_id, higher_id) with its
/// snippets aligned to that ordering, ready to reduce into a [`DuplicateGroup`].
#[derive(Debug, Clone)]
struct CanonicalObservation {
    pair: (String, String),
    distance: f32,
    snippet_lo: Option<String>,
    snippet_hi: Option<String>,
}

/// Canonicalise a neighbour observation to (lower_id, higher_id), keeping each
/// member's snippet aligned with the reordered ids.
/// Precondition: `doc_a != doc_b` (callers filter same-document observations).
fn canonical_observation(obs: &NeighbourObservation) -> CanonicalObservation {
    if obs.doc_a <= obs.doc_b {
        CanonicalObservation {
            pair: (obs.doc_a.clone(), obs.doc_b.clone()),
            distance: obs.normalized_distance,
            snippet_lo: obs.content_a.clone(),
            snippet_hi: obs.content_b.clone(),
        }
    } else {
        CanonicalObservation {
            pair: (obs.doc_b.clone(), obs.doc_a.clone()),
            distance: obs.normalized_distance,
            snippet_lo: obs.content_b.clone(),
            snippet_hi: obs.content_a.clone(),
        }
    }
}
