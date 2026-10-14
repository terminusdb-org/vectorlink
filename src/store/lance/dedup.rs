// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 DFRNT AB

//! Pure document-level dedup logic (no I/O, no LanceStore dependency).

use std::collections::HashMap;

use crate::chunk::chunk_location;
use crate::kernel::distance::normalized_cosine_from_lance;
use crate::kernel::model::{ChunkInfo, DuplicateGroup, DuplicateMember, LegScore, SearchHit};

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
            let final_distance = display_distance(&hit);
            let chunk = chunk_info(&hit, snippet, location);
            SearchHit {
                id: hit.doc_id,
                distance: final_distance,
                chunk,
                rrf_score: None,
                vector: None,
                fts: None,
            }
        })
        .collect();

    // Sort by distance (nearest first); doc id breaks ties deterministically.
    results.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });

    results
}

/// Document-level Reciprocal Rank Fusion over the vector and FTS legs of a
/// hybrid search (Cormack et al. 2009, k=60).
///
/// Each leg collapses to a document ranking via MaxP — a document's rank in a
/// leg is the position of its best chunk among the leg's distinct documents,
/// so multi-chunk fan-out cannot buy rank. The fused score sums
/// `1/(K + rank + 1)` over the legs the document appears in: a document
/// present in both legs accrues both contributions.
///
/// The representative chunk (snippet, location) is the vector leg's best chunk
/// when present — it carries a real cosine distance — else the FTS best chunk.
/// The hit's top-level `distance` is the fused-quality distance (monotone in
/// the RRF score), so result order and distance order agree; the raw per-leg
/// ranks and distances surface on the hit so callers can see why a document
/// ranked where it did.
pub fn rrf_fuse_documents(
    vector_hits: Vec<ChunkHit>,
    fts_hits: Vec<ChunkHit>,
    snippet: bool,
) -> Vec<SearchHit> {
    const K: f32 = 60.0;

    struct FusedDoc {
        score: f32,
        vector: Option<(u32, ChunkHit)>,
        fts: Option<(u32, ChunkHit)>,
    }

    let mut docs: HashMap<String, FusedDoc> = HashMap::new();
    let mut absorb = |hits: Vec<ChunkHit>, is_vector: bool| {
        let mut rank = 0u32;
        for hit in hits {
            let entry = docs.entry(hit.doc_id.clone()).or_insert(FusedDoc {
                score: 0.0,
                vector: None,
                fts: None,
            });
            let slot = if is_vector { &mut entry.vector } else { &mut entry.fts };
            if slot.is_some() {
                continue; // best chunk for this doc already seen in this leg
            }
            entry.score += 1.0 / (K + rank as f32 + 1.0);
            *slot = Some((rank, hit));
            rank += 1;
        }
    };
    absorb(vector_hits, true);
    absorb(fts_hits, false);

    let mut results: Vec<SearchHit> = docs
        .into_iter()
        .map(|(doc_id, fused)| {
            let rep = fused
                .vector
                .as_ref()
                .map(|(_, h)| h)
                .or(fused.fts.as_ref().map(|(_, h)| h))
                .expect("fused doc must have at least one leg");
            let location = chunk_location(rep.chunk_token_start as u32, rep.doc_token_len as u32);
            SearchHit {
                id: doc_id,
                distance: fused_distance(fused.score),
                chunk: chunk_info(rep, snippet, location),
                rrf_score: Some(fused.score),
                vector: fused.vector.map(|(rank, hit)| leg_score(rank, &hit)),
                fts: fused.fts.map(|(rank, hit)| leg_score(rank, &hit)),
            }
        })
        .collect();

    // Fused score descending; doc id breaks ties deterministically.
    results.sort_by(|a, b| {
        b.rrf_score
            .partial_cmp(&a.rrf_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });

    results
}

/// Reference-scale [0,1] distance for display: RawCosine hits go through
/// `normalized_cosine_from_lance`; Normalised hits pass through.
fn display_distance(hit: &ChunkHit) -> f32 {
    match hit.distance_kind {
        DistanceKind::RawCosine => normalized_cosine_from_lance(hit.distance),
        DistanceKind::Normalised => hit.distance,
    }
}

fn chunk_info(hit: &ChunkHit, snippet: bool, location: f32) -> ChunkInfo {
    ChunkInfo {
        index: hit.chunk_index as u32,
        count: hit.chunk_count as u32,
        token_start: hit.chunk_token_start as u32,
        doc_token_len: hit.doc_token_len as u32,
        location,
        snippet: if snippet { Some(hit.content.clone()) } else { None },
    }
}

/// Fused-quality distance for hybrid hits: `1 − score/score_max` where
/// score_max is the RRF score of a document ranked 0 in BOTH legs.
/// Monotone in the fused score, so result order and distance order agree.
/// Anchors: 0 = top of both legs, 0.5 = top of only one leg, →1 = barely
/// surfaced.
fn fused_distance(score: f32) -> f32 {
    const K: f32 = 60.0;
    let score_max = 2.0 / (K + 1.0);
    (1.0 - score / score_max).max(0.0)
}

fn leg_score(rank: u32, hit: &ChunkHit) -> LegScore {
    LegScore {
        rank,
        distance: display_distance(hit),
    }
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
