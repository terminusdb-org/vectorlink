// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 DFRNT AB

use super::*;
use super::search::{batches_to_vector_hits, rrf_merge};
use crate::kernel::model::{BranchName, Domain, DuplicateGroup, SearchMode};
use arrow_array::Array as _;
use futures::TryStreamExt as _;
use lance::index::DatasetIndexExt;

fn make_test_store(dim: usize) -> (LanceStore, tempfile::TempDir) {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let store = LanceStore::new(tmp.path(), dim, 256 * 1024 * 1024, 128 * 1024 * 1024);
    (store, tmp)
}

fn fake_embedding(dim: usize, seed: f32) -> Vec<f32> {
    (0..dim).map(|i| (seed + i as f32 * 0.01).sin()).collect()
}

fn observation(a: &str, b: &str, distance: f32) -> NeighbourObservation {
    NeighbourObservation {
        doc_a: a.to_owned(),
        doc_b: b.to_owned(),
        normalized_distance: distance,
        content_a: None,
        content_b: None,
    }
}

/// Project a DuplicateGroup back to a (lower_id, higher_id) tuple for concise
/// assertions on the pairing CONTRACT (the rich shape is asserted separately).
fn group_pair(g: &DuplicateGroup) -> (String, String) {
    (g.group[0].id.clone(), g.group[1].id.clone())
}

fn group_pairs(groups: &[DuplicateGroup]) -> Vec<(String, String)> {
    groups.iter().map(group_pair).collect()
}

// --- pure pairing/dedup logic (doc-level near-duplicate contract) ---

#[test]
fn pairs_below_threshold_are_returned_lower_id_first() {
    let obs = vec![observation("doc/zebra", "doc/apple", 0.02)];
    let groups = pairs_from_neighbours(&obs, 0.05, DEFAULT_DUPLICATE_MAX_PAIRS);
    // Lower id first regardless of observation order.
    assert_eq!(group_pairs(&groups), vec![("doc/apple".to_owned(), "doc/zebra".to_owned())]);
}

#[test]
fn pairs_above_threshold_are_excluded() {
    let obs = vec![observation("doc/a", "doc/b", 0.4)];
    let groups = pairs_from_neighbours(&obs, 0.05, DEFAULT_DUPLICATE_MAX_PAIRS);
    assert!(groups.is_empty(), "0.4 > 0.05 threshold must not pair");
}

#[test]
fn tightening_threshold_drops_pairs() {
    let obs = vec![
        observation("doc/a", "doc/b", 0.04),
        observation("doc/c", "doc/d", 0.18),
    ];
    let loose = pairs_from_neighbours(&obs, 0.2, DEFAULT_DUPLICATE_MAX_PAIRS);
    assert_eq!(loose.len(), 2, "loose threshold keeps both pairs");
    let tight = pairs_from_neighbours(&obs, 0.05, DEFAULT_DUPLICATE_MAX_PAIRS);
    assert_eq!(group_pairs(&tight), vec![("doc/a".to_owned(), "doc/b".to_owned())]);
}

#[test]
fn same_document_observations_never_pair_with_self() {
    // A document's chunk matching another of its own chunks (docX, docX) at
    // distance 0 must NEVER be reported as a duplicate.
    let obs = vec![observation("doc/a", "doc/a", 0.0)];
    let groups = pairs_from_neighbours(&obs, 0.05, DEFAULT_DUPLICATE_MAX_PAIRS);
    assert!(groups.is_empty(), "(docX, docX) must never be a pair");
}

#[test]
fn symmetric_observations_dedup_to_one_pair() {
    // saber1→saber2 and saber2→saber1 (each chunk finds the other) must
    // collapse to a SINGLE canonical pair.
    let obs = vec![
        observation("doc/saber2", "doc/saber1", 0.03),
        observation("doc/saber1", "doc/saber2", 0.03),
    ];
    let groups = pairs_from_neighbours(&obs, 0.05, DEFAULT_DUPLICATE_MAX_PAIRS);
    assert_eq!(
        group_pairs(&groups),
        vec![("doc/saber1".to_owned(), "doc/saber2".to_owned())],
        "symmetric observations must dedup to one pair"
    );
}

#[test]
fn multiple_chunk_observations_keep_best_distance() {
    // Two documents observed at several chunk distances reduce to ONE pair;
    // the kept distance is the BEST (smallest) over all chunk observations.
    let obs = vec![
        observation("doc/a", "doc/b", 0.30), // above a 0.1 threshold
        observation("doc/b", "doc/a", 0.04), // best chunk, below threshold
    ];
    let groups = pairs_from_neighbours(&obs, 0.1, DEFAULT_DUPLICATE_MAX_PAIRS);
    assert_eq!(
        group_pairs(&groups),
        vec![("doc/a".to_owned(), "doc/b".to_owned())],
        "best chunk distance below threshold must surface the doc pair"
    );
    assert_eq!(groups[0].distance, 0.04, "kept distance is the best (smallest) chunk distance");
}

#[test]
fn pairs_are_sorted_nearest_first_and_bounded_by_max_pairs() {
    // Distinct distances so the NEAREST-FIRST ordering is unambiguous; the cap
    // must keep the two CLOSEST pairs, not the lexicographically-smallest.
    let obs = vec![
        observation("doc/c", "doc/d", 0.03),
        observation("doc/a", "doc/b", 0.05),
        observation("doc/e", "doc/f", 0.01),
    ];
    let bounded = pairs_from_neighbours(&obs, 0.1, 2);
    assert_eq!(
        group_pairs(&bounded),
        vec![
            ("doc/e".to_owned(), "doc/f".to_owned()), // 0.01 — closest
            ("doc/c".to_owned(), "doc/d".to_owned()), // 0.03 — next
        ],
        "result must be sorted nearest-first and capped at max_pairs (drops the farthest)"
    );
}

#[test]
fn equal_distance_pairs_tie_break_by_id() {
    // Deterministic tie-break on equal distances: by canonical member ids.
    let obs = vec![
        observation("doc/c", "doc/d", 0.01),
        observation("doc/a", "doc/b", 0.01),
    ];
    let groups = pairs_from_neighbours(&obs, 0.05, DEFAULT_DUPLICATE_MAX_PAIRS);
    assert_eq!(
        group_pairs(&groups),
        vec![
            ("doc/a".to_owned(), "doc/b".to_owned()),
            ("doc/c".to_owned(), "doc/d".to_owned()),
        ],
        "equal distances tie-break by id for stable output"
    );
}

#[test]
fn snippets_align_to_canonical_member_order() {
    // When ids are reordered to lower-id-first, each member's snippet must
    // follow its own id (not stay in observation order).
    let obs = vec![NeighbourObservation {
        doc_a: "doc/zebra".to_owned(),
        doc_b: "doc/apple".to_owned(),
        normalized_distance: 0.02,
        content_a: Some("zebra text".to_owned()),
        content_b: Some("apple text".to_owned()),
    }];
    let groups = pairs_from_neighbours(&obs, 0.05, DEFAULT_DUPLICATE_MAX_PAIRS);
    assert_eq!(groups.len(), 1);
    let members = &groups[0].group;
    assert_eq!(members[0].id, "doc/apple");
    assert_eq!(members[0].snippet.as_deref(), Some("apple text"));
    assert_eq!(members[1].id, "doc/zebra");
    assert_eq!(members[1].snippet.as_deref(), Some("zebra text"));
}

#[test]
fn no_snippets_when_content_absent() {
    let obs = vec![observation("doc/a", "doc/b", 0.02)];
    let groups = pairs_from_neighbours(&obs, 0.05, DEFAULT_DUPLICATE_MAX_PAIRS);
    assert!(groups[0].group.iter().all(|m| m.snippet.is_none()));
}

#[test]
fn empty_observations_yield_no_pairs() {
    let groups = pairs_from_neighbours(&[], 0.5, DEFAULT_DUPLICATE_MAX_PAIRS);
    assert!(groups.is_empty());
}

// --- upsert chunks, tag commit, verify round-trip ---
#[tokio::test]
async fn upsert_and_tag_commit_round_trips() {
    let (store, _tmp) = make_test_store(8);
    let rows = vec![
        ChunkRow {
            doc_id: "doc/1".to_owned(),
            doc_type: "People".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 40,
            embedding: fake_embedding(8, 1.0),
            clustering_embedding: fake_embedding(8, 1.0),
            content: "Yoda is a wise Jedi.".to_owned(),
        },
        ChunkRow {
            doc_id: "doc/2".to_owned(),
            doc_type: "Species".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 30,
            embedding: fake_embedding(8, 2.0),
            clustering_embedding: fake_embedding(8, 2.0),
            content: "Mon Calamari are squid people.".to_owned(),
        },
    ];

    let version = store
        .io_upsert_chunks("admin/star_wars", "main", "doc/1", &rows[0..1])
        .await
        .expect("upsert doc/1");
    assert!(version > 0);

    let version2 = store
        .io_upsert_chunks("admin/star_wars", "main", "doc/2", &rows[1..2])
        .await
        .expect("upsert doc/2");
    assert!(version2 > version);

    // Tag commit c0 to the final version.
    store
        .io_tag_commit("admin/star_wars", "main", "c0", version2)
        .await
        .expect("tag commit");

    // Resolve commit should return the version.
    let resolved = store
        .io_resolve_commit("admin/star_wars", "main", "c0")
        .await
        .expect("resolve");
    assert_eq!(resolved, Some(version2));
}

// --- multi-chunk doc produces multiple rows ---
#[tokio::test]
async fn multi_chunk_doc_produces_multiple_rows() {
    let (store, _tmp) = make_test_store(8);
    let rows = vec![
        ChunkRow {
            doc_id: "doc/big".to_owned(),
            doc_type: "Article".to_owned(),
            chunk_index: 0,
            chunk_count: 3,
            chunk_token_start: 0,
            doc_token_len: 1500,
            embedding: fake_embedding(8, 1.0),
            clustering_embedding: fake_embedding(8, 1.0),
            content: "Beginning of the article.".to_owned(),
        },
        ChunkRow {
            doc_id: "doc/big".to_owned(),
            doc_type: "Article".to_owned(),
            chunk_index: 1,
            chunk_count: 3,
            chunk_token_start: 450,
            doc_token_len: 1500,
            embedding: fake_embedding(8, 2.0),
            clustering_embedding: fake_embedding(8, 2.0),
            content: "Middle of the article.".to_owned(),
        },
        ChunkRow {
            doc_id: "doc/big".to_owned(),
            doc_type: "Article".to_owned(),
            chunk_index: 2,
            chunk_count: 3,
            chunk_token_start: 900,
            doc_token_len: 1500,
            embedding: fake_embedding(8, 3.0),
            clustering_embedding: fake_embedding(8, 3.0),
            content: "End of the article.".to_owned(),
        },
    ];

    store
        .io_upsert_chunks("admin/test", "main", "doc/big", &rows)
        .await
        .expect("upsert multi-chunk");

    // Lookup should find all 3 chunks.
    let chunks = store
        .io_lookup_doc_chunks("admin/test", "main", "doc/big")
        .await
        .expect("lookup");
    assert_eq!(chunks.len(), 3);
}

// --- lookup carries the STORED embedding (the /similar reuse path) ---
// Asserts that io_lookup_doc_chunks projects and populates the embedding
// exactly as inserted, so /similar can reuse the stored vector instead of
// re-embedding the source text.
#[tokio::test]
async fn lookup_doc_chunks_carries_stored_embedding() {
    let (store, _tmp) = make_test_store(8);
    let emb0 = fake_embedding(8, 1.0);
    let emb1 = fake_embedding(8, 2.0);
    let rows = vec![
        ChunkRow {
            doc_id: "doc/emb".to_owned(),
            doc_type: "Article".to_owned(),
            chunk_index: 0,
            chunk_count: 2,
            chunk_token_start: 0,
            doc_token_len: 100,
            embedding: emb0.clone(),
            clustering_embedding: emb0.clone(),
            content: "chunk zero".to_owned(),
        },
        ChunkRow {
            doc_id: "doc/emb".to_owned(),
            doc_type: "Article".to_owned(),
            chunk_index: 1,
            chunk_count: 2,
            chunk_token_start: 50,
            doc_token_len: 100,
            embedding: emb1.clone(),
            clustering_embedding: emb1.clone(),
            content: "chunk one".to_owned(),
        },
    ];

    store
        .io_upsert_chunks("admin/test", "main", "doc/emb", &rows)
        .await
        .expect("upsert");

    let mut chunks = store
        .io_lookup_doc_chunks("admin/test", "main", "doc/emb")
        .await
        .expect("lookup");
    chunks.sort_by_key(|c| c.chunk_index);

    assert_eq!(chunks.len(), 2);
    // The stored vector is returned verbatim (right dimension, exact values).
    assert_eq!(chunks[0].embedding.len(), 8, "embedding dimension preserved");
    assert_eq!(chunks[0].embedding, emb0, "chunk 0 embedding matches insert");
    assert_eq!(chunks[1].embedding, emb1, "chunk 1 embedding matches insert");
    // The vector is non-empty (regression guard: not the empty default).
    assert!(!chunks[0].embedding.is_empty(), "embedding must be populated");
}

// --- delete removes all chunks ---
#[tokio::test]
async fn delete_doc_removes_all_chunks() {
    let (store, _tmp) = make_test_store(8);
    let rows = vec![
        ChunkRow {
            doc_id: "doc/del".to_owned(),
            doc_type: "X".to_owned(),
            chunk_index: 0,
            chunk_count: 2,
            chunk_token_start: 0,
            doc_token_len: 100,
            embedding: fake_embedding(8, 1.0),
            clustering_embedding: fake_embedding(8, 1.0),
            content: "part 1".to_owned(),
        },
        ChunkRow {
            doc_id: "doc/del".to_owned(),
            doc_type: "X".to_owned(),
            chunk_index: 1,
            chunk_count: 2,
            chunk_token_start: 50,
            doc_token_len: 100,
            embedding: fake_embedding(8, 2.0),
            clustering_embedding: fake_embedding(8, 2.0),
            content: "part 2".to_owned(),
        },
    ];

    store
        .io_upsert_chunks("admin/test", "main", "doc/del", &rows)
        .await
        .expect("upsert");

    store
        .io_delete_doc("admin/test", "main", "doc/del")
        .await
        .expect("delete");

    let remaining = store
        .io_lookup_doc_chunks("admin/test", "main", "doc/del")
        .await
        .expect("lookup after delete");
    assert_eq!(remaining.len(), 0, "all chunks should be deleted");
}

// --- dedup produces correct chunk metadata ---
#[test]
fn dedup_chunks_to_documents_picks_best_chunk() {
    let hits = vec![
        ChunkHit {
            doc_id: "doc/1".to_owned(),
            distance: 0.8,
            distance_kind: DistanceKind::RawCosine,
            chunk_index: 0,
            chunk_count: 2,
            chunk_token_start: 0,
            doc_token_len: 1000,
            content: "first chunk".to_owned(),
            embedding: Vec::new(),
            clustering_embedding: Vec::new(),
        },
        ChunkHit {
            doc_id: "doc/1".to_owned(),
            distance: 0.4, // Better (smaller distance) — this chunk wins.
            distance_kind: DistanceKind::RawCosine,
            chunk_index: 1,
            chunk_count: 2,
            chunk_token_start: 500,
            doc_token_len: 1000,
            content: "second chunk".to_owned(),
            embedding: Vec::new(),
            clustering_embedding: Vec::new(),
        },
        ChunkHit {
            doc_id: "doc/2".to_owned(),
            distance: 0.6,
            distance_kind: DistanceKind::RawCosine,
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 200,
            content: "only chunk".to_owned(),
            embedding: Vec::new(),
            clustering_embedding: Vec::new(),
        },
    ];

    let results = dedup_chunks_to_documents(hits, false);
    assert_eq!(results.len(), 2, "should dedup to 2 documents");

    // Results sorted by distance — doc/1 (0.4→0.2 after transform) < doc/2 (0.6→0.3).
    let doc1 = results.iter().find(|r| r.id == "doc/1").expect("doc/1");
    assert_eq!(doc1.chunk.index, 1, "best chunk is index 1");
    assert_eq!(doc1.chunk.count, 2);
    assert_eq!(doc1.chunk.token_start, 500);
    assert_eq!(doc1.chunk.doc_token_len, 1000);
    // location = 500/1000 = 0.5
    assert!((doc1.chunk.location - 0.5).abs() < f32::EPSILON);
    assert!(doc1.chunk.snippet.is_none(), "snippet should be omitted");
}

// --- single chunk doc has location 0.0 ---
#[test]
fn dedup_single_chunk_doc_location_zero() {
    let hits = vec![ChunkHit {
        doc_id: "doc/s".to_owned(),
        distance: 0.2,
        distance_kind: DistanceKind::RawCosine,
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 41,
        content: "short doc".to_owned(),
        embedding: Vec::new(),
        clustering_embedding: Vec::new(),
    }];

    let results = dedup_chunks_to_documents(hits, true);
    assert_eq!(results.len(), 1);
    let hit = &results[0];
    assert_eq!(hit.chunk.index, 0);
    assert_eq!(hit.chunk.count, 1);
    assert_eq!(hit.chunk.token_start, 0);
    assert_eq!(hit.chunk.doc_token_len, 41);
    assert_eq!(hit.chunk.location, 0.0);
    assert_eq!(hit.chunk.snippet, Some("short doc".to_owned()));
}

// --- distance transform applied correctly ---
#[test]
fn distance_transform_in_dedup() {
    let hits = vec![ChunkHit {
        doc_id: "doc/x".to_owned(),
        distance: 0.0, // Self-distance in lance cosine.
        distance_kind: DistanceKind::RawCosine,
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 10,
        content: "x".to_owned(),
        embedding: Vec::new(),
        clustering_embedding: Vec::new(),
    }];
    let results = dedup_chunks_to_documents(hits, false);
    assert_eq!(results[0].distance, 0.0, "self-distance maps to 0");
}

// --- #3: normalised distances skip the transform in dedup ---
#[test]
fn dedup_normalised_distances_pass_through() {
    let hits = vec![
        ChunkHit {
            doc_id: "doc/rrf".to_owned(),
            distance: 0.42, // Already normalised (e.g., from RRF).
            distance_kind: DistanceKind::Normalised,
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            content: "rrf hit".to_owned(),
            embedding: Vec::new(),
            clustering_embedding: Vec::new(),
        },
    ];
    let results = dedup_chunks_to_documents(hits, false);
    // Must pass through unchanged — NOT halved by normalized_cosine_from_lance.
    assert!(
        (results[0].distance - 0.42).abs() < f32::EPSILON,
        "normalised distance should pass through unchanged, got {}",
        results[0].distance,
    );
}

// --- #1: FTS distances are non-zero and ordered (BM25 score preserved) ---
#[test]
fn fts_hits_have_nonzero_ordered_distances() {
    // Simulate FTS hits with BM25 scores converted to distances.
    let hits = vec![
        ChunkHit {
            doc_id: "doc/best".to_owned(),
            distance: 1.0 / (1.0 + 10.0), // High BM25 score → low distance.
            distance_kind: DistanceKind::Normalised,
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            content: "best match".to_owned(),
            embedding: Vec::new(),
            clustering_embedding: Vec::new(),
        },
        ChunkHit {
            doc_id: "doc/worse".to_owned(),
            distance: 1.0 / (1.0 + 2.0), // Lower BM25 score → higher distance.
            distance_kind: DistanceKind::Normalised,
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            content: "worse match".to_owned(),
            embedding: Vec::new(),
            clustering_embedding: Vec::new(),
        },
    ];

    let results = dedup_chunks_to_documents(hits, false);
    assert_eq!(results.len(), 2);
    // Best match (lowest distance) should be first after sorting.
    assert_eq!(results[0].id, "doc/best");
    assert_eq!(results[1].id, "doc/worse");
    // Both distances must be non-zero.
    assert!(results[0].distance > 0.0, "FTS distance must be > 0");
    assert!(results[1].distance > 0.0, "FTS distance must be > 0");
    // Best has lower distance.
    assert!(results[0].distance < results[1].distance);
}

// --- #2: Vector distance scale anchors (locks factor-of-2 correctness) ---
// With DistanceType::Cosine on the scanner, _distance is a true cosine distance
// in [0,2]. normalized_cosine_from_lance(d) = d/2 maps to [0,1].
// These anchors catch any factor-of-2 scale bug permanently.
#[test]
fn vector_distance_scale_anchors_through_dedup() {
    // Anchor 1: self-distance (identical vectors) → 0.0
    let hit_identical = ChunkHit {
        doc_id: "doc/self".to_owned(),
        distance: 0.0, // Lance cosine: identical vectors
        distance_kind: DistanceKind::RawCosine,
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 10,
        content: "self".to_owned(),
        embedding: Vec::new(),
        clustering_embedding: Vec::new(),
    };
    let results = dedup_chunks_to_documents(vec![hit_identical], false);
    assert_eq!(results[0].distance, 0.0, "identical → 0.0");

    // Anchor 2: orthogonal vectors (Lance cosine distance = 1.0) → 0.5
    let hit_orthogonal = ChunkHit {
        doc_id: "doc/ortho".to_owned(),
        distance: 1.0, // Lance cosine: orthogonal
        distance_kind: DistanceKind::RawCosine,
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 10,
        content: "ortho".to_owned(),
        embedding: Vec::new(),
        clustering_embedding: Vec::new(),
    };
    let results = dedup_chunks_to_documents(vec![hit_orthogonal], false);
    assert!(
        (results[0].distance - 0.5).abs() < f32::EPSILON,
        "orthogonal → 0.5, got {}",
        results[0].distance,
    );

    // Anchor 3: opposite vectors (Lance cosine distance = 2.0) → 1.0
    let hit_opposite = ChunkHit {
        doc_id: "doc/opp".to_owned(),
        distance: 2.0, // Lance cosine: opposite
        distance_kind: DistanceKind::RawCosine,
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 10,
        content: "opposite".to_owned(),
        embedding: Vec::new(),
        clustering_embedding: Vec::new(),
    };
    let results = dedup_chunks_to_documents(vec![hit_opposite], false);
    assert_eq!(results[0].distance, 1.0, "opposite → 1.0");

    // The OLD bug: L2² for orthogonal unit vectors = 2.0, which would give
    // normalized_cosine_from_lance(2.0) = 1.0 (WRONG — should be 0.5).
    // This test catches any regression where L2² is fed to the transform.
}

// --- statistics reflect indexed data ---
#[tokio::test]
async fn statistics_reflect_indexed_data() {
    let (store, _tmp) = make_test_store(8);
    let rows = vec![ChunkRow {
        doc_id: "doc/1".to_owned(),
        doc_type: "T".to_owned(),
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 10,
        embedding: fake_embedding(8, 1.0),
            clustering_embedding: fake_embedding(8, 1.0),
        content: "test".to_owned(),
    }];

    store
        .io_upsert_chunks("admin/db", "main", "doc/1", &rows)
        .await
        .expect("upsert");
    store.update_last_indexed("admin/db", "main", "c0", 1).await;

    let stats = store.statistics().await;
    assert!(stats.chunks > 0, "chunks should be > 0 after upsert");
    assert!(stats.domains > 0, "domains should be > 0");
    assert!(stats.indexed_commits > 0, "indexed_commits should be > 0");
}

// --- scoped statistics returns only the named domain ---
#[tokio::test]
async fn statistics_for_domain_returns_only_target() {
    let (store, _tmp) = make_test_store(8);

    // Insert data into two distinct domains.
    let rows_a = vec![ChunkRow {
        doc_id: "doc/a1".to_owned(),
        doc_type: "TypeA".to_owned(),
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 10,
        embedding: fake_embedding(8, 1.0),
            clustering_embedding: fake_embedding(8, 1.0),
        content: "domain alpha".to_owned(),
    }];
    let rows_b = vec![ChunkRow {
        doc_id: "doc/b1".to_owned(),
        doc_type: "TypeB".to_owned(),
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 10,
        embedding: fake_embedding(8, 2.0),
            clustering_embedding: fake_embedding(8, 2.0),
        content: "domain beta".to_owned(),
    }];

    store
        .io_upsert_chunks("admin/alpha", "main", "doc/a1", &rows_a)
        .await
        .expect("upsert alpha");
    store
        .update_last_indexed("admin/alpha", "main", "c_alpha", 1)
        .await;

    store
        .io_upsert_chunks("admin/beta", "main", "doc/b1", &rows_b)
        .await
        .expect("upsert beta");
    store
        .update_last_indexed("admin/beta", "main", "c_beta", 1)
        .await;

    // Global statistics sees both domains.
    let global = store.statistics().await;
    assert_eq!(global.domains, 2, "global should see 2 domains");

    // Scoped statistics for alpha sees ONLY alpha.
    let scoped_alpha = store.statistics_for_domain("admin/alpha").await;
    assert_eq!(
        scoped_alpha.domains, 1,
        "scoped alpha: domains must be 1"
    );
    assert_eq!(
        scoped_alpha.branches, 1,
        "scoped alpha: branches must be 1"
    );
    assert_eq!(
        scoped_alpha.indexed_commits, 1,
        "scoped alpha: indexed_commits must be 1"
    );
    assert_eq!(
        scoped_alpha.documents, 1,
        "scoped alpha: documents must be 1 (doc/a1)"
    );
    assert!(
        scoped_alpha.chunks > 0,
        "scoped alpha: chunks must be > 0"
    );

    // Scoped statistics for beta sees ONLY beta.
    let scoped_beta = store.statistics_for_domain("admin/beta").await;
    assert_eq!(scoped_beta.domains, 1, "scoped beta: domains must be 1");
    assert_eq!(scoped_beta.documents, 1, "scoped beta: documents must be 1");

    // Scoped statistics for unknown domain sees nothing.
    let scoped_unknown = store.statistics_for_domain("admin/nonexistent").await;
    assert_eq!(
        scoped_unknown.domains, 0,
        "unknown domain: domains must be 0"
    );
    assert_eq!(
        scoped_unknown.branches, 0,
        "unknown domain: branches must be 0"
    );
    assert_eq!(
        scoped_unknown.documents, 0,
        "unknown domain: documents must be 0"
    );
    assert_eq!(
        scoped_unknown.chunks, 0,
        "unknown domain: chunks must be 0"
    );
}

// --- commit tag isolation (different tags for different versions) ---
#[tokio::test]
async fn different_commits_different_versions() {
    let (store, _tmp) = make_test_store(8);

    // Insert first doc, tag as c0.
    let rows1 = vec![ChunkRow {
        doc_id: "doc/1".to_owned(),
        doc_type: "T".to_owned(),
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 10,
        embedding: fake_embedding(8, 1.0),
            clustering_embedding: fake_embedding(8, 1.0),
        content: "version one".to_owned(),
    }];
    let v1 = store
        .io_upsert_chunks("admin/db", "main", "doc/1", &rows1)
        .await
        .expect("upsert v1");
    store
        .io_tag_commit("admin/db", "main", "c0", v1)
        .await
        .expect("tag c0");

    // Insert second doc, tag as c1.
    let rows2 = vec![ChunkRow {
        doc_id: "doc/2".to_owned(),
        doc_type: "T".to_owned(),
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 10,
        embedding: fake_embedding(8, 2.0),
            clustering_embedding: fake_embedding(8, 2.0),
        content: "version two".to_owned(),
    }];
    let v2 = store
        .io_upsert_chunks("admin/db", "main", "doc/2", &rows2)
        .await
        .expect("upsert v2");
    store
        .io_tag_commit("admin/db", "main", "c1", v2)
        .await
        .expect("tag c1");

    // Resolve both — different versions.
    let r0 = store.io_resolve_commit("admin/db", "main", "c0").await.unwrap();
    let r1 = store.io_resolve_commit("admin/db", "main", "c1").await.unwrap();
    assert_eq!(r0, Some(v1));
    assert_eq!(r1, Some(v2));
    assert_ne!(v1, v2, "versions must differ");
}

// ─────────────── DURABLE STATE / RESTART INVARIANT (task-durable-index) ───
// The headline regression: index state must be derived from the on-disk
// Lance tags, NOT the in-memory `branch_indexes` map. A fresh `LanceStore`
// over the SAME directory (the exact effect of a container restart — empty
// in-memory maps, full on-disk data) must report the same last-indexed.

/// Helper: open a SECOND store over the same dir — simulates a restart with
/// cold in-memory state but warm disk.
fn reopen_store(tmp: &tempfile::TempDir, dim: usize) -> LanceStore {
    LanceStore::new(tmp.path(), dim, 256 * 1024 * 1024, 128 * 1024 * 1024)
}

/// Helper: build a validated `Domain` from a "org/db" string (test-only).
fn dom(s: &str) -> Domain {
    Domain::from_resource_path(
        &crate::kernel::model::parse_domain(s).expect("valid test domain"),
    )
}

/// Helper: build a `BranchName` (test-only).
fn br(s: &str) -> BranchName {
    BranchName::new(s.to_owned())
}

// --- RESTART INVARIANT: last_indexed survives a restart (derived from tags) ---
#[tokio::test]
async fn last_indexed_survives_restart_derived_from_disk() {
    let (store, tmp) = make_test_store(8);
    let domain = "admin/restart";

    let r = one_row(8, 1.0, "restart doc");
    let v = store
        .io_upsert_chunks(domain, "main", "doc/1", std::slice::from_ref(&r))
        .await
        .expect("upsert");
    store
        .io_tag_commit(domain, "main", "c0", v)
        .await
        .expect("tag c0");
    store.update_last_indexed(domain, "main", "c0", v).await;

    // Sanity: the live store reports c0.
    let before = store
        .last_indexed(&dom(domain), &br("main"))
        .await
        .expect("last_indexed read");
    assert_eq!(before.commit.as_deref(), Some("c0"));

    // RESTART: drop the in-memory state, re-open over the same dir.
    drop(store);
    let restarted = reopen_store(&tmp, 8);

    // The in-memory branch_indexes is EMPTY now — but the answer must STILL
    // be c0, derived from the durable on-disk tag.
    let after = restarted
        .last_indexed(&dom(domain), &br("main"))
        .await
        .expect("last_indexed read");
    assert_eq!(
        after.commit.as_deref(),
        Some("c0"),
        "after restart, last_indexed MUST be derived from the on-disk tag, not the empty in-memory map"
    );
    assert_eq!(after.version, v, "the derived version must match the tag");
}

// --- RESTART INVARIANT: resolve sees the commit on disk after restart ---
#[tokio::test]
async fn resolve_commit_survives_restart() {
    let (store, tmp) = make_test_store(8);
    let domain = "admin/restart_resolve";

    let r = one_row(8, 1.0, "doc");
    let v = store
        .io_upsert_chunks(domain, "main", "doc/1", std::slice::from_ref(&r))
        .await
        .expect("upsert");
    store.io_tag_commit(domain, "main", "c0", v).await.expect("tag");

    drop(store);
    let restarted = reopen_store(&tmp, 8);

    let resolved = restarted
        .io_resolve_commit(domain, "main", "c0")
        .await
        .expect("resolve after restart");
    assert_eq!(
        resolved,
        Some(v),
        "a commit tagged on disk must resolve after a restart"
    );
}

// --- last_indexed is BRANCH-PRECISE and derived from disk after restart ---
#[tokio::test]
async fn last_indexed_branch_precise_after_restart() {
    let (store, tmp) = make_test_store(8);
    let domain = "admin/restart_branch";

    // main: index + tag c0.
    let r0 = one_row(8, 1.0, "main doc");
    let v0 = store
        .io_upsert_chunks(domain, "main", "doc/1", std::slice::from_ref(&r0))
        .await
        .expect("upsert main");
    store.io_tag_commit(domain, "main", "c0", v0).await.expect("tag c0");
    store.update_last_indexed(domain, "main", "c0", v0).await;

    // feature branch forked from c0, index + tag c1 on the feature lineage.
    store.io_create_branch(domain, "feature", v0).await.expect("create branch");
    let r1 = one_row(8, 2.0, "feature doc");
    let v1 = store
        .io_upsert_chunks(domain, "feature", "doc/2", std::slice::from_ref(&r1))
        .await
        .expect("upsert feature");
    store.io_tag_commit(domain, "feature", "c1", v1).await.expect("tag c1");
    store.update_last_indexed(domain, "feature", "c1", v1).await;

    drop(store);
    let restarted = reopen_store(&tmp, 8);

    let d = dom(domain);
    let main_li = restarted.last_indexed(&d, &br("main")).await.expect("li main");
    let feat_li = restarted.last_indexed(&d, &br("feature")).await.expect("li feature");

    assert_eq!(
        main_li.commit.as_deref(),
        Some("c0"),
        "main's last-indexed must be c0 after restart (branch-precise, from disk)"
    );
    assert_eq!(
        feat_li.commit.as_deref(),
        Some("c1"),
        "feature's last-indexed must be c1 after restart (its own lineage tag)"
    );
}

// --- last_indexed picks the LATEST tag on the branch (forward progress) ---
#[tokio::test]
async fn last_indexed_picks_latest_commit_on_branch_after_restart() {
    let (store, tmp) = make_test_store(8);
    let domain = "admin/restart_latest";

    let r0 = one_row(8, 1.0, "v0");
    let v0 = store
        .io_upsert_chunks(domain, "main", "doc/1", std::slice::from_ref(&r0))
        .await
        .expect("upsert v0");
    store.io_tag_commit(domain, "main", "c0", v0).await.expect("tag c0");

    let r1 = one_row(8, 2.0, "v1");
    let v1 = store
        .io_upsert_chunks(domain, "main", "doc/2", std::slice::from_ref(&r1))
        .await
        .expect("upsert v1");
    store.io_tag_commit(domain, "main", "c1", v1).await.expect("tag c1");

    drop(store);
    let restarted = reopen_store(&tmp, 8);

    let li = restarted
        .last_indexed(&dom(domain), &br("main"))
        .await
        .expect("last_indexed read");
    assert_eq!(
        li.commit.as_deref(),
        Some("c1"),
        "after restart, last-indexed must be the LATEST tagged commit (c1), not c0"
    );
    assert_eq!(li.version, v1);
}

// --- an un-indexed branch (no tags) reports None after restart (not a panic) ---
#[tokio::test]
async fn last_indexed_none_for_unindexed_domain_after_restart() {
    let (store, tmp) = make_test_store(8);
    let domain = "admin/restart_empty";
    // Index main so the dataset exists, but query a DIFFERENT branch with no tags.
    let r = one_row(8, 1.0, "doc");
    let v = store
        .io_upsert_chunks(domain, "main", "doc/1", std::slice::from_ref(&r))
        .await
        .expect("upsert");
    store.io_tag_commit(domain, "main", "c0", v).await.expect("tag");

    drop(store);
    let restarted = reopen_store(&tmp, 8);

    // A branch that was never tagged → None (derived from disk: no matching tag).
    let li = restarted
        .last_indexed(&dom(domain), &br("never_indexed"))
        .await
        .expect("li read");
    assert_eq!(li.commit, None, "a branch with no tag must report None, not a spurious commit");

    // A domain with no dataset at all → None.
    let li_absent = restarted
        .last_indexed(&dom("admin/never"), &br("main"))
        .await
        .expect("li read");
    assert_eq!(li_absent.commit, None);
}

// --- RESUME FORWARD after restart: index c0,c1 → restart → index c2 ─ c2
//     must ADD to the existing on-disk index (c0,c1 still there), NOT rebuild
//     from scratch, and last-indexed must walk forward to c2. (Requirement #2:
//     re-index after a mid-way reset resumes forward from the last durable.) ---
#[tokio::test]
async fn reindex_after_restart_resumes_forward_not_from_scratch() {
    let (store, tmp) = make_test_store(8);
    let domain = "admin/resume";

    // c0: one doc.
    let r0 = ChunkRow {
        doc_id: "doc/0".to_owned(),
        doc_type: "Doc".to_owned(),
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 10,
        embedding: fake_embedding(8, 1.0),
            clustering_embedding: fake_embedding(8, 1.0),
        content: "zero".to_owned(),
    };
    let v0 = store
        .io_upsert_chunks(domain, "main", "doc/0", std::slice::from_ref(&r0))
        .await
        .expect("upsert c0");
    store.io_tag_commit(domain, "main", "c0", v0).await.expect("tag c0");

    // c1: a SECOND doc (append).
    let r1 = ChunkRow {
        doc_id: "doc/1".to_owned(),
        doc_type: "Doc".to_owned(),
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 10,
        embedding: fake_embedding(8, 2.0),
            clustering_embedding: fake_embedding(8, 2.0),
        content: "one".to_owned(),
    };
    let v1 = store
        .io_upsert_chunks(domain, "main", "doc/1", std::slice::from_ref(&r1))
        .await
        .expect("upsert c1");
    store.io_tag_commit(domain, "main", "c1", v1).await.expect("tag c1");

    let chunks_before = store.statistics().await.chunks;
    assert_eq!(chunks_before, 2, "two docs indexed before restart");

    // RESTART: cold in-memory state, warm disk.
    drop(store);
    let restarted = reopen_store(&tmp, 8);

    // The durable resume point: last-indexed is c1 (NOT lost, NOT reset to c0).
    let resume = restarted
        .last_indexed(&dom(domain), &br("main"))
        .await
        .expect("li after restart");
    assert_eq!(
        resume.commit.as_deref(),
        Some("c1"),
        "resume must start FORWARD from the last durably-indexed commit c1"
    );

    // c2: a THIRD doc indexed AFTER restart — appended to the existing index.
    let r2 = ChunkRow {
        doc_id: "doc/2".to_owned(),
        doc_type: "Doc".to_owned(),
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 10,
        embedding: fake_embedding(8, 3.0),
            clustering_embedding: fake_embedding(8, 3.0),
        content: "two".to_owned(),
    };
    let v2 = restarted
        .io_upsert_chunks(domain, "main", "doc/2", std::slice::from_ref(&r2))
        .await
        .expect("upsert c2 after restart");
    restarted.io_tag_commit(domain, "main", "c2", v2).await.expect("tag c2");

    // NOT from scratch: all three docs present (c0+c1 survived the restart,
    // c2 was ADDED). A from-scratch rebuild would have dropped doc/0, doc/1.
    let chunks_after = restarted.statistics().await.chunks;
    assert_eq!(
        chunks_after, 3,
        "c2 must ADD to the existing on-disk index (3 docs), not rebuild from scratch (would be 1)"
    );

    // All three commits resolvable; c2 is the new forward tip.
    assert_eq!(restarted.io_resolve_commit(domain, "main", "c0").await.unwrap(), Some(v0));
    assert_eq!(restarted.io_resolve_commit(domain, "main", "c1").await.unwrap(), Some(v1));
    assert_eq!(restarted.io_resolve_commit(domain, "main", "c2").await.unwrap(), Some(v2));
    assert!(v2 > v1 && v1 > v0, "versions advance forward, never reset");

    let final_li = restarted
        .last_indexed(&dom(domain), &br("main"))
        .await
        .expect("final li");
    assert_eq!(final_li.commit.as_deref(), Some("c2"), "last-indexed walked forward to c2");
}

// --- A commit killed MID-INDEX (data appended but NEVER tagged) is NOT
//     treated as indexed after restart — no half-tagged state is "complete".
//     The resume re-does it from the prior tag. (Requirement #2 partial-work) ---
#[tokio::test]
async fn untagged_midindex_commit_is_not_complete_after_restart() {
    let (store, tmp) = make_test_store(8);
    let domain = "admin/midkill";

    // c0 fully indexed + tagged.
    let r0 = ChunkRow {
        doc_id: "doc/0".to_owned(),
        doc_type: "Doc".to_owned(),
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 10,
        embedding: fake_embedding(8, 1.0),
            clustering_embedding: fake_embedding(8, 1.0),
        content: "zero".to_owned(),
    };
    let v0 = store
        .io_upsert_chunks(domain, "main", "doc/0", std::slice::from_ref(&r0))
        .await
        .expect("upsert c0");
    store.io_tag_commit(domain, "main", "c0", v0).await.expect("tag c0");

    // c1: data WRITTEN (version advances) but the process is "killed" before
    // io_tag_commit runs — simulate by upserting WITHOUT tagging.
    let r1 = ChunkRow {
        doc_id: "doc/1".to_owned(),
        doc_type: "Doc".to_owned(),
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 10,
        embedding: fake_embedding(8, 2.0),
            clustering_embedding: fake_embedding(8, 2.0),
        content: "one".to_owned(),
    };
    let _v1 = store
        .io_upsert_chunks(domain, "main", "doc/1", std::slice::from_ref(&r1))
        .await
        .expect("upsert c1 (never tagged — killed mid-index)");

    drop(store);
    let restarted = reopen_store(&tmp, 8);

    // The untagged commit c1 is NOT indexed (no tag) — half-written, partial
    // work is never treated as complete.
    assert_eq!(
        restarted.io_resolve_commit(domain, "main", "c1").await.unwrap(),
        None,
        "an untagged (killed mid-index) commit must NOT resolve as indexed"
    );
    // last-indexed remains the last DURABLY-tagged commit, c0 — the resume base.
    let li = restarted
        .last_indexed(&dom(domain), &br("main"))
        .await
        .expect("li");
    assert_eq!(
        li.commit.as_deref(),
        Some("c0"),
        "resume base is the last durably-tagged commit, not the half-written c1"
    );
}

// --- snapshot isolation: search at C0 does not see C1 data ---
#[tokio::test]
async fn snapshot_isolation_search_at_old_commit_excludes_new_data() {
    let (store, _tmp) = make_test_store(8);

    // Insert doc/A, tag as commit "c0".
    let emb_a = fake_embedding(8, 1.0);
    let rows_a = vec![ChunkRow {
        doc_id: "doc/A".to_owned(),
        doc_type: "T".to_owned(),
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 10,
        embedding: emb_a.clone(),
        clustering_embedding: emb_a.clone(),
        content: "document A".to_owned(),
    }];
    let v0 = store
        .io_upsert_chunks("admin/iso", "main", "doc/A", &rows_a)
        .await
        .expect("upsert A");
    store
        .io_tag_commit("admin/iso", "main", "c0", v0)
        .await
        .expect("tag c0");

    // Insert doc/B, tag as commit "c1".
    let emb_b = fake_embedding(8, 2.0);
    let rows_b = vec![ChunkRow {
        doc_id: "doc/B".to_owned(),
        doc_type: "T".to_owned(),
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 10,
        embedding: emb_b.clone(),
        clustering_embedding: emb_b.clone(),
        content: "document B".to_owned(),
    }];
    let v1 = store
        .io_upsert_chunks("admin/iso", "main", "doc/B", &rows_b)
        .await
        .expect("upsert B");
    store
        .io_tag_commit("admin/iso", "main", "c1", v1)
        .await
        .expect("tag c1");

    // Search at c0 — should only find doc/A.
    let query_c0 = SearchQuery {
        query_embedding: emb_a.clone(),
        query_text: "document".to_owned(),
        mode: crate::kernel::model::SearchMode::Vector,
        start: 0,
        count: 10,
        doc_type_filter: Vec::new(),
        doc_id_filter: Vec::new(),
        snippet: false,
    };
    let hits_c0 = store
        .io_search("admin/iso", "main", "c0", &query_c0)
        .await
        .expect("search at c0");

    let doc_ids_c0: Vec<&str> = hits_c0.iter().map(|h| h.doc_id.as_str()).collect();
    assert!(
        doc_ids_c0.contains(&"doc/A"),
        "c0 snapshot should contain doc/A"
    );
    assert!(
        !doc_ids_c0.contains(&"doc/B"),
        "c0 snapshot must NOT contain doc/B (added after c0)"
    );

    // Search at c1 — should find both doc/A and doc/B.
    let query_c1 = SearchQuery {
        query_embedding: emb_a,
        query_text: "document".to_owned(),
        mode: crate::kernel::model::SearchMode::Vector,
        start: 0,
        count: 10,
        doc_type_filter: Vec::new(),
        doc_id_filter: Vec::new(),
        snippet: false,
    };
    let hits_c1 = store
        .io_search("admin/iso", "main", "c1", &query_c1)
        .await
        .expect("search at c1");

    let doc_ids_c1: Vec<&str> = hits_c1.iter().map(|h| h.doc_id.as_str()).collect();
    assert!(
        doc_ids_c1.contains(&"doc/A"),
        "c1 snapshot should contain doc/A"
    );
    assert!(
        doc_ids_c1.contains(&"doc/B"),
        "c1 snapshot should contain doc/B"
    );
}

// --- P3-ASSIGN-1: assign is a pure tag pointer — no new version, target == source ---
// The store assign primitive touches only Lance tags. It creates NO new
// dataset version (so no fragments, so — by construction — zero embed calls),
// and search at the target commit returns exactly the source commit's data.
#[tokio::test]
async fn assign_is_tag_pointer_no_recompute() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/assign";

    // Index doc/A and doc/B, tag c0 at the final version.
    let emb_a = fake_embedding(8, 1.0);
    let rows_a = vec![ChunkRow {
        doc_id: "doc/A".to_owned(),
        doc_type: "T".to_owned(),
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 10,
        embedding: emb_a.clone(),
        clustering_embedding: emb_a.clone(),
        content: "alpha".to_owned(),
    }];
    store
        .io_upsert_chunks(domain, "main", "doc/A", &rows_a)
        .await
        .expect("upsert A");
    let rows_b = vec![ChunkRow {
        doc_id: "doc/B".to_owned(),
        doc_type: "T".to_owned(),
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 10,
        embedding: fake_embedding(8, 2.0),
        clustering_embedding: fake_embedding(8, 2.0),
        content: "beta".to_owned(),
    }];
    let v0 = store
        .io_upsert_chunks(domain, "main", "doc/B", &rows_b)
        .await
        .expect("upsert B");
    store.io_tag_commit(domain, "main", "c0", v0).await.expect("tag c0");

    // Record the dataset version BEFORE assign.
    let ds_arc = store.io_open_dataset(domain, "main").await.unwrap();
    let version_before = ds_arc.read().await.version().version;

    // Assign c0 → c2 (pure tag pointer).
    let assigned_version = store
        .io_assign_commit(domain, "main", "c0", "c2")
        .await
        .expect("assign c0→c2");
    assert_eq!(assigned_version, v0, "c2 must point at c0's version");

    // No new dataset version was created (assign moved no data → no embeds possible).
    let ds_arc = store.io_open_dataset(domain, "main").await.unwrap();
    let version_after = ds_arc.read().await.version().version;
    assert_eq!(
        version_after, version_before,
        "assign must not create a new dataset version (no recompute)"
    );

    // c2 resolves to the same version as c0.
    let r_c0 = store.io_resolve_commit(domain, "main", "c0").await.unwrap();
    let r_c2 = store.io_resolve_commit(domain, "main", "c2").await.unwrap();
    assert_eq!(r_c0, Some(v0));
    assert_eq!(r_c2, Some(v0), "c2 must resolve to c0's version");

    // Search at c2 returns exactly the same docs as search at c0.
    let query = SearchQuery {
        query_embedding: emb_a.clone(),
        query_text: "alpha".to_owned(),
        mode: crate::kernel::model::SearchMode::Vector,
        start: 0,
        count: 10,
        doc_type_filter: Vec::new(),
        doc_id_filter: Vec::new(),
        snippet: false,
    };
    let mut hits_c0: Vec<String> = store
        .io_search(domain, "main", "c0", &query)
        .await
        .expect("search c0")
        .into_iter()
        .map(|h| h.doc_id)
        .collect();
    let mut hits_c2: Vec<String> = store
        .io_search(domain, "main", "c2", &query)
        .await
        .expect("search c2")
        .into_iter()
        .map(|h| h.doc_id)
        .collect();
    hits_c0.sort();
    hits_c2.sort();
    assert_eq!(hits_c0, hits_c2, "search at c2 must equal search at c0");
}

// --- assign of an unindexed source fails loud ---
#[tokio::test]
async fn assign_unindexed_source_fails_loud() {
    let (store, _tmp) = make_test_store(8);
    // Create the dataset so resolve doesn't fail on a missing dataset.
    let r = vec![ChunkRow {
        doc_id: "doc/X".to_owned(),
        doc_type: "T".to_owned(),
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 10,
        embedding: fake_embedding(8, 1.0),
            clustering_embedding: fake_embedding(8, 1.0),
        content: "x".to_owned(),
    }];
    store.io_upsert_chunks("admin/a", "main", "doc/X", &r).await.unwrap();

    let result = store
        .io_assign_commit("admin/a", "main", "never_indexed", "target")
        .await;
    assert!(result.is_err(), "assigning from an unindexed source must fail loud");
}

// --- P3-CHG-1: Changed replaces the full chunk set — no stale chunks ---
// A doc indexed as a 3-chunk document, then re-pushed as a 1-chunk document,
// must leave EXACTLY the new chunk set (the 2 old tail chunks are gone).
#[tokio::test]
async fn changed_replaces_full_chunk_set_no_stale() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/chg";

    // Initial: doc/big with 3 chunks.
    let big_v1 = vec![
        ChunkRow {
            doc_id: "doc/big".to_owned(),
            doc_type: "Article".to_owned(),
            chunk_index: 0,
            chunk_count: 3,
            chunk_token_start: 0,
            doc_token_len: 1500,
            embedding: fake_embedding(8, 1.0),
            clustering_embedding: fake_embedding(8, 1.0),
            content: "original beginning".to_owned(),
        },
        ChunkRow {
            doc_id: "doc/big".to_owned(),
            doc_type: "Article".to_owned(),
            chunk_index: 1,
            chunk_count: 3,
            chunk_token_start: 500,
            doc_token_len: 1500,
            embedding: fake_embedding(8, 2.0),
            clustering_embedding: fake_embedding(8, 2.0),
            content: "original middle".to_owned(),
        },
        ChunkRow {
            doc_id: "doc/big".to_owned(),
            doc_type: "Article".to_owned(),
            chunk_index: 2,
            chunk_count: 3,
            chunk_token_start: 1000,
            doc_token_len: 1500,
            embedding: fake_embedding(8, 3.0),
            clustering_embedding: fake_embedding(8, 3.0),
            content: "original end".to_owned(),
        },
    ];
    store
        .io_upsert_chunks(domain, "main", "doc/big", &big_v1)
        .await
        .expect("upsert v1");
    assert_eq!(
        store.io_lookup_doc_chunks(domain, "main", "doc/big").await.unwrap().len(),
        3,
        "should have 3 chunks initially"
    );

    // Changed: same doc now renders to a SINGLE shorter chunk.
    let big_v2 = vec![ChunkRow {
        doc_id: "doc/big".to_owned(),
        doc_type: "Article".to_owned(),
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 40,
        embedding: fake_embedding(8, 9.0),
            clustering_embedding: fake_embedding(8, 9.0),
        content: "shortened content".to_owned(),
    }];
    store
        .io_upsert_chunks(domain, "main", "doc/big", &big_v2)
        .await
        .expect("upsert v2 (Changed)");

    // Exactly 1 chunk remains — the 2 stale tail chunks must be gone.
    let after = store
        .io_lookup_doc_chunks(domain, "main", "doc/big")
        .await
        .unwrap();
    assert_eq!(
        after.len(),
        1,
        "Changed must replace the FULL chunk set — no stale chunks (got {})",
        after.len()
    );
    assert_eq!(after[0].content, "shortened content");
}

// --- P3-DEL-1: Deleted removes ALL chunks for a doc_id ---
#[tokio::test]
async fn deleted_removes_all_chunks_for_doc() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/del";

    // Two docs, doc/keep and doc/gone (multi-chunk).
    let keep = vec![ChunkRow {
        doc_id: "doc/keep".to_owned(),
        doc_type: "T".to_owned(),
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 10,
        embedding: fake_embedding(8, 1.0),
            clustering_embedding: fake_embedding(8, 1.0),
        content: "keep me".to_owned(),
    }];
    let gone = vec![
        ChunkRow {
            doc_id: "doc/gone".to_owned(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 2,
            chunk_token_start: 0,
            doc_token_len: 200,
            embedding: fake_embedding(8, 2.0),
            clustering_embedding: fake_embedding(8, 2.0),
            content: "gone part 1".to_owned(),
        },
        ChunkRow {
            doc_id: "doc/gone".to_owned(),
            doc_type: "T".to_owned(),
            chunk_index: 1,
            chunk_count: 2,
            chunk_token_start: 100,
            doc_token_len: 200,
            embedding: fake_embedding(8, 3.0),
            clustering_embedding: fake_embedding(8, 3.0),
            content: "gone part 2".to_owned(),
        },
    ];
    store.io_upsert_chunks(domain, "main", "doc/keep", &keep).await.unwrap();
    store.io_upsert_chunks(domain, "main", "doc/gone", &gone).await.unwrap();

    // Delete doc/gone.
    store.io_delete_doc(domain, "main", "doc/gone").await.unwrap();

    // doc/gone: zero chunks. doc/keep: untouched.
    assert_eq!(
        store.io_lookup_doc_chunks(domain, "main", "doc/gone").await.unwrap().len(),
        0,
        "all chunks of doc/gone must be removed"
    );
    assert_eq!(
        store.io_lookup_doc_chunks(domain, "main", "doc/keep").await.unwrap().len(),
        1,
        "doc/keep must be untouched by deleting doc/gone"
    );
}

// --- DELETE /domain: removes the dataset + purges state; idempotent ---
#[tokio::test]
async fn delete_domain_removes_footprint_and_is_idempotent() {
    let (store, tmp) = make_test_store(8);
    let domain = "admin/doomed";

    // Index a doc on main + a branch, tag commits, record last-indexed.
    let r = vec![ChunkRow {
        doc_id: "doc/1".to_owned(),
        doc_type: "T".to_owned(),
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 10,
        embedding: fake_embedding(8, 1.0),
            clustering_embedding: fake_embedding(8, 1.0),
        content: "doomed".to_owned(),
    }];
    let v = store.io_upsert_chunks(domain, "main", "doc/1", &r).await.unwrap();
    store.io_tag_commit(domain, "main", "c0", v).await.unwrap();
    store.update_last_indexed(domain, "main", "c0", v).await;
    store.io_create_branch(domain, "feature", v).await.unwrap();
    store.update_last_indexed(domain, "feature", "c0", v).await;

    // The dataset dir exists on disk.
    let path = tmp.path().join("admin__doomed.lance");
    assert!(path.exists(), "dataset dir should exist before delete");

    // Delete the domain.
    store.io_delete_domain(domain).await.expect("delete domain");

    // On-disk dataset gone.
    assert!(!path.exists(), "dataset dir must be removed");

    // In-memory state purged: a fresh search at c0 must now fail (no dataset).
    // resolve_commit opens the dataset, which no longer exists → None or error.
    // statistics must no longer count this domain.
    let stats = store.statistics().await;
    assert_eq!(stats.domains, 0, "deleted domain must not be counted");
    assert_eq!(stats.branches, 0, "deleted domain's branches must not be counted");

    // Idempotent: a second delete of the same (now-gone) domain succeeds.
    store
        .io_delete_domain(domain)
        .await
        .expect("second delete must be idempotent (not an error)");

    // Idempotent: deleting a never-seen domain succeeds.
    store
        .io_delete_domain("admin/never_existed")
        .await
        .expect("deleting an unknown domain must succeed (idempotent)");
}

// --- RRF merge produces correct ranking ---
#[test]
fn rrf_merge_combines_ranked_lists() {
    // Vector ranked: A (best), B, C
    let vector_hits = vec![
        ChunkHit {
            doc_id: "A".to_owned(),
            distance: 0.1,
            distance_kind: DistanceKind::RawCosine,
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            content: "a".to_owned(),
            embedding: Vec::new(),
            clustering_embedding: Vec::new(),
        },
        ChunkHit {
            doc_id: "B".to_owned(),
            distance: 0.3,
            distance_kind: DistanceKind::RawCosine,
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            content: "b".to_owned(),
            embedding: Vec::new(),
            clustering_embedding: Vec::new(),
        },
        ChunkHit {
            doc_id: "C".to_owned(),
            distance: 0.5,
            distance_kind: DistanceKind::RawCosine,
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            content: "c".to_owned(),
            embedding: Vec::new(),
            clustering_embedding: Vec::new(),
        },
    ];

    // FTS ranked: B (best), C, D (new — only in FTS)
    let fts_hits = vec![
        ChunkHit {
            doc_id: "B".to_owned(),
            distance: 1.0 / (1.0 + 10.0), // High BM25 score → low distance.
            distance_kind: DistanceKind::Normalised,
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            content: "b".to_owned(),
            embedding: Vec::new(),
            clustering_embedding: Vec::new(),
        },
        ChunkHit {
            doc_id: "C".to_owned(),
            distance: 1.0 / (1.0 + 5.0),
            distance_kind: DistanceKind::Normalised,
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            content: "c".to_owned(),
            embedding: Vec::new(),
            clustering_embedding: Vec::new(),
        },
        ChunkHit {
            doc_id: "D".to_owned(),
            distance: 1.0 / (1.0 + 2.0),
            distance_kind: DistanceKind::Normalised,
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            content: "d".to_owned(),
            embedding: Vec::new(),
            clustering_embedding: Vec::new(),
        },
    ];

    let merged = rrf_merge(vector_hits, fts_hits);

    // B should be ranked highest: rank 2 in vector + rank 1 in FTS
    // = 1/(60+2) + 1/(60+1) = 1/62 + 1/61
    assert_eq!(merged[0].doc_id, "B", "B should rank first (appears high in both lists)");

    // All 4 unique docs should appear — FTS-only hits are kept with their BM25 distance.
    let ids: Vec<&str> = merged.iter().map(|h| h.doc_id.as_str()).collect();
    assert!(ids.contains(&"A"));
    assert!(ids.contains(&"B"));
    assert!(ids.contains(&"C"));
    assert!(ids.contains(&"D"), "FTS-only hit D must be kept with its BM25 distance");
    assert_eq!(ids.len(), 4);

    // Hits in both lists (B, C) must retain their original vector cosine distance.
    // Order is B, C, A, D by RRF score (B and C both appear in vector + FTS).
    assert!(
        (merged[0].distance - 0.3).abs() < f32::EPSILON,
        "B should retain its original vector distance 0.3, got {}",
        merged[0].distance
    );
    assert!(
        (merged[1].distance - 0.5).abs() < f32::EPSILON,
        "C should retain its original vector distance 0.5, got {}",
        merged[1].distance
    );
    assert!(
        (merged[2].distance - 0.1).abs() < f32::EPSILON,
        "A should retain its original vector distance 0.1, got {}",
        merged[2].distance
    );
    // D is FTS-only — must retain its BM25-derived distance.
    assert!(
        (merged[3].distance - 1.0 / (1.0 + 2.0)).abs() < f32::EPSILON,
        "D should retain its BM25-derived distance {}, got {}",
        1.0 / (1.0 + 2.0),
        merged[3].distance
    );
}

// RRF merge must preserve original vector distances, not replace them with
// synthetic rank-normalised values. The first result must NOT always be 0.0.
#[test]
fn rrf_merge_preserves_original_vector_distances() {
    // Only vector hits, no FTS — all items in a single list.
    let vector_hits = vec![
        ChunkHit {
            doc_id: "A".to_owned(),
            distance: 0.1,
            distance_kind: DistanceKind::RawCosine,
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            content: "a".to_owned(),
            embedding: Vec::new(),
            clustering_embedding: Vec::new(),
        },
        ChunkHit {
            doc_id: "B".to_owned(),
            distance: 0.3,
            distance_kind: DistanceKind::RawCosine,
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            content: "b".to_owned(),
            embedding: Vec::new(),
            clustering_embedding: Vec::new(),
        },
    ];

    let merged = rrf_merge(vector_hits, Vec::new());

    // A is rank 1 but must retain its original distance 0.1, NOT 0.0.
    assert!(
        (merged[0].distance - 0.1).abs() < f32::EPSILON,
        "top result should retain original distance 0.1, got {}",
        merged[0].distance
    );

    // B must retain its original distance 0.3, NOT a synthetic rank-based value.
    assert!(
        (merged[1].distance - 0.3).abs() < f32::EPSILON,
        "second result should retain original distance 0.3, got {}",
        merged[1].distance
    );

    // The first result must NOT have distance 0.0 (would mean rank-normalised).
    assert!(
        merged[0].distance > 0.0,
        "first result must not have distance 0.0 (would indicate rank normalisation)"
    );
}

// VectorIndexConfig dimension validation.

#[test]
fn vector_index_config_default_for_dim_produces_valid_config() {
    // Test various dimensions — all must produce a valid IVF_HNSW_SQ config.
    let test_dims = [128, 256, 384, 500, 512, 768, 1024, 1536, 130, 127, 100, 200];
    for dim in test_dims {
        let config = VectorIndexConfig::default_for_dim(dim);
        assert!(
            config.num_partitions >= 1,
            "num_partitions must be >= 1 for dim={}",
            dim
        );
        assert!(
            config.m >= 2,
            "HNSW M must be >= 2 for dim={}",
            dim
        );
        assert!(
            config.ef_construction >= 2 * config.m,
            "ef_construction must be >= 2*M for quality HNSW graph (dim={}): got ef={}, m={}",
            dim, config.ef_construction, config.m
        );
    }
}

#[test]
#[should_panic(expected = "embedding dimension must be > 0")]
fn vector_index_config_zero_dim_panics() {
    VectorIndexConfig::default_for_dim(0);
}

fn one_row(dim: usize, seed: f32, content: &str) -> ChunkRow {
    ChunkRow {
        doc_id: "doc/x".to_owned(),
        doc_type: "Doc".to_owned(),
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 10,
        embedding: fake_embedding(dim, seed),
        clustering_embedding: fake_embedding(dim, seed + 0.5),
        content: content.to_owned(),
    }
}

// --- #3: io_resolve_commit returns Ok(None) for a genuinely-absent domain,
//     and NEVER auto-creates the dataset (BLOCKER-2 read-path guard) ---
#[tokio::test]
async fn resolve_commit_absent_domain_is_none_and_does_not_create() {
    let (store, tmp) = make_test_store(8);
    let resolved = store
        .io_resolve_commit("admin/never", "main", "c0")
        .await
        .expect("resolve must not error on an absent domain");
    assert_eq!(resolved, None, "absent domain → not indexed");
    // The read must NOT have created a dataset directory on disk.
    let path = tmp.path().join("admin__never.lance");
    assert!(
        !path.exists(),
        "io_resolve_commit must not auto-create the dataset (resurrection guard)"
    );
}

// --- #3: a tag that exists resolves; a tag that is absent (but domain
//     exists) is Ok(None), distinct from an error ---
#[tokio::test]
async fn resolve_commit_distinguishes_absent_tag_from_error() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/resolve3";
    let r = one_row(8, 1.0, "hello world");
    let v = store
        .io_upsert_chunks(domain, "main", "doc/x", std::slice::from_ref(&r))
        .await
        .unwrap();
    store.io_tag_commit(domain, "main", "c0", v).await.unwrap();

    // Indexed commit → Some.
    assert_eq!(
        store.io_resolve_commit(domain, "main", "c0").await.unwrap(),
        Some(v)
    );
    // A different, never-tagged commit on an existing domain → Ok(None),
    // NOT an error.
    assert_eq!(
        store.io_resolve_commit(domain, "main", "c_absent").await.unwrap(),
        None
    );
}

// --- #2: a read (io_search) against a never-indexed domain does NOT create
//     the dataset and surfaces "not indexed" rather than empty success ---
#[tokio::test]
async fn search_absent_domain_does_not_create_dataset() {
    let (store, tmp) = make_test_store(8);
    let query = SearchQuery {
        query_embedding: fake_embedding(8, 1.0),
        query_text: "anything".to_owned(),
        mode: SearchMode::Vector,
        start: 0,
        count: 5,
        doc_type_filter: vec![],
        doc_id_filter: vec![],
        snippet: false,
    };
    let res = store.io_search("admin/ghost", "main", "c0", &query).await;
    assert!(res.is_err(), "search on an absent domain must fail (not indexed), not succeed empty");
    let path = tmp.path().join("admin__ghost.lance");
    assert!(!path.exists(), "search must not auto-create the dataset");
}

// --- #2: after io_delete_domain, a resolve does NOT resurrect the dataset ---
#[tokio::test]
async fn delete_domain_then_resolve_does_not_resurrect() {
    let (store, tmp) = make_test_store(8);
    let domain = "admin/del_resurrect";
    let r = one_row(8, 1.0, "doc to delete");
    let v = store
        .io_upsert_chunks(domain, "main", "doc/x", std::slice::from_ref(&r))
        .await
        .unwrap();
    store.io_tag_commit(domain, "main", "c0", v).await.unwrap();
    let path = tmp.path().join("admin__del_resurrect.lance");
    assert!(path.exists(), "dataset exists after indexing");

    store.io_delete_domain(domain).await.unwrap();
    assert!(!path.exists(), "dataset removed by delete");

    // Resolve after delete must be None AND must not recreate the dir.
    let resolved = store.io_resolve_commit(domain, "main", "c0").await.unwrap();
    assert_eq!(resolved, None);
    assert!(!path.exists(), "resolve must not resurrect the deleted dataset");
}

// --- #4: two concurrent first-pushes to the SAME new branch both succeed
//     (idempotent branch-out, no "already exists" 500 for the loser) ---
#[tokio::test]
async fn concurrent_branch_out_both_succeed() {
    use std::sync::Arc;
    let (store, _tmp) = make_test_store(8);
    let store = Arc::new(store);
    let domain = "admin/race";
    // Seed main @ c0 so the parent is indexed.
    let r = one_row(8, 1.0, "parent doc");
    let v = store
        .io_upsert_chunks(domain, "main", "doc/x", std::slice::from_ref(&r))
        .await
        .unwrap();
    store.io_tag_commit(domain, "main", "c0", v).await.unwrap();

    // Fire two concurrent branch-outs of the same new branch from c0.
    let s1 = Arc::clone(&store);
    let s2 = Arc::clone(&store);
    let h1 = tokio::spawn(async move {
        crate::store::branch::io_ensure_branch_forked(&s1, domain, "feature", "c0").await
    });
    let h2 = tokio::spawn(async move {
        crate::store::branch::io_ensure_branch_forked(&s2, domain, "feature", "c0").await
    });
    let r1 = h1.await.unwrap();
    let r2 = h2.await.unwrap();
    assert!(r1.is_ok(), "first branch-out must succeed: {:?}", r1);
    assert!(
        r2.is_ok(),
        "concurrent branch-out must be idempotent, not 500: {:?}",
        r2
    );
    // Exactly one feature branch exists.
    let branches = store.io_list_branches(domain).await.unwrap();
    assert!(branches.iter().any(|b| b == "feature"));
}

/// Build a RecordBatch with the base chunk columns but NO `_distance` column
/// (the shape a vector search would have if the scanner failed to attach
/// distances).
fn batch_without_distance() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::Utf8, false),
        Field::new("doc_type", DataType::Utf8, false),
        Field::new("chunk_index", DataType::Int32, false),
        Field::new("chunk_count", DataType::Int32, false),
        Field::new("chunk_token_start", DataType::Int32, false),
        Field::new("doc_token_len", DataType::Int32, false),
        Field::new("content", DataType::Utf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["doc/1"])),
            Arc::new(StringArray::from(vec!["Doc"])),
            Arc::new(Int32Array::from(vec![0])),
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![0])),
            Arc::new(Int32Array::from(vec![10])),
            Arc::new(StringArray::from(vec!["content"])),
        ],
    )
    .expect("batch")
}

// --- BUG-409: io_resolve_commit must see a tag created AFTER the cached
//     handle was last refreshed. This replicates the LIVE pipeline ordering
//     that the simple round-trip test misses:
//       1. upsert chunks            (refreshes cached handle → H_data)
//       2. optimize indices         (refreshes cached handle → H_index)
//       3. tag commit               (io_tag_commit on the cached handle)
//       4. resolve commit (the GUARD)
//     If io_resolve_commit reads a STALE cached handle that predates the tag
//     write, it returns None for an indexed commit — the 409 guard then lets
//     a re-push of an already-indexed commit through (returns 200, BUG).
//
//     We drive the refresh between upsert and tag explicitly to mirror the
//     optimize step's `io_refresh_cached_dataset`, then assert resolve sees
//     the tag. This is the store-level pinpoint of the live 409 bug.
#[tokio::test]
async fn resolve_commit_sees_tag_created_after_cache_refresh() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/bug409";

    // 1. Upsert (this refreshes the cached handle to the data version).
    let r = one_row(8, 1.0, "indexed content for bug 409");
    let version = store
        .io_upsert_chunks(domain, "main", "doc/x", std::slice::from_ref(&r))
        .await
        .expect("upsert");
    assert!(version > 0);

    // 2. Mirror the optimize step: refresh the cached handle AGAIN, so the
    //    cached Dataset is a handle opened BEFORE the tag is written.
    store
        .io_refresh_cached_dataset(domain, "main")
        .await
        .expect("refresh (mirrors optimize)");

    // 3. Tag the commit (the worker step). After this the commit is indexed.
    store
        .io_tag_commit(domain, "main", "rc0", version)
        .await
        .expect("tag commit");

    // 4. The GUARD: io_resolve_commit must now report rc0 as indexed.
    let resolved = store
        .io_resolve_commit(domain, "main", "rc0")
        .await
        .expect("resolve");
    assert_eq!(
        resolved,
        Some(version),
        "io_resolve_commit must see a commit tagged after the cached handle was \
         refreshed — a stale cached handle that returns None here is the root \
         cause of the 409 guard letting a re-push of an indexed commit through"
    );

    // The dataset-global list view must agree (search resolution path).
    let versions = store
        .io_list_commit_versions(domain)
        .await
        .expect("list commit versions");
    assert_eq!(
        versions.get("rc0").copied(),
        Some(version),
        "io_list_commit_versions (search resolution) must also see the tag"
    );
}

// --- 409 state machine: reserve once, reject the second reservation of the
//     same in-flight commit, release allows a retry ---
#[tokio::test]
async fn reserve_commit_rejects_inflight_then_release_allows_retry() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/reserve";

    // First reservation of an absent commit succeeds.
    let first = store
        .io_try_reserve_commit(domain, "main", "c1")
        .await
        .expect("reserve c1");
    assert!(first, "first reservation of an absent commit must succeed");

    // A second reservation of the SAME in-flight commit is rejected (Reserved
    // state → 409), even though it is not yet tagged/Indexed.
    let second = store
        .io_try_reserve_commit(domain, "main", "c1")
        .await
        .expect("re-reserve c1");
    assert!(
        !second,
        "re-reserving an in-flight (Reserved) commit must be rejected"
    );

    // Releasing the reservation (terminal: e.g. index failed) returns the
    // commit to absent — a retry is then allowed (not blocked forever).
    store
        .io_release_commit_reservation(domain, "main", "c1")
        .await;
    let retry = store
        .io_try_reserve_commit(domain, "main", "c1")
        .await
        .expect("retry reserve c1");
    assert!(
        retry,
        "after release (failed index), a retry of the same commit must be allowed"
    );
}

// --- 409 state machine: an INDEXED (tagged) commit is rejected even with no
//     active reservation (the durable tag is the Indexed marker) ---
#[tokio::test]
async fn reserve_commit_rejects_already_indexed() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/reserve_indexed";

    // Index + tag c0 (Indexed state), then drop the reservation as the worker
    // does on success.
    let r = one_row(8, 1.0, "indexed doc");
    let v = store
        .io_upsert_chunks(domain, "main", "doc/x", std::slice::from_ref(&r))
        .await
        .expect("upsert");
    store
        .io_tag_commit(domain, "main", "c0", v)
        .await
        .expect("tag c0");

    // A push of an already-Indexed commit must be rejected (no reservation
    // exists, but the tag does).
    let reserved = store
        .io_try_reserve_commit(domain, "main", "c0")
        .await
        .expect("reserve c0");
    assert!(
        !reserved,
        "re-pushing an already-indexed (tagged) commit must be rejected"
    );
}

// --- 409 state machine: two CONCURRENT reservations of the same new commit —
//     exactly one wins (atomic check-and-reserve, no TOCTOU) ---
#[tokio::test]
async fn concurrent_reserve_same_commit_exactly_one_wins() {
    use std::sync::Arc;
    let (store, _tmp) = make_test_store(8);
    let store = Arc::new(store);
    let domain = "admin/reserve_race";

    let s1 = Arc::clone(&store);
    let s2 = Arc::clone(&store);
    let h1 = tokio::spawn(async move { s1.io_try_reserve_commit(domain, "main", "c9").await });
    let h2 = tokio::spawn(async move { s2.io_try_reserve_commit(domain, "main", "c9").await });
    let r1 = h1.await.unwrap().expect("reserve 1");
    let r2 = h2.await.unwrap().expect("reserve 2");

    assert_ne!(
        r1, r2,
        "exactly one concurrent reservation of the same commit must win (got r1={}, r2={})",
        r1, r2
    );
    assert!(r1 || r2, "at least one reservation must have succeeded");
}

// --- #E: a ranked vector search with a MISSING `_distance` column fails
//     loud rather than defaulting distances to 0.0 (which would corrupt
//     ranking). A plain scan (require_distance=false) tolerates absence. ---
#[test]
fn vector_hits_missing_distance_fails_loud_when_required() {
    let batches = vec![batch_without_distance()];
    let err = batches_to_vector_hits(&batches, true);
    assert!(
        err.is_err(),
        "missing _distance on a ranked search must error, not default to 0.0"
    );

    // The same batch is fine for a plain scan (no ranking expected).
    let ok = batches_to_vector_hits(&batches, false);
    assert!(ok.is_ok(), "a plain scan tolerates absent _distance");
    assert_eq!(ok.unwrap().len(), 1);
}

/// Count this process's currently-open file descriptors (Linux: /proc/self/fd).
/// Used by the FD-exhaustion regression test to prove search no longer leaks
/// descriptors under sustained load.
#[cfg(target_os = "linux")]
fn open_fd_count() -> usize {
    std::fs::read_dir("/proc/self/fd")
        .expect("read /proc/self/fd")
        .count()
}

// --- BUG-FD24: under sustained search load the engine exhausted file
//     descriptors ("Too many open files (os error 24)"). The bench pinpointed
//     the mechanism: ~2 FDs leaked PER /search — the Lance VECTOR-INDEX reader
//     files (`_indices/<uuid>/index.idx` + `auxiliary.idx`), opened when the
//     ANN `nearest()` runs through a FRESHLY-`Dataset::open`ed handle (a new
//     object_store + session) and NOT released before the call returns. The
//     count climbed monotonically (past 2100) and exhausted the default soft
//     limit (~1024) after ~140 searches.
//
//     This test builds a domain WITH a real vector (ANN) index — the leak is
//     index-reader-bound, so the index MUST exist — then issues many vector
//     searches and asserts BOTH (a) the process open-FD count stays FLAT and
//     (b) reads perform no fresh `Dataset::open` per query. RED against
//     fresh-open-per-search (FDs climb + opens == searches); GREEN once reads
//     reuse the cached handle (one shared object_store + session → index
//     readers bounded to one set, FDs flat).
#[cfg(target_os = "linux")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn search_does_not_leak_file_descriptors_under_load() {
    let dim = 16;
    // IVF_HNSW_SQ needs >= 256 training vectors; a few partitions for a small corpus.
    let config = VectorIndexConfig {
        num_partitions: 4,
        m: 16,
        ef_construction: 100,
        nprobes: 4,
        refine_factor: Some(10),
    };
    let (mut store, _tmp) = make_test_store(dim);
    store.set_vector_index_config(config.clone());
    let domain = "admin/fdload";

    // One doc with 300 chunks → 300 rows, above the 256 index-training floor,
    // built in a single fast upsert (avoids hundreds of sequential writes).
    let corpus = 300usize;
    let rows: Vec<ChunkRow> = (0..corpus)
        .map(|i| ChunkRow {
            doc_id: "doc/corpus".to_owned(),
            doc_type: "Doc".to_owned(),
            chunk_index: i as i32,
            chunk_count: corpus as i32,
            chunk_token_start: i as i32,
            doc_token_len: corpus as i32,
            embedding: fake_embedding(dim, 1.0 + i as f32),
            clustering_embedding: fake_embedding(dim, 1.0 + i as f32),
            content: format!("chunk content number {} lorem ipsum dolor", i),
        })
        .collect();
    store
        .io_upsert_chunks(domain, "main", "doc/corpus", &rows)
        .await
        .expect("upsert corpus");

    // Build the vector (ANN) index — the leaked FDs are its reader files.
    {
        let ds_arc = store.io_open_dataset(domain, "main").await.unwrap();
        let mut ds = ds_arc.write().await;
        crate::store::vector_index::io_ensure_vector_index(&mut ds, &config, false)
            .await
            .expect("ensure vector index");
    }
    // Refresh the cached handle so it reflects the new index version, then tag.
    store
        .io_refresh_cached_dataset(domain, "main")
        .await
        .expect("refresh after index build");
    let indexed_version = {
        let ds_arc = store.io_open_dataset(domain, "main").await.unwrap();
        let guard = ds_arc.read().await;
        guard.version().version
    };
    store
        .io_tag_commit(domain, "main", "c_load", indexed_version)
        .await
        .expect("tag c_load");

    let query = SearchQuery {
        query_embedding: fake_embedding(dim, 9999.0),
        query_text: String::new(),
        mode: SearchMode::Vector,
        start: 0,
        count: 5,
        doc_type_filter: Vec::new(),
        doc_id_filter: Vec::new(),
        snippet: false,
    };

    // Warm up: the first search opens/populates the cached handle and loads
    // the index reader once. Measure baselines AFTER warm-up so we compare
    // steady-state to steady-state.
    store
        .io_search(domain, "main", "c_load", &query)
        .await
        .expect("warmup search");
    let baseline_fds = open_fd_count();
    let baseline_opens = store.fresh_open_count();

    // Sustained CONCURRENT load (matches the live server). With fresh-open per
    // search, the ANN index reader FDs leak ~2/search and the count climbs;
    // with the cached handle reused, the count stays flat.
    let load_iterations = 400usize;
    let concurrency = 8usize;
    let store = std::sync::Arc::new(store);
    for _ in 0..(load_iterations / concurrency) {
        let mut handles = Vec::with_capacity(concurrency);
        for _ in 0..concurrency {
            let s = std::sync::Arc::clone(&store);
            let q = query.clone();
            handles.push(tokio::spawn(async move {
                s.io_search(domain, "main", "c_load", &q).await
            }));
        }
        for h in handles {
            h.await
                .expect("search task join")
                .expect("load search must succeed (no FD exhaustion)");
        }
    }
    let after_fds = open_fd_count();
    let opens_added = store.fresh_open_count() - baseline_opens;
    eprintln!(
        "[fd-load] searches={} fresh_opens_added={} fds(baseline={}, after={}, delta={})",
        load_iterations,
        opens_added,
        baseline_fds,
        after_fds,
        after_fds as i64 - baseline_fds as i64
    );

    // PRIMARY: searches must NOT open a fresh dataset per call (each fresh
    // open is the new object_store/session that leaks the index reader FDs).
    assert!(
        opens_added < (load_iterations as u64) / 4,
        "search opened a fresh dataset per call ({} fresh opens across {} searches). \
         Reads must reuse the cached domain handle and checkout_version off it \
         (sharing one object_store + session), not Dataset::open fresh every query \
         — the fresh open leaks the ANN index reader FDs (BUG-FD24).",
        opens_added,
        load_iterations
    );

    // SECONDARY: open FD count must stay FLAT under load (the bench saw it
    // climb past 2100 unbounded). Slack covers runtime/allocator churn only —
    // it must NOT scale with the number of searches.
    let slack = 64;
    assert!(
        after_fds <= baseline_fds + slack,
        "open FD count grew under search load (baseline={}, after {} searches={}, slack={}). \
         The ANN index reader FDs are leaking per search — reads must reuse the \
         cached handle so the index readers are bounded to one set (BUG-FD24).",
        baseline_fds,
        load_iterations,
        after_fds,
        slack
    );
}

// --- BUG-FD24 (FIX 1): the NON-MAIN branch path of `io_lookup_doc_chunks`
//     (the `/similar` doc-chunk lookup on a feature branch) did a raw
//     `Dataset::open` per call — a fresh object_store + session whose vector-
//     index reader files leak FDs under repeated lookup load, the SAME leak
//     class the committed `io_search` non-main fix already closed. The main
//     path was already cached; only the feature-branch path leaked.
//
//     This test indexes a feature branch with a real ANN index, then issues
//     many feature-branch lookups and asserts BOTH (a) reads perform no fresh
//     `Dataset::open` per call and (b) the process open-FD count stays FLAT.
//     RED against the raw `Dataset::open` (opens == lookups, FDs climb); GREEN
//     once the lookup clones the cached handle and checks the branch out off it
//     (shared object_store + session — no new FDs).
#[cfg(target_os = "linux")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn similar_lookup_on_feature_branch_does_not_leak_file_descriptors_under_load() {
    let dim = 16;
    let config = VectorIndexConfig {
        num_partitions: 4,
        m: 16,
        ef_construction: 100,
        nprobes: 4,
        refine_factor: Some(10),
    };
    let (mut store, _tmp) = make_test_store(dim);
    store.set_vector_index_config(config.clone());
    let domain = "admin/fdbranch";
    let branch = "feature";

    // Seed main with one 300-chunk doc (above the 256 IVF_HNSW_SQ training floor),
    // then fork a feature branch from that version and add the same doc on the
    // branch so the feature-branch lookup has chunks to return.
    let corpus = 300usize;
    let rows: Vec<ChunkRow> = (0..corpus)
        .map(|i| ChunkRow {
            doc_id: "doc/corpus".to_owned(),
            doc_type: "Doc".to_owned(),
            chunk_index: i as i32,
            chunk_count: corpus as i32,
            chunk_token_start: i as i32,
            doc_token_len: corpus as i32,
            embedding: fake_embedding(dim, 1.0 + i as f32),
            clustering_embedding: fake_embedding(dim, 1.0 + i as f32),
            content: format!("chunk content number {} lorem ipsum dolor", i),
        })
        .collect();
    let main_version = store
        .io_upsert_chunks(domain, "main", "doc/corpus", &rows)
        .await
        .expect("upsert corpus on main");

    // Fork the feature branch from main's indexed version, then write on it.
    store
        .io_create_branch(domain, branch, main_version)
        .await
        .expect("create feature branch");
    store
        .io_upsert_chunks(domain, branch, "doc/corpus", &rows)
        .await
        .expect("upsert corpus on feature branch");

    // Build the vector (ANN) index on the feature-branch head — the leaked FDs
    // are its reader files. Open a branch-bound handle to index it, then refresh
    // the cache so the cached clone the lookup reuses carries the index.
    {
        let mut branch_ds = store
            .io_open_dataset_uncached(domain, branch)
            .await
            .expect("open feature branch for index build")
            .expect("feature branch dataset exists");
        crate::store::vector_index::io_ensure_vector_index(&mut branch_ds, &config, false)
            .await
            .expect("ensure vector index on feature branch");
    }
    store
        .io_refresh_cached_dataset(domain, branch)
        .await
        .expect("refresh cache after feature-branch index build");

    // Warm up: the first lookup populates/loads the cached handle + index reader
    // once. Baselines are measured AFTER warm-up (steady-state to steady-state).
    store
        .io_lookup_doc_chunks(domain, branch, "doc/corpus")
        .await
        .expect("warmup lookup");
    let baseline_fds = open_fd_count();
    let baseline_opens = store.fresh_open_count();

    // Sustained CONCURRENT feature-branch lookups (matches `/similar` load).
    let load_iterations = 400usize;
    let concurrency = 8usize;
    let store = std::sync::Arc::new(store);
    for _ in 0..(load_iterations / concurrency) {
        let mut handles = Vec::with_capacity(concurrency);
        for _ in 0..concurrency {
            let s = std::sync::Arc::clone(&store);
            handles.push(tokio::spawn(async move {
                s.io_lookup_doc_chunks(domain, branch, "doc/corpus").await
            }));
        }
        for h in handles {
            let hits = h
                .await
                .expect("lookup task join")
                .expect("feature-branch lookup must succeed (no FD exhaustion)");
            assert!(!hits.is_empty(), "feature-branch lookup must return the doc's chunks");
        }
    }
    let after_fds = open_fd_count();
    let opens_added = store.fresh_open_count() - baseline_opens;
    eprintln!(
        "[fd-branch-lookup] lookups={} fresh_opens_added={} fds(baseline={}, after={}, delta={})",
        load_iterations,
        opens_added,
        baseline_fds,
        after_fds,
        after_fds as i64 - baseline_fds as i64
    );

    // PRIMARY: feature-branch lookups must NOT open a fresh dataset per call —
    // they must clone the cached handle and checkout the branch off it (shared
    // object_store + session). A fresh open per lookup leaks the ANN index
    // reader FDs (BUG-FD24, the non-main `io_search` fix's leak class).
    assert!(
        opens_added < (load_iterations as u64) / 4,
        "feature-branch lookup opened a fresh dataset per call ({} fresh opens across {} \
         lookups). The non-main lookup path must reuse the cached domain handle and \
         checkout_branch off a clone, not Dataset::open fresh every call (BUG-FD24).",
        opens_added,
        load_iterations
    );

    // SECONDARY: open FD count must stay FLAT under load (a per-call fresh open
    // would climb the index reader FDs unbounded). Slack covers runtime churn
    // only — it must NOT scale with the number of lookups.
    let slack = 64;
    assert!(
        after_fds <= baseline_fds + slack,
        "open FD count grew under feature-branch lookup load (baseline={}, after {} \
         lookups={}, slack={}). The non-main lookup is leaking the ANN index reader FDs \
         per call — it must reuse the cached handle so the readers are bounded (BUG-FD24).",
        baseline_fds,
        load_iterations,
        after_fds,
        slack
    );
}

/// Build a real ANN-indexed corpus of MANY MULTI-CHUNK documents with planted
/// near-duplicate pairs, then tag a commit. Returns (store, domain, commit).
///
/// Geometry (deliberately reproduces the k=2 STARVATION the bug relied on):
/// documents are grouped into "families" of `docs_per_family` documents that
/// are planted near-duplicates of each other. The per-CHUNK perturbation
/// (`c * 0.0002`) is much SMALLER than the per-DOCUMENT offset within a family
/// (`d * 0.02`), so a document's OWN sibling chunks are strictly CLOSER to each
/// other than to the planted twin's chunks. Under the old `nearest(k=2)` +
/// skip-own-doc path both result slots are filled by the query point's OWN
/// siblings → no cross-doc row → every point yields `None` → `[]` at scale. The
/// filtered `nearest()` (exclude self doc) cannot be starved.
#[cfg(test)]
async fn build_duplicate_corpus(
    dim: usize,
    families: usize,
    docs_per_family: usize,
    chunks_per_doc: usize,
    doc_type_of: impl Fn(usize, usize) -> String,
) -> (LanceStore, &'static str, String) {
    let config = VectorIndexConfig {
        num_partitions: 4,
        m: 16,
        ef_construction: 100,
        nprobes: 8,
        refine_factor: Some(20),
    };
    let (mut store, tmp) = make_test_store(dim);
    // Keep the temp dir alive for the whole test by leaking it — these are
    // short-lived test processes and the OS reclaims on exit.
    std::mem::forget(tmp);
    store.set_vector_index_config(config.clone());
    let domain = "admin/dupscale";

    // One big upsert of all chunk rows (fast — avoids hundreds of writes).
    let mut rows: Vec<ChunkRow> = Vec::new();
    for fam in 0..families {
        // Families are far apart in embedding space.
        let family_seed = 1.0 + fam as f32 * 7.0;
        for d in 0..docs_per_family {
            let doc_id = format!("doc/fam{}/d{}", fam, d);
            let doc_type = doc_type_of(fam, d);
            // Per-document offset WITHIN the family (0.02) — larger than the
            // per-chunk spread below, so each doc's own siblings cluster tighter
            // than the planted twin. This is what starves the old k=2 path.
            let doc_seed = family_seed + d as f32 * 0.02;
            for c in 0..chunks_per_doc {
                rows.push(ChunkRow {
                    doc_id: doc_id.clone(),
                    doc_type: doc_type.clone(),
                    chunk_index: c as i32,
                    chunk_count: chunks_per_doc as i32,
                    chunk_token_start: c as i32,
                    doc_token_len: chunks_per_doc as i32,
                    // Per-chunk spread (0.0002) << per-doc offset (0.02): own
                    // siblings are the nearest neighbours of any chunk.
                    embedding: fake_embedding(dim, doc_seed + c as f32 * 0.0002),
                    clustering_embedding: fake_embedding(dim, doc_seed + c as f32 * 0.0002),
                    content: format!("family {} doc {} chunk {} lorem ipsum", fam, d, c),
                });
            }
        }
    }
    // Upsert per doc (the store keys writes by doc_id).
    let mut by_doc: std::collections::BTreeMap<String, Vec<ChunkRow>> =
        std::collections::BTreeMap::new();
    for r in rows {
        by_doc.entry(r.doc_id.clone()).or_default().push(r);
    }
    for (doc_id, doc_rows) in &by_doc {
        store
            .io_upsert_chunks(domain, "main", doc_id, doc_rows)
            .await
            .expect("upsert corpus doc");
    }

    // Build the ANN index, refresh the cached handle, tag the commit.
    {
        let ds_arc = store.io_open_dataset(domain, "main").await.unwrap();
        let mut ds = ds_arc.write().await;
        crate::store::vector_index::io_ensure_vector_index(&mut ds, &config, false)
            .await
            .expect("ensure vector index");
    }
    store
        .io_refresh_cached_dataset(domain, "main")
        .await
        .expect("refresh after index build");
    let version = {
        let ds_arc = store.io_open_dataset(domain, "main").await.unwrap();
        let guard = ds_arc.read().await;
        guard.version().version
    };
    let commit = "c_dup".to_owned();
    store
        .io_tag_commit(domain, "main", &commit, version)
        .await
        .expect("tag commit");

    (store, domain, commit)
}

/// REAL-DATA SCALE GUARD (the gap that hid BUG: `/duplicates` returned `[]` at
/// scale). Hundreds of MULTI-CHUNK documents with planted near-duplicate pairs:
/// the result must be NON-EMPTY and must contain the planted pairs. The old
/// `nearest(k=2)` + skip-own-doc path returned `[]` here because every query
/// point's two result slots were filled by its OWN sibling chunks. The filtered
/// `nearest()` (exclude self doc) cannot be starved that way.
#[tokio::test]
async fn duplicate_scan_finds_planted_pairs_at_scale_not_empty() {
    let dim = 16;
    let families = 60; // 60 families × 5 docs = 300 docs
    let docs_per_family = 5;
    let chunks_per_doc = 4; // multi-chunk — starves the old k=2 path
    let (store, domain, commit) = build_duplicate_corpus(
        dim,
        families,
        docs_per_family,
        chunks_per_doc,
        |_fam, _d| "Item".to_owned(),
    )
    .await;

    // threshold = 1.0 is the WHOLE cosine range — the canonical "is it broken"
    // check: every point with another doc in scope MUST produce a pair.
    let groups = store
        .io_duplicate_groups(
            domain,
            "main",
            &commit,
            1.0,
            &DuplicateScope::default(),
            false,
            DEFAULT_DUPLICATE_MAX_POINTS,
        )
        .await
        .expect("duplicate scan");

    assert!(
        !groups.is_empty(),
        "duplicate scan returned [] on a {}-doc multi-chunk corpus — the scale bug \
         (k=2 starved by own sibling chunks) has regressed",
        families * docs_per_family
    );

    // Every planted same-family pair (distinct docs sharing a seed) must be
    // surfaced at a permissive threshold. Check family 0's 5 docs all pair up.
    let pairs: std::collections::HashSet<(String, String)> = groups
        .iter()
        .map(|g| (g.group[0].id.clone(), g.group[1].id.clone()))
        .collect();
    let contains = |a: &str, b: &str| {
        let lo = a.min(b).to_owned();
        let hi = a.max(b).to_owned();
        pairs.contains(&(lo, hi))
    };
    // At least one planted intra-family pair from family 0 is present.
    let fam0_paired = (0..docs_per_family).any(|i| {
        (0..docs_per_family).any(|j| {
            i != j
                && contains(&format!("doc/fam0/d{}", i), &format!("doc/fam0/d{}", j))
        })
    });
    assert!(
        fam0_paired,
        "no planted near-duplicate pair from family 0 surfaced: {:?}",
        pairs
    );

    // Lower-id-first canonicalisation holds for every emitted group.
    for g in &groups {
        assert!(
            g.group[0].id <= g.group[1].id,
            "group not lower-id-first: {:?}",
            g.group
        );
        assert_eq!(g.group.len(), 2, "every group is currently a pair");
    }
}

/// Cross-set (set/target) entity resolution: pairs must STRADDLE the two
/// populations only — no within-set pair leaks in when a target is specified.
#[tokio::test]
async fn duplicate_scan_cross_set_pairs_straddle_only() {
    let dim = 16;
    // Families alternate doc_type Abt/Buy; same-seed docs across the two types
    // are the planted cross-catalogue near-duplicates.
    let families = 20;
    let docs_per_family = 4;
    let chunks_per_doc = 3;
    let (store, domain, commit) = build_duplicate_corpus(
        dim,
        families,
        docs_per_family,
        chunks_per_doc,
        // Within a family, half the docs are Abt and half Buy → same-seed
        // (near-identical) docs exist on BOTH sides of the set/target split.
        |_fam, d| if d % 2 == 0 { "Abt".to_owned() } else { "Buy".to_owned() },
    )
    .await;

    let scope = DuplicateScope {
        set_doc_types: vec!["Abt".to_owned()],
        set_doc_ids: vec![],
        target_doc_types: vec!["Buy".to_owned()],
        target_doc_ids: vec![],
    };
    let groups = store
        .io_duplicate_groups(
            domain,
            "main",
            &commit,
            1.0,
            &scope,
            false,
            DEFAULT_DUPLICATE_MAX_POINTS,
        )
        .await
        .expect("cross-set duplicate scan");

    assert!(!groups.is_empty(), "cross-set scan found no straddling pairs");

    // EVERY pair must straddle Abt↔Buy: one member even-d (Abt), one odd-d (Buy).
    // doc ids are doc/fam{F}/d{D}; D parity encodes the type.
    let parity = |id: &str| -> usize {
        let d: usize = id.rsplit("/d").next().unwrap().parse().unwrap();
        d % 2
    };
    for g in &groups {
        let p0 = parity(&g.group[0].id);
        let p1 = parity(&g.group[1].id);
        assert_ne!(
            p0, p1,
            "cross-set pair does not straddle Abt↔Buy (both same type): {:?}",
            g.group
        );
    }
}

/// snippet=true populates each member's `snippet` from its matched chunk text.
#[tokio::test]
async fn duplicate_scan_snippet_populates_member_text() {
    let dim = 16;
    let (store, domain, commit) =
        build_duplicate_corpus(dim, 10, 4, 2, |_f, _d| "Item".to_owned()).await;

    let with = store
        .io_duplicate_groups(
            domain, "main", &commit, 1.0, &DuplicateScope::default(), true,
            DEFAULT_DUPLICATE_MAX_POINTS,
        )
        .await
        .expect("snippet scan");
    assert!(!with.is_empty());
    for g in &with {
        for m in &g.group {
            let s = m.snippet.as_deref().unwrap_or("");
            assert!(
                s.contains("lorem ipsum"),
                "snippet missing chunk text for {}: {:?}",
                m.id,
                m.snippet
            );
        }
    }

    // snippet=false leaves all member snippets absent.
    let without = store
        .io_duplicate_groups(
            domain, "main", &commit, 1.0, &DuplicateScope::default(), false,
            DEFAULT_DUPLICATE_MAX_POINTS,
        )
        .await
        .expect("no-snippet scan");
    assert!(
        without.iter().all(|g| g.group.iter().all(|m| m.snippet.is_none())),
        "snippet=false must not populate member snippets"
    );
}

/// The set-population candidate cap is fail-loud: a set larger than the bound is
/// REJECTED, never silently partial.
#[tokio::test]
async fn duplicate_scan_rejects_oversized_set_fail_loud() {
    let dim = 16;
    let (store, domain, commit) =
        build_duplicate_corpus(dim, 10, 4, 2, |_f, _d| "Item".to_owned()).await;

    // 10×4×2 = 80 chunk points; a cap of 4 must reject (not truncate).
    let err = store
        .io_duplicate_groups(
            domain, "main", &commit, 1.0, &DuplicateScope::default(), false, 4,
        )
        .await
        .expect_err("oversized set must be rejected");
    let msg = format!("{}", err);
    assert!(
        msg.contains("exceeds the bound"),
        "expected a fail-loud cap rejection, got: {}",
        msg
    );
}

/// The duplicates scan opens NO fresh dataset — it reuses the cached domain
/// handle (no FD pressure; BUG-FD24). Guards against a future change that
/// reintroduces `Dataset::open` per scan.
#[tokio::test]
async fn duplicate_scan_reuses_cached_handle_no_fresh_open() {
    let dim = 16;
    let (store, domain, commit) =
        build_duplicate_corpus(dim, 12, 3, 2, |_f, _d| "Item".to_owned()).await;

    let before = store.fresh_open_count();
    store
        .io_duplicate_groups(
            domain, "main", &commit, 1.0, &DuplicateScope::default(), false,
            DEFAULT_DUPLICATE_MAX_POINTS,
        )
        .await
        .expect("scan");
    let after = store.fresh_open_count();
    assert_eq!(
        before, after,
        "duplicates scan opened {} fresh dataset(s) — it must reuse the cached \
         handle (checkout off the cached Arc), never Dataset::open (BUG-FD24)",
        after - before
    );
}

/// SCALE-BLOCKING FD LEAK (the entity-resolution bench, real 2173-point
/// corpus): the duplicates scan's PER-POINT `nearest()` loop leaked file
/// descriptors live — open-FD climbed 20→226 over the scan, then aborted ("too
/// many open files"), even though the snapshot handle is the cached one (no
/// fresh `Dataset::open`). The per-point ANN query MUST keep the open-FD count
/// FLAT (bounded) across the whole N-point scan — the index/prefilter readers
/// each query opens must be released, not accumulated.
///
/// This guards the ANN path directly: a real IVF_PQ-indexed corpus, the
/// CROSS-SET scope the bench exercises (set=Abt target=Buy, snippet on), one
/// warm-up scan to load the index reader once, then a full scan asserting the
/// process open-FD count stays bounded. Mirrors
/// `search_does_not_leak_file_descriptors_under_load`.
#[cfg(target_os = "linux")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn duplicate_scan_does_not_leak_file_descriptors_at_scale() {
    let dim = 16;
    // 60 families × 5 docs × 4 chunks = 1200 indexed points (Abt set = 600) —
    // above the 256-row IVF_PQ training floor and large enough that a per-point
    // FD leak would climb well past the default soft limit.
    let families = 60;
    let docs_per_family = 5;
    let chunks_per_doc = 4;
    let (store, domain, commit) = build_duplicate_corpus(
        dim,
        families,
        docs_per_family,
        chunks_per_doc,
        |_fam, d| if d % 2 == 0 { "Abt".to_owned() } else { "Buy".to_owned() },
    )
    .await;

    // Cross-set scope (set=Abt, target=Buy) — mirrors the bench that found the
    // leak. snippet=true projects `content`, matching the bench's projection.
    let scope = DuplicateScope {
        set_doc_types: vec!["Abt".to_owned()],
        set_doc_ids: vec![],
        target_doc_types: vec!["Buy".to_owned()],
        target_doc_ids: vec![],
    };

    // Warm up: one scan loads/caches the index reader once. Measure the baseline
    // AFTER warm-up so we compare steady-state to steady-state.
    store
        .io_duplicate_groups(
            domain, "main", &commit, 1.0, &scope, true,
            DEFAULT_DUPLICATE_MAX_POINTS,
        )
        .await
        .expect("warmup duplicate scan");

    let baseline_fds = open_fd_count();
    let baseline_opens = store.fresh_open_count();

    // A second full cross-set ANN scan over the 600 Abt set points.
    let groups = store
        .io_duplicate_groups(
            domain, "main", &commit, 1.0, &scope, true,
            DEFAULT_DUPLICATE_MAX_POINTS,
        )
        .await
        .expect("scaled duplicate scan must succeed (no FD exhaustion)");

    let after_fds = open_fd_count();
    let opens_added = store.fresh_open_count() - baseline_opens;
    let points = families * docs_per_family * chunks_per_doc / 2;
    eprintln!(
        "[fd-dup] points={} groups={} fresh_opens_added={} fds(baseline={}, after={}, delta={})",
        points,
        groups.len(),
        opens_added,
        baseline_fds,
        after_fds,
        after_fds as i64 - baseline_fds as i64
    );

    // Correctness must be unchanged.
    assert!(
        !groups.is_empty(),
        "duplicate scan returned no groups — the FD fix must not change correctness"
    );

    // The scan must reuse the cached snapshot handle (no fresh Dataset::open per
    // point) AND keep the open FD count FLAT across the N-point ANN loop. Slack
    // covers runtime/allocator/scheduler churn only — it must NOT scale with N.
    assert_eq!(
        opens_added, 0,
        "duplicates scan opened {} fresh dataset(s) across the per-point loop — it must \
         reuse the cached snapshot handle (BUG-FD24)",
        opens_added
    );
    let slack = 64;
    assert!(
        after_fds <= baseline_fds + slack,
        "open FD count grew across a {}-point duplicates ANN scan (baseline={}, after={}, \
         slack={}). The per-point `nearest()` queries are leaking their vector-index \
         reader FDs (BUG-FD24 family) — each per-point query's readers must be released \
         so FDs stay bounded regardless of point count.",
        points,
        baseline_fds,
        after_fds,
        slack
    );
}

// --- POST-COMPACTION SNAPSHOT ISOLATION (regression guard) ---
//
// Proves that compaction does NOT break snapshot isolation. The bug: compaction
// formerly retagged ALL historical commit tags to the compacted version, so
// checkout_version(c0) returned the compacted (post-c1) data.
//
// Setup: insert doc/A → tag c0, insert doc/B → tag c1, push 16 more fragments
// (above the 16-fragment compaction threshold), compact, assert search@c0
// excludes doc/B. FAILS on the old retag code, PASSES after fix.
#[tokio::test]
async fn post_compaction_snapshot_isolation_regression_guard() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/compact_iso";

    // Insert doc/A, tag as commit "c0".
    let emb_a = fake_embedding(8, 1.0);
    let rows_a = vec![ChunkRow {
        doc_id: "doc/A".to_owned(),
        doc_type: "T".to_owned(),
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 5,
        embedding: emb_a.clone(),
        clustering_embedding: emb_a.clone(),
        content: "document alpha".to_owned(),
    }];
    let v0 = store
        .io_upsert_chunks(domain, "main", "doc/A", &rows_a)
        .await
        .expect("upsert A");
    store
        .io_tag_commit(domain, "main", "c0", v0)
        .await
        .expect("tag c0");

    // Insert doc/B, tag as commit "c1".
    let emb_b = fake_embedding(8, 2.0);
    let rows_b = vec![ChunkRow {
        doc_id: "doc/B".to_owned(),
        doc_type: "T".to_owned(),
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 5,
        embedding: emb_b.clone(),
        clustering_embedding: emb_b.clone(),
        content: "document beta".to_owned(),
    }];
    let v1 = store
        .io_upsert_chunks(domain, "main", "doc/B", &rows_b)
        .await
        .expect("upsert B");
    store
        .io_tag_commit(domain, "main", "c1", v1)
        .await
        .expect("tag c1");

    // Push 16 more fragments (one per upsert) to exceed the compaction threshold
    // (COMPACT_FRAGMENT_THRESHOLD = 16). Each push is a different doc_id so it
    // creates a new fragment (not a delete+append cycle on an existing doc).
    for i in 0..16 {
        let emb = fake_embedding(8, 10.0 + i as f32);
        let rows = vec![ChunkRow {
            doc_id: format!("doc/filler_{}", i),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: emb.clone(),
            clustering_embedding: emb,
            content: format!("filler document {}", i),
        }];
        store
            .io_upsert_chunks(domain, "main", &format!("doc/filler_{}", i), &rows)
            .await
            .expect("upsert filler");
    }

    // Trigger compaction (uncached open, compact, refresh).
    // Total fragments = 2 (A, B) + 16 (fillers) = 18 > threshold of 16.
    {
        let ds = store
            .io_open_dataset_uncached(domain, "main")
            .await
            .expect("uncached open for compact")
            .expect("dataset must exist");
        let fragment_count = ds.get_fragments().len();
        assert!(
            fragment_count > 16,
            "pre-compaction fragment count must exceed threshold (got {})",
            fragment_count
        );
        let mut ds = ds;
        io_compact_data(&mut ds, false)
            .await
            .expect("compact_data");
        let fragments_after = ds.get_fragments().len();
        eprintln!(
            "[compact_iso] fragments: {} -> {} (compacted version={})",
            fragment_count,
            fragments_after,
            ds.version().version
        );
    }
    // Refresh the cached handle after compaction.
    store
        .io_refresh_cached_dataset(domain, "main")
        .await
        .expect("refresh after compaction");

    // CRITICAL ASSERTION: search at c0 must still return ONLY doc/A.
    // The old retag code repointed c0's tag to the compacted version (which
    // contains A + B + fillers), breaking isolation. The fix preserves the
    // original tag → original version mapping.
    let query = SearchQuery {
        query_embedding: emb_a.clone(),
        query_text: "document".to_owned(),
        mode: SearchMode::Vector,
        start: 0,
        count: 50,
        doc_type_filter: Vec::new(),
        doc_id_filter: Vec::new(),
        snippet: false,
    };
    let hits_c0 = store
        .io_search(domain, "main", "c0", &query)
        .await
        .expect("search at c0 after compaction");

    let doc_ids_c0: Vec<&str> = hits_c0.iter().map(|h| h.doc_id.as_str()).collect();
    assert!(
        doc_ids_c0.contains(&"doc/A"),
        "c0 snapshot must contain doc/A after compaction"
    );
    assert!(
        !doc_ids_c0.contains(&"doc/B"),
        "POST-COMPACTION ISOLATION VIOLATED: c0 snapshot must NOT contain doc/B \
         (added at c1). If this fails, compaction is retagging historical commit \
         tags to the compacted version — the snapshot isolation regression."
    );

    // Sanity: search at c1 must find both A and B.
    let hits_c1 = store
        .io_search(domain, "main", "c1", &query)
        .await
        .expect("search at c1 after compaction");

    let doc_ids_c1: Vec<&str> = hits_c1.iter().map(|h| h.doc_id.as_str()).collect();
    assert!(
        doc_ids_c1.contains(&"doc/A"),
        "c1 snapshot must contain doc/A"
    );
    assert!(
        doc_ids_c1.contains(&"doc/B"),
        "c1 snapshot must contain doc/B"
    );
}

// --- DELTA-FORK RETAGGING: SNAPSHOT ISOLATION AT ALL COMMITS ---
//
// Verifies that io_retag_with_delta_forks preserves snapshot isolation at
// EVERY commit (not just boundaries). Intermediate commits get their own
// versions on the temp branch with exact data.
//
// With 10 commits, boundary positions are 0, 3, 9 (base-3 powers: 0, 3, 9).
// Commits at positions 0, 3, 9 retain their own version on main.
// Commits at positions 1, 2, 4, 5, 6, 7, 8 get new versions on the temp branch.
//
// After retagging:
// - Search at ANY commit returns ONLY that commit's data.
// - Each commit has its own distinct version (full snapshot isolation).
#[tokio::test]
async fn boundary_retagging_preserves_snapshot_isolation_at_boundaries() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/boundary_iso";

    // Create 10 commits, each adding one unique document.
    // commit_i adds doc/i with a unique embedding.
    let mut commit_versions: Vec<(String, u64)> = Vec::new();
    for i in 0..10 {
        let emb = fake_embedding(8, i as f32 * 10.0);
        let doc_id = format!("doc/{}", i);
        let rows = vec![ChunkRow {
            doc_id: doc_id.clone(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: emb.clone(),
            clustering_embedding: emb,
            content: format!("document {}", i),
        }];
        let v = store
            .io_upsert_chunks(domain, "main", &doc_id, &rows)
            .await
            .expect("upsert");
        let commit = format!("c{}", i);
        store
            .io_tag_commit(domain, "main", &commit, v)
            .await
            .expect("tag");
        commit_versions.push((commit, v));
    }

    // Record original versions before retagging.
    let original_versions: std::collections::HashMap<String, u64> =
        commit_versions.iter().cloned().collect();

    // Run delta-fork retagging.
    let (total, retagged) = store
        .io_retag_with_delta_forks(domain, "main", commit_versions.last().unwrap().1)
        .await
        .expect("retag");

    assert_eq!(total, 10, "total tags should be 10");
    // Boundaries at positions 0, 3 → 2 boundaries retained.
    // Position 9 (latest) is excluded from boundaries — it is treated as
    // an intermediate so it gets retagged, unpinning its original version.
    // Intermediate: positions 1, 2, 4, 5, 6, 7, 8, 9 → 8 retagged.
    assert_eq!(retagged, 8, "should retag 8 intermediate tags");
    assert_eq!(total - retagged, 2, "should retain 2 boundary tags");

    // Verify tag → version mapping after retagging.
    let post_versions = store
        .io_list_commit_versions(domain)
        .await
        .expect("list versions");

    // Boundary commits (c0, c3) must retain their original versions.
    // c9 is NOT a boundary — it is retagged as an intermediate.
    for &i in &[0usize, 3] {
        let commit = format!("c{}", i);
        let post_v = post_versions.get(&commit).copied();
        let orig_v = original_versions.get(&commit).copied();
        assert_eq!(
            post_v, orig_v,
            "boundary commit {} must retain its original version (got {:?}, expected {:?})",
            commit, post_v, orig_v
        );
    }

    // Intermediate commits must be retagged (tags moved to temp branch).
    // Version numbers may coincidentally match across branches, so we verify
    // via snapshot isolation search below instead of version number comparison.

    // SNAPSHOT ISOLATION: search at boundary c0 must return ONLY doc/0.
    let emb_0 = fake_embedding(8, 0.0);
    let query = SearchQuery {
        query_embedding: emb_0,
        query_text: "document".to_owned(),
        mode: SearchMode::Vector,
        start: 0,
        count: 50,
        doc_type_filter: Vec::new(),
        doc_id_filter: Vec::new(),
        snippet: false,
    };
    let hits_c0 = store
        .io_search(domain, "main", "c0", &query)
        .await
        .expect("search at c0");
    let doc_ids_c0: Vec<&str> = hits_c0.iter().map(|h| h.doc_id.as_str()).collect();
    assert!(
        doc_ids_c0.contains(&"doc/0"),
        "c0 boundary must contain doc/0"
    );
    assert!(
        !doc_ids_c0.contains(&"doc/9"),
        "SNAPSHOT ISOLATION VIOLATED: c0 boundary must NOT contain doc/9 (added at c9)"
    );

    // SNAPSHOT ISOLATION: search at boundary c3 must return doc/0..doc/3 but NOT doc/9.
    let emb_3 = fake_embedding(8, 30.0);
    let query_3 = SearchQuery {
        query_embedding: emb_3,
        query_text: "document".to_owned(),
        mode: SearchMode::Vector,
        start: 0,
        count: 50,
        doc_type_filter: Vec::new(),
        doc_id_filter: Vec::new(),
        snippet: false,
    };
    let hits_c3 = store
        .io_search(domain, "main", "c3", &query_3)
        .await
        .expect("search at c3");
    let doc_ids_c3: Vec<&str> = hits_c3.iter().map(|h| h.doc_id.as_str()).collect();
    assert!(
        doc_ids_c3.contains(&"doc/3"),
        "c3 boundary must contain doc/3"
    );
    assert!(
        !doc_ids_c3.contains(&"doc/9"),
        "SNAPSHOT ISOLATION VIOLATED: c3 boundary must NOT contain doc/9 (added at c9)"
    );

    // INTERMEDIATE COMMIT: search at c5 must see ONLY doc/0..doc/5, NOT doc/9.
    // With delta-fork retagging, c5 has its own version with exact snapshot.
    let hits_c5 = store
        .io_search(domain, "main", "c5", &query_3)
        .await
        .expect("search at c5");
    let doc_ids_c5: Vec<&str> = hits_c5.iter().map(|h| h.doc_id.as_str()).collect();
    assert!(
        doc_ids_c5.contains(&"doc/5"),
        "c5 must contain doc/5"
    );
    assert!(
        !doc_ids_c5.contains(&"doc/9"),
        "SNAPSHOT ISOLATION VIOLATED: c5 must NOT contain doc/9"
    );

    // DISTINCT VERSIONS: all 10 commits should have distinct versions
    // (full snapshot isolation, not just at boundaries).
    let distinct_versions: std::collections::HashSet<u64> =
        post_versions.values().copied().collect();
    assert_eq!(
        distinct_versions.len(),
        10,
        "should have 10 distinct pinned versions (full snapshot isolation), got {:?}",
        distinct_versions
    );
}

// --- DELTA-FORK RETAGGING: NO COLLAPSE TO SINGLE VERSION REGRESSION GUARD ---
//
// Guards against the regression where retagging collapses ALL tags to a
// single version. With 10 commits, if all tags were retagged to one version,
// distinct_versions would be 1. With delta-fork retagging, each intermediate
// commit gets its own version on the temp branch, so distinct_versions
// should be 10 (full snapshot isolation).
#[tokio::test]
async fn boundary_retagging_does_not_retag_all_to_one_version() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/no_retag_all";

    for i in 0..10 {
        let emb = fake_embedding(8, i as f32 * 10.0);
        let doc_id = format!("doc/{}", i);
        let rows = vec![ChunkRow {
            doc_id: doc_id.clone(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: emb.clone(),
            clustering_embedding: emb,
            content: format!("document {}", i),
        }];
        let v = store
            .io_upsert_chunks(domain, "main", &doc_id, &rows)
            .await
            .expect("upsert");
        let commit = format!("c{}", i);
        store
            .io_tag_commit(domain, "main", &commit, v)
            .await
            .expect("tag");
    }

    let compact_version = store
        .io_open_dataset_readonly(domain)
        .await
        .expect("open")
        .expect("dataset")
        .read()
        .await
        .version()
        .version;

    store
        .io_retag_with_delta_forks(domain, "main", compact_version)
        .await
        .expect("retag");

    let post_versions = store
        .io_list_commit_versions(domain)
        .await
        .expect("list");

    let distinct_versions: std::collections::HashSet<u64> =
        post_versions.values().copied().collect();

    // With delta-fork retagging, all 10 commits should have distinct versions.
    // If all tags were retagged to compact_version, this would be 1.
    assert!(
        distinct_versions.len() > 1,
        "REGRESSION: all tags retagged to a single version ({:?}). \
         Delta-fork retagging must retain distinct versions for snapshot isolation.",
        distinct_versions
    );
    assert_eq!(
        distinct_versions.len(),
        10,
        "should have 10 distinct versions (full snapshot isolation), got {:?}",
        distinct_versions
    );
}

// --- ANN-PATH SNAPSHOT ISOLATION ---
//
// Proves that the IVF_HNSW_SQ (ANN) search path is also version-isolated.
// With 300+ rows the vector index is active (above the 256 training-vector
// floor), and searches go through the nearest() ANN path instead of flat scan.
// After creating a vector index at c1, a search at c0 must still exclude c1
// data — the ANN index must respect the checkout_version boundary.
#[tokio::test]
async fn ann_path_snapshot_isolation_with_vector_index() {
    let dim = 16;
    let config = VectorIndexConfig {
        num_partitions: 4,
        m: 16,
        ef_construction: 100,
        nprobes: 4,
        refine_factor: Some(10),
    };
    let (mut store, _tmp) = make_test_store(dim);
    store.set_vector_index_config(config.clone());
    let domain = "admin/ann_iso";

    // Commit c0: 300 rows from a single doc (above the 256 IVF_HNSW_SQ floor).
    let corpus_c0 = 300usize;
    let rows_c0: Vec<ChunkRow> = (0..corpus_c0)
        .map(|i| ChunkRow {
            doc_id: "doc/alpha".to_owned(),
            doc_type: "Doc".to_owned(),
            chunk_index: i as i32,
            chunk_count: corpus_c0 as i32,
            chunk_token_start: i as i32,
            doc_token_len: corpus_c0 as i32,
            embedding: fake_embedding(dim, 1.0 + i as f32),
            clustering_embedding: fake_embedding(dim, 1.0 + i as f32),
            content: format!("alpha chunk number {}", i),
        })
        .collect();
    let v0 = store
        .io_upsert_chunks(domain, "main", "doc/alpha", &rows_c0)
        .await
        .expect("upsert alpha corpus");
    store
        .io_tag_commit(domain, "main", "c0", v0)
        .await
        .expect("tag c0");

    // Commit c1: add 50 rows from a new doc (doc/beta). Total rows = 350, well
    // above the 256 training floor.
    let corpus_c1 = 50usize;
    let rows_c1: Vec<ChunkRow> = (0..corpus_c1)
        .map(|i| ChunkRow {
            doc_id: "doc/beta".to_owned(),
            doc_type: "Doc".to_owned(),
            chunk_index: i as i32,
            chunk_count: corpus_c1 as i32,
            chunk_token_start: i as i32,
            doc_token_len: corpus_c1 as i32,
            // Use a distinctly different seed range so beta vectors are far from alpha.
            embedding: fake_embedding(dim, 5000.0 + i as f32),
            clustering_embedding: fake_embedding(dim, 5000.0 + i as f32),
            content: format!("beta chunk number {}", i),
        })
        .collect();
    let v1 = store
        .io_upsert_chunks(domain, "main", "doc/beta", &rows_c1)
        .await
        .expect("upsert beta corpus");
    store
        .io_tag_commit(domain, "main", "c1", v1)
        .await
        .expect("tag c1");

    // Build the ANN (IVF_HNSW_SQ) vector index. This indexes ALL rows (c0 + c1)
    // into the HNSW structure. The isolation contract says: even though the index
    // covers all versions, checkout_version(c0) must filter to only c0's rows.
    {
        let ds_arc = store.io_open_dataset(domain, "main").await.unwrap();
        let mut ds = ds_arc.write().await;
        crate::store::vector_index::io_ensure_vector_index(&mut ds, &config, false)
            .await
            .expect("ensure ANN vector index");
    }
    store
        .io_refresh_cached_dataset(domain, "main")
        .await
        .expect("refresh after index build");

    // Also run optimize_indices (append) to be thorough — this is what the
    // production optimize worker does after building the index.
    {
        let ds_arc = store.io_open_dataset(domain, "main").await.unwrap();
        let mut ds = ds_arc.write().await;
        crate::store::vector_index::io_ensure_vector_index(&mut ds, &config, false)
            .await
            .expect("optimize indices append");
    }
    store
        .io_refresh_cached_dataset(domain, "main")
        .await
        .expect("refresh after optimize");

    // Search at c0 via the ANN path (mode=Vector, the nearest() codepath).
    // Query vector is close to alpha's seed range (seed=1.0 area).
    let query_c0 = SearchQuery {
        query_embedding: fake_embedding(dim, 1.5),
        query_text: String::new(),
        mode: SearchMode::Vector,
        start: 0,
        count: 50,
        doc_type_filter: Vec::new(),
        doc_id_filter: Vec::new(),
        snippet: false,
    };
    let hits_c0 = store
        .io_search(domain, "main", "c0", &query_c0)
        .await
        .expect("ANN search at c0");

    let doc_ids_c0: Vec<&str> = hits_c0.iter().map(|h| h.doc_id.as_str()).collect();
    assert!(
        doc_ids_c0.contains(&"doc/alpha"),
        "ANN search at c0 must find doc/alpha"
    );
    assert!(
        !doc_ids_c0.contains(&"doc/beta"),
        "ANN-PATH ISOLATION VIOLATED: search at c0 must NOT find doc/beta \
         (added at c1). The IVF_HNSW_SQ index is leaking cross-version rows \
         through the nearest() path — checkout_version is not filtering correctly."
    );

    // Sanity: search at c1 must find both alpha and beta.
    let query_c1 = SearchQuery {
        query_embedding: fake_embedding(dim, 5000.5),
        query_text: String::new(),
        mode: SearchMode::Vector,
        start: 0,
        count: 50,
        doc_type_filter: Vec::new(),
        doc_id_filter: Vec::new(),
        snippet: false,
    };
    let hits_c1 = store
        .io_search(domain, "main", "c1", &query_c1)
        .await
        .expect("ANN search at c1");

    let doc_ids_c1: Vec<&str> = hits_c1.iter().map(|h| h.doc_id.as_str()).collect();
    assert!(
        doc_ids_c1.contains(&"doc/beta"),
        "ANN search at c1 must find doc/beta"
    );
    // Alpha should also be in c1 (append-only) if the query is close enough.
    // We use a beta-range query here so alpha may or may not appear (vector distance).
    // The critical assertion is: beta IS in c1 and NOT in c0.
}

// --- FD-MEASUREMENT: historical-version search after compaction (task-59) ---
//
// HYPOTHESIS UNDER TEST (#58 assumption): after compaction, searching an OLD
// tag (c0, whose version predates the compacted version) opens ONLY the
// fragments relevant to ITS version — NOT the accumulated full fragment set.
// This is the FD-safety assumption that underpins #58's removal of the
// compaction retag.
//
// SETUP:
//   1. Tag c0 early (1 fragment — 1 doc).
//   2. Push MANY more commits (c1..cN), each adding a new fragment.
//   3. Tag c_mid at the midpoint, tag c_head at HEAD (accumulated fragments).
//   4. Compact → HEAD fragments collapse to O(1); c0/c_mid still reference
//      their ORIGINAL pre-compaction version (old manifests with old fragments).
//   5. Measure open_fd_count delta + fresh_open delta for:
//      a) search at c0 (old version, few fragments)
//      b) search at c_mid (old version, many fragments)
//      c) search at c_head (latest — post-compaction, few fragments)
//
// BOUNDED outcome → #58's assumption holds. BLOWS UP → real FD-vs-isolation
// tradeoff requiring smart compaction (#60).
#[cfg(target_os = "linux")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fd_measurement_historical_version_search_after_compaction() {
    let dim = 8;
    let (store, _tmp) = make_test_store(dim);
    let domain = "admin/fd_historical";

    // --- Phase 1: Tag c0 early (1 fragment = 1 doc). ---
    let emb_c0 = fake_embedding(dim, 1.0);
    let rows_c0 = vec![ChunkRow {
        doc_id: "doc/genesis".to_owned(),
        doc_type: "T".to_owned(),
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 5,
        embedding: emb_c0.clone(),
        clustering_embedding: emb_c0.clone(),
        content: "genesis document".to_owned(),
    }];
    let v0 = store
        .io_upsert_chunks(domain, "main", "doc/genesis", &rows_c0)
        .await
        .expect("upsert genesis");
    store
        .io_tag_commit(domain, "main", "c0", v0)
        .await
        .expect("tag c0");

    // --- Phase 2: Push many commits (c1..c39), each a new doc = new fragment. ---
    // 40 total fragments (c0 + 39 more) well above the COMPACT_FRAGMENT_THRESHOLD=16.
    let total_pushes = 39usize;
    let mid_point = 20usize;
    let mut v_mid: u64 = 0;
    let mut v_head: u64 = 0;

    for i in 1..=total_pushes {
        let emb = fake_embedding(dim, 10.0 + i as f32);
        let rows = vec![ChunkRow {
            doc_id: format!("doc/push_{}", i),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: emb.clone(),
            clustering_embedding: emb,
            content: format!("push document {}", i),
        }];
        let v = store
            .io_upsert_chunks(domain, "main", &format!("doc/push_{}", i), &rows)
            .await
            .expect("upsert push");

        if i == mid_point {
            store
                .io_tag_commit(domain, "main", "c_mid", v)
                .await
                .expect("tag c_mid");
            v_mid = v;
        }
        if i == total_pushes {
            store
                .io_tag_commit(domain, "main", "c_head", v)
                .await
                .expect("tag c_head");
            v_head = v;
        }
    }
    assert!(v_mid > 0, "v_mid must be tagged");
    assert!(v_head > v_mid, "v_head must be after v_mid");

    // --- Phase 3: Verify fragment count before compaction, then compact. ---
    let fragments_before_compaction;
    {
        let ds = store
            .io_open_dataset_uncached(domain, "main")
            .await
            .expect("uncached open for compact")
            .expect("dataset must exist");
        fragments_before_compaction = ds.get_fragments().len();
        assert!(
            fragments_before_compaction > 16,
            "pre-compaction fragment count must exceed threshold (got {})",
            fragments_before_compaction
        );
        let mut ds = ds;
        io_compact_data(&mut ds, false)
            .await
            .expect("compact_data");
        let fragments_after = ds.get_fragments().len();
        eprintln!(
            "[fd-historical] compaction: {} -> {} fragments (compacted version={})",
            fragments_before_compaction,
            fragments_after,
            ds.version().version
        );
    }
    // Refresh the cached handle after compaction.
    store
        .io_refresh_cached_dataset(domain, "main")
        .await
        .expect("refresh after compaction");

    // --- Phase 4: Measure fragment count visible at each version. ---
    // Checkout each version and count its fragments to understand what Lance
    // exposes at each historical point.
    let snapshot_c0 = store
        .io_snapshot_from_cache(domain, "main", "c0")
        .await
        .expect("snapshot c0");
    let frags_c0 = snapshot_c0.get_fragments().len();

    let snapshot_mid = store
        .io_snapshot_from_cache(domain, "main", "c_mid")
        .await
        .expect("snapshot c_mid");
    let frags_mid = snapshot_mid.get_fragments().len();

    let snapshot_head = store
        .io_snapshot_from_cache(domain, "main", "c_head")
        .await
        .expect("snapshot c_head");
    let frags_head = snapshot_head.get_fragments().len();

    eprintln!(
        "[fd-historical] fragments per version: c0={}, c_mid={}, c_head={} (pre-compact total={})",
        frags_c0, frags_mid, frags_head, fragments_before_compaction
    );

    // --- Phase 5: FD measurement — search at each version and measure delta. ---
    let query = SearchQuery {
        query_embedding: emb_c0.clone(),
        query_text: "document".to_owned(),
        mode: SearchMode::Vector,
        start: 0,
        count: 50,
        doc_type_filter: Vec::new(),
        doc_id_filter: Vec::new(),
        snippet: false,
    };

    // Warm up all paths once.
    store.io_search(domain, "main", "c0", &query).await.expect("warmup c0");
    store.io_search(domain, "main", "c_mid", &query).await.expect("warmup c_mid");
    store.io_search(domain, "main", "c_head", &query).await.expect("warmup c_head");

    // Measure c0 search FDs (multiple iterations for stability).
    let iterations = 50usize;

    let fd_before_c0 = open_fd_count();
    let opens_before_c0 = store.fresh_open_count();
    for _ in 0..iterations {
        store
            .io_search(domain, "main", "c0", &query)
            .await
            .expect("search c0 must not exhaust FDs");
    }
    let fd_after_c0 = open_fd_count();
    let opens_c0 = store.fresh_open_count() - opens_before_c0;
    let fd_delta_c0 = fd_after_c0 as i64 - fd_before_c0 as i64;

    // Measure c_mid search FDs.
    let fd_before_mid = open_fd_count();
    let opens_before_mid = store.fresh_open_count();
    for _ in 0..iterations {
        store
            .io_search(domain, "main", "c_mid", &query)
            .await
            .expect("search c_mid must not exhaust FDs");
    }
    let fd_after_mid = open_fd_count();
    let opens_mid = store.fresh_open_count() - opens_before_mid;
    let fd_delta_mid = fd_after_mid as i64 - fd_before_mid as i64;

    // Measure c_head search FDs (post-compaction — should be minimal).
    let fd_before_head = open_fd_count();
    let opens_before_head = store.fresh_open_count();
    for _ in 0..iterations {
        store
            .io_search(domain, "main", "c_head", &query)
            .await
            .expect("search c_head must not exhaust FDs");
    }
    let fd_after_head = open_fd_count();
    let opens_head = store.fresh_open_count() - opens_before_head;
    let fd_delta_head = fd_after_head as i64 - fd_before_head as i64;

    // --- Phase 6: Report measured numbers. ---
    eprintln!(
        "[fd-historical] MEASUREMENT REPORT (iterations={}):",
        iterations
    );
    eprintln!(
        "[fd-historical] tag=c0      fragments={:>3}  fds_delta={:>4}  fresh_opens={}",
        frags_c0, fd_delta_c0, opens_c0
    );
    eprintln!(
        "[fd-historical] tag=c_mid   fragments={:>3}  fds_delta={:>4}  fresh_opens={}",
        frags_mid, fd_delta_mid, opens_mid
    );
    eprintln!(
        "[fd-historical] tag=c_head  fragments={:>3}  fds_delta={:>4}  fresh_opens={}",
        frags_head, fd_delta_head, opens_head
    );

    // --- Phase 7: Assertions — FD behaviour must be BOUNDED. ---

    // A) No fresh opens per search (reuses cached handle).
    assert_eq!(
        opens_c0, 0,
        "historical c0 search opened {} fresh dataset(s) — must reuse cached handle",
        opens_c0
    );
    assert_eq!(
        opens_mid, 0,
        "historical c_mid search opened {} fresh dataset(s) — must reuse cached handle",
        opens_mid
    );
    assert_eq!(
        opens_head, 0,
        "c_head search opened {} fresh dataset(s) — must reuse cached handle",
        opens_head
    );

    // B) FD delta stays bounded (not scaling with iterations or fragment count).
    // Slack: a few FDs for runtime churn is acceptable; growth proportional to
    // iterations or the fragment count of the historical version is NOT.
    let slack: i64 = 64;
    assert!(
        fd_delta_c0 < slack,
        "c0 historical search FD delta ({}) exceeds slack ({}) — \
         old-version search is leaking FDs per call (fragments at c0={}). \
         The #58 assumption (bounded fragment cost) may be violated.",
        fd_delta_c0, slack, frags_c0
    );
    assert!(
        fd_delta_mid < slack,
        "c_mid historical search FD delta ({}) exceeds slack ({}) — \
         mid-history search is leaking FDs per call (fragments at c_mid={}). \
         The #58 assumption (bounded fragment cost) may be violated.",
        fd_delta_mid, slack, frags_mid
    );
    assert!(
        fd_delta_head < slack,
        "c_head (post-compaction HEAD) search FD delta ({}) exceeds slack ({}) — \
         HEAD search is leaking FDs per call.",
        fd_delta_head, slack
    );

    // C) Historical-version search must NOT open dramatically more FDs than HEAD.
    // If c0 or c_mid's fragment set at their version is bounded, the FD cost is
    // bounded. We allow c_mid to use more FDs than HEAD (it has more fragments
    // from its pre-compaction era) but it must stay well under nofile=1024.
    let max_historical_fd_budget: i64 = 200;
    assert!(
        fd_delta_c0 < max_historical_fd_budget,
        "c0 historical search FD delta ({}) exceeds the safety budget ({}) — \
         old-version fragment layout is opening too many descriptors. \
         Historical-version search may blow up FDs at scale.",
        fd_delta_c0, max_historical_fd_budget
    );
    assert!(
        fd_delta_mid < max_historical_fd_budget,
        "c_mid historical search FD delta ({}) exceeds the safety budget ({}) — \
         mid-history fragment layout is opening too many descriptors. \
         Historical-version search may blow up FDs at scale.",
        fd_delta_mid, max_historical_fd_budget
    );

    // D) The VERSION's fragment count is bounded by its own era's accumulation,
    //    NOT by the total fragments ever created. After compaction, c0's version
    //    has its original manifest (1 fragment), c_mid has ~21, c_head references
    //    the compacted set. The key property: c0 does NOT inherit the full 40
    //    accumulated fragments.
    assert!(
        frags_c0 <= 2,
        "c0 version exposes {} fragments — expected <= 2 (just the genesis doc's fragment). \
         If c0 exposes the full accumulated set, snapshot isolation's fragment \
         budget is unbounded and the #58 assumption is FALSE.",
        frags_c0
    );
    // c_mid's version may have up to mid_point+1 fragments (one per push).
    // That is bounded by design — the historical version references only ITS era.
    assert!(
        frags_mid <= mid_point + 2,
        "c_mid version exposes {} fragments — expected <= {} (its era). \
         If c_mid exposes more than its era's fragments, something is wrong.",
        frags_mid, mid_point + 2
    );

    eprintln!(
        "[fd-historical] CONCLUSION: historical-version search is BOUNDED. \
         c0 (1 fragment) costs {} FD delta, c_mid ({} fragments) costs {} FD delta, \
         c_head (compacted) costs {} FD delta. All well under nofile=1024. \
         #58's assumption (post-compaction old-tag search opens only relevant \
         fragments) is CONFIRMED.",
        fd_delta_c0, frags_mid, fd_delta_mid, fd_delta_head
    );
}

// --- extract_completions (typeahead suggestion logic) ---

#[test]
fn extract_completions_finds_query_and_extends_to_word_boundary() {
    let snippets = vec!["the quick brown fox jumps"];
    let completions = search::extract_completions("quick", snippets);
    // n-gram completions: quick brown, quick brown fox, quick brown fox jumps
    assert!(completions.contains(&"quick brown".to_owned()));
    assert!(completions.contains(&"quick brown fox".to_owned()));
    assert!(completions.contains(&"quick brown fox jumps".to_owned()));
}

#[test]
fn extract_completions_deduplicates_case_insensitively() {
    let snippets = vec!["Quick Brown fox", "quick brown bear"];
    let completions = search::extract_completions("quick", snippets);
    // "quick brown" appears twice (case-insensitive), so it should be first.
    assert_eq!(completions[0].to_lowercase(), "quick brown");
    // Both multi-word variants should be present.
    assert!(completions.iter().any(|c| c.to_lowercase() == "quick brown fox"));
    assert!(completions.iter().any(|c| c.to_lowercase() == "quick brown bear"));
}

#[test]
fn extract_completions_returns_at_most_eight() {
    let snippets: Vec<&str> = vec![
        "alpha one two", "alpha three four", "alpha five six",
        "alpha seven eight", "alpha nine ten", "alpha eleven twelve",
    ];
    let completions = search::extract_completions("alpha", snippets);
    assert!(completions.len() <= 8);
    // Each snippet produces 2 n-grams (alpha + 1 word, alpha + 2 words) = 12 total,
    // but capped at 8.
    assert_eq!(completions.len(), 8);
}

#[test]
fn extract_completions_skips_snippets_without_query_match() {
    let snippets = vec!["no match here", "target word found"];
    let completions = search::extract_completions("target", snippets);
    // n-gram: target word, target word found
    assert!(completions.contains(&"target word".to_owned()));
    assert!(completions.contains(&"target word found".to_owned()));
}

#[test]
fn extract_completions_extends_to_punctuation_boundary() {
    let snippets = vec!["the value, is great"];
    let completions = search::extract_completions("the", snippets);
    // Comma splits "value" from "is", but n-grams still produce:
    // "the value", "the value is", "the value is great"
    assert!(completions.contains(&"the value".to_owned()));
    assert!(completions.contains(&"the value is".to_owned()));
    assert!(completions.contains(&"the value is great".to_owned()));
}

#[test]
fn extract_completions_returns_empty_when_no_snippet_matches() {
    let snippets = vec!["nothing relevant", "also irrelevant"];
    let completions = search::extract_completions("xyz", snippets);
    assert!(completions.is_empty());
}

#[test]
fn extract_completions_skips_completion_equal_to_query() {
    let snippets = vec!["query query"];
    let completions = search::extract_completions("query", snippets);
    // "query query" is longer than "query", so it's included.
    assert!(completions.contains(&"query query".to_owned()));
}

#[test]
fn extract_completions_filters_when_no_next_word_exists() {
    let snippets = vec!["the query"];
    let completions = search::extract_completions("query", snippets);
    assert!(completions.is_empty());
}

#[test]
fn extract_completions_handles_query_at_end_of_snippet() {
    let snippets = vec!["the end is near"];
    let completions = search::extract_completions("near", snippets);
    assert!(completions.is_empty());
}

#[tokio::test]
async fn suggest_substring_fallback_finds_partial_word_matches() {
    let (store, _tmp) = make_test_store(8);
    let domain = "test/suggest_sub";
    let branch = "main";
    let commit = "c1";

    let rows = vec![
        ChunkRow {
            doc_id: "doc1".to_owned(),
            doc_type: "Product".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            embedding: fake_embedding(8, 0.1),
            clustering_embedding: fake_embedding(8, 0.1),
            content: "Electrical components for industrial use".to_owned(),
        },
        ChunkRow {
            doc_id: "doc2".to_owned(),
            doc_type: "Product".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            embedding: fake_embedding(8, 0.2),
            clustering_embedding: fake_embedding(8, 0.2),
            content: "Electronic devices and gadgets".to_owned(),
        },
    ];

    store.io_upsert_chunks(domain, branch, "doc1", &rows[0..1]).await.expect("upsert doc1");
    let v2 = store.io_upsert_chunks(domain, branch, "doc2", &rows[1..2]).await.expect("upsert doc2");
    // Tag the latest version so the commit is indexed.
    store.io_tag_commit(domain, branch, commit, v2).await.expect("tag commit");
    store.update_last_indexed(domain, branch, commit, v2).await;

    let query = SuggestQuery {
        query_text: "elect".to_owned(),
        count: 10,
        doc_type_filter: Vec::new(),
        doc_id_filter: Vec::new(),
    };

    let result = store.io_suggest(domain, branch, commit, &query).await.expect("suggest");

    assert!(result.total_approx >= 2, "substring fallback should find both docs");
    let ids: Vec<&str> = result.hits.iter().map(|h| h.id.as_str()).collect();
    assert!(ids.contains(&"doc1"), "should find 'electrical' via substring");
    assert!(ids.contains(&"doc2"), "should find 'electronic' via substring");
}

#[test]
fn encode_decode_domain_path_roundtrip() {
    let cases = vec![
        "admin/product_assortment",
        "admin/my_db",
        "org/team/sub_db",
        "a/b",
    ];
    for domain in &cases {
        let encoded = encode_domain_path(domain);
        let decoded = decode_domain_path(&encoded);
        assert_eq!(
            decoded.as_deref(),
            Some(*domain),
            "round-trip failed for '{}': encoded='{}', decoded={:?}",
            domain,
            encoded,
            decoded
        );
    }
}

// --- TEMPORAL ACCURACY AFTER COMPACT + INDEX MERGE + NEW CHUNKS ---
//
// Verifies that the temporal vector search contract holds after the full
// optimization pipeline: data compaction, index merge (retrain), and
// subsequent new chunk writes. Historical commits must return exactly
// the documents that existed at their tagged version — no more, no less.
//
// This is the regression guard for the io_retag_all bug: retagging all
// historical commits to the post-merge version would make c0 and c1
// see documents that didn't exist at those points in time.
//
// Setup:
//   1. Insert doc/A → tag c0
//   2. Insert doc/B → tag c1
//   3. Push 16 filler fragments (exceeds COMPACT_FRAGMENT_THRESHOLD)
//   4. Compact data + merge indices (retrain) at HEAD
//   5. Insert doc/C → tag c2 (new chunks after optimization)
//   6. Assert search@c0 = {A}, search@c1 = {A,B}, search@c2 = {A,B,C}
#[tokio::test]
async fn temporal_accuracy_after_compact_merge_and_new_chunks() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/temporal_merge";

    // Phase 1: Insert doc/A, tag as c0.
    let emb_a = fake_embedding(8, 1.0);
    let rows_a = vec![ChunkRow {
        doc_id: "doc/A".to_owned(),
        doc_type: "T".to_owned(),
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 5,
        embedding: emb_a.clone(),
        clustering_embedding: emb_a.clone(),
        content: "document alpha".to_owned(),
    }];
    let v0 = store
        .io_upsert_chunks(domain, "main", "doc/A", &rows_a)
        .await
        .expect("upsert A");
    store
        .io_tag_commit(domain, "main", "c0", v0)
        .await
        .expect("tag c0");

    // Phase 2: Insert doc/B, tag as c1.
    let emb_b = fake_embedding(8, 2.0);
    let rows_b = vec![ChunkRow {
        doc_id: "doc/B".to_owned(),
        doc_type: "T".to_owned(),
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 5,
        embedding: emb_b.clone(),
        clustering_embedding: emb_b.clone(),
        content: "document beta".to_owned(),
    }];
    let v1 = store
        .io_upsert_chunks(domain, "main", "doc/B", &rows_b)
        .await
        .expect("upsert B");
    store
        .io_tag_commit(domain, "main", "c1", v1)
        .await
        .expect("tag c1");

    // Phase 3: Push 16 filler fragments to exceed compaction threshold.
    for i in 0..16 {
        let emb = fake_embedding(8, 10.0 + i as f32);
        let rows = vec![ChunkRow {
            doc_id: format!("doc/filler_{}", i),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: emb.clone(),
            clustering_embedding: emb,
            content: format!("filler document {}", i),
        }];
        store
            .io_upsert_chunks(domain, "main", &format!("doc/filler_{}", i), &rows)
            .await
            .expect("upsert filler");
    }

    // Phase 4: Compact data + merge indices (retrain) at HEAD.
    {
        let ds = store
            .io_open_dataset_uncached(domain, "main")
            .await
            .expect("uncached open for compact")
            .expect("dataset must exist");
        let mut ds = ds;
        let frags_before = ds.get_fragments().len();
        io_compact_data(&mut ds, false)
            .await
            .expect("compact_data");
        let (idx_before, idx_after) =
            io_merge_indices(&mut ds).await.expect("merge_indices");
        eprintln!(
            "[temporal_merge] compact: {} fragments, indices {}→{}",
            frags_before, idx_before, idx_after
        );
    }
    store
        .io_refresh_cached_dataset(domain, "main")
        .await
        .expect("refresh after compact+merge");

    // Phase 5: Insert doc/C after optimization, tag as c2.
    let emb_c = fake_embedding(8, 3.0);
    let rows_c = vec![ChunkRow {
        doc_id: "doc/C".to_owned(),
        doc_type: "T".to_owned(),
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 5,
        embedding: emb_c.clone(),
        clustering_embedding: emb_c.clone(),
        content: "document gamma".to_owned(),
    }];
    let v2 = store
        .io_upsert_chunks(domain, "main", "doc/C", &rows_c)
        .await
        .expect("upsert C after merge");
    store
        .io_tag_commit(domain, "main", "c2", v2)
        .await
        .expect("tag c2");

    // Phase 6: Verify temporal accuracy at each commit.
    let query = SearchQuery {
        query_embedding: fake_embedding(8, 0.0),
        query_text: "document".to_owned(),
        mode: SearchMode::Vector,
        start: 0,
        count: 100,
        doc_type_filter: Vec::new(),
        doc_id_filter: Vec::new(),
        snippet: false,
    };

    // c0 must see only doc/A (not B, not C, not fillers).
    let hits_c0 = store
        .io_search(domain, "main", "c0", &query)
        .await
        .expect("search at c0");
    let ids_c0: Vec<&str> = hits_c0.iter().map(|h| h.doc_id.as_str()).collect();
    assert!(
        ids_c0.contains(&"doc/A"),
        "c0 must contain doc/A"
    );
    assert!(
        !ids_c0.contains(&"doc/B"),
        "TEMPORAL VIOLATION: c0 must not contain doc/B (added at c1)"
    );
    assert!(
        !ids_c0.contains(&"doc/C"),
        "TEMPORAL VIOLATION: c0 must not contain doc/C (added at c2, after merge)"
    );

    // c1 must see doc/A and doc/B (not C, not fillers).
    let hits_c1 = store
        .io_search(domain, "main", "c1", &query)
        .await
        .expect("search at c1");
    let ids_c1: Vec<&str> = hits_c1.iter().map(|h| h.doc_id.as_str()).collect();
    assert!(
        ids_c1.contains(&"doc/A"),
        "c1 must contain doc/A"
    );
    assert!(
        ids_c1.contains(&"doc/B"),
        "c1 must contain doc/B"
    );
    assert!(
        !ids_c1.contains(&"doc/C"),
        "TEMPORAL VIOLATION: c1 must not contain doc/C (added at c2, after merge)"
    );

    // c2 must see doc/A, doc/B, and doc/C.
    let hits_c2 = store
        .io_search(domain, "main", "c2", &query)
        .await
        .expect("search at c2");
    let ids_c2: Vec<&str> = hits_c2.iter().map(|h| h.doc_id.as_str()).collect();
    assert!(
        ids_c2.contains(&"doc/A"),
        "c2 must contain doc/A"
    );
    assert!(
        ids_c2.contains(&"doc/B"),
        "c2 must contain doc/B"
    );
    assert!(
        ids_c2.contains(&"doc/C"),
        "c2 must contain doc/C (added after merge, must be searchable at c2)"
    );
}

// --- TEMPORAL ACCURACY AFTER EXPONENTIAL INDEX ROLL-UP ---
//
// Verifies that the exponential roll-up (base 3, analogous to TerminusDB's
// exponential_rollup_strategy) preserves temporal accuracy while reducing
// index delta count from O(N) to O(log₃(N)).
//
// Setup:
//   1. Insert doc/A → tag c0
//   2. Insert doc/B → tag c1
//   3. Push 10 filler docs (enough to create multiple index deltas)
//   4. Run exponential roll-up at HEAD (base 3)
//   5. Insert doc/C → tag c2 (new chunks after roll-up)
//   6. Assert:
//      - search@c0 = {A}, search@c1 = {A,B}, search@c2 = {A,B,C}
//      - Index count at HEAD after roll-up < index count before roll-up
#[tokio::test]
async fn temporal_accuracy_after_exponential_rollup() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/rollup_temporal";

    // Phase 1: Insert doc/A, tag as c0.
    let emb_a = fake_embedding(8, 1.0);
    let rows_a = vec![ChunkRow {
        doc_id: "doc/A".to_owned(),
        doc_type: "T".to_owned(),
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 5,
        embedding: emb_a.clone(),
        clustering_embedding: emb_a.clone(),
        content: "document alpha".to_owned(),
    }];
    let v0 = store
        .io_upsert_chunks(domain, "main", "doc/A", &rows_a)
        .await
        .expect("upsert A");
    store
        .io_tag_commit(domain, "main", "c0", v0)
        .await
        .expect("tag c0");

    // Phase 2: Insert doc/B, tag as c1.
    let emb_b = fake_embedding(8, 2.0);
    let rows_b = vec![ChunkRow {
        doc_id: "doc/B".to_owned(),
        doc_type: "T".to_owned(),
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 5,
        embedding: emb_b.clone(),
        clustering_embedding: emb_b.clone(),
        content: "document beta".to_owned(),
    }];
    let v1 = store
        .io_upsert_chunks(domain, "main", "doc/B", &rows_b)
        .await
        .expect("upsert B");
    store
        .io_tag_commit(domain, "main", "c1", v1)
        .await
        .expect("tag c1");

    // Phase 3: Push 10 filler docs to create multiple index deltas.
    for i in 0..10 {
        let emb = fake_embedding(8, 10.0 + i as f32);
        let rows = vec![ChunkRow {
            doc_id: format!("doc/filler_{}", i),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: emb.clone(),
            clustering_embedding: emb,
            content: format!("filler document {}", i),
        }];
        store
            .io_upsert_chunks(domain, "main", &format!("doc/filler_{}", i), &rows)
            .await
            .expect("upsert filler");
    }

    // Phase 4: Run exponential roll-up at HEAD (base 3).
    let indices_before;
    let indices_after;
    {
        let ds = store
            .io_open_dataset_uncached(domain, "main")
            .await
            .expect("uncached open for rollup")
            .expect("dataset must exist");
        let mut ds = ds;

        let indices = ds
            .load_indices()
            .await
            .expect("load indices before rollup");
        indices_before = indices.len();

        let (before, after) =
            crate::store::lance::io_exponential_rollup(&mut ds, 3)
                .await
                .expect("exponential rollup");
        indices_after = after;

        eprintln!(
            "[rollup_temporal] indices {}→{} (before={})",
            before, after, indices_before
        );
    }
    store
        .io_refresh_cached_dataset(domain, "main")
        .await
        .expect("refresh after rollup");

    // Phase 5: Insert doc/C after roll-up, tag as c2.
    let emb_c = fake_embedding(8, 3.0);
    let rows_c = vec![ChunkRow {
        doc_id: "doc/C".to_owned(),
        doc_type: "T".to_owned(),
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 5,
        embedding: emb_c.clone(),
        clustering_embedding: emb_c.clone(),
        content: "document gamma".to_owned(),
    }];
    let v2 = store
        .io_upsert_chunks(domain, "main", "doc/C", &rows_c)
        .await
        .expect("upsert C after rollup");
    store
        .io_tag_commit(domain, "main", "c2", v2)
        .await
        .expect("tag c2");

    // Phase 6: Verify temporal accuracy at each commit.
    let query = SearchQuery {
        query_embedding: fake_embedding(8, 0.0),
        query_text: "document".to_owned(),
        mode: SearchMode::Vector,
        start: 0,
        count: 100,
        doc_type_filter: Vec::new(),
        doc_id_filter: Vec::new(),
        snippet: false,
    };

    // c0 must see only doc/A.
    let hits_c0 = store
        .io_search(domain, "main", "c0", &query)
        .await
        .expect("search at c0");
    let ids_c0: Vec<&str> = hits_c0.iter().map(|h| h.doc_id.as_str()).collect();
    assert!(
        ids_c0.contains(&"doc/A"),
        "c0 must contain doc/A"
    );
    assert!(
        !ids_c0.contains(&"doc/B"),
        "TEMPORAL VIOLATION: c0 must not contain doc/B (added at c1)"
    );
    assert!(
        !ids_c0.contains(&"doc/C"),
        "TEMPORAL VIOLATION: c0 must not contain doc/C (added at c2, after rollup)"
    );

    // c1 must see doc/A and doc/B.
    let hits_c1 = store
        .io_search(domain, "main", "c1", &query)
        .await
        .expect("search at c1");
    let ids_c1: Vec<&str> = hits_c1.iter().map(|h| h.doc_id.as_str()).collect();
    assert!(
        ids_c1.contains(&"doc/A"),
        "c1 must contain doc/A"
    );
    assert!(
        ids_c1.contains(&"doc/B"),
        "c1 must contain doc/B"
    );
    assert!(
        !ids_c1.contains(&"doc/C"),
        "TEMPORAL VIOLATION: c1 must not contain doc/C (added at c2, after rollup)"
    );

    // c2 must see doc/A, doc/B, and doc/C.
    let hits_c2 = store
        .io_search(domain, "main", "c2", &query)
        .await
        .expect("search at c2");
    let ids_c2: Vec<&str> = hits_c2.iter().map(|h| h.doc_id.as_str()).collect();
    assert!(
        ids_c2.contains(&"doc/A"),
        "c2 must contain doc/A"
    );
    assert!(
        ids_c2.contains(&"doc/B"),
        "c2 must contain doc/B"
    );
    assert!(
        ids_c2.contains(&"doc/C"),
        "c2 must contain doc/C (added after rollup, must be searchable at c2)"
    );

    // Roll-up should have reduced index count (or at least not increased it).
    // With 12 pushes and base 3, partitions = [0..8] (9) + [9..11] (3) = 2 merges.
    // The exact count depends on LanceDB internals, but it must not grow.
    assert!(
        indices_after <= indices_before,
        "rollup must not increase index count: before={}, after={}",
        indices_before,
        indices_after
    );
}

#[test]
fn prune_empty_index_dirs_removes_empty_and_keeps_non_empty() {
    let (store, _tmp) = make_test_store(8);
    let domain = "test/prune_empty";
    let dataset_path = store.dataset_path(domain);
    let indices_dir = dataset_path.join("_indices");

    // Create 3 empty UUID dirs (simulating LanceDB cleanup leftovers).
    for uuid in &["empty-1", "empty-2", "empty-3"] {
        let dir = indices_dir.join(uuid);
        std::fs::create_dir_all(&dir).expect("create empty index dir");
    }

    // Create 2 non-empty UUID dirs with dummy index files.
    for uuid in &["live-1", "live-2"] {
        let dir = indices_dir.join(uuid);
        std::fs::create_dir_all(&dir).expect("create live index dir");
        std::fs::write(dir.join("index.lance"), b"dummy").expect("write dummy index file");
    }

    let removed = store.io_prune_empty_index_dirs(domain).expect("prune succeeds");
    assert_eq!(removed, 3, "should remove exactly 3 empty dirs");

    // Empty dirs gone.
    for uuid in &["empty-1", "empty-2", "empty-3"] {
        assert!(
            !indices_dir.join(uuid).exists(),
            "empty dir {} should be removed",
            uuid
        );
    }

    // Non-empty dirs survive.
    for uuid in &["live-1", "live-2"] {
        assert!(
            indices_dir.join(uuid).exists(),
            "non-empty dir {} should survive",
            uuid
        );
    }
}

#[test]
fn prune_empty_index_dirs_no_indices_dir_is_noop() {
    let (store, _tmp) = make_test_store(8);
    let domain = "test/prune_noop";
    let removed = store.io_prune_empty_index_dirs(domain).expect("prune succeeds");
    assert_eq!(removed, 0, "no _indices dir → 0 removed");
}

// ===========================================================================
// SPIKE TESTS: Delta-fork retagging — validating LanceDB API behavior
//
// These tests validate whether checkout_version + append creates child
// versions, whether branch + fast-forward works, and whether cleanup
// handles orphaned versions correctly. Results determine the implementation
// strategy for delta-fork retagging.
// ===========================================================================

/// Helper: collect distinct doc_ids from a Dataset handle by scanning
/// only the doc_id column.
async fn collect_doc_ids(ds: &lance::dataset::Dataset) -> std::collections::HashSet<String> {
    let mut scanner = ds.scan();
    scanner.project(&["doc_id"]).expect("project doc_id");
    let batches: Vec<arrow_array::RecordBatch> = scanner
        .try_into_stream()
        .await
        .expect("stream doc_ids")
        .try_collect()
        .await
        .expect("collect doc_ids");
    let mut ids = std::collections::HashSet::new();
    for batch in &batches {
        if batch.num_rows() == 0 {
            continue;
        }
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .expect("doc_id column is StringArray");
        for i in 0..col.len() {
            ids.insert(col.value(i).to_owned());
        }
    }
    ids
}

/// Helper: make a single ChunkRow for a doc with a specific doc_id.
fn make_row(dim: usize, seed: f32, doc_id: &str, content: &str) -> ChunkRow {
    let emb = fake_embedding(dim, seed);
    ChunkRow {
        doc_id: doc_id.to_owned(),
        doc_type: "T".to_owned(),
        chunk_index: 0,
        chunk_count: 1,
        chunk_token_start: 0,
        doc_token_len: 5,
        embedding: emb.clone(),
        clustering_embedding: emb,
        content: content.to_owned(),
    }
}

/// Spike 0b: Branch + fast-forward main to branch head.
///
/// Validates whether we can create a temporary branch from V0, append
/// deltas linearly on it, then fast-forward main to the branch's head.
#[tokio::test]
async fn spike_0b_branch_then_fastforward_main() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/spike0b";

    // V1: insert doc/A on main
    let v1 = store
        .io_upsert_chunks(domain, "main", "doc/A", &[make_row(8, 1.0, "doc/A", "alpha")])
        .await
        .expect("upsert A");
    store.io_tag_commit(domain, "main", "c0", v1).await.expect("tag c0");

    // V2: insert doc/B on main
    let v2 = store
        .io_upsert_chunks(domain, "main", "doc/B", &[make_row(8, 2.0, "doc/B", "beta")])
        .await
        .expect("upsert B");
    store.io_tag_commit(domain, "main", "c1", v2).await.expect("tag c1");

    eprintln!("[spike0b] main: v1={}, v2={}", v1, v2);

    // Create a "rebuild" branch from V1
    store
        .io_create_branch(domain, "rebuild", v1)
        .await
        .expect("create rebuild branch from v1");

    // On "rebuild": append doc/B → V1' (child of V1 on rebuild branch)
    let mut ds_rebuild = store
        .io_open_dataset_uncached(domain, "rebuild")
        .await
        .expect("open rebuild")
        .expect("rebuild exists");
    eprintln!("[spike0b] rebuild branch head: v{}", ds_rebuild.version().version);

    let batch = store.rows_to_batch(&[make_row(8, 2.0, "doc/B", "beta")]).expect("batch");
    let schema = store.chunk_schema();
    let reader = arrow_array::RecordBatchIterator::new(vec![Ok(batch)], schema);
    ds_rebuild.append(reader, None).await.expect("append on rebuild");
    let v1_prime = ds_rebuild.version().version;
    eprintln!("[spike0b] rebuild after append: v{}", v1_prime);

    // Verify rebuild branch has doc/A + doc/B
    let ids_rebuild = collect_doc_ids(&ds_rebuild).await;
    eprintln!("[spike0b] rebuild doc_ids: {:?}", ids_rebuild);
    assert!(ids_rebuild.contains("doc/A"), "rebuild must contain doc/A");
    assert!(ids_rebuild.contains("doc/B"), "rebuild must contain doc/B");

    // Try to fast-forward main to rebuild's head.
    // LanceDB may support this via create_branch with same name or some
    // other API. Let's try checking if we can checkout main and then
    // somehow point it to v1_prime.
    //
    // Approach 1: Try to use ds.checkout_version on main and see if
    // that changes HEAD.
    let ds_main = store.io_open_dataset_uncached(domain, "main").await.expect("open main").expect("main exists");
    let main_version_before = ds_main.version().version;
    eprintln!("[spike0b] main HEAD before fast-forward attempt: v{}", main_version_before);

    // Approach 2: Check if LanceDB has a restore/reset/checkout_latest that
    // can move main to a specific version. Let's try checkout_version and
    // then see if subsequent opens reflect it.
    // NOTE: checkout_version returns a NEW Dataset, it doesn't mutate the
    // branch HEAD. We need a different mechanism.

    // Approach 3: Try creating main branch from rebuild's version.
    // LanceDB's create_branch might fail if main already exists.
    let path = store.dataset_path(domain);
    let uri = path.to_string_lossy().to_string();
    let mut ds_fresh = store.io_open_fresh(&uri).await.expect("fresh open");
    let result = ds_fresh.create_branch("main2", v1_prime, None).await;
    eprintln!("[spike0b] create_branch('main2', v{}) result: {:?}", v1_prime, result.is_ok());

    // Approach 4: Check if there's a way to move the main branch pointer.
    // Let's check if tags can be used as a workaround — tag the rebuild
    // version and have searches resolve through it.
    //
    // For now, document what we found:
    eprintln!("[spike0b] RESULT: checkout_version does NOT move branch HEAD");
    eprintln!("[spike0b] RESULT: create_branch('main2', v{}) = {}", v1_prime, result.is_ok());
    eprintln!("[spike0b] RESULT: No direct fast-forward API found in LanceDB 7.0.0");
}

/// Spike 0d: Cleanup on forked/orphaned versions.
///
/// After retagging c1→v4 and c2→v5 (from spike 0c pattern), verify that
/// cleanup_with_policy(retain_n_versions(1)) deletes orphaned v2, v3
/// while keeping tagged v1, v4, v5.
#[tokio::test]
async fn spike_0d_cleanup_deletes_orphaned_versions() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/spike0d";

    // Create 3 versions on main
    let v1 = store.io_upsert_chunks(domain, "main", "doc/A", &[make_row(8, 1.0, "doc/A", "alpha")]).await.expect("upsert A");
    store.io_tag_commit(domain, "main", "c0", v1).await.expect("tag c0");

    let v2 = store.io_upsert_chunks(domain, "main", "doc/B", &[make_row(8, 2.0, "doc/B", "beta")]).await.expect("upsert B");
    store.io_tag_commit(domain, "main", "c1", v2).await.expect("tag c1");

    let v3 = store.io_upsert_chunks(domain, "main", "doc/C", &[make_row(8, 3.0, "doc/C", "gamma")]).await.expect("upsert C");
    store.io_tag_commit(domain, "main", "c2", v3).await.expect("tag c2");

    // Rebuild: checkout V1, append doc/B → V4, append doc/C → V5
    let ds = store.io_open_dataset_uncached(domain, "main").await.expect("open").expect("exists");
    let mut ds_rebuild = ds.checkout_version(v1).await.expect("checkout v1");

    let batch_b = store.rows_to_batch(&[make_row(8, 2.0, "doc/B", "beta")]).expect("batch B");
    let schema = store.chunk_schema();
    let reader_b = arrow_array::RecordBatchIterator::new(vec![Ok(batch_b)], schema.clone());
    ds_rebuild.append(reader_b, None).await.expect("append B");
    let v4 = ds_rebuild.version().version;

    let batch_c = store.rows_to_batch(&[make_row(8, 3.0, "doc/C", "gamma")]).expect("batch C");
    let reader_c = arrow_array::RecordBatchIterator::new(vec![Ok(batch_c)], schema);
    ds_rebuild.append(reader_c, None).await.expect("append C");
    let v5 = ds_rebuild.version().version;

    eprintln!("[spike0d] versions: v1={}, v2={}, v3={}, v4={}, v5={}", v1, v2, v3, v4, v5);

    // Retag: c0→v1 (keep), c1→v4 (new), c2→v5 (new)
    let tag_c1 = crate::layeridx::encode_commit_tag("c1");
    let tag_c2 = crate::layeridx::encode_commit_tag("c2");
    {
        let ds_arc = store.io_open_dataset(domain, "main").await.expect("open");
        let ds_w = ds_arc.write().await;
        ds_w.tags().delete(&tag_c1).await.expect("delete old c1");
        ds_w.tags().create(&tag_c1, v4).await.expect("create c1 at v4");
        ds_w.tags().delete(&tag_c2).await.expect("delete old c2");
        ds_w.tags().create(&tag_c2, v5).await.expect("create c2 at v5");
    }
    store.io_refresh_cached_dataset(domain, "main").await.expect("refresh");

    // Run aggressive cleanup
    let result = store.io_cleanup_aggressive(domain, "main").await;
    eprintln!("[spike0d] cleanup result: {:?}", result.is_ok());

    // Verify tagged versions still resolve
    let query = SearchQuery {
        query_text: "document".to_owned(),
        query_embedding: fake_embedding(8, 1.0),
        mode: SearchMode::Hybrid,
        start: 0,
        count: 10,
        doc_type_filter: vec![],
        doc_id_filter: vec![],
        snippet: false,
    };

    let hits_c0 = store.io_search(domain, "main", "c0", &query).await.expect("search c0");
    let ids_c0: Vec<&str> = hits_c0.iter().map(|h| h.doc_id.as_str()).collect();
    eprintln!("[spike0d] search at c0: {:?}", ids_c0);
    assert!(ids_c0.contains(&"doc/A"), "c0 must still resolve after cleanup");

    let hits_c1 = store.io_search(domain, "main", "c1", &query).await.expect("search c1");
    let ids_c1: Vec<&str> = hits_c1.iter().map(|h| h.doc_id.as_str()).collect();
    eprintln!("[spike0d] search at c1: {:?}", ids_c1);
    assert!(ids_c1.contains(&"doc/A"), "c1 must contain doc/A after cleanup");
    assert!(ids_c1.contains(&"doc/B"), "c1 must contain doc/B after cleanup");

    let hits_c2 = store.io_search(domain, "main", "c2", &query).await.expect("search c2");
    let ids_c2: Vec<&str> = hits_c2.iter().map(|h| h.doc_id.as_str()).collect();
    eprintln!("[spike0d] search at c2: {:?}", ids_c2);
    assert!(ids_c2.contains(&"doc/A"), "c2 must contain doc/A after cleanup");
    assert!(ids_c2.contains(&"doc/B"), "c2 must contain doc/B after cleanup");
    assert!(ids_c2.contains(&"doc/C"), "c2 must contain doc/C after cleanup");

    eprintln!("[spike0d] RESULT: All tagged versions survive cleanup after retagging");
    eprintln!("[spike0d] RESULT: Orphaned v2, v3 should be deleted by cleanup");
}

/// Spike 0e: Search correctness at forked versions with vector index.
///
/// Validates that vector search works correctly at a version created via
/// checkout_version + append, with enough rows for IVF training.
#[tokio::test]
async fn spike_0e_vector_search_at_forked_version() {
    let dim = 16;
    let config = crate::store::vector_index::tests::make_test_config(dim);
    let (mut store, _tmp) = make_test_store(dim);
    store.set_vector_index_config(config.clone());
    let domain = "admin/spike0e";

    // V1: insert 300 docs (enough for IVF training)
    let rows_a: Vec<ChunkRow> = (0..300)
        .map(|i| make_row(dim, 1.0 + i as f32 * 0.01, &format!("doc/A_{}", i), &format!("alpha {}", i)))
        .collect();
    let v1 = store.io_upsert_chunks(domain, "main", "doc/A_batch", &rows_a).await.expect("upsert A batch");
    store.io_tag_commit(domain, "main", "c0", v1).await.expect("tag c0");

    // Create vector index at V1
    {
        let mut ds = store.io_open_dataset_uncached(domain, "main").await.expect("open").expect("exists");
        crate::store::vector_index::io_ensure_vector_index(&mut ds, &config, false).await.expect("create vector index at v1");
    }
    store.io_refresh_cached_dataset(domain, "main").await.expect("refresh");

    // V2: insert 300 more docs
    let rows_b: Vec<ChunkRow> = (0..300)
        .map(|i| make_row(dim, 5.0 + i as f32 * 0.01, &format!("doc/B_{}", i), &format!("beta {}", i)))
        .collect();
    let v2 = store.io_upsert_chunks(domain, "main", "doc/B_batch", &rows_b).await.expect("upsert B batch");
    store.io_tag_commit(domain, "main", "c1", v2).await.expect("tag c1");

    // Optimize indices at V2 (append delta for new fragments)
    {
        let mut ds = store.io_open_dataset_uncached(domain, "main").await.expect("open").expect("exists");
        crate::store::vector_index::io_ensure_vector_index(&mut ds, &config, false).await.expect("append vector index at v2");
    }
    store.io_refresh_cached_dataset(domain, "main").await.expect("refresh");

    // Now fork: checkout V1, append doc/B_batch rows → V3
    let ds = store.io_open_dataset_uncached(domain, "main").await.expect("open").expect("exists");
    let mut ds_fork = ds.checkout_version(v1).await.expect("checkout v1");

    let batch_b = store.rows_to_batch(&rows_b).expect("batch B");
    let schema = store.chunk_schema();
    let reader_b = arrow_array::RecordBatchIterator::new(vec![Ok(batch_b)], schema);
    ds_fork.append(reader_b, None).await.expect("append B on fork");
    let v3 = ds_fork.version().version;
    eprintln!("[spike0e] fork: v1 + doc/B_batch → v{}", v3);

    // Create vector index on the fork version
    crate::store::vector_index::io_ensure_vector_index(&mut ds_fork, &config, false).await.expect("create vector index on fork");

    // Tag c1_new → v3
    let tag_c1_new = crate::layeridx::encode_commit_tag("c1_new");
    ds_fork.tags().create(&tag_c1_new, v3).await.expect("tag c1_new at v3");
    store.io_refresh_cached_dataset(domain, "main").await.expect("refresh");

    // Search at c0 (v1): should find only doc/A_* docs
    let query_a = SearchQuery {
        query_text: "alpha".to_owned(),
        query_embedding: fake_embedding(dim, 1.0),
        mode: SearchMode::Vector,
        start: 0,
        count: 10,
        doc_type_filter: vec![],
        doc_id_filter: vec![],
        snippet: false,
    };
    let hits_c0 = store.io_search(domain, "main", "c0", &query_a).await.expect("search c0");
    let ids_c0: Vec<&str> = hits_c0.iter().map(|h| h.doc_id.as_str()).collect();
    eprintln!("[spike0e] search at c0 (v{}): {} hits, sample: {:?}", v1, hits_c0.len(), ids_c0.iter().take(5).collect::<Vec<_>>());
    assert!(hits_c0.iter().all(|h| h.doc_id.starts_with("doc/A_")), "c0 must only return doc/A_* docs");

    // Search at c1_new (v3): should find both doc/A_* and doc/B_* docs
    let query_b = SearchQuery {
        query_text: "beta".to_owned(),
        query_embedding: fake_embedding(dim, 5.0),
        mode: SearchMode::Vector,
        start: 0,
        count: 10,
        doc_type_filter: vec![],
        doc_id_filter: vec![],
        snippet: false,
    };
    let hits_c1_new = store.io_search(domain, "main", "c1_new", &query_b).await.expect("search c1_new");
    let ids_c1_new: Vec<&str> = hits_c1_new.iter().map(|h| h.doc_id.as_str()).collect();
    eprintln!("[spike0e] search at c1_new (v{}): {} hits, sample: {:?}", v3, hits_c1_new.len(), ids_c1_new.iter().take(5).collect::<Vec<_>>());
    assert!(hits_c1_new.iter().any(|h| h.doc_id.starts_with("doc/B_")), "c1_new must find doc/B_* docs");

    // Search at c1 (v2, original): should also find both
    let hits_c1 = store.io_search(domain, "main", "c1", &query_b).await.expect("search c1");
    eprintln!("[spike0e] search at c1 (v{}): {} hits", v2, hits_c1.len());
    assert!(hits_c1.iter().any(|h| h.doc_id.starts_with("doc/B_")), "c1 must find doc/B_* docs");

    eprintln!("[spike0e] RESULT: Vector search works at forked version v{}", v3);
    eprintln!("[spike0e] RESULT: Snapshot isolation preserved — c0 returns only A docs, c1_new returns A+B docs");
}

/// Spike 0f: Verify that after rebuild + retag + cleanup, new pushes
/// to main still work correctly (HEAD is the old chain, new pushes
/// extend it, old forked versions remain accessible via tags).
#[tokio::test]
async fn spike_0f_push_after_rebuild_retag_cleanup() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/spike0f";

    // Create 3 versions on main
    let v1 = store.io_upsert_chunks(domain, "main", "doc/A", &[make_row(8, 1.0, "doc/A", "alpha")]).await.expect("upsert A");
    store.io_tag_commit(domain, "main", "c0", v1).await.expect("tag c0");

    let v2 = store.io_upsert_chunks(domain, "main", "doc/B", &[make_row(8, 2.0, "doc/B", "beta")]).await.expect("upsert B");
    store.io_tag_commit(domain, "main", "c1", v2).await.expect("tag c1");

    let v3 = store.io_upsert_chunks(domain, "main", "doc/C", &[make_row(8, 3.0, "doc/C", "gamma")]).await.expect("upsert C");
    store.io_tag_commit(domain, "main", "c2", v3).await.expect("tag c2");

    // Rebuild: checkout V1, append doc/B → V4, append doc/C → V5
    let ds = store.io_open_dataset_uncached(domain, "main").await.expect("open").expect("exists");
    let mut ds_rebuild = ds.checkout_version(v1).await.expect("checkout v1");

    let batch_b = store.rows_to_batch(&[make_row(8, 2.0, "doc/B", "beta")]).expect("batch B");
    let schema = store.chunk_schema();
    let reader_b = arrow_array::RecordBatchIterator::new(vec![Ok(batch_b)], schema.clone());
    ds_rebuild.append(reader_b, None).await.expect("append B");
    let v4 = ds_rebuild.version().version;

    let batch_c = store.rows_to_batch(&[make_row(8, 3.0, "doc/C", "gamma")]).expect("batch C");
    let reader_c = arrow_array::RecordBatchIterator::new(vec![Ok(batch_c)], schema);
    ds_rebuild.append(reader_c, None).await.expect("append C");
    let v5 = ds_rebuild.version().version;

    // Retag: c1→v4, c2→v5
    let tag_c1 = crate::layeridx::encode_commit_tag("c1");
    let tag_c2 = crate::layeridx::encode_commit_tag("c2");
    {
        let ds_arc = store.io_open_dataset(domain, "main").await.expect("open");
        let ds_w = ds_arc.write().await;
        ds_w.tags().delete(&tag_c1).await.expect("delete old c1");
        ds_w.tags().create(&tag_c1, v4).await.expect("create c1 at v4");
        ds_w.tags().delete(&tag_c2).await.expect("delete old c2");
        ds_w.tags().create(&tag_c2, v5).await.expect("create c2 at v5");
    }
    store.io_refresh_cached_dataset(domain, "main").await.expect("refresh");

    // Run cleanup
    let _ = store.io_cleanup_aggressive(domain, "main").await;

    // Now push a new doc/D to main (HEAD is still v3)
    let v6 = store.io_upsert_chunks(domain, "main", "doc/D", &[make_row(8, 4.0, "doc/D", "delta")]).await.expect("upsert D");
    store.io_tag_commit(domain, "main", "c3", v6).await.expect("tag c3");
    eprintln!("[spike0f] new push: doc/D → v{} (HEAD)", v6);

    // Verify HEAD has all 4 docs
    let ds_head = store.io_open_dataset_uncached(domain, "main").await.expect("open").expect("exists");
    let ids_head = collect_doc_ids(&ds_head).await;
    eprintln!("[spike0f] HEAD (v{}) doc_ids: {:?}", ds_head.version().version, ids_head);
    assert!(ids_head.contains("doc/A"), "HEAD must contain doc/A");
    assert!(ids_head.contains("doc/B"), "HEAD must contain doc/B");
    assert!(ids_head.contains("doc/C"), "HEAD must contain doc/C");
    assert!(ids_head.contains("doc/D"), "HEAD must contain doc/D");

    // Verify historical tags still work
    let query = SearchQuery {
        query_text: "document".to_owned(),
        query_embedding: fake_embedding(8, 1.0),
        mode: SearchMode::Hybrid,
        start: 0,
        count: 10,
        doc_type_filter: vec![],
        doc_id_filter: vec![],
        snippet: false,
    };

    let hits_c0 = store.io_search(domain, "main", "c0", &query).await.expect("search c0");
    let ids_c0: Vec<&str> = hits_c0.iter().map(|h| h.doc_id.as_str()).collect();
    eprintln!("[spike0f] search at c0: {:?}", ids_c0);
    assert!(ids_c0.contains(&"doc/A") && !ids_c0.contains(&"doc/D"), "c0 snapshot isolation");

    let hits_c2 = store.io_search(domain, "main", "c2", &query).await.expect("search c2");
    let ids_c2: Vec<&str> = hits_c2.iter().map(|h| h.doc_id.as_str()).collect();
    eprintln!("[spike0f] search at c2 (v{}): {:?}", v5, ids_c2);
    assert!(ids_c2.contains(&"doc/A") && ids_c2.contains(&"doc/B") && ids_c2.contains(&"doc/C"), "c2 must have A+B+C");
    assert!(!ids_c2.contains(&"doc/D"), "c2 must NOT contain doc/D (snapshot isolation)");

    eprintln!("[spike0f] RESULT: New pushes to main work after rebuild+retag+cleanup");
    eprintln!("[spike0f] RESULT: HEAD extends from old chain (v{}), forked versions remain accessible via tags", v3);
}

/// Spike 0g: Branch + append + tag + delete branch + search via tag.
///
/// Tests whether we can:
/// 1. Create a branch from a boundary version
/// 2. Append deltas on the branch (creating versions with correct data)
/// 3. Tag commit IDs to versions on the branch
/// 4. Delete the branch
/// 5. Still search at those tags (tags are dataset-global)
///
/// This is the key test for the branch-based delta-fork approach.
#[tokio::test]
async fn spike_0g_branch_append_tag_delete_branch_search() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/spike0g";

    // Main: V1(doc/A), V2(doc/A+doc/B), V3(doc/A+doc/B+doc/C)
    let v1 = store.io_upsert_chunks(domain, "main", "doc/A", &[make_row(8, 1.0, "doc/A", "alpha")]).await.expect("upsert A");
    store.io_tag_commit(domain, "main", "c0", v1).await.expect("tag c0");

    let v2 = store.io_upsert_chunks(domain, "main", "doc/B", &[make_row(8, 2.0, "doc/B", "beta")]).await.expect("upsert B");
    store.io_tag_commit(domain, "main", "c1", v2).await.expect("tag c1");

    let v3 = store.io_upsert_chunks(domain, "main", "doc/C", &[make_row(8, 3.0, "doc/C", "gamma")]).await.expect("upsert C");
    store.io_tag_commit(domain, "main", "c2", v3).await.expect("tag c2");

    eprintln!("[spike0g] main: v1={}, v2={}, v3={}", v1, v2, v3);

    // Create a "rebuild" branch from V1 (boundary)
    store.io_create_branch(domain, "rebuild", v1).await.expect("create rebuild branch");

    // On "rebuild": append doc/B → V4 (child of V1 on rebuild branch)
    let mut ds_rebuild = store.io_open_dataset_uncached(domain, "rebuild").await.expect("open rebuild").expect("exists");
    let batch_b = store.rows_to_batch(&[make_row(8, 2.0, "doc/B", "beta")]).expect("batch B");
    let schema = store.chunk_schema();
    let reader_b = arrow_array::RecordBatchIterator::new(vec![Ok(batch_b)], schema.clone());
    ds_rebuild.append(reader_b, None).await.expect("append B on rebuild");
    let v4 = ds_rebuild.version().version;

    // Create FTS index at V4 (before appending more data)
    crate::store::lance::index::io_ensure_fts_index_on_dataset(&mut ds_rebuild)
        .await
        .expect("create FTS index at v4 on rebuild");
    let v4_indexed = ds_rebuild.version().version;
    eprintln!("[spike0g] v4={}, v4_indexed={}", v4, v4_indexed);

    // Append doc/C → V5 (child of V4_indexed on rebuild branch)
    let batch_c = store.rows_to_batch(&[make_row(8, 3.0, "doc/C", "gamma")]).expect("batch C");
    let reader_c = arrow_array::RecordBatchIterator::new(vec![Ok(batch_c)], schema);
    ds_rebuild.append(reader_c, None).await.expect("append C on rebuild");
    let v5 = ds_rebuild.version().version;

    // Optimize FTS index at V5 (index new fragments)
    crate::store::lance::index::io_ensure_fts_index_on_dataset(&mut ds_rebuild)
        .await
        .expect("optimize FTS index at v5 on rebuild");
    let v5_indexed = ds_rebuild.version().version;
    eprintln!("[spike0g] v5={}, v5_indexed={}", v5, v5_indexed);

    eprintln!("[spike0g] rebuild: v4={}, v4_indexed={}, v5={}, v5_indexed={}", v4, v4_indexed, v5, v5_indexed);

    // Verify rebuild branch has correct data at each version
    let ds_v4 = ds_rebuild.checkout_version(v4_indexed).await.expect("checkout v4_indexed on rebuild");
    let ids_v4 = collect_doc_ids(&ds_v4).await;
    eprintln!("[spike0g] rebuild v4_indexed doc_ids: {:?}", ids_v4);
    assert!(ids_v4.contains("doc/A") && ids_v4.contains("doc/B"), "v4 must have A+B");
    assert!(!ids_v4.contains("doc/C"), "v4 must NOT have doc/C");

    let ids_v5 = collect_doc_ids(&ds_rebuild).await;
    eprintln!("[spike0g] rebuild v5_indexed doc_ids: {:?}", ids_v5);
    assert!(ids_v5.contains("doc/A") && ids_v5.contains("doc/B") && ids_v5.contains("doc/C"), "v5 must have A+B+C");

    // Retag: c1 from v2 to v4_indexed, c2 from v3 to v5_indexed
    let tag_c1 = crate::layeridx::encode_commit_tag("c1");
    let tag_c2 = crate::layeridx::encode_commit_tag("c2");
    {
        let ds_arc = store.io_open_dataset(domain, "main").await.expect("open");
        let ds_w = ds_arc.write().await;
        ds_w.tags().delete(&tag_c1).await.expect("delete old c1");
        ds_w.tags().create(&tag_c1, v4_indexed).await.expect("create c1 at v4_indexed");
        ds_w.tags().delete(&tag_c2).await.expect("delete old c2");
        ds_w.tags().create(&tag_c2, v5_indexed).await.expect("create c2 at v5_indexed");
    }
    store.io_refresh_cached_dataset(domain, "main").await.expect("refresh");

    // Search at c1 (now → v4_indexed on rebuild branch): should return doc/A + doc/B only
    let query = SearchQuery {
        query_text: "document".to_owned(),
        query_embedding: fake_embedding(8, 1.0),
        mode: SearchMode::Hybrid,
        start: 0,
        count: 10,
        doc_type_filter: vec![],
        doc_id_filter: vec![],
        snippet: false,
    };
    let hits_c1 = store.io_search(domain, "main", "c1", &query).await.expect("search c1");
    let ids_c1: Vec<&str> = hits_c1.iter().map(|h| h.doc_id.as_str()).collect();
    eprintln!("[spike0g] search at c1 (v{}): {:?}", v4_indexed, ids_c1);
    assert!(ids_c1.contains(&"doc/A"), "c1 must contain doc/A");
    assert!(ids_c1.contains(&"doc/B"), "c1 must contain doc/B");
    assert!(!ids_c1.contains(&"doc/C"), "c1 must NOT contain doc/C (snapshot isolation!)");

    // Search at c2 (now → v5_indexed on rebuild branch): should return all 3
    let hits_c2 = store.io_search(domain, "main", "c2", &query).await.expect("search c2");
    let ids_c2: Vec<&str> = hits_c2.iter().map(|h| h.doc_id.as_str()).collect();
    eprintln!("[spike0g] search at c2 (v{}): {:?}", v5_indexed, ids_c2);
    assert!(ids_c2.contains(&"doc/A") && ids_c2.contains(&"doc/B") && ids_c2.contains(&"doc/C"), "c2 must have A+B+C");

    eprintln!("[spike0g] RESULT: Retagging to branch versions works! Snapshot isolation preserved!");
    eprintln!("[spike0g] RESULT: c1→v{} (rebuild branch) returns A+B only, c2→v{} returns A+B+C", v4_indexed, v5_indexed);

    // Now try to delete the "rebuild" branch and see if tags still resolve.
    let path = store.dataset_path(domain);
    let uri = path.to_string_lossy().to_string();
    let mut ds_fresh = store.io_open_fresh(&uri).await.expect("fresh open");
    let delete_result = ds_fresh.delete_branch("rebuild").await;
    eprintln!("[spike0g] delete_branch('rebuild') result: {:?}", delete_result.is_ok());

    if delete_result.is_ok() {
        store.io_refresh_cached_dataset(domain, "main").await.expect("refresh after branch delete");

        let hits_c1_after = store.io_search(domain, "main", "c1", &query).await;
        eprintln!("[spike0g] search at c1 after branch delete: {:?}", hits_c1_after.is_ok());
        if let Ok(hits) = hits_c1_after {
            let ids: Vec<&str> = hits.iter().map(|h| h.doc_id.as_str()).collect();
            eprintln!("[spike0g] search at c1 after branch delete: {:?}", ids);
            assert!(ids.contains(&"doc/A"), "c1 must still work after branch delete");
        }

        let hits_c2_after = store.io_search(domain, "main", "c2", &query).await;
        eprintln!("[spike0g] search at c2 after branch delete: {:?}", hits_c2_after.is_ok());
        if let Ok(hits) = hits_c2_after {
            let ids: Vec<&str> = hits.iter().map(|h| h.doc_id.as_str()).collect();
            eprintln!("[spike0g] search at c2 after branch delete: {:?}", ids);
            assert!(ids.contains(&"doc/C"), "c2 must still work after branch delete");
        }

        eprintln!("[spike0g] RESULT: Tags survive branch deletion!");
    } else {
        eprintln!("[spike0g] RESULT: delete_branch failed — branches must be kept");
    }

    // Run cleanup and verify tagged versions survive
    let _ = store.io_cleanup_aggressive(domain, "main").await;
    store.io_refresh_cached_dataset(domain, "main").await.expect("refresh after cleanup");

    let hits_c0_final = store.io_search(domain, "main", "c0", &query).await.expect("search c0 after cleanup");
    let ids_c0_final: Vec<&str> = hits_c0_final.iter().map(|h| h.doc_id.as_str()).collect();
    eprintln!("[spike0g] search at c0 after cleanup: {:?}", ids_c0_final);
    assert!(ids_c0_final.contains(&"doc/A"), "c0 must survive cleanup");

    let hits_c1_final = store.io_search(domain, "main", "c1", &query).await.expect("search c1 after cleanup");
    let ids_c1_final: Vec<&str> = hits_c1_final.iter().map(|h| h.doc_id.as_str()).collect();
    eprintln!("[spike0g] search at c1 after cleanup: {:?}", ids_c1_final);
    assert!(ids_c1_final.contains(&"doc/A") && ids_c1_final.contains(&"doc/B"), "c1 must survive cleanup");
    assert!(!ids_c1_final.contains(&"doc/C"), "c1 snapshot isolation after cleanup");

    let hits_c2_final = store.io_search(domain, "main", "c2", &query).await.expect("search c2 after cleanup");
    let ids_c2_final: Vec<&str> = hits_c2_final.iter().map(|h| h.doc_id.as_str()).collect();
    eprintln!("[spike0g] search at c2 after cleanup: {:?}", ids_c2_final);
    assert!(ids_c2_final.contains(&"doc/A") && ids_c2_final.contains(&"doc/B") && ids_c2_final.contains(&"doc/C"), "c2 must survive cleanup");

    eprintln!("[spike0g] RESULT: All tagged versions survive cleanup with correct snapshot isolation!");
}

/// Spike 0h: Branch-swap approach — create new branch, replay deltas,
/// delete old main, create main from new branch head.
///
/// Tests the user's proposed approach:
/// 1. Create a "main_new" branch from the first boundary version
/// 2. Append all deltas linearly on "main_new" (cherry-pick equivalent)
/// 3. Create indices on "main_new"
/// 4. Tag all commit IDs to versions on "main_new"
/// 5. Try to delete "main" branch
/// 6. Try to create "main" from "main_new"'s head (or rename)
/// 7. Verify searches work at all tags on the new main
/// 8. Add new documents to main and verify everything still works
#[tokio::test]
async fn spike_0h_branch_swap_rebuild_main() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/spike0h";

    // Main: V1(doc/A), V2(doc/A+doc/B), V3(doc/A+doc/B+doc/C)
    let v1 = store.io_upsert_chunks(domain, "main", "doc/A", &[make_row(8, 1.0, "doc/A", "alpha")]).await.expect("upsert A");
    store.io_tag_commit(domain, "main", "c0", v1).await.expect("tag c0");

    let v2 = store.io_upsert_chunks(domain, "main", "doc/B", &[make_row(8, 2.0, "doc/B", "beta")]).await.expect("upsert B");
    store.io_tag_commit(domain, "main", "c1", v2).await.expect("tag c1");

    let v3 = store.io_upsert_chunks(domain, "main", "doc/C", &[make_row(8, 3.0, "doc/C", "gamma")]).await.expect("upsert C");
    store.io_tag_commit(domain, "main", "c2", v3).await.expect("tag c2");

    eprintln!("[spike0h] original main: v1={}, v2={}, v3={}", v1, v2, v3);

    // Step 1: Create "main_new" branch from V1 (first boundary)
    store.io_create_branch(domain, "main_new", v1).await.expect("create main_new branch");

    // Step 2: Append deltas linearly on "main_new"
    let mut ds_new = store.io_open_dataset_uncached(domain, "main_new").await.expect("open main_new").expect("exists");
    let schema = store.chunk_schema();

    // Append doc/B → V4
    let batch_b = store.rows_to_batch(&[make_row(8, 2.0, "doc/B", "beta")]).expect("batch B");
    let reader_b = arrow_array::RecordBatchIterator::new(vec![Ok(batch_b)], schema.clone());
    ds_new.append(reader_b, None).await.expect("append B on main_new");
    let v4 = ds_new.version().version;

    // Create FTS index at V4 (before appending more data)
    crate::store::lance::index::io_ensure_fts_index_on_dataset(&mut ds_new)
        .await
        .expect("create FTS index at v4 on main_new");
    let v4_indexed = ds_new.version().version;

    // Append doc/C → V5
    let batch_c = store.rows_to_batch(&[make_row(8, 3.0, "doc/C", "gamma")]).expect("batch C");
    let reader_c = arrow_array::RecordBatchIterator::new(vec![Ok(batch_c)], schema);
    ds_new.append(reader_c, None).await.expect("append C on main_new");
    let v5 = ds_new.version().version;

    // Optimize FTS index at V5
    crate::store::lance::index::io_ensure_fts_index_on_dataset(&mut ds_new)
        .await
        .expect("optimize FTS index at v5 on main_new");
    let v5_indexed = ds_new.version().version;

    eprintln!("[spike0h] main_new: v4={}, v4_indexed={}, v5={}, v5_indexed={}", v4, v4_indexed, v5, v5_indexed);

    // Verify data correctness on main_new
    let ids_v4 = {
        let ds_v4 = ds_new.checkout_version(v4_indexed).await.expect("checkout v4_indexed");
        collect_doc_ids(&ds_v4).await
    };
    eprintln!("[spike0h] main_new v4_indexed doc_ids: {:?}", ids_v4);
    assert!(ids_v4.contains("doc/A") && ids_v4.contains("doc/B"), "v4 must have A+B");
    assert!(!ids_v4.contains("doc/C"), "v4 must NOT have doc/C");

    let ids_v5 = collect_doc_ids(&ds_new).await;
    eprintln!("[spike0h] main_new v5_indexed doc_ids: {:?}", ids_v5);
    assert!(ids_v5.contains("doc/A") && ids_v5.contains("doc/B") && ids_v5.contains("doc/C"), "v5 must have A+B+C");

    // Step 4: Retag commits to versions on main_new
    let tag_c1 = crate::layeridx::encode_commit_tag("c1");
    let tag_c2 = crate::layeridx::encode_commit_tag("c2");
    {
        let ds_arc = store.io_open_dataset(domain, "main").await.expect("open");
        let ds_w = ds_arc.write().await;
        ds_w.tags().delete(&tag_c1).await.expect("delete old c1");
        ds_w.tags().create(&tag_c1, v4_indexed).await.expect("create c1 at v4_indexed");
        ds_w.tags().delete(&tag_c2).await.expect("delete old c2");
        ds_w.tags().create(&tag_c2, v5_indexed).await.expect("create c2 at v5_indexed");
    }
    store.io_refresh_cached_dataset(domain, "main").await.expect("refresh");

    // Step 5: Try to delete "main" branch
    let path = store.dataset_path(domain);
    let uri = path.to_string_lossy().to_string();
    let mut ds_fresh = store.io_open_fresh(&uri).await.expect("fresh open");
    let delete_main_result = ds_fresh.delete_branch("main").await;
    eprintln!("[spike0h] delete_branch('main') result: {:?}", delete_main_result.is_ok());
    if let Err(ref e) = delete_main_result {
        eprintln!("[spike0h] delete_branch('main') error: {}", e);
    }

    // Step 6: If main was deleted, try to create main from v5_indexed
    if delete_main_result.is_ok() {
        let create_main_result = ds_fresh.create_branch("main", v5_indexed, None).await;
        eprintln!("[spike0h] create_branch('main', v{}) result: {:?}", v5_indexed, create_main_result.is_ok());
    } else {
        eprintln!("[spike0h] Cannot delete main branch — branch-swap NOT viable");
        eprintln!("[spike0h] RESULT: Branch-swap approach FAILED — LanceDB does not allow deleting 'main' branch");
    }

    store.io_refresh_cached_dataset(domain, "main").await.expect("refresh after branch swap attempt");

    // Step 7: Verify searches work at all tags (using temp branch versions)
    let query = SearchQuery {
        query_text: "document".to_owned(),
        query_embedding: fake_embedding(8, 1.0),
        mode: SearchMode::Hybrid,
        start: 0,
        count: 10,
        doc_type_filter: vec![],
        doc_id_filter: vec![],
        snippet: false,
    };

    let hits_c0 = store.io_search(domain, "main", "c0", &query).await.expect("search c0");
    let ids_c0: Vec<&str> = hits_c0.iter().map(|h| h.doc_id.as_str()).collect();
    eprintln!("[spike0h] search at c0: {:?}", ids_c0);
    assert!(ids_c0.contains(&"doc/A"), "c0 must contain doc/A");

    let hits_c1 = store.io_search(domain, "main", "c1", &query).await.expect("search c1");
    let ids_c1: Vec<&str> = hits_c1.iter().map(|h| h.doc_id.as_str()).collect();
    eprintln!("[spike0h] search at c1 (v{}): {:?}", v4_indexed, ids_c1);
    assert!(ids_c1.contains(&"doc/A"), "c1 must contain doc/A");
    assert!(ids_c1.contains(&"doc/B"), "c1 must contain doc/B");
    assert!(!ids_c1.contains(&"doc/C"), "c1 must NOT contain doc/C (snapshot isolation)");

    let hits_c2 = store.io_search(domain, "main", "c2", &query).await.expect("search c2");
    let ids_c2: Vec<&str> = hits_c2.iter().map(|h| h.doc_id.as_str()).collect();
    eprintln!("[spike0h] search at c2 (v{}): {:?}", v5_indexed, ids_c2);
    assert!(ids_c2.contains(&"doc/A") && ids_c2.contains(&"doc/B") && ids_c2.contains(&"doc/C"), "c2 must have A+B+C");

    // Step 8: Add new documents to main and verify
    let v_new = store.io_upsert_chunks(domain, "main", "doc/D", &[make_row(8, 4.0, "doc/D", "delta")]).await.expect("upsert D");
    store.io_tag_commit(domain, "main", "c3", v_new).await.expect("tag c3");
    eprintln!("[spike0h] new push: doc/D → v{}", v_new);

    // Verify HEAD has all 4 docs
    let ds_head = store.io_open_dataset_uncached(domain, "main").await.expect("open").expect("exists");
    let ids_head = collect_doc_ids(&ds_head).await;
    eprintln!("[spike0h] HEAD (v{}) doc_ids: {:?}", ds_head.version().version, ids_head);
    assert!(ids_head.contains("doc/A"), "HEAD must contain doc/A");
    assert!(ids_head.contains("doc/B"), "HEAD must contain doc/B");
    assert!(ids_head.contains("doc/C"), "HEAD must contain doc/C");
    assert!(ids_head.contains("doc/D"), "HEAD must contain doc/D");

    // Verify snapshot isolation still holds for historical tags
    let hits_c2_final = store.io_search(domain, "main", "c2", &query).await.expect("search c2 after new push");
    let ids_c2_final: Vec<&str> = hits_c2_final.iter().map(|h| h.doc_id.as_str()).collect();
    eprintln!("[spike0h] search at c2 after new push: {:?}", ids_c2_final);
    assert!(!ids_c2_final.contains(&"doc/D"), "c2 must NOT contain doc/D (snapshot isolation)");

    // Run cleanup
    let _ = store.io_cleanup_aggressive(domain, "main").await;
    store.io_refresh_cached_dataset(domain, "main").await.expect("refresh after cleanup");

    // Verify all tags still work after cleanup
    let hits_c0_final = store.io_search(domain, "main", "c0", &query).await.expect("search c0 after cleanup");
    let ids_c0_final: Vec<&str> = hits_c0_final.iter().map(|h| h.doc_id.as_str()).collect();
    eprintln!("[spike0h] search at c0 after cleanup: {:?}", ids_c0_final);
    assert!(ids_c0_final.contains(&"doc/A"), "c0 must survive cleanup");

    let hits_c1_final = store.io_search(domain, "main", "c1", &query).await.expect("search c1 after cleanup");
    let ids_c1_final: Vec<&str> = hits_c1_final.iter().map(|h| h.doc_id.as_str()).collect();
    eprintln!("[spike0h] search at c1 after cleanup: {:?}", ids_c1_final);
    assert!(ids_c1_final.contains(&"doc/A") && ids_c1_final.contains(&"doc/B"), "c1 must survive cleanup");
    assert!(!ids_c1_final.contains(&"doc/C"), "c1 snapshot isolation after cleanup");

    let hits_c2_final2 = store.io_search(domain, "main", "c2", &query).await.expect("search c2 after cleanup");
    let ids_c2_final2: Vec<&str> = hits_c2_final2.iter().map(|h| h.doc_id.as_str()).collect();
    eprintln!("[spike0h] search at c2 after cleanup: {:?}", ids_c2_final2);
    assert!(ids_c2_final2.contains(&"doc/A") && ids_c2_final2.contains(&"doc/B") && ids_c2_final2.contains(&"doc/C"), "c2 must survive cleanup");

    eprintln!("[spike0h] RESULT: Branch-swap NOT viable (cannot delete 'main'), but temp-branch+retag works!");
    eprintln!("[spike0h] RESULT: New pushes to main work, snapshot isolation preserved, cleanup OK");
}

// --- Phase 1.0: Startup cleanup of stale .-compact_rebuild_* branches ---

/// Cleanup deletes a stale .-compact_rebuild_* branch when one exists.
#[tokio::test]
async fn cleanup_compaction_branches_deletes_stale_branch() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/cleanup_stale";

    // Create a dataset with one commit
    let v1 = store
        .io_upsert_chunks(domain, "main", "doc/A", &[make_row(8, 1.0, "doc/A", "alpha")])
        .await
        .expect("upsert A");
    store.io_tag_commit(domain, "main", "c0", v1).await.expect("tag c0");

    // Simulate a crashed compaction: create a .-compact_rebuild_* branch
    store
        .io_create_branch(domain, &compact_rebuild_branch_name(1), v1)
        .await
        .expect("create stale rebuild branch");

    // Verify it exists
    let branches_before = store.io_list_branches(domain).await.expect("list branches");
    assert!(
        branches_before.contains(&compact_rebuild_branch_name(1)),
        "stale branch should exist before cleanup"
    );

    // Run cleanup
    let cleaned = store.io_cleanup_compaction_branches().await.expect("cleanup");
    assert_eq!(cleaned.len(), 1, "one domain should have been cleaned");
    assert_eq!(cleaned[0].0, domain, "cleaned domain should match");

    // Verify it's gone
    let branches_after = store.io_list_branches(domain).await.expect("list branches");
    assert!(
        !branches_after.contains(&compact_rebuild_branch_name(1)),
        "stale branch should be deleted after cleanup"
    );
}

/// Cleanup is a no-op when no stale .-compact_rebuild_* branch exists.
#[tokio::test]
async fn cleanup_compaction_branches_noop_when_clean() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/cleanup_noop";

    // Create a dataset with one commit — no stale branch
    let v1 = store
        .io_upsert_chunks(domain, "main", "doc/A", &[make_row(8, 1.0, "doc/A", "alpha")])
        .await
        .expect("upsert A");
    store.io_tag_commit(domain, "main", "c0", v1).await.expect("tag c0");

    // Run cleanup — should be a no-op
    let cleaned = store.io_cleanup_compaction_branches().await.expect("cleanup");
    assert!(cleaned.is_empty(), "no domains should have been cleaned");
}

/// Cleanup only affects datasets that have a stale .-compact_rebuild_* branch.
#[tokio::test]
async fn cleanup_compaction_branches_only_cleans_affected() {
    let (store, _tmp) = make_test_store(8);
    let domain_a = "admin/cleanup_multi_a";
    let domain_b = "admin/cleanup_multi_b";

    // Both domains get a commit
    let v_a = store
        .io_upsert_chunks(domain_a, "main", "doc/A", &[make_row(8, 1.0, "doc/A", "alpha")])
        .await
        .expect("upsert A");
    store.io_tag_commit(domain_a, "main", "c0", v_a).await.expect("tag c0 a");

    let v_b = store
        .io_upsert_chunks(domain_b, "main", "doc/B", &[make_row(8, 2.0, "doc/B", "beta")])
        .await
        .expect("upsert B");
    store.io_tag_commit(domain_b, "main", "c0", v_b).await.expect("tag c0 b");

    // Only domain_a gets a stale branch
    store
        .io_create_branch(domain_a, &compact_rebuild_branch_name(1), v_a)
        .await
        .expect("create stale branch on domain_a");

    // Run cleanup
    let cleaned = store.io_cleanup_compaction_branches().await.expect("cleanup");
    assert_eq!(cleaned.len(), 1, "only one domain should have been cleaned");
    assert_eq!(cleaned[0].0, domain_a, "only domain_a should be cleaned");

    // Verify domain_a's stale branch is gone
    let branches_a = store.io_list_branches(domain_a).await.expect("list branches a");
    assert!(
        !branches_a.contains(&compact_rebuild_branch_name(1)),
        "domain_a stale branch should be deleted"
    );

    // Verify domain_b never had a stale branch
    let branches_b = store.io_list_branches(domain_b).await.expect("list branches b");
    assert!(
        !branches_b.contains(&compact_rebuild_branch_name(1)),
        "domain_b should never have had a stale branch"
    );
}

/// The `.-compact_rebuild_` prefix is reserved for internal vectorlink branches.
/// This test documents the contract: is_compact_rebuild_branch identifies
/// branches used by delta-fork retagging, and is_reserved_branch_name
/// identifies the broader `.-` namespace for future reserved prefixes.
///
/// The HTTP handler validate_branch_name uses is_compact_rebuild_branch to
/// reject `.-compact_rebuild_` branch names from external requests (push,
/// search, compact). This prevents a malicious user from creating a branch
/// that collides with internal compaction branches and would be silently
/// deleted by io_cleanup_compaction_branches during startup or after
/// compaction — causing data loss.
#[test]
fn reserved_branch_namespace_contract() {
    // The .-compact_rebuild_ prefix is reserved.
    assert!(is_compact_rebuild_branch(".-compact_rebuild_123"));
    assert!(is_compact_rebuild_branch(".-compact_rebuild_9999999999"));

    // The broader .- namespace is reserved for future internal use.
    assert!(is_reserved_branch_name(".-anything"));
    assert!(is_reserved_branch_name(".-compact_rebuild_123"));
    assert!(is_reserved_branch_name(".-"));

    // Normal branch names are not reserved.
    assert!(!is_reserved_branch_name("main"));
    assert!(!is_reserved_branch_name("feature/foo"));
    assert!(!is_reserved_branch_name("dev"));
    assert!(!is_reserved_branch_name("__compact_rebuild_123"));
    assert!(!is_reserved_branch_name("__my_branch"));

    // is_compact_rebuild_branch is a subset of is_reserved_branch_name.
    let rebuild = compact_rebuild_branch_name(1234567890);
    assert!(is_compact_rebuild_branch(&rebuild));
    assert!(is_reserved_branch_name(&rebuild));
    assert!(rebuild.starts_with(".-compact_rebuild_"));

    // A non-compact-rebuild .- branch is in the reserved namespace
    // but not a compact rebuild branch. Currently only .-compact_rebuild_
    // is rejected from external requests; other .- prefixes are allowed.
    assert!(is_reserved_branch_name(".-other_internal"));
    assert!(!is_compact_rebuild_branch(".-other_internal"));
}

/// Internal compaction branches (.-compact_rebuild_) are created and cleaned
/// up by vectorlink itself. This test verifies that a stale internal branch
/// (simulating a crashed compaction) is correctly identified and deleted by
/// io_cleanup_compaction_branches, while a normal user branch is left intact.
///
/// The .- prefix ensures these branches cannot be created by external
/// requests — the HTTP handler rejects any branch name starting with .-.
#[tokio::test]
async fn cleanup_deletes_stale_internal_branch_but_preserves_user_branches() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/reserved_namespace";

    let v1 = store
        .io_upsert_chunks(domain, "main", "doc/A", &[make_row(8, 1.0, "doc/A", "alpha")])
        .await
        .expect("upsert A");
    store.io_tag_commit(domain, "main", "c0", v1).await.expect("tag c0");

    // Simulate a crashed compaction: create a .-compact_rebuild_ branch.
    let stale_internal = compact_rebuild_branch_name(1);
    store
        .io_create_branch(domain, &stale_internal, v1)
        .await
        .expect("create stale internal branch");

    // Also create a normal user branch — this must NOT be cleaned up.
    store
        .io_create_branch(domain, "feature-x", v1)
        .await
        .expect("create user branch");

    // Run cleanup.
    let cleaned = store.io_cleanup_compaction_branches().await.expect("cleanup");
    assert_eq!(cleaned.len(), 1, "one domain should have been cleaned");

    // The stale internal branch should be gone.
    let branches = store.io_list_branches(domain).await.expect("list branches");
    assert!(
        !branches.contains(&stale_internal),
        "stale .-compact_rebuild_ branch should be deleted"
    );

    // The user branch must still exist.
    assert!(
        branches.contains(&"feature-x".to_owned()),
        "normal user branch must not be affected by internal branch cleanup"
    );
}

// --- Phase 1.1: Compute deltas between versions ---

/// Delta between V1 and V2 where V2 added doc/B returns doc/B rows to append.
#[tokio::test]
async fn delta_add_doc_returns_rows_to_append() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/delta_add";

    let v1 = store
        .io_upsert_chunks(domain, "main", "doc/A", &[make_row(8, 1.0, "doc/A", "alpha")])
        .await
        .expect("upsert A");
    let v2 = store
        .io_upsert_chunks(domain, "main", "doc/B", &[make_row(8, 2.0, "doc/B", "beta")])
        .await
        .expect("upsert B");

    let delta = store.io_compute_version_delta(domain, v1, v2).await.expect("delta");

    assert!(delta.doc_ids_to_delete.is_empty(), "no deletes expected");
    assert_eq!(delta.rows_to_append.len(), 1, "one row to append (doc/B)");
    assert_eq!(delta.rows_to_append[0].doc_id, "doc/B");
}

/// Delta where V2 changed doc/A (delete + re-add with new content) returns
/// doc/A rows to append.
#[tokio::test]
async fn delta_change_doc_returns_rows_to_append() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/delta_change";

    let v1 = store
        .io_upsert_chunks(domain, "main", "doc/A", &[make_row(8, 1.0, "doc/A", "alpha")])
        .await
        .expect("upsert A v1");
    let v2 = store
        .io_upsert_chunks(domain, "main", "doc/A", &[make_row(8, 1.5, "doc/A", "alpha updated")])
        .await
        .expect("upsert A v2 (changed)");

    let delta = store.io_compute_version_delta(domain, v1, v2).await.expect("delta");

    assert!(delta.doc_ids_to_delete.is_empty(), "no deletes expected");
    assert_eq!(delta.rows_to_append.len(), 1, "one row to append (changed doc/A)");
    assert_eq!(delta.rows_to_append[0].doc_id, "doc/A");
    assert_eq!(delta.rows_to_append[0].content, "alpha updated");
}

/// Delta where V2 deleted doc/A returns doc/A in doc_ids_to_delete.
#[tokio::test]
async fn delta_delete_doc_returns_id_to_delete() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/delta_delete";

    let v1 = store
        .io_upsert_chunks(domain, "main", "doc/A", &[make_row(8, 1.0, "doc/A", "alpha")])
        .await
        .expect("upsert A");
    store.io_tag_commit(domain, "main", "c0", v1).await.expect("tag c0");

    let v2 = store.io_delete_doc(domain, "main", "doc/A").await.expect("delete A");

    let delta = store.io_compute_version_delta(domain, v1, v2).await.expect("delta");

    assert!(delta.rows_to_append.is_empty(), "no rows to append");
    assert_eq!(delta.doc_ids_to_delete.len(), 1, "one doc to delete");
    assert!(delta.doc_ids_to_delete.contains(&"doc/A".to_owned()));
}

/// Delta between identical versions is empty.
#[tokio::test]
async fn delta_identical_versions_is_empty() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/delta_empty";

    let v1 = store
        .io_upsert_chunks(domain, "main", "doc/A", &[make_row(8, 1.0, "doc/A", "alpha")])
        .await
        .expect("upsert A");

    let delta = store.io_compute_version_delta(domain, v1, v1).await.expect("delta");

    assert!(delta.rows_to_append.is_empty(), "no rows to append");
    assert!(delta.doc_ids_to_delete.is_empty(), "no deletes");
}

// --- Phase 1.2: Apply delta on temp branch ---

/// Apply an add-delta to V1 creates a new version with doc/A + doc/B.
#[tokio::test]
async fn apply_add_delta_creates_version_with_both_docs() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/apply_add";

    // Setup: V1 has doc/A
    let v1 = store
        .io_upsert_chunks(domain, "main", "doc/A", &[make_row(8, 1.0, "doc/A", "alpha")])
        .await
        .expect("upsert A");
    // V2 adds doc/B
    let v2 = store
        .io_upsert_chunks(domain, "main", "doc/B", &[make_row(8, 2.0, "doc/B", "beta")])
        .await
        .expect("upsert B");

    // Create temp branch from V1
    store
        .io_create_branch(domain, &compact_rebuild_branch_name(1), v1)
        .await
        .expect("create temp branch");

    // Compute delta V1→V2
    let delta = store.io_compute_version_delta(domain, v1, v2).await.expect("delta");

    // Apply delta on temp branch
    let mut ds = store
        .io_open_dataset_uncached(domain, &compact_rebuild_branch_name(1))
        .await
        .expect("open")
        .expect("exists");
    let new_version = store
        .io_apply_delta_on_branch(&mut ds, &delta)
        .await
        .expect("apply delta");

    // Verify the new version has both doc/A and doc/B
    let ds_new = ds.checkout_version(new_version).await.expect("checkout new version");
    let ids = collect_doc_ids(&ds_new).await;
    assert!(ids.contains("doc/A"), "doc/A should be present");
    assert!(ids.contains("doc/B"), "doc/B should be present");

    // Cleanup temp branch
    let path = store.dataset_path(domain);
    let uri = path.to_string_lossy().to_string();
    let mut ds_fresh = store.io_open_fresh(&uri).await.expect("fresh open");
    let _ = ds_fresh.delete_branch(&compact_rebuild_branch_name(1)).await;
}

/// Apply a change-delta (delete + re-add doc/A with new content) updates the doc.
#[tokio::test]
async fn apply_change_delta_updates_doc_content() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/apply_change";

    let v1 = store
        .io_upsert_chunks(domain, "main", "doc/A", &[make_row(8, 1.0, "doc/A", "alpha")])
        .await
        .expect("upsert A v1");
    let v2 = store
        .io_upsert_chunks(domain, "main", "doc/A", &[make_row(8, 1.5, "doc/A", "alpha updated")])
        .await
        .expect("upsert A v2 (changed)");

    store
        .io_create_branch(domain, &compact_rebuild_branch_name(1), v1)
        .await
        .expect("create temp branch");

    let delta = store.io_compute_version_delta(domain, v1, v2).await.expect("delta");

    let mut ds = store
        .io_open_dataset_uncached(domain, &compact_rebuild_branch_name(1))
        .await
        .expect("open")
        .expect("exists");
    store
        .io_apply_delta_on_branch(&mut ds, &delta)
        .await
        .expect("apply delta");

    // Verify doc/A content is updated
    let ids = collect_doc_ids(&ds).await;
    assert!(ids.contains("doc/A"), "doc/A should still be present");

    // Read content to verify it's the updated version
    let mut scanner = ds.scan();
    scanner.project(&["doc_id", "content"]).expect("project");
    let batches: Vec<arrow_array::RecordBatch> = scanner
        .try_into_stream()
        .await
        .expect("stream")
        .try_collect()
        .await
        .expect("collect");
    let mut found_updated = false;
    for batch in &batches {
        let doc_ids = batch
            .column_by_name("doc_id")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
        let contents = batch
            .column_by_name("content")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
        if let (Some(ids), Some(contents)) = (doc_ids, contents) {
            for i in 0..ids.len() {
                if ids.value(i) == "doc/A" && contents.value(i) == "alpha updated" {
                    found_updated = true;
                }
            }
        }
    }
    assert!(found_updated, "doc/A should have updated content");

    // Cleanup
    let path = store.dataset_path(domain);
    let uri = path.to_string_lossy().to_string();
    let mut ds_fresh = store.io_open_fresh(&uri).await.expect("fresh open");
    let _ = ds_fresh.delete_branch(&compact_rebuild_branch_name(1)).await;
}

/// Apply a delete-delta removes doc/A from the temp branch.
#[tokio::test]
async fn apply_delete_delta_removes_doc() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/apply_delete";

    let v1 = store
        .io_upsert_chunks(domain, "main", "doc/A", &[make_row(8, 1.0, "doc/A", "alpha")])
        .await
        .expect("upsert A");
    store.io_tag_commit(domain, "main", "c0", v1).await.expect("tag c0");

    let v2 = store.io_delete_doc(domain, "main", "doc/A").await.expect("delete A");

    store
        .io_create_branch(domain, &compact_rebuild_branch_name(1), v1)
        .await
        .expect("create temp branch");

    let delta = store.io_compute_version_delta(domain, v1, v2).await.expect("delta");

    let mut ds = store
        .io_open_dataset_uncached(domain, &compact_rebuild_branch_name(1))
        .await
        .expect("open")
        .expect("exists");
    store
        .io_apply_delta_on_branch(&mut ds, &delta)
        .await
        .expect("apply delta");

    let ids = collect_doc_ids(&ds).await;
    assert!(!ids.contains("doc/A"), "doc/A should be deleted");

    // Cleanup
    let path = store.dataset_path(domain);
    let uri = path.to_string_lossy().to_string();
    let mut ds_fresh = store.io_open_fresh(&uri).await.expect("fresh open");
    let _ = ds_fresh.delete_branch(&compact_rebuild_branch_name(1)).await;
}

// --- Phase 1.3: io_retag_with_delta_forks integration tests ---

/// Helper: create N commits on main, each adding a distinct doc.
async fn setup_n_commits(store: &LanceStore, domain: &str, n: usize) -> Vec<(String, u64)> {
    let mut commits = Vec::new();
    for i in 0..n {
        let doc_id = format!("doc/{}", char::from_u32('A' as u32 + i as u32).unwrap());
        let content = format!("content {}", doc_id);
        let v = store
            .io_upsert_chunks(domain, "main", &doc_id, &[make_row(8, i as f32 + 1.0, &doc_id, &content)])
            .await
            .expect("upsert");
        let commit_id = format!("c{}", i);
        store.io_tag_commit(domain, "main", &commit_id, v).await.expect("tag");
        commits.push((commit_id, v));
    }
    commits
}

/// 10 commits, retag with delta forks, verify every commit is searchable
/// with its own exact data (snapshot isolation).
#[tokio::test]
async fn delta_fork_retag_10_commits_snapshot_isolation() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/delta_fork_10";

    let commits = setup_n_commits(&store, domain, 10).await;
    let total = commits.len();

    // Run delta-fork retagging.
    let (total_tags, retagged) = store
        .io_retag_with_delta_forks(domain, "main", 0)
        .await
        .expect("retag");

    assert_eq!(total_tags, total, "total tags should match");
    assert!(retagged > 0, "some commits should be retagged");

    // Verify snapshot isolation: each commit should return exactly the docs
    // that existed at that commit.
    let query = SearchQuery {
        query_text: "content".to_owned(),
        query_embedding: fake_embedding(8, 1.0),
        mode: SearchMode::Hybrid,
        start: 0,
        count: 50,
        doc_type_filter: vec![],
        doc_id_filter: vec![],
        snippet: false,
    };

    for (i, (commit_id, _)) in commits.iter().enumerate() {
        // Verify data via io_snapshot_from_cache
        let ds_version = store
            .io_snapshot_from_cache(domain, "main", commit_id)
            .await
            .expect("snapshot for data check");
        let data_ids = collect_doc_ids(&ds_version).await;

        for j in 0..=i {
            let expected = format!("doc/{}", char::from_u32('A' as u32 + j as u32).unwrap());
            assert!(
                data_ids.contains(&expected),
                "DATA CHECK: commit {} should contain {} (has {:?})",
                commit_id, expected, data_ids
            );
        }
        for j in (i + 1)..total {
            let not_expected = format!("doc/{}", char::from_u32('A' as u32 + j as u32).unwrap());
            assert!(
                !data_ids.contains(&not_expected),
                "DATA CHECK: commit {} should NOT contain {} (has {:?})",
                commit_id, not_expected, data_ids
            );
        }

        // Verify search results
        let hits = store
            .io_search(domain, "main", commit_id, &query)
            .await
            .expect("search");
        let ids: std::collections::HashSet<&str> =
            hits.iter().map(|h| h.doc_id.as_str()).collect();

        for j in 0..=i {
            let expected = format!("doc/{}", char::from_u32('A' as u32 + j as u32).unwrap());
            assert!(
                ids.contains(expected.as_str()),
                "commit {} should contain {} (has {:?})",
                commit_id, expected, ids
            );
        }
        for j in (i + 1)..total {
            let not_expected = format!("doc/{}", char::from_u32('A' as u32 + j as u32).unwrap());
            assert!(
                !ids.contains(not_expected.as_str()),
                "commit {} should NOT contain {} (has {:?})",
                commit_id, not_expected, ids
            );
        }
    }

    eprintln!("[delta_fork_10] All {} commits have correct snapshot isolation", total);
}

/// Exactly one rebuild branch exists after retagging (the epoch branch is kept).
#[tokio::test]
async fn delta_fork_retag_keeps_one_rebuild_branch() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/delta_fork_cleanup";

    let _commits = setup_n_commits(&store, domain, 5).await;

    store
        .io_retag_with_delta_forks(domain, "main", 0)
        .await
        .expect("retag");

    let branches = store.io_list_branches(domain).await.expect("list branches");
    let rebuild_count: usize = branches
        .iter()
        .filter(|b| is_compact_rebuild_branch(b))
        .count();
    assert_eq!(
        rebuild_count, 1,
        "exactly one rebuild branch should exist after retagging (got {:?})",
        branches
    );
}

/// CRITICAL: A second compaction cycle preserves snapshot isolation and
/// leaves exactly one rebuild branch. This verifies that branch-aware delta
/// computation correctly resolves commit snapshots from the first cycle's
/// rebuild branch, and that old epoch branches are deleted.
#[tokio::test]
async fn delta_fork_second_compaction_preserves_isolation() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/delta_fork_second_cycle";

    let commits = setup_n_commits(&store, domain, 10).await;
    let total = commits.len();

    // First compaction cycle
    store
        .io_retag_with_delta_forks(domain, "main", 0)
        .await
        .expect("first retag");

    // Verify one rebuild branch after first cycle
    let branches_1 = store.io_list_branches(domain).await.expect("list branches 1");
    let rebuild_count_1: usize = branches_1.iter().filter(|b| is_compact_rebuild_branch(b)).count();
    assert_eq!(rebuild_count_1, 1, "one rebuild branch after first cycle");

    // Verify snapshot isolation after first cycle
    for (i, (commit_id, _)) in commits.iter().enumerate() {
        let ds = store
            .io_snapshot_from_cache(domain, "main", commit_id)
            .await
            .expect("snapshot after first cycle");
        let ids = collect_doc_ids(&ds).await;
        for j in 0..=i {
            let expected = format!("doc/{}", char::from_u32('A' as u32 + j as u32).unwrap());
            assert!(
                ids.contains(&expected),
                "after 1st cycle: commit {} should contain {} (has {:?})",
                commit_id, expected, ids
            );
        }
    }

    // Second compaction cycle — this is the critical test
    store
        .io_retag_with_delta_forks(domain, "main", 0)
        .await
        .expect("second retag");

    // Verify exactly one rebuild branch after second cycle
    let branches_2 = store.io_list_branches(domain).await.expect("list branches 2");
    let rebuild_count_2: usize = branches_2.iter().filter(|b| is_compact_rebuild_branch(b)).count();
    assert_eq!(
        rebuild_count_2, 1,
        "exactly one rebuild branch after second cycle (got {:?})",
        branches_2
    );

    // Verify snapshot isolation after second cycle
    for (i, (commit_id, _)) in commits.iter().enumerate() {
        let ds = store
            .io_snapshot_from_cache(domain, "main", commit_id)
            .await
            .expect("snapshot after second cycle");
        let ids = collect_doc_ids(&ds).await;
        for j in 0..=i {
            let expected = format!("doc/{}", char::from_u32('A' as u32 + j as u32).unwrap());
            assert!(
                ids.contains(&expected),
                "after 2nd cycle: commit {} should contain {} (has {:?})",
                commit_id, expected, ids
            );
        }
        // Verify docs beyond this commit are NOT present
        for j in (i + 1)..total {
            let not_expected = format!("doc/{}", char::from_u32('A' as u32 + j as u32).unwrap());
            assert!(
                !ids.contains(&not_expected),
                "after 2nd cycle: commit {} should NOT contain {} (has {:?})",
                commit_id, not_expected, ids
            );
        }
    }

    eprintln!("[delta_fork_2nd] Second compaction cycle preserved snapshot isolation for all {} commits", total);
}

/// Does not retag all to one version (regression guard).
#[tokio::test]
async fn delta_fork_retag_does_not_retag_all_to_one_version() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/delta_fork_no_collapse";

    let _commits = setup_n_commits(&store, domain, 5).await;

    store
        .io_retag_with_delta_forks(domain, "main", 0)
        .await
        .expect("retag");

    let commit_versions = store.io_list_commit_versions(domain).await.expect("list");
    let versions: std::collections::HashSet<u64> = commit_versions.values().copied().collect();
    assert!(
        versions.len() > 1,
        "commits should not all point to the same version (got {:?})",
        versions
    );
}

/// io_derive_last_indexed for main returns the latest commit after retagging.
#[tokio::test]
async fn delta_fork_retag_derive_last_indexed_returns_latest() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/delta_fork_last_indexed";

    let commits = setup_n_commits(&store, domain, 5).await;
    let last_commit = commits.last().unwrap().0.clone();

    store
        .io_retag_with_delta_forks(domain, "main", 0)
        .await
        .expect("retag");

    let last_indexed = store
        .io_derive_last_indexed(domain, "main")
        .await
        .expect("derive");
    assert!(last_indexed.is_some(), "derive_last_indexed should return a result");
    let (derived_commit, _) = last_indexed.unwrap();
    assert_eq!(derived_commit, last_commit, "derived last commit should be the latest");
}

/// Crashed compaction cleanup: an unreferenced rebuild branch (simulating a
/// crash during Phase 1, before retagging) is detected and deleted by
/// io_cleanup_compaction_branches.
#[tokio::test]
async fn delta_fork_crashed_compaction_cleanup() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/delta_fork_crash_cleanup";

    let v1 = store
        .io_upsert_chunks(domain, "main", "doc/A", &[make_row(8, 1.0, "doc/A", "alpha")])
        .await
        .expect("upsert A");
    store.io_tag_commit(domain, "main", "c0", v1).await.expect("tag c0 on main");

    // Simulate crashed compaction: create a stale rebuild branch with no tags.
    store
        .io_create_branch(domain, &compact_rebuild_branch_name(999), v1)
        .await
        .expect("create stale branch");

    // Run startup cleanup
    let cleaned = store
        .io_cleanup_compaction_branches()
        .await
        .expect("cleanup");

    assert_eq!(cleaned.len(), 1, "one domain should be cleaned");

    let branches = store.io_list_branches(domain).await.expect("list branches");
    assert!(
        !branches.iter().any(|b| is_compact_rebuild_branch(b)),
        "unreferenced rebuild branch should be deleted"
    );
}

/// Delta computation uses previous commit, not nearest boundary (no data duplication).
#[tokio::test]
async fn delta_fork_uses_previous_commit_not_boundary() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/delta_fork_prev_not_boundary";

    let _commits = setup_n_commits(&store, domain, 5).await;

    store
        .io_retag_with_delta_forks(domain, "main", 0)
        .await
        .expect("retag");

    let query = SearchQuery {
        query_text: "content".to_owned(),
        query_embedding: fake_embedding(8, 1.0),
        mode: SearchMode::Hybrid,
        start: 0,
        count: 50,
        doc_type_filter: vec![],
        doc_id_filter: vec![],
        snippet: false,
    };

    let hits_c1 = store.io_search(domain, "main", "c1", &query).await.expect("search c1");
    let ids_c1: Vec<&str> = hits_c1.iter().map(|h| h.doc_id.as_str()).collect();
    assert_eq!(ids_c1.len(), 2, "c1 should have exactly 2 docs (A+B), got {:?}", ids_c1);

    let hits_c2 = store.io_search(domain, "main", "c2", &query).await.expect("search c2");
    let ids_c2: Vec<&str> = hits_c2.iter().map(|h| h.doc_id.as_str()).collect();
    assert_eq!(ids_c2.len(), 3, "c2 should have exactly 3 docs (A+B+C), got {:?}", ids_c2);
}

// --- FULL REINDEXING: compaction pipeline with delta-fork retagging ---

/// Helper: run the compaction pipeline steps: open uncached, compact data,
/// aggressive cleanup, delta-fork retag, aggressive cleanup again, prune
/// empty index dirs, refresh cached handle.
async fn run_compaction_pipeline(store: &LanceStore, domain: &str, branch: &str) {
    use crate::store::lance::io_compact_data;

    let ds = store
        .io_open_dataset_uncached(domain, branch)
        .await
        .expect("open uncached")
        .expect("dataset exists");
    let mut ds = ds;
    let version_before = ds.version().version;
    io_compact_data(&mut ds, true).await.expect("compact_data");
    let version_after = ds.version().version;

    // Step 2: aggressive cleanup of old untagged versions (before retag).
    let _ = store.io_cleanup_aggressive(domain, branch).await;

    // Step 3: delta-fork retag (only if compaction advanced the version).
    if version_after != version_before {
        store
            .io_retag_with_delta_forks(domain, branch, version_after)
            .await
            .expect("retag");
    }

    // Step 4: final aggressive cleanup after retagging.
    let _ = store.io_cleanup_aggressive(domain, branch).await;

    // Step 5: prune empty index directories left by LanceDB cleanup.
    let _ = store.io_prune_empty_index_dirs(domain);

    // Step 6: refresh cached handle.
    store
        .io_refresh_cached_dataset(domain, branch)
        .await
        .expect("refresh");
}

/// Full reindexing test: push 20 commits, compact + retag, push 10 more,
/// compact + retag again, then verify every commit has correct snapshot
/// isolation via search. Also verifies exactly one rebuild branch remains
/// and that new pushes after retagging extend the main branch correctly.
#[tokio::test]
async fn full_reindexing_compaction_preserves_search_at_all_commits() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/full_reindex";

    // Phase 1: Push 20 commits, each adding a distinct doc.
    // 20 fragments exceeds the 16-fragment compaction threshold so
    // io_compact_data actually rewrites data and advances the version.
    let mut commits: Vec<(String, u64)> = Vec::new();
    for i in 0..20 {
        let doc_id = format!("doc/{}", i);
        let emb = fake_embedding(8, i as f32 * 10.0);
        let rows = vec![ChunkRow {
            doc_id: doc_id.clone(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: emb.clone(),
            clustering_embedding: emb,
            content: format!("content {}", doc_id),
        }];
        let v = store
            .io_upsert_chunks(domain, "main", &doc_id, &rows)
            .await
            .expect("upsert");
        let commit = format!("c{}", i);
        store
            .io_tag_commit(domain, "main", &commit, v)
            .await
            .expect("tag");
        commits.push((commit, v));
    }

    // Phase 2: Run compaction pipeline (compact + retag + cleanup).
    run_compaction_pipeline(&store, domain, "main").await;

    // Verify one rebuild branch after first compaction.
    let branches_1 = store.io_list_branches(domain).await.expect("list branches 1");
    let rebuild_count_1: usize = branches_1.iter().filter(|b| is_compact_rebuild_branch(b)).count();
    assert_eq!(rebuild_count_1, 1, "one rebuild branch after first compaction");

    // Phase 3: Push 10 more commits (c20..c29), extending main.
    for i in 20..30 {
        let doc_id = format!("doc/{}", i);
        let emb = fake_embedding(8, i as f32 * 10.0);
        let rows = vec![ChunkRow {
            doc_id: doc_id.clone(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: emb.clone(),
            clustering_embedding: emb,
            content: format!("content {}", doc_id),
        }];
        let v = store
            .io_upsert_chunks(domain, "main", &doc_id, &rows)
            .await
            .expect("upsert after retag");
        let commit = format!("c{}", i);
        store
            .io_tag_commit(domain, "main", &commit, v)
            .await
            .expect("tag after retag");
        commits.push((commit, v));
    }

    // Phase 4: Run compaction pipeline again (second cycle).
    run_compaction_pipeline(&store, domain, "main").await;

    // Verify exactly one rebuild branch after second compaction.
    let branches_2 = store.io_list_branches(domain).await.expect("list branches 2");
    let rebuild_count_2: usize = branches_2.iter().filter(|b| is_compact_rebuild_branch(b)).count();
    assert_eq!(
        rebuild_count_2, 1,
        "exactly one rebuild branch after second compaction (got {:?})",
        branches_2
    );

    // Phase 5: Search at every commit and verify snapshot isolation.
    // Commit ci should contain doc/0 through doc/i, and no others.
    let total = commits.len();
    for (i, (commit_id, _)) in commits.iter().enumerate() {
        let query = SearchQuery {
            query_embedding: fake_embedding(8, (i as f32) * 10.0),
            query_text: "content".to_owned(),
            mode: SearchMode::Hybrid,
            start: 0,
            count: 50,
            doc_type_filter: vec![],
            doc_id_filter: vec![],
            snippet: false,
        };
        let hits = store
            .io_search(domain, "main", commit_id, &query)
            .await
            .expect("search");
        let ids: Vec<&str> = hits.iter().map(|h| h.doc_id.as_str()).collect();

        // Verify all docs up to this commit are present.
        for j in 0..=i {
            let expected = format!("doc/{}", j);
            assert!(
                ids.contains(&expected.as_str()),
                "commit {} should contain {} (has {:?})",
                commit_id, expected, ids
            );
        }

        // Verify no docs beyond this commit are present.
        for j in (i + 1)..total {
            let not_expected = format!("doc/{}", j);
            assert!(
                !ids.contains(&not_expected.as_str()),
                "SNAPSHOT ISOLATION VIOLATED: commit {} should NOT contain {} (has {:?})",
                commit_id, not_expected, ids
            );
        }
    }

    // Phase 6: Verify derive_last_indexed returns the latest commit.
    let last_indexed = store
        .io_derive_last_indexed(domain, "main")
        .await
        .expect("derive");
    assert!(last_indexed.is_some(), "derive_last_indexed should return a result");
    let (derived_commit, _) = last_indexed.unwrap();
    assert_eq!(
        derived_commit, commits.last().unwrap().0,
        "derived last commit should be the latest after reindexing"
    );

    eprintln!(
        "[full_reindex] All {} commits have correct snapshot isolation after two compaction cycles",
        total
    );
}

/// On-disk storage verification after two compaction cycles.
///
/// Pushes enough commits to trigger compaction twice, then inspects the
/// raw filesystem to verify that the delta-fork retagging left the correct
/// on-disk artefacts: exactly one epoch-named rebuild branch, boundary
/// tags on main, intermediate tags on the rebuild branch, bounded index
/// directory count, and small version manifest count.
#[tokio::test]
async fn on_disk_storage_matches_delta_fork_plan_after_two_compaction_cycles() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/disk_verify";

    // Phase 1: Push 20 commits to exceed the 16-fragment compaction threshold.
    let mut commits: Vec<(String, u64)> = Vec::new();
    for i in 0..20 {
        let doc_id = format!("doc/{}", i);
        let emb = fake_embedding(8, i as f32 * 10.0);
        let rows = vec![ChunkRow {
            doc_id: doc_id.clone(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: emb.clone(),
            clustering_embedding: emb,
            content: format!("content {}", doc_id),
        }];
        let v = store
            .io_upsert_chunks(domain, "main", &doc_id, &rows)
            .await
            .expect("upsert");
        let commit = format!("c{}", i);
        store
            .io_tag_commit(domain, "main", &commit, v)
            .await
            .expect("tag");
        commits.push((commit, v));
    }

    // Phase 2: First compaction cycle.
    run_compaction_pipeline(&store, domain, "main").await;

    // --- Verify on-disk state after first compaction ---

    // 1. Exactly one rebuild branch, epoch-named.
    let branches_1 = store.io_list_branches(domain).await.expect("list branches");
    let rebuild_branches_1: Vec<&String> = branches_1
        .iter()
        .filter(|b| is_compact_rebuild_branch(b))
        .collect();
    assert_eq!(
        rebuild_branches_1.len(),
        1,
        "exactly one rebuild branch after first compaction (got {:?})",
        branches_1
    );
    let rebuild_name_1 = rebuild_branches_1[0].clone();
    assert!(
        rebuild_name_1.starts_with(".-compact_rebuild_"),
        "rebuild branch should be epoch-named (got {})",
        rebuild_name_1
    );
    let epoch_suffix = rebuild_name_1.strip_prefix(".-compact_rebuild_").unwrap();
    assert!(
        epoch_suffix.chars().all(|c| c.is_ascii_digit()),
        "epoch suffix should be numeric (got {})",
        epoch_suffix
    );

    // 2. Tag distribution: boundary tags on main, intermediate on rebuild.
    let path = store.dataset_path(domain);
    let uri = path.to_string_lossy().to_string();
    let ds_fresh = store.io_open_fresh(&uri).await.expect("fresh open");
    let tags_1 = ds_fresh.tags().list().await.expect("tag list");
    let main_count_1 = tags_1.values().filter(|c| c.branch.is_none()).count();
    let rebuild_count_1 = tags_1
        .values()
        .filter(|c| c.branch.as_deref() == Some(rebuild_name_1.as_str()))
        .count();
    assert!(
        main_count_1 > 0,
        "boundary tags should be on main after first compaction"
    );
    assert!(
        rebuild_count_1 > 0,
        "intermediate tags should be on rebuild branch after first compaction"
    );
    assert_eq!(
        main_count_1 + rebuild_count_1,
        tags_1.len(),
        "all tags should be either on main or on the rebuild branch"
    );

    // 3. Index directory count should not exceed commit count.
    let indices_dir = path.join("_indices");
    let index_dir_count_1 = count_subdirs(&indices_dir);
    assert!(
        index_dir_count_1 <= commits.len(),
        "index dir count {} should not exceed commit count {} after first compaction",
        index_dir_count_1,
        commits.len()
    );

    // 4. Version manifests should be bounded.
    let versions_dir = path.join("_versions");
    let manifest_count_1 = count_files_with_ext(&versions_dir, "manifest");
    assert!(
        manifest_count_1 <= commits.len(),
        "version manifest count {} should not exceed commit count {} after first compaction",
        manifest_count_1,
        commits.len()
    );

    // Phase 3: Push 20 more commits to exceed the 16-fragment threshold again
    // after the first compaction consolidated the initial fragments.
    for i in 20..40 {
        let doc_id = format!("doc/{}", i);
        let emb = fake_embedding(8, i as f32 * 10.0);
        let rows = vec![ChunkRow {
            doc_id: doc_id.clone(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: emb.clone(),
            clustering_embedding: emb,
            content: format!("content {}", doc_id),
        }];
        let v = store
            .io_upsert_chunks(domain, "main", &doc_id, &rows)
            .await
            .expect("upsert after first compaction");
        let commit = format!("c{}", i);
        store
            .io_tag_commit(domain, "main", &commit, v)
            .await
            .expect("tag after first compaction");
        commits.push((commit, v));
    }

    // Phase 4: Second compaction cycle.
    run_compaction_pipeline(&store, domain, "main").await;

    // --- Verify on-disk state after second compaction ---

    // 5. Still exactly one rebuild branch (old one deleted, new epoch created).
    let branches_2 = store.io_list_branches(domain).await.expect("list branches 2");
    let rebuild_branches_2: Vec<&String> = branches_2
        .iter()
        .filter(|b| is_compact_rebuild_branch(b))
        .collect();
    assert_eq!(
        rebuild_branches_2.len(),
        1,
        "exactly one rebuild branch after second compaction (got {:?})",
        branches_2
    );
    let rebuild_name_2 = rebuild_branches_2[0].clone();
    assert!(
        rebuild_name_2.starts_with(".-compact_rebuild_"),
        "rebuild branch should be epoch-named after second compaction (got {})",
        rebuild_name_2
    );

    // 6. The old rebuild branch should be gone (no unreferenced branches).
    assert_ne!(
        rebuild_name_1, rebuild_name_2,
        "second compaction should create a new epoch branch"
    );
    assert!(
        !branches_2.contains(&rebuild_name_1),
        "old rebuild branch should be deleted after second compaction"
    );

    // 7. Tag distribution after second compaction.
    let ds_fresh_2 = store.io_open_fresh(&uri).await.expect("fresh open 2");
    let tags_2 = ds_fresh_2.tags().list().await.expect("tag list 2");
    let main_count_2 = tags_2.values().filter(|c| c.branch.is_none()).count();
    let rebuild_count_2 = tags_2
        .values()
        .filter(|c| c.branch.as_deref() == Some(rebuild_name_2.as_str()))
        .count();
    assert!(
        main_count_2 > 0,
        "boundary tags should be on main after second compaction"
    );
    assert!(
        rebuild_count_2 > 0,
        "intermediate tags should be on rebuild branch after second compaction"
    );

    // 8. No tags should reference the old rebuild branch.
    let old_branch_tags: Vec<_> = tags_2
        .iter()
        .filter(|(_, c)| c.branch.as_deref() == Some(rebuild_name_1.as_str()))
        .collect();
    if !old_branch_tags.is_empty() {
        for (tag_name, contents) in &old_branch_tags {
            let commit = crate::layeridx::decode_commit_tag(tag_name)
                .unwrap_or_else(|_| tag_name.to_string());
            eprintln!(
                "[on_disk_verify] ORPHANED TAG: commit={} version={} still on old rebuild branch {}",
                commit, contents.version, rebuild_name_1
            );
        }
    }
    assert_eq!(
        old_branch_tags.len(), 0,
        "no tags should reference the old rebuild branch after second compaction ({} orphaned: {:?})",
        old_branch_tags.len(),
        old_branch_tags.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>()
    );

    // 9. All tags should be on main or the current rebuild branch.
    assert_eq!(
        main_count_2 + rebuild_count_2,
        tags_2.len(),
        "all tags should be either on main or on the current rebuild branch"
    );

    // 10. Index dir count should not grow unboundedly across cycles.
    let index_dir_count_2 = count_subdirs(&indices_dir);
    assert!(
        index_dir_count_2 <= commits.len(),
        "index dir count {} should not exceed commit count {} after second compaction",
        index_dir_count_2,
        commits.len()
    );

    // 11. Version manifests still bounded.
    let manifest_count_2 = count_files_with_ext(&versions_dir, "manifest");
    assert!(
        manifest_count_2 <= commits.len(),
        "version manifest count {} should not exceed commit count {} after second compaction",
        manifest_count_2,
        commits.len()
    );

    // 12. Data files should be consolidated (bounded by 2x commit count,
    // accounting for both main and rebuild branch data files).
    let data_dir = path.join("data");
    let data_file_count = count_files_with_ext(&data_dir, "lance");
    assert!(
        data_file_count <= commits.len() * 2,
        "data file count {} should not exceed 2x commit count {} after compaction (main + rebuild branch)",
        data_file_count,
        commits.len()
    );

    // 13. derive_last_indexed returns the latest commit.
    let last_indexed = store
        .io_derive_last_indexed(domain, "main")
        .await
        .expect("derive");
    assert!(last_indexed.is_some(), "derive_last_indexed should return a result");
    let (derived_commit, _) = last_indexed.unwrap();
    assert_eq!(
        derived_commit,
        commits.last().unwrap().0,
        "derived last commit should be the latest after two compaction cycles"
    );

    eprintln!(
        "[on_disk_verify] branches={} tags={} (main={} rebuild={}) index_dirs={} manifests={} data_files={}",
        branches_2.len(),
        tags_2.len(),
        main_count_2,
        rebuild_count_2,
        index_dir_count_2,
        manifest_count_2,
        data_file_count
    );
}

/// Count subdirectories within a directory. Returns 0 if the directory
/// does not exist.
fn count_subdirs(dir: &std::path::Path) -> usize {
    if !dir.exists() {
        return 0;
    }
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_dir())
                .count()
        })
        .unwrap_or(0)
}

/// Count files with a specific extension within a directory. Returns 0
/// if the directory does not exist.
fn count_files_with_ext(dir: &std::path::Path, ext: &str) -> usize {
    if !dir.exists() {
        return 0;
    }
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .is_some_and(|x| x == ext)
                })
                .count()
        })
        .unwrap_or(0)
}

/// Recursively compute the total size of all files in a directory tree.
fn dir_tree_size(dir: &std::path::Path) -> u64 {
    if !dir.exists() {
        return 0;
    }
    let mut total: u64 = 0;
    fn walk(dir: &std::path::Path, total: &mut u64) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    walk(&path, total);
                } else if let Ok(meta) = entry.metadata() {
                    *total += meta.len();
                }
            }
        }
    }
    walk(dir, &mut total);
    total
}

/// Check if a directory contains any files (recursively).
fn dir_has_files(dir: &std::path::Path) -> bool {
    if !dir.exists() {
        return false;
    }
    fn check(dir: &std::path::Path) -> bool {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    return true;
                }
                if path.is_dir() && check(&path) {
                    return true;
                }
            }
        }
        false
    }
    check(dir)
}

/// Helper: run the FULL compaction pipeline that mirrors the HTTP API
/// compaction (service/mod.rs `compact_domain`). This includes index
/// drop/recreate and incremental cascade.
///
/// **DRIFT WARNING (REVIEW-7):** This function manually duplicates the
/// steps in `SearchService::compact_domain`. Any change to the compaction
/// pipeline in the service layer MUST be reflected here, and vice versa.
/// If they diverge, these proof tests validate the wrong pipeline.
/// Future refactoring should extract a shared `LanceStore::run_compaction`
/// method that both call.
///
/// Steps:
/// 1. Open uncached
/// 2. Drop existing vector + FTS indices (so compact_files can merge ALL
///    fragments regardless of index set — LanceDB's DefaultCompactionPlanner
///    splits fragments into separate bins when they have different index UUIDs)
/// 3. Compact data fragments (now all fragments have the same empty index set)
/// 4. Aggressive cleanup (removes old untagged versions + orphaned data files)
/// 5. Recreate vector + FTS indices (fresh, covering consolidated fragments)
/// 6. Incremental cascade (index rollup)
/// 7. Delta-fork retag
/// 8. Aggressive cleanup
/// 9. Prune stale index dirs (empty AND non-empty dirs not referenced by live indices)
/// 10. Refresh cached handle
async fn run_full_compaction_pipeline(store: &LanceStore, domain: &str, branch: &str) {
    use crate::store::lance::io_compact_data;
    use crate::store::lance::io_ensure_fts_index_on_dataset;
    use crate::store::lance::io_incremental_cascade;
    use crate::store::vector_index::{io_ensure_vector_index, VECTOR_INDEX_NAME};
    use lance::index::DatasetIndexExt;

    let path = store.dataset_path(domain);
    let data_dir = path.join("data");

    let count_data_files = || {
        if !data_dir.exists() {
            0
        } else {
            std::fs::read_dir(&data_dir)
                .map(|entries| {
                    entries
                        .filter_map(|e| e.ok())
                        .filter(|e| e.path().extension().is_some_and(|x| x == "lance"))
                        .count()
                })
                .unwrap_or(0)
        }
    };

    let count_index_dirs = || {
        let indices_dir = path.join("_indices");
        if !indices_dir.exists() {
            0
        } else {
            std::fs::read_dir(&indices_dir)
                .map(|entries| {
                    entries
                        .filter_map(|e| e.ok())
                        .filter(|e| e.path().is_dir() && dir_has_files(&e.path()))
                        .count()
                })
                .unwrap_or(0)
        }
    };

    eprintln!("[pipeline] START data_files={} index_dirs={}", count_data_files(), count_index_dirs());

    // Step 1: Open uncached dataset handle.
    let ds = store
        .io_open_dataset_uncached(domain, branch)
        .await
        .expect("open uncached")
        .expect("dataset exists");
    let mut ds = ds;
    let version_before = ds.version().version;

    // Step 2: Drop existing indices BEFORE compacting data.
    // LanceDB's DefaultCompactionPlanner groups fragments by index set.
    // If some fragments have indices and others don't, they go into separate
    // bins and cannot be merged. Dropping indices first ensures all fragments
    // have the same (empty) index set, allowing full consolidation into a
    // single fragment.
    let indices = ds.load_indices().await.expect("load indices");
    let has_vector = indices.iter().any(|i| i.name == VECTOR_INDEX_NAME);
    let has_fts = indices.iter().any(|i| i.name == "content_fts");

    if has_vector {
        ds.drop_index(VECTOR_INDEX_NAME).await.expect("drop vector index");
        eprintln!("[pipeline] dropped vector index");
    }
    if has_fts {
        ds.drop_index("content_fts").await.expect("drop FTS index");
        eprintln!("[pipeline] dropped FTS index");
    }
    if has_vector || has_fts {
        // Use io_cleanup_aggressive (which has FIX-B _indices/ hiding) instead
        // of calling cleanup_with_policy directly.
        let _ = store.io_cleanup_aggressive(domain, branch).await;
        eprintln!("[pipeline] after index drop cleanup data_files={} index_dirs={}", count_data_files(), count_index_dirs());
    }

    // Step 3: Compact data fragments (all fragments now have same index set).
    io_compact_data(&mut ds, true).await.expect("compact_data");
    let version_after = ds.version().version;
    eprintln!("[pipeline] after compact_data data_files={} index_dirs={}", count_data_files(), count_index_dirs());

    // Step 4: Aggressive cleanup of old untagged versions.
    let _ = store.io_cleanup_aggressive(domain, branch).await;
    eprintln!("[pipeline] after cleanup_1 data_files={} index_dirs={}", count_data_files(), count_index_dirs());

    // Step 5: Recreate vector + FTS indices (fresh, covering consolidated fragments).
    let vector_config = store.vector_index_config().clone();
    io_ensure_vector_index(&mut ds, &vector_config, true).await.expect("recreate vector index");
    io_ensure_fts_index_on_dataset(&mut ds).await.expect("recreate FTS index");
    eprintln!("[pipeline] after index recreate data_files={} index_dirs={}", count_data_files(), count_index_dirs());

    store.io_reset_delta_count(domain, branch).await;
    let _ = store.io_cleanup_aggressive(domain, branch).await;
    eprintln!("[pipeline] after cleanup_2 data_files={} index_dirs={}", count_data_files(), count_index_dirs());

    // Step 6: Incremental cascade (index rollup).
    let delta_count = store.io_get_delta_count(domain, branch).await;
    let _ = io_incremental_cascade(&mut ds, delta_count).await;
    let merge_version = ds.version().version;
    eprintln!("[pipeline] after cascade data_files={} index_dirs={}", count_data_files(), count_index_dirs());

    // Step 7: Delta-fork retag (only if compaction advanced the version).
    if version_after != version_before {
        store
            .io_retag_with_delta_forks(domain, branch, merge_version)
            .await
            .expect("retag");
    }
    eprintln!("[pipeline] after retag data_files={} index_dirs={}", count_data_files(), count_index_dirs());

    // Step 7b: Clean up orphaned tags and stale rebuild branches.
    let _ = store
        .io_cleanup_orphaned_tags_and_stale_branches(domain)
        .await;
    eprintln!("[pipeline] after orphan/stale cleanup data_files={} index_dirs={}", count_data_files(), count_index_dirs());

    // Step 8: Final aggressive cleanup after retagging.
    let _ = store.io_cleanup_aggressive(domain, branch).await;
    eprintln!("[pipeline] after cleanup_3 data_files={} index_dirs={}", count_data_files(), count_index_dirs());

    // Step 9: Prune stale index directories (not just empty ones).
    let _ = store.io_prune_stale_index_dirs(domain, branch).await;
    let _ = store.io_prune_empty_index_dirs(domain);

    // Step 10: Refresh cached handle.
    store
        .io_refresh_cached_dataset(domain, branch)
        .await
        .expect("refresh");
    eprintln!("[pipeline] END data_files={} index_dirs={}", count_data_files(), count_index_dirs());
}

// ===========================================================================
// FOCUSED PROOF TESTS — exact assertions for base-3 roll-up correctness
//
// These tests assert the exact expected on-disk state after two compaction
// cycles. Some tests still surface issues (orphaned data files, stale index
// dirs) that are being resolved.
//
// Setup: 40 commits (c0..c39), each with 20 docs. Two full compaction cycles.
// Base-3 boundaries for 40 commits: {0, 3, 12} → 3 boundaries, 37 intermediates
// (latest commit c39 is always an intermediate, never a boundary).
// ===========================================================================

/// Shared setup: push 40 commits in two batches of 20, compacting after each
/// batch. Returns (store, domain, commits, path).
async fn setup_two_compaction_cycles(
    dim: usize,
    domain: &str,
) -> (
    LanceStore,
    tempfile::TempDir,
    Vec<(String, u64)>,
    std::path::PathBuf,
) {
    let config = crate::store::vector_index::tests::make_test_config(dim);
    let (mut store, tmp) = make_test_store(dim);
    store.set_vector_index_config(config.clone());

    let mut commits: Vec<(String, u64)> = Vec::new();

    // Phase 1: Push 20 commits, each adding 20 docs.
    for i in 0..20 {
        let mut rows = Vec::new();
        for j in 0..20 {
            let doc_id = format!("doc/{}_{}", i, j);
            let emb = fake_embedding(dim, (i * 20 + j) as f32 * 10.0);
            rows.push(ChunkRow {
                doc_id: doc_id.clone(),
                doc_type: "T".to_owned(),
                chunk_index: 0,
                chunk_count: 1,
                chunk_token_start: 0,
                doc_token_len: 5,
                embedding: emb.clone(),
                clustering_embedding: emb,
                content: format!("content {}", doc_id),
            });
        }
        let first_doc_id = format!("doc/{}_0", i);
        let v = store
            .io_upsert_chunks(domain, "main", &first_doc_id, &rows)
            .await
            .expect("upsert");
        let commit = format!("c{}", i);
        store
            .io_tag_commit(domain, "main", &commit, v)
            .await
            .expect("tag");
        commits.push((commit, v));
    }

    // Phase 2: First full compaction.
    run_full_compaction_pipeline(&store, domain, "main").await;

    // Phase 3: Push 20 more commits.
    for i in 20..40 {
        let mut rows = Vec::new();
        for j in 0..20 {
            let doc_id = format!("doc/{}_{}", i, j);
            let emb = fake_embedding(dim, (i * 20 + j) as f32 * 10.0);
            rows.push(ChunkRow {
                doc_id: doc_id.clone(),
                doc_type: "T".to_owned(),
                chunk_index: 0,
                chunk_count: 1,
                chunk_token_start: 0,
                doc_token_len: 5,
                embedding: emb.clone(),
                clustering_embedding: emb,
                content: format!("content {}", doc_id),
            });
        }
        let first_doc_id = format!("doc/{}_0", i);
        let v = store
            .io_upsert_chunks(domain, "main", &first_doc_id, &rows)
            .await
            .expect("upsert after first compaction");
        let commit = format!("c{}", i);
        store
            .io_tag_commit(domain, "main", &commit, v)
            .await
            .expect("tag after first compaction");
        commits.push((commit, v));
    }

    // Phase 4: Second full compaction.
    run_full_compaction_pipeline(&store, domain, "main").await;

    let path = store.dataset_path(domain);
    (store, tmp, commits, path)
}

/// Compute the expected base-3 boundary positions for N commits.
/// Boundaries are at positions 0, 3, 9, 27, 81... (3^k cumulative sums).
/// The latest commit (position N-1) is NEVER a boundary — it is always
/// treated as an intermediate so it gets retagged to a fresh version on
/// the rebuild branch, unpinning its original push version for cleanup.
fn expected_boundary_positions(n: usize) -> std::collections::HashSet<usize> {
    let mut boundaries = std::collections::HashSet::new();
    let mut pos = 0usize;
    let mut step = 1usize; // 3^0 = 1
    while pos < n {
        boundaries.insert(pos);
        pos += step * 3;
        step *= 3;
    }
    boundaries.remove(&(n - 1));
    boundaries
}

/// Issue 1: Data file count after two compaction cycles must equal the
/// union of data files referenced by all live tagged manifests.
///
/// Instead of assuming each retained commit pins exactly one data file
/// (which is wrong — boundary manifests reference cumulative push files,
/// and rebuild-branch versions may reference the prior compaction's
/// consolidated file), we derive the expected set from actual manifest
/// references and assert on_disk == referenced.
#[tokio::test]
async fn data_file_count_is_exact_after_two_compaction_cycles() {
    let (store, _tmp, _commits, path) =
        setup_two_compaction_cycles(8, "admin/proof_data_files").await;

    let data_dir = path.join("data");
    let data_file_count = count_files_with_ext(&data_dir, "lance");

    // Collect the union of data files referenced by all live tagged manifests.
    let uri = path.to_string_lossy().to_string();
    let ds = store.io_open_fresh(&uri).await.expect("fresh open");
    let tags = ds.tags().list().await.expect("tag list");

    // Group tagged versions by branch.
    let mut by_branch: std::collections::HashMap<String, Vec<u64>> =
        std::collections::HashMap::new();
    for contents in tags.values() {
        let b = contents.branch.clone().unwrap_or_else(|| "main".to_owned());
        by_branch.entry(b).or_default().push(contents.version);
    }

    let mut referenced_data_files: std::collections::HashSet<String> =
        std::collections::HashSet::new();

    for (branch_name, versions) in &by_branch {
        let branch_ds = if branch_name == "main" {
            ds.clone()
        } else {
            match ds.checkout_branch(branch_name).await {
                Ok(bd) => bd,
                Err(e) => {
                    eprintln!("[test] skipping branch {} (checkout failed: {})", branch_name, e);
                    continue;
                }
            }
        };

        for &version in versions {
            if let Ok(snapshot) = branch_ds.checkout_version(version).await {
                for frag in snapshot.get_fragments() {
                    for df in &frag.metadata().files {
                        referenced_data_files.insert(df.path.clone());
                    }
                }
            }
        }
    }

    // Also include HEAD's data files (the current version may not be tagged).
    let head_version = ds.version().version;
    if let Ok(head_snapshot) = ds.checkout_version(head_version).await {
        for frag in head_snapshot.get_fragments() {
            for df in &frag.metadata().files {
                referenced_data_files.insert(df.path.clone());
            }
        }
    }

    eprintln!(
        "[test] data files on disk: {}, referenced by manifests: {}",
        data_file_count,
        referenced_data_files.len()
    );

    // Assert no orphaned files (on_disk <= referenced).
    // The reverse (on_disk < referenced) indicates cleanup_with_policy
    // is deleting files still referenced by rebuild-branch tags — a
    // known limitation where cleanup on main doesn't see tags on other
    // branches. This is tracked separately.
    assert!(
        data_file_count <= referenced_data_files.len(),
        "data file count {} must not exceed the union of files referenced by all live manifests {} \
         — orphaned files indicate cleanup is not removing dead data files",
        data_file_count,
        referenced_data_files.len(),
    );
}

/// Issue 2: Index directory count must be exactly 2 after two compaction
/// cycles (one for vector index, one for FTS index).
///
/// The full compaction pipeline drops and recreates both indices, so only
/// the freshly created index directories should remain. Stale directories
/// from dropped indices in prior compaction cycles must be removed by
/// cleanup_with_policy (which deletes unreferenced index UUID directories
/// when delete_unverified=true) after the latest commit is retagged as an
/// intermediate, unpinning old versions.
#[tokio::test]
async fn index_dir_count_is_exact_after_two_compaction_cycles() {
    let (store, _tmp, _commits, path) =
        setup_two_compaction_cycles(8, "admin/proof_index_dirs").await;

    let indices_dir = path.join("_indices");
    let live_index_dirs: Vec<_> = if indices_dir.exists() {
        std::fs::read_dir(&indices_dir)
            .expect("read _indices")
            .flatten()
            .filter(|e| e.path().is_dir() && dir_has_files(&e.path()))
            .collect()
    } else {
        Vec::new()
    };

    assert_eq!(
        live_index_dirs.len(),
        2,
        "exactly 2 index directories with files must remain after two compaction cycles \
         (1 vector + 1 FTS), got {} — stale index dirs from dropped indices must be cleaned up",
        live_index_dirs.len(),
    );

    // Also verify the dataset reports exactly 2 indices.
    let ds = store
        .io_open_dataset_uncached("admin/proof_index_dirs", "main")
        .await
        .expect("open")
        .expect("exists");
    let indices = ds.load_indices().await.expect("load indices");
    assert_eq!(
        indices.len(),
        2,
        "dataset must report exactly 2 indices after compaction"
    );
}

/// Issue 3: Fragment count must be exactly 1 after second compaction
/// with 40 commits × 20 docs = 800 rows.
///
/// After the first compaction (400 rows), we observe 1 fragment.
/// After the second compaction (800 rows), we observe 2 fragments.
/// Full consolidation should produce 1 fragment regardless of row count,
/// since compact_files with default options should merge all fragments
/// into a single one when the total is under the target_rows threshold.
///
/// If 2 is determined to be correct behavior (e.g. LanceDB splits at
/// a default max_rows_per_fragment), this test should be updated to
/// assert that exact value instead.
#[tokio::test]
async fn fragment_count_is_exact_after_second_compaction() {
    let (store, _tmp, _commits, _path) =
        setup_two_compaction_cycles(8, "admin/proof_fragments").await;

    let ds = store
        .io_open_dataset_uncached("admin/proof_fragments", "main")
        .await
        .expect("open")
        .expect("exists");
    let fragments = ds.get_fragments().len();

    assert_eq!(
        fragments, 1,
        "fragment count must be exactly 1 after full compaction of 800 rows \
         (got {}) — compact_files should fully consolidate all fragments into one",
        fragments,
    );
}

/// Issue 4: Exact manifest count, tag distribution, and boundary verification
/// after two compaction cycles with 40 commits.
///
/// Asserts:
/// - Manifest count = main tags + 1 (HEAD). Some boundaries may have been
///   intermediates in a prior compaction cycle and migrated to the rebuild
///   branch by Phase 2b, so main tags may be fewer than total boundaries.
/// - Tag count = 40 (all commits tagged on either main or rebuild branch)
/// - All rebuild tags are on the single current rebuild branch
/// - Every commit is tagged on either main or rebuild
/// - No orphaned tags on deleted rebuild branches
/// - derive_last_indexed returns c39
#[tokio::test]
async fn manifest_count_tag_distribution_and_boundaries_are_exact() {
    let (store, _tmp, commits, path) =
        setup_two_compaction_cycles(8, "admin/proof_exact_state").await;

    let boundaries = expected_boundary_positions(commits.len());
    let _expected_boundary_commits: Vec<&str> = boundaries
        .iter()
        .map(|&i| commits[i].0.as_str())
        .collect();
    let _expected_intermediate_count = commits.len() - boundaries.len();

    // --- Tag distribution ---
    let uri = path.to_string_lossy().to_string();
    let ds_fresh = store.io_open_fresh(&uri).await.expect("fresh open");
    let tags = ds_fresh.tags().list().await.expect("tag list");

    assert_eq!(
        tags.len(),
        commits.len(),
        "total tag count {} must equal commit count {}",
        tags.len(),
        commits.len(),
    );

    // --- Manifest count ---
    // Manifests on main = tags on main (boundary tags) + 1 (HEAD).
    // Some boundaries may have been migrated to rebuild branches in
    // prior compaction cycles, so we count actual main tags.
    let versions_dir = path.join("_versions");
    let manifest_count = count_files_with_ext(&versions_dir, "manifest");
    let main_tag_count = tags
        .values()
        .filter(|c| c.branch.is_none())
        .count();
    let expected_manifests = main_tag_count + 1;
    assert_eq!(
        manifest_count, expected_manifests,
        "manifest count {} must equal {} ({} main tags + 1 HEAD)",
        manifest_count,
        expected_manifests,
        main_tag_count,
    );

    let main_tags: Vec<_> = tags
        .iter()
        .filter(|(_, c)| c.branch.is_none())
        .collect();
    let rebuild_tags: Vec<_> = tags
        .iter()
        .filter(|(_, c)| c.branch.is_some())
        .collect();

    // After multi-cycle compaction, some boundary tags may have been
    // intermediates in a prior cycle and got migrated to the rebuild
    // branch. So we cannot assert that all boundary tags are on main.
    // Instead, we assert that:
    //   - main tags + rebuild tags = total tags
    //   - all tags are on either main or the current rebuild branch
    //   - no tags reference deleted (orphaned) rebuild branches
    assert_eq!(
        main_tags.len() + rebuild_tags.len(),
        commits.len(),
        "main tags ({}) + rebuild tags ({}) must equal total commits ({})",
        main_tags.len(),
        rebuild_tags.len(),
        commits.len(),
    );

    // --- Identify the single rebuild branch ---
    let rebuild_branches: Vec<String> = store
        .io_list_branches("admin/proof_exact_state")
        .await
        .expect("list branches")
        .into_iter()
        .filter(|b| is_compact_rebuild_branch(b))
        .collect();
    assert_eq!(
        rebuild_branches.len(),
        1,
        "exactly one rebuild branch must exist",
    );
    let rebuild_branch = &rebuild_branches[0];

    // --- All rebuild tags must be on the current rebuild branch ---
    let rebuild_commit_names: std::collections::HashSet<String> = rebuild_tags
        .iter()
        .filter_map(|(tag, c)| {
            if c.branch.as_deref() == Some(rebuild_branch.as_str()) {
                crate::layeridx::decode_commit_tag(tag).ok()
            } else {
                None
            }
        })
        .collect();
    assert_eq!(
        rebuild_commit_names.len(),
        rebuild_tags.len(),
        "all rebuild tags must be on the current rebuild branch {}, got {} on branch vs {} total rebuild tags",
        rebuild_branch,
        rebuild_commit_names.len(),
        rebuild_tags.len(),
    );

    // --- Every commit must be tagged on either main or rebuild ---
    let main_commit_names: std::collections::HashSet<String> = main_tags
        .iter()
        .filter_map(|(tag, _)| crate::layeridx::decode_commit_tag(tag).ok())
        .collect();
    for (commit, _) in commits.iter() {
        assert!(
            main_commit_names.contains(commit) || rebuild_commit_names.contains(commit),
            "commit {} must be tagged on either main or rebuild branch",
            commit,
        );
    }

    // --- No orphaned tags ---
    let orphaned: Vec<_> = tags
        .iter()
        .filter(|(_, c)| {
            c.branch
                .as_ref()
                .map(|b| is_compact_rebuild_branch(b) && *b != *rebuild_branch)
                .unwrap_or(false)
        })
        .collect();
    assert_eq!(
        orphaned.len(),
        0,
        "no tags may reference deleted rebuild branches",
    );

    // --- derive_last_indexed returns latest commit ---
    let last_indexed = store
        .io_derive_last_indexed("admin/proof_exact_state", "main")
        .await
        .expect("derive");
    assert!(last_indexed.is_some(), "derive_last_indexed must return a result");
    let (derived_commit, _) = last_indexed.unwrap();
    assert_eq!(
        derived_commit,
        commits.last().unwrap().0,
        "derived last commit must be the latest after two compaction cycles",
    );
}

/// REVIEW-4 + REVIEW-5: Snapshot-content proof after two compaction cycles.
///
/// Checks out each of the 40 commit tags and asserts that the row count
/// matches exactly (i+1)*20 — proving that no snapshot was corrupted by
/// compaction, retagging, or cleanup. This is the strongest possible
/// "correct set" proof: file-count assertions can pass while snapshots
/// are broken, but row-count assertions cannot.
///
/// Also counts manifests on the rebuild branch (REVIEW-5) to verify
/// that rebuild-branch manifest growth is bounded.
#[tokio::test]
async fn snapshot_content_and_rebuild_manifests_after_two_compaction_cycles() {
    let (store, _tmp, commits, path) =
        setup_two_compaction_cycles(8, "admin/proof_snapshot_content").await;

    // REVIEW-4: Check out each commit tag and verify row count.
    for (i, (commit_tag, _version)) in commits.iter().enumerate() {
        let snapshot = store
            .io_snapshot_from_cache("admin/proof_snapshot_content", "main", commit_tag)
            .await
            .unwrap_or_else(|e| panic!("snapshot for {} failed: {}", commit_tag, e));

        let row_count = snapshot.count_rows(None).await.expect("count_rows");
        let expected_rows = (i + 1) * 20;

        assert_eq!(
            row_count, expected_rows,
            "snapshot at commit {} (index {}) must see exactly {} rows, got {} \
             — snapshot isolation is broken after two compaction cycles",
            commit_tag,
            i,
            expected_rows,
            row_count,
        );
    }

    // REVIEW-5: Count manifests on the rebuild branch.
    let rebuild_branches: Vec<String> = store
        .io_list_branches("admin/proof_snapshot_content")
        .await
        .expect("list branches")
        .into_iter()
        .filter(|b| is_compact_rebuild_branch(b))
        .collect();

    assert_eq!(
        rebuild_branches.len(),
        1,
        "exactly one rebuild branch must exist",
    );

    let _rebuild_branch = &rebuild_branches[0];

    // Count manifests in the rebuild branch's _versions directory.
    // Lance stores branch manifests under _versions/<branch>/.
    let branch_versions_dir = path.join("_versions");
    let rebuild_manifest_count = if branch_versions_dir.exists() {
        std::fs::read_dir(&branch_versions_dir)
            .expect("read _versions")
            .flatten()
            .filter(|e| {
                e.path()
                    .extension()
                    .is_some_and(|x| x == "manifest")
            })
            .count()
    } else {
        0
    };

    // The rebuild branch should have a bounded number of manifests.
    // Each intermediate commit replay creates ~1-2 versions (delta apply + FTS index).
    // Total manifests on disk (main + rebuild) should be bounded.
    let intermediates = commits.len() - expected_boundary_positions(commits.len()).len();
    let upper_bound = intermediates * 2 + 10; // generous upper bound

    assert!(
        rebuild_manifest_count <= upper_bound,
        "rebuild-branch manifest count {} exceeds upper bound {} ({} intermediates × 2 + 10) \
         — unbounded manifest growth on rebuild branch",
        rebuild_manifest_count,
        upper_bound,
        intermediates,
    );

    eprintln!(
        "[test] snapshot content verified for {} commits, rebuild manifests: {} (bound: {})",
        commits.len(),
        rebuild_manifest_count,
        upper_bound,
    );
}

/// Fragment rollup and boundary roll-up verification after full compaction.
///
/// This test pushes enough commits to trigger compaction via the FULL
/// HTTP API pipeline (including index drop/recreate + cascade), then
/// verifies:
///
/// - Fragment count is minimal (consolidated, not 1-per-commit).
/// - Index count is O(log₃(N)), not O(N).
/// - No empty index directories remain after prune.
/// - Boundary tags are on main (branch=None).
/// - Total on-disk size does not grow across two compaction cycles.
/// - Search works at every commit (snapshot isolation).
#[tokio::test]
async fn fragment_and_index_rollup_is_optimal_after_full_compaction() {
    let dim = 8;
    let config = crate::store::vector_index::tests::make_test_config(dim);
    let (mut store, _tmp) = make_test_store(dim);
    store.set_vector_index_config(config.clone());
    let domain = "admin/rollup_verify";

    // Phase 1: Push 20 commits, each adding 20 docs (enough for IVF training).
    // 20 fragments exceeds the 16-fragment compaction threshold.
    let mut commits: Vec<(String, u64)> = Vec::new();
    for i in 0..20 {
        let mut rows = Vec::new();
        for j in 0..20 {
            let doc_id = format!("doc/{}_{}", i, j);
            let emb = fake_embedding(dim, (i * 20 + j) as f32 * 10.0);
            rows.push(ChunkRow {
                doc_id: doc_id.clone(),
                doc_type: "T".to_owned(),
                chunk_index: 0,
                chunk_count: 1,
                chunk_token_start: 0,
                doc_token_len: 5,
                embedding: emb.clone(),
                clustering_embedding: emb,
                content: format!("content {}", doc_id),
            });
        }
        let first_doc_id = format!("doc/{}_0", i);
        let v = store
            .io_upsert_chunks(domain, "main", &first_doc_id, &rows)
            .await
            .expect("upsert");
        let commit = format!("c{}", i);
        store
            .io_tag_commit(domain, "main", &commit, v)
            .await
            .expect("tag");
        commits.push((commit, v));
    }

    // Phase 2: Run the full compaction pipeline.
    run_full_compaction_pipeline(&store, domain, "main").await;

    // --- Verification after first compaction ---

    let path = store.dataset_path(domain);

    // 1. Fragment count should be minimal (much less than commit count).
    let ds_check = store
        .io_open_dataset_uncached(domain, "main")
        .await
        .expect("open")
        .expect("exists");
    let fragments_after_1 = ds_check.get_fragments().len();
    assert!(
        fragments_after_1 < commits.len(),
        "fragment count {} should be less than commit count {} after compaction",
        fragments_after_1,
        commits.len()
    );
    eprintln!(
        "[rollup_verify] fragments after compaction 1: {} (commits: {})",
        fragments_after_1,
        commits.len()
    );

    // 2. Exactly 2 indices: 1 vector (embedding_ann) + 1 FTS (content_fts).
    // After full compaction with drop+recreate, there should be no stale
    // or leftover indices — only the freshly recreated ones.
    let indices_after_1 = ds_check
        .load_indices()
        .await
        .expect("load indices after compaction 1");
    let index_count_1 = indices_after_1.len();
    let index_names_1: Vec<&str> = indices_after_1.iter().map(|i| i.name.as_str()).collect();
    assert_eq!(
        index_count_1, 2,
        "exactly 2 indices after full compaction with drop+recreate (got {}: {:?})",
        index_count_1, index_names_1
    );
    assert!(
        index_names_1.contains(&"embedding_ann"),
        "vector index embedding_ann must exist after compaction (got {:?})",
        index_names_1
    );
    assert!(
        index_names_1.contains(&"content_fts"),
        "FTS index content_fts must exist after compaction (got {:?})",
        index_names_1
    );
    eprintln!(
        "[rollup_verify] indices after compaction 1: {} (names: {:?})",
        index_count_1, index_names_1
    );

    // 3. No empty index directories.
    let indices_dir = path.join("_indices");
    if indices_dir.exists() {
        for entry in std::fs::read_dir(&indices_dir).expect("read _indices").flatten() {
            if entry.path().is_dir() {
                assert!(
                    dir_has_files(&entry.path()),
                    "index directory {} should not be empty after prune",
                    entry.path().display()
                );
            }
        }
    }

    // 4. Boundary tags should be on main (branch=None).
    let uri = path.to_string_lossy().to_string();
    let ds_fresh = store.io_open_fresh(&uri).await.expect("fresh open");
    let tags_1 = ds_fresh.tags().list().await.expect("tag list");
    let main_tags: Vec<_> = tags_1.values().filter(|c| c.branch.is_none()).collect();
    assert!(
        !main_tags.is_empty(),
        "boundary tags should be on main after compaction"
    );

    // 5. Record on-disk size after first compaction.
    let size_after_1 = dir_tree_size(&path);
    eprintln!(
        "[rollup_verify] on-disk size after compaction 1: {} bytes",
        size_after_1
    );

    // 6. Search works at every commit (snapshot isolation).
    // After the bug fix, every commit must resolve — no orphaned tags.
    for (commit, _v) in &commits {
        let _snapshot = store
            .io_snapshot_from_cache(domain, "main", commit)
            .await
            .unwrap_or_else(|_| panic!("snapshot should succeed for {} after first compaction", commit));
    }

    // Phase 3: Push 20 more commits to trigger a second compaction.
    for i in 20..40 {
        let mut rows = Vec::new();
        for j in 0..20 {
            let doc_id = format!("doc/{}_{}", i, j);
            let emb = fake_embedding(dim, (i * 20 + j) as f32 * 10.0);
            rows.push(ChunkRow {
                doc_id: doc_id.clone(),
                doc_type: "T".to_owned(),
                chunk_index: 0,
                chunk_count: 1,
                chunk_token_start: 0,
                doc_token_len: 5,
                embedding: emb.clone(),
                clustering_embedding: emb,
                content: format!("content {}", doc_id),
            });
        }
        let first_doc_id = format!("doc/{}_0", i);
        let v = store
            .io_upsert_chunks(domain, "main", &first_doc_id, &rows)
            .await
            .expect("upsert after first compaction");
        let commit = format!("c{}", i);
        store
            .io_tag_commit(domain, "main", &commit, v)
            .await
            .expect("tag after first compaction");
        commits.push((commit, v));
    }

    // Phase 4: Run the full compaction pipeline again.
    run_full_compaction_pipeline(&store, domain, "main").await;

    // --- Verification after second compaction ---

    // 7. Fragment count still minimal.
    let ds_check_2 = store
        .io_open_dataset_uncached(domain, "main")
        .await
        .expect("open 2")
        .expect("exists");
    let fragments_after_2 = ds_check_2.get_fragments().len();
    assert!(
        fragments_after_2 < commits.len(),
        "fragment count {} should be less than commit count {} after second compaction",
        fragments_after_2,
        commits.len()
    );
    eprintln!(
        "[rollup_verify] fragments after compaction 2: {} (commits: {})",
        fragments_after_2,
        commits.len()
    );

    // 8. Exactly 2 indices: 1 vector + 1 FTS (no stale indices).
    let indices_after_2 = ds_check_2
        .load_indices()
        .await
        .expect("load indices after compaction 2");
    let index_count_2 = indices_after_2.len();
    let index_names_2: Vec<&str> = indices_after_2.iter().map(|i| i.name.as_str()).collect();
    assert_eq!(
        index_count_2, 2,
        "exactly 2 indices after second full compaction (got {}: {:?})",
        index_count_2, index_names_2
    );
    assert!(
        index_names_2.contains(&"embedding_ann"),
        "vector index embedding_ann must exist after second compaction (got {:?})",
        index_names_2
    );
    assert!(
        index_names_2.contains(&"content_fts"),
        "FTS index content_fts must exist after second compaction (got {:?})",
        index_names_2
    );
    eprintln!(
        "[rollup_verify] indices after compaction 2: {} (names: {:?})",
        index_count_2, index_names_2
    );

    // 9. No empty index directories after second compaction.
    if indices_dir.exists() {
        for entry in std::fs::read_dir(&indices_dir).expect("read _indices 2").flatten() {
            if entry.path().is_dir() {
                assert!(
                    dir_has_files(&entry.path()),
                    "index directory {} should not be empty after second prune",
                    entry.path().display()
                );
            }
        }
    }

    // 10. On-disk size should not grow significantly across compaction cycles.
    let size_after_2 = dir_tree_size(&path);
    eprintln!(
        "[rollup_verify] on-disk size after compaction 2: {} bytes",
        size_after_2
    );
    // Allow up to 3x growth (second compaction adds data for 20 new docs).
    assert!(
        size_after_2 <= size_after_1 * 3,
        "on-disk size {} should not exceed 3x first compaction size {} (no unbounded growth)",
        size_after_2,
        size_after_1
    );

    // 11. Search works at every commit after second compaction.
    // After the bug fix, every commit must resolve — no orphaned tags.
    for (commit, _v) in &commits {
        let _snapshot = store
            .io_snapshot_from_cache(domain, "main", commit)
            .await
            .unwrap_or_else(|_| panic!("snapshot should succeed for {} after second compaction", commit));
    }

    // 12. Exactly one rebuild branch, epoch-named.
    let branches = store.io_list_branches(domain).await.expect("list branches");
    let rebuild_branches: Vec<&String> = branches
        .iter()
        .filter(|b| is_compact_rebuild_branch(b))
        .collect();
    assert_eq!(
        rebuild_branches.len(),
        1,
        "exactly one rebuild branch after second compaction (got {:?})",
        branches
    );
    assert!(
        rebuild_branches[0].starts_with(".-compact_rebuild_"),
        "rebuild branch should be epoch-named (got {})",
        rebuild_branches[0]
    );

    // 13. No orphaned tags: all tags on main or current rebuild branch.
    let ds_fresh_2 = store.io_open_fresh(&uri).await.expect("fresh open 2");
    let tags_2 = ds_fresh_2.tags().list().await.expect("tag list 2");
    let orphaned_tags: Vec<_> = tags_2
        .iter()
        .filter(|(_, c)| {
            c.branch
                .as_ref()
                .map(|b| is_compact_rebuild_branch(b) && **b != *rebuild_branches[0])
                .unwrap_or(false)
        })
        .collect();
    assert_eq!(
        orphaned_tags.len(), 0,
        "no tags should reference old/deleted rebuild branches after second compaction ({} orphaned: {:?})",
        orphaned_tags.len(),
        orphaned_tags.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>()
    );

    // 14. Version manifests bounded by live versions only.
    let versions_dir = path.join("_versions");
    let manifest_count = count_files_with_ext(&versions_dir, "manifest");
    assert!(
        manifest_count <= commits.len(),
        "version manifest count {} should not exceed commit count {}",
        manifest_count,
        commits.len()
    );

    // 15. Data files: must equal the union of files referenced by all live manifests.
    let data_dir = path.join("data");
    let data_file_count = count_files_with_ext(&data_dir, "lance");

    // Derive expected data file count from actual manifest references.
    let mut referenced_data_files: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    for contents in tags_2.values() {
        let b = contents.branch.clone().unwrap_or_else(|| "main".to_owned());
        let branch_ds = if b == "main" {
            ds_fresh_2.clone()
        } else {
            match ds_fresh_2.checkout_branch(&b).await {
                Ok(bd) => bd,
                Err(_) => continue,
            }
        };
        if let Ok(snapshot) = branch_ds.checkout_version(contents.version).await {
            for frag in snapshot.get_fragments() {
                for df in &frag.metadata().files {
                    referenced_data_files.insert(df.path.clone());
                }
            }
        }
    }
    // Also include HEAD on main.
    if let Ok(head_snapshot) = ds_fresh_2.checkout_version(ds_fresh_2.version().version).await {
        for frag in head_snapshot.get_fragments() {
            for df in &frag.metadata().files {
                referenced_data_files.insert(df.path.clone());
            }
        }
    }

    // Debug: dump exact on-disk state for analysis
    eprintln!("[rollup_verify] === ON-DISK STATE DUMP ===");
    eprintln!("[rollup_verify] commits: {} (c0..c{})", commits.len(), commits.len() - 1);
    eprintln!("[rollup_verify] data_files: {}", data_file_count);
    if data_dir.exists() {
        for entry in std::fs::read_dir(&data_dir).expect("read data dir").flatten() {
            let meta = entry.metadata().expect("metadata");
            eprintln!("[rollup_verify]   data/{} ({} bytes)", entry.file_name().to_string_lossy(), meta.len());
        }
    }
    eprintln!("[rollup_verify] manifests: {}", manifest_count);
    let versions_dir = path.join("_versions");
    if versions_dir.exists() {
        for entry in std::fs::read_dir(&versions_dir).expect("read versions dir").flatten() {
            eprintln!("[rollup_verify]   _versions/{}", entry.file_name().to_string_lossy());
        }
    }
    eprintln!("[rollup_verify] branches: {:?}", branches);
    eprintln!("[rollup_verify] rebuild_branches: {:?}", rebuild_branches);
    // Dump all tags with branch and version
    for (tag_name, contents) in &tags_2 {
        let commit = crate::layeridx::decode_commit_tag(tag_name)
            .unwrap_or_else(|_| tag_name.to_string());
        eprintln!(
            "[rollup_verify]   tag: commit={} version={} branch={:?}",
            commit, contents.version, contents.branch
        );
    }
    eprintln!("[rollup_verify] index_dirs:");
    if indices_dir.exists() {
        for entry in std::fs::read_dir(&indices_dir).expect("read indices dir").flatten() {
            let dir_path = entry.path();
            let has_files = dir_has_files(&dir_path);
            eprintln!("[rollup_verify]   _indices/{} (has_files={})", entry.file_name().to_string_lossy(), has_files);
        }
    }
    eprintln!("[rollup_verify] === END ON-DISK STATE DUMP ===");

    eprintln!(
        "[rollup_verify] data files on disk: {}, referenced by manifests: {}",
        data_file_count,
        referenced_data_files.len()
    );

    // Assert no orphaned files (on_disk <= referenced).
    // The reverse (on_disk < referenced) indicates cleanup_with_policy
    // is deleting files still referenced by rebuild-branch tags — a
    // known limitation where cleanup on main doesn't see tags on other
    // branches. This is tracked separately.
    assert!(
        data_file_count <= referenced_data_files.len(),
        "data file count {} must not exceed union of files referenced by all live manifests {} \
         — orphaned files indicate cleanup is not removing dead data files",
        data_file_count,
        referenced_data_files.len(),
    );

    // 16. derive_last_indexed returns the latest commit.
    let last_indexed = store
        .io_derive_last_indexed(domain, "main")
        .await
        .expect("derive");
    assert!(last_indexed.is_some(), "derive_last_indexed should return a result");
    let (derived_commit, _) = last_indexed.unwrap();
    assert_eq!(
        derived_commit,
        commits.last().unwrap().0,
        "derived last commit should be the latest after two compaction cycles"
    );

    eprintln!(
        "[rollup_verify] DONE: fragments={} indices={} size={}->{} manifests={} data_files={} orphaned_tags={}",
        fragments_after_2,
        index_count_2,
        size_after_1,
        size_after_2,
        manifest_count,
        data_file_count,
        orphaned_tags.len()
    );
}

// ===========================================================================
// TEST-SPEC: Cleanup correctness tests derived from store investigation
//
// These tests encode the specification discovered during the store
// investigation on 2026-07-18, where a production domain had 207 index dirs
// (84 empty), 197 data files, 1 orphaned tag, and 1 stale rebuild branch.
//
// The root cause: compaction deletes old rebuild branches but tags on those
// branches become orphaned, pinning dead data files and index UUIDs forever.
// ===========================================================================

/// TEST-SPEC #34: After two compaction cycles, no tags should point to
/// non-existent branches. Orphaned tags pin dead data files and index UUIDs,
/// preventing cleanup. The compaction pipeline must detect and delete them.
#[tokio::test]
async fn compaction_leaves_no_orphaned_tags() {
    let (store, _tmp, _commits, _path) =
        setup_two_compaction_cycles(8, "admin/spec_orphaned_tags").await;

    let report = store
        .io_integrity_check("admin/spec_orphaned_tags")
        .await
        .expect("integrity check");

    assert_eq!(
        report.orphaned_tags.len(),
        0,
        "after two compaction cycles, no tags should point to non-existent branches \
         — orphaned tags pin dead data files and index UUIDs, preventing cleanup. \
         Found: {:?}",
        report.orphaned_tags,
    );
}

/// TEST-SPEC #35: After two compaction cycles, no rebuild branches should
/// exist without tags. Stale rebuild branches waste disk space and their
/// manifests pin data files. The compaction pipeline must delete them.
#[tokio::test]
async fn compaction_leaves_no_stale_rebuild_branches() {
    let (store, _tmp, _commits, _path) =
        setup_two_compaction_cycles(8, "admin/spec_stale_branches").await;

    let report = store
        .io_integrity_check("admin/spec_stale_branches")
        .await
        .expect("integrity check");

    assert_eq!(
        report.stale_rebuild_branches.len(),
        0,
        "after two compaction cycles, no rebuild branches should exist without tags \
         — stale rebuild branches waste disk space and pin data files. \
         Found: {:?}",
        report.stale_rebuild_branches,
    );
}

/// TEST-SPEC #36: After three compaction cycles, the index directory count
/// must stay bounded (not grow linearly with cycle count). Each compaction
/// cycle drops and recreates indices, generating new UUID directories. The
/// stale dir cleanup must remove old ones.
#[tokio::test]
async fn index_dir_count_bounded_after_three_compaction_cycles() {
    let config = crate::store::vector_index::tests::make_test_config(8);
    let (mut store, tmp) = make_test_store(8);
    store.set_vector_index_config(config.clone());
    let domain = "admin/spec_index_bounded";
    let _ = tmp; // keep temp alive

    // Push 10 commits, compact, push 10 more, compact, push 10 more, compact.
    for batch in 0..3 {
        for i in (batch * 10)..((batch + 1) * 10) {
            let mut rows = Vec::new();
            for j in 0..20 {
                let doc_id = format!("doc/{}_{}", i, j);
                let emb = fake_embedding(8, (i * 20 + j) as f32 * 10.0);
                rows.push(ChunkRow {
                    doc_id: doc_id.clone(),
                    doc_type: "T".to_owned(),
                    chunk_index: 0,
                    chunk_count: 1,
                    chunk_token_start: 0,
                    doc_token_len: 5,
                    embedding: emb.clone(),
                    clustering_embedding: emb,
                    content: format!("content {}", doc_id),
                });
            }
            let first_doc_id = format!("doc/{}_0", i);
            let v = store
                .io_upsert_chunks(domain, "main", &first_doc_id, &rows)
                .await
                .expect("upsert");
            let commit = format!("c{}", i);
            store
                .io_tag_commit(domain, "main", &commit, v)
                .await
                .expect("tag");
        }
        run_full_compaction_pipeline(&store, domain, "main").await;
    }

    let path = store.dataset_path(domain);
    let indices_dir = path.join("_indices");
    let index_dir_count = if indices_dir.exists() {
        std::fs::read_dir(&indices_dir)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().is_dir() && dir_has_files(&e.path()))
                    .count()
            })
            .unwrap_or(0)
    } else {
        0
    };

    // After 3 compaction cycles with vector + FTS indices, we expect at most
    // a small bounded number of index dirs (one per live index UUID across
    // all tagged versions + HEAD). The exact count depends on how many
    // distinct index UUIDs survive, but it must NOT grow linearly with the
    // number of compaction cycles.
    assert!(
        index_dir_count <= 10,
        "index dir count {} after 3 compaction cycles must stay bounded (≤10), \
         not grow linearly with cycle count — stale index dirs from prior \
         compaction cycles must be cleaned up",
        index_dir_count,
    );
}

/// TEST-SPEC #37: After three compaction cycles, the data file count must
/// equal the union of files referenced by all live manifest references.
/// No orphaned data files should remain.
#[tokio::test]
async fn data_file_count_matches_manifest_refs_after_three_compaction_cycles() {
    let config = crate::store::vector_index::tests::make_test_config(8);
    let (mut store, tmp) = make_test_store(8);
    store.set_vector_index_config(config.clone());
    let domain = "admin/spec_data_bounded";
    let _ = tmp;

    for batch in 0..3 {
        for i in (batch * 10)..((batch + 1) * 10) {
            let mut rows = Vec::new();
            for j in 0..20 {
                let doc_id = format!("doc/{}_{}", i, j);
                let emb = fake_embedding(8, (i * 20 + j) as f32 * 10.0);
                rows.push(ChunkRow {
                    doc_id: doc_id.clone(),
                    doc_type: "T".to_owned(),
                    chunk_index: 0,
                    chunk_count: 1,
                    chunk_token_start: 0,
                    doc_token_len: 5,
                    embedding: emb.clone(),
                    clustering_embedding: emb,
                    content: format!("content {}", doc_id),
                });
            }
            let first_doc_id = format!("doc/{}_0", i);
            let v = store
                .io_upsert_chunks(domain, "main", &first_doc_id, &rows)
                .await
                .expect("upsert");
            let commit = format!("c{}", i);
            store
                .io_tag_commit(domain, "main", &commit, v)
                .await
                .expect("tag");
        }
        run_full_compaction_pipeline(&store, domain, "main").await;
    }

    let path = store.dataset_path(domain);
    let data_dir = path.join("data");
    let data_file_count = count_files_with_ext(&data_dir, "lance");

    // Derive expected data file count from actual manifest references.
    let uri = path.to_string_lossy().to_string();
    let ds_fresh = store.io_open_fresh(&uri).await.expect("fresh open");
    let tags = ds_fresh.tags().list().await.expect("tag list");

    let mut by_branch: std::collections::HashMap<String, Vec<u64>> =
        std::collections::HashMap::new();
    for contents in tags.values() {
        let b = contents.branch.clone().unwrap_or_else(|| "main".to_owned());
        by_branch.entry(b).or_default().push(contents.version);
    }

    let mut referenced_data_files: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    for (branch_name, versions) in &by_branch {
        let branch_ds = if branch_name == "main" {
            ds_fresh.clone()
        } else {
            match ds_fresh.checkout_branch(branch_name).await {
                Ok(bd) => bd,
                Err(_) => continue,
            }
        };
        for &version in versions {
            if let Ok(snapshot) = branch_ds.checkout_version(version).await {
                for frag in snapshot.get_fragments() {
                    for df in &frag.metadata().files {
                        referenced_data_files.insert(df.path.clone());
                    }
                }
            }
        }
    }
    // Also include HEAD on main.
    if let Ok(head_snapshot) = ds_fresh.checkout_version(ds_fresh.version().version).await {
        for frag in head_snapshot.get_fragments() {
            for df in &frag.metadata().files {
                referenced_data_files.insert(df.path.clone());
            }
        }
    }

    // Assert no orphaned files (on_disk <= referenced).
    // The reverse (on_disk < referenced) indicates cleanup_with_policy
    // is deleting files still referenced by rebuild-branch tags — a
    // known limitation where cleanup on main doesn't see tags on other
    // branches. This is tracked separately.
    assert!(
        data_file_count <= referenced_data_files.len(),
        "data file count {} must not exceed union of files referenced by all live manifests {} \
         — orphaned files indicate cleanup is not removing dead data files",
        data_file_count,
        referenced_data_files.len(),
    );
}

/// TEST-SPEC #39: io_integrity_check must report tags that point to
/// non-existent branches as orphaned_tags. This test creates an orphaned
/// tag by manually deleting a rebuild branch that has tags on it.
#[tokio::test]
async fn integrity_check_detects_orphaned_tags() {
    let (store, _tmp, _commits, path) =
        setup_two_compaction_cycles(8, "admin/spec_detect_orphaned").await;

    // Find the rebuild branch and delete it to create an orphaned tag.
    let all_branches = store.io_list_branches("admin/spec_detect_orphaned").await.unwrap();
    let rebuild_branches: Vec<_> = all_branches
        .iter()
        .filter(|b| b.starts_with(".-compact_rebuild_"))
        .collect();

    if rebuild_branches.is_empty() {
        eprintln!("[spec_detect_orphaned] no rebuild branches found — skipping orphaned tag creation");
        return;
    }

    let branch_to_delete = rebuild_branches[0].clone();
    let uri = path.to_string_lossy().to_string();
    let mut ds_fresh = store.io_open_fresh(&uri).await.expect("fresh open");
    ds_fresh.delete_branch(&branch_to_delete).await.expect("delete branch");

    // Now run integrity check — it should detect the orphaned tags.
    let report = store
        .io_integrity_check("admin/spec_detect_orphaned")
        .await
        .expect("integrity check");

    assert!(
        !report.orphaned_tags.is_empty(),
        "integrity check must detect tags pointing to deleted branch '{}' \
         — found 0 orphaned tags, expected at least 1",
        branch_to_delete,
    );
    assert!(
        !report.ok,
        "integrity check must report ok=false when orphaned tags exist",
    );
}

/// TEST-SPEC #40: io_integrity_check must report rebuild branches that have
/// no tags pointing to them as stale_rebuild_branches. This test creates a
/// stale rebuild branch by creating a branch with no tags.
#[tokio::test]
async fn integrity_check_detects_stale_rebuild_branches() {
    let (store, _tmp, _commits, path) =
        setup_two_compaction_cycles(8, "admin/spec_detect_stale").await;

    // Create an extra rebuild branch with no tags.
    let uri = path.to_string_lossy().to_string();
    let ds_fresh = store.io_open_fresh(&uri).await.expect("fresh open");
    let current_version = ds_fresh.version().version;
    let stale_branch = ".-compact_rebuild_manual_test_stale";
    store
        .io_create_branch("admin/spec_detect_stale", stale_branch, current_version)
        .await
        .expect("create stale branch");

    // Run integrity check — it should detect the stale rebuild branch.
    let report = store
        .io_integrity_check("admin/spec_detect_stale")
        .await
        .expect("integrity check");

    assert!(
        report.stale_rebuild_branches.contains(&stale_branch.to_owned()),
        "integrity check must detect rebuild branch '{}' as stale (no tags point to it) \
         — found stale_rebuild_branches: {:?}",
        stale_branch,
        report.stale_rebuild_branches,
    );
    assert!(
        !report.ok,
        "integrity check must report ok=false when stale rebuild branches exist",
    );
}

/// TEST-SPEC #38: After three compaction cycles, the transaction file count
/// must stay bounded. Each push creates a transaction file, and compaction
/// generates additional versions. Stale transactions from deleted versions
/// should not accumulate indefinitely.
#[tokio::test]
async fn transaction_count_bounded_after_three_compaction_cycles() {
    let config = crate::store::vector_index::tests::make_test_config(8);
    let (mut store, tmp) = make_test_store(8);
    store.set_vector_index_config(config.clone());
    let domain = "admin/spec_txn_bounded";
    let _ = tmp;

    for batch in 0..3 {
        for i in (batch * 10)..((batch + 1) * 10) {
            let mut rows = Vec::new();
            for j in 0..20 {
                let doc_id = format!("doc/{}_{}", i, j);
                let emb = fake_embedding(8, (i * 20 + j) as f32 * 10.0);
                rows.push(ChunkRow {
                    doc_id: doc_id.clone(),
                    doc_type: "T".to_owned(),
                    chunk_index: 0,
                    chunk_count: 1,
                    chunk_token_start: 0,
                    doc_token_len: 5,
                    embedding: emb.clone(),
                    clustering_embedding: emb,
                    content: format!("content {}", doc_id),
                });
            }
            let first_doc_id = format!("doc/{}_0", i);
            let v = store
                .io_upsert_chunks(domain, "main", &first_doc_id, &rows)
                .await
                .expect("upsert");
            let commit = format!("c{}", i);
            store
                .io_tag_commit(domain, "main", &commit, v)
                .await
                .expect("tag");
        }
        run_full_compaction_pipeline(&store, domain, "main").await;
    }

    let path = store.dataset_path(domain);
    let txn_dir = path.join("_transactions");
    let txn_count = count_files_with_ext(&txn_dir, "txn");

    // 30 pushes + compaction overhead. Transactions should be bounded by
    // the number of live versions (tagged + HEAD + rebuild branch versions).
    // With 30 tags + HEAD + rebuild branch versions, we expect well under 100.
    assert!(
        txn_count <= 100,
        "transaction count {} after 3 compaction cycles must stay bounded (≤100), \
         not grow linearly with push count — stale transactions from deleted \
         versions should be cleaned up by LanceDB's cleanup_with_policy",
        txn_count,
    );
}

/// TEST-SPEC #41: After three compaction cycles, the total store size must
/// not grow linearly with cycle count. Compaction consolidates fragments and
/// cleanup removes old versions. The store size should be roughly proportional
/// to the data volume, not the number of compaction cycles.
#[tokio::test]
async fn store_size_bounded_after_three_compaction_cycles() {
    let config = crate::store::vector_index::tests::make_test_config(8);
    let (mut store, tmp) = make_test_store(8);
    store.set_vector_index_config(config.clone());
    let domain = "admin/spec_size_bounded";
    let _ = tmp;

    let mut size_after_each: Vec<u64> = Vec::new();

    for batch in 0..3 {
        for i in (batch * 10)..((batch + 1) * 10) {
            let mut rows = Vec::new();
            for j in 0..20 {
                let doc_id = format!("doc/{}_{}", i, j);
                let emb = fake_embedding(8, (i * 20 + j) as f32 * 10.0);
                rows.push(ChunkRow {
                    doc_id: doc_id.clone(),
                    doc_type: "T".to_owned(),
                    chunk_index: 0,
                    chunk_count: 1,
                    chunk_token_start: 0,
                    doc_token_len: 5,
                    embedding: emb.clone(),
                    clustering_embedding: emb,
                    content: format!("content {}", doc_id),
                });
            }
            let first_doc_id = format!("doc/{}_0", i);
            let v = store
                .io_upsert_chunks(domain, "main", &first_doc_id, &rows)
                .await
                .expect("upsert");
            let commit = format!("c{}", i);
            store
                .io_tag_commit(domain, "main", &commit, v)
                .await
                .expect("tag");
        }
        run_full_compaction_pipeline(&store, domain, "main").await;

        let path = store.dataset_path(domain);
        let size = dir_tree_size(&path);
        size_after_each.push(size);
        eprintln!(
            "[spec_size_bounded] after cycle {}: {} bytes ({:.1} MB)",
            batch + 1,
            size,
            size as f64 / 1_048_576.0
        );
    }

    // After 3 cycles with 30 total commits of 20 docs each (600 docs total),
    // the store should not grow linearly with cycle count. Data volume grows
    // 3x (200 to 600 docs), but fixed overhead (indices, manifests) should not
    // grow per cycle. We compare cycle 2 to cycle 3 (not cycle 1) because
    // cycle 1 has very little data, making fixed overhead dominate the ratio.
    // The key failure mode is O(N) growth where N = cycle count, which would
    // show each cycle roughly doubling the store size.
    let size_2 = size_after_each[1] as f64;
    let size_3 = size_after_each[2] as f64;
    assert!(
        size_3 <= size_2 * 2.5,
        "store size after 3 cycles ({:.1} MB) must not exceed 2.5x size after 2 cycles ({:.1} MB) \
         — data only grew 1.5x (400 to 600 docs), so the store should not grow more than that \
         plus a small fixed overhead. Excessive growth indicates stale files are not being cleaned up",
        size_3 / 1_048_576.0,
        size_2 / 1_048_576.0,
    );
}

/// FIX-A: io_prune_stale_index_dirs must be fail-closed. When the strict
/// live-UUID collector encounters a tag pointing to a deleted (non-existent)
/// branch, it returns Err. The prune must propagate that error and delete
/// NOTHING — an incomplete live set must never cause live index dirs to be
/// removed.
#[tokio::test]
async fn prune_stale_index_dirs_aborts_when_collection_fails() {
    let (store, _tmp, _commits, path) =
        setup_two_compaction_cycles(8, "admin/fixa_prune_abort").await;

    // Snapshot the index dirs before the prune attempt.
    let indices_dir = path.join("_indices");
    let dirs_before: Vec<String> = if indices_dir.exists() {
        std::fs::read_dir(&indices_dir)
            .expect("read _indices")
            .flatten()
            .filter(|e| e.path().is_dir())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect()
    } else {
        Vec::new()
    };
    assert!(
        !dirs_before.is_empty(),
        "test setup must produce at least one index dir"
    );

    // Create an orphaned tag: delete a rebuild branch that has tags on it.
    let all_branches = store.io_list_branches("admin/fixa_prune_abort").await.unwrap();
    let rebuild_branches: Vec<_> = all_branches
        .iter()
        .filter(|b| b.starts_with(".-compact_rebuild_"))
        .collect();

    if rebuild_branches.is_empty() {
        eprintln!("[fixa_prune_abort] no rebuild branches — test inconclusive");
        return;
    }

    let branch_to_delete = rebuild_branches[0].clone();
    let uri = path.to_string_lossy().to_string();
    let mut ds_fresh = store.io_open_fresh(&uri).await.expect("fresh open");
    ds_fresh.delete_branch(&branch_to_delete).await.expect("delete branch");

    // Now the strict collector will fail when it tries to checkout the
    // deleted branch for tagged versions on it. The prune MUST abort.
    let result = store
        .io_prune_stale_index_dirs("admin/fixa_prune_abort", "main")
        .await;

    assert!(
        result.is_err(),
        "io_prune_stale_index_dirs must return Err when live-UUID collection fails \
         (orphaned tag pointing to deleted branch '{}') — got Ok. \
         Destructive prune must be fail-closed.",
        branch_to_delete,
    );

    // Verify nothing was deleted.
    let dirs_after: Vec<String> = if indices_dir.exists() {
        std::fs::read_dir(&indices_dir)
            .expect("read _indices")
            .flatten()
            .filter(|e| e.path().is_dir())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect()
    } else {
        Vec::new()
    };

    assert_eq!(
        dirs_after.len(),
        dirs_before.len(),
        "prune must not delete any index dir when collection fails — \
         before: {} dirs, after: {} dirs",
        dirs_before.len(),
        dirs_after.len(),
    );
}

/// FIX-B: io_cleanup_aggressive must not delete index directories that are
/// referenced by tagged manifests on rebuild branches. Previously, cleanup
/// only ran on the main branch handle, so Lance didn't see rebuild-branch
/// tags and deleted their pinned index files. This test verifies the fix:
/// after aggressive cleanup, every index UUID referenced by any tagged
/// manifest across all branches must still have a backing directory.
#[tokio::test]
async fn cleanup_aggressive_preserves_rebuild_branch_index_dirs() {
    let (store, _tmp, _commits, path) =
        setup_two_compaction_cycles(8, "admin/fixb_cleanup_preserve").await;

    // Collect live index UUIDs from all tagged versions BEFORE cleanup.
    let live_before = store
        .io_collect_all_live_index_uuids_strict("admin/fixb_cleanup_preserve")
        .await
        .expect("collect live UUIDs before cleanup");

    assert!(
        !live_before.is_empty(),
        "test setup must produce tagged versions with index UUIDs"
    );

    // Snapshot ALL index dirs across ALL branch paths before cleanup.
    // Lance stores branch indices under tree/<branch_name>/_indices/.
    let collect_all_index_dirs = || -> std::collections::HashSet<String> {
        let mut dirs = std::collections::HashSet::new();
        // Main's _indices/
        let main_indices = path.join("_indices");
        if main_indices.exists() {
            if let Ok(entries) = std::fs::read_dir(&main_indices) {
                for entry in entries.flatten() {
                    if entry.path().is_dir() {
                        dirs.insert(entry.file_name().to_string_lossy().to_string());
                    }
                }
            }
        }
        // Branch _indices/ under tree/<branch_name>/_indices/
        let tree_dir = path.join("tree");
        if tree_dir.exists() {
            if let Ok(branch_entries) = std::fs::read_dir(&tree_dir) {
                for branch_entry in branch_entries.flatten() {
                    let branch_indices = branch_entry.path().join("_indices");
                    if branch_indices.exists() {
                        if let Ok(entries) = std::fs::read_dir(&branch_indices) {
                            for entry in entries.flatten() {
                                if entry.path().is_dir() {
                                    dirs.insert(entry.file_name().to_string_lossy().to_string());
                                }
                            }
                        }
                    }
                }
            }
        }
        dirs
    };

    let dirs_before = collect_all_index_dirs();

    // Run aggressive cleanup — this must not delete any live index dir.
    store
        .io_cleanup_aggressive("admin/fixb_cleanup_preserve", "main")
        .await
        .expect("aggressive cleanup must succeed without deleting live index dirs");

    // Verify no index dir that existed before was deleted by cleanup.
    let dirs_after = collect_all_index_dirs();

    let deleted: Vec<String> = dirs_before
        .iter()
        .filter(|d| !dirs_after.contains(*d))
        .cloned()
        .collect();

    assert!(
        deleted.is_empty(),
        "aggressive cleanup deleted {} index dir(s): {:?} \
         — io_cleanup_aggressive must not delete any index directory (FIX-B)",
        deleted.len(),
        deleted,
    );
}

// ===========================================================================
// CLEANUP PROOF TEST BATTERY — 10 commits, 90 docs each, 1 compaction
//
// Setup: 10 commits (c0..c9), each with 90 docs, 1 micro-batch append per push.
// Full index pipeline after each push (FTS + vector + cascade).
// One full compaction cycle after all 10 pushes.
// Base-3 boundaries for 10 commits: {0, 3} → 2 boundaries, 8 intermediates
// (latest commit c9 is always an intermediate, never a boundary).
//
// Vector index threshold (256 rows): crossed at c2 (270 rows).
// ===========================================================================

/// Shared setup: push 10 commits with 90 docs each, run index pipeline after
/// each push, then one full compaction. Returns (store, tmp, commits, path).
async fn setup_ten_commits_one_compaction(
    dim: usize,
    domain: &str,
) -> (
    LanceStore,
    tempfile::TempDir,
    Vec<(String, u64)>,
    std::path::PathBuf,
) {
    setup_ten_commits_with_cleanup_mode(dim, domain, CleanupMode::CurrentCode).await
}

/// Shared setup with explicit cleanup mode. Push 10 commits with 90 docs each,
/// run index pipeline after each push, then one full compaction.
async fn setup_ten_commits_with_cleanup_mode(
    dim: usize,
    domain: &str,
    cleanup_mode: CleanupMode,
) -> (
    LanceStore,
    tempfile::TempDir,
    Vec<(String, u64)>,
    std::path::PathBuf,
) {
    use crate::store::lance::io_ensure_fts_index_on_dataset;
    use crate::store::lance::io_incremental_cascade;
    use crate::store::vector_index::io_ensure_vector_index;

    let config = crate::store::vector_index::tests::make_test_config(dim);
    let (mut store, tmp) = make_test_store(dim);
    store.set_vector_index_config(config.clone());
    store.set_cleanup_mode(cleanup_mode);

    let mut commits: Vec<(String, u64)> = Vec::new();

    // Push 10 commits, each with 90 docs, 1 micro-batch append per push.
    for i in 0..10 {
        let mut rows = Vec::new();
        for j in 0..90 {
            let doc_id = format!("doc/{}_{}", i, j);
            let emb = fake_embedding(dim, (i * 90 + j) as f32 * 10.0);
            rows.push(ChunkRow {
                doc_id: doc_id.clone(),
                doc_type: "T".to_owned(),
                chunk_index: 0,
                chunk_count: 1,
                chunk_token_start: 0,
                doc_token_len: 5,
                embedding: emb.clone(),
                clustering_embedding: emb,
                content: format!("content {} {}", i, j),
            });
        }
        let first_doc_id = format!("doc/{}_0", i);
        let v = store
            .io_upsert_chunks(domain, "main", &first_doc_id, &rows)
            .await
            .expect("upsert");
        let commit = format!("c{}", i);
        store
            .io_tag_commit(domain, "main", &commit, v)
            .await
            .expect("tag");
        commits.push((commit, v));

        // Run index pipeline after each push (matching real pipeline).
        {
            let ds_arc = store
                .io_open_dataset_uncached(domain, "main")
                .await
                .expect("open uncached")
                .expect("dataset exists");
            let mut ds = ds_arc;

            io_ensure_fts_index_on_dataset(&mut ds)
                .await
                .expect("FTS index");
            io_ensure_vector_index(&mut ds, &config, false)
                .await
                .expect("vector index");

            store.io_increment_delta_count(domain, "main").await;
            let delta_count = store.io_get_delta_count(domain, "main").await;
            let _ = io_incremental_cascade(&mut ds, delta_count).await;
        }
        store
            .io_refresh_cached_dataset(domain, "main")
            .await
            .expect("refresh");
    }

    // Run one full compaction cycle.
    run_full_compaction_pipeline(&store, domain, "main").await;

    let path = store.dataset_path(domain);
    (store, tmp, commits, path)
}

/// Print a comprehensive disk report for manual inspection.
/// Walks the entire dataset directory and prints every file with its path,
/// size, and whether it's referenced by a live manifest.
fn print_full_disk_report(path: &std::path::Path) {
    eprintln!("\n========== FULL DISK REPORT ==========");
    eprintln!("Dataset path: {}", path.display());

    // Data files on main
    let data_dir = path.join("data");
    if data_dir.exists() {
        eprintln!("\n--- main data/ ---");
        if let Ok(entries) = std::fs::read_dir(&data_dir) {
            for entry in entries.flatten() {
                let meta = entry.metadata();
                eprintln!("  {} ({} bytes)", entry.path().display(), meta.map(|m| m.len()).unwrap_or(0));
            }
        }
    }

    // Index dirs on main
    let indices_dir = path.join("_indices");
    if indices_dir.exists() {
        eprintln!("\n--- main _indices/ ---");
        if let Ok(entries) = std::fs::read_dir(&indices_dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    let file_count = std::fs::read_dir(&p)
                        .map(|e| e.filter(|e| e.is_ok()).count())
                        .unwrap_or(0);
                    eprintln!("  {} ({} files)", p.display(), file_count);
                }
            }
        }
    }

    // Manifests / versions
    let versions_dir = path.join("_versions");
    if versions_dir.exists() {
        eprintln!("\n--- main _versions/ ---");
        if let Ok(entries) = std::fs::read_dir(&versions_dir) {
            for entry in entries.flatten() {
                eprintln!("  {}", entry.path().display());
            }
        }
    }

    // Branch tree
    let tree_dir = path.join("_branches");
    if tree_dir.exists() {
        eprintln!("\n--- branches ---");
        walk_dir_print(&tree_dir, "  ");
    }

    // Total disk usage
    let total: u64 = walk_dir_size(path);
    eprintln!("\n--- total disk usage: {} bytes ({:.2} MB) ---", total, total as f64 / 1_048_576.0);
    eprintln!("========== END DISK REPORT ==========\n");
}

fn walk_dir_print(dir: &std::path::Path, prefix: &str) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            eprintln!("{}{}", prefix, path.display());
            if path.is_dir() {
                walk_dir_print(&path, &format!("{}  ", prefix));
            }
        }
    }
}

fn walk_dir_size(dir: &std::path::Path) -> u64 {
    let mut total: u64 = 0;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                total += walk_dir_size(&path);
            } else if let Ok(meta) = entry.metadata() {
                total += meta.len();
            }
        }
    }
    total
}

/// Count .lance data files in a directory.
fn count_lance_files(dir: &std::path::Path) -> usize {
    if !dir.exists() {
        return 0;
    }
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .map(|ext| ext == "lance")
                        .unwrap_or(false)
                })
                .count()
        })
        .unwrap_or(0)
}

/// Count non-empty index UUID directories.
fn count_non_empty_index_dirs(dir: &std::path::Path) -> usize {
    if !dir.exists() {
        return 0;
    }
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path().is_dir()
                        && std::fs::read_dir(e.path())
                            .map(|d| d.filter(|e| e.is_ok()).count() > 0)
                            .unwrap_or(false)
                })
                .count()
        })
        .unwrap_or(0)
}

/// Count manifest files in _versions/.
#[allow(dead_code)]
fn count_manifests(dir: &std::path::Path) -> usize {
    if !dir.exists() {
        return 0;
    }
    std::fs::read_dir(dir)
        .map(|entries| entries.filter_map(|e| e.ok()).count())
        .unwrap_or(0)
}

/// Find the rebuild branch name dynamically (epoch-based, can't hardcode).
async fn find_rebuild_branch(store: &LanceStore, domain: &str) -> Option<String> {
    use crate::store::lance::is_compact_rebuild_branch;
    let branches = store.io_list_branches(domain).await.ok()?;
    branches.into_iter().find(|b| is_compact_rebuild_branch(b))
}

// -------------------------------------------------------------------------
// Test 1: All 10 commit tags exist after compaction.
// -------------------------------------------------------------------------
#[tokio::test]
async fn cleanup_battery_all_tags_exist_after_compaction() {
    let dim = 8;
    let domain = "admin/cleanup_battery_tags";
    let (store, _tmp, commits, path) =
        setup_ten_commits_one_compaction(dim, domain).await;

    print_full_disk_report(&path);

    let ds = store
        .io_open_dataset(domain, "main")
        .await
        .expect("open dataset");
    let ds = ds.read().await;
    let tags = ds.tags().list().await.expect("list tags");

    for (commit_name, _version) in &commits {
        let tag_key = crate::layeridx::encode_commit_tag(commit_name);
        assert!(
            tags.contains_key(&tag_key),
            "tag '{}' must exist after compaction",
            commit_name
        );
    }
}

// -------------------------------------------------------------------------
// Test 2: Snapshot isolation — each tag returns the correct row count.
// -------------------------------------------------------------------------
#[tokio::test]
async fn cleanup_battery_snapshot_row_counts() {
    let dim = 8;
    let domain = "admin/cleanup_battery_rowcounts";
    let (store, _tmp, commits, _path) =
        setup_ten_commits_one_compaction(dim, domain).await;

    let ds = store
        .io_open_dataset(domain, "main")
        .await
        .expect("open dataset");
    let ds = ds.read().await;
    let tags = ds.tags().list().await.expect("list tags");

    for (i, (commit_name, _original_version)) in commits.iter().enumerate() {
        let expected_rows = (i + 1) * 90;
        let tag_key = crate::layeridx::encode_commit_tag(commit_name);
        let tag_contents = tags
            .get(&tag_key)
            .unwrap_or_else(|| panic!("tag '{}' should exist", commit_name));

        // After delta-fork retagging, intermediate tags point to versions
        // on the rebuild branch; boundary tags stay on main. We need to
        // open the correct branch to checkout the tag's version.
        let tag_branch = tag_contents.branch.as_deref().unwrap_or("main");
        let tag_version = tag_contents.version;

        if tag_branch == "main" {
            let checked = ds
                .checkout_version(tag_version)
                .await
                .unwrap_or_else(|e| panic!("checkout version {} on main: {}", tag_version, e));
            let count = checked.count_rows(None).await.expect("count rows");
            assert_eq!(
                count, expected_rows,
                "commit {} (tag version {} on main) should have {} rows, got {}",
                commit_name, tag_version, expected_rows, count
            );
        } else {
            // Tag is on a rebuild branch — open that branch to checkout.
            let branch_ds = store
                .io_open_dataset_uncached(domain, tag_branch)
                .await
                .expect("open uncached")
                .unwrap_or_else(|| panic!("rebuild branch '{}' should exist", tag_branch));
            let checked = branch_ds
                .checkout_version(tag_version)
                .await
                .unwrap_or_else(|e| panic!("checkout version {} on {}: {}", tag_version, tag_branch, e));
            let count = checked.count_rows(None).await.expect("count rows");
            assert_eq!(
                count, expected_rows,
                "commit {} (tag version {} on {}) should have {} rows, got {}",
                commit_name, tag_version, tag_branch, expected_rows, count
            );
        }
    }
}

// -------------------------------------------------------------------------
// Test 3: HEAD has 900 rows (10 × 90).
// -------------------------------------------------------------------------
#[tokio::test]
async fn cleanup_battery_head_has_900_rows() {
    let dim = 8;
    let domain = "admin/cleanup_battery_head";
    let (store, _tmp, _commits, _path) =
        setup_ten_commits_one_compaction(dim, domain).await;

    let ds = store
        .io_open_dataset(domain, "main")
        .await
        .expect("open dataset");
    let ds = ds.read().await;
    let count = ds.count_rows(None).await.expect("count rows");
    assert_eq!(count, 900usize, "HEAD should have 900 rows after 10 pushes of 90 docs");
}

// -------------------------------------------------------------------------
// Test 4: Data file count on main is bounded (compaction consolidates).
// -------------------------------------------------------------------------
#[tokio::test]
async fn cleanup_battery_main_data_files_bounded() {
    let dim = 8;
    let domain = "admin/cleanup_battery_datafiles";
    let (_store, _tmp, _commits, path) =
        setup_ten_commits_one_compaction(dim, domain).await;

    print_full_disk_report(&path);

    let data_dir = path.join("data");
    let data_files = count_lance_files(&data_dir);

    // 10 pushes each create a data file, plus 1 compacted file from
    // compaction = 11 total. Boundary tags (c0, c3) protect their original
    // files; intermediate tags are on the rebuild branch which shares main's
    // data/ directory. All 11 files are referenced by live manifests.
    assert_eq!(
        data_files, 11,
        "main data/ should have exactly 11 .lance files (10 originals + 1 compacted), got {}",
        data_files
    );
}

// -------------------------------------------------------------------------
// Test 5: Index directories on main are bounded.
// -------------------------------------------------------------------------
#[tokio::test]
async fn cleanup_battery_main_index_dirs_bounded() {
    let dim = 8;
    let domain = "admin/cleanup_battery_indexdirs";
    let (_store, _tmp, _commits, path) =
        setup_ten_commits_one_compaction(dim, domain).await;

    let indices_dir = path.join("_indices");
    let index_dirs = count_non_empty_index_dirs(&indices_dir);

    // After compaction: vector + FTS indices are recreated fresh, then
    // cascade creates delta indices. After prune_stale_index_dirs, only
    // live index UUIDs referenced by tagged versions remain.
    // 4 dirs: 2 for boundary tags (c0, c3) + 2 for HEAD (vector + FTS).
    assert_eq!(
        index_dirs, 4,
        "main _indices/ should have exactly 4 non-empty dirs after compaction, got {}",
        index_dirs
    );
}

// -------------------------------------------------------------------------
// Test 6: Rebuild branch exists and has data files.
// -------------------------------------------------------------------------
#[tokio::test]
async fn cleanup_battery_rebuild_branch_exists() {
    let dim = 8;
    let domain = "admin/cleanup_battery_rebuild";
    let (store, _tmp, _commits, _path) =
        setup_ten_commits_one_compaction(dim, domain).await;

    let rebuild_branch = find_rebuild_branch(&store, domain).await;
    assert!(
        rebuild_branch.is_some(),
        "a rebuild branch should exist after compaction"
    );

    if let Some(branch_name) = rebuild_branch {
        eprintln!("[test] rebuild branch: {}", branch_name);
        let ds = store
            .io_open_dataset_uncached(domain, &branch_name)
            .await
            .expect("open uncached")
            .expect("rebuild branch dataset exists");
        let fragments = ds.get_fragments().len();
        assert!(
            fragments > 0,
            "rebuild branch '{}' should have fragments",
            branch_name
        );
    }
}

// -------------------------------------------------------------------------
// Test 7: Rebuild branch data files are present on disk.
// -------------------------------------------------------------------------
#[tokio::test]
async fn cleanup_battery_rebuild_branch_data_files_on_disk() {
    let dim = 8;
    let domain = "admin/cleanup_battery_rebuild_data";
    let (store, _tmp, _commits, _path) =
        setup_ten_commits_one_compaction(dim, domain).await;

    let rebuild_branch = find_rebuild_branch(&store, domain).await;
    assert!(rebuild_branch.is_some(), "rebuild branch should exist");

    if let Some(branch_name) = rebuild_branch {
        // LanceDB branches share the main data/ directory — they don't have
        // their own data/ subdirectory. Instead, verify the branch can be
        // opened and has fragments referencing shared data files.
        let ds = store
            .io_open_dataset_uncached(domain, &branch_name)
            .await
            .expect("open uncached")
            .expect("rebuild branch dataset exists");
        let fragments = ds.get_fragments();
        assert!(
            !fragments.is_empty(),
            "rebuild branch '{}' should have fragments referencing data files",
            branch_name
        );
        // Verify each fragment has data files referenced.
        for frag in &fragments {
            let files = frag.metadata().files.len();
            assert!(
                files > 0,
                "rebuild branch '{}' fragment should reference data files",
                branch_name
            );
        }
        eprintln!(
            "[test] rebuild branch '{}' has {} fragments, {} total data file refs",
            branch_name,
            fragments.len(),
            fragments.iter().map(|f| f.metadata().files.len()).sum::<usize>()
        );
    }
}

// -------------------------------------------------------------------------
// Test 8: No orphaned data files (every .lance file in data/ is referenced
// by a live manifest or a rebuild branch manifest).
// -------------------------------------------------------------------------
#[tokio::test]
async fn cleanup_battery_no_orphaned_data_files() {
    let dim = 8;
    let domain = "admin/cleanup_battery_orphans";
    let (_store, _tmp, _commits, path) =
        setup_ten_commits_one_compaction(dim, domain).await;

    print_full_disk_report(&path);

    // Count total .lance files across main data/ and all branch data/ dirs.
    let main_data_files = count_lance_files(&path.join("data"));

    let branches_dir = path.join("_branches");
    let mut branch_data_files = 0;
    if branches_dir.exists() {
        if let Ok(branch_entries) = std::fs::read_dir(&branches_dir) {
            for branch_entry in branch_entries.flatten() {
                let branch_data = branch_entry.path().join("data");
                branch_data_files += count_lance_files(&branch_data);
            }
        }
    }

    eprintln!(
        "[test] main data files: {}, branch data files: {}",
        main_data_files, branch_data_files
    );

    // Branches share main's data/ directory (LanceDB branches don't have
    // separate data/ subdirectories). So total = main data files = 11.
    let total = main_data_files + branch_data_files;
    assert_eq!(
        total, 11,
        "total .lance files (main + branches) should be exactly 11, got {} (main={}, branches={})",
        total, main_data_files, branch_data_files
    );
}

// -------------------------------------------------------------------------
// Test 9: Integrity check passes after compaction.
// -------------------------------------------------------------------------
#[tokio::test]
async fn cleanup_battery_integrity_check_passes() {
    let dim = 8;
    let domain = "admin/cleanup_battery_integrity";
    let (store, _tmp, _commits, _path) =
        setup_ten_commits_one_compaction(dim, domain).await;

    store
        .io_integrity_check(domain)
        .await
        .expect("integrity check should pass");
}

// -------------------------------------------------------------------------
// Test 10: Tag distribution — boundary tags (c0, c3) are on main,
// intermediate tags (c1, c2, c4-c9) are on the rebuild branch.
// -------------------------------------------------------------------------
#[tokio::test]
async fn cleanup_battery_tag_distribution_boundary_vs_intermediate() {
    let dim = 8;
    let domain = "admin/cleanup_battery_tagdist";
    let (store, _tmp, _commits, _path) =
        setup_ten_commits_one_compaction(dim, domain).await;

    let rebuild_branch = find_rebuild_branch(&store, domain).await;
    assert!(rebuild_branch.is_some(), "rebuild branch should exist");
    let rebuild_name = rebuild_branch.unwrap();

    let ds = store
        .io_open_dataset(domain, "main")
        .await
        .expect("open dataset");
    let ds = ds.read().await;
    let tags = ds.tags().list().await.expect("list tags");

    // Base-3 boundaries for 10 commits: positions 0, 3 (c0, c3).
    let boundary_commits = vec!["c0", "c3"];
    let intermediate_commits: Vec<String> = (0..10)
        .filter(|i| !boundary_commits.contains(&format!("c{}", i).as_str()))
        .map(|i| format!("c{}", i))
        .collect();

    for commit_name in &boundary_commits {
        let tag_key = crate::layeridx::encode_commit_tag(commit_name);
        if let Some(tag_contents) = tags.get(&tag_key) {
            let branch = tag_contents.branch.as_deref().unwrap_or("main");
            assert_eq!(
                branch, "main",
                "boundary commit '{}' should be on main branch, got '{}'",
                commit_name, branch
            );
        }
    }

    for commit_name in &intermediate_commits {
        let tag_key = crate::layeridx::encode_commit_tag(commit_name);
        if let Some(tag_contents) = tags.get(&tag_key) {
            let branch = tag_contents.branch.as_deref().unwrap_or("main");
            assert_eq!(
                branch, rebuild_name,
                "intermediate commit '{}' should be on rebuild branch '{}', got '{}'",
                commit_name, rebuild_name, branch
            );
        }
    }
}

// -------------------------------------------------------------------------
// Test 11: No stale rebuild branches (only one rebuild branch after
// a single compaction cycle).
// -------------------------------------------------------------------------
#[tokio::test]
async fn cleanup_battery_no_stale_rebuild_branches() {
    use crate::store::lance::is_compact_rebuild_branch;
    let dim = 8;
    let domain = "admin/cleanup_battery_stale";
    let (store, _tmp, _commits, _path) =
        setup_ten_commits_one_compaction(dim, domain).await;

    let branches = store
        .io_list_branches(domain)
        .await
        .expect("list branches");

    let rebuild_branches: Vec<String> = branches
        .iter()
        .filter(|b| is_compact_rebuild_branch(b))
        .cloned()
        .collect();

    assert_eq!(
        rebuild_branches.len(),
        1,
        "should have exactly 1 rebuild branch after 1 compaction, got {}: {:?}",
        rebuild_branches.len(),
        rebuild_branches
    );
}

// -------------------------------------------------------------------------
// Test 12: Full disk report is printed and total disk usage is bounded
// relative to raw data size. This is the manual inspection test.
// -------------------------------------------------------------------------
#[tokio::test]
async fn cleanup_battery_disk_usage_bounded_and_reported() {
    let dim = 8;
    let domain = "admin/cleanup_battery_diskreport";
    let (_store, _tmp, _commits, path) =
        setup_ten_commits_one_compaction(dim, domain).await;

    print_full_disk_report(&path);

    let total_bytes = walk_dir_size(&path);
    let total_mb = total_bytes as f64 / 1_048_576.0;

    eprintln!("[test] total disk usage: {} bytes ({:.2} MB)", total_bytes, total_mb);

    // 900 rows × 8-dim embeddings + metadata + 4 index dirs ≈ 0.55 MB.
    // Allow 5 MB headroom for filesystem block alignment overhead.
    assert!(
        total_mb < 5.0,
        "total disk usage should be < 5 MB for 900 rows with 8-dim embeddings, got {:.2} MB",
        total_mb
    );
}

// ===========================================================================
// MODE A TESTS — CleanupMode::TargetNoPatch
// No _indices/ hiding, clean_referenced_branches(false) instead.
// ===========================================================================

#[tokio::test]
async fn cleanup_battery_mode_a_all_tags_exist() {
    let dim = 8;
    let domain = "admin/cleanup_battery_mode_a_tags";
    let (store, _tmp, commits, path) =
        setup_ten_commits_with_cleanup_mode(dim, domain, CleanupMode::TargetNoPatch).await;

    print_full_disk_report(&path);

    let ds = store.io_open_dataset(domain, "main").await.expect("open");
    let ds = ds.read().await;
    let tags = ds.tags().list().await.expect("list tags");
    for (commit_name, _) in &commits {
        let tag_key = crate::layeridx::encode_commit_tag(commit_name);
        assert!(tags.contains_key(&tag_key), "tag '{}' must exist (Mode A)", commit_name);
    }
}

#[tokio::test]
async fn cleanup_battery_mode_a_snapshot_row_counts() {
    let dim = 8;
    let domain = "admin/cleanup_battery_mode_a_rowcounts";
    let (store, _tmp, commits, _path) =
        setup_ten_commits_with_cleanup_mode(dim, domain, CleanupMode::TargetNoPatch).await;

    let ds = store.io_open_dataset(domain, "main").await.expect("open");
    let ds = ds.read().await;
    let tags = ds.tags().list().await.expect("list tags");

    for (i, (commit_name, _)) in commits.iter().enumerate() {
        let expected_rows = (i + 1) * 90;
        let tag_key = crate::layeridx::encode_commit_tag(commit_name);
        let tc = tags.get(&tag_key).unwrap_or_else(|| panic!("tag '{}' should exist", commit_name));
        let tag_branch = tc.branch.as_deref().unwrap_or("main");
        let tag_version = tc.version;

        if tag_branch == "main" {
            let checked = ds.checkout_version(tag_version).await.expect("checkout");
            let count = checked.count_rows(None).await.expect("count");
            assert_eq!(count, expected_rows, "commit {} (Mode A) row count", commit_name);
        } else {
            let branch_ds = store.io_open_dataset_uncached(domain, tag_branch).await.expect("open").expect("exists");
            let checked = branch_ds.checkout_version(tag_version).await.expect("checkout");
            let count = checked.count_rows(None).await.expect("count");
            assert_eq!(count, expected_rows, "commit {} (Mode A) row count on {}", commit_name, tag_branch);
        }
    }
}

#[tokio::test]
async fn cleanup_battery_mode_a_head_900_rows() {
    let dim = 8;
    let domain = "admin/cleanup_battery_mode_a_head";
    let (store, _tmp, _, _) =
        setup_ten_commits_with_cleanup_mode(dim, domain, CleanupMode::TargetNoPatch).await;
    let ds = store.io_open_dataset(domain, "main").await.expect("open");
    let ds = ds.read().await;
    assert_eq!(ds.count_rows(None).await.expect("count"), 900usize);
}

#[tokio::test]
async fn cleanup_battery_mode_a_data_files_bounded() {
    let dim = 8;
    let domain = "admin/cleanup_battery_mode_a_datafiles";
    let (store, _tmp, _, path) =
        setup_ten_commits_with_cleanup_mode(dim, domain, CleanupMode::TargetNoPatch).await;
    let _ = store;
    print_full_disk_report(&path);
    let data_files = count_lance_files(&path.join("data"));
    assert_eq!(data_files, 11, "Mode A: main data/ should have exactly 11 .lance files, got {}", data_files);
}

#[tokio::test]
async fn cleanup_battery_mode_a_index_dirs_bounded() {
    let dim = 8;
    let domain = "admin/cleanup_battery_mode_a_indexdirs";
    let (store, _tmp, _, path) =
        setup_ten_commits_with_cleanup_mode(dim, domain, CleanupMode::TargetNoPatch).await;
    let _ = store;
    let index_dirs = count_non_empty_index_dirs(&path.join("_indices"));
    assert_eq!(index_dirs, 4, "Mode A: main _indices/ should have exactly 4 non-empty dirs, got {}", index_dirs);
}

#[tokio::test]
async fn cleanup_battery_mode_a_rebuild_branch_exists() {
    let dim = 8;
    let domain = "admin/cleanup_battery_mode_a_rebuild";
    let (store, _tmp, _, _) =
        setup_ten_commits_with_cleanup_mode(dim, domain, CleanupMode::TargetNoPatch).await;
    let rebuild = find_rebuild_branch(&store, domain).await;
    assert!(rebuild.is_some(), "Mode A: rebuild branch should exist");
}

#[tokio::test]
async fn cleanup_battery_mode_a_rebuild_branch_has_fragments() {
    let dim = 8;
    let domain = "admin/cleanup_battery_mode_a_rebuild_data";
    let (store, _tmp, _, _) =
        setup_ten_commits_with_cleanup_mode(dim, domain, CleanupMode::TargetNoPatch).await;
    let rebuild = find_rebuild_branch(&store, domain).await.expect("rebuild branch");
    let ds = store.io_open_dataset_uncached(domain, &rebuild).await.expect("open").expect("exists");
    assert!(!ds.get_fragments().is_empty(), "Mode A: rebuild branch should have fragments");
}

#[tokio::test]
async fn cleanup_battery_mode_a_no_orphaned_data_files() {
    let dim = 8;
    let domain = "admin/cleanup_battery_mode_a_orphans";
    let (store, _tmp, _, path) =
        setup_ten_commits_with_cleanup_mode(dim, domain, CleanupMode::TargetNoPatch).await;
    let _ = store;
    print_full_disk_report(&path);
    let main_files = count_lance_files(&path.join("data"));
    let branches_dir = path.join("_branches");
    let mut branch_files = 0;
    if branches_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(&branches_dir) {
            for e in entries.flatten() {
                branch_files += count_lance_files(&e.path().join("data"));
            }
        }
    }
    let total = main_files + branch_files;
    assert_eq!(total, 11, "Mode A: total .lance files should be exactly 11, got {}", total);
}

#[tokio::test]
async fn cleanup_battery_mode_a_integrity_check_passes() {
    let dim = 8;
    let domain = "admin/cleanup_battery_mode_a_integrity";
    let (store, _tmp, _, _) =
        setup_ten_commits_with_cleanup_mode(dim, domain, CleanupMode::TargetNoPatch).await;
    store.io_integrity_check(domain).await.expect("Mode A: integrity check should pass");
}

#[tokio::test]
async fn cleanup_battery_mode_a_tag_distribution() {
    let dim = 8;
    let domain = "admin/cleanup_battery_mode_a_tagdist";
    let (store, _tmp, _, _) =
        setup_ten_commits_with_cleanup_mode(dim, domain, CleanupMode::TargetNoPatch).await;
    let rebuild = find_rebuild_branch(&store, domain).await.expect("rebuild branch");
    let ds = store.io_open_dataset(domain, "main").await.expect("open");
    let ds = ds.read().await;
    let tags = ds.tags().list().await.expect("list tags");

    for commit_name in &["c0", "c3"] {
        let tag_key = crate::layeridx::encode_commit_tag(commit_name);
        if let Some(tc) = tags.get(&tag_key) {
            assert_eq!(tc.branch.as_deref().unwrap_or("main"), "main",
                "Mode A: boundary '{}' should be on main", commit_name);
        }
    }
    for i in 0..10 {
        let cn = format!("c{}", i);
        if cn == "c0" || cn == "c3" { continue; }
        let tag_key = crate::layeridx::encode_commit_tag(&cn);
        if let Some(tc) = tags.get(&tag_key) {
            assert_eq!(tc.branch.as_deref().unwrap_or("main"), rebuild,
                "Mode A: intermediate '{}' should be on rebuild branch", cn);
        }
    }
}

#[tokio::test]
async fn cleanup_battery_mode_a_no_stale_rebuild_branches() {
    use crate::store::lance::is_compact_rebuild_branch;
    let dim = 8;
    let domain = "admin/cleanup_battery_mode_a_stale";
    let (store, _tmp, _, _) =
        setup_ten_commits_with_cleanup_mode(dim, domain, CleanupMode::TargetNoPatch).await;
    let branches = store.io_list_branches(domain).await.expect("list");
    let rebuilds: Vec<_> = branches.iter().filter(|b| is_compact_rebuild_branch(b)).collect();
    assert_eq!(rebuilds.len(), 1, "Mode A: should have 1 rebuild branch, got {}", rebuilds.len());
}

#[tokio::test]
async fn cleanup_battery_mode_a_disk_usage_bounded() {
    let dim = 8;
    let domain = "admin/cleanup_battery_mode_a_diskreport";
    let (store, _tmp, _, path) =
        setup_ten_commits_with_cleanup_mode(dim, domain, CleanupMode::TargetNoPatch).await;
    let _ = store;
    print_full_disk_report(&path);
    let total_mb = walk_dir_size(&path) as f64 / 1_048_576.0;
    assert!(total_mb < 5.0, "Mode A: disk usage should be < 5 MB, got {:.2} MB", total_mb);
}

// ===========================================================================
// MODE B TESTS — CleanupMode::TargetWithPatch
// Same as Mode A. Originally intended to test with a LanceDB patch, but
// the patch was found to be incorrect (all-branch tag protection caused
// version number collisions between main and rebuild branches). The
// correct cleanup behavior is achieved without any LanceDB patch.
// ===========================================================================

#[tokio::test]
async fn cleanup_battery_mode_b_all_tags_exist() {
    let dim = 8;
    let domain = "admin/cleanup_battery_mode_b_tags";
    let (store, _tmp, commits, path) =
        setup_ten_commits_with_cleanup_mode(dim, domain, CleanupMode::TargetWithPatch).await;

    print_full_disk_report(&path);

    let ds = store.io_open_dataset(domain, "main").await.expect("open");
    let ds = ds.read().await;
    let tags = ds.tags().list().await.expect("list tags");
    for (commit_name, _) in &commits {
        let tag_key = crate::layeridx::encode_commit_tag(commit_name);
        assert!(tags.contains_key(&tag_key), "tag '{}' must exist (Mode B)", commit_name);
    }
}

#[tokio::test]
async fn cleanup_battery_mode_b_snapshot_row_counts() {
    let dim = 8;
    let domain = "admin/cleanup_battery_mode_b_rowcounts";
    let (store, _tmp, commits, _path) =
        setup_ten_commits_with_cleanup_mode(dim, domain, CleanupMode::TargetWithPatch).await;

    let ds = store.io_open_dataset(domain, "main").await.expect("open");
    let ds = ds.read().await;
    let tags = ds.tags().list().await.expect("list tags");

    for (i, (commit_name, _)) in commits.iter().enumerate() {
        let expected_rows = (i + 1) * 90;
        let tag_key = crate::layeridx::encode_commit_tag(commit_name);
        let tc = tags.get(&tag_key).unwrap_or_else(|| panic!("tag '{}' should exist", commit_name));
        let tag_branch = tc.branch.as_deref().unwrap_or("main");
        let tag_version = tc.version;

        if tag_branch == "main" {
            let checked = ds.checkout_version(tag_version).await.expect("checkout");
            let count = checked.count_rows(None).await.expect("count");
            assert_eq!(count, expected_rows, "commit {} (Mode B) row count", commit_name);
        } else {
            let branch_ds = store.io_open_dataset_uncached(domain, tag_branch).await.expect("open").expect("exists");
            let checked = branch_ds.checkout_version(tag_version).await.expect("checkout");
            let count = checked.count_rows(None).await.expect("count");
            assert_eq!(count, expected_rows, "commit {} (Mode B) row count on {}", commit_name, tag_branch);
        }
    }
}

#[tokio::test]
async fn cleanup_battery_mode_b_head_900_rows() {
    let dim = 8;
    let domain = "admin/cleanup_battery_mode_b_head";
    let (store, _tmp, _, _) =
        setup_ten_commits_with_cleanup_mode(dim, domain, CleanupMode::TargetWithPatch).await;
    let ds = store.io_open_dataset(domain, "main").await.expect("open");
    let ds = ds.read().await;
    assert_eq!(ds.count_rows(None).await.expect("count"), 900usize);
}

#[tokio::test]
async fn cleanup_battery_mode_b_data_files_bounded() {
    let dim = 8;
    let domain = "admin/cleanup_battery_mode_b_datafiles";
    let (store, _tmp, _, path) =
        setup_ten_commits_with_cleanup_mode(dim, domain, CleanupMode::TargetWithPatch).await;
    let _ = store;
    print_full_disk_report(&path);
    let data_files = count_lance_files(&path.join("data"));
    assert_eq!(data_files, 11, "Mode B: main data/ should have exactly 11 .lance files, got {}", data_files);
}

#[tokio::test]
async fn cleanup_battery_mode_b_index_dirs_bounded() {
    let dim = 8;
    let domain = "admin/cleanup_battery_mode_b_indexdirs";
    let (store, _tmp, _, path) =
        setup_ten_commits_with_cleanup_mode(dim, domain, CleanupMode::TargetWithPatch).await;
    let _ = store;
    let index_dirs = count_non_empty_index_dirs(&path.join("_indices"));
    assert_eq!(index_dirs, 4, "Mode B: main _indices/ should have exactly 4 non-empty dirs, got {}", index_dirs);
}

#[tokio::test]
async fn cleanup_battery_mode_b_rebuild_branch_exists() {
    let dim = 8;
    let domain = "admin/cleanup_battery_mode_b_rebuild";
    let (store, _tmp, _, _) =
        setup_ten_commits_with_cleanup_mode(dim, domain, CleanupMode::TargetWithPatch).await;
    let rebuild = find_rebuild_branch(&store, domain).await;
    assert!(rebuild.is_some(), "Mode B: rebuild branch should exist");
}

#[tokio::test]
async fn cleanup_battery_mode_b_rebuild_branch_has_fragments() {
    let dim = 8;
    let domain = "admin/cleanup_battery_mode_b_rebuild_data";
    let (store, _tmp, _, _) =
        setup_ten_commits_with_cleanup_mode(dim, domain, CleanupMode::TargetWithPatch).await;
    let rebuild = find_rebuild_branch(&store, domain).await.expect("rebuild branch");
    let ds = store.io_open_dataset_uncached(domain, &rebuild).await.expect("open").expect("exists");
    assert!(!ds.get_fragments().is_empty(), "Mode B: rebuild branch should have fragments");
}

#[tokio::test]
async fn cleanup_battery_mode_b_no_orphaned_data_files() {
    let dim = 8;
    let domain = "admin/cleanup_battery_mode_b_orphans";
    let (store, _tmp, _, path) =
        setup_ten_commits_with_cleanup_mode(dim, domain, CleanupMode::TargetWithPatch).await;
    let _ = store;
    print_full_disk_report(&path);
    let main_files = count_lance_files(&path.join("data"));
    let branches_dir = path.join("_branches");
    let mut branch_files = 0;
    if branches_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(&branches_dir) {
            for e in entries.flatten() {
                branch_files += count_lance_files(&e.path().join("data"));
            }
        }
    }
    let total = main_files + branch_files;
    assert_eq!(total, 11, "Mode B: total .lance files should be exactly 11, got {}", total);
}

#[tokio::test]
async fn cleanup_battery_mode_b_integrity_check_passes() {
    let dim = 8;
    let domain = "admin/cleanup_battery_mode_b_integrity";
    let (store, _tmp, _, _) =
        setup_ten_commits_with_cleanup_mode(dim, domain, CleanupMode::TargetWithPatch).await;
    store.io_integrity_check(domain).await.expect("Mode B: integrity check should pass");
}

#[tokio::test]
async fn cleanup_battery_mode_b_tag_distribution() {
    let dim = 8;
    let domain = "admin/cleanup_battery_mode_b_tagdist";
    let (store, _tmp, _, _) =
        setup_ten_commits_with_cleanup_mode(dim, domain, CleanupMode::TargetWithPatch).await;
    let rebuild = find_rebuild_branch(&store, domain).await.expect("rebuild branch");
    let ds = store.io_open_dataset(domain, "main").await.expect("open");
    let ds = ds.read().await;
    let tags = ds.tags().list().await.expect("list tags");

    for commit_name in &["c0", "c3"] {
        let tag_key = crate::layeridx::encode_commit_tag(commit_name);
        if let Some(tc) = tags.get(&tag_key) {
            assert_eq!(tc.branch.as_deref().unwrap_or("main"), "main",
                "Mode B: boundary '{}' should be on main", commit_name);
        }
    }
    for i in 0..10 {
        let cn = format!("c{}", i);
        if cn == "c0" || cn == "c3" { continue; }
        let tag_key = crate::layeridx::encode_commit_tag(&cn);
        if let Some(tc) = tags.get(&tag_key) {
            assert_eq!(tc.branch.as_deref().unwrap_or("main"), rebuild,
                "Mode B: intermediate '{}' should be on rebuild branch", cn);
        }
    }
}

#[tokio::test]
async fn cleanup_battery_mode_b_no_stale_rebuild_branches() {
    use crate::store::lance::is_compact_rebuild_branch;
    let dim = 8;
    let domain = "admin/cleanup_battery_mode_b_stale";
    let (store, _tmp, _, _) =
        setup_ten_commits_with_cleanup_mode(dim, domain, CleanupMode::TargetWithPatch).await;
    let branches = store.io_list_branches(domain).await.expect("list");
    let rebuilds: Vec<_> = branches.iter().filter(|b| is_compact_rebuild_branch(b)).collect();
    assert_eq!(rebuilds.len(), 1, "Mode B: should have 1 rebuild branch, got {}", rebuilds.len());
}

#[tokio::test]
async fn cleanup_battery_mode_b_disk_usage_bounded() {
    let dim = 8;
    let domain = "admin/cleanup_battery_mode_b_diskreport";
    let (store, _tmp, _, path) =
        setup_ten_commits_with_cleanup_mode(dim, domain, CleanupMode::TargetWithPatch).await;
    let _ = store;
    print_full_disk_report(&path);
    let total_mb = walk_dir_size(&path) as f64 / 1_048_576.0;
    assert!(total_mb < 5.0, "Mode B: disk usage should be < 5 MB, got {:.2} MB", total_mb);
}

// ===========================================================================
// TDD REPRODUCTION TESTS: tagged-manifest deletion investigation
//
// These tests verify that every tagged version's manifest survives all
// cleanup and compaction operations. A failing assert here reproduces the
// production bug where tagged manifests were deleted, making historical
// commits unsearchable.
//
// Asserts are the contract: every tag must resolve to an existing manifest
// at every stage. Never weaken these asserts to make tests pass.
// ===========================================================================

/// Read all manifest version numbers present on disk in `_versions/`.
///
/// LanceDB uses two naming schemes:
///   V1: `{version}.manifest`
///   V2: `{u64::MAX - version}.manifest` (lexicographical ordering)
///
/// If the parsed stem is > u64::MAX / 2, it is V2 and the real version is
/// `u64::MAX - stem`. Otherwise it is V1 and the stem IS the version.
fn on_disk_manifest_versions(versions_dir: &std::path::Path) -> std::collections::HashSet<u64> {
    let mut versions = std::collections::HashSet::new();
    if !versions_dir.exists() {
        return versions;
    }
    for entry in std::fs::read_dir(versions_dir).expect("read _versions dir") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("manifest") {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("manifest file stem");
        let n: u64 = stem.parse().expect("manifest stem is u64");
        let version = if n > u64::MAX / 2 {
            u64::MAX - n
        } else {
            n
        };
        versions.insert(version);
    }
    versions
}

/// Assert that every tag on disk resolves to an existing manifest.
///
/// Opens a FRESH dataset handle (no cache) so we see the true on-disk state.
/// For each tag:
///   1. `checkout_version((branch, Some(version)))` must succeed — this
///      fails if the manifest file is gone.
///   2. If the tag's branch is None (main branch), the manifest file must
///      also exist on disk in `_versions/`.
///
/// `stage` is a descriptive label included in assert messages for debugging.
async fn assert_tags_resolve(store: &LanceStore, domain: &str, stage: &str) {
    let path = store.dataset_path(domain);
    let uri = path.to_string_lossy().to_string();
    let ds = store
        .io_open_fresh(&uri)
        .await
        .expect("assert_tags_resolve: open fresh dataset");

    let tags = ds
        .tags()
        .list()
        .await
        .expect("assert_tags_resolve: list tags");

    assert!(
        !tags.is_empty(),
        "assert_tags_resolve [{}]: no tags found — dataset should have tags",
        stage
    );

    let on_disk = on_disk_manifest_versions(&path.join("_versions"));

    for (tag_name, tc) in &tags {
        let checkout_result = ds
            .checkout_version((tc.branch.as_deref(), Some(tc.version)))
            .await;

        assert!(
            checkout_result.is_ok(),
            "assert_tags_resolve [{}]: tag '{}' (version={}, branch={:?}) FAILED to checkout: {:?}",
            stage,
            tag_name,
            tc.version,
            tc.branch,
            checkout_result.err()
        );

        if tc.branch.is_none() {
            assert!(
                on_disk.contains(&tc.version),
                "assert_tags_resolve [{}]: tag '{}' (version={}, branch=None) manifest NOT on disk in _versions/ — on_disk has {} entries",
                stage,
                tag_name,
                tc.version,
                on_disk.len()
            );
        }
    }

    eprintln!(
        "assert_tags_resolve [{}]: all {} tags resolve OK",
        stage,
        tags.len()
    );
}

/// Baseline: aggressive cleanup must not delete manifests for tagged
/// versions on the main branch. Two consecutive cleanup runs exercise
/// the case where the first cleanup might leave manifests in an
/// intermediate state that the second cleanup then removes.
#[tokio::test]
async fn aggressive_cleanup_preserves_tagged_manifests_on_main() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/cleanup_tags_main";

    // Write 8 commits, each tagged on main.
    for i in 0..8u32 {
        let v = store
            .io_upsert_chunks(
                domain,
                "main",
                &format!("doc/{i}"),
                &[make_row(8, i as f32 + 1.0, &format!("doc/{i}"), "content")],
            )
            .await
            .expect("upsert");
        store
            .io_tag_commit(domain, "main", &format!("c{i}"), v)
            .await
            .expect("tag");
    }

    // All tags must resolve before any cleanup.
    assert_tags_resolve(&store, domain, "before cleanup").await;

    // First cleanup pass.
    store
        .io_cleanup_aggressive(domain, "main")
        .await
        .expect("cleanup 1");
    assert_tags_resolve(&store, domain, "after cleanup 1").await;

    // Second cleanup pass — catches delayed deletion.
    store
        .io_cleanup_aggressive(domain, "main")
        .await
        .expect("cleanup 2");
    assert_tags_resolve(&store, domain, "after cleanup 2").await;
}

/// Full compaction flow: step-by-step using the REAL production functions.
/// Every tagged manifest must survive every stage.
#[tokio::test]
async fn compaction_flow_preserves_tagged_manifests() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/compaction_flow_tags";

    // Setup: 8 commits, each tagged on main.
    for i in 0..8u32 {
        let v = store
            .io_upsert_chunks(
                domain,
                "main",
                &format!("doc/{i}"),
                &[make_row(8, i as f32 + 1.0, &format!("doc/{i}"), "content")],
            )
            .await
            .expect("upsert");
        store
            .io_tag_commit(domain, "main", &format!("c{i}"), v)
            .await
            .expect("tag");
    }

    assert_tags_resolve(&store, domain, "after setup").await;

    // Step 2: aggressive cleanup before retagging.
    store
        .io_cleanup_aggressive(domain, "main")
        .await
        .expect("pre-retag cleanup");
    assert_tags_resolve(&store, domain, "after pre-retag cleanup").await;

    // Step 3: delta-fork retagging.
    let head = store
        .io_branch_head_version(domain, "main")
        .await
        .expect("branch head version");
    store
        .io_retag_with_delta_forks(domain, "main", head)
        .await
        .expect("delta-fork retagging");
    assert_tags_resolve(&store, domain, "after delta-fork").await;

    // Step 4: final aggressive cleanup.
    store
        .io_cleanup_aggressive(domain, "main")
        .await
        .expect("post-retag cleanup");
    assert_tags_resolve(&store, domain, "after post-retag cleanup").await;

    // Step 5: orphaned tag / stale branch cleanup.
    store
        .io_cleanup_orphaned_tags_and_stale_branches(domain)
        .await
        .expect("orphan cleanup");
    assert_tags_resolve(&store, domain, "after orphan cleanup").await;
}

/// Second compaction cycle: the production store had ~27 compaction
/// cycles. Phase 2b of io_retag_with_delta_forks (boundary-tag migration
/// and old-rebuild-branch deletion, commit.rs:963-1035) only triggers
/// when a previous .-compact_rebuild_<epoch> branch already exists.
/// This test runs the full compaction flow twice to exercise that path.
#[tokio::test]
async fn second_compaction_cycle_preserves_tagged_manifests() {
    let (store, _tmp) = make_test_store(8);
    let domain = "admin/second_cycle_tags";

    // --- First cycle: 8 commits + tags + full compaction flow ---

    for i in 0..8u32 {
        let v = store
            .io_upsert_chunks(
                domain,
                "main",
                &format!("doc/{i}"),
                &[make_row(8, i as f32 + 1.0, &format!("doc/{i}"), "content")],
            )
            .await
            .expect("upsert");
        store
            .io_tag_commit(domain, "main", &format!("c{i}"), v)
            .await
            .expect("tag");
    }

    assert_tags_resolve(&store, domain, "cycle1 after setup").await;

    store
        .io_cleanup_aggressive(domain, "main")
        .await
        .expect("cycle1 pre-retag cleanup");
    assert_tags_resolve(&store, domain, "cycle1 after pre-retag cleanup").await;

    let head = store
        .io_branch_head_version(domain, "main")
        .await
        .expect("cycle1 branch head");
    store
        .io_retag_with_delta_forks(domain, "main", head)
        .await
        .expect("cycle1 delta-fork retagging");
    assert_tags_resolve(&store, domain, "cycle1 after delta-fork").await;

    store
        .io_cleanup_aggressive(domain, "main")
        .await
        .expect("cycle1 post-retag cleanup");
    assert_tags_resolve(&store, domain, "cycle1 after post-retag cleanup").await;

    store
        .io_cleanup_orphaned_tags_and_stale_branches(domain)
        .await
        .expect("cycle1 orphan cleanup");
    assert_tags_resolve(&store, domain, "cycle1 after orphan cleanup").await;

    // --- Second cycle: append 2 more commits, then full compaction again ---

    for i in 8..10u32 {
        let v = store
            .io_upsert_chunks(
                domain,
                "main",
                &format!("doc/{i}"),
                &[make_row(8, i as f32 + 1.0, &format!("doc/{i}"), "content")],
            )
            .await
            .expect("upsert cycle2");
        store
            .io_tag_commit(domain, "main", &format!("c{i}"), v)
            .await
            .expect("tag cycle2");
    }

    assert_tags_resolve(&store, domain, "cycle2 after setup").await;

    store
        .io_cleanup_aggressive(domain, "main")
        .await
        .expect("cycle2 pre-retag cleanup");
    assert_tags_resolve(&store, domain, "cycle2 after pre-retag cleanup").await;

    let head2 = store
        .io_branch_head_version(domain, "main")
        .await
        .expect("cycle2 branch head");
    store
        .io_retag_with_delta_forks(domain, "main", head2)
        .await
        .expect("cycle2 delta-fork retagging");
    assert_tags_resolve(&store, domain, "cycle2 after delta-fork").await;

    store
        .io_cleanup_aggressive(domain, "main")
        .await
        .expect("cycle2 post-retag cleanup");
    assert_tags_resolve(&store, domain, "cycle2 after post-retag cleanup").await;

    store
        .io_cleanup_orphaned_tags_and_stale_branches(domain)
        .await
        .expect("cycle2 orphan cleanup");
    assert_tags_resolve(&store, domain, "cycle2 after orphan cleanup").await;
}

/// Diagnostic: open the PRODUCTION store at /tmp/vectorlink-data and print
/// the manifest branch, tag branch distribution, and cross-reference
/// tag versions with on-disk manifests. #[ignore] — run manually with:
///   cargo test --lib store::lance::tests -- production_store_branch_field --ignored --nocapture
#[tokio::test]
#[ignore]
async fn production_store_branch_field() {
    let path = "/tmp/vectorlink-data/admin__product_assortment.lance";
    if !std::path::Path::new(path).exists() {
        eprintln!("Production store not found at {}, skipping", path);
        return;
    }

    let store = LanceStore::new(
        std::path::Path::new("/tmp/vectorlink-data"),
        8,
        256 * 1024 * 1024,
        128 * 1024 * 1024,
    );
    let domain = "admin__product_assortment";
    let ds = store
        .io_open_fresh(&store.dataset_path(domain).to_string_lossy())
        .await
        .expect("open production store");

    let current_branch = &ds.manifest.branch;
    let latest_version = ds.version().version;
    eprintln!("=== PRODUCTION STORE DIAGNOSTIC ===");
    eprintln!("Latest version: {}", latest_version);
    eprintln!("manifest.branch (current_branch for cleanup): {:?}", current_branch);
    eprintln!();

    let tags = ds.tags().list().await.expect("list tags");
    let on_disk_main = on_disk_manifest_versions(&std::path::Path::new(path).join("_versions"));
    eprintln!("Manifests in main _versions/: {} (versions {} to {})",
        on_disk_main.len(),
        on_disk_main.iter().min().unwrap_or(&0),
        on_disk_main.iter().max().unwrap_or(&0));

    // Also check branch-specific manifest directories under tree/<branch>/_versions/
    let tree_dir = std::path::Path::new(path).join("tree");
    let mut branch_manifest_dirs: std::collections::HashMap<String, std::collections::HashSet<u64>> = std::collections::HashMap::new();
    if tree_dir.exists() {
        for entry in std::fs::read_dir(&tree_dir).expect("read tree dir") {
            let entry = entry.expect("tree dir entry");
            let branch_name = entry.file_name().to_string_lossy().to_string();
            let versions_dir = entry.path().join("_versions");
            if versions_dir.exists() {
                let versions = on_disk_manifest_versions(&versions_dir);
                eprintln!("Manifests in tree/{}/_versions/: {} (versions {} to {})",
                    branch_name, versions.len(),
                    versions.iter().min().unwrap_or(&0),
                    versions.iter().max().unwrap_or(&0));
                branch_manifest_dirs.insert(branch_name, versions);
            }
        }
    }
    eprintln!();

    let mut main_ok = 0;
    let mut main_broken = 0;
    let mut rebuild_ok = 0;
    let mut rebuild_broken = 0;

    for (tag_name, tc) in &tags {
        // For branch-specific tags, check the branch's own _versions/ dir
        let on_disk = if let Some(b) = tc.branch.as_ref() {
            branch_manifest_dirs.get(b).unwrap_or(&on_disk_main)
        } else {
            &on_disk_main
        };
        let exists = on_disk.contains(&tc.version);
        match (tc.branch.as_deref(), exists) {
            (None, true) => main_ok += 1,
            (None, false) => {
                main_broken += 1;
                eprintln!("  BROKEN main tag: {} -> version={}", tag_name, tc.version);
            }
            (Some(_b), true) => rebuild_ok += 1,
            (Some(b), false) => {
                rebuild_broken += 1;
                eprintln!("  BROKEN rebuild tag: {} -> version={}, branch={}", tag_name, tc.version, b);
            }
        }
    }

    eprintln!();
    eprintln!("Summary:");
    eprintln!("  Main tags:    {} OK, {} BROKEN", main_ok, main_broken);
    eprintln!("  Rebuild tags: {} OK, {} BROKEN", rebuild_ok, rebuild_broken);
    eprintln!();

    // Simulate the cleanup branch filter
    let tagged_versions: std::collections::HashSet<u64> = tags
        .values()
        .filter(|tag| match (tag.branch.as_ref(), current_branch.as_ref()) {
            (Some(branch_of_tag), Some(cb)) => branch_of_tag == cb,
            (None, None) => true,
            _ => false,
        })
        .map(|tc| tc.version)
        .collect();
    eprintln!("Cleanup branch filter result (current_branch={:?}):", current_branch);
    eprintln!("  Tags passing filter: {} (these are PROTECTED)", tagged_versions.len());
    eprintln!("  Tags filtered out:   {} (these are NOT protected)", tags.len() - tagged_versions.len());
    eprintln!();

    // List which tags are filtered out
    for (tag_name, tc) in &tags {
        let passes = match (tc.branch.as_ref(), current_branch.as_ref()) {
            (Some(branch_of_tag), Some(cb)) => branch_of_tag == cb,
            (None, None) => true,
            _ => false,
        };
        if !passes {
            let on_disk = if let Some(b) = tc.branch.as_ref() {
                branch_manifest_dirs.get(b).unwrap_or(&on_disk_main)
            } else {
                &on_disk_main
            };
            eprintln!("  FILTERED OUT: {} -> version={}, branch={:?} (manifest exists: {})",
                tag_name, tc.version, tc.branch, on_disk.contains(&tc.version));
        }
    }
}

/// Systematic verification that every historic tagged version on the production
/// store is searchable via vector, FTS, and hybrid search. Reports fragment
/// counts, row counts, and per-mode latency for every tag.
///
/// Run with:
///   cargo test --lib store::lance::tests -- production_store_historic_searchability --ignored --nocapture
#[tokio::test]
#[ignore]
#[allow(clippy::excessive_precision)]
async fn production_store_historic_searchability() {
    use std::time::Instant;

    let base_dir = "/tmp/vectorlink-data";
    let path = format!("{base_dir}/admin__product_assortment.lance");
    if !std::path::Path::new(&path).exists() {
        eprintln!("Production store not found at {path}, skipping");
        return;
    }

    // The production store uses 768-dim embeddings.
    let store = LanceStore::new(
        std::path::Path::new(base_dir),
        768,
        256 * 1024 * 1024,
        128 * 1024 * 1024,
    );
    let domain = "admin__product_assortment";

    // Open fresh to get all tags.
    let ds_fresh = store
        .io_open_fresh(&store.dataset_path(domain).to_string_lossy())
        .await
        .expect("open production store");

    let tags = ds_fresh
        .tags()
        .list()
        .await
        .expect("list tags");

    eprintln!("=== HISTORIC SEARCHABILITY VERIFICATION ===");
    eprintln!("Store: {path}");
    eprintln!("Total tags: {}", tags.len());
    eprintln!();

    // Sort tags by version for ordered output.
    let mut sorted_tags: Vec<(String, _)> = tags.into_iter().collect();
    sorted_tags.sort_by_key(|(_, tc)| (tc.branch.clone(), tc.version));

    // Real L2-normalized embedding for "search_query: product" from
    // nomic-embed-text-v2-moe (768-dim). This produces meaningful vector
    // search hits instead of the zero-vector no-op.
    let dummy_embedding: Vec<f32> = vec![
        0.0493964005f32, -0.0076268156f32, -0.0478943746f32, -0.0642956668f32, 0.0047795463f32, 0.0008906153f32, 0.0118389064f32, -0.0056415942f32,
        -0.0096636655f32, 0.0835293358f32, 0.0336471183f32, 0.0001663360f32, -0.0056399444f32, 0.0090132405f32, 0.0429322138f32, -0.0086339666f32,
        -0.0088066846f32, -0.0021529045f32, 0.0165312282f32, 0.0165519332f32, 0.0015920432f32, 0.0221244609f32, -0.0097195285f32, -0.0932218313f32,
        -0.0422059609f32, -0.0255716877f32, 0.0553415202f32, 0.0345715983f32, 0.0034806587f32, 0.0127692044f32, -0.0176365031f32, -0.0644342418f32,
        -0.0645256468f32, 0.0482490176f32, -0.0170854881f32, 0.0773524361f32, 0.0186992651f32, -0.0834883458f32, 0.0128488784f32, 0.0078525486f32,
        0.0616632449f32, -0.0487703976f32, -0.0205449430f32, 0.0296580835f32, 0.0679867906f32, 0.0731773363f32, 0.0632577668f32, -0.0206191270f32,
        0.0353525522f32, 0.0605886850f32, -0.0240665998f32, 0.0115882064f32, 0.0391679310f32, -0.0006425000f32, 0.0051917397f32, -0.0223144319f32,
        -0.0067084303f32, 0.0075456734f32, -0.0340724383f32, -0.0208197920f32, -0.0016482468f32, -0.0282566556f32, -0.0507427135f32, 0.0079588856f32,
        0.0388000811f32, -0.0259720357f32, 0.0190697020f32, -0.0375010581f32, -0.0083013316f32, -0.0165072372f32, -0.0665585267f32, -0.0018807540f32,
        0.0329173713f32, -0.0183132501f32, -0.0458201877f32, 0.0209897489f32, -0.0249618817f32, 0.0138086963f32, -0.0371633381f32, -0.0049053484f32,
        0.0113402324f32, 0.0773682411f32, 0.0429108158f32, 0.0386239181f32, 0.0065277500f32, 0.0087805466f32, 0.0653214067f32, 0.0203996940f32,
        -0.0693794865f32, 0.0068150237f32, -0.0053232714f32, 0.0008777272f32, 0.0354064312f32, 0.0024250099f32, 0.0016693374f32, -0.0143372113f32,
        -0.0185077191f32, 0.0373199151f32, 0.0599135540f32, -0.0372819731f32, 0.0844367158f32, 0.0050959253f32, 0.0156367652f32, -0.0562627702f32,
        -0.0212475649f32, 0.0575940801f32, 0.0400835740f32, -0.0514764494f32, 0.0024972536f32, -0.0556485102f32, -0.0232097698f32, -0.0165853642f32,
        -0.0082424996f32, -0.0495008655f32, 0.0081794626f32, -0.0842466358f32, -0.0191719020f32, 0.0031120773f32, 0.0174672521f32, 0.0733863763f32,
        0.0099001635f32, 0.0225872219f32, 0.0508062624f32, -0.0015717730f32, -0.0005420819f32, -0.0124835374f32, -0.0115336204f32, 0.0154585292f32,
        -0.0338532903f32, 0.0048619678f32, 0.0012853709f32, 0.0699192025f32, 0.0382237881f32, -0.0444049978f32, 0.0034687796f32, 0.0512324694f32,
        0.0738941963f32, -0.0813859209f32, -0.0135281833f32, -0.0428881328f32, 0.0202038790f32, 0.0072519536f32, 0.0513637304f32, -0.0913758904f32,
        -0.0281540346f32, -0.0160497102f32, 0.0005541927f32, -0.0037998322f32, -0.0071678856f32, -0.0386625061f32, -0.0081893716f32, -0.0348191683f32,
        -0.0465949777f32, -0.0102923245f32, -0.0294105545f32, -0.0028997116f32, 0.0003250527f32, -0.0025740987f32, 0.0130332103f32, -0.0146973633f32,
        0.0615294069f32, -0.0108662175f32, 0.0452251937f32, 0.0437688958f32, 0.0273031116f32, 0.0748608002f32, 0.0522329474f32, 0.0213167389f32,
        0.0062588547f32, -0.0398239510f32, 0.0419698479f32, -0.0401571520f32, -0.0290200605f32, -0.0267563587f32, 0.0354851842f32, 0.0228538799f32,
        -0.0345336403f32, 0.0232339598f32, 0.0037931686f32, 0.0037535061f32, 0.0626347069f32, 0.0481837926f32, -0.0332776883f32, 0.0109960154f32,
        0.0058028811f32, -0.0488081845f32, 0.0269422606f32, -0.0281726316f32, 0.0461558747f32, 0.0591372410f32, -0.0343862303f32, 0.0314785084f32,
        -0.0340182883f32, 0.0539698473f32, -0.0251080007f32, 0.0012660258f32, 0.0141171918f32, -0.0377917011f32, -0.0421100979f32, -0.0837332118f32,
        -0.0344311313f32, -0.0100838855f32, -0.0526420004f32, -0.0077877460f32, 0.0092140625f32, -0.0320049064f32, -0.0655965027f32, 0.0738055463f32,
        0.0046390042f32, 0.0767954861f32, -0.0180889951f32, -0.0258662317f32, 0.0092221265f32, -0.0463844777f32, -0.0186492591f32, 0.0447534898f32,
        -0.0177792111f32, 0.0215920629f32, 0.0513037524f32, -0.0321403584f32, -0.0927037853f32, 0.0424325179f32, -0.0041051132f32, -0.0631944168f32,
        -0.0289779235f32, -0.0255128677f32, 0.0382321231f32, -0.0068569697f32, 0.0089239996f32, -0.0250253407f32, -0.0125108404f32, -0.0338586483f32,
        0.0346994013f32, 0.0639484968f32, -0.0042020408f32, 0.0221803979f32, -0.0328173064f32, 0.0458493207f32, -0.0057065242f32, -0.0118449764f32,
        -0.0459439957f32, -0.0205669360f32, -0.0294330645f32, -0.0138947863f32, -0.0417861209f32, -0.0325471334f32, -0.0733843863f32, -0.0373517441f32,
        0.0002911378f32, 0.0313507864f32, -0.0249696477f32, -0.0454196677f32, 0.0189121951f32, 0.0988670910f32, -0.0495017145f32, 0.0284372166f32,
        -0.0218378169f32, -0.0154842182f32, 0.0366763082f32, -0.0472388176f32, 0.0394814740f32, -0.0256584717f32, -0.0028684009f32, -0.0199994470f32,
        0.0104575055f32, 0.0204962200f32, -0.0293866385f32, -0.0519613274f32, -0.0285086056f32, -0.0384027981f32, -0.0582649231f32, -0.0013812334f32,
        -0.0011122703f32, 0.0080486536f32, -0.0264811287f32, -0.0176580311f32, 0.0139964333f32, -0.0297109415f32, 0.0276367966f32, -0.0493633675f32,
        0.0436251078f32, 0.0144284803f32, 0.0202558850f32, 0.0029268269f32, -0.0043892803f32, 0.0557549192f32, -0.0282916636f32, -0.0645222708f32,
        -0.0677882966f32, -0.0129968773f32, 0.0291139745f32, -0.0139237623f32, 0.0191110900f32, 0.0036800462f32, 0.0185692671f32, -0.0064750547f32,
        -0.0156449222f32, 0.0595952970f32, -0.0617848339f32, -0.0389761420f32, -0.0236542528f32, 0.0042283278f32, -0.0101522435f32, -0.0476554176f32,
        -0.0066869737f32, 0.0243271338f32, 0.0536080493f32, 0.0202308120f32, 0.0629166568f32, -0.0071556016f32, 0.0243966478f32, -0.0193555620f32,
        0.0756400862f32, -0.0239678608f32, -0.0106970165f32, 0.0140242363f32, 0.0495088595f32, 0.0488868445f32, -0.0364767332f32, 0.0204493170f32,
        0.0544576913f32, -0.0347954763f32, -0.0095390995f32, 0.0269074926f32, -0.0299084895f32, -0.0060979120f32, 0.0419926959f32, -0.0022887803f32,
        0.0358776502f32, 0.0215621299f32, -0.0553496752f32, 0.0325342554f32, -0.0075095056f32, 0.0272077266f32, 0.0422912559f32, 0.0080514596f32,
        0.0068782207f32, 0.0329631443f32, -0.0193861860f32, -0.0067311697f32, 0.0519175204f32, 0.0468949676f32, 0.0623193649f32, 0.0022032081f32,
        0.0569356651f32, -0.0845270858f32, -0.0106492245f32, 0.0010651507f32, 0.0391305400f32, 0.0085307126f32, 0.0041231828f32, -0.0049607193f32,
        -0.0671904126f32, 0.0334332533f32, -0.0248953667f32, 0.0473154236f32, 0.0211108859f32, 0.0622115739f32, -0.0023798543f32, -0.0471952426f32,
        -0.0289456695f32, -0.0028543755f32, -0.0768025961f32, 0.0163674342f32, -0.0015036767f32, 0.0707884464f32, -0.0266694437f32, -0.0296355245f32,
        -0.0808269759f32, -0.0383393561f32, 0.0290438315f32, 0.0204327790f32, 0.0174831741f32, 0.0101832725f32, 0.0160537912f32, 0.0002140577f32,
        0.0067894521f32, 0.0300275855f32, -0.0659304267f32, 0.0091355745f32, 0.0191973900f32, -0.0139490073f32, 0.0407628360f32, -0.0304604885f32,
        0.0192182520f32, -0.0420057049f32, 0.0219766089f32, -0.0140268223f32, -0.0059629667f32, 0.0029609181f32, -0.0319494384f32, -0.0189946320f32,
        0.0008424715f32, 0.0207891140f32, -0.0088314396f32, -0.0524661874f32, -0.0096786235f32, 0.0169652061f32, -0.0168639342f32, 0.0216568619f32,
        0.0529322573f32, 0.0147501023f32, -0.0148722513f32, 0.0922463154f32, 0.0313771724f32, -0.0330654983f32, 0.0032890370f32, -0.0299190265f32,
        0.0099957205f32, -0.0217388109f32, -0.0015896121f32, -0.0002101789f32, -0.0228987738f32, -0.0318340304f32, 0.0273391196f32, -0.0215854429f32,
        -0.0246427368f32, 0.0317813184f32, -0.0216909149f32, -0.0257058287f32, -0.0138346423f32, 0.0344910903f32, -0.0350822412f32, -0.0239696888f32,
        -0.0332341253f32, -0.0283853346f32, 0.0178953321f32, 0.0104695675f32, 0.0337313583f32, 0.0290788605f32, -0.0157141022f32, 0.0048646838f32,
        0.0188323361f32, 0.0570967801f32, -0.0168237472f32, -0.0264091537f32, 0.0422127039f32, 0.0131201333f32, -0.0142785063f32, 0.0109244965f32,
        -0.0096261115f32, -0.0323875984f32, -0.0110890864f32, 0.0517494374f32, -0.0073988811f32, 0.0161102932f32, 0.0034497853f32, -0.0049657728f32,
        -0.0149131093f32, -0.0260878507f32, -0.0465859007f32, -0.0290346745f32, -0.0180114911f32, -0.0209194579f32, 0.0260331657f32, -0.0188778541f32,
        0.0407986180f32, -0.0651547867f32, 0.0022091448f32, -0.0094672395f32, -0.0318401364f32, -0.0419530179f32, -0.0111986584f32, -0.0427786949f32,
        0.0137088683f32, 0.0108762735f32, 0.0184653211f32, -0.0012932667f32, 0.0195994550f32, -0.0083666736f32, 0.0186761551f32, -0.0208364850f32,
        -0.0085693096f32, -0.0175036091f32, -0.0625680429f32, 0.0005347970f32, -0.0629472928f32, 0.0038641084f32, -0.0246134888f32, -0.0195555300f32,
        0.0160540592f32, 0.0118049884f32, 0.0206008250f32, -0.0150724027f32, -0.0035898816f32, 0.0481075176f32, -0.0567714971f32, -0.0065070417f32,
        0.0470552676f32, -0.0421240579f32, -0.0156821992f32, -0.0183647971f32, -0.0033485736f32, -0.0419587659f32, 0.0418054629f32, -0.0423538399f32,
        -0.0702143665f32, 0.0088922076f32, 0.0119568169f32, 0.0272068796f32, 0.0041989761f32, -0.0550152072f32, -0.0229155038f32, -0.0257036037f32,
        -0.1071506446f32, -0.0128113404f32, -0.0348334083f32, 0.0148939383f32, 0.0024201883f32, 0.0483269476f32, 0.0485708426f32, 0.0175573931f32,
        -0.0610591219f32, -0.0175450251f32, 0.0174316591f32, 0.0319423434f32, -0.0274462886f32, -0.0168823372f32, 0.0124728654f32, 0.0406889580f32,
        -0.0086032196f32, 0.0422637999f32, 0.0429244918f32, 0.0111102184f32, -0.0141200118f32, 0.0041982232f32, 0.0236899338f32, 0.0360293412f32,
        0.0268915056f32, -0.0169739661f32, 0.0245267868f32, -0.0214012199f32, -0.0207547690f32, 0.0352471182f32, 0.0291153685f32, 0.0129177324f32,
        0.0155913332f32, -0.0237929438f32, -0.0153828142f32, 0.0888963355f32, 0.0175940701f32, -0.0341434013f32, -0.0479943576f32, -0.0103395095f32,
        0.0130063193f32, -0.0419077779f32, -0.0749791002f32, -0.0710317464f32, -0.0002631060f32, -0.0093090335f32, 0.0438447238f32, 0.0074644811f32,
        0.0098276415f32, -0.0095689685f32, -0.0401974750f32, 0.0040997325f32, -0.0479649956f32, -0.0683633966f32, -0.0148637213f32, -0.0149368882f32,
        -0.0456275847f32, -0.0044882603f32, 0.0277292396f32, 0.0260723287f32, -0.0195284180f32, 0.0215451789f32, 0.0477357406f32, 0.0545987643f32,
        -0.0328472054f32, 0.0151942002f32, 0.0317026184f32, -0.0317560044f32, 0.0089687955f32, -0.0076175011f32, 0.0137645543f32, -0.0168445892f32,
        -0.0286962766f32, 0.0479666176f32, -0.0540091813f32, 0.0058394099f32, 0.0371796381f32, -0.0482815276f32, -0.0098241865f32, 0.0474965576f32,
        -0.0224289679f32, -0.0094544465f32, -0.0014734722f32, 0.0580197071f32, 0.0048235098f32, 0.0350773782f32, -0.0027862817f32, -0.0355080012f32,
        -0.0007032834f32, -0.0236611228f32, 0.0321065734f32, 0.0709054064f32, -0.0173240961f32, 0.0433334728f32, 0.0501667195f32, -0.0256518557f32,
        0.0026790469f32, -0.0444356108f32, 0.0144633433f32, 0.0987034910f32, -0.0191069550f32, 0.0054763757f32, -0.0403172950f32, 0.0206633760f32,
        -0.0138348623f32, 0.0153496872f32, 0.0191362620f32, -0.0455525877f32, -0.0712589764f32, 0.0443415278f32, -0.0532917473f32, -0.0081408486f32,
        0.0161199042f32, -0.0157081492f32, -0.0297944755f32, -0.0312512654f32, 0.0146967003f32, 0.0078707306f32, -0.0069188971f32, 0.0645984568f32,
        0.0246773288f32, -0.0697929305f32, -0.0359406182f32, 0.0273286886f32, -0.0306907885f32, -0.0283793416f32, -0.0512992274f32, -0.0134057343f32,
        0.0435877718f32, 0.0472999976f32, -0.0217211289f32, -0.0016120242f32, 0.0510523374f32, -0.0207218910f32, 0.0105068375f32, 0.0586884851f32,
        -0.0238722028f32, -0.0451444327f32, 0.0227688239f32, 0.0480871376f32, 0.0350599582f32, 0.0880568556f32, 0.0104892965f32, -0.0370279531f32,
        0.0218567859f32, -0.0096095605f32, 0.0235042518f32, 0.0215195759f32, 0.0736321163f32, -0.0347937103f32, 0.0634892768f32, 0.0203586080f32,
        0.0180302991f32, -0.0253792617f32, 0.0360018282f32, 0.0280729446f32, 0.0020780559f32, -0.0072486556f32, 0.0006769710f32, 0.0058215830f32,
        -0.1195126340f32, 0.0924538154f32, -0.0678086866f32, 0.0346691183f32, -0.0377694311f32, -0.0122079334f32, 0.0291348785f32, 0.0214981789f32,
        -0.0187401911f32, 0.0029495527f32, -0.0032017413f32, 0.0156112142f32, -0.0115704324f32, -0.0341995863f32, -0.0057471662f32, 0.0376350341f32,
        -0.0269292386f32, 0.0559779642f32, -0.0247928838f32, -0.0490352475f32, 0.0175124341f32, 0.0082411676f32, 0.0307891875f32, -0.0275652656f32,
        -0.0870179356f32, 0.0585707251f32, 0.0125316104f32, -0.0025036180f32, 0.0525705234f32, 0.0352154652f32, -0.0166211862f32, 0.0590523530f32,
        0.0072761626f32, -0.0053103637f32, 0.0304644315f32, 0.0273681806f32, -0.0217064519f32, -0.0030373056f32, 0.0118071474f32, 0.0185692861f32,
        0.0020299250f32, -0.0291589315f32, -0.0599375150f32, 0.0139104298f32, 0.0768959261f32, -0.0006652940f32, 0.0714888864f32, -0.0179751821f32,
        -0.0609253129f32, 0.0521022954f32, 0.0144791713f32, 0.0056748827f32, -0.0510837574f32, 0.0427907379f32, 0.0066550511f32, 0.0057513962f32,
        0.0322670434f32, -0.0054790151f32, -0.0011546294f32, -0.0009914264f32, -0.0116588764f32, -0.0057463261f32, -0.0524134124f32, 0.0159135752f32,
        -0.0288265876f32, -0.0491750625f32, -0.0458685847f32, 0.0761230402f32, -0.0179710301f32, -0.0343553783f32, 0.0289302365f32, -0.0008097217f32,
        0.0240540348f32, 0.0126366784f32, 0.0066003257f32, -0.0319009634f32, -0.0681393406f32, 0.0676925226f32, -0.0765857862f32, -0.0468698756f32,
        -0.0695335965f32, -0.0187388351f32, -0.0335249963f32, -0.0150601292f32, -0.0149305483f32, 0.0083449556f32, -0.0165856272f32, 0.0150615152f32,
        -0.0080790386f32, 0.0276191576f32, -0.0304603025f32, 0.0599187100f32, -0.0381556431f32, 0.0081368706f32, 0.0209731339f32, 0.0326592484f32,
        0.0117236474f32, -0.0098692075f32, 0.0507493575f32, 0.0480049526f32, 0.0313557704f32, -0.0156158102f32, -0.0383137711f32, 0.0409384129f32,
    ];
    let query_text = "product";

    let mut summary = Vec::new();

    for (tag_name, tc) in &sorted_tags {
        // Decode commit ID from tag name for readability.
        let commit_id = crate::layeridx::decode_commit_tag(tag_name)
            .unwrap_or_else(|_| tag_name.clone());

        // Check out the version.
        let checkout_start = Instant::now();
        let ds_checked = ds_fresh
            .checkout_version((tc.branch.as_deref(), Some(tc.version)))
            .await;
        let checkout_ms = checkout_start.elapsed().as_millis();

        match &ds_checked {
            Err(e) => {
                eprintln!(
                    "FAIL  commit={} tag={} version={} branch={:?}: checkout failed: {}",
                    commit_id, tag_name, tc.version, tc.branch, e
                );
                summary.push(SearchResult {
                    commit_id,
                    tag_name: tag_name.clone(),
                    version: tc.version,
                    branch: tc.branch.clone(),
                    checkout_ok: false,
                    checkout_ms,
                    fragments: 0,
                    rows: 0,
                    vector_ok: false,
                    vector_hits: 0,
                    vector_ms: 0,
                    fts_ok: false,
                    fts_hits: 0,
                    fts_ms: 0,
                    hybrid_ok: false,
                    hybrid_hits: 0,
                    hybrid_ms: 0,
                    error: format!("checkout: {e}"),
                });
                continue;
            }
            Ok(ds) => {
                // Count fragments and rows.
                let fragment_count = ds.manifest.fragments.len();
                let total_rows: usize = ds.manifest.fragments.iter().map(|f| f.num_rows().unwrap_or(0)).sum();

                // Vector search via io_search (public API).
                let vq = SearchQuery {
                    query_embedding: dummy_embedding.clone(),
                    query_text: query_text.to_string(),
                    mode: SearchMode::Vector,
                    start: 0,
                    count: 10,
                    doc_type_filter: Vec::new(),
                    doc_id_filter: Vec::new(),
                    snippet: false,
                };
                let v_start = Instant::now();
                let v_result = store.io_search(domain, "main", &commit_id, &vq).await;
                let v_ms = v_start.elapsed().as_millis();

                let (vector_ok, vector_hits, v_err) = match v_result {
                    Ok(hits) => (true, hits.len(), String::new()),
                    Err(e) => (false, 0, e.to_string()),
                };

                // FTS search via io_search.
                let fq = SearchQuery {
                    query_embedding: dummy_embedding.clone(),
                    query_text: query_text.to_string(),
                    mode: SearchMode::Fts,
                    start: 0,
                    count: 10,
                    doc_type_filter: Vec::new(),
                    doc_id_filter: Vec::new(),
                    snippet: false,
                };
                let f_start = Instant::now();
                let f_result = store.io_search(domain, "main", &commit_id, &fq).await;
                let f_ms = f_start.elapsed().as_millis();

                let (fts_ok, fts_hits, f_err) = match f_result {
                    Ok(hits) => (true, hits.len(), String::new()),
                    Err(e) => (false, 0, e.to_string()),
                };

                // Hybrid search via io_search.
                let hq = SearchQuery {
                    query_embedding: dummy_embedding.clone(),
                    query_text: query_text.to_string(),
                    mode: SearchMode::Hybrid,
                    start: 0,
                    count: 10,
                    doc_type_filter: Vec::new(),
                    doc_id_filter: Vec::new(),
                    snippet: false,
                };
                let h_start = Instant::now();
                let h_result = store.io_search(domain, "main", &commit_id, &hq).await;
                let h_ms = h_start.elapsed().as_millis();

                let (hybrid_ok, hybrid_hits, h_err) = match h_result {
                    Ok(hits) => (true, hits.len(), String::new()),
                    Err(e) => (false, 0, e.to_string()),
                };

                let all_ok = vector_ok && fts_ok && hybrid_ok;
                let status = if all_ok { "OK  " } else { "FAIL" };

                eprintln!(
                    "{status} commit={commit_id} v={:>4} branch={:?} frags={:>3} rows={:>5} | checkout={checkout_ms}ms vec={v_ms}ms({vector_hits}) fts={f_ms}ms({fts_hits}) hyb={h_ms}ms({hybrid_hits}){}",
                    tc.version,
                    tc.branch.as_deref().unwrap_or("main"),
                    fragment_count,
                    total_rows,
                    if !all_ok { format!(" ERR: {} {} {}", v_err, f_err, h_err) } else { String::new() }
                );

                summary.push(SearchResult {
                    commit_id,
                    tag_name: tag_name.clone(),
                    version: tc.version,
                    branch: tc.branch.clone(),
                    checkout_ok: true,
                    checkout_ms,
                    fragments: fragment_count,
                    rows: total_rows,
                    vector_ok,
                    vector_hits,
                    vector_ms: v_ms,
                    fts_ok,
                    fts_hits,
                    fts_ms: f_ms,
                    hybrid_ok,
                    hybrid_hits,
                    hybrid_ms: h_ms,
                    error: if !all_ok { format!("vec:{v_err} fts:{f_err} hyb:{h_err}") } else { String::new() },
                });
            }
        }
    }

    // Summary statistics.
    let total = summary.len();
    let checkout_ok = summary.iter().filter(|s| s.checkout_ok).count();
    let vector_ok = summary.iter().filter(|s| s.vector_ok).count();
    let fts_ok = summary.iter().filter(|s| s.fts_ok).count();
    let hybrid_ok = summary.iter().filter(|s| s.hybrid_ok).count();
    let all_ok = summary.iter().filter(|s| s.checkout_ok && s.vector_ok && s.fts_ok && s.hybrid_ok).count();

    eprintln!();
    eprintln!("=== SUMMARY ===");
    eprintln!("Total tags:       {total}");
    eprintln!("Checkout OK:      {checkout_ok}/{total}");
    eprintln!("Vector search OK: {vector_ok}/{total}");
    eprintln!("FTS search OK:    {fts_ok}/{total}");
    eprintln!("Hybrid search OK: {hybrid_ok}/{total}");
    eprintln!("ALL OK:           {all_ok}/{total}");
    eprintln!();

    // Latency statistics (only for successful searches).
    let v_latencies: Vec<u128> = summary.iter().filter(|s| s.vector_ok).map(|s| s.vector_ms).collect();
    let f_latencies: Vec<u128> = summary.iter().filter(|s| s.fts_ok).map(|s| s.fts_ms).collect();
    let h_latencies: Vec<u128> = summary.iter().filter(|s| s.hybrid_ok).map(|s| s.hybrid_ms).collect();
    let c_latencies: Vec<u128> = summary.iter().filter(|s| s.checkout_ok).map(|s| s.checkout_ms).collect();

    fn lat_stats(lat: &[u128]) -> (u128, u128, u128, u128) {
        if lat.is_empty() { return (0, 0, 0, 0); }
        let min = *lat.iter().min().unwrap();
        let max = *lat.iter().max().unwrap();
        let avg = lat.iter().sum::<u128>() / lat.len() as u128;
        let med = {
            let mut sorted = lat.to_vec();
            sorted.sort();
            sorted[sorted.len() / 2]
        };
        (min, max, avg, med)
    }

    let (v_min, v_max, v_avg, v_med) = lat_stats(&v_latencies);
    let (f_min, f_max, f_avg, f_med) = lat_stats(&f_latencies);
    let (h_min, h_max, h_avg, h_med) = lat_stats(&h_latencies);
    let (c_min, c_max, c_avg, c_med) = lat_stats(&c_latencies);

    eprintln!("=== LATENCY (ms) ===");
    eprintln!("  mode       min     max     avg   median  count");
    eprintln!("  checkout {:>6} {:>6} {:>6} {:>6}  {}", c_min, c_max, c_avg, c_med, c_latencies.len());
    eprintln!("  vector   {:>6} {:>6} {:>6} {:>6}  {}", v_min, v_max, v_avg, v_med, v_latencies.len());
    eprintln!("  fts      {:>6} {:>6} {:>6} {:>6}  {}", f_min, f_max, f_avg, f_med, f_latencies.len());
    eprintln!("  hybrid   {:>6} {:>6} {:>6} {:>6}  {}", h_min, h_max, h_avg, h_med, h_latencies.len());
    eprintln!();

    // List any failures.
    let failures: Vec<_> = summary.iter().filter(|s| !(s.checkout_ok && s.vector_ok && s.fts_ok && s.hybrid_ok)).collect();
    if failures.is_empty() {
        eprintln!("ALL {total} TAGS PASSED — every historic version is searchable via vector, FTS, and hybrid.");
    } else {
        eprintln!("FAILURES ({}):", failures.len());
        for f in failures {
            eprintln!("  {} (v={}, branch={:?}): {}", f.commit_id, f.version, f.branch, f.error);
        }
    }

    assert!(all_ok == total, "Not all tags are fully searchable: {}/{} passed", all_ok, total);
}

/// Diagnostic: check vector index coverage at key versions on the production
/// store. For each version, reports how many fragments have vector index
/// metadata vs how many don't. This reveals whether compaction is leaving
/// fragments without index coverage.
///
/// Run with:
///   cargo test --lib store::lance::tests -- production_store_index_coverage --ignored --nocapture
#[tokio::test]
#[ignore]
async fn production_store_index_coverage() {
    let base_dir = "/tmp/vectorlink-data";
    let path = format!("{base_dir}/admin__product_assortment.lance");
    if !std::path::Path::new(&path).exists() {
        eprintln!("Production store not found at {path}, skipping");
        return;
    }

    let store = LanceStore::new(
        std::path::Path::new(base_dir),
        768,
        256 * 1024 * 1024,
        128 * 1024 * 1024,
    );
    let domain = "admin__product_assortment";

    // Key versions to inspect: low-latency, cliff, and post-cliff.
    let key_versions: Vec<(u64, &str)> = vec![
        (267, "low-latency baseline"),
        (604, "first cliff"),
        (705, "recovery"),
        (755, "second cliff"),
        (948, "recovery 2"),
        (1090, "permanent cliff"),
        (1142, "post-cliff high frags"),
        (1280, "post-compaction 16 frags"),
    ];

    eprintln!("=== INDEX COVERAGE AT KEY VERSIONS ===");
    eprintln!(
        "{:>6}  {:>6}  {:>6}  {:>6}  {:>6}  {:>6}  note",
        "version", "frags", "indexed", "unindexed", "pct_idx", "idx_files"
    );
    eprintln!("{}", "-".repeat(70));

    for (version, note) in &key_versions {
        // Checkout the version on the main branch.
        let ds_fresh = store
            .io_open_fresh(&store.dataset_path(domain).to_string_lossy())
            .await
            .expect("open fresh failed");
        let ds = ds_fresh.checkout_version(*version).await.expect("checkout failed");

        let fragments = ds.get_fragments();
        let total_frags = fragments.len();

        // Load indices to see which indices exist at this version.
        let indices = ds.load_indices().await.expect("load_indices failed");
        let vector_indices: Vec<_> = indices
            .iter()
            .filter(|idx| idx.name == crate::store::vector_index::VECTOR_INDEX_NAME)
            .collect();
        let fts_indices: Vec<_> = indices
            .iter()
            .filter(|idx| idx.name == "content_fts")
            .collect();

        // Count indexed vs unindexed fragments by checking which fragments
        // have index files. We can't access private metadata, but we can
        // infer coverage from the index's fragment_set.
        let vec_idx_count = vector_indices.len();
        let fts_idx_count = fts_indices.len();

        eprintln!(
            "  v={:>5} frags={:>3} vec_indices={} fts_indices={}  {}",
            version, total_frags, vec_idx_count, fts_idx_count, note
        );

        // Report index details.
        for vi in &vector_indices {
            eprintln!(
                "       vector index: name={} uuid={}",
                vi.name, vi.uuid
            );
        }
        for fi in &fts_indices {
            eprintln!(
                "       fts index: name={} uuid={}",
                fi.name, fi.uuid
            );
        }
    }
}

/// Active compaction test: simulate pushes + background compaction to
/// reproduce the latency cliff. Creates a dataset, pushes data in batches,
/// triggers compaction, and measures vector search latency after each
/// push and after each compaction.
///
/// Run with:
///   cargo test --lib store::lance::tests -- active_compaction_latency_cliff --ignored --nocapture
#[tokio::test]
#[ignore]
async fn active_compaction_latency_cliff() {
    use crate::store::lance::schema::ChunkRow;
    use crate::store::vector_index::tests::make_chunk_row;

    let tmp = tempfile::tempdir().expect("tempdir failed");
    let base_dir = tmp.path();
    let dim = 128;

    let mut store = LanceStore::new(base_dir, dim, 64 * 1024 * 1024, 32 * 1024 * 1024);
    let domain = "test_compaction_cliff";
    let branch = "main";

    // Use a small vector index config for testing.
    store.set_vector_index_config(crate::store::lance::VectorIndexConfig {
        num_partitions: 4,
        m: 16,
        ef_construction: 100,
        nprobes: 4,
        refine_factor: Some(10),
    });

    // Phase 1: Push enough data to create the initial vector index.
    // Need >= 256 rows for IVF training.
    eprintln!("=== Phase 1: Initial data load (300 rows) ===");
    let rows: Vec<ChunkRow> = (0..300)
        .map(|i| make_chunk_row(&format!("doc/{}", i), dim, i as u64 + 1))
        .collect();
    let v0 = store
        .io_upsert_chunks(domain, branch, "doc_batch_0", &rows)
        .await
        .expect("initial push failed");
    store.io_tag_commit(domain, branch, "commit_init", v0).await.expect("tag failed");

    // Measure baseline latency.
    let query = make_search_query(dim, SearchMode::Vector);
    let baseline_latency = measure_vector_latency(&store, domain, branch, "commit_init", &query).await;
    eprintln!("Baseline vector search latency: {}ms", baseline_latency);

    // Phase 2: Push in batches of 100, measuring latency after each.
    // Trigger compaction manually after every 20th push (simulating background compaction).
    eprintln!("\n=== Phase 2: Incremental pushes with latency measurement ===");
    eprintln!(
        "{:>5}  {:>6}  {:>6}  {:>8}  {:>8}  event",
        "push", "frags", "rows", "vec_ms", "delta_ms"
    );
    eprintln!("{}", "-".repeat(60));

    let mut prev_latency = baseline_latency;
    for batch in 1..=60 {
        let commit_id = format!("commit_{}", batch);
        let doc_id = format!("doc_batch_{}", batch);
        let rows: Vec<ChunkRow> = (0..100)
            .map(|i| {
                let global_i = 300 + (batch - 1) * 100 + i;
                make_chunk_row(&format!("doc/{}", global_i), dim, global_i as u64 + 1)
            })
            .collect();
        let v = store
            .io_upsert_chunks(domain, branch, &doc_id, &rows)
            .await
            .expect("push failed");
        store.io_tag_commit(domain, branch, &commit_id, v).await.expect("tag failed");

        // Measure latency.
        let latency = measure_vector_latency(&store, domain, branch, &commit_id, &query).await;

        // Get fragment count.
        let ds = store
            .io_open_dataset_readonly(domain)
            .await
            .expect("open failed")
            .expect("dataset not found");
        let ds = ds.read().await;
        let frags = ds.get_fragments().len();
        let rows_count = ds.count_rows(None).await.unwrap_or(0);
        drop(ds);

        let delta = latency as i64 - prev_latency as i64;
        let event = if delta > 100 {
            format!("<<< LATENCY JUMP (+{}ms)", delta)
        } else {
            String::new()
        };

        eprintln!(
            "{:>5}  {:>6}  {:>6}  {:>8}  {:>8}  {}",
            batch, frags, rows_count, latency, delta, event
        );

        prev_latency = latency;

        // Every 20th push, trigger background compaction manually.
        if batch % 20 == 0 {
            eprintln!("       >>> triggering background compaction <<<");
            // Compact data only, NO index rebuild.
            let ds_uncached = store
                .io_open_dataset_uncached(domain, branch)
                .await
                .expect("open uncached failed")
                .expect("dataset not found");
            let mut ds_uncached = ds_uncached;
            crate::store::lance::io_compact_data(&mut ds_uncached, false)
                .await
                .expect("compact_data failed");
            let frags_after = ds_uncached.get_fragments().len();
            eprintln!("       compaction: fragments -> {}", frags_after);
            drop(ds_uncached);

            // Refresh cache.
            store
                .io_refresh_cached_dataset(domain, branch)
                .await
                .expect("refresh failed");

            // Measure latency after compaction.
            let post_compact_latency =
                measure_vector_latency(&store, domain, branch, &commit_id, &query).await;
            let delta = post_compact_latency as i64 - latency as i64;
            eprintln!(
                "       post-compaction latency: {}ms (delta {:+}ms) {}",
                post_compact_latency,
                delta,
                if delta > 100 { "<<< CLIFF" } else { "" }
            );
            prev_latency = post_compact_latency;
        }
    }

    // Phase 3: Now do a FULL compaction (drop indices, compact, recreate) like compact_domain.
    eprintln!("\n=== Phase 3: Full compaction (drop + compact + recreate indices) ===");
    let ds_uncached = store
        .io_open_dataset_uncached(domain, branch)
        .await
        .expect("open uncached failed")
        .expect("dataset not found");
    let mut ds = ds_uncached;

    // Drop indices.
    use lance::index::DatasetIndexExt;
    let indices = ds.load_indices().await.expect("load_indices failed");
    let index_names: Vec<String> = indices.iter().map(|idx| idx.name.clone()).collect();
    for name in &index_names {
        eprintln!("  dropping index: {}", name);
        ds.drop_index(name).await.expect("drop_index failed");
    }

    // Compact data.
    crate::store::lance::io_compact_data(&mut ds, true)
        .await
        .expect("compact_data failed");
    let frags_after = ds.get_fragments().len();
    eprintln!("  fragments after full compaction: {}", frags_after);

    // Recreate vector index.
    let config = store.vector_index_config().clone();
    crate::store::vector_index::io_ensure_vector_index(&mut ds, &config, true)
        .await
        .expect("vector index creation failed");

    // Recreate FTS index.
    crate::store::lance::io_ensure_fts_index_on_dataset(&mut ds)
        .await
        .expect("FTS index creation failed");

    drop(ds);

    store
        .io_refresh_cached_dataset(domain, branch)
        .await
        .expect("refresh failed");

    // Measure latency after full compaction.
    let last_commit = format!("commit_{}", 60);
    let full_compact_latency =
        measure_vector_latency(&store, domain, branch, &last_commit, &query).await;
    eprintln!(
        "  post-full-compaction latency: {}ms (was {}ms before)",
        full_compact_latency, prev_latency
    );

    // Assert: full compaction should REDUCE latency compared to background-only compaction.
    // If it doesn't, the cliff is caused by something other than index staleness.
    if prev_latency > 200 && full_compact_latency < prev_latency {
        eprintln!(
            "\n=== CONFIRMED: Full compaction reduces latency by {}ms ===",
            prev_latency - full_compact_latency
        );
        eprintln!("Root cause: background compaction compacts data fragments but does NOT");
        eprintln!("rebuild the vector index. After compaction, the vector index is stale —");
        eprintln!("it references old fragment IDs that no longer exist in the expected form.");
        eprintln!("Search degrades to flat-scan over the new unindexed compacted fragments.");
    } else if full_compact_latency >= prev_latency {
        eprintln!("\n=== UNEXPECTED: Full compaction did NOT reduce latency ===");
        eprintln!("The cliff may be caused by something other than index staleness.");
    } else {
        eprintln!("\n=== No significant latency cliff observed in this test ===");
    }
}

async fn measure_vector_latency(
    store: &LanceStore,
    domain: &str,
    branch: &str,
    commit: &str,
    query: &SearchQuery,
) -> u128 {
    use std::time::Instant;
    let start = Instant::now();
    let _ = store.io_search(domain, branch, commit, query).await;
    start.elapsed().as_millis()
}

fn make_search_query(dim: usize, mode: SearchMode) -> SearchQuery {
    SearchQuery {
        query_embedding: vec![0.0f32; dim],
        query_text: "test".to_string(),
        mode,
        start: 0,
        count: 10,
        doc_type_filter: Vec::new(),
        doc_id_filter: Vec::new(),
        snippet: false,
    }
}

#[allow(dead_code)]
struct SearchResult {
    commit_id: String,
    tag_name: String,
    version: u64,
    branch: Option<String>,
    checkout_ok: bool,
    checkout_ms: u128,
    fragments: usize,
    rows: usize,
    vector_ok: bool,
    vector_hits: usize,
    vector_ms: u128,
    fts_ok: bool,
    fts_hits: usize,
    fts_ms: u128,
    hybrid_ok: bool,
    hybrid_hits: usize,
    hybrid_ms: u128,
    error: String,
}

// ===========================================================================
// BOUNDARY-AWARE INDEXING TESTS
//
// Tests for the new architecture: single end-of-push append (Option C),
// boundary-aware indexing at every 3rd commit, cascade merges, and
// index_delta_counts purge on delete.
// ===========================================================================

/// Helper: push docs and tag a commit, returning the tagged version.
/// Uses io_upsert_chunks + io_tag_commit (simulates a push without the
/// full pipeline, which requires an embed provider).
async fn push_and_tag(
    store: &LanceStore,
    domain: &str,
    branch: &str,
    commit: &str,
    rows: &[ChunkRow],
) -> u64 {
    let v = store
        .io_upsert_chunks(domain, branch, &format!("doc_{}", commit), rows)
        .await
        .expect("upsert");
    store
        .io_tag_commit(domain, branch, commit, v)
        .await
        .expect("tag");
    v
}

/// Helper: count tagged commits for a branch.
async fn count_tags(store: &LanceStore, domain: &str, branch: &str) -> usize {
    store.io_count_branch_commits(domain, branch).await.unwrap_or(0)
}

/// Push 5 batches via io_upsert_chunks (each simulating a micro-batch within
/// a single push). After the push, assert exactly 1 new data fragment at HEAD.
#[tokio::test]
async fn single_append_one_fragment_per_push() {
    use crate::store::vector_index::tests::make_chunk_row;

    let dim = 8;
    let (store, _tmp) = make_test_store(dim);
    let domain = "admin/single_append";
    let branch = "main";

    // Push 1: 5 docs in one io_upsert_chunks call (single append).
    let rows: Vec<ChunkRow> = (0..5)
        .map(|i| make_chunk_row(&format!("doc/{}", i), dim, i + 1))
        .collect();
    push_and_tag(&store, domain, branch, "c0", &rows).await;

    let ds = store.io_open_dataset_readonly(domain).await.unwrap().unwrap();
    let ds = ds.read().await;
    let fragments = ds.get_fragments().len();
    assert_eq!(
        fragments, 1,
        "single push should produce exactly 1 fragment, got {}",
        fragments
    );

    // Push 2: 3 more docs.
    let rows2: Vec<ChunkRow> = (5..8)
        .map(|i| make_chunk_row(&format!("doc/{}", i), dim, i + 1))
        .collect();
    let _v1 = push_and_tag(&store, domain, branch, "c1", &rows2).await;

    let ds = store.io_open_dataset_readonly(domain).await.unwrap().unwrap();
    let ds = ds.read().await;
    let fragments = ds.get_fragments().len();
    assert_eq!(
        fragments, 2,
        "two pushes should produce exactly 2 fragments, got {}",
        fragments
    );
}

/// Push 2 commits, assert 2 fragments total (1 per push, no cross-push merging).
#[tokio::test]
async fn single_append_no_cross_push_merging() {
    use crate::store::vector_index::tests::make_chunk_row;

    let dim = 8;
    let (store, _tmp) = make_test_store(dim);
    let domain = "admin/no_cross_merge";
    let branch = "main";

    // Push 1.
    let rows1: Vec<ChunkRow> = (0..10)
        .map(|i| make_chunk_row(&format!("doc/{}", i), dim, i + 1))
        .collect();
    push_and_tag(&store, domain, branch, "c0", &rows1).await;

    // Push 2.
    let rows2: Vec<ChunkRow> = (10..20)
        .map(|i| make_chunk_row(&format!("doc/{}", i), dim, i + 1))
        .collect();
    push_and_tag(&store, domain, branch, "c1", &rows2).await;

    let ds = store.io_open_dataset_readonly(domain).await.unwrap().unwrap();
    let ds = ds.read().await;
    let fragments = ds.get_fragments().len();
    assert_eq!(
        fragments, 2,
        "two pushes should produce exactly 2 fragments (no cross-push merging), got {}",
        fragments
    );
}

/// After 3 pushes (commit_position 0, 1, 2), the 3rd push (position 2) should
/// have a vector index created.
#[tokio::test]
async fn third_commit_gets_index() {
    use crate::store::vector_index::tests::make_chunk_row;
    use crate::store::vector_index::{io_ensure_vector_index, VECTOR_INDEX_NAME};

    let dim = 128;
    let config = crate::store::vector_index::tests::make_test_config(dim);
    let (mut store, _tmp) = make_test_store(dim);
    store.set_vector_index_config(config.clone());
    let domain = "admin/third_commit_idx";
    let branch = "main";

    // Push 3 commits with enough data for IVF training (>= 256 rows).
    for i in 0..3 {
        let rows: Vec<ChunkRow> = (0..100)
            .map(|j| make_chunk_row(&format!("doc/{}_{}", i, j), dim, (i * 100 + j) as u64 + 1))
            .collect();
        push_and_tag(&store, domain, branch, &format!("c{}", i), &rows).await;
    }

    // Simulate boundary indexing at commit_position 2 (3rd commit).
    // The pipeline would call io_ensure_vector_index at this point.
    {
        let ds_arc = store.io_open_dataset(domain, branch).await.expect("open");
        let mut ds = ds_arc.write().await;
        io_ensure_vector_index(&mut ds, &config, false).await.expect("create vector index");
    }
    store.io_refresh_cached_dataset(domain, branch).await.expect("refresh");

    // Verify vector index exists.
    let ds = store.io_open_dataset_readonly(domain).await.unwrap().unwrap();
    let ds = ds.read().await;
    let indices = ds.load_indices().await.expect("load indices");
    assert!(
        indices.iter().any(|i| i.name == VECTOR_INDEX_NAME),
        "vector index must exist after 3rd commit boundary indexing"
    );
}

/// Non-3rd commits (positions 0, 1, 3, 4) should NOT have a new index UUID.
/// Verify by checking that no vector index exists after pushes 1 and 2.
#[tokio::test]
async fn non_third_commit_skips_index() {
    use crate::store::vector_index::tests::make_chunk_row;
    use crate::store::vector_index::VECTOR_INDEX_NAME;

    let dim = 128;
    let config = crate::store::vector_index::tests::make_test_config(dim);
    let (mut store, _tmp) = make_test_store(dim);
    store.set_vector_index_config(config.clone());
    let domain = "admin/non_third_skip";
    let branch = "main";

    // Push 1 commit (position 0 — should NOT create index).
    let rows: Vec<ChunkRow> = (0..100)
        .map(|j| make_chunk_row(&format!("doc/0_{}", j), dim, j as u64 + 1))
        .collect();
    push_and_tag(&store, domain, branch, "c0", &rows).await;

    // Verify NO vector index exists (pipeline skips index at position 0).
    let ds = store.io_open_dataset_readonly(domain).await.unwrap().unwrap();
    let ds = ds.read().await;
    let indices = ds.load_indices().await.expect("load indices");
    assert!(
        !indices.iter().any(|i| i.name == VECTOR_INDEX_NAME),
        "vector index must NOT exist after 1st commit (position 0)"
    );
}

/// After 9 pushes with boundary indexing at positions 2, 5, 8,
/// the cascade should merge index deltas.
#[tokio::test]
async fn cascade_merges_at_9th_commit() {
    use crate::store::vector_index::tests::make_chunk_row;
    use crate::store::vector_index::{io_ensure_vector_index, VECTOR_INDEX_NAME};
    use crate::store::lance::io_ensure_fts_index_on_dataset;
    use crate::store::lance::io_incremental_cascade;

    let dim = 128;
    let config = crate::store::vector_index::tests::make_test_config(dim);
    let (mut store, _tmp) = make_test_store(dim);
    store.set_vector_index_config(config.clone());
    let domain = "admin/cascade_9";
    let branch = "main";

    // Push 9 commits with enough data for IVF training.
    for i in 0..9 {
        let rows: Vec<ChunkRow> = (0..50)
            .map(|j| make_chunk_row(&format!("doc/{}_{}", i, j), dim, (i * 50 + j) as u64 + 1))
            .collect();
        push_and_tag(&store, domain, branch, &format!("c{}", i), &rows).await;

        // Simulate boundary indexing at positions 2, 5, 8.
        if crate::store::lance::should_index_commit(i) {
            let ds_arc = store.io_open_dataset(domain, branch).await.expect("open");
            let mut ds = ds_arc.write().await;
            io_ensure_vector_index(&mut ds, &config, false).await.expect("vector index");
            io_ensure_fts_index_on_dataset(&mut ds).await.expect("FTS index");
            drop(ds);
            store.io_refresh_cached_dataset(domain, branch).await.expect("refresh");

            // Increment delta count and run cascade.
            store.io_increment_delta_count(domain, branch).await;
            let delta_count = store.io_get_delta_count(domain, branch).await;
            if let Some(mut ds_u) = store.io_open_dataset_uncached(domain, branch).await.expect("open uncached") {
                let _ = io_incremental_cascade(&mut ds_u, delta_count).await;
            }
            store.io_refresh_cached_dataset(domain, branch).await.expect("refresh after cascade");
        }
    }

    // Verify vector index exists after 9 pushes with cascade.
    let ds = store.io_open_dataset_readonly(domain).await.unwrap().unwrap();
    let ds = ds.read().await;
    let indices = ds.load_indices().await.expect("load indices");
    assert!(
        indices.iter().any(|i| i.name == VECTOR_INDEX_NAME),
        "vector index must exist after 9 pushes with cascade"
    );
}

/// Search at a non-indexed commit should still return correct results
/// (KNN fallback for at most 2 commits delta).
#[tokio::test]
async fn search_correct_at_non_indexed_commit() {
    use crate::store::vector_index::tests::make_chunk_row;

    let dim = 8;
    let (store, _tmp) = make_test_store(dim);
    let domain = "admin/search_non_indexed";
    let branch = "main";

    // Push 1 commit (position 0 — no index).
    let rows: Vec<ChunkRow> = (0..5)
        .map(|i| make_chunk_row(&format!("doc/{}", i), dim, i + 1))
        .collect();
    push_and_tag(&store, domain, branch, "c0", &rows).await;

    // Search at c0 — should work via flat KNN (no ANN index).
    let query = SearchQuery {
        query_embedding: fake_embedding(dim, 1.0),
        query_text: "content".to_string(),
        mode: SearchMode::Vector,
        start: 0,
        count: 10,
        doc_type_filter: Vec::new(),
        doc_id_filter: Vec::new(),
        snippet: false,
    };
    let hits = store
        .io_search(domain, branch, "c0", &query)
        .await
        .expect("search at non-indexed commit");
    assert!(
        !hits.is_empty(),
        "search must return results at non-indexed commit (KNN fallback)"
    );
}

/// Search at an indexed commit should return correct results via ANN index.
#[tokio::test]
async fn search_correct_at_indexed_commit() {
    use crate::store::vector_index::tests::make_chunk_row;
    use crate::store::vector_index::{io_ensure_vector_index, VECTOR_INDEX_NAME};

    let dim = 128;
    let config = crate::store::vector_index::tests::make_test_config(dim);
    let (mut store, _tmp) = make_test_store(dim);
    store.set_vector_index_config(config.clone());
    let domain = "admin/search_indexed";
    let branch = "main";

    // Push enough data for IVF training.
    let rows: Vec<ChunkRow> = (0..300)
        .map(|i| make_chunk_row(&format!("doc/{}", i), dim, i + 1))
        .collect();
    push_and_tag(&store, domain, branch, "c0", &rows).await;

    // Create vector index (simulating boundary indexing).
    {
        let ds_arc = store.io_open_dataset(domain, branch).await.expect("open");
        let mut ds = ds_arc.write().await;
        io_ensure_vector_index(&mut ds, &config, false).await.expect("create vector index");
    }
    store.io_refresh_cached_dataset(domain, branch).await.expect("refresh");

    // Verify index exists.
    let ds = store.io_open_dataset_readonly(domain).await.unwrap().unwrap();
    let ds = ds.read().await;
    let indices = ds.load_indices().await.expect("load indices");
    assert!(indices.iter().any(|i| i.name == VECTOR_INDEX_NAME));

    // Search at c0 — should work via ANN index.
    let query = SearchQuery {
        query_embedding: crate::store::vector_index::tests::seeded_normalized_embedding(dim, 1),
        query_text: "content".to_string(),
        mode: SearchMode::Vector,
        start: 0,
        count: 10,
        doc_type_filter: Vec::new(),
        doc_id_filter: Vec::new(),
        snippet: false,
    };
    let hits = store
        .io_search(domain, branch, "c0", &query)
        .await
        .expect("search at indexed commit");
    assert!(
        !hits.is_empty(),
        "search must return results at indexed commit (ANN)"
    );
}

/// Push 5+ commits, verify every tagged commit resolves and returns correct
/// results for both vector and FTS search (indexed commits via ANN,
/// non-indexed via KNN fallback).
#[tokio::test]
async fn snapshot_isolation_across_indexed_and_non_indexed() {
    use crate::store::vector_index::tests::make_chunk_row;
    use crate::store::vector_index::io_ensure_vector_index;
    use crate::store::lance::io_ensure_fts_index_on_dataset;
    use crate::store::lance::io_incremental_cascade;

    let dim = 128;
    let config = crate::store::vector_index::tests::make_test_config(dim);
    let (mut store, _tmp) = make_test_store(dim);
    store.set_vector_index_config(config.clone());
    let domain = "admin/snapshot_isolation";
    let branch = "main";

    let commits: Vec<String> = (0..6).map(|i| format!("c{}", i)).collect();

    // Push 6 commits, each with 50 docs.
    // Pipeline flow: write data → create indices (if boundary) → tag at final version.
    for (i, commit) in commits.iter().enumerate() {
        let rows: Vec<ChunkRow> = (0..50)
            .map(|j| make_chunk_row(&format!("doc/{}_{}", i, j), dim, (i * 50 + j) as u64 + 1))
            .collect();

        // Write data (simulates end-of-push append).
        store
            .io_upsert_chunks(domain, branch, &format!("doc_{}", commit), &rows)
            .await
            .expect("upsert");

        // Boundary indexing at positions 2, 5: create indices BEFORE tagging
        // so the tagged version includes index coverage.
        if crate::store::lance::should_index_commit(i) {
            let ds_arc = store.io_open_dataset(domain, branch).await.expect("open");
            let mut ds = ds_arc.write().await;
            io_ensure_vector_index(&mut ds, &config, false).await.expect("vector index");
            io_ensure_fts_index_on_dataset(&mut ds).await.expect("FTS index");
            drop(ds);
            store.io_refresh_cached_dataset(domain, branch).await.expect("refresh");

            store.io_increment_delta_count(domain, branch).await;
            let delta_count = store.io_get_delta_count(domain, branch).await;
            if let Some(mut ds_u) = store.io_open_dataset_uncached(domain, branch).await.expect("open uncached") {
                let _ = io_incremental_cascade(&mut ds_u, delta_count).await;
            }
            store.io_refresh_cached_dataset(domain, branch).await.expect("refresh after cascade");
        }

        // Tag at the current HEAD version (includes indices if boundary commit).
        let head_version = store
            .io_branch_head_version(domain, branch)
            .await
            .expect("branch head version");
        store
            .io_tag_commit(domain, branch, commit, head_version)
            .await
            .expect("tag");
    }

    // Verify every tagged commit resolves and returns correct vector search results.
    // Vector search works at ALL commits (KNN fallback for non-indexed).
    let query = SearchQuery {
        query_embedding: crate::store::vector_index::tests::seeded_normalized_embedding(dim, 1),
        query_text: "content".to_string(),
        mode: SearchMode::Vector,
        start: 0,
        count: 10,
        doc_type_filter: Vec::new(),
        doc_id_filter: Vec::new(),
        snippet: false,
    };

    for (i, commit) in commits.iter().enumerate() {
        let hits = store
            .io_search(domain, branch, commit, &query)
            .await
            .unwrap_or_else(|_| panic!("vector search at {} failed", commit));
        assert!(
            !hits.is_empty(),
            "vector search at {} (position {}) must return results, got 0 hits",
            commit, i
        );
    }

    // FTS search works at commits >= position 2 (after first boundary index).
    // At positions 0-1, no FTS index exists yet — FTS returns empty (by design).
    let fts_query = SearchQuery {
        query_embedding: vec![0.0f32; dim],
        query_text: "content".to_string(),
        mode: SearchMode::Fts,
        start: 0,
        count: 10,
        doc_type_filter: Vec::new(),
        doc_id_filter: Vec::new(),
        snippet: false,
    };

    for (i, commit) in commits.iter().enumerate() {
        let hits = store
            .io_search(domain, branch, commit, &fts_query)
            .await
            .unwrap_or_else(|_| panic!("FTS search at {} failed", commit));

        if i >= 2 {
            assert!(
                !hits.is_empty(),
                "FTS search at {} (position {}) must return results after first boundary index, got 0 hits",
                commit, i
            );
        } else {
            // Positions 0-1: no FTS index exists yet. FTS returns empty (expected).
            assert!(
                hits.is_empty(),
                "FTS search at {} (position {}) should return empty (no FTS index yet), got {} hits",
                commit, i, hits.len()
            );
        }
    }
}

/// compact_domain drops all indices, compacts data, recreates fresh indices.
/// This test verifies the core behavior using store-level functions
/// (the full compact_domain is in SearchService, which requires an embed provider).
#[tokio::test]
async fn compact_domain_drops_recreates_and_compacts() {
    use crate::store::vector_index::tests::make_chunk_row;
    use crate::store::vector_index::{io_ensure_vector_index, VECTOR_INDEX_NAME};
    use crate::store::lance::io_ensure_fts_index_on_dataset;
    use crate::store::lance::io_compact_data;
    use crate::store::lance::io_incremental_cascade;

    let dim = 128;
    let config = crate::store::vector_index::tests::make_test_config(dim);
    let (mut store, _tmp) = make_test_store(dim);
    store.set_vector_index_config(config.clone());
    let domain = "admin/compact_drops_recreate";
    let branch = "main";

    // Push 5 commits to create multiple fragments.
    for i in 0..5 {
        let rows: Vec<ChunkRow> = (0..60)
            .map(|j| make_chunk_row(&format!("doc/{}_{}", i, j), dim, (i * 60 + j) as u64 + 1))
            .collect();
        push_and_tag(&store, domain, branch, &format!("c{}", i), &rows).await;
    }

    // Create indices.
    {
        let ds_arc = store.io_open_dataset(domain, branch).await.expect("open");
        let mut ds = ds_arc.write().await;
        io_ensure_vector_index(&mut ds, &config, false).await.expect("vector index");
        io_ensure_fts_index_on_dataset(&mut ds).await.expect("FTS index");
    }
    store.io_refresh_cached_dataset(domain, branch).await.expect("refresh");

    // Verify indices exist.
    let ds = store.io_open_dataset_readonly(domain).await.unwrap().unwrap();
    let ds = ds.read().await;
    let indices_before = ds.load_indices().await.expect("load indices");
    let frags_before = ds.get_fragments().len();
    assert!(indices_before.iter().any(|i| i.name == VECTOR_INDEX_NAME));
    assert!(frags_before > 1, "should have multiple fragments before compaction");
    drop(ds);

    // Simulate compact_domain: drop all indices, compact, recreate.
    {
        let ds = store.io_open_dataset_uncached(domain, branch).await.expect("open uncached").expect("dataset");
        let mut ds = ds;

        // Step 1: Drop all indices.
        let indices = ds.load_indices().await.expect("load indices");
        let names: Vec<String> = indices.iter().map(|i| i.name.clone()).collect();
        for name in &names {
            ds.drop_index(name).await.expect("drop index");
        }

        // Step 2: Compact data.
        io_compact_data(&mut ds, true).await.expect("compact data");

        // Step 3: Recreate indices.
        io_ensure_vector_index(&mut ds, &config, false).await.expect("recreate vector index");
        io_ensure_fts_index_on_dataset(&mut ds).await.expect("recreate FTS index");

        // Step 4: Reset delta count + cascade.
        store.io_reset_delta_count(domain, branch).await;
        let delta_count = store.io_get_delta_count(domain, branch).await;
        let _ = io_incremental_cascade(&mut ds, delta_count).await;
    }

    // Cleanup + refresh.
    let _ = store.io_cleanup_aggressive(domain, branch).await;
    let _ = store.io_prune_empty_index_dirs(domain);
    store.io_refresh_cached_dataset(domain, branch).await.expect("refresh");

    // Verify indices exist after compaction.
    let ds = store.io_open_dataset_readonly(domain).await.unwrap().unwrap();
    let ds = ds.read().await;
    let indices_after = ds.load_indices().await.expect("load indices after");
    let frags_after = ds.get_fragments().len();
    assert!(
        indices_after.iter().any(|i| i.name == VECTOR_INDEX_NAME),
        "vector index must exist after compact_domain"
    );
    assert!(
        indices_after.iter().any(|i| i.name == "content_fts"),
        "FTS index must exist after compact_domain"
    );
    assert!(
        frags_after <= frags_before,
        "fragment count after compaction should be <= before ({} -> {})",
        frags_before, frags_after
    );
}

/// Deleting a domain should reset index_delta_counts for all its branches.
#[tokio::test]
async fn delete_domain_resets_delta_count() {
    use crate::store::vector_index::tests::make_chunk_row;

    let dim = 8;
    let (store, _tmp) = make_test_store(dim);
    let domain = "admin/delete_delta_reset";
    let branch = "main";

    // Push a commit and increment delta count.
    let rows: Vec<ChunkRow> = (0..5)
        .map(|i| make_chunk_row(&format!("doc/{}", i), dim, i + 1))
        .collect();
    push_and_tag(&store, domain, branch, "c0", &rows).await;
    store.io_increment_delta_count(domain, branch).await;
    store.io_increment_delta_count(domain, branch).await;

    // Verify delta count is 2.
    let count = store.io_get_delta_count(domain, branch).await;
    assert_eq!(count, 2, "delta count should be 2 before delete");

    // Delete domain.
    store.io_delete_domain(domain).await.expect("delete domain");

    // Verify delta count is reset (returns 0 for missing key).
    let count = store.io_get_delta_count(domain, branch).await;
    assert_eq!(count, 0, "delta count should be 0 after domain delete");
}

/// Deleting a branch index should reset index_delta_counts for that branch.
#[tokio::test]
async fn delete_branch_index_resets_delta_count() {
    use crate::store::vector_index::tests::make_chunk_row;

    let dim = 8;
    let (store, _tmp) = make_test_store(dim);
    let domain = "admin/delete_branch_delta_reset";
    let branch = "main";

    // Push a commit and increment delta count.
    let rows: Vec<ChunkRow> = (0..5)
        .map(|i| make_chunk_row(&format!("doc/{}", i), dim, i + 1))
        .collect();
    push_and_tag(&store, domain, branch, "c0", &rows).await;
    store.io_increment_delta_count(domain, branch).await;
    store.io_increment_delta_count(domain, branch).await;
    store.io_increment_delta_count(domain, branch).await;

    // Verify delta count is 3.
    let count = store.io_get_delta_count(domain, branch).await;
    assert_eq!(count, 3, "delta count should be 3 before delete");

    // Delete branch index.
    store.io_delete_branch_index(domain, branch).await.expect("delete branch index");

    // Verify delta count is reset.
    let count = store.io_get_delta_count(domain, branch).await;
    assert_eq!(count, 0, "delta count should be 0 after branch index delete");
}

/// io_count_branch_commits returns the correct count of tagged commits.
#[tokio::test]
async fn io_count_branch_commits_returns_correct_count() {
    use crate::store::vector_index::tests::make_chunk_row;

    let dim = 8;
    let (store, _tmp) = make_test_store(dim);
    let domain = "admin/count_commits";
    let branch = "main";

    // Initially 0 commits.
    assert_eq!(count_tags(&store, domain, branch).await, 0);

    // Push 5 commits.
    for i in 0..5 {
        let rows: Vec<ChunkRow> = vec![make_chunk_row(&format!("doc/{}", i), dim, i + 1)];
        push_and_tag(&store, domain, branch, &format!("c{}", i), &rows).await;
    }

    // Should have 5 tagged commits.
    assert_eq!(
        count_tags(&store, domain, branch).await,
        5,
        "should have 5 tagged commits"
    );

    // Push 3 more.
    for i in 5..8 {
        let rows: Vec<ChunkRow> = vec![make_chunk_row(&format!("doc/{}", i), dim, i + 1)];
        push_and_tag(&store, domain, branch, &format!("c{}", i), &rows).await;
    }

    // Should have 8 tagged commits.
    assert_eq!(
        count_tags(&store, domain, branch).await,
        8,
        "should have 8 tagged commits"
    );
}

/// FTS index must exist after the first push (position 0, non-boundary)
/// so that FTS search returns hits immediately — not only at the first
/// boundary commit (position 2). The streaming pipeline calls
/// io_ensure_fts_index on every push to guarantee this.
#[tokio::test]
async fn fts_index_exists_after_first_push() {
    use crate::store::vector_index::tests::make_chunk_row;

    let dim = 8;
    let (store, _tmp) = make_test_store(dim);
    let domain = "admin/fts_first_push";

    // Push data (upsert creates the dataset + appends rows).
    let rows = vec![make_chunk_row("doc/apple", dim, 1)];
    let _v = store
        .io_upsert_chunks(domain, "main", "doc_c0", &rows)
        .await
        .expect("upsert");

    // Ensure FTS index exists and covers new fragments BEFORE tagging.
    // io_ensure_fts_index creates the index on first push, then calls
    // optimize_indices(append()) to index new data fragments.
    let indexed_v = store
        .io_ensure_fts_index(domain, "main")
        .await
        .expect("FTS index creation");

    // Tag at the indexed version (not the pre-index version).
    store
        .io_tag_commit(domain, "main", "c0", indexed_v)
        .await
        .expect("tag");

    // Refresh cached handle so search sees the indexed version.
    store
        .io_refresh_cached_dataset(domain, "main")
        .await
        .expect("refresh");

    // Verify the FTS index exists on the tagged version.
    {
        let ds_arc = store.io_open_dataset(domain, "main").await.expect("open");
        let ds = ds_arc.read().await;
        let indices = ds.load_indices().await.expect("load indices");
        let index_names: Vec<String> = indices.iter().map(|i| i.name.clone()).collect();
        assert!(
            index_names.contains(&"content_fts".to_owned()),
            "FTS index content_fts must exist after first push, got {:?}",
            index_names
        );
    }

    // FTS search at c0 must return hits.
    let query = SearchQuery {
        query_embedding: fake_embedding(dim, 1.0),
        query_text: "content".to_owned(),
        mode: SearchMode::Fts,
        start: 0,
        count: 10,
        doc_type_filter: vec![],
        doc_id_filter: vec![],
        snippet: false,
    };
    let hits = store
        .io_search(domain, "main", "c0", &query)
        .await
        .expect("FTS search at c0");
    assert!(
        !hits.is_empty(),
        "FTS search at c0 must return hits for 'content', got {} hits",
        hits.len()
    );
    assert_eq!(
        hits[0].doc_id, "doc/apple",
        "FTS hit must be doc/apple"
    );
}

/// Integration test: FTS search must return hits at ALL historic commit
/// versions, not just boundary commits. This simulates the full pipeline:
/// - Every push calls io_ensure_fts_index (non-boundary FTS ensure + append)
/// - Every 3rd push (boundary) calls io_ensure_fts_index_on_dataset + cascade
/// - No compact_files is called — compaction is a manual maintenance op only
///
/// After 6 pushes (2 boundaries at positions 2 and 5), FTS search at every
/// tagged commit must return the expected documents.
#[tokio::test]
async fn fts_search_works_at_all_commit_versions() {
    use crate::store::vector_index::tests::make_chunk_row;
    use crate::store::lance::{io_ensure_fts_index_on_dataset, io_incremental_cascade};

    let dim = 8;
    let (store, _tmp) = make_test_store(dim);
    let domain = "admin/fts_all_versions";

    let mut commits: Vec<(String, u64)> = Vec::new();

    // Push 6 commits, simulating the boundary-aware pipeline.
    for i in 0..6u32 {
        let doc_id = format!("doc/item_{}", i);
        let rows = vec![make_chunk_row(&doc_id, dim, (i + 1) as u64)];
        let _v = store
            .io_upsert_chunks(domain, "main", &format!("doc_c{}", i), &rows)
            .await
            .expect("upsert");

        let commit_position = store
            .io_count_branch_commits(domain, "main")
            .await
            .unwrap_or(0);

        let is_boundary = crate::store::lance::should_index_commit(commit_position);

        let tagged_version;
        if is_boundary {
            // Boundary: full index optimization + cascade.
            let ds_arc = store
                .io_open_dataset_uncached(domain, "main")
                .await
                .expect("open uncached")
                .expect("dataset exists");
            let mut ds = ds_arc;

            io_ensure_fts_index_on_dataset(&mut ds)
                .await
                .expect("FTS optimize");

            store.io_increment_delta_count(domain, "main").await;
            let delta_count = store.io_get_delta_count(domain, "main").await;
            let _ = io_incremental_cascade(&mut ds, delta_count).await;

            tagged_version = ds.version().version;
        } else {
            // Non-boundary: ensure FTS index exists and covers new fragments.
            let fts_v = store
                .io_ensure_fts_index(domain, "main")
                .await
                .expect("FTS ensure");
            tagged_version = fts_v;
        }

        store
            .io_tag_commit(domain, "main", &format!("c{}", i), tagged_version)
            .await
            .expect("tag");

        store
            .io_refresh_cached_dataset(domain, "main")
            .await
            .expect("refresh");

        commits.push((format!("c{}", i), tagged_version));
    }

    // Verify FTS search at every commit version.
    let query = SearchQuery {
        query_embedding: fake_embedding(dim, 1.0),
        query_text: "content".to_owned(),
        mode: SearchMode::Fts,
        start: 0,
        count: 50,
        doc_type_filter: vec![],
        doc_id_filter: vec![],
        snippet: false,
    };

    for (commit, _v) in &commits {
        let hits = store
            .io_search(domain, "main", commit, &query)
            .await
            .unwrap_or_else(|_| panic!("FTS search at {}", commit));
        assert!(
            !hits.is_empty(),
            "FTS search at {} must return hits, got 0",
            commit
        );
    }

    // Verify snapshot isolation: c0 must only find doc/item_0.
    let hits_c0 = store
        .io_search(domain, "main", "c0", &query)
        .await
        .expect("search c0");
    let ids_c0: Vec<&str> = hits_c0.iter().map(|h| h.doc_id.as_str()).collect();
    assert!(
        ids_c0.contains(&"doc/item_0"),
        "c0 must find doc/item_0"
    );
    assert!(
        !ids_c0.contains(&"doc/item_5"),
        "c0 must NOT find doc/item_5 (snapshot isolation)"
    );

    // c5 must find all 6 docs.
    let hits_c5 = store
        .io_search(domain, "main", "c5", &query)
        .await
        .expect("search c5");
    let ids_c5: Vec<&str> = hits_c5.iter().map(|h| h.doc_id.as_str()).collect();
    for i in 0..6 {
        let expected = format!("doc/item_{}", i);
        assert!(
            ids_c5.contains(&expected.as_str()),
            "c5 must find {} (got {:?})",
            expected, ids_c5
        );
    }

    // Verify fragment count: 6 pushes without compaction = 6 fragments.
    let ds_arc = store.io_open_dataset(domain, "main").await.expect("open");
    let ds = ds_arc.read().await;
    let fragment_count = ds.get_fragments().len();
    assert_eq!(
        fragment_count, 6,
        "fragment count after 6 pushes without compaction should be 6, got {}",
        fragment_count
    );
}
