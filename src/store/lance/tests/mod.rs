use super::*;
use super::config::largest_divisor_leq;
use super::search::{batches_to_vector_hits, rrf_merge};
use crate::kernel::model::{BranchName, Domain, DuplicateGroup, SearchMode};

fn make_test_store(dim: usize) -> (LanceStore, tempfile::TempDir) {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let store = LanceStore::new(tmp.path(), dim);
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
    LanceStore::new(tmp.path(), dim)
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
        },
    ];

    // FTS ranked: B (best), C, D (new — only in FTS)
    let fts_hits = vec![
        ChunkHit {
            doc_id: "B".to_owned(),
            distance: 0.1, // FTS distance (from BM25 conversion).
            distance_kind: DistanceKind::Normalised,
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            content: "b".to_owned(),
            embedding: Vec::new(),
        },
        ChunkHit {
            doc_id: "C".to_owned(),
            distance: 0.2,
            distance_kind: DistanceKind::Normalised,
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            content: "c".to_owned(),
            embedding: Vec::new(),
        },
        ChunkHit {
            doc_id: "D".to_owned(),
            distance: 0.3,
            distance_kind: DistanceKind::Normalised,
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            content: "d".to_owned(),
            embedding: Vec::new(),
        },
    ];

    let merged = rrf_merge(vector_hits, fts_hits);

    // B should be ranked highest: rank 2 in vector + rank 1 in FTS
    // = 1/(60+2) + 1/(60+1) = 1/62 + 1/61
    assert_eq!(merged[0].doc_id, "B", "B should rank first (appears high in both lists)");

    // All 4 unique docs should appear.
    let ids: Vec<&str> = merged.iter().map(|h| h.doc_id.as_str()).collect();
    assert!(ids.contains(&"A"));
    assert!(ids.contains(&"B"));
    assert!(ids.contains(&"C"));
    assert!(ids.contains(&"D"));
    assert_eq!(ids.len(), 4);
}

// Fix #4: VectorIndexConfig dimension validation and divisor guarantee.

#[test]
fn largest_divisor_leq_standard_dims() {
    // 768-d (nomic-embed-v2): target = 768/16 = 48, 768%48 = 0 → 48
    assert_eq!(largest_divisor_leq(768, 48), 48);
    // 128-d: target = 128/8 = 16, 128%16 = 0 → 16
    assert_eq!(largest_divisor_leq(128, 16), 16);
    // 384-d: target = 384/16 = 24, 384%24 = 0 → 24
    assert_eq!(largest_divisor_leq(384, 24), 24);
    // 1536-d: target = 1536/16 = 96, 1536%96 = 0 → 96
    assert_eq!(largest_divisor_leq(1536, 96), 96);
}

#[test]
fn largest_divisor_leq_non_standard_dims() {
    // 500-d: target = 500/16 = 31. 500%31 = 500-31*16 = 500-496 = 4 ≠ 0.
    // Largest divisor of 500 <= 31: 500 = 2^2 * 5^3. Divisors: 1,2,4,5,10,20,25,50,100...
    // 25 <= 31 and 500%25 = 0.
    assert_eq!(largest_divisor_leq(500, 31), 25);
    // 130-d: target = 130/8 = 16. 130%16 = 2 ≠ 0.
    // Divisors of 130: 1,2,5,10,13,26,65,130. Largest <= 16: 13.
    assert_eq!(largest_divisor_leq(130, 16), 13);
}

#[test]
fn largest_divisor_leq_prime_dim() {
    // 127 is prime. Target = 127/8 = 15. Only divisors: 1, 127.
    // Largest <= 15: 1.
    assert_eq!(largest_divisor_leq(127, 15), 1);
}

#[test]
fn vector_index_config_guarantees_divisibility() {
    // Test various dimensions — all must produce num_sub_vectors that divides dim.
    let test_dims = [128, 256, 384, 500, 512, 768, 1024, 1536, 130, 127, 100, 200];
    for dim in test_dims {
        let config = VectorIndexConfig::default_for_dim(dim);
        assert_eq!(
            dim % config.num_sub_vectors, 0,
            "dim={} must be divisible by num_sub_vectors={} (got remainder {})",
            dim, config.num_sub_vectors, dim % config.num_sub_vectors
        );
        assert!(
            config.num_sub_vectors >= 1,
            "num_sub_vectors must be at least 1 for dim={}",
            dim
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
    // IVF_PQ needs >= 256 training vectors; a few partitions for a small corpus.
    let config = VectorIndexConfig {
        num_partitions: 4,
        num_sub_vectors: 8,
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
        crate::store::vector_index::io_ensure_vector_index(&mut ds, &config)
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
        num_sub_vectors: 8,
        nprobes: 4,
        refine_factor: Some(10),
    };
    let (mut store, _tmp) = make_test_store(dim);
    store.set_vector_index_config(config.clone());
    let domain = "admin/fdbranch";
    let branch = "feature";

    // Seed main with one 300-chunk doc (above the 256 IVF_PQ training floor),
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
        crate::store::vector_index::io_ensure_vector_index(&mut branch_ds, &config)
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
        num_sub_vectors: 8,
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
        crate::store::vector_index::io_ensure_vector_index(&mut ds, &config)
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
