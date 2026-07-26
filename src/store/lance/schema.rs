// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 DFRNT AB

//! Types and constants for the Lance store schema.

use crate::kernel::model::SearchMode;

/// A chunk row ready for insertion into Lance.
#[derive(Debug, Clone)]
pub struct ChunkRow {
    pub doc_id: String,
    pub doc_type: String,
    pub chunk_index: i32,
    pub chunk_count: i32,
    pub chunk_token_start: i32,
    pub doc_token_len: i32,
    pub embedding: Vec<f32>,
    /// Clustering-role embedding (clustering: prefix). STORED but NOT INDEXED by
    /// default — the only ANN index remains on `embedding`. When clustering is
    /// enabled for the domain, an ANN index is also created on this column.
    /// Populated with zeros when `store_clustering` is disabled.
    pub clustering_embedding: Vec<f32>,
    pub content: String,
}

/// Search query parameters.
#[derive(Debug, Clone)]
pub struct SearchQuery {
    pub query_embedding: Vec<f32>,
    pub query_text: String,
    pub mode: SearchMode,
    pub start: usize,
    pub count: usize,
    pub doc_type_filter: Vec<String>,
    pub doc_id_filter: Vec<String>,
    pub snippet: bool,
}

/// Suggest (typeahead) query parameters.
/// FTS-only (no embedding), optimised for sub-100ms partial-query responses.
#[derive(Debug, Clone)]
pub struct SuggestQuery {
    pub query_text: String,
    pub count: usize,
    pub doc_type_filter: Vec<String>,
    pub doc_id_filter: Vec<String>,
}

/// A document-level suggest hit with snippet and next-words for smart compose.
/// Suggest is a typeahead endpoint — ordering is meaningful, distance is not.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SuggestHit {
    pub id: String,
    /// The matched chunk content snippet.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
    /// Start byte offset of the query match within `snippet` (for UI highlighting).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub match_start: Option<usize>,
    /// End byte offset of the query match within `snippet` (for UI highlighting).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub match_end: Option<usize>,
    /// Likely next words after the query match, ordered by proximity.
    /// The first entry is the most likely next word. A tab key can cycle
    /// through these to advance the cursor word by word.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub next_words: Vec<String>,
}

/// Result of a suggest (typeahead) operation.
#[derive(Debug, Clone)]
pub struct SuggestResult {
    pub total_approx: usize,
    pub completions: Vec<String>,
    pub hits: Vec<SuggestHit>,
}

/// Whether a ChunkHit's distance is raw (needs transform) or already normalised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceKind {
    /// Raw Lance cosine distance [0, 2] — needs `normalized_cosine_from_lance`.
    RawCosine,
    /// Already normalised to [0, 1] (e.g., from RRF or FTS conversion).
    Normalised,
}

/// Internal chunk-level hit before dedup to documents.
#[derive(Debug, Clone)]
pub struct ChunkHit {
    pub doc_id: String,
    pub distance: f32,
    pub distance_kind: DistanceKind,
    pub chunk_index: i32,
    pub chunk_count: i32,
    pub chunk_token_start: i32,
    pub doc_token_len: i32,
    pub content: String,
    /// The L2-normalised DOCUMENT-role embedding vector as STORED at insert time.
    /// Populated only by the plain doc-chunk lookup path (`io_lookup_doc_chunks`),
    /// where it is projected from the snapshot so `/similar` can reuse the stored
    /// vector directly instead of re-embedding the source text. Ranked search/FTS
    /// paths leave this empty (they rank by `_distance`/`_score`, not the raw vector).
    pub embedding: Vec<f32>,
    /// The L2-normalised CLUSTERING-role embedding vector as STORED at insert
    /// time. Populated only by the plain doc-chunk lookup path
    /// (`io_lookup_doc_chunks`). Ranked search/FTS paths leave this empty.
    pub clustering_embedding: Vec<f32>,
}

/// Default Lance branch name. Layout (A) maps TerminusDB's `main` branch to
/// the Lance dataset's native default branch.
pub const MAIN_BRANCH: &str = "main";

/// Reserved prefix for internal tdb-search branches. The `.-` namespace is
/// reserved for internal use and must not be accepted from external requests
/// (push, search, compact, etc.).
pub const RESERVED_PREFIX: &str = ".-";

/// Reserved prefix for rebuild branches used by delta-fork retagging during
/// compaction. Each compaction creates a fresh branch named
/// `.-compact_rebuild_<epoch>` to avoid colliding with the previous
/// compaction's branch (which still owns live tagged versions). After
/// retagging, older epoch branches are deleted. If the process crashes,
/// an unreferenced branch lingers until startup cleanup.
pub const COMPACT_REBUILD_PREFIX: &str = ".-compact_rebuild_";

/// Generate a rebuild branch name with an epoch component.
pub fn compact_rebuild_branch_name(epoch: u64) -> String {
    format!("{}{}", COMPACT_REBUILD_PREFIX, epoch)
}

/// Check whether a branch name belongs to the reserved rebuild namespace.
pub fn is_compact_rebuild_branch(branch: &str) -> bool {
    branch.starts_with(COMPACT_REBUILD_PREFIX)
}

/// Check whether a branch name is in the reserved `.-` namespace.
/// Branch names starting with `.-` are internal to tdb-search and must not
/// be accepted from external requests.
pub fn is_reserved_branch_name(branch: &str) -> bool {
    branch.starts_with(RESERVED_PREFIX)
}

/// Hard upper bound on the number of indexed points (chunk vectors) the
/// near-duplicate scan will consider in a single snapshot. The scan issues ONE
/// ANN `nearest(k=2)` query per point — O(n) cheap indexed queries, not O(n²) —
/// but `n` is still bounded so a pathologically large corpus can never trigger an
/// unbounded run. A snapshot with more points than this is rejected (fail-loud)
/// rather than silently truncated, so duplicate results are never quietly partial.
/// The requester needs to chunk ID:s if necessary.
pub const DEFAULT_DUPLICATE_MAX_POINTS: usize = 50_000;

/// Hard upper bound on the number of distinct document pairs the near-duplicate
/// detector will collect before stopping. Bounds output size (and memory)
/// independently of the point cap. The requester needs to chunk if necessary.
pub const DEFAULT_DUPLICATE_MAX_PAIRS: usize = 10_000;

/// Scope of a near-duplicate scan (Spec 02/13 set/target model).
///
/// The `set_*` filters define the POPULATION whose documents we look for
/// near-duplicates within. The `target_*` filters, when non-empty, define a
/// SECOND population: each set member's nearest neighbour is then restricted to
/// the target, so every emitted pair STRADDLES set↔target (cross-catalogue
/// entity resolution). When the target filters are empty, the scan is within-set
/// dedup: each set member's nearest neighbour is any OTHER document in the set.
///
/// Filters are doc-type and/or doc-id IN-lists (the same machinery `/search` and
/// `/similar` use). Empty `set_*` means "the whole snapshot" for the set side.
#[derive(Debug, Clone, Default)]
pub struct DuplicateScope {
    pub set_doc_types: Vec<String>,
    pub set_doc_ids: Vec<String>,
    pub target_doc_types: Vec<String>,
    pub target_doc_ids: Vec<String>,
}

impl DuplicateScope {
    /// True when a distinct target population is configured (cross-set mode).
    pub(super) fn has_target(&self) -> bool {
        !self.target_doc_types.is_empty() || !self.target_doc_ids.is_empty()
    }
}

/// A single nearest-neighbour observation between two chunk vectors, with the
/// distance already normalised to the reference [0, 1] cosine scale
/// (`normalized_cosine_from_lance`). `doc_a`/`doc_b` are the owning document ids
/// of the two chunks (they MAY be equal — the pairing step discards same-document
/// observations so a document is never reported as a duplicate of itself).
///
/// `content_a`/`content_b` carry the two chunks' text so the pairing step can
/// surface a snippet per member when requested; they are `None` when snippets
/// were not collected (snippet=false), keeping the scan cheap.
#[derive(Debug, Clone, PartialEq)]
pub struct NeighbourObservation {
    pub doc_a: String,
    pub doc_b: String,
    pub normalized_distance: f32,
    pub content_a: Option<String>,
    pub content_b: Option<String>,
}

/// The two directional cross-NN maps for entity resolution.
pub struct ResolveNeighbourMaps {
    /// For each set doc_id: its top-K neighbours in the target population.
    pub set_to_target:
        std::collections::HashMap<String, Vec<crate::resolve::Neighbour>>,
    /// For each target doc_id: its top-K neighbours in the set population.
    pub target_to_set:
        std::collections::HashMap<String, Vec<crate::resolve::Neighbour>>,
}
