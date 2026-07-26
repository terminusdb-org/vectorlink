// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 DFRNT AB

#![forbid(unsafe_code)]

//! Domain types — the vocabulary every module speaks.
//! Pure constructors and validation; no I/O.

use std::fmt;

use serde::{Deserialize, Serialize};

/// A validated database domain identifier (e.g. "admin/star_wars").
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Domain(String);

impl Domain {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Domain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Opaque TerminusDB commit hash.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CommitId(String);

impl CommitId {
    pub fn new(s: String) -> Self {
        Self(s)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Branch name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BranchName(String);

impl BranchName {
    pub fn new(s: String) -> Self {
        Self(s)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Reference kind within a resource path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ref {
    Branch(BranchName),
    Commit(CommitId),
}

/// Fully-qualified resource path (mirrors TerminusDB's graphspec grammar).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourcePath {
    pub org: String,
    pub db: String,
    pub repo: String,
    pub reference: Ref,
}

/// Parse error for resource path / domain normalisation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ParseError {
    #[error("invalid domain: expected at least org/db, got {0:?}")]
    InvalidDomain(String),
    #[error("invalid reference kind: expected 'branch' or 'commit', got {0:?}")]
    InvalidRefKind(String),
    #[error("incomplete reference: missing value after '{0}'")]
    IncompleteRef(String),
}

/// Normalise a domain string into a fully-qualified ResourcePath.
///
/// Follows TerminusDB's `resolve_absolute_descriptor` grammar:
/// - `org/db`                         -> org/db/local/branch/main
/// - `org/db/repo`                    -> org/db/repo/branch/main
/// - `org/db/repo/branch/name`        -> that branch
/// - `org/db/repo/commit/hash`        -> that commit
pub fn parse_domain(input: &str) -> Result<ResourcePath, ParseError> {
    let segments: Vec<&str> = input.split('/').collect();
    match segments.len() {
        2 => Ok(ResourcePath {
            org: segments[0].to_owned(),
            db: segments[1].to_owned(),
            repo: "local".to_owned(),
            reference: Ref::Branch(BranchName::new("main".to_owned())),
        }),
        3 => Ok(ResourcePath {
            org: segments[0].to_owned(),
            db: segments[1].to_owned(),
            repo: segments[2].to_owned(),
            reference: Ref::Branch(BranchName::new("main".to_owned())),
        }),
        5 => {
            let ref_kind = segments[3];
            let ref_value = segments[4];
            match ref_kind {
                "branch" => Ok(ResourcePath {
                    org: segments[0].to_owned(),
                    db: segments[1].to_owned(),
                    repo: segments[2].to_owned(),
                    reference: Ref::Branch(BranchName::new(ref_value.to_owned())),
                }),
                "commit" => Ok(ResourcePath {
                    org: segments[0].to_owned(),
                    db: segments[1].to_owned(),
                    repo: segments[2].to_owned(),
                    reference: Ref::Commit(CommitId::new(ref_value.to_owned())),
                }),
                other => Err(ParseError::InvalidRefKind(other.to_owned())),
            }
        }
        4 => {
            // Could be org/db/repo/branch or org/db/repo/commit — ambiguous without keyword.
            // Treat segments[3] as incomplete reference specifier.
            let ref_kind = segments[3];
            if ref_kind == "branch" || ref_kind == "commit" {
                Err(ParseError::IncompleteRef(ref_kind.to_owned()))
            } else {
                Err(ParseError::InvalidDomain(input.to_owned()))
            }
        }
        _ => Err(ParseError::InvalidDomain(input.to_owned())),
    }
}

/// Construct a Domain from a raw validated string (used after parse_domain succeeds).
impl Domain {
    pub fn from_resource_path(rp: &ResourcePath) -> Self {
        Self(format!("{}/{}", rp.org, rp.db))
    }
}

/// Search mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    Vector,
    Fts,
    Hybrid,
}

impl SearchMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "vector" => Some(Self::Vector),
            "fts" => Some(Self::Fts),
            "hybrid" => Some(Self::Hybrid),
            _ => None,
        }
    }
}

/// NDJSON push operation (one line of the push body).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum Operation {
    Inserted { id: String, string: String },
    Changed { id: String, string: String },
    Deleted { id: String },
    Error { message: String },
    Abort,
}

/// Task status for async push operations.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "status")]
pub enum TaskStatus {
    Pending { percentage: f64 },
    Complete { indexed_documents: u64, skipped: Vec<SkippedDoc> },
    Error { error: String },
}

/// Progress update sent via the streaming push response (NDJSON, one line per update).
///
/// When `stream=true` is sent to `POST /push`, the response body is an NDJSON stream
/// of `ProgressUpdate` objects. The pipeline emits `Progress` after each embedding
/// batch, then a terminal `Complete`, `Error`, or `Aborted` as the final line.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "status", rename_all = "lowercase")]
pub enum ProgressUpdate {
    /// Incremental progress: docs embedded so far.
    Progress {
        indexed: u64,
        total_seen: u64,
        skipped: u64,
    },
    /// Terminal success: all docs processed, written to Lance, commit tagged.
    Complete {
        indexed_documents: u64,
        skipped: Vec<SkippedDoc>,
    },
    /// Terminal error: pipeline failed (e.g. Lance write error).
    Error {
        error: String,
    },
    /// Terminal abort: client sent `{"op":"Abort"}` in the NDJSON request body.
    Aborted,
}

/// A document skipped during indexing.
#[derive(Debug, Clone, Serialize)]
pub struct SkippedDoc {
    pub id: String,
    pub message: String,
}

/// Search result hit (frozen wire shape).
#[derive(Debug, Clone, Serialize)]
pub struct SearchHit {
    pub id: String,
    pub distance: f32,
    pub chunk: ChunkInfo,
}

/// Chunk location metadata within a search hit.
#[derive(Debug, Clone, Serialize)]
pub struct ChunkInfo {
    pub index: u32,
    pub count: u32,
    pub token_start: u32,
    pub doc_token_len: u32,
    pub location: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
}

/// Last-indexed response.
#[derive(Debug, Clone, Serialize)]
pub struct LastIndexed {
    pub branch: String,
    pub commit: Option<String>,
    pub version: u64,
}

/// Engine statistics.
#[derive(Debug, Clone, Serialize)]
pub struct Statistics {
    pub domains: u64,
    pub branches: u64,
    pub indexed_commits: u64,
    pub documents: u64,
    /// Live chunks (excludes soft-deleted rows). This is the accurate count
    /// of searchable text segments.
    pub chunks: u64,
    /// Number of data fragments not yet covered by the vector ANN index.
    /// Reflects the indexing backlog — higher values mean more fragments are
    /// being flat-scanned (correct but slower). Returns to 0 after
    /// `optimize_indices(append())` drains the queue.
    pub pending_index_fragments: u64,
    /// Total rows (chunks) in fragments not yet covered by the vector ANN index.
    /// Measures flat-scan cost in rows, complementing `pending_index_fragments`
    /// which measures it in fragment count. Returns to 0 after
    /// `optimize_indices(append())` drains the queue.
    pub pending_index_documents: u64,
    /// Whether clustering embeddings are stored for this domain (domain-scoped only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store_clustering: Option<bool>,
}

/// Internal instrumentation stats — sizes of in-memory data structures.
/// Exposed via the global /statistics endpoint (no domain filter) for
/// memory leak monitoring. Not included in domain-scoped stats.
#[derive(Debug, Clone, Serialize)]
pub struct InternalStats {
    /// Number of cached dataset handles (Lance Dataset objects).
    pub cached_datasets: usize,
    /// Number of tracked tasks in the tasks HashMap.
    pub tasks: usize,
    /// Number of per-(domain, branch) branch index entries.
    pub branch_indexes: usize,
    /// Number of per-(domain, branch) pipeline locks.
    pub pipeline_locks: usize,
    /// Number of per-domain guard mutexes.
    pub domain_guards: usize,
    /// Number of in-flight commit reservations.
    pub inflight_commits: usize,
    /// Pipeline progress: chunks chunked but not yet embedded.
    pub pipeline_pending_chunks: u64,
    /// Pipeline progress: chunks embedded but not yet written to Lance.
    pub pipeline_embedded_chunks: u64,
    /// Pipeline progress: chunks written to Lance (append committed).
    pub pipeline_written_chunks: u64,
    /// Number of pipeline tasks currently running (spawned but not completed).
    pub pipeline_active_tasks: u64,
    /// Total count of fresh Dataset::open calls (cumulative counter).
    pub fresh_open_count: u64,
    /// Number of entries in the embedding cache.
    pub embed_cache_entries: usize,
    /// Approximate memory usage of the embedding cache in bytes.
    pub embed_cache_size_bytes: usize,
    /// Configured Lance index cache capacity in bytes.
    pub lance_index_cache_capacity_bytes: usize,
    /// Configured Lance metadata cache capacity in bytes.
    pub lance_metadata_cache_capacity_bytes: usize,
}

/// A single member of a near-duplicate group: the document id, and its chunk
/// text when `snippet=true` was requested.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DuplicateMember {
    pub id: String,
    /// The matched chunk's text. Present only when `snippet=true` was requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
}

/// A near-duplicate group: a symmetric set of member documents whose best chunks
/// are within `threshold`, with the [0, 1] cosine `distance` of the best pair of
/// chunks. Currently always two members (a pair); the shape is `group` (a
/// symmetric array) rather than `a`/`b` so it extends to clusters of >2 without a
/// wire-shape change.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DuplicateGroup {
    pub group: Vec<DuplicateMember>,
    pub distance: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_domain_org_db_defaults_local_main() {
        let rp = parse_domain("admin/star_wars").unwrap();
        assert_eq!(rp.org, "admin");
        assert_eq!(rp.db, "star_wars");
        assert_eq!(rp.repo, "local");
        assert_eq!(rp.reference, Ref::Branch(BranchName::new("main".to_owned())));
    }

    #[test]
    fn parse_domain_with_repo_defaults_branch_main() {
        let rp = parse_domain("org/db/myrepo").unwrap();
        assert_eq!(rp.repo, "myrepo");
        assert_eq!(rp.reference, Ref::Branch(BranchName::new("main".to_owned())));
    }

    #[test]
    fn parse_domain_explicit_branch() {
        let rp = parse_domain("org/db/local/branch/dev").unwrap();
        assert_eq!(rp.reference, Ref::Branch(BranchName::new("dev".to_owned())));
    }

    #[test]
    fn parse_domain_explicit_commit() {
        let rp = parse_domain("org/db/local/commit/abc123").unwrap();
        assert_eq!(rp.reference, Ref::Commit(CommitId::new("abc123".to_owned())));
    }

    #[test]
    fn parse_domain_single_segment_fails() {
        assert!(parse_domain("onlyone").is_err());
    }

    #[test]
    fn parse_domain_invalid_ref_kind() {
        let result = parse_domain("org/db/local/tag/v1");
        assert!(matches!(result, Err(ParseError::InvalidRefKind(_))));
    }

    #[test]
    fn parse_domain_incomplete_ref() {
        let result = parse_domain("org/db/local/branch");
        assert!(matches!(result, Err(ParseError::IncompleteRef(_))));
    }

    // ── ProgressUpdate serialization tests ──

    #[test]
    fn progress_update_progress_serializes_correctly() {
        let update = ProgressUpdate::Progress {
            indexed: 32,
            total_seen: 64,
            skipped: 1,
        };
        let json = serde_json::to_string(&update).expect("serialize");
        assert_eq!(
            json,
            r#"{"status":"progress","indexed":32,"total_seen":64,"skipped":1}"#
        );
    }

    #[test]
    fn progress_update_complete_serializes_correctly() {
        let update = ProgressUpdate::Complete {
            indexed_documents: 100,
            skipped: vec![SkippedDoc {
                id: "doc/bad".to_owned(),
                message: "chunking failed: empty".to_owned(),
            }],
        };
        let json = serde_json::to_string(&update).expect("serialize");
        assert_eq!(
            json,
            r#"{"status":"complete","indexed_documents":100,"skipped":[{"id":"doc/bad","message":"chunking failed: empty"}]}"#
        );
    }

    #[test]
    fn progress_update_complete_with_empty_skipped() {
        let update = ProgressUpdate::Complete {
            indexed_documents: 50,
            skipped: vec![],
        };
        let json = serde_json::to_string(&update).expect("serialize");
        assert_eq!(
            json,
            r#"{"status":"complete","indexed_documents":50,"skipped":[]}"#
        );
    }

    #[test]
    fn progress_update_error_serializes_correctly() {
        let update = ProgressUpdate::Error {
            error: "batch delete-append failed: I/O error".to_owned(),
        };
        let json = serde_json::to_string(&update).expect("serialize");
        assert_eq!(
            json,
            r#"{"status":"error","error":"batch delete-append failed: I/O error"}"#
        );
    }

    #[test]
    fn progress_update_aborted_serializes_correctly() {
        let update = ProgressUpdate::Aborted;
        let json = serde_json::to_string(&update).expect("serialize");
        assert_eq!(json, r#"{"status":"aborted"}"#);
    }

    #[test]
    fn progress_update_progress_zero_values() {
        let update = ProgressUpdate::Progress {
            indexed: 0,
            total_seen: 0,
            skipped: 0,
        };
        let json = serde_json::to_string(&update).expect("serialize");
        assert_eq!(
            json,
            r#"{"status":"progress","indexed":0,"total_seen":0,"skipped":0}"#
        );
    }
}
