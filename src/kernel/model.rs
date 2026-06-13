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
}

/// Task status for async push operations.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "status")]
pub enum TaskStatus {
    Pending { percentage: f64 },
    Complete { indexed_documents: u64, skipped: Vec<SkippedDoc> },
    Error { error: String },
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
    pub chunks: u64,
}

/// Duplicate pair.
pub type DuplicatePair = (String, String);

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
}
