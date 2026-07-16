// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 DFRNT AB

#![forbid(unsafe_code)]

//! Layer index — commit→version mapping via Lance tags.
//!
//! The single authority for resolving a commit to a Lance dataset version.
//! Currently supports linear history (single branch per domain); future work
//! adds branch-out for multi-branch isolation.
//!
//! Includes the pure encode/decode helpers for commit-id → Lance tag-name
//! encoding (reversible, collision-free, restricted to Lance's allowed alphabet).

use thiserror::Error;

/// Errors from the layer index module.
#[derive(Debug, Error)]
pub enum LayerIdxError {
    #[error("layer index error: {0}")]
    Internal(String),
    #[error("tag decode error: {0}")]
    DecodeError(String),
}

/// Encode a commit id into a Lance-safe tag name.
///
/// Lance tag names must be non-empty, contain only [A-Za-z0-9 . - _],
/// and must not start/end with '.', contain '..', or end with '.lock'.
///
/// Scheme: prefix "c_" + per-byte encoding:
/// - [A-Za-z0-9_] → verbatim
/// - '-' → "--"
/// - anything else → "-HH" (lowercase hex of the byte)
///
/// This guarantees the tag name stays within [A-Za-z0-9_-], is non-empty,
/// never starts with '.', and is exactly reversible.
pub fn encode_commit_tag(commit: &str) -> String {
    let mut out = String::with_capacity(commit.len() + 4);
    out.push_str("c_");
    for &b in commit.as_bytes() {
        let c = b as char;
        if c == '-' {
            out.push_str("--");
        } else if c.is_ascii_alphanumeric() || c == '_' {
            out.push(c);
        } else {
            out.push_str(&format!("-{:02x}", b));
        }
    }
    out
}

/// Decode a Lance tag name back to the original commit id.
///
/// Inverse of `encode_commit_tag`. Returns an error if the tag
/// does not have the "c_" prefix or contains malformed escape sequences.
pub fn decode_commit_tag(tag: &str) -> Result<String, LayerIdxError> {
    let s = tag
        .strip_prefix("c_")
        .ok_or_else(|| LayerIdxError::DecodeError("missing 'c_' prefix".to_owned()))?;

    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'-' {
            if i + 1 >= bytes.len() {
                return Err(LayerIdxError::DecodeError(format!(
                    "truncated escape at position {}",
                    i
                )));
            }
            if bytes[i + 1] == b'-' {
                // "--" → literal '-'
                out.push(b'-');
                i += 2;
            } else {
                // "-HH" → byte from hex
                if i + 2 >= bytes.len() {
                    return Err(LayerIdxError::DecodeError(format!(
                        "truncated hex escape at position {}",
                        i
                    )));
                }
                let hex_str = std::str::from_utf8(&bytes[i + 1..i + 3]).map_err(|_| {
                    LayerIdxError::DecodeError(format!("invalid UTF-8 in hex at position {}", i))
                })?;
                let byte_val = u8::from_str_radix(hex_str, 16).map_err(|_| {
                    LayerIdxError::DecodeError(format!(
                        "invalid hex '{}' at position {}",
                        hex_str, i
                    ))
                })?;
                out.push(byte_val);
                i += 3;
            }
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }

    String::from_utf8(out)
        .map_err(|e| LayerIdxError::DecodeError(format!("decoded bytes not valid UTF-8: {}", e)))
}

/// Per-branch last-indexed tracking (in-memory, persisted by the store layer).
#[derive(Debug, Clone, Default)]
pub struct BranchIndex {
    pub commit: Option<String>,
    pub version: u64,
}

// ─────────────────────────── Nearest-ancestor resolution ──────────────────

/// Outcome of resolving a requested commit to a searchable layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolvedLayer {
    /// The requested commit is itself indexed — exact, not stale.
    Exact { commit: String, version: u64 },
    /// The requested commit is not indexed; the nearest indexed ancestor is
    /// served instead (STALE — `served_commit` ≠ requested).
    Ancestor { served_commit: String, version: u64 },
    /// No indexed ancestor found within the supplied window — 404.
    None,
}

/// Resolve a requested commit to its searchable layer using the progressive
/// ancestor window (Spec 10 §5).
///
/// `requested_commit` — the commit the caller asked to search.
/// `ancestors` — ordered nearest-first ancestor commit ids supplied by
///   TerminusDB (it owns the DAG); the window grows 10 → 1000 across calls, but
///   resolution itself is a pure walk over whatever list is supplied here.
/// `resolve` — maps a commit id to its indexed version, if any (the layer
///   index / dataset-global tag lookup).
///
/// Returns `Exact` if the requested commit itself resolves; else the FIRST
/// ancestor (nearest-first) that resolves as `Ancestor` (stale); else `None`.
///
/// Pure control flow over an injected async resolver — no I/O of its own, so it
/// is unit-testable with an in-memory resolver and exercised end-to-end with the
/// real tag lookup.
pub async fn resolve_nearest_layer<F, Fut>(
    requested_commit: &str,
    ancestors: &[String],
    mut resolve: F,
) -> Result<ResolvedLayer, LayerIdxError>
where
    F: FnMut(String) -> Fut,
    Fut: std::future::Future<Output = Result<Option<u64>, LayerIdxError>>,
{
    // 1. Exact match first.
    if let Some(version) = resolve(requested_commit.to_owned()).await? {
        return Ok(ResolvedLayer::Exact {
            commit: requested_commit.to_owned(),
            version,
        });
    }

    // 2. Walk ancestors nearest-first; serve the first that resolves.
    for ancestor in ancestors {
        if let Some(version) = resolve(ancestor.clone()).await? {
            return Ok(ResolvedLayer::Ancestor {
                served_commit: ancestor.clone(),
                version,
            });
        }
    }

    // 3. No indexed ancestor in the supplied window.
    Ok(ResolvedLayer::None)
}

// Per-branch enablement + the 404 negative cache that used to live here have
// been REMOVED (task-durable-index-state §6). Index state — "is this commit
// indexed", "does this branch have indexed lineage", "what is the last-indexed
// commit" — is now derived from the durable on-disk Lance tags (see
// `LanceStore::io_derive_last_indexed` / `last_indexed`). The negative cache was
// the source of the restart-loses-state bug (it cached "no lineage" for a branch
// whose index was on disk) and, on the now-fast durable lookup path, guarded
// nothing worth keeping. Removing it makes the restart invariant trivially true.

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // ───────────────────── nearest-ancestor resolution ─────────────────────

    /// Build an in-memory resolver over a commit→version map.
    fn map_resolver(
        map: HashMap<String, u64>,
    ) -> impl FnMut(String) -> std::future::Ready<Result<Option<u64>, LayerIdxError>> {
        move |commit: String| std::future::ready(Ok(map.get(&commit).copied()))
    }

    // --- exact match wins (not stale) ---
    #[tokio::test]
    async fn nearest_layer_exact_match() {
        let map = HashMap::from([("c2".to_owned(), 5u64)]);
        let ancestors = vec!["c1".to_owned(), "c0".to_owned()];
        let resolved = resolve_nearest_layer("c2", &ancestors, map_resolver(map))
            .await
            .unwrap();
        assert_eq!(
            resolved,
            ResolvedLayer::Exact { commit: "c2".to_owned(), version: 5 }
        );
    }

    // --- nearest indexed ancestor served when requested is not indexed ---
    #[tokio::test]
    async fn nearest_layer_serves_nearest_ancestor() {
        // c3 requested (not indexed); ancestors nearest-first: c2 (not indexed),
        // c1 (indexed v7), c0 (indexed v3). Must serve c1 — the FIRST that resolves.
        let map = HashMap::from([("c1".to_owned(), 7u64), ("c0".to_owned(), 3u64)]);
        let ancestors = vec!["c2".to_owned(), "c1".to_owned(), "c0".to_owned()];
        let resolved = resolve_nearest_layer("c3", &ancestors, map_resolver(map))
            .await
            .unwrap();
        assert_eq!(
            resolved,
            ResolvedLayer::Ancestor { served_commit: "c1".to_owned(), version: 7 }
        );
    }

    // --- no indexed ancestor → None (caller turns this into a cached 404) ---
    #[tokio::test]
    async fn nearest_layer_none_when_no_ancestor_indexed() {
        let map: HashMap<String, u64> = HashMap::new();
        let ancestors = vec!["c1".to_owned(), "c0".to_owned()];
        let resolved = resolve_nearest_layer("c2", &ancestors, map_resolver(map))
            .await
            .unwrap();
        assert_eq!(resolved, ResolvedLayer::None);
    }

    // --- BLOCKER-1: requesting an OLDER un-indexed commit must NOT resolve to a
    //     NEWER indexed commit that is NOT in the ancestor window. The newer tip
    //     `cNew` is indexed, but it is a DESCENDANT of the requested `cOld`, so
    //     it is absent from cOld's ancestor window — resolution must be None,
    //     never `cNew` (no snapshot-isolation leak). ---
    #[tokio::test]
    async fn nearest_layer_never_serves_newer_non_ancestor() {
        // Only the NEWER commit is indexed. The requested OLDER commit's honest
        // ancestor window contains only older roots (none indexed here).
        let map = HashMap::from([("cNew".to_owned(), 42u64)]);
        let ancestors = vec!["cOlderRoot".to_owned()];
        let resolved = resolve_nearest_layer("cOld", &ancestors, map_resolver(map))
            .await
            .unwrap();
        assert_eq!(
            resolved,
            ResolvedLayer::None,
            "must NOT serve the newer indexed tip for an older request — it is a descendant, not an ancestor"
        );
    }

    // --- BLOCKER-1: with NO ancestor window, a non-exact commit cannot be
    //     proven to descend from any indexed commit → None (the engine 404s
    //     rather than serving the tip). ---
    #[tokio::test]
    async fn nearest_layer_empty_window_non_exact_is_none() {
        let map = HashMap::from([("cIndexed".to_owned(), 7u64)]);
        let ancestors: Vec<String> = vec![];
        let resolved = resolve_nearest_layer("cUnknown", &ancestors, map_resolver(map))
            .await
            .unwrap();
        assert_eq!(resolved, ResolvedLayer::None);
    }

    // --- encode/decode round-trip: normal commit ids ---
    #[test]
    fn encode_decode_simple_hex_commit() {
        let commit = "o2uq7k1mrun1vp4urktmw55962vlpto";
        let tag = encode_commit_tag(commit);
        assert_eq!(tag, "c_o2uq7k1mrun1vp4urktmw55962vlpto");
        let decoded = decode_commit_tag(&tag).unwrap();
        assert_eq!(decoded, commit);
    }

    // --- encode/decode round-trip: adversarial commit id with special chars ---
    #[test]
    fn encode_decode_adversarial_commit() {
        let commit = "branch:feature/v.1..2";
        let tag = encode_commit_tag(commit);
        // Should NOT contain any raw ':', '/', '.' characters.
        assert!(!tag.contains(':'));
        assert!(!tag.contains('/'));
        assert!(!tag.contains('.'));
        let decoded = decode_commit_tag(&tag).unwrap();
        assert_eq!(decoded, commit);
    }

    // --- encode/decode: dash in commit id ---
    #[test]
    fn encode_decode_dash_commit() {
        let commit = "abc-def-123";
        let tag = encode_commit_tag(commit);
        assert!(tag.contains("--")); // dash is escaped as --
        let decoded = decode_commit_tag(&tag).unwrap();
        assert_eq!(decoded, commit);
    }

    // --- encode/decode: underscore passes through ---
    #[test]
    fn encode_decode_underscore() {
        let commit = "abc_def_123";
        let tag = encode_commit_tag(commit);
        assert_eq!(tag, "c_abc_def_123");
        let decoded = decode_commit_tag(&tag).unwrap();
        assert_eq!(decoded, commit);
    }

    // --- encode/decode: empty commit id ---
    #[test]
    fn encode_decode_empty() {
        let commit = "";
        let tag = encode_commit_tag(commit);
        assert_eq!(tag, "c_");
        let decoded = decode_commit_tag(&tag).unwrap();
        assert_eq!(decoded, commit);
    }

    // --- decode: missing prefix errors ---
    #[test]
    fn decode_missing_prefix_errors() {
        let result = decode_commit_tag("no_prefix_here");
        assert!(result.is_err());
    }

    // --- decode: truncated escape errors ---
    #[test]
    fn decode_truncated_escape_errors() {
        // "c_abc-" — dash at end with no following char
        let result = decode_commit_tag("c_abc-");
        assert!(result.is_err());
    }

    // --- decode: invalid hex errors ---
    #[test]
    fn decode_invalid_hex_errors() {
        // "c_abc-zz" — 'zz' is not valid hex
        let result = decode_commit_tag("c_abc-zz");
        assert!(result.is_err());
    }

    // --- tag stays within Lance's allowed alphabet ---
    #[test]
    fn encoded_tag_uses_only_allowed_chars() {
        let adversarial_commits = [
            "branch:feature/v.1..2",
            "commit with spaces",
            "terminusdb:///star-wars/People/20",
            ".leading-dot",
            "trailing-dot.",
            "has..double..dots",
            "ends-with.lock",
            "main",
            "日本語",
        ];

        for commit in &adversarial_commits {
            let tag = encode_commit_tag(commit);
            // Must be non-empty (always has "c_" prefix).
            assert!(!tag.is_empty(), "tag for {:?} is empty", commit);
            // Must only contain [A-Za-z0-9_-].
            for ch in tag.chars() {
                assert!(
                    ch.is_ascii_alphanumeric() || ch == '_' || ch == '-',
                    "tag for {:?} contains illegal char {:?}: full tag = {:?}",
                    commit,
                    ch,
                    tag
                );
            }
            // Must not start with '.'.
            assert!(!tag.starts_with('.'), "tag starts with dot: {:?}", tag);
            // Must round-trip.
            let decoded = decode_commit_tag(&tag).unwrap();
            assert_eq!(&decoded, *commit, "round-trip failed for {:?}", commit);
        }
    }
}
