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

#[cfg(test)]
mod tests {
    use super::*;

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
