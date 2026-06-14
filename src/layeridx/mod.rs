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

// ───────────────────── Per-branch enablement + negative cache ─────────────

/// Process-local, advisory per-branch state for catch-up resolution (Spec 10 §5,
/// RISK-15). The TRUTH is always the layer index (Lance tags); this state only
/// accelerates the common paths:
///  - `enabled`: which `(domain, branch)` lineages are indexing-enabled (one
///    explicit bootstrap per lineage; descendants auto-enroll on first search).
///  - `negative_cache`: branches with NO indexed ancestor → 404, cached with a
///    TTL so we don't re-walk up-to-1000-commit history on every search. Busted
///    immediately on an indexing request for the branch (direct); TTL backstops
///    the indirect/ancestor case.
///
/// The clock is injectable for deterministic TTL tests.
#[derive(Debug)]
pub struct BranchState {
    enabled: std::sync::RwLock<std::collections::HashSet<(String, String)>>,
    negative_cache: std::sync::RwLock<std::collections::HashMap<(String, String), NegativeEntry>>,
    ttl: std::time::Duration,
    clock: Clock,
}

#[derive(Debug, Clone, Copy)]
struct NegativeEntry {
    cached_at: std::time::Instant,
}

/// Injectable clock — monotonic time plus a manual offset. In production the
/// offset is always `ZERO` (real time); tests can advance it deterministically
/// via `BranchState::advance`. Modelled as a single always-compiled struct (not
/// a real/test enum) so there is no test-only dead code under `-D warnings`.
#[derive(Debug)]
struct Clock {
    /// Extra time to add on top of real monotonic time. Always `ZERO` in
    /// production; advanced by tests for deterministic TTL expiry.
    manual_offset: std::sync::RwLock<std::time::Duration>,
}

impl Clock {
    fn new() -> Self {
        Self {
            manual_offset: std::sync::RwLock::new(std::time::Duration::ZERO),
        }
    }

    fn now(&self) -> std::time::Instant {
        let offset = *self
            .manual_offset
            .read()
            .expect("clock offset lock poisoned — a thread panicked holding it");
        std::time::Instant::now() + offset
    }

    /// Advance time by `delta`. Only used by tests to exercise TTL expiry; in
    /// production the offset stays `ZERO`.
    #[cfg(test)]
    fn advance(&self, delta: std::time::Duration) {
        let mut o = self
            .manual_offset
            .write()
            .expect("clock offset lock poisoned");
        *o += delta;
    }
}

/// Default negative-cache TTL (Spec 10 §5 / RISK-15): 3600 seconds.
pub const DEFAULT_NEGATIVE_CACHE_TTL_SECS: u64 = 3600;

impl Default for BranchState {
    fn default() -> Self {
        Self::with_ttl(std::time::Duration::from_secs(DEFAULT_NEGATIVE_CACHE_TTL_SECS))
    }
}

impl BranchState {
    /// Production constructor with the given negative-cache TTL.
    pub fn with_ttl(ttl: std::time::Duration) -> Self {
        Self {
            enabled: std::sync::RwLock::new(std::collections::HashSet::new()),
            negative_cache: std::sync::RwLock::new(std::collections::HashMap::new()),
            ttl,
            clock: Clock::new(),
        }
    }

    /// Test alias: construct with a TTL and a clock whose time can be advanced
    /// deterministically via `advance`.
    #[cfg(test)]
    pub fn with_manual_clock(ttl: std::time::Duration) -> Self {
        Self::with_ttl(ttl)
    }

    /// Advance the internal clock by `delta` (deterministic TTL testing).
    #[cfg(test)]
    pub fn advance(&self, delta: std::time::Duration) {
        self.clock.advance(delta);
    }

    fn key(domain: &str, branch: &str) -> (String, String) {
        (domain.to_owned(), branch.to_owned())
    }

    /// Mark a `(domain, branch)` lineage indexing-enabled (explicit bootstrap or
    /// auto-enroll). Idempotent.
    pub fn enable(&self, domain: &str, branch: &str) {
        let mut set = self.enabled.write().expect("enabled lock poisoned");
        set.insert(Self::key(domain, branch));
    }

    /// Is this `(domain, branch)` indexing-enabled? Enabling one branch does NOT
    /// enable a sibling (per-branch precision, RISK-22 / P3-LIX-2).
    pub fn is_enabled(&self, domain: &str, branch: &str) -> bool {
        let set = self.enabled.read().expect("enabled lock poisoned");
        set.contains(&Self::key(domain, branch))
    }

    /// Record a 404 (no indexed ancestor) for a branch, with the current time.
    pub fn record_negative(&self, domain: &str, branch: &str) {
        let mut cache = self
            .negative_cache
            .write()
            .expect("negative cache lock poisoned");
        cache.insert(
            Self::key(domain, branch),
            NegativeEntry {
                cached_at: self.clock.now(),
            },
        );
    }

    /// Is there a LIVE (non-expired) negative-cache entry for this branch?
    /// Expired entries are treated as absent (and pruned lazily here).
    pub fn is_negative_cached(&self, domain: &str, branch: &str) -> bool {
        let key = Self::key(domain, branch);
        let now = self.clock.now();
        // Fast read path.
        {
            let cache = self.negative_cache.read().expect("negative cache lock poisoned");
            match cache.get(&key) {
                None => return false,
                Some(entry) => {
                    if now.duration_since(entry.cached_at) < self.ttl {
                        return true;
                    }
                    // Expired — fall through to prune under the write lock.
                }
            }
        }
        // Prune the expired entry so it can't accumulate.
        let mut cache = self.negative_cache.write().expect("negative cache lock poisoned");
        if let Some(entry) = cache.get(&key) {
            if now.duration_since(entry.cached_at) >= self.ttl {
                cache.remove(&key);
            }
        }
        false
    }

    /// Invalidate (bust) the negative-cache entry for a branch — called
    /// immediately on an indexing request for that branch (direct enablement).
    pub fn invalidate_negative(&self, domain: &str, branch: &str) {
        let mut cache = self
            .negative_cache
            .write()
            .expect("negative cache lock poisoned");
        cache.remove(&Self::key(domain, branch));
    }

    /// Purge ALL per-branch state for a domain (every branch). Used by
    /// `DELETE /domain` so no enablement or negative-cache entry outlives the
    /// deleted domain's footprint.
    pub fn purge_domain(&self, domain: &str) {
        {
            let mut set = self.enabled.write().expect("enabled lock poisoned");
            set.retain(|(d, _b)| d != domain);
        }
        {
            let mut cache = self
                .negative_cache
                .write()
                .expect("negative cache lock poisoned");
            cache.retain(|(d, _b), _| d != domain);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::time::Duration;

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

    // ───────────────────── per-branch state (P3-LIX-2) ─────────────────────

    // --- enabling one branch does NOT enable a sibling ---
    #[test]
    fn enablement_is_per_branch() {
        let state = BranchState::default();
        state.enable("admin/db", "main");
        assert!(state.is_enabled("admin/db", "main"));
        assert!(
            !state.is_enabled("admin/db", "feature"),
            "enabling main must not enable a sibling branch"
        );
        // A different domain's main is also independent.
        assert!(!state.is_enabled("admin/other", "main"));
    }

    // --- negative cache is per-branch ---
    #[test]
    fn negative_cache_is_per_branch() {
        let state = BranchState::default();
        state.record_negative("admin/db", "feature");
        assert!(state.is_negative_cached("admin/db", "feature"));
        assert!(
            !state.is_negative_cached("admin/db", "main"),
            "a 404 on feature must not negatively cache main"
        );
    }

    // --- negative cache busts immediately on invalidate (direct index request) ---
    #[test]
    fn negative_cache_invalidates_on_index() {
        let state = BranchState::default();
        state.record_negative("admin/db", "feature");
        assert!(state.is_negative_cached("admin/db", "feature"));
        state.invalidate_negative("admin/db", "feature");
        assert!(
            !state.is_negative_cached("admin/db", "feature"),
            "negative cache must bust immediately on an indexing request"
        );
    }

    // --- negative cache expires after the TTL (manual clock) ---
    #[test]
    fn negative_cache_expires_after_ttl() {
        let ttl = Duration::from_secs(3600);
        let state = BranchState::with_manual_clock(ttl);
        state.record_negative("admin/db", "feature");
        assert!(state.is_negative_cached("admin/db", "feature"));

        // Advance just under the TTL — still cached.
        state.advance(Duration::from_secs(3599));
        assert!(state.is_negative_cached("admin/db", "feature"));

        // Cross the TTL — expired (treated as absent, pruned).
        state.advance(Duration::from_secs(2));
        assert!(
            !state.is_negative_cached("admin/db", "feature"),
            "negative cache entry must expire after the TTL"
        );
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
