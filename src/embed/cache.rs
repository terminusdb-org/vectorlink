// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 DFRNT AB

#![forbid(unsafe_code)]

//! Disk-backed embedding cache using sled + zstd.
//!
//! Cache key: SHA-256(model_name || "\n" || prefixed_text) → 32 bytes.
//! Cache value: zstd-compressed raw f32 bytes.
//!
//! When disabled (max_entries = None), all operations are no-ops / passthrough.

use std::path::Path;

use sha2::{Digest, Sha256};
use ruzstd::encoding::CompressionLevel;

/// Zstd compression level for embedding storage.
/// ruzstd uses a simplified level enum (Fastest/Fast/Default/High).
const ZSTD_LEVEL: CompressionLevel = CompressionLevel::Fastest;

/// Sled background flush interval in milliseconds.
const SLED_FLUSH_MS: u64 = 2000;

// Sled's `cache_capacity` is a soft limit on the in-memory page cache.
// Sled persists ALL entries to disk (unbounded for now) and uses an internal
// LRU to evict in-memory pages when the cache is full.
// We keep the in-memory page cache small (~20MB) — just enough for hot
// entries. The old formula (max_entries × 4KB) caused 12 GB RAM usage when
// max_entries was set to 3M. Disk-side LRU eviction will be added later.
const SLED_CACHE_CAPACITY_BYTES: u64 = 5_000 * 4096;

/// Disk-backed embedding cache.
///
/// When `enabled` is false, `get` always returns `None` and `put` is a no-op.
/// This allows the cache to be transparently disabled via configuration.
pub struct EmbedCache {
    db: Option<sled::Db>,
    enabled: bool,
    /// Atomic entry counter shared across all clones via Arc.
    /// Incremented on `put`, never decremented (sled doesn't expose eviction
    /// callbacks). Initialized lazily on first `len()` call by scanning the
    /// sled tree. After that, reads are O(1).
    /// Uses `AtomicUsize::MAX` as a sentinel for "not yet initialized".
    entry_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl Clone for EmbedCache {
    fn clone(&self) -> Self {
        Self {
            db: self.db.clone(),
            enabled: self.enabled,
            entry_count: std::sync::Arc::clone(&self.entry_count),
        }
    }
}

impl EmbedCache {
    /// Create a disabled cache (no-op for get/put).
    pub fn disabled() -> Self {
        Self {
            db: None,
            enabled: false,
            entry_count: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    /// Open or create the sled cache at `path`.
    ///
    /// `max_entries` controls whether the cache is enabled (Some) or disabled
    /// (None). The in-memory page cache is fixed at ~20 MB regardless — sled
    /// persists all entries to disk and uses LRU eviction for in-memory pages.
    pub fn open(path: &Path, max_entries: Option<usize>) -> Self {
        match max_entries {
            None => Self {
                db: None,
                enabled: false,
                entry_count: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            },
            Some(_) => {
                let db = sled::Config::default()
                    .path(path)
                    .flush_every_ms(Some(SLED_FLUSH_MS))
                    .cache_capacity(SLED_CACHE_CAPACITY_BYTES)
                    .open()
                    .expect("failed to open sled embed cache");
                Self {
                    db: Some(db),
                    enabled: true,
                    entry_count: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(usize::MAX)),
                }
            }
        }
    }

    /// Compute the 32-byte SHA-256 cache key from model name and prefixed text.
    pub fn cache_key(model: &str, prefixed_text: &str) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(model.as_bytes());
        hasher.update(b"\n");
        hasher.update(prefixed_text.as_bytes());
        hasher.finalize().into()
    }

    /// Look up a cached embedding by key.
    ///
    /// Returns `None` if: cache disabled, key missing, decompression fails,
    /// or the decompressed data doesn't match `expected_dim`.
    pub fn get(&self, key: &[u8; 32], expected_dim: usize) -> Option<Vec<f32>> {
        if !self.enabled {
            return None;
        }
        let db = self.db.as_ref()?;
        let compressed = db.get(key).ok().flatten()?;
        let decompressed = decompress_zstd(&compressed).ok()?;
        let bytes = decompressed.len();
        let expected_bytes = expected_dim * std::mem::size_of::<f32>();
        if bytes != expected_bytes {
            return None;
        }
        let floats: Vec<f32> = decompressed
            .chunks_exact(4)
            .map(|chunk| {
                f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
            })
            .collect();
        Some(floats)
    }

    /// Store an embedding in the cache. No-op if disabled.
    pub fn put(&self, key: &[u8; 32], embedding: &[f32]) {
        if !self.enabled {
            return;
        }
        if let Some(db) = &self.db {
            let raw: Vec<u8> = embedding
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            let compressed = ruzstd::encoding::compress_to_vec(&raw[..], ZSTD_LEVEL);
            let was_new = db.insert(key, compressed).ok().flatten().is_none();
            if was_new {
                self.entry_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    /// Whether the cache is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Number of entries currently in the cache. Returns 0 if disabled.
    /// Uses an atomic counter for O(1) reads. On first call after open,
    /// initializes the counter by scanning the sled tree (one-time cost).
    pub fn len(&self) -> usize {
        if !self.enabled {
            return 0;
        }
        let cached = self.entry_count.load(std::sync::atomic::Ordering::Relaxed);
        if cached != usize::MAX {
            return cached;
        }
        // First call: scan the sled tree to initialize the counter.
        let db = match &self.db {
            Some(db) => db,
            None => return 0,
        };
        let count = db.len();
        self.entry_count.store(count, std::sync::atomic::Ordering::Relaxed);
        count
    }

    /// Returns true if the cache is empty or disabled.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Flush the cache to disk. No-op if disabled.
    pub fn flush(&self) {
        if let Some(db) = &self.db {
            let _ = db.flush();
        }
    }
}

/// Decompress zstd-compressed data using ruzstd's StreamingDecoder.
fn decompress_zstd(compressed: &[u8]) -> Result<Vec<u8>, std::io::Error> {
    use std::io::Read;
    let mut decoder = ruzstd::decoding::StreamingDecoder::new(compressed)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let mut out = Vec::new();
    decoder.read_to_end(&mut out)?;
    Ok(out)
}

// ─────────────────────────────── Tests ───────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_tmp_cache() -> (TempDir, EmbedCache) {
        let dir = TempDir::new().expect("tempdir");
        let cache = EmbedCache::open(dir.path(), Some(1000));
        (dir, cache)
    }

    #[test]
    fn cache_key_derivation() {
        let k1 = EmbedCache::cache_key("nomic-embed-v2", "search_document: hello");
        let k2 = EmbedCache::cache_key("nomic-embed-v2", "search_document: hello");
        assert_eq!(k1, k2, "same model + text → same key");

        let k3 = EmbedCache::cache_key("nomic-embed-v2", "search_query: hello");
        assert_ne!(k1, k3, "different prefix → different key");

        let k4 = EmbedCache::cache_key("other-model", "search_document: hello");
        assert_ne!(k1, k4, "different model → different key");
    }

    #[test]
    fn compression_roundtrip() {
        let embedding: Vec<f32> = (0..768).map(|i| i as f32 * 0.1).collect();
        let raw: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
        let compressed = ruzstd::encoding::compress_to_vec(&raw[..], ZSTD_LEVEL);
        let decompressed = decompress_zstd(&compressed).expect("decode");
        let floats: Vec<f32> = decompressed
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(floats, embedding, "roundtrip preserves exact values");
    }

    #[test]
    fn cache_put_get_roundtrip() {
        let (_dir, cache) = make_tmp_cache();
        assert!(cache.is_enabled());

        let key = EmbedCache::cache_key("nomic-embed-v2", "search_document: hello");
        let embedding: Vec<f32> = (0..768).map(|i| i as f32 * 0.001).collect();

        cache.put(&key, &embedding);

        let retrieved = cache.get(&key, 768);
        assert!(retrieved.is_some(), "cache hit after put");
        assert_eq!(retrieved.unwrap(), embedding, "retrieved embedding matches");
    }

    #[test]
    fn cache_disabled_when_none() {
        let dir = TempDir::new().expect("tempdir");
        let cache = EmbedCache::open(dir.path(), None);

        assert!(!cache.is_enabled());

        let key = EmbedCache::cache_key("model", "text");
        let embedding = vec![0.1_f32; 768];

        cache.put(&key, &embedding);
        assert_eq!(cache.len(), 0, "put is no-op when disabled");

        assert!(cache.get(&key, 768).is_none(), "get returns None when disabled");
    }

    #[test]
    fn dimension_mismatch_on_cached() {
        let (_dir, cache) = make_tmp_cache();

        let key = EmbedCache::cache_key("model", "text");
        let embedding = vec![0.5_f32; 768];
        cache.put(&key, &embedding);

        // Request with wrong dimension → None (defensive)
        assert!(cache.get(&key, 512).is_none(), "wrong dim → cache miss");
    }

    #[test]
    fn cache_len_tracks_entries() {
        let (_dir, cache) = make_tmp_cache();
        assert_eq!(cache.len(), 0);

        let k1 = EmbedCache::cache_key("model", "text1");
        let k2 = EmbedCache::cache_key("model", "text2");
        let emb = vec![0.1_f32; 768];

        cache.put(&k1, &emb);
        assert_eq!(cache.len(), 1);

        cache.put(&k2, &emb);
        assert_eq!(cache.len(), 2);

        // Overwrite same key → count stays the same
        cache.put(&k1, &emb);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn cache_miss_returns_none() {
        let (_dir, cache) = make_tmp_cache();
        let key = EmbedCache::cache_key("model", "never_stored");
        assert!(cache.get(&key, 768).is_none(), "missing key → None");
    }
}
