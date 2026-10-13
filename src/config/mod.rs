// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 DFRNT AB

#![forbid(unsafe_code)]

//! Configuration — resolve CLI + environment variables.
//! Pure validation with one effectful entry point for env loading.

use crate::embed::Provider;

/// Per-domain settings persisted in `domain_settings.json`.
///
/// Controls optional indexing features such as clustering embeddings.
/// Settings are set on first push for a new domain and are immutable
/// thereafter — changing a setting requires a full reindex (delete + re-push).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct DomainSettings {
    /// When `true`, the indexing pipeline generates a second embedding per chunk
    /// using `EmbeddingRole::Clustering` and stores it in the `clustering_embedding`
    /// column. The `/candidates` endpoint performs a dual KNN gather to produce
    /// `clustering_distance`. Defaults to `false` (one embedding call per chunk).
    #[serde(default)]
    pub store_clustering: bool,
}

/// In-memory map of domain → settings, loaded from `domain_settings.json` at
/// startup and held for the lifetime of the process. Writes are rare (first
/// push only), so a simple `RwLock<HashMap>` suffices.
#[derive(Debug, Clone, Default)]
pub struct DomainSettingsMap {
    settings: std::sync::Arc<tokio::sync::RwLock<std::collections::HashMap<String, DomainSettings>>>,
    /// Path to the `domain_settings.json` file (for persistence).
    path: std::path::PathBuf,
}

impl DomainSettingsMap {
    /// Load domain settings from `domain_settings.json` in the given data
    /// directory. If the file does not exist, starts with an empty map (all
    /// domains default to `store_clustering: false`).
    pub fn load(data_dir: &std::path::Path) -> Self {
        let path = data_dir.join("domain_settings.json");
        let settings = if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(contents) => {
                    serde_json::from_str::<std::collections::HashMap<String, DomainSettings>>(
                        &contents,
                    )
                    .unwrap_or_default()
                }
                Err(_) => {
                    eprintln!(
                        "[config] warning: failed to read {}, starting with empty settings",
                        path.display()
                    );
                    std::collections::HashMap::new()
                }
            }
        } else {
            std::collections::HashMap::new()
        };

        Self {
            settings: std::sync::Arc::new(tokio::sync::RwLock::new(settings)),
            path,
        }
    }

    /// Create an empty settings map (for tests).
    pub fn new_empty() -> Self {
        Self {
            settings: std::sync::Arc::new(tokio::sync::RwLock::new(
                std::collections::HashMap::new(),
            )),
            path: std::path::PathBuf::new(),
        }
    }

    /// Get the settings for a domain. Returns `DomainSettings::default()` if
    /// the domain is not in the map (clustering OFF by default).
    pub async fn get(&self, domain: &str) -> DomainSettings {
        let settings = self.settings.read().await;
        settings
            .get(domain)
            .cloned()
            .unwrap_or_default()
    }

    /// Set settings for a domain. Persists to `domain_settings.json` if a path
    /// was configured. Should only be called for new domains (first push).
    pub async fn set(&self, domain: &str, settings: DomainSettings) {
        let mut map = self.settings.write().await;
        map.insert(domain.to_owned(), settings.clone());
        drop(map);

        if !self.path.as_os_str().is_empty() {
            let map = self.settings.read().await;
            if let Ok(json) = serde_json::to_string_pretty(&*map) {
                let _ = std::fs::write(&self.path, json);
            }
        }
    }

    /// Check whether a domain has clustering enabled.
    pub async fn store_clustering(&self, domain: &str) -> bool {
        self.get(domain).await.store_clustering
    }
}

/// Embedding parameters locked into `vectorlink-params.json` at the store root
/// when the store is first initialized. These define the vector space — every
/// stored embedding is only valid for the exact provider/model/dim that produced
/// it, so the parameters are immutable for the lifetime of the store. Secrets
/// (VECTORLINK_EMBED_API_KEY) are never persisted here.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LockedEmbeddingParams {
    pub provider: String,
    pub base_url: String,
    pub model: String,
    pub dim: usize,
}

/// Name of the embedding-parameters lock file at the store root.
pub const EMBEDDING_PARAMS_FILE: &str = "vectorlink-params.json";

impl LockedEmbeddingParams {
    pub fn from_provider(provider: &Provider) -> Self {
        Self {
            provider: provider.kind().to_owned(),
            // Normalized like io_embed does, so a trailing slash in
            // VECTORLINK_EMBED_URL cannot false-positive the lock check.
            base_url: provider.base_url().trim_end_matches('/').to_owned(),
            model: provider.model_name().to_owned(),
            dim: provider.expected_dim(),
        }
    }
}

/// Verify the configured embedding provider against the parameters locked into
/// the store at creation, writing the lock file on first initialization.
///
/// A mismatch means the store's vectors were produced by a different embedding
/// setup and cannot be served correctly — this returns an error describing both
/// parameter sets so the operator can restore the original configuration or
/// recreate the store with `vectorlink drop-store --yes`.
pub fn verify_or_lock_embedding_params(
    data_dir: &std::path::Path,
    provider: &Provider,
) -> Result<(), String> {
    let path = data_dir.join(EMBEDDING_PARAMS_FILE);
    let current = LockedEmbeddingParams::from_provider(provider);

    if path.exists() {
        let contents = std::fs::read_to_string(&path)
            .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
        let locked: LockedEmbeddingParams = serde_json::from_str(&contents).map_err(|e| {
            format!(
                "corrupt {}: {} — the store's embedding parameters cannot be verified. \
                 Restore the file or recreate the store with `vectorlink drop-store --yes`.",
                path.display(),
                e
            )
        })?;
        if locked != current {
            return Err(format!(
                "embedding parameters mismatch: the vectorlink store at {} was created with\n{}\n\
                 but the current configuration resolves to\n{}\n\
                 All stored embeddings are only valid for the locked parameters. Restore the \
                 original parameters, or drop and recreate the entire store with \
                 `vectorlink drop-store --yes` — embedded in TerminusDB, delete the vectorlink \
                 directory under the storage path instead (deletes all indexed vectors).",
                data_dir.display(),
                serde_json::to_string_pretty(&locked).unwrap_or_default(),
                serde_json::to_string_pretty(&current).unwrap_or_default()
            ));
        }
        return Ok(());
    }

    std::fs::create_dir_all(data_dir)
        .map_err(|e| format!("failed to create {}: {}", data_dir.display(), e))?;
    // A store built before the lock file existed has datasets but no params —
    // adopt the current parameters, but warn that they were never verified.
    let has_datasets = std::fs::read_dir(data_dir)
        .map(|rd| {
            rd.flatten()
                .any(|e| e.file_name().to_string_lossy().ends_with(".lance"))
        })
        .unwrap_or(false);
    if has_datasets {
        eprintln!(
            "[config] warning: existing store at {} has no {}; locking current embedding parameters",
            data_dir.display(),
            EMBEDDING_PARAMS_FILE
        );
    }
    let json = serde_json::to_string_pretty(&current)
        .map_err(|e| format!("failed to serialize embedding parameters: {}", e))?;
    std::fs::write(&path, format!("{}\n", json))
        .map_err(|e| format!("failed to write {}: {}", path.display(), e))?;
    Ok(())
}

/// Default batch size for cross-document embedding (PO decision, 2026-06-15).
/// Past the knee of the latency curve (88.8 ms/embed at bs=32 vs 228.8 at bs=1).
const DEFAULT_EMBED_BATCH_SIZE: usize = 32;

/// Application configuration.
#[derive(Debug, Clone)]
pub struct Config {
    pub admin_user: String,
    pub admin_secret: String,
    pub port: u16,
    pub embed_provider: Provider,
    /// Optional bearer token for the embedding endpoint (e.g. OpenAI API key).
    /// `None` sends no Authorization header. `VECTORLINK_EMBED_API_KEY`;
    /// default None. Never persisted to disk.
    pub embed_api_key: Option<String>,
    pub data_dir: String,
    pub tokenizer_path: String,
    /// Number of texts to batch per embedding HTTP call (cross-document batching).
    /// Configurable via `VECTORLINK_EMBED_BATCH_SIZE`; default 32.
    pub embed_batch_size: usize,
    /// Maximum entries for the disk-backed embedding cache (sled + zstd).
    /// `None` disables the cache (passthrough to embedding provider).
    /// Configurable via `VECTORLINK_EMBED_CACHE_SIZE`; default 20,000.
    pub embed_cache_size: Option<usize>,
    /// Optional Prometheus metrics endpoint port. When set, a separate HTTP
    /// server listens on this port and exposes `/metrics` in Prometheus text
    /// exposition format. When `None`, no metrics server is started.
    /// Configurable via `VECTORLINK_PROMETHEUS_PORT`; default None (disabled).
    pub prometheus_port: Option<u16>,
    /// Lance index cache capacity in bytes. Controls how much RAM Lance uses
    /// to cache vector index data for fast ANN queries. Lance's own default is
    /// 6 GiB; we use 2 GiB as a balanced default for typical deployments.
    /// Configurable via `VECTORLINK_LANCE_INDEX_CACHE_BYTES`; default 2 GiB.
    pub lance_index_cache_bytes: usize,
    /// Lance metadata cache capacity in bytes. Controls how much RAM Lance
    /// uses to cache dataset metadata (manifests, fragment info). Lance's own
    /// default is 1 GiB; we use 512 MiB as a balanced default.
    /// Configurable via `VECTORLINK_LANCE_METADATA_CACHE_BYTES`; default 512 MiB.
    pub lance_metadata_cache_bytes: usize,
}

impl Config {
    /// Load configuration from environment variables with defaults.
    pub fn from_env() -> Self {
        let provider_str = std::env::var("VECTORLINK_EMBED_PROVIDER")
            .unwrap_or_else(|_| "openai_compatible".to_owned());
        let embed_url = std::env::var("VECTORLINK_EMBED_URL")
            .unwrap_or_else(|_| "http://localhost:11434".to_owned());
        let model = std::env::var("VECTORLINK_MODEL")
            .unwrap_or_else(|_| "nomic-embed-text-v2-moe".to_owned());
        let dim: usize = std::env::var("VECTORLINK_DIM")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(768);

        let embed_provider = match provider_str.as_str() {
            "openai" => Provider::OpenAi {
                base_url: embed_url,
                model,
                dim,
            },
            "generic_http" => Provider::GenericHttp {
                base_url: embed_url,
                model,
                dim,
            },
            _ => Provider::OpenAiCompatible {
                base_url: embed_url,
                model,
                dim,
            },
        };

        let embed_batch_size: usize = std::env::var("VECTORLINK_EMBED_BATCH_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_EMBED_BATCH_SIZE);

        // Fail-fast at boot: batch_size=0 would panic on first push (division / empty batches).
        // Surface the misconfiguration immediately with a clear diagnostic.
        assert!(
            embed_batch_size > 0,
            "VECTORLINK_EMBED_BATCH_SIZE must be >= 1, got 0 (check environment)"
        );

        Self {
            admin_user: std::env::var("VECTORLINK_ADMIN_USER")
                .unwrap_or_else(|_| "admin".to_owned()),
            admin_secret: std::env::var("VECTORLINK_ADMIN_SECRET")
                .unwrap_or_else(|_| "root".to_owned()),
            port: std::env::var("VECTORLINK_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(8080),
            embed_provider,
            embed_api_key: std::env::var("VECTORLINK_EMBED_API_KEY")
                .ok()
                .filter(|s| !s.is_empty()),
            data_dir: std::env::var("VECTORLINK_DATA_DIR")
                .unwrap_or_else(|_| "/data".to_owned()),
            tokenizer_path: std::env::var("VECTORLINK_TOKENIZER_PATH")
                .unwrap_or_else(|_| "/data/tokenizer.json.bz2".to_owned()),
            embed_batch_size,
            embed_cache_size: match std::env::var("VECTORLINK_EMBED_CACHE_SIZE") {
                Ok(s) if s.eq_ignore_ascii_case("none") => None,
                Ok(s) => Some(s.parse::<usize>().unwrap_or(20_000)),
                Err(_) => Some(20_000),
            },
            prometheus_port: std::env::var("VECTORLINK_PROMETHEUS_PORT")
                .ok()
                .and_then(|s| s.parse::<u16>().ok()),
            lance_index_cache_bytes: std::env::var("VECTORLINK_LANCE_INDEX_CACHE_BYTES")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(2 * 1024 * 1024 * 1024),
            lance_metadata_cache_bytes: std::env::var("VECTORLINK_LANCE_METADATA_CACHE_BYTES")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(512 * 1024 * 1024),
        }
    }

    /// Create a config with explicit values (used in tests).
    pub fn new(admin_user: String, admin_secret: String, port: u16) -> Self {
        Self {
            admin_user,
            admin_secret,
            port,
            embed_provider: Provider::OpenAiCompatible {
                base_url: "http://localhost:11434".to_owned(),
                model: "nomic-embed-text-v2-moe".to_owned(),
                dim: 768,
            },
            embed_api_key: None,
            data_dir: "/tmp/vectorlink-test".to_owned(),
            tokenizer_path: "assets/tokenizer.json.bz2".to_owned(),
            embed_batch_size: DEFAULT_EMBED_BATCH_SIZE,
            embed_cache_size: None,
            prometheus_port: None,
            lance_index_cache_bytes: 2 * 1024 * 1024 * 1024,
            lance_metadata_cache_bytes: 512 * 1024 * 1024,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            admin_user: "admin".to_owned(),
            admin_secret: "root".to_owned(),
            port: 8080,
            embed_provider: Provider::OpenAiCompatible {
                base_url: "http://localhost:11434".to_owned(),
                model: "nomic-embed-text-v2-moe".to_owned(),
                dim: 768,
            },
            embed_api_key: None,
            data_dir: "/tmp/vectorlink-test".to_owned(),
            tokenizer_path: "assets/tokenizer.json.bz2".to_owned(),
            embed_batch_size: DEFAULT_EMBED_BATCH_SIZE,
            embed_cache_size: None,
            prometheus_port: None,
            lance_index_cache_bytes: 2 * 1024 * 1024 * 1024,
            lance_metadata_cache_bytes: 512 * 1024 * 1024,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider(model: &str, dim: usize) -> Provider {
        Provider::OpenAiCompatible {
            base_url: "http://localhost:11434".to_owned(),
            model: model.to_owned(),
            dim,
        }
    }

    #[test]
    fn params_lock_written_on_first_init() {
        let dir = tempfile::tempdir().unwrap();
        verify_or_lock_embedding_params(dir.path(), &provider("nomic-embed-v2", 768)).unwrap();
        let locked: LockedEmbeddingParams = serde_json::from_str(
            &std::fs::read_to_string(dir.path().join(EMBEDDING_PARAMS_FILE)).unwrap(),
        )
        .unwrap();
        assert_eq!(
            locked,
            LockedEmbeddingParams::from_provider(&provider("nomic-embed-v2", 768))
        );
    }

    #[test]
    fn params_lock_accepts_matching_provider() {
        let dir = tempfile::tempdir().unwrap();
        verify_or_lock_embedding_params(dir.path(), &provider("nomic-embed-v2", 768)).unwrap();
        verify_or_lock_embedding_params(dir.path(), &provider("nomic-embed-v2", 768)).unwrap();
    }

    #[test]
    fn params_lock_rejects_changed_model_with_recovery_hint() {
        let dir = tempfile::tempdir().unwrap();
        verify_or_lock_embedding_params(dir.path(), &provider("nomic-embed-v2", 768)).unwrap();
        let err =
            verify_or_lock_embedding_params(dir.path(), &provider("text-embedding-3-small", 1536))
                .expect_err("mismatched params must fail");
        assert!(err.contains("nomic-embed-v2"), "stored params missing: {err}");
        assert!(
            err.contains("text-embedding-3-small"),
            "current params missing: {err}"
        );
        assert!(err.contains("drop-store"), "recovery hint missing: {err}");
    }

    #[test]
    fn params_lock_rejects_corrupt_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(EMBEDDING_PARAMS_FILE), "not json").unwrap();
        let err = verify_or_lock_embedding_params(dir.path(), &provider("nomic-embed-v2", 768))
            .expect_err("corrupt lock file must fail");
        assert!(err.contains(EMBEDDING_PARAMS_FILE), "path missing: {err}");
    }
}
