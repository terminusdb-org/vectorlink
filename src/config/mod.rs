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
    /// Slot-2 (clustering) provider parameters, recorded when
    /// `VECTORLINK_MODEL2` is configured. `None` means the slot-1 provider
    /// serves the clustering role too — the vectors in `clustering_embedding`
    /// belong to the slot-1 space either way, so this is part of the lock.
    #[serde(default)]
    pub clustering: Option<Box<LockedEmbeddingParams>>,
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
            clustering: None,
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
    provider2: Option<&Provider>,
) -> Result<(), String> {
    let path = data_dir.join(EMBEDDING_PARAMS_FILE);
    let mut current = LockedEmbeddingParams::from_provider(provider);
    current.clustering = provider2.map(|p| Box::new(LockedEmbeddingParams::from_provider(p)));

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

/// Build a provider from its kind string + endpoint parts.
fn resolve_provider(kind: &str, base_url: String, model: String, dim: usize) -> Provider {
    match kind {
        "openai" => Provider::OpenAi {
            base_url,
            model,
            dim,
        },
        "generic_http" => Provider::GenericHttp {
            base_url,
            model,
            dim,
        },
        _ => Provider::OpenAiCompatible {
            base_url,
            model,
            dim,
        },
    }
}

/// Resolve the optional slot-2 (clustering) embedding provider.
///
/// `None` when `VECTORLINK_MODEL2` is unset — the slot-1 provider then serves
/// all embedding roles (single-model operation, the default).
///
/// When set, the `VECTORLINK_MODEL2_*` variables override endpoint, provider
/// kind and dimension; anything unset falls back to the slot-1 configuration,
/// so `VECTORLINK_MODEL2=nomic-embed-v2` alone is enough to pin the second
/// model on the same endpoint.
pub fn resolve_provider2<F>(slot1: &Provider, get: F) -> Option<Provider>
where
    F: Fn(&str) -> Option<String>,
{
    let model = get("VECTORLINK_MODEL2")?;
    let kind = get("VECTORLINK_MODEL2_PROVIDER").unwrap_or_else(|| slot1.kind().to_owned());
    let base_url = get("VECTORLINK_MODEL2_URL").unwrap_or_else(|| slot1.base_url().to_owned());
    let dim = get("VECTORLINK_MODEL2_DIM")
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| slot1.expected_dim());
    Some(resolve_provider(&kind, base_url, model, dim))
}

/// Slot-2 embed endpoint bearer token: `VECTORLINK_MODEL2_API_KEY`, falling
/// back to `VECTORLINK_EMBED_API_KEY` only when the slot-2 endpoint is the
/// same as slot 1's — credentials must never leak to a different host.
pub fn resolve_api_key2<F>(get: F, same_endpoint: bool) -> Option<String>
where
    F: Fn(&str) -> Option<String>,
{
    get("VECTORLINK_MODEL2_API_KEY")
        .or_else(|| {
            if same_endpoint {
                get("VECTORLINK_EMBED_API_KEY")
            } else {
                None
            }
        })
        .filter(|s| !s.is_empty())
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
    /// Optional second (slot-2) embedding provider used for the clustering
    /// role — e.g. `VECTORLINK_MODEL2=nomic-embed-v2`. When `None`, the
    /// slot-1 provider serves all roles.
    pub provider2: Option<Provider>,
    /// Optional bearer token for the embedding endpoint (e.g. OpenAI API key).
    /// `None` sends no Authorization header. `VECTORLINK_EMBED_API_KEY`;
    /// default None. Never persisted to disk.
    pub embed_api_key: Option<String>,
    /// Bearer token for the slot-2 endpoint (`VECTORLINK_MODEL2_API_KEY`,
    /// falling back to `VECTORLINK_EMBED_API_KEY`).
    pub api_key2: Option<String>,
    /// ANN index for the `clustering_embedding` column. `Some(config)`
    /// enables it (default) with independently tunable parameters;
    /// `None` (`VECTORLINK_MODEL2_INDEX=false`) leaves the column
    /// unindexed — /candidates gathers via flat scan.
    pub clustering_index: Option<crate::store::lance::VectorIndexConfig>,
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
            .unwrap_or_else(|_| "multilingual-e5-base".to_owned());
        let dim: usize = std::env::var("VECTORLINK_DIM")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(768);

        let embed_provider = resolve_provider(&provider_str, embed_url, model, dim);

        let env = |k: &str| std::env::var(k).ok();
        let provider2 = resolve_provider2(&embed_provider, env);
        let api_key2 = provider2.as_ref().and_then(|p2| {
            let same_endpoint = p2.base_url().trim_end_matches('/')
                == embed_provider.base_url().trim_end_matches('/');
            resolve_api_key2(env, same_endpoint)
        });
        if let Some(p2) = &provider2 {
            // The `clustering_embedding` column is FixedSizeList(dim) — a
            // different slot-2 width cannot be stored. Fail loud at boot.
            assert_eq!(
                p2.expected_dim(),
                dim,
                "VECTORLINK_MODEL2_DIM ({}) must equal VECTORLINK_DIM ({}) — \
                 clustering_embedding shares the store's vector width",
                p2.expected_dim(),
                dim
            );
        }

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

        // Slot-2 (clustering) ANN index: on by default, independently
        // parameterised so the second column can carry a different precision
        // profile than the slot-1 `embedding` index.
        let clustering_index = match std::env::var("VECTORLINK_MODEL2_INDEX") {
            Ok(s) if s.eq_ignore_ascii_case("false") || s == "0" => None,
            _ => {
                let mut cfg = crate::store::lance::VectorIndexConfig::default_for_dim(dim);
                let p = |name: &str| std::env::var(name).ok();
                if let Some(v) = p("VECTORLINK_MODEL2_INDEX_NUM_PARTITIONS") {
                    cfg.num_partitions = v.parse().unwrap_or(cfg.num_partitions);
                }
                if let Some(v) = p("VECTORLINK_MODEL2_INDEX_M") {
                    cfg.m = v.parse().unwrap_or(cfg.m);
                }
                if let Some(v) = p("VECTORLINK_MODEL2_INDEX_EF_CONSTRUCTION") {
                    cfg.ef_construction = v.parse().unwrap_or(cfg.ef_construction);
                }
                if let Some(v) = p("VECTORLINK_MODEL2_INDEX_NPROBES") {
                    cfg.nprobes = v.parse().unwrap_or(cfg.nprobes);
                }
                if let Some(v) = p("VECTORLINK_MODEL2_INDEX_REFINE_FACTOR") {
                    cfg.refine_factor = if v.eq_ignore_ascii_case("none") {
                        None
                    } else {
                        v.parse().ok().or(cfg.refine_factor)
                    };
                }
                Some(cfg)
            }
        };

        Self {
            admin_user: std::env::var("VECTORLINK_ADMIN_USER")
                .unwrap_or_else(|_| "admin".to_owned()),
            // No default: an empty secret means unset. The server binary
            // refuses to start without it; the embedded (trusted_router)
            // path does no auth and never reads it.
            admin_secret: std::env::var("VECTORLINK_ADMIN_SECRET").unwrap_or_default(),
            port: std::env::var("VECTORLINK_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(7372),
            embed_provider,
            provider2,
            embed_api_key: std::env::var("VECTORLINK_EMBED_API_KEY")
                .ok()
                .filter(|s| !s.is_empty()),
            api_key2,
            clustering_index,
            data_dir: std::env::var("VECTORLINK_DATA_DIR").unwrap_or_else(|_| "/data".to_owned()),
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
                model: "multilingual-e5-base".to_owned(),
                dim: 768,
            },
            provider2: None,
            embed_api_key: None,
            api_key2: None,
            clustering_index: Some(crate::store::lance::VectorIndexConfig::default_for_dim(768)),
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
            port: 7372,
            embed_provider: Provider::OpenAiCompatible {
                base_url: "http://localhost:11434".to_owned(),
                model: "multilingual-e5-base".to_owned(),
                dim: 768,
            },
            provider2: None,
            embed_api_key: None,
            api_key2: None,
            clustering_index: Some(crate::store::lance::VectorIndexConfig::default_for_dim(768)),
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
        verify_or_lock_embedding_params(dir.path(), &provider("nomic-embed-v2", 768), None)
            .unwrap();
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
        verify_or_lock_embedding_params(dir.path(), &provider("nomic-embed-v2", 768), None)
            .unwrap();
        verify_or_lock_embedding_params(dir.path(), &provider("nomic-embed-v2", 768), None)
            .unwrap();
    }

    #[test]
    fn params_lock_rejects_changed_model_with_recovery_hint() {
        let dir = tempfile::tempdir().unwrap();
        verify_or_lock_embedding_params(dir.path(), &provider("nomic-embed-v2", 768), None)
            .unwrap();
        let err = verify_or_lock_embedding_params(
            dir.path(),
            &provider("text-embedding-3-small", 1536),
            None,
        )
        .expect_err("mismatched params must fail");
        assert!(
            err.contains("nomic-embed-v2"),
            "stored params missing: {err}"
        );
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
        let err =
            verify_or_lock_embedding_params(dir.path(), &provider("nomic-embed-v2", 768), None)
                .expect_err("corrupt lock file must fail");
        assert!(err.contains(EMBEDDING_PARAMS_FILE), "path missing: {err}");
    }

    // --- slot-2 (clustering) provider resolution ---

    fn env<'a>(vars: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k| {
            vars.iter()
                .find(|(name, _)| *name == k)
                .map(|(_, v)| v.to_string())
        }
    }

    #[test]
    fn provider2_absent_without_model2() {
        let slot1 = provider("multilingual-e5-base", 768);
        assert!(resolve_provider2(&slot1, env(&[])).is_none());
    }

    #[test]
    fn provider2_inherits_slot1_kind_url_and_dim() {
        let slot1 = Provider::GenericHttp {
            base_url: "http://custom:9000".to_owned(),
            model: "multilingual-e5-base".to_owned(),
            dim: 768,
        };
        let p2 = resolve_provider2(&slot1, env(&[("VECTORLINK_MODEL2", "nomic-embed-v2")]))
            .expect("MODEL2 set → provider2 must resolve");
        assert_eq!(p2.kind(), "generic_http", "kind must inherit slot 1");
        assert_eq!(
            p2.base_url(),
            "http://custom:9000",
            "url must inherit slot 1"
        );
        assert_eq!(p2.model_name(), "nomic-embed-v2");
        assert_eq!(p2.expected_dim(), 768, "dim must inherit slot 1");
    }

    #[test]
    fn provider2_custom_endpoint_overrides() {
        let slot1 = provider("multilingual-e5-base", 768);
        let p2 = resolve_provider2(
            &slot1,
            env(&[
                ("VECTORLINK_MODEL2", "nomic-embed-v2"),
                ("VECTORLINK_MODEL2_PROVIDER", "openai"),
                ("VECTORLINK_MODEL2_URL", "http://other:11434"),
                ("VECTORLINK_MODEL2_DIM", "768"),
            ]),
        )
        .unwrap();
        assert_eq!(p2.kind(), "openai");
        assert_eq!(p2.base_url(), "http://other:11434");
        assert_eq!(p2.model_name(), "nomic-embed-v2");
        assert_eq!(p2.expected_dim(), 768);
    }

    #[test]
    fn api_key2_falls_back_to_slot1_key() {
        let vars = env(&[("VECTORLINK_EMBED_API_KEY", "sk-main")]);
        // Same endpoint → the slot-1 key serves both providers.
        assert_eq!(resolve_api_key2(&vars, true), Some("sk-main".to_owned()));
        let vars = env(&[
            ("VECTORLINK_EMBED_API_KEY", "sk-main"),
            ("VECTORLINK_MODEL2_API_KEY", "sk-second"),
        ]);
        assert_eq!(resolve_api_key2(&vars, false), Some("sk-second".to_owned()));
        assert_eq!(resolve_api_key2(env(&[]), true), None);
    }

    #[test]
    fn api_key2_never_leaks_slot1_key_to_a_different_endpoint() {
        // Different endpoint + no EMBED2 key → None (unauthenticated), never
        // the slot-1 credential.
        let vars = env(&[("VECTORLINK_EMBED_API_KEY", "sk-main")]);
        assert_eq!(resolve_api_key2(&vars, false), None);
    }

    // --- params lock covers slot 2 ---

    #[test]
    fn params_lock_covers_clustering_provider() {
        let dir = tempfile::tempdir().unwrap();
        verify_or_lock_embedding_params(
            dir.path(),
            &provider("multilingual-e5-base", 768),
            Some(&provider("nomic-embed-v2", 768)),
        )
        .unwrap();
        let locked: LockedEmbeddingParams = serde_json::from_str(
            &std::fs::read_to_string(dir.path().join(EMBEDDING_PARAMS_FILE)).unwrap(),
        )
        .unwrap();
        assert_eq!(
            locked.clustering.as_deref().map(|c| c.model.as_str()),
            Some("nomic-embed-v2"),
            "clustering model must be recorded in the lock"
        );
    }

    #[test]
    fn params_lock_rejects_changed_clustering_model() {
        let dir = tempfile::tempdir().unwrap();
        verify_or_lock_embedding_params(
            dir.path(),
            &provider("multilingual-e5-base", 768),
            Some(&provider("nomic-embed-v2", 768)),
        )
        .unwrap();
        let err = verify_or_lock_embedding_params(
            dir.path(),
            &provider("multilingual-e5-base", 768),
            Some(&provider("text-embedding-3-small", 768)),
        )
        .expect_err("changed clustering model must fail");
        assert!(err.contains("text-embedding-3-small"), "{err}");
    }

    #[test]
    fn params_lock_rejects_adding_provider2_to_existing_store() {
        // A store created without slot 2 holds clustering embeddings from the
        // slot-1 model. Adding a second model silently would mix spaces.
        let dir = tempfile::tempdir().unwrap();
        verify_or_lock_embedding_params(dir.path(), &provider("multilingual-e5-base", 768), None)
            .unwrap();
        let err = verify_or_lock_embedding_params(
            dir.path(),
            &provider("multilingual-e5-base", 768),
            Some(&provider("nomic-embed-v2", 768)),
        )
        .expect_err("enabling slot 2 on a locked store must fail");
        assert!(err.contains("drop-store"), "recovery hint missing: {err}");
    }
}
