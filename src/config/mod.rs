#![forbid(unsafe_code)]

//! Configuration — resolve CLI + environment variables.
//! Pure validation with one effectful entry point for env loading.

use crate::embed::Provider;

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
    pub data_dir: String,
    pub tokenizer_path: String,
    /// Number of texts to batch per embedding HTTP call (cross-document batching).
    /// Configurable via `TDB_SEARCH_EMBED_BATCH_SIZE`; default 32.
    pub embed_batch_size: usize,
}

impl Config {
    /// Load configuration from environment variables with defaults.
    pub fn from_env() -> Self {
        let provider_str = std::env::var("TDB_SEARCH_EMBED_PROVIDER")
            .unwrap_or_else(|_| "openai_compatible".to_owned());
        let embed_url = std::env::var("TDB_SEARCH_EMBED_URL")
            .unwrap_or_else(|_| "http://localhost:11434".to_owned());
        let model = std::env::var("TDB_SEARCH_MODEL")
            .unwrap_or_else(|_| "nomic-embed-v2".to_owned());
        let dim: usize = std::env::var("TDB_SEARCH_DIM")
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

        let embed_batch_size: usize = std::env::var("TDB_SEARCH_EMBED_BATCH_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_EMBED_BATCH_SIZE);

        // Fail-fast at boot: batch_size=0 would panic on first push (division / empty batches).
        // Surface the misconfiguration immediately with a clear diagnostic.
        assert!(
            embed_batch_size > 0,
            "TDB_SEARCH_EMBED_BATCH_SIZE must be >= 1, got 0 (check environment)"
        );

        Self {
            admin_user: std::env::var("TDB_SEARCH_ADMIN_USER")
                .unwrap_or_else(|_| "admin".to_owned()),
            admin_secret: std::env::var("TDB_SEARCH_ADMIN_SECRET")
                .unwrap_or_else(|_| "root".to_owned()),
            port: std::env::var("TDB_SEARCH_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(8080),
            embed_provider,
            data_dir: std::env::var("TDB_SEARCH_DATA_DIR")
                .unwrap_or_else(|_| "/data".to_owned()),
            tokenizer_path: std::env::var("TDB_SEARCH_TOKENIZER_PATH")
                .unwrap_or_else(|_| "/data/tokenizer.json.bz2".to_owned()),
            embed_batch_size,
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
                model: "nomic-embed-v2".to_owned(),
                dim: 768,
            },
            data_dir: "/tmp/tdb-search-test".to_owned(),
            tokenizer_path: "spikes/tokenizer/tokenizer.json".to_owned(),
            embed_batch_size: DEFAULT_EMBED_BATCH_SIZE,
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
                model: "nomic-embed-v2".to_owned(),
                dim: 768,
            },
            data_dir: "/tmp/tdb-search-test".to_owned(),
            tokenizer_path: "spikes/tokenizer/tokenizer.json".to_owned(),
            embed_batch_size: DEFAULT_EMBED_BATCH_SIZE,
        }
    }
}
