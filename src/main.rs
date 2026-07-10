#![forbid(unsafe_code)]

//! tdb-search server — standalone semantic search engine.
//!
//! Binds [::]:8080, answers liveness immediately, defers all heavy work.

use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;

use tdb_search::chunk;
use tdb_search::config::Config;
use tdb_search::http_api;
use tdb_search::service::SearchService;
use tdb_search::store::lance::LanceStore;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::from_env();

    // Load the tokenizer (required for chunking).
    let tokenizer_path = Path::new(&config.tokenizer_path);
    let tokenizer = chunk::io_load_tokenizer(tokenizer_path).map_err(|e| {
        format!(
            "failed to load tokenizer from {}: {}",
            config.tokenizer_path, e
        )
    })?;

    // Create the LanceStore backed by the data directory.
    let dim = config.embed_provider.expected_dim();
    let data_dir = Path::new(&config.data_dir);
    let lance_store = Arc::new(LanceStore::new(data_dir, dim));

    // Disable probabilistic auto-compaction when opted out (deterministic test mode).
    // Explicit POST /compact still works regardless.
    if std::env::var("TDB_SEARCH_DISABLE_AUTO_COMPACTION").unwrap_or_default() == "1" {
        lance_store.set_auto_compaction_enabled(false);
        eprintln!("[config] auto-compaction DISABLED (TDB_SEARCH_DISABLE_AUTO_COMPACTION=1)");
    }

    let service = SearchService::new(lance_store, config.clone(), tokenizer);
    let app = http_api::router(service, config);

    let addr = SocketAddr::from(([0, 0, 0, 0, 0, 0, 0, 0], 8080));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};
    let mut sigterm = signal(SignalKind::terminate()).expect("failed to install SIGTERM handler");
    let mut sigint = signal(SignalKind::interrupt()).expect("failed to install SIGINT handler");
    tokio::select! {
        _ = sigterm.recv() => {}
        _ = sigint.recv() => {}
    }
}
