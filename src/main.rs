#![forbid(unsafe_code)]
// Some kernel types are defined ahead of the code paths that exercise them
// through the HTTP layer. Suppress dead-code for the kernel and store.
#![allow(dead_code)]

//! tdb-search — standalone semantic search engine.
//!
//! Binds [::]:8080, answers liveness immediately, defers all heavy work.

mod config;
mod http_api;
mod kernel;
mod service;
mod store;

use std::net::SocketAddr;

use config::Config;
use service::SearchService;
use store::InMemoryStore;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::from_env();
    let store = InMemoryStore::new();
    let service = SearchService::new(store, config.clone());
    let app = http_api::router(service, config);

    let addr = SocketAddr::from(([0, 0, 0, 0, 0, 0, 0, 0], 8080));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
