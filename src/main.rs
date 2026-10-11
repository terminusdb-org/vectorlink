// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 DFRNT AB

#![forbid(unsafe_code)]

//! vectorlink server — standalone semantic search engine.
//!
//! Binds [::]:7372, answers liveness immediately, defers all heavy work.
//!
//! Subcommands:
//!   (no args)            — start the HTTP server
//!   prime-embed-cache    — read LanceDB embeddings and prime the sled cache

use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{Array, FixedSizeListArray, Float32Array, StringArray};
use futures::TryStreamExt;
use lance::dataset::Dataset;
use vectorlink::chunk;
use vectorlink::config::Config;
use vectorlink::embed::{self, EmbeddingRole};
use vectorlink::embed::cache::EmbedCache;
use vectorlink::http_api;
use vectorlink::kernel::model::{parse_domain, Domain};
use vectorlink::metrics;
use vectorlink::service::SearchService;
use vectorlink::store::lance::{encode_domain_path, LanceStore};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 && args[1] == "prime-embed-cache" {
        return prime_embed_cache(&args[2..]).await;
    }

    run_server().await
}

async fn run_server() -> Result<(), Box<dyn std::error::Error>> {
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
    let lance_store = Arc::new(LanceStore::new(
        data_dir,
        dim,
        config.lance_index_cache_bytes,
        config.lance_metadata_cache_bytes,
    ));

    // Startup cleanup: delete any stale __compact_rebuild branches left by
    // a crashed compaction. This is a critical safety check — a lingering
    // rebuild branch indicates a crashed compaction and should not be served.
    let cleaned = lance_store.io_cleanup_compaction_branches().await?;
    if !cleaned.is_empty() {
        eprintln!(
            "[startup] cleaned {} stale {} branch(es) from crashed compaction(s)",
            cleaned.len(),
            vectorlink::store::lance::COMPACT_REBUILD_PREFIX
        );
    }

    let port = config.port;
    let data_dir = config.data_dir.clone();
    let model_name = config.embed_provider.model_name().to_owned();
    let prometheus_port = config.prometheus_port;
    // Clone the store Arc for the shutdown handler (before it's moved into SearchService).
    let store_for_shutdown = Arc::clone(&lance_store);

    let service = SearchService::new(lance_store, config.clone(), tokenizer);
    let app = http_api::router(service.clone(), config);

    // Spawn Prometheus metrics server on a separate port if configured.
    if let Some(metrics_port) = prometheus_port {
        tokio::spawn(async move {
            if let Err(e) = metrics::start_metrics_server(metrics_port, service).await {
                eprintln!("[metrics] server error: {}", e);
            }
        });
    }

    let addr = SocketAddr::from(([0, 0, 0, 0, 0, 0, 0, 0], port));
    eprintln!("[vectorlink] starting on port {} | data_dir={} | model={}", port, data_dir, model_name);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("[vectorlink] listening on {}", addr);

    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            shutdown_signal().await;

            // Wait for all in-flight pipeline tasks to complete.
            // These tasks are tokio::spawn'd background tasks that write data to
            // Lance. Killing them mid-write loses data (versions get lost).
            // We poll every 200ms with a timeout.
            let wait_start = std::time::Instant::now();
            let max_wait = std::time::Duration::from_secs(120);
            loop {
                let active = store_for_shutdown
                    .pipeline_active_tasks
                    .load(std::sync::atomic::Ordering::Relaxed);
                if active == 0 {
                    eprintln!("[shutdown] all pipeline tasks complete, exiting gracefully");
                    break;
                }
                if wait_start.elapsed() > max_wait {
                    eprintln!(
                        "[shutdown] WARNING: {} pipeline task(s) still active after {}s, forcing exit",
                        active, max_wait.as_secs()
                    );
                    break;
                }
                eprintln!(
                    "[shutdown] waiting for {} pipeline task(s)... ({}s elapsed)",
                    active,
                    wait_start.elapsed().as_secs()
                );
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            }

            // Give axum a short grace period to finish sending HTTP
            // responses on in-flight connections, then force exit.
            // Without this, axum waits indefinitely for long-running
            // /push streams from TerminusDB to close, blocking shutdown.
            eprintln!("[shutdown] allowing 5s for in-flight HTTP responses, then exiting");
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            std::process::exit(0);
        })
        .await?;
    Ok(())
}

/// Parse `--from-domain <value>` and `--include-clustering` from args.
struct PrimeCacheArgs {
    from_domain: String,
    include_clustering: bool,
}

fn parse_prime_cache_args(args: &[String]) -> Result<PrimeCacheArgs, String> {
    let mut from_domain = None;
    let mut include_clustering = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--from-domain" => {
                i += 1;
                from_domain = args.get(i).cloned();
            }
            "--include-clustering" => {
                include_clustering = true;
            }
            other => {
                return Err(format!("unknown argument: {}", other));
            }
        }
        i += 1;
    }

    Ok(PrimeCacheArgs {
        from_domain: from_domain.ok_or("--from-domain is required")?,
        include_clustering,
    })
}

/// Read embeddings from a LanceDB dataset and populate the sled embed cache.
async fn prime_embed_cache(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let parsed = parse_prime_cache_args(args)?;
    let config = Config::from_env();

    let model = config.embed_provider.model_name();
    let dim = config.embed_provider.expected_dim();
    let data_dir = Path::new(&config.data_dir);

    // Parse the graph spec → domain string.
    let rp = parse_domain(&parsed.from_domain)
        .map_err(|e| format!("invalid --from-domain '{}': {}", parsed.from_domain, e))?;
    let domain = Domain::from_resource_path(&rp);
    let domain_str = domain.as_str().to_owned();

    // Construct the dataset path.
    let safe_domain = encode_domain_path(&domain_str);
    let dataset_path = data_dir.join(format!("{}.lance", safe_domain));

    if !dataset_path.exists() {
        return Err(format!(
            "dataset not found: {} (domain={})",
            dataset_path.display(),
            domain_str
        ).into());
    }

    let uri = dataset_path.to_string_lossy().to_string();
    eprintln!("[prime] opening dataset: {}", uri);
    eprintln!("[prime] model={} dim={} domain={}", model, dim, domain_str);

    // Open sled cache.
    let cache_dir = data_dir.join("embed_cache");
    let cache_size: Option<usize> = match std::env::var("VECTORLINK_EMBED_CACHE_SIZE") {
        Ok(s) if s.eq_ignore_ascii_case("none") => None,
        Ok(s) => Some(s.parse::<usize>().unwrap_or(20_000)),
        Err(_) => Some(20_000),
    };

    eprintln!("[prime] cache_dir={} cache_size={:?}", cache_dir.display(), cache_size);
    let cache = EmbedCache::open(&cache_dir, cache_size);

    if !cache.is_enabled() {
        return Err("cache is disabled (VECTORLINK_EMBED_CACHE_SIZE=None). Cannot prime.".into());
    }

    // Open the Lance dataset.
    let ds = Dataset::open(&uri).await?;

    // Build projection columns.
    let columns: Vec<&str> = if parsed.include_clustering {
        vec!["content", "embedding", "clustering_embedding"]
    } else {
        vec!["content", "embedding"]
    };

    let mut scanner = ds.scan();
    scanner.project(&columns)?;

    let stream = scanner.try_into_stream().await?;

    let mut total_rows: u64 = 0;
    let mut total_cached: u64 = 0;
    let start = std::time::Instant::now();

    futures::pin_mut!(stream);
    while let Some(batch) = stream.try_next().await? {
        let contents = batch
            .column_by_name("content")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let embeddings = batch
            .column_by_name("embedding")
            .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>());
        let clustering_embeddings = if parsed.include_clustering {
            batch
                .column_by_name("clustering_embedding")
                .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>())
        } else {
            None
        };

        let (Some(contents), Some(embeddings)) = (contents, embeddings) else {
            eprintln!("[prime] WARNING: batch missing content or embedding column, skipping");
            continue;
        };

        for row_idx in 0..contents.len() {
            let content = contents.value(row_idx);

            // Document embedding.
            let emb_values = embeddings.value(row_idx);
            let emb_flat = emb_values
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or("embedding column is not Float32")?;
            let embedding: Vec<f32> = emb_flat.values().to_vec();

            if embedding.len() == dim {
                let prefixed = embed::apply_prefix(
                    &[content.to_owned()],
                    model,
                    EmbeddingRole::Document,
                );
                let key = EmbedCache::cache_key(model, &prefixed[0]);
                cache.put(&key, &embedding);
                total_cached += 1;
            }

            // Clustering embedding (optional).
            if let Some(clust_arr) = clustering_embeddings {
                let clust_values = clust_arr.value(row_idx);
                let clust_flat = clust_values
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or("clustering_embedding column is not Float32")?;
                let clust_emb: Vec<f32> = clust_flat.values().to_vec();

                if clust_emb.len() == dim {
                    let prefixed = embed::apply_prefix(
                        &[content.to_owned()],
                        model,
                        EmbeddingRole::Clustering,
                    );
                    let key = EmbedCache::cache_key(model, &prefixed[0]);
                    cache.put(&key, &clust_emb);
                    total_cached += 1;
                }
            }

            total_rows += 1;
            if total_rows.is_multiple_of(10_000) {
                eprintln!(
                    "[prime] domain={} rows={} cached={} elapsed={}ms",
                    domain_str,
                    total_rows,
                    total_cached,
                    start.elapsed().as_millis()
                );
            }
        }
    }

    cache.flush();

    eprintln!(
        "[prime] DONE domain={} rows={} cached={} entries={} elapsed={}ms",
        domain_str,
        total_rows,
        total_cached,
        cache.len(),
        start.elapsed().as_millis()
    );

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
