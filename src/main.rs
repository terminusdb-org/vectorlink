#![forbid(unsafe_code)]

//! tdb-search server — standalone semantic search engine.
//!
//! Binds [::]:8080, answers liveness immediately, defers all heavy work.
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
use tdb_search::chunk;
use tdb_search::config::Config;
use tdb_search::embed::{self, EmbeddingRole};
use tdb_search::embed::cache::EmbedCache;
use tdb_search::http_api;
use tdb_search::kernel::model::{parse_domain, Domain};
use tdb_search::service::SearchService;
use tdb_search::store::lance::{encode_domain_path, LanceStore};

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
    let lance_store = Arc::new(LanceStore::new(data_dir, dim));

    // Disable probabilistic auto-compaction when opted out (deterministic test mode).
    // Explicit POST /compact still works regardless.
    if std::env::var("TDB_SEARCH_DISABLE_AUTO_COMPACTION").unwrap_or_default() == "1" {
        lance_store.set_auto_compaction_enabled(false);
        eprintln!("[config] auto-compaction DISABLED (TDB_SEARCH_DISABLE_AUTO_COMPACTION=1)");
    }

    let port = config.port;
    let data_dir = config.data_dir.clone();
    let model_name = config.embed_provider.model_name().to_owned();
    let service = SearchService::new(lance_store, config.clone(), tokenizer);
    let app = http_api::router(service, config);

    let addr = SocketAddr::from(([0, 0, 0, 0, 0, 0, 0, 0], port));
    eprintln!("[tdb-search] starting on port {} | data_dir={} | model={}", port, data_dir, model_name);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("[tdb-search] listening on {}", addr);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
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
    let cache_size: Option<usize> = match std::env::var("TDB_SEARCH_EMBED_CACHE_SIZE") {
        Ok(s) if s.eq_ignore_ascii_case("none") => None,
        Ok(s) => Some(s.parse::<usize>().unwrap_or(20_000)),
        Err(_) => Some(20_000),
    };

    eprintln!("[prime] cache_dir={} cache_size={:?}", cache_dir.display(), cache_size);
    let cache = EmbedCache::open(&cache_dir, cache_size);

    if !cache.is_enabled() {
        return Err("cache is disabled (TDB_SEARCH_EMBED_CACHE_SIZE=None). Cannot prime.".into());
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
            if total_rows % 10_000 == 0 {
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
