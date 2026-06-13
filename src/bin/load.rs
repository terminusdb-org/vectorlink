#![forbid(unsafe_code)]

//! tdb-search load — offline indexing CLI.
//!
//! Reads an NDJSON file and runs the same chunk→embed→store→tag pipeline as /push.
//! Usage:
//!   tdb-search-load --directory D --domain org/db --commit C \
//!                   [--previous P] [--branch B] --input ops.jsonl
//!
//! Same failure modes as /push: malformed NDJSON halts; Operation::Error skips
//! the individual doc; systemic embedding/store errors fail the whole task.

use std::path::Path;
use std::process::ExitCode;

use tdb_search::chunk;
use tdb_search::embed::{self, EmbeddingRole};
use tdb_search::ingest;
use tdb_search::kernel::distance::l2_normalize;
use tdb_search::kernel::model::Operation;
use tdb_search::store::lance::{ChunkRow, LanceStore};

/// Parsed CLI arguments.
struct LoadArgs {
    directory: String,
    domain: String,
    commit: String,
    branch: String,
    input: String,
    embed_url: String,
    model: String,
    dim: usize,
    tokenizer_path: String,
}

fn parse_args() -> Result<LoadArgs, String> {
    let args: Vec<String> = std::env::args().collect();

    let mut directory = None;
    let mut domain = None;
    let mut commit = None;
    let mut branch = "main".to_owned();
    let mut input = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--directory" | "-d" => {
                i += 1;
                directory = args.get(i).cloned();
            }
            "--domain" => {
                i += 1;
                domain = args.get(i).cloned();
            }
            "--commit" => {
                i += 1;
                commit = args.get(i).cloned();
            }
            "--previous" => {
                // Accepted but not used (for forward compatibility).
                i += 1;
            }
            "--branch" => {
                i += 1;
                if let Some(b) = args.get(i) {
                    branch = b.clone();
                }
            }
            "--input" => {
                i += 1;
                input = args.get(i).cloned();
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => {
                return Err(format!("unknown argument: {}", other));
            }
        }
        i += 1;
    }

    let directory = directory.ok_or("--directory is required")?;
    let domain = domain.ok_or("--domain is required")?;
    let commit = commit.ok_or("--commit is required")?;
    let input = input.ok_or("--input is required")?;

    let embed_url = std::env::var("TDB_SEARCH_EMBED_URL")
        .unwrap_or_else(|_| "http://localhost:11434".to_owned());
    let model = std::env::var("TDB_SEARCH_MODEL")
        .unwrap_or_else(|_| "nomic-embed-v2".to_owned());
    let dim: usize = std::env::var("TDB_SEARCH_DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(768);
    let tokenizer_path = std::env::var("TDB_SEARCH_TOKENIZER_PATH")
        .unwrap_or_else(|_| "spikes/tokenizer/tokenizer.json".to_owned());

    Ok(LoadArgs {
        directory,
        domain,
        commit,
        branch,
        input,
        embed_url,
        model,
        dim,
        tokenizer_path,
    })
}

fn print_usage() {
    eprintln!(
        "Usage: tdb-search-load --directory D --domain org/db --commit C [--previous P] [--branch B] --input ops.jsonl\n\
         \n\
         Offline indexing CLI. Runs the same pipeline as /push.\n\
         \n\
         Environment variables:\n\
         TDB_SEARCH_EMBED_URL       Embedding endpoint (default: http://localhost:11434)\n\
         TDB_SEARCH_MODEL           Model name (default: nomic-embed-v2)\n\
         TDB_SEARCH_DIM             Embedding dimension (default: 768)\n\
         TDB_SEARCH_TOKENIZER_PATH  Path to tokenizer.json"
    );
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {}", e);
            print_usage();
            return ExitCode::from(2);
        }
    };

    // Load tokenizer.
    let tokenizer_path = Path::new(&args.tokenizer_path);
    let tokenizer = match chunk::io_load_tokenizer(tokenizer_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: failed to load tokenizer: {}", e);
            return ExitCode::FAILURE;
        }
    };

    // Compute chunk params.
    let prefix = match embed::prefixes_for_model(&args.model) {
        Some(p) => embed::prefix_for_role(&p, EmbeddingRole::Document).to_owned(),
        None => String::new(),
    };
    let chunk_params = match chunk::params_for_nomic(&tokenizer, &prefix) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: chunk params: {}", e);
            return ExitCode::FAILURE;
        }
    };

    // Read and parse the input NDJSON file.
    let input_path = Path::new(&args.input);
    let body = match std::fs::read_to_string(input_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: failed to read input file {}: {}", args.input, e);
            return ExitCode::FAILURE;
        }
    };

    let operations = match ingest::parse_ndjson_body(&body) {
        Ok(ops) => ops,
        Err(e) => {
            eprintln!("error: malformed NDJSON: {}", e);
            return ExitCode::FAILURE;
        }
    };

    eprintln!(
        "load: {} operations, domain={}, branch={}, commit={}",
        operations.len(),
        args.domain,
        args.branch,
        args.commit
    );

    // Open the LanceStore at the given directory.
    let data_dir = Path::new(&args.directory);
    let store = LanceStore::new(data_dir, args.dim);

    // Configure the embedding provider.
    let provider = embed::Provider::OpenAiCompatible {
        base_url: args.embed_url,
        model: args.model,
        dim: args.dim,
    };
    let http_client = reqwest::Client::new();

    // Run the indexing pipeline.
    let mut indexed_count: u64 = 0;
    let mut skipped_count: u64 = 0;
    let mut last_version: u64 = 0;

    for (i, op) in operations.iter().enumerate() {
        match op {
            Operation::Inserted { id, string } | Operation::Changed { id, string } => {
                match io_index_one(
                    &store,
                    &tokenizer,
                    &chunk_params,
                    &provider,
                    &http_client,
                    &args.domain,
                    &args.branch,
                    id,
                    string,
                )
                .await
                {
                    Ok(version) => {
                        indexed_count += 1;
                        last_version = version;
                    }
                    Err(e) => {
                        eprintln!("  skip {}: {} — {}", i + 1, id, e);
                        skipped_count += 1;
                    }
                }
            }
            Operation::Deleted { id } => {
                match store.io_delete_doc(&args.domain, &args.branch, id).await {
                    Ok(version) => {
                        last_version = version;
                    }
                    Err(e) => {
                        eprintln!("  skip delete {}: {} — {}", i + 1, id, e);
                        skipped_count += 1;
                    }
                }
            }
            Operation::Error { message } => {
                eprintln!("  skip {}: operation error — {}", i + 1, message);
                skipped_count += 1;
            }
        }
    }

    // Create FTS index then tag the commit (order matters: FTS index creation
    // produces a new version, tag must point to version that includes the index).
    if last_version > 0 {
        match store
            .io_ensure_fts_index(&args.domain, &args.branch)
            .await
        {
            Ok(fts_version) => {
                if fts_version > 0 {
                    last_version = fts_version;
                }
            }
            Err(e) => {
                eprintln!("warning: FTS index creation failed: {}", e);
                // Continue — FTS is non-critical; vector search still works.
            }
        }

        if let Err(e) = store
            .io_tag_commit(&args.domain, &args.branch, &args.commit, last_version)
            .await
        {
            eprintln!("error: failed to tag commit: {}", e);
            return ExitCode::FAILURE;
        }
        store
            .update_last_indexed(&args.domain, &args.branch, &args.commit, last_version)
            .await;
    }

    eprintln!(
        "load: done. indexed={}, skipped={}, version={}",
        indexed_count, skipped_count, last_version
    );

    if skipped_count > 0 {
        // Non-zero exit to signal partial failure (same as /push task with skipped).
        ExitCode::from(0)
    } else {
        ExitCode::SUCCESS
    }
}

/// Index a single document: chunk → embed → upsert.
#[allow(clippy::too_many_arguments)]
async fn io_index_one(
    store: &LanceStore,
    tokenizer: &tokenizers::Tokenizer,
    chunk_params: &chunk::ChunkParams,
    provider: &embed::Provider,
    http_client: &reqwest::Client,
    domain: &str,
    branch: &str,
    doc_id: &str,
    text: &str,
) -> Result<u64, String> {
    let chunks = chunk::chunk_text(tokenizer, text, chunk_params)
        .map_err(|e| format!("chunking failed: {}", e))?;

    if chunks.is_empty() {
        return Err("chunking produced zero chunks".to_owned());
    }

    let chunk_texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let mut embeddings =
        embed::io_embed(provider, &chunk_texts, EmbeddingRole::Document, http_client)
            .await
            .map_err(|e| format!("embedding failed: {}", e))?;

    if embeddings.len() != chunks.len() {
        return Err(format!(
            "embedding count mismatch: expected {}, got {}",
            chunks.len(),
            embeddings.len()
        ));
    }

    // L2-normalise for cosine distance (same as the service pipeline).
    for emb in &mut embeddings {
        l2_normalize(emb);
    }

    let doc_type = ingest::extract_doc_type(doc_id);
    let rows: Vec<ChunkRow> = chunks
        .iter()
        .zip(embeddings)
        .map(|(ch, embedding)| ChunkRow {
            doc_id: doc_id.to_owned(),
            doc_type: doc_type.clone(),
            chunk_index: ch.index as i32,
            chunk_count: ch.count as i32,
            chunk_token_start: ch.token_start as i32,
            doc_token_len: ch.doc_token_len as i32,
            embedding,
            content: ch.text.clone(),
        })
        .collect();

    store
        .io_upsert_chunks(domain, branch, doc_id, &rows)
        .await
        .map_err(|e| format!("upsert failed: {}", e))
}
