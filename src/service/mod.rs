// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 DFRNT AB

#![forbid(unsafe_code)]

//! Service — the transport-agnostic core API surface.
//! Owns no framework types (no axum/hyper in signatures).
//! Composes store operations and validates domain logic.
//! Wires the full pipeline: parse → chunk → embed → store → tag.

use std::sync::Arc;

use tokio::sync::mpsc;
use tokenizers::Tokenizer;

use crate::chunk::{self, ChunkParams};
use crate::config::Config;
use crate::embed::{self, EmbedResult, EmbeddingRole, Provider};
use crate::ingest;
use crate::kernel::distance::l2_normalize;
use crate::kernel::error::ServiceError;
use crate::kernel::model::{
    parse_domain, BranchName, Domain, DuplicateGroup, InternalStats, LastIndexed, Operation,
    ProgressUpdate, Ref, SearchHit, SearchMode, SkippedDoc, Statistics, TaskStatus,
};
use crate::store::lance::{
    ChunkRow, DuplicateScope, EmbeddingRecord, LanceStore, SearchQuery, dedup_chunks_to_documents,
};

use serde::Serialize;

/// Response body for GET /embeddings.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingsResult {
    /// Document-role embeddings keyed by doc_id.
    pub doc_embeddings: std::collections::HashMap<String, Vec<f32>>,
    /// Clustering-role embeddings keyed by doc_id (empty when clustering disabled).
    pub clustering_embeddings: std::collections::HashMap<String, Vec<f32>>,
    /// Whether clustering embeddings are stored for this domain.
    pub store_clustering: bool,
    /// The commit that was searched.
    pub served_commit: String,
}

/// Streaming response for GET /embeddings (NDJSON mode).
/// Contains metadata + a channel receiver for individual embedding records.
pub struct EmbeddingsStreamResult {
    /// The commit that was searched.
    pub served_commit: String,
    /// Whether clustering embeddings are stored for this domain.
    pub store_clustering: bool,
    /// Total number of unique doc_ids that will be streamed.
    pub total_count: u64,
    /// Channel receiver yielding `EmbeddingRecord` values.
    pub receiver: tokio::sync::mpsc::Receiver<EmbeddingRecord>,
}

/// A single candidate neighbour in the /candidates response.
#[derive(Debug, Clone, Serialize)]
pub struct CandidateEntry {
    pub id: String,
    pub distance: f32,
    /// Clustering-role distance (only present when clustering is enabled for the domain).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clustering_distance: Option<f32>,
    /// Embedding of the neighbour doc (only present if requested via `include`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    /// Concatenated chunk text of the neighbour doc (only present if requested).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// Stats for the /candidates response.
#[derive(Debug, Clone, Serialize)]
pub struct CandidatesStats {
    pub elapsed_ms: u64,
    pub set_points: usize,
    pub target_points: usize,
    pub set_to_target_edges: usize,
    pub target_to_set_edges: usize,
    /// Whether clustering_distance is populated for this domain.
    pub store_clustering: bool,
}

/// Response body for POST /candidates.
#[derive(Debug, Clone, Serialize)]
pub struct CandidatesResult {
    pub set_to_target: std::collections::HashMap<String, Vec<CandidateEntry>>,
    pub target_to_set: std::collections::HashMap<String, Vec<CandidateEntry>>,
    pub stats: CandidatesStats,
}

/// Max concurrent heavy-scan requests (/resolve, /duplicates). Each such request
/// may spike ~64-96 working FDs via ANN queries; capping concurrency prevents
/// stacked spikes from exhausting the default nofile=1024 limit under load.
const HEAVY_SCAN_MAX_CONCURRENT: usize = 4;

/// The search service — owns the store and config, provides the transport-agnostic API.
#[derive(Clone)]
pub struct SearchService {
    pub store: Arc<LanceStore>,
    config: Config,
    tokenizer: Arc<Tokenizer>,
    chunk_params: ChunkParams,
    http_client: reqwest::Client,
    embed_cache: crate::embed::cache::EmbedCache,
    /// Per-capability readiness: index readiness is independent of search
    /// readiness (search additionally requires a warm embedding backend).
    ready_index: Arc<std::sync::atomic::AtomicBool>,
    ready_search: Arc<std::sync::atomic::AtomicBool>,
    /// Semaphore limiting concurrent heavy-scan operations (/resolve, /duplicates)
    /// to prevent stacked FD spikes from exceeding nofile under load (BUG-FD24).
    heavy_scan_semaphore: Arc<tokio::sync::Semaphore>,
}

/// Outcome of a search that resolves through the catch-up layer (RISK-15).
/// Carries the commit actually SERVED so the transport can report it truthfully
/// via `TerminusDB-Data-Version` — a stale (ancestor) result is never dressed up
/// as fresh.
#[derive(Debug, Clone)]
pub struct SearchOutcome {
    pub hits: Vec<SearchHit>,
    /// The commit whose snapshot was actually searched. Equals the requested
    /// commit when exact; the nearest indexed ancestor under lag (⇒ stale).
    pub served_commit: String,
}

/// Outcome of a `/similar` request — same staleness contract as `SearchOutcome`
/// (#A): carries the commit actually served so the transport reports it via
/// `TerminusDB-Data-Version` and never dresses a stale ancestor up as fresh.
#[derive(Debug, Clone)]
pub struct SimilarOutcome {
    pub hits: Vec<SearchHit>,
    /// The commit whose snapshot was actually searched (exact or proven ancestor).
    pub served_commit: String,
}

/// Result of a `/compare` request — stateless distance between two texts.
#[derive(Debug, Clone)]
pub struct CompareResult {
    /// Normalized cosine distance on the [0, 1] reference scale.
    /// 0 = identical, ~0.5 = unrelated, 1 = opposite.
    pub distance: f32,
    /// Which embedding role was applied to the source text.
    pub source_role: String,
    /// Which embedding role was applied to the target text.
    pub target_role: String,
}

/// Outcome of a `/suggest` (typeahead) request.
/// FTS-only (no embedding), carries approximate match count, completions,
/// and the first N document IDs for UI typeahead assistance.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SuggestOutcome {
    pub served_commit: String,
    pub total_approx: usize,
    pub completions: Vec<String>,
    pub hits: Vec<crate::store::lance::SuggestHit>,
}

impl std::fmt::Debug for SearchService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchService")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl SearchService {
    pub fn new(
        store: Arc<LanceStore>,
        config: Config,
        tokenizer: Tokenizer,
    ) -> Self {
        let embed_cache = {
            let cache_dir = std::path::Path::new(&config.data_dir).join("embed_cache");
            crate::embed::cache::EmbedCache::open(&cache_dir, config.embed_cache_size)
        };
        let prefix = match embed::prefixes_for_model(config.embed_provider.model_name()) {
            Some(p) => embed::prefix_for_role(&p, EmbeddingRole::Document).to_owned(),
            None => String::new(),
        };
        let chunk_params = chunk::params_for_nomic(&tokenizer, &prefix)
            .unwrap_or(ChunkParams { max_tokens: 480, overlap: 64 });

        let ready_index = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let search_ready = std::env::var("VECTORLINK_SEARCH_READY")
            .map(|v| v != "false")
            .unwrap_or(true);
        let ready_search = Arc::new(std::sync::atomic::AtomicBool::new(search_ready));

        Self {
            store,
            config,
            tokenizer: Arc::new(tokenizer),
            chunk_params,
            http_client: reqwest::Client::new(),
            embed_cache,
            ready_index,
            ready_search,
            heavy_scan_semaphore: Arc::new(tokio::sync::Semaphore::new(
                HEAVY_SCAN_MAX_CONCURRENT,
            )),
        }
    }

    /// Set search readiness (used in tests to control the readiness state).
    pub fn set_search_ready(&self, ready: bool) {
        self.ready_search
            .store(ready, std::sync::atomic::Ordering::SeqCst);
    }

    /// Set index readiness.
    pub fn set_index_ready(&self, ready: bool) {
        self.ready_index
            .store(ready, std::sync::atomic::Ordering::SeqCst);
    }

    /// Check if the service is ready for indexing.
    pub fn is_index_ready(&self) -> bool {
        self.ready_index.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Check if the service is ready for search.
    pub fn is_search_ready(&self) -> bool {
        self.ready_search.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Get last-indexed commit for a (domain, branch).
    pub async fn last_indexed(
        &self,
        domain_raw: &str,
        branch_raw: &str,
    ) -> Result<LastIndexed, ServiceError> {
        let rp =
            parse_domain(domain_raw).map_err(|e| ServiceError::Validation(e.to_string()))?;
        let domain = Domain::from_resource_path(&rp);
        let branch = BranchName::new(branch_raw.to_owned());
        self.store
            .last_indexed(&domain, &branch)
            .await
            .map_err(|e| ServiceError::Internal(e.to_string()))
    }

    /// Resolve a commit from either an explicit commit ID or a branch name.
    ///
    /// When `commit` is provided and non-empty, it is returned as-is.
    /// When `commit` is absent/empty but `branch` is provided, the latest
    /// indexed commit for that branch is returned. When both are absent/empty,
    /// returns 400 (commit or branch is required — no implicit latest).
    pub async fn resolve_commit_or_branch(
        &self,
        domain_raw: &str,
        commit: &str,
        branch: Option<&str>,
    ) -> Result<String, ServiceError> {
        if !commit.is_empty() {
            return Ok(commit.to_owned());
        }
        let branch_name = match branch.filter(|b| !b.is_empty()) {
            Some(b) => b,
            None => {
                return Err(ServiceError::Validation(
                    "missing required query parameter: commit or branch".to_owned(),
                ));
            }
        };
        let last = self.last_indexed(domain_raw, branch_name).await?;
        last.commit
            .ok_or_else(|| ServiceError::NotFound(format!(
                "no indexed commit for branch {} on domain {}",
                branch_name, domain_raw
            )))
    }

    /// Start an async push/index task.
    /// Real pipeline: parse NDJSON → chunk → embed → upsert → tag → update last-indexed.
    pub async fn push(
        &self,
        domain_raw: &str,
        branch_raw: &str,
        target_commit: &str,
        parent_commit: Option<&str>,
        operations: Vec<Operation>,
    ) -> Result<String, ServiceError> {
        let rp =
            parse_domain(domain_raw).map_err(|e| ServiceError::Validation(e.to_string()))?;
        let domain = Domain::from_resource_path(&rp);
        let domain_str = domain.as_str().to_owned();
        let branch = branch_raw.to_owned();
        let commit = target_commit.to_owned();

        // GUARD (409 state machine): a commit is "committed" the moment its push
        // is ACCEPTED, not only once indexing tags it. Atomically check-and-reserve:
        // reject (409) if the commit is in ANY non-absent state — Reserved/Indexing
        // (a push already in flight) OR Indexed (already tagged). The check and the
        // reservation happen under one lock so two concurrent pushes of the same
        // commit cannot both pass (exactly one wins). On a proven-absent commit we
        // take the reservation here and MUST release it on every terminal path
        // below (success, failure, or panic) — otherwise a failed index would block
        // a legitimate retry forever.
        let reserved = self
            .store
            .io_try_reserve_commit(&domain_str, &branch, &commit)
            .await
            .map_err(|e| ServiceError::Internal(e.to_string()))?;
        if !reserved {
            return Err(ServiceError::Conflict(format!(
                "commit {} already pushed (in-flight or indexed)",
                commit
            )));
        }

        // Branch-out: if a parent commit is supplied and the target branch does
        // not yet exist, fork it from the parent's indexed version (block reuse).
        // A push to "main" or to an existing branch skips this (no-op). Fail loud
        // if the parent isn't indexed (cannot fork from nothing).
        //
        // RELEASE-ON-FAILURE: this runs AFTER the reservation is taken. If the
        // fork fails we must release the reservation before returning, otherwise
        // the commit stays permanently reserved and a legitimate retry is blocked.
        if branch != crate::store::lance::MAIN_BRANCH {
            if let Some(parent) = parent_commit {
                if let Err(e) = crate::store::branch::io_ensure_branch_forked(
                    &self.store,
                    &domain_str,
                    &branch,
                    parent,
                )
                .await
                {
                    self.store
                        .io_release_commit_reservation(&domain_str, &branch, &commit)
                        .await;
                    let msg = e.to_string();
                    return Err(if msg.contains("not indexed") || msg.contains("does not exist") {
                        ServiceError::NotFound(msg)
                    } else {
                        ServiceError::Internal(msg)
                    });
                }
            }
        }

        // No process-local enablement/negative-cache to manage here: a search
        // that 404'd before this push resolves on its next attempt purely from
        // the durable tag this push will write (task-durable-index-state — the
        // negative cache that previously needed busting here is gone).

        // Generate a task ID immediately.
        let task_id = format!("task-{}", uuid::Uuid::new_v4().as_simple());

        // Mark task as pending.
        self.store
            .record_task(
                &task_id,
                TaskStatus::Pending { percentage: 0.0 },
            )
            .await;

        // Clone what we need for the background task.
        let store = Arc::clone(&self.store);

        // Acquire the pipeline lock with a timeout. A previous pipeline for
        // this branch may still be winding down (releasing its lock). A 30s
        // timeout covers the edge case where the previous pipeline is
        // mid-embed-call; only true compaction (which doesn't check cancel
        // tokens) will exceed the timeout and get a 503 so TerminusDB retries.
        let pipeline_guard = match tokio::time::timeout(
            std::time::Duration::from_secs(30),
            store.acquire_pipeline_lock(&domain_str, &branch),
        ).await {
            Ok(g) => g,
            Err(_) => {
                self.store
                    .io_release_commit_reservation(&domain_str, &branch, &commit)
                    .await;
                return Err(ServiceError::Unavailable(format!(
                    "pipeline lock busy for {}/{}, retry later",
                    domain_str, branch
                )));
            }
        };

        let tokenizer = Arc::clone(&self.tokenizer);
        let chunk_params = self.chunk_params.clone();
        let provider = self.config.embed_provider.clone();
        let http_client = self.http_client.clone();
        let embed_batch_size = self.config.embed_batch_size;
        let embed_cache = self.embed_cache.clone();
        let task_id_clone = task_id.clone();

        // Clones for use after the spawned task (which moves the originals).
        let store_after = Arc::clone(&store);
        let domain_after = domain_str.clone();
        let branch_after = branch.clone();
        let commit_after = commit.clone();

        // Spawn the indexing pipeline as a background task.
        let store_for_panic = Arc::clone(&store);
        let task_id_for_panic = task_id.clone();
        let reservation_coords = (domain_str.clone(), branch.clone(), commit.clone());
        let handle = tokio::spawn(async move {
            let _pipeline_guard = pipeline_guard;
            store.pipeline_active_tasks.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let ctx = PipelineCtx {
                store: &store,
                tokenizer: &tokenizer,
                chunk_params: &chunk_params,
                provider: &provider,
                http_client: &http_client,
                embed_cache: &embed_cache,
                domain: &domain_str,
                branch: &branch,
                embed_batch_size,
            };
            let result = io_run_index_pipeline(&ctx, &commit, operations).await;
            store.pipeline_active_tasks.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            result
        });

        // Await the pipeline task before returning. This makes the push handler
        // synchronous — TerminusDB pushes commits sequentially, so awaiting
        // ensures the previous commit's pipeline completes (and its memory is
        // freed) before the next push is accepted.
        let result = match handle.await {
            Ok(inner) => inner,
            Err(join_err) => {
                let msg = if join_err.is_panic() {
                    format!("pipeline panicked: {:?}", join_err)
                } else {
                    format!("pipeline cancelled: {:?}", join_err)
                };
                store_for_panic
                    .record_task(&task_id_for_panic, TaskStatus::Error { error: msg })
                    .await;
                let (d, b, c) = &reservation_coords;
                store_for_panic
                    .io_release_commit_reservation(d, b, c)
                    .await;
                store_for_panic.pipeline_active_tasks.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(task_id);
            }
        };

        match result {
            Ok((indexed, skipped)) => {
                store_after
                    .record_task(
                        &task_id_clone,
                        TaskStatus::Complete {
                            indexed_documents: indexed,
                            skipped,
                        },
                    )
                    .await;
            }
            Err(error_msg) => {
                store_after
                    .record_task(
                        &task_id_clone,
                        TaskStatus::Error { error: error_msg },
                    )
                    .await;
            }
        }
        store_after
            .io_release_commit_reservation(&domain_after, &branch_after, &commit_after)
            .await;

        Ok(task_id)
    }

    /// Start an async push/index task that reads operations incrementally
    /// from a channel rather than buffering all operations upfront.
    /// This enables true streaming: the HTTP handler can start sending
    /// operations through the channel while the pipeline begins chunking
    /// and embedding them concurrently.
    ///
    /// When `progress_tx` is `Some`, the pipeline sends `ProgressUpdate` messages
    /// through it as NDJSON progress lines. The caller holds the `progress_rx`
    /// and streams its items as the HTTP response body.
    pub async fn push_stream(
        &self,
        domain_raw: &str,
        branch_raw: &str,
        target_commit: &str,
        parent_commit: Option<&str>,
        operation_rx: mpsc::Receiver<Operation>,
        progress_tx: Option<mpsc::Sender<ProgressUpdate>>,
    ) -> Result<String, ServiceError> {
        let rp =
            parse_domain(domain_raw).map_err(|e| ServiceError::Validation(e.to_string()))?;
        let domain = Domain::from_resource_path(&rp);
        let domain_str = domain.as_str().to_owned();
        let branch = branch_raw.to_owned();
        let commit = target_commit.to_owned();

        let reserved = self
            .store
            .io_try_reserve_commit(&domain_str, &branch, &commit)
            .await
            .map_err(|e| ServiceError::Internal(e.to_string()))?;
        if !reserved {
            return Err(ServiceError::Conflict(format!(
                "commit {} already pushed (in-flight or indexed)",
                commit
            )));
        }

        if branch != crate::store::lance::MAIN_BRANCH {
            if let Some(parent) = parent_commit {
                if let Err(e) = crate::store::branch::io_ensure_branch_forked(
                    &self.store,
                    &domain_str,
                    &branch,
                    parent,
                )
                .await
                {
                    self.store
                        .io_release_commit_reservation(&domain_str, &branch, &commit)
                        .await;
                    let msg = e.to_string();
                    return Err(if msg.contains("not indexed") || msg.contains("does not exist") {
                        ServiceError::NotFound(msg)
                    } else {
                        ServiceError::Internal(msg)
                    });
                }
            }
        }

        let task_id = format!("task-{}", uuid::Uuid::new_v4().as_simple());

        self.store
            .record_task(
                &task_id,
                TaskStatus::Pending { percentage: 0.0 },
            )
            .await;

        let store = Arc::clone(&self.store);
        let tokenizer = Arc::clone(&self.tokenizer);
        let chunk_params = self.chunk_params.clone();
        let provider = self.config.embed_provider.clone();
        let http_client = self.http_client.clone();
        let embed_batch_size = self.config.embed_batch_size;
        let embed_cache = self.embed_cache.clone();
        let task_id_clone = task_id.clone();

        let store_for_panic = Arc::clone(&store);
        let task_id_for_panic = task_id.clone();
        let reservation_coords = (domain_str.clone(), branch.clone(), commit.clone());
        let cancel_domain = domain_str.clone();
        let cancel_branch = branch.clone();

        // Create cancellation token for this pipeline task.
        let cancel_token = tokio_util::sync::CancellationToken::new();

        // Cancel any previous in-flight pipeline for this branch and register
        // the new token. This ensures stale tasks from a previous run (e.g.
        // before a reindex) are cancelled and cannot write old chunks.
        self.store
            .cancel_previous_pipeline_and_register(&cancel_domain, &cancel_branch, cancel_token.clone())
            .await;

        let cancel_token_for_task = cancel_token.clone();

        // Acquire the pipeline lock with a timeout. The cancel token above
        // already cancelled any previous push pipeline for this branch, so it
        // should release the lock within seconds. A 30s timeout covers the
        // edge case where the previous pipeline is mid-embed-call; only true
        // compaction (which doesn't check cancel tokens) will exceed the
        // timeout and get a 503 so TerminusDB retries.
        let pipeline_guard = match tokio::time::timeout(
            std::time::Duration::from_secs(30),
            store.acquire_pipeline_lock(&domain_str, &branch),
        ).await {
            Ok(g) => g,
            Err(_) => {
                self.store
                    .io_release_commit_reservation(&domain_str, &branch, &commit)
                    .await;
                self.store
                    .cancel_previous_pipeline_and_register(&cancel_domain, &cancel_branch, cancel_token.clone())
                    .await;
                return Err(ServiceError::Unavailable(format!(
                    "pipeline lock busy for {}/{}, retry later",
                    domain_str, branch
                )));
            }
        };

        let handle = tokio::spawn(async move {
            let _pipeline_guard = pipeline_guard;
            store.pipeline_active_tasks.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let ctx = PipelineCtx {
                store: &store,
                tokenizer: &tokenizer,
                chunk_params: &chunk_params,
                provider: &provider,
                http_client: &http_client,
                embed_cache: &embed_cache,
                domain: &domain_str,
                branch: &branch,
                embed_batch_size,
            };

            // The pipeline checks the cancellation token at every key point:
            // - before each operation recv
            // - before each sub-batch embed
            // - during each embed HTTP call via tokio::select!
            // This ensures immediate cancellation when a reindex triggers a
            // new push, releasing memory from the old pipeline.
            let result = io_run_index_pipeline_stream(
                &ctx, &commit, operation_rx, progress_tx, &cancel_token_for_task,
            ).await;

            // Unregister cancel token — pipeline completed (not cancelled).
            store.unregister_pipeline_cancel_token(&domain_str, &branch).await;

            // Record task status and release reservation inside the spawned task.
            match result {
                Ok((indexed, skipped)) => {
                    store
                        .record_task(
                            &task_id_clone,
                            TaskStatus::Complete {
                                indexed_documents: indexed,
                                skipped,
                            },
                        )
                        .await;
                }
                Err(error_msg) => {
                    store
                        .record_task(
                            &task_id_clone,
                            TaskStatus::Error { error: error_msg },
                        )
                        .await;
                }
            }
            store
                .io_release_commit_reservation(&domain_str, &branch, &commit)
                .await;
            store.pipeline_active_tasks.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        });

        // Spawn a monitor to catch panics — if the pipeline task panics,
        // record Error and release the reservation.
        tokio::spawn(async move {
            if let Err(join_err) = handle.await {
                let msg = if join_err.is_panic() {
                    format!("pipeline panicked: {:?}", join_err)
                } else {
                    format!("pipeline cancelled: {:?}", join_err)
                };
                store_for_panic
                    .record_task(&task_id_for_panic, TaskStatus::Error { error: msg })
                    .await;
                let (d, b, c) = &reservation_coords;
                store_for_panic
                    .io_release_commit_reservation(d, b, c)
                    .await;
                store_for_panic.unregister_pipeline_cancel_token(d, b).await;
                store_for_panic.pipeline_active_tasks.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            }
        });

        Ok(task_id)
    }

    /// Check the status of a push task.
    pub async fn check_task(&self, task_id: &str) -> Result<TaskStatus, ServiceError> {
        self.store
            .check_task(task_id)
            .await
            .ok_or_else(|| ServiceError::NotFound(format!("unknown task: {}", task_id)))
    }

    /// Assign a target commit to an existing source commit's index.
    pub async fn assign(
        &self,
        domain_raw: &str,
        source_commit: &str,
        target_commit: &str,
    ) -> Result<(), ServiceError> {
        let rp =
            parse_domain(domain_raw).map_err(|e| ServiceError::Validation(e.to_string()))?;
        let domain = Domain::from_resource_path(&rp);
        let domain_str = domain.as_str();

        // Use the branch carried by the domain graphspec, NOT a hardcoded "main"
        // (#B): `/assign` on `org/db/local/branch/feature` must resolve the
        // source tag on `feature`. Tag resolution is dataset-global, but the
        // create side of `io_assign_commit` needs the owning branch for a
        // non-main lineage.
        let branch = extract_branch(&rp);

        // Pure tag-pointer assign: resolve source → tag target to the same
        // version. NO embedding, NO recompute (P3-ASSIGN-1). Fail loud if the
        // source commit is not indexed.
        match self
            .store
            .io_assign_commit(domain_str, &branch, source_commit, target_commit)
            .await
        {
            Ok(_version) => Ok(()),
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("is not indexed") {
                    Err(ServiceError::NotFound(format!(
                        "source commit {} is not indexed",
                        source_commit
                    )))
                } else {
                    Err(ServiceError::Internal(msg))
                }
            }
        }
    }

    /// Search: embed query → vector/fts/hybrid search → dedup → return.
    /// Returns the `SearchOutcome` (hits + served commit for staleness reporting).
    pub async fn search(
        &self,
        domain_raw: &str,
        commit: &str,
        q: &str,
    ) -> Result<SearchOutcome, ServiceError> {
        self.search_with_options(
            domain_raw,
            commit,
            q,
            SearchMode::Hybrid,
            0,
            10,
            &[],
            &[],
            false,
            &[],
        )
        .await
    }

    /// Full search with all options. Returns a `SearchOutcome` carrying the
    /// commit actually SERVED (exact, or the nearest indexed ancestor under lag)
    /// so the transport reports staleness truthfully (RISK-15, P3-LAG-1).
    ///
    /// Catch-up resolution (never blocks, never silently stale, never leaks
    /// newer-than-requested data):
    ///  1. Exact: requested commit is indexed → serve it.
    ///  2. Lag: not indexed → walk the TerminusDB-supplied `ancestors` window
    ///     nearest-first; serve the FIRST indexed one (a PROVEN ancestor of the
    ///     requested commit). Report it as served (served ≠ requested ⇒ stale).
    ///     We NEVER serve the branch tip merely because it exists — only a
    ///     commit proven to be an ancestor (BLOCKER-1 snapshot isolation).
    ///  3. None: no indexed ancestor in the window → `NotFound` (404), negatively
    ///     cached per branch (TTL) so a repeat search doesn't re-walk history.
    ///
    /// `ancestors` is the nearest-first ancestor window (Spec 10 §5; last 10,
    /// then up to 1000). Empty window + not-exact ⇒ no provable ancestor ⇒ 404.
    #[allow(clippy::too_many_arguments)]
    pub async fn search_with_options(
        &self,
        domain_raw: &str,
        commit: &str,
        q: &str,
        mode: SearchMode,
        start: usize,
        count: usize,
        doc_type_filter: &[String],
        doc_id_filter: &[String],
        snippet: bool,
        ancestors: &[String],
    ) -> Result<SearchOutcome, ServiceError> {
        let rp =
            parse_domain(domain_raw).map_err(|e| ServiceError::Validation(e.to_string()))?;

        if !self.is_search_ready() {
            return Err(ServiceError::Unavailable(
                "search capability not ready (embedding backend cold)".to_owned(),
            ));
        }

        let domain = Domain::from_resource_path(&rp);
        let domain_str = domain.as_str().to_owned();
        let branch = extract_branch(&rp);

        // Resolve the searchable commit via catch-up (exact → proven nearest
        // ancestor in the supplied window → 404). Never serves newer data.
        // Pass the already-parsed Domain through (#7 — no re-parse of the
        // normalized string, no spurious 500 path).
        let served_commit = self
            .resolve_searchable_commit(&domain, &branch, commit, ancestors)
            .await?;

        // Embed the query text.
        let query_texts = vec![q.to_owned()];
        let embeddings = embed::io_embed(
            &self.config.embed_provider,
            &query_texts,
            EmbeddingRole::Query,
            &self.http_client,
            Some(&self.embed_cache),
        )
        .await
        .map_err(|e| ServiceError::Internal(format!("embedding failed: {}", e)))?;

        let mut query_embedding = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| ServiceError::Internal("no embedding returned".to_owned()))?;
        l2_normalize(&mut query_embedding);

        let search_query = SearchQuery {
            query_embedding: query_embedding.clone(),
            query_text: q.to_owned(),
            mode,
            start,
            count,
            doc_type_filter: doc_type_filter.to_vec(),
            doc_id_filter: doc_id_filter.to_vec(),
            snippet,
        };

        let chunk_hits = self
            .store
            .io_search(&domain_str, &branch, &served_commit, &search_query)
            .await
            .map_err(|e| ServiceError::Internal(e.to_string()))?;

        let results = dedup_chunks_to_documents(chunk_hits, snippet);

        // Apply pagination.
        let paginated: Vec<SearchHit> = results
            .into_iter()
            .skip(start)
            .take(count)
            .collect();

        Ok(SearchOutcome {
            hits: paginated,
            served_commit,
        })
    }

    /// Suggest (typeahead): FTS-only partial-query assistance.
    /// No embedding call — fast path using the existing FTS inverted index.
    /// Returns approximate match count, completion suggestions, and first N doc IDs.
    #[allow(clippy::too_many_arguments)]
    pub async fn suggest(
        &self,
        domain_raw: &str,
        commit: &str,
        q: &str,
        count: usize,
        doc_type_filter: &[String],
        doc_id_filter: &[String],
        ancestors: &[String],
    ) -> Result<SuggestOutcome, ServiceError> {
        let rp =
            parse_domain(domain_raw).map_err(|e| ServiceError::Validation(e.to_string()))?;

        if !self.is_search_ready() {
            return Err(ServiceError::Unavailable(
                "search capability not ready (embedding backend cold)".to_owned(),
            ));
        }

        let domain = Domain::from_resource_path(&rp);
        let domain_str = domain.as_str().to_owned();
        let branch = extract_branch(&rp);

        let served_commit = self
            .resolve_searchable_commit(&domain, &branch, commit, ancestors)
            .await?;

        let suggest_query = crate::store::lance::SuggestQuery {
            query_text: q.to_owned(),
            count,
            doc_type_filter: doc_type_filter.to_vec(),
            doc_id_filter: doc_id_filter.to_vec(),
        };

        let result = self
            .store
            .io_suggest(&domain_str, &branch, &served_commit, &suggest_query)
            .await
            .map_err(|e| ServiceError::Internal(e.to_string()))?;

        Ok(SuggestOutcome {
            served_commit,
            total_approx: result.total_approx,
            completions: result.completions,
            hits: result.hits,
        })
    }

    /// Resolve the requested commit to the commit whose snapshot will actually
    /// be searched (catch-up, RISK-15, BLOCKER-1). Exact if indexed; else the
    /// nearest PROVEN ancestor in the supplied window; else `NotFound`.
    ///
    /// SNAPSHOT ISOLATION (BLOCKER-1): we only ever serve a commit that is the
    /// requested commit itself OR a member of the nearest-first `ancestors`
    /// window TerminusDB supplied. We NEVER serve the branch tip just because it
    /// is the latest indexed commit — that could be a DESCENDANT of an older
    /// requested commit, leaking newer data the requested snapshot never had.
    ///
    /// Lineage gate (#5, Spec 10 §5): a branch with NO indexed lineage on disk
    /// is 404'd before any ancestor walk.
    ///
    /// DURABLE STATE (task-durable-index-state): every signal here — "is this
    /// commit indexed", "does this branch have any indexed lineage" — is read
    /// from the on-disk Lance tags, NOT a process-local map. There is NO negative
    /// cache: the previous 404 negative cache was the source of the
    /// restart-loses-state bug (it cached "no lineage" for a branch whose index
    /// was on disk) and, now that the lineage check is a cheap durable tag lookup
    /// and the ancestor walk is a pure in-memory pass over a single tag-list
    /// read, it guarded nothing worth keeping. Its removal makes the restart
    /// invariant trivially true: a branch with an on-disk index can NEVER be
    /// blocked, because nothing process-local outlives the truth on disk.
    async fn resolve_searchable_commit(
        &self,
        domain: &Domain,
        branch: &str,
        requested_commit: &str,
        ancestors: &[String],
    ) -> Result<String, ServiceError> {
        let domain_str = domain.as_str();

        // 1. Load the dataset-global commit→version map ONCE (fail-loud on a
        //    real tag-list error; absent dataset → empty map). This single
        //    durable read backs both the exact match and the pure in-memory
        //    nearest-ancestor walk below — no per-candidate I/O.
        let commit_versions = self
            .store
            .io_list_commit_versions(domain_str)
            .await
            .map_err(|e| ServiceError::Internal(e.to_string()))?;

        // 2. EXACT match: a directly-indexed commit is always serveable.
        if commit_versions.contains_key(requested_commit) {
            return Ok(requested_commit.to_owned());
        }

        // 3. Lineage gate (#5): a branch with NO indexed lineage at all → 404,
        //    before any ancestor walk. The signal is the durable on-disk
        //    last-indexed tag for the branch (survives a restart).
        if !self.branch_has_indexed_lineage(domain, branch).await? {
            return Err(ServiceError::NotFound(format!(
                "no indexed lineage for branch {} (commit {})",
                branch, requested_commit
            )));
        }

        // 4. Pure nearest-ancestor resolution over the SUPPLIED window only.
        //    The resolver re-checks exact (already handled above; harmless),
        //    then walks ancestors nearest-first, resolving each via the
        //    in-memory map — no per-candidate I/O. We never consult the tip.
        let resolved = crate::layeridx::resolve_nearest_layer(
            requested_commit,
            ancestors,
            |c: String| {
                let v = commit_versions.get(&c).copied();
                std::future::ready(Ok(v))
            },
        )
        .await
        .map_err(|e| ServiceError::Internal(e.to_string()))?;

        match resolved {
            crate::layeridx::ResolvedLayer::Exact { commit, .. } => Ok(commit),
            crate::layeridx::ResolvedLayer::Ancestor { served_commit, .. } => {
                // A PROVEN ancestor (member of the supplied window). Stale (the
                // caller reports it via TerminusDB-Data-Version), but never newer
                // than requested. The lag between requested and served is the
                // catch-up nudge signal — TerminusDB's push driver converges it.
                Ok(served_commit)
            }
            crate::layeridx::ResolvedLayer::None => {
                // Requested not indexed and no SUPPLIED ancestor is indexed. We
                // will NOT serve the tip (could be a descendant of the requested
                // commit) — 404. TerminusDB nudges a push to catch up.
                Err(ServiceError::NotFound(format!(
                    "no indexed ancestor for commit {} on branch {} within the supplied window",
                    requested_commit, branch
                )))
            }
        }
    }

    /// Whether `branch` has any indexed lineage — the gate for resolution (#5,
    /// Spec 10 §5).
    ///
    /// DURABLE (task-durable-index-state): the signal is the on-disk
    /// last-indexed tag for the branch (`store.last_indexed` derives it from
    /// the Lance tags on a cache miss), so it is correct immediately after a
    /// restart with no process-local enablement state. FAIL-LOUD: a real
    /// tag-read error propagates rather than being misread as "no lineage".
    async fn branch_has_indexed_lineage(
        &self,
        domain: &Domain,
        branch: &str,
    ) -> Result<bool, ServiceError> {
        let last = self
            .store
            .last_indexed(domain, &BranchName::new(branch.to_owned()))
            .await
            .map_err(|e| ServiceError::Internal(e.to_string()))?;
        Ok(last.commit.is_some())
    }

    /// Similar: look up doc by id → use its best embedding → vector search.
    pub async fn similar(
        &self,
        domain_raw: &str,
        commit: &str,
        id: &str,
    ) -> Result<SimilarOutcome, ServiceError> {
        self.similar_with_options(domain_raw, commit, id, 0, 10, &[], &[], false, &[])
            .await
    }

    /// Similar with full options. Routes through the SAME catch-up resolution as
    /// `/search` (#A): an un-indexed/lagging commit resolves to the nearest
    /// PROVEN ancestor (never a descendant) and the served commit is returned so
    /// the transport reports staleness via `TerminusDB-Data-Version`. A branch
    /// with no indexed lineage → 404 (not a raw "commit not indexed" 500).
    ///
    /// PROBE SIGNAL: uses the source document's stored DOCUMENT-role embedding
    /// (`embedding`) to probe the DOCUMENT-role embedding ANN index. This is a
    /// same-role doc→doc comparison — the correct signal for finding similar
    /// documents.
    ///
    /// SAME-ROLE EXACT-DUPLICATE OVERRIDE: after ranking, if the source document's
    /// DOCUMENT-role embedding is bit-identical to a result's DOCUMENT-role
    /// embedding (doc↔doc same-role equality), that result's distance is
    /// overridden to 0. This surfaces exact content duplicates at distance 0
    /// regardless of the ranking distance.
    #[allow(clippy::too_many_arguments)]
    pub async fn similar_with_options(
        &self,
        domain_raw: &str,
        commit: &str,
        id: &str,
        start: usize,
        count: usize,
        doc_type_filter: &[String],
        doc_id_filter: &[String],
        snippet: bool,
        ancestors: &[String],
    ) -> Result<SimilarOutcome, ServiceError> {
        let rp =
            parse_domain(domain_raw).map_err(|e| ServiceError::Validation(e.to_string()))?;

        if !self.is_search_ready() {
            return Err(ServiceError::Unavailable(
                "search capability not ready (embedding backend cold)".to_owned(),
            ));
        }

        let domain = Domain::from_resource_path(&rp);
        let domain_str = domain.as_str().to_owned();
        let branch = extract_branch(&rp);

        // Resolve the searchable commit via the SHARED catch-up path (#A) BEFORE
        // any store read — exact, nearest proven ancestor, or 404. Never serves
        // newer-than-requested data.
        let served_commit = self
            .resolve_searchable_commit(&domain, &branch, commit, ancestors)
            .await?;

        // Look up the document's chunks (indexed lookup, not a scan).
        let doc_chunks = self
            .store
            .io_lookup_doc_chunks(&domain_str, &branch, id)
            .await
            .map_err(|e| ServiceError::Internal(e.to_string()))?;

        if doc_chunks.is_empty() {
            return Err(ServiceError::NotFound(format!(
                "document {} not found in index",
                id
            )));
        }

        // PROBE: use the source document's stored DOCUMENT-role embedding (same-role
        // doc→doc probe). This is the correct comparison for finding similar documents.
        let probe_embedding = doc_chunks[0].embedding.clone();

        // FAIL-LOUD sanity check: the stored vector must be present and the right
        // dimension. A violation means the projection or the insert invariant is
        // broken — surface it rather than silently search on garbage.
        let expected_dim = self.config.embed_provider.expected_dim();
        if probe_embedding.len() != expected_dim {
            return Err(ServiceError::Internal(format!(
                "stored embedding for {} has dimension {}, expected {}",
                id,
                probe_embedding.len(),
                expected_dim
            )));
        }
        let norm_sq: f32 = probe_embedding.iter().map(|v| v * v).sum();
        if (norm_sq - 1.0).abs() > 1e-3 {
            return Err(ServiceError::Internal(format!(
                "stored embedding for {} is not L2-normalised (norm^2 = {}); \
                 the insert-time normalisation invariant is violated",
                id, norm_sq
            )));
        }

        let search_query = SearchQuery {
            query_embedding: probe_embedding.clone(),
            query_text: String::new(),
            mode: SearchMode::Vector,
            start,
            count: count + 1, // Over-fetch to exclude self.
            doc_type_filter: doc_type_filter.to_vec(),
            doc_id_filter: doc_id_filter.to_vec(),
            snippet,
        };

        let chunk_hits = self
            .store
            .io_search(&domain_str, &branch, &served_commit, &search_query)
            .await
            .map_err(|e| ServiceError::Internal(e.to_string()))?;

        let all_results = dedup_chunks_to_documents(chunk_hits, snippet);

        // Exclude self from results and apply pagination.
        let mut paginated: Vec<SearchHit> = all_results
            .into_iter()
            .filter(|hit| hit.id != id)
            .skip(start)
            .take(count)
            .collect();

        // SAME-ROLE EXACT-DUPLICATE OVERRIDE (doc↔doc): for each result, fetch its
        // stored DOCUMENT-role embedding and compare with the source's. If bit-
        // identical → override distance to 0.
        let result_ids: Vec<String> = paginated.iter().map(|h| h.id.clone()).collect();
        if !result_ids.is_empty() && !probe_embedding.is_empty() {
            let result_embeddings = self
                .store
                .io_fetch_result_embeddings(
                    &domain_str,
                    &branch,
                    &served_commit,
                    &result_ids,
                    &[],
                    "embedding",
                )
                .await
                .map_err(|e| ServiceError::Internal(e.to_string()))?;

            for hit in &mut paginated {
                if let Some(result_emb) = result_embeddings.get(&hit.id) {
                    if crate::kernel::distance::vectors_equal(&probe_embedding, result_emb) {
                        hit.distance = 0.0;
                    }
                }
            }

            // Re-sort after override: a hit that moved to 0.0 should sort first.
            paginated.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        Ok(SimilarOutcome {
            hits: paginated,
            served_commit,
        })
    }

    /// Similar by text: embed the input text with Document role and run vector
    /// search against all stored document embeddings. Uses search_document:
    /// prefix since the text represents document content, not a query.
    #[allow(clippy::too_many_arguments)]
    pub async fn similar_with_text(
        &self,
        domain_raw: &str,
        commit: &str,
        text: &str,
        start: usize,
        count: usize,
        doc_type_filter: &[String],
        doc_id_filter: &[String],
        snippet: bool,
        ancestors: &[String],
    ) -> Result<SimilarOutcome, ServiceError> {
        let rp =
            parse_domain(domain_raw).map_err(|e| ServiceError::Validation(e.to_string()))?;

        if !self.is_search_ready() {
            return Err(ServiceError::Unavailable(
                "search capability not ready (embedding backend cold)".to_owned(),
            ));
        }

        let domain = Domain::from_resource_path(&rp);
        let domain_str = domain.as_str().to_owned();
        let branch = extract_branch(&rp);

        let served_commit = self
            .resolve_searchable_commit(&domain, &branch, commit, ancestors)
            .await?;

        // Embed the text with Document role (search_document: prefix).
        let texts = vec![text.to_owned()];
        let embeddings = embed::io_embed(
            &self.config.embed_provider,
            &texts,
            EmbeddingRole::Document,
            &self.http_client,
            Some(&self.embed_cache),
        )
        .await
        .map_err(|e| ServiceError::Internal(format!("embedding failed: {}", e)))?;

        let mut probe_embedding = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| ServiceError::Internal("no embedding returned".to_owned()))?;
        l2_normalize(&mut probe_embedding);

        let search_query = SearchQuery {
            query_embedding: probe_embedding,
            query_text: text.to_owned(),
            mode: SearchMode::Vector,
            start,
            count,
            doc_type_filter: doc_type_filter.to_vec(),
            doc_id_filter: doc_id_filter.to_vec(),
            snippet,
        };

        let chunk_hits = self
            .store
            .io_search(&domain_str, &branch, &served_commit, &search_query)
            .await
            .map_err(|e| ServiceError::Internal(e.to_string()))?;

        let results = dedup_chunks_to_documents(chunk_hits, snippet);

        let paginated: Vec<SearchHit> = results
            .into_iter()
            .skip(start)
            .take(count)
            .collect();

        Ok(SimilarOutcome {
            hits: paginated,
            served_commit,
        })
    }

    /// Duplicates: near-duplicate document groups in the whole snapshot (bounded).
    ///
    /// Delegates to [`duplicates_with_options`] with the contract defaults
    /// (threshold 0.0, whole-snapshot scope, no snippets, page from 0, full page).
    pub async fn duplicates(
        &self,
        domain_raw: &str,
        commit: &str,
    ) -> Result<Vec<DuplicateGroup>, ServiceError> {
        self.duplicates_with_options(
            domain_raw,
            commit,
            0.0,
            &DuplicateScope::default(),
            false,
            0,
            usize::MAX,
            &[],
        )
        .await
    }

    /// Scoped near-duplicate document groups with full options.
    ///
    /// SCOPING (set/target — Spec 02/13): `scope.set_*` defines the population we
    /// look for near-duplicates within; `scope.target_*`, when present, defines a
    /// second population so every emitted group STRADDLES set↔target (cross-set
    /// entity resolution). Absent target → within-set dedup.
    ///
    /// ALGORITHM: for each SET point in the commit's snapshot, the store runs ONE
    /// FILTERED ANN `nearest()` query (excluding the point's own document, or
    /// restricted to the target) — every returned row is a genuine cross-document
    /// neighbour, so the scan is never starved by a multi-chunk document's own
    /// sibling chunks (the defect that returned `[]` at scale). O(n) cheap indexed
    /// queries, never an O(n²) all-pairs scan.
    ///
    /// UNITS: `threshold` is on the reference [0, 1] cosine scale used by
    /// `/search` (0 = identical, 0.5 = orthogonal, 1 = opposite); e.g. `<= 0.05`
    /// is "near-identical".
    ///
    /// BOUNDED: a candidate cap (`DEFAULT_DUPLICATE_MAX_POINTS`) rejects an
    /// oversized set population (fail-loud, never a silent partial run);
    /// `start`/`count` paginate the sorted (nearest-first) group list. Resolution
    /// uses the SAME catch-up path as `/search`/`/similar`, so commit-resolution
    /// and not-indexed (404) behaviour are consistent.
    #[allow(clippy::too_many_arguments)]
    pub async fn duplicates_with_options(
        &self,
        domain_raw: &str,
        commit: &str,
        threshold: f32,
        scope: &DuplicateScope,
        snippet: bool,
        start: usize,
        count: usize,
        ancestors: &[String],
    ) -> Result<Vec<DuplicateGroup>, ServiceError> {
        let rp =
            parse_domain(domain_raw).map_err(|e| ServiceError::Validation(e.to_string()))?;

        if !self.is_search_ready() {
            return Err(ServiceError::Unavailable(
                "search capability not ready (embedding backend cold)".to_owned(),
            ));
        }

        // Acquire heavy-scan permit (BUG-FD24): bounds concurrent /duplicates
        // requests so stacked FD spikes cannot exhaust nofile under load.
        let _permit = self
            .heavy_scan_semaphore
            .acquire()
            .await
            .map_err(|_| ServiceError::Internal("heavy-scan semaphore closed".to_owned()))?;

        let domain = Domain::from_resource_path(&rp);
        let domain_str = domain.as_str().to_owned();
        let branch = extract_branch(&rp);

        // Resolve the searchable commit via the SHARED catch-up path BEFORE any
        // store read — exact, nearest proven ancestor, or 404. Identical to
        // `/search`/`/similar` so not-indexed behaviour is consistent.
        let served_commit = self
            .resolve_searchable_commit(&domain, &branch, commit, ancestors)
            .await?;

        let mut groups = self
            .store
            .io_duplicate_groups(
                &domain_str,
                &branch,
                &served_commit,
                threshold,
                scope,
                snippet,
                crate::store::lance::DEFAULT_DUPLICATE_MAX_POINTS,
            )
            .await
            .map_err(|e| ServiceError::Internal(e.to_string()))?;

        // SAME-ROLE EXACT-DUPLICATE OVERRIDE (doc↔doc): for each candidate pair,
        // compare the two members' DOCUMENT-role embeddings. If bit-identical
        // (genuinely the same content) → override the pair's distance to 0.
        if !groups.is_empty() {
            // Collect all unique doc_ids mentioned across all groups.
            let all_doc_ids: Vec<String> = groups
                .iter()
                .flat_map(|g| g.group.iter().map(|m| m.id.clone()))
                .collect::<std::collections::HashSet<String>>()
                .into_iter()
                .collect();

            let doc_embeddings = self
                .store
                .io_fetch_result_embeddings(
                    &domain_str,
                    &branch,
                    &served_commit,
                    &all_doc_ids,
                    &[],
                    "embedding",
                )
                .await
                .map_err(|e| ServiceError::Internal(e.to_string()))?;

            for group in &mut groups {
                if group.group.len() == 2 {
                    let emb_a = doc_embeddings.get(&group.group[0].id);
                    let emb_b = doc_embeddings.get(&group.group[1].id);
                    if let (Some(a), Some(b)) = (emb_a, emb_b) {
                        if crate::kernel::distance::vectors_equal(a, b) {
                            group.distance = 0.0;
                        }
                    }
                }
            }

            // Re-sort after override: groups at distance 0 should sort first.
            groups.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.group[0].id.cmp(&b.group[0].id))
                    .then_with(|| a.group[1].id.cmp(&b.group[1].id))
            });
        }

        // Bounded pagination over the deterministic (sorted) group list.
        let paginated = groups.into_iter().skip(start).take(count).collect();
        Ok(paginated)
    }

    /// Raw bidirectional KNN gather for the /candidates endpoint.
    ///
    /// Returns directional candidate maps (set→target and target→set) with
    /// optional embeddings and content. No matching algorithm, no tau thresholds.
    #[allow(clippy::too_many_arguments)]
    pub async fn candidates_gather(
        &self,
        domain_raw: &str,
        commit: &str,
        scope: &DuplicateScope,
        k: usize,
        threshold_set: f32,
        threshold_target: f32,
        include_embeddings: bool,
        include_content: bool,
        ancestors: &[String],
    ) -> Result<CandidatesResult, ServiceError> {
        let rp =
            parse_domain(domain_raw).map_err(|e| ServiceError::Validation(e.to_string()))?;

        if !self.is_search_ready() {
            return Err(ServiceError::Unavailable(
                "search capability not ready (embedding backend cold)".to_owned(),
            ));
        }

        let _permit = self
            .heavy_scan_semaphore
            .acquire()
            .await
            .map_err(|_| ServiceError::Internal("heavy-scan semaphore closed".to_owned()))?;

        let domain = Domain::from_resource_path(&rp);
        let domain_str = domain.as_str().to_owned();
        let branch = extract_branch(&rp);

        let served_commit = self
            .resolve_searchable_commit(&domain, &branch, commit, ancestors)
            .await?;

        let start_time = std::time::Instant::now();

        let maps = self
            .store
            .io_resolve_cross_neighbours(
                &domain_str,
                &branch,
                &served_commit,
                scope,
                k,
                threshold_set,
                threshold_target,
                crate::store::lance::DEFAULT_DUPLICATE_MAX_POINTS,
            )
            .await
            .map_err(|e| ServiceError::Internal(e.to_string()))?;

        // Collect all doc_ids for optional embedding/content fetch.
        let all_doc_ids: Vec<String> = if include_embeddings || include_content {
            maps.set_to_target
                .values()
                .flat_map(|nbrs| nbrs.iter().map(|n| n.id.clone()))
                .chain(maps.set_to_target.keys().cloned())
                .chain(maps.target_to_set.values().flat_map(|nbrs| nbrs.iter().map(|n| n.id.clone())))
                .chain(maps.target_to_set.keys().cloned())
                .collect::<std::collections::HashSet<String>>()
                .into_iter()
                .collect()
        } else {
            Vec::new()
        };

        // Fetch embeddings if requested.
        let doc_embeddings = if include_embeddings {
            self.store
                .io_fetch_result_embeddings(
                    &domain_str,
                    &branch,
                    &served_commit,
                    &all_doc_ids,
                    &[],
                    "embedding",
                )
                .await
                .map_err(|e| ServiceError::Internal(e.to_string()))?
        } else {
            std::collections::HashMap::new()
        };

        // Fetch content if requested.
        let doc_contents = if include_content {
            self.store
                .io_fetch_doc_contents(
                    &domain_str,
                    &branch,
                    &served_commit,
                    &all_doc_ids,
                )
                .await
                .map_err(|e| ServiceError::Internal(e.to_string()))?
        } else {
            std::collections::HashMap::new()
        };

        // Fetch clustering embeddings if clustering is enabled for the domain.
        let store_clustering = self.store.domain_settings.store_clustering(&domain_str).await;
        let clustering_embeddings: std::collections::HashMap<String, Vec<f32>> = if store_clustering {
            // Fetch clustering embeddings for ALL docs involved (both sources and neighbours).
            let mut all_ids: std::collections::HashSet<String> = all_doc_ids.iter().cloned().collect();
            for id in maps.set_to_target.keys() {
                all_ids.insert(id.clone());
            }
            for id in maps.target_to_set.keys() {
                all_ids.insert(id.clone());
            }
            let all_ids_vec: Vec<String> = all_ids.into_iter().collect();
            self.store
                .io_fetch_result_embeddings(
                    &domain_str,
                    &branch,
                    &served_commit,
                    &all_ids_vec,
                    &[],
                    "clustering_embedding",
                )
                .await
                .map_err(|e| ServiceError::Internal(e.to_string()))?
        } else {
            std::collections::HashMap::new()
        };

        // Convert Neighbour maps to CandidateEntry maps.
        // When clustering is enabled, compute clustering_distance as the cosine
        // distance between the source doc's clustering_embedding and the neighbour's
        // clustering_embedding (both L2-normalised → distance = 1 - dot_product).
        let convert = |src_id: &str, nbrs: &Vec<crate::resolve::Neighbour>| -> Vec<CandidateEntry> {
            let src_clustering = clustering_embeddings.get(src_id);
            nbrs.iter()
                .map(|n| {
                    let clustering_distance = if store_clustering {
                        match (src_clustering, clustering_embeddings.get(&n.id)) {
                            (Some(src), Some(nbr)) => {
                                let dot: f32 = src.iter().zip(nbr.iter()).map(|(a, b)| a * b).sum();
                                Some((1.0 - dot).max(0.0))
                            }
                            _ => None,
                        }
                    } else {
                        None
                    };
                    CandidateEntry {
                        id: n.id.clone(),
                        distance: n.distance,
                        clustering_distance,
                        embedding: if include_embeddings {
                            doc_embeddings.get(&n.id).cloned()
                        } else {
                            None
                        },
                        content: if include_content {
                            doc_contents.get(&n.id).cloned()
                        } else {
                            None
                        },
                    }
                })
                .collect()
        };

        let set_to_target: std::collections::HashMap<String, Vec<CandidateEntry>> =
            maps.set_to_target
                .iter()
                .map(|(id, nbrs)| (id.clone(), convert(id, nbrs)))
                .collect();

        let target_to_set: std::collections::HashMap<String, Vec<CandidateEntry>> =
            maps.target_to_set
                .iter()
                .map(|(id, nbrs)| (id.clone(), convert(id, nbrs)))
                .collect();

        let set_to_target_edges: usize = set_to_target.values().map(|v| v.len()).sum();
        let target_to_set_edges: usize = target_to_set.values().map(|v| v.len()).sum();

        let elapsed_ms = start_time.elapsed().as_millis() as u64;

        Ok(CandidatesResult {
            set_to_target,
            target_to_set,
            stats: CandidatesStats {
                elapsed_ms,
                set_points: maps.set_to_target.len(),
                target_points: maps.target_to_set.len(),
                set_to_target_edges,
                target_to_set_edges,
                store_clustering,
            },
        })
    }

    /// Global statistics — aggregates across ALL domains (admin/internal only).
    pub async fn statistics(&self) -> Result<Statistics, ServiceError> {
        Ok(self.store.statistics().await)
    }

    /// Internal instrumentation stats — sizes of in-memory data structures and
    /// process RSS. Exposed via the global /statistics endpoint for leak monitoring.
    pub async fn internal_stats(&self) -> InternalStats {
        let mut stats = self.store.internal_stats().await;
        stats.embed_cache_entries = self.embed_cache.len();
        stats.embed_cache_size_bytes = self.embed_cache.len() * self.store.dim * std::mem::size_of::<f32>();
        stats
    }

    /// Domain-scoped statistics — aggregates ONLY the named domain's footprint.
    /// Validates the domain string via parse_domain (fail-loud on invalid input).
    pub async fn statistics_for_domain(&self, domain_raw: &str) -> Result<Statistics, ServiceError> {
        let rp =
            parse_domain(domain_raw).map_err(|e| ServiceError::Validation(e.to_string()))?;
        let domain = Domain::from_resource_path(&rp);
        let domain_str = domain.as_str().to_owned();

        Ok(self.store.statistics_for_domain(&domain_str).await)
    }

    /// Run a store integrity check on a domain. Read-only — compares on-disk
    /// state against live references from all tagged manifests. Exposed via
    /// `GET /integrity?domain=...` for operator troubleshooting.
    pub async fn integrity_check(
        &self,
        domain_raw: &str,
    ) -> Result<crate::store::lance::integrity::IntegrityReport, ServiceError> {
        let rp =
            parse_domain(domain_raw).map_err(|e| ServiceError::Validation(e.to_string()))?;
        let domain = Domain::from_resource_path(&rp);
        let domain_str = domain.as_str().to_owned();

        self.store
            .io_integrity_check(&domain_str)
            .await
            .map_err(|e| ServiceError::Internal(e.to_string()))
    }

    /// Fetch stored embeddings for a set of doc IDs and/or doc types.
    /// Returns both document and clustering embeddings (clustering only when enabled).
    /// When both doc_ids and doc_types are empty, all embeddings are returned.
    pub async fn fetch_embeddings(
        &self,
        domain_raw: &str,
        commit: &str,
        doc_ids: &[String],
        doc_types: &[String],
        ancestors: &[String],
    ) -> Result<EmbeddingsResult, ServiceError> {
        let rp =
            parse_domain(domain_raw).map_err(|e| ServiceError::Validation(e.to_string()))?;

        if !self.is_search_ready() {
            return Err(ServiceError::Unavailable(
                "search capability not ready (embedding backend cold)".to_owned(),
            ));
        }

        let domain = Domain::from_resource_path(&rp);
        let domain_str = domain.as_str().to_owned();
        let branch = extract_branch(&rp);

        let served_commit = self
            .resolve_searchable_commit(&domain, &branch, commit, ancestors)
            .await?;

        let doc_embeddings = self
            .store
            .io_fetch_result_embeddings(&domain_str, &branch, &served_commit, doc_ids, doc_types, "embedding")
            .await
            .map_err(|e| ServiceError::Internal(e.to_string()))?;

        let store_clustering = self.store.domain_settings.store_clustering(&domain_str).await;
        let clustering_embeddings = if store_clustering {
            self.store
                .io_fetch_result_embeddings(&domain_str, &branch, &served_commit, doc_ids, doc_types, "clustering_embedding")
                .await
                .map_err(|e| ServiceError::Internal(e.to_string()))?
        } else {
            std::collections::HashMap::new()
        };

        Ok(EmbeddingsResult {
            doc_embeddings,
            clustering_embeddings,
            store_clustering,
            served_commit,
        })
    }

    /// Streaming variant of `fetch_embeddings`. Returns metadata (served_commit,
    /// store_clustering, total_count) plus a tokio mpsc receiver that yields
    /// `EmbeddingRecord` values as they are read from Lance.
    pub async fn fetch_embeddings_stream(
        &self,
        domain_raw: &str,
        commit: &str,
        doc_ids: &[String],
        doc_types: &[String],
        ancestors: &[String],
    ) -> Result<EmbeddingsStreamResult, ServiceError> {
        let rp =
            parse_domain(domain_raw).map_err(|e| ServiceError::Validation(e.to_string()))?;

        if !self.is_search_ready() {
            return Err(ServiceError::Unavailable(
                "search capability not ready (embedding backend cold)".to_owned(),
            ));
        }

        let domain = Domain::from_resource_path(&rp);
        let domain_str = domain.as_str().to_owned();
        let branch = extract_branch(&rp);

        let served_commit = self
            .resolve_searchable_commit(&domain, &branch, commit, ancestors)
            .await?;

        let store_clustering = self.store.domain_settings.store_clustering(&domain_str).await;

        let (total_count, rx) = self
            .store
            .io_stream_embeddings(&domain_str, &branch, &served_commit, doc_ids, doc_types, store_clustering)
            .await
            .map_err(|e| ServiceError::Internal(e.to_string()))?;

        Ok(EmbeddingsStreamResult {
            served_commit,
            store_clustering,
            total_count,
            receiver: rx,
        })
    }

    /// Delete a domain's ENTIRE footprint: the `{domain}.lance` dataset (all
    /// branches/versions/tags) plus the store's in-memory caches. IDEMPOTENT —
    /// an unknown/already-gone domain succeeds (TerminusDB may retry). Fail-loud
    /// on a real I/O error.
    ///
    /// The store removes the on-disk dataset FIRST, then purges its own in-memory
    /// maps (fail-loud partial guard inside the store). There is no separate
    /// service-owned per-branch enablement/negative-cache to purge any more —
    /// index state is derived from the (now-deleted) on-disk tags
    /// (task-durable-index-state).
    pub async fn delete_domain(&self, domain_raw: &str) -> Result<(), ServiceError> {
        let rp =
            parse_domain(domain_raw).map_err(|e| ServiceError::Validation(e.to_string()))?;
        let domain = Domain::from_resource_path(&rp);
        let domain_str = domain.as_str().to_owned();

        self.store
            .io_delete_domain(&domain_str)
            .await
            .map_err(|e| ServiceError::Internal(e.to_string()))?;

        Ok(())
    }

    /// Delete a single branch's index (tags + in-memory caches) without
    /// affecting other branches in the same domain dataset. Used by
    /// TerminusDB when re-indexing a branch from scratch.
    pub async fn delete_branch_index(
        &self,
        domain_raw: &str,
        branch: &str,
    ) -> Result<(), ServiceError> {
        let rp =
            parse_domain(domain_raw).map_err(|e| ServiceError::Validation(e.to_string()))?;
        let domain = Domain::from_resource_path(&rp);
        let domain_str = domain.as_str().to_owned();

        self.store
            .io_delete_branch_index(&domain_str, branch)
            .await
            .map_err(|e| ServiceError::Internal(e.to_string()))?;

        Ok(())
    }

    /// Compare two texts by embedding them and computing their normalized cosine distance.
    /// Stateless: NO dataset, NO ANN index, NO domain required.
    ///
    /// `source` is embedded with `EmbeddingRole::Document`, `target` with `EmbeddingRole::Document`.
    /// Both vectors are L2-normalized, then the normalized cosine distance is computed on
    /// the same [0, 1] reference scale that `/search`, `/similar`, `/duplicates`, and
    /// `/candidates` return (0 = identical, ~0.5 = unrelated, 1 = opposite).
    pub async fn compare(
        &self,
        source: &str,
        target: &str,
    ) -> Result<CompareResult, ServiceError> {
        self.compare_with_roles(source, target, EmbeddingRole::Document, EmbeddingRole::Document)
            .await
    }

    /// Compare two texts with explicit embedding roles for each side.
    /// Supports all four nomic task prefixes: search_query, search_document,
    /// clustering, classification.
    pub async fn compare_with_roles(
        &self,
        source: &str,
        target: &str,
        source_role: EmbeddingRole,
        target_role: EmbeddingRole,
    ) -> Result<CompareResult, ServiceError> {
        // If source and target are identical strings, return distance 0
        // without embedding. Even with different role prefixes (e.g. query vs
        // document), the user-visible contract is that comparing a text with
        // itself yields zero distance.
        if source == target {
            return Ok(CompareResult {
                distance: 0.0,
                source_role: role_name(source_role).to_owned(),
                target_role: role_name(target_role).to_owned(),
            });
        }

        if !self.is_search_ready() {
            return Err(ServiceError::Unavailable(
                "search capability not ready (embedding backend cold)".to_owned(),
            ));
        }

        // Embed both texts, each with its specified role.
        let source_texts = vec![source.to_owned()];
        let target_texts = vec![target.to_owned()];

        let source_embeddings = embed::io_embed(
            &self.config.embed_provider,
            &source_texts,
            source_role,
            &self.http_client,
            Some(&self.embed_cache),
        )
        .await
        .map_err(|e| ServiceError::Internal(format!("embedding source failed: {}", e)))?;

        let target_embeddings = embed::io_embed(
            &self.config.embed_provider,
            &target_texts,
            target_role,
            &self.http_client,
            Some(&self.embed_cache),
        )
        .await
        .map_err(|e| ServiceError::Internal(format!("embedding target failed: {}", e)))?;

        let mut source_vec = source_embeddings
            .into_iter()
            .next()
            .ok_or_else(|| ServiceError::Internal("no embedding returned for source".to_owned()))?;
        let mut target_vec = target_embeddings
            .into_iter()
            .next()
            .ok_or_else(|| ServiceError::Internal("no embedding returned for target".to_owned()))?;

        // L2-normalize both vectors.
        l2_normalize(&mut source_vec);
        l2_normalize(&mut target_vec);

        // Compute the normalized cosine distance on the [0, 1] reference scale.
        let distance =
            crate::kernel::distance::cosine_distance_normalized(&source_vec, &target_vec);

        Ok(CompareResult {
            distance,
            source_role: role_name(source_role).to_owned(),
            target_role: role_name(target_role).to_owned(),
        })
    }

    /// Validate a domain string without side effects.
    pub fn validate_domain(&self, domain_raw: &str) -> Result<(), ServiceError> {
        parse_domain(domain_raw).map_err(|e| ServiceError::Validation(e.to_string()))?;
        Ok(())
    }

    /// Record an error task (used when NDJSON parse fails before processing).
    pub async fn record_error_task(&self, task_id: &str, error: String) {
        self.store
            .record_task(task_id, TaskStatus::Error { error })
            .await;
    }
}

/// Convert an EmbeddingRole to its string name for API responses.
fn role_name(role: EmbeddingRole) -> &'static str {
    match role {
        EmbeddingRole::Document => "document",
        EmbeddingRole::Query => "query",
        EmbeddingRole::Clustering => "clustering",
        EmbeddingRole::Classification => "classification",
    }
}

/// Extract the branch name from a ResourcePath (defaults to "main").
fn extract_branch(rp: &crate::kernel::model::ResourcePath) -> String {
    match &rp.reference {
        Ref::Branch(b) => b.as_str().to_owned(),
        Ref::Commit(_) => "main".to_owned(),
    }
}

/// Bundled context for the indexing pipeline (avoids too-many-arguments).
struct PipelineCtx<'a> {
    store: &'a LanceStore,
    tokenizer: &'a Tokenizer,
    chunk_params: &'a ChunkParams,
    provider: &'a Provider,
    http_client: &'a reqwest::Client,
    embed_cache: &'a crate::embed::cache::EmbedCache,
    domain: &'a str,
    branch: &'a str,
    /// Cross-document embedding batch size (from Config, env-configurable).
    embed_batch_size: usize,
}

/// Run the push pipeline for a set of operations (Phase 6A: one-push-one-fragment +
/// cross-document embedding batching).
/// Returns (indexed_count, skipped_docs) on success, or error message on failure.
///
/// Architecture:
///   COLLECT phase (cross-document batched embedding):
///     A. Chunk all Insert/Changed docs (pure, no IO). Record errors per doc.
///     B. Flatten all chunk texts into ONE ordered list with a mapping back to
///        (doc_index, chunk_index). Accumulate Deleted doc_ids separately.
///     C. Batch-embed: split the flat text list into batches of embed_batch_size,
///        call io_embed per batch via io_embed_batched. Per-doc failure isolation:
///        if a batch fails, retry individually; only the toxic text(s) are marked
///        Failed — the rest succeed.
///     D. Scatter embeddings back: for each doc, check all its chunk embeddings
///        succeeded. If any failed, skip that doc entirely. Otherwise L2-normalise
///        and build ChunkRows.
///   WRITE phase (batched, under pipeline lock):
///     1. Acquire per-(domain, branch) pipeline lock
///     2. io_batch_delete_append: two-version delete-then-append for all
///        Insert/Changed rows + Deleted doc_ids → 1–2 fragments per commit
///     3. Optimize indices (FTS + vector ANN) on uncached handle — O(delta)
///     4. Tag commit → indexed version (commit becomes searchable)
///     5. Update last-indexed tracking
///     6. Release pipeline lock
///
/// Crash safety (two-version delete-then-append):
///   - Delete first, append second. Crash between them leaves only the
///     pure-Deleted docs removed (correct) and the commit untagged (re-pushable).
///   - Tag/last-indexed advance ONLY after both writes land.
///   - NOT atomic single-version; crash-safety comes from the
///     untagged-commit→invisible→re-pushable property.
///
/// Serialised per-(domain, branch): at most ONE unindexed commit per branch at
/// any time. The pipeline lock prevents a second push from starting until the
/// current commit is fully indexed and tagged.
///
/// Latency is decoupled from the HTTP handler: this runs in a tokio::spawn'd
/// background task. The HTTP /push endpoint returns the task_id immediately.
async fn io_run_index_pipeline(
    ctx: &PipelineCtx<'_>,
    commit: &str,
    operations: Vec<Operation>,
) -> Result<(u64, Vec<SkippedDoc>), String> {
    // ─── COLLECT phase A: chunk all docs (pure, no IO) ──────────────────────
    // Separate operations into:
    //   - docs_to_embed: Vec<DocToEmbed> for Insert/Changed (chunked, ready for embed)
    //   - delete_ids: doc_ids for Changed + Deleted (old chunks to remove)
    //   - skipped: docs that failed chunking or Operation::Error
    //
    // Changed semantics: the old chunks for a Changed doc MUST be deleted before
    // inserting the replacement rows. A doc that shrinks from 5 chunks to 1 chunk
    // must not leave orphan chunks (chunk_index 1-4) behind. The delete_ids set
    // covers both Changed (replacement) and Deleted (pure removal) doc_ids.

    struct DocToEmbed {
        id: String,
        chunks: Vec<chunk::Chunk>,
        is_changed: bool,
    }

    let mut docs_to_embed: Vec<DocToEmbed> = Vec::new();
    let mut delete_ids: Vec<String> = Vec::new();
    let mut skipped: Vec<SkippedDoc> = Vec::new();

    for op in &operations {
        match op {
            Operation::Inserted { id, string } => {
                match chunk::chunk_text(ctx.tokenizer, string, ctx.chunk_params) {
                    Ok(chunks) if chunks.is_empty() => {
                        skipped.push(SkippedDoc {
                            id: id.clone(),
                            message: "chunking produced zero chunks".to_owned(),
                        });
                    }
                    Ok(chunks) => {
                        docs_to_embed.push(DocToEmbed {
                            id: id.clone(),
                            chunks,
                            is_changed: false,
                        });
                    }
                    Err(e) => {
                        skipped.push(SkippedDoc {
                            id: id.clone(),
                            message: format!("chunking failed: {}", e),
                        });
                    }
                }
            }
            Operation::Changed { id, string } => {
                match chunk::chunk_text(ctx.tokenizer, string, ctx.chunk_params) {
                    Ok(chunks) if chunks.is_empty() => {
                        skipped.push(SkippedDoc {
                            id: id.clone(),
                            message: "chunking produced zero chunks".to_owned(),
                        });
                    }
                    Ok(chunks) => {
                        docs_to_embed.push(DocToEmbed {
                            id: id.clone(),
                            chunks,
                            is_changed: true,
                        });
                    }
                    Err(e) => {
                        skipped.push(SkippedDoc {
                            id: id.clone(),
                            message: format!("chunking failed: {}", e),
                        });
                    }
                }
            }
            Operation::Deleted { id } => {
                delete_ids.push(id.clone());
            }
            Operation::Error { message } => {
                // Per spec: Operation::Error → skip+record, do not fail the whole task.
                skipped.push(SkippedDoc {
                    id: "unknown".to_owned(),
                    message: format!("operation error: {}", message),
                });
            }
            Operation::Abort => {
                // Abort: cancel in-process work immediately.
                return Err("push aborted by client".to_string());
            }
        }
    }

    // ─── COLLECT phase B: flatten chunk texts into one ordered list ──────────
    // Build a flat text list and a parallel mapping of (doc_index, chunk_index)
    // so we can scatter embeddings back after the batched call.
    let mut flat_texts: Vec<String> = Vec::new();
    // Range per doc: (start_in_flat, count_in_flat) — lets us slice the results.
    let mut doc_flat_ranges: Vec<(usize, usize)> = Vec::with_capacity(docs_to_embed.len());

    for doc in &docs_to_embed {
        let start = flat_texts.len();
        for chunk in &doc.chunks {
            flat_texts.push(chunk.text.clone());
        }
        doc_flat_ranges.push((start, doc.chunks.len()));
    }

    // ─── COLLECT phase C: batch-embed all texts (DUAL — Document + Clustering) ───
    // Embed each chunk TWICE — once with Document role (for the ANN-indexed
    // `embedding` column) and once with Clustering role (for the stored
    // `clustering_embedding` column used by /candidates dual KNN gather).
    // When clustering is disabled for the domain, the second embedding call is
    // skipped and zeros are stored instead (saves embed time + bandwidth).
    let store_clustering = ctx.store.domain_settings.store_clustering(ctx.domain).await;

    let embed_results_doc = embed::io_embed_batched(
        ctx.provider,
        &flat_texts,
        ctx.embed_batch_size,
        EmbeddingRole::Document,
        ctx.http_client,
        Some(ctx.embed_cache),
    )
    .await;

    let embed_results_query = if store_clustering {
        embed::io_embed_batched(
            ctx.provider,
            &flat_texts,
            ctx.embed_batch_size,
            EmbeddingRole::Clustering,
            ctx.http_client,
            Some(ctx.embed_cache),
        )
        .await
    } else {
        // Clustering disabled — fill with zeros (same dimension, no embed call).
        flat_texts
            .iter()
            .map(|_| EmbedResult::Ok(vec![0.0; ctx.store.dim]))
            .collect::<Vec<_>>()
    };

    // ─── COLLECT phase D: scatter BOTH embedding sets → build ChunkRows ─────
    // For each doc, check that ALL its chunk embeddings (BOTH roles) succeeded.
    // If any failed, skip the entire doc (per-doc isolation). Otherwise
    // L2-normalise both sets and build rows with both vectors.
    let mut all_rows: Vec<ChunkRow> = Vec::new();
    let mut indexed_count: u64 = 0;

    for (doc_idx, doc) in docs_to_embed.iter().enumerate() {
        let (start, count) = doc_flat_ranges[doc_idx];
        let doc_results_doc = &embed_results_doc[start..start + count];
        let doc_results_query = &embed_results_query[start..start + count];

        // Check for any failed embeddings in EITHER role for this doc's chunks.
        let first_failure = doc_results_doc
            .iter()
            .chain(doc_results_query.iter())
            .find_map(|r| match r {
                EmbedResult::Failed(msg) => Some(msg.clone()),
                EmbedResult::Ok(_) => None,
            });

        if let Some(failure_msg) = first_failure {
            skipped.push(SkippedDoc {
                id: doc.id.clone(),
                message: failure_msg,
            });
            continue;
        }

        // All embeddings succeeded — extract both sets, normalise, and build rows.
        let mut embeddings_doc: Vec<Vec<f32>> = Vec::with_capacity(doc_results_doc.len());
        let mut embeddings_query: Vec<Vec<f32>> = Vec::with_capacity(doc_results_query.len());
        let mut extraction_failed = false;

        for r in doc_results_doc {
            match r {
                EmbedResult::Ok(emb) => embeddings_doc.push(emb.clone()),
                EmbedResult::Failed(msg) => {
                    skipped.push(SkippedDoc {
                        id: doc.id.clone(),
                        message: format!(
                            "internal: EmbedResult::Failed after first_failure check (doc): {}",
                            msg
                        ),
                    });
                    extraction_failed = true;
                    break;
                }
            }
        }
        if extraction_failed {
            continue;
        }

        for r in doc_results_query {
            match r {
                EmbedResult::Ok(emb) => embeddings_query.push(emb.clone()),
                EmbedResult::Failed(msg) => {
                    skipped.push(SkippedDoc {
                        id: doc.id.clone(),
                        message: format!(
                            "internal: EmbedResult::Failed after first_failure check (query): {}",
                            msg
                        ),
                    });
                    extraction_failed = true;
                    break;
                }
            }
        }
        if extraction_failed {
            continue;
        }

        // L2-normalise BOTH embedding sets so cosine metric works correctly.
        for emb in &mut embeddings_doc {
            l2_normalize(emb);
        }
        for emb in &mut embeddings_query {
            l2_normalize(emb);
        }

        // Build chunk rows with both vectors.
        let doc_type = ingest::extract_doc_type(&doc.id);
        let rows: Vec<ChunkRow> = doc
            .chunks
            .iter()
            .zip(embeddings_doc.into_iter().zip(embeddings_query))
            .map(|(chunk, (embedding, clustering_embedding))| ChunkRow {
                doc_id: doc.id.clone(),
                doc_type: doc_type.clone(),
                chunk_index: chunk.index as i32,
                chunk_count: chunk.count as i32,
                chunk_token_start: chunk.token_start as i32,
                doc_token_len: chunk.doc_token_len as i32,
                embedding,
                clustering_embedding,
                content: chunk.text.clone(),
            })
            .collect();

        // If Changed, register for deletion of old chunks.
        if doc.is_changed {
            delete_ids.push(doc.id.clone());
        }

        all_rows.extend(rows);
        indexed_count += 1;
    }

    // ─── WRITE phase (batched, under pipeline lock) ─────────────────────────
    // Only proceed with data writes if there is actual work (rows to insert/update
    // or docs to delete). HOWEVER: every commit — including no-op/all-error commits
    // — MUST be tagged and last_indexed advanced. Skipping the tag for a no-op
    // commit causes catch-up stalls (RISK-26): the engine never progresses past
    // the empty commit and subsequent normal commits cannot be reached.
    let has_work = !all_rows.is_empty() || !delete_ids.is_empty();

    // The pipeline lock was acquired in push before spawning and is held
    // by _pipeline_guard in the spawned task. No need to acquire here.

    if has_work {
        // Batched write (Option B): two-version delete-then-append under pipeline lock.
        //   1. Delete: all Changed + Deleted doc_ids (removes old chunks, handles shrinking).
        //   2. Append: all new rows for Insert + Changed docs (one fragment).
        // Crash safety: crash after (1) but before (2) leaves commit untagged → re-pushable.
        // Produces 1–2 fragments total (vs N per-doc fragments before Phase 6A).
        let last_version = ctx
            .store
            .io_batch_delete_append(ctx.domain, ctx.branch, &delete_ids, &all_rows)
            .await
            .map_err(|e| format!("batch delete-append failed: {}", e))?;

        // Boundary-aware indexing: create indices only at every 3rd commit
        // (positions 2, 5, 8, ... 0-indexed). Non-indexed commits rely on
        // KNN fallback for search (at most 2 commits delta).
        let commit_position = ctx.store.io_count_branch_commits(ctx.domain, ctx.branch).await.unwrap_or(0);
        let action = should_create_index(commit_position);

        let mut indexed_version: Option<u64> = None;

        if action == IndexAction::Create {
            let vector_config = ctx.store.vector_index_config().clone();

            let ds_opt = ctx.store.io_open_dataset_uncached(ctx.domain, ctx.branch).await;
            if let Ok(Some(mut ds_opt)) = ds_opt {
                crate::store::lance::io_ensure_fts_index_on_dataset(&mut ds_opt)
                    .await
                    .map_err(|e| format!("FTS index creation failed: {}", e))?;

                crate::store::vector_index::io_ensure_vector_index(&mut ds_opt, &vector_config, false)
                    .await
                    .map_err(|e| format!("vector index creation failed: {}", e))?;

                if store_clustering {
                    crate::store::vector_index::io_ensure_clustering_vector_index(&mut ds_opt, &vector_config, false)
                        .await
                        .map_err(|e| format!("clustering vector index creation failed: {}", e))?;
                }

                indexed_version = Some(ds_opt.version().version);

                ctx.store.io_increment_delta_count(ctx.domain, ctx.branch).await;
                let delta_count = ctx.store.io_get_delta_count(ctx.domain, ctx.branch).await;

                let (idx_before, idx_after) =
                    crate::store::lance::io_incremental_cascade(&mut ds_opt, delta_count)
                        .await
                        .unwrap_or((0, 0));

                if idx_after != idx_before {
                    eprintln!(
                        "[pipeline] cascade: domain={} branch={} delta_count={} indices {}→{}",
                        ctx.domain, ctx.branch, delta_count, idx_before, idx_after
                    );
                }

                ctx.store.io_refresh_cached_dataset(ctx.domain, ctx.branch).await
                    .map_err(|e| format!("cascade cache refresh failed: {}", e))?;

                eprintln!(
                    "[pipeline] boundary index created: domain={} branch={} commit_position={}",
                    ctx.domain, ctx.branch, commit_position
                );
            }
        } else {
            // Non-boundary commit: FTS has no flat-scan fallback (unlike vector
            // search), so the inverted index must exist and cover new fragments.
            // Open a branch-bound handle — NOT io_ensure_fts_index which opens
            // the cached MAIN handle and would tag the wrong version for non-main
            // branches.
            let ds_opt = ctx.store.io_open_dataset_uncached(ctx.domain, ctx.branch).await
                .map_err(|e| format!("FTS index: failed to open dataset: {}", e))?;
            if let Some(mut ds_opt) = ds_opt {
                let fts_version = crate::store::lance::io_ensure_fts_index_on_dataset(&mut ds_opt)
                    .await
                    .map_err(|e| format!("FTS index ensure failed: {}", e))?;
                indexed_version = Some(fts_version);
            }
        }

        // Tag the commit to the INDEXED version (data + indices), or fall back
        // to the data-only version if optimization was skipped/failed.
        // After this point, the commit is resolvable and searchable.
        let tag_version = indexed_version.unwrap_or(last_version);
        ctx.store
            .io_tag_commit(ctx.domain, ctx.branch, commit, tag_version)
            .await
            .map_err(|e| format!("failed to tag commit: {}", e))?;

        // Update last-indexed tracking.
        ctx.store
            .update_last_indexed(ctx.domain, ctx.branch, commit, tag_version)
            .await;

        // Aggressively prune untagged intermediate versions after tagging.
        if let Err(e) = ctx.store.io_cleanup_aggressive(ctx.domain, ctx.branch).await {
            eprintln!("[pipeline] aggressive cleanup failed (soft): {}", e);
        }
    } else {
        // ─── NO-OP path (RISK-26 fix): tag + advance without writing data ──────
        // The commit contains no indexable content (all Operation::Error, empty ops,
        // or all docs skipped). We MUST still tag it and advance last_indexed so
        // catch-up progresses past this commit. The tag points to the current/prior
        // latest data version — there is nothing new to search, but the commit is
        // marked as "processed" and will not block subsequent commits.
        //
        // Ensure the dataset exists (auto-creates an empty one for a fresh domain).
        // This is needed before we can read the branch head version or tag.
        ctx.store
            .io_open_dataset(ctx.domain, ctx.branch)
            .await
            .map_err(|e| format!("no-op tag: failed to ensure dataset: {}", e))?;

        // Get the current head version for the BRANCH (not main). For non-main
        // branches, the version space is branch-scoped — using main's version would
        // fail because that version may not exist in the branch's lineage.
        let current_version = ctx
            .store
            .io_branch_head_version(ctx.domain, ctx.branch)
            .await
            .map_err(|e| format!("no-op tag: failed to get branch head version: {}", e))?;

        // Tag the empty commit to the existing version (no new data).
        ctx.store
            .io_tag_commit(ctx.domain, ctx.branch, commit, current_version)
            .await
            .map_err(|e| format!("no-op tag: failed to tag commit: {}", e))?;

        // Advance last-indexed past this commit.
        ctx.store
            .update_last_indexed(ctx.domain, ctx.branch, commit, current_version)
            .await;
    }

    Ok((indexed_count, skipped))
}

/// Streaming variant of `io_run_index_pipeline`.
///
/// Reads operations incrementally from an mpsc channel, chunking and embedding
/// them in micro-batches as they arrive. Each micro-batch is written to Lance
/// immediately via `ds.append()`, bounding memory to the micro-batch size rather
/// than the entire commit.
///
/// Architecture:
///   1. Acquire pipeline lock + domain guard + open write handle
///   2. Micro-batch loop: read N docs → chunk → embed → delete old (Changed) → append → drop
///   3. After all ops: optimize indices + tag commit
///
// ─── Boundary-aware indexing logic ──────────────────────────────────
/// Indices are created at every 3rd commit (positions 2, 5, 8, ... 0-indexed).
/// Non-indexed commits rely on KNN fallback for search (at most 2 commits delta).

#[derive(Debug, PartialEq)]
enum IndexAction {
    Skip,
    Create,
}

fn should_create_index(commit_position: usize) -> IndexAction {
    if crate::store::lance::should_index_commit(commit_position) {
        IndexAction::Create
    } else {
        IndexAction::Skip
    }
}

/// The pipeline lock is held for the entire loop (still one commit at a time).
/// Multiple appends produce multiple fragments — compaction merges them later.
/// The commit is untagged until the end, so no search sees partial data.
async fn io_run_index_pipeline_stream(
    ctx: &PipelineCtx<'_>,
    commit: &str,
    mut operation_rx: mpsc::Receiver<Operation>,
    progress_tx: Option<mpsc::Sender<ProgressUpdate>>,
    cancel_token: &tokio_util::sync::CancellationToken,
) -> Result<(u64, Vec<SkippedDoc>), String> {
    struct DocToEmbed {
        id: String,
        chunks: Vec<chunk::Chunk>,
        is_changed: bool,
    }

    // Reset pipeline progress counters.
    ctx.store.pipeline_pending_chunks.store(0, std::sync::atomic::Ordering::Relaxed);
    ctx.store.pipeline_embedded_chunks.store(0, std::sync::atomic::Ordering::Relaxed);
    ctx.store.pipeline_written_chunks.store(0, std::sync::atomic::Ordering::Relaxed);

    let mut all_delete_ids: Vec<String> = Vec::new();
    let mut skipped: Vec<SkippedDoc> = Vec::new();
    let mut indexed_count: u64 = 0;
    let mut total_docs_seen: u64 = 0;

    // Micro-batch buffer: collect docs until we have enough to embed+write.
    const MICRO_BATCH_DOCS: usize = 2000;
    let mut batch_docs: Vec<DocToEmbed> = Vec::with_capacity(MICRO_BATCH_DOCS);

    // The pipeline lock was acquired in push_stream before spawning and is
    // held by _pipeline_guard in the spawned task. No need to acquire here.

    // Acquire domain guard to serialise against DELETE /domain.
    let _domain_guard = ctx.store.acquire_domain_guard(ctx.domain).await;

    // Ensure the dataset exists (creates the main branch on first use).
    ctx.store.io_open_dataset(ctx.domain, ctx.branch).await
        .map_err(|e| format!("failed to ensure dataset: {}", e))?;

    // Open a fresh branch-bound write handle. This handle is reused for all
    // micro-batch appends + deletes, so all fragments land on the same branch
    // lineage.
    let mut ds = ctx.store.io_open_branch_for_write(ctx.domain, ctx.branch).await
        .map_err(|e| format!("failed to open write handle: {}", e))?;

    let store_clustering = ctx.store.domain_settings.store_clustering(ctx.domain).await;

    // Boundary-aware indexing: determine commit position (0-indexed) for this
    // push. Indices are created at every 3rd commit (positions 2, 5, 8, ...).
    let commit_position = ctx.store.io_count_branch_commits(ctx.domain, ctx.branch).await.unwrap_or(0);

    // Push-level row accumulator: all micro-batch rows are collected here and
    // appended in a single ds.append() call at the end, producing exactly 1
    // new data fragment per push (Option C architecture).
    let mut all_rows: Vec<ChunkRow> = Vec::new();

    // ─── Micro-batch flush helper ───────────────────────────────────────
    // Embeds, deletes old chunks (for Changed docs), appends new rows,
    // then drops all Vecs to free memory. Returns (indexed, skipped) for this batch.
    #[allow(clippy::too_many_arguments)]
    async fn flush_microbatch(
        ctx: &PipelineCtx<'_>,
        ds: &mut lance::dataset::Dataset,
        batch_docs: &mut Vec<DocToEmbed>,
        all_delete_ids: &mut Vec<String>,
        all_rows: &mut Vec<ChunkRow>,
        skipped: &mut Vec<SkippedDoc>,
        store_clustering: bool,
        progress_tx: &Option<mpsc::Sender<ProgressUpdate>>,
        total_docs_seen: &mut u64,
        indexed_count: &mut u64,
        cancel_token: &tokio_util::sync::CancellationToken,
    ) -> Result<(), String> {
        if batch_docs.is_empty() {
            return Ok(());
        }

        let batch_chunk_count: u64 = batch_docs.iter().map(|d| d.chunks.len() as u64).sum();
        ctx.store.pipeline_pending_chunks.fetch_add(batch_chunk_count, std::sync::atomic::Ordering::Relaxed);

        // Process docs in sub-batches of embed_batch_size so progress is
        // reported frequently (per ~32 docs) rather than per 2000-doc batch.
        // Each sub-batch: flatten chunks → embed doc+clustering → scatter →
        // report progress → accumulate rows for the single Lance append.
        let mut batch_rows: Vec<ChunkRow> = Vec::new();

        for sub_batch in batch_docs.chunks(ctx.embed_batch_size.max(1)) {
            // Check cancellation before starting this sub-batch's embed.
            // This ensures we stop immediately when a reindex triggers a new
            // push, rather than continuing to embed and accumulate memory.
            if cancel_token.is_cancelled() {
                return Err("pipeline superseded by new push (reindex)".to_string());
            }
            // Flatten chunk texts for this sub-batch.
            let mut flat_texts: Vec<String> = Vec::new();
            let mut doc_flat_ranges: Vec<(usize, usize)> = Vec::with_capacity(sub_batch.len());

            for doc in sub_batch {
                let start = flat_texts.len();
                for chunk in &doc.chunks {
                    flat_texts.push(chunk.text.clone());
                }
                doc_flat_ranges.push((start, doc.chunks.len()));
            }

            let sub_chunk_count = flat_texts.len() as u64;

            // Embed document role (cancellation-aware).
            let embed_results_doc = tokio::select! {
                r = embed::io_embed_batched(
                    ctx.provider,
                    &flat_texts,
                    ctx.embed_batch_size,
                    EmbeddingRole::Document,
                    ctx.http_client,
                    Some(ctx.embed_cache),
                ) => r,
                _ = cancel_token.cancelled() => {
                    return Err("pipeline superseded by new push (reindex)".to_string());
                }
            };

            // Embed clustering role (or zeros if disabled, cancellation-aware).
            let embed_results_query = if store_clustering {
                tokio::select! {
                    r = embed::io_embed_batched(
                        ctx.provider,
                        &flat_texts,
                        ctx.embed_batch_size,
                        EmbeddingRole::Clustering,
                        ctx.http_client,
                        Some(ctx.embed_cache),
                    ) => r,
                    _ = cancel_token.cancelled() => {
                        return Err("pipeline superseded by new push (reindex)".to_string());
                    }
                }
            } else {
                flat_texts.iter().map(|_| EmbedResult::Ok(vec![0.0; ctx.store.dim])).collect::<Vec<_>>()
            };

            ctx.store.pipeline_embedded_chunks.fetch_add(sub_chunk_count, std::sync::atomic::Ordering::Relaxed);

            // Scatter embeddings → build ChunkRows for this sub-batch.
            for (doc_idx, doc) in sub_batch.iter().enumerate() {
                let (start, count) = doc_flat_ranges[doc_idx];
                let doc_results_doc = &embed_results_doc[start..start + count];
                let doc_results_query = &embed_results_query[start..start + count];

                let first_failure = doc_results_doc.iter().chain(doc_results_query.iter()).find_map(|r| match r {
                    EmbedResult::Failed(msg) => Some(msg.clone()),
                    EmbedResult::Ok(_) => None,
                });

                if let Some(failure_msg) = first_failure {
                    skipped.push(SkippedDoc { id: doc.id.clone(), message: failure_msg });
                    continue;
                }

                let mut embeddings_doc: Vec<Vec<f32>> = Vec::with_capacity(doc_results_doc.len());
                let mut embeddings_query: Vec<Vec<f32>> = Vec::with_capacity(doc_results_query.len());

                for r in doc_results_doc {
                    match r {
                        EmbedResult::Ok(emb) => embeddings_doc.push(emb.clone()),
                        EmbedResult::Failed(msg) => {
                            skipped.push(SkippedDoc {
                                id: doc.id.clone(),
                                message: format!("internal: EmbedResult::Failed after check (doc): {}", msg),
                            });
                            embeddings_doc.clear();
                            break;
                        }
                    }
                }
                if embeddings_doc.is_empty() && !doc_results_doc.is_empty() {
                    continue;
                }

                for r in doc_results_query {
                    match r {
                        EmbedResult::Ok(emb) => embeddings_query.push(emb.clone()),
                        EmbedResult::Failed(msg) => {
                            skipped.push(SkippedDoc {
                                id: doc.id.clone(),
                                message: format!("internal: EmbedResult::Failed after check (query): {}", msg),
                            });
                            embeddings_query.clear();
                            break;
                        }
                    }
                }
                if embeddings_query.is_empty() && !doc_results_query.is_empty() {
                    continue;
                }

                for emb in &mut embeddings_doc { l2_normalize(emb); }
                for emb in &mut embeddings_query { l2_normalize(emb); }

                let doc_type = ingest::extract_doc_type(&doc.id);

                // For Changed docs: delete old chunks BEFORE appending new ones.
                // FAIL-LOUD: a failed delete leaves stale chunks alongside new
                // ones (data corruption). Abort the pipeline rather than silently
                // appending on top of old data.
                if doc.is_changed {
                    let expr = lance::deps::datafusion::logical_expr::col("doc_id")
                        .eq(lance::deps::datafusion::logical_expr::lit(doc.id.as_str()));
                    let result = lance::dataset::write::DeleteBuilder::from_expr(
                        std::sync::Arc::new(ds.clone()),
                        expr,
                    ).execute().await
                        .map_err(|e| format!(
                            "Changed doc delete failed for {}: {} (aborting to prevent stale+new chunk corruption)",
                            doc.id, e
                        ))?;
                    *ds = result.new_dataset.as_ref().clone();
                    all_delete_ids.push(doc.id.clone());
                }

                let rows: Vec<ChunkRow> = doc.chunks.iter()
                    .zip(embeddings_doc.into_iter().zip(embeddings_query))
                    .map(|(chunk, (embedding, clustering_embedding))| ChunkRow {
                        doc_id: doc.id.clone(),
                        doc_type: doc_type.clone(),
                        chunk_index: chunk.index as i32,
                        chunk_count: chunk.count as i32,
                        chunk_token_start: chunk.token_start as i32,
                        doc_token_len: chunk.doc_token_len as i32,
                        embedding,
                        clustering_embedding,
                        content: chunk.text.clone(),
                    })
                    .collect();

                batch_rows.extend(rows);
                *indexed_count += 1;

                // Report progress per document — every document that
                // receives its embedding triggers a progress update so
                // the client (TerminusDB) sees continuous per-document
                // advancement, not batch-boundary jumps.
                if let Some(tx) = progress_tx {
                    let _ = tx.send(ProgressUpdate::Progress {
                        indexed: *indexed_count,
                        total_seen: *total_docs_seen,
                        skipped: skipped.len() as u64,
                    }).await;
                }
            }
        }

        // Accumulate rows for the single end-of-push append (Option C).
        // No per-micro-batch append — all rows are appended once after the
        // main loop, producing exactly 1 new data fragment per push.
        if !batch_rows.is_empty() {
            ctx.store.pipeline_written_chunks.fetch_add(batch_rows.len() as u64, std::sync::atomic::Ordering::Relaxed);
            all_rows.extend(batch_rows);
        }

        // Decrement pending by the batch chunk count (they're now embedded+written).
        ctx.store.pipeline_pending_chunks.fetch_sub(batch_chunk_count, std::sync::atomic::Ordering::Relaxed);

        // Drop all Vecs to free memory before the next micro-batch.
        // clear() keeps capacity; shrink_to_fit() returns it to the allocator
        // so RSS doesn't creep upward across micro-batches.
        batch_docs.clear();
        batch_docs.shrink_to_fit();

        Ok(())
    }

    // ─── Main loop: read ops from channel, micro-batch embed+write ──────
    loop {
        if cancel_token.is_cancelled() {
            return Err("pipeline superseded by new push (reindex)".to_string());
        }
        let op = tokio::select! {
            op = operation_rx.recv() => match op {
                Some(op) => op,
                None => break, // channel closed — flush remaining
            },
            _ = cancel_token.cancelled() => {
                return Err("pipeline superseded by new push (reindex)".to_string());
            }
        };
        match op {
            Operation::Inserted { id, string } => {
                total_docs_seen += 1;
                match chunk::chunk_text(ctx.tokenizer, &string, ctx.chunk_params) {
                    Ok(chunks) if chunks.is_empty() => {
                        skipped.push(SkippedDoc { id, message: "chunking produced zero chunks".to_owned() });
                    }
                    Ok(chunks) => {
                        batch_docs.push(DocToEmbed { id, chunks, is_changed: false });
                    }
                    Err(e) => {
                        skipped.push(SkippedDoc { id, message: format!("chunking failed: {}", e) });
                    }
                }
            }
            Operation::Changed { id, string } => {
                total_docs_seen += 1;
                match chunk::chunk_text(ctx.tokenizer, &string, ctx.chunk_params) {
                    Ok(chunks) if chunks.is_empty() => {
                        skipped.push(SkippedDoc { id, message: "chunking produced zero chunks".to_owned() });
                    }
                    Ok(chunks) => {
                        batch_docs.push(DocToEmbed { id, chunks, is_changed: true });
                    }
                    Err(e) => {
                        skipped.push(SkippedDoc { id, message: format!("chunking failed: {}", e) });
                    }
                }
            }
            Operation::Deleted { id } => {
                all_delete_ids.push(id);
            }
            Operation::Error { message } => {
                skipped.push(SkippedDoc { id: "unknown".to_owned(), message: format!("operation error: {}", message) });
            }
            Operation::Abort => {
                if let Some(tx) = &progress_tx {
                    let _ = tx.send(ProgressUpdate::Aborted).await;
                }
                return Err("push aborted by client".to_string());
            }
        }

        // Flush when the micro-batch is full.
        if batch_docs.len() >= MICRO_BATCH_DOCS {
            flush_microbatch(
                ctx, &mut ds, &mut batch_docs, &mut all_delete_ids, &mut all_rows,
                &mut skipped, store_clustering, &progress_tx,
                &mut total_docs_seen, &mut indexed_count,
                cancel_token,
            ).await?;
        }
    }

    // Channel closed — flush any remaining docs in the buffer.
    flush_microbatch(
        ctx, &mut ds, &mut batch_docs, &mut all_delete_ids, &mut all_rows,
        &mut skipped, store_clustering, &progress_tx,
        &mut total_docs_seen, &mut indexed_count,
        cancel_token,
    ).await?;

    // ─── Write phase: single append + boundary indexing + tag ───────────
    let has_work = indexed_count > 0 || !all_delete_ids.is_empty();

    // For pure Deleted docs (not Changed), do one delete now.
    if !all_delete_ids.is_empty() && indexed_count == 0 {
        ctx.store.io_microbatch_delete(&mut ds, &all_delete_ids).await
            .map_err(|e| format!("final delete failed: {}", e))?;
    }

    // Single end-of-push append: all rows accumulated during micro-batch
    // processing are appended in one ds.append() call, producing exactly
    // 1 new data fragment per push (Option C architecture).
    if !all_rows.is_empty() {
        ctx.store.io_microbatch_append(&mut ds, &all_rows).await
            .map_err(|e| format!("end-of-push append failed: {}", e))?;
    }

    // Refresh the cached handle so subsequent reads see the new data.
    ctx.store.io_refresh_cached_dataset(ctx.domain, ctx.branch).await
        .map_err(|e| format!("cache refresh failed: {}", e))?;

    if has_work {
        // Boundary-aware indexing: create indices only at every 3rd commit
        // (positions 2, 5, 8, ... 0-indexed). Non-indexed commits rely on
        // KNN fallback for search (at most 2 commits delta).
        let action = should_create_index(commit_position);

        let indexed_version: Option<u64>;

        if action == IndexAction::Create {
            let vector_config = ctx.store.vector_index_config().clone();

            // Create/update indices on the write handle.
            crate::store::lance::io_ensure_fts_index_on_dataset(&mut ds)
                .await
                .map_err(|e| format!("FTS index creation failed: {}", e))?;

            crate::store::vector_index::io_ensure_vector_index(&mut ds, &vector_config, false)
                .await
                .map_err(|e| format!("vector index creation failed: {}", e))?;

            if store_clustering {
                crate::store::vector_index::io_ensure_clustering_vector_index(&mut ds, &vector_config, false)
                    .await
                    .map_err(|e| format!("clustering vector index creation failed: {}", e))?;
            }

            indexed_version = Some(ds.version().version);

            // Increment delta count and run the incremental merge(3) cascade.
            ctx.store.io_increment_delta_count(ctx.domain, ctx.branch).await;
            let delta_count = ctx.store.io_get_delta_count(ctx.domain, ctx.branch).await;

            let (idx_before, idx_after) =
                crate::store::lance::io_incremental_cascade(&mut ds, delta_count)
                    .await
                    .unwrap_or((0, 0));

            if idx_after != idx_before {
                eprintln!(
                    "[pipeline] cascade: domain={} branch={} delta_count={} indices {}→{}",
                    ctx.domain, ctx.branch, delta_count, idx_before, idx_after
                );
            }

            ctx.store.io_refresh_cached_dataset(ctx.domain, ctx.branch).await
                .map_err(|e| format!("cascade cache refresh failed: {}", e))?;

            eprintln!(
                "[pipeline] boundary index created: domain={} branch={} commit_position={}",
                ctx.domain, ctx.branch, commit_position
            );
        } else {
            // Non-boundary commit: FTS has no flat-scan fallback (unlike vector
            // search), so the inverted index must exist and cover new fragments.
            // Use the branch-bound write handle (ds) — NOT io_ensure_fts_index
            // which opens the cached MAIN handle and would tag the wrong version
            // for non-main branches.
            let fts_version = crate::store::lance::io_ensure_fts_index_on_dataset(&mut ds)
                .await
                .map_err(|e| format!("FTS index ensure failed: {}", e))?;
            indexed_version = Some(fts_version);

            ctx.store.io_refresh_cached_dataset(ctx.domain, ctx.branch).await
                .map_err(|e| format!("FTS cache refresh failed: {}", e))?;
        }

        let last_version = ds.version().version;
        let tag_version = indexed_version.unwrap_or(last_version);
        if let Err(e) = ctx.store.io_tag_commit(ctx.domain, ctx.branch, commit, tag_version).await {
            let msg = format!("failed to tag commit: {}", e);
            if let Some(tx) = &progress_tx {
                let _ = tx.send(ProgressUpdate::Error { error: msg.clone() }).await;
            }
            return Err(msg);
        }

        ctx.store.update_last_indexed(ctx.domain, ctx.branch, commit, tag_version).await;

        // Aggressively prune untagged intermediate versions after tagging.
        if let Err(e) = ctx.store.io_cleanup_aggressive(ctx.domain, ctx.branch).await {
            eprintln!("[pipeline] aggressive cleanup failed (soft): {}", e);
        }
    } else {
        // No-op commit: tag at current version.
        if let Err(e) = ctx.store.io_open_dataset(ctx.domain, ctx.branch).await {
            let msg = format!("no-op tag: failed to ensure dataset: {}", e);
            if let Some(tx) = &progress_tx {
                let _ = tx.send(ProgressUpdate::Error { error: msg.clone() }).await;
            }
            return Err(msg);
        }

        let current_version = ctx.store.io_branch_head_version(ctx.domain, ctx.branch).await
            .map_err(|e| format!("no-op tag: failed to get branch head version: {}", e))?;

        if let Err(e) = ctx.store.io_tag_commit(ctx.domain, ctx.branch, commit, current_version).await {
            let msg = format!("no-op tag: failed to tag commit: {}", e);
            if let Some(tx) = &progress_tx {
                let _ = tx.send(ProgressUpdate::Error { error: msg.clone() }).await;
            }
            return Err(msg);
        }

        ctx.store.update_last_indexed(ctx.domain, ctx.branch, commit, current_version).await;
    }

    // Reset pipeline counters after completion.
    ctx.store.pipeline_pending_chunks.store(0, std::sync::atomic::Ordering::Relaxed);
    ctx.store.pipeline_embedded_chunks.store(0, std::sync::atomic::Ordering::Relaxed);
    ctx.store.pipeline_written_chunks.store(0, std::sync::atomic::Ordering::Relaxed);

    // Send Complete progress update if streaming.
    if let Some(tx) = &progress_tx {
        let _ = tx.send(ProgressUpdate::Complete {
            indexed_documents: indexed_count,
            skipped: skipped.clone(),
        }).await;
    }

    Ok((indexed_count, skipped))
}

// ═══════════════════════════════════════════════════════════════════════════════
// RISK-26 regression tests: no-op/empty commits must tag + advance last_indexed.
// ═══════════════════════════════════════════════════════════════════════════════
#[cfg(test)]
mod tests_risk26 {
    use super::*;
    use crate::kernel::model::BranchName;
    use std::path::Path;

    static DISABLED_CACHE: std::sync::LazyLock<crate::embed::cache::EmbedCache> = std::sync::LazyLock::new(crate::embed::cache::EmbedCache::disabled);

    /// Build a test tokenizer from the checked-in fixture.
    fn test_tokenizer() -> Tokenizer {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("assets")
            .join("tokenizer.json.bz2");
        crate::chunk::io_load_tokenizer(&path).expect("test tokenizer must load")
    }

    /// A dummy provider that will never be called (no texts to embed in no-op commits).
    fn dummy_provider() -> Provider {
        Provider::OpenAiCompatible {
            base_url: "http://127.0.0.1:0/never-called".to_owned(),
            model: "test-noop".to_owned(),
            dim: 8,
        }
    }

    /// Helper: build a PipelineCtx for test use.
    fn make_ctx<'a>(
        store: &'a LanceStore,
        tokenizer: &'a Tokenizer,
        chunk_params: &'a ChunkParams,
        provider: &'a Provider,
        http_client: &'a reqwest::Client,
        domain: &'a str,
        branch: &'a str,
    ) -> PipelineCtx<'a> {
        PipelineCtx {
            store,
            tokenizer,
            chunk_params,
            provider,
            http_client,
            embed_cache: &DISABLED_CACHE,
            domain,
            branch,
            embed_batch_size: 32,
        }
    }

    /// regression (a): a commit of ONLY Operation::Error must tag + advance last_indexed.
    #[tokio::test]
    async fn all_error_commit_tags_and_advances_last_indexed() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let store = LanceStore::new(tmp.path(), 8, 256 * 1024 * 1024, 128 * 1024 * 1024);
        let tokenizer = test_tokenizer();
        let chunk_params = ChunkParams { max_tokens: 512, overlap: 64 };
        let provider = dummy_provider();
        let http_client = reqwest::Client::new();
        let domain = "admin/risk26_error";
        let branch = "main";

        // Pre-condition: seed the domain with one real doc so a prior version exists.
        let seed_row = ChunkRow {
            doc_id: "doc/seed".to_owned(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: vec![0.1; 8],
            clustering_embedding: vec![0.1; 8],
            content: "seed".to_owned(),
        };
        let v_seed = store
            .io_upsert_chunks(domain, branch, "doc/seed", std::slice::from_ref(&seed_row))
            .await
            .expect("seed upsert");
        store.io_tag_commit(domain, branch, "c0_seed", v_seed).await.expect("tag seed");
        store.update_last_indexed(domain, branch, "c0_seed", v_seed).await;

        // ACT: push a commit with ONLY Operation::Error entries.
        let operations = vec![
            Operation::Error { message: "render failed: doc/broken1".to_owned() },
            Operation::Error { message: "render failed: doc/broken2".to_owned() },
        ];
        let ctx = make_ctx(&store, &tokenizer, &chunk_params, &provider, &http_client, domain, branch);
        let (indexed, skipped) = io_run_index_pipeline(&ctx, "c1_all_error", operations)
            .await
            .expect("pipeline must succeed for all-error commit");

        // ASSERT: zero indexed, errors recorded as skipped.
        assert_eq!(indexed, 0, "no docs should be indexed for all-error commit");
        assert_eq!(skipped.len(), 2, "both errors should be in skipped");

        // ASSERT: last_indexed advanced to c1_all_error.
        let li = store
            .last_indexed(
                &Domain::from_resource_path(&parse_domain(domain).unwrap()),
                &BranchName::new(branch.to_owned()),
            )
            .await
            .expect("last_indexed read");
        assert_eq!(
            li.commit.as_deref(),
            Some("c1_all_error"),
            "last_indexed MUST advance past the all-error commit"
        );

        // ASSERT: the commit is tagged (resolvable).
        let resolved = store.io_resolve_commit(domain, branch, "c1_all_error").await.expect("resolve");
        assert!(
            resolved.is_some(),
            "all-error commit MUST be tagged and resolvable"
        );
        // The tag version should be the seed version (no new data written).
        assert_eq!(resolved.unwrap(), v_seed, "tag must point to the prior data version");
    }

    /// Abort operation: pipeline must return Err and NOT advance last_indexed.
    #[tokio::test]
    async fn abort_operation_returns_error() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let store = LanceStore::new(tmp.path(), 8, 256 * 1024 * 1024, 128 * 1024 * 1024);
        let tokenizer = test_tokenizer();
        let chunk_params = ChunkParams { max_tokens: 512, overlap: 64 };
        let provider = dummy_provider();
        let http_client = reqwest::Client::new();
        let domain = "admin/abort_test";
        let branch = "main";

        // Seed with one doc so a prior version exists.
        let seed_row = ChunkRow {
            doc_id: "doc/seed".to_owned(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: vec![0.1; 8],
            clustering_embedding: vec![0.1; 8],
            content: "seed".to_owned(),
        };
        let v_seed = store
            .io_upsert_chunks(domain, branch, "doc/seed", std::slice::from_ref(&seed_row))
            .await
            .expect("seed upsert");
        store.io_tag_commit(domain, branch, "c0_seed", v_seed).await.expect("tag seed");
        store.update_last_indexed(domain, branch, "c0_seed", v_seed).await;

        // ACT: push a commit with an Abort operation.
        let operations = vec![Operation::Abort];
        let ctx = make_ctx(&store, &tokenizer, &chunk_params, &provider, &http_client, domain, branch);
        let result = io_run_index_pipeline(&ctx, "c1_abort", operations).await;

        // ASSERT: pipeline returns Err.
        assert!(result.is_err(), "Abort must return an error");
        assert!(result.unwrap_err().contains("aborted"), "error message must mention abort");

        // ASSERT: last_indexed NOT advanced (still at seed commit).
        let li = store
            .last_indexed(
                &Domain::from_resource_path(&parse_domain(domain).unwrap()),
                &BranchName::new(branch.to_owned()),
            )
            .await
            .expect("last_indexed read");
        assert_eq!(
            li.commit.as_deref(),
            Some("c0_seed"),
            "last_indexed must NOT advance after abort"
        );
    }

    /// regression (b): an empty-operations commit must tag + advance last_indexed.
    #[tokio::test]
    async fn empty_operations_commit_tags_and_advances_last_indexed() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let store = LanceStore::new(tmp.path(), 8, 256 * 1024 * 1024, 128 * 1024 * 1024);
        let tokenizer = test_tokenizer();
        let chunk_params = ChunkParams { max_tokens: 512, overlap: 64 };
        let provider = dummy_provider();
        let http_client = reqwest::Client::new();
        let domain = "admin/risk26_empty";
        let branch = "main";

        // Seed with one real doc.
        let seed_row = ChunkRow {
            doc_id: "doc/seed".to_owned(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: vec![0.1; 8],
            clustering_embedding: vec![0.1; 8],
            content: "seed".to_owned(),
        };
        let v_seed = store
            .io_upsert_chunks(domain, branch, "doc/seed", std::slice::from_ref(&seed_row))
            .await
            .expect("seed upsert");
        store.io_tag_commit(domain, branch, "c0_seed", v_seed).await.expect("tag seed");
        store.update_last_indexed(domain, branch, "c0_seed", v_seed).await;

        // ACT: push a commit with ZERO operations.
        let operations: Vec<Operation> = vec![];
        let ctx = make_ctx(&store, &tokenizer, &chunk_params, &provider, &http_client, domain, branch);
        let (indexed, skipped) = io_run_index_pipeline(&ctx, "c1_empty", operations)
            .await
            .expect("pipeline must succeed for empty commit");

        // ASSERT: zero indexed, zero skipped.
        assert_eq!(indexed, 0);
        assert_eq!(skipped.len(), 0);

        // ASSERT: last_indexed advanced to c1_empty.
        let li = store
            .last_indexed(
                &Domain::from_resource_path(&parse_domain(domain).unwrap()),
                &BranchName::new(branch.to_owned()),
            )
            .await
            .expect("last_indexed read");
        assert_eq!(
            li.commit.as_deref(),
            Some("c1_empty"),
            "last_indexed MUST advance past the empty commit"
        );

        // ASSERT: the commit is tagged.
        let resolved = store.io_resolve_commit(domain, branch, "c1_empty").await.expect("resolve");
        assert!(resolved.is_some(), "empty commit MUST be tagged");
        assert_eq!(resolved.unwrap(), v_seed, "tag must point to the prior data version");
    }

    /// regression (c): a NORMAL commit after a no-op commit must still index correctly.
    /// This proves catch-up is not stalled — the engine progresses past the empty commit.
    #[tokio::test]
    async fn normal_commit_after_noop_indexes_correctly() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let store = LanceStore::new(tmp.path(), 8, 256 * 1024 * 1024, 128 * 1024 * 1024);
        let tokenizer = test_tokenizer();
        let chunk_params = ChunkParams { max_tokens: 512, overlap: 64 };
        let provider = dummy_provider();
        let http_client = reqwest::Client::new();
        let domain = "admin/risk26_resume";
        let branch = "main";

        // Seed with one doc.
        let seed_row = ChunkRow {
            doc_id: "doc/seed".to_owned(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: vec![0.1; 8],
            clustering_embedding: vec![0.1; 8],
            content: "seed".to_owned(),
        };
        let v_seed = store
            .io_upsert_chunks(domain, branch, "doc/seed", std::slice::from_ref(&seed_row))
            .await
            .expect("seed upsert");
        store.io_tag_commit(domain, branch, "c0_seed", v_seed).await.expect("tag seed");
        store.update_last_indexed(domain, branch, "c0_seed", v_seed).await;

        // Push an all-error commit (no-op).
        let error_ops = vec![
            Operation::Error { message: "transient failure".to_owned() },
        ];
        let ctx = make_ctx(&store, &tokenizer, &chunk_params, &provider, &http_client, domain, branch);
        io_run_index_pipeline(&ctx, "c1_noop", error_ops)
            .await
            .expect("noop commit must succeed");

        // Verify catch-up is unblocked: last_indexed is c1_noop.
        let li_after_noop = store
            .last_indexed(
                &Domain::from_resource_path(&parse_domain(domain).unwrap()),
                &BranchName::new(branch.to_owned()),
            )
            .await
            .expect("last_indexed read");
        assert_eq!(li_after_noop.commit.as_deref(), Some("c1_noop"));

        // Now push a NORMAL commit with a real Insert (simulated: directly upsert
        // rows and tag, mirroring what the pipeline would do with a working embedder).
        // This proves the engine can advance PAST the empty commit.
        let normal_row = ChunkRow {
            doc_id: "doc/new".to_owned(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            embedding: vec![0.2; 8],
            clustering_embedding: vec![0.2; 8],
            content: "new document".to_owned(),
        };
        let v_normal = store
            .io_upsert_chunks(domain, branch, "doc/new", std::slice::from_ref(&normal_row))
            .await
            .expect("normal upsert");
        store.io_tag_commit(domain, branch, "c2_normal", v_normal).await.expect("tag normal");
        store.update_last_indexed(domain, branch, "c2_normal", v_normal).await;

        // ASSERT: last_indexed advanced to c2_normal (past the no-op).
        let li_final = store
            .last_indexed(
                &Domain::from_resource_path(&parse_domain(domain).unwrap()),
                &BranchName::new(branch.to_owned()),
            )
            .await
            .expect("last_indexed read");
        assert_eq!(
            li_final.commit.as_deref(),
            Some("c2_normal"),
            "last_indexed MUST advance past the no-op to the normal commit"
        );
        assert!(
            li_final.version > v_seed,
            "version must have advanced beyond the seed"
        );

        // ASSERT: both the no-op commit AND the normal commit are resolvable.
        let r_noop = store.io_resolve_commit(domain, branch, "c1_noop").await.expect("resolve noop");
        let r_normal = store.io_resolve_commit(domain, branch, "c2_normal").await.expect("resolve normal");
        assert!(r_noop.is_some(), "no-op commit must remain resolvable");
        assert!(r_normal.is_some(), "normal commit must be resolvable");
        assert!(
            r_normal.unwrap() > r_noop.unwrap(),
            "normal commit version must be newer than the no-op tag version"
        );
    }

    /// Edge case: no-op commit on a never-before-indexed domain (first commit is empty).
    /// The dataset auto-creates and the commit gets tagged to version 1.
    #[tokio::test]
    async fn noop_commit_on_fresh_domain_creates_dataset_and_tags() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let store = LanceStore::new(tmp.path(), 8, 256 * 1024 * 1024, 128 * 1024 * 1024);
        let tokenizer = test_tokenizer();
        let chunk_params = ChunkParams { max_tokens: 512, overlap: 64 };
        let provider = dummy_provider();
        let http_client = reqwest::Client::new();
        let domain = "admin/risk26_fresh";
        let branch = "main";

        // NO SEED — domain has never been indexed.
        // Push an empty commit.
        let operations: Vec<Operation> = vec![];
        let ctx = make_ctx(&store, &tokenizer, &chunk_params, &provider, &http_client, domain, branch);
        let (indexed, skipped) = io_run_index_pipeline(&ctx, "c0_first_empty", operations)
            .await
            .expect("pipeline must succeed for first-ever empty commit");

        assert_eq!(indexed, 0);
        assert_eq!(skipped.len(), 0);

        // ASSERT: last_indexed is set (commit is known).
        let li = store
            .last_indexed(
                &Domain::from_resource_path(&parse_domain(domain).unwrap()),
                &BranchName::new(branch.to_owned()),
            )
            .await
            .expect("last_indexed read");
        assert_eq!(
            li.commit.as_deref(),
            Some("c0_first_empty"),
            "first-ever empty commit must still advance last_indexed"
        );

        // ASSERT: the commit is tagged.
        let resolved = store.io_resolve_commit(domain, branch, "c0_first_empty").await.expect("resolve");
        assert!(resolved.is_some(), "first-ever empty commit must be tagged");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Streaming pipeline tests: verify io_run_index_pipeline_stream produces the
// same results as the batch io_run_index_pipeline.
// ═══════════════════════════════════════════════════════════════════════════════
#[cfg(test)]
mod tests_streaming {
    use super::*;
    use crate::kernel::model::BranchName;
    use std::path::Path;

    static DISABLED_CACHE: std::sync::LazyLock<crate::embed::cache::EmbedCache> = std::sync::LazyLock::new(crate::embed::cache::EmbedCache::disabled);

    fn test_tokenizer() -> Tokenizer {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("assets")
            .join("tokenizer.json.bz2");
        crate::chunk::io_load_tokenizer(&path).expect("test tokenizer must load")
    }

    fn dummy_provider() -> Provider {
        Provider::OpenAiCompatible {
            base_url: "http://127.0.0.1:0/never-called".to_owned(),
            model: "test-noop".to_owned(),
            dim: 8,
        }
    }

    fn make_ctx<'a>(
        store: &'a LanceStore,
        tokenizer: &'a Tokenizer,
        chunk_params: &'a ChunkParams,
        provider: &'a Provider,
        http_client: &'a reqwest::Client,
        domain: &'a str,
        branch: &'a str,
    ) -> PipelineCtx<'a> {
        PipelineCtx {
            store,
            tokenizer,
            chunk_params,
            provider,
            http_client,
            embed_cache: &DISABLED_CACHE,
            domain,
            branch,
            embed_batch_size: 32,
        }
    }

    /// The streaming pipeline must handle an empty channel (no operations)
    /// the same way the batch pipeline handles an empty Vec.
    #[tokio::test]
    async fn stream_empty_operations_tags_and_advances_last_indexed() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let store = LanceStore::new(tmp.path(), 8, 256 * 1024 * 1024, 128 * 1024 * 1024);
        let tokenizer = test_tokenizer();
        let chunk_params = ChunkParams { max_tokens: 512, overlap: 64 };
        let provider = dummy_provider();
        let http_client = reqwest::Client::new();
        let domain = "admin/stream_empty";
        let branch = "main";

        // Seed with one doc.
        let seed_row = ChunkRow {
            doc_id: "doc/seed".to_owned(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: vec![0.1; 8],
            clustering_embedding: vec![0.1; 8],
            content: "seed".to_owned(),
        };
        let v_seed = store
            .io_upsert_chunks(domain, branch, "doc/seed", std::slice::from_ref(&seed_row))
            .await
            .expect("seed upsert");
        store.io_tag_commit(domain, branch, "c0_seed", v_seed).await.expect("tag seed");
        store.update_last_indexed(domain, branch, "c0_seed", v_seed).await;

        // ACT: streaming pipeline with an empty channel (immediately closed).
        let (_tx, rx) = mpsc::channel::<Operation>(16);
        drop(_tx); // Close immediately — no operations.

        let ctx = make_ctx(&store, &tokenizer, &chunk_params, &provider, &http_client, domain, branch);
        let (indexed, skipped) = io_run_index_pipeline_stream(&ctx, "c1_empty", rx, None, &tokio_util::sync::CancellationToken::new())
            .await
            .expect("streaming pipeline must succeed for empty channel");

        assert_eq!(indexed, 0);
        assert_eq!(skipped.len(), 0);

        let li = store
            .last_indexed(
                &Domain::from_resource_path(&parse_domain(domain).unwrap()),
                &BranchName::new(branch.to_owned()),
            )
            .await
            .expect("last_indexed read");
        assert_eq!(
            li.commit.as_deref(),
            Some("c1_empty"),
            "streaming pipeline must advance last_indexed for empty channel"
        );
    }

    /// The streaming pipeline must handle all-Error operations the same way
    /// as the batch pipeline.
    #[tokio::test]
    async fn stream_all_error_operations_tags_and_advances() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let store = LanceStore::new(tmp.path(), 8, 256 * 1024 * 1024, 128 * 1024 * 1024);
        let tokenizer = test_tokenizer();
        let chunk_params = ChunkParams { max_tokens: 512, overlap: 64 };
        let provider = dummy_provider();
        let http_client = reqwest::Client::new();
        let domain = "admin/stream_errors";
        let branch = "main";

        // Seed.
        let seed_row = ChunkRow {
            doc_id: "doc/seed".to_owned(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: vec![0.1; 8],
            clustering_embedding: vec![0.1; 8],
            content: "seed".to_owned(),
        };
        let v_seed = store
            .io_upsert_chunks(domain, branch, "doc/seed", std::slice::from_ref(&seed_row))
            .await
            .expect("seed upsert");
        store.io_tag_commit(domain, branch, "c0_seed", v_seed).await.expect("tag seed");
        store.update_last_indexed(domain, branch, "c0_seed", v_seed).await;

        // ACT: streaming pipeline with all-Error operations.
        let (tx, rx) = mpsc::channel::<Operation>(16);
        tx.send(Operation::Error { message: "error1".to_owned() }).await.unwrap();
        tx.send(Operation::Error { message: "error2".to_owned() }).await.unwrap();
        drop(tx);

        let ctx = make_ctx(&store, &tokenizer, &chunk_params, &provider, &http_client, domain, branch);
        let (indexed, skipped) = io_run_index_pipeline_stream(&ctx, "c1_errors", rx, None, &tokio_util::sync::CancellationToken::new())
            .await
            .expect("streaming pipeline must succeed for all-error channel");

        assert_eq!(indexed, 0, "no docs should be indexed for all-error stream");
        assert_eq!(skipped.len(), 2, "both errors should be in skipped");

        let li = store
            .last_indexed(
                &Domain::from_resource_path(&parse_domain(domain).unwrap()),
                &BranchName::new(branch.to_owned()),
            )
            .await
            .expect("last_indexed read");
        assert_eq!(
            li.commit.as_deref(),
            Some("c1_errors"),
            "streaming pipeline must advance last_indexed past all-error commit"
        );
    }

    /// The streaming pipeline must handle Abort operations correctly.
    #[tokio::test]
    async fn stream_abort_returns_error() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let store = LanceStore::new(tmp.path(), 8, 256 * 1024 * 1024, 128 * 1024 * 1024);
        let tokenizer = test_tokenizer();
        let chunk_params = ChunkParams { max_tokens: 512, overlap: 64 };
        let provider = dummy_provider();
        let http_client = reqwest::Client::new();
        let domain = "admin/stream_abort";
        let branch = "main";

        // Seed.
        let seed_row = ChunkRow {
            doc_id: "doc/seed".to_owned(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: vec![0.1; 8],
            clustering_embedding: vec![0.1; 8],
            content: "seed".to_owned(),
        };
        let v_seed = store
            .io_upsert_chunks(domain, branch, "doc/seed", std::slice::from_ref(&seed_row))
            .await
            .expect("seed upsert");
        store.io_tag_commit(domain, branch, "c0_seed", v_seed).await.expect("tag seed");
        store.update_last_indexed(domain, branch, "c0_seed", v_seed).await;

        // ACT: streaming pipeline with Abort.
        let (tx, rx) = mpsc::channel::<Operation>(16);
        tx.send(Operation::Abort).await.unwrap();
        drop(tx);

        let ctx = make_ctx(&store, &tokenizer, &chunk_params, &provider, &http_client, domain, branch);
        let result = io_run_index_pipeline_stream(&ctx, "c1_abort", rx, None, &tokio_util::sync::CancellationToken::new()).await;

        assert!(result.is_err(), "Abort must return error in streaming pipeline");
        assert!(result.unwrap_err().contains("aborted"));

        // last_indexed must NOT advance.
        let li = store
            .last_indexed(
                &Domain::from_resource_path(&parse_domain(domain).unwrap()),
                &BranchName::new(branch.to_owned()),
            )
            .await
            .expect("last_indexed read");
        assert_eq!(
            li.commit.as_deref(),
            Some("c0_seed"),
            "last_indexed must NOT advance after abort in streaming pipeline"
        );
    }

    /// The streaming pipeline must handle Deleted operations and advance last_indexed.
    #[tokio::test]
    async fn stream_deleted_operations_tags_and_advances() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let store = LanceStore::new(tmp.path(), 8, 256 * 1024 * 1024, 128 * 1024 * 1024);
        let tokenizer = test_tokenizer();
        let chunk_params = ChunkParams { max_tokens: 512, overlap: 64 };
        let provider = dummy_provider();
        let http_client = reqwest::Client::new();
        let domain = "admin/stream_deleted";
        let branch = "main";

        // Seed with a doc.
        let seed_row = ChunkRow {
            doc_id: "doc/seed".to_owned(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: vec![0.1; 8],
            clustering_embedding: vec![0.1; 8],
            content: "seed".to_owned(),
        };
        let v_seed = store
            .io_upsert_chunks(domain, branch, "doc/seed", std::slice::from_ref(&seed_row))
            .await
            .expect("seed upsert");
        store.io_tag_commit(domain, branch, "c0_seed", v_seed).await.expect("tag seed");
        store.update_last_indexed(domain, branch, "c0_seed", v_seed).await;

        // ACT: streaming pipeline with a Deleted operation.
        let (tx, rx) = mpsc::channel::<Operation>(16);
        tx.send(Operation::Deleted { id: "doc/seed".to_owned() }).await.unwrap();
        drop(tx);

        let ctx = make_ctx(&store, &tokenizer, &chunk_params, &provider, &http_client, domain, branch);
        let (indexed, skipped) = io_run_index_pipeline_stream(&ctx, "c1_delete", rx, None, &tokio_util::sync::CancellationToken::new())
            .await
            .expect("streaming pipeline must succeed for delete-only channel");

        assert_eq!(indexed, 0, "no docs indexed for delete-only stream");
        assert_eq!(skipped.len(), 0, "no skipped for delete-only stream");

        let li = store
            .last_indexed(
                &Domain::from_resource_path(&parse_domain(domain).unwrap()),
                &BranchName::new(branch.to_owned()),
            )
            .await
            .expect("last_indexed read");
        assert_eq!(
            li.commit.as_deref(),
            Some("c1_delete"),
            "streaming pipeline must advance last_indexed for delete-only commit"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Same-role exact-duplicate distance override tests.
// Verifies that bit-identical stored embeddings produce distance=0 across all
// four search-family endpoints (/similar, /duplicates, /resolve). /search is
// tested indirectly via io_fetch_result_embeddings because it requires a live
// embedding server for the query embedding step.
// ═══════════════════════════════════════════════════════════════════════════════
#[cfg(test)]
mod tests_exact_duplicate_override {
    use super::*;
    use crate::kernel::distance::l2_normalize;
    use crate::store::lance::{ChunkRow, DuplicateScope, LanceStore};
    use std::sync::Arc;

    /// Build a test SearchService with a small embedding dimension and no live
    /// embedding server (search_ready=true to pass the gate, but no real embedding
    /// calls will succeed — tests that don't need live embedding are safe).
    fn make_test_service(store: Arc<LanceStore>, dim: usize) -> SearchService {
        let config = Config {
            admin_user: "admin".to_owned(),
            admin_secret: "root".to_owned(),
            port: 0,
            embed_provider: crate::embed::Provider::OpenAiCompatible {
                base_url: "http://127.0.0.1:0/no-server".to_owned(),
                model: "test-model".to_owned(),
                dim,
            },
            data_dir: "/tmp/test".to_owned(),
            tokenizer_path: "assets/tokenizer.json.bz2".to_owned(),
            embed_batch_size: 32,
            embed_cache_size: None,
            prometheus_port: None,
            lance_index_cache_bytes: 256 * 1024 * 1024,
            lance_metadata_cache_bytes: 128 * 1024 * 1024,
        };
        let tokenizer_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("assets")
            .join("tokenizer.json.bz2");
        let tokenizer =
            crate::chunk::io_load_tokenizer(&tokenizer_path).expect("test tokenizer must load");
        let svc = SearchService::new(store, config, tokenizer);
        svc.set_search_ready(true);
        svc
    }

    /// Create a normalized embedding from a seed. Deterministic, L2-normalized.
    fn make_embedding(dim: usize, seed: f32) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim).map(|i| (seed + i as f32 * 0.1).sin()).collect();
        l2_normalize(&mut v);
        v
    }

    /// Seed a store with two documents:
    /// - doc_a with embedding `emb_a` and clustering_embedding `qemb_a`
    /// - doc_b with embedding `emb_b` and clustering_embedding `qemb_b`
    ///
    /// Tags the commit and returns the domain string.
    async fn seed_two_docs(
        store: &LanceStore,
        emb_a: Vec<f32>,
        qemb_a: Vec<f32>,
        emb_b: Vec<f32>,
        qemb_b: Vec<f32>,
        domain: &str,
    ) -> String {
        let row_a = ChunkRow {
            doc_id: "doc/a".to_owned(),
            doc_type: "Item".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: emb_a,
            clustering_embedding: qemb_a,
            content: "content of doc a".to_owned(),
        };
        let row_b = ChunkRow {
            doc_id: "doc/b".to_owned(),
            doc_type: "Item".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: emb_b,
            clustering_embedding: qemb_b,
            content: "content of doc b".to_owned(),
        };

        store
            .io_upsert_chunks(domain, "main", "doc/a", std::slice::from_ref(&row_a))
            .await
            .expect("upsert doc/a");
        store
            .io_upsert_chunks(domain, "main", "doc/b", std::slice::from_ref(&row_b))
            .await
            .expect("upsert doc/b");

        // Refresh and tag.
        store
            .io_refresh_cached_dataset(domain, "main")
            .await
            .expect("refresh");
        let ds_arc = store.io_open_dataset(domain, "main").await.unwrap();
        let version = ds_arc.read().await.version().version;
        let commit = "c_test".to_owned();
        store
            .io_tag_commit(domain, "main", &commit, version)
            .await
            .expect("tag");
        store
            .update_last_indexed(domain, "main", &commit, version)
            .await;
        commit
    }

    // ─── /similar: exact-duplicate override (doc↔doc) ─────────────────────────

    /// /similar with bit-identical DOCUMENT embeddings: distance MUST be 0.
    #[tokio::test]
    async fn similar_identical_doc_embeddings_distance_zero() {
        let dim = 16;
        let tmp = tempfile::tempdir().expect("temp dir");
        let store = Arc::new(LanceStore::new(tmp.path(), dim, 256 * 1024 * 1024, 128 * 1024 * 1024));

        let emb = make_embedding(dim, 1.0);
        let qemb_a = make_embedding(dim, 2.0);
        let qemb_b = make_embedding(dim, 3.0);

        // IDENTICAL doc-role embeddings for both docs.
        let commit = seed_two_docs(
            &store,
            emb.clone(),
            qemb_a.clone(),
            emb.clone(),
            qemb_b.clone(),
            "admin/sim_ident",
        )
        .await;

        let svc = make_test_service(Arc::clone(&store), dim);

        let result = svc
            .similar_with_options(
                "admin/sim_ident",
                &commit,
                "doc/a",
                0,
                10,
                &[],
                &[],
                false,
                &[],
            )
            .await
            .expect("similar must succeed");

        // doc/b should appear with distance=0 (same doc-role embedding as doc/a).
        let doc_b_hit = result.hits.iter().find(|h| h.id == "doc/b");
        assert!(
            doc_b_hit.is_some(),
            "doc/b must appear in /similar results for doc/a"
        );
        assert_eq!(
            doc_b_hit.unwrap().distance,
            0.0,
            "bit-identical doc-role embeddings must produce distance=0 via the override"
        );
    }

    /// /similar with DIFFERENT document embeddings: distance MUST be > 0.
    #[tokio::test]
    async fn similar_different_doc_embeddings_distance_nonzero() {
        let dim = 16;
        let tmp = tempfile::tempdir().expect("temp dir");
        let store = Arc::new(LanceStore::new(tmp.path(), dim, 256 * 1024 * 1024, 128 * 1024 * 1024));

        let emb_a = make_embedding(dim, 1.0);
        let emb_b = make_embedding(dim, 5.0); // Different seed → different embedding.
        let qemb_a = make_embedding(dim, 2.0);
        let qemb_b = make_embedding(dim, 3.0);

        let commit = seed_two_docs(
            &store,
            emb_a,
            qemb_a,
            emb_b,
            qemb_b,
            "admin/sim_diff",
        )
        .await;

        let svc = make_test_service(Arc::clone(&store), dim);

        let result = svc
            .similar_with_options(
                "admin/sim_diff",
                &commit,
                "doc/a",
                0,
                10,
                &[],
                &[],
                false,
                &[],
            )
            .await
            .expect("similar must succeed");

        let doc_b_hit = result.hits.iter().find(|h| h.id == "doc/b");
        assert!(
            doc_b_hit.is_some(),
            "doc/b must appear in /similar results for doc/a"
        );
        assert!(
            doc_b_hit.unwrap().distance > 0.0,
            "different doc-role embeddings must produce distance > 0, got {}",
            doc_b_hit.unwrap().distance
        );
    }

    // ─── /duplicates: exact-duplicate override (doc↔doc) ──────────────────────

    /// /duplicates with bit-identical DOCUMENT embeddings: distance MUST be 0.
    #[tokio::test]
    async fn duplicates_identical_doc_embeddings_distance_zero() {
        let dim = 16;
        let tmp = tempfile::tempdir().expect("temp dir");
        let store = Arc::new(LanceStore::new(tmp.path(), dim, 256 * 1024 * 1024, 128 * 1024 * 1024));

        let emb = make_embedding(dim, 1.0);
        let qemb_a = make_embedding(dim, 2.0);
        let qemb_b = make_embedding(dim, 3.0);

        // IDENTICAL doc-role embeddings.
        let commit = seed_two_docs(
            &store,
            emb.clone(),
            qemb_a,
            emb.clone(),
            qemb_b,
            "admin/dup_ident",
        )
        .await;

        let svc = make_test_service(Arc::clone(&store), dim);

        let groups = svc
            .duplicates_with_options(
                "admin/dup_ident",
                &commit,
                1.0, // Permissive threshold — find everything.
                &DuplicateScope::default(),
                false,
                0,
                100,
                &[],
            )
            .await
            .expect("duplicates must succeed");

        // The pair (doc/a, doc/b) should appear with distance=0.
        let pair = groups.iter().find(|g| {
            (g.group[0].id == "doc/a" && g.group[1].id == "doc/b")
                || (g.group[0].id == "doc/b" && g.group[1].id == "doc/a")
        });
        assert!(
            pair.is_some(),
            "doc/a + doc/b must appear as a duplicate pair, got groups: {:?}",
            groups.iter().map(|g| (&g.group[0].id, &g.group[1].id, g.distance)).collect::<Vec<_>>()
        );
        assert_eq!(
            pair.unwrap().distance,
            0.0,
            "bit-identical doc-role embeddings must produce distance=0 via the override"
        );
    }

    /// /duplicates with DIFFERENT document embeddings: distance MUST be > 0.
    #[tokio::test]
    async fn duplicates_different_doc_embeddings_distance_nonzero() {
        let dim = 16;
        let tmp = tempfile::tempdir().expect("temp dir");
        let store = Arc::new(LanceStore::new(tmp.path(), dim, 256 * 1024 * 1024, 128 * 1024 * 1024));

        let emb_a = make_embedding(dim, 1.0);
        let emb_b = make_embedding(dim, 5.0); // Different seed.
        let qemb_a = make_embedding(dim, 2.0);
        let qemb_b = make_embedding(dim, 3.0);

        let commit = seed_two_docs(
            &store,
            emb_a,
            qemb_a,
            emb_b,
            qemb_b,
            "admin/dup_diff",
        )
        .await;

        let svc = make_test_service(Arc::clone(&store), dim);

        let groups = svc
            .duplicates_with_options(
                "admin/dup_diff",
                &commit,
                1.0,
                &DuplicateScope::default(),
                false,
                0,
                100,
                &[],
            )
            .await
            .expect("duplicates must succeed");

        let pair = groups.iter().find(|g| {
            (g.group[0].id == "doc/a" && g.group[1].id == "doc/b")
                || (g.group[0].id == "doc/b" && g.group[1].id == "doc/a")
        });
        assert!(
            pair.is_some(),
            "doc/a + doc/b must appear as a pair with permissive threshold=1.0"
        );
        assert!(
            pair.unwrap().distance > 0.0,
            "different doc-role embeddings must produce distance > 0, got {}",
            pair.unwrap().distance
        );
    }

    // ─── /resolve: exact-duplicate override (doc↔doc) ─────────────────────────

    /// /candidates with bit-identical DOCUMENT embeddings: candidate distance MUST be 0.
    #[tokio::test]
    async fn candidates_identical_doc_embeddings_distance_zero() {
        let dim = 16;
        let tmp = tempfile::tempdir().expect("temp dir");
        let store = Arc::new(LanceStore::new(tmp.path(), dim, 256 * 1024 * 1024, 128 * 1024 * 1024));

        let emb = make_embedding(dim, 1.0);
        let qemb_a = make_embedding(dim, 2.0);
        let qemb_b = make_embedding(dim, 3.0);

        // Seed with two doc types for cross-set resolution.
        let row_a = ChunkRow {
            doc_id: "doc/set_item".to_owned(),
            doc_type: "SetType".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: emb.clone(),
            clustering_embedding: qemb_a,
            content: "set item content".to_owned(),
        };
        let row_b = ChunkRow {
            doc_id: "doc/target_item".to_owned(),
            doc_type: "TargetType".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: emb.clone(), // IDENTICAL doc embedding.
            clustering_embedding: qemb_b,
            content: "target item content".to_owned(),
        };

        let domain = "admin/res_ident";
        store
            .io_upsert_chunks(domain, "main", "doc/set_item", std::slice::from_ref(&row_a))
            .await
            .expect("upsert set_item");
        store
            .io_upsert_chunks(domain, "main", "doc/target_item", std::slice::from_ref(&row_b))
            .await
            .expect("upsert target_item");
        store
            .io_refresh_cached_dataset(domain, "main")
            .await
            .expect("refresh");
        let ds_arc = store.io_open_dataset(domain, "main").await.unwrap();
        let version = ds_arc.read().await.version().version;
        let commit = "c_res";
        store
            .io_tag_commit(domain, "main", commit, version)
            .await
            .expect("tag");
        store
            .update_last_indexed(domain, "main", commit, version)
            .await;

        let svc = make_test_service(Arc::clone(&store), dim);

        let scope = DuplicateScope {
            set_doc_types: vec!["SetType".to_owned()],
            target_doc_types: vec!["TargetType".to_owned()],
            ..Default::default()
        };

        let result = svc
            .candidates_gather(
                domain,
                commit,
                &scope,
                5,     // k
                1.0,   // threshold_set (permissive)
                1.0,   // threshold_target (permissive)
                false, // include_embeddings
                false, // include_content
                &[],   // ancestors
            )
            .await
            .expect("candidates must succeed");

        // set_item should have target_item as a candidate with distance=0.
        let set_nbrs = result.set_to_target.get("doc/set_item");
        assert!(set_nbrs.is_some(), "set_item must have candidates");
        let candidate = set_nbrs.unwrap().iter().find(|c| c.id == "doc/target_item");
        assert!(
            candidate.is_some(),
            "set_item must have target_item as a candidate"
        );
        assert_eq!(
            candidate.unwrap().distance,
            0.0,
            "bit-identical doc-role embeddings must produce distance=0"
        );
    }

    /// /candidates with DIFFERENT document embeddings: candidate distance MUST be > 0.
    #[tokio::test]
    async fn candidates_different_doc_embeddings_distance_nonzero() {
        let dim = 16;
        let tmp = tempfile::tempdir().expect("temp dir");
        let store = Arc::new(LanceStore::new(tmp.path(), dim, 256 * 1024 * 1024, 128 * 1024 * 1024));

        let emb_a = make_embedding(dim, 1.0);
        let emb_b = make_embedding(dim, 1.1); // Slightly different — close enough to match.
        let qemb_a = make_embedding(dim, 2.0);
        let qemb_b = make_embedding(dim, 3.0);

        let row_a = ChunkRow {
            doc_id: "doc/set_item".to_owned(),
            doc_type: "SetType".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: emb_a,
            clustering_embedding: qemb_a,
            content: "set item content".to_owned(),
        };
        let row_b = ChunkRow {
            doc_id: "doc/target_item".to_owned(),
            doc_type: "TargetType".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: emb_b, // DIFFERENT doc embedding.
            clustering_embedding: qemb_b,
            content: "target item content".to_owned(),
        };

        let domain = "admin/res_diff";
        store
            .io_upsert_chunks(domain, "main", "doc/set_item", std::slice::from_ref(&row_a))
            .await
            .expect("upsert set_item");
        store
            .io_upsert_chunks(domain, "main", "doc/target_item", std::slice::from_ref(&row_b))
            .await
            .expect("upsert target_item");
        store
            .io_refresh_cached_dataset(domain, "main")
            .await
            .expect("refresh");
        let ds_arc = store.io_open_dataset(domain, "main").await.unwrap();
        let version = ds_arc.read().await.version().version;
        let commit = "c_res";
        store
            .io_tag_commit(domain, "main", commit, version)
            .await
            .expect("tag");
        store
            .update_last_indexed(domain, "main", commit, version)
            .await;

        let svc = make_test_service(Arc::clone(&store), dim);

        let scope = DuplicateScope {
            set_doc_types: vec!["SetType".to_owned()],
            target_doc_types: vec!["TargetType".to_owned()],
            ..Default::default()
        };

        let result = svc
            .candidates_gather(
                domain,
                commit,
                &scope,
                5,
                1.0,
                1.0,
                false,
                false,
                &[],
            )
            .await
            .expect("candidates must succeed");

        let set_nbrs = result.set_to_target.get("doc/set_item");
        assert!(set_nbrs.is_some(), "set_item must have candidates");
        let candidate = set_nbrs.unwrap().iter().find(|c| c.id == "doc/target_item");
        assert!(
            candidate.is_some(),
            "set_item must have target_item as a candidate (close embeddings with permissive threshold)"
        );
        assert!(
            candidate.unwrap().distance > 0.0,
            "different doc-role embeddings must produce distance > 0, got {}",
            candidate.unwrap().distance
        );
    }

    // ─── io_fetch_result_embeddings (store-level): correctness guard ──────────

    /// io_fetch_result_embeddings correctly returns stored doc-role embeddings.
    #[tokio::test]
    async fn fetch_result_embeddings_returns_stored_vectors() {
        let dim = 8;
        let tmp = tempfile::tempdir().expect("temp dir");
        let store = LanceStore::new(tmp.path(), dim, 256 * 1024 * 1024, 128 * 1024 * 1024);

        let emb_a = make_embedding(dim, 1.0);
        let emb_b = make_embedding(dim, 5.0);

        let row_a = ChunkRow {
            doc_id: "doc/x".to_owned(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: emb_a.clone(),
            clustering_embedding: make_embedding(dim, 2.0),
            content: "x".to_owned(),
        };
        let row_b = ChunkRow {
            doc_id: "doc/y".to_owned(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: emb_b.clone(),
            clustering_embedding: make_embedding(dim, 3.0),
            content: "y".to_owned(),
        };

        let domain = "admin/fetch_emb";
        store
            .io_upsert_chunks(domain, "main", "doc/x", std::slice::from_ref(&row_a))
            .await
            .unwrap();
        store
            .io_upsert_chunks(domain, "main", "doc/y", std::slice::from_ref(&row_b))
            .await
            .unwrap();
        store
            .io_refresh_cached_dataset(domain, "main")
            .await
            .unwrap();
        let ds_arc = store.io_open_dataset(domain, "main").await.unwrap();
        let version = ds_arc.read().await.version().version;
        store
            .io_tag_commit(domain, "main", "c1", version)
            .await
            .unwrap();

        let result = store
            .io_fetch_result_embeddings(
                domain,
                "main",
                "c1",
                &["doc/x".to_owned(), "doc/y".to_owned()],
                &[],
                "embedding",
            )
            .await
            .expect("fetch must succeed");

        assert_eq!(result.len(), 2);
        assert!(
            crate::kernel::distance::vectors_equal(result.get("doc/x").unwrap(), &emb_a),
            "fetched embedding for doc/x must match what was stored"
        );
        assert!(
            crate::kernel::distance::vectors_equal(result.get("doc/y").unwrap(), &emb_b),
            "fetched embedding for doc/y must match what was stored"
        );
    }

    /// io_fetch_result_embeddings with clustering_embedding column.
    #[tokio::test]
    async fn fetch_result_clustering_embeddings_returns_stored_vectors() {
        let dim = 8;
        let tmp = tempfile::tempdir().expect("temp dir");
        let store = LanceStore::new(tmp.path(), dim, 256 * 1024 * 1024, 128 * 1024 * 1024);

        let qemb_a = make_embedding(dim, 7.0);
        let qemb_b = make_embedding(dim, 9.0);

        let row_a = ChunkRow {
            doc_id: "doc/p".to_owned(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: make_embedding(dim, 1.0),
            clustering_embedding: qemb_a.clone(),
            content: "p".to_owned(),
        };
        let row_b = ChunkRow {
            doc_id: "doc/q".to_owned(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: make_embedding(dim, 2.0),
            clustering_embedding: qemb_b.clone(),
            content: "q".to_owned(),
        };

        let domain = "admin/fetch_qemb";
        store
            .io_upsert_chunks(domain, "main", "doc/p", std::slice::from_ref(&row_a))
            .await
            .unwrap();
        store
            .io_upsert_chunks(domain, "main", "doc/q", std::slice::from_ref(&row_b))
            .await
            .unwrap();
        store
            .io_refresh_cached_dataset(domain, "main")
            .await
            .unwrap();
        let ds_arc = store.io_open_dataset(domain, "main").await.unwrap();
        let version = ds_arc.read().await.version().version;
        store
            .io_tag_commit(domain, "main", "c1", version)
            .await
            .unwrap();

        let result = store
            .io_fetch_result_embeddings(
                domain,
                "main",
                "c1",
                &["doc/p".to_owned(), "doc/q".to_owned()],
                &[],
                "clustering_embedding",
            )
            .await
            .expect("fetch must succeed");

        assert_eq!(result.len(), 2);
        assert!(
            crate::kernel::distance::vectors_equal(result.get("doc/p").unwrap(), &qemb_a),
            "fetched clustering_embedding for doc/p must match stored"
        );
        assert!(
            crate::kernel::distance::vectors_equal(result.get("doc/q").unwrap(), &qemb_b),
            "fetched clustering_embedding for doc/q must match stored"
        );
    }

    /// io_fetch_result_embeddings with doc_types filter returns only matching types.
    #[tokio::test]
    async fn fetch_result_embeddings_filters_by_doc_type() {
        let dim = 8;
        let tmp = tempfile::tempdir().expect("temp dir");
        let store = LanceStore::new(tmp.path(), dim, 256 * 1024 * 1024, 128 * 1024 * 1024);

        let row_a = ChunkRow {
            doc_id: "doc/a".to_owned(),
            doc_type: "Product".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: make_embedding(dim, 1.0),
            clustering_embedding: make_embedding(dim, 2.0),
            content: "a".to_owned(),
        };
        let row_b = ChunkRow {
            doc_id: "doc/b".to_owned(),
            doc_type: "Customer".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: make_embedding(dim, 5.0),
            clustering_embedding: make_embedding(dim, 6.0),
            content: "b".to_owned(),
        };

        let domain = "admin/filter_type";
        store
            .io_upsert_chunks(domain, "main", "doc/a", std::slice::from_ref(&row_a))
            .await
            .unwrap();
        store
            .io_upsert_chunks(domain, "main", "doc/b", std::slice::from_ref(&row_b))
            .await
            .unwrap();
        store
            .io_refresh_cached_dataset(domain, "main")
            .await
            .unwrap();
        let ds_arc = store.io_open_dataset(domain, "main").await.unwrap();
        let version = ds_arc.read().await.version().version;
        store
            .io_tag_commit(domain, "main", "c1", version)
            .await
            .unwrap();

        let result = store
            .io_fetch_result_embeddings(
                domain,
                "main",
                "c1",
                &[],
                &["Product".to_owned()],
                "embedding",
            )
            .await
            .expect("fetch must succeed");

        assert_eq!(result.len(), 1, "only Product docs should be returned");
        assert!(
            result.contains_key("doc/a"),
            "doc/a (Product) should be present"
        );
        assert!(
            !result.contains_key("doc/b"),
            "doc/b (Customer) should be filtered out"
        );
    }

    /// io_fetch_result_embeddings with both doc_ids and doc_types empty returns all embeddings.
    #[tokio::test]
    async fn fetch_result_embeddings_all_when_no_filters() {
        let dim = 8;
        let tmp = tempfile::tempdir().expect("temp dir");
        let store = LanceStore::new(tmp.path(), dim, 256 * 1024 * 1024, 128 * 1024 * 1024);

        let row_a = ChunkRow {
            doc_id: "doc/a".to_owned(),
            doc_type: "Product".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: make_embedding(dim, 1.0),
            clustering_embedding: make_embedding(dim, 2.0),
            content: "a".to_owned(),
        };
        let row_b = ChunkRow {
            doc_id: "doc/b".to_owned(),
            doc_type: "Customer".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: make_embedding(dim, 5.0),
            clustering_embedding: make_embedding(dim, 6.0),
            content: "b".to_owned(),
        };

        let domain = "admin/all_emb";
        store
            .io_upsert_chunks(domain, "main", "doc/a", std::slice::from_ref(&row_a))
            .await
            .unwrap();
        store
            .io_upsert_chunks(domain, "main", "doc/b", std::slice::from_ref(&row_b))
            .await
            .unwrap();
        store
            .io_refresh_cached_dataset(domain, "main")
            .await
            .unwrap();
        let ds_arc = store.io_open_dataset(domain, "main").await.unwrap();
        let version = ds_arc.read().await.version().version;
        store
            .io_tag_commit(domain, "main", "c1", version)
            .await
            .unwrap();

        let result = store
            .io_fetch_result_embeddings(
                domain,
                "main",
                "c1",
                &[],
                &[],
                "embedding",
            )
            .await
            .expect("fetch must succeed");

        assert_eq!(result.len(), 2, "all embeddings should be returned when no filters");
        assert!(result.contains_key("doc/a"));
        assert!(result.contains_key("doc/b"));
    }

    /// io_fetch_result_embeddings with both doc_ids and doc_types applies AND filter.
    #[tokio::test]
    async fn fetch_result_embeddings_combined_filter() {
        let dim = 8;
        let tmp = tempfile::tempdir().expect("temp dir");
        let store = LanceStore::new(tmp.path(), dim, 256 * 1024 * 1024, 128 * 1024 * 1024);

        let row_a = ChunkRow {
            doc_id: "doc/a".to_owned(),
            doc_type: "Product".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: make_embedding(dim, 1.0),
            clustering_embedding: make_embedding(dim, 2.0),
            content: "a".to_owned(),
        };
        let row_b = ChunkRow {
            doc_id: "doc/b".to_owned(),
            doc_type: "Product".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: make_embedding(dim, 5.0),
            clustering_embedding: make_embedding(dim, 6.0),
            content: "b".to_owned(),
        };
        let row_c = ChunkRow {
            doc_id: "doc/c".to_owned(),
            doc_type: "Customer".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: make_embedding(dim, 9.0),
            clustering_embedding: make_embedding(dim, 10.0),
            content: "c".to_owned(),
        };

        let domain = "admin/combined_filter";
        store
            .io_upsert_chunks(domain, "main", "doc/a", std::slice::from_ref(&row_a))
            .await
            .unwrap();
        store
            .io_upsert_chunks(domain, "main", "doc/b", std::slice::from_ref(&row_b))
            .await
            .unwrap();
        store
            .io_upsert_chunks(domain, "main", "doc/c", std::slice::from_ref(&row_c))
            .await
            .unwrap();
        store
            .io_refresh_cached_dataset(domain, "main")
            .await
            .unwrap();
        let ds_arc = store.io_open_dataset(domain, "main").await.unwrap();
        let version = ds_arc.read().await.version().version;
        store
            .io_tag_commit(domain, "main", "c1", version)
            .await
            .unwrap();

        // Filter: doc_type=Product AND doc_id IN (doc/a, doc/b) → doc/a and doc/b
        // doc/c is excluded: it's Customer AND not in doc_ids list
        let result = store
            .io_fetch_result_embeddings(
                domain,
                "main",
                "c1",
                &["doc/a".to_owned(), "doc/b".to_owned()],
                &["Product".to_owned()],
                "embedding",
            )
            .await
            .expect("fetch must succeed");

        assert_eq!(result.len(), 2, "AND filter should return doc/a and doc/b (both Product and in doc_ids)");
        assert!(result.contains_key("doc/a"));
        assert!(result.contains_key("doc/b"));
        assert!(!result.contains_key("doc/c"), "doc/c is Customer and not in doc_ids");
    }
}

#[cfg(test)]
mod tests_concurrent_search {
    use super::*;
    use crate::store::lance::ChunkRow;
    use std::path::Path;
    use std::time::Duration;

    #[allow(dead_code)]
    static DISABLED_CACHE: std::sync::LazyLock<crate::embed::cache::EmbedCache> =
        std::sync::LazyLock::new(crate::embed::cache::EmbedCache::disabled);

    #[allow(dead_code)]
    fn test_tokenizer() -> Tokenizer {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("assets")
            .join("tokenizer.json.bz2");
        crate::chunk::io_load_tokenizer(&path).expect("test tokenizer must load")
    }

    #[allow(dead_code)]
    fn dummy_provider() -> Provider {
        Provider::OpenAiCompatible {
            base_url: "http://127.0.0.1:0/never-called".to_owned(),
            model: "test-noop".to_owned(),
            dim: 8,
        }
    }

    #[allow(dead_code)]
    fn make_row(doc_id: &str, content: &str, emb: Vec<f32>) -> ChunkRow {
        ChunkRow {
            doc_id: doc_id.to_owned(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 5,
            embedding: emb,
            clustering_embedding: vec![0.1; 8],
            content: content.to_owned(),
        }
    }

    /// Search must return results while the pipeline lock is held by an
    /// in-flight indexing task. The search operates on an already-tagged
    /// commit snapshot and must not block on the pipeline lock.
    #[tokio::test]
    async fn search_completes_while_pipeline_lock_held() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let store = LanceStore::new(tmp.path(), 8, 256 * 1024 * 1024, 128 * 1024 * 1024);
        let domain = "admin/concurrent_search";
        let branch = "main";

        // Seed: insert two docs and tag as c0.
        let row_a = make_row("doc/a", "king of the hill", vec![0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let row_b = make_row("doc/b", "queen of hearts", vec![0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let v0 = store
            .io_upsert_chunks(domain, branch, "doc/a", std::slice::from_ref(&row_a))
            .await
            .expect("seed upsert a");
        store
            .io_upsert_chunks(domain, branch, "doc/b", std::slice::from_ref(&row_b))
            .await
            .expect("seed upsert b");
        store.io_tag_commit(domain, branch, "c0", v0).await.expect("tag c0");
        store.update_last_indexed(domain, branch, "c0", v0).await;

        // Acquire the pipeline lock — simulating an in-flight indexing task.
        let _pipeline_guard = store.acquire_pipeline_lock(domain, branch).await;

        // While the pipeline lock is held, perform a search on the tagged
        // commit c0. This must NOT block on the pipeline lock.
        let query = SearchQuery {
            query_embedding: vec![0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            query_text: "king".to_owned(),
            mode: SearchMode::Vector,
            start: 0,
            count: 10,
            doc_type_filter: vec![],
            doc_id_filter: vec![],
            snippet: false,
        };

        // Wrap in timeout — if search blocks on the pipeline lock, this fails.
        let hits = tokio::time::timeout(
            Duration::from_secs(5),
            store.io_search(domain, branch, "c0", &query),
        )
        .await
        .expect("search must not block while pipeline lock is held")
        .expect("search must succeed");

        // Should find doc/a (closest to the query embedding).
        assert!(!hits.is_empty(), "search must return hits");
        assert_eq!(hits[0].doc_id, "doc/a", "closest hit should be doc/a");

        // Release the pipeline lock.
        drop(_pipeline_guard);
    }

    /// Search must return results while a full indexing pipeline is running
    /// with active operations in the channel. This tests real concurrency,
    /// not just lock acquisition.
    #[tokio::test]
    async fn search_completes_during_active_indexing() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let store = LanceStore::new(tmp.path(), 8, 256 * 1024 * 1024, 128 * 1024 * 1024);
        let chunk_params = ChunkParams { max_tokens: 512, overlap: 64 };
        let _provider = dummy_provider();
        let _http_client = reqwest::Client::new();
        let domain = "admin/concurrent_active";
        let branch = "main";

        // Seed: insert one doc and tag as c0.
        let row_seed = make_row("doc/seed", "king of the hill", vec![0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let v0 = store
            .io_upsert_chunks(domain, branch, "doc/seed", std::slice::from_ref(&row_seed))
            .await
            .expect("seed upsert");
        store.io_tag_commit(domain, branch, "c0", v0).await.expect("tag c0");
        store.update_last_indexed(domain, branch, "c0", v0).await;

        // Start a pipeline with a slow operation channel — we'll keep the
        // channel open with a delay so the pipeline is actively running
        // (holding the pipeline lock) when we search.
        let (tx, rx) = mpsc::channel::<Operation>(16);
        let cancel_token = tokio_util::sync::CancellationToken::new();

        let store_arc = Arc::new(store);
        let store_for_pipeline = Arc::clone(&store_arc);
        let tokenizer_clone = test_tokenizer();
        let chunk_params_clone = chunk_params.clone();
        let provider_clone = _provider.clone();
        let http_client_clone = _http_client.clone();
        let cancel_for_pipeline = cancel_token.clone();

        let pipeline_handle = tokio::spawn(async move {
            let ctx = PipelineCtx {
                store: &store_for_pipeline,
                tokenizer: &tokenizer_clone,
                chunk_params: &chunk_params_clone,
                provider: &provider_clone,
                http_client: &http_client_clone,
                embed_cache: &DISABLED_CACHE,
                domain,
                branch,
                embed_batch_size: 32,
            };
            let _ = io_run_index_pipeline_stream(
                &ctx, "c1", rx, None, &cancel_for_pipeline,
            ).await;
        });

        // Give the pipeline a moment to acquire the pipeline lock.
        tokio::time::sleep(Duration::from_millis(50)).await;

        // While the pipeline is running, search on the tagged commit c0.
        let query = SearchQuery {
            query_embedding: vec![0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            query_text: "king".to_owned(),
            mode: SearchMode::Vector,
            start: 0,
            count: 10,
            doc_type_filter: vec![],
            doc_id_filter: vec![],
            snippet: false,
        };

        let hits = tokio::time::timeout(
            Duration::from_secs(10),
            store_arc.io_search(domain, branch, "c0", &query),
        )
        .await
        .expect("search must not block during active indexing")
        .expect("search must succeed");

        assert!(!hits.is_empty(), "search must return hits during indexing");
        assert_eq!(hits[0].doc_id, "doc/seed", "closest hit should be doc/seed");

        // Clean up: cancel the pipeline and wait for it to finish.
        cancel_token.cancel();
        drop(tx);
        let _ = tokio::time::timeout(Duration::from_secs(5), pipeline_handle)
            .await
            .expect("pipeline must finish after cancel");
    }

    /// FTS search must also work concurrently with indexing.
    #[tokio::test]
    async fn fts_search_completes_while_pipeline_lock_held() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let store = LanceStore::new(tmp.path(), 8, 256 * 1024 * 1024, 128 * 1024 * 1024);
        let domain = "admin/concurrent_fts";
        let branch = "main";

        // Seed: insert a doc with searchable text.
        let row = make_row("doc/a", "the king is in the castle", vec![0.1; 8]);
        let _v0 = store
            .io_upsert_chunks(domain, branch, "doc/a", std::slice::from_ref(&row))
            .await
            .expect("seed upsert");

        // Create the FTS inverted index BEFORE tagging so the tagged
        // version includes the index.
        store
            .io_ensure_fts_index_created(domain, branch)
            .await
            .expect("FTS index creation");

        // Now tag the version that includes the FTS index.
        // Re-read the head version since FTS index creation may have advanced it.
        let v_after_fts = {
            let ds_arc = store.io_open_dataset(domain, branch).await.expect("open dataset");
            let ds = ds_arc.read().await;
            ds.version().version
        };
        store.io_tag_commit(domain, branch, "c0", v_after_fts).await.expect("tag c0");
        store.update_last_indexed(domain, branch, "c0", v_after_fts).await;

        // Acquire the pipeline lock — simulating an in-flight indexing task.
        let _pipeline_guard = store.acquire_pipeline_lock(domain, branch).await;

        // FTS search while pipeline lock is held.
        let query = SearchQuery {
            query_embedding: vec![],
            query_text: "king".to_owned(),
            mode: SearchMode::Fts,
            start: 0,
            count: 10,
            doc_type_filter: vec![],
            doc_id_filter: vec![],
            snippet: true,
        };

        let hits = tokio::time::timeout(
            Duration::from_secs(5),
            store.io_search(domain, branch, "c0", &query),
        )
        .await
        .expect("FTS search must not block while pipeline lock is held")
        .expect("FTS search must succeed");

        assert!(!hits.is_empty(), "FTS search must return hits");
        assert_eq!(hits[0].doc_id, "doc/a", "FTS hit should be doc/a");

        drop(_pipeline_guard);
    }

    /// Search must return results when the latest indexed commit has indices.
    /// This verifies that io_search on a properly indexed commit returns results,
    /// which is the core requirement for parallel search and indexing.
    #[tokio::test]
    async fn search_returns_results_when_snapshot_lacks_indices() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let store = LanceStore::new(tmp.path(), 8, 256 * 1024 * 1024, 128 * 1024 * 1024);
        let domain = "admin/index_fallback";
        let branch = "main";

        // Seed: insert a doc.
        let row = make_row("doc/a", "the king is in the castle", vec![0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let _v0 = store
            .io_upsert_chunks(domain, branch, "doc/a", std::slice::from_ref(&row))
            .await
            .expect("seed upsert");

        // Create the FTS index BEFORE tagging so the tagged version includes it.
        store
            .io_ensure_fts_index_created(domain, branch)
            .await
            .expect("FTS index creation");

        // Tag at the version that includes the FTS index.
        let v_after_fts = {
            let ds_arc = store.io_open_dataset(domain, branch).await.expect("open dataset");
            let ds = ds_arc.read().await;
            ds.version().version
        };
        store.io_tag_commit(domain, branch, "c0", v_after_fts).await.expect("tag c0");
        store.update_last_indexed(domain, branch, "c0", v_after_fts).await;

        // Add more data after tagging (simulates pipeline advancing the head).
        let row2 = make_row("doc/b", "queen of hearts", vec![0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        store
            .io_upsert_chunks(domain, branch, "doc/b", std::slice::from_ref(&row2))
            .await
            .expect("upsert b");

        // Search FTS on the tagged commit c0. Since c0 has the FTS index,
        // the search should return results quickly.
        let query = SearchQuery {
            query_embedding: vec![],
            query_text: "king".to_owned(),
            mode: SearchMode::Fts,
            start: 0,
            count: 10,
            doc_type_filter: vec![],
            doc_id_filter: vec![],
            snippet: true,
        };

        let hits = tokio::time::timeout(
            Duration::from_secs(10),
            store.io_search(domain, branch, "c0", &query),
        )
        .await
        .expect("FTS search must not hang when indices are present")
        .expect("FTS search must succeed");

        assert!(
            !hits.is_empty(),
            "FTS search must return results when indices are present"
        );
        assert_eq!(hits[0].doc_id, "doc/a", "FTS hit should be doc/a");
    }

    /// Vector search must return results when the latest indexed commit has
    /// the IVF vector index. This verifies that vector search works on a
    /// properly indexed commit, which is the core requirement for parallel
    /// search and indexing.
    #[tokio::test]
    async fn vector_search_returns_results_when_snapshot_lacks_indices() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let store = LanceStore::new(tmp.path(), 8, 256 * 1024 * 1024, 128 * 1024 * 1024);
        let domain = "admin/vector_fallback";
        let branch = "main";

        // Seed: insert enough docs to meet IVF minimum training threshold.
        let row = make_row("doc/a", "king of the hill", vec![0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let _v0 = store
            .io_upsert_chunks(domain, branch, "doc/a", std::slice::from_ref(&row))
            .await
            .expect("seed upsert");

        for i in 0..300 {
            let emb = vec![0.1 * (i as f32 / 300.0), 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            let r = make_row(&format!("doc/b{}", i), &format!("doc number {}", i), emb);
            store
                .io_upsert_chunks(domain, branch, &format!("doc/b{}", i), std::slice::from_ref(&r))
                .await
                .expect("upsert");
        }

        // Create vector index on the current head (enough rows now).
        {
            let ds_arc = store.io_open_dataset(domain, branch).await.expect("open dataset");
            let mut ds = ds_arc.write().await;
            crate::store::vector_index::io_ensure_vector_index(&mut ds, store.vector_index_config(), false)
                .await
                .expect("vector index creation");
        }

        // Tag at the version that includes the vector index.
        let v_after_vec = {
            let ds_arc = store.io_open_dataset(domain, branch).await.expect("open dataset");
            let ds = ds_arc.read().await;
            ds.version().version
        };
        store.io_tag_commit(domain, branch, "c0", v_after_vec).await.expect("tag c0");
        store.update_last_indexed(domain, branch, "c0", v_after_vec).await;

        // Add more data after tagging (simulates pipeline advancing the head).
        let row2 = make_row("doc/c", "queen of hearts", vec![0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        store
            .io_upsert_chunks(domain, branch, "doc/c", std::slice::from_ref(&row2))
            .await
            .expect("upsert c");

        // Vector search on the tagged commit c0 (has vector index).
        let query = SearchQuery {
            query_embedding: vec![0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            query_text: "king".to_owned(),
            mode: SearchMode::Vector,
            start: 0,
            count: 10,
            doc_type_filter: vec![],
            doc_id_filter: vec![],
            snippet: false,
        };

        let hits = tokio::time::timeout(
            Duration::from_secs(10),
            store.io_search(domain, branch, "c0", &query),
        )
        .await
        .expect("vector search must not hang when indices are present")
        .expect("vector search must succeed");

        assert!(
            !hits.is_empty(),
            "vector search must return results when indices are present"
        );
    }
}

#[cfg(test)]
mod tests_index_trigger {
    use super::*;

    #[test]
    fn should_create_index_at_3rd_commit() {
        assert_eq!(should_create_index(2), IndexAction::Create);
    }

    #[test]
    fn should_create_index_at_6th_commit() {
        assert_eq!(should_create_index(5), IndexAction::Create);
    }

    #[test]
    fn should_create_index_at_9th_commit() {
        assert_eq!(should_create_index(8), IndexAction::Create);
    }

    #[test]
    fn should_skip_index_for_non_3rd() {
        assert_eq!(should_create_index(0), IndexAction::Skip);
        assert_eq!(should_create_index(1), IndexAction::Skip);
        assert_eq!(should_create_index(3), IndexAction::Skip);
        assert_eq!(should_create_index(4), IndexAction::Skip);
        assert_eq!(should_create_index(6), IndexAction::Skip);
        assert_eq!(should_create_index(7), IndexAction::Skip);
    }
}
