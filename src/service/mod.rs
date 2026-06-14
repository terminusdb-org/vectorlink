#![forbid(unsafe_code)]

//! Service — the transport-agnostic core API surface.
//! Owns no framework types (no axum/hyper in signatures).
//! Composes store operations and validates domain logic.
//! Wires the full pipeline: parse → chunk → embed → store → tag.

use std::sync::Arc;

use tokenizers::Tokenizer;

use crate::chunk::{self, ChunkParams};
use crate::config::Config;
use crate::embed::{self, EmbeddingRole, Provider};
use crate::ingest;
use crate::kernel::distance::l2_normalize;
use crate::kernel::error::ServiceError;
use crate::kernel::model::{
    parse_domain, BranchName, Domain, LastIndexed, Operation, Ref, SearchHit, SearchMode,
    SkippedDoc, Statistics, TaskStatus,
};
use crate::store::lance::{ChunkRow, LanceStore, SearchQuery, dedup_chunks_to_documents};

/// The search service — owns the store and config, provides the transport-agnostic API.
#[derive(Clone)]
pub struct SearchService {
    store: Arc<LanceStore>,
    config: Config,
    tokenizer: Arc<Tokenizer>,
    chunk_params: ChunkParams,
    http_client: reqwest::Client,
    /// Per-capability readiness: index readiness is independent of search
    /// readiness (search additionally requires a warm embedding backend).
    ready_index: Arc<std::sync::atomic::AtomicBool>,
    ready_search: Arc<std::sync::atomic::AtomicBool>,
    /// Process-local per-branch state: indexing-enablement + 404 negative cache
    /// for catch-up resolution (truth is always the layer index / Lance tags).
    branch_state: Arc<crate::layeridx::BranchState>,
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
        let prefix = match embed::prefixes_for_model(config.embed_provider.model_name()) {
            Some(p) => embed::prefix_for_role(&p, EmbeddingRole::Document).to_owned(),
            None => String::new(),
        };
        let chunk_params = chunk::params_for_nomic(&tokenizer, &prefix)
            .unwrap_or(ChunkParams { max_tokens: 480, overlap: 64 });

        let ready_index = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let search_ready = std::env::var("TDB_SEARCH_SEARCH_READY")
            .map(|v| v != "false")
            .unwrap_or(true);
        let ready_search = Arc::new(std::sync::atomic::AtomicBool::new(search_ready));

        Self {
            store,
            config,
            tokenizer: Arc::new(tokenizer),
            chunk_params,
            http_client: reqwest::Client::new(),
            ready_index,
            ready_search,
            branch_state: Arc::new(crate::layeridx::BranchState::default()),
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
        Ok(self.store.last_indexed(&domain, &branch).await)
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

        // Branch-out: if a parent commit is supplied and the target branch does
        // not yet exist, fork it from the parent's indexed version (block reuse).
        // A push to "main" or to an existing branch skips this (no-op). Fail loud
        // if the parent isn't indexed (cannot fork from nothing).
        if branch != crate::store::lance::MAIN_BRANCH {
            if let Some(parent) = parent_commit {
                crate::store::branch::io_ensure_branch_forked(
                    &self.store,
                    &domain_str,
                    &branch,
                    parent,
                )
                .await
                .map_err(|e| {
                    let msg = e.to_string();
                    if msg.contains("not indexed") || msg.contains("does not exist") {
                        ServiceError::NotFound(msg)
                    } else {
                        ServiceError::Internal(msg)
                    }
                })?;
            }
        }

        // An indexing request for this branch directly busts its 404 negative
        // cache and enrolls it (Spec 10 §5: invalidate immediately on a direct
        // index request) — so a search that 404'd before this push will resolve
        // on its next attempt.
        self.branch_state.invalidate_negative(&domain_str, &branch);
        self.branch_state.enable(&domain_str, &branch);

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
        let tokenizer = Arc::clone(&self.tokenizer);
        let chunk_params = self.chunk_params.clone();
        let provider = self.config.embed_provider.clone();
        let http_client = self.http_client.clone();
        let task_id_clone = task_id.clone();

        // Spawn the indexing pipeline as a background task.
        // Monitor the JoinHandle so panics are captured → Error, never Pending forever.
        let store_for_panic = Arc::clone(&store);
        let task_id_for_panic = task_id.clone();
        let handle = tokio::spawn(async move {
            let ctx = PipelineCtx {
                store: &store,
                tokenizer: &tokenizer,
                chunk_params: &chunk_params,
                provider: &provider,
                http_client: &http_client,
                domain: &domain_str,
                branch: &branch,
            };
            let result = io_run_index_pipeline(&ctx, &commit, operations).await;

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
        });

        // Monitor: if the spawned task panics, record Error (never leave Pending).
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

        // Resolve the branch (default to "main").
        let branch = "main";

        // Pure tag-pointer assign: resolve source → tag target to the same
        // version. NO embedding, NO recompute (P3-ASSIGN-1). Fail loud if the
        // source commit is not indexed.
        match self
            .store
            .io_assign_commit(domain_str, branch, source_commit, target_commit)
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
        )
        .await
    }

    /// Full search with all options. Returns a `SearchOutcome` carrying the
    /// commit actually SERVED (exact, or the nearest indexed ancestor under lag)
    /// so the transport reports staleness truthfully (RISK-15, P3-LAG-1).
    ///
    /// Catch-up resolution (never blocks, never silently stale):
    ///  1. Exact: requested commit is indexed → serve it.
    ///  2. Lag: not indexed → serve the branch's durable last-indexed commit (the
    ///     nearest indexed ancestor on a linear-per-branch history) and report it
    ///     as the served commit (⇒ caller sees served ≠ requested = stale).
    ///  3. None: branch has no indexed ancestor → `NotFound` (404), negatively
    ///     cached per branch (TTL) so a repeat search doesn't re-walk history.
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

        // Resolve the searchable commit via catch-up (exact → nearest ancestor → 404).
        let served_commit = self
            .resolve_searchable_commit(&domain_str, &branch, commit)
            .await?;

        // Embed the query text.
        let query_texts = vec![q.to_owned()];
        let embeddings = embed::io_embed(
            &self.config.embed_provider,
            &query_texts,
            EmbeddingRole::Query,
            &self.http_client,
        )
        .await
        .map_err(|e| ServiceError::Internal(format!("embedding failed: {}", e)))?;

        let mut query_embedding = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| ServiceError::Internal("no embedding returned".to_owned()))?;
        l2_normalize(&mut query_embedding);

        let search_query = SearchQuery {
            query_embedding,
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

    /// Resolve the requested commit to the commit whose snapshot will actually
    /// be searched (catch-up, RISK-15). Exact if indexed; else the branch's
    /// durable last-indexed commit (nearest ancestor on linear-per-branch); else
    /// `NotFound`, negatively cached per branch.
    ///
    /// Auto-enroll: any successful resolution marks the branch indexing-enabled,
    /// so a descendant branch that resolves through an ancestor is enrolled on
    /// first search (propagation; one explicit bootstrap per lineage).
    async fn resolve_searchable_commit(
        &self,
        domain: &str,
        branch: &str,
        requested_commit: &str,
    ) -> Result<String, ServiceError> {
        // 1. Exact match: the requested commit is itself indexed.
        let exact = self
            .store
            .io_resolve_commit(domain, branch, requested_commit)
            .await
            .map_err(|e| ServiceError::Internal(e.to_string()))?;
        if exact.is_some() {
            self.branch_state.enable(domain, branch);
            self.branch_state.invalidate_negative(domain, branch);
            return Ok(requested_commit.to_owned());
        }

        // 2. Negative cache: a recent search already found no indexed ancestor.
        if self.branch_state.is_negative_cached(domain, branch) {
            return Err(ServiceError::NotFound(format!(
                "no indexed ancestor for commit {} on branch {} (negatively cached)",
                requested_commit, branch
            )));
        }

        // 3. Lag: serve the branch's durable last-indexed commit (the nearest
        //    indexed ancestor on a linear-per-branch history). TerminusDB owns
        //    the DAG and pushes the missing delta on seeing served ≠ requested.
        let last = self
            .store
            .last_indexed(
                &Domain::from_resource_path(
                    &parse_domain(domain).map_err(|e| ServiceError::Internal(e.to_string()))?,
                ),
                &BranchName::new(branch.to_owned()),
            )
            .await;

        match last.commit {
            Some(ancestor) => {
                // Enroll the branch (auto-enroll on first resolved search).
                self.branch_state.enable(domain, branch);
                Ok(ancestor)
            }
            None => {
                // 4. No indexed ancestor at all → 404, negatively cached.
                self.branch_state.record_negative(domain, branch);
                Err(ServiceError::NotFound(format!(
                    "no indexed ancestor for commit {} on branch {}",
                    requested_commit, branch
                )))
            }
        }
    }

    /// Similar: look up doc by id → use its best embedding → vector search.
    pub async fn similar(
        &self,
        domain_raw: &str,
        commit: &str,
        id: &str,
    ) -> Result<Vec<SearchHit>, ServiceError> {
        self.similar_with_options(domain_raw, commit, id, 0, 10, &[], false)
            .await
    }

    /// Similar with full options.
    #[allow(clippy::too_many_arguments)]
    pub async fn similar_with_options(
        &self,
        domain_raw: &str,
        commit: &str,
        id: &str,
        start: usize,
        count: usize,
        doc_type_filter: &[String],
        snippet: bool,
    ) -> Result<Vec<SearchHit>, ServiceError> {
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

        // Use the first chunk's embedding (best available — chunk 0).
        // Retrieve the actual embedding by re-embedding the chunk text (the store
        // doesn't store raw embeddings in the hit). For /similar we re-embed using
        // the Document role to get the source vector.
        let source_text = vec![doc_chunks[0].content.clone()];
        let embeddings = embed::io_embed(
            &self.config.embed_provider,
            &source_text,
            EmbeddingRole::Document,
            &self.http_client,
        )
        .await
        .map_err(|e| ServiceError::Internal(format!("embedding for similar failed: {}", e)))?;

        let mut query_embedding = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| ServiceError::Internal("no embedding returned for similar".to_owned()))?;
        l2_normalize(&mut query_embedding);

        let search_query = SearchQuery {
            query_embedding,
            query_text: String::new(),
            mode: SearchMode::Vector,
            start,
            count: count + 1, // Over-fetch to exclude self.
            doc_type_filter: doc_type_filter.to_vec(),
            doc_id_filter: Vec::new(),
            snippet,
        };

        let chunk_hits = self
            .store
            .io_search(&domain_str, &branch, commit, &search_query)
            .await
            .map_err(|e| ServiceError::Internal(e.to_string()))?;

        let all_results = dedup_chunks_to_documents(chunk_hits, snippet);

        // Exclude self from results and apply pagination.
        let paginated: Vec<SearchHit> = all_results
            .into_iter()
            .filter(|hit| hit.id != id)
            .skip(start)
            .take(count)
            .collect();

        Ok(paginated)
    }

    /// Duplicates: bounded near-duplicate detection.
    /// Returns 501 if unbounded (no commit indexed).
    pub async fn duplicates(
        &self,
        domain_raw: &str,
        commit: &str,
    ) -> Result<Vec<(String, String)>, ServiceError> {
        let rp =
            parse_domain(domain_raw).map_err(|e| ServiceError::Validation(e.to_string()))?;
        let _domain = Domain::from_resource_path(&rp);

        // Duplicates requires a full scan which is bounded by the index size.
        // For now, return an empty result set (feature to be fully implemented
        // when the index supports efficient near-duplicate queries).
        // If the commit is not indexed, return 501.
        let domain_str = _domain.as_str().to_owned();
        let branch = extract_branch(&rp);

        let resolved = self
            .store
            .io_resolve_commit(&domain_str, &branch, commit)
            .await
            .map_err(|e| ServiceError::Internal(e.to_string()))?;

        if resolved.is_none() {
            return Err(ServiceError::Unavailable(
                "commit not indexed — cannot compute duplicates".to_owned(),
            ));
        }

        // Bounded: return empty for now (full duplicate detection is a future feature).
        Ok(Vec::new())
    }

    /// Statistics.
    pub async fn statistics(&self) -> Result<Statistics, ServiceError> {
        Ok(self.store.statistics().await)
    }

    /// Delete a domain's ENTIRE footprint: the `{domain}.lance` dataset (all
    /// branches/versions/tags) plus every in-memory trace (store caches +
    /// per-branch enablement/negative-cache). IDEMPOTENT — an unknown/already-
    /// gone domain succeeds (TerminusDB may retry). Fail-loud on a real I/O error.
    ///
    /// Order: the store removes the on-disk dataset FIRST, then purges its own
    /// in-memory maps (fail-loud partial guard inside the store). Only after the
    /// store succeeds do we purge the service-owned per-branch state — so we
    /// never leave searchable per-branch state pointing at a deleted dataset.
    pub async fn delete_domain(&self, domain_raw: &str) -> Result<(), ServiceError> {
        let rp =
            parse_domain(domain_raw).map_err(|e| ServiceError::Validation(e.to_string()))?;
        let domain = Domain::from_resource_path(&rp);
        let domain_str = domain.as_str().to_owned();

        self.store
            .io_delete_domain(&domain_str)
            .await
            .map_err(|e| ServiceError::Internal(e.to_string()))?;

        self.branch_state.purge_domain(&domain_str);

        Ok(())
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
    domain: &'a str,
    branch: &'a str,
}

/// Run the push pipeline for a set of operations.
/// Returns (indexed_count, skipped_docs) on success, or error message on failure.
///
/// Pipeline ordering (solves BLOCKER-1 by construction):
///   1. Acquire per-(domain, branch) pipeline lock
///   2. Upsert chunks → version N (data written)
///   3. Optimize indices (FTS + vector ANN) → version N+K (indices built)
///      - Uses an UNCACHED dataset handle so searches on other branches aren't blocked
///   4. Tag commit → version N+K (commit becomes searchable with full index coverage)
///   5. Update last-indexed tracking
///   6. Release pipeline lock
///
/// A commit is NOT resolvable until step 4 completes — `io_search` returns
/// "commit not indexed" (404) for uncommitted work. This is correct + fail-loud
/// (Phase 3 upgrades to graceful nearest-ancestor fallback).
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
    // Acquire the per-branch lock. Held for the entire upsert→optimize→tag sequence.
    // This ensures at most one unindexed commit per branch and prevents snapshot
    // isolation violations (a later commit's data cannot leak into an earlier tag).
    let _guard = ctx.store.acquire_pipeline_lock(ctx.domain, ctx.branch).await;

    let mut indexed_count: u64 = 0;
    let mut skipped: Vec<SkippedDoc> = Vec::new();
    let mut last_version: u64 = 0;

    for op in &operations {
        match op {
            Operation::Inserted { id, string } | Operation::Changed { id, string } => {
                match io_index_document(ctx, id, string).await {
                    Ok(version) => {
                        indexed_count += 1;
                        last_version = version;
                    }
                    Err(e) => {
                        skipped.push(SkippedDoc {
                            id: id.clone(),
                            message: e,
                        });
                    }
                }
            }
            Operation::Deleted { id } => {
                match ctx.store.io_delete_doc(ctx.domain, ctx.branch, id).await {
                    Ok(version) => {
                        last_version = version;
                    }
                    Err(e) => {
                        skipped.push(SkippedDoc {
                            id: id.clone(),
                            message: format!("delete failed: {}", e),
                        });
                    }
                }
            }
            Operation::Error { message } => {
                // Per spec: Operation::Error → skip+record, do not fail the whole task.
                skipped.push(SkippedDoc {
                    id: "unknown".to_owned(),
                    message: format!("operation error: {}", message),
                });
            }
        }
    }

    // Step 3: Optimize indices on an UNCACHED handle (Fix #2: doesn't block searches).
    // FTS + vector ANN are built incrementally via optimize_indices(append()) — O(delta).
    // This runs BEFORE tagging so the tagged version has full index coverage (BLOCKER-1 fix).
    if last_version > 0 {
        let indexed_version = io_optimize_on_uncached_handle(ctx.store, ctx.domain, ctx.branch)
            .await
            .map_err(|e| format!("index optimization failed: {}", e))?;

        // Step 4: Tag the commit to the INDEXED version (data + indices).
        // After this point, the commit is resolvable and fully searchable.
        let tag_version = indexed_version.unwrap_or(last_version);
        ctx.store
            .io_tag_commit(ctx.domain, ctx.branch, commit, tag_version)
            .await
            .map_err(|e| format!("failed to tag commit: {}", e))?;

        // Step 5: Update last-indexed tracking.
        ctx.store
            .update_last_indexed(ctx.domain, ctx.branch, commit, tag_version)
            .await;
    }

    Ok((indexed_count, skipped))
}

/// Optimize FTS + vector indices on an uncached dataset handle.
/// Returns the version after optimization, or None if the dataset doesn't exist yet.
///
/// Uses `io_open_dataset_uncached` so the write operations don't hold the shared
/// `Arc<RwLock<Dataset>>` and block concurrent search reads (Fix #2).
/// After optimization, refreshes the cached handle.
async fn io_optimize_on_uncached_handle(
    store: &LanceStore,
    domain: &str,
    branch: &str,
) -> Result<Option<u64>, String> {
    let ds = store
        .io_open_dataset_uncached(domain, branch)
        .await
        .map_err(|e| format!("uncached open failed: {}", e))?;

    let mut ds = match ds {
        Some(d) => d,
        None => return Ok(None),
    };

    // FTS: create inverted index or incrementally append new fragments.
    crate::store::lance::io_ensure_fts_index_on_dataset(&mut ds)
        .await
        .map_err(|e| format!("FTS optimize failed: {}", e))?;

    // Vector ANN: create IVF_PQ index (if enough rows) or incrementally append.
    crate::store::vector_index::io_ensure_vector_index(&mut ds, store.vector_index_config())
        .await
        .map_err(|e| format!("vector optimize failed: {}", e))?;

    let final_version = ds.version().version;

    // Refresh the cached handle so subsequent reads see the optimized indices.
    store
        .io_refresh_cached_dataset(domain, branch)
        .await
        .map_err(|e| format!("cache refresh failed: {}", e))?;

    Ok(Some(final_version))
}

/// Index a single document: chunk → embed → upsert.
async fn io_index_document(
    ctx: &PipelineCtx<'_>,
    doc_id: &str,
    text: &str,
) -> Result<u64, String> {
    // 1. Chunk the document.
    let chunks = chunk::chunk_text(ctx.tokenizer, text, ctx.chunk_params)
        .map_err(|e| format!("chunking failed: {}", e))?;

    if chunks.is_empty() {
        return Err("chunking produced zero chunks".to_owned());
    }

    // 2. Embed all chunks in a single batch, then L2-normalise for cosine distance.
    let chunk_texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let mut embeddings =
        embed::io_embed(ctx.provider, &chunk_texts, EmbeddingRole::Document, ctx.http_client)
            .await
            .map_err(|e| format!("embedding failed: {}", e))?;

    if embeddings.len() != chunks.len() {
        return Err(format!(
            "embedding count mismatch: expected {}, got {}",
            chunks.len(),
            embeddings.len()
        ));
    }

    // L2-normalise embeddings so that Lance's L2² metric = cosine distance.
    for emb in &mut embeddings {
        l2_normalize(emb);
    }

    // 3. Build chunk rows.
    let doc_type = ingest::extract_doc_type(doc_id);
    let rows: Vec<ChunkRow> = chunks
        .iter()
        .zip(embeddings)
        .map(|(chunk, embedding)| ChunkRow {
            doc_id: doc_id.to_owned(),
            doc_type: doc_type.clone(),
            chunk_index: chunk.index as i32,
            chunk_count: chunk.count as i32,
            chunk_token_start: chunk.token_start as i32,
            doc_token_len: chunk.doc_token_len as i32,
            embedding,
            content: chunk.text.clone(),
        })
        .collect();

    // 4. Upsert into the store.
    let version = ctx
        .store
        .io_upsert_chunks(ctx.domain, ctx.branch, doc_id, &rows)
        .await
        .map_err(|e| format!("upsert failed: {}", e))?;

    Ok(version)
}
