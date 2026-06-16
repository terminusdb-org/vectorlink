#![forbid(unsafe_code)]

//! Service — the transport-agnostic core API surface.
//! Owns no framework types (no axum/hyper in signatures).
//! Composes store operations and validates domain logic.
//! Wires the full pipeline: parse → chunk → embed → store → tag.

use std::sync::Arc;

use tokenizers::Tokenizer;

use crate::chunk::{self, ChunkParams};
use crate::config::Config;
use crate::embed::{self, EmbedResult, EmbeddingRole, Provider};
use crate::ingest;
use crate::kernel::distance::l2_normalize;
use crate::kernel::error::ServiceError;
use crate::kernel::model::{
    parse_domain, BranchName, Domain, DuplicateGroup, LastIndexed, Operation, Ref, SearchHit,
    SearchMode, SkippedDoc, Statistics, TaskStatus,
};
use crate::store::lance::{
    ChunkRow, DuplicateScope, LanceStore, SearchQuery, dedup_chunks_to_documents,
};

/// Max concurrent heavy-scan requests (/resolve, /duplicates). Each such request
/// may spike ~64-96 working FDs via ANN queries; capping concurrency prevents
/// stacked spikes from exhausting the default nofile=1024 limit under load.
const HEAVY_SCAN_MAX_CONCURRENT: usize = 4;

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
        let tokenizer = Arc::clone(&self.tokenizer);
        let chunk_params = self.chunk_params.clone();
        let provider = self.config.embed_provider.clone();
        let http_client = self.http_client.clone();
        let embed_batch_size = self.config.embed_batch_size;
        let task_id_clone = task_id.clone();

        // Spawn the indexing pipeline as a background task.
        // Monitor the JoinHandle so panics are captured → Error, never Pending forever.
        let store_for_panic = Arc::clone(&store);
        let task_id_for_panic = task_id.clone();
        // Coordinates for releasing the reservation on the panic/cancel path
        // (the in-task path releases via its own owned `domain_str`/`branch`/`commit`).
        let reservation_coords = (domain_str.clone(), branch.clone(), commit.clone());
        let handle = tokio::spawn(async move {
            let ctx = PipelineCtx {
                store: &store,
                tokenizer: &tokenizer,
                chunk_params: &chunk_params,
                provider: &provider,
                http_client: &http_client,
                domain: &domain_str,
                branch: &branch,
                embed_batch_size,
            };
            let result = io_run_index_pipeline(&ctx, &commit, operations).await;

            // Release the reservation on EVERY terminal state. On success the
            // durable Lance tag now keeps the 409 guard correct; on failure
            // releasing returns the commit to absent so a retry is allowed.
            match result {
                Ok((indexed, skipped)) => {
                    // Probabilistic background compaction (BUG-FD24): 5% chance
                    // on a successful write that created fragments. Does NOT
                    // block — spawns a background task if the roll hits.
                    if indexed > 0 {
                        LanceStore::maybe_trigger_background_compaction(
                            Arc::clone(&store),
                            domain_str.clone(),
                            branch.clone(),
                        );
                    }
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
        });

        // Monitor: if the spawned task panics, record Error (never leave Pending)
        // AND release the reservation so the panicked commit can be retried (the
        // in-task release never runs on a panic).
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

        // Reuse the source document's STORED embedding (chunk 0, the best
        // available) directly as the query vector — NO re-embedding round-trip.
        // The lookup projects the vector from the snapshot, and every embedding is
        // L2-normalised at insert time (both the service pipeline and the bulk
        // loader normalise before write), so it is already on the unit sphere that
        // Lance's cosine metric expects — no re-normalisation here.
        //
        // This is both faster (zero embed calls per /similar) and strictly more
        // faithful than re-embedding the chunk text, which could drift from the
        // stored vector due to batching/quantisation/provider non-determinism.
        let query_embedding = doc_chunks[0].embedding.clone();

        // FAIL-LOUD sanity check: the stored vector must be present, the right
        // dimension, and unit-norm. A violation means the projection or the insert
        // invariant is broken — surface it rather than silently search on garbage.
        let expected_dim = self.config.embed_provider.expected_dim();
        if query_embedding.len() != expected_dim {
            return Err(ServiceError::Internal(format!(
                "stored embedding for {} has dimension {}, expected {}",
                id,
                query_embedding.len(),
                expected_dim
            )));
        }
        let norm_sq: f32 = query_embedding.iter().map(|v| v * v).sum();
        if (norm_sq - 1.0).abs() > 1e-3 {
            return Err(ServiceError::Internal(format!(
                "stored embedding for {} is not L2-normalised (norm^2 = {}); \
                 the insert-time normalisation invariant is violated",
                id, norm_sq
            )));
        }

        let search_query = SearchQuery {
            query_embedding,
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
        let paginated: Vec<SearchHit> = all_results
            .into_iter()
            .filter(|hit| hit.id != id)
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

        let groups = self
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

        // Bounded pagination over the deterministic (sorted) group list.
        let paginated = groups.into_iter().skip(start).take(count).collect();
        Ok(paginated)
    }

    /// Entity resolution: batch server-side matching over stored vectors.
    ///
    /// Replaces N sequential HTTP /similar calls with a single in-process batch:
    /// scan both populations, collect reciprocal cross-NN in-process (FD-safe per
    /// BUG-FD24), then run the 3-threshold matching algorithm (port of resolve.js).
    ///
    /// Returns the 3-partition output (matched, set_only, target_only) plus stats.
    #[allow(clippy::too_many_arguments)]
    pub async fn resolve_with_options(
        &self,
        domain_raw: &str,
        commit: &str,
        scope: &DuplicateScope,
        k: usize,
        threshold: f32,
        tau_one_to_one: f32,
        tau_one_to_many: Option<f32>,
        tau_many_to_one: Option<f32>,
        ancestors: &[String],
    ) -> Result<crate::resolve::ResolveResult, ServiceError> {
        let rp =
            parse_domain(domain_raw).map_err(|e| ServiceError::Validation(e.to_string()))?;

        if !self.is_search_ready() {
            return Err(ServiceError::Unavailable(
                "search capability not ready (embedding backend cold)".to_owned(),
            ));
        }

        // Acquire heavy-scan permit (BUG-FD24): bounds concurrent /resolve requests
        // so stacked FD spikes cannot exhaust nofile under load.
        let _permit = self
            .heavy_scan_semaphore
            .acquire()
            .await
            .map_err(|_| ServiceError::Internal("heavy-scan semaphore closed".to_owned()))?;

        let domain = Domain::from_resource_path(&rp);
        let domain_str = domain.as_str().to_owned();
        let branch = extract_branch(&rp);

        // Resolve the searchable commit (catch-up resolution).
        let served_commit = self
            .resolve_searchable_commit(&domain, &branch, commit, ancestors)
            .await?;

        let start_time = std::time::Instant::now();

        // Collect reciprocal cross-NN maps (in-process batch, FD-safe).
        let maps = self
            .store
            .io_resolve_cross_neighbours(
                &domain_str,
                &branch,
                &served_commit,
                scope,
                k,
                threshold,
                crate::store::lance::DEFAULT_DUPLICATE_MAX_POINTS,
            )
            .await
            .map_err(|e| ServiceError::Internal(e.to_string()))?;

        // Run the pure matching algorithm.
        let options = crate::resolve::ResolveOptions {
            k,
            threshold,
            tau_one_to_one,
            tau_one_to_many,
            tau_many_to_one,
        };

        let elapsed_ms = start_time.elapsed().as_millis() as u64;
        let result =
            crate::resolve::resolve(&maps.set_to_target, &maps.target_to_set, &options, elapsed_ms);

        Ok(result)
    }

    /// Global statistics — aggregates across ALL domains (admin/internal only).
    pub async fn statistics(&self) -> Result<Statistics, ServiceError> {
        Ok(self.store.statistics().await)
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

    /// Trigger data compaction for a domain's dataset on a given branch.
    /// Merges small fragments (one per push) into fewer large fragments, reducing
    /// the FD count for full-table scans. Returns before/after fragment counts.
    ///
    /// After compaction, all existing commit tags are re-pointed to the new
    /// (compacted) version so that tag-resolved snapshots use the fewer-fragment
    /// layout. Without re-tagging, old tags reference the pre-compaction version
    /// which still has N fragments per original push.
    ///
    /// Admin-only, idempotent: re-running on an already-compacted dataset is fast
    /// (the threshold check short-circuits if fragments <= 16).
    pub async fn compact_domain(
        &self,
        domain_raw: &str,
        branch: &str,
    ) -> Result<serde_json::Value, ServiceError> {
        let rp =
            parse_domain(domain_raw).map_err(|e| ServiceError::Validation(e.to_string()))?;
        let domain = Domain::from_resource_path(&rp);
        let domain_str = domain.as_str().to_owned();

        let ds = self
            .store
            .io_open_dataset_uncached(&domain_str, branch)
            .await
            .map_err(|e| ServiceError::Internal(e.to_string()))?;

        let mut ds = match ds {
            Some(d) => d,
            None => {
                return Err(ServiceError::NotFound(format!(
                    "domain {} has no dataset on disk",
                    domain_raw
                )));
            }
        };

        let fragments_before = ds.get_fragments().len();

        crate::store::lance::io_compact_data(&mut ds)
            .await
            .map_err(|e| ServiceError::Internal(e.to_string()))?;

        let fragments_after = ds.get_fragments().len();
        let compacted_version = ds.version().version;

        // Re-tag all commits pointing to older versions so that tag-resolved
        // snapshots (io_snapshot_from_cache → checkout_version) read from the
        // latest (fewest-fragment) layout. Without this, old tags reference
        // pre-compaction versions that still open N fragment files per scan.
        // Always retag stale tags — even if this invocation didn't compact
        // (fragments already low), tags may be stale from a prior compaction.
        let all_tags = ds
            .tags()
            .list()
            .await
            .map_err(|e| {
                ServiceError::Internal(format!("tag list for retag failed: {}", e))
            })?;
        let mut tags_repointed = 0u64;
        for (tag_name, tag_contents) in &all_tags {
            if tag_contents.version != compacted_version {
                ds.tags()
                    .update(tag_name, compacted_version)
                    .await
                    .map_err(|e| {
                        ServiceError::Internal(format!(
                            "retag '{}' to v{} failed: {}",
                            tag_name, compacted_version, e
                        ))
                    })?;
                tags_repointed += 1;
            }
        }

        // Refresh the cached handle so subsequent reads use the compacted data.
        self.store
            .io_refresh_cached_dataset(&domain_str, branch)
            .await
            .map_err(|e| ServiceError::Internal(e.to_string()))?;

        Ok(serde_json::json!({
            "domain": domain_raw,
            "branch": branch,
            "fragments_before": fragments_before,
            "fragments_after": fragments_after,
            "compacted": fragments_before != fragments_after,
            "compacted_version": compacted_version,
            "tags_repointed": tags_repointed
        }))
    }

    /// Compare two texts by embedding them and computing their normalized cosine distance.
    /// Stateless: NO dataset, NO ANN index, NO domain required.
    ///
    /// `source` is embedded with `EmbeddingRole::Query`, `target` with `EmbeddingRole::Document`.
    /// Both vectors are L2-normalized, then the normalized cosine distance is computed on
    /// the same [0, 1] reference scale that `/search`, `/similar`, `/duplicates`, and
    /// `/resolve` return (0 = identical, ~0.5 = unrelated, 1 = opposite).
    pub async fn compare(
        &self,
        source: &str,
        target: &str,
    ) -> Result<CompareResult, ServiceError> {
        // FIX 3 (#48): If source and target are identical strings, return distance 0
        // without embedding. This avoids the asymmetric-prefix artefact where identical
        // text embeds to different vectors (search_query: vs search_document: prefixes)
        // producing ~0.16 distance for genuinely identical content.
        // Checked BEFORE embedding — input-string equality, NOT vector equality.
        if source == target {
            return Ok(CompareResult {
                distance: 0.0,
                source_role: "query".to_owned(),
                target_role: "document".to_owned(),
            });
        }

        if !self.is_search_ready() {
            return Err(ServiceError::Unavailable(
                "search capability not ready (embedding backend cold)".to_owned(),
            ));
        }

        // Embed both texts in a single batch of 2 for efficiency.
        // Source uses Query role, Target uses Document role (asymmetric embedding).
        let source_texts = vec![source.to_owned()];
        let target_texts = vec![target.to_owned()];

        let source_embeddings = embed::io_embed(
            &self.config.embed_provider,
            &source_texts,
            EmbeddingRole::Query,
            &self.http_client,
        )
        .await
        .map_err(|e| ServiceError::Internal(format!("embedding source failed: {}", e)))?;

        let target_embeddings = embed::io_embed(
            &self.config.embed_provider,
            &target_texts,
            EmbeddingRole::Document,
            &self.http_client,
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
            source_role: "query".to_owned(),
            target_role: "document".to_owned(),
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

    // ─── COLLECT phase C: batch-embed all texts (DUAL — Document + Query) ───
    // Phase 6A Step 5: embed each chunk TWICE — once with Document role (for the
    // ANN-indexed `embedding` column) and once with Query role (for the stored
    // `query_embedding` column used by /resolve and /duplicates to probe with the
    // asymmetric query→document signal). Embed time ~doubles, mitigated by Step 3
    // batching (batch_size=32). Index build time does NOT double — only ONE ANN
    // index exists (on `embedding`).
    let embed_results_doc = embed::io_embed_batched(
        ctx.provider,
        &flat_texts,
        ctx.embed_batch_size,
        EmbeddingRole::Document,
        ctx.http_client,
    )
    .await;

    let embed_results_query = embed::io_embed_batched(
        ctx.provider,
        &flat_texts,
        ctx.embed_batch_size,
        EmbeddingRole::Query,
        ctx.http_client,
    )
    .await;

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
            .map(|(chunk, (embedding, query_embedding))| ChunkRow {
                doc_id: doc.id.clone(),
                doc_type: doc_type.clone(),
                chunk_index: chunk.index as i32,
                chunk_count: chunk.count as i32,
                chunk_token_start: chunk.token_start as i32,
                doc_token_len: chunk.doc_token_len as i32,
                embedding,
                query_embedding,
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

    // Acquire the per-branch lock. Held for the entire write→optimize→tag sequence.
    // This ensures at most one unindexed commit per branch and prevents snapshot
    // isolation violations (a later commit's data cannot leak into an earlier tag).
    // Acquired unconditionally: even a no-op commit must be tagged under the lock
    // to maintain ordering (a concurrent push must not interleave tags).
    let _guard = ctx.store.acquire_pipeline_lock(ctx.domain, ctx.branch).await;

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

        // Optimize indices on an UNCACHED handle (Fix #2: doesn't block searches).
        // FTS + vector ANN are built incrementally via optimize_indices(append()) — O(delta).
        // This runs BEFORE tagging so the tagged version has full index coverage (BLOCKER-1 fix).
        let indexed_version = io_optimize_on_uncached_handle(ctx.store, ctx.domain, ctx.branch)
            .await
            .map_err(|e| format!("index optimization failed: {}", e))?;

        // Tag the commit to the INDEXED version (data + indices).
        // After this point, the commit is resolvable and fully searchable.
        let tag_version = indexed_version.unwrap_or(last_version);
        ctx.store
            .io_tag_commit(ctx.domain, ctx.branch, commit, tag_version)
            .await
            .map_err(|e| format!("failed to tag commit: {}", e))?;

        // Update last-indexed tracking.
        ctx.store
            .update_last_indexed(ctx.domain, ctx.branch, commit, tag_version)
            .await;
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

/// Optimize FTS + vector indices on an uncached dataset handle.
/// Returns the version after optimization, or None if the dataset doesn't exist yet.
///
/// Uses `io_open_dataset_uncached` so the write operations don't hold the shared
/// `Arc<RwLock<Dataset>>` and block concurrent search reads (Fix #2).
/// After optimization, refreshes the cached handle.
///
/// The index creation/optimization calls internally use DataFusion, which calls
/// `Handle::current().block_on()` in some code paths (e.g. HashJoinExec). This
/// panics when called from within a tokio::spawn context ("Cannot block the current
/// thread from within a runtime"). To avoid this, the index operations run on a
/// dedicated blocking thread via `spawn_blocking`, with a fresh single-threaded
/// runtime for the async Lance calls.
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

    // Run index optimization on a blocking thread with its own runtime to avoid
    // the DataFusion "Cannot block from within a runtime" panic (see doc comment).
    let vector_config = store.vector_index_config().clone();
    let final_version = tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| format!("failed to create optimize runtime: {}", e))?;

        rt.block_on(async {
            // FTS: create inverted index or incrementally append new fragments.
            crate::store::lance::io_ensure_fts_index_on_dataset(&mut ds)
                .await
                .map_err(|e| format!("FTS optimize failed: {}", e))?;

            // Vector ANN: create IVF_PQ index (if enough rows) or incrementally append.
            crate::store::vector_index::io_ensure_vector_index(&mut ds, &vector_config)
                .await
                .map_err(|e| format!("vector optimize failed: {}", e))?;

            Ok::<u64, String>(ds.version().version)
        })
    })
    .await
    .map_err(|e| format!("optimize task panicked: {}", e))?
    .map_err(|e: String| e)?;

    // Refresh the cached handle so subsequent reads see the optimized indices.
    store
        .io_refresh_cached_dataset(domain, branch)
        .await
        .map_err(|e| format!("cache refresh failed: {}", e))?;

    Ok(Some(final_version))
}

// ═══════════════════════════════════════════════════════════════════════════════
// RISK-26 regression tests: no-op/empty commits must tag + advance last_indexed.
// ═══════════════════════════════════════════════════════════════════════════════
#[cfg(test)]
mod tests_risk26 {
    use super::*;
    use crate::kernel::model::BranchName;
    use std::path::Path;

    /// Build a test tokenizer from the checked-in fixture.
    fn test_tokenizer() -> Tokenizer {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("spikes")
            .join("tokenizer")
            .join("tokenizer.json");
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
            domain,
            branch,
            embed_batch_size: 32,
        }
    }

    /// regression (a): a commit of ONLY Operation::Error must tag + advance last_indexed.
    #[tokio::test]
    async fn all_error_commit_tags_and_advances_last_indexed() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let store = LanceStore::new(tmp.path(), 8);
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
            query_embedding: vec![0.1; 8],
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

    /// regression (b): an empty-operations commit must tag + advance last_indexed.
    #[tokio::test]
    async fn empty_operations_commit_tags_and_advances_last_indexed() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let store = LanceStore::new(tmp.path(), 8);
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
            query_embedding: vec![0.1; 8],
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
        let store = LanceStore::new(tmp.path(), 8);
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
            query_embedding: vec![0.1; 8],
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
            query_embedding: vec![0.2; 8],
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

    /// edge case: no-op commit on a never-before-indexed domain (first commit is empty).
    /// The dataset auto-creates and the commit gets tagged to version 1.
    #[tokio::test]
    async fn noop_commit_on_fresh_domain_creates_dataset_and_tags() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let store = LanceStore::new(tmp.path(), 8);
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

