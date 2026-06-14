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

/// Outcome of a `/similar` request — same staleness contract as `SearchOutcome`
/// (#A): carries the commit actually served so the transport reports it via
/// `TerminusDB-Data-Version` and never dresses a stale ancestor up as fresh.
#[derive(Debug, Clone)]
pub struct SimilarOutcome {
    pub hits: Vec<SearchHit>,
    /// The commit whose snapshot was actually searched (exact or proven ancestor).
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
            };
            let result = io_run_index_pipeline(&ctx, &commit, operations).await;

            // Release the reservation on EVERY terminal state. On success the
            // durable Lance tag now keeps the 409 guard correct; on failure
            // releasing returns the commit to absent so a retry is allowed.
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
    /// Enablement gate (#5, Spec 10 §5 / RISK-22): a branch with NO indexed
    /// lineage is 404'd and negatively cached BEFORE any ancestor walk. A branch
    /// that resolves is marked enabled (auto-enroll propagation).
    async fn resolve_searchable_commit(
        &self,
        domain: &Domain,
        branch: &str,
        requested_commit: &str,
        ancestors: &[String],
    ) -> Result<String, ServiceError> {
        let domain_str = domain.as_str();

        // 1. Negative cache FIRST — short-circuit without any I/O. Correct
        //    because the ONLY way a commit becomes indexed is via a push, and
        //    the push path busts this branch's negative cache (direct
        //    invalidation), so a freshly-indexed exact commit is never masked by
        //    a stale negative entry. This preserves the cache's purpose: a 404'd
        //    branch does not re-walk history (and does not re-list tags).
        if self.branch_state.is_negative_cached(domain_str, branch) {
            return Err(ServiceError::NotFound(format!(
                "no indexed ancestor for commit {} on branch {} (negatively cached)",
                requested_commit, branch
            )));
        }

        // 2. Load the dataset-global commit→version map ONCE (fail-loud on a
        //    real tag-list error; absent dataset → empty map). The
        //    nearest-ancestor walk is then a pure, in-memory resolution.
        let commit_versions = self
            .store
            .io_list_commit_versions(domain_str)
            .await
            .map_err(|e| ServiceError::Internal(e.to_string()))?;

        // 3. EXACT match: a directly-indexed commit is always serveable.
        //    Busts any (now-stale) negative entry + enrolls.
        if commit_versions.contains_key(requested_commit) {
            self.branch_state.enable(domain_str, branch);
            self.branch_state.invalidate_negative(domain_str, branch);
            return Ok(requested_commit.to_owned());
        }

        // 4. Enablement gate (#5): a branch with NO indexed lineage at all →
        //    404 + negative cache, before any ancestor walk. The durable
        //    last-indexed commit (or the enabled set) is the lineage signal.
        if !self.branch_has_indexed_lineage(domain, branch).await? {
            self.branch_state.record_negative(domain_str, branch);
            return Err(ServiceError::NotFound(format!(
                "no indexed lineage for branch {} (commit {})",
                branch, requested_commit
            )));
        }

        // 5. Pure nearest-ancestor resolution over the SUPPLIED window only.
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
            crate::layeridx::ResolvedLayer::Exact { commit, .. } => {
                self.branch_state.enable(domain_str, branch);
                self.branch_state.invalidate_negative(domain_str, branch);
                Ok(commit)
            }
            crate::layeridx::ResolvedLayer::Ancestor { served_commit, .. } => {
                // A PROVEN ancestor (member of the supplied window). Stale, but
                // never newer than requested. Auto-enroll the branch.
                self.branch_state.enable(domain_str, branch);
                Ok(served_commit)
            }
            crate::layeridx::ResolvedLayer::None => {
                // Requested not indexed and no SUPPLIED ancestor is indexed. We
                // will NOT serve the tip (could be a descendant of the requested
                // commit) — 404 + negative cache. TerminusDB nudges a push.
                self.branch_state.record_negative(domain_str, branch);
                Err(ServiceError::NotFound(format!(
                    "no indexed ancestor for commit {} on branch {} within the supplied window",
                    requested_commit, branch
                )))
            }
        }
    }

    /// Whether `branch` has any indexed lineage — the enablement signal for the
    /// resolution gate (#5, Spec 10 §5). The in-memory `enabled` set is the fast
    /// path (a branch is enabled on its first push / first resolved search); the
    /// durable `last_indexed.commit` is the authoritative backstop that survives
    /// a process restart (the `enabled` set is process-local). Reading
    /// `is_enabled` here is what makes the enablement set load-bearing rather
    /// than write-only.
    async fn branch_has_indexed_lineage(
        &self,
        domain: &Domain,
        branch: &str,
    ) -> Result<bool, ServiceError> {
        if self.branch_state.is_enabled(domain.as_str(), branch) {
            return Ok(true);
        }
        let last = self
            .store
            .last_indexed(domain, &BranchName::new(branch.to_owned()))
            .await;
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
