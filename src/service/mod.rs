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
}

impl std::fmt::Debug for SearchService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchService")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl SearchService {
    pub fn new(store: Arc<LanceStore>, config: Config, tokenizer: Tokenizer) -> Self {
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
        _parent_commit: Option<&str>,
        operations: Vec<Operation>,
    ) -> Result<String, ServiceError> {
        let rp =
            parse_domain(domain_raw).map_err(|e| ServiceError::Validation(e.to_string()))?;
        let domain = Domain::from_resource_path(&rp);
        let domain_str = domain.as_str().to_owned();
        let branch = branch_raw.to_owned();
        let commit = target_commit.to_owned();

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
        tokio::spawn(async move {
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

        // Check if source commit is indexed.
        let source_version = self
            .store
            .io_resolve_commit(domain_str, branch, source_commit)
            .await
            .map_err(|e| ServiceError::Internal(e.to_string()))?;

        match source_version {
            Some(version) => {
                // Tag the target commit to the same version.
                self.store
                    .io_tag_commit(domain_str, branch, target_commit, version)
                    .await
                    .map_err(|e| ServiceError::Internal(e.to_string()))?;
                Ok(())
            }
            None => Err(ServiceError::NotFound(format!(
                "source commit {} is not indexed",
                source_commit
            ))),
        }
    }

    /// Search: embed query → vector/fts/hybrid search → dedup → return.
    pub async fn search(
        &self,
        domain_raw: &str,
        commit: &str,
        q: &str,
    ) -> Result<Vec<SearchHit>, ServiceError> {
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

    /// Full search with all options.
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

        let query_embedding = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| ServiceError::Internal("no embedding returned".to_owned()))?;

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
            .io_search(&domain_str, &branch, commit, &search_query)
            .await
            .map_err(|e| ServiceError::Internal(e.to_string()))?;

        let results = dedup_chunks_to_documents(chunk_hits, snippet);

        // Apply pagination.
        let paginated: Vec<SearchHit> = results
            .into_iter()
            .skip(start)
            .take(count)
            .collect();

        Ok(paginated)
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

        let query_embedding = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| ServiceError::Internal("no embedding returned for similar".to_owned()))?;

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

/// Run the full indexing pipeline for a set of operations.
/// Returns (indexed_count, skipped_docs) on success, or error message on failure.
async fn io_run_index_pipeline(
    ctx: &PipelineCtx<'_>,
    commit: &str,
    operations: Vec<Operation>,
) -> Result<(u64, Vec<SkippedDoc>), String> {
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

    // Tag the commit to the final version (only if we have any data).
    // Order matters: FTS index creation produces a new version, so we must
    // create the index FIRST, then tag the commit to the version that includes
    // both data and the FTS index.
    if last_version > 0 {
        // Ensure FTS index exists — this may produce a new version.
        let fts_version = ctx
            .store
            .io_ensure_fts_index(ctx.domain, ctx.branch)
            .await
            .map_err(|e| format!("FTS index creation failed: {}", e))?;

        // Tag the commit to the version that includes the FTS index.
        let tag_version = if fts_version > 0 { fts_version } else { last_version };
        ctx.store
            .io_tag_commit(ctx.domain, ctx.branch, commit, tag_version)
            .await
            .map_err(|e| format!("failed to tag commit: {}", e))?;

        last_version = tag_version;
    }

    // Update last-indexed tracking.
    ctx.store
        .update_last_indexed(ctx.domain, ctx.branch, commit, last_version)
        .await;

    Ok((indexed_count, skipped))
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

    // 2. Embed all chunks in a single batch.
    let chunk_texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let embeddings =
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
