#![forbid(unsafe_code)]

//! Service — the transport-agnostic core API surface.
//! Owns no framework types (no axum/hyper in signatures).
//! Composes store operations and validates domain logic.

use std::sync::Arc;

use crate::config::Config;
use crate::kernel::error::ServiceError;
use crate::kernel::model::{
    parse_domain, BranchName, Domain, LastIndexed, Operation, SearchHit,
    Statistics, TaskStatus,
};
use crate::store::InMemoryStore;

/// The search service — owns the store and config, provides the transport-agnostic API.
#[derive(Debug, Clone)]
pub struct SearchService {
    store: InMemoryStore,
    config: Config,
    /// Per-capability readiness: index readiness is independent of search
    /// readiness (search additionally requires a warm embedding backend).
    ready_index: Arc<std::sync::atomic::AtomicBool>,
    ready_search: Arc<std::sync::atomic::AtomicBool>,
}

impl SearchService {
    pub fn new(store: InMemoryStore, config: Config) -> Self {
        let ready_index = Arc::new(std::sync::atomic::AtomicBool::new(true));
        // Search readiness controlled by env var; defaults to true when there
        // is no embedding backend to warm. Set TDB_SEARCH_SEARCH_READY=false to
        // exercise the cold-search path.
        let search_ready = std::env::var("TDB_SEARCH_SEARCH_READY")
            .map(|v| v != "false")
            .unwrap_or(true);
        let ready_search = Arc::new(std::sync::atomic::AtomicBool::new(search_ready));
        Self { store, config, ready_index, ready_search }
    }

    /// Set search readiness (used in tests to control the readiness state).
    pub fn set_search_ready(&self, ready: bool) {
        self.ready_search.store(ready, std::sync::atomic::Ordering::SeqCst);
    }

    /// Set index readiness.
    pub fn set_index_ready(&self, ready: bool) {
        self.ready_index.store(ready, std::sync::atomic::Ordering::SeqCst);
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
        let _rp = parse_domain(domain_raw)
            .map_err(|e| ServiceError::Validation(e.to_string()))?;
        let domain = Domain::from_resource_path(&_rp);
        let branch = BranchName::new(branch_raw.to_owned());
        Ok(self.store.last_indexed(&domain, &branch).await)
    }

    /// Start an async push/index task.
    pub async fn push(
        &self,
        domain_raw: &str,
        branch_raw: &str,
        target_commit: &str,
        _parent_commit: Option<&str>,
        _operations: Vec<Operation>,
    ) -> Result<String, ServiceError> {
        let rp = parse_domain(domain_raw)
            .map_err(|e| ServiceError::Validation(e.to_string()))?;
        let domain = Domain::from_resource_path(&rp);
        let branch = BranchName::new(branch_raw.to_owned());
        self.store
            .start_push(&domain, &branch, target_commit)
            .await
            .map_err(ServiceError::Conflict)
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
        let rp = parse_domain(domain_raw)
            .map_err(|e| ServiceError::Validation(e.to_string()))?;
        let domain = Domain::from_resource_path(&rp);
        self.store
            .assign(&domain, source_commit, target_commit)
            .await
            .map_err(ServiceError::Internal)
    }

    /// Search (stub returns empty).
    pub async fn search(
        &self,
        domain_raw: &str,
        _commit: &str,
        _q: &str,
    ) -> Result<Vec<SearchHit>, ServiceError> {
        let _rp = parse_domain(domain_raw)
            .map_err(|e| ServiceError::Validation(e.to_string()))?;
        if !self.is_search_ready() {
            return Err(ServiceError::Unavailable(
                "search capability not ready (embedding backend cold)".to_owned(),
            ));
        }
        Ok(self.store.search().await)
    }

    /// Similar (stub returns empty).
    pub async fn similar(
        &self,
        domain_raw: &str,
        _commit: &str,
        _id: &str,
    ) -> Result<Vec<SearchHit>, ServiceError> {
        let _rp = parse_domain(domain_raw)
            .map_err(|e| ServiceError::Validation(e.to_string()))?;
        Ok(self.store.similar().await)
    }

    /// Duplicates (stub returns empty).
    pub async fn duplicates(
        &self,
        domain_raw: &str,
        _commit: &str,
    ) -> Result<Vec<(String, String)>, ServiceError> {
        let _rp = parse_domain(domain_raw)
            .map_err(|e| ServiceError::Validation(e.to_string()))?;
        Ok(self.store.duplicates().await)
    }

    /// Statistics.
    pub async fn statistics(&self) -> Result<Statistics, ServiceError> {
        Ok(self.store.statistics().await)
    }

    /// Validate a domain string without side effects.
    pub fn validate_domain(&self, domain_raw: &str) -> Result<(), ServiceError> {
        parse_domain(domain_raw)
            .map_err(|e| ServiceError::Validation(e.to_string()))?;
        Ok(())
    }

    /// Record an error task (used when NDJSON parse fails before processing).
    pub async fn record_error_task(&self, task_id: &str, error: String) {
        self.store.record_error_task(task_id, error).await;
    }
}
