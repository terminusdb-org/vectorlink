// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 DFRNT AB

#![forbid(unsafe_code)]

//! Store — storage backends.
//!
//! InMemoryStore: stub for contract testing.
//! LanceStore: real LanceDB-backed persistence with versioned vector search.
//! vector_index: ANN index creation and incremental maintenance.

pub mod branch;
pub mod lance;
pub mod vector_index;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::RwLock;

use crate::kernel::model::{
    BranchName, Domain, LastIndexed, SearchHit, Statistics, TaskStatus,
};

/// Key for the pending-set guard: (domain string, branch name).
type PendingKey = (String, String);

/// In-memory stub store. Thread-safe via Arc<RwLock<...>>.
/// Retained for contract tests; production uses LanceStore.
#[derive(Debug, Clone)]
pub struct InMemoryStore {
    inner: Arc<RwLock<StoreInner>>,
}

#[derive(Debug, Default)]
struct StoreInner {
    /// Tasks by task ID, with insertion timestamp.
    tasks: HashMap<String, (TaskStatus, Instant)>,
    /// Pending push keys (domain, branch) — for concurrency guard.
    pending: HashMap<PendingKey, String>,
    /// Commit assignments: (domain, target_commit) -> source_commit.
    assignments: HashMap<(String, String), String>,
    /// Last-indexed tracking: (domain, branch) -> (commit, version).
    last_indexed: HashMap<(String, String), (Option<String>, u64)>,
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(StoreInner::default())),
        }
    }

    /// Get last-indexed for a (domain, branch) pair.
    pub async fn last_indexed(&self, domain: &Domain, branch: &BranchName) -> LastIndexed {
        let inner = self.inner.read().await;
        let key = (domain.as_str().to_owned(), branch.as_str().to_owned());
        match inner.last_indexed.get(&key) {
            Some((commit, version)) => LastIndexed {
                branch: branch.as_str().to_owned(),
                commit: commit.clone(),
                version: *version,
            },
            None => LastIndexed {
                branch: branch.as_str().to_owned(),
                commit: None,
                version: 0,
            },
        }
    }

    /// Start a push task. Returns Ok(task_id) or Err if already in progress.
    pub async fn start_push(
        &self,
        domain: &Domain,
        branch: &BranchName,
        target_commit: &str,
    ) -> Result<String, String> {
        let mut inner = self.inner.write().await;
        let key = (domain.as_str().to_owned(), branch.as_str().to_owned());
        if inner.pending.contains_key(&key) {
            return Err(format!(
                "a push for {}/{} is already in progress",
                domain.as_str(),
                branch.as_str()
            ));
        }
        let task_id = format!("task-{}", uuid::Uuid::new_v4().as_simple());
        inner.pending.insert(key.clone(), task_id.clone());
        // Mark task as complete immediately (stub — no real indexing).
        inner.tasks.insert(
            task_id.clone(),
            (TaskStatus::Complete {
                indexed_documents: 0,
                skipped: Vec::new(),
            }, Instant::now()),
        );
        // Update last-indexed.
        let version = inner
            .last_indexed
            .get(&key)
            .map(|(_, v)| v + 1)
            .unwrap_or(1);
        inner.last_indexed.insert(key.clone(), (Some(target_commit.to_owned()), version));
        // Release pending lock.
        inner.pending.remove(&key);
        Ok(task_id)
    }

    /// Check task status.
    pub async fn check_task(&self, task_id: &str) -> Option<TaskStatus> {
        let inner = self.inner.read().await;
        inner.tasks.get(task_id).map(|(s, _)| s.clone())
    }

    /// Assign a target commit to a source commit's index (no recompute).
    pub async fn assign(
        &self,
        domain: &Domain,
        source_commit: &str,
        target_commit: &str,
    ) -> Result<(), String> {
        let mut inner = self.inner.write().await;
        let key = (domain.as_str().to_owned(), target_commit.to_owned());
        inner.assignments.insert(key, source_commit.to_owned());
        Ok(())
    }

    /// Stub search — returns empty results.
    pub async fn search(&self) -> Vec<SearchHit> {
        Vec::new()
    }

    /// Stub similar — returns empty results.
    pub async fn similar(&self) -> Vec<SearchHit> {
        Vec::new()
    }

    /// Stub duplicates — returns empty results.
    pub async fn duplicates(&self) -> Vec<(String, String)> {
        Vec::new()
    }

    /// Record an error task (used when NDJSON parse fails).
    pub async fn record_error_task(&self, task_id: &str, error: String) {
        let mut inner = self.inner.write().await;
        inner.tasks.insert(
            task_id.to_owned(),
            (TaskStatus::Error { error }, Instant::now()),
        );
    }

    /// Stub statistics.
    pub async fn statistics(&self) -> Statistics {
        Statistics {
            domains: 0,
            branches: 0,
            indexed_commits: 0,
            documents: 0,
            chunks: 0,
            pending_index_fragments: 0,
            pending_index_documents: 0,
            store_clustering: None,
        }
    }
}
