//! Indexing queue — decouples push-commit latency from index maintenance.
//!
//! After a push upserts data and tags the commit, it enqueues an indexing
//! request for the (domain, branch). A background worker drains the queue,
//! calling `io_ensure_fts_index` + `io_ensure_vector_index` with
//! `OptimizeOptions::append()` semantics (O(new_fragments), not O(corpus)).
//!
//! Design properties:
//! - **Per-branch coalescing:** multiple enqueues for the same (domain, branch)
//!   collapse into a single optimize call. Only one optimize per branch runs at
//!   a time regardless of how many pushes queued.
//! - **Search-during-lag correctness:** Lance flat-searches unindexed fragments
//!   alongside indexed ones. Pending indexing affects latency, not correctness.
//! - **Fail-loud:** worker errors are logged and surfaced — a stuck queue is
//!   visible via `pending_index_fragments` in `/statistics`, never silent.
//! - **Graceful shutdown:** the worker drains remaining items before stopping.

use std::collections::HashSet;
use std::sync::Arc;

use tokio::sync::{Mutex, Notify};

use crate::store::lance::LanceStore;

/// An indexing request: ensure indices are up-to-date for this (domain, branch).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IndexRequest {
    pub domain: String,
    pub branch: String,
}

/// The indexing queue: a coalescing set + notify signal.
///
/// Per-branch coalescing: if (domain, branch) is already in the pending set,
/// a second enqueue is a no-op (the worker will cover all new fragments in
/// one `optimize_indices(append())` call regardless).
#[derive(Debug)]
pub struct IndexQueue {
    /// Pending requests (coalescing set — duplicates are no-ops).
    pending: Mutex<HashSet<IndexRequest>>,
    /// Notify the worker that new work is available.
    notify: Notify,
    /// Shutdown signal.
    shutdown: Mutex<bool>,
}

impl Default for IndexQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl IndexQueue {
    pub fn new() -> Self {
        Self {
            pending: Mutex::new(HashSet::new()),
            notify: Notify::new(),
            shutdown: Mutex::new(false),
        }
    }

    /// Enqueue an indexing request. Coalesces with any existing pending request
    /// for the same (domain, branch) — O(1), never blocks.
    pub async fn enqueue(&self, request: IndexRequest) {
        let mut pending = self.pending.lock().await;
        pending.insert(request);
        // Wake the worker (if it's waiting for work).
        self.notify.notify_one();
    }

    /// Drain all pending requests. Returns the set of unique requests to process.
    /// Called by the worker — takes everything currently pending in one batch.
    async fn drain(&self) -> Vec<IndexRequest> {
        let mut pending = self.pending.lock().await;
        let drained: Vec<IndexRequest> = pending.drain().collect();
        drained
    }

    /// Signal the worker to shut down after draining remaining work.
    pub async fn shutdown(&self) {
        let mut shutdown = self.shutdown.lock().await;
        *shutdown = true;
        self.notify.notify_one();
    }

    /// Check if shutdown has been requested.
    async fn is_shutdown(&self) -> bool {
        let shutdown = self.shutdown.lock().await;
        *shutdown
    }

    /// Check if there's pending work (used in tests to observe queue state).
    pub async fn pending_count(&self) -> usize {
        let pending = self.pending.lock().await;
        pending.len()
    }
}

/// Spawn the background index worker. Returns the JoinHandle for shutdown.
///
/// The worker loops: wait for notify → drain pending → optimize indices for
/// each unique (domain, branch). On shutdown, drains remaining work then exits.
///
/// Failure isolation: if `optimize_indices` fails for a (domain, branch), the
/// error is logged loudly. The request is NOT re-enqueued (the next push will
/// enqueue it again, triggering a retry). This avoids infinite retry loops
/// while ensuring the backlog is observable via `pending_index_fragments`.
pub fn spawn_index_worker(
    queue: Arc<IndexQueue>,
    store: Arc<LanceStore>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            // Wait for work or shutdown signal.
            queue.notify.notified().await;

            // Drain all pending requests.
            let requests = queue.drain().await;

            // Process each unique (domain, branch).
            for req in &requests {
                if let Err(e) = io_optimize_branch(&store, &req.domain, &req.branch).await {
                    // FAIL LOUD: log the error. The backlog remains visible via
                    // `pending_index_fragments` in /statistics (reads from Lance's
                    // index coverage bitmap, not this queue). The next push will
                    // re-enqueue this branch, triggering a retry.
                    eprintln!(
                        "[index-worker] ERROR: optimize for {}/{} failed: {}",
                        req.domain, req.branch, e
                    );
                }
            }

            // Check shutdown after processing (drain remaining work first).
            if queue.is_shutdown().await {
                // Final drain to ensure nothing is dropped on clean shutdown.
                let final_requests = queue.drain().await;
                for req in &final_requests {
                    if let Err(e) = io_optimize_branch(&store, &req.domain, &req.branch).await {
                        eprintln!(
                            "[index-worker] ERROR (shutdown): optimize for {}/{} failed: {}",
                            req.domain, req.branch, e
                        );
                    }
                }
                break;
            }
        }
    })
}

/// Optimize all indices for a single (domain, branch).
///
/// Opens a FRESH (uncached) dataset handle for optimization so that the
/// write-locked `optimize_indices` calls don't block concurrent search reads
/// on the cached handle. After optimization completes, refreshes the cache.
///
/// Sequence: open uncached → FTS optimize → vector optimize → refresh cache.
async fn io_optimize_branch(
    store: &LanceStore,
    domain: &str,
    branch: &str,
) -> Result<(), String> {
    // Open a fresh handle directly from disk (not from the shared RwLock cache).
    // This allows searches to continue reading from the cached handle without
    // being blocked by the write lock held during optimize_indices.
    let ds = store
        .io_open_dataset_uncached(domain, branch)
        .await
        .map_err(|e| format!("uncached open failed: {}", e))?;

    let mut ds = match ds {
        Some(d) => d,
        None => return Ok(()), // Dataset doesn't exist on disk yet — nothing to optimize.
    };

    // FTS index: create-once or incremental append on the fresh handle.
    crate::store::lance::io_ensure_fts_index_on_dataset(&mut ds)
        .await
        .map_err(|e| format!("FTS index optimize failed: {}", e))?;

    // Vector ANN index: create-once (if enough rows) or incremental append.
    crate::store::vector_index::io_ensure_vector_index(&mut ds, store.vector_index_config())
        .await
        .map_err(|e| format!("vector index optimize failed: {}", e))?;

    // Refresh the cached handle so subsequent reads see the optimized indices.
    store
        .io_refresh_cached_dataset(domain, branch)
        .await
        .map_err(|e| format!("cache refresh failed: {}", e))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn enqueue_coalesces_duplicates() {
        let queue = IndexQueue::new();
        let req = IndexRequest {
            domain: "admin/db".to_owned(),
            branch: "main".to_owned(),
        };

        queue.enqueue(req.clone()).await;
        queue.enqueue(req.clone()).await;
        queue.enqueue(req.clone()).await;

        assert_eq!(queue.pending_count().await, 1, "duplicates should coalesce");
    }

    #[tokio::test]
    async fn drain_empties_the_queue() {
        let queue = IndexQueue::new();
        queue
            .enqueue(IndexRequest {
                domain: "a/b".to_owned(),
                branch: "main".to_owned(),
            })
            .await;
        queue
            .enqueue(IndexRequest {
                domain: "c/d".to_owned(),
                branch: "dev".to_owned(),
            })
            .await;

        let drained = queue.drain().await;
        assert_eq!(drained.len(), 2);
        assert_eq!(queue.pending_count().await, 0, "queue should be empty after drain");
    }

    #[tokio::test]
    async fn shutdown_signals_worker_to_exit() {
        let queue = Arc::new(IndexQueue::new());
        let store = Arc::new(LanceStore::new(
            std::path::Path::new("/tmp/indexqueue-test-shutdown"),
            128,
        ));

        let handle = spawn_index_worker(Arc::clone(&queue), store);

        // Signal shutdown.
        queue.shutdown().await;

        // Worker should exit within a reasonable time.
        let result = tokio::time::timeout(std::time::Duration::from_secs(5), handle).await;
        assert!(result.is_ok(), "worker should exit after shutdown signal");
    }
}
