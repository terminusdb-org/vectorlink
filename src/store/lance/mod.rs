// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 DFRNT AB

#![forbid(unsafe_code)]

//! LanceDB-backed persistent store (single-branch linear history).
//!
//! Schema: one row per chunk, keyed by (doc_id, chunk_index).
//! Supports vector search, FTS, and hybrid search. Commit→version binding via
//! Lance tags (managed by layeridx).

mod config;
mod schema;
mod write;
mod commit;
mod index;
pub mod integrity;
mod search;
mod resolve;
mod rollup;
pub mod dedup;
mod stats;

#[cfg(test)]
#[allow(clippy::useless_vec)]
mod tests;

// --- Public API facade (preserves all existing import paths) ---

pub use self::config::VectorIndexConfig;
pub use self::schema::{
    ChunkHit, ChunkRow, DistanceKind, DuplicateScope, NeighbourObservation,
    ResolveNeighbourMaps, SearchQuery, SuggestHit, SuggestQuery, SuggestResult,
    DEFAULT_DUPLICATE_MAX_PAIRS, DEFAULT_DUPLICATE_MAX_POINTS, MAIN_BRANCH,
    COMPACT_REBUILD_PREFIX, compact_rebuild_branch_name, is_compact_rebuild_branch,
    is_reserved_branch_name,
};
pub use self::dedup::{dedup_chunks_to_documents, pairs_from_neighbours};
pub use self::index::{io_compact_data, io_ensure_fts_index_on_dataset, io_merge_indices};
pub use self::rollup::{io_exponential_rollup, io_incremental_cascade, merges_needed, rollup_partitions, should_index_commit};
pub use self::commit::VersionDelta;

/// Cleanup mode controls how aggressive version cleanup interacts with
/// rebuild-branch index files. The default (`CurrentCode`) preserves the
/// existing FIX-B behaviour (hiding `_indices/` during cleanup when rebuild
/// branches exist). `TargetNoPatch` removes the hiding and relies on
/// `clean_referenced_branches(false)` to protect rebuild-branch files.
/// `TargetWithPatch` is identical to `TargetNoPatch` — a LanceDB patch was
/// originally planned but found to be incorrect (all-branch tag protection
/// caused version number collisions between main and rebuild branches).
/// The enum variant is retained for future experimentation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum CleanupMode {
    /// Mode 0: current behaviour — hide `_indices/` when rebuild branches exist.
    #[default]
    CurrentCode,
    /// Mode A: remove hiding, set `clean_referenced_branches(false)`.
    TargetNoPatch,
    /// Mode B: identical to Mode A (no LanceDB patch needed).
    TargetWithPatch,
}

impl CleanupMode {
    /// Returns true if the FIX-B `_indices/` hiding should be active.
    pub fn hide_indices(self) -> bool {
        matches!(self, CleanupMode::CurrentCode)
    }

    /// Returns true if `clean_referenced_branches` should be set to false
    /// (i.e., do NOT clean referenced branches, preserving rebuild-branch files).
    pub fn disable_clean_referenced_branches(self) -> bool {
        matches!(self, CleanupMode::TargetNoPatch | CleanupMode::TargetWithPatch)
    }
}

/// A single embedding record for streaming. Contains the doc_id, the document
/// embedding, and optionally the clustering embedding (when store_clustering
/// is true).
#[derive(Debug, Clone, serde::Serialize)]
pub struct EmbeddingRecord {
    pub doc_id: String,
    pub embedding: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clustering_embedding: Option<Vec<f32>>,
}


use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{
    FixedSizeListArray, Float32Array, Int32Array, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::cleanup::CleanupPolicyBuilder;
// DatasetIndexExt is used in the integrity submodule.
use lance::session::Session;
use lance::io::ObjectStoreRegistry;
use tokio::sync::RwLock;

use crate::config::DomainSettingsMap;
use crate::kernel::error::StoreError;
use crate::layeridx::BranchIndex;

/// Per-branch metadata key: (domain_str, branch_str).
/// Used for last-indexed, pipeline locks, and index tracking — branch-precise
/// state per RISK-22. The Lance DATASET itself is domain-keyed (layout A): one
/// `{domain}.lance` dataset holds all TerminusDB branches as Lance branches.
pub(super) type BranchKey = (String, String);

/// The Lance-backed store (layout A: one dataset per domain, TerminusDB
/// branches as Lance branches inside it; tags are dataset-global).
#[derive(Debug)]
pub struct LanceStore {
    base_dir: PathBuf,
    pub dim: usize,
    /// Vector index configuration (nprobes, refine_factor for search).
    pub(super) vector_index_config: VectorIndexConfig,
    /// Open dataset handles, keyed by DOMAIN (layout A). The cached handle is
    /// the domain dataset opened at its default (main) branch head; branch
    /// writes check out a branch-bound handle without mutating this one.
    pub(super) datasets: RwLock<HashMap<String, Arc<RwLock<Dataset>>>>,
    /// Per-(domain, branch) index tracking (branch-precise).
    pub(super) branch_indexes: RwLock<HashMap<BranchKey, BranchIndex>>,
    /// Tasks by task ID, with insertion timestamp for TTL-based eviction.
    pub(super) tasks: RwLock<HashMap<String, (crate::kernel::model::TaskStatus, std::time::Instant)>>,
    /// Per-(domain, branch) pipeline serialisation lock.
    /// Ensures concurrent pushes to the same branch are serialised so that
    /// commit→version tags are correctly isolated.
    pub(super) pipeline_locks: RwLock<HashMap<BranchKey, Arc<tokio::sync::Mutex<()>>>>,
    /// Per-(domain, branch) cancellation token for the currently-running pipeline task.
    /// When a new push arrives for a branch that already has an active pipeline,
    /// the previous task's token is cancelled so it can send a proper Error
    /// progress update before exiting. This prevents stale chunks from polluting
    /// the new index and gives the client (TerminusDB) a clean error message.
    pub(super) pipeline_cancel_tokens: RwLock<HashMap<BranchKey, tokio_util::sync::CancellationToken>>,
    /// Per-DOMAIN guard serialising dataset CREATE (write path) against
    /// `DELETE /domain` (BLOCKER-2 / #6). A delete holds this for the whole
    /// remove-then-purge sequence; a write that must create the domain dataset
    /// holds it across the create. This makes "remove the footprint" atomic with
    /// respect to a concurrent first-write — the delete never races a half-created
    /// dataset, and a create never silently revives a domain mid-delete.
    domain_guards: RwLock<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    /// In-flight commit reservations keyed by (domain, branch, commit).
    ///
    /// A commit is "committed" the moment its push is ACCEPTED — not only once
    /// indexing finishes and tags it. This set holds the ACCEPTED→INDEXING window:
    /// a commit is inserted when its push passes the guard (reserved), and removed
    /// when indexing reaches a terminal state. On SUCCESS the durable Lance tag
    /// becomes the permanent "Indexed" marker (and the reservation is dropped); on
    /// FAILURE the reservation is dropped so a legitimate retry is allowed.
    ///
    /// The 409 guard rejects a re-push whose commit is EITHER reserved (in this
    /// set) OR already Indexed (tagged). Mutations are serialised by
    /// `reservation_lock` so the check-and-reserve is atomic (no TOCTOU race
    /// between two concurrent pushes of the same commit).
    pub(super) inflight_commits: RwLock<std::collections::HashSet<(String, String, String)>>,
    /// Serialises the atomic check-and-reserve in `io_try_reserve_commit` so two
    /// concurrent pushes of the SAME (domain, branch, commit) cannot both pass.
    pub(super) reservation_lock: tokio::sync::Mutex<()>,
    /// Count of `Dataset::open` (fresh from disk) calls. Each fresh open spins up
    /// a NEW Lance object-store + session holding its own file readers; under
    /// concurrent load these transient opens are the FD-pressure source that
    /// exhausted descriptors (BUG-FD24). Read paths must reuse the cached domain
    /// handle and `checkout_version`/`checkout_branch` off it (which SHARE the
    /// handle's object_store + session — see `Dataset::checkout_by_ref`), so the
    /// fresh-open count does NOT grow with the number of searches. Instrumented so
    /// the regression test can assert reads reuse the cache rather than opening
    /// fresh per query.
    fresh_open_count: std::sync::atomic::AtomicU64,
    /// Per-domain settings (clustering, etc.). Loaded from
    /// `domain_settings.json` at startup.
    pub domain_settings: DomainSettingsMap,
    /// Shared Lance session — all Dataset::open calls reuse this session,
    /// sharing the index cache, metadata cache, and object store registry.
    /// This prevents memory accumulation from N independent sessions (each
    /// with its own 6 GiB index cache + 1 GiB metadata cache by default).
    /// Configured with reduced cache sizes: 512 MiB index + 256 MiB metadata.
    pub(super) lance_session: Arc<Session>,
    /// Pipeline progress: chunks read from channel but not yet written to Lance.
    /// Reset to 0 at the start of each pipeline run, incremented as chunks are
    /// produced by the chunker, decremented as they are written.
    pub pipeline_pending_chunks: std::sync::atomic::AtomicU64,
    /// Pipeline progress: chunks that have been embedded (embedding result received).
    pub pipeline_embedded_chunks: std::sync::atomic::AtomicU64,
    /// Pipeline progress: chunks that have been written to Lance (append committed).
    pub pipeline_written_chunks: std::sync::atomic::AtomicU64,
    /// Number of pipeline tasks currently running (spawned but not yet completed).
    /// Incremented in push_stream's tokio::spawn, decremented on completion/error/panic.
    pub pipeline_active_tasks: std::sync::atomic::AtomicU64,
    /// Configured Lance index cache capacity (for stats reporting).
    pub(super) lance_index_cache_capacity: usize,
    /// Configured Lance metadata cache capacity (for stats reporting).
    pub(super) lance_metadata_cache_capacity: usize,
    /// Per-(domain, branch) count of index deltas created since last compaction.
    /// Incremented after each push that creates a delta via `optimize_indices(append())`.
    /// Used by `io_incremental_cascade` to determine how many `merge(3)` calls
    /// are needed to maintain the base-3 hierarchy. Reset to 0 on compaction
    /// (when indices are dropped and recreated).
    pub(super) index_delta_counts: RwLock<HashMap<BranchKey, u64>>,
    /// Cleanup mode controlling how version cleanup interacts with
    /// rebuild-branch index files. Default is `CurrentCode` (existing
    /// FIX-B behaviour). Can be changed at runtime for testing.
    pub(super) cleanup_mode: std::sync::atomic::AtomicU8,
}

impl LanceStore {
    /// Create a new LanceStore backed by the given directory.
    pub fn new(base_dir: &Path, dim: usize, lance_index_cache_bytes: usize, lance_metadata_cache_bytes: usize) -> Self {
        let domain_settings = DomainSettingsMap::load(base_dir);
        let vector_index_config = VectorIndexConfig::default_for_dim(dim);
        Self {
            base_dir: base_dir.to_owned(),
            dim,
            vector_index_config,
            datasets: RwLock::new(HashMap::new()),
            branch_indexes: RwLock::new(HashMap::new()),
            tasks: RwLock::new(HashMap::new()),
            pipeline_locks: RwLock::new(HashMap::new()),
            pipeline_cancel_tokens: RwLock::new(HashMap::new()),
            domain_guards: RwLock::new(HashMap::new()),
            inflight_commits: RwLock::new(std::collections::HashSet::new()),
            reservation_lock: tokio::sync::Mutex::new(()),
            fresh_open_count: std::sync::atomic::AtomicU64::new(0),
            domain_settings,
            lance_session: Arc::new(Session::new(
                lance_index_cache_bytes,
                lance_metadata_cache_bytes,
                Arc::new(ObjectStoreRegistry::default()),
            )),
            pipeline_pending_chunks: std::sync::atomic::AtomicU64::new(0),
            pipeline_embedded_chunks: std::sync::atomic::AtomicU64::new(0),
            pipeline_written_chunks: std::sync::atomic::AtomicU64::new(0),
            pipeline_active_tasks: std::sync::atomic::AtomicU64::new(0),
            lance_index_cache_capacity: lance_index_cache_bytes,
            lance_metadata_cache_capacity: lance_metadata_cache_bytes,
            index_delta_counts: RwLock::new(HashMap::new()),
            cleanup_mode: std::sync::atomic::AtomicU8::new(CleanupMode::CurrentCode as u8),
        }
    }

    /// Set the cleanup mode. Controls how version cleanup interacts with
    /// rebuild-branch index files. See `CleanupMode` for details.
    pub fn set_cleanup_mode(&self, mode: CleanupMode) {
        self.cleanup_mode
            .store(mode as u8, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get the current cleanup mode.
    pub fn cleanup_mode(&self) -> CleanupMode {
        let raw = self.cleanup_mode.load(std::sync::atomic::Ordering::Relaxed);
        match raw {
            0 => CleanupMode::CurrentCode,
            1 => CleanupMode::TargetNoPatch,
            2 => CleanupMode::TargetWithPatch,
            _ => CleanupMode::CurrentCode,
        }
    }

    /// Get the index delta count for a (domain, branch) — the number of delta
    /// indices created since last compaction. Used by the cascade to determine
    /// how many merge(3) calls are needed.
    pub async fn io_get_delta_count(&self, domain: &str, branch: &str) -> u64 {
        let key = (domain.to_owned(), branch.to_owned());
        let counts = self.index_delta_counts.read().await;
        *counts.get(&key).unwrap_or(&0)
    }

    /// Increment the index delta count for a (domain, branch) after a push
    /// creates a new delta index via `optimize_indices(append())`.
    pub async fn io_increment_delta_count(&self, domain: &str, branch: &str) {
        let key = (domain.to_owned(), branch.to_owned());
        let mut counts = self.index_delta_counts.write().await;
        *counts.entry(key).or_insert(0) += 1;
    }

    /// Reset the index delta count for a (domain, branch) to 0. Called after
    /// compaction drops and recreates indices (the delta count restarts).
    pub async fn io_reset_delta_count(&self, domain: &str, branch: &str) {
        let key = (domain.to_owned(), branch.to_owned());
        let mut counts = self.index_delta_counts.write().await;
        counts.insert(key, 0);
    }

    /// Open the domain dataset FRESH from disk, tracking the open for FD/perf
    /// instrumentation. Every `Dataset::open` allocates a fresh object_store +
    /// session (with their own file readers), so this is the FD-pressure entry
    /// point we minimise (BUG-FD24). Callers that can reuse a cached handle MUST
    /// do so via `io_open_dataset_readonly` instead.
    ///
    /// INVARIANT (poka-yoke): every `Dataset::open` in this module goes through
    /// `io_open_fresh` (it is the ONLY counted fresh-open constructor). A bare
    /// `Dataset::open` is FORBIDDEN — it bypasses `fresh_open_count` and so HIDES
    /// the open from the FD-leak regression guard
    /// (`search_does_not_leak_file_descriptors_under_load`), which is exactly how
    /// the BUG-FD24 hot-path leak went unnoticed. Read paths that can reuse a
    /// cached handle must NOT open at all (clone the cached `Dataset` +
    /// `checkout_*` off it — that shares object_store + session, no new FDs). The
    /// only non-open dataset constructor is `Dataset::write` (create path), which
    /// is a distinct semantic (it makes a new empty dataset, not a re-open of an
    /// existing one) and is the single deliberate exception.
    pub(super) async fn io_open_fresh(&self, uri: &str) -> Result<Dataset, StoreError> {
        self.fresh_open_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        DatasetBuilder::from_uri(uri)
            .with_session(self.lance_session.clone())
            .load()
            .await
            .map_err(|e| StoreError::Internal(format!("dataset open failed: {}", e)))
    }

    /// Number of fresh `Dataset::open` calls performed so far (test
    /// instrumentation for the FD-exhaustion regression guard).
    #[cfg(test)]
    pub fn fresh_open_count(&self) -> u64 {
        self.fresh_open_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Acquire the per-domain create/delete guard (BLOCKER-2 / #6). Held by a
    /// dataset-creating write across the create, and by `DELETE /domain` across
    /// the whole remove-then-purge — so the two never interleave.
    pub async fn acquire_domain_guard(&self, domain: &str) -> tokio::sync::OwnedMutexGuard<()> {
        let lock = {
            let guards = self.domain_guards.read().await;
            if let Some(l) = guards.get(domain) {
                Arc::clone(l)
            } else {
                drop(guards);
                let mut guards = self.domain_guards.write().await;
                guards
                    .entry(domain.to_owned())
                    .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
                    .clone()
            }
        };
        lock.lock_owned().await
    }

    /// Get the vector index configuration (used by the ingest pipeline).
    pub fn vector_index_config(&self) -> &VectorIndexConfig {
        &self.vector_index_config
    }

    /// Override the vector index configuration (used in tests to control search params).
    #[cfg(test)]
    pub fn set_vector_index_config(&mut self, config: VectorIndexConfig) {
        self.vector_index_config = config;
    }

    /// Get the Arrow schema for chunk rows (embedding dimension from config).
    /// Includes BOTH `embedding` (document-role, ANN-INDEXED) and
    /// `clustering_embedding` (clustering-role, STORED, optionally ANN-indexed
    /// when clustering is enabled for the domain).
    pub(super) fn chunk_schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Utf8, false),
            Field::new("doc_type", DataType::Utf8, false),
            Field::new("chunk_index", DataType::Int32, false),
            Field::new("chunk_count", DataType::Int32, false),
            Field::new("chunk_token_start", DataType::Int32, false),
            Field::new("doc_token_len", DataType::Int32, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    self.dim as i32,
                ),
                false,
            ),
            // Clustering-role embedding: stored for /candidates dual KNN gather.
            // ANN-indexed when clustering is enabled for the domain; otherwise
            // stored as zeros and not indexed.
            Field::new(
                "clustering_embedding",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    self.dim as i32,
                ),
                false,
            ),
            Field::new("content", DataType::Utf8, false),
        ]))
    }

    /// Get the on-disk path for a domain's Lance dataset (layout A: one dataset
    /// per domain; branches live inside it as Lance branches).
    ///
    /// Each segment of the domain (`org`, `team`) is URI-percent-encoded so
    /// any character is safe in the filesystem path. This is a breaking change
    /// from the old `replace('/', "__")` encoding — a clean re-index is required.
    pub fn dataset_path(&self, domain: &str) -> PathBuf {
        let safe_domain = encode_domain_path(domain);
        self.base_dir.join(format!("{}.lance", safe_domain))
    }

    /// Open or create the cached domain dataset handle (at its default branch
    /// head). The `branch` argument is accepted for signature stability across
    /// the store API; branch selection for reads/writes happens via
    /// `checkout_branch` / `checkout_version` on the returned handle's data, not
    /// by opening separate directories (layout A).
    ///
    /// INVARIANT: the cached handle is opened at the dataset's native default
    /// branch. Callers that need a specific branch's head check out a branch-
    /// bound handle (`io_open_branch_for_write`); callers that need a commit's
    /// snapshot check out a version (`io_search`). The cache is never advanced
    /// to a non-default branch.
    pub async fn io_open_dataset(
        &self,
        domain: &str,
        _branch: &str,
    ) -> Result<Arc<RwLock<Dataset>>, StoreError> {
        // Check cache first (keyed by domain — layout A).
        {
            let datasets = self.datasets.read().await;
            if let Some(ds) = datasets.get(domain) {
                return Ok(Arc::clone(ds));
            }
        }

        let path = self.dataset_path(domain);
        let uri = path.to_string_lossy().to_string();

        // Try to open existing dataset (counted — even a cache-miss open is
        // funnelled through `io_open_fresh` so it is visible to the FD guard).
        let ds = if path.exists() {
            self.io_open_fresh(&uri).await?
        } else {
            // Create a new empty dataset with the schema.
            let schema = self.chunk_schema();
            let empty_batch = self.empty_batch();
            let reader = RecordBatchIterator::new(vec![Ok(empty_batch)], schema);
            let params = WriteParams {
                mode: WriteMode::Create,
                ..Default::default()
            };
            Dataset::write(reader, &uri, Some(params))
                .await
                .map_err(|e| StoreError::Internal(format!("failed to create dataset: {}", e)))?
        };

        let arc_ds = Arc::new(RwLock::new(ds));
        let mut datasets = self.datasets.write().await;
        // Re-check: a concurrent opener may have inserted while we were opening.
        if let Some(existing) = datasets.get(domain) {
            return Ok(Arc::clone(existing));
        }
        datasets.insert(domain.to_owned(), Arc::clone(&arc_ds));
        Ok(arc_ds)
    }

    /// READ-ONLY domain dataset open (layout A) — NEVER auto-creates.
    ///
    /// Returns `Some(handle)` if the dataset exists on disk (or is already
    /// cached), `None` if it does not exist. Unlike `io_open_dataset`, this is
    /// the path used by resolve/search/lookup: a read against a domain that was
    /// never indexed — or was deleted via `DELETE /domain` — must NOT bring an
    /// empty dataset back into existence (BLOCKER-2 resurrection class). Only a
    /// genuine write/index (`io_upsert_chunks` / `io_delete_doc`) may create.
    ///
    /// FAIL-LOUD: a real open error (corrupt manifest / I/O) propagates as
    /// `StoreError`; only a genuinely-absent directory yields `Ok(None)`.
    pub async fn io_open_dataset_readonly(
        &self,
        domain: &str,
    ) -> Result<Option<Arc<RwLock<Dataset>>>, StoreError> {
        // Check cache first (keyed by domain — layout A).
        {
            let datasets = self.datasets.read().await;
            if let Some(ds) = datasets.get(domain) {
                return Ok(Some(Arc::clone(ds)));
            }
        }

        let path = self.dataset_path(domain);
        if !path.exists() {
            // Genuinely absent — do NOT create. (Resurrection guard.)
            return Ok(None);
        }

        let uri = path.to_string_lossy().to_string();
        // Counted fresh open (cache miss). Once cached, subsequent reads reuse the
        // handle and never re-open (BUG-FD24).
        let ds = self.io_open_fresh(&uri).await?;

        let arc_ds = Arc::new(RwLock::new(ds));
        let mut datasets = self.datasets.write().await;
        // Re-check: a concurrent opener may have inserted while we were opening.
        if let Some(existing) = datasets.get(domain) {
            return Ok(Some(Arc::clone(existing)));
        }
        datasets.insert(domain.to_owned(), Arc::clone(&arc_ds));
        Ok(Some(arc_ds))
    }

    /// Open a FRESH, branch-bound dataset handle for WRITING to `branch`
    /// (layout A). Opens the domain dataset from disk and checks out the named
    /// branch so that appends/deletes target that branch's head, leaving sibling
    /// branches (and the cached main handle) untouched.
    ///
    /// INVARIANT: the returned handle is bound to `branch`; an append on it
    /// advances only `branch`'s head. For `branch == MAIN_BRANCH` the dataset's
    /// native default branch is the write target (no checkout needed).
    ///
    /// Returns an error if the dataset or the branch does not exist — callers
    /// must ensure the dataset (and any non-main branch) exists first.
    pub async fn io_open_branch_for_write(
        &self,
        domain: &str,
        branch: &str,
    ) -> Result<Dataset, StoreError> {
        let path = self.dataset_path(domain);
        let uri = path.to_string_lossy().to_string();
        // Write path: a fresh branch-bound handle is intentional (it must not
        // share the cached main handle), but it is still counted via `io_open_fresh`
        // so the open is VISIBLE to the FD guard. Write opens are one-shot per
        // mutation (not a hot read loop), so the count growth is bounded and
        // expected.
        let ds = self.io_open_fresh(&uri).await?;

        if branch == MAIN_BRANCH {
            // Native default branch — writes already target it.
            return Ok(ds);
        }

        ds.checkout_branch(branch).await.map_err(|e| {
            StoreError::Internal(format!(
                "failed to checkout branch '{}' for write (does it exist?): {}",
                branch, e
            ))
        })
    }

    /// Get the current head version for a (domain, branch) pair.
    ///
    /// For main: returns the cached handle's version (same as `io_open_dataset`).
    /// For non-main: opens a fresh branch-bound handle and reads its version.
    ///
    /// PRECONDITION: the dataset and branch must already exist — this never
    /// auto-creates. Use `io_open_dataset` first if auto-creation is needed.
    ///
    /// Used by the no-op pipeline path to determine the version to tag
    /// an empty commit to. The version must be from the BRANCH's lineage — using
    /// the main handle's version for a non-main branch would produce a tag pointing
    /// to a version that doesn't exist in the branch's lineage (Lance rejects it).
    pub async fn io_branch_head_version(
        &self,
        domain: &str,
        branch: &str,
    ) -> Result<u64, StoreError> {
        if branch == MAIN_BRANCH {
            let ds_arc = self.io_open_dataset(domain, branch).await?;
            let ds = ds_arc.read().await;
            return Ok(ds.version().version);
        }

        // Non-main: open a fresh branch-bound handle to read its head version.
        let ds = self.io_open_branch_for_write(domain, branch).await?;
        Ok(ds.version().version)
    }

    /// List the Lance branch names that currently exist in the domain dataset.
    /// Returns an empty list if the dataset doesn't exist yet.
    pub async fn io_list_branches(&self, domain: &str) -> Result<Vec<String>, StoreError> {
        let path = self.dataset_path(domain);
        if !path.exists() {
            return Ok(Vec::new());
        }
        let uri = path.to_string_lossy().to_string();
        // Counted fresh open (administrative/branch-listing path, not a hot loop).
        let ds = self.io_open_fresh(&uri).await?;
        let branches = ds
            .list_branches()
            .await
            .map_err(|e| StoreError::Internal(format!("list_branches failed: {}", e)))?;
        Ok(branches.into_keys().collect())
    }

    /// Create a new Lance branch `branch` forked from `from_version` (layout A,
    /// block reuse). The new branch shares the parent version's fragment files
    /// (shallow clone — no data copied; RISK-01, proven by Phase-0 spike 0a1.3).
    /// Subsequent writes on the branch add only delta fragments.
    ///
    /// INVARIANT: `from_version` must be a real version of the domain dataset
    /// (typically resolved from a parent commit's tag). Creating a branch that
    /// already exists is an error (fail loud — the caller decides idempotency).
    ///
    /// After creation the cache is left untouched (it tracks the default branch);
    /// the new branch is reached via `checkout_branch` on subsequent operations.
    pub async fn io_create_branch(
        &self,
        domain: &str,
        branch: &str,
        from_version: u64,
    ) -> Result<(), StoreError> {
        if branch == MAIN_BRANCH {
            return Err(StoreError::Internal(
                "cannot create the default 'main' branch — it always exists".to_owned(),
            ));
        }

        // The domain dataset must already exist (the parent was indexed there).
        let path = self.dataset_path(domain);
        if !path.exists() {
            return Err(StoreError::Internal(format!(
                "cannot branch domain '{}': dataset does not exist (index the parent first)",
                domain
            )));
        }
        let uri = path.to_string_lossy().to_string();
        // Counted fresh open (one-shot branch-create path, not a hot loop).
        let mut ds = self.io_open_fresh(&uri).await?;

        ds.create_branch(branch, from_version, None)
            .await
            .map_err(|e| {
                StoreError::Internal(format!(
                    "create_branch '{}' from version {} failed: {}",
                    branch, from_version, e
                ))
            })?;

        Ok(())
    }

    /// Create a new Lance branch forked from a commit tag. Uses LanceDB's
    /// `Ref::Tag` to resolve the tag's owning branch and version, so this
    /// works correctly even when the tag lives on a rebuild branch (after
    /// delta-fork retagging). This is the branch-aware counterpart to
    /// `io_create_branch`, which only works for main-branch versions.
    pub async fn io_create_branch_from_tag(
        &self,
        domain: &str,
        branch: &str,
        commit: &str,
    ) -> Result<u64, StoreError> {
        if branch == MAIN_BRANCH {
            return Err(StoreError::Internal(
                "cannot create the default 'main' branch — it always exists".to_owned(),
            ));
        }

        let path = self.dataset_path(domain);
        if !path.exists() {
            return Err(StoreError::Internal(format!(
                "cannot branch domain '{}': dataset does not exist (index the parent first)",
                domain
            )));
        }

        let tag = crate::layeridx::encode_commit_tag(commit);
        let uri = path.to_string_lossy().to_string();
        let mut ds = self.io_open_fresh(&uri).await?;

        use lance::dataset::refs::Ref;
        let created = ds
            .create_branch(branch, Ref::Tag(tag.clone()), None)
            .await
            .map_err(|e| {
                StoreError::Internal(format!(
                    "create_branch '{}' from tag '{}' failed: {}",
                    branch, tag, e
                ))
            })?;

        Ok(created.version().version)
    }

    /// Collect the set of physical data-file paths referenced by a branch's
    /// current head (layout A). Used to PROVE block reuse by path identity
    /// (P3-BR-1): a branch forked from a parent shares the parent's fragment
    /// files, so the two path sets intersect on the shared fragments.
    pub async fn io_branch_data_file_paths(
        &self,
        domain: &str,
        branch: &str,
    ) -> Result<std::collections::HashSet<String>, StoreError> {
        let path = self.dataset_path(domain);
        if !path.exists() {
            return Ok(std::collections::HashSet::new());
        }
        let uri = path.to_string_lossy().to_string();
        // Counted fresh open (block-reuse proof / diagnostic path, not a hot loop).
        let ds = self.io_open_fresh(&uri).await?;
        let ds = if branch == MAIN_BRANCH {
            ds
        } else {
            ds.checkout_branch(branch).await.map_err(|e| {
                StoreError::Internal(format!("checkout '{}' for data-file paths failed: {}", branch, e))
            })?
        };

        let mut set = std::collections::HashSet::new();
        for frag in ds.get_fragments() {
            for df in &frag.metadata().files {
                set.insert(df.path.clone());
            }
        }
        Ok(set)
    }

    /// Open a FRESH dataset handle directly from disk (NOT from the shared cache).
    /// Used by the background index worker to perform `optimize_indices` without
    /// holding the cached `Arc<RwLock<Dataset>>`'s write lock — which would block
    /// concurrent search read locks and stall queries during optimization.
    ///
    /// Layout A: opens the domain dataset and checks out `branch` so the optimize
    /// targets the right branch head. After optimization completes, the caller
    /// should refresh the cache via `io_refresh_cached_dataset`.
    ///
    /// Returns None if the dataset does not exist on disk (nothing to optimize).
    pub async fn io_open_dataset_uncached(
        &self,
        domain: &str,
        branch: &str,
    ) -> Result<Option<Dataset>, StoreError> {
        let path = self.dataset_path(domain);
        if !path.exists() {
            return Ok(None);
        }
        let uri = path.to_string_lossy().to_string();
        // Counted fresh open. This is deliberately uncached (the background
        // optimize worker must not hold the cached handle's write lock), but it is
        // still funnelled through `io_open_fresh` so it is VISIBLE to the FD guard.
        // It runs once per optimize cycle, not per query.
        let ds = self.io_open_fresh(&uri).await?;

        if branch == MAIN_BRANCH {
            return Ok(Some(ds));
        }

        let branch_ds = ds.checkout_branch(branch).await.map_err(|e| {
            StoreError::Internal(format!(
                "uncached checkout branch '{}' failed: {}",
                branch, e
            ))
        })?;
        Ok(Some(branch_ds))
    }

    /// Refresh the cached domain dataset handle by re-opening from disk at the
    /// default branch head. Called after the background worker completes
    /// optimization so subsequent reads see the new version/indices.
    ///
    /// `_branch` is accepted for signature stability; the cache is domain-keyed
    /// and always reflects the dataset's default branch (layout A).
    ///
    /// In-flight readers hold an `Arc::clone` of the old `Dataset`, so it
    /// stays alive until those operations complete. Lance's own internal
    /// caches (manifest, fragment metadata) may retain memory after drop —
    /// that is a Lance library issue, not fixable here.
    pub async fn io_refresh_cached_dataset(
        &self,
        domain: &str,
        _branch: &str,
    ) -> Result<(), StoreError> {
        let path = self.dataset_path(domain);
        if !path.exists() {
            return Ok(());
        }
        let uri = path.to_string_lossy().to_string();
        // Counted fresh open. Invalidate-on-write re-opens at the default branch
        // head so subsequent cached reads see the new version/tags; routed through
        // `io_open_fresh` so even this write-side re-open is VISIBLE to the FD guard.
        let ds = self.io_open_fresh(&uri).await?;

        let arc_ds = Arc::new(RwLock::new(ds));
        let mut datasets = self.datasets.write().await;
        datasets.insert(domain.to_owned(), arc_ds);
        Ok(())
    }

    /// Prune old untagged intermediate versions created by micro-batch appends.
    /// Called after each commit is tagged to prevent manifest growth across
    /// many commits. Retains the last 3 versions (for rollback safety) plus
    /// all tagged versions. Uses delete_unverified=true since we know no other
    /// operations are in progress (pipeline lock is held).
    ///
    /// **FIX-B**: Hides `_indices/` during cleanup to prevent Lance from
    /// deleting index files referenced by rebuild-branch tags.
    pub async fn io_cleanup_old_versions(
        &self,
        domain: &str,
        _branch: &str,
    ) -> Result<(), StoreError> {
        let path = self.dataset_path(domain);
        if !path.exists() {
            return Ok(());
        }
        let uri = path.to_string_lossy().to_string();
        let ds = self.io_open_fresh(&uri).await?;

        // FIX-B: Hide _indices/ during cleanup, but ONLY when rebuild
        // branches exist AND we're in CurrentCode mode. In TargetNoPatch/
        // TargetWithPatch modes, we skip hiding and instead set
        // clean_referenced_branches(false) to protect rebuild-branch files.
        let mode = self.cleanup_mode();
        let has_rebuild_branches = self
            .io_list_branches(domain)
            .await
            .map(|branches| branches.iter().any(|b| is_compact_rebuild_branch(b)))
            .unwrap_or(false);

        let indices_dir = path.join("_indices");
        let indices_backup = path.join("_indices_backup_fixb");
        let hid_indices = if mode.hide_indices() && has_rebuild_branches && indices_dir.exists() {
            std::fs::rename(&indices_dir, &indices_backup)
                .map_err(|e| StoreError::Internal(format!(
                    "cleanup old: failed to move _indices to backup: {}", e
                )))?;
            true
        } else {
            false
        };

        let mut policy_builder = CleanupPolicyBuilder::default()
            .retain_n_versions(&ds, 3).await
            .map_err(|e| StoreError::Internal(format!("cleanup policy build failed: {}", e)))?
            .delete_unverified(true)
            .error_if_tagged_old_versions(false);

        if mode.disable_clean_referenced_branches() {
            policy_builder = policy_builder.clean_referenced_branches(false);
        }

        let policy = policy_builder.build();

        let cleanup_result = ds.cleanup_with_policy(policy).await;

        // Restore _indices/.
        if hid_indices {
            if indices_dir.exists() {
                if let Ok(entries) = std::fs::read_dir(&indices_dir) {
                    for entry in entries.flatten() {
                        let entry_path = entry.path();
                        let dest = indices_backup.join(entry.file_name());
                        let _ = std::fs::rename(&entry_path, &dest);
                    }
                }
                let _ = std::fs::remove_dir(&indices_dir);
            }
            std::fs::rename(&indices_backup, &indices_dir)
                .map_err(|e| StoreError::Internal(format!(
                    "cleanup old: failed to restore _indices from backup: {}", e
                )))?;
        }

        cleanup_result
            .map_err(|e| StoreError::Internal(format!("version cleanup failed: {}", e)))?;

        Ok(())
    }

    /// Aggressive cleanup that retains only the current version plus tagged
    /// versions. Used after compaction to reclaim disk space from orphaned
    /// index files created by index rebuilds (drop_index + create_index).
    /// The standard io_cleanup_old_versions retains 3 versions which can
    /// keep old index files alive; this function retains only 1 to ensure
    /// orphaned index directories are removed.
    ///
    /// **FIX-B**: Temporarily moves `_indices/` out of the way before
    /// `cleanup_with_policy` so Lance cannot delete index files referenced
    /// by rebuild-branch tags. Lance's `retain_branch_lineage_files` only
    /// protects indices whose `base_id` matches the current branch URI —
    /// indices created on rebuild branches have a different `base_id` and
    /// are not protected. Index dir cleanup is exclusively handled by
    /// `io_prune_stale_index_dirs` (fail-closed from FIX-A).
    pub async fn io_cleanup_aggressive(
        &self,
        domain: &str,
        _branch: &str,
    ) -> Result<(), StoreError> {
        let path = self.dataset_path(domain);
        if !path.exists() {
            return Ok(());
        }
        let uri = path.to_string_lossy().to_string();
        let ds = self.io_open_fresh(&uri).await?;

        // FIX-B: Temporarily hide _indices/ from Lance's cleanup, but ONLY
        // when rebuild branches exist AND we're in CurrentCode mode. During
        // compaction, rebuild-branch indices have a different base_id and are
        // not protected by retain_branch_lineage_files. Between compactions
        // (regular pushes), no rebuild branches exist, so hiding is unnecessary.
        // In TargetNoPatch/TargetWithPatch modes, we skip hiding and instead
        // set clean_referenced_branches(false) to protect rebuild-branch files.
        let mode = self.cleanup_mode();
        let has_rebuild_branches = self
            .io_list_branches(domain)
            .await
            .map(|branches| branches.iter().any(|b| is_compact_rebuild_branch(b)))
            .unwrap_or(false);

        let indices_dir = path.join("_indices");
        let indices_backup = path.join("_indices_backup_fixb");
        let hid_indices = if mode.hide_indices() && has_rebuild_branches && indices_dir.exists() {
            std::fs::rename(&indices_dir, &indices_backup)
                .map_err(|e| StoreError::Internal(format!(
                    "aggressive cleanup: failed to move _indices to backup: {}", e
                )))?;
            true
        } else {
            false
        };

        let mut policy_builder = CleanupPolicyBuilder::default()
            .retain_n_versions(&ds, 1).await
            .map_err(|e| StoreError::Internal(format!("aggressive cleanup policy build failed: {}", e)))?
            .delete_unverified(true)
            .error_if_tagged_old_versions(false);

        if mode.disable_clean_referenced_branches() {
            policy_builder = policy_builder.clean_referenced_branches(false);
        }

        let policy = policy_builder.build();

        let cleanup_result = ds.cleanup_with_policy(policy).await;

        // Restore _indices/ immediately after cleanup.
        if hid_indices {
            if indices_dir.exists() {
                if let Ok(entries) = std::fs::read_dir(&indices_dir) {
                    for entry in entries.flatten() {
                        let entry_path = entry.path();
                        let dest = indices_backup.join(entry.file_name());
                        let _ = std::fs::rename(&entry_path, &dest);
                    }
                }
                let _ = std::fs::remove_dir(&indices_dir);
            }
            std::fs::rename(&indices_backup, &indices_dir)
                .map_err(|e| StoreError::Internal(format!(
                    "aggressive cleanup: failed to restore _indices from backup: {}", e
                )))?;
        }

        let stats = cleanup_result
            .map_err(|e| StoreError::Internal(format!("aggressive cleanup failed: {}", e)))?;

        eprintln!(
            "[cleanup_aggressive] main: removed {} bytes, {} old versions, {} data files, {} index files",
            stats.bytes_removed, stats.old_versions, stats.data_files_removed, stats.index_files_removed
        );

        // Prune stale index directories left behind by index optimization
        // (cascade merges create new UUIDs; old ones become orphaned).
        // This runs under the pipeline lock (caller holds it), so the
        // TOCTOU race with concurrent index creation is prevented.
        if let Err(e) = self.io_prune_stale_index_dirs(domain, _branch).await {
            eprintln!("[cleanup_aggressive] prune stale index dirs failed (soft): {}", e);
        }
        if let Err(e) = self.io_prune_empty_index_dirs(domain) {
            eprintln!("[cleanup_aggressive] prune empty index dirs failed (soft): {}", e);
        }

        Ok(())
    }

    /// Remove empty index UUID directories left behind by LanceDB's
    /// `cleanup_with_policy`. LanceDB deletes index files inside
    /// `_indices/<uuid>/` but leaves the empty `<uuid>` directories.
    /// Over many push/compaction cycles, hundreds of empty directories
    /// accumulate. This function scans `_indices/` and removes any
    /// directory that contains no files (recursively empty).
    ///
    /// Returns the count of directories removed.
    pub fn io_prune_empty_index_dirs(&self, domain: &str) -> Result<usize, StoreError> {
        let path = self.dataset_path(domain);
        let indices_dir = path.join("_indices");

        if !indices_dir.exists() {
            return Ok(0);
        }

        let mut removed = 0usize;
        for entry in std::fs::read_dir(&indices_dir)
            .map_err(|e| StoreError::Internal(format!("prune: read_dir failed: {}", e)))?
        {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("[cleanup] prune: skipping unreadable entry: {}", e);
                    continue;
                }
            };

            let dir_path = entry.path();
            if !dir_path.is_dir() {
                continue;
            }

            // Check if the directory contains any files (recursively).
            let has_files = std::fs::read_dir(&dir_path)
                .map(|mut it| it.any(|e| e.map(|e| e.path().is_file()).unwrap_or(false)))
                .unwrap_or(false);

            // Also check subdirectories for files (LanceDB index dirs
            // may have nested structure like <uuid>/0/index.lance).
            let has_files_recursive = if !has_files {
                dir_has_files_recursive(&dir_path)
            } else {
                true
            };

            if !has_files_recursive {
                match std::fs::remove_dir_all(&dir_path) {
                    Ok(_) => removed += 1,
                    Err(e) => {
                        eprintln!(
                            "[cleanup] prune: failed to remove empty dir {:?}: {}",
                            dir_path, e
                        );
                    }
                }
            }
        }

        if removed > 0 {
            eprintln!("[cleanup] pruned {} empty index directories", removed);
        }

        Ok(removed)
    }

    /// Remove stale index UUID directories that are not referenced by any
    /// live index on any tagged version across all branches. Unlike
    /// `io_prune_empty_index_dirs`, this also removes directories that
    /// still contain files — they are stale index files from prior
    /// compaction cycles that `cleanup_with_policy` failed to delete.
    ///
    /// This function collects live index UUIDs by scanning ALL tagged
    /// versions across ALL branches (not just HEAD). This is critical
    /// because boundary commits retain their original manifests, which
    /// may reference index UUIDs that differ from HEAD's indices.
    /// Deleting a UUID referenced only by a non-HEAD tagged manifest
    /// would silently break historical search at that boundary.
    ///
    /// MUST be called under the compaction lock — there is a TOCTOU race
    /// between `load_indices()` and the directory scan if a concurrent
    /// index creation produces a new UUID dir not yet in any manifest.
    ///
    /// Returns the count of directories removed.
    pub async fn io_prune_stale_index_dirs(
        &self,
        domain: &str,
        _branch: &str,
    ) -> Result<usize, StoreError> {
        let path = self.dataset_path(domain);
        let indices_dir = path.join("_indices");

        if !indices_dir.exists() {
            return Ok(0);
        }

        // Collect live index UUIDs from ALL tagged versions across ALL branches.
        // MUST use the strict (fail-closed) variant — if collection is incomplete
        // for any reason, we must NOT prune, or we risk deleting live index dirs.
        let live_uuids = self.io_collect_all_live_index_uuids_strict(domain).await?;

        eprintln!(
            "[cleanup] prune stale: {} live index UUIDs from all tagged versions",
            live_uuids.len(),
        );

        // Scan _indices and remove any directory not in the live set.
        let mut removed = 0usize;
        for entry in std::fs::read_dir(&indices_dir)
            .map_err(|e| StoreError::Internal(format!("prune stale: read_dir failed: {}", e)))?
        {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("[cleanup] prune stale: skipping unreadable entry: {}", e);
                    continue;
                }
            };

            let dir_path = entry.path();
            if !dir_path.is_dir() {
                continue;
            }

            let dir_name = entry.file_name();
            let dir_uuid = dir_name.to_string_lossy().to_string();

            if !live_uuids.contains(&dir_uuid) {
                eprintln!(
                    "[cleanup] prune stale: removing stale index dir {} (not in live set of {} UUIDs)",
                    dir_uuid, live_uuids.len()
                );
                match std::fs::remove_dir_all(&dir_path) {
                    Ok(_) => removed += 1,
                    Err(e) => {
                        eprintln!(
                            "[cleanup] prune stale: failed to remove dir {:?}: {}",
                            dir_path, e
                        );
                    }
                }
            }
        }

        if removed > 0 {
            eprintln!("[cleanup] pruned {} stale index directories", removed);
        }

        Ok(removed)
    }

    /// Create an empty RecordBatch with the chunk schema (for dataset initialization).
    pub(super) fn empty_batch(&self) -> RecordBatch {
        let schema = self.chunk_schema();
        let dim = self.dim as i32;

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(Vec::<&str>::new())),
                Arc::new(StringArray::from(Vec::<&str>::new())),
                Arc::new(Int32Array::from(Vec::<i32>::new())),
                Arc::new(Int32Array::from(Vec::<i32>::new())),
                Arc::new(Int32Array::from(Vec::<i32>::new())),
                Arc::new(Int32Array::from(Vec::<i32>::new())),
                Arc::new(FixedSizeListArray::new(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim,
                    Arc::new(Float32Array::from(Vec::<f32>::new())),
                    None,
                )),
                // clustering_embedding (same shape as embedding, stored).
                Arc::new(FixedSizeListArray::new(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim,
                    Arc::new(Float32Array::from(Vec::<f32>::new())),
                    None,
                )),
                Arc::new(StringArray::from(Vec::<&str>::new())),
            ],
        )
        .expect("empty batch construction must not fail")
    }

    /// Build a RecordBatch from chunk rows (includes both embedding columns).
    pub(super) fn rows_to_batch(&self, rows: &[ChunkRow]) -> Result<RecordBatch, StoreError> {
        let schema = self.chunk_schema();

        let doc_ids: Vec<&str> = rows.iter().map(|r| r.doc_id.as_str()).collect();
        let doc_types: Vec<&str> = rows.iter().map(|r| r.doc_type.as_str()).collect();
        let chunk_indexes: Vec<i32> = rows.iter().map(|r| r.chunk_index).collect();
        let chunk_counts: Vec<i32> = rows.iter().map(|r| r.chunk_count).collect();
        let chunk_token_starts: Vec<i32> = rows.iter().map(|r| r.chunk_token_start).collect();
        let doc_token_lens: Vec<i32> = rows.iter().map(|r| r.doc_token_len).collect();
        let contents: Vec<&str> = rows.iter().map(|r| r.content.as_str()).collect();

        // Build the embedding FixedSizeList (document-role, ANN-indexed).
        let flat_embeddings: Vec<f32> = rows.iter().flat_map(|r| r.embedding.iter().copied()).collect();
        let values = Float32Array::from(flat_embeddings);
        let embedding_array = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            self.dim as i32,
            Arc::new(values),
            None,
        );

        // Build the clustering_embedding FixedSizeList (clustering-role, stored).
        let flat_clustering_embeddings: Vec<f32> = rows
            .iter()
            .flat_map(|r| r.clustering_embedding.iter().copied())
            .collect();
        let clustering_values = Float32Array::from(flat_clustering_embeddings);
        let clustering_embedding_array = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            self.dim as i32,
            Arc::new(clustering_values),
            None,
        );

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(doc_ids)),
                Arc::new(StringArray::from(doc_types)),
                Arc::new(Int32Array::from(chunk_indexes)),
                Arc::new(Int32Array::from(chunk_counts)),
                Arc::new(Int32Array::from(chunk_token_starts)),
                Arc::new(Int32Array::from(doc_token_lens)),
                Arc::new(embedding_array) as Arc<dyn arrow_array::Array>,
                Arc::new(clustering_embedding_array) as Arc<dyn arrow_array::Array>,
                Arc::new(StringArray::from(contents)),
            ],
        )
        .map_err(|e| StoreError::Internal(format!("batch construction failed: {}", e)))
    }
}

/// URI-percent-encode each segment of a domain path (`org/team`) so any
/// character is safe in a filesystem directory name. The `/` separator is
/// preserved as the segment boundary.
///
/// This replaces the old `domain.replace('/', "__")` encoding. Existing
/// on-disk datasets using `__` are incompatible — a clean re-index is required.
pub fn encode_domain_path(domain: &str) -> String {
    use form_urlencoded::byte_serialize;
    domain
        .split('/')
        .map(|segment| {
            // Encode each segment. `byte_serialize` percent-encodes everything
            // that is not in the unreserved set (alphanumerics + `-._~`).
            // This is safe for filesystem paths on all platforms.
            byte_serialize(segment.as_bytes()).collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("__")
}

/// Decode a filesystem-safe domain path back to the original domain string.
/// Reverses `encode_domain_path`: splits by `__`, percent-decodes each segment,
/// joins with `/`. Returns `None` if any segment fails to decode.
pub fn decode_domain_path(encoded: &str) -> Option<String> {
    use form_urlencoded::parse;
    let segments: Vec<Option<String>> = encoded
        .split("__")
        .map(|seg| {
            if seg.is_empty() {
                Some(String::new())
            } else {
                let decoded = parse(seg.as_bytes())
                    .map(|(key, val)| {
                        if val.is_empty() {
                            key.into_owned()
                        } else {
                            format!("{}={}", key, val)
                        }
                    })
                    .collect::<Vec<_>>();
                if decoded.is_empty() {
                    None
                } else {
                    Some(decoded.join(""))
                }
            }
        })
        .collect();

    if segments.iter().any(|s| s.is_none()) {
        return None;
    }

    Some(
        segments
            .into_iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>()
            .join("/"),
    )
}

/// Recursively check if a directory contains any files.
/// Returns true if at least one file is found at any depth.
fn dir_has_files_recursive(path: &std::path::Path) -> bool {
    let Ok(entries) = std::fs::read_dir(path) else {
        return false;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_file() {
            return true;
        }
        if p.is_dir() && dir_has_files_recursive(&p) {
            return true;
        }
    }
    false
}
