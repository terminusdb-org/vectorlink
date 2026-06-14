#![forbid(unsafe_code)]

//! LanceDB-backed persistent store (single-branch linear history).
//!
//! Schema: one row per chunk, keyed by (doc_id, chunk_index).
//! Supports vector search, FTS, and hybrid search. Commit→version binding via
//! Lance tags (managed by layeridx).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{
    Array, FixedSizeListArray, Float32Array, Int32Array, RecordBatch, RecordBatchIterator,
    StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance::index::DatasetIndexExt;
use lance_index::IndexType;
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::{FullTextSearchQuery, InvertedIndexParams};
use lance_linalg::distance::DistanceType;
use tokio::sync::RwLock;

use crate::kernel::error::StoreError;
use crate::kernel::model::{
    BranchName, ChunkInfo, Domain, LastIndexed, SearchHit, SearchMode, Statistics, TaskStatus,
};
use crate::layeridx::{self, BranchIndex};

/// Configuration for the vector ANN index (IVF_PQ).
/// Parameters are pinned as constants — changing them perturbs ranking
/// and should be treated like a model version bump.
#[derive(Debug, Clone)]
pub struct VectorIndexConfig {
    /// Number of IVF partitions. More partitions = faster search at the cost of
    /// index build time and recall. Recommended: sqrt(n) for corpus size n.
    pub num_partitions: usize,
    /// Number of PQ sub-vectors. Must divide the embedding dimension evenly.
    pub num_sub_vectors: usize,
    /// Number of probes during search (how many partitions to scan).
    /// Higher = better recall, slower search.
    pub nprobes: usize,
    /// Refine factor: re-rank this many candidates with full-precision vectors.
    /// Higher = better recall at the cost of latency. None = no refinement.
    pub refine_factor: Option<u32>,
}

impl VectorIndexConfig {
    /// Sane defaults for a given embedding dimension.
    /// These are pinned — treat changes as a model bump.
    ///
    /// INVARIANT: `num_sub_vectors` always divides `dim` evenly (PQ requirement).
    /// If `dim` is not evenly divisible by the target sub-vector count, we find
    /// the largest divisor of `dim` that is <= the target. This guarantees the
    /// index build never fails due to dimension/sub-vector mismatch.
    ///
    /// # Panics
    ///
    /// Panics if `dim == 0`. This is called at service startup; a zero-dim
    /// embedding is a configuration error that must fail loud at boot.
    pub fn default_for_dim(dim: usize) -> Self {
        assert!(dim > 0, "embedding dimension must be > 0 (check TDB_SEARCH_DIM)");

        // Target sub-vector counts by dimension range:
        // 128-d → target 16 (8 dims per sub-vector)
        // 256-d → target 32 (8 dims each)
        // 768-d → target 48 (16 dims each)
        // >768  → target dim/16
        let target = match dim {
            d if d <= 128 => (d / 8).max(1),
            d if d <= 256 => (d / 8).max(1),
            d if d <= 768 => (d / 16).max(1),
            d => (d / 16).max(1),
        };

        // Find the largest divisor of `dim` that is <= target.
        // This guarantees dim % num_sub_vectors == 0 (PQ requirement).
        let num_sub_vectors = largest_divisor_leq(dim, target);

        Self {
            // Start with 16 partitions; scale up with corpus via
            // `recommended_num_partitions` when corpus size is known.
            num_partitions: 16,
            num_sub_vectors,
            nprobes: 8,
            refine_factor: Some(10),
        }
    }
}

/// Find the largest divisor of `n` that is <= `target`.
/// Returns 1 if no divisor in [2, target] exists (1 always divides anything).
///
/// Used to ensure `dim % num_sub_vectors == 0` for PQ index creation.
fn largest_divisor_leq(n: usize, target: usize) -> usize {
    // Search downward from target to find a divisor of n.
    // For typical embedding dimensions (128, 256, 384, 512, 768, 1024, 1536)
    // this terminates quickly because they have many small factors.
    let mut candidate = target;
    while candidate > 1 {
        if n.is_multiple_of(candidate) {
            return candidate;
        }
        candidate -= 1;
    }
    // 1 always divides any positive number.
    1
}

/// A chunk row ready for insertion into Lance.
#[derive(Debug, Clone)]
pub struct ChunkRow {
    pub doc_id: String,
    pub doc_type: String,
    pub chunk_index: i32,
    pub chunk_count: i32,
    pub chunk_token_start: i32,
    pub doc_token_len: i32,
    pub embedding: Vec<f32>,
    pub content: String,
}

/// Search query parameters.
#[derive(Debug, Clone)]
pub struct SearchQuery {
    pub query_embedding: Vec<f32>,
    pub query_text: String,
    pub mode: SearchMode,
    pub start: usize,
    pub count: usize,
    pub doc_type_filter: Vec<String>,
    pub doc_id_filter: Vec<String>,
    pub snippet: bool,
}

/// Whether a ChunkHit's distance is raw (needs transform) or already normalised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceKind {
    /// Raw Lance cosine distance [0, 2] — needs `normalized_cosine_from_lance`.
    RawCosine,
    /// Already normalised to [0, 1] (e.g., from RRF or FTS conversion).
    Normalised,
}

/// Internal chunk-level hit before dedup to documents.
#[derive(Debug, Clone)]
pub struct ChunkHit {
    pub doc_id: String,
    pub distance: f32,
    pub distance_kind: DistanceKind,
    pub chunk_index: i32,
    pub chunk_count: i32,
    pub chunk_token_start: i32,
    pub doc_token_len: i32,
    pub content: String,
}

/// Per-branch metadata key: (domain_str, branch_str).
/// Used for last-indexed, pipeline locks, and index tracking — branch-precise
/// state per RISK-22. The Lance DATASET itself is domain-keyed (layout A): one
/// `{domain}.lance` dataset holds all TerminusDB branches as Lance branches.
type BranchKey = (String, String);

/// Default Lance branch name. Layout (A) maps TerminusDB's `main` branch to
/// the Lance dataset's native default branch.
pub const MAIN_BRANCH: &str = "main";

/// The Lance-backed store (layout A: one dataset per domain, TerminusDB
/// branches as Lance branches inside it; tags are dataset-global).
#[derive(Debug)]
pub struct LanceStore {
    base_dir: PathBuf,
    dim: usize,
    /// Vector index configuration (nprobes, refine_factor for search).
    vector_index_config: VectorIndexConfig,
    /// Open dataset handles, keyed by DOMAIN (layout A). The cached handle is
    /// the domain dataset opened at its default (main) branch head; branch
    /// writes check out a branch-bound handle without mutating this one.
    datasets: RwLock<HashMap<String, Arc<RwLock<Dataset>>>>,
    /// Per-(domain, branch) index tracking (branch-precise).
    branch_indexes: RwLock<HashMap<BranchKey, BranchIndex>>,
    /// Tasks by task ID.
    tasks: RwLock<HashMap<String, TaskStatus>>,
    /// Per-(domain, branch) pipeline serialisation lock.
    /// Ensures concurrent pushes to the same branch are serialised so that
    /// commit→version tags are correctly isolated.
    pipeline_locks: RwLock<HashMap<BranchKey, Arc<tokio::sync::Mutex<()>>>>,
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
    inflight_commits: RwLock<std::collections::HashSet<(String, String, String)>>,
    /// Serialises the atomic check-and-reserve in `io_try_reserve_commit` so two
    /// concurrent pushes of the SAME (domain, branch, commit) cannot both pass.
    reservation_lock: tokio::sync::Mutex<()>,
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
}

impl LanceStore {
    /// Create a new LanceStore backed by the given directory.
    pub fn new(base_dir: &Path, dim: usize) -> Self {
        let vector_index_config = VectorIndexConfig::default_for_dim(dim);
        Self {
            base_dir: base_dir.to_owned(),
            dim,
            vector_index_config,
            datasets: RwLock::new(HashMap::new()),
            branch_indexes: RwLock::new(HashMap::new()),
            tasks: RwLock::new(HashMap::new()),
            pipeline_locks: RwLock::new(HashMap::new()),
            domain_guards: RwLock::new(HashMap::new()),
            inflight_commits: RwLock::new(std::collections::HashSet::new()),
            reservation_lock: tokio::sync::Mutex::new(()),
            fresh_open_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Open the domain dataset FRESH from disk, tracking the open for FD/perf
    /// instrumentation. Every `Dataset::open` allocates a fresh object_store +
    /// session (with their own file readers), so this is the FD-pressure entry
    /// point we minimise (BUG-FD24). Callers that can reuse a cached handle MUST
    /// do so via `io_open_dataset_readonly` instead.
    async fn io_open_fresh(&self, uri: &str) -> Result<Dataset, StoreError> {
        self.fresh_open_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Dataset::open(uri)
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

    /// Atomically check-and-reserve a commit for indexing (the 409 state machine).
    ///
    /// A commit is rejected (returns `Ok(false)` — caller must respond 409) if it
    /// is in ANY non-absent state:
    ///   - Reserved/Indexing: present in `inflight_commits` (a push is in flight).
    ///   - Indexed: a durable Lance tag exists for it (`io_resolve_commit`).
    ///
    /// Otherwise the commit is absent: we INSERT the reservation and return
    /// `Ok(true)` (caller proceeds to spawn the index pipeline).
    ///
    /// ATOMICITY: the whole check-and-insert runs under `reservation_lock`, so two
    /// concurrent pushes of the same (domain, branch, commit) cannot both observe
    /// "absent" — exactly one wins the reservation, the other gets `Ok(false)`.
    ///
    /// FAIL-LOUD: a real tag-resolution error (I/O / corruption) propagates as
    /// `Err` — it is NOT collapsed into "absent" (which would let a re-push of a
    /// possibly-indexed commit through). The reservation is only taken on a proven
    /// absence.
    pub async fn io_try_reserve_commit(
        &self,
        domain: &str,
        branch: &str,
        commit: &str,
    ) -> Result<bool, StoreError> {
        let _atomic = self.reservation_lock.lock().await;

        // In-flight (Reserved/Indexing) → reject.
        let key = (domain.to_owned(), branch.to_owned(), commit.to_owned());
        {
            let inflight = self.inflight_commits.read().await;
            if inflight.contains(&key) {
                return Ok(false);
            }
        }

        // Durably Indexed (tagged) → reject. Fail-loud on a real resolution error.
        if self.io_resolve_commit(domain, branch, commit).await?.is_some() {
            return Ok(false);
        }

        // Absent → reserve and proceed.
        let mut inflight = self.inflight_commits.write().await;
        inflight.insert(key);
        Ok(true)
    }

    /// Release a commit reservation (terminal state of its push).
    ///
    /// Called on BOTH success and failure of the index pipeline:
    ///   - SUCCESS: the durable Lance tag now marks the commit Indexed, so the
    ///     in-flight reservation is no longer needed (the tag keeps the 409 guard
    ///     correct). Dropping it keeps the set bounded.
    ///   - FAILURE (task → Error): no tag was written, so dropping the reservation
    ///     returns the commit to the absent state — a legitimate retry of the same
    ///     commit is then allowed (NOT blocked forever).
    ///
    /// Idempotent: releasing an absent reservation is a no-op.
    pub async fn io_release_commit_reservation(&self, domain: &str, branch: &str, commit: &str) {
        let key = (domain.to_owned(), branch.to_owned(), commit.to_owned());
        let mut inflight = self.inflight_commits.write().await;
        inflight.remove(&key);
    }

    /// Acquire the per-domain create/delete guard (BLOCKER-2 / #6). Held by a
    /// dataset-creating write across the create, and by `DELETE /domain` across
    /// the whole remove-then-purge — so the two never interleave.
    async fn acquire_domain_guard(&self, domain: &str) -> tokio::sync::OwnedMutexGuard<()> {
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

    /// Acquire the per-(domain, branch) pipeline lock.
    /// Serialises upsert→tag operations so concurrent pushes don't interleave.
    pub async fn acquire_pipeline_lock(
        &self,
        domain: &str,
        branch: &str,
    ) -> tokio::sync::OwnedMutexGuard<()> {
        let key: BranchKey = (domain.to_owned(), branch.to_owned());

        // Get or create the lock for this key.
        let lock = {
            let locks = self.pipeline_locks.read().await;
            if let Some(l) = locks.get(&key) {
                Arc::clone(l)
            } else {
                drop(locks);
                let mut locks = self.pipeline_locks.write().await;
                let l = locks
                    .entry(key)
                    .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
                    .clone();
                l
            }
        };

        lock.lock_owned().await
    }

    /// Get the Arrow schema for chunk rows (embedding dimension from config).
    pub fn chunk_schema(&self) -> Arc<Schema> {
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
            Field::new("content", DataType::Utf8, false),
        ]))
    }

    /// Get the on-disk path for a domain's Lance dataset (layout A: one dataset
    /// per domain; branches live inside it as Lance branches).
    fn dataset_path(&self, domain: &str) -> PathBuf {
        // Use a safe directory name: domain slashes replaced with double-underscore.
        let safe_domain = domain.replace('/', "__");
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

        // Try to open existing dataset.
        let ds = if path.exists() {
            Dataset::open(&uri)
                .await
                .map_err(|e| StoreError::Internal(format!("failed to open dataset: {}", e)))?
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
        let ds = Dataset::open(&uri)
            .await
            .map_err(|e| StoreError::Internal(format!("failed to open dataset: {}", e)))?;

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
    async fn io_open_branch_for_write(
        &self,
        domain: &str,
        branch: &str,
    ) -> Result<Dataset, StoreError> {
        let path = self.dataset_path(domain);
        let uri = path.to_string_lossy().to_string();
        let ds = Dataset::open(&uri)
            .await
            .map_err(|e| StoreError::Internal(format!("failed to open dataset for write: {}", e)))?;

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

    /// List the Lance branch names that currently exist in the domain dataset.
    /// Returns an empty list if the dataset doesn't exist yet.
    pub async fn io_list_branches(&self, domain: &str) -> Result<Vec<String>, StoreError> {
        let path = self.dataset_path(domain);
        if !path.exists() {
            return Ok(Vec::new());
        }
        let uri = path.to_string_lossy().to_string();
        let ds = Dataset::open(&uri)
            .await
            .map_err(|e| StoreError::Internal(format!("open for list_branches failed: {}", e)))?;
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
        let mut ds = Dataset::open(&uri)
            .await
            .map_err(|e| StoreError::Internal(format!("open for create_branch failed: {}", e)))?;

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
        let ds = Dataset::open(&uri)
            .await
            .map_err(|e| StoreError::Internal(format!("open for data-file paths failed: {}", e)))?;
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
        let ds = Dataset::open(&uri)
            .await
            .map_err(|e| StoreError::Internal(format!("uncached open failed: {}", e)))?;

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
        let ds = Dataset::open(&uri)
            .await
            .map_err(|e| StoreError::Internal(format!("refresh open failed: {}", e)))?;

        let arc_ds = Arc::new(RwLock::new(ds));
        let mut datasets = self.datasets.write().await;
        datasets.insert(domain.to_owned(), arc_ds);
        Ok(())
    }

    /// Create an empty RecordBatch with the chunk schema (for dataset initialization).
    fn empty_batch(&self) -> RecordBatch {
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
                Arc::new(StringArray::from(Vec::<&str>::new())),
            ],
        )
        .expect("empty batch construction must not fail")
    }

    /// Build a RecordBatch from chunk rows.
    fn rows_to_batch(&self, rows: &[ChunkRow]) -> Result<RecordBatch, StoreError> {
        let schema = self.chunk_schema();

        let doc_ids: Vec<&str> = rows.iter().map(|r| r.doc_id.as_str()).collect();
        let doc_types: Vec<&str> = rows.iter().map(|r| r.doc_type.as_str()).collect();
        let chunk_indexes: Vec<i32> = rows.iter().map(|r| r.chunk_index).collect();
        let chunk_counts: Vec<i32> = rows.iter().map(|r| r.chunk_count).collect();
        let chunk_token_starts: Vec<i32> = rows.iter().map(|r| r.chunk_token_start).collect();
        let doc_token_lens: Vec<i32> = rows.iter().map(|r| r.doc_token_len).collect();
        let contents: Vec<&str> = rows.iter().map(|r| r.content.as_str()).collect();

        // Build the embedding FixedSizeList.
        let flat_embeddings: Vec<f32> = rows.iter().flat_map(|r| r.embedding.iter().copied()).collect();
        let values = Float32Array::from(flat_embeddings);
        let embedding_array = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            self.dim as i32,
            Arc::new(values),
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
                Arc::new(StringArray::from(contents)),
            ],
        )
        .map_err(|e| StoreError::Internal(format!("batch construction failed: {}", e)))
    }

    /// Upsert chunk rows for a document on `branch` (layout A). First deletes all
    /// existing rows for the doc_id, then appends the new rows — the
    /// delete-then-append that implements real `Changed` (replace full chunk set,
    /// no stale chunks; RISK-13).
    ///
    /// Writes target `branch`'s head via a branch-bound handle so sibling
    /// branches are untouched. Ensures the domain dataset exists first.
    pub async fn io_upsert_chunks(
        &self,
        domain: &str,
        branch: &str,
        doc_id: &str,
        rows: &[ChunkRow],
    ) -> Result<u64, StoreError> {
        if rows.is_empty() {
            return Ok(0);
        }

        // Serialise dataset creation against DELETE /domain (BLOCKER-2 / #6):
        // hold the per-domain guard across the ensure-exists so a concurrent
        // delete can't observe a half-created dataset.
        let _domain_guard = self.acquire_domain_guard(domain).await;

        // Ensure the domain dataset exists (creates the main branch on first use).
        self.io_open_dataset(domain, branch).await?;

        // Open a fresh branch-bound handle so the write targets `branch`'s head.
        let mut ds = self.io_open_branch_for_write(domain, branch).await?;

        // Delete existing rows for this doc_id (replace semantics).
        let filter = format!("doc_id = '{}'", doc_id.replace('\'', "''"));
        ds.delete(&filter)
            .await
            .map_err(|e| StoreError::Internal(format!("delete failed: {}", e)))?;

        // Append new rows.
        let batch = self.rows_to_batch(rows)?;
        let schema = self.chunk_schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        ds.append(reader, None)
            .await
            .map_err(|e| StoreError::Internal(format!("append failed: {}", e)))?;

        // Keep the cache consistent with the branch we just advanced (main).
        // For non-main branches the cached main handle is unaffected; refreshing
        // is harmless (re-opens at default branch head).
        self.io_refresh_cached_dataset(domain, branch).await?;

        Ok(ds.version().version)
    }

    /// Delete all chunks for a doc_id on `branch` (`Deleted` op; RISK-13).
    /// Writes target `branch`'s head via a branch-bound handle.
    pub async fn io_delete_doc(
        &self,
        domain: &str,
        branch: &str,
        doc_id: &str,
    ) -> Result<u64, StoreError> {
        // A delete against a domain that does not exist is a no-op — it must NOT
        // create the dataset (BLOCKER-2 resurrection guard). Only a genuine
        // insert/change creates a domain.
        if self.io_open_dataset_readonly(domain).await?.is_none() {
            return Ok(0);
        }

        let mut ds = self.io_open_branch_for_write(domain, branch).await?;

        let filter = format!("doc_id = '{}'", doc_id.replace('\'', "''"));
        ds.delete(&filter)
            .await
            .map_err(|e| StoreError::Internal(format!("delete failed: {}", e)))?;

        self.io_refresh_cached_dataset(domain, branch).await?;

        Ok(ds.version().version)
    }

    /// Bind a commit to a Lance version via a tag.
    ///
    /// Layout A: Lance versions are scoped to the branch that created them (the
    /// version number `v` means "version v on `branch`'s lineage"). The tag MUST
    /// therefore be created on a handle checked out to `branch` — otherwise Lance
    /// rejects it ("version <branch>:<v> does not exist"). Tag *resolution* is
    /// dataset-global (a tag created on any branch resolves from any other —
    /// Phase-0 spike 0a2.2), which is what gives us branch-from-anywhere.
    pub async fn io_tag_commit(
        &self,
        domain: &str,
        branch: &str,
        commit: &str,
        version: u64,
    ) -> Result<(), StoreError> {
        let tag = layeridx::encode_commit_tag(commit);

        if branch == MAIN_BRANCH {
            // Default branch — the cached handle is on main; tag there.
            {
                let ds_arc = self.io_open_dataset(domain, branch).await?;
                let ds = ds_arc.read().await;
                ds.tags()
                    .create(&tag, version)
                    .await
                    .map_err(|e| StoreError::Internal(format!("tag creation failed: {}", e)))?;
            }
            // Invalidate-on-write (BUG-FD24): refresh the cached handle so a
            // subsequent cached-handle read (resolve/list/search) is guaranteed to
            // see this tag. Tag listing reads disk live, but refreshing makes the
            // contract explicit and robust against any future handle-level caching.
            self.io_refresh_cached_dataset(domain, branch).await?;
            return Ok(());
        }

        // Non-main: create the tag on a branch-bound handle so `version` resolves
        // on the branch's own lineage.
        let path = self.dataset_path(domain);
        let uri = path.to_string_lossy().to_string();
        let base = self.io_open_fresh(&uri).await?;
        let branch_ds = base.checkout_branch(branch).await.map_err(|e| {
            StoreError::Internal(format!("checkout '{}' for tag failed: {}", branch, e))
        })?;
        branch_ds
            .tags()
            .create(&tag, version)
            .await
            .map_err(|e| StoreError::Internal(format!("tag creation failed: {}", e)))?;
        // Invalidate-on-write (BUG-FD24): tag resolution is dataset-global, so the
        // cached (default-branch) handle must be refreshed to guarantee this
        // branch-scoped tag is visible to subsequent cached-handle reads.
        self.io_refresh_cached_dataset(domain, branch).await?;
        Ok(())
    }

    /// Assign `target_commit` to point at `source_commit`'s already-indexed
    /// version — a pure tag pointer, NO data movement and NO embedding
    /// (`/assign` semantics; P3-ASSIGN-1). Resolves the source via the
    /// dataset-global tag and tags the target to the same version.
    ///
    /// INVARIANT (fail loud): the source commit MUST already be indexed; an
    /// unindexed source is an error, never a silent no-op.
    ///
    /// This touches only Lance tags (no `io_upsert_chunks`, no `io_embed`), so by
    /// construction `/assign` performs zero embed-provider calls and creates no
    /// new dataset version.
    pub async fn io_assign_commit(
        &self,
        domain: &str,
        branch: &str,
        source_commit: &str,
        target_commit: &str,
    ) -> Result<u64, StoreError> {
        let source_version = self
            .io_resolve_commit(domain, branch, source_commit)
            .await?
            .ok_or_else(|| {
                StoreError::Internal(format!(
                    "cannot assign: source commit '{}' is not indexed",
                    source_commit
                ))
            })?;

        self.io_tag_commit(domain, branch, target_commit, source_version)
            .await?;

        Ok(source_version)
    }

    /// Ensure an FTS (INVERTED) index exists on the "content" column and is
    /// up-to-date with all fragments. On first call, creates the index; on
    /// subsequent calls, incrementally indexes new (unindexed) fragments via
    /// `optimize_indices` — O(new_data), not O(corpus).
    ///
    /// Lance tracks which fragments are covered by the index via a bitmap.
    /// Queries always scan unindexed fragments via brute-force, so correctness
    /// is guaranteed even before optimize runs. This call improves FTS query
    /// performance by ensuring all fragments are indexed.
    /// Ensure the FTS inverted index EXISTS (create if not present, no-op otherwise).
    /// Used inline during push to guarantee FTS is available for search immediately.
    /// Does NOT call optimize_indices — that's deferred to the background worker.
    ///
    /// Unlike vector search (which flat-scans unindexed fragments), FTS REQUIRES
    /// the inverted index to exist. Calling this inline on each push is O(1) after
    /// the first creation because it's a metadata check only.
    pub async fn io_ensure_fts_index_created(
        &self,
        domain: &str,
        branch: &str,
    ) -> Result<u64, StoreError> {
        let ds_arc = self.io_open_dataset(domain, branch).await?;
        let mut ds = ds_arc.write().await;

        let indices = ds.load_indices().await
            .map_err(|e| StoreError::Internal(format!("load indices failed: {}", e)))?;
        let has_fts = indices.iter().any(|idx| idx.name == "content_fts");

        if !has_fts {
            // First time: create the inverted index covering all current fragments.
            let params = InvertedIndexParams::default();
            ds.create_index(
                &["content"],
                IndexType::Inverted,
                Some("content_fts".to_owned()),
                &params,
                false,
            )
            .await
            .map_err(|e| StoreError::Internal(format!("FTS index creation failed: {}", e)))?;
        }

        Ok(ds.version().version)
    }

    /// Ensure the FTS inverted index is fully up-to-date: create if missing, then
    /// incrementally index any new (unindexed) fragments via optimize_indices(append).
    /// Called by the background index worker — NOT on the push hot path.
    pub async fn io_ensure_fts_index(
        &self,
        domain: &str,
        branch: &str,
    ) -> Result<u64, StoreError> {
        let ds_arc = self.io_open_dataset(domain, branch).await?;
        let mut ds = ds_arc.write().await;

        // Check if the FTS index already exists.
        let indices = ds.load_indices().await
            .map_err(|e| StoreError::Internal(format!("load indices failed: {}", e)))?;
        let has_fts = indices.iter().any(|idx| idx.name == "content_fts");

        if !has_fts {
            // First time: create the inverted index.
            let params = InvertedIndexParams::default();
            ds.create_index(
                &["content"],
                IndexType::Inverted,
                Some("content_fts".to_owned()),
                &params,
                false,
            )
            .await
            .map_err(|e| StoreError::Internal(format!("FTS index creation failed: {}", e)))?;
        } else {
            // Index exists: incrementally index only new (unindexed) fragments.
            // OptimizeOptions::append() sets num_indices_to_merge=0, meaning:
            // create a new delta index from unindexed fragments only (O(delta)).
            ds.optimize_indices(&OptimizeOptions::append())
                .await
                .map_err(|e| StoreError::Internal(format!("FTS index optimize failed: {}", e)))?;
        }

        Ok(ds.version().version)
    }

    /// Resolve a commit to a Lance version via tag lookup.
    ///
    /// Returns `Ok(Some(version))` if the commit is indexed (its tag exists),
    /// `Ok(None)` if the commit is genuinely NOT indexed (no such tag), and
    /// `Err(..)` for any REAL failure (corrupt manifest, I/O, lock poison).
    ///
    /// FAIL-LOUD (BLOCKER-3): we resolve via the full tag list rather than a
    /// per-tag `get_version` whose `Err` cannot be distinguished from genuine
    /// absence. A `tags().list()` error is a real error and propagates; a tag
    /// missing from a successfully-listed map is genuine "not indexed". This
    /// closes the class where a transient error silently downgraded a search to
    /// "not indexed" (and, combined with catch-up, silently served stale data).
    ///
    /// READ-ONLY: never auto-creates the dataset (BLOCKER-2 resurrection guard).
    /// A domain with no dataset on disk resolves to `Ok(None)`.
    ///
    /// Reads the CACHED domain handle (BUG-FD24): a fresh `Dataset::open` per
    /// resolve spins up a new object_store + session and leaks file descriptors
    /// under load. TAG VISIBILITY is preserved because every mutation that writes
    /// a tag/version (`io_tag_commit`, `io_upsert_chunks`, `io_delete_doc`,
    /// optimize, assign) refreshes the cached handle via `io_refresh_cached_dataset`
    /// — so a commit tagged by the worker is visible to a subsequent resolve. This
    /// is the invalidate-on-write contract that lets reads reuse the cache without
    /// regressing the 409 guard (which previously required fresh-open to avoid a
    /// stale listing).
    pub async fn io_resolve_commit(
        &self,
        domain: &str,
        _branch: &str,
        commit: &str,
    ) -> Result<Option<u64>, StoreError> {
        let cached = match self.io_open_dataset_readonly(domain).await? {
            Some(ds) => ds,
            None => return Ok(None), // No dataset on disk → genuinely not indexed.
        };
        let ds = cached.read().await;

        let tag = layeridx::encode_commit_tag(commit);
        // `list()` reads the refs directory; a failure here is a REAL error
        // (I/O / corruption), surfaced loudly — NOT collapsed into "not indexed".
        let tags = ds
            .tags()
            .list()
            .await
            .map_err(|e| StoreError::Internal(format!("tag list failed: {}", e)))?;
        Ok(tags.get(&tag).map(|contents| contents.version))
    }

    /// List ALL indexed commit→version mappings for a domain (decoded commit
    /// ids → Lance version). One I/O read of the dataset-global tag set, used by
    /// the catch-up resolver to walk an ancestor window purely in memory rather
    /// than issuing one tag lookup per candidate.
    ///
    /// FAIL-LOUD: a `list()` error propagates; a tag whose name does not decode
    /// to a commit id is skipped (it was not written by `encode_commit_tag` —
    /// e.g. a Lance-internal ref), never treated as an error.
    ///
    /// READ-ONLY: never auto-creates. Absent dataset → empty map.
    ///
    /// Reads the CACHED domain handle (BUG-FD24), not a fresh `Dataset::open`
    /// per call. `tags().list()` reads the on-disk refs directory live each call
    /// (`ObjectStore::read_dir` → `list_with_delimiter`, no listing cache), and the
    /// cache is refreshed by every tag/version mutation, so a recently-tagged
    /// commit is visible to catch-up resolution without opening fresh per query.
    pub async fn io_list_commit_versions(
        &self,
        domain: &str,
    ) -> Result<HashMap<String, u64>, StoreError> {
        let cached = match self.io_open_dataset_readonly(domain).await? {
            Some(ds) => ds,
            None => return Ok(HashMap::new()),
        };
        let ds = cached.read().await;
        let tags = ds
            .tags()
            .list()
            .await
            .map_err(|e| StoreError::Internal(format!("tag list failed: {}", e)))?;

        let mut out = HashMap::with_capacity(tags.len());
        for (tag_name, contents) in tags {
            // Only our commit tags decode; skip anything else (fail-soft on
            // decode is correct here — a non-commit ref is simply not a commit).
            if let Ok(commit) = layeridx::decode_commit_tag(&tag_name) {
                out.insert(commit, contents.version);
            }
        }
        Ok(out)
    }

    /// Delete a domain's ENTIRE store footprint (layout A): drop the single
    /// `{domain}.lance` dataset (all branches/versions/tags) AND purge the
    /// domain's in-memory state (dataset cache entry, every `(domain, *)` entry
    /// in `branch_indexes` and `pipeline_locks`).
    ///
    /// IDEMPOTENT: deleting an unknown/already-removed domain succeeds (returns
    /// `Ok(())`) — TerminusDB may retry. We fail loud only on a genuine I/O error
    /// (the directory exists but cannot be removed).
    ///
    /// FAIL-LOUD partial-delete guard: the on-disk dataset is removed FIRST; only
    /// if that succeeds (or there was nothing on disk) do we purge in-memory
    /// state. If the disk removal errors, we surface it and leave in-memory state
    /// intact — never a half-deleted footprint (searchable map entry with no
    /// dataset, or a dataset with no map entry).
    pub async fn io_delete_domain(&self, domain: &str) -> Result<(), StoreError> {
        // Hold the per-domain guard across the WHOLE remove-then-purge so a
        // concurrent first-write that creates the dataset cannot interleave
        // (BLOCKER-2 / #6): either the create completes before we remove (and we
        // remove it), or it waits until after we have removed+purged (and then
        // legitimately re-creates as a fresh index). Never a half-deleted state.
        let _guard = self.acquire_domain_guard(domain).await;

        // 1. Drop the on-disk dataset directory first.
        let path = self.dataset_path(domain);
        if path.exists() {
            tokio::fs::remove_dir_all(&path).await.map_err(|e| {
                StoreError::Internal(format!(
                    "failed to remove dataset directory {}: {}",
                    path.display(),
                    e
                ))
            })?;
        }

        // 2. Disk removal succeeded (or nothing was there) — purge in-memory
        //    state. Each map is purged under its own lock.
        {
            let mut datasets = self.datasets.write().await;
            datasets.remove(domain);
        }
        {
            let mut indexes = self.branch_indexes.write().await;
            indexes.retain(|(d, _b), _| d != domain);
        }
        {
            let mut locks = self.pipeline_locks.write().await;
            locks.retain(|(d, _b), _| d != domain);
        }
        {
            // Purge any in-flight reservations for this domain so a delete +
            // re-push of the same commit is not wrongly 409'd by a stale
            // reservation left over from a push that the delete superseded.
            let mut inflight = self.inflight_commits.write().await;
            inflight.retain(|(d, _b, _c)| d != domain);
        }

        Ok(())
    }

    /// Get last-indexed for a (domain, branch) pair.
    pub async fn last_indexed(&self, domain: &Domain, branch: &BranchName) -> LastIndexed {
        let key = (domain.as_str().to_owned(), branch.as_str().to_owned());
        let indexes = self.branch_indexes.read().await;
        match indexes.get(&key) {
            Some(bi) => LastIndexed {
                branch: branch.as_str().to_owned(),
                commit: bi.commit.clone(),
                version: bi.version,
            },
            None => LastIndexed {
                branch: branch.as_str().to_owned(),
                commit: None,
                version: 0,
            },
        }
    }

    /// Update last-indexed tracking.
    pub async fn update_last_indexed(
        &self,
        domain: &str,
        branch: &str,
        commit: &str,
        version: u64,
    ) {
        let key = (domain.to_owned(), branch.to_owned());
        let mut indexes = self.branch_indexes.write().await;
        indexes.insert(
            key,
            BranchIndex {
                commit: Some(commit.to_owned()),
                version,
            },
        );
    }

    /// Record a task status.
    pub async fn record_task(&self, task_id: &str, status: TaskStatus) {
        let mut tasks = self.tasks.write().await;
        tasks.insert(task_id.to_owned(), status);
    }

    /// Check task status.
    pub async fn check_task(&self, task_id: &str) -> Option<TaskStatus> {
        let tasks = self.tasks.read().await;
        tasks.get(task_id).cloned()
    }

    /// Count total rows across all datasets (for statistics).
    pub async fn statistics(&self) -> Statistics {
        let datasets = self.datasets.read().await;
        let indexes = self.branch_indexes.read().await;

        // Layout A: each cache entry is one domain dataset. Branch count comes
        // from the per-(domain, branch) index map, not the dataset count.
        let mut domains = std::collections::HashSet::new();
        let mut distinct_branches: std::collections::HashSet<BranchKey> =
            std::collections::HashSet::new();
        let mut indexed_commits = 0u64;
        let mut chunks = 0u64;
        let mut distinct_docs: std::collections::HashSet<String> = std::collections::HashSet::new();

        for domain in datasets.keys() {
            domains.insert(domain.clone());
        }

        for (key, bi) in indexes.iter() {
            distinct_branches.insert(key.clone());
            domains.insert(key.0.clone());
            if bi.commit.is_some() {
                indexed_commits += 1;
            }
        }
        let branches = distinct_branches.len() as u64;

        // Count rows and distinct doc_ids from datasets (best-effort).
        for (_key, ds_arc) in datasets.iter() {
            let ds = ds_arc.read().await;
            if let Ok(count) = ds.count_rows(None).await {
                chunks += count as u64;
            }
            // Scan doc_id column for distinct count.
            let mut scanner = ds.scan();
            if scanner.project(&["doc_id"]).is_ok() {
                if let Ok(stream) = scanner.try_into_stream().await {
                    if let Ok(batches) = stream.try_collect::<Vec<RecordBatch>>().await {
                        for batch in &batches {
                            if let Some(ids) = batch
                                .column_by_name("doc_id")
                                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                            {
                                for i in 0..ids.len() {
                                    distinct_docs.insert(ids.value(i).to_owned());
                                }
                            }
                        }
                    }
                }
            }
        }

        // Count pending index fragments across all datasets.
        let mut pending_index_fragments: u64 = 0;
        for (_key, ds_arc) in datasets.iter() {
            let ds = ds_arc.read().await;
            if let Ok(pending) = crate::store::vector_index::count_unindexed_fragments(&ds).await {
                pending_index_fragments += pending;
            }
        }

        Statistics {
            domains: domains.len() as u64,
            branches,
            indexed_commits,
            documents: distinct_docs.len() as u64,
            chunks,
            pending_index_fragments,
        }
    }

    /// Vector/FTS/hybrid search over chunk rows at a given version.
    /// Returns chunk-level hits (caller dedups to documents).
    /// INVARIANT: searches the snapshot at the commit's tagged version, NOT the latest.
    pub async fn io_search(
        &self,
        domain: &str,
        branch: &str,
        commit: &str,
        query: &SearchQuery,
    ) -> Result<Vec<ChunkHit>, StoreError> {
        // Resolve `commit` to its versioned read-only snapshot off the CACHED
        // domain handle (shared object_store + session — see
        // `io_snapshot_from_cache`). Reusing the cache instead of `Dataset::open`
        // per search bounds the open file-descriptor count under load (BUG-FD24);
        // writers refresh the cache on mutation so freshly-tagged commits are still
        // visible (the 409 / search-at-just-indexed-commit invariant).
        let snapshot = self.io_snapshot_from_cache(domain, branch, commit).await?;

        // Build the search based on mode — always against the versioned snapshot.
        let hits = match query.mode {
            SearchMode::Vector => self.vector_search(&snapshot, query).await?,
            SearchMode::Fts => self.fts_search(&snapshot, query).await?,
            SearchMode::Hybrid => self.hybrid_search(&snapshot, query).await?,
        };

        Ok(hits)
    }

    /// Vector search using Lance's `nearest()` with `DistanceType::Cosine`.
    ///
    /// When a vector ANN index exists on the `embedding` column (created by
    /// `io_ensure_vector_index`), Lance routes the query through the index for
    /// sub-linear performance. Unindexed fragments are flat-searched alongside
    /// the index (correct results, higher latency on the unindexed portion).
    ///
    /// Without an index (legacy path or pre-index versions), this degrades to
    /// O(n) flat KNN — functionally correct but not suitable for large corpora.
    ///
    /// Distance: embeddings are L2-normalised before insert, so cosine distance
    /// is in [0, 2]. The `DistanceType::Cosine` metric is set explicitly on the
    /// scanner to ensure correct distance computation regardless of index type.
    async fn vector_search(
        &self,
        ds: &Dataset,
        query: &SearchQuery,
    ) -> Result<Vec<ChunkHit>, StoreError> {
        // Over-fetch to allow for dedup (multiple chunks per doc).
        let k = (query.start + query.count) * 3;

        let mut scanner = ds.scan();
        scanner
            .nearest("embedding", &Float32Array::from(query.query_embedding.clone()), k)
            .map_err(|e| StoreError::Internal(format!("vector search setup failed: {}", e)))?;
        scanner.distance_metric(DistanceType::Cosine);
        // ANN tuning: nprobes controls how many IVF partitions are scanned;
        // refine re-ranks the top candidates with full-precision vectors.
        scanner.nprobes(self.vector_index_config.nprobes);
        if let Some(rf) = self.vector_index_config.refine_factor {
            scanner.refine(rf);
        }

        // Apply filters if present.
        if !query.doc_type_filter.is_empty() || !query.doc_id_filter.is_empty() {
            let filter = build_filter_expression(&query.doc_type_filter, &query.doc_id_filter);
            if !filter.is_empty() {
                scanner.filter(&filter)
                    .map_err(|e| StoreError::Internal(format!("filter failed: {}", e)))?;
            }
        }

        let batches: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .map_err(|e| StoreError::Internal(format!("scan stream failed: {}", e)))?
            .try_collect()
            .await
            .map_err(|e| StoreError::Internal(format!("batch collect failed: {}", e)))?;

        // KNN search via nearest() — a `_distance` column is REQUIRED; fail loud
        // if absent rather than corrupt ranking with 0.0 (#E).
        batches_to_vector_hits(&batches, true)
    }

    /// Full-text search.
    /// Returns empty results if no FTS (INVERTED) index exists on the dataset.
    async fn fts_search(
        &self,
        ds: &Dataset,
        query: &SearchQuery,
    ) -> Result<Vec<ChunkHit>, StoreError> {
        let k = (query.start + query.count) * 3;

        let mut scanner = ds.scan();
        // Lance FTS via full_text_search on the "content" column.
        scanner
            .full_text_search(FullTextSearchQuery::new(query.query_text.clone()))
            .map_err(|e| StoreError::Internal(format!("FTS search setup failed: {}", e)))?;
        scanner
            .limit(Some(k as i64), None)
            .map_err(|e| StoreError::Internal(format!("limit failed: {}", e)))?;

        // Apply filters if present.
        if !query.doc_type_filter.is_empty() || !query.doc_id_filter.is_empty() {
            let filter = build_filter_expression(&query.doc_type_filter, &query.doc_id_filter);
            if !filter.is_empty() {
                scanner
                    .filter(&filter)
                    .map_err(|e| StoreError::Internal(format!("filter failed: {}", e)))?;
            }
        }

        let stream = match scanner.try_into_stream().await {
            Ok(s) => s,
            Err(e) => {
                let msg = e.to_string();
                // WHY: FTS is called from hybrid_search, which fuses vector + FTS.
                // At historical versions (checkout_version), the INVERTED index may
                // not yet exist (created after that version was tagged). Returning
                // empty degrades hybrid to vector-only — still useful and correct.
                // INVARIANT: the vector side of hybrid always runs (vector_search is
                // called independently); hybrid is never empty solely because FTS is.
                // CONSEQUENCE: search returns vector-only results at versions that
                // predate FTS index creation. No data loss; ranking is less rich.
                if msg.contains("INVERTED index") || msg.contains("full text search") {
                    return Ok(Vec::new());
                }
                return Err(StoreError::Internal(format!("FTS stream failed: {}", msg)));
            }
        };

        match stream.try_collect::<Vec<RecordBatch>>().await {
            Ok(batches) => Ok(batches_to_fts_hits(&batches)),
            Err(e) => {
                let msg = e.to_string();
                // WHY/INVARIANT/CONSEQUENCE: same as above — historical snapshot
                // may lack the INVERTED index; hybrid degrades to vector-only.
                if msg.contains("INVERTED index") || msg.contains("full text search") {
                    return Ok(Vec::new());
                }
                Err(StoreError::Internal(format!("FTS collect failed: {}", msg)))
            }
        }
    }

    /// Hybrid search: Reciprocal Rank Fusion (RRF) over vector + FTS ranked lists.
    /// Deterministic given deterministic inputs.
    /// RRF score = sum over lists of 1/(k + rank_in_list) where k=60 (standard).
    async fn hybrid_search(
        &self,
        ds: &Dataset,
        query: &SearchQuery,
    ) -> Result<Vec<ChunkHit>, StoreError> {
        let vector_hits = self.vector_search(ds, query).await?;
        let fts_hits = self.fts_search(ds, query).await?;
        Ok(rrf_merge(vector_hits, fts_hits))
    }

    /// Look up a document's chunks by doc_id at the head of `branch` (indexed
    /// lookup for /similar).
    /// INVARIANT: uses a filter on the doc_id column (indexed), NOT a full scan.
    /// Layout A: reads the BRANCH head (checked out), not the cached main handle —
    /// so a lookup on a feature branch sees the branch's data, not main's.
    pub async fn io_lookup_doc_chunks(
        &self,
        domain: &str,
        branch: &str,
        doc_id: &str,
    ) -> Result<Vec<ChunkHit>, StoreError> {
        // Read the branch head. For main, the cached handle is the default
        // branch; for a feature branch, check out a branch-bound handle.
        let path = self.dataset_path(domain);
        if !path.exists() {
            return Ok(Vec::new());
        }
        let owned_ds;
        let ds: &Dataset = if branch == MAIN_BRANCH {
            // READ-ONLY (BLOCKER-2): never resurrect a deleted domain on lookup.
            // The `path.exists()` guard above means we only reach here when the
            // dataset exists, but use the non-creating open for defence in depth.
            let ds_arc = match self.io_open_dataset_readonly(domain).await? {
                Some(ds) => ds,
                None => return Ok(Vec::new()),
            };
            let guard = ds_arc.read().await;
            // Read+collect under the guard, then return early — simplest correct path.
            return Self::scan_doc_chunks(&guard, doc_id).await;
        } else {
            let uri = path.to_string_lossy().to_string();
            let base = Dataset::open(&uri)
                .await
                .map_err(|e| StoreError::Internal(format!("lookup open failed: {}", e)))?;
            owned_ds = base.checkout_branch(branch).await.map_err(|e| {
                StoreError::Internal(format!("lookup checkout '{}' failed: {}", branch, e))
            })?;
            &owned_ds
        };

        Self::scan_doc_chunks(ds, doc_id).await
    }

    /// Scan a dataset handle for all chunks of `doc_id` (filter on indexed column).
    async fn scan_doc_chunks(ds: &Dataset, doc_id: &str) -> Result<Vec<ChunkHit>, StoreError> {
        let filter = format!("doc_id = '{}'", doc_id.replace('\'', "''"));
        let mut scanner = ds.scan();
        scanner
            .filter(&filter)
            .map_err(|e| StoreError::Internal(format!("lookup filter failed: {}", e)))?;

        let batches: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .map_err(|e| StoreError::Internal(format!("lookup stream failed: {}", e)))?
            .try_collect()
            .await
            .map_err(|e| StoreError::Internal(format!("lookup collect failed: {}", e)))?;

        // Plain filter scan (no KNN) — no `_distance` column by design; the
        // caller (/similar) re-embeds and ranks separately.
        batches_to_vector_hits(&batches, false)
    }

    /// Resolve `commit` on `branch` to its versioned, read-only snapshot off the
    /// CACHED domain handle — the single shared resolution path for `/search`,
    /// `/similar` and `/duplicates` (so commit→snapshot resolution and the
    /// not-indexed behaviour are CONSISTENT across all three).
    ///
    /// CACHED, NOT FRESH-PER-CALL (BUG-FD24): we clone the cached domain `Dataset`
    /// (cheap — `Dataset::clone` shares the object_store + session via `Arc`) and
    /// `checkout_branch`/`checkout_version` off that clone. `checkout_by_ref`
    /// REUSES the handle's object_store and session (it does not allocate a new
    /// one), so resolving a snapshot from the cache opens NO new file descriptors —
    /// unlike `Dataset::open`, which spins up a fresh object_store + session
    /// holding its own file readers per call (the FD-exhaustion source under load).
    ///
    /// TAG VISIBILITY (the reason fresh-open previously existed): the cached handle
    /// is refreshed by every mutation that changes tags/versions (`io_tag_commit`,
    /// `io_upsert_chunks`, `io_delete_doc`, optimize, assign) via
    /// `io_refresh_cached_dataset`, so a commit tagged by the worker is visible to
    /// a subsequent search/resolve. This preserves the 409 / search-at-just-
    /// indexed-commit invariant WITHOUT opening fresh every query.
    ///
    /// READ-ONLY (BLOCKER-2): an absent dataset is "not indexed", never resurrected
    /// (`io_open_dataset_readonly` does not auto-create).
    async fn io_snapshot_from_cache(
        &self,
        domain: &str,
        branch: &str,
        commit: &str,
    ) -> Result<Dataset, StoreError> {
        let cached = match self.io_open_dataset_readonly(domain).await? {
            Some(ds) => ds,
            None => {
                return Err(StoreError::Internal(format!(
                    "commit not indexed: domain '{}' has no dataset",
                    domain
                )));
            }
        };

        // Clone the cached Dataset (shares object_store + session via Arc) so the
        // checkouts below run without holding the cache lock for the whole search.
        let base = {
            let guard = cached.read().await;
            guard.clone()
        };

        let owned_branch_ds;
        let base: &Dataset = if branch == MAIN_BRANCH {
            &base
        } else {
            owned_branch_ds = base.checkout_branch(branch).await.map_err(|e| {
                StoreError::Internal(format!("checkout '{}' for snapshot failed: {}", branch, e))
            })?;
            &owned_branch_ds
        };

        let tag = layeridx::encode_commit_tag(commit);
        let version = base
            .tags()
            .get_version(&tag)
            .await
            .map_err(|e| StoreError::Internal(format!("commit not indexed: {}", e)))?;

        base.checkout_version(version).await.map_err(|e| {
            StoreError::Internal(format!("checkout version {} failed: {}", version, e))
        })
    }

    /// Resolve `commit` on `branch` to its versioned, read-only snapshot.
    /// Thin alias over `io_snapshot_from_cache` retained for `/duplicates` and
    /// `/similar` call sites.
    async fn io_open_snapshot(
        &self,
        domain: &str,
        branch: &str,
        commit: &str,
    ) -> Result<Dataset, StoreError> {
        self.io_snapshot_from_cache(domain, branch, commit).await
    }

}

/// Ensure the FTS inverted index is up-to-date on an already-open Dataset handle.
/// Used by the background index worker to operate on an uncached handle (no RwLock
/// contention with search reads).
///
/// Creates the index if missing; if it already exists, runs optimize_indices(append())
/// to index only new (unindexed) fragments — O(delta).
pub async fn io_ensure_fts_index_on_dataset(ds: &mut Dataset) -> Result<u64, StoreError> {
    let indices = ds
        .load_indices()
        .await
        .map_err(|e| StoreError::Internal(format!("load indices failed: {}", e)))?;
    let has_fts = indices.iter().any(|idx| idx.name == "content_fts");

    if !has_fts {
        let params = InvertedIndexParams::default();
        ds.create_index(
            &["content"],
            IndexType::Inverted,
            Some("content_fts".to_owned()),
            &params,
            false,
        )
        .await
        .map_err(|e| StoreError::Internal(format!("FTS index creation failed: {}", e)))?;
    } else {
        ds.optimize_indices(&OptimizeOptions::append())
            .await
            .map_err(|e| StoreError::Internal(format!("FTS index optimize failed: {}", e)))?;
    }

    Ok(ds.version().version)
}

/// Build a SQL-like filter expression for doc_type and doc_id IN (...) filters.
fn build_filter_expression(doc_types: &[String], doc_ids: &[String]) -> String {
    let mut parts = Vec::new();

    if !doc_types.is_empty() {
        let values: Vec<String> = doc_types
            .iter()
            .map(|t| format!("'{}'", t.replace('\'', "''")))
            .collect();
        parts.push(format!("doc_type IN ({})", values.join(", ")));
    }

    if !doc_ids.is_empty() {
        let values: Vec<String> = doc_ids
            .iter()
            .map(|id| format!("'{}'", id.replace('\'', "''")))
            .collect();
        parts.push(format!("doc_id IN ({})", values.join(", ")));
    }

    // AND across the two sets (OR within each via IN).
    parts.join(" AND ")
}

/// Extract ChunkHit records from RecordBatches.
///
/// `require_distance` distinguishes the two read shapes:
///  - A vector `nearest()` search MUST carry a `_distance` column; a missing one
///    means the scanner did not run as a KNN query — defaulting every hit to 0.0
///    would silently corrupt ranking (all-equal distances → arbitrary order),
///    so we FAIL LOUD with `StoreError::Internal` (#E).
///  - A plain filter scan (e.g. doc-chunk lookup for `/similar`) has NO
///    `_distance` column by design and does NOT rank — `require_distance=false`
///    yields a neutral 0.0 there, which is correct (the caller re-embeds and
///    ranks separately).
fn batches_to_vector_hits(
    batches: &[RecordBatch],
    require_distance: bool,
) -> Result<Vec<ChunkHit>, StoreError> {
    let mut hits = Vec::new();

    for batch in batches {
        let n = batch.num_rows();
        let doc_ids = batch
            .column_by_name("doc_id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let chunk_indexes = batch
            .column_by_name("chunk_index")
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
        let chunk_counts = batch
            .column_by_name("chunk_count")
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
        let chunk_token_starts = batch
            .column_by_name("chunk_token_start")
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
        let doc_token_lens = batch
            .column_by_name("doc_token_len")
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
        let contents = batch
            .column_by_name("content")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let distances = batch
            .column_by_name("_distance")
            .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

        // FAIL LOUD (#E): a ranked vector search with no usable `_distance`
        // column is a real error — never default to 0.0 and corrupt ranking.
        if require_distance && distances.is_none() && n > 0 {
            return Err(StoreError::Internal(
                "vector search result is missing the `_distance` column — \
                 refusing to default distances to 0.0 (would corrupt ranking)"
                    .to_owned(),
            ));
        }

        if let (Some(ids), Some(ci), Some(cc), Some(cts), Some(dtl), Some(cnt)) =
            (doc_ids, chunk_indexes, chunk_counts, chunk_token_starts, doc_token_lens, contents)
        {
            for i in 0..n {
                // For a ranked search `distances` is guaranteed Some by the guard
                // above; for a plain scan a neutral 0.0 is correct (no ranking).
                let distance = distances.map(|d| d.value(i)).unwrap_or(0.0);
                hits.push(ChunkHit {
                    doc_id: ids.value(i).to_owned(),
                    distance,
                    distance_kind: DistanceKind::RawCosine,
                    chunk_index: ci.value(i),
                    chunk_count: cc.value(i),
                    chunk_token_start: cts.value(i),
                    doc_token_len: dtl.value(i),
                    content: cnt.value(i).to_owned(),
                });
            }
        }
    }

    Ok(hits)
}

/// Extract ChunkHit records from RecordBatches (FTS search — reads `_score`).
/// BM25 scores are converted to distances: `distance = 1/(1+score)` so 0=best.
/// Results are preserved in stream order (BM25 rank order from Lance).
fn batches_to_fts_hits(batches: &[RecordBatch]) -> Vec<ChunkHit> {
    let mut hits = Vec::new();

    for batch in batches {
        let n = batch.num_rows();
        let doc_ids = batch
            .column_by_name("doc_id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let chunk_indexes = batch
            .column_by_name("chunk_index")
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
        let chunk_counts = batch
            .column_by_name("chunk_count")
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
        let chunk_token_starts = batch
            .column_by_name("chunk_token_start")
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
        let doc_token_lens = batch
            .column_by_name("doc_token_len")
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
        let contents = batch
            .column_by_name("content")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let scores = batch
            .column_by_name("_score")
            .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

        if let (Some(ids), Some(ci), Some(cc), Some(cts), Some(dtl), Some(cnt)) =
            (doc_ids, chunk_indexes, chunk_counts, chunk_token_starts, doc_token_lens, contents)
        {
            for i in 0..n {
                // BM25 score → distance: higher score = more relevant = lower distance.
                let score = scores.map(|s| s.value(i)).unwrap_or(0.0);
                let distance = 1.0 / (1.0 + score);
                hits.push(ChunkHit {
                    doc_id: ids.value(i).to_owned(),
                    distance,
                    distance_kind: DistanceKind::Normalised,
                    chunk_index: ci.value(i),
                    chunk_count: cc.value(i),
                    chunk_token_start: cts.value(i),
                    doc_token_len: dtl.value(i),
                    content: cnt.value(i).to_owned(),
                });
            }
        }
    }

    hits
}

/// Reciprocal Rank Fusion: merge two ranked lists into a single list.
/// RRF score = sum of 1/(k + rank) across all lists where the item appears.
/// k=60 is the standard constant. Items are keyed by (doc_id, chunk_index).
/// The output is sorted by descending RRF score (highest relevance first).
/// Distance is set to 1.0 - normalized_rrf_score (so smaller = more relevant).
fn rrf_merge(vector_hits: Vec<ChunkHit>, fts_hits: Vec<ChunkHit>) -> Vec<ChunkHit> {
    use std::collections::HashMap;
    const K: f32 = 60.0;

    // Key: (doc_id, chunk_index) → (rrf_score, best ChunkHit)
    let mut scores: HashMap<(String, i32), (f32, ChunkHit)> = HashMap::new();

    // Score vector results by rank (already sorted by distance — best first).
    for (rank, hit) in vector_hits.into_iter().enumerate() {
        let key = (hit.doc_id.clone(), hit.chunk_index);
        let rrf_score = 1.0 / (K + rank as f32 + 1.0);
        let entry = scores.entry(key).or_insert_with(|| (0.0, hit.clone()));
        entry.0 += rrf_score;
        // Keep the hit with the better (smaller) original distance.
        if hit.distance < entry.1.distance {
            entry.1 = hit;
        }
    }

    // Score FTS results by rank (already sorted by relevance — best first).
    for (rank, hit) in fts_hits.into_iter().enumerate() {
        let key = (hit.doc_id.clone(), hit.chunk_index);
        let rrf_score = 1.0 / (K + rank as f32 + 1.0);
        let entry = scores.entry(key).or_insert_with(|| (0.0, hit.clone()));
        entry.0 += rrf_score;
        if hit.distance < entry.1.distance {
            entry.1 = hit;
        }
    }

    // Sort by RRF score descending (highest relevance first).
    let mut ranked: Vec<(f32, ChunkHit)> = scores.into_values().collect();
    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Convert RRF score into a synthetic distance (lower = better).
    // Normalise: max possible single-list score is 1/(K+1) ≈ 0.0164.
    // Max combined (rank 1 in both) is 2/(K+1) ≈ 0.0328.
    // We map to [0, 1] via: distance = 1.0 - (score / max_score).
    let max_score = 2.0 / (K + 1.0);
    ranked
        .into_iter()
        .map(|(score, mut hit)| {
            hit.distance = (1.0 - score / max_score).clamp(0.0, 1.0);
            hit.distance_kind = DistanceKind::Normalised;
            hit
        })
        .collect()
}

/// Dedup chunk-level hits to document-level hits (best chunk per doc_id).
/// Distance transform is mode-aware: only RawCosine hits get `normalized_cosine_from_lance`;
/// already-normalised hits (FTS, RRF) pass through unchanged.
pub fn dedup_chunks_to_documents(hits: Vec<ChunkHit>, snippet: bool) -> Vec<SearchHit> {
    use std::collections::HashMap;
    use crate::kernel::distance::normalized_cosine_from_lance;
    use crate::chunk::chunk_location;

    // Group by doc_id, keep the best (smallest distance) chunk per document.
    let mut best_per_doc: HashMap<String, ChunkHit> = HashMap::new();
    for hit in hits {
        let entry = best_per_doc
            .entry(hit.doc_id.clone())
            .or_insert_with(|| hit.clone());
        if hit.distance < entry.distance {
            *entry = hit;
        }
    }

    let mut results: Vec<SearchHit> = best_per_doc
        .into_values()
        .map(|hit| {
            let location = chunk_location(hit.chunk_token_start as u32, hit.doc_token_len as u32);
            let final_distance = match hit.distance_kind {
                DistanceKind::RawCosine => normalized_cosine_from_lance(hit.distance),
                DistanceKind::Normalised => hit.distance,
            };
            SearchHit {
                id: hit.doc_id,
                distance: final_distance,
                chunk: ChunkInfo {
                    index: hit.chunk_index as u32,
                    count: hit.chunk_count as u32,
                    token_start: hit.chunk_token_start as u32,
                    doc_token_len: hit.doc_token_len as u32,
                    location,
                    snippet: if snippet { Some(hit.content) } else { None },
                },
            }
        })
        .collect();

    // Sort by distance (nearest first).
    results.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

#[cfg(test)]
#[allow(clippy::useless_vec)]
mod tests {
    use super::*;

    fn make_test_store(dim: usize) -> (LanceStore, tempfile::TempDir) {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let store = LanceStore::new(tmp.path(), dim);
        (store, tmp)
    }

    fn fake_embedding(dim: usize, seed: f32) -> Vec<f32> {
        (0..dim).map(|i| (seed + i as f32 * 0.01).sin()).collect()
    }

    // --- upsert chunks, tag commit, verify round-trip ---
    #[tokio::test]
    async fn upsert_and_tag_commit_round_trips() {
        let (store, _tmp) = make_test_store(8);
        let rows = vec![
            ChunkRow {
                doc_id: "doc/1".to_owned(),
                doc_type: "People".to_owned(),
                chunk_index: 0,
                chunk_count: 1,
                chunk_token_start: 0,
                doc_token_len: 40,
                embedding: fake_embedding(8, 1.0),
                content: "Yoda is a wise Jedi.".to_owned(),
            },
            ChunkRow {
                doc_id: "doc/2".to_owned(),
                doc_type: "Species".to_owned(),
                chunk_index: 0,
                chunk_count: 1,
                chunk_token_start: 0,
                doc_token_len: 30,
                embedding: fake_embedding(8, 2.0),
                content: "Mon Calamari are squid people.".to_owned(),
            },
        ];

        let version = store
            .io_upsert_chunks("admin/star_wars", "main", "doc/1", &rows[0..1])
            .await
            .expect("upsert doc/1");
        assert!(version > 0);

        let version2 = store
            .io_upsert_chunks("admin/star_wars", "main", "doc/2", &rows[1..2])
            .await
            .expect("upsert doc/2");
        assert!(version2 > version);

        // Tag commit c0 to the final version.
        store
            .io_tag_commit("admin/star_wars", "main", "c0", version2)
            .await
            .expect("tag commit");

        // Resolve commit should return the version.
        let resolved = store
            .io_resolve_commit("admin/star_wars", "main", "c0")
            .await
            .expect("resolve");
        assert_eq!(resolved, Some(version2));
    }

    // --- multi-chunk doc produces multiple rows ---
    #[tokio::test]
    async fn multi_chunk_doc_produces_multiple_rows() {
        let (store, _tmp) = make_test_store(8);
        let rows = vec![
            ChunkRow {
                doc_id: "doc/big".to_owned(),
                doc_type: "Article".to_owned(),
                chunk_index: 0,
                chunk_count: 3,
                chunk_token_start: 0,
                doc_token_len: 1500,
                embedding: fake_embedding(8, 1.0),
                content: "Beginning of the article.".to_owned(),
            },
            ChunkRow {
                doc_id: "doc/big".to_owned(),
                doc_type: "Article".to_owned(),
                chunk_index: 1,
                chunk_count: 3,
                chunk_token_start: 450,
                doc_token_len: 1500,
                embedding: fake_embedding(8, 2.0),
                content: "Middle of the article.".to_owned(),
            },
            ChunkRow {
                doc_id: "doc/big".to_owned(),
                doc_type: "Article".to_owned(),
                chunk_index: 2,
                chunk_count: 3,
                chunk_token_start: 900,
                doc_token_len: 1500,
                embedding: fake_embedding(8, 3.0),
                content: "End of the article.".to_owned(),
            },
        ];

        store
            .io_upsert_chunks("admin/test", "main", "doc/big", &rows)
            .await
            .expect("upsert multi-chunk");

        // Lookup should find all 3 chunks.
        let chunks = store
            .io_lookup_doc_chunks("admin/test", "main", "doc/big")
            .await
            .expect("lookup");
        assert_eq!(chunks.len(), 3);
    }

    // --- delete removes all chunks ---
    #[tokio::test]
    async fn delete_doc_removes_all_chunks() {
        let (store, _tmp) = make_test_store(8);
        let rows = vec![
            ChunkRow {
                doc_id: "doc/del".to_owned(),
                doc_type: "X".to_owned(),
                chunk_index: 0,
                chunk_count: 2,
                chunk_token_start: 0,
                doc_token_len: 100,
                embedding: fake_embedding(8, 1.0),
                content: "part 1".to_owned(),
            },
            ChunkRow {
                doc_id: "doc/del".to_owned(),
                doc_type: "X".to_owned(),
                chunk_index: 1,
                chunk_count: 2,
                chunk_token_start: 50,
                doc_token_len: 100,
                embedding: fake_embedding(8, 2.0),
                content: "part 2".to_owned(),
            },
        ];

        store
            .io_upsert_chunks("admin/test", "main", "doc/del", &rows)
            .await
            .expect("upsert");

        store
            .io_delete_doc("admin/test", "main", "doc/del")
            .await
            .expect("delete");

        let remaining = store
            .io_lookup_doc_chunks("admin/test", "main", "doc/del")
            .await
            .expect("lookup after delete");
        assert_eq!(remaining.len(), 0, "all chunks should be deleted");
    }

    // --- dedup produces correct chunk metadata ---
    #[test]
    fn dedup_chunks_to_documents_picks_best_chunk() {
        let hits = vec![
            ChunkHit {
                doc_id: "doc/1".to_owned(),
                distance: 0.8,
                distance_kind: DistanceKind::RawCosine,
                chunk_index: 0,
                chunk_count: 2,
                chunk_token_start: 0,
                doc_token_len: 1000,
                content: "first chunk".to_owned(),
            },
            ChunkHit {
                doc_id: "doc/1".to_owned(),
                distance: 0.4, // Better (smaller distance) — this chunk wins.
                distance_kind: DistanceKind::RawCosine,
                chunk_index: 1,
                chunk_count: 2,
                chunk_token_start: 500,
                doc_token_len: 1000,
                content: "second chunk".to_owned(),
            },
            ChunkHit {
                doc_id: "doc/2".to_owned(),
                distance: 0.6,
                distance_kind: DistanceKind::RawCosine,
                chunk_index: 0,
                chunk_count: 1,
                chunk_token_start: 0,
                doc_token_len: 200,
                content: "only chunk".to_owned(),
            },
        ];

        let results = dedup_chunks_to_documents(hits, false);
        assert_eq!(results.len(), 2, "should dedup to 2 documents");

        // Results sorted by distance — doc/1 (0.4→0.2 after transform) < doc/2 (0.6→0.3).
        let doc1 = results.iter().find(|r| r.id == "doc/1").expect("doc/1");
        assert_eq!(doc1.chunk.index, 1, "best chunk is index 1");
        assert_eq!(doc1.chunk.count, 2);
        assert_eq!(doc1.chunk.token_start, 500);
        assert_eq!(doc1.chunk.doc_token_len, 1000);
        // location = 500/1000 = 0.5
        assert!((doc1.chunk.location - 0.5).abs() < f32::EPSILON);
        assert!(doc1.chunk.snippet.is_none(), "snippet should be omitted");
    }

    // --- single chunk doc has location 0.0 ---
    #[test]
    fn dedup_single_chunk_doc_location_zero() {
        let hits = vec![ChunkHit {
            doc_id: "doc/s".to_owned(),
            distance: 0.2,
            distance_kind: DistanceKind::RawCosine,
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 41,
            content: "short doc".to_owned(),
        }];

        let results = dedup_chunks_to_documents(hits, true);
        assert_eq!(results.len(), 1);
        let hit = &results[0];
        assert_eq!(hit.chunk.index, 0);
        assert_eq!(hit.chunk.count, 1);
        assert_eq!(hit.chunk.token_start, 0);
        assert_eq!(hit.chunk.doc_token_len, 41);
        assert_eq!(hit.chunk.location, 0.0);
        assert_eq!(hit.chunk.snippet, Some("short doc".to_owned()));
    }

    // --- distance transform applied correctly ---
    #[test]
    fn distance_transform_in_dedup() {
        let hits = vec![ChunkHit {
            doc_id: "doc/x".to_owned(),
            distance: 0.0, // Self-distance in lance cosine.
            distance_kind: DistanceKind::RawCosine,
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            content: "x".to_owned(),
        }];
        let results = dedup_chunks_to_documents(hits, false);
        assert_eq!(results[0].distance, 0.0, "self-distance maps to 0");
    }

    // --- #3: normalised distances skip the transform in dedup ---
    #[test]
    fn dedup_normalised_distances_pass_through() {
        let hits = vec![
            ChunkHit {
                doc_id: "doc/rrf".to_owned(),
                distance: 0.42, // Already normalised (e.g., from RRF).
                distance_kind: DistanceKind::Normalised,
                chunk_index: 0,
                chunk_count: 1,
                chunk_token_start: 0,
                doc_token_len: 10,
                content: "rrf hit".to_owned(),
            },
        ];
        let results = dedup_chunks_to_documents(hits, false);
        // Must pass through unchanged — NOT halved by normalized_cosine_from_lance.
        assert!(
            (results[0].distance - 0.42).abs() < f32::EPSILON,
            "normalised distance should pass through unchanged, got {}",
            results[0].distance,
        );
    }

    // --- #1: FTS distances are non-zero and ordered (BM25 score preserved) ---
    #[test]
    fn fts_hits_have_nonzero_ordered_distances() {
        // Simulate FTS hits with BM25 scores converted to distances.
        let hits = vec![
            ChunkHit {
                doc_id: "doc/best".to_owned(),
                distance: 1.0 / (1.0 + 10.0), // High BM25 score → low distance.
                distance_kind: DistanceKind::Normalised,
                chunk_index: 0,
                chunk_count: 1,
                chunk_token_start: 0,
                doc_token_len: 10,
                content: "best match".to_owned(),
            },
            ChunkHit {
                doc_id: "doc/worse".to_owned(),
                distance: 1.0 / (1.0 + 2.0), // Lower BM25 score → higher distance.
                distance_kind: DistanceKind::Normalised,
                chunk_index: 0,
                chunk_count: 1,
                chunk_token_start: 0,
                doc_token_len: 10,
                content: "worse match".to_owned(),
            },
        ];

        let results = dedup_chunks_to_documents(hits, false);
        assert_eq!(results.len(), 2);
        // Best match (lowest distance) should be first after sorting.
        assert_eq!(results[0].id, "doc/best");
        assert_eq!(results[1].id, "doc/worse");
        // Both distances must be non-zero.
        assert!(results[0].distance > 0.0, "FTS distance must be > 0");
        assert!(results[1].distance > 0.0, "FTS distance must be > 0");
        // Best has lower distance.
        assert!(results[0].distance < results[1].distance);
    }

    // --- #2: Vector distance scale anchors (locks factor-of-2 correctness) ---
    // With DistanceType::Cosine on the scanner, _distance is a true cosine distance
    // in [0,2]. normalized_cosine_from_lance(d) = d/2 maps to [0,1].
    // These anchors catch any factor-of-2 scale bug permanently.
    #[test]
    fn vector_distance_scale_anchors_through_dedup() {
        // Anchor 1: self-distance (identical vectors) → 0.0
        let hit_identical = ChunkHit {
            doc_id: "doc/self".to_owned(),
            distance: 0.0, // Lance cosine: identical vectors
            distance_kind: DistanceKind::RawCosine,
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            content: "self".to_owned(),
        };
        let results = dedup_chunks_to_documents(vec![hit_identical], false);
        assert_eq!(results[0].distance, 0.0, "identical → 0.0");

        // Anchor 2: orthogonal vectors (Lance cosine distance = 1.0) → 0.5
        let hit_orthogonal = ChunkHit {
            doc_id: "doc/ortho".to_owned(),
            distance: 1.0, // Lance cosine: orthogonal
            distance_kind: DistanceKind::RawCosine,
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            content: "ortho".to_owned(),
        };
        let results = dedup_chunks_to_documents(vec![hit_orthogonal], false);
        assert!(
            (results[0].distance - 0.5).abs() < f32::EPSILON,
            "orthogonal → 0.5, got {}",
            results[0].distance,
        );

        // Anchor 3: opposite vectors (Lance cosine distance = 2.0) → 1.0
        let hit_opposite = ChunkHit {
            doc_id: "doc/opp".to_owned(),
            distance: 2.0, // Lance cosine: opposite
            distance_kind: DistanceKind::RawCosine,
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            content: "opposite".to_owned(),
        };
        let results = dedup_chunks_to_documents(vec![hit_opposite], false);
        assert_eq!(results[0].distance, 1.0, "opposite → 1.0");

        // The OLD bug: L2² for orthogonal unit vectors = 2.0, which would give
        // normalized_cosine_from_lance(2.0) = 1.0 (WRONG — should be 0.5).
        // This test catches any regression where L2² is fed to the transform.
    }

    // --- statistics reflect indexed data ---
    #[tokio::test]
    async fn statistics_reflect_indexed_data() {
        let (store, _tmp) = make_test_store(8);
        let rows = vec![ChunkRow {
            doc_id: "doc/1".to_owned(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            embedding: fake_embedding(8, 1.0),
            content: "test".to_owned(),
        }];

        store
            .io_upsert_chunks("admin/db", "main", "doc/1", &rows)
            .await
            .expect("upsert");
        store.update_last_indexed("admin/db", "main", "c0", 1).await;

        let stats = store.statistics().await;
        assert!(stats.chunks > 0, "chunks should be > 0 after upsert");
        assert!(stats.domains > 0, "domains should be > 0");
        assert!(stats.indexed_commits > 0, "indexed_commits should be > 0");
    }

    // --- commit tag isolation (different tags for different versions) ---
    #[tokio::test]
    async fn different_commits_different_versions() {
        let (store, _tmp) = make_test_store(8);

        // Insert first doc, tag as c0.
        let rows1 = vec![ChunkRow {
            doc_id: "doc/1".to_owned(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            embedding: fake_embedding(8, 1.0),
            content: "version one".to_owned(),
        }];
        let v1 = store
            .io_upsert_chunks("admin/db", "main", "doc/1", &rows1)
            .await
            .expect("upsert v1");
        store
            .io_tag_commit("admin/db", "main", "c0", v1)
            .await
            .expect("tag c0");

        // Insert second doc, tag as c1.
        let rows2 = vec![ChunkRow {
            doc_id: "doc/2".to_owned(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            embedding: fake_embedding(8, 2.0),
            content: "version two".to_owned(),
        }];
        let v2 = store
            .io_upsert_chunks("admin/db", "main", "doc/2", &rows2)
            .await
            .expect("upsert v2");
        store
            .io_tag_commit("admin/db", "main", "c1", v2)
            .await
            .expect("tag c1");

        // Resolve both — different versions.
        let r0 = store.io_resolve_commit("admin/db", "main", "c0").await.unwrap();
        let r1 = store.io_resolve_commit("admin/db", "main", "c1").await.unwrap();
        assert_eq!(r0, Some(v1));
        assert_eq!(r1, Some(v2));
        assert_ne!(v1, v2, "versions must differ");
    }

    // --- snapshot isolation: search at C0 does not see C1 data ---
    #[tokio::test]
    async fn snapshot_isolation_search_at_old_commit_excludes_new_data() {
        let (store, _tmp) = make_test_store(8);

        // Insert doc/A, tag as commit "c0".
        let emb_a = fake_embedding(8, 1.0);
        let rows_a = vec![ChunkRow {
            doc_id: "doc/A".to_owned(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            embedding: emb_a.clone(),
            content: "document A".to_owned(),
        }];
        let v0 = store
            .io_upsert_chunks("admin/iso", "main", "doc/A", &rows_a)
            .await
            .expect("upsert A");
        store
            .io_tag_commit("admin/iso", "main", "c0", v0)
            .await
            .expect("tag c0");

        // Insert doc/B, tag as commit "c1".
        let emb_b = fake_embedding(8, 2.0);
        let rows_b = vec![ChunkRow {
            doc_id: "doc/B".to_owned(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            embedding: emb_b.clone(),
            content: "document B".to_owned(),
        }];
        let v1 = store
            .io_upsert_chunks("admin/iso", "main", "doc/B", &rows_b)
            .await
            .expect("upsert B");
        store
            .io_tag_commit("admin/iso", "main", "c1", v1)
            .await
            .expect("tag c1");

        // Search at c0 — should only find doc/A.
        let query_c0 = SearchQuery {
            query_embedding: emb_a.clone(),
            query_text: "document".to_owned(),
            mode: crate::kernel::model::SearchMode::Vector,
            start: 0,
            count: 10,
            doc_type_filter: Vec::new(),
            doc_id_filter: Vec::new(),
            snippet: false,
        };
        let hits_c0 = store
            .io_search("admin/iso", "main", "c0", &query_c0)
            .await
            .expect("search at c0");

        let doc_ids_c0: Vec<&str> = hits_c0.iter().map(|h| h.doc_id.as_str()).collect();
        assert!(
            doc_ids_c0.contains(&"doc/A"),
            "c0 snapshot should contain doc/A"
        );
        assert!(
            !doc_ids_c0.contains(&"doc/B"),
            "c0 snapshot must NOT contain doc/B (added after c0)"
        );

        // Search at c1 — should find both doc/A and doc/B.
        let query_c1 = SearchQuery {
            query_embedding: emb_a,
            query_text: "document".to_owned(),
            mode: crate::kernel::model::SearchMode::Vector,
            start: 0,
            count: 10,
            doc_type_filter: Vec::new(),
            doc_id_filter: Vec::new(),
            snippet: false,
        };
        let hits_c1 = store
            .io_search("admin/iso", "main", "c1", &query_c1)
            .await
            .expect("search at c1");

        let doc_ids_c1: Vec<&str> = hits_c1.iter().map(|h| h.doc_id.as_str()).collect();
        assert!(
            doc_ids_c1.contains(&"doc/A"),
            "c1 snapshot should contain doc/A"
        );
        assert!(
            doc_ids_c1.contains(&"doc/B"),
            "c1 snapshot should contain doc/B"
        );
    }

    // --- P3-ASSIGN-1: assign is a pure tag pointer — no new version, target == source ---
    // The store assign primitive touches only Lance tags. It creates NO new
    // dataset version (so no fragments, so — by construction — zero embed calls),
    // and search at the target commit returns exactly the source commit's data.
    #[tokio::test]
    async fn assign_is_tag_pointer_no_recompute() {
        let (store, _tmp) = make_test_store(8);
        let domain = "admin/assign";

        // Index doc/A and doc/B, tag c0 at the final version.
        let emb_a = fake_embedding(8, 1.0);
        let rows_a = vec![ChunkRow {
            doc_id: "doc/A".to_owned(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            embedding: emb_a.clone(),
            content: "alpha".to_owned(),
        }];
        store
            .io_upsert_chunks(domain, "main", "doc/A", &rows_a)
            .await
            .expect("upsert A");
        let rows_b = vec![ChunkRow {
            doc_id: "doc/B".to_owned(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            embedding: fake_embedding(8, 2.0),
            content: "beta".to_owned(),
        }];
        let v0 = store
            .io_upsert_chunks(domain, "main", "doc/B", &rows_b)
            .await
            .expect("upsert B");
        store.io_tag_commit(domain, "main", "c0", v0).await.expect("tag c0");

        // Record the dataset version BEFORE assign.
        let ds_arc = store.io_open_dataset(domain, "main").await.unwrap();
        let version_before = ds_arc.read().await.version().version;

        // Assign c0 → c2 (pure tag pointer).
        let assigned_version = store
            .io_assign_commit(domain, "main", "c0", "c2")
            .await
            .expect("assign c0→c2");
        assert_eq!(assigned_version, v0, "c2 must point at c0's version");

        // No new dataset version was created (assign moved no data → no embeds possible).
        let ds_arc = store.io_open_dataset(domain, "main").await.unwrap();
        let version_after = ds_arc.read().await.version().version;
        assert_eq!(
            version_after, version_before,
            "assign must not create a new dataset version (no recompute)"
        );

        // c2 resolves to the same version as c0.
        let r_c0 = store.io_resolve_commit(domain, "main", "c0").await.unwrap();
        let r_c2 = store.io_resolve_commit(domain, "main", "c2").await.unwrap();
        assert_eq!(r_c0, Some(v0));
        assert_eq!(r_c2, Some(v0), "c2 must resolve to c0's version");

        // Search at c2 returns exactly the same docs as search at c0.
        let query = SearchQuery {
            query_embedding: emb_a.clone(),
            query_text: "alpha".to_owned(),
            mode: crate::kernel::model::SearchMode::Vector,
            start: 0,
            count: 10,
            doc_type_filter: Vec::new(),
            doc_id_filter: Vec::new(),
            snippet: false,
        };
        let mut hits_c0: Vec<String> = store
            .io_search(domain, "main", "c0", &query)
            .await
            .expect("search c0")
            .into_iter()
            .map(|h| h.doc_id)
            .collect();
        let mut hits_c2: Vec<String> = store
            .io_search(domain, "main", "c2", &query)
            .await
            .expect("search c2")
            .into_iter()
            .map(|h| h.doc_id)
            .collect();
        hits_c0.sort();
        hits_c2.sort();
        assert_eq!(hits_c0, hits_c2, "search at c2 must equal search at c0");
    }

    // --- assign of an unindexed source fails loud ---
    #[tokio::test]
    async fn assign_unindexed_source_fails_loud() {
        let (store, _tmp) = make_test_store(8);
        // Create the dataset so resolve doesn't fail on a missing dataset.
        let r = vec![ChunkRow {
            doc_id: "doc/X".to_owned(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            embedding: fake_embedding(8, 1.0),
            content: "x".to_owned(),
        }];
        store.io_upsert_chunks("admin/a", "main", "doc/X", &r).await.unwrap();

        let result = store
            .io_assign_commit("admin/a", "main", "never_indexed", "target")
            .await;
        assert!(result.is_err(), "assigning from an unindexed source must fail loud");
    }

    // --- P3-CHG-1: Changed replaces the full chunk set — no stale chunks ---
    // A doc indexed as a 3-chunk document, then re-pushed as a 1-chunk document,
    // must leave EXACTLY the new chunk set (the 2 old tail chunks are gone).
    #[tokio::test]
    async fn changed_replaces_full_chunk_set_no_stale() {
        let (store, _tmp) = make_test_store(8);
        let domain = "admin/chg";

        // Initial: doc/big with 3 chunks.
        let big_v1 = vec![
            ChunkRow {
                doc_id: "doc/big".to_owned(),
                doc_type: "Article".to_owned(),
                chunk_index: 0,
                chunk_count: 3,
                chunk_token_start: 0,
                doc_token_len: 1500,
                embedding: fake_embedding(8, 1.0),
                content: "original beginning".to_owned(),
            },
            ChunkRow {
                doc_id: "doc/big".to_owned(),
                doc_type: "Article".to_owned(),
                chunk_index: 1,
                chunk_count: 3,
                chunk_token_start: 500,
                doc_token_len: 1500,
                embedding: fake_embedding(8, 2.0),
                content: "original middle".to_owned(),
            },
            ChunkRow {
                doc_id: "doc/big".to_owned(),
                doc_type: "Article".to_owned(),
                chunk_index: 2,
                chunk_count: 3,
                chunk_token_start: 1000,
                doc_token_len: 1500,
                embedding: fake_embedding(8, 3.0),
                content: "original end".to_owned(),
            },
        ];
        store
            .io_upsert_chunks(domain, "main", "doc/big", &big_v1)
            .await
            .expect("upsert v1");
        assert_eq!(
            store.io_lookup_doc_chunks(domain, "main", "doc/big").await.unwrap().len(),
            3,
            "should have 3 chunks initially"
        );

        // Changed: same doc now renders to a SINGLE shorter chunk.
        let big_v2 = vec![ChunkRow {
            doc_id: "doc/big".to_owned(),
            doc_type: "Article".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 40,
            embedding: fake_embedding(8, 9.0),
            content: "shortened content".to_owned(),
        }];
        store
            .io_upsert_chunks(domain, "main", "doc/big", &big_v2)
            .await
            .expect("upsert v2 (Changed)");

        // Exactly 1 chunk remains — the 2 stale tail chunks must be gone.
        let after = store
            .io_lookup_doc_chunks(domain, "main", "doc/big")
            .await
            .unwrap();
        assert_eq!(
            after.len(),
            1,
            "Changed must replace the FULL chunk set — no stale chunks (got {})",
            after.len()
        );
        assert_eq!(after[0].content, "shortened content");
    }

    // --- P3-DEL-1: Deleted removes ALL chunks for a doc_id ---
    #[tokio::test]
    async fn deleted_removes_all_chunks_for_doc() {
        let (store, _tmp) = make_test_store(8);
        let domain = "admin/del";

        // Two docs, doc/keep and doc/gone (multi-chunk).
        let keep = vec![ChunkRow {
            doc_id: "doc/keep".to_owned(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            embedding: fake_embedding(8, 1.0),
            content: "keep me".to_owned(),
        }];
        let gone = vec![
            ChunkRow {
                doc_id: "doc/gone".to_owned(),
                doc_type: "T".to_owned(),
                chunk_index: 0,
                chunk_count: 2,
                chunk_token_start: 0,
                doc_token_len: 200,
                embedding: fake_embedding(8, 2.0),
                content: "gone part 1".to_owned(),
            },
            ChunkRow {
                doc_id: "doc/gone".to_owned(),
                doc_type: "T".to_owned(),
                chunk_index: 1,
                chunk_count: 2,
                chunk_token_start: 100,
                doc_token_len: 200,
                embedding: fake_embedding(8, 3.0),
                content: "gone part 2".to_owned(),
            },
        ];
        store.io_upsert_chunks(domain, "main", "doc/keep", &keep).await.unwrap();
        store.io_upsert_chunks(domain, "main", "doc/gone", &gone).await.unwrap();

        // Delete doc/gone.
        store.io_delete_doc(domain, "main", "doc/gone").await.unwrap();

        // doc/gone: zero chunks. doc/keep: untouched.
        assert_eq!(
            store.io_lookup_doc_chunks(domain, "main", "doc/gone").await.unwrap().len(),
            0,
            "all chunks of doc/gone must be removed"
        );
        assert_eq!(
            store.io_lookup_doc_chunks(domain, "main", "doc/keep").await.unwrap().len(),
            1,
            "doc/keep must be untouched by deleting doc/gone"
        );
    }

    // --- DELETE /domain: removes the dataset + purges state; idempotent ---
    #[tokio::test]
    async fn delete_domain_removes_footprint_and_is_idempotent() {
        let (store, tmp) = make_test_store(8);
        let domain = "admin/doomed";

        // Index a doc on main + a branch, tag commits, record last-indexed.
        let r = vec![ChunkRow {
            doc_id: "doc/1".to_owned(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            embedding: fake_embedding(8, 1.0),
            content: "doomed".to_owned(),
        }];
        let v = store.io_upsert_chunks(domain, "main", "doc/1", &r).await.unwrap();
        store.io_tag_commit(domain, "main", "c0", v).await.unwrap();
        store.update_last_indexed(domain, "main", "c0", v).await;
        store.io_create_branch(domain, "feature", v).await.unwrap();
        store.update_last_indexed(domain, "feature", "c0", v).await;

        // The dataset dir exists on disk.
        let path = tmp.path().join("admin__doomed.lance");
        assert!(path.exists(), "dataset dir should exist before delete");

        // Delete the domain.
        store.io_delete_domain(domain).await.expect("delete domain");

        // On-disk dataset gone.
        assert!(!path.exists(), "dataset dir must be removed");

        // In-memory state purged: a fresh search at c0 must now fail (no dataset).
        // resolve_commit opens the dataset, which no longer exists → None or error.
        // statistics must no longer count this domain.
        let stats = store.statistics().await;
        assert_eq!(stats.domains, 0, "deleted domain must not be counted");
        assert_eq!(stats.branches, 0, "deleted domain's branches must not be counted");

        // Idempotent: a second delete of the same (now-gone) domain succeeds.
        store
            .io_delete_domain(domain)
            .await
            .expect("second delete must be idempotent (not an error)");

        // Idempotent: deleting a never-seen domain succeeds.
        store
            .io_delete_domain("admin/never_existed")
            .await
            .expect("deleting an unknown domain must succeed (idempotent)");
    }

    // --- RRF merge produces correct ranking ---
    #[test]
    fn rrf_merge_combines_ranked_lists() {
        // Vector ranked: A (best), B, C
        let vector_hits = vec![
            ChunkHit {
                doc_id: "A".to_owned(),
                distance: 0.1,
                distance_kind: DistanceKind::RawCosine,
                chunk_index: 0,
                chunk_count: 1,
                chunk_token_start: 0,
                doc_token_len: 10,
                content: "a".to_owned(),
            },
            ChunkHit {
                doc_id: "B".to_owned(),
                distance: 0.3,
                distance_kind: DistanceKind::RawCosine,
                chunk_index: 0,
                chunk_count: 1,
                chunk_token_start: 0,
                doc_token_len: 10,
                content: "b".to_owned(),
            },
            ChunkHit {
                doc_id: "C".to_owned(),
                distance: 0.5,
                distance_kind: DistanceKind::RawCosine,
                chunk_index: 0,
                chunk_count: 1,
                chunk_token_start: 0,
                doc_token_len: 10,
                content: "c".to_owned(),
            },
        ];

        // FTS ranked: B (best), C, D (new — only in FTS)
        let fts_hits = vec![
            ChunkHit {
                doc_id: "B".to_owned(),
                distance: 0.1, // FTS distance (from BM25 conversion).
                distance_kind: DistanceKind::Normalised,
                chunk_index: 0,
                chunk_count: 1,
                chunk_token_start: 0,
                doc_token_len: 10,
                content: "b".to_owned(),
            },
            ChunkHit {
                doc_id: "C".to_owned(),
                distance: 0.2,
                distance_kind: DistanceKind::Normalised,
                chunk_index: 0,
                chunk_count: 1,
                chunk_token_start: 0,
                doc_token_len: 10,
                content: "c".to_owned(),
            },
            ChunkHit {
                doc_id: "D".to_owned(),
                distance: 0.3,
                distance_kind: DistanceKind::Normalised,
                chunk_index: 0,
                chunk_count: 1,
                chunk_token_start: 0,
                doc_token_len: 10,
                content: "d".to_owned(),
            },
        ];

        let merged = rrf_merge(vector_hits, fts_hits);

        // B should be ranked highest: rank 2 in vector + rank 1 in FTS
        // = 1/(60+2) + 1/(60+1) = 1/62 + 1/61
        assert_eq!(merged[0].doc_id, "B", "B should rank first (appears high in both lists)");

        // All 4 unique docs should appear.
        let ids: Vec<&str> = merged.iter().map(|h| h.doc_id.as_str()).collect();
        assert!(ids.contains(&"A"));
        assert!(ids.contains(&"B"));
        assert!(ids.contains(&"C"));
        assert!(ids.contains(&"D"));
        assert_eq!(ids.len(), 4);
    }

    // Fix #4: VectorIndexConfig dimension validation and divisor guarantee.

    #[test]
    fn largest_divisor_leq_standard_dims() {
        // 768-d (nomic-embed-v2): target = 768/16 = 48, 768%48 = 0 → 48
        assert_eq!(super::largest_divisor_leq(768, 48), 48);
        // 128-d: target = 128/8 = 16, 128%16 = 0 → 16
        assert_eq!(super::largest_divisor_leq(128, 16), 16);
        // 384-d: target = 384/16 = 24, 384%24 = 0 → 24
        assert_eq!(super::largest_divisor_leq(384, 24), 24);
        // 1536-d: target = 1536/16 = 96, 1536%96 = 0 → 96
        assert_eq!(super::largest_divisor_leq(1536, 96), 96);
    }

    #[test]
    fn largest_divisor_leq_non_standard_dims() {
        // 500-d: target = 500/16 = 31. 500%31 = 500-31*16 = 500-496 = 4 ≠ 0.
        // Largest divisor of 500 <= 31: 500 = 2^2 * 5^3. Divisors: 1,2,4,5,10,20,25,50,100...
        // 25 <= 31 and 500%25 = 0.
        assert_eq!(super::largest_divisor_leq(500, 31), 25);
        // 130-d: target = 130/8 = 16. 130%16 = 2 ≠ 0.
        // Divisors of 130: 1,2,5,10,13,26,65,130. Largest <= 16: 13.
        assert_eq!(super::largest_divisor_leq(130, 16), 13);
    }

    #[test]
    fn largest_divisor_leq_prime_dim() {
        // 127 is prime. Target = 127/8 = 15. Only divisors: 1, 127.
        // Largest <= 15: 1.
        assert_eq!(super::largest_divisor_leq(127, 15), 1);
    }

    #[test]
    fn vector_index_config_guarantees_divisibility() {
        // Test various dimensions — all must produce num_sub_vectors that divides dim.
        let test_dims = [128, 256, 384, 500, 512, 768, 1024, 1536, 130, 127, 100, 200];
        for dim in test_dims {
            let config = VectorIndexConfig::default_for_dim(dim);
            assert_eq!(
                dim % config.num_sub_vectors, 0,
                "dim={} must be divisible by num_sub_vectors={} (got remainder {})",
                dim, config.num_sub_vectors, dim % config.num_sub_vectors
            );
            assert!(
                config.num_sub_vectors >= 1,
                "num_sub_vectors must be at least 1 for dim={}",
                dim
            );
        }
    }

    #[test]
    #[should_panic(expected = "embedding dimension must be > 0")]
    fn vector_index_config_zero_dim_panics() {
        VectorIndexConfig::default_for_dim(0);
    }

    fn one_row(dim: usize, seed: f32, content: &str) -> ChunkRow {
        ChunkRow {
            doc_id: "doc/x".to_owned(),
            doc_type: "Doc".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            embedding: fake_embedding(dim, seed),
            content: content.to_owned(),
        }
    }

    // --- #3: io_resolve_commit returns Ok(None) for a genuinely-absent domain,
    //     and NEVER auto-creates the dataset (BLOCKER-2 read-path guard) ---
    #[tokio::test]
    async fn resolve_commit_absent_domain_is_none_and_does_not_create() {
        let (store, tmp) = make_test_store(8);
        let resolved = store
            .io_resolve_commit("admin/never", "main", "c0")
            .await
            .expect("resolve must not error on an absent domain");
        assert_eq!(resolved, None, "absent domain → not indexed");
        // The read must NOT have created a dataset directory on disk.
        let path = tmp.path().join("admin__never.lance");
        assert!(
            !path.exists(),
            "io_resolve_commit must not auto-create the dataset (resurrection guard)"
        );
    }

    // --- #3: a tag that exists resolves; a tag that is absent (but domain
    //     exists) is Ok(None), distinct from an error ---
    #[tokio::test]
    async fn resolve_commit_distinguishes_absent_tag_from_error() {
        let (store, _tmp) = make_test_store(8);
        let domain = "admin/resolve3";
        let r = one_row(8, 1.0, "hello world");
        let v = store
            .io_upsert_chunks(domain, "main", "doc/x", std::slice::from_ref(&r))
            .await
            .unwrap();
        store.io_tag_commit(domain, "main", "c0", v).await.unwrap();

        // Indexed commit → Some.
        assert_eq!(
            store.io_resolve_commit(domain, "main", "c0").await.unwrap(),
            Some(v)
        );
        // A different, never-tagged commit on an existing domain → Ok(None),
        // NOT an error.
        assert_eq!(
            store.io_resolve_commit(domain, "main", "c_absent").await.unwrap(),
            None
        );
    }

    // --- #2: a read (io_search) against a never-indexed domain does NOT create
    //     the dataset and surfaces "not indexed" rather than empty success ---
    #[tokio::test]
    async fn search_absent_domain_does_not_create_dataset() {
        let (store, tmp) = make_test_store(8);
        let query = SearchQuery {
            query_embedding: fake_embedding(8, 1.0),
            query_text: "anything".to_owned(),
            mode: SearchMode::Vector,
            start: 0,
            count: 5,
            doc_type_filter: vec![],
            doc_id_filter: vec![],
            snippet: false,
        };
        let res = store.io_search("admin/ghost", "main", "c0", &query).await;
        assert!(res.is_err(), "search on an absent domain must fail (not indexed), not succeed empty");
        let path = tmp.path().join("admin__ghost.lance");
        assert!(!path.exists(), "search must not auto-create the dataset");
    }

    // --- #2: after io_delete_domain, a resolve does NOT resurrect the dataset ---
    #[tokio::test]
    async fn delete_domain_then_resolve_does_not_resurrect() {
        let (store, tmp) = make_test_store(8);
        let domain = "admin/del_resurrect";
        let r = one_row(8, 1.0, "doc to delete");
        let v = store
            .io_upsert_chunks(domain, "main", "doc/x", std::slice::from_ref(&r))
            .await
            .unwrap();
        store.io_tag_commit(domain, "main", "c0", v).await.unwrap();
        let path = tmp.path().join("admin__del_resurrect.lance");
        assert!(path.exists(), "dataset exists after indexing");

        store.io_delete_domain(domain).await.unwrap();
        assert!(!path.exists(), "dataset removed by delete");

        // Resolve after delete must be None AND must not recreate the dir.
        let resolved = store.io_resolve_commit(domain, "main", "c0").await.unwrap();
        assert_eq!(resolved, None);
        assert!(!path.exists(), "resolve must not resurrect the deleted dataset");
    }

    // --- #4: two concurrent first-pushes to the SAME new branch both succeed
    //     (idempotent branch-out, no "already exists" 500 for the loser) ---
    #[tokio::test]
    async fn concurrent_branch_out_both_succeed() {
        use std::sync::Arc;
        let (store, _tmp) = make_test_store(8);
        let store = Arc::new(store);
        let domain = "admin/race";
        // Seed main @ c0 so the parent is indexed.
        let r = one_row(8, 1.0, "parent doc");
        let v = store
            .io_upsert_chunks(domain, "main", "doc/x", std::slice::from_ref(&r))
            .await
            .unwrap();
        store.io_tag_commit(domain, "main", "c0", v).await.unwrap();

        // Fire two concurrent branch-outs of the same new branch from c0.
        let s1 = Arc::clone(&store);
        let s2 = Arc::clone(&store);
        let h1 = tokio::spawn(async move {
            crate::store::branch::io_ensure_branch_forked(&s1, domain, "feature", "c0").await
        });
        let h2 = tokio::spawn(async move {
            crate::store::branch::io_ensure_branch_forked(&s2, domain, "feature", "c0").await
        });
        let r1 = h1.await.unwrap();
        let r2 = h2.await.unwrap();
        assert!(r1.is_ok(), "first branch-out must succeed: {:?}", r1);
        assert!(
            r2.is_ok(),
            "concurrent branch-out must be idempotent, not 500: {:?}",
            r2
        );
        // Exactly one feature branch exists.
        let branches = store.io_list_branches(domain).await.unwrap();
        assert!(branches.iter().any(|b| b == "feature"));
    }

    /// Build a RecordBatch with the base chunk columns but NO `_distance` column
    /// (the shape a vector search would have if the scanner failed to attach
    /// distances).
    fn batch_without_distance() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Utf8, false),
            Field::new("doc_type", DataType::Utf8, false),
            Field::new("chunk_index", DataType::Int32, false),
            Field::new("chunk_count", DataType::Int32, false),
            Field::new("chunk_token_start", DataType::Int32, false),
            Field::new("doc_token_len", DataType::Int32, false),
            Field::new("content", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["doc/1"])),
                Arc::new(StringArray::from(vec!["Doc"])),
                Arc::new(Int32Array::from(vec![0])),
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(Int32Array::from(vec![0])),
                Arc::new(Int32Array::from(vec![10])),
                Arc::new(StringArray::from(vec!["content"])),
            ],
        )
        .expect("batch")
    }

    // --- BUG-409: io_resolve_commit must see a tag created AFTER the cached
    //     handle was last refreshed. This replicates the LIVE pipeline ordering
    //     that the simple round-trip test misses:
    //       1. upsert chunks            (refreshes cached handle → H_data)
    //       2. optimize indices         (refreshes cached handle → H_index)
    //       3. tag commit               (io_tag_commit on the cached handle)
    //       4. resolve commit (the GUARD)
    //     If io_resolve_commit reads a STALE cached handle that predates the tag
    //     write, it returns None for an indexed commit — the 409 guard then lets
    //     a re-push of an already-indexed commit through (returns 200, BUG).
    //
    //     We drive the refresh between upsert and tag explicitly to mirror the
    //     optimize step's `io_refresh_cached_dataset`, then assert resolve sees
    //     the tag. This is the store-level pinpoint of the live 409 bug.
    #[tokio::test]
    async fn resolve_commit_sees_tag_created_after_cache_refresh() {
        let (store, _tmp) = make_test_store(8);
        let domain = "admin/bug409";

        // 1. Upsert (this refreshes the cached handle to the data version).
        let r = one_row(8, 1.0, "indexed content for bug 409");
        let version = store
            .io_upsert_chunks(domain, "main", "doc/x", std::slice::from_ref(&r))
            .await
            .expect("upsert");
        assert!(version > 0);

        // 2. Mirror the optimize step: refresh the cached handle AGAIN, so the
        //    cached Dataset is a handle opened BEFORE the tag is written.
        store
            .io_refresh_cached_dataset(domain, "main")
            .await
            .expect("refresh (mirrors optimize)");

        // 3. Tag the commit (the worker step). After this the commit is indexed.
        store
            .io_tag_commit(domain, "main", "rc0", version)
            .await
            .expect("tag commit");

        // 4. The GUARD: io_resolve_commit must now report rc0 as indexed.
        let resolved = store
            .io_resolve_commit(domain, "main", "rc0")
            .await
            .expect("resolve");
        assert_eq!(
            resolved,
            Some(version),
            "io_resolve_commit must see a commit tagged after the cached handle was \
             refreshed — a stale cached handle that returns None here is the root \
             cause of the 409 guard letting a re-push of an indexed commit through"
        );

        // The dataset-global list view must agree (search resolution path).
        let versions = store
            .io_list_commit_versions(domain)
            .await
            .expect("list commit versions");
        assert_eq!(
            versions.get("rc0").copied(),
            Some(version),
            "io_list_commit_versions (search resolution) must also see the tag"
        );
    }

    // --- 409 state machine: reserve once, reject the second reservation of the
    //     same in-flight commit, release allows a retry ---
    #[tokio::test]
    async fn reserve_commit_rejects_inflight_then_release_allows_retry() {
        let (store, _tmp) = make_test_store(8);
        let domain = "admin/reserve";

        // First reservation of an absent commit succeeds.
        let first = store
            .io_try_reserve_commit(domain, "main", "c1")
            .await
            .expect("reserve c1");
        assert!(first, "first reservation of an absent commit must succeed");

        // A second reservation of the SAME in-flight commit is rejected (Reserved
        // state → 409), even though it is not yet tagged/Indexed.
        let second = store
            .io_try_reserve_commit(domain, "main", "c1")
            .await
            .expect("re-reserve c1");
        assert!(
            !second,
            "re-reserving an in-flight (Reserved) commit must be rejected"
        );

        // Releasing the reservation (terminal: e.g. index failed) returns the
        // commit to absent — a retry is then allowed (not blocked forever).
        store
            .io_release_commit_reservation(domain, "main", "c1")
            .await;
        let retry = store
            .io_try_reserve_commit(domain, "main", "c1")
            .await
            .expect("retry reserve c1");
        assert!(
            retry,
            "after release (failed index), a retry of the same commit must be allowed"
        );
    }

    // --- 409 state machine: an INDEXED (tagged) commit is rejected even with no
    //     active reservation (the durable tag is the Indexed marker) ---
    #[tokio::test]
    async fn reserve_commit_rejects_already_indexed() {
        let (store, _tmp) = make_test_store(8);
        let domain = "admin/reserve_indexed";

        // Index + tag c0 (Indexed state), then drop the reservation as the worker
        // does on success.
        let r = one_row(8, 1.0, "indexed doc");
        let v = store
            .io_upsert_chunks(domain, "main", "doc/x", std::slice::from_ref(&r))
            .await
            .expect("upsert");
        store
            .io_tag_commit(domain, "main", "c0", v)
            .await
            .expect("tag c0");

        // A push of an already-Indexed commit must be rejected (no reservation
        // exists, but the tag does).
        let reserved = store
            .io_try_reserve_commit(domain, "main", "c0")
            .await
            .expect("reserve c0");
        assert!(
            !reserved,
            "re-pushing an already-indexed (tagged) commit must be rejected"
        );
    }

    // --- 409 state machine: two CONCURRENT reservations of the same new commit —
    //     exactly one wins (atomic check-and-reserve, no TOCTOU) ---
    #[tokio::test]
    async fn concurrent_reserve_same_commit_exactly_one_wins() {
        use std::sync::Arc;
        let (store, _tmp) = make_test_store(8);
        let store = Arc::new(store);
        let domain = "admin/reserve_race";

        let s1 = Arc::clone(&store);
        let s2 = Arc::clone(&store);
        let h1 = tokio::spawn(async move { s1.io_try_reserve_commit(domain, "main", "c9").await });
        let h2 = tokio::spawn(async move { s2.io_try_reserve_commit(domain, "main", "c9").await });
        let r1 = h1.await.unwrap().expect("reserve 1");
        let r2 = h2.await.unwrap().expect("reserve 2");

        assert_ne!(
            r1, r2,
            "exactly one concurrent reservation of the same commit must win (got r1={}, r2={})",
            r1, r2
        );
        assert!(r1 || r2, "at least one reservation must have succeeded");
    }

    // --- #E: a ranked vector search with a MISSING `_distance` column fails
    //     loud rather than defaulting distances to 0.0 (which would corrupt
    //     ranking). A plain scan (require_distance=false) tolerates absence. ---
    #[test]
    fn vector_hits_missing_distance_fails_loud_when_required() {
        let batches = vec![batch_without_distance()];
        let err = batches_to_vector_hits(&batches, true);
        assert!(
            err.is_err(),
            "missing _distance on a ranked search must error, not default to 0.0"
        );

        // The same batch is fine for a plain scan (no ranking expected).
        let ok = batches_to_vector_hits(&batches, false);
        assert!(ok.is_ok(), "a plain scan tolerates absent _distance");
        assert_eq!(ok.unwrap().len(), 1);
    }

    /// Count this process's currently-open file descriptors (Linux: /proc/self/fd).
    /// Used by the FD-exhaustion regression test to prove search no longer leaks
    /// descriptors under sustained load.
    #[cfg(target_os = "linux")]
    fn open_fd_count() -> usize {
        std::fs::read_dir("/proc/self/fd")
            .expect("read /proc/self/fd")
            .count()
    }

    // --- BUG-FD24: under sustained search load the engine exhausted file
    //     descriptors ("Too many open files (os error 24)"). The bench pinpointed
    //     the mechanism: ~2 FDs leaked PER /search — the Lance VECTOR-INDEX reader
    //     files (`_indices/<uuid>/index.idx` + `auxiliary.idx`), opened when the
    //     ANN `nearest()` runs through a FRESHLY-`Dataset::open`ed handle (a new
    //     object_store + session) and NOT released before the call returns. The
    //     count climbed monotonically (past 2100) and exhausted the default soft
    //     limit (~1024) after ~140 searches.
    //
    //     This test builds a domain WITH a real vector (ANN) index — the leak is
    //     index-reader-bound, so the index MUST exist — then issues many vector
    //     searches and asserts BOTH (a) the process open-FD count stays FLAT and
    //     (b) reads perform no fresh `Dataset::open` per query. RED against
    //     fresh-open-per-search (FDs climb + opens == searches); GREEN once reads
    //     reuse the cached handle (one shared object_store + session → index
    //     readers bounded to one set, FDs flat).
    #[cfg(target_os = "linux")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn search_does_not_leak_file_descriptors_under_load() {
        let dim = 16;
        // IVF_PQ needs >= 256 training vectors; a few partitions for a small corpus.
        let config = VectorIndexConfig {
            num_partitions: 4,
            num_sub_vectors: 8,
            nprobes: 4,
            refine_factor: Some(10),
        };
        let (mut store, _tmp) = make_test_store(dim);
        store.set_vector_index_config(config.clone());
        let domain = "admin/fdload";

        // One doc with 300 chunks → 300 rows, above the 256 index-training floor,
        // built in a single fast upsert (avoids hundreds of sequential writes).
        let corpus = 300usize;
        let rows: Vec<ChunkRow> = (0..corpus)
            .map(|i| ChunkRow {
                doc_id: "doc/corpus".to_owned(),
                doc_type: "Doc".to_owned(),
                chunk_index: i as i32,
                chunk_count: corpus as i32,
                chunk_token_start: i as i32,
                doc_token_len: corpus as i32,
                embedding: fake_embedding(dim, 1.0 + i as f32),
                content: format!("chunk content number {} lorem ipsum dolor", i),
            })
            .collect();
        store
            .io_upsert_chunks(domain, "main", "doc/corpus", &rows)
            .await
            .expect("upsert corpus");

        // Build the vector (ANN) index — the leaked FDs are its reader files.
        {
            let ds_arc = store.io_open_dataset(domain, "main").await.unwrap();
            let mut ds = ds_arc.write().await;
            crate::store::vector_index::io_ensure_vector_index(&mut ds, &config)
                .await
                .expect("ensure vector index");
        }
        // Refresh the cached handle so it reflects the new index version, then tag.
        store
            .io_refresh_cached_dataset(domain, "main")
            .await
            .expect("refresh after index build");
        let indexed_version = {
            let ds_arc = store.io_open_dataset(domain, "main").await.unwrap();
            let guard = ds_arc.read().await;
            guard.version().version
        };
        store
            .io_tag_commit(domain, "main", "c_load", indexed_version)
            .await
            .expect("tag c_load");

        let query = SearchQuery {
            query_embedding: fake_embedding(dim, 9999.0),
            query_text: String::new(),
            mode: SearchMode::Vector,
            start: 0,
            count: 5,
            doc_type_filter: Vec::new(),
            doc_id_filter: Vec::new(),
            snippet: false,
        };

        // Warm up: the first search opens/populates the cached handle and loads
        // the index reader once. Measure baselines AFTER warm-up so we compare
        // steady-state to steady-state.
        store
            .io_search(domain, "main", "c_load", &query)
            .await
            .expect("warmup search");
        let baseline_fds = open_fd_count();
        let baseline_opens = store.fresh_open_count();

        // Sustained CONCURRENT load (matches the live server). With fresh-open per
        // search, the ANN index reader FDs leak ~2/search and the count climbs;
        // with the cached handle reused, the count stays flat.
        let load_iterations = 400usize;
        let concurrency = 8usize;
        let store = std::sync::Arc::new(store);
        for _ in 0..(load_iterations / concurrency) {
            let mut handles = Vec::with_capacity(concurrency);
            for _ in 0..concurrency {
                let s = std::sync::Arc::clone(&store);
                let q = query.clone();
                handles.push(tokio::spawn(async move {
                    s.io_search(domain, "main", "c_load", &q).await
                }));
            }
            for h in handles {
                h.await
                    .expect("search task join")
                    .expect("load search must succeed (no FD exhaustion)");
            }
        }
        let after_fds = open_fd_count();
        let opens_added = store.fresh_open_count() - baseline_opens;
        eprintln!(
            "[fd-load] searches={} fresh_opens_added={} fds(baseline={}, after={}, delta={})",
            load_iterations,
            opens_added,
            baseline_fds,
            after_fds,
            after_fds as i64 - baseline_fds as i64
        );

        // PRIMARY: searches must NOT open a fresh dataset per call (each fresh
        // open is the new object_store/session that leaks the index reader FDs).
        assert!(
            opens_added < (load_iterations as u64) / 4,
            "search opened a fresh dataset per call ({} fresh opens across {} searches). \
             Reads must reuse the cached domain handle and checkout_version off it \
             (sharing one object_store + session), not Dataset::open fresh every query \
             — the fresh open leaks the ANN index reader FDs (BUG-FD24).",
            opens_added,
            load_iterations
        );

        // SECONDARY: open FD count must stay FLAT under load (the bench saw it
        // climb past 2100 unbounded). Slack covers runtime/allocator churn only —
        // it must NOT scale with the number of searches.
        let slack = 64;
        assert!(
            after_fds <= baseline_fds + slack,
            "open FD count grew under search load (baseline={}, after {} searches={}, slack={}). \
             The ANN index reader FDs are leaking per search — reads must reuse the \
             cached handle so the index readers are bounded to one set (BUG-FD24).",
            baseline_fds,
            load_iterations,
            after_fds,
            slack
        );
    }
}
