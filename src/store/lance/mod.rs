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
mod search;
mod resolve;
pub mod dedup;
mod stats;

#[cfg(test)]
#[allow(clippy::useless_vec)]
mod tests;

// --- Public API facade (preserves all existing import paths) ---

pub use self::config::VectorIndexConfig;
pub use self::schema::{
    ChunkHit, ChunkRow, DistanceKind, DuplicateScope, NeighbourObservation,
    ResolveNeighbourMaps, SearchQuery, DEFAULT_DUPLICATE_MAX_PAIRS,
    DEFAULT_DUPLICATE_MAX_POINTS, MAIN_BRANCH,
};
pub use self::dedup::{dedup_chunks_to_documents, pairs_from_neighbours};
pub use self::index::{io_compact_data, io_ensure_fts_index_on_dataset};


use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{
    FixedSizeListArray, Float32Array, Int32Array, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use lance::dataset::{Dataset, WriteMode, WriteParams};
use tokio::sync::RwLock;

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
    dim: usize,
    /// Vector index configuration (nprobes, refine_factor for search).
    pub(super) vector_index_config: VectorIndexConfig,
    /// Open dataset handles, keyed by DOMAIN (layout A). The cached handle is
    /// the domain dataset opened at its default (main) branch head; branch
    /// writes check out a branch-bound handle without mutating this one.
    pub(super) datasets: RwLock<HashMap<String, Arc<RwLock<Dataset>>>>,
    /// Per-(domain, branch) index tracking (branch-precise).
    pub(super) branch_indexes: RwLock<HashMap<BranchKey, BranchIndex>>,
    /// Tasks by task ID.
    pub(super) tasks: RwLock<HashMap<String, crate::kernel::model::TaskStatus>>,
    /// Per-(domain, branch) pipeline serialisation lock.
    /// Ensures concurrent pushes to the same branch are serialised so that
    /// commit→version tags are correctly isolated.
    pub(super) pipeline_locks: RwLock<HashMap<BranchKey, Arc<tokio::sync::Mutex<()>>>>,
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
    /// Per-(domain, branch) guard preventing concurrent background compactions.
    /// A domain/branch key is present while a background compaction task is
    /// in-flight; a 5% roll that hits while a compaction is already running is
    /// a no-op. Guards are removed on task completion (success or failure).
    pub(super) compaction_in_progress: RwLock<std::collections::HashSet<BranchKey>>,
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
            compaction_in_progress: RwLock::new(std::collections::HashSet::new()),
        }
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

    /// Acquire the per-domain create/delete guard (BLOCKER-2 / #6). Held by a
    /// dataset-creating write across the create, and by `DELETE /domain` across
    /// the whole remove-then-purge — so the two never interleave.
    pub(super) async fn acquire_domain_guard(&self, domain: &str) -> tokio::sync::OwnedMutexGuard<()> {
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
    /// Includes BOTH `embedding` (document-role, ANN-INDEXED) and `query_embedding`
    /// (query-role, STORED but NOT INDEXED — Phase 6A Step 5 dual-vector).
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
            // Query-role embedding: stored for /resolve and /duplicates to probe
            // with the asymmetric query→document signal. NOT ANN-indexed — the
            // only index lives on `embedding`. This column is a pure storage
            // column read by `io_scan_points` during the resolve/duplicates scan.
            Field::new(
                "query_embedding",
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
    pub(super) fn dataset_path(&self, domain: &str) -> PathBuf {
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
    pub(super) async fn io_open_branch_for_write(
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
                // query_embedding (same shape as embedding, stored but NOT indexed).
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

        // Build the query_embedding FixedSizeList (query-role, stored NOT indexed).
        let flat_query_embeddings: Vec<f32> = rows
            .iter()
            .flat_map(|r| r.query_embedding.iter().copied())
            .collect();
        let query_values = Float32Array::from(flat_query_embeddings);
        let query_embedding_array = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            self.dim as i32,
            Arc::new(query_values),
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
                Arc::new(query_embedding_array) as Arc<dyn arrow_array::Array>,
                Arc::new(StringArray::from(contents)),
            ],
        )
        .map_err(|e| StoreError::Internal(format!("batch construction failed: {}", e)))
    }
}
