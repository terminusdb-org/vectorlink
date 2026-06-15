//! Commit/version lifecycle: tag, assign, resolve, reserve, last-indexed.

use std::collections::HashMap;
use std::sync::Arc;

use crate::kernel::error::StoreError;
use crate::kernel::model::{BranchName, Domain, LastIndexed, TaskStatus};
use crate::layeridx::{self, BranchIndex};

use super::{BranchKey, LanceStore, MAIN_BRANCH};

impl LanceStore {
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
            // CACHE COHERENCE (task-durable-index-state): keep the in-memory
            // last-indexed cache from EVER lagging the durable tag we just wrote.
            self.cache_last_indexed(domain, branch, commit, version).await;
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
        // CACHE COHERENCE (task-durable-index-state): the just-written tag is the
        // newest indexed commit on this branch — update the cache so it never
        // lags the durable truth (the resume base after any restart).
        self.cache_last_indexed(domain, branch, commit, version).await;
        Ok(())
    }

    /// Update the in-memory `branch_indexes` last-indexed CACHE to reflect a
    /// just-tagged commit. The durable authority is always the Lance tag (read
    /// by `io_derive_last_indexed`); this keeps the accelerator coherent so a
    /// cache HIT can never report a commit OLDER than the latest durable tag.
    /// Called from every path that writes a commit tag.
    pub(super) async fn cache_last_indexed(&self, domain: &str, branch: &str, commit: &str, version: u64) {
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

    /// DURABLE per-branch last-indexed, derived ENTIRELY from the on-disk Lance
    /// tags (task-durable-index-state). This is the authority that survives a
    /// process restart — the in-memory `branch_indexes` map is only a cache that
    /// is rebuilt from this on a miss.
    ///
    /// HOW IT WORKS: every indexed commit is a dataset-global Lance tag, and
    /// Lance records the branch it was created on in `TagContents.branch`
    /// (`None` == the default/main branch; `Some(b)` for a non-main branch — see
    /// Lance `standardize_branch`). We therefore filter the tag set to the tags
    /// belonging to `branch` and pick the most-recently-created one. "Last
    /// indexed" is ordered by `created_at` (the durable wall-clock the tag was
    /// written), with the Lance `version` as a deterministic tie-break for the
    /// rare same-instant case. The version returned is the tagged dataset
    /// version, identical to what `update_last_indexed` would have cached.
    ///
    /// INVARIANT (the restart fix): a branch with an on-disk tag NEVER resolves
    /// to `None` here just because the process was restarted — the answer comes
    /// from disk, not from a volatile map. A branch with no matching tag (never
    /// indexed) correctly resolves to `None`.
    ///
    /// READ-ONLY: an absent dataset → `None` (never auto-creates — resurrection
    /// guard). FAIL-LOUD: a real `tags().list()` I/O error propagates.
    async fn io_derive_last_indexed(
        &self,
        domain: &str,
        branch: &str,
    ) -> Result<Option<(String, u64)>, StoreError> {
        let cached = match self.io_open_dataset_readonly(domain).await? {
            Some(ds) => ds,
            None => return Ok(None),
        };
        let ds = cached.read().await;
        let tags = ds
            .tags()
            .list()
            .await
            .map_err(|e| StoreError::Internal(format!("tag list failed: {}", e)))?;
        drop(ds);

        // A tag belongs to `branch` iff its recorded branch matches Lance's
        // canonical form: `None` for main, `Some(branch)` otherwise.
        let want_branch: Option<&str> = if branch == MAIN_BRANCH { None } else { Some(branch) };

        let best = tags
            .into_iter()
            .filter_map(|(tag_name, contents)| {
                layeridx::decode_commit_tag(&tag_name)
                    .ok()
                    .map(|commit| (commit, contents))
            })
            .filter(|(_, contents)| contents.branch.as_deref() == want_branch)
            .max_by(|(_, a), (_, b)| {
                // Latest wins: order by creation time, then version as a stable
                // tie-break. `created_at` is the durable ordering signal.
                a.created_at
                    .cmp(&b.created_at)
                    .then(a.version.cmp(&b.version))
            });

        Ok(best.map(|(commit, contents)| (commit, contents.version)))
    }

    /// Get last-indexed for a (domain, branch) pair.
    ///
    /// DURABLE-FIRST (task-durable-index-state): the authority is the on-disk
    /// Lance tags, NOT the in-memory `branch_indexes` map. The map is only a
    /// cache. On a cache HIT we serve it (the fast path for a freshly-pushed
    /// branch in this process). On a cache MISS — which is exactly the state
    /// after a restart, when the map is empty but the tags are on disk — we
    /// derive the answer from disk and populate the cache so the next read is
    /// fast. This is the fix for "a restart makes an indexed branch look
    /// un-indexed": the answer can never be a spurious `None` for a branch whose
    /// index is on disk.
    ///
    /// FAIL-LOUD: a real tag-list I/O error during derivation propagates as the
    /// durable-derivation error rather than being silently downgraded to `None`
    /// — a corrupt/unreadable store must not masquerade as "never indexed".
    pub async fn last_indexed(
        &self,
        domain: &Domain,
        branch: &BranchName,
    ) -> Result<LastIndexed, StoreError> {
        let key = (domain.as_str().to_owned(), branch.as_str().to_owned());

        // Fast path: cache hit.
        {
            let indexes = self.branch_indexes.read().await;
            if let Some(bi) = indexes.get(&key) {
                return Ok(LastIndexed {
                    branch: branch.as_str().to_owned(),
                    commit: bi.commit.clone(),
                    version: bi.version,
                });
            }
        }

        // Cache miss (cold process / post-restart): derive from durable disk.
        match self.io_derive_last_indexed(domain.as_str(), branch.as_str()).await? {
            Some((commit, version)) => {
                // Populate the cache so subsequent reads are O(1). A concurrent
                // writer's entry (if any) is authoritative — it reflects a more
                // recent push — so only insert when still absent.
                {
                    let mut indexes = self.branch_indexes.write().await;
                    indexes.entry(key).or_insert(BranchIndex {
                        commit: Some(commit.clone()),
                        version,
                    });
                }
                Ok(LastIndexed {
                    branch: branch.as_str().to_owned(),
                    commit: Some(commit),
                    version,
                })
            }
            None => Ok(LastIndexed {
                branch: branch.as_str().to_owned(),
                commit: None,
                version: 0,
            }),
        }
    }

    /// Update last-indexed tracking.
    ///
    /// NOTE (task-durable-index-state): this refreshes the in-memory cache only.
    /// `io_tag_commit` ALREADY refreshes the same cache entry as part of writing
    /// the durable tag, so calling this after a tag is now redundant-but-harmless
    /// (it writes the identical value). The durable authority is the Lance tag,
    /// read by `io_derive_last_indexed`; this cache is only an accelerator.
    pub async fn update_last_indexed(
        &self,
        domain: &str,
        branch: &str,
        commit: &str,
        version: u64,
    ) {
        self.cache_last_indexed(domain, branch, commit, version).await;
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
}
