#![forbid(unsafe_code)]

//! Branch-out — forking a TerminusDB branch from a parent commit (layout A).
//!
//! A branch-out forks linearly from a parent commit's tagged Lance version. The
//! child branch SHARES the parent's fragment files (block reuse, RISK-01) and
//! writes only deltas thereafter. Parent-commit resolution comes from the push
//! handshake (`parent_commit` on `POST /push`).
//!
//! Linear-per-branch only (locked decision): there is no merge here. A merge in
//! TerminusDB arrives as ordinary linear appends of changed docs on the target
//! branch. This module supports BRANCH-OUT exclusively.

use crate::kernel::error::StoreError;
use crate::store::lance::{LanceStore, MAIN_BRANCH};

/// Ensure `branch` exists in the domain dataset, forking from `parent_commit`'s
/// tagged version if it does not.
///
/// Returns `BranchOutcome` describing what happened so the caller can act
/// (e.g. auto-enroll the new branch in the layer index).
///
/// Pure-by-contract on inputs; all I/O is delegated to the store (`io_*`).
///
/// INVARIANT (fail loud): if `branch` does not exist AND `parent_commit` is not
/// resolvable to a version in the domain dataset, this is an error — we never
/// silently create an empty branch or guess a parent.
///
/// `branch == "main"` always exists (the Lance native default branch); a
/// branch-out request for main is a no-op `AlreadyExists`.
pub async fn io_ensure_branch_forked(
    store: &LanceStore,
    domain: &str,
    branch: &str,
    parent_commit: &str,
) -> Result<BranchOutcome, StoreError> {
    if branch == MAIN_BRANCH {
        return Ok(BranchOutcome::AlreadyExists);
    }

    let existing = store.io_list_branches(domain).await?;
    if existing.iter().any(|b| b == branch) {
        return Ok(BranchOutcome::AlreadyExists);
    }

    // Branch does not exist — resolve the parent commit to a version and fork.
    let parent_version = store
        .io_resolve_commit(domain, MAIN_BRANCH, parent_commit)
        .await?
        .ok_or_else(|| {
            StoreError::Internal(format!(
                "cannot branch '{}' in domain '{}': parent commit '{}' is not indexed \
                 (no tagged version to fork from)",
                branch, domain, parent_commit
            ))
        })?;

    store
        .io_create_branch(domain, branch, parent_version)
        .await?;

    Ok(BranchOutcome::Created { parent_version })
}

/// Result of a branch-out attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchOutcome {
    /// The branch already existed (no fork performed).
    AlreadyExists,
    /// The branch was newly forked from `parent_version`.
    Created { parent_version: u64 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::lance::{ChunkRow, LanceStore};

    fn make_test_store(dim: usize) -> (LanceStore, tempfile::TempDir) {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let store = LanceStore::new(tmp.path(), dim);
        (store, tmp)
    }

    fn fake_embedding(dim: usize, seed: f32) -> Vec<f32> {
        (0..dim).map(|i| (seed + i as f32 * 0.01).sin()).collect()
    }

    fn row(doc_id: &str, dim: usize, seed: f32, content: &str) -> ChunkRow {
        ChunkRow {
            doc_id: doc_id.to_owned(),
            doc_type: "Doc".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            embedding: fake_embedding(dim, seed),
            content: content.to_owned(),
        }
    }

    /// Index `n` docs on main, tag a commit, return the tagged version.
    async fn seed_main_commit(
        store: &LanceStore,
        domain: &str,
        commit: &str,
        doc_ids: &[&str],
    ) -> u64 {
        let mut version = 0;
        for (i, id) in doc_ids.iter().enumerate() {
            let r = row(id, 8, i as f32 + 1.0, &format!("content for {}", id));
            version = store
                .io_upsert_chunks(domain, "main", id, std::slice::from_ref(&r))
                .await
                .expect("upsert");
        }
        store
            .io_tag_commit(domain, "main", commit, version)
            .await
            .expect("tag");
        version
    }

    // --- branch-out from a parent that doesn't exist fails loud ---
    #[tokio::test]
    async fn branch_out_unindexed_parent_fails_loud() {
        let (store, _tmp) = make_test_store(8);
        // Domain dataset doesn't even exist yet → must error, not silently create.
        let result =
            io_ensure_branch_forked(&store, "admin/empty", "feature", "no_such_commit").await;
        assert!(result.is_err(), "branching an unindexed parent must fail loud");
    }

    // --- branch-out on main is a no-op AlreadyExists ---
    #[tokio::test]
    async fn branch_out_main_is_noop() {
        let (store, _tmp) = make_test_store(8);
        let outcome = io_ensure_branch_forked(&store, "admin/db", "main", "anything")
            .await
            .expect("main branch-out");
        assert_eq!(outcome, BranchOutcome::AlreadyExists);
    }

    // --- P3-BR-1: branch from c0 sees c0's docs WITHOUT re-indexing; shared fragments ---
    #[tokio::test]
    async fn branch_shares_parent_fragment_files_by_path_identity() {
        let (store, _tmp) = make_test_store(8);
        let domain = "admin/sw";

        // Seed main with two docs, tag c0.
        let _v0 = seed_main_commit(&store, domain, "c0", &["People/1", "People/2"]).await;

        // Capture main's fragment file paths at this point.
        let main_files = store
            .io_branch_data_file_paths(domain, "main")
            .await
            .expect("main files");
        assert!(!main_files.is_empty(), "main must have fragment files");

        // Branch out at c0.
        let outcome = io_ensure_branch_forked(&store, domain, "feature", "c0")
            .await
            .expect("branch out");
        assert!(matches!(outcome, BranchOutcome::Created { .. }));

        // The branch must reference the SAME physical fragment files (block reuse).
        let branch_files = store
            .io_branch_data_file_paths(domain, "feature")
            .await
            .expect("branch files");
        let shared: Vec<_> = branch_files.intersection(&main_files).collect();
        assert_eq!(
            shared.len(),
            main_files.len(),
            "branch must share ALL parent fragment files by path identity (block reuse). \
             main={:?} branch={:?}",
            main_files,
            branch_files,
        );
        // And the branch must carry exactly the parent's files (no copies, no extras
        // before any branch write).
        assert_eq!(
            branch_files, main_files,
            "freshly-forked branch fragment set must equal the parent's (shallow clone)"
        );
    }

    // --- P3-BR-2: appends on the branch don't mutate main ---
    #[tokio::test]
    async fn branch_append_does_not_mutate_main() {
        let (store, _tmp) = make_test_store(8);
        let domain = "admin/iso";

        seed_main_commit(&store, domain, "c0", &["A", "B"]).await;
        let main_files_before = store
            .io_branch_data_file_paths(domain, "main")
            .await
            .unwrap();

        io_ensure_branch_forked(&store, domain, "feature", "c0")
            .await
            .expect("branch out");

        // Write a new doc on the feature branch.
        let new = row("C", 8, 99.0, "branch-only doc C");
        store
            .io_upsert_chunks(domain, "feature", "C", std::slice::from_ref(&new))
            .await
            .expect("upsert on branch");

        // Main's fragment files must be unchanged (parent untouched).
        let main_files_after = store
            .io_branch_data_file_paths(domain, "main")
            .await
            .unwrap();
        assert_eq!(
            main_files_after, main_files_before,
            "appending on the branch must not change main's fragment files"
        );

        // The branch now has MORE files than main (the delta fragment).
        let branch_files = store
            .io_branch_data_file_paths(domain, "feature")
            .await
            .unwrap();
        assert!(
            branch_files.is_superset(&main_files_after),
            "branch must still reference the shared parent files"
        );
        assert!(
            branch_files.len() > main_files_after.len(),
            "branch must have added a delta fragment for the new doc"
        );
    }

    // --- P3-BR-3: branch-from-anywhere — fork at c0 resolves c0's snapshot ---
    // regardless of later commits on main.
    #[tokio::test]
    async fn branch_from_anywhere_resolves_fork_point_snapshot() {
        let (store, _tmp) = make_test_store(8);
        let domain = "admin/anywhere";

        // c0: docs A,B.
        let v_c0 = seed_main_commit(&store, domain, "c0", &["A", "B"]).await;

        // Advance main to c1 with an extra doc C (after the eventual fork point).
        let r = row("C", 8, 50.0, "doc C added at c1");
        let v_c1 = store
            .io_upsert_chunks(domain, "main", "C", std::slice::from_ref(&r))
            .await
            .unwrap();
        store.io_tag_commit(domain, "main", "c1", v_c1).await.unwrap();
        assert!(v_c1 > v_c0);

        // Fork a branch at c0 (the OLDER commit), even though main is now at c1.
        let outcome = io_ensure_branch_forked(&store, domain, "feature", "c0")
            .await
            .expect("fork at c0");
        assert_eq!(outcome, BranchOutcome::Created { parent_version: v_c0 });

        // The branch's fragment set must equal main's fragments AT c0 — i.e. it
        // must NOT contain c1's extra fragment. We compare against the recorded
        // c0 file set by re-deriving it: checkout the branch and ensure doc C is
        // absent at the branch head (it was added at c1, after the fork point).
        let branch_files = store
            .io_branch_data_file_paths(domain, "feature")
            .await
            .unwrap();
        // Branch was forked from v_c0; verify by searching the branch head has
        // exactly the c0 docs. We use the store's lookup to confirm C is absent.
        let c_chunks = store
            .io_lookup_doc_chunks(domain, "feature", "C")
            .await
            .unwrap();
        assert!(
            c_chunks.is_empty(),
            "branch forked at c0 must NOT see doc C (added at c1) — branch-from-anywhere"
        );
        assert!(!branch_files.is_empty());
    }
}
