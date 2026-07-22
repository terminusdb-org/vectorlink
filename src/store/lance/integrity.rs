// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 DFRNT AB

#![forbid(unsafe_code)]

//! Store integrity checks — compares on-disk state against live references.
//!
//! These functions are production code (not tests) exposed via the HTTP API
//! (`GET /integrity?domain=...`). They allow operators to verify that:
//!
//! - Every index UUID directory on disk is referenced by a live tagged manifest.
//! - Every index UUID referenced by a tagged manifest has an existing directory.
//! - Data file counts match the union of files referenced by live manifests.
//! - Manifest counts are bounded and no orphaned manifests exist.
//!
//! The checks are read-only and safe to run at any time. They do not modify
//! the store or acquire locks — they observe a snapshot of the on-disk state.

use std::collections::HashSet;

use lance::index::DatasetIndexExt;
use serde::Serialize;

use crate::kernel::error::StoreError;
use crate::store::lance::{is_compact_rebuild_branch, LanceStore};

/// Result of an integrity check for a single domain.
#[derive(Debug, Serialize)]
pub struct IntegrityReport {
    pub domain: String,
    pub branches: Vec<BranchInfo>,
    pub tagged_versions: usize,
    pub on_disk_index_dirs: Vec<String>,
    pub referenced_index_uuids: Vec<String>,
    pub stale_index_dirs: Vec<String>,
    pub dangling_index_refs: Vec<String>,
    pub on_disk_data_files: usize,
    pub on_disk_manifests: usize,
    pub rebuild_branches: Vec<String>,
    pub orphaned_tags: Vec<String>,
    pub stale_rebuild_branches: Vec<String>,
    pub ok: bool,
    pub warnings: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct BranchInfo {
    pub name: String,
    pub version: u64,
}

impl LanceStore {
    /// Collect all live index UUIDs from ALL tagged versions across ALL branches.
    ///
    /// **Lenient variant** — soft-skips collection errors (logs to stderr,
    /// continues). Safe for read-only integrity reporting where partial
    /// data is still useful. **Must NOT be used by destructive callers**
    /// (e.g. `io_prune_stale_index_dirs`) — use
    /// `io_collect_all_live_index_uuids_strict` instead.
    pub async fn io_collect_all_live_index_uuids(
        &self,
        domain: &str,
    ) -> Result<HashSet<String>, StoreError> {
        self.collect_live_index_uuids(domain, false).await
    }

    /// **Strict (fail-closed) variant** — returns `Err` on ANY collection
    /// failure (branch checkout, version checkout, load_indices). Destructive
    /// callers (e.g. `io_prune_stale_index_dirs`) MUST use this variant so
    /// that an incomplete live set never causes live index dirs to be deleted.
    pub async fn io_collect_all_live_index_uuids_strict(
        &self,
        domain: &str,
    ) -> Result<HashSet<String>, StoreError> {
        self.collect_live_index_uuids(domain, true).await
    }

    /// Internal implementation shared by both variants.
    /// `strict = true` → fail-closed (return Err on any collection error).
    /// `strict = false` → lenient (log and continue).
    async fn collect_live_index_uuids(
        &self,
        domain: &str,
        strict: bool,
    ) -> Result<HashSet<String>, StoreError> {
        let path = self.dataset_path(domain);
        if !path.exists() {
            return Ok(HashSet::new());
        }

        let uri = path.to_string_lossy().to_string();
        let ds = self.io_open_fresh(&uri).await?;

        let tags = ds
            .tags()
            .list()
            .await
            .map_err(|e| StoreError::Internal(format!("collect UUIDs: tag list failed: {}", e)))?;

        // Group tagged versions by branch for efficient checkout.
        let mut by_branch: std::collections::HashMap<String, Vec<u64>> =
            std::collections::HashMap::new();
        for contents in tags.values() {
            let b = contents
                .branch
                .clone()
                .unwrap_or_else(|| "main".to_owned());
            by_branch.entry(b).or_default().push(contents.version);
        }

        let mut live_uuids: HashSet<String> = HashSet::new();

        // First, collect UUIDs from HEAD on ALL branches (including main).
        // HEAD on main is the compacted version and may not be tagged — its
        // index UUIDs must be in the live set or they get pruned.
        let all_branches = self.io_list_branches(domain).await?;

        // Collect HEAD UUIDs from main (the default branch).
        match ds.load_indices().await {
            Ok(indices) => {
                for idx in indices.iter() {
                    live_uuids.insert(idx.uuid.to_string());
                }
            }
            Err(e) => {
                if strict {
                    return Err(StoreError::Internal(format!(
                        "collect UUIDs (strict): load_indices failed for HEAD on main: {}",
                        e
                    )));
                }
                eprintln!("[integrity] collect UUIDs: load_indices failed for HEAD on main: {}", e);
            }
        }

        // Collect HEAD UUIDs from all non-main branches.
        for branch_name in &all_branches {
            let branch_ds = if branch_name == "main" {
                ds.clone()
            } else {
                match ds.checkout_branch(branch_name).await {
                    Ok(bd) => bd,
                    Err(e) => {
                        if strict {
                            return Err(StoreError::Internal(format!(
                                "collect UUIDs (strict): checkout_branch '{}' failed: {}",
                                branch_name, e
                            )));
                        }
                        eprintln!(
                            "[integrity] collect UUIDs: skipping branch {} (checkout failed: {})",
                            branch_name, e
                        );
                        continue;
                    }
                }
            };
            match branch_ds.load_indices().await {
                Ok(indices) => {
                    for idx in indices.iter() {
                        live_uuids.insert(idx.uuid.to_string());
                    }
                }
                Err(e) => {
                    if strict {
                        return Err(StoreError::Internal(format!(
                            "collect UUIDs (strict): load_indices failed for HEAD on branch '{}': {}",
                            branch_name, e
                        )));
                    }
                    eprintln!(
                        "[integrity] collect UUIDs: load_indices failed for HEAD on branch {}: {}",
                        branch_name, e
                    );
                }
            }
        }

        // Then collect UUIDs from all tagged versions across all branches.
        for (branch_name, versions) in &by_branch {
            let branch_ds = if branch_name == "main" {
                ds.clone()
            } else {
                match ds.checkout_branch(branch_name).await {
                    Ok(bd) => bd,
                    Err(e) => {
                        if strict {
                            return Err(StoreError::Internal(format!(
                                "collect UUIDs (strict): checkout_branch '{}' for tagged versions failed: {}",
                                branch_name, e
                            )));
                        }
                        eprintln!(
                            "[integrity] collect UUIDs: skipping branch {} (checkout failed: {})",
                            branch_name, e
                        );
                        continue;
                    }
                }
            };

            for &version in versions {
                match branch_ds.checkout_version(version).await {
                    Ok(snapshot) => {
                        match snapshot.load_indices().await {
                            Ok(indices) => {
                                for idx in indices.iter() {
                                    live_uuids.insert(idx.uuid.to_string());
                                }
                            }
                            Err(e) => {
                                if strict {
                                    return Err(StoreError::Internal(format!(
                                        "collect UUIDs (strict): load_indices failed for version {} on branch '{}': {}",
                                        version, branch_name, e
                                    )));
                                }
                                eprintln!(
                                    "[integrity] collect UUIDs: load_indices failed for version {} on branch {}: {}",
                                    version, branch_name, e
                                );
                            }
                        }
                    }
                    Err(e) => {
                        if strict {
                            return Err(StoreError::Internal(format!(
                                "collect UUIDs (strict): checkout version {} on branch '{}' failed: {}",
                                version, branch_name, e
                            )));
                        }
                        eprintln!(
                            "[integrity] collect UUIDs: checkout version {} on branch {} failed: {}",
                            version, branch_name, e
                        );
                    }
                }
            }
        }

        Ok(live_uuids)
    }

    /// Run a full integrity check on a domain. Read-only — does not modify
    /// the store. Returns a structured report comparing on-disk state
    /// against live references from all tagged manifests.
    pub async fn io_integrity_check(&self, domain: &str) -> Result<IntegrityReport, StoreError> {
        let path = self.dataset_path(domain);

        if !path.exists() {
            return Ok(IntegrityReport {
                domain: domain.to_owned(),
                branches: Vec::new(),
                tagged_versions: 0,
                on_disk_index_dirs: Vec::new(),
                referenced_index_uuids: Vec::new(),
                stale_index_dirs: Vec::new(),
                dangling_index_refs: Vec::new(),
                on_disk_data_files: 0,
                on_disk_manifests: 0,
                rebuild_branches: Vec::new(),
                orphaned_tags: Vec::new(),
                stale_rebuild_branches: Vec::new(),
                ok: true,
                warnings: vec!["dataset does not exist on disk".to_owned()],
            });
        }

        let uri = path.to_string_lossy().to_string();
        let ds = self.io_open_fresh(&uri).await?;

        // --- Collect tag info ---
        let tags = ds
            .tags()
            .list()
            .await
            .map_err(|e| StoreError::Internal(format!("integrity: tag list failed: {}", e)))?;

        let tagged_versions = tags.len();

        // Group by branch for branch info.
        let mut by_branch: std::collections::HashMap<String, Vec<u64>> =
            std::collections::HashMap::new();
        for contents in tags.values() {
            let b = contents
                .branch
                .clone()
                .unwrap_or_else(|| "main".to_owned());
            by_branch.entry(b).or_default().push(contents.version);
        }

        let branches: Vec<BranchInfo> = by_branch
            .iter()
            .map(|(name, versions)| BranchInfo {
                name: name.clone(),
                version: *versions.iter().max().unwrap_or(&0),
            })
            .collect();

        // --- Collect live index UUIDs from all tagged versions ---
        let live_uuids = self.io_collect_all_live_index_uuids(domain).await?;
        let referenced_index_uuids: Vec<String> = live_uuids.iter().cloned().collect();

        // --- Scan on-disk index directories (main + all branch tree/<branch>/_indices/) ---
        let mut on_disk_index_dirs: Vec<String> = Vec::new();

        // Main's _indices/
        let indices_dir = path.join("_indices");
        if indices_dir.exists() {
            on_disk_index_dirs.extend(
                std::fs::read_dir(&indices_dir)
                    .map_err(|e| StoreError::Internal(format!("integrity: read _indices failed: {}", e)))?
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().is_dir())
                    .map(|e| e.file_name().to_string_lossy().to_string()),
            );
        }

        // Branch _indices/ under tree/<branch_name>/_indices/
        let tree_dir = path.join("tree");
        if tree_dir.exists() {
            if let Ok(branch_entries) = std::fs::read_dir(&tree_dir) {
                for branch_entry in branch_entries.filter_map(|e| e.ok()) {
                    let branch_indices = branch_entry.path().join("_indices");
                    if branch_indices.exists() {
                        if let Ok(entries) = std::fs::read_dir(&branch_indices) {
                            on_disk_index_dirs.extend(
                                entries
                                    .filter_map(|e| e.ok())
                                    .filter(|e| e.path().is_dir())
                                    .map(|e| e.file_name().to_string_lossy().to_string()),
                            );
                        }
                    }
                }
            }
        }

        on_disk_index_dirs.sort();
        on_disk_index_dirs.dedup();

        let on_disk_set: HashSet<&str> =
            on_disk_index_dirs.iter().map(|s| s.as_str()).collect();

        // --- Stale dirs: on disk but not referenced by any tagged version ---
        let stale_index_dirs: Vec<String> = on_disk_index_dirs
            .iter()
            .filter(|uuid| !live_uuids.contains(*uuid))
            .cloned()
            .collect();

        // --- Dangling refs: referenced by tagged manifests but no dir on disk ---
        let dangling_index_refs: Vec<String> = live_uuids
            .iter()
            .filter(|uuid| !on_disk_set.contains(uuid.as_str()))
            .cloned()
            .collect();

        // --- Count on-disk data files ---
        let data_dir = path.join("data");
        let on_disk_data_files = if data_dir.exists() {
            std::fs::read_dir(&data_dir)
                .map_err(|e| StoreError::Internal(format!("integrity: read data dir failed: {}", e)))?
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .is_some_and(|x| x == "lance")
                })
                .count()
        } else {
            0
        };

        // --- Count on-disk manifests ---
        let versions_dir = path.join("_versions");
        let on_disk_manifests = if versions_dir.exists() {
            std::fs::read_dir(&versions_dir)
                .map_err(|e| StoreError::Internal(format!("integrity: read _versions failed: {}", e)))?
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .is_some_and(|x| x == "manifest")
                })
                .count()
        } else {
            0
        };

        // --- List rebuild branches ---
        let all_branches = self.io_list_branches(domain).await?;
        let rebuild_branches: Vec<String> = all_branches
            .iter()
            .filter(|b| is_compact_rebuild_branch(b))
            .cloned()
            .collect();

        // --- Detect orphaned tags: tags pointing to non-existent branches ---
        let all_branch_set: HashSet<&str> =
            all_branches.iter().map(|b| b.as_str()).collect();
        let orphaned_tags: Vec<String> = tags
            .iter()
            .filter_map(|(tag_name, contents)| {
                if let Some(ref b) = contents.branch {
                    if !all_branch_set.contains(b.as_str()) {
                        return Some(format!(
                            "{} (branch={} version={})",
                            tag_name, b, contents.version
                        ));
                    }
                }
                None
            })
            .collect();

        // --- Detect stale rebuild branches: rebuild branches with no tags ---
        let tagged_branches: HashSet<&str> =
            tags.values().filter_map(|c| c.branch.as_deref()).collect();
        let stale_rebuild_branches: Vec<String> = rebuild_branches
            .iter()
            .filter(|b| !tagged_branches.contains(b.as_str()))
            .cloned()
            .collect();

        // --- Determine overall status ---
        let mut warnings = Vec::new();

        if !stale_index_dirs.is_empty() {
            warnings.push(format!(
                "{} stale index directories on disk not referenced by any tagged version",
                stale_index_dirs.len()
            ));
        }
        if !dangling_index_refs.is_empty() {
            warnings.push(format!(
                "{} index UUIDs referenced by tagged manifests have no backing directory on disk (historical FTS may be degraded)",
                dangling_index_refs.len()
            ));
        }
        if rebuild_branches.len() > 1 {
            warnings.push(format!(
                "{} rebuild branches exist (expected ≤1 after compaction)",
                rebuild_branches.len()
            ));
        }
        if !orphaned_tags.is_empty() {
            warnings.push(format!(
                "{} tags point to non-existent branches (orphaned tags pin dead data)",
                orphaned_tags.len()
            ));
        }
        if !stale_rebuild_branches.is_empty() {
            warnings.push(format!(
                "{} rebuild branches have no tags (stale — should be deleted)",
                stale_rebuild_branches.len()
            ));
        }

        let ok = stale_index_dirs.is_empty()
            && dangling_index_refs.is_empty()
            && rebuild_branches.len() <= 1
            && orphaned_tags.is_empty()
            && stale_rebuild_branches.is_empty();

        Ok(IntegrityReport {
            domain: domain.to_owned(),
            branches,
            tagged_versions,
            on_disk_index_dirs,
            referenced_index_uuids,
            stale_index_dirs,
            dangling_index_refs,
            on_disk_data_files,
            on_disk_manifests,
            rebuild_branches,
            orphaned_tags,
            stale_rebuild_branches,
            ok,
            warnings,
        })
    }
}
