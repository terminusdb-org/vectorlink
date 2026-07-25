# Evidence — 0a-1 Lance branching & block reuse (RISK-01/02)

**Verdict: PASS.** lance 7.0.0, lancedb 0.30.0, Rust 1.96, aarch64, in `rust:1-bookworm`.
Full logs: `logs/0a1-build-run.log`, `logs/0a1-run.log`. Source: `../lance-branch/`.

## Results (from `logs/0a1-run.log`)
```
after create: version=1 rows=1000 data_files=1
RESULT 0a1.1_append_version: PASS — rows=1000, files=1
RESULT 0a1.2_create_branch: PASS — branches=["feature"]
parent_files=1 branch_files=1 shared=1
RESULT 0a1.3_block_reuse: PASS — branch references identical parent data files (shared=1/1) — shallow clone, no copy
RESULT 0a1.4_branch_isolation: PASS — branch_rows=1100 main_rows=1000 new_branch_files=1 parent_unchanged=true
RESULT 0a1.5_checkout: PASS — checkout(feature)=1100 checkout(main)=1000
RESULT 0a1.6_fork_from_past: PASS — fork_rows=1000 shares_original_fragments=true
RESULT 0a1.7_coexistence: INFO — lancedb::connect succeeded on lance-written dir
RUN_EXIT=0
```

## What this proves
- **`lance` 7.0 `create_branch(branch, version, None)` is a shallow clone** — the branch's data-file set is *identical* to the parent's (path identity, not row-count). No data copied. (RISK-01, the 5/5 risk, resolved.)
- **Branch appends isolate:** parent rows + parent fragment files unchanged; only new fragment(s) written for the delta. Block reuse + isolation both hold — the core history property.
- **Branch-from-anywhere:** a branch forked from an *older* version sees exactly that version's rows and shares its fragments. Confirms the layer index can resolve a commit's layer for forks from any point.
- **lancedb/lance coexistence:** `lancedb::connect` opens the directory `lance` core wrote — so we can write/branch via `lance` core and query/FTS via `lancedb` on one dataset (RISK-02 pattern confirmed at the connect level; FTS/hybrid query exercised in Phase 2).

## Caveat / robustness
- 1000 rows produced a single data file, so block-reuse shows as `1/1` identical. The property (branch points at the parent's physical file, not a copy) is genuine. A multi-fragment re-run (more rows / smaller `max_rows_per_file`) makes the "N/N shared" assertion more emphatic — noted for the Phase-2 store tests; not required to clear the gate.

## API confirmed (carries into implementation)
- `Dataset::create_branch(&mut self, branch: &str, version: impl Into<Ref>, store_params: Option<ObjectStoreParams>) -> Result<Self>` — doc: "two-phase: create the branch dataset by **shallow cloning**."
- `checkout_branch(&self, &str)`, `list_branches() -> HashMap<String, BranchContents>`, `get_fragments()[].metadata().files[].path` for physical-file inspection.
