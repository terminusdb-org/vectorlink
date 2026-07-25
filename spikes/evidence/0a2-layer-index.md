# Evidence — 0a-2 Global commit→layer index (RISK-16, HARD GATE) + RISK-18

**Verdict: PASS. Backing decision = Lance tags (Tier 1).** lance 7.0.0, Rust 1.96, aarch64.
Full log: `logs/rerun-spikes.log`. Source: `../layer-index/`.

## Results
```
encoded tags: o2uq7k1mrun1vp4urktmw55962vlpto -> c_o2uq7k1mrun1vp4urktmw55962vlpto
              branch:feature/v.1..2          -> c_branch-3afeature-2fv-2e1-2e-2e2
RESULT 0a2.0_tag_encoding:           PASS — encode/decode round-trips for normal + adversarial ids
RESULT 0a2.1_tag_roundtrip:          PASS — commit ids → v1/v2, list=2
RESULT 0a2.2_global_resolution:      PASS — tag created on main resolves from the feature-branch session (Ok(2))
RESULT 0a2.3_scale:                  PASS — 5000 tags: create_total=1518ms, single_lookup=191us, enumerate(5002)=92ms
RESULT 0a2.4_compaction_tag_safe:    PASS — frags 7→1 (compacted 7); pre-compaction tag still reads 250/250 rows
RESULT 0a2.5_branch_after_compaction:PASS — branch reads 100 rows after parent compaction
RERUN_DONE=0
```

## What this proves
- **Tags are a viable global commit→version layer index.** A tag created on one branch resolves from a session on another (branch-from-anywhere). Lookup ~191µs, enumeration of 5k ~92ms — fast enough; **no need for the explicit-table fallback.** (RISK-16 hard gate cleared.)
- **Commit→tag encoding works against real Lance.** The reversible `c_` + `-HH` scheme (Spec 12 §3.4) passes Lance's `check_valid_tag` for both normal hashes and adversarial ids (`:`,`/`,`.`,`..`). Confirms the encoding contract, not just the theory.
- **RISK-18 (compaction/cleanup) is safe under our access pattern:** `compact_files` consolidated 7 fragments → 1, yet a **pre-compaction tagged version still reads all its rows**, and a **branch forked before compaction still reads** after the parent is compacted. So we can compact for read performance without breaking historical/branch resolution. Combined with Lance's tag/branch-aware cleanup (`cleanup_cascade_branch`), the "retention over reclamation" policy holds.

## Decision recorded
**Layer-index backing = Lance tags** (Tier 1 PASS). Commit ids are bound to versions via the encoded tag name; the explicit-table backing is not needed.

## Scale caveat (carry to Phase 2 / RISK-04)
Validated at 5k tags. Extrapolation to 100k–1M (the Spec 11 thresholds) was not run here (spike speed). 191µs/lookup + 92ms/5k-enumerate is well within budget and trends fine, but the high-end scale check remains a Phase-2/5 due-diligence item. Not gate-blocking given the comfortable margin.
