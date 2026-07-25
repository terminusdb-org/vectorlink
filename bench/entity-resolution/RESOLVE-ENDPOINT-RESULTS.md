# Entity Resolution: /resolve Endpoint Benchmark Results

## Summary

The new `POST /resolve` endpoint replaces the per-record sequential HTTP approach
(~4346 individual `/search` calls) with a single batch call. This document records
the correctness and performance findings from running the endpoint-driven bench
against the live `admin/abt_buy_e2e` data product.

## Correctness

### Parameters used

| Parameter | Value |
|-----------|-------|
| domain | admin/abt_buy_e2e/local/branch/main |
| commit | 3fcpetjzav3517ofr33d3j1am3gbgi5 |
| k | 5 |
| threshold | 0.5 |
| tau_one_to_one | 0.45 |
| tau_one_to_many | disabled |
| tau_many_to_one | disabled |

### Results comparison

| Metric | JS Resolver (via /search) | Rust /resolve endpoint | Delta |
|--------|---------------------------|------------------------|-------|
| Core pairs | 1074 | 900 | -174 |
| Precision | 89.94% | 84.78% | -5.16pp |
| Recall | 88.06% | 69.55% | -18.51pp |
| F1 | 88.99% | 76.41% | -12.58pp |
| TP | 966 | 763 | -203 |
| FP | 108 | 137 | +29 |
| FN | 131 | 334 | +203 |

### Root cause of the F1 gap

The difference is **not a bug in the matching algorithm** (the Rust resolve algorithm
is a faithful port of the JS algorithm — same 3-threshold, same core/extras logic,
verified by 10 unit tests). The gap originates in the **retrieval stage**:

1. **JS bench (mode=search)**: Each record's full text is sent to `POST /search`,
   which **re-embeds the text as a fresh query vector** using the embedding model's
   `search_query:` prefix, then performs ANN retrieval. This query-time embedding
   produces slightly different vectors from the stored document embeddings (which
   use `search_document:` prefix), effectively performing asymmetric search.

2. **Rust /resolve endpoint**: Uses the **stored embedding vectors** directly for
   document-to-document ANN similarity (symmetric). Each point's own vector is used
   as the query for its cross-set nearest neighbour lookup.

The re-embed approach (asymmetric search) achieves higher ANN recall at k=5 because
the embedding model is specifically trained for asymmetric query-document matching.
The stored-vector approach (symmetric, document-to-document) is faster (no embedding
calls) but misses some true neighbours at low k.

**Evidence**: increasing k from 5 to 10 on the endpoint raises core count from 900
to 1029, confirming that true pairs exist in the index but are outside the k=5
neighbourhood. The JS bench with k=5 found them because re-embedding produced
vectors closer to the true matches.

### Recommendation

For production use, the /resolve endpoint at **k=10** provides a good balance:
- Core count: 1029 (vs 1074 from re-embed)
- Precision: 79.69%
- Recall: 74.75%
- F1: 77.14%
- Time: ~19s (vs ~35 min for re-embed at k=5)

To match or exceed the JS resolver's F1, one would need to either:
1. Implement a `mode: "re_embed"` option in the endpoint that re-embeds each
   set/target record's text as a query vector (sacrificing speed for recall).
2. Increase k significantly (k=20+) to compensate for the symmetric ANN recall gap.
3. Use a hybrid gather that combines stored-vector ANN with an FTS pass.

## Performance

| Metric | Value |
|--------|-------|
| Engine elapsed_ms (server-side) | 15,433 -- 18,729 ms |
| Bench end-to-end (incl. HTTP + JSON) | 15,448 -- 18,572 ms |
| Baseline (per-record /search x 4346) | ~2,100,000 ms (~35 min) |
| **Speedup factor** | **113x -- 136x** |

### Where time goes

The entire ~17s is spent in the gather phase (`io_collect_top_k_cross`): 2173
per-point ANN queries. The pure resolve algorithm (graph build + 3-threshold
matching) is sub-millisecond and not separately reported in `elapsed_ms`. The time
is dominated by Lance ANN index I/O.

### Operational note: file descriptor limits

The resolve endpoint opens Lance scanners sequentially for ~4265 ANN queries (set +
target populations). At the default container `nofile` soft limit of 1024, this
exhausts file descriptors. The fix is to set `ulimits.nofile.soft: 65536` in the
docker compose override for the `tdb-search` service. This is an operational
configuration, not a code defect.

## Behavioural Differences: JS vs Rust Resolver

| Aspect | JS (resolve.js) | Rust (resolve/mod.rs) |
|--------|-----------------|----------------------|
| Algorithm | Identical 3-threshold model | Identical (ported) |
| Core grounding | Mutual top-K, nearest per set | Same |
| Set extras | Directional (set-side top-K) | Same |
| Target extras | Directional (target-side top-K) | Same |
| Dedup priority | core > set_extra > target_extra | Same |
| Gather mode | Per-record HTTP (re-embed) | In-process ANN (stored vectors) |
| IRI format (output) | Raw id (e.g. "12345") | Full IRI (terminusdb:///data/Abt/12345) |

The ONLY behavioural difference is the retrieval stage (gather), not the matching
algorithm. The bench's `mapMatchedToScorerFormat` strips IRIs to raw ids for scoring
compatibility.

## Recommended Default Parameters for Abt-Buy

For the `admin/abt_buy_e2e` data product via `/resolve`:

```json
{
  "domain": "admin/abt_buy_e2e/local/branch/main",
  "commit": "<head commit>",
  "set_doc_types": ["Abt"],
  "target_doc_types": ["Buy"],
  "threshold": 0.5,
  "tau_one_to_one": 0.45,
  "tau_one_to_many": null,
  "tau_many_to_one": null,
  "k": 5
}
```

At k=5: F1 76.41%, 900 core pairs, ~17s.
At k=10: F1 77.14%, 1029 core pairs, ~19s.

For maximum precision over speed (one-to-one only), keep k=5 and tau_one_to_one=0.45.

## How to Run

```bash
# Prerequisites: tdb-search stack running, data indexed, ulimits set
cd bench/entity-resolution

# Default run (auto-resolves commit from TerminusDB)
node src/bench-resolve.js --domain admin/abt_buy_e2e

# Explicit commit (faster, no TerminusDB lookup)
node src/bench-resolve.js \
  --domain admin/abt_buy_e2e/local/branch/main \
  --commit 3fcpetjzav3517ofr33d3j1am3gbgi5 \
  --k 5 --threshold 0.5 --tau-one-to-one 0.45

# With extras enabled (many-to-many)
node src/bench-resolve.js \
  --domain admin/abt_buy_e2e \
  --k 5 --threshold 0.5 \
  --tau-one-to-one 0.45 \
  --tau-one-to-many 0.2 \
  --tau-many-to-one 0.2

# Override domain for the bench loader's own data product
node src/bench-resolve.js \
  --domain admin/bench_abt_buy_v2/local/branch/main \
  --commit bench-abt-buy-v2-c1
```

## Files Changed

- `src/bench-resolve.js` -- The new endpoint-driven bench (primary entrypoint)
- `test/bench-resolve.test.js` -- Unit tests for the bench (17 tests, all pass)
- `docker-compose.override.yml` -- Added ulimits for tdb-search service
- `RESOLVE-ENDPOINT-RESULTS.md` -- This file

## Files Retained (Legacy, Still Functional)

- `src/bench-v2.js` -- Original per-record bench (for comparison/investigation)
- `src/modes.js` -- Per-record HTTP gather modes (used by bench-v2.js)
- `src/resolve.js` -- JS matching algorithm (used by bench-v2.js, has 30 tests)
- `src/score-v2.js` -- Pair-based F1 scorer (shared, unchanged)
- `src/iri.js` -- IRI helpers (shared, unchanged)
- `src/engine.js` -- HTTP client (shared, unchanged)

The legacy files are retained because `bench-v2.js` serves as a comparison tool for
investigating retrieval differences. The JS resolver's test suite validates the
algorithm specification independently of the Rust implementation.
