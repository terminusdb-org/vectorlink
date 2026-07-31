# 06 — Integration test strategy

### Learning objectives
After this document you will be able to:
- **Stand up** the full system under test from `docker-compose.yml`.
- **Choose** the right test level for a given behaviour.
- **Write** deterministic, repeatable assertions against a live stack.
- **Guarantee** reproducibility by controlling the embedding path.

### Prerequisites
All prior docs, especially [05 — Embeddings & determinism](./05-embeddings-and-determinism.md).

---

## 1. Goal and principle

> **An integration test runs against the real components wired together by `docker-compose.yml`, exercises the public HTTP contract, and asserts exact, reproducible results.**

Principle: **test the contract through the front door** (the HTTP API), against **real** dependencies (real LanceDB store, real embedding model, real TerminusDB), with **deterministic** inputs and outputs. No mocking of the things under test. Determinism comes from the stable local model run in eval mode with controlled batching (doc 05).

This complements — does not replace — the unit and store-level tests in `specs/06-test-strategy.md` (layers 1–2). This document is layers 3–4.

---

## 2. The system under test (docker-compose.yml)

```
┌─────────────────────────────────────────────────────────────┐
│ docker-compose.yml  (CPU-only, no external network)         │
│                                                             │
│  ┌────────────┐   push (NDJSON)   ┌──────────────────────┐  │
│  │ terminusdb │──────────────────▶│      tdb-search      │  │
│  │  :6363     │   operation ops   │        :7372         │  │
│  └────────────┘                   │  LanceDB store (vol) │  │
│                                   └─────────┬────────────┘  │
│                                             │ /v1/embeddings│
│                                   ┌─────────▼────────────┐  │
│                                   │ embeddings (Ollama)  │  │
│                                   │  nomic-embed-text-   │  │
│                                   │  v2-moe (GGUF, Q8_0) │  │
│                                   │  @ pinned digests,CPU│  │
│                                   └──────────────────────┘  │
│                                                             │
│  test-runner (optional 4th service or host process)         │
└─────────────────────────────────────────────────────────────┘
```

| Service | Role in tests | Determinism controls |
|---------|---------------|----------------------|
| `tdb-search` | System under test | Pinned image/build; fixed config; LanceDB volume reset per run |
| `embeddings` (Ollama) | Real embeddings | **Pinned image + GGUF digests, Q8_0**, CPU, one-input-per-request for tests |
| `terminusdb` | Real push source | Seeded from a fixed dataset bundle (Star Wars) at a known commit |
| `test-runner` | Drives HTTP, asserts | Talks only to public endpoints |

A **fixture-only profile** may swap `terminusdb` for a tiny "fake push driver" (doc 02 §5) that POSTs a frozen operation stream to `/push` — faster, and used for the bulk of contract tests. The full-fidelity profile uses the real TerminusDB for the end-to-end smoke tests.

---

## 3. Determinism contract for the suite

Every assertion that compares vectors, distances, or rankings relies on these being pinned (doc 05 §3–§4):

1. **Embeddings image + GGUF pinned** to digests in the Ollama service definition — never a moving tag.
2. **Q8_0 quantization, CPU** — one numerical path (goldens are baseline-specific to this quantization).
3. **Inference mode** — Ollama/llama.cpp serves inference-mode models (no dropout); embeddings have no sampling step.
4. **One-input-per-request in tests** — no padding partners, so each input's vector is independent of its neighbours.
5. **Prefixes injected deterministically** — `search_document:` for content, `search_query:` for queries.
6. **Store reset per run** — fresh LanceDB volume so versions/commits start clean.

If all six hold, the suite is intended to be **bit-for-bit reproducible**: identical vectors → identical distances → identical rankings. ⚠️ **Cross-restart bit-identity is a Phase-0 carried caveat** (only same-process repeatability is proven, `spikes/evidence/embeddings-stack.md`); the determinism gate (Level E) is the place it is confirmed before golden-vector assertions are enforced strictly.

**All three search modes are in the deterministic battery.** `hybrid` is the default mode, so determinism cannot be a vector-only guarantee. `vector`, `fts`, and `hybrid` each get golden/ranking assertions. This holds because: vector is deterministic (above); FTS over fixed `content` with a **pinned FTS tokenizer/analyzer** is deterministic; and RRF fusion is a deterministic function of the two input rankings. The FTS index configuration (tokenizer, stop-words, stemming) is therefore **also pinned** as a determinism input, alongside the embedding model.

### Two assertion styles
- **Golden-vector (strict):** embed the fixture corpus once, commit the resulting vectors as `tests/golden/*.json`, and assert the live service reproduces them exactly. Catches any environment/model drift as a diff. Used for a small, representative fixture set.
- **Ranking/relative (robust):** assert *ordering* and *thresholds* ("Yoda is the top hit for 'Wise old man'", "distance(self) ≈ 0", "distance(a,b) < distance(a,c)"). Survives benign numeric noise; primary style for most end-to-end tests.

Prefer golden-vector for the determinism guarantee tests; ranking assertions for behavioural tests.

---

## 4. Test levels (what runs against the stack)

### Level A — HTTP contract (fake push driver profile)
Verifies every endpoint in `specs/02-interface-contract.md`:
- `/last-indexed?domain=&branch=` returns the indexer's last-indexed commit (or empty before first index).
- `/push?domain=&branch=&target_commit=&parent_commit=` ingests an NDJSON stream and, on completion, advances the last-indexed commit; a malformed line fails the push loudly.
- `/search`, `/similar`, `/duplicates`, `/statistics`: exact params, headers, status codes, JSON shapes.
- Missing admin secret → 401; missing domain/commit → 400.
- Push contract: assert TerminusDB→indexer push carries `target_commit` and `parent_commit`, the NDJSON body is parsed **incrementally** (not buffered whole), and the HTTP Basic admin secret is present (capture at the indexer; a fake push driver feeds it).

### Level B — Indexing & history (real embeddings, fake or real content)
- **Index-then-search:** index commit `C0`, search a known query, assert the expected top hit (ranking style).
- **Incremental reuse:** index `C1` from `C0` after changing one doc; assert (a) only the changed doc was re-embedded (observe embedding-service call count or row provenance), (b) `/search?commit=C0` still returns the old result, `commit=C1` the new — proving snapshot isolation.
- **`/assign`:** assign `C0`→`C2`; assert search at `C2` == search at `C0` with no new embedding calls.
- **Branch-out (the core):** branch from `C0`; append on the branch; assert the branch sees C0's docs (block reuse) and appends don't affect `main`. Assert fragment sharing via row/version inspection.

### Level C — Search modes (all three, all deterministic)
The default mode is **hybrid**, so every mode is a first-class, determinism-gated path — not an optional extra:
- **`mode=hybrid` (default):** a query where neither pure vector nor pure FTS alone ranks the target first, but hybrid does; also assert that omitting `mode` == `mode=hybrid`.
- **`mode=vector`:** matches Level B (reference parity).
- **`mode=fts`:** keyword query returns the doc containing an exact rare term; `doc_type` filter works.
- **All three run in the deterministic battery (Level E):** golden + ranking assertions for vector, fts, and hybrid. Requires the **FTS tokenizer/analyzer config to be pinned** (a determinism input alongside the embedding model).

### Level C′ — Large documents & chunking (real embeddings)
Guards against silent truncation — nothing is dropped:
- Index a document whose rendered string exceeds the model window (>512 tokens); assert it produces multiple chunk rows (`chunk_count > 1`) covering the whole text.
- **Tail-recall test:** a query matching a passage near the *end* of a large document returns that document — proving the tail was embedded, not truncated.
- `Changed` on a large doc replaces its full chunk set (no stale chunks remain); `Deleted` removes all its chunks.
- Chunk→document dedup: `/search` returns each matching document once, with its best-chunk distance; chunk ids never appear in responses.

### Level C″ — Lag, outage & catch-up (eventual consistency)
Background indexing means search lags the write head; these scenarios are first-class, not edge cases:
- **No indexed lineage:** search a branch with no indexed ancestor → **404**, verdict **negatively cached** (second search does not re-walk history).
- **Negative-cache invalidation:** after a 404, explicitly index a commit in the branch's ancestry → next search is no longer 404.
- **Auto-enroll propagation:** index P; branch `B` from P; search `B` (never explicitly enabled) → resolves P's layer immediately and `B` enrolls for catch-up.
- **Lag:** commit `C`, search `C` *before* indexing completes → response serves the nearest indexed ancestor **with an explicit stale/`indexed_commit` marker** (never silently current), and catch-up converges so a later search at `C` is exact.
- **Outage:** stop the indexer, make N commits, restart → on the next `/last-indexed` handshake TerminusDB pushes the delta from the durable last-indexed commit per `(domain,branch)` to head; search at head eventually correct.
- **Search-during-catch-up:** returns best-available (flagged) and converges to exact.
> These require a controllable clock/trigger on indexing in the test harness (pause/resume the push driver and the indexer).

### Level D — End-to-end (real TerminusDB profile)
The fidelity smoke test, reproducing the VectorLink blog:
- Seed TerminusDB with the Star Wars bundle; configure the `People`/`Species` embedding templates; create commit `C0`.
- Let TerminusDB push `C0` to the indexer; wait until the indexer's `/last-indexed` reports `C0`.
- Assert: `"Wise old man"` → top hit `People/20` (Yoda); `"Who are the squid people"` → top hit `Species/8` (Mon Calamari).
- **Prefix-table test:** the default model receives nomic prefixes; an unknown model name receives none; embedding the same text as document vs query yields different vectors.
- **Distance-scale test:** self-distance ≈ `0.0`; reported distance ∈ `[0,1]`; equals `lance_cosine_distance / 2` within tolerance.
- **CPU/no-network assertion:** the stack completes with no GPU and no egress.

### Level E — Determinism gate (all three search modes)
- Run Level D twice on a clean volume; assert identical rankings and (for the golden subset) identical vectors/distances.
- Re-embed a golden fixture via the live service; assert byte-equality with `tests/golden/`.
- **Run the full query catalogue in each mode — `vector`, `fts`, `hybrid` — and assert each mode's results are identical across the two runs** (golden ranking per mode). The default (omitted `mode`) must equal `hybrid`.
- Pin and assert the **FTS analyzer config** (tokenizer/stop-words/stemming) as a determinism input; a change to it is treated like a model-revision bump (reviewed, regenerates goldens).

---

## 5. Fixtures

| Fixture | Purpose | Stability |
|---------|---------|-----------|
| `tests/fixtures/ops/*.jsonl` | Frozen operation streams for the fake content endpoint | Hand-fixed; version-controlled |
| Star Wars bundle | Real TerminusDB seed for Level D | Pinned bundle file |
| `tests/golden/embeddings/*.json` | Golden vectors for the strict determinism gate | Regenerated only on an intentional, reviewed model-revision bump |
| Query catalogue | (query → expected top id) pairs | Version-controlled |

When the model revision is intentionally bumped, golden vectors are regenerated in a dedicated, reviewed commit — never silently.

---

## 6. Lifecycle of a test run

```
1. docker compose up --wait        (services report healthy)
2. health gates:
     GET tdb-search /statistics → 200
     GET embeddings  (Ollama readiness) → 200
     terminusdb info           → 200
3. seed: load Star Wars bundle (Level D) OR mount fixture ops (Levels A–C)
4. run test-runner: drive HTTP, poll /last-indexed until the target commit is indexed, assert
5. (determinism gate) repeat Level D, diff results
6. docker compose down -v          (drop the LanceDB volume → clean next run)
```

Health gates must use **DOM/endpoint readiness polling, not fixed sleeps** (a project rule): poll `/last-indexed`, `/statistics`, and the Ollama embeddings endpoint until ready or timeout.

---

## 7. What success looks like (exit criteria)

- All Level A contract assertions green against the live server.
- Level B proves incremental reuse, snapshot isolation, `/assign`, and branch-out block-sharing.
- Level D reproduces the blog search results on CPU with the local model.
- Level E passes twice with identical output, and golden vectors match.
- The whole suite runs from `docker compose up` with no external network and no GPU.

---

## Check your understanding
1. Why can the suite assert *exact* distances rather than just orderings? *(The local model is pinned — Ollama image + GGUF digests, Q8_0, CPU, one-input-per-request → bit-identical embeddings within a run; cross-restart identity confirmed at the Level E gate.)*
2. What is the single configuration choice that most directly removes the batching nondeterminism in tests? *(One input per embedding request — no padding partners.)*
3. How do you prove incremental indexing reused parent blocks rather than recomputing? *(Show only the changed doc was embedded — call count/provenance — and that the old commit's search result is unchanged.)*
4. Why reset the LanceDB volume between runs? *(So versions/commits/tags start clean and results are reproducible.)*
5. Which test guards the nomic prefix requirement? *(Embedding the same text as document vs query and asserting the vectors differ.)*
