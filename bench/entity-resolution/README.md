# Entity-resolution accuracy benchmark (tdb-search)

Measures the tdb-search vector engine's **entity-resolution accuracy** against a
standard public benchmark — starting with **Abt-Buy** from the
[Leipzig ER benchmark datasets](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution).
The harness is dataset-agnostic: a second dataset plugs in with a new config
entry + template (see *Adding another dataset*).

## What it measures

Two product catalogues (Abt, Buy) describe overlapping real-world products under
different SKUs and text. The ground-truth `abt_buy_perfectMapping.csv` lists the
true `(idAbt ↔ idBuy)` matches. The bench asks: **does the engine's
nearest-vector match agree with the ground truth?**

- **Corpus = Buy** — all Buy records are pushed into the engine as the indexed
  population.
- **Queries = Abt** — for each Abt record, its rendered text is sent to
  `/search` (the engine embeds it server-side) and the top-K Buy hits are taken.
- **Score** against the perfect mapping:
  - **precision@1** — fraction of Abt records whose **top-1** Buy hit is a
    ground-truth match (the headline accuracy number).
  - **recall@K** for K ∈ {1, 5, 10} — fraction whose correct Buy match appears
    in the top K.

The mapping is many-to-many (1081 unique Abt ids → 1097 pairs), so a query's
ground truth is a **set** of valid Buy ids; a hit counts if the returned id is
any of them.

## Measured result (Abt-Buy, run 2026-06-14)

Engine: `tdb-search` on the compose stack at `:8081`, model `nomic-embed-v2`
(dim 768), `mode=vector`, top-10, all 1081 mapped Abt records scored.

```
precision@1 : 83.16%  (899/1081)
recall@1    : 83.16%  (899/1081)
recall@5    : 96.48%  (1043/1081)
recall@10   : 98.06%  (1060/1081)
```

Many of the 182 top-1 misses have the correct Buy id at rank 2–3 (the embeddings
are close but a slightly nearer non-match wins rank 1) — consistent with the
high recall@5/@10. This is a strong unsupervised vector-only result for Abt-Buy.

## Data quality / encoding caveats

- **Encoding**: `Abt.csv` is **iso-8859-1 (latin-1)**; `Buy.csv` and the mapping
  are ASCII. Everything is read as latin-1 (byte-for-byte safe, never throws on
  odd bytes, ASCII round-trips identically) so no row is silently dropped.
- **No silent row drops**: the CSV parser is strict and **fails loud** on any
  column-count mismatch or unterminated quote — a dropped row would skew the
  accuracy number. Verified counts: Abt 1081, Buy 1092, mapping 1097 pairs.
- **Price normalisation**: raw prices carry a leading `$` and `.00`
  (e.g. `$399.00`). The templates render `${{price}}` and the spec's worked
  example shows `$399`, so the loader strips the currency symbol and a trailing
  `.00`/`.0` to render exactly `$399`. An unparseable price fails loud (it is
  never silently embedded as garbage). Empty prices render the
  "with no price indicated" branch (Abt) or omit the clause (Buy).
- Some Abt records have an empty `price` and many Buy records have empty
  `description`/`manufacturer`; the Handlebars `{{#if}}` guards omit those
  optional clauses, per spec.

## Embedding templates (Handlebars)

Editable `.hbs` files under `templates/`, rendered with the real `handlebars`
npm package (`src/render.js`).

**Abt** (`templates/abt.hbs`):
```handlebars
The name of the SKU is "{{name}}" and it's description is "{{description}}"{{#if price}} with a price of ${{price}}{{else}} with no price indicated{{/if}}
```

**Buy** (`templates/buy.hbs`):
```handlebars
The name of the SKU is "{{name}}"{{#if description}} and it's description is "{{description}}"{{/if}}{{#if manufacturer}}, the manufacturer is {{manufacturer}}{{/if}}{{#if price}} and the price is ${{price}}{{/if}}
```

## Fetch the data

The raw dataset is **NOT committed** (gitignored — re-fetchable, not ours to
redistribute). Fetch + extract it on demand:

```bash
node src/fetch.js abt-buy
```

This downloads `https://dbs.uni-leipzig.de/files/datasets/Abt-Buy.zip` and
extracts `Abt.csv`, `Buy.csv`, `abt_buy_perfectMapping.csv` into `data/`
(idempotent — skips the download if the CSVs are already present). Extraction
uses `python3` (the host has no `unzip`). If network access is blocked, fetch the
zip by hand and drop the three CSVs into `data/`.

## Dependencies

`handlebars` (declared in `package.json`). `npm install` was not run in the build
environment; the dependency is vendored into `node_modules/` (gitignored). In a
normal environment, `npm install` resolves it.

## Run it

Engine must be up at `:8081` with `/health/ready` showing `index:true,
search:true`. Override `ENGINE_URL` / `ENGINE_CRED` if different.

```bash
# Full pipeline: idempotent load → score, prints the scorecard.
node src/bench.js abt-buy        # or: npm run bench:abt-buy

# Or run the steps separately:
node src/fetch.js  abt-buy       # fetch + extract data (idempotent)
node src/load.js   abt-buy       # reset domain, push Buy corpus, wait indexed
node src/verify.js abt-buy       # per-Abt search → top-K → score
```

Tunables (env vars): `BENCH_QUERY_DELAY_MS` (inter-query pause, default 25),
`BENCH_LIMIT` (cap the number of Abt queries for a partial sweep, default 0 =
all).

### Idempotency / reset

`load` first issues `DELETE /domain?domain=admin/bench_abt_buy` (idempotent —
returns 204 even if absent), then pushes a fresh deterministic commit
(`bench-abt-buy-c1`). Re-running never 409s. The corpus count is checked against
the pushed count after indexing — a mismatch fails loud (we never score against
an incomplete corpus).

## How the score is computed

`src/score.js` is pure (no I/O). Given, per Abt record, the ranked Buy ids the
engine returned and the ground-truth Buy-id set:
- `precision@1` = fraction whose top-1 Buy id is in the truth set.
- `recall@K` = fraction with at least one truth id within the top K.
Only Abt records that have a ground-truth mapping are scored.

## Adding another dataset (extensibility)

The whole harness is driven by `src/datasets.js`. To add e.g. Amazon-Google or
DBLP-ACM:

1. Add an entry to `datasets` in `src/datasets.js` — its zip URL + file names,
   per-side encodings, id fields, mapping column names, the IRI base, and the
   `.hbs` template path for each side.
2. Add the matching `.hbs` template(s) under `templates/`.
3. Run `node src/bench.js <new-key>`.

No loader/verifier/scorer changes are needed — they read everything from the
config. The per-query matching step is isolated in
`src/verify.js::ioMatchPerQuery`, so it can later be swapped for the bulk
`/duplicates` path (being built separately) without touching loading or scoring.

---

# v2 — reciprocal cross-NN entity resolution (spec 17 §4)

v2 is the full **reciprocal / mutual-nearest-neighbour** ER algorithm of
[`specs/17-entity-resolution-framework.md`](../../../../projects/2026-06-terminusdb-vectorlink/specs/17-entity-resolution-framework.md).
Where v1 indexes only Buy and queries with un-indexed Abt text (re-embedding
every query), v2 indexes **both** catalogues into **one snapshot** and anchors all
comparisons on already-stored vectors. v1 remains runnable as the **baseline**
(`node src/bench.js abt-buy`, precision@1 83.16%).

## The algorithm (pure, `src/resolve.js`)

Given each record's top-K cross-catalogue neighbours in **both** directions
(Abt→Buy and Buy→Abt, distances on the `[0,1]` cosine scale):

1. **Index both populations** into one snapshot, distinct id namespaces
   (`…/Abt/<id>`, `…/Buy/<id>`) — so a pair's provenance is in its ids, and the
   engine's `doc_type` (= the IRI's second-to-last segment, `Abt`/`Buy`) scopes
   each cross-NN direction to the opposite catalogue.
2. **Reciprocal cross top-K NN** → a sparse bipartite candidate graph: an edge
   `(a,b)` exists iff `b ∈ topK(a)` **or** `a ∈ topK(b)`.
3. **Threshold cap τ (default 0.5)** — prune every edge with distance `> τ`. A
   record left with no edge is a legitimate non-match.
4. **Mutual top-K grounding** — accept directly every pair mutually within the
   other's top-K (`b ∈ topK(a)` **and** `a ∈ topK(b)`). Near-linear; resolves the
   confident majority and removes those nodes before the only expensive step.
5. **Per-cluster optimal assignment** — the residual fragments into small
   independent **connected components**; within **each component** run the
   **minimum-cost bipartite (Hungarian)** assignment (`src/hungarian.js`), never a
   global N×M matrix. A **runaway-component guard** caps component size and falls
   back to greedy within an oversized component (logged), so a pathological knot
   can never hang the run.
6. **Leave the rest unmatched** — abstain rather than force a least-bad pair.

**Performance contract (§6):** the O(n³) assignment is structurally confined to
clusters whose size **k** bounds. `k`, `τ`, and grounding-first are simultaneously
the recall knobs and the cost knobs. Each run reports cluster-size distribution,
the max component observed, runaway count, and per-cluster assignment time.

## Refinements baked in (A, E)

The v2 templates (`templates/abt.v2.hbs`, `templates/buy.v2.hbs`) differ from v1:

- **A — price removed** from the embedded text entirely (the same product has
  different prices across catalogues → price tokens push true matches apart).
  Price is retained only as record metadata, never in the vector.
- **E — vendor front + sentence-case.** Buy leads with its `manufacturer`
  (matching Abt, which leads with the brand inside `name`), and a `{{sentenceCase}}`
  Handlebars helper de-uppercases all-caps **brand words** (`LINKSYS` → `Linksys`)
  so the brand token aligns lexically across catalogues. **Model numbers are
  preserved verbatim** — `sentenceCase` only de-allcaps purely-alphabetic tokens,
  leaving alphanumeric codes (`EZXS88W`, `PSLX350H`) untouched, because their
  exact casing is identity-bearing (`src/text.js`).

Rendered examples (`node -e` against the templates):
```
BUY : Linksys. Linksys EtherFast EZXS88W Ethernet Switch - EZXS88W. Linksys EtherFast 8-Port 10/100 Switch (New/Workgroup)
ABT : Sony Turntable - PSLX350H. Sony Turntable - PSLX350H/ Belt Drive System/ 33-1/3 and 45 RPM Speeds
```

## Three selectable matching MODES

The algorithm above is generic; the **retrieval primitive** that supplies its
cross-NN candidates is selectable (`--mode`, `src/modes.js`):

| mode | endpoint | vectors | top-K? | run order | note |
|------|----------|---------|--------|-----------|------|
| **`search`** | per-record `POST /search` | embeds query text | yes (true top-K) | **first** | v1-style retrieval feeding the v2 algorithm; no engine dependency beyond what's live |
| **`duplicates`** | bulk `GET /duplicates` set/target | stored vectors | **top-1 per direction** | **second** | fast bulk path; the endpoint emits ONE nearest neighbour per set point, so it grounds at effective k=1 and the residual is isolated pairs |
| **`similar`** | per-record `POST /similar` (scoped) | stored (but see note) | yes (true top-K) | **last, gated** | the engine still **re-embeds** the anchor text (fix pending), so the headline speed-up is deferred; the mode is wired and correct, its RUN waits on the engine fix |

**Mode caveat — `duplicates` is top-1, not top-K.** The `/duplicates` set/target
endpoint runs one filtered ANN per set point and returns its single nearest
in-target neighbour (`io_nearest_neighbour` over-fetches k=8 only to absorb ANN
recall slack, then takes the first row). So `duplicates` mode feeds the resolver a
top-1 list per direction: mutual grounding reduces to mutual-nearest-neighbour
(k=1), and the per-cluster residual is isolated pairs. This is a valid (and the
spec's k=1) configuration — documented honestly rather than faked as top-K. True
top-K bulk would need an engine change to return k>1 per set point.

## Run it (after the engine is confirmed up)

```bash
# search mode FIRST (loads both catalogues, scores, prints config + scorecard)
node src/bench-v2.js --mode search abt-buy-v2          # or: npm run bench:v2:search

# duplicates mode SECOND (reuses the snapshot already indexed by the first run)
node src/bench-v2.js --mode duplicates --no-load abt-buy-v2   # npm run bench:v2:duplicates

# similar mode — wired, RUN DEFERRED until the reuse-stored-vector engine fix lands
node src/bench-v2.js --mode similar --no-load abt-buy-v2      # npm run bench:v2:similar
```

Knobs (all printed per run): `--mode search|duplicates|similar`, `--k N`
(fan-out, default 5), `--threshold T` (τ cap, default 0.5), `--max-component N`
(runaway guard, default 200), `--no-load` (reuse an already-indexed snapshot for
cheap A/B sweeps), `--query-delay-ms N`. Unknown flags **fail loud**.

The scorecard reports: grounded (Step 4) vs assigned (Step 5) **pair counts and
per-stage precision**, overall precision + recall vs the perfect mapping,
**cluster-size distribution + assignment time** (the §6 contract), and wall-clock
gather time vs the v1 baseline line.

## Offline unit tests (pure logic — no engine)

The pure core is fully unit-tested offline with the Node built-in runner (no
`npm install`):

```bash
npm test            # node --test test/*.test.js
```

Covers: `hungarian` (min-cost assignment incl. the greedy-failure case,
rectangular matrices), `text`/`sentenceCase` (brand de-allcaps, model-number
preservation), the v2 `render` templates (refinements A + E), the `resolve`
pipeline (τ cap, mutual-top-K grounding incl. rank-2 recovery, connected
components, per-cluster optimal assignment beating greedy, runaway fallback,
unmatched abstention), and `score-v2` (per-stage precision split, recall).

## Adding another dataset (v2)

Add an entry to `datasets.js` with a `sides: { abt, buy }` block (each side's CSV,
id field, `doc_type`, and v2 `.hbs` template), a distinct `domain`/`commit`, and
the mapping columns. No resolver/scorer/mode changes are needed — they are
dataset-agnostic.

---

## Layout

```
bench/entity-resolution/
  README.md            ← this file (method spec, v1 + v2)
  CURL-PLAYBOOK.md     ← drive the engine by hand with curl
  package.json         ← scripts + handlebars dependency
  templates/
    abt.hbs            ← v1 Abt embedding template (baseline)
    buy.hbs            ← v1 Buy embedding template (baseline)
    abt.v2.hbs         ← v2 Abt template (price-free, sentence-cased)
    buy.v2.hbs         ← v2 Buy template (vendor-front, sentence-cased, no price)
  src/
    datasets.js        ← dataset registry (abt-buy + abt-buy-v2)
    csv.js             ← strict, fail-loud CSV parser
    render.js          ← handlebars renderer (+ sentenceCase helper)
    text.js            ← pure brand-alignment text helpers (sentenceCase)
    engine.js          ← tdb-search HTTP client (push/check/search/similar/duplicates/delete)
    load-records.js    ← v1 load + render the CSV sides and the mapping
    load.js            ← v1 idempotent push of the Buy corpus, wait indexed
    load-v2.js         ← v2 push of BOTH catalogues into one snapshot, wait indexed
    modes.js           ← the three retrieval-mode adapters (search/duplicates/similar)
    resolve.js         ← PURE v2 algorithm (graph → ground → components → assign)
    hungarian.js       ← PURE min-cost bipartite (Hungarian) assignment
    score.js           ← v1 pure precision@1 / recall@K scoring
    score-v2.js        ← v2 pure scoring (grounded vs assigned split + recall)
    verify.js          ← v1 per-Abt search → top-K → score → scorecard
    bench.js           ← v1 entrypoint: load → verify
    bench-v2.js        ← v2 entrypoint: load → gather(mode) → resolve → score
    fetch.js           ← download + extract the dataset zip
  test/                ← offline unit tests (node --test)
  data/                ← raw CSVs (gitignored; fetched on demand)
```
