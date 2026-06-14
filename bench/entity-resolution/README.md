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

## Layout

```
bench/entity-resolution/
  README.md            ← this file (method spec)
  CURL-PLAYBOOK.md     ← drive the engine by hand with curl
  package.json         ← scripts + handlebars dependency
  templates/
    abt.hbs            ← Abt embedding template
    buy.hbs            ← Buy embedding template
  src/
    datasets.js        ← dataset registry (the extensibility point)
    csv.js             ← strict, fail-loud CSV parser
    render.js          ← handlebars renderer + price normalisation
    engine.js          ← tdb-search HTTP client (push/check/search/delete)
    load-records.js    ← load + render the CSV sides and the mapping
    load.js            ← idempotent push of the Buy corpus, wait indexed
    score.js           ← pure precision@1 / recall@K scoring
    verify.js          ← per-Abt search → top-K → score → print scorecard
    fetch.js           ← download + extract the dataset zip
    bench.js           ← single entrypoint: load → verify
  data/                ← raw CSVs (gitignored; fetched on demand)
```
