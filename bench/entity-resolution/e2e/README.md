# E2E Test Fixture: Abt/Buy through TerminusDB -> tdb-search

End-to-end test of the full embedding pipeline: TerminusDB renders embedding
text via its Handlebars template engine, pushes deltas to tdb-search, and
search/duplicates queries return results that can be validated against the
ground-truth perfect mapping.

This fixture drives indexing THROUGH TerminusDB (the Phase-6 push driver),
NOT directly against the tdb-search engine (which is what the standalone bench
does).

---

## Prerequisites

### 1. Compose stack running

The 3-service stack (TerminusDB + Ollama + tdb-search) must be running:

```bash
cd twinfoxdb/tdb-search
docker compose up -d
```

Wait for all services to be healthy:

```bash
docker compose ps   # all should show "healthy"
```

### 2. Endpoints (default ports)

| Service    | Host endpoint          | Container-internal     |
|------------|------------------------|------------------------|
| TerminusDB | http://localhost:6365  | http://terminusdb:6363 |
| tdb-search | http://localhost:8081  | http://tdb-search:8080 |
| Ollama     | http://localhost:11434 | http://embeddings:11434|

### 3. Credentials

| Service    | User  | Secret | Env var                   |
|------------|-------|--------|---------------------------|
| TerminusDB | admin | root   | TERMINUSDB_ADMIN_PASS     |
| tdb-search | admin | root   | TDB_SEARCH_ADMIN_SECRET   |

### 4. TerminusDB indexer configuration

TerminusDB must be configured with the push-driver backend. Set these
environment variables on the TerminusDB container (in docker-compose.yml or
via `docker compose exec`):

```yaml
environment:
  - TERMINUSDB_INDEXER_BACKEND=http_tdb_search
  - TERMINUSDB_TDB_SEARCH_ENDPOINT=http://tdb-search:8080
  - TERMINUSDB_SEARCH_ADMIN_USER=admin
  - TERMINUSDB_SEARCH_ADMIN_SECRET=root
```

> **Note:** The default docker-compose.yml in this repo does NOT set these yet.
> You must add them to the `terminusdb` service's environment block and restart.

### 5. Node.js

Node.js 18+ is required for the converter script.

---

## Step-by-step procedure

### Step 1: Convert CSV data to JSON documents

```bash
cd twinfoxdb/tdb-search/bench/entity-resolution/e2e
node convert.js
```

This produces:
- `abt-documents.json` — array of Abt instance documents
- `buy-documents.json` — array of Buy instance documents

Verify the output looks correct (the script prints samples).

### Step 2: Create the database

```bash
curl -u admin:root -X POST "http://localhost:6365/api/db/admin/abt_buy_e2e" \
  -H "Content-Type: application/json" \
  -d '{"label": "Abt-Buy E2E", "comment": "Entity resolution e2e test fixture"}'
```

Expected: HTTP 200 with `{"api:status": "api:success"}`.

### Step 3: Insert the schema

```bash
curl -u admin:root -X POST \
  "http://localhost:6365/api/document/admin/abt_buy_e2e?graph_type=schema&author=admin&message=Add+Abt+Buy+schema+with+embeddings&full_replace=true" \
  -H "Content-Type: application/json" \
  -d @schema.json
```

Expected: HTTP 200. The response body is an empty array `[]` (schema documents
do not return IDs by default).

### Step 4: Load Abt documents

```bash
curl -u admin:root -X POST \
  "http://localhost:6365/api/document/admin/abt_buy_e2e?author=admin&message=Load+Abt+catalogue" \
  -H "Content-Type: application/json" \
  -d @abt-documents.json
```

Expected: HTTP 200 with a JSON array of inserted document IDs.

### Step 5: Load Buy documents (separate commit)

```bash
curl -u admin:root -X POST \
  "http://localhost:6365/api/document/admin/abt_buy_e2e?author=admin&message=Load+Buy+catalogue" \
  -H "Content-Type: application/json" \
  -d @buy-documents.json
```

Expected: HTTP 200 with inserted document IDs.

> **Why two commits?** The push driver processes per-commit deltas. Two separate
> inserts produce two commits, exercising the incremental (per-commit) push path
> — Case 3 in `io_push_delta_/10`.

### Step 6: Trigger the push driver

> **STATUS: PENDING PHASE-6 T4** — There is currently no HTTP endpoint that
> triggers `io_index_branch/3`. The Prolog predicate exists and is fully
> implemented, but its HTTP handler route has not been merged yet.

**Intended command (once the HTTP trigger exists):**

```bash
curl -u admin:root -X POST \
  "http://localhost:6365/api/index/admin/abt_buy_e2e/local/branch/main"
```

**Workaround (swipl -x saved state, proven working):**

The TerminusDB binary is a SWI-Prolog saved state. Invoke the push driver via:

```bash
docker compose exec terminusdb swipl -x /app/terminusdb/terminusdb -g "
triple:super_user_authority(Auth),
transaction:open_descriptor(system_descriptor{}, System_DB),
catch(
    (   api_indexer:io_index_branch(System_DB, Auth, \"admin/abt_buy_e2e/local/branch/main\"),
        format(\"SUCCESS: Push completed~n\"),
        halt(0)
    ),
    Error,
    (   format(\"ERROR: ~w~n\", [Error]),
        halt(1)
    )
)"
```

Note: module-qualified calls are required (`triple:`, `transaction:`,
`api_indexer:`) because the saved state does not auto-import into the user
module.

This will:
1. Call `GET /last-indexed` on tdb-search for the domain
2. Resolve the branch HEAD commit(s)
3. Stream NDJSON push ops for each unseen commit to `POST /push` on tdb-search
4. tdb-search embeds each document via Ollama and indexes into Lance

**What to watch for:**
- The first push on a fresh domain uses Case 2 (full index of HEAD as a single
  `none`-diff push). Even with multiple data commits, the driver pushes ONE diff
  of the HEAD state when the engine has never seen the domain (commit=null).
  The per-commit incremental path (Case 3) only activates on SUBSEQUENT pushes
  where the engine already has an indexed commit.
- Indexing ~2,173 documents takes approximately 8-12 minutes on CPU-only Ollama
  (ARM64, ~3-4 docs/sec). Monitor progress via:
  ```bash
  curl -u admin:root http://localhost:8081/statistics
  ```
  Wait until `documents` reaches 2173 and `indexed_commits` becomes 1.
- A re-push of the same commit returns **409** ("already pushed / in-flight") —
  this is correct idempotency behaviour, not an error.

### Step 7: Verify search works

Once indexing completes (check `/statistics` shows `indexed_commits: 1`), query
the engine directly. The search endpoint is **POST** with parameter `q` (not
`query`) and requires the indexed `commit` hash:

```bash
# Get the indexed commit hash
COMMIT=$(curl -s -u admin:root \
  "http://localhost:8081/last-indexed?domain=admin/abt_buy_e2e/local/branch/main&branch=main" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['commit'])")

# Search for a known Abt product (Sony turntable)
curl -u admin:root -X POST "http://localhost:8081/search" \
  -H "Content-Type: application/json" \
  -d "{
    \"domain\": \"admin/abt_buy_e2e/local/branch/main\",
    \"branch\": \"main\",
    \"commit\": \"$COMMIT\",
    \"q\": \"Sony turntable belt drive\",
    \"k\": 5
  }"
```

Expected: a JSON array of results with IDs. `Abt/552` (Sony PSLX350H) should
appear in the top 5 with distance < 0.05.

### Step 8: Verify entity resolution via /similar

The `/similar` endpoint finds the nearest neighbours to a specific document —
this is the primary cross-catalogue entity resolution mechanism:

```bash
# Find matches for Abt/38477 (Linksys EtherFast EZXS88W)
curl -u admin:root \
  "http://localhost:8081/similar?domain=admin/abt_buy_e2e/local/branch/main&branch=main&commit=$COMMIT&id=terminusdb:///data/Abt/38477&k=5"
```

Expected: `Buy/10011646` at rank 1 with distance ~0.05 (the true pair).

### Step 8b: Verify duplicates (intra-catalogue near-duplicates)

The `/duplicates` endpoint is **GET** (not POST) and finds document pairs within
a distance threshold:

```bash
curl -u admin:root \
  "http://localhost:8081/duplicates?domain=admin/abt_buy_e2e/local/branch/main&branch=main&commit=$COMMIT&threshold=0.05&k=5"
```

Expected: pairs of similar documents (mostly intra-catalogue near-duplicates at
tight thresholds). For cross-catalogue entity resolution, use `/similar` on
individual documents instead.

> **Note on search/duplicates fronting through TerminusDB:** This is Phase-6 T4
> work (not yet merged). For now, query tdb-search directly at port 8081.

### Step 9: Validate against ground truth

Cross-reference known true pairs from `abt_buy_perfectMapping.csv` using
`/similar`:

| idAbt | idBuy    | Expected match                    | Verified result       |
|-------|----------|-----------------------------------|-----------------------|
| 38477 | 10011646 | Linksys EtherFast EZXS88W switch  | Rank 1, distance 0.057 |
| 38475 | 10140760 | Linksys EtherFast EZXS55W switch  | Rank 3 from Abt/38477 |
| 33053 | 10221960 | Netgear ProSafe FS105 switch      | Rank 1, distance 0.048 |

```bash
# Verify Abt/38477 -> Buy/10011646 (should be rank 1)
curl -s -u admin:root \
  "http://localhost:8081/similar?domain=admin/abt_buy_e2e/local/branch/main&branch=main&commit=$COMMIT&id=terminusdb:///data/Abt/38477&k=5"

# Verify Abt/33053 -> Buy/10221960 (should be rank 1)
curl -s -u admin:root \
  "http://localhost:8081/similar?domain=admin/abt_buy_e2e/local/branch/main&branch=main&commit=$COMMIT&id=terminusdb:///data/Abt/33053&k=5"
```

---

## What good looks like

When the fixture runs successfully:

1. `GET /last-indexed?domain=admin/abt_buy_e2e/local/branch/main&branch=main`
   returns a commit hash matching the branch HEAD.
2. `/statistics` shows `documents: 2173`, `indexed_commits: 1`.
3. `/similar` for known Abt records returns their true Buy pairs at rank 1
   with distance < 0.10.
4. `/search` with a product name returns relevant results from both catalogues.

**Expected embedding renderings** (what TerminusDB's Handlebars produces):

- **Abt record 552** (Sony Turntable):
  ```
  Sony Turntable - PSLX350H. Sony Turntable - PSLX350H/ Belt Drive System/ 33-1/3 and 45 RPM Speeds/ Servo Speed Control/ Supplied Moving Magnet Phono Cartridge/ Bonded Diamond Stylus/ Static Balance Tonearm/ Pitch Control
  ```

- **Buy record 10011646** (Linksys switch):
  ```
  Linksys. Linksys EtherFast EZXS88W Ethernet Switch - EZXS88W. Linksys EtherFast 8-Port 10/100 Switch (New/Workgroup)
  ```

These match the bench's v2 rendering (with sentenceCase pre-applied in the data).

---

## Helper decision: pre-normalise in the converter

**Decision:** Option (a) — pre-normalise case in the converter so the template
needs no custom helper.

**Rationale:**

1. TerminusDB's Rust Handlebars renderer (`src/rust/terminusdb-community/src/
   template.rs`) creates a plain `Handlebars::new()` with NO registered helpers.
   A template referencing `{{sentenceCase name}}` would throw:
   `handlebars_render_error("Helper sentenceCase not found", ...)`.

2. The bench's `sentenceCase` helper (defined in `src/text.js`, registered in
   `src/render.js`) de-uppercases purely-alphabetic ALL-CAPS brand words
   (e.g. "LINKSYS" -> "Linksys") while preserving mixed-case tokens and
   alphanumeric model numbers (e.g. "EZXS88W" stays "EZXS88W").

3. By applying this transformation in the converter (`convert.js`), the
   `name` and `manufacturer` fields stored in TerminusDB already carry the
   normalised form. The template then uses plain `{{name}}` / `{{manufacturer}}`
   — no helper needed.

4. This faithfully reproduces the bench's v2 rendered output because the
   normalisation is identical (same algorithm, same token rules).

**Trade-off:** The stored field values differ from the raw CSV (e.g. the `name`
field has de-uppercased brand words). This is acceptable because:
- The `record_id` field preserves the original CSV id for ground-truth scoring.
- The normalisation is deterministic and reversible for display purposes.
- The embedding text — the critical output — matches the proven v2 bench.

---

## Troubleshooting

### Schema render error ("Helper not found")

If you see `handlebars_render_error("Helper sentenceCase not found", ...)`,
the schema template still references `{{sentenceCase ...}}`. Ensure you are
using the `schema.json` from this fixture (plain `{{name}}`, no helpers).

### Authentication failure (401)

- TerminusDB default: `admin:root` (env `TERMINUSDB_ADMIN_PASS`)
- tdb-search default: `admin:root` (env `TDB_SEARCH_ADMIN_SECRET`)

If you changed the admin pass, update the curl `-u` accordingly.

### Push driver refuses ("indexer_backend_not_tdb_search")

TerminusDB's `io_index_branch` predicate gates on `TERMINUSDB_INDEXER_BACKEND=
http_tdb_search`. Ensure this env var is set on the TerminusDB container and
restart it (tabled predicates cache on first call).

### Branch addressing

The push driver derives the `domain` parameter from the resolved descriptor.
The full form is `admin/abt_buy_e2e/local/branch/main`. The engine parses this
via `parse_domain` — if the path is malformed, you'll get a parse error from
the engine.

### Embedding timeout

CPU-only Ollama is slow. If indexing times out or the engine reports incomplete
indexing, increase the timeout or wait longer. Check engine logs:
```bash
docker compose logs -f tdb-search
```

### Database already exists

If you re-run and the database exists:
```bash
curl -u admin:root -X DELETE "http://localhost:6365/api/db/admin/abt_buy_e2e"
```
Then repeat from Step 2.

---

## Pending Phase-6 capabilities

| Capability | Status | Impact on this fixture |
|------------|--------|------------------------|
| Push driver HTTP trigger (T4) | Not merged | Step 6 must use `swipl -x` workaround |
| Search fronting through TerminusDB (T4) | Not merged | Steps 7-8 query tdb-search directly |
| `format=embedding` HTTP endpoint | Feature branch only | Cannot verify renderings via HTTP GET |
| Automatic push-on-commit hook | Not designed | Steps 4-5 would auto-trigger push |

**Note on `format=embedding`:** The endpoint `GET /api/document?format=embedding`
exists on branch `feature/search-v2-embeddings-api` but is NOT on main. The
Docker image built from main returns "requires enterprise edition". However, the
internal rendering path (used by the push driver via `$embedding` Rust module)
IS available in the community build. Successful push + indexing proves the
rendering works correctly without needing the HTTP endpoint.

---

## File manifest

```
e2e/
  README.md           — this manual
  schema.json         — TerminusDB schema (Abt + Buy classes with embedding metadata)
  convert.js          — CSV -> JSON converter (Node.js, reuses bench CSV parser)
  abt-documents.json  — (generated) Abt instance documents
  buy-documents.json  — (generated) Buy instance documents
```
