# 7. Complete example — Star Wars, end to end

A full search lifecycle driven entirely with `curl`: check readiness, index a commit, watch indexing complete, then run vector, full-text, hybrid, filtered, and similarity searches — plus history (a second commit) and a couple of operational calls.

Every command uses two short environment variables so you can paste the whole thing against your server:

```bash
export URL=http://localhost:8081      # the engine (docker compose maps it to :8081)
export CRED=admin:root                # admin secret (HTTP Basic) — the compose default
```

> `curl -u "$CRED"` sends the admin secret on every call. Health probes don't need it.
> Pipe responses through `jq` for readable output (optional).

Throughout, the data product (domain) is **`admin/star_wars`** and we index two commits, `c1` then `c2`.

---

## 1. Wait until the engine is ready

```bash
curl -fsS "$URL/health/live"
# {"status":"ok"}

curl -fsS "$URL/health/ready" | jq
# { "ready": true, "index": true, "search": true }
```

`index: true` → it can accept pushes. `search: true` → the embedding backend is warm.
Poll until **both** are true — don't sleep a fixed amount:

```bash
until curl -fsS "$URL/health/ready" | jq -e '.index and .search' >/dev/null; do
  echo "waiting for engine…"; sleep 2
done
echo "ready"
```

---

## 2. Ask where the engine is up to

Before indexing, find the last-indexed commit for this `(domain, branch)`. On a fresh engine it's `null`.

```bash
curl -fsS -u "$CRED" \
  "$URL/last-indexed?domain=admin/star_wars&branch=main" | jq
# { "branch": "main", "commit": null, "version": 0 }
```

`commit: null` means this lineage has never been indexed — so the first push has **no** `parent_commit` (every document is an `Inserted`).

---

## 3. Index the first commit (`c1`) — push an NDJSON delta

The push body is an **NDJSON stream**: one operation per line. Each `Inserted`/`Changed` carries the rendered text the engine will chunk and embed.

And capture the task id straight from the push as TASK. Use `@file.ndjson` to reference a file:

```bash
TASK=$(curl -fsS -u "$CRED" -X POST \
  -H "Content-Type: application/x-ndjson" \
  "$URL/push?domain=admin/star_wars&branch=main&target_commit=c1" \
  --data-binary $'
{"op":"Inserted","id":"terminusdb:///star-wars/People/luke","string":"The person\'s name is Luke Skywalker. A Jedi Knight from Tatooine, son of Anakin, trained by Yoda and Obi-Wan to use the light side of the Force."}
{"op":"Inserted","id":"terminusdb:///star-wars/People/vader","string":"The person\'s name is Darth Vader. A Sith Lord consumed by the dark side of the Force, once the Jedi Anakin Skywalker, now enforcer of the Galactic Empire."}
{"op":"Inserted","id":"terminusdb:///star-wars/People/yoda","string":"The person\'s name is Yoda. An ancient and wise Jedi Master who trained generations of Jedi in the ways of the Force on Dagobah."}
{"op":"Inserted","id":"terminusdb:///star-wars/Species/wookiee","string":"The species is Wookiee. Tall, strong, hairy warriors from the forest planet Kashyyyk; Chewbacca is the best known."}
')
echo "TASK=$TASK"
# TASK=task-7f3a9c...
```

The response is an opaque **task id** — indexing runs asynchronously.

> Note the leading newline in `--data-binary $'\n…'` is harmless (blank lines are skipped). `$'…'` lets the embedded `\'` escapes work in the rendered strings.

Poll `/check`. `/check` reports status in the body:

```bash
curl -s -u "$CRED" "$URL/check?task_id=$TASK"
# { "status": "Pending", "percentage": 50.0 }            [http 200]
# …then…
# { "status": "Complete", "indexed_documents": 4, "skipped": [] }   [http 200]
```

Confirm the lineage now resolves:

```bash
curl -fsS -u "$CRED" \
  "$URL/last-indexed?domain=admin/star_wars&branch=main" | jq
# { "branch": "main", "commit": "c1", "version": 1 }
```

---

## 5. Search

All searches name the data product (`domain`) and the snapshot (`commit`). Results are `[{ "id", "distance" }]`, **nearest first** — `distance` is in `[0, 1]`, `0` = identical.

### 5a. Vector search (semantic) — the default is hybrid, so ask for `vector` explicitly

```bash
curl -fsS -u "$CRED" -G "$URL/search" \
  --data-urlencode "domain=admin/star_wars" \
  --data-urlencode "commit=c1" \
  --data-urlencode "mode=vector" \
  --data-urlencode "q=wise old Jedi master who trains others" | jq
# [
#   { "id": "terminusdb:///star-wars/People/yoda",  "distance": 0.07 },
#   { "id": "terminusdb:///star-wars/People/luke",  "distance": 0.21 },
#   { "id": "terminusdb:///star-wars/People/vader", "distance": 0.34 }
# ]
```

Yoda ranks first — semantic match, even though the query shares no exact words with his description.

### 5b. Full-text search (keyword)

```bash
curl -fsS -u "$CRED" -G "$URL/search" \
  --data-urlencode "domain=admin/star_wars" \
  --data-urlencode "commit=c1" \
  --data-urlencode "mode=fts" \
  --data-urlencode "q=Kashyyyk" | jq
# [ { "id": "terminusdb:///star-wars/Species/wookiee", "distance": 0.12 } ]
```

Exact rare term → only the Wookiee document.

### 5c. Hybrid search (default) — combines vector + FTS via RRF

`mode` omitted ⇒ hybrid:

```bash
curl -fsS -u "$CRED" -G "$URL/search" \
  --data-urlencode "domain=admin/star_wars" \
  --data-urlencode "commit=c1" \
  --data-urlencode "q=dark side of the Force" | jq
# [
#   { "id": "terminusdb:///star-wars/People/vader", "distance": 0.05 },
#   { "id": "terminusdb:///star-wars/People/luke",  "distance": 0.40 },
#   ...
# ]
```

Vader first — strongest on both the semantic and keyword signals.

### 5d. Filter by document type

`doc_type`/`doc_id` are **repeated** params (not comma-joined). Restrict to species only:

```bash
curl -fsS -u "$CRED" -G "$URL/search" \
  --data-urlencode "domain=admin/star_wars" \
  --data-urlencode "commit=c1" \
  --data-urlencode "q=warrior from a forest world" \
  --data-urlencode "doc_type=Species" | jq
# [ { "id": "terminusdb:///star-wars/Species/wookiee", "distance": 0.18 } ]
```

### 5e. Paginate, and get a text snippet

```bash
curl -fsS -u "$CRED" -G "$URL/search" \
  --data-urlencode "domain=admin/star_wars" \
  --data-urlencode "commit=c1" \
  --data-urlencode "q=Jedi" \
  --data-urlencode "start=0" \
  --data-urlencode "count=2" \
  --data-urlencode "snippet=true" | jq
# [
#   { "id": ".../yoda", "distance": 0.11, "chunk": { "index": 0, "count": 1, "snippet": "…wise Jedi Master…" } },
#   { "id": ".../luke", "distance": 0.19, "chunk": { ... } }
# ]
```

### 5f. The same search via POST (JSON body)

Use POST for long or programmatically-built queries. Body fields override query params.

```bash
curl -fsS -u "$CRED" -X POST "$URL/search" \
  -H "Content-Type: application/json" \
  -d '{
        "domain": "admin/star_wars",
        "commit": "c1",
        "mode": "hybrid",
        "q": "Sith Lord of the Galactic Empire",
        "count": 3
      }' | jq
```

### 5g. Find documents similar to a known one

```bash
curl -fsS -u "$CRED" -G "$URL/similar" \
  --data-urlencode "domain=admin/star_wars" \
  --data-urlencode "commit=c1" \
  --data-urlencode "id=terminusdb:///star-wars/People/vader" | jq
# Neighbours of Vader — Anakin/Jedi-adjacent people rank highest.
```

---

## 6. A second commit (`c2`) — change and delete

History moves forward by pushing the delta from the last-indexed commit. Now `parent_commit=c1`. We **change** Vader's text and **delete** the Wookiee.

```bash
curl -fsS -u "$CRED" -X POST \
  -H "Content-Type: application/x-ndjson" \
  "$URL/push?domain=admin/star_wars&branch=main&target_commit=c2&parent_commit=c1" \
  --data-binary $'
{"op":"Changed","id":"terminusdb:///star-wars/People/vader","string":"The person\'s name is Anakin Skywalker, redeemed. He turned away from the dark side and destroyed the Emperor, fulfilling the prophecy of the Chosen One."}
{"op":"Deleted","id":"terminusdb:///star-wars/Species/wookiee"}
'
# task-9b2e10  → poll /check as in step 4
```

After it completes, the two snapshots differ — and **history is preserved**:

```bash
# c2: Vader's text is now the redemption arc; "Kashyyyk" no longer matches anything.
curl -fsS -u "$CRED" -G "$URL/search" \
  --data-urlencode "domain=admin/star_wars" \
  --data-urlencode "commit=c2" \
  --data-urlencode "mode=fts" \
  --data-urlencode "q=Kashyyyk" | jq
# []   ← the Wookiee was deleted at c2

# c1: the old snapshot is unchanged — Kashyyyk still matches.
curl -fsS -u "$CRED" -G "$URL/search" \
  --data-urlencode "domain=admin/star_wars" \
  --data-urlencode "commit=c1" \
  --data-urlencode "mode=fts" \
  --data-urlencode "q=Kashyyyk" | jq
# [ { "id": ".../Species/wookiee", "distance": 0.12 } ]
```

Searching `c1` always sees `c1`'s data; searching `c2` sees the updated set. The engine never mutates an old snapshot.

---

## 7. Operational calls

### Statistics

```bash
curl -fsS -u "$CRED" "$URL/statistics" | jq
# {
#   "domains": 1,
#   "branches": 1,
#   "indexed_commits": 2,
#   "documents": 3,
#   "chunks": 3,
#   "pending_index_fragments": 0
# }
```

`pending_index_fragments: 0` means indexing has caught up (no backlog).

### Assign a commit to an existing snapshot (no re-index)

If a new commit `c3` indexes to the same content as `c2` (e.g. a no-op metadata commit), point it at `c2`'s snapshot instead of re-embedding:

```bash
curl -fsS -u "$CRED" -X POST \
  "$URL/assign?domain=admin/star_wars&source_commit=c2&target_commit=c3"
# 204 No Content   ← c3 now searches identically to c2, zero embedding cost
```

### Delete the whole data product

When the data product is removed, purge its entire search footprint (idempotent — a repeat returns success, not 404):

```bash
curl -fsS -u "$CRED" -X DELETE "$URL/domain?domain=admin/star_wars"
# 204 No Content
```

---

## What this exercised

| Step | Endpoint | Lifecycle stage |
|------|----------|-----------------|
| 1 | `GET /health/{live,ready}` | readiness |
| 2 | `GET /last-indexed` | catch-up handshake |
| 3–4 | `POST /push` → `GET /check` | indexing (async) |
| 5a–5g | `GET/POST /search`, `GET /similar` | vector / FTS / hybrid / filter / paginate / similar |
| 6 | `POST /push` (Changed/Deleted) | history — a second commit, snapshot isolation |
| 7 | `GET /statistics`, `POST /assign`, `DELETE /domain` | operations |

Distances and exact rankings depend on the embedding model; the *ordering* shown (right document first) is the behaviour to expect.
