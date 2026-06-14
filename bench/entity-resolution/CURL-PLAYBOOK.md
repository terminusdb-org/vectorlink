# Curl playbook — drive the Abt-Buy bench by hand

Raw `curl` commands to exercise the same engine lifecycle the JS harness drives,
for poking at the engine without Node. Mirrors `docs/user/07-complete-example.md`
but with the Abt-Buy bench domain and the exact rendered-text shapes the harness
pushes.

```bash
export URL=http://localhost:8081      # the engine (compose maps it to :8081)
export CRED=admin:root                # admin secret (HTTP Basic) — compose default
export DOMAIN=admin/bench_abt_buy     # the dedicated bench domain
export COMMIT=bench-abt-buy-c1        # the deterministic bench commit
```

> `curl -u "$CRED"` sends the admin secret. Health probes don't need it.
> Pipe through `jq` for readable output (optional).

---

## 1. Readiness — wait for index + search both true

```bash
curl -fsS "$URL/health/ready" | jq
# { "ready": true, "index": true, "search": true }
```

`index:true` → can `/push`. `search:true` → embedding backend warm. Poll until both:

```bash
until curl -fsS "$URL/health/ready" | jq -e '.index and .search' >/dev/null; do
  echo "waiting for engine…"; sleep 2
done; echo ready
```

---

## 2. Reset the domain (idempotent) — so a re-run never 409s

```bash
curl -fsS -u "$CRED" -X DELETE "$URL/domain?domain=$DOMAIN"
# 204 No Content (also 204 if it never existed — idempotent)
```

---

## 3. Push a few Buy corpus records (NDJSON) — engine embeds server-side

These three lines are exactly what the `buy.hbs` template renders (note: prices
are normalised — the raw CSV `$99.00` becomes `$99`; optional clauses omitted
when the field is empty).

```bash
TASK=$(curl -fsS -u "$CRED" -X POST \
  -H "Content-Type: application/x-ndjson" \
  "$URL/push?domain=$DOMAIN&branch=main&target_commit=$COMMIT" \
  --data-binary '{"op":"Inserted","id":"terminusdb:///bench/abt_buy/Buy/10011646","string":"The name of the SKU is \"Linksys EtherFast EZXS88W Ethernet Switch - EZXS88W\" and it'"'"'s description is \"Linksys EtherFast 8-Port 10/100 Switch (New/Workgroup)\", the manufacturer is LINKSYS"}
{"op":"Inserted","id":"terminusdb:///bench/abt_buy/Buy/10140760","string":"The name of the SKU is \"Linksys EtherFast EZXS55W Ethernet Switch\" and it'"'"'s description is \"5 x 10/100Base-TX LAN\", the manufacturer is LINKSYS"}
{"op":"Inserted","id":"terminusdb:///bench/abt_buy/Buy/207910213","string":"The name of the SKU is \"Sony PSLX350H Belt Drive Turntable\", the manufacturer is Sony"}')
echo "TASK=$TASK"
```

> The harness pushes all 1092 Buy rows in one NDJSON stream; here we send three
> to demonstrate the wire shape. To push the full corpus by hand, generate the
> NDJSON with the harness: `node src/load.js abt-buy` (it does push + poll).

---

## 4. Poll the push task to completion

```bash
curl -s -u "$CRED" "$URL/check?task_id=$TASK" | jq
# { "status": "Pending", "percentage": 0.0 }     → then …
# { "status": "Complete", "indexed_documents": 3, "skipped": [] }
```

Poll until Complete (the harness watches the `documents` counter in
`/statistics` as the real progress signal — `percentage` is coarse):

```bash
until curl -s -u "$CRED" "$URL/check?task_id=$TASK" | jq -e '.status=="Complete"' >/dev/null; do
  curl -s -u "$CRED" "$URL/statistics" | jq -c '{documents,chunks,pending_index_fragments}'
  sleep 2
done; echo "indexing complete"
```

---

## 5. Search the corpus with an Abt query record (engine embeds the query)

This `q` is exactly what `abt.hbs` renders for an Abt record. The engine embeds
it server-side — do NOT pre-embed.

```bash
curl -fsS -u "$CRED" -X POST "$URL/search" \
  -H "Content-Type: application/json" \
  -d '{
        "domain": "'"$DOMAIN"'",
        "commit": "'"$COMMIT"'",
        "q": "The name of the SKU is \"Linksys EtherFast EZXS88W Ethernet Switch\" and it'"'"'s description is \"8 port 10/100 switch\" with no price indicated",
        "mode": "vector",
        "count": 10
      }' | jq
# [ { "id": "terminusdb:///bench/abt_buy/Buy/10011646", "distance": 0.07, "chunk": {…} }, … ]
```

GET form (link-safe, simple queries):

```bash
curl -fsS -u "$CRED" -G "$URL/search" \
  --data-urlencode "domain=$DOMAIN" \
  --data-urlencode "commit=$COMMIT" \
  --data-urlencode "mode=vector" \
  --data-urlencode "count=10" \
  --data-urlencode "q=Linksys EtherFast 8-port ethernet switch" | jq
```

The **top-1** `id` is what `precision@1` scores against the ground-truth Buy id
for that Abt record (`abt_buy_perfectMapping.csv`).

---

## 6. Inspect engine state

```bash
# Where the lineage is up to:
curl -fsS -u "$CRED" "$URL/last-indexed?domain=$DOMAIN&branch=main" | jq
# { "branch": "main", "commit": "bench-abt-buy-c1", "version": 1 }

# Global counters:
curl -fsS -u "$CRED" "$URL/statistics" | jq
# { "domains": 1, "indexed_commits": 1, "documents": 1092, "chunks": 1092, … }
```

---

## 7. Find duplicates of a known indexed corpus record (similarity)

```bash
curl -fsS -u "$CRED" -G "$URL/similar" \
  --data-urlencode "domain=$DOMAIN" \
  --data-urlencode "commit=$COMMIT" \
  --data-urlencode "id=terminusdb:///bench/abt_buy/Buy/10011646" \
  --data-urlencode "count=5" | jq
```

---

## 8. Corpus-wide near-duplicate pairs — `/duplicates`

`/duplicates` is the bulk path the harness will later swap in to replace the
per-Abt `/search` loop. It scans the indexed snapshot and returns DOCUMENT-level
pairs whose best chunks are within `threshold` (the same `[0,1]` cosine-distance
scale `/search` reports: `0` = identical, `0.5` = unrelated). It is `O(n)`
indexed `nearest(k=2)` queries, not an `O(n²)` all-pairs scan. Lower id first,
deduplicated, a document is never paired with itself.

```bash
curl -fsS -u "$CRED" -G "$URL/duplicates" \
  --data-urlencode "domain=$DOMAIN" \
  --data-urlencode "commit=$COMMIT" \
  --data-urlencode "threshold=0.05" \
  --data-urlencode "start=0" \
  --data-urlencode "count=50" | jq
# Array of [id1, id2] pairs, e.g.
# [
#   ["terminusdb:///bench/abt_buy/Buy/10011646", "terminusdb:///bench/abt_buy/Buy/10140760"]
# ]
```

Parameters:

| param       | meaning                                                                 |
|-------------|-------------------------------------------------------------------------|
| `domain`    | the data product (required)                                             |
| `commit`    | the snapshot to scan (required)                                         |
| `threshold` | max `[0,1]` cosine distance for a pair to count (default `0.0`; `0.05` ≈ near-identical) |
| `start`     | pagination offset over the sorted pair list (default `0`)               |
| `count`     | page size (default `50`)                                                |

> On the Abt-Buy **Buy** corpus this returns `[]` even at `threshold=0.4` — the
> Buy catalogue has no near-identical intra-corpus duplicates (they are distinct
> products). `/duplicates` finds duplicates WITHIN one indexed corpus; the
> Abt↔Buy cross-catalogue match the bench scores is done by querying with the
> Abt text (section 5). When the harness swaps to a bulk path, it would index
> Abt + Buy together and read `/duplicates` for cross-catalogue pairs.

To detect cross-catalogue (Abt↔Buy) duplicates this way, both sides must be in
the SAME indexed snapshot, then filter the returned pairs to those that straddle
the two id namespaces (`…/Abt/…` paired with `…/Buy/…`).

---

## Operational note — file-descriptor pressure under a long search loop (FIXED)

**Resolved (BUG-FD24).** Previously each `/search` opened a fresh `Dataset`
handle (`Dataset::open`) per query, which spins up a NEW Lance object-store +
session whose file readers — including the vector-index files
`_indices/<uuid>/index.idx` + `auxiliary.idx` — held ~2 FDs per search. The
default container soft limit is `nofile=1024`, so a tight sweep hit
`LanceError(IO): Too many open files (os error 24)` after ~140 searches.

The read paths (`io_search`, `io_open_snapshot`, `io_resolve_commit`,
`io_list_commit_versions`) now reuse the CACHED domain handle and
`checkout_version`/`checkout_branch` off it — those SHARE the handle's
object-store + session (`Dataset::checkout_by_ref`), so the index readers are
bounded to one set per cached handle instead of one per query. Writers refresh
the cached handle on every tag/version mutation (`io_tag_commit`,
`io_upsert_chunks`, `io_delete_doc`, optimize, assign), so a freshly-indexed
commit is still resolvable by a subsequent search (the 409 / just-indexed-commit
invariant is preserved). Verified live at the DEFAULT `nofile=1024`: 300+
searches with the open-FD count FLAT (no growth) and zero failures.

Confirm the open-FD count is now stable under load:

```bash
CID=$(docker ps -q -f name=tdb-search-tdb-search-1)
docker exec "$CID" sh -c 'cat /proc/1/limits | grep "open files"'   # default 1024 is fine
docker exec "$CID" sh -c 'ls /proc/1/fd | wc -l'                    # stays flat across a search sweep
```

The raised-ulimit compose override that the bench used as a stopgap is NO LONGER
needed and must not be committed — the fix is in the Rust search path.
