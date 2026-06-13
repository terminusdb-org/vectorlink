# 2. Quickstart

Bring up the stack, index one commit, and run a search — all over HTTP.

## 1. Start the stack

```bash
docker compose up
```

This starts three services on CPU, with no external network after the one-time model pull:

- `tdb-search` — the engine (HTTP API on `:8080`)
- `embeddings` — a local embedding model server
- `terminusdb` — a TerminusDB server (only needed for the end-to-end example; the steps below don't use it)

Wait until the engine is ready:

```bash
curl -fsS http://localhost:8080/health/ready | jq
# { "ready": true, "index": true, "search": true }
```

`index: true` means it can accept pushes; `search: true` means the embedding backend is warm. Poll until both are true — don't sleep a fixed amount.

## 2. Index a commit (push)

Create a small NDJSON delta — one operation per line:

```bash
cat > delta.ndjson <<'EOF'
{"op":"Inserted","id":"terminusdb:///star-wars/People/20","string":"The person's name is Yoda. A wise old Jedi master, small and green."}
{"op":"Inserted","id":"terminusdb:///star-wars/Species/8","string":"The Mon Calamari are an amphibious species resembling squid, known as skilled starship engineers."}
EOF
```

Find where the engine is up to (empty on first use), then push the delta as commit `c1`:

```bash
curl -u admin:root 'http://localhost:8080/last-indexed?domain=admin/star_wars&branch=main'
# { "branch": "main", "commit": null, "version": 0 }

curl -u admin:root -X POST \
  'http://localhost:8080/push?domain=admin/star_wars&branch=main&target_commit=c1' \
  -H 'Content-Type: application/x-ndjson' \
  --data-binary @delta.ndjson
# task-7f3a9c
```

Poll the task until complete:

```bash
curl -u admin:root 'http://localhost:8080/check?task_id=task-7f3a9c'
# { "status": "Complete", "indexed_documents": 2, "skipped": [] }
```

## 3. Search

```bash
# GET — simple and cacheable
curl -u admin:root 'http://localhost:8080/search?domain=admin/star_wars&commit=c1&q=wise+old+man'

# POST — structured JSON
curl -u admin:root -X POST 'http://localhost:8080/search' \
  -H 'Content-Type: application/json' \
  -d '{"domain":"admin/star_wars","commit":"c1","q":"who are the squid people"}'
```

Expected (hybrid, the default): `People/20` (Yoda) tops "wise old man"; `Species/8` (Mon Calamari) tops "squid people" — even though neither rendered text contains those exact words. That is the semantic payoff.

```json
[
  { "id": "terminusdb:///star-wars/Species/8", "distance": 0.0941 }
]
```

---

Next: [Indexing by push](./03-indexing.md).
