# tdb-search

A semantic search engine built primarily for [TerminusDB](https://terminusdb.com), built in Rust on [LanceDB](https://lancedb.com). It is designed to work standalone.

`tdb-search` is an independent reimplementation of TerminusDB's VectorLink semantic indexer. It preserves VectorLink's **search-family** HTTP shapes and its TerminusDB integration, but replaces the bespoke HNSW index and hand-rolled vector store with **LanceDB** — gaining full-text search, hybrid (vector + keyword) search, and a maintained, versioned, branchable columnar vector store. Embeddings come from a **configurable provider**, defaulting to a local CPU model (served by an Ollama sidecar) so the whole stack runs offline.

---

## Why

The original [`terminusdb-semantic-indexer`](https://github.com/terminusdb-labs/terminusdb-semantic-indexer) (the source of the `terminusdb/vectorlink` image) hardcodes OpenAI embeddings and uses a custom HNSW engine that requires maintenance by the original team. Additionally, it was never completed and proven in production for actual workloads.

`tdb-search`:

- **Preserves the search interface** — the same `/search`, `/similar`, `/duplicates`, `/check`, `/statistics` shapes, so existing search callers keep working (search-family parity; additive search modes layered on).
- **Inverts indexing to a push model** — instead of the indexer pulling content, **TerminusDB pushes** rendered text deltas to it (`GET /last-indexed` → `POST /push`). The indexer never calls back into TerminusDB; it owns the embedding model and is a complete, standalone HTTP search service.
- **Uses LanceDB at the core** — versioned, branchable storage with full-text and hybrid search out of the box (selectable using a mode parameter).
- **Configurable embeddings** — local CPU model via an Ollama sidecar (default), direct OpenAI, or any OpenAI-compatible / generic HTTP endpoint.
- **Reflects TerminusDB history faithfully** — each commit is a versioned snapshot; branch-out shares the parent commit's vector blocks rather than copying them.

## How history maps to LanceDB

TerminusDB history is linear per branch. `tdb-search` mirrors this:

| TerminusDB | tdb-search (LanceDB / Lance) |
|------------|------------------------------|
| domain `org/db` | a Lance dataset |
| commit | a dataset version, bound by a Lance **tag** (`commit:<id>`) |
| index commit `C` from parent `P` | append only the changed documents on top of `P`'s version |
| branch-out at `P` | a Lance branch forked from `P`'s version, **sharing its data blocks** |
| reassign commit pointer | a tag pointing at an existing version (no recompute) |

Vector blocks from the parent commit are reused, never duplicated — the same property the original indexer achieved by sharing an append-only vector store across commits. A **global commit→layer index** (keyed by commit id, per domain, backed by Lance tags) resolves any commit's layer, so a branch forked at commit `P` finds `P`'s vectors regardless of which branch first indexed it.

## Search

A single endpoint serves three modes:

- **`hybrid` (default)** — vector + full-text fused with reciprocal-rank fusion; best out-of-the-box relevance.
- **`vector`** — semantic nearest-neighbour only; reproduces the original VectorLink behaviour exactly.
- **`fts`** — keyword/full-text over the rendered text (filterable by `doc_type`).

The same raw query text drives both sides in hybrid mode. Distances use the reference normalised cosine scale in `[0, 1]` (0 = identical, 0.5 = unrelated, 1 = opposite).

## Embeddings

The embedding provider is configurable:

- **Local CPU (default)** — `nomic-ai/nomic-embed-text-v2-moe` (768-d, multilingual), served by a local **[Ollama](https://ollama.com) sidecar** (from Nomic's official GGUF, Q8_0) exposing an OpenAI-compatible API. arm64-native, runs on CPU, no external network after the one-time model pull.
- **Direct OpenAI** — `api.openai.com` for parity with the original.
- **Generic / OpenAI-compatible HTTP** — any embeddings endpoint, configurable base URL, model, and dimension (e.g. TEI or vLLM if you prefer a different sidecar).

> The default model requires task prefixes (`search_document:` for indexed text, `search_query:` for queries); `tdb-search` injects these automatically from a hard-coded, model-keyed table, so they cannot be misconfigured per deployment.

> An in-process (no-sidecar) embedding mode via `fastembed` is a deferred future option — it does not yet build on aarch64, so the Ollama sidecar is the canonical local runtime.

## Access

The indexer is a **trusted component** behind a single shared admin secret (HTTP Basic, default `admin:root`), checked on every request (`401` on miss) — authentication only, **no RBAC**. TerminusDB fronts search and authorises the caller against its own capability system before calling the indexer. Deploy the indexer on a private network with TerminusDB as the front door.

## Quickstart (planned)

```bash
docker compose up
```

Brings up three services on CPU, no external calls (after the one-time model pull):

- `tdb-search` — this server (HTTP API on `:8080`)
- `embeddings` — Ollama serving `nomic-embed-text-v2-moe` (GGUF, Q8_0)
- `terminusdb` — a TerminusDB server for an end-to-end example

Drive indexing by push, then search (all requests carry the admin secret):

```bash
# 1. ask where the indexer is up to (TerminusDB uses this to compute the delta)
curl -u admin:root 'localhost:8080/last-indexed?domain=admin/star_wars&branch=main'

# 2. push the rendered NDJSON delta for a commit (one operation per line)
curl -u admin:root -X POST \
  'localhost:8080/push?domain=admin/star_wars&branch=main&target_commit=<C>&parent_commit=<P>' \
  -H 'Content-Type: application/x-ndjson' \
  --data-binary @delta.ndjson

# 3. semantic search (raw text body; hybrid by default)
curl -u admin:root 'localhost:8080/search?domain=admin/star_wars&commit=<C>' -d 'Wise old man'
```

In normal operation TerminusDB performs steps 1–2 automatically; the `curl` calls above are how you drive the indexer standalone.

## HTTP API

- **Indexing (push):** `GET /last-indexed`, `POST /push` (NDJSON operation stream, incremental), `GET /check` (poll a push task), `POST /assign` (point a commit at an existing index — a mutation, so POST, not the reference's `GET`).
- **Search:** `POST /search?mode=vector|fts|hybrid` (hybrid default; `doc_type`/`doc_id` filters), `GET /similar`, `GET /duplicates`.
- **Ops:** `GET /statistics`.

Indexing is driven by push (`/last-indexed` + `/push`), not a pull trigger. The full contract is the OpenAPI document: [`openapi.yaml`](./openapi.yaml) — render it with `make docs`.

## Deployment & startup

The engine targets **fast cold-start for KEDA-style scale-from-zero**: a fresh pod binds and answers its liveness probe near-instantly, defers all heavy work (datasets opened on first touch, layer index resolved per request, no boot-time scans), and the embedding tier is scaled separately so an indexer pod never pays model-load cost on cold start.

## Documentation

- Design & specs: [`projects/2026-06-terminusdb-vectorlink/`](../../projects/2026-06-terminusdb-vectorlink/)
- Architecture reference (teaching docs): [`docs/architecture/`](./docs/architecture/)
- Tests not ported from the reference implementations: [`OMITTED_TESTS.md`](./OMITTED_TESTS.md)

## License

Apache-2.0 (see `LICENSE`).
