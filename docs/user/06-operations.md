# 6. Operations

Auth, health, statistics, and configuration for running the engine.

## Authentication

Every functional endpoint requires a shared **admin secret** over HTTP Basic:

```bash
curl -u admin:root 'http://localhost:8080/statistics'
```

- Default `admin:root` — **change it for any exposed deployment** (see configuration below).
- Missing or wrong secret → `401` on every endpoint.
- This is authentication only — the engine performs no per-user authorisation (RBAC). It is a trusted component: run it on a private network with your front door (e.g. TerminusDB) authorising callers.
- The **health probes are the only unauthenticated endpoints**, so an orchestrator can probe an instance that is scaling up.

There is **no embedding-key request header** — the engine owns its model and calls the provider itself; any provider key is the engine's own server-side configuration.

## Health & readiness

```bash
curl -fsS http://localhost:8080/health/live    # process up — answers immediately
curl -fsS http://localhost:8080/health/ready    # can it actually serve?
```

Readiness is **per-capability** so an orchestrator can route/scale on the right signal:

```json
{ "ready": true, "index": true, "search": false }
```

- `index` — store reachable, can accept `/push`.
- `search` — store **and** embedding backend warm, can serve `/search`.

An instance can be ready-to-index before ready-to-search. While the embedding backend is warming, `/search` returns `503` with `Retry-After` rather than blocking or returning empty results. Probes do no heavy work, so a fresh instance becomes serveable quickly.

## Statistics

```bash
curl -u admin:root http://localhost:8080/statistics
```

```json
{
  "domains": 3,
  "branches": 9,
  "indexed_commits": 412,
  "documents": 18044,
  "chunks": 51234
}
```

Counters are best-effort/advisory (may be approximate under concurrency) — don't gate logic on exact values.

## Configuration

Resolution order per setting: **command-line flag > environment variable > built-in default**. Invalid or missing-required values **fail at startup** with a message naming the setting — the engine never starts in a half-configured state.

| Setting | Flag | Env | Default |
|---------|------|-----|---------|
| storage directory | `--directory` | — | (required) |
| listen port | `--port` | — | `8080` |
| embedding provider | `--provider` | `..._PROVIDER` | local sidecar (OpenAI-compatible) |
| provider URL | `--embed-url` | `..._EMBED_URL` | the bundled embeddings service |
| embedding model | `--model` | `..._MODEL` | `nomic-ai/nomic-embed-text-v2-moe` |
| embedding dimension | `--dim` | `..._DIM` | `768` (fixed per index once created) |
| provider key (if the provider needs one) | `--embed-key` | `..._EMBED_KEY` / `OPENAI_API_KEY` | — |
| negative-cache TTL (seconds) | `--neg-cache-ttl` | `..._NEG_CACHE_TTL` | `3600` |
| admin user | `--admin-user` | `TDB_SEARCH_ADMIN_USER` | `admin` |
| admin secret | `--admin-secret` | `TDB_SEARCH_ADMIN_SECRET` | `root` |

The embedding dimension is fixed when an index is first created and is immutable; a query whose embedding dimension doesn't match the index fails loudly (`409`) — it is never silently padded.

## Failure behaviour

The engine fails loudly and never masks errors:

| Situation | Response |
|-----------|----------|
| Missing/wrong admin secret | `401` |
| Missing/invalid parameter | `400`, naming the parameter |
| Branch has no indexed ancestor | `404` |
| Embedding backend cold/unreachable | `503` + `Retry-After` |
| Query embedding dimension ≠ index | `409` |
| Provider error / store read failure | `5xx` with the cause in the body |

An empty result array always means "genuinely nothing matched" — never a hidden failure.
