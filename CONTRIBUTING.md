# Contributing to tdb-search

## What this is

A standalone semantic search engine in Rust on LanceDB with version history and branching support. TerminusDB pushes rendered text deltas to it; tdb-search embeds, indexes, and serves vector/keyword/hybrid search. No pull from TerminusDB — indexing is entirely push-driven.

Borrows some concepts from the original VectorLink implementation.

## Build

Builds run inside a pinned Docker image (`Dockerfile.build`) with all Rust deps pre-baked. No local toolchain needed beyond Docker and Make.

```bash
make build-image     # one-time: build the CI/dev container image
make dev             # incremental debug build → target/debug/tdb-search
make build-release   # release build (no LTO, fast enough for CI)
make release-image   # production build (LTO, opt-level=s — slow, for shipping only)
```

If you prefer building without Docker: `cargo build` works directly with a local Rust toolchain.

## Test

```bash
make test              # cargo test (Rust unit tests)
make test-integration  # mocha integration suite against a live container
make pr                # full pre-PR gate: lint + test + integration + release build + docs
```

The rust debug build is 10x slower than the release build. The release build is recommended generally.

### Local test servers

`tests/tdb-search-server.sh` starts both tdb-search (port 7372) and a paired TerminusDB test server (port 7373), in ../terminusdb:

```bash
make server-start     # start both
make server-stop      # stop both
make server-clean     # stop, wipe storage, start fresh
make server-status    # show status
```

### E2E test with TerminusDB

The E2E test (`tests/e2e/indexer-push-e2e.js` in the terminusdb repo) exercises the full pipeline: schema with embedding metadata, document insertion, automatic indexer push, and search through both the TerminusDB proxy and tdb-search directly.

```bash
make server-start
cd ../terminusdb
TERMINUSDB_BASE_URL=http://127.0.0.1:7373 \
TDB_SEARCH_URL=http://127.0.0.1:7372 \
npx mocha --timeout 180000 tests/e2e/indexer-push-e2e.js
cd ../tdb-search
make server-stop
```

## Lint

```bash
make lint             # OpenAPI (Redocly) + clippy (-D warnings) + eslint
```

`#![forbid(unsafe_code)]` is in every crate. Introducing `unsafe` requires a reviewed, signed commit that removes the forbid attribute.

## Source layout

```
src/
  main.rs            Server binary + prime-embed-cache CLI subcommand
  bin/load.rs        Standalone bulk-load binary (tdb-search-load)
  lib.rs             Shared module declarations
  config/            Environment-based configuration
  chunk/             Tokenizer-driven text chunking
  embed/             Embedding provider client + disk-backed cache (sled/zstd)
    cache.rs         EmbedCache — sled KV store with zstd-compressed vectors
  http_api/          Axum HTTP handlers + routing
  service/           Core search service: index pipeline, push, resolve, search
  store/lance/       LanceDB storage layer: datasets, tags, search, resolve, dedup
  kernel/            Domain models, graph specs, resource paths
  layeridx/          Commit→layer index (Lance tags per domain)
  resolve/           Entity resolution (flat-KNN candidate materialization)
  ingest/            NDJSON push parsing
  indexqueue/        Phase-3 placeholder (unused, retained for future parallel indexing)
```

## Configuration

All config is environment variables, read in `src/config/mod.rs`:

| Variable | Default | Description |
|---|---|---|
| `TDB_SEARCH_PORT` | `8080` | Listen port |
| `TDB_SEARCH_DATA_DIR` | `/data` | Lance dataset storage path |
| `TDB_SEARCH_EMBED_URL` | `http://127.0.0.1:11434` | Embedding provider URL (Ollama) |
| `TDB_SEARCH_MODEL` | `nomic-embed-text-v2-moe` | Embedding model name |
| `TDB_SEARCH_DIM` | `768` | Embedding vector dimension |
| `TDB_SEARCH_EMBED_BATCH_SIZE` | `32` | Texts per embedding HTTP call |
| `TDB_SEARCH_EMBED_CACHE_SIZE` | `20000` | Max embed cache entries (`none` to disable) |
| `TDB_SEARCH_TOKENIZER_PATH` | `/data/tokenizer.json.bz2` | Tokenizer file (bz2 or plain JSON) |
| `TDB_SEARCH_ADMIN_USER` | `admin` | HTTP Basic auth user |
| `TDB_SEARCH_ADMIN_PASSWORD` | `root` | HTTP Basic auth password |

## TerminusDB integration

TerminusDB pushes to tdb-search via the indexer plugin. Set these in the TerminusDB environment:

| Variable | Default | Description |
|---|---|---|
| `TERMINUSDB_INDEXER_BACKEND` | `http_tdb_search` | Indexer backend |
| `TERMINUSDB_TDB_SEARCH_ENDPOINT` | `http://127.0.0.1:7372` | Push target URL |

## Workflow

1. `make dev` — build
2. `make lint` — must pass before commit
3. `make test` — unit tests
4. `make test-integration` — integration suite
5. `make pr` — full gate before opening a PR

Commits are GPG-signed. The `make pr` target is the aggregate gate — it must be green.

## Security

### No `unsafe` code

`#![forbid(unsafe_code)]` is enforced crate-wide. Introducing `unsafe` requires a reviewed, signed commit that removes the forbid attribute.

### Lance filter construction — use `filter_expr`, never string interpolation

Lance (via DataFusion) accepts both SQL string filters (`scanner.filter(&str)`) and pre-parsed `Expr` filters (`scanner.filter_expr(Expr)`). **Always use `filter_expr` with `Expr` values.** Never construct SQL filter strings by interpolating user-supplied data.

The SQL string path is vulnerable to injection. Single-quote doubling (`'` → `''`) is insufficient because:

- **Backslash escaping**: In SQL dialects that interpret `\` as an escape character (MySQL-style), a doc_id like `x\'` breaks out of the string literal after quote doubling. The `\` survives `replace('\'', "''")`, producing `\''` — which the parser may interpret as an escaped quote followed by a string terminator.
- **Control characters**: Newlines, null bytes, and other control characters pass through `replace` unchanged and can break SQL parsers or inject comment-based payloads.
- **Dialect uncertainty**: DataFusion's SQL dialect behavior around backslash escapes is not guaranteed to match the standard SQL behavior we assume.

**Correct pattern** — use `Expr` values from `lance::deps::datafusion::logical_expr`:

```rust
use lance::deps::datafusion::logical_expr::{col, in_list, lit, Expr};

// Single doc_id equality
let expr = col("doc_id").eq(lit(doc_id));
scanner.filter_expr(expr);

// IN-list
let values: Vec<Expr> = doc_ids.iter().map(|id| lit(id.as_str())).collect();
let expr = in_list(col("doc_id"), values, false);
scanner.filter_expr(expr);

// Combined doc_type + doc_id filter (returns Option<Expr>)
let filter = build_filter_expr(&doc_types, &doc_ids);
if let Some(expr) = filter {
    scanner.filter_expr(expr);
}
```

For deletes, use `DeleteBuilder::from_expr` (not `DeleteBuilder::new`, which takes a SQL string):

```rust
use lance::dataset::write::DeleteBuilder;

let expr = col("doc_id").eq(lit(doc_id));
let result = DeleteBuilder::from_expr(Arc::new(ds.clone()), expr)
    .execute().await?;
ds = result.new_dataset.as_ref().clone();
```

**Helper functions** in `src/store/lance/search.rs`:
- `build_filter_expr(doc_types, doc_ids) -> Option<Expr>` — combined IN-list filter
- `doc_id_eq_expr(doc_id) -> Expr` — single doc_id equality

**Review checklist** for new filter code:
- No `format!("doc_id = '{}'", ...)` or similar string interpolation
- No `scanner.filter(&str)` with user-supplied values
- No `ds.delete(&str)` with user-supplied values
- All filters use `filter_expr(Expr)` or `DeleteBuilder::from_expr`

**Enforcement**: `clippy.toml` declares `Scanner::filter`, `Dataset::delete`, and `DeleteBuilder::new` as `disallowed-methods`. `make clippy` (`-D warnings`) turns any call into a hard build failure.

### Security issue classes to check against

When reviewing or writing code that handles user input, verify resistance against these classes:

**1. SQL / filter injection**
- User-supplied strings (doc_id, doc_type, commit, branch, domain) must never be interpolated into SQL filter strings.
- All filters use DataFusion `Expr` values (`col`, `lit`, `in_list`, `eq`, `not_eq`, `and`) which are type-safe and cannot be injected.
- The `clippy.toml` disallowed-methods lint prevents reintroduction of string-based filter APIs.

**2. Path traversal**
- User-supplied `domain` strings flow into filesystem paths via `dataset_path` → `encode_domain_path`.
- `encode_domain_path` URI-percent-encodes each segment, so `../` and similar traversal sequences are encoded to `%2E%2E%2F` and cannot escape the data directory.
- `parse_domain` validates segment structure (2, 3, or 5 slash-separated segments).
- Never use `domain.replace('/', "_")` or raw string concatenation for filesystem paths.

**3. Integer overflow**
- User-supplied numeric params (`start`, `count`, `k`) are parsed by serde as `i64` or `usize`.
- `validate_pagination` enforces `start >= 0`, `start <= 1_000_000`, `count >= 1`, `count <= 200`.
- Internal arithmetic like `k = (start + count) * 3` uses `saturating_add` / `saturating_mul` to prevent silent overflow.
- Never use plain `+` or `*` on user-derived `usize` values without checked or saturating arithmetic.

**4. UTF-8 validation**
- Push body chunks are validated with `std::str::from_utf8` (fail-loud). Invalid UTF-8 aborts the push and records an error on the task.
- Never use `String::from_utf8_lossy` on request bodies — it silently replaces invalid bytes with U+FFFD, corrupting doc_ids and text content.
- NDJSON lines are parsed with `serde_json::from_str`, which rejects malformed JSON.

**5. Float validation**
- User-supplied floats (`threshold`, `threshold_set`, `threshold_target`) are checked with `is_finite()` to reject NaN and infinity.
- Range checks (`[0.0, 1.0]` for thresholds, `>= 0.0` for distance thresholds) prevent semantic misuse.

**6. Resource exhaustion (DoS)**
- `MAX_RESULT_COUNT` (200) and `MAX_START` (1_000_000) bound pagination cost.
- Embedding batch size is validated at boot (`assert!(embed_batch_size > 0)`).
- Pipeline backpressure: mpsc channels with bounded buffers prevent unbounded memory growth under load.
- Future mitigation: streaming responses for large result sets to avoid buffering entire datasets in memory.

**7. Cross-document contamination**
- A filter for one doc_id must never match other documents. The `Expr`-based filters guarantee this because `col("doc_id").eq(lit(id))` is a strict equality — no wildcard or injection can broaden the match.
- Integration tests in `tests/contract/security-filter-injection.js` verify this invariant with malicious doc_ids containing SQL injection payloads.

**8. Commit ID injection**
- Commit IDs flow into Lance tag names via `encode_commit_tag`, which escapes all non-`[A-Za-z0-9_-]` characters.
- Never use raw commit IDs as tag names or filesystem components.

**9. Header injection**
- The `TerminusDB-Data-Version` header is constructed via `format!("commit:{}", served_commit).parse()` — if the commit contains control characters or newlines, the `HeaderValue::parse` call fails and returns a 500.
- Never use `format!` to build header values without parsing the result through `HeaderValue::from_str`.
