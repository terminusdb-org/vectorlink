# Vectorlink Integration Tests

Integration tests for the vectorlink semantic indexer running against a live
TerminusDB instance on `localhost:7373` and a vectorlink server on
`localhost:7374`.

## Prerequisites

1. **TerminusDB** with the vectorlink plugin, running on `http://localhost:7373` with default admin/root credentials
2. **Vectorlink server** running on `http://localhost:7374` with content endpoint pointing to TerminusDB:
   ```bash
   cargo run -- serve \
     --content-endpoint http://localhost:7373/api/index \
     --user-forward-header x-terminusdb-user \
     --directory /tmp/vl_index_dir \
     --port 7374
   ```
   The `--user-forward-header` flag is passed for compatibility, but authentication
   is handled by forwarding the `authorization` header from the incoming request to
   TerminusDB. The test client sends `authorization: Basic admin:root`.
3. **Ollama** (optional) running on `http://localhost:11434` with `qwen3-embedding:4b` model pulled

## Quick Start

Use the included restart script to start both TerminusDB and vectorlink with the
 correct configuration:

```bash
bash tests/restart-servers.sh
```

## Running Tests

```bash
cd tests
npm install

# Run all tests (uses OpenAI by default, or Ollama if VECTORLINK_EMBEDDING_PROVIDER=ollama)
npm test

# Run with Ollama embeddings
VECTORLINK_EMBEDDING_PROVIDER=ollama npm test

# Run only the Ollama-specific tests
npm run test:ollama
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TERMINUSDB_BASE_URL` | `http://localhost:7373` | TerminusDB server URL |
| `TERMINUSDB_USER` | `admin` | TerminusDB user |
| `TERMINUSDB_PASSWORD` | `root` | TerminusDB password |
| `TERMINUSDB_ORG` | `admin` | TerminusDB organization |
| `VECTORLINK_BASE_URL` | `http://localhost:7374` | Vectorlink server URL |
| `VECTORLINK_EMBEDDING_PROVIDER` | (unset, uses OpenAI) | Set to `ollama` to use Ollama |
| `VECTORLINK_OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `VECTORLINK_OLLAMA_MODEL` | `qwen3-embedding:4b` | Ollama embedding model |
| `VECTORLINK_OLLAMA_DIMENSIONS` | `1536` | Embedding dimensions |
| `OPENAI_KEY` | (required for OpenAI) | OpenAI API key |

## Test Coverage

### `integration.test.js`
- **Index lifecycle**: start indexing, poll task status, task not found, pending status
- **Search**: text query results, relevance ranking, sort order, count limit, non-existent commit
- **Similar documents**: find by id, self-inclusion, non-existent id
- **Duplicate candidates**: low/high threshold behavior
- **Statistics**: vector store stats endpoint
- **Assign index**: no-op copy between commits, search on assigned index
- **Incremental indexing**: delta indexing with updated, added, and deleted documents
- **Multi-commit**: search across original and updated commit indexes

### `ollama.test.js`
- **Ollama indexing**: index documents with `qwen3-embedding:4b`
- **Ollama search**: relevant results with Ollama embeddings
- **Ollama similar**: similar document lookup
- **Embedding dimensions**: verify 1536-dimensional output
