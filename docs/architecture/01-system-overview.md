# 01 — System overview

### Learning objectives
After this document you will be able to:
- **Name** every component in the TerminusDB + tdb-search system and state its single responsibility.
- **Trace** the lifecycle of a document from a user edit to it being semantically searchable.
- **Distinguish** what is unchanged from the reference VectorLink versus 🆕 new in tdb-search.

### Prerequisites
The glossary in [README](./README.md). No prior VectorLink knowledge assumed.

---

## 1. The components

```
┌──────────────┐   search request    ┌─────────────────────┐
│   Caller     │────────────────────▶│     TerminusDB      │
│ (or any app) │◀────────────────────│  (graph database +  │
└──────────────┘      results        │   commit history)   │
                                     └─────────┬───────────┘
                                               │
              POST /push  (NDJSON delta) ◀──────┤  "here is what changed
              POST /search (front search) ◀─────┤   from commit P to commit C"
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────┐
│                          tdb-search                           │
│  HTTP API · indexing · search                                 │
│  ┌───────────┐   ┌───────────────┐                            │
│  │  LanceDB  │   │  embed client │──────────▶ ┌──────────────────────┐
│  │  store    │   └───────────────┘   /v1/     │  Embedding provider   │
│  │ (vectors, │                        embed   │  default: local Ollama │
│  │  FTS,     │                                │  sidecar (nomic-embed- │
│  │  versions)│                                │  text-v2-moe)          │
│  └───────────┘                                └───────────────────────┘
└────────────────────────────────────────────────────────────────────────┘
```

| Component | Responsibility | Status |
|-----------|----------------|--------|
| **TerminusDB** | Stores documents and their full commit history; knows the diff between any two commits; renders documents to embedding strings via GraphQL + Handlebars; **renders text and PUSHES deltas** to the indexer; **fronts search** (authorises the caller, then calls the indexer). | Unchanged |
| **tdb-search** | The semantic indexer: HTTP API that **receives pushes, embeds them, stores them versioned, and answers search**. | Reimplemented (this project) |
| **Embedding provider** | Turns text into vectors. | 🆕 configurable; default is a local Ollama sidecar serving a CPU model (nomic-embed-text-v2-moe) |
| **LanceDB store** | Versioned columnar storage of vectors + full-text index. | 🆕 replaces bespoke HNSW + paged vector store |
| **Caller / app** | Configures embedding templates, edits documents, runs searches via TerminusDB. | Unchanged |

---

## 2. Responsibility boundary (who owns what)

A frequent source of confusion: **TerminusDB produces the text to embed; tdb-search produces and stores the vectors.** tdb-search never reads the graph directly and never renders Handlebars — it only consumes the rendered `string` in each operation. This boundary is load-bearing for testing: you can test tdb-search end-to-end by feeding it an operation stream, with or without a real TerminusDB.

---

## 3. The lifecycle of a document (the worked example)

Follow one document — `People/20` (Yoda) — from edit to searchable.

1. **Author edits** the Star Wars database in the dashboard and approves a change request. TerminusDB records a new **commit** `C1` (parent `C0`).
2. **TerminusDB detects the new commit** and drives indexing: it calls the indexer `GET /last-indexed?domain=admin/star_wars&branch=main` to learn the indexer's last-indexed commit (`C0`), computes the diff `C0→C1`, and prepares to push the delta.
3. **Push**: TerminusDB renders the changed documents and POSTs them to tdb-search as an NDJSON stream, `POST /push?domain=admin/star_wars&branch=main&target_commit=C1&parent_commit=C0`:
   ```json
   {"op":"Changed","id":"terminusdb:///star-wars/People/20","string":"The person's name is Yoda. ..."}
   ```
   tdb-search processes the stream incrementally (never buffering it whole).
4. **Embedding**: tdb-search sends each `string` to the embedding provider and receives a vector.
5. **Storage**: tdb-search upserts a row `{doc_id, doc_type, embedding, content}` into the LanceDB table for `admin/star_wars`, producing a new version tagged `commit:C1`. Vectors from `C0` that did not change are **reused, not recomputed** (doc 03).
6. **Completion**: tdb-search finishes the stream and records `C1` as the last-indexed commit for `(admin/star_wars, main)`.
7. **Search**: a caller asks TerminusDB to search; TerminusDB authorises the caller, then calls the indexer `POST /search?domain=admin/star_wars&commit=C1` with body `"Wise old man"`. The indexer embeds the query and returns `[{"id":".../People/20","distance":…}]`.

```
edit → commit C1 → TerminusDB GET /last-indexed → diff(C0→C1) → push NDJSON delta
     → embed changed docs → store new version (reuse C0 blocks) → search at C1 finds Yoda
```

---

## 4. What is unchanged vs 🆕 new

**Unchanged (reproduced exactly):**
- The NDJSON operation line format (`Inserted`/`Changed`/`Deleted`/`Error`), produced by TerminusDB's existing rendering (doc 02).
- The per-commit, parent-reusing indexing model (doc 03).

**Different from the reference (this project's design):**
- The push protocol (TerminusDB → indexer over HTTP) replaces the reference's content-pull mechanism; the indexer never calls back to TerminusDB.
- TerminusDB fronts and authorises search; the indexer is gated by a shared admin secret.

**🆕 New in tdb-search:**
- LanceDB replaces the HNSW file + paged `.vecs` store.
- Configurable embedding provider; default is a **local Ollama sidecar** running a CPU model (no OpenAI key needed).
- Full-text and hybrid search (doc 04).
- Real `Deleted`/`Changed` semantics available (feature-flagged; reference left these unimplemented).
- Branch-out modelled on Lance branches that share parent blocks (doc 03).

---

## Check your understanding
1. If you wanted to test tdb-search without running TerminusDB, what would you substitute, and why is that sufficient? *(Answer: a fake push driver that POSTs an NDJSON operation stream to `/push` — because tdb-search only consumes operations, never the graph.)*
2. Which component renders a document into the text that gets embedded? *(TerminusDB, via GraphQL + Handlebars.)*
3. Name one thing that is reused rather than recomputed when indexing `C1` from `C0`. *(The vectors/blocks of documents unchanged between C0 and C1.)*
