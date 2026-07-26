# 04 — Search

### Learning objectives
After this document you will be able to:
- **Use** every search endpoint correctly (`/search`, `/similar`, `/duplicates`).
- **Interpret** the `distance` values returned.
- **Explain** 🆕 full-text and hybrid search and when to use each.

### Prerequisites
[03 — Indexing & history](./03-indexing-and-history.md).

---

## 1. `/search` — semantic search by free text

```
POST /search?domain=D&commit=C[&count=N]
Header: Authorization: Basic <admin secret>   (the only per-request credential)
Body:   raw UTF-8 query text   (NOT JSON)
```
> There is no embedding-key request header: the engine owns the embedding model and calls the provider itself, so the provider key is the engine's own server-side config, never a per-request header. The only per-request credential is the admin secret.
Mechanism: embed the query text → vector search the `(domain,commit)` index → return the `N` nearest (default 10).

Response:
```json
[{"id":"terminusdb:///star-wars/Species/8","distance":0.0939}, ...]
```
Index not found → `404`.

### Worked example (from the blog)
Body `"Who are the squid people"` against indexed Star Wars returns `Species/8` (Mon Calamari) first — even though the embedding text never says "squid." This is the semantic-similarity payoff.

---

## 2. `/similar` — documents like a known document

```
GET /similar?domain=D&commit=C&id=<doc-iri>[&count=N]
```
Mechanism: look up the stored vector for `id`, then search by that vector. Same response shape as `/search`. If the id isn't in the index → error; index missing → `404`.

Use it for "more like this" / recommendations.

---

## 3. `/duplicates` — scoped near-duplicate groups

```
GET /duplicates?domain=D&commit=C[&threshold=T]
              [&doc_type=…&doc_id=…]            # the SET population
              [&target_doc_type=…&target_doc_id=…]  # optional TARGET population
              [&snippet=true][&start=…&count=…]
```
Mechanism: for each indexed point in the **set** population, run ONE ANN nearest-neighbour query whose filter EXCLUDES the point's own document (within-set) or RESTRICTS to the **target** population (cross-set). Because the filter removes the query point's own document, every returned neighbour is a genuine cross-document match — the scan can never be starved by a multi-chunk document's own sibling chunks (the defect that returned `[]` on real corpora with a fixed `k=2`). Matches reduce to document-level groups (distinct ids, best chunk distance, lower id first) below `threshold` (default `0.0`).

- **Scope.** `doc_type`/`doc_id` (repeated) define the set; `target_doc_type`/`target_doc_id` (repeated) define a second population so every group straddles set↔target (cross-catalogue entity resolution). Absent target → within-set dedup.
- **Snippets.** `snippet=true` includes each member's matched chunk text.

Returns groups, sorted nearest-first:
```json
[
  { "group": [ { "id": "id_a" }, { "id": "id_b" } ], "distance": 0.06 }
]
```
With `snippet=true`, each member also carries `"snippet": "…"`. The `group` array is symmetric (lower id first) and extends to clusters of >2 members without a shape change. Use it for entity-resolution / dedup workflows.

---

## 4. Interpreting `distance`

tdb-search reports a **normalized cosine distance in `[0, 1]`**, identical to the reference scale:

| cosine similarity | reported `distance` | meaning |
|-------------------|---------------------|---------|
| `1.0` (same direction) | `0.0` | closest |
| `0.0` (orthogonal) | `0.5` | unrelated |
| `−1.0` (opposite) | `1.0` | farthest |

Smaller is closer. The reference computed this as `clamp01((dot − 1) / −2)` = `(1 − cos_sim) / 2` over L2-normalized vectors.

LanceDB natively returns the *standard* cosine distance `1 − cos_sim` ∈ `[0, 2]`. These two scales differ by **exactly ×2**, so tdb-search converts deterministically:

```
distance = clamp01( lance_cosine_distance / 2 )
```

Vectors are L2-normalized before insertion, so Lance's cosine distance equals `1 − dot`, and the transform yields the `[0,1]` scale exactly. This keeps distances bounded, intuitive, and well-behaved.

---

## 5. 🆕 Full-text and hybrid search

The reference has **vector search only**. tdb-search adds full-text and hybrid through a single endpoint:

```
POST /search?domain=D&commit=C&count=N&mode=vector|fts|hybrid[&doc_type=T]
Body: raw UTF-8 query text   (unchanged from the reference)
```

| Mode | What it does | Backed by |
|------|--------------|-----------|
| **hybrid** (DEFAULT) | Combine vector + FTS, fused with reciprocal-rank fusion — best out-of-the-box relevance | LanceDB `execute_hybrid` + `RRFReranker` |
| **vector** | Semantic nearest-neighbour only — **exact reference behaviour** | LanceDB vector index |
| **fts** | Keyword/full-text over the rendered `content` (filterable by `doc_type`) | LanceDB `Index::FTS` |

- **`hybrid` is the default** when `mode` is omitted — chosen for best results out of the box. Callers wanting the legacy pure-vector behaviour pass `mode=vector`.
- **One query, both sides:** in hybrid the *same* raw query text is embedded for the vector side **and** used as the FTS terms. No new request shape — the body stays a single raw string.
- FTS/hybrid run over **chunk** rows, then dedup to documents (best chunk per `doc_id`), preserving the `[{id,distance}]` response shape.

Why this matters: full-text search catches exact terms, IDs, and rare tokens that embeddings blur; hybrid gives the best of both. The FTS index targets the **rendered embedding string plus doc id/type**, so keyword and semantic search stay aligned to the same text.

> **Determinism note.** Because `hybrid` is the default, the deterministic test battery (doc 06) exercises **all three modes** — vector, fts, and hybrid. RRF fusion is deterministic given deterministic vector + FTS inputs, and FTS over fixed `content` with a pinned tokenizer is deterministic, so hybrid ranking is reproducible bit-for-bit like the others.

---

## 6. The search snapshot is commit-scoped

Every search names a `commit`. tdb-search resolves it to the tagged Lance version and reads that snapshot (doc 03 §4). So search results are **reproducible per commit**: the same query against the same commit returns the same documents, regardless of later indexing. This property is the backbone of the deterministic integration tests (doc 06).

## 7. 🆕 Eventual consistency: search can lag the write head

Indexing is **post-commit and asynchronous** (doc 03 §3) — a commit lands in TerminusDB first, and the engine catches up in the background. So a search may name a commit that **isn't indexed yet** (normal lag, or after an outage).

The behaviour is explicit and standard — **serve from the last known layer immediately, catch up in the background**:

1. **If the branch has any indexed ancestor**: resolve the **nearest indexed ancestor `≤` the requested commit** and **return its results immediately**, with a marker (`indexed_commit`, plus `stale` when it isn't the exact commit) — **never block, never present stale results as current**. The branch then **auto-enrolls** and catch-up is triggered.
2. **If the branch has no indexed ancestor at all**: return **404**, and **cache that verdict** so the (up-to-1000-commit) history walk isn't repeated for an unindexed lineage.
3. **Trigger catch-up** toward the requested commit in the background so subsequent searches converge.

**Enrollment / propagation model.** Bootstrapping a lineage requires **one** explicit index/enable (the first index for a domain/branch). After that, **any descendant branch becomes searchable and auto-enrolls the first time it is searched** — "if any ancestor was ever indexed, all children get indexed when searched."

**404 negative cache (in-memory, time-bound).** A 404 is cached per branch so the (up-to-1000-commit) ancestor walk isn't repeated. It is invalidated two ways: (1) **immediately** when an indexing request targets that branch (direct enablement); (2) by a **TTL (default 1 h)** that backstops the indirect case — a commit indexed on an *ancestor* branch makes the descendant searchable without any request touching it, so the entry simply expires and the next search re-walks and finds it. Ordinary new commits never invalidate it (an unindexed branch's new commits are themselves unindexed). The cache is advisory and process-local; a restart clears it.

**Finding the nearest ancestor fast (the progressive window).** TerminusDB owns the commit DAG, so it hands the indexer the **last 10 ancestor commits** of the requested commit first; the indexer checks its layer index for any of them. If none are indexed, it asks for **more, progressively, up to 1000 at a time**, until it finds an indexed ancestor — or returns 404 if the lineage has none. This keeps the common case to a 10-item check while still resolving deep history.

**Branching from anywhere.** The indexer keeps a **global commit→layer index** (keyed by commit id, per domain), so a branch forked at commit `P` resolves `P`'s layer *regardless of which branch originally indexed `P`*. Lance gives block reuse via branch lineage; commit identity is the indexer's own index (doc 03 §4).

After an outage the indexer walks forward from its durable last-indexed commit per `(domain, branch)`, reusing parent blocks per commit, until it reaches the head. This is why the integration tests include explicit **indexing-disabled (404)**, **lag**, **branch-from-anywhere**, **outage**, and **search-during-catch-up** scenarios (doc 06).

> Determinism (§6) and eventual consistency (§7) are compatible: *once a commit is fully indexed*, search at that commit is exact and reproducible forever. Lag only affects the window before a commit finishes indexing.

---

## Check your understanding
1. How is the `/search` query supplied — JSON or raw body? *(Raw UTF-8 body.)*
2. What does a `distance` of `0` mean? *(Identical direction / closest possible.)*
3. When would you choose `fts` over `vector`? *(Exact terms, identifiers, rare tokens that semantic search blurs.)*
4. Why are search results reproducible for a fixed commit? *(Search reads an immutable per-commit snapshot/version.)*
