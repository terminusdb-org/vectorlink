# 03 — Indexing & history

### Learning objectives
After this document you will be able to:
- **Explain** how the reference builds one index per `(domain, commit)` and how it reuses a parent commit's vector blocks.
- **Describe** the push-driven indexing flow (`/last-indexed` handshake → `/push`).
- **Map** the reference's commit/parent model onto LanceDB versions, tags, and branches.
- **Predict** what a search at a historical commit should return.

### Prerequisites
[01](./01-system-overview.md), [02](./02-terminusdb-integration.md).

---

## 1. Concept: an index is a parent index plus this commit's deltas

The defining property of the system: indexing commit `C` with parent `P` is **not** a full rebuild. It is "take `P`'s index, add only what changed, save as `C`." The expensive artifacts — the vectors — are **shared** between `P` and `C`. Only changed documents are embedded and stored anew.

This is what makes per-commit indexing affordable and is the behaviour tdb-search must preserve.

---

## 2. Mechanism (reference): two stores, one shared

The reference splits storage into two layers (see `specs/01-reference-architecture.md` for line-level detail):

**Vector store — append-only, domain-global (`vectors.rs`)**
- One `{domain}.vecs` file per *domain* (not per commit).
- Paged into 12 KB blocks (2 vectors per block). An `LRU` arena caches blocks.
- `add_vecs` only appends; each vector gets a permanent integer `vec_id`. Never moved, never deleted → **shareable across every commit and branch of the domain.**

**Index — pointers per commit (`indexer.rs`)**
- One HNSW per `(domain, commit)`: `{domain}@{commit}.hnsw`.
- Graph nodes are just `{doc_id, vec_id}` — pointers into the shared vector store.
- Serialized as JSON; deserialized by re-attaching each `vec_id` to the shared store.

**Parent reuse (`server.rs::load_hnsw_for_indexing`)**
```
if previous P given:  clone P's pointer-graph  (inherits all parent blocks)
else:                 start a fresh empty index
then: embed only changed docs → append new blocks → insert → save {domain}@C
```

**`/assign`** copies a source commit's index to a target commit name with no recomputation — used when a new commit doesn't change anything indexable.

> So today, "use the block of the parent point" literally means: clone the parent's pointer graph and let it keep referencing the shared, append-only vectors; only deltas get new blocks.

---

## 3. The push-driven indexing flow

TerminusDB drives indexing; the indexer reacts to a push:

```
TerminusDB:
  GET /last-indexed?domain=D&branch=B        → indexer's last-indexed commit P
  compute diff(P→C), render changed docs to NDJSON
  POST /push?domain=D&branch=B&target_commit=C&parent_commit=P
       Authorization: Basic admin secret
       Body: NDJSON operation stream

indexer (on /push):
  read the NDJSON stream incrementally (never buffered whole)
  ... process operations in chunks ...
    embed changed docs → append/upsert rows → reuse parent fragments
  on stream end: serialize new version, tag commit:C,
                 record C as last-indexed for (D, B)
  on failure:    fail loudly with status + body (no silent fallback)
```
The indexer never pulls from TerminusDB; the parent commit comes from the `/last-indexed` handshake. A `(domain,commit)` already at HEAD needs no push.

**Post-commit, background, lagging.** Indexing runs *after* the commit exists, in the background — never on the write path. Search is therefore eventually consistent and can lag the write head; a search for a not-yet-indexed commit serves the nearest indexed ancestor with a stale marker (doc 04 §7). Catch-up is **push-driven**: TerminusDB knows HEAD and pushes the missing delta — the indexer never pulls to catch up. The indexer records a durable last-indexed commit per `(domain, branch)`, and after an outage TerminusDB resumes pushing from that point.

---

## 4. 🆕 Mapping history onto LanceDB

tdb-search keeps the *behaviour* of §2–§3 but replaces the storage with LanceDB. The product decision is **linear-per-branch** history (no merge semantics; merges land on the target branch as ordinary linear appends). Only **branch-out** must be supported.

| Reference concept | tdb-search (LanceDB / Lance) |
|-------------------|------------------------------|
| `{domain}.vecs` (shared vectors) | A Lance dataset for the domain; rows carry the embedding |
| `vec_id` (stable, shared) | A Lance row in a fragment; fragments shared across versions/branches. One row **per chunk** of a document (doc 05 §7), keyed `(doc_id, chunk_index)` |
| `{domain}@{commit}.hnsw` (commit index) | A Lance **version**, bound to the commit by a **tag** `commit:<id>` |
| Index `C` from parent `P` | Append only changed docs onto `P`'s version → new version, tag `commit:C`. Unchanged rows' fragments are reused. |
| Branch-out at `P` | A Lance **branch** created from `P`'s version (`lance` core `create_branch`) — **shares P's fragments**, appends go only to the branch |
| `/assign src→tgt` | A tag `commit:tgt` pointing at `commit:src`'s version — no data movement |

### Why block reuse is preserved
A Lance branch forked from a parent version references the parent's data fragments rather than copying them, recording `parentVersion`/`parentBranch`. A search on the branch transparently reads parent fragments + branch-local fragments. This is the storage-layer equivalent of the reference's "clone the pointer graph over a shared vector store."

> **⚠️ Implementation dependency:** branching is **not** in the `lancedb` high-level API — it lives in the lower-level `lance` core crate. Block-sharing branching is validated empirically before being built on. Fallback: one dataset per `(domain, branch)`.

### Commit ↔ version binding, and the global layer index
Lance versions are monotonic `u64`; commits are opaque hashes. **Lance does not know about TerminusDB commits** — it tracks its own version lineage (`parentVersion`/`parentBranch`) and named tags, but the mapping *commit → layer* is the indexer's to maintain.

The indexer keeps a **global layer index keyed by commit id, per domain**:
```
(domain, commit_id) → (lance_branch, lance_version)   + per-branch: indexing-enabled, last-indexed commit
```
After indexing `C`, record `C → (branch, version)`. `/search?commit=C` resolves `C`, checks out that snapshot (MVCC = consistent read; later commits invisible), and queries.

Keying **globally by commit id (not per branch)** makes this a *comprehensive parent resolver*: a branch forked at commit `P` finds `P`'s layer regardless of which branch originally indexed `P`. Block reuse still comes from Lance branch lineage (the fork shares `P`'s fragments); the layer index only records *where* `P`'s layer lives. Backed by Lance tags.

---

## 5. Worked example: index, amend, branch

```
1. first index of admin/star_wars at C0 (no parent)
   TerminusDB pushes every doc as Inserted → POST /push?...&target_commit=C0
   → create dataset, embed all docs, append rows, tag commit:C0
2. edit People/20 → commit C1
   TerminusDB GET /last-indexed → P=C0; diff(C0→C1) = {Changed People/20}
   POST /push?...&target_commit=C1&parent_commit=C0
   → embed only People/20, upsert → new version, tag commit:C1
   → all other People/* rows reused from C0 (no re-embed)
3. branch out at C0 → commit B0 on branch "feature"
   → create_branch("feature", version_of(commit:C0))
   → shares all C0 fragments; appends on feature don't touch main
4. /search?commit=C0  vs  /search?commit=C1
   → C0 sees the old Yoda string; C1 sees the edited one (MVCC snapshots)
```

---

## Check your understanding
1. When indexing `C1` from `C0`, which documents get embedded? *(Only those changed between C0 and C1.)*
2. What does `/assign` avoid doing? *(Recomputing/re-embedding — it just points a new commit at an existing index/version.)*
3. Why does a search at `commit=C0` not see changes made in `C1`? *(Each commit is an isolated snapshot — a Lance version via MVCC, an HNSW file in the reference.)*
4. Where does branching support come from in tdb-search, and what is the risk? *(The `lance` core crate's `create_branch`; it isn't in the high-level lancedb API and is not stability-labelled.)*
