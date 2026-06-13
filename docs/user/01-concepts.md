# 1. Concepts

A small vocabulary covers the whole system.

## Domain — what you address

A **domain** is a database, addressed as a graphspec. The repository segment is always part of the address and defaults to `local`:

| You write | It means |
|-----------|----------|
| `org/db` | `org/db/local/branch/main` |
| `org/db/<repo>` | `org/db/<repo>/branch/main` |
| `org/db/<repo>/branch/<branch>` | that branch |
| `org/db/<repo>/commit/<commit>` | that commit's snapshot, directly |

A branch names a moving line of history; a **commit** names one fixed point on it. The engine never guesses "latest" — the snapshot you search is always an explicit commit.

## Commit — a searchable snapshot

Each indexed **commit** is an independent, immutable snapshot. Searching `commit=C1` always returns the same results regardless of later indexing — this reproducibility is the backbone of the system. Indexing commit `C1` from its parent `C0` only processes what changed; everything unchanged is reused.

## Branch — a line of history

History is **linear per branch**. Branching out at a commit forks a new line that **shares the parent's stored vectors** (no recompute, no copy) and only adds what changes on the branch. The engine tracks branches; the upstream system (TerminusDB) owns branch heads and any merge logic.

## Document and chunk

You index **documents** (each identified by a full IRI, e.g. `terminusdb:///star-wars/People/20`). A long document is split into **chunks** that fit the embedding model's window, with overlap so nothing is lost at boundaries — each chunk is embedded separately. Search runs over chunks but **deduplicates back to documents**: you always get documents back, never chunk fragments. Each result still tells you *which* chunk matched and roughly *where* in the document it is (`chunk.index`/`count`/`location`), so you can jump to the passage — and optionally the chunk's text via `snippet=true`.

## Search modes

One search endpoint, three modes:

| Mode | What it does | Use when |
|------|--------------|----------|
| **hybrid** (default) | vector + full-text, fused | best general relevance |
| **vector** | semantic nearest-neighbour | pure "find similar meaning" |
| **fts** | keyword / full-text | exact terms, identifiers, rare tokens |

## Distance

Results are ranked by **distance** in `[0, 1]`: `0` is identical, `0.5` unrelated, `1` opposite. Smaller is closer.

## Embedding

The engine **owns its embedding model** and runs it locally (a CPU model by default), so the stack works offline. You never pass an embedding key on a request — the model is the engine's own configuration. The only per-request credential is the admin secret.

---

Next: [Quickstart](./02-quickstart.md).
