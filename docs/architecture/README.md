# tdb-search Architecture Reference

This folder is the **teaching reference** for how TerminusDB and `tdb-search` work together. It is written to two audiences at once:

1. **Implementers** building `tdb-search` — it describes the existing VectorLink behaviour precisely enough to reproduce it, and flags every place `tdb-search` deliberately diverges.
2. **Testers** — it explains the system well enough to know *what correct looks like* and how to verify it (culminating in the integration-test strategy, doc 06).

> **Ground truth:** unless a section is marked **🆕 New in tdb-search**, it is a direct description of how the reference implementation (`terminusdb-semantic-indexer`, the source of the `terminusdb/vectorlink` image) works today. New additions are always called out explicitly.

---

## How these docs are structured (instructional design)

Each document follows the same learning-oriented shape so you can read actively rather than skim:

- **Learning objectives** — what you will be able to *do* after reading (Bloom: understand → apply → analyse).
- **Prerequisites** — what to read first.
- **Concept → mechanism → worked example** — progressive disclosure: the idea, then how it actually works, then a concrete trace through real data (the Star Wars dataset, as in the original VectorLink blog).
- **🆕 New in tdb-search** callouts — divergences from the reference.
- **Check your understanding** — a few questions whose answers are recoverable from the text; these double as the seed for test cases.

This mirrors Gagné's events of instruction (gain attention → state objective → recall prior → present material → worked example → elicit performance → assess) applied to technical reference material.

---

## Reading path

| # | Document | You will understand… |
|---|----------|----------------------|
| 01 | [System overview](./01-system-overview.md) | The whole system: every component, who calls whom, and the lifecycle of a document from edit to searchable. |
| 02 | [TerminusDB integration](./02-terminusdb-integration.md) | How commits, the change-request flow, GraphQL+Handlebars embedding strings, and the push protocol deliver the operation stream. |
| 03 | [Indexing & history](./03-indexing-and-history.md) | How an index is built per commit, how the parent commit reuses parent blocks, and how branch-out is modelled. |
| 04 | [Search](./04-search.md) | `/search`, `/similar`, `/duplicates`, and 🆕 full-text & hybrid search. |
| 05 | [Embeddings & determinism](./05-embeddings-and-determinism.md) | The configurable embedding provider, the default local model, and exactly what makes embedding output reproducible. |
| 06 | [Integration test strategy](./06-integration-test-strategy.md) | How to integration-test the full stack from `docker-compose.yml` with deterministic, repeatable assertions. |

Start at 01 and read in order; later docs assume the earlier ones.

---

## The one-sentence mental model

> TerminusDB knows *what changed between two commits*; `tdb-search` turns those changes into vectors and stores them in a versioned LanceDB table so that any commit can be searched semantically — reusing a parent commit's vectors instead of recomputing them.

---

## Glossary (used throughout)

| Term | Meaning |
|------|---------|
| **Domain** | A TerminusDB database, addressed as `org/db` (e.g. `admin/star_wars`). The unit of vector isolation. |
| **Commit** | An opaque TerminusDB commit hash. Each indexed commit is independently searchable. |
| **Operation** | A single change to index: `Inserted`, `Changed`, `Deleted`, or `Error`, streamed as JSON. |
| **Embedding string** | The natural-language text rendered from a document via a GraphQL query + Handlebars template, then embedded. |
| **Index** | The searchable structure for one `(domain, commit)`. Reference: an HNSW file. tdb-search: a LanceDB table version. |
| **Block / fragment** | The physical unit of stored vectors. Reused across commits/branches rather than copied. |
| **Provider** | The embedding backend (local model, OpenAI, or generic HTTP). |
