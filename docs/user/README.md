# tdb-search — user guide

How to run tdb-search and use it as a standalone semantic search engine, driven entirely over HTTP. You do **not** need TerminusDB to follow this guide — every example uses `curl`.

> tdb-search integrates with TerminusDB (which fronts search and drives indexing in production), but the engine is self-contained: it receives pushed text, embeds and stores it versioned-per-commit, and answers search. This guide drives it directly.

## Contents

1. [Concepts](./01-concepts.md) — the model: domains, commits, branches, chunks, search modes.
2. [Quickstart](./02-quickstart.md) — `docker compose up`, index a commit, run a search.
3. [Indexing by push](./03-indexing.md) — `/last-indexed`, the NDJSON `/push` protocol, polling `/check`.
4. [Searching](./04-searching.md) — GET vs POST, the three modes, filters, pagination, distances.
5. [History & branching](./05-history-and-branching.md) — per-commit snapshots, branch-out with block reuse, `/assign`, staleness.
6. [Operations](./06-operations.md) — auth, health/readiness, statistics, configuration.

## The one-paragraph mental model

You **push** the text that changed between two commits; tdb-search embeds it and stores a new searchable **snapshot** bound to the target commit. Each commit is independently searchable and reproducible. Branching from a commit shares the parent's vectors rather than recomputing them. Search names a commit and returns the matching documents, nearest first.

## Conventions in these docs

- Base URL is `http://localhost:8080` (the default).
- Every request carries the admin secret as HTTP Basic auth: `-u admin:root` (change it for any exposed deployment).
- The full machine-readable contract is [`../../openapi.yaml`](../../openapi.yaml); render it with `make docs`.
