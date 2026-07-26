# 02 — TerminusDB integration

### Learning objectives
After this document you will be able to:
- **Explain** how TerminusDB turns a commit diff into an operation stream.
- **Describe** how embedding strings are produced from documents (GraphQL query + Handlebars template).
- **Reproduce** the exact push HTTP contract (TerminusDB → indexer).
- **Construct** a fake push driver for testing.

### Prerequisites
[01 — System overview](./01-system-overview.md).

---

## 1. Concept: TerminusDB is the source of *what changed*

TerminusDB is a versioned graph database. Every approved change request creates a commit. Crucially, TerminusDB can compute the set of documents that changed between any two commits — this is what makes **incremental** indexing possible: it pushes only the delta, never the whole database.

The rendering and diffing logic lives in the TerminusDB server (Rust `embedding.rs` / `write_op_for`, driven by TerminusDB's commit-diff machinery).

---

## 2. Mechanism: from commit diff to operations

To build the delta from `parent_commit` to `target_commit`, TerminusDB does the following:

1. **Find embeddable types.** For each document class, TerminusDB schema may carry `@metadata.embedding` with a GraphQL `query` and an optional Handlebars `template`. Only classes with an embedding query are indexed.
2. **Enumerate changed documents.**
   - If a `parent_commit` is known: the commit diff yields each changed id and classifies it:
     - present after, present before → `Changed`
     - present after, absent before → `Inserted`
     - absent after → `Deleted`
   - If there is no parent (first index): every document of each embeddable type → `Inserted`.
3. **Render each document to an embedding string.** Via `write_op_for` (`terminusdb-community/src/embedding.rs`):
   - run the type's GraphQL query for that document id,
   - feed the result into the type's Handlebars template (or, if no template, the JSON document),
   - emit one NDJSON line.

   > Note: this rendered string can be **arbitrarily long**. tdb-search chunks it to fit the embedding model's context window (doc 05 §7) so nothing is silently truncated — TerminusDB itself imposes no length limit here.
4. **Push** the lines to the indexer as a chunked NDJSON stream.

### Worked example: the Star Wars `People` embedding template
Schema metadata for the `People` class:
```json
{
  "embedding": {
    "query": "query($id: ID){ People(id : $id) { birth_year, desc, eye_color, gender, hair_colors, height, homeworld { label }, label, mass, skin_colors, species { label } } }",
    "template": "The person's name is {{label}}.{{#if desc}} They are described with the following synopsis: {{#each desc}} *{{this}} {{/each}}.{{/if}}{{#if gender}} Their gender is {{gender}}.{{/if}} ..."
  }
}
```
For `People/20` this renders to:
```
The person's name is Yoda. They are described with the following synopsis: Yoda is a fictional character ... Their gender is male. They have the following hair colours: white. ...
```
That rendered string — not the raw JSON — is what gets embedded. (The blog's insight: embedding the meaningful sentence gives far better semantics than embedding raw JSON.)

> **Boundary reminder:** all of §2 happens *inside TerminusDB*. tdb-search receives only the finished operation lines.

---

## 3. The push contract (TerminusDB → indexer)

TerminusDB drives indexing. First it learns where the indexer is up to, then it pushes the delta:

```
GET  /last-indexed?domain={domain}&branch={branch}
       → { "commit": "<parent_commit>" }   (the indexer's last-indexed commit)

POST /push?domain={domain}&branch={branch}&target_commit={C}&parent_commit={P}
       Authorization: Basic <base64(admin:secret)>
       Body: NDJSON operation stream
```

- `{domain}` — e.g. `admin/star_wars`.
- `{branch}` — e.g. `main`.
- `{parent_commit}` — comes from the `/last-indexed` handshake; TerminusDB computes the diff from there to `{target_commit}`.
- **Auth** — every indexer request is gated by a shared admin secret over **HTTP Basic** (`admin:root` by default). This is authentication only — there is no RBAC at the indexer. TerminusDB fronts search and authorises the caller against its own capability system before calling the indexer.
- **Body** — an NDJSON stream, parsed line-by-line and processed incrementally in chunks (never buffered whole).
- **Failure** → the push fails loudly with the status code and body (no silent fallback).

The indexer makes **no** outbound call to TerminusDB: there is no callback and no content-pull. The parent commit it needs comes entirely from the `/last-indexed` handshake.

---

## 4. The operation stream format

NDJSON; each line is one tagged operation (`#[serde(tag="op")]`). This is the **same** line format as the reference:

```json
{"op":"Inserted","id":"terminusdb:///star-wars/People/20","string":"The person's name is Yoda. ..."}
{"op":"Changed","id":"terminusdb:///star-wars/People/22","string":"The person's name is Boba Fett. ..."}
{"op":"Deleted","id":"terminusdb:///star-wars/People/21"}
{"op":"Error","message":"Failed to retrieve embedding for id ..."}
```

| op | Meaning | Reference behaviour | 🆕 tdb-search |
|----|---------|---------------------|---------------|
| `Inserted` | New document | Embed `string`, insert point | Same |
| `Changed` | Modified document | Treated as insert (duplicate) | May upsert (replace) behind a feature flag |
| `Deleted` | Removed document | Parsed then **dropped** (no-op) | May delete the row behind a feature flag |
| `Error` | Upstream couldn't render | Logged, skipped | Same |

Default tdb-search behaviour must not regress the reference: insert-only semantics remain the default; delete/replace are opt-in.

---

## 5. Building a fake push driver (for tests)

Because the contract is just "POST an NDJSON stream to `/push`," a test double is trivial: a client that POSTs a fixed file of operation lines to `/push?domain=…&branch=…&target_commit=…&parent_commit=…` with the admin secret. This lets you integration-test all of tdb-search's indexing without TerminusDB, with fully controlled input. (The full deterministic stack in doc 06 uses a *real* TerminusDB for end-to-end fidelity, but the fake driver is invaluable for focused tests.)

---

## Check your understanding
1. What two schema artifacts turn a document into its embedding string? *(A GraphQL query and a Handlebars template in `@metadata.embedding`.)*
2. How does TerminusDB decide whether a changed document is `Inserted` vs `Changed`? *(Whether it existed in the `parent_commit`.)*
3. How is a push request authenticated, and where does the parent commit come from? *(HTTP Basic with a shared admin secret; the parent commit comes from the `/last-indexed` handshake.)*
