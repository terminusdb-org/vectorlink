# 3. Indexing by push

You drive indexing. The engine never pulls — it reacts to what you push.

## The handshake

Before pushing, ask where the engine is up to for a `(domain, branch)`:

```
GET /last-indexed?domain=admin/star_wars&branch=main
→ { "branch": "main", "commit": "c1", "version": 7 }     # or commit: null if never indexed
```

You compute the delta from that commit to your new head, render it to NDJSON, and push.

## The push

```
POST /push?domain=admin/star_wars&branch=main&target_commit=<C>&parent_commit=<P>
Authorization: Basic <admin secret>
Content-Type: application/x-ndjson

<NDJSON body — one operation per line>
```

- `parent_commit` is the commit you diffed from (the prior last-indexed). **Omit it** for the first index of a lineage — every document is then an insert.
- The body is streamed and processed incrementally — a large initial index is one request, never held whole in memory.
- The response is an opaque **task id**; indexing runs asynchronously.

## The operation lines

Each line is one tagged operation:

```json
{"op":"Inserted","id":"<iri>","string":"<rendered text to embed>"}
{"op":"Changed","id":"<iri>","string":"<new text>"}
{"op":"Deleted","id":"<iri>"}
{"op":"Error","message":"<upstream could not render this doc>"}
```

| op | Meaning |
|----|---------|
| `Inserted` | New document — embed and store its chunks. |
| `Changed` | Modified document — **replaces its whole chunk set** (no stale chunks remain). |
| `Deleted` | Removed document — **all its chunks are deleted**. |
| `Error` | The upstream couldn't render one document — that document is **skipped and recorded**, never silently dropped; indexing continues. |

The `string` may be arbitrarily long; the engine chunks it to fit the model window — nothing is silently truncated.

## Polling completion

```
GET /check?task_id=<id>
→ { "status": "Pending",  "percentage": 42.5 }
→ { "status": "Complete", "indexed_documents": 1284, "skipped": [ {"id":"...","message":"..."} ] }
```

- `skipped` lists any `Error` operations — visible, never hidden.
- A **systemic** failure (malformed stream, embedding backend down, store write error) fails the whole task with a `500` and the cause in the body — a partial index is never committed and never reported as success.

## Concurrency

A push already in progress for the same `(domain, branch)` is rejected with `409` — it is not started twice.

## Reassigning a commit (no recompute)

When a new commit changes nothing indexable, bind it to an existing snapshot instead of re-indexing:

```
POST /assign?domain=admin/star_wars&source_commit=<C>&target_commit=<C2>
→ 204
```

This is a pointer operation — no embedding, no data movement. (It mutates state, so it is `POST`.)

---

Next: [Searching](./04-searching.md).
