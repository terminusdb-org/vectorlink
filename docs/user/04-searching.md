# 4. Searching

One endpoint, two interfaces (GET and POST), three modes.

## GET vs POST

- **`GET /search`** — read-only and cacheable. The query text is `q`. All parameters are query parameters. Best for simple, link-safe searches.
- **`POST /search`** — a structured **JSON body** (better for long queries or programmatic composition).

Every parameter may be given as a query parameter **or** in the JSON body. **The JSON body wins**: if a field is present in the body, the same-named query parameter is ignored (no merge). This lets you set defaults in the URL and override them in the body.

```bash
# GET
curl -u admin:root 'http://localhost:8080/search?domain=admin/star_wars&commit=c1&q=wise+old+man&mode=hybrid'

# POST
curl -u admin:root -X POST http://localhost:8080/search \
  -H 'Content-Type: application/json' \
  -d '{"domain":"admin/star_wars","commit":"c1","q":"wise old man","mode":"hybrid","count":5}'
```

`domain`, `commit`, and `q` are required (in the body or the query); missing any gives `400`.

## Parameters

| Param | Default | Meaning |
|-------|---------|---------|
| `domain` | — | The database (graphspec; see [Concepts](./01-concepts.md)). |
| `commit` | — | The snapshot to search. |
| `q` | — | The query text. |
| `mode` | `hybrid` | `vector` \| `fts` \| `hybrid`. |
| `start` | `0` | Zero-based offset of the first result (pagination). |
| `count` | `50` | Page size. |
| `doc_type` | — | Restrict to these types. |
| `doc_id` | — | Restrict to these document IRIs. |
| `snippet` | `false` | Include the matched chunk's text in each hit (`chunk.snippet`). |

## Filters

`doc_type` and `doc_id` restrict the result set. They **AND** together (a hit must match both sets) and **OR** within a set.

- As query parameters, repeat them: `?doc_type=People&doc_type=Species`.
- In JSON, use arrays: `"doc_type": ["People", "Species"]`.

```bash
curl -u admin:root -X POST http://localhost:8080/search \
  -H 'Content-Type: application/json' \
  -d '{"domain":"admin/star_wars","commit":"c1","q":"engineer","doc_type":["Species"]}'
```

## Pagination

`start` + `count` page the ranked results. To walk a result set: `start=0&count=50`, then `start=50&count=50`, and so on.

## Modes in practice

- **hybrid** (default): combines meaning and keywords — the best general default.
- **vector**: pure semantic similarity — finds related meaning even with no shared words.
- **fts**: exact keywords, identifiers, rare tokens that embeddings blur.

```bash
curl -u admin:root 'http://localhost:8080/search?domain=admin/star_wars&commit=c1&q=Mon+Calamari&mode=fts'
```

## Reading the response

```json
[
  { "id": "terminusdb:///star-wars/People/20", "distance": 0.0939,
    "chunk": { "index": 0, "count": 1, "location": 0.0 } },
  { "id": "terminusdb:///star-wars/Species/8", "distance": 0.1421,
    "chunk": { "index": 3, "count": 12, "location": 0.27 } }
]
```

- Nearest first; **distance** in `[0,1]` (0 identical, 0.5 unrelated, 1 opposite) — it is the distance of the document's **best-matching chunk**.
- One hit per document — chunk fragments are never separate rows.
- **`chunk`** tells you *where* in the document the match was, so you can jump to it:
  - `index` — which chunk matched (0-based).
  - `count` — how many chunks the document was split into (`1` if it fit in one).
  - `location` — approximate fractional position of that chunk's start, `0.0` (beginning) … `1.0` (end). Multiply by 100 for a percentage — e.g. `0.27` ≈ 27% of the way through. Derived from token offsets (accounts for overlap), so it is approximate, not a character index.
- An **empty array means genuinely no match** — never an error in disguise. Errors are status codes (see [Operations](./06-operations.md)).

### Getting the matched text

Add `snippet=true` to include the matched chunk's text as `chunk.snippet` (omitted by default to keep responses small):

```bash
curl -u admin:root 'http://localhost:8080/search?domain=admin/star_wars&commit=c1&q=wise+old+man&snippet=true'
```
```json
[ { "id": "terminusdb:///star-wars/People/20", "distance": 0.0939,
    "chunk": { "index": 0, "count": 1, "location": 0.0,
               "snippet": "The person's name is Yoda. A wise old Jedi master ..." } } ]
```

## Similar and duplicates

```bash
# more like a known document
curl -u admin:root 'http://localhost:8080/similar?domain=admin/star_wars&commit=c1&id=terminusdb:///star-wars/People/20'

# corpus-wide near-duplicate pairs (bounded by threshold + pagination)
curl -u admin:root 'http://localhost:8080/duplicates?domain=admin/star_wars&commit=c1&threshold=0.05'
```

`/duplicates` is always bounded; it never runs an unbounded all-pairs scan.

---

Next: [History & branching](./05-history-and-branching.md).
