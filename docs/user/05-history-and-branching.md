# 5. History & branching

This is what makes tdb-search more than a vector index: every commit is an independent, reproducible snapshot, and branching shares a parent's vectors instead of recomputing them. This page proves it end-to-end with `curl`.

## Per-commit snapshots are reproducible

Index commit `c1` (from [Quickstart](./02-quickstart.md)), then change one document and index `c2` on top of it.

```bash
# c2: Yoda's description changes; Mon Calamari is unchanged
cat > delta-c2.ndjson <<'EOF'
{"op":"Changed","id":"terminusdb:///star-wars/People/20","string":"The person's name is Yoda. He trained Luke Skywalker on Dagobah."}
EOF

curl -u admin:root -X POST \
  'http://localhost:8080/push?domain=admin/star_wars&branch=main&target_commit=c2&parent_commit=c1' \
  -H 'Content-Type: application/x-ndjson' --data-binary @delta-c2.ndjson
# poll /check until Complete
```

Now search the **same query at each commit**:

```bash
curl -u admin:root 'http://localhost:8080/search?domain=admin/star_wars&commit=c1&q=trained+Luke+on+Dagobah'
curl -u admin:root 'http://localhost:8080/search?domain=admin/star_wars&commit=c2&q=trained+Luke+on+Dagobah'
```

- At `c2`, Yoda ranks for "trained Luke on Dagobah" (the new text).
- At `c1`, it does **not** — `c1` is frozen as it was. The old snapshot is unchanged by later indexing.

Only `People/20` was re-embedded for `c2`; `Species/8` was reused from `c1`, not recomputed.

## Branch-out shares the parent's vectors

Fork a branch at `c1` and add a document on it:

```bash
cat > delta-b1.ndjson <<'EOF'
{"op":"Inserted","id":"terminusdb:///star-wars/People/21","string":"The person's name is Boba Fett, a feared bounty hunter in Mandalorian armour."}
EOF

# branch "experiment" forks from c1; first push on it names c1 as parent
curl -u admin:root -X POST \
  'http://localhost:8080/push?domain=admin/star_wars&branch=experiment&target_commit=b1&parent_commit=c1' \
  -H 'Content-Type: application/x-ndjson' --data-binary @delta-b1.ndjson
```

Now observe the two properties:

```bash
# the branch sees c1's documents (Yoda, Mon Calamari) WITHOUT re-indexing them — block reuse
curl -u admin:root 'http://localhost:8080/search?domain=admin/star_wars&commit=b1&q=wise+old+man'
#   → returns People/20, inherited from c1's shared vectors

# the new document exists only on the branch
curl -u admin:root 'http://localhost:8080/search?domain=admin/star_wars&commit=b1&q=bounty+hunter'
#   → returns People/21

# main is untouched by the branch
curl -u admin:root 'http://localhost:8080/search?domain=admin/star_wars&commit=c2&q=bounty+hunter'
#   → does NOT return People/21
```

The branch was created by sharing `c1`'s stored vectors, not copying them; only `People/21` (the delta) was embedded. Appends on `experiment` never touch `main`.

## Branch from anywhere

A branch can fork from **any** commit, regardless of which branch originally indexed it. The engine resolves a commit's snapshot globally per domain, so `parent_commit=c1` works whether `c1` was indexed on `main` or elsewhere.

## Reassign with no recompute

If a commit changes nothing indexable, point it at an existing snapshot:

```bash
curl -u admin:root -X POST \
  'http://localhost:8080/assign?domain=admin/star_wars&source_commit=c2&target_commit=c3'
# 204 — searching c3 now equals searching c2, with no embedding work
```

## Staleness: searching a not-yet-indexed commit

Indexing is asynchronous, so search can lag the write head. If you search a commit that isn't indexed yet, the engine **serves the nearest indexed ancestor immediately** (never blocks) and tells you what it actually served via the **`TerminusDB-Data-Version`** response header:

```bash
curl -u admin:root -D - 'http://localhost:8080/search?domain=admin/star_wars&commit=c9&q=wise+old+man'
# HTTP/1.1 200 OK
# TerminusDB-Data-Version: commit:c2          ← served from c2, not c9
# [ ... results from c2 ... ]
```

Because the served data-version (`c2`) differs from what you asked for (`c9`), the result is **stale** — the caller compares the two and can push the missing delta to catch up. You can also send the data-version you expect in the same header on the request; the engine treats it as advisory and always reports what it served.

If a branch has **no** indexed ancestor at all, search returns `404` (and that verdict is briefly cached so the history walk isn't repeated).

---

Next: [Operations](./06-operations.md).
