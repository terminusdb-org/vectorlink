/**
 * Chunking deep-test — through the REAL HTTP pipeline (push → poll → search),
 * against the live engine + real embeddings + real chunker.
 *
 * The previous sole multi-chunk test only asserted `chunk.count >= 2` and that
 * the TAIL was findable. This battery is rigorous and DETERMINISTIC:
 *
 *   1. EXACT chunk count — a doc engineered (in TOKENS) to a known N → count == N.
 *   2. HEAD + MIDDLE + TAIL recall — distinct phrases planted at the start,
 *      a true interior chunk, and the end; each must be retrievable. A dropped
 *      middle chunk fails the test (the old test never exercised the middle).
 *   3. DEDUP-to-one-hit — a query matching content in SEVERAL chunks of one doc
 *      returns exactly ONE hit for that doc_id (chunk→document dedup).
 *   4. 512-TOKEN BOUNDARY crossing — a doc just over the model's single-chunk
 *      budget goes 1→2 chunks with ALL content retrievable (no silent
 *      truncation — RISK-06).
 *   5. `Changed` SHRINKS chunk count — a 5-chunk doc Changed to a 1-chunk doc
 *      leaves no stale orphan chunks findable.
 *
 * ── Fixture token-sizing rationale (why the chunk counts are deterministic) ──
 * The engine sizes chunks with `params_for_nomic(tokenizer, "search_document: ")`
 * (src/chunk/mod.rs): WINDOW=512, minus the document prefix's token cost. With
 * the shipped nomic tokenizer (spikes/tokenizer/tokenizer.json) the prefix costs
 * 4 tokens, so:
 *     max_tokens = 508,  overlap = 508/7 = 72,  step = max_tokens - overlap = 436
 * The filler unit "alpha " encodes to EXACTLY 2 tokens (uniform — no multi-digit
 * subword splits, unlike "word{N} "). Chunk boundaries are [start, start+508),
 * advancing by `step` until the doc end is reached.
 *
 * These exact token totals + chunk counts were measured against the real
 * tokenizer (a temporary probe in src/chunk/mod.rs, since removed) and are
 * reproduced by the engine's own chunker — they are NOT guessed:
 *   - ("alpha "×251) + "endmarker undermk omega" = 508 tokens → 1 chunk (at budget)
 *   - ("alpha "×252) + "endmarker overmk omega"  = 510 tokens → 2 chunks (over budget)
 *   - head + ("alpha "×550) + middle + ("alpha "×550) + tail = 2243 tokens
 *       → EXACTLY 5 chunks; head lands in chunk 0, middle in chunk 2 (a true
 *         interior chunk, not an overlap-shared edge), tail in chunk 4 (last).
 *
 * If the chunker's boundary/overlap math regresses, the exact-count assertions
 * below fail loudly — that is the point.
 */

const { expect } = require("chai")
const { agent, authHeader } = require("../lib/agent")

// ── deterministic fixture constants (see header rationale) ──
const FILLER_UNIT = "alpha " // 2 tokens each, uniform.
const HEAD_PHRASE = "The hidden artefact is the Kyber crystal of Ilum. "
const MIDDLE_PHRASE = "The secret garrison commander is Admiral Thrawn. "
const TAIL_PHRASE = "The final passphrase is xyzzy spoken by Mon Mothma."

// head + 550 filler + middle + 550 filler + tail = 2243 tokens → EXACTLY 5 chunks.
const FIVE_CHUNK_FILLER_REPS = 550
const FIVE_CHUNK_EXPECTED = 5

// ("alpha "×251) + "endmarker undermk omega" = 508 tokens → 1 chunk.
// ("alpha "×252) + "endmarker overmk omega"  = 510 tokens → 2 chunks.
const BOUNDARY_ONE_CHUNK_REPS = 251
const BOUNDARY_TWO_CHUNK_REPS = 252

function buildFiveChunkDoc () {
  const filler = FILLER_UNIT.repeat(FIVE_CHUNK_FILLER_REPS)
  return `${HEAD_PHRASE}${filler}${MIDDLE_PHRASE}${filler}${TAIL_PHRASE}`
}

async function waitForTask (taskId, timeoutMs = 90000) {
  const start = Date.now()
  while (Date.now() - start < timeoutMs) {
    const res = await agent()
      .get("/check")
      .query({ task_id: taskId })
      .set("Authorization", authHeader())
    if (res.status === 200 && res.body.status === "Complete") {
      return res.body
    }
    if (res.status === 500) {
      throw new Error(`task failed: ${res.text}`)
    }
    await new Promise(resolve => setTimeout(resolve, 500))
  }
  throw new Error(`task ${taskId} did not complete within ${timeoutMs}ms`)
}

async function pushAndWait (domain, branch, commit, ops, parentCommit) {
  const body = ops.map(l => JSON.stringify(l)).join("\n")
  const query = { domain, branch, target_commit: commit }
  if (parentCommit) {
    query.parent_commit = parentCommit
  }
  const pushRes = await agent()
    .post("/push")
    .query(query)
    .set("Authorization", authHeader())
    .set("Content-Type", "application/x-ndjson")
    .send(body)
    .expect(200)
  return waitForTask(pushRes.text)
}

async function searchMode (domain, commit, q, mode, extra = {}) {
  const res = await agent()
    .get("/search")
    .query({ domain, commit, q, mode, ...extra })
    .set("Authorization", authHeader())
    .expect(200)
  expect(res.body).to.be.an("array")
  return res.body
}

async function search (domain, commit, q, extra = {}) {
  return searchMode(domain, commit, q, "vector", extra)
}

describe("Chunking deep-test (real embeddings + real chunker)", function () {
  this.timeout(240000) // Long embedding work on CPU across several multi-chunk docs.

  const DOMAIN = "admin/chunk_deeptest"
  const BRANCH = "main"
  const FIVE_CHUNK_ID = "terminusdb:///chunk/Documents/fivechunk"
  const BOUNDARY_ID = "terminusdb:///chunk/Documents/boundary"

  before(async function () {
    // Clean up any stale state from prior runs (each test must be independently runnable).
    await agent().delete("/domain").query({ domain: DOMAIN }).set("Authorization", authHeader())
    // c0: the deterministic 5-chunk document with planted head/middle/tail phrases.
    const result = await pushAndWait(DOMAIN, BRANCH, "c0", [
      { op: "Inserted", id: FIVE_CHUNK_ID, string: buildFiveChunkDoc() },
    ])
    expect(result.status).to.equal("Complete")
    expect(result.indexed_documents).to.equal(1)
  })

  after(async function () {
    await agent().delete("/domain").query({ domain: DOMAIN }).set("Authorization", authHeader())
  })

  describe("1. exact chunk count", function () {
    it(`a token-engineered doc splits into EXACTLY ${FIVE_CHUNK_EXPECTED} chunks (not just >= 2)`, async function () {
      // Query the head phrase so the doc is surfaced; every hit for this doc must
      // report the SAME total chunk count, and it must equal the predicted N.
      const hits = await search(DOMAIN, "c0", "hidden artefact Kyber crystal Ilum")
      const docHits = hits.filter(h => h.id === FIVE_CHUNK_ID)
      expect(docHits.length).to.be.at.least(1, "five-chunk doc must be retrievable")
      for (const hit of docHits) {
        expect(hit.chunk.count).to.equal(
          FIVE_CHUNK_EXPECTED,
          `chunk.count must be exactly ${FIVE_CHUNK_EXPECTED} (boundary/overlap math regression if not)`,
        )
      }
    })
  })

  describe("2. head + MIDDLE + tail recall", function () {
    // Each planted phrase lives in a distinct chunk (head→0, middle→2, tail→4).
    // A dropped or mis-embedded interior chunk makes the middle query fail.
    const cases = [
      // Each marker is a UNIQUE literal term living in a DISTINCT chunk
      // (Kyber→chunk 0/head, Thrawn→chunk 2/interior, xyzzy→chunk 4/tail). We
      // query via FTS — which matches literal text, so a dropped chunk makes its
      // marker unretrievable and fails the test. (Vector mode would return this
      // single-doc commit for ANY query, so it cannot prove per-chunk recall.)
      { label: "HEAD", marker: "Kyber" },
      { label: "MIDDLE", marker: "Thrawn" },
      { label: "TAIL", marker: "xyzzy" },
    ]
    for (const { label, marker } of cases) {
      it(`recalls the ${label} chunk's unique marker (${marker}) via FTS`, async function () {
        const hits = await searchMode(DOMAIN, "c0", marker, "fts", { count: 50 })
        const found = hits.some(h => h.id === FIVE_CHUNK_ID)
        expect(found).to.equal(
          true,
          `${label} marker "${marker}" must be FTS-retrievable — a dropped ${label} chunk would fail this`,
        )
      })
    }

    it("the MIDDLE hit's locator points at an interior chunk (not head, not tail)", async function () {
      const hits = await search(DOMAIN, "c0", "secret garrison commander Admiral Thrawn")
      const docHit = hits.find(h => h.id === FIVE_CHUNK_ID)
      expect(docHit, "middle query must surface the doc").to.not.equal(undefined)
      // The middle phrase was planted in chunk index 2 of 5 (a true interior chunk).
      // location is token_start/doc_token_len, so it must be strictly between the
      // first chunk (location 0) and the last chunk.
      expect(docHit.chunk.index).to.be.greaterThan(
        0, "middle hit must not be the head chunk (index 0)",
      )
      expect(docHit.chunk.index).to.be.lessThan(
        docHit.chunk.count - 1, "middle hit must not be the tail chunk (last index)",
      )
      expect(docHit.chunk.location).to.be.greaterThan(0)
      expect(docHit.chunk.location).to.be.lessThan(1)
    })
  })

  describe("3. dedup to one hit", function () {
    it("a query matching content across SEVERAL chunks returns exactly ONE hit for the doc", async function () {
      // "alpha" is the filler that appears in every chunk of the five-chunk doc.
      // Without chunk→document dedup this would return up to 5 hits for one id.
      const hits = await search(DOMAIN, "c0", "alpha", { count: 50 })
      const docHits = hits.filter(h => h.id === FIVE_CHUNK_ID)
      expect(docHits.length).to.equal(
        1,
        `expected exactly ONE deduped hit for ${FIVE_CHUNK_ID}, got ${docHits.length} (chunk→doc dedup broken)`,
      )
      // The single hit must still carry a valid best-chunk locator.
      const hit = docHits[0]
      expect(hit.chunk.index).to.be.at.least(0)
      expect(hit.chunk.index).to.be.lessThan(hit.chunk.count)
      expect(hit.chunk.count).to.equal(FIVE_CHUNK_EXPECTED)
    })
  })

  describe("4. 512-token boundary crossing (no silent truncation — RISK-06)", function () {
    before(async function () {
      // Two sibling docs on a fresh commit: one just AT the single-chunk budget
      // (508 tokens → 1 chunk) and one just OVER it (510 tokens → 2 chunks). Each
      // has a UNIQUE retrievable phrase at the very end so we prove the tail of
      // the over-budget doc is not silently dropped.
      const justUnder =
        FILLER_UNIT.repeat(BOUNDARY_ONE_CHUNK_REPS) + "endmarker undermk omega"
      const justOver =
        FILLER_UNIT.repeat(BOUNDARY_TWO_CHUNK_REPS) + "endmarker overmk omega"
      const result = await pushAndWait(DOMAIN, BRANCH, "c_boundary", [
        { op: "Inserted", id: `${BOUNDARY_ID}/under`, string: justUnder },
        { op: "Inserted", id: `${BOUNDARY_ID}/over`, string: justOver },
      ], "c0")
      expect(result.status).to.equal("Complete")
      expect(result.indexed_documents).to.equal(2)
    })

    // Each doc carries a UNIQUE literal tail marker (undermk / overmk). We use
    // FTS so the lookup keys on the EXACT marker text — a robust, doc-specific
    // probe rather than a vector nearest-neighbour that could surface the sibling.
    it("the at-budget doc (508 tokens) is a single chunk", async function () {
      const hits = await searchMode(DOMAIN, "c_boundary", "undermk", "fts", { count: 50 })
      const docHit = hits.find(h => h.id === `${BOUNDARY_ID}/under`)
      expect(docHit, "at-budget doc must be retrievable by its marker").to.not.equal(undefined)
      expect(docHit.chunk.count).to.equal(1, "508-token doc must be a single chunk")
    })

    it("the over-budget doc (510 tokens) splits 1→2 chunks", async function () {
      const hits = await searchMode(DOMAIN, "c_boundary", "overmk", "fts", { count: 50 })
      const docHit = hits.find(h => h.id === `${BOUNDARY_ID}/over`)
      expect(docHit, "over-budget doc must be retrievable by its marker").to.not.equal(undefined)
      expect(docHit.chunk.count).to.equal(2, "510-token doc must split into exactly 2 chunks")
    })

    it("the over-budget doc's TAIL content survives (no silent truncation)", async function () {
      // "overmk" sits in the final part of the over-budget doc, past the 508
      // single-chunk budget. FTS matches literal text: if the chunker truncated
      // instead of splitting, this marker would be unindexed and unreachable.
      const hits = await searchMode(DOMAIN, "c_boundary", "overmk", "fts", { count: 50 })
      const found = hits.some(h => h.id === `${BOUNDARY_ID}/over`)
      expect(found).to.equal(
        true,
        "tail marker of the over-budget doc must be FTS-retrievable — silent truncation otherwise",
      )
    })
  })

  describe("5. Changed shrinks chunk count (no orphan chunks)", function () {
    const SHRINK_ID = "terminusdb:///chunk/Documents/shrink"

    it("a 5-chunk doc Changed to a 1-chunk doc leaves no stale orphan chunks findable", async function () {
      // Push the 5-chunk doc with a UNIQUE tail marker we can hunt for later.
      const bigDoc =
        HEAD_PHRASE + FILLER_UNIT.repeat(FIVE_CHUNK_FILLER_REPS) + MIDDLE_PHRASE +
        FILLER_UNIT.repeat(FIVE_CHUNK_FILLER_REPS) +
        "The vanished tail marker is orphanmk quux."
      const big = await pushAndWait(DOMAIN, BRANCH, "shrink_c0", [
        { op: "Inserted", id: SHRINK_ID, string: bigDoc },
      ], "c_boundary")
      expect(big.status).to.equal("Complete")

      // Sanity: the 5-chunk doc is multi-chunk and the orphan marker is findable.
      const before = await search(DOMAIN, "shrink_c0", "vanished tail marker orphanmk quux", { count: 50 })
      const beforeHit = before.find(h => h.id === SHRINK_ID)
      expect(beforeHit, "doc must be findable before Changed").to.not.equal(undefined)
      expect(beforeHit.chunk.count).to.equal(FIVE_CHUNK_EXPECTED, "must start as a 5-chunk doc")

      // Sanity: the unique orphan token is FTS-matchable before the Changed (FTS
      // matches LITERAL text, so it is the honest instrument for "is this text
      // still in the index?" — unlike vector search, which always returns the
      // nearest neighbour regardless of whether the query terms are present).
      const ftsBefore = await searchMode(DOMAIN, "shrink_c0", "orphanmk", "fts", { count: 50 })
      expect(ftsBefore.some(h => h.id === SHRINK_ID), "orphan token must be FTS-findable before Changed")
        .to.equal(true)

      // Changed → a tiny single-chunk doc with totally different content. The
      // orphan tail marker must NOT survive into the new commit's chunk set.
      const small = await pushAndWait(DOMAIN, BRANCH, "shrink_c1", [
        { op: "Changed", id: SHRINK_ID, string: "A brief replacement about a quiet moisture farm on Tatooine." },
      ], "shrink_c0")
      expect(small.status).to.equal("Complete")

      // New content is found and reports a single chunk — the 5-chunk set was
      // fully replaced (a surviving orphan would keep the count at 5).
      const newHits = await search(DOMAIN, "shrink_c1", "moisture farm Tatooine", { count: 50 })
      const newHit = newHits.find(h => h.id === SHRINK_ID)
      expect(newHit, "replacement content must be findable").to.not.equal(undefined)
      expect(newHit.chunk.count).to.equal(1, "Changed must collapse the doc to a single chunk")

      // The orphan token must be GONE at the new commit. FTS (literal-text match)
      // is decisive here: a stale orphan chunk would still contain "orphanmk" and
      // FTS would surface it. Vector mode cannot prove absence (it returns the
      // nearest neighbour for any query), so we assert via FTS.
      const ftsAfter = await searchMode(DOMAIN, "shrink_c1", "orphanmk", "fts", { count: 50 })
      const orphanStillThere = ftsAfter.some(h => h.id === SHRINK_ID)
      expect(orphanStillThere).to.equal(
        false,
        "stale orphan chunk from the pre-Changed 5-chunk doc must NOT remain FTS-findable",
      )
    })
  })
})
