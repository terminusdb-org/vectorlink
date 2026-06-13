/**
 * Indexing pipeline integration tests.
 *
 * Exercises the REAL pipeline: push → check → search, with a live Ollama backend.
 * These tests require the engine to be started with a working embeddings provider.
 * Run via `make test-integration` (which stands up Ollama + the engine).
 */

const { expect } = require("chai")
const { agent, authHeader } = require("../lib/agent")

// Helper: wait for a task to complete (poll /check).
async function waitForTask (taskId, timeoutMs = 30000) {
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

// Helper: push NDJSON and wait for completion.
async function pushAndWait (domain, branch, commit, ndjsonLines) {
  const body = ndjsonLines.map(l => JSON.stringify(l)).join("\n")
  const pushRes = await agent()
    .post("/push")
    .query({ domain, branch, target_commit: commit })
    .set("Authorization", authHeader())
    .set("Content-Type", "application/x-ndjson")
    .send(body)
    .expect(200)

  const taskId = pushRes.text
  expect(taskId).to.be.a("string")
  expect(taskId.length).to.be.greaterThan(0)
  return waitForTask(taskId)
}

describe("Indexing pipeline (real embeddings)", function () {
  this.timeout(120000) // Embedding can be slow on CPU.

  const DOMAIN = "admin/integration_test"
  const BRANCH = "main"

  // Index a batch of documents.
  before(async function () {
    const operations = [
      { op: "Inserted", id: "terminusdb:///itest/People/yoda", string: "Yoda is a wise and ancient Jedi master who teaches the ways of the Force." },
      { op: "Inserted", id: "terminusdb:///itest/People/luke", string: "Luke Skywalker is a young Jedi knight who brings hope to the galaxy." },
      { op: "Inserted", id: "terminusdb:///itest/Species/wookiee", string: "Wookiees are tall furry beings from the planet Kashyyyk known for their strength." },
      { op: "Inserted", id: "terminusdb:///itest/Vehicles/xwing", string: "The X-wing starfighter is a versatile Rebel Alliance attack craft used in many battles." },
      { op: "Inserted", id: "terminusdb:///itest/People/obiwan", string: "Obi-Wan Kenobi is a legendary Jedi master and mentor to both Anakin and Luke Skywalker." },
    ]

    const result = await pushAndWait(DOMAIN, BRANCH, "c0", operations)
    expect(result.status).to.equal("Complete")
    expect(result.indexed_documents).to.be.at.least(4)
  })

  describe("GET /search — vector mode", function () {
    it("returns results with correct shape and top hit for Jedi query", async function () {
      const res = await agent()
        .get("/search")
        .query({
          domain: DOMAIN,
          commit: "c0",
          q: "wise Jedi master",
          mode: "vector",
        })
        .set("Authorization", authHeader())
        .expect(200)

      expect(res.body).to.be.an("array")
      expect(res.body.length).to.be.at.least(1)

      // Verify hit shape.
      const hit = res.body[0]
      expect(hit).to.have.property("id")
      expect(hit).to.have.property("distance")
      expect(hit.distance).to.be.a("number")
      expect(hit.distance).to.be.at.least(0)
      expect(hit.distance).to.be.at.most(1)
      expect(hit).to.have.property("chunk")
      expect(hit.chunk).to.have.property("index")
      expect(hit.chunk).to.have.property("count")
      expect(hit.chunk).to.have.property("token_start")
      expect(hit.chunk).to.have.property("doc_token_len")
      expect(hit.chunk).to.have.property("location")

      // Top hit should be Yoda or Obi-Wan (both are Jedi masters).
      const topIds = res.body.slice(0, 3).map(h => h.id)
      const jediMasters = ["terminusdb:///itest/People/yoda", "terminusdb:///itest/People/obiwan"]
      const hasJediMaster = topIds.some(id => jediMasters.includes(id))
      expect(hasJediMaster).to.equal(true, `expected a Jedi master in top 3, got: ${JSON.stringify(topIds)}`)
    })

    it("vector distances are correctly scaled (fix #2: cosine metric, not L2)", async function () {
      const res = await agent()
        .get("/search")
        .query({
          domain: DOMAIN,
          commit: "c0",
          q: "starfighter battle",
          mode: "vector",
        })
        .set("Authorization", authHeader())
        .expect(200)

      expect(res.body).to.be.an("array")
      expect(res.body.length).to.be.at.least(2)

      // All distances must be strictly between 0 and 1 for distinct documents.
      for (const hit of res.body) {
        expect(hit.distance).to.be.greaterThan(0, `distance for ${hit.id} should be > 0`)
        expect(hit.distance).to.be.lessThan(1, `distance for ${hit.id} should be < 1`)
      }

      // Not all distances should be identical (the old bug clamped everything to 1.0).
      const distances = res.body.map(h => h.distance)
      const allSame = distances.every(d => d === distances[0])
      expect(allSame).to.equal(false, "distances should vary between documents, not all be identical")

      // SCALE ASSERTION: The factor-of-2 bug causes unrelated docs to saturate at ~1.0.
      // With correct cosine metric, the WORST (most distant) hit should be well below 1.0.
      // Real embedding spaces: even unrelated docs rarely exceed ~0.5 cosine distance.
      // We use 0.8 as the threshold — generous enough for real models, but catches the
      // factor-of-2 bug where orthogonal (should be 0.5) was reported as 1.0.
      const worstDistance = Math.max(...distances)
      expect(worstDistance).to.be.lessThan(
        0.8,
        `worst distance ${worstDistance} suggests factor-of-2 scale bug (L2 fed to cosine transform)`,
      )
    })
  })

  describe("GET /search — FTS mode", function () {
    it("finds an exact rare term (Kashyyyk) via FTS with non-zero distance", async function () {
      const res = await agent()
        .get("/search")
        .query({
          domain: DOMAIN,
          commit: "c0",
          q: "Kashyyyk",
          mode: "fts",
        })
        .set("Authorization", authHeader())
        .expect(200)

      expect(res.body).to.be.an("array")
      expect(res.body.length).to.be.at.least(1)
      // The rare-term doc MUST be first (BM25 ranking preserved).
      expect(res.body[0].id).to.equal("terminusdb:///itest/Species/wookiee")
      // Distance must be non-zero (fix #1: was 0.0 when reading _distance instead of _score).
      expect(res.body[0].distance).to.be.a("number")
      expect(res.body[0].distance).to.be.greaterThan(0)
      expect(res.body[0].distance).to.be.lessThan(1)
    })
  })

  describe("GET /search — hybrid mode", function () {
    it("returns results using hybrid (RRF) fusion", async function () {
      const res = await agent()
        .get("/search")
        .query({
          domain: DOMAIN,
          commit: "c0",
          q: "Jedi knight",
          mode: "hybrid",
        })
        .set("Authorization", authHeader())
        .expect(200)

      expect(res.body).to.be.an("array")
      expect(res.body.length).to.be.at.least(1)
      // Luke or Yoda or Obi-Wan should appear.
      const ids = res.body.map(h => h.id)
      const hasJedi = ids.some(id => id.includes("People"))
      expect(hasJedi).to.equal(true, "hybrid should return people docs for 'Jedi knight'")
    })
  })

  describe("GET /search — doc_type filter", function () {
    it("filters results to only Species", async function () {
      const res = await agent()
        .get("/search")
        .query({
          domain: DOMAIN,
          commit: "c0",
          q: "strong beings",
          mode: "vector",
          "doc_type[]": "Species",
        })
        .set("Authorization", authHeader())
        .expect(200)

      expect(res.body).to.be.an("array")
      // All results should be Species type (Wookiee).
      for (const hit of res.body) {
        expect(hit.id).to.include("Species")
      }
    })
  })

  describe("GET /search — doc_id filter", function () {
    it("filters results to specific document IDs", async function () {
      const res = await agent()
        .get("/search")
        .query({
          domain: DOMAIN,
          commit: "c0",
          q: "Jedi",
          mode: "vector",
          "doc_id[]": "terminusdb:///itest/People/yoda",
        })
        .set("Authorization", authHeader())
        .expect(200)

      expect(res.body).to.be.an("array")
      for (const hit of res.body) {
        expect(hit.id).to.equal("terminusdb:///itest/People/yoda")
      }
    })
  })

  describe("GET /search — pagination", function () {
    it("start=0 count=2 returns at most 2 results", async function () {
      const res = await agent()
        .get("/search")
        .query({
          domain: DOMAIN,
          commit: "c0",
          q: "the",
          mode: "vector",
          start: 0,
          count: 2,
        })
        .set("Authorization", authHeader())
        .expect(200)

      expect(res.body).to.be.an("array")
      expect(res.body.length).to.be.at.most(2)
    })

    it("start=2 skips the first two results", async function () {
      const resAll = await agent()
        .get("/search")
        .query({
          domain: DOMAIN,
          commit: "c0",
          q: "the",
          mode: "vector",
          start: 0,
          count: 10,
        })
        .set("Authorization", authHeader())
        .expect(200)

      const resSkip = await agent()
        .get("/search")
        .query({
          domain: DOMAIN,
          commit: "c0",
          q: "the",
          mode: "vector",
          start: 2,
          count: 10,
        })
        .set("Authorization", authHeader())
        .expect(200)

      if (resAll.body.length > 2) {
        expect(resSkip.body[0].id).to.equal(resAll.body[2].id)
      }
    })
  })

  describe("GET /search — snippet field", function () {
    it("snippet=true includes snippet in chunk", async function () {
      const res = await agent()
        .get("/search")
        .query({
          domain: DOMAIN,
          commit: "c0",
          q: "Yoda",
          mode: "vector",
          snippet: true,
        })
        .set("Authorization", authHeader())
        .expect(200)

      expect(res.body).to.be.an("array")
      expect(res.body.length).to.be.at.least(1)
      expect(res.body[0].chunk).to.have.property("snippet")
      expect(res.body[0].chunk.snippet).to.be.a("string")
      expect(res.body[0].chunk.snippet.length).to.be.greaterThan(0)
    })

    it("snippet=false or omitted does not include snippet", async function () {
      const res = await agent()
        .get("/search")
        .query({
          domain: DOMAIN,
          commit: "c0",
          q: "Yoda",
          mode: "vector",
        })
        .set("Authorization", authHeader())
        .expect(200)

      expect(res.body).to.be.an("array")
      if (res.body.length > 0) {
        expect(res.body[0].chunk).to.not.have.property("snippet")
      }
    })
  })

  describe("Chunk locator fields", function () {
    it("chunk metadata is complete and valid", async function () {
      const res = await agent()
        .get("/search")
        .query({
          domain: DOMAIN,
          commit: "c0",
          q: "Jedi",
          mode: "vector",
        })
        .set("Authorization", authHeader())
        .expect(200)

      expect(res.body.length).to.be.at.least(1)
      for (const hit of res.body) {
        expect(hit.chunk.index).to.be.a("number")
        expect(hit.chunk.index).to.be.at.least(0)
        expect(hit.chunk.count).to.be.a("number")
        expect(hit.chunk.count).to.be.at.least(1)
        expect(hit.chunk.token_start).to.be.a("number")
        expect(hit.chunk.token_start).to.be.at.least(0)
        expect(hit.chunk.doc_token_len).to.be.a("number")
        expect(hit.chunk.doc_token_len).to.be.at.least(1)
        expect(hit.chunk.location).to.be.a("number")
        expect(hit.chunk.location).to.be.at.least(0)
        expect(hit.chunk.location).to.be.at.most(1)
        // index < count
        expect(hit.chunk.index).to.be.lessThan(hit.chunk.count)
      }
    })
  })

  describe("Large document (multi-chunk) tail recall", function () {
    before(async function () {
      // Push a large document where the relevant info is near the end.
      const filler = Array.from({ length: 200 }, (_, i) => `Sentence number ${i} contains filler content about nothing in particular.`).join(" ")
      const tail = " The secret password is swordfish and only Admiral Ackbar knows it."
      const largeDoc = filler + tail

      const result = await pushAndWait(DOMAIN, BRANCH, "c1", [
        { op: "Inserted", id: "terminusdb:///itest/Documents/large", string: largeDoc },
      ])
      expect(result.status).to.equal("Complete")
      expect(result.indexed_documents).to.equal(1)
    })

    it("finds content from the tail of a multi-chunk document", async function () {
      const res = await agent()
        .get("/search")
        .query({
          domain: DOMAIN,
          commit: "c1",
          q: "secret password swordfish Admiral Ackbar",
          mode: "vector",
        })
        .set("Authorization", authHeader())
        .expect(200)

      expect(res.body).to.be.an("array")
      expect(res.body.length).to.be.at.least(1)

      const largeDocHit = res.body.find(h => h.id === "terminusdb:///itest/Documents/large")
      expect(largeDocHit).to.not.equal(undefined)
      // Should be a multi-chunk document.
      expect(largeDocHit.chunk.count).to.be.at.least(2)
      // The matching chunk should be from the later part (location > 0).
      expect(largeDocHit.chunk.location).to.be.greaterThan(0)
    })
  })

  describe("BLOCKER-1: FTS completeness at own commit (queue/worker decoupling gap)", function () {
    // This test documents BLOCKER-1: with async indexing, FTS/hybrid searches
    // at a commit's own version don't find the newly-pushed docs because:
    // 1. The commit is tagged to the data-only version (no FTS index yet)
    // 2. Lance FTS requires the INVERTED index to exist (no flat-scan fallback)
    // 3. The background worker builds the index later at a higher version
    // 4. The tag is immutable — it never gains FTS coverage of its own docs
    //
    // This test MUST FAIL until BLOCKER-1 is resolved (option A or B).
    // When the fix is implemented, this test should pass.

    it("FTS finds just-pushed docs at their own commit (currently failing — BLOCKER-1)", async function () {
      // Use a fresh commit (c_blocker1) to isolate from other tests.
      const operations = [
        { op: "Inserted", id: "terminusdb:///itest/Blocker/quux", string: "The planet Zygerria is known for its slave trade and ancient architecture." },
      ]

      const result = await pushAndWait(DOMAIN, BRANCH, "c_blocker1", operations)
      expect(result.status).to.equal("Complete")
      expect(result.indexed_documents).to.equal(1)

      // FTS search at the just-pushed commit should find the doc.
      // BLOCKER-1: this currently returns empty because the INVERTED index
      // doesn't exist at the tagged version.
      const res = await agent()
        .get("/search")
        .query({
          domain: DOMAIN,
          commit: "c_blocker1",
          q: "Zygerria",
          mode: "fts",
        })
        .set("Authorization", authHeader())
        .expect(200)

      expect(res.body).to.be.an("array")
      expect(res.body.length).to.be.at.least(1, "BLOCKER-1: FTS must find just-pushed docs at their own commit")
      expect(res.body[0].id).to.equal("terminusdb:///itest/Blocker/quux")
    })

    it("hybrid finds just-pushed docs at their own commit (currently failing — BLOCKER-1)", async function () {
      // Hybrid = vector + FTS fusion. If FTS is broken, hybrid degrades to vector-only.
      // The doc should still be findable via vector, but we specifically test that
      // the hybrid result includes the FTS-matchable term.
      const res = await agent()
        .get("/search")
        .query({
          domain: DOMAIN,
          commit: "c_blocker1",
          q: "Zygerria slave trade",
          mode: "hybrid",
        })
        .set("Authorization", authHeader())
        .expect(200)

      expect(res.body).to.be.an("array")
      expect(res.body.length).to.be.at.least(1, "BLOCKER-1: hybrid must find just-pushed docs")
      // The Zygerria doc should be in results (via vector at minimum).
      const found = res.body.some(h => h.id === "terminusdb:///itest/Blocker/quux")
      expect(found).to.equal(true, "BLOCKER-1: hybrid should include the just-pushed doc")
    })
  })

  describe("GET /statistics reflects indexed data", function () {
    it("shows non-zero counts after indexing", async function () {
      const res = await agent()
        .get("/statistics")
        .set("Authorization", authHeader())
        .expect(200)

      expect(res.body.domains).to.be.at.least(1)
      expect(res.body.chunks).to.be.at.least(5) // At least 5 docs indexed.
      expect(res.body.indexed_commits).to.be.at.least(1)
    })
  })

  describe("GET /last-indexed after push", function () {
    it("shows the latest commit", async function () {
      const res = await agent()
        .get("/last-indexed")
        .query({ domain: DOMAIN, branch: BRANCH })
        .set("Authorization", authHeader())
        .expect(200)

      expect(res.body.commit).to.not.equal(null)
      expect(res.body.version).to.be.at.least(1)
    })
  })
})
