/**
 * Vectorlink parity tests — scenarios ported from the vectorlink integration
 * test suite (tests/test/integration.test.js) that were not already covered
 * by existing tdb-search contract tests.
 *
 * Ported gaps:
 *   1. Search results are sorted by distance ascending (closest first).
 *   2. /similar with a non-existent document ID against an indexed commit
 *      returns an error (404), not an empty array or 200.
 *   3. Top-1 relevance: the single most relevant document is the first hit
 *      for a query that strongly matches one document.
 *
 * These tests use the same push→poll→search pipeline as the other contract
 * tests and require a live engine with a working embeddings provider.
 */

const { expect } = require("chai")
const { agent, authHeader } = require("../lib/agent")

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

describe("Vectorlink parity — ported scenarios", function () {
  this.timeout(180000)

  const DOMAIN = "admin/vl_parity"
  const BRANCH = "main"

  before(async function () {
    await agent().delete("/domain").query({ domain: DOMAIN }).set("Authorization", authHeader())

    await pushAndWait(DOMAIN, BRANCH, "vl_c0", [
      { op: "Inserted", id: "terminusdb:///vl/Doc/red", string: "The red fox jumps over the lazy dog near the riverbank at sunset" },
      { op: "Inserted", id: "terminusdb:///vl/Doc/blue", string: "The ocean is deep blue and full of fish swimming in the coral reef" },
      { op: "Inserted", id: "terminusdb:///vl/Doc/green", string: "The forest is lush green with tall trees and singing birds in spring" },
      { op: "Inserted", id: "terminusdb:///vl/Doc/yellow", string: "The sun shines bright yellow over the desert sand dunes at noon" },
      { op: "Inserted", id: "terminusdb:///vl/Doc/purple", string: "The lavender fields stretch across the hills in a sea of purple flowers" },
    ])
  })

  after(async function () {
    await agent().delete("/domain").query({ domain: DOMAIN }).set("Authorization", authHeader())
  })

  describe("Search result ordering", function () {
    it("search results are sorted by distance ascending (closest first)", async function () {
      const res = await agent()
        .get("/search")
        .query({ domain: DOMAIN, commit: "vl_c0", q: "forest trees green birds", mode: "vector", count: 5 })
        .set("Authorization", authHeader())
        .expect(200)

      expect(res.body).to.be.an("array")
      expect(res.body.length).to.be.greaterThan(1, "need at least 2 results to verify sort order")
      for (let i = 1; i < res.body.length; i++) {
        expect(res.body[i].distance).to.be.at.least(
          res.body[i - 1].distance,
          `result ${i} distance ${res.body[i].distance} must be >= result ${i - 1} distance ${res.body[i - 1].distance}`,
        )
      }
    })

    it("top-1 result is the most relevant document for a strongly matching query", async function () {
      const res = await agent()
        .get("/search")
        .query({ domain: DOMAIN, commit: "vl_c0", q: "ocean deep blue fish coral reef", mode: "vector", count: 5 })
        .set("Authorization", authHeader())
        .expect(200)

      expect(res.body).to.be.an("array")
      expect(res.body.length).to.be.greaterThan(0)
      expect(res.body[0].id).to.include("blue",
        `expected the blue/ocean doc as top hit, got ${res.body[0].id}`)
    })
  })

  describe("/similar with non-existent document ID", function () {
    it("returns 404 for a non-existent document ID against an indexed commit", async function () {
      const res = await agent()
        .get("/similar")
        .query({ domain: DOMAIN, commit: "vl_c0", id: "terminusdb:///vl/Doc/nonexistent", count: 3 })
        .set("Authorization", authHeader())

      expect(res.status).to.be.oneOf([400, 404],
        `expected 400 or 404 for non-existent doc ID, got ${res.status}: ${res.text}`)
    })
  })

  describe("Count parameter limits results", function () {
    it("count=2 returns at most 2 results", async function () {
      const res = await agent()
        .get("/search")
        .query({ domain: DOMAIN, commit: "vl_c0", q: "colors nature", mode: "vector", count: 2 })
        .set("Authorization", authHeader())
        .expect(200)

      expect(res.body).to.be.an("array")
      expect(res.body.length).to.be.at.most(2)
    })
  })

  describe("Incremental indexing with Changed op (delta from parent)", function () {
    it("changed content is findable at the new commit but not the old", async function () {
      await pushAndWait(DOMAIN, BRANCH, "vl_c1", [
        { op: "Changed", id: "terminusdb:///vl/Doc/red", string: "The crimson fox leaps across the sleeping hound by the riverside at dusk" },
      ], "vl_c0")

      const newFts = await agent()
        .get("/search")
        .query({ domain: DOMAIN, commit: "vl_c1", q: "crimson sleeping hound", mode: "fts", count: 10 })
        .set("Authorization", authHeader())
        .expect(200)
      expect(newFts.body.some(h => h.id === "terminusdb:///vl/Doc/red"),
        "changed content must be findable at vl_c1").to.equal(true)

      const oldFts = await agent()
        .get("/search")
        .query({ domain: DOMAIN, commit: "vl_c1", q: "lazy dog riverbank sunset", mode: "fts", count: 10 })
        .set("Authorization", authHeader())
        .expect(200)
      expect(oldFts.body.some(h => h.id === "terminusdb:///vl/Doc/red"),
        "old content must be gone at vl_c1 after Changed").to.equal(false)
    })

    it("original commit index is preserved after incremental update", async function () {
      const res = await agent()
        .get("/search")
        .query({ domain: DOMAIN, commit: "vl_c0", q: "red fox jumps lazy dog", mode: "fts", count: 10 })
        .set("Authorization", authHeader())
        .expect(200)

      expect(res.body.some(h => h.id === "terminusdb:///vl/Doc/red"),
        "original content must still be findable at vl_c0").to.equal(true)
    })
  })

  describe("Delete document and reindex", function () {
    it("deleted document is absent from search at the new commit", async function () {
      await pushAndWait(DOMAIN, BRANCH, "vl_c2", [
        { op: "Deleted", id: "terminusdb:///vl/Doc/yellow" },
      ], "vl_c1")

      const res = await agent()
        .get("/search")
        .query({ domain: DOMAIN, commit: "vl_c2", q: "sun yellow desert sand dunes", mode: "fts", count: 10 })
        .set("Authorization", authHeader())
        .expect(200)

      expect(res.body.some(h => h.id === "terminusdb:///vl/Doc/yellow"),
        "deleted doc must be absent at vl_c2").to.equal(false)
    })

    it("deleted document is still findable at the older commit (snapshot isolation)", async function () {
      const res = await agent()
        .get("/search")
        .query({ domain: DOMAIN, commit: "vl_c0", q: "sun yellow desert sand dunes", mode: "fts", count: 10 })
        .set("Authorization", authHeader())
        .expect(200)

      expect(res.body.some(h => h.id === "terminusdb:///vl/Doc/yellow"),
        "deleted doc must still be findable at vl_c0 (snapshot isolation)").to.equal(true)
    })
  })

  describe("/assign creates a searchable pointer to an existing index", function () {
    it("assign makes the target commit searchable identically to the source", async function () {
      await agent()
        .post("/assign")
        .query({ domain: DOMAIN, source_commit: "vl_c0", target_commit: "vl_assigned" })
        .set("Authorization", authHeader())
        .expect(204)

      const src = await agent()
        .get("/search")
        .query({ domain: DOMAIN, commit: "vl_c0", q: "ocean blue fish", mode: "vector", count: 5 })
        .set("Authorization", authHeader())
        .expect(200)
      const tgt = await agent()
        .get("/search")
        .query({ domain: DOMAIN, commit: "vl_assigned", q: "ocean blue fish", mode: "vector", count: 5 })
        .set("Authorization", authHeader())
        .expect(200)

      const srcIds = src.body.map(h => h.id).sort()
      const tgtIds = tgt.body.map(h => h.id).sort()
      expect(tgtIds).to.deep.equal(srcIds,
        "assigned commit must return the same results as the source")
    })
  })
})
