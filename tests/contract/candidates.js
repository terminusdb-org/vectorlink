/**
 * Contract test for POST /candidates — raw bidirectional KNN gather endpoint.
 *
 * Tests:
 *   1. Basic directional KNN gather (set→target and target→set maps).
 *   2. Candidates sorted nearest-first, capped at k.
 *   3. Separate threshold_set and threshold_target control recall per direction.
 *   4. include=embeddings returns embedding vectors per candidate.
 *   5. include=content returns concatenated chunk text per candidate.
 *   6. Validation: missing required fields → 400.
 *   7. Requires authentication.
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

describe("POST /candidates — raw bidirectional KNN gather", function () {
  this.timeout(180000)

  const DOMAIN = "admin/candidates_test"
  const BRANCH = "main"

  before(async function () {
    await agent().delete("/domain").query({ domain: DOMAIN }).set("Authorization", authHeader())
    await pushAndWait(DOMAIN, BRANCH, "cand_c0", [
      {
        op: "Inserted",
        id: "terminusdb:///cand/Product/A",
        string: "Premium wireless noise-cancelling headphones with 30-hour battery life and active noise reduction technology",
      },
      {
        op: "Inserted",
        id: "terminusdb:///cand/Product/B",
        string: "The ancient art of Japanese tea ceremony emphasises mindfulness and precise ceremonial movements",
      },
      {
        op: "Inserted",
        id: "terminusdb:///cand/Catalogue/X",
        string: "High-end wireless headphones featuring noise cancellation and extended battery life for long listening sessions",
      },
      {
        op: "Inserted",
        id: "terminusdb:///cand/Catalogue/Y",
        string: "Industrial grade concrete mixing equipment for large construction projects and infrastructure development",
      },
    ])
  })

  after(async function () {
    await agent().delete("/domain").query({ domain: DOMAIN }).set("Authorization", authHeader())
  })

  it("returns directional KNN maps with correct structure", async function () {
    const res = await agent()
      .post("/candidates")
      .set("Authorization", authHeader())
      .send({
        domain: DOMAIN,
        commit: "cand_c0",
        set_doc_types: ["Product"],
        target_doc_types: ["Catalogue"],
        threshold_set: 0.5,
        threshold_target: 0.5,
        k: 5,
      })
      .expect(200)

    expect(res.body).to.have.property("set_to_target").that.is.an("object")
    expect(res.body).to.have.property("target_to_set").that.is.an("object")
    expect(res.body).to.have.property("stats").that.is.an("object")

    // Stats must report point counts and edge counts.
    expect(res.body.stats.set_points).to.be.at.least(1)
    expect(res.body.stats.target_points).to.be.at.least(1)
    expect(res.body.stats.set_to_target_edges).to.be.a("number")
    expect(res.body.stats.target_to_set_edges).to.be.a("number")
    expect(res.body.stats.elapsed_ms).to.be.a("number")
  })

  it("Product/A has Catalogue/X as nearest candidate (headphones match)", async function () {
    const res = await agent()
      .post("/candidates")
      .set("Authorization", authHeader())
      .send({
        domain: DOMAIN,
        commit: "cand_c0",
        set_doc_types: ["Product"],
        target_doc_types: ["Catalogue"],
        threshold_set: 0.5,
        k: 5,
      })
      .expect(200)

    const setMap = res.body.set_to_target
    const productA = "terminusdb:///cand/Product/A"
    expect(setMap).to.have.property(productA)

    const candidates = setMap[productA]
    expect(candidates).to.be.an("array").with.length.at.least(1)

    // Each candidate has id and distance.
    expect(candidates[0]).to.have.property("id")
    expect(candidates[0]).to.have.property("distance")
    expect(candidates[0].distance).to.be.at.most(0.5)

    // Catalogue/X should be in the candidate list (both about headphones).
    const hasX = candidates.some(c => c.id === "terminusdb:///cand/Catalogue/X")
    expect(hasX, "Product/A should have Catalogue/X as a candidate").to.equal(true)

    // Candidates should be sorted nearest-first.
    for (let i = 1; i < candidates.length; i++) {
      expect(candidates[i].distance).to.be.at.least(candidates[i - 1].distance,
        "candidates must be sorted nearest-first")
    }
  })

  it("k limits the number of candidates per doc", async function () {
    const res = await agent()
      .post("/candidates")
      .set("Authorization", authHeader())
      .send({
        domain: DOMAIN,
        commit: "cand_c0",
        set_doc_types: ["Product"],
        target_doc_types: ["Catalogue"],
        threshold_set: 1.0,
        k: 1,
      })
      .expect(200)

    const setMap = res.body.set_to_target
    for (const docId of Object.keys(setMap)) {
      expect(setMap[docId].length).to.be.at.most(1,
        `k=1 must cap candidates at 1 per doc, got ${setMap[docId].length} for ${docId}`)
    }
  })

  it("separate threshold_set and threshold_target control directional recall", async function () {
    // Tight set threshold, loose target threshold.
    const res = await agent()
      .post("/candidates")
      .set("Authorization", authHeader())
      .send({
        domain: DOMAIN,
        commit: "cand_c0",
        set_doc_types: ["Product"],
        target_doc_types: ["Catalogue"],
        threshold_set: 0.01,
        threshold_target: 0.5,
        k: 5,
      })
      .expect(200)

    // With tight threshold_set, set→target should have fewer edges.
    const setEdges = res.body.stats.set_to_target_edges
    const targetEdges = res.body.stats.target_to_set_edges

    // set→target with tight threshold should have 0 or very few edges.
    expect(setEdges).to.be.at.most(targetEdges,
      "tight threshold_set should produce fewer set→target edges than loose target→set")
  })

  it("include=embeddings returns embedding vectors", async function () {
    const res = await agent()
      .post("/candidates")
      .set("Authorization", authHeader())
      .send({
        domain: DOMAIN,
        commit: "cand_c0",
        set_doc_types: ["Product"],
        target_doc_types: ["Catalogue"],
        threshold_set: 0.5,
        k: 5,
        include: "embeddings",
      })
      .expect(200)

    const setMap = res.body.set_to_target
    for (const docId of Object.keys(setMap)) {
      for (const candidate of setMap[docId]) {
        expect(candidate).to.have.property("embedding")
        expect(candidate.embedding).to.be.an("array").with.length.at.least(1)
      }
    }
  })

  it("include=content returns concatenated chunk text", async function () {
    const res = await agent()
      .post("/candidates")
      .set("Authorization", authHeader())
      .send({
        domain: DOMAIN,
        commit: "cand_c0",
        set_doc_types: ["Product"],
        target_doc_types: ["Catalogue"],
        threshold_set: 0.5,
        k: 5,
        include: "content",
      })
      .expect(200)

    const setMap = res.body.set_to_target
    for (const docId of Object.keys(setMap)) {
      for (const candidate of setMap[docId]) {
        expect(candidate).to.have.property("content")
        expect(candidate.content).to.be.a("string")
      }
    }
  })

  it("default response (no include) has only id and distance", async function () {
    const res = await agent()
      .post("/candidates")
      .set("Authorization", authHeader())
      .send({
        domain: DOMAIN,
        commit: "cand_c0",
        set_doc_types: ["Product"],
        target_doc_types: ["Catalogue"],
        threshold_set: 0.5,
        k: 5,
      })
      .expect(200)

    const setMap = res.body.set_to_target
    for (const docId of Object.keys(setMap)) {
      for (const candidate of setMap[docId]) {
        expect(candidate).to.have.property("id")
        expect(candidate).to.have.property("distance")
        expect(candidate).to.not.have.property("embedding")
        expect(candidate).to.not.have.property("content")
      }
    }
  })

  it("missing required fields return 400", async function () {
    // Missing domain.
    const r1 = await agent()
      .post("/candidates")
      .set("Authorization", authHeader())
      .send({ commit: "c0", threshold_set: 0.5 })
    expect(r1.status).to.equal(400)

    // Missing commit.
    const r2 = await agent()
      .post("/candidates")
      .set("Authorization", authHeader())
      .send({ domain: DOMAIN, threshold_set: 0.5 })
    expect(r2.status).to.equal(400)

    // Missing threshold_set.
    const r3 = await agent()
      .post("/candidates")
      .set("Authorization", authHeader())
      .send({ domain: DOMAIN, commit: "c0" })
    expect(r3.status).to.equal(400)
  })

  it("out-of-range threshold values are rejected", async function () {
    const res = await agent()
      .post("/candidates")
      .set("Authorization", authHeader())
      .send({
        domain: DOMAIN,
        commit: "cand_c0",
        threshold_set: 1.5,
        k: 5,
      })
    expect(res.status).to.equal(400)
  })

  it("requires authentication", async function () {
    const res = await agent()
      .post("/candidates")
      .send({
        domain: DOMAIN,
        commit: "cand_c0",
        threshold_set: 0.5,
        k: 5,
      })
    expect(res.status).to.equal(401)
  })
})
