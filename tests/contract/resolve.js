/**
 * Contract test for POST /resolve — batch entity resolution endpoint.
 *
 * Tests:
 *   1. Basic reciprocal NN core grounding (two similar docs → matched).
 *   2. 3-partition output (matched + set_only + target_only).
 *   3. tau > threshold rejected (the silent-recall trap guard).
 *   4. Threshold controls recall: wider threshold recovers more matches.
 *   5. Independent tau: disabling extras produces only core matches.
 *   6. Validation: missing required fields → 400.
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

describe("POST /resolve — batch entity resolution", function () {
  this.timeout(180000)

  const DOMAIN = "admin/resolve_test"
  const BRANCH = "main"

  before(async function () {
    // Clean up any stale state from prior runs (each test must be independently runnable).
    await agent().delete("/domain").query({ domain: DOMAIN }).set("Authorization", authHeader())
    // Index a small corpus with two populations:
    //   SET: Product/A and Product/B (similar products)
    //   TARGET: Catalogue/X and Catalogue/Y (X matches A, Y is unrelated)
    await pushAndWait(DOMAIN, BRANCH, "resolve_c0", [
      {
        op: "Inserted",
        id: "terminusdb:///resolve/Product/A",
        string: "Premium wireless noise-cancelling headphones with 30-hour battery life and active noise reduction technology",
      },
      {
        op: "Inserted",
        id: "terminusdb:///resolve/Product/B",
        string: "The ancient art of Japanese tea ceremony emphasises mindfulness and precise ceremonial movements",
      },
      {
        op: "Inserted",
        id: "terminusdb:///resolve/Catalogue/X",
        string: "High-end wireless headphones featuring noise cancellation and extended battery life for long listening sessions",
      },
      {
        op: "Inserted",
        id: "terminusdb:///resolve/Catalogue/Y",
        string: "Industrial grade concrete mixing equipment for large construction projects and infrastructure development",
      },
    ])
  })

  after(async function () {
    await agent().delete("/domain").query({ domain: DOMAIN }).set("Authorization", authHeader())
  })

  it("resolves matching cross-set records into matched with 3-partition output", async function () {
    const res = await agent()
      .post("/resolve")
      .set("Authorization", authHeader())
      .send({
        domain: DOMAIN,
        commit: "resolve_c0",
        set_doc_types: ["Product"],
        target_doc_types: ["Catalogue"],
        threshold: 0.5,
        tau_one_to_one: 0.45,
        tau_one_to_many: 0.3,
        tau_many_to_one: 0.3,
        k: 5,
      })
      .expect(200)

    // Must have the 3-partition structure.
    expect(res.body).to.have.property("matched").that.is.an("array")
    expect(res.body).to.have.property("set_only").that.is.an("array")
    expect(res.body).to.have.property("target_only").that.is.an("array")
    expect(res.body).to.have.property("stats").that.is.an("object")

    // Product/A should match Catalogue/X (both about wireless headphones).
    const matchedPairs = res.body.matched.map(m => [m.set_id, m.target_id])
    const hasHeadphoneMatch = matchedPairs.some(
      ([s, t]) =>
        s === "terminusdb:///resolve/Product/A" &&
        t === "terminusdb:///resolve/Catalogue/X",
    )
    expect(hasHeadphoneMatch, "Product/A should match Catalogue/X (headphones)").to.equal(true)

    // Stats must report point counts.
    expect(res.body.stats.set_points).to.be.at.least(1)
    expect(res.body.stats.target_points).to.be.at.least(1)
    expect(res.body.stats.elapsed_ms).to.be.a("number")
  })

  it("wider threshold recovers more matches (recall lever)", async function () {
    // Tight threshold: very few or no matches.
    const tight = await agent()
      .post("/resolve")
      .set("Authorization", authHeader())
      .send({
        domain: DOMAIN,
        commit: "resolve_c0",
        set_doc_types: ["Product"],
        target_doc_types: ["Catalogue"],
        threshold: 0.05,
        tau_one_to_one: 0.05,
        k: 5,
      })
      .expect(200)

    // Loose threshold: at least as many matches.
    const loose = await agent()
      .post("/resolve")
      .set("Authorization", authHeader())
      .send({
        domain: DOMAIN,
        commit: "resolve_c0",
        set_doc_types: ["Product"],
        target_doc_types: ["Catalogue"],
        threshold: 0.5,
        tau_one_to_one: 0.45,
        k: 5,
      })
      .expect(200)

    expect(loose.body.matched.length).to.be.at.least(tight.body.matched.length,
      "wider threshold must recover at least as many matches (monotonic recall)")
  })

  it("tau > threshold is rejected (silent-recall trap)", async function () {
    const res = await agent()
      .post("/resolve")
      .set("Authorization", authHeader())
      .send({
        domain: DOMAIN,
        commit: "resolve_c0",
        set_doc_types: ["Product"],
        target_doc_types: ["Catalogue"],
        threshold: 0.3,
        tau_one_to_one: 0.5, // > threshold -> error
        k: 5,
      })

    expect(res.status).to.equal(400)
    expect(res.body.error).to.match(/tau_one_to_one.*threshold/)
  })

  it("tau_one_to_many > threshold is rejected", async function () {
    const res = await agent()
      .post("/resolve")
      .set("Authorization", authHeader())
      .send({
        domain: DOMAIN,
        commit: "resolve_c0",
        set_doc_types: ["Product"],
        target_doc_types: ["Catalogue"],
        threshold: 0.3,
        tau_one_to_one: 0.2,
        tau_one_to_many: 0.4, // > threshold -> error
        k: 5,
      })

    expect(res.status).to.equal(400)
    expect(res.body.error).to.match(/tau_one_to_many.*threshold/)
  })

  it("missing required fields return 400", async function () {
    // Missing domain.
    const r1 = await agent()
      .post("/resolve")
      .set("Authorization", authHeader())
      .send({ commit: "c0", threshold: 0.5, tau_one_to_one: 0.3 })
    expect(r1.status).to.equal(400)

    // Missing commit.
    const r2 = await agent()
      .post("/resolve")
      .set("Authorization", authHeader())
      .send({ domain: DOMAIN, threshold: 0.5, tau_one_to_one: 0.3 })
    expect(r2.status).to.equal(400)

    // Missing threshold.
    const r3 = await agent()
      .post("/resolve")
      .set("Authorization", authHeader())
      .send({ domain: DOMAIN, commit: "c0", tau_one_to_one: 0.3 })
    expect(r3.status).to.equal(400)

    // Missing tau_one_to_one.
    const r4 = await agent()
      .post("/resolve")
      .set("Authorization", authHeader())
      .send({ domain: DOMAIN, commit: "c0", threshold: 0.5 })
    expect(r4.status).to.equal(400)
  })

  it("out-of-range tau values are rejected", async function () {
    const res = await agent()
      .post("/resolve")
      .set("Authorization", authHeader())
      .send({
        domain: DOMAIN,
        commit: "resolve_c0",
        threshold: 0.5,
        tau_one_to_one: 1.5, // out of [0, 1]
        k: 5,
      })
    expect(res.status).to.equal(400)
  })

  it("requires authentication", async function () {
    const res = await agent()
      .post("/resolve")
      .send({
        domain: DOMAIN,
        commit: "resolve_c0",
        threshold: 0.5,
        tau_one_to_one: 0.3,
        k: 5,
      })
    expect(res.status).to.equal(401)
  })
})
