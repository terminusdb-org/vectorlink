/**
 * no-op/empty commits must tag + advance last_indexed.
 *
 * Exercises the REAL pipeline with commits that produce no indexable rows:
 *   (a) A commit of only Operation::Error
 *   (b) An empty-operations commit
 * Both must tag + advance last_indexed, and a subsequent normal commit must
 * index correctly (prove no stall).
 *
 * Requires the live engine + Ollama backend (run via `make test-integration`).
 */

const { expect } = require("chai")
const { agent, authHeader } = require("../lib/agent")

// Poll /check until the task reaches a terminal state.
async function waitForTerminal (taskId, timeoutMs = 60000) {
  const start = Date.now()
  while (Date.now() - start < timeoutMs) {
    const res = await agent()
      .get("/check")
      .query({ task_id: taskId })
      .set("Authorization", authHeader())
    if (res.status === 200 && res.body.status === "Complete") {
      return { status: "Complete", body: res.body }
    }
    if (res.status === 200 && res.body.status === "Error") {
      return { status: "Error", body: res.body }
    }
    if (res.status === 500) {
      return { status: "Error", body: res.text }
    }
    await new Promise(resolve => setTimeout(resolve, 300))
  }
  throw new Error(`task ${taskId} did not reach a terminal state within ${timeoutMs}ms`)
}

// Push NDJSON and wait for completion. Returns the task result body.
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
  const result = await waitForTerminal(taskId)
  expect(result.status).to.equal("Complete", `task failed: ${JSON.stringify(result.body)}`)
  return result.body
}

// Get last-indexed for a domain/branch.
async function getLastIndexed (domain, branch) {
  const res = await agent()
    .get("/last-indexed")
    .query({ domain, branch })
    .set("Authorization", authHeader())
    .expect(200)
  return res.body
}

describe("RISK-26: no-op/empty commits must tag + advance last_indexed", function () {
  this.timeout(120000) // Embedding can be slow on CPU.

  const DOMAIN = "admin/risk26_integration"
  const BRANCH = "main"

  before(async function () {
    // Clean up stale state from prior runs.
    await agent().delete("/domain").query({ domain: DOMAIN }).set("Authorization", authHeader())
  })

  after(async function () {
    await agent().delete("/domain").query({ domain: DOMAIN }).set("Authorization", authHeader())
  })

  it("(a) all-error commit advances last_indexed past the empty commit", async function () {
    // Step 1: seed with a real document so the domain has data.
    const seedOps = [
      { op: "Inserted", id: "terminusdb:///risk26/Doc/seed", string: "This is a seed document for testing RISK-26 catch-up stall behaviour." },
    ]
    const seedResult = await pushAndWait(DOMAIN, BRANCH, "c0_seed", seedOps)
    expect(seedResult.indexed_documents).to.equal(1)

    // Verify last-indexed is at c0_seed.
    const liAfterSeed = await getLastIndexed(DOMAIN, BRANCH)
    expect(liAfterSeed.commit).to.equal("c0_seed")

    // Step 2: push an all-error commit.
    const errorOps = [
      { op: "Error", message: "render failed: doc/broken1" },
      { op: "Error", message: "render failed: doc/broken2" },
    ]
    const errorResult = await pushAndWait(DOMAIN, BRANCH, "c1_all_error", errorOps)
    expect(errorResult.indexed_documents).to.equal(0)
    expect(errorResult.skipped).to.be.an("array").with.lengthOf(2)

    // ASSERT: last_indexed advanced past the all-error commit.
    const liAfterError = await getLastIndexed(DOMAIN, BRANCH)
    expect(liAfterError.commit).to.equal(
      "c1_all_error",
      "RISK-26: last_indexed MUST advance past an all-error commit",
    )
  })

  it("(b) empty-operations commit advances last_indexed", async function () {
    // Push an empty commit (zero operations, completely empty NDJSON body).
    const emptyResult = await pushAndWait(DOMAIN, BRANCH, "c2_empty", [])
    expect(emptyResult.indexed_documents).to.equal(0)

    // ASSERT: last_indexed advanced.
    const liAfterEmpty = await getLastIndexed(DOMAIN, BRANCH)
    expect(liAfterEmpty.commit).to.equal(
      "c2_empty",
      "RISK-26: last_indexed MUST advance past an empty commit",
    )
  })

  it("(c) normal commit after no-op indexes correctly (no stall)", async function () {
    // Push a normal commit AFTER the two no-ops — this proves catch-up is unblocked.
    const normalOps = [
      { op: "Inserted", id: "terminusdb:///risk26/Doc/after_noop", string: "This document proves the engine did not stall on the empty commit and continued indexing normally." },
    ]
    const normalResult = await pushAndWait(DOMAIN, BRANCH, "c3_normal", normalOps)
    expect(normalResult.indexed_documents).to.equal(1)

    // ASSERT: last_indexed advanced to the normal commit.
    const liAfterNormal = await getLastIndexed(DOMAIN, BRANCH)
    expect(liAfterNormal.commit).to.equal(
      "c3_normal",
      "RISK-26: catch-up MUST NOT stall — normal commit after no-ops must advance",
    )

    // ASSERT: the normal document is searchable.
    const searchRes = await agent()
      .get("/search")
      .query({
        domain: DOMAIN,
        commit: "c3_normal",
        q: "stall engine indexing",
        mode: "fts",
      })
      .set("Authorization", authHeader())
      .expect(200)
    expect(searchRes.body).to.be.an("array")
    const docIds = searchRes.body.map(h => h.id)
    expect(docIds).to.include(
      "terminusdb:///risk26/Doc/after_noop",
      "document indexed after no-op commits must be searchable",
    )
  })
})
