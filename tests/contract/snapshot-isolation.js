/**
 * Snapshot isolation integration tests.
 *
 * Proves that /search at commit C0 returns only data indexed at C0,
 * not data added by a later commit C1.
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

async function pushAndWait (domain, branch, commit, ops) {
  const body = ops.map(l => JSON.stringify(l)).join("\n")
  const pushRes = await agent()
    .post("/push")
    .query({ domain, branch, target_commit: commit })
    .set("Authorization", authHeader())
    .set("Content-Type", "application/x-ndjson")
    .send(body)
    .expect(200)
  return waitForTask(pushRes.text)
}

describe("Snapshot isolation", function () {
  this.timeout(120000)

  const DOMAIN = "admin/snapshot_test"
  const BRANCH = "main"

  before(async function () {
    // Clean up any stale state from prior runs (each test must be independently runnable).
    await agent().delete("/domain").query({ domain: DOMAIN }).set("Authorization", authHeader())
    // Commit iso_c0: only has doc/alpha.
    await pushAndWait(DOMAIN, BRANCH, "iso_c0", [
      { op: "Inserted", id: "terminusdb:///snap/Animals/alpha", string: "The alpha wolf leads the pack through the forest at dawn." },
    ])

    // Commit iso_c1: adds doc/beta (alpha still present via append-only).
    await pushAndWait(DOMAIN, BRANCH, "iso_c1", [
      { op: "Inserted", id: "terminusdb:///snap/Animals/beta", string: "The beta fish is a colourful freshwater species popular in aquariums." },
    ])
  })

  after(async function () {
    await agent().delete("/domain").query({ domain: DOMAIN }).set("Authorization", authHeader())
  })

  it("search at iso_c0 finds alpha but NOT beta", async function () {
    const res = await agent()
      .get("/search")
      .query({
        domain: DOMAIN,
        commit: "iso_c0",
        q: "animal",
        mode: "vector",
        count: 10,
      })
      .set("Authorization", authHeader())
      .expect(200)

    const ids = res.body.map(h => h.id)
    expect(ids).to.include("terminusdb:///snap/Animals/alpha")
    expect(ids).to.not.include("terminusdb:///snap/Animals/beta")
  })

  it("search at iso_c1 finds both alpha and beta", async function () {
    const res = await agent()
      .get("/search")
      .query({
        domain: DOMAIN,
        commit: "iso_c1",
        q: "animal",
        mode: "vector",
        count: 10,
      })
      .set("Authorization", authHeader())
      .expect(200)

    const ids = res.body.map(h => h.id)
    expect(ids).to.include("terminusdb:///snap/Animals/alpha")
    expect(ids).to.include("terminusdb:///snap/Animals/beta")
  })
})
