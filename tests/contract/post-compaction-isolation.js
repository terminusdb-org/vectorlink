/**
 * Post-compaction snapshot isolation (deterministic regression guard).
 *
 * Proves that explicit /compact does NOT break snapshot isolation.
 * The bug: compaction formerly retagged ALL historical commit tags to the
 * compacted version, so checkout_version(c0) returned post-c1 data.
 *
 * This test deterministically triggers compaction via POST /compact (not the
 * probabilistic 5% trigger) and verifies the historical-accuracy guarantee:
 * search at c0 EXCLUDES data added at c1, even after compaction has run.
 *
 * Prerequisite: TDB_SEARCH_DISABLE_AUTO_COMPACTION=1 in the engine env
 * (eliminates non-determinism from the 5% probabilistic trigger).
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

describe("Post-compaction snapshot isolation (deterministic)", function () {
  this.timeout(120000)

  const DOMAIN = "admin/compact_isolation_test"
  const BRANCH = "main"

  before(async function () {
    // Clean up any stale state.
    await agent().delete("/domain").query({ domain: DOMAIN }).set("Authorization", authHeader())

    // Commit compact_c0: only has doc/alpha.
    await pushAndWait(DOMAIN, BRANCH, "compact_c0", [
      { op: "Inserted", id: "terminusdb:///compact/Animals/alpha", string: "The alpha wolf leads the pack through the northern forest." },
    ])

    // Commit compact_c1: adds doc/beta (alpha still present via append-only).
    await pushAndWait(DOMAIN, BRANCH, "compact_c1", [
      { op: "Inserted", id: "terminusdb:///compact/Animals/beta", string: "The beta fish is a vibrant freshwater species kept in aquariums." },
    ])
  })

  after(async function () {
    await agent().delete("/domain").query({ domain: DOMAIN }).set("Authorization", authHeader())
  })

  it("POST /compact succeeds and reports compaction metadata", async function () {
    const res = await agent()
      .post("/compact")
      .query({ domain: DOMAIN, branch: BRANCH })
      .set("Authorization", authHeader())
      .expect(200)

    expect(res.body).to.have.property("domain", DOMAIN)
    expect(res.body).to.have.property("branch", BRANCH)
    expect(res.body).to.have.property("fragments_before")
    expect(res.body).to.have.property("fragments_after")
    expect(res.body).to.have.property("compacted_version")
  })

  it("search at compact_c0 finds alpha but NOT beta AFTER compaction", async function () {
    // This is THE critical assertion: after compaction has provably run,
    // historical isolation is preserved. If the retag bug were present, c0's
    // tag would point to the compacted version (which includes beta).
    const res = await agent()
      .get("/search")
      .query({
        domain: DOMAIN,
        commit: "compact_c0",
        q: "animal",
        mode: "vector",
        count: 10,
      })
      .set("Authorization", authHeader())
      .expect(200)

    const ids = res.body.map(h => h.id)
    expect(ids).to.include("terminusdb:///compact/Animals/alpha")
    expect(ids).to.not.include(
      "terminusdb:///compact/Animals/beta",
      "POST-COMPACTION ISOLATION VIOLATED: c0 must NOT see beta (added at c1). " +
      "If this fails, compaction is retagging historical commit tags.",
    )
  })

  it("search at compact_c1 finds both alpha and beta AFTER compaction", async function () {
    const res = await agent()
      .get("/search")
      .query({
        domain: DOMAIN,
        commit: "compact_c1",
        q: "animal",
        mode: "vector",
        count: 10,
      })
      .set("Authorization", authHeader())
      .expect(200)

    const ids = res.body.map(h => h.id)
    expect(ids).to.include("terminusdb:///compact/Animals/alpha")
    expect(ids).to.include("terminusdb:///compact/Animals/beta")
  })
})
