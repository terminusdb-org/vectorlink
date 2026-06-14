/**
 * P3 branching + assign through the REAL HTTP pipeline (push with parent_commit,
 * /assign), against the live engine + embeddings. Complements the store-level
 * block-reuse proof (path identity) in `src/store/branch.rs` by proving the
 * externally-observable behaviour end-to-end.
 *
 *  - branch-out (push with parent_commit + a new branch) sees the parent's docs
 *    at the branch WITHOUT re-pushing them (block reuse, observable);
 *  - appends on the branch do not appear on main;
 *  - /assign makes the target commit search identically to the source, with no
 *    re-index (the assign call returns 204 and is instantaneous — no embedding).
 */

const { expect } = require("chai")
const { agent, authHeader } = require("../lib/agent")

async function waitForTask (taskId, timeoutMs = 60000) {
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

function ids (body) {
  return body.map(h => h.id)
}

describe("P3 branching + assign (HTTP pipeline)", function () {
  this.timeout(180000)

  const DOMAIN = "admin/branch_test"

  before(async function () {
    // main @ bc0: two docs.
    await pushAndWait(DOMAIN, "main", "bc0", [
      { op: "Inserted", id: "terminusdb:///br/People/leia", string: "Princess Leia leads the Rebel Alliance against the Empire." },
      { op: "Inserted", id: "terminusdb:///br/People/vader", string: "Darth Vader is a Sith Lord clad in black armour." },
    ])
  })

  it("branch-out from bc0 sees the parent's docs without re-pushing them", async function () {
    // Fork branch `feature` from bc0, pushing ONE new doc with parent_commit=bc0.
    await pushAndWait(
      DOMAIN, "feature", "fc0",
      [{ op: "Inserted", id: "terminusdb:///br/People/rey", string: "Rey is a scavenger from Jakku strong with the Force." }],
      "bc0",
    )

    // Search the feature branch at fc0: must find BOTH the inherited parent docs
    // (leia/vader, never re-pushed on feature) AND the branch-local doc (rey).
    const res = await agent()
      .get("/search")
      .query({ domain: `${DOMAIN}/local/branch/feature`, commit: "fc0", q: "Rebel Alliance Force", mode: "vector", count: 10 })
      .set("Authorization", authHeader())
      .expect(200)
    const found = ids(res.body)
    expect(found).to.include("terminusdb:///br/People/rey", "branch-local doc present")
    expect(found).to.include("terminusdb:///br/People/leia", "inherited parent doc present without re-push (block reuse)")
  })

  it("appends on the branch do not appear on main", async function () {
    // main @ bc0 must NOT contain rey (only added on feature).
    const res = await agent()
      .get("/search")
      .query({ domain: DOMAIN, commit: "bc0", q: "scavenger Jakku", mode: "vector", count: 10 })
      .set("Authorization", authHeader())
      .expect(200)
    expect(ids(res.body)).to.not.include("terminusdb:///br/People/rey", "branch doc must not leak into main")
  })

  it("P3-ASSIGN-1: /assign makes the target search identically to the source", async function () {
    // Assign bc0 → bc_assigned (pure tag pointer, 204, no re-index).
    await agent()
      .post("/assign")
      .query({ domain: DOMAIN, source_commit: "bc0", target_commit: "bc_assigned" })
      .set("Authorization", authHeader())
      .expect(204)

    // Search at the assigned commit equals search at the source.
    const q = { q: "Rebel Alliance", mode: "vector", count: 10 }
    const src = await agent()
      .get("/search")
      .query({ domain: DOMAIN, commit: "bc0", ...q })
      .set("Authorization", authHeader())
      .expect(200)
    const tgt = await agent()
      .get("/search")
      .query({ domain: DOMAIN, commit: "bc_assigned", ...q })
      .set("Authorization", authHeader())
      .expect(200)
    expect(ids(tgt.body).sort()).to.deep.equal(ids(src.body).sort(), "assigned commit must search identically to source")
  })
})
