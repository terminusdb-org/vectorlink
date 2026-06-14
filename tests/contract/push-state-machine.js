/**
 * Push 409 state-machine integration tests (real pipeline, live server).
 *
 * A commit is "committed" the moment its push is ACCEPTED — not only once
 * indexing finishes and tags it. The 409 guard must therefore reject a re-push
 * of a target_commit that is in ANY non-absent state:
 *   - Reserved/Indexing: a push is in flight (task not yet terminal).
 *   - Indexed: already tagged (durable).
 * The check-and-reserve is atomic: two concurrent pushes of the same commit
 * yield exactly one 200 and one 409. A failed index releases the reservation so
 * a legitimate retry is allowed.
 *
 * Requires the live engine + embeddings backend (run via `make test-integration`).
 */

const { expect } = require("chai")
const { agent, authHeader } = require("../lib/agent")

// Poll /check until the task reaches a terminal state. Returns { status, body }.
// status is "Complete" (200 Complete), "Error" (500), or throws on timeout.
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
    if (res.status === 500) {
      return { status: "Error", body: res.text }
    }
    await new Promise(resolve => setTimeout(resolve, 300))
  }
  throw new Error(`task ${taskId} did not reach a terminal state within ${timeoutMs}ms`)
}

function pushRequest (domain, branch, commit, ndjsonBody) {
  return agent()
    .post("/push")
    .query({ domain, branch, target_commit: commit })
    .set("Authorization", authHeader())
    .set("Content-Type", "application/x-ndjson")
    .send(ndjsonBody)
}

const oneDoc = JSON.stringify({
  op: "Inserted",
  id: "terminusdb:///sm/People/yoda",
  string: "Yoda is a wise Jedi master who trains padawans in the ways of the Force.",
})

describe("POST /push — 409 state machine (accepted / indexing / indexed + atomicity)", function () {
  this.timeout(120000)

  const BRANCH = "main"

  it("re-pushing an INDEXED commit returns 409 (the original bug)", async function () {
    const domain = "admin/sm_indexed"
    const first = await pushRequest(domain, BRANCH, "rc0", oneDoc).expect(200)
    const result = await waitForTerminal(first.text)
    expect(result.status).to.equal("Complete")

    // Re-push the same indexed commit → 409.
    const repush = await pushRequest(domain, BRANCH, "rc0", oneDoc)
    expect(repush.status).to.equal(409,
      `expected 409 for already-indexed commit, got ${repush.status}: ${repush.text}`)
    expect(repush.text).to.not.match(/^task-/)
  })

  it("re-pushing a commit whose first push is STILL INDEXING returns 409", async function () {
    const domain = "admin/sm_inflight"
    // First push: accept it but do NOT wait for completion — it is now in the
    // Reserved/Indexing window.
    const first = await pushRequest(domain, BRANCH, "rc_inflight", oneDoc).expect(200)
    expect(first.text).to.match(/^task-/)

    // Immediately re-push the same commit while the first is in flight → 409.
    const repush = await pushRequest(domain, BRANCH, "rc_inflight", oneDoc)
    expect(repush.status).to.equal(409,
      `expected 409 for in-flight commit, got ${repush.status}: ${repush.text}`)
    expect(repush.text).to.not.match(/^task-/)

    // Let the first push finish so it does not bleed into other tests.
    await waitForTerminal(first.text)
  })

  it("TWO concurrent pushes of the same new commit → exactly one 200, one 409", async function () {
    const domain = "admin/sm_race"
    // Fire both pushes concurrently (do not await sequentially).
    const [a, b] = await Promise.all([
      pushRequest(domain, BRANCH, "rc_race", oneDoc),
      pushRequest(domain, BRANCH, "rc_race", oneDoc),
    ])
    const statuses = [a.status, b.status].sort()
    expect(statuses).to.deep.equal([200, 409],
      `expected exactly one 200 and one 409, got ${JSON.stringify(statuses)} ` +
      `(a=${a.status}:${a.text}, b=${b.status}:${b.text})`)

    // Drain the accepted task.
    const winner = a.status === 200 ? a : b
    await waitForTerminal(winner.text)
  })

  it("the reservation is released on terminal — a different commit on the same branch is accepted (no leak)", async function () {
    // The release-on-FAILURE semantics (a failed index frees the commit for
    // retry) are proven deterministically at the store level
    // (`reserve_commit_rejects_inflight_then_release_allows_retry`). Here we prove
    // the live release-on-terminal: after a push completes, the reservation does
    // not leak and a subsequent push of a DIFFERENT commit on the same branch is
    // accepted, while the SAME commit stays 409 (durable Indexed).
    const domain = "admin/sm_release"
    const first = await pushRequest(domain, BRANCH, "rc_a", oneDoc).expect(200)
    const firstResult = await waitForTerminal(first.text)
    expect(firstResult.status).to.equal("Complete")

    // Same commit → 409 (Indexed, durable).
    const repush = await pushRequest(domain, BRANCH, "rc_a", oneDoc)
    expect(repush.status).to.equal(409,
      `same indexed commit must be 409, got ${repush.status}: ${repush.text}`)

    // Different commit on the same branch → 200 (reservation released; no leak).
    const second = await pushRequest(domain, BRANCH, "rc_b", oneDoc).expect(200)
    expect(second.text).to.match(/^task-/)
    await waitForTerminal(second.text)
  })
})
