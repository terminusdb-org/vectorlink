/**
 * Contract tests for streaming POST /push.
 *
 * Verifies incremental NDJSON parsing, abort detection (422),
 * malformed line handling, and empty body acceptance.
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

describe("POST /push — streaming behaviour", function () {
  this.timeout(120000)

  const DOMAIN = "admin/streaming_push_test"
  const BRANCH = "main"

  before(async function () {
    await agent()
      .delete("/domain")
      .query({ domain: DOMAIN })
      .set("Authorization", authHeader())
  })

  after(async function () {
    await agent()
      .delete("/domain")
      .query({ domain: DOMAIN })
      .set("Authorization", authHeader())
  })

  it("should accept streaming NDJSON with 3 documents and return 200 + task_id", async function () {
    const ops = [
      { op: "Inserted", id: "doc1", string: "First document about vector databases" },
      { op: "Inserted", id: "doc2", string: "Second document about graph databases" },
      { op: "Inserted", id: "doc3", string: "Third document about semantic search" },
    ]
    const body = ops.map(l => JSON.stringify(l)).join("\n")
    const res = await agent()
      .post("/push")
      .query({ domain: DOMAIN, branch: BRANCH, target_commit: "commit-001" })
      .set("Authorization", authHeader())
      .set("Content-Type", "application/x-ndjson")
      .send(body)
    expect(res.status).to.equal(200)
    expect(res.text).to.match(/^task-/)
    await waitForTask(res.text)
  })

  it("should return 422 when Abort is sent mid-stream", async function () {
    const ops = [
      { op: "Inserted", id: "doc_abort_1", string: "Document before abort" },
      { op: "Abort" },
    ]
    const body = ops.map(l => JSON.stringify(l)).join("\n")
    const res = await agent()
      .post("/push")
      .query({ domain: DOMAIN, branch: BRANCH, target_commit: "commit-abort-001" })
      .set("Authorization", authHeader())
      .set("Content-Type", "application/x-ndjson")
      .send(body)
    expect(res.status).to.equal(422)
    expect(res.body).to.have.property("error")
  })

  it("should return 200 + task_id for malformed NDJSON (error surfaces via /check)", async function () {
    const body = [
      JSON.stringify({ op: "Inserted", id: "doc_ok", string: "Valid document" }),
      "{malformed json}",
    ].join("\n")
    const res = await agent()
      .post("/push")
      .query({ domain: DOMAIN, branch: BRANCH, target_commit: "commit-malformed-001" })
      .set("Authorization", authHeader())
      .set("Content-Type", "application/x-ndjson")
      .send(body)
    expect(res.status).to.equal(200)
    expect(res.text).to.match(/^task-/)
    // The task should eventually show an error via /check
    const start = Date.now()
    let taskStatus = null
    while (Date.now() - start < 30000) {
      const checkRes = await agent()
        .get("/check")
        .query({ task_id: res.text })
        .set("Authorization", authHeader())
      if (checkRes.status === 500) {
        taskStatus = "Error"
        break
      }
      if (checkRes.status === 200 && checkRes.body.status === "Error") {
        taskStatus = "Error"
        break
      }
      if (checkRes.status === 200 && checkRes.body.status === "Complete") {
        taskStatus = "Complete"
        break
      }
      await new Promise(resolve => setTimeout(resolve, 500))
    }
    expect(taskStatus).to.equal("Error")
  })

  it("should return 200 + task_id for empty body (no-op commit)", async function () {
    const res = await agent()
      .post("/push")
      .query({ domain: DOMAIN, branch: BRANCH, target_commit: "commit-empty-001" })
      .set("Authorization", authHeader())
      .set("Content-Type", "application/x-ndjson")
      .send("")
    expect(res.status).to.equal(200)
    expect(res.text).to.match(/^task-/)
    await waitForTask(res.text)
  })
})
