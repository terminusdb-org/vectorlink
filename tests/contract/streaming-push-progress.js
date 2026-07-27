/**
 * Contract tests for streaming push with progress feedback (stream=true).
 *
 * These tests define the full streaming push contract with 100% clarity,
 * testable locally against vectorlink alone (no TerminusDB required).
 * The contract matches exactly what TerminusDB will consume:
 *
 *   POST /push?stream=true&domain=...&target_commit=...
 *   Content-Type: application/x-ndjson
 *
 *   → 200 OK
 *   → Content-Type: application/x-ndjson
 *   → X-Task-Id: task-<uuid>
 *   → Body: NDJSON lines of ProgressUpdate objects
 *
 * ProgressUpdate shapes:
 *   {"status":"progress","indexed":N,"total_seen":M,"skipped":K}
 *   {"status":"complete","indexed_documents":N,"skipped":[...]}
 *   {"status":"error","error":"..."}
 *   {"status":"aborted"}
 */

const { expect } = require("chai")
const http = require("http")
const { agent, authHeader, BASE_URL } = require("../lib/agent")

const ADMIN_USER = process.env.VECTORLINK_ADMIN_USER || "admin"
const ADMIN_SECRET = process.env.VECTORLINK_ADMIN_SECRET || "root"

/**
 * Parse the BASE_URL into host and port for raw HTTP requests.
 */
function parseBaseUrl () {
  const url = new URL(BASE_URL)
  return {
    hostname: url.hostname,
    port: url.port || (url.protocol === "https:" ? 443 : 80),
  }
}

/**
 * Send a streaming push request and collect all NDJSON response lines.
 *
 * Uses raw http.request instead of supertest because supertest buffers
 * the entire response body, preventing streaming reading.
 *
 * @param {object} opts
 * @param {string} opts.domain
 * @param {string} opts.branch
 * @param {string} opts.targetCommit
 * @param {string} opts.body - NDJSON string
 * @param {boolean} opts.stream - if true, adds stream=true query param
 * @param {number} opts.timeoutMs - response timeout (default 120000)
 * @returns {Promise<{status: number, headers: object, lines: string[], rawBody: string}>}
 */
function streamingPush (opts) {
  const { hostname, port } = parseBaseUrl()
  const streamParam = opts.stream ? "&stream=true" : ""
  const path = `/push?domain=${encodeURIComponent(opts.domain)}&branch=${encodeURIComponent(opts.branch || "main")}&target_commit=${encodeURIComponent(opts.targetCommit)}${streamParam}`
  const auth = Buffer.from(`${ADMIN_USER}:${ADMIN_SECRET}`).toString("base64")

  return new Promise((resolve, reject) => {
    const req = http.request(
      {
        hostname,
        port,
        path,
        method: "POST",
        headers: {
          "Content-Type": "application/x-ndjson",
          Authorization: `Basic ${auth}`,
        },
        timeout: opts.timeoutMs || 120000,
      },
      (res) => {
        const chunks = []
        res.on("data", (chunk) => chunks.push(chunk))
        res.on("end", () => {
          const rawBody = Buffer.concat(chunks).toString("utf-8")
          const lines = rawBody
            .split("\n")
            .map((l) => l.trim())
            .filter((l) => l.length > 0)
          resolve({
            status: res.statusCode,
            headers: res.headers,
            lines,
            rawBody,
          })
        })
        res.on("error", reject)
      },
    )
    req.on("error", reject)
    req.on("timeout", () => {
      req.destroy(new Error("request timed out"))
    })
    req.write(opts.body || "")
    req.end()
  })
}

/**
 * Parse NDJSON lines into JSON objects.
 * @param {string[]} lines
 * @returns {object[]}
 */
function parseLines (lines) {
  return lines.map((l) => JSON.parse(l))
}

/**
 * Wait for a task to reach a terminal state via /check.
 * @param {string} taskId
 * @param {number} timeoutMs
 * @returns {Promise<object>}
 */
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
    if (res.status === 200 && res.body.status === "Error") {
      throw new Error(`task error: ${res.body.error}`)
    }
    await new Promise((resolve) => setTimeout(resolve, 500))
  }
  throw new Error(`task ${taskId} did not complete within ${timeoutMs}ms`)
}

describe("POST /push?stream=true — streaming push with progress", function () {
  this.timeout(120000)

  const DOMAIN = "admin/streaming_progress_test"
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

  // Test 6: stream=true returns 200 with application/x-ndjson content type
  it("returns 200 with application/x-ndjson content type", async function () {
    const ops = [
      { op: "Inserted", id: "doc1", string: "First document about vector databases" },
    ]
    const body = ops.map((l) => JSON.stringify(l)).join("\n")
    const res = await streamingPush({
      domain: DOMAIN,
      branch: BRANCH,
      targetCommit: "commit-content-type-001",
      body,
      stream: true,
    })
    expect(res.status).to.equal(200)
    expect(res.headers["content-type"]).to.include("application/x-ndjson")
  })

  // Test 7: stream=true returns X-Task-Id header with valid task-<uuid> value
  it("returns X-Task-Id header with valid task-<uuid> value", async function () {
    const ops = [
      { op: "Inserted", id: "doc1", string: "Document for task-id test" },
    ]
    const body = ops.map((l) => JSON.stringify(l)).join("\n")
    const res = await streamingPush({
      domain: DOMAIN,
      branch: BRANCH,
      targetCommit: "commit-task-id-001",
      body,
      stream: true,
    })
    expect(res.status).to.equal(200)
    expect(res.headers).to.have.property("x-task-id")
    expect(res.headers["x-task-id"]).to.match(/^task-/)
  })

  // Test 8: stream=true with 3 docs returns NDJSON lines: at least one progress, then complete
  it("with 3 docs returns at least one progress line then a complete line", async function () {
    const ops = [
      { op: "Inserted", id: "doc1", string: "First document about vector databases" },
      { op: "Inserted", id: "doc2", string: "Second document about graph databases" },
      { op: "Inserted", id: "doc3", string: "Third document about semantic search" },
    ]
    const body = ops.map((l) => JSON.stringify(l)).join("\n")
    const res = await streamingPush({
      domain: DOMAIN,
      branch: BRANCH,
      targetCommit: "commit-3docs-001",
      body,
      stream: true,
    })
    expect(res.status).to.equal(200)
    expect(res.lines.length).to.be.greaterThan(0)

    const updates = parseLines(res.lines)

    // Must have at least one progress line
    const progressLines = updates.filter((u) => u.status === "progress")
    expect(progressLines.length).to.be.greaterThan(0)

    // Last line must be complete
    const lastUpdate = updates[updates.length - 1]
    expect(lastUpdate.status).to.equal("complete")
  })

  // Test 9: progress line shape
  it("progress line has correct shape: {status, indexed, total_seen, skipped}", async function () {
    const ops = [
      { op: "Inserted", id: "doc1", string: "Document for progress shape test" },
    ]
    const body = ops.map((l) => JSON.stringify(l)).join("\n")
    const res = await streamingPush({
      domain: DOMAIN,
      branch: BRANCH,
      targetCommit: "commit-progress-shape-001",
      body,
      stream: true,
    })
    expect(res.status).to.equal(200)

    const updates = parseLines(res.lines)
    const progressLines = updates.filter((u) => u.status === "progress")

    for (const p of progressLines) {
      expect(p).to.have.property("status", "progress")
      expect(p).to.have.property("indexed")
      expect(p.indexed).to.be.a("number")
      expect(p).to.have.property("total_seen")
      expect(p.total_seen).to.be.a("number")
      expect(p).to.have.property("skipped")
      expect(p.skipped).to.be.a("number")
    }
  })

  // Test 10: complete line shape matches /check Complete shape
  it("complete line has correct shape: {status, indexed_documents, skipped}", async function () {
    const ops = [
      { op: "Inserted", id: "doc1", string: "Document for complete shape test" },
    ]
    const body = ops.map((l) => JSON.stringify(l)).join("\n")
    const res = await streamingPush({
      domain: DOMAIN,
      branch: BRANCH,
      targetCommit: "commit-complete-shape-001",
      body,
      stream: true,
    })
    expect(res.status).to.equal(200)

    const updates = parseLines(res.lines)
    const completeLine = updates[updates.length - 1]

    expect(completeLine.status).to.equal("complete")
    expect(completeLine).to.have.property("indexed_documents")
    expect(completeLine.indexed_documents).to.be.a("number")
    expect(completeLine).to.have.property("skipped")
    expect(completeLine.skipped).to.be.an("array")
  })

  // Test 11: stream=true with empty body returns complete with indexed_documents=0
  it("with empty body returns complete with indexed_documents=0 and skipped=[]", async function () {
    const res = await streamingPush({
      domain: DOMAIN,
      branch: BRANCH,
      targetCommit: "commit-empty-001",
      body: "",
      stream: true,
    })
    expect(res.status).to.equal(200)

    const updates = parseLines(res.lines)
    const lastUpdate = updates[updates.length - 1]

    expect(lastUpdate.status).to.equal("complete")
    expect(lastUpdate.indexed_documents).to.equal(0)
    expect(lastUpdate.skipped).to.deep.equal([])
  })

  // Test 12: stream=true with abort returns {"status":"aborted"} as the final line
  it("with abort returns {\"status\":\"aborted\"} as the final line", async function () {
    const ops = [
      { op: "Inserted", id: "doc_before_abort", string: "Document before abort" },
      { op: "Abort" },
    ]
    const body = ops.map((l) => JSON.stringify(l)).join("\n")
    const res = await streamingPush({
      domain: DOMAIN,
      branch: BRANCH,
      targetCommit: "commit-abort-001",
      body,
      stream: true,
    })
    expect(res.status).to.equal(200)

    const updates = parseLines(res.lines)
    const lastUpdate = updates[updates.length - 1]

    expect(lastUpdate.status).to.equal("aborted")
  })

  // Test 13: stream=true with malformed NDJSON still returns progress + complete
  it("with malformed NDJSON still returns progress + complete (error goes to skipped)", async function () {
    const body = [
      JSON.stringify({ op: "Inserted", id: "doc_ok", string: "Valid document for malformed test" }),
      "{malformed json}",
    ].join("\n")
    const res = await streamingPush({
      domain: DOMAIN,
      branch: BRANCH,
      targetCommit: "commit-malformed-001",
      body,
      stream: true,
    })
    expect(res.status).to.equal(200)

    const updates = parseLines(res.lines)
    const lastUpdate = updates[updates.length - 1]

    // The malformed line should cause the task to error, but the stream
    // should still contain progress updates and a terminal line.
    // The terminal line may be "complete" or "error" depending on how
    // the pipeline handles malformed input in stream mode.
    expect(["complete", "error"]).to.include(lastUpdate.status)
  })

  // Test 14: stream=false (or omitted) returns text/plain with task-<uuid> (backward compat)
  it("stream=false returns text/plain with task-<uuid> (backward compat)", async function () {
    const ops = [
      { op: "Inserted", id: "doc1", string: "Backward compat test document" },
    ]
    const body = ops.map((l) => JSON.stringify(l)).join("\n")
    const res = await streamingPush({
      domain: DOMAIN,
      branch: BRANCH,
      targetCommit: "commit-backward-compat-001",
      body,
      stream: false,
    })
    expect(res.status).to.equal(200)
    expect(res.headers["content-type"]).to.include("text/plain")
    expect(res.rawBody).to.match(/^task-/)
  })

  it("stream omitted returns text/plain with task-<uuid> (backward compat)", async function () {
    const ops = [
      { op: "Inserted", id: "doc1", string: "No stream param test document" },
    ]
    const body = ops.map((l) => JSON.stringify(l)).join("\n")
    const res = await streamingPush({
      domain: DOMAIN,
      branch: BRANCH,
      targetCommit: "commit-no-stream-param-001",
      body,
      // stream not set — no stream=true in query
    })
    expect(res.status).to.equal(200)
    expect(res.headers["content-type"]).to.include("text/plain")
    expect(res.rawBody).to.match(/^task-/)
  })

  // Test 15: stream=true with 409 conflict returns 409 Conflict (no streaming body)
  it("with 409 conflict returns 409 Conflict with JSON error (no streaming body)", async function () {
    // First push to claim the commit
    const ops = [
      { op: "Inserted", id: "doc1", string: "First push for conflict test" },
    ]
    const body = ops.map((l) => JSON.stringify(l)).join("\n")
    const firstRes = await streamingPush({
      domain: DOMAIN,
      branch: BRANCH,
      targetCommit: "commit-conflict-001",
      body,
      stream: true,
    })
    expect(firstRes.status).to.equal(200)

    // Second push to same commit — should get 409
    const secondRes = await streamingPush({
      domain: DOMAIN,
      branch: BRANCH,
      targetCommit: "commit-conflict-001",
      body,
      stream: true,
    })
    expect(secondRes.status).to.equal(409)
    expect(secondRes.headers["content-type"]).to.include("application/json")
    // Should NOT be a streaming NDJSON body (no progress/complete lines).
    // A 409 response has a single JSON error object, not NDJSON progress updates.
    const parsed = JSON.parse(secondRes.rawBody)
    expect(parsed).to.have.property("error")
    expect(parsed.status).to.not.equal("progress")
    expect(parsed.status).to.not.equal("complete")
  })

  // Test 16: stream=true task-id from X-Task-Id is queryable via /check
  it("task-id from X-Task-Id header is queryable via /check and returns Complete", async function () {
    const ops = [
      { op: "Inserted", id: "doc1", string: "Document for check queryability test" },
    ]
    const body = ops.map((l) => JSON.stringify(l)).join("\n")
    const res = await streamingPush({
      domain: DOMAIN,
      branch: BRANCH,
      targetCommit: "commit-check-001",
      body,
      stream: true,
    })
    expect(res.status).to.equal(200)
    const taskId = res.headers["x-task-id"]
    expect(taskId).to.match(/^task-/)

    // Wait for the task to complete via /check
    const taskResult = await waitForTask(taskId)
    expect(taskResult.status).to.equal("Complete")
    expect(taskResult).to.have.property("indexed_documents")
    expect(taskResult).to.have.property("skipped")
  })

  // Test 17: stream=true with Error operations returns complete with indexed_documents=0
  it("with Error operations returns complete with indexed_documents=0 and skipped entries", async function () {
    const ops = [
      { op: "Error", message: "render failed: doc/broken1" },
      { op: "Error", message: "render failed: doc/broken2" },
    ]
    const body = ops.map((l) => JSON.stringify(l)).join("\n")
    const res = await streamingPush({
      domain: DOMAIN,
      branch: BRANCH,
      targetCommit: "commit-error-ops-001",
      body,
      stream: true,
    })
    expect(res.status).to.equal(200)

    const updates = parseLines(res.lines)
    const lastUpdate = updates[updates.length - 1]

    expect(lastUpdate.status).to.equal("complete")
    expect(lastUpdate.indexed_documents).to.equal(0)
    expect(lastUpdate.skipped.length).to.equal(2)
  })

  // Test 18: stream=true response stream ends after terminal line (no extra data)
  it("response stream ends after terminal line (no extra data)", async function () {
    const ops = [
      { op: "Inserted", id: "doc1", string: "Document for stream end test" },
    ]
    const body = ops.map((l) => JSON.stringify(l)).join("\n")
    const res = await streamingPush({
      domain: DOMAIN,
      branch: BRANCH,
      targetCommit: "commit-stream-end-001",
      body,
      stream: true,
    })
    expect(res.status).to.equal(200)

    const updates = parseLines(res.lines)
    const lastUpdate = updates[updates.length - 1]

    // Last update must be a terminal status
    expect(["complete", "error", "aborted"]).to.include(lastUpdate.status)

    // There should be no lines after the terminal line
    // (already guaranteed by how we split, but verify terminal is truly last)
    const terminalIdx = updates.findIndex(
      (u) => u.status === "complete" || u.status === "error" || u.status === "aborted",
    )
    expect(terminalIdx).to.equal(updates.length - 1)
  })

  // Test 19: stream=true with Deleted operations returns complete with correct count
  it("with Deleted operations returns complete with correct indexed_documents", async function () {
    // First insert a doc
    const insertOps = [
      { op: "Inserted", id: "doc_to_delete", string: "Document that will be deleted" },
    ]
    const insertBody = insertOps.map((l) => JSON.stringify(l)).join("\n")
    const insertRes = await streamingPush({
      domain: DOMAIN,
      branch: BRANCH,
      targetCommit: "commit-delete-insert-001",
      body: insertBody,
      stream: true,
    })
    expect(insertRes.status).to.equal(200)

    // Now push a commit that deletes the doc
    const deleteOps = [
      { op: "Deleted", id: "doc_to_delete" },
    ]
    const deleteBody = deleteOps.map((l) => JSON.stringify(l)).join("\n")
    const res = await streamingPush({
      domain: DOMAIN,
      branch: BRANCH,
      targetCommit: "commit-delete-002",
      body: deleteBody,
      stream: true,
    })
    expect(res.status).to.equal(200)

    const updates = parseLines(res.lines)
    const lastUpdate = updates[updates.length - 1]

    expect(lastUpdate.status).to.equal("complete")
    // Deleted-only commit: 0 new docs indexed
    expect(lastUpdate.indexed_documents).to.equal(0)
    expect(lastUpdate.skipped).to.deep.equal([])
  })

  // Test 20: stream=true client disconnect — pipeline aborts or completes, no orphaned data
  it("client disconnect (abort HTTP request mid-stream) — pipeline reaches terminal state, no orphaned data", async function () {
    const { hostname, port } = parseBaseUrl()
    const targetCommit = "commit-disconnect-001"
    const path = `/push?domain=${encodeURIComponent(DOMAIN)}&branch=${encodeURIComponent(BRANCH)}&target_commit=${encodeURIComponent(targetCommit)}&stream=true`
    const auth = Buffer.from(`${ADMIN_USER}:${ADMIN_SECRET}`).toString("base64")

    // Create a large body with many documents to keep the stream busy
    const manyOps = []
    for (let i = 0; i < 100; i++) {
      manyOps.push({
        op: "Inserted",
        id: `doc_disconnect_${i}`,
        string: `Document ${i} for disconnect test with enough content to keep the pipeline busy`,
      })
    }
    const body = manyOps.map((l) => JSON.stringify(l)).join("\n")

    // Capture the X-Task-Id from the response headers before disconnecting
    let capturedTaskId = null

    // Send the request but abort it after receiving the first progress line
    await new Promise((resolve) => {
      const req = http.request(
        {
          hostname,
          port,
          path,
          method: "POST",
          headers: {
            "Content-Type": "application/x-ndjson",
            Authorization: `Basic ${auth}`,
          },
          timeout: 30000,
        },
        (res) => {
          // Capture task ID from header
          if (res.headers["x-task-id"]) {
            capturedTaskId = res.headers["x-task-id"]
          }
          // As soon as we get any data, abort the request
          res.on("data", () => {
            req.destroy()
          })
          res.on("error", () => {
            // Expected — connection was destroyed
          })
        },
      )
      req.on("error", () => {
        // Expected — we destroyed the request
      })
      req.write(body)
      req.end()
      // Give it a moment to send data, then resolve
      setTimeout(resolve, 5000)
    })

    // Wait for the pipeline to process the disconnect or complete.
    // Poll /check until the task reaches a terminal state (Complete or Error),
    // or timeout. A fixed wait is insufficient in CI where Ollama is slow.
    expect(capturedTaskId).to.not.equal(null)
    expect(capturedTaskId).to.match(/^task-/)

    let checkRes = null
    const pollDeadline = Date.now() + 60000
    while (Date.now() < pollDeadline) {
      checkRes = await agent()
        .get("/check")
        .query({ task_id: capturedTaskId })
        .set("Authorization", authHeader())
      if (checkRes.status === 200 && checkRes.body.status !== "Pending") {
        break
      }
      if (checkRes.status === 500) {
        break
      }
      await new Promise(resolve => setTimeout(resolve, 1000))
    }

    // Must be 200 (Complete) or 500 (Error) — never 200 with Pending
    if (checkRes.status === 200) {
      expect(checkRes.body.status).to.not.equal("Pending")
      // If Complete, the data is consistent — /last-indexed should point to the commit
      if (checkRes.body.status === "Complete") {
        const liRes = await agent()
          .get("/last-indexed")
          .query({ domain: DOMAIN, branch: BRANCH })
          .set("Authorization", authHeader())
          .expect(200)
        // last_indexed may or may not be at this commit (depending on timing),
        // but it must be a valid commit (not null)
        expect(liRes.body.commit).to.not.equal(null)
      }
    } else if (checkRes.status === 500) {
      // Error is expected when the pipeline detected the disconnect
      expect(checkRes.text).to.include("disconnected")
    } else {
      throw new Error(`Unexpected /check status: ${checkRes.status}`)
    }
  })
})
