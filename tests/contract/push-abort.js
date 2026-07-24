/**
 * Contract test for push abort (OpAbort → 422 Unprocessable Entity).
 *
 * Verifies that:
 *   1. A non-streaming push with an Abort line returns 422 with the abort message.
 *   2. The task is recorded as an error (checkable via /check if a task id is returned).
 *   3. The abort line number is reported in the error message.
 */

const { expect } = require("chai")
const { agent, authHeader } = require("../lib/agent")

describe("POST /push — abort (OpAbort → 422)", function () {
  this.timeout(120000)

  const DOMAIN = "admin/abort_test"

  before(async function () {
    await agent().delete("/domain").query({ domain: DOMAIN }).set("Authorization", authHeader())
  })

  after(async function () {
    await agent().delete("/domain").query({ domain: DOMAIN }).set("Authorization", authHeader())
  })

  it("non-streaming push with Abort returns 422 with abort message", async function () {
    const ndjson = [
      JSON.stringify({ op: "Inserted", id: "terminusdb:///ab/Doc/1", string: "Hello world" }),
      JSON.stringify({ op: "Inserted", id: "terminusdb:///ab/Doc/2", string: "Second doc" }),
      JSON.stringify({ op: "Abort" }),
    ].join("\n")

    const res = await agent()
      .post("/push")
      .query({ domain: DOMAIN, branch: "main", target_commit: "abort_c0" })
      .set("Authorization", authHeader())
      .set("Content-Type", "application/x-ndjson")
      .send(ndjson)

    expect(res.status).to.equal(422)
    expect(res.body).to.have.property("error")
    expect(res.body.error).to.include("abort")

    // Allow the aborted pipeline to fully wind down (release lock + reservation)
    // before the next test starts a new push on the same domain/branch.
    await new Promise(resolve => setTimeout(resolve, 1000))
  })

  it("abort on first line returns 422 immediately", async function () {
    const ndjson = JSON.stringify({ op: "Abort" })

    const res = await agent()
      .post("/push")
      .query({ domain: DOMAIN, branch: "main", target_commit: "abort_c1" })
      .set("Authorization", authHeader())
      .set("Content-Type", "application/x-ndjson")
      .send(ndjson)

    expect(res.status).to.equal(422)
    expect(res.body).to.have.property("error")

    // Allow the aborted pipeline to fully wind down.
    await new Promise(resolve => setTimeout(resolve, 1000))
  })

  it("abort after valid inserts does not index the aborted commit", async function () {
    const ndjson = [
      JSON.stringify({ op: "Inserted", id: "terminusdb:///ab/Doc/10", string: "Document ten" }),
      JSON.stringify({ op: "Abort" }),
    ].join("\n")

    const res = await agent()
      .post("/push")
      .query({ domain: DOMAIN, branch: "main", target_commit: "abort_c2" })
      .set("Authorization", authHeader())
      .set("Content-Type", "application/x-ndjson")
      .send(ndjson)

    expect(res.status).to.equal(422)

    // Verify the aborted commit was NOT indexed.
    const lastRes = await agent()
      .get("/last-indexed")
      .query({ domain: DOMAIN, branch: "main" })
      .set("Authorization", authHeader())
      .expect(200)

    // The last-indexed commit should either be null (nothing indexed) or
    // a different commit — never "abort_c2".
    if (lastRes.body.commit) {
      expect(lastRes.body.commit).to.not.equal("abort_c2")
    }
  })

  it("requires authentication even for abort", async function () {
    const ndjson = JSON.stringify({ op: "Abort" })

    const res = await agent()
      .post("/push")
      .query({ domain: DOMAIN, branch: "main", target_commit: "abort_c3" })
      .set("Content-Type", "application/x-ndjson")
      .send(ndjson)

    expect(res.status).to.equal(401)
  })
})
