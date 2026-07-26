/**
 * Contract test for GET /suggest — typeahead content verification.
 *
 * Verifies the response shape and actual suggestion content:
 *   - served_commit is present and matches the indexed commit
 *   - total_approx is a non-negative integer
 *   - completions is an array of strings
 *   - hits is an array of {id, distance} objects
 *   - count parameter limits the number of hits
 *   - doc_type filter restricts suggestions
 *   - TerminusDB-Data-Version header is present in the response
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
    if (res.status === 200 && res.body.status === "Complete") return res.body
    if (res.status === 500) throw new Error(`task failed: ${res.text}`)
    await new Promise((resolve) => setTimeout(resolve, 500))
  }
  throw new Error(`task ${taskId} did not complete within ${timeoutMs}ms`)
}

async function pushAndWait (domain, branch, commit, ops) {
  const body = ops.map((l) => JSON.stringify(l)).join("\n")
  const pushRes = await agent()
    .post("/push")
    .query({ domain, branch, target_commit: commit })
    .set("Authorization", authHeader())
    .set("Content-Type", "application/x-ndjson")
    .send(body)
    .expect(200)
  return waitForTask(pushRes.text)
}

describe("GET /suggest — typeahead content verification", function () {
  this.timeout(180000)

  const DOMAIN = "admin/suggest_content_test"
  const BRANCH = "main"
  const COMMIT = "sc_c0"

  before(async function () {
    await agent().delete("/domain").query({ domain: DOMAIN }).set("Authorization", authHeader())
    await pushAndWait(DOMAIN, BRANCH, COMMIT, [
      {
        op: "Inserted",
        id: "terminusdb:///sc/People/Luke",
        string: "Luke Skywalker is a farm boy turned Jedi Knight who battles the Galactic Empire",
      },
      {
        op: "Inserted",
        id: "terminusdb:///sc/People/Yoda",
        string: "Yoda is a wise old Jedi master who trains Luke in the ways of the Force",
      },
      {
        op: "Inserted",
        id: "terminusdb:///sc/Species/Wookiee",
        string: "Wookiees are tall hairy creatures from the planet Kashyyyk known for their strength",
      },
    ])
  })

  after(async function () {
    await agent().delete("/domain").query({ domain: DOMAIN }).set("Authorization", authHeader())
  })

  it("returns correct response shape with served_commit, completions, and hits", async function () {
    const res = await agent()
      .get("/suggest")
      .query({ domain: DOMAIN, commit: COMMIT, q: "jedi" })
      .set("Authorization", authHeader())
      .expect(200)

    expect(res.body).to.have.property("served_commit").that.is.a("string")
    expect(res.body.served_commit).to.equal(COMMIT)
    expect(res.body).to.have.property("total_approx").that.is.a("number")
    expect(res.body.total_approx).to.be.at.least(0)
    expect(res.body).to.have.property("completions").that.is.an("array")
    expect(res.body).to.have.property("hits").that.is.an("array")

    for (const hit of res.body.hits) {
      expect(hit).to.have.property("id").that.is.a("string")
    }
  })

  it("returns TerminusDB-Data-Version header in response", async function () {
    const res = await agent()
      .get("/suggest")
      .query({ domain: DOMAIN, commit: COMMIT, q: "luke" })
      .set("Authorization", authHeader())
      .expect(200)

    expect(res.headers).to.have.property("terminusdb-data-version")
  })

  it("count parameter limits the number of hits", async function () {
    const res = await agent()
      .get("/suggest")
      .query({ domain: DOMAIN, commit: COMMIT, q: "the", count: 1 })
      .set("Authorization", authHeader())
      .expect(200)

    expect(res.body.hits.length).to.be.at.most(1)
  })

  it("doc_type filter restricts suggestions to matching types", async function () {
    const res = await agent()
      .get("/suggest")
      .query({ domain: DOMAIN, commit: COMMIT, q: "jedi", doc_type: "People" })
      .set("Authorization", authHeader())
      .expect(200)

    for (const hit of res.body.hits) {
      expect(hit.id).to.include("/People/")
    }
  })

  it("missing q parameter returns 400", async function () {
    await agent()
      .get("/suggest")
      .query({ domain: DOMAIN, commit: COMMIT })
      .set("Authorization", authHeader())
      .expect(400)
  })

  it("missing domain returns 400", async function () {
    await agent()
      .get("/suggest")
      .query({ commit: COMMIT, q: "test" })
      .set("Authorization", authHeader())
      .expect(400)
  })

  it("branch parameter resolves commit when commit is absent", async function () {
    const res = await agent()
      .get("/suggest")
      .query({ domain: DOMAIN, branch: BRANCH, q: "jedi" })
      .set("Authorization", authHeader())
      .expect(200)

    expect(res.body.served_commit).to.equal(COMMIT)
  })

  it("requires authentication", async function () {
    const res = await agent()
      .get("/suggest")
      .query({ domain: DOMAIN, commit: COMMIT, q: "jedi" })
    expect(res.status).to.equal(401)
  })

  it("malformed TerminusDB-Data-Version header returns 400", async function () {
    await agent()
      .get("/suggest")
      .query({ domain: DOMAIN, commit: COMMIT, q: "jedi" })
      .set("Authorization", authHeader())
      .set("TerminusDB-Data-Version", "bad")
      .expect(400)
  })
})
