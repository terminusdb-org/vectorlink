/**
 * P1-CON-* — Contract shape tests.
 * Verify every endpoint returns the correct openapi shapes against the stub.
 */

const { expect } = require("chai")
const { agent, authHeader } = require("../lib/agent")

describe("P1-CON: Contract shapes", function () {
  // P1-CON-1: GET /last-indexed
  describe("P1-CON-1: GET /last-indexed", function () {
    it("returns 200 with LastIndexed shape {branch, commit, version}", async function () {
      const res = await agent()
        .get("/last-indexed")
        .query({ domain: "admin/star_wars", branch: "main" })
        .set("Authorization", authHeader())
        .expect(200)

      expect(res.body).to.have.property("branch", "main")
      expect(res.body).to.have.property("commit")
      expect(res.body.commit).to.equal(null)
      expect(res.body).to.have.property("version")
      expect(res.body.version).to.be.a("number")
    })
  })

  // P1-CON-2: POST /push
  describe("P1-CON-2: POST /push", function () {
    it("returns 200 with a non-empty text task id", async function () {
      const ndjson = [
        JSON.stringify({ op: "Inserted", id: "terminusdb:///test/Doc/1", string: "Hello world" }),
        JSON.stringify({ op: "Deleted", id: "terminusdb:///test/Doc/2" }),
      ].join("\n")

      const res = await agent()
        .post("/push")
        .query({ domain: "admin/star_wars", branch: "main", target_commit: "c1" })
        .set("Authorization", authHeader())
        .set("Content-Type", "application/x-ndjson")
        .send(ndjson)
        .expect(200)

      expect(res.text).to.be.a("string")
      expect(res.text.length).to.be.greaterThan(0)
    })
  })

  // P1-CON-3: GET /check
  describe("P1-CON-3: GET /check", function () {
    it("returns 200 with TaskPending or TaskComplete shape", async function () {
      // First push to get a task id.
      const pushRes = await agent()
        .post("/push")
        .query({ domain: "admin/db", branch: "main", target_commit: "check-test" })
        .set("Authorization", authHeader())
        .set("Content-Type", "application/x-ndjson")
        .send(JSON.stringify({ op: "Inserted", id: "terminusdb:///test/Doc/1", string: "test" }))
        .expect(200)

      const taskId = pushRes.text

      const res = await agent()
        .get("/check")
        .query({ task_id: taskId })
        .set("Authorization", authHeader())
        .expect(200)

      // Must be TaskPending or TaskComplete.
      expect(res.body).to.have.property("status")
      expect(["Pending", "Complete"]).to.include(res.body.status)

      if (res.body.status === "Pending") {
        expect(res.body).to.have.property("percentage")
        expect(res.body.percentage).to.be.a("number")
      } else {
        expect(res.body).to.have.property("indexed_documents")
        expect(res.body.indexed_documents).to.be.a("number")
        expect(res.body).to.have.property("skipped")
        expect(res.body.skipped).to.be.an("array")
      }
    })

    it("returns 404 for unknown task id", async function () {
      const res = await agent()
        .get("/check")
        .query({ task_id: "nonexistent-task-id" })
        .set("Authorization", authHeader())
        .expect(404)

      expect(res.body).to.have.property("error")
    })
  })

  // P1-CON-4: GET /search
  // Real engine: 200 = success (exact or stale ancestor), 404 = no indexed
  // lineage (the documented NoIndexedLineage contract response for an unindexed
  // commit on a domain that was never pushed), 503 = search cold.
  describe("P1-CON-4: GET /search", function () {
    it("returns 200 (array), 404 (no indexed lineage), or 503 (search cold)", async function () {
      const res = await agent()
        .get("/search")
        .query({ domain: "admin/star_wars", commit: "abc123", q: "wise" })
        .set("Authorization", authHeader())

      expect([200, 404, 503]).to.include(res.status)
      if (res.status === 200) {
        expect(res.body).to.be.an("array")
      } else if (res.status === 503) {
        expect(res.body).to.have.property("error")
        expect(res.headers).to.have.property("retry-after")
      }
    })
  })

  // P1-CON-5: POST /search with body overriding query params
  describe("P1-CON-5: POST /search", function () {
    it("returns 200, 404, or 503 with JSON body overriding query params", async function () {
      const res = await agent()
        .post("/search")
        .query({ domain: "admin/ignored", commit: "ignored", q: "ignored" })
        .set("Authorization", authHeader())
        .send({
          domain: "admin/star_wars",
          commit: "abc123",
          q: "wise old man",
        })

      expect([200, 404, 503]).to.include(res.status)
      if (res.status === 200) {
        expect(res.body).to.be.an("array")
      }
    })

    it("body field overrides same-named query param (precedence test)", async function () {
      // With an invalid domain in query but valid in body, should not get
      // a validation error for the query-param domain.
      const res = await agent()
        .post("/search")
        .query({ domain: "x", commit: "y", q: "z" })
        .set("Authorization", authHeader())
        .send({
          domain: "admin/star_wars",
          commit: "abc123",
          q: "wise old man",
        })

      // Should not be 400 for invalid domain (body wins over query); 404 is the
      // valid no-indexed-lineage response for the unindexed body domain/commit.
      expect([200, 404, 503]).to.include(res.status)
    })
  })

  // P1-CON-6: POST /assign
  describe("P1-CON-6: POST /assign", function () {
    it("returns 204 or 404 (source not indexed) for assign", async function () {
      const res = await agent()
        .post("/assign")
        .query({
          domain: "admin/star_wars",
          source_commit: "c0",
          target_commit: "c1",
        })
        .set("Authorization", authHeader())

      // 204 = success (source commit was indexed).
      // 404 = source commit not indexed (real store, no prior push).
      expect([204, 404]).to.include(res.status)
    })

    it("GET /assign is not routed (404 or 405)", async function () {
      const res = await agent()
        .get("/assign")
        .set("Authorization", authHeader())

      // axum returns 405 for wrong method if route exists, 404 if not.
      expect([404, 405]).to.include(res.status)
    })
  })

  // P1-CON-7: GET /statistics
  describe("P1-CON-7: GET /statistics", function () {
    it("returns 200 with {domains, branches, indexed_commits, documents, chunks}", async function () {
      const res = await agent()
        .get("/statistics")
        .set("Authorization", authHeader())
        .expect(200)

      expect(res.body).to.have.property("domains")
      expect(res.body).to.have.property("branches")
      expect(res.body).to.have.property("indexed_commits")
      expect(res.body).to.have.property("documents")
      expect(res.body).to.have.property("chunks")
      // Must NOT have arena/lance keys.
      expect(res.body).to.not.have.property("arena")
      expect(res.body).to.not.have.property("lance")
    })
  })

  // P1-CON-8: GET /similar and GET /duplicates
  describe("P1-CON-8: GET /similar and GET /duplicates", function () {
    it("/similar returns 200 (array), 404 (doc not found), or 500 (commit not indexed)", async function () {
      const res = await agent()
        .get("/similar")
        .query({ domain: "admin/star_wars", commit: "abc123", id: "terminusdb:///test/Doc/1" })
        .set("Authorization", authHeader())

      // 200 = success (doc found). 404 = doc not in index. 500 = commit not indexed.
      expect([200, 404, 500]).to.include(res.status)
      if (res.status === 200) {
        expect(res.body).to.be.an("array")
      }
    })

    it("/duplicates returns 200 (array) or a non-2xx for an unresolvable commit", async function () {
      const res = await agent()
        .get("/duplicates")
        .query({ domain: "admin/star_wars", commit: "abc123" })
        .set("Authorization", authHeader())

      // /duplicates resolves the commit via the SAME catch-up path as /search
      // and /similar, so an unresolvable commit yields 404 (no indexed lineage)
      // or 503 (search backend cold), and 200 when an ancestor resolves. Assert
      // "resolved or declined", not one fixed code.
      expect([200, 404, 503]).to.include(res.status)
      if (res.status === 200) {
        expect(res.body).to.be.an("array")
      }
    })
  })
})
