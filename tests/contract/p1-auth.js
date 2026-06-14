/**
 * P1-AUTH-* — Admin secret gate tests.
 * Verify authentication enforcement on all endpoints.
 */

const { expect } = require("chai")
const { agent, wrongAuthHeader } = require("../lib/agent")

describe("P1-AUTH: Admin secret gate", function () {
  const functionalEndpoints = [
    { method: "get", path: "/last-indexed", query: { domain: "admin/db", branch: "main" } },
    { method: "get", path: "/search", query: { domain: "admin/db", commit: "c1", q: "test" } },
    { method: "post", path: "/search", query: {}, body: { domain: "admin/db", commit: "c1", q: "test" } },
    { method: "get", path: "/similar", query: { domain: "admin/db", commit: "c1", id: "x" } },
    { method: "post", path: "/similar", query: {}, body: { domain: "admin/db", commit: "c1", id: "x" } },
    { method: "get", path: "/duplicates", query: { domain: "admin/db", commit: "c1" } },
    { method: "get", path: "/statistics", query: {} },
    { method: "get", path: "/check", query: { task_id: "fake" } },
    { method: "post", path: "/push", query: { domain: "admin/db", branch: "main", target_commit: "c1" } },
    { method: "post", path: "/assign", query: { domain: "admin/db", source_commit: "c0", target_commit: "c1" } },
  ]

  // P1-AUTH-1: No Authorization header -> 401
  describe("P1-AUTH-1: no Authorization header -> 401", function () {
    functionalEndpoints.forEach(({ method, path, query, body }) => {
      it(`${method.toUpperCase()} ${path} without auth returns 401`, async function () {
        let req = agent()[method](path).query(query)
        if (body) {
          req = req.send(body)
        }
        const res = await req.expect(401)
        expect(res.body).to.have.property("error")
      })
    })
  })

  // P1-AUTH-2: Wrong secret -> 401
  describe("P1-AUTH-2: wrong secret -> 401", function () {
    functionalEndpoints.forEach(({ method, path, query, body }) => {
      it(`${method.toUpperCase()} ${path} with wrong secret returns 401`, async function () {
        let req = agent()[method](path)
          .query(query)
          .set("Authorization", wrongAuthHeader())
        if (body) {
          req = req.send(body)
        }
        const res = await req.expect(401)
        expect(res.body).to.have.property("error")
      })
    })
  })

  // P1-AUTH-3: Health probes unauthenticated -> never 401
  describe("P1-AUTH-3: health probes unauthenticated", function () {
    it("GET /health/live without auth returns 200", async function () {
      const res = await agent()
        .get("/health/live")
        .expect(200)

      expect(res.body).to.have.property("status", "ok")
    })

    it("GET /health/ready without auth returns 200 or 503 (never 401)", async function () {
      const res = await agent()
        .get("/health/ready")

      expect([200, 503]).to.include(res.status)
      expect(res.body).to.have.property("ready")
      expect(res.body).to.have.property("index")
      expect(res.body).to.have.property("search")
    })
  })
})
