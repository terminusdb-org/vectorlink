/**
 * P1-VAL-* and P1-GS-* — Parameter validation and graphspec tests (Phase 1).
 */

const { expect } = require("chai")
const { agent, authHeader } = require("../lib/agent")

describe("P1-VAL: Parameter validation", function () {
  // P1-VAL-1: Missing required params -> 400 naming the param.
  describe("P1-VAL-1: missing domain/commit/q on /search -> 400", function () {
    it("GET /search without domain returns 400 naming domain", async function () {
      const res = await agent()
        .get("/search")
        .query({ commit: "c1", q: "test" })
        .set("Authorization", authHeader())
        .expect(400)

      expect(res.body.error).to.match(/domain/i)
    })

    it("GET /search without commit returns 400 naming commit", async function () {
      const res = await agent()
        .get("/search")
        .query({ domain: "admin/db", q: "test" })
        .set("Authorization", authHeader())
        .expect(400)

      expect(res.body.error).to.match(/commit/i)
    })

    it("GET /search without q returns 400 naming q", async function () {
      const res = await agent()
        .get("/search")
        .query({ domain: "admin/db", commit: "c1" })
        .set("Authorization", authHeader())
        .expect(400)

      expect(res.body.error).to.match(/q/i)
    })

    it("POST /search with empty body and no query params returns 400", async function () {
      const res = await agent()
        .post("/search")
        .set("Authorization", authHeader())
        .send({})
        .expect(400)

      expect(res.body.error).to.match(/domain/i)
    })
  })

  // P1-VAL-2: Invalid mode/start/count -> 400.
  describe("P1-VAL-2: invalid mode/start/count -> 400", function () {
    it("unknown mode returns 400 naming mode", async function () {
      const res = await agent()
        .get("/search")
        .query({ domain: "admin/db", commit: "c1", q: "test", mode: "unknown" })
        .set("Authorization", authHeader())
        .expect(400)

      expect(res.body.error).to.match(/mode/i)
    })

    it("negative start returns 400", async function () {
      const res = await agent()
        .get("/search")
        .query({ domain: "admin/db", commit: "c1", q: "test", start: -1 })
        .set("Authorization", authHeader())
        .expect(400)

      expect(res.body.error).to.match(/start/i)
    })

    it("count < 1 returns 400", async function () {
      const res = await agent()
        .get("/search")
        .query({ domain: "admin/db", commit: "c1", q: "test", count: 0 })
        .set("Authorization", authHeader())
        .expect(400)

      expect(res.body.error).to.match(/count/i)
    })
  })

  // P1-VAL-3: Malformed TerminusDB-Data-Version header -> 400.
  describe("P1-VAL-3: malformed TerminusDB-Data-Version -> 400", function () {
    it("short label part returns 400", async function () {
      const res = await agent()
        .get("/search")
        .query({ domain: "admin/db", commit: "c1", q: "test" })
        .set("Authorization", authHeader())
        .set("TerminusDB-Data-Version", "co:abc123def456")
        .expect(400)

      expect(res.body.error).to.match(/data-version/i)
    })

    it("no colon returns 400", async function () {
      const res = await agent()
        .get("/search")
        .query({ domain: "admin/db", commit: "c1", q: "test" })
        .set("Authorization", authHeader())
        .set("TerminusDB-Data-Version", "commitabc123")
        .expect(400)

      expect(res.body.error).to.match(/data-version/i)
    })

    it("valid header passes through", async function () {
      // This should not return 400 for the header (may return 503 for cold search).
      const res = await agent()
        .get("/search")
        .query({ domain: "admin/db", commit: "c1", q: "test" })
        .set("Authorization", authHeader())
        .set("TerminusDB-Data-Version", "commit:abc123def456")

      // Should NOT be 400 for header — it may be 503 (search not ready) or 200.
      expect(res.status).to.not.equal(400)
    })
  })
})

describe("P1-GS: Graphspec normalisation", function () {
  // P1-GS-1: Domain normalisation is tested at the unit level (Rust tests).
  // At HTTP level, verify that short-form domains are accepted.
  describe("P1-GS-1: short-form domain accepted", function () {
    it("org/db form is accepted (not rejected as invalid)", async function () {
      const res = await agent()
        .get("/last-indexed")
        .query({ domain: "admin/star_wars", branch: "main" })
        .set("Authorization", authHeader())

      expect(res.status).to.equal(200)
    })

    it("org/db/repo/branch/name form is accepted", async function () {
      const res = await agent()
        .get("/last-indexed")
        .query({ domain: "admin/star_wars/local/branch/dev", branch: "dev" })
        .set("Authorization", authHeader())

      expect(res.status).to.equal(200)
    })
  })

  // P1-GS-2: No implicit latest — search without explicit commit -> 400.
  describe("P1-GS-2: no implicit latest", function () {
    it("search without commit returns 400", async function () {
      const res = await agent()
        .get("/search")
        .query({ domain: "admin/db", q: "test" })
        .set("Authorization", authHeader())
        .expect(400)

      expect(res.body.error).to.match(/commit/i)
    })
  })
})
