/**
 * Contract tests for POST /embeddings — batch JSON, streaming NDJSON, and
 * bidirectional NDJSON modes.
 *
 * Also covers GET /embeddings with Accept: application/x-ndjson directly
 * against tdb-search (not through the TerminusDB proxy).
 */

const { expect } = require("chai")
const http = require("http")
const { agent, authHeader } = require("../lib/agent")

const BASE_URL = process.env.TDB_SEARCH_URL || "http://localhost:7372"

function rawNdjsonRequest (path, method, body, headers) {
  return new Promise((resolve, reject) => {
    const url = new URL(path, BASE_URL)
    const opts = {
      hostname: url.hostname,
      port: url.port,
      path: url.pathname + url.search,
      method,
      headers: Object.assign(
        { Authorization: authHeader() },
        headers || {},
      ),
    }
    if (body) {
      opts.headers["Content-Length"] = Buffer.byteLength(body)
    }
    const req = http.request(opts, (res) => {
      let raw = ""
      res.on("data", (c) => { raw += c })
      res.on("end", () => {
        const lines = raw.split("\n").filter((l) => l.trim().length > 0)
        const parsed = []
        for (const line of lines) {
          try { parsed.push(JSON.parse(line)) } catch { /* skip */ }
        }
        let json
        try { json = JSON.parse(raw) } catch { /* not JSON */ }
        resolve({ status: res.statusCode, body: json, text: raw, lines: parsed, headers: res.headers })
      })
    })
    req.on("error", reject)
    if (body) req.write(body)
    req.end()
  })
}

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

describe("POST /embeddings — batch and streaming modes", function () {
  this.timeout(120000)

  const DOMAIN = "admin/embeddings_post_test"
  const BRANCH = "main"
  const COMMIT = "ep_c0"

  before(async function () {
    await agent().delete("/domain").query({ domain: DOMAIN }).set("Authorization", authHeader())
    await pushAndWait(DOMAIN, BRANCH, COMMIT, [
      {
        op: "Inserted",
        id: "terminusdb:///ep/Product/A",
        string: "Premium wireless noise-cancelling headphones with 30-hour battery life",
      },
      {
        op: "Inserted",
        id: "terminusdb:///ep/Product/B",
        string: "The ancient art of Japanese tea ceremony emphasises mindfulness",
      },
      {
        op: "Inserted",
        id: "terminusdb:///ep/Catalogue/X",
        string: "High-end wireless headphones featuring noise cancellation",
      },
    ])
  })

  after(async function () {
    await agent().delete("/domain").query({ domain: DOMAIN }).set("Authorization", authHeader())
  })

  it("batch mode (JSON body, stream=false) returns JSON with doc_embeddings", async function () {
    const res = await agent()
      .post("/embeddings")
      .set("Authorization", authHeader())
      .send({ domain: DOMAIN, commit: COMMIT })
      .expect(200)

    expect(res.body).to.have.property("doc_embeddings").that.is.an("object")
    expect(res.body).to.have.property("clustering_embeddings").that.is.an("object")
    expect(res.body).to.have.property("store_clustering").that.is.a("boolean")
    expect(res.body).to.have.property("served_commit").that.is.a("string")
    expect(Object.keys(res.body.doc_embeddings).length).to.be.greaterThan(0)
  })

  it("batch mode with doc_ids filter returns only requested documents", async function () {
    const res = await agent()
      .post("/embeddings")
      .set("Authorization", authHeader())
      .send({ domain: DOMAIN, commit: COMMIT, doc_ids: ["terminusdb:///ep/Product/A"] })
      .expect(200)

    expect(res.body).to.have.property("doc_embeddings").that.is.an("object")
    const ids = Object.keys(res.body.doc_embeddings)
    expect(ids).to.include("terminusdb:///ep/Product/A")
    expect(ids.length).to.equal(1)
  })

  it("batch mode with doc_types filter returns only matching types", async function () {
    const res = await agent()
      .post("/embeddings")
      .set("Authorization", authHeader())
      .send({ domain: DOMAIN, commit: COMMIT, doc_types: ["Product"] })
      .expect(200)

    const ids = Object.keys(res.body.doc_embeddings)
    for (const id of ids) {
      expect(id).to.include("/Product/")
    }
  })

  it("streaming mode (JSON body, stream=true) returns NDJSON lines", async function () {
    const res = await rawNdjsonRequest(
      "/embeddings",
      "POST",
      JSON.stringify({ domain: DOMAIN, commit: COMMIT, stream: true }),
      { "Content-Type": "application/json" },
    )

    expect(res.status).to.equal(200)
    expect(res.headers["content-type"]).to.include("application/x-ndjson")
    expect(res.headers["x-served-commit"]).to.be.a("string")
    expect(res.headers).to.have.property("x-store-clustering")
    expect(res.headers).to.have.property("x-total-count")
    expect(res.lines.length).to.be.greaterThan(0)
    for (const line of res.lines) {
      expect(line).to.have.property("doc_id")
      expect(line).to.have.property("embedding")
    }
  })

  it("bidirectional NDJSON mode streams embeddings per doc_id", async function () {
    const ndjsonBody = [
      JSON.stringify({ domain: DOMAIN, commit: COMMIT }),
      JSON.stringify({ domain: DOMAIN, commit: COMMIT, doc_id: "terminusdb:///ep/Product/A" }),
      JSON.stringify({ domain: DOMAIN, commit: COMMIT, doc_id: "terminusdb:///ep/Product/B" }),
    ].join("\n")

    const res = await rawNdjsonRequest(
      "/embeddings",
      "POST",
      ndjsonBody,
      { "Content-Type": "application/x-ndjson" },
    )

    expect(res.status).to.equal(200)
    expect(res.headers["content-type"]).to.include("application/x-ndjson")
    expect(res.lines.length).to.be.greaterThan(0)
    for (const line of res.lines) {
      expect(line).to.have.property("doc_id")
      expect(line).to.have.property("embedding")
    }
  })

  it("missing domain in JSON body returns 400", async function () {
    const res = await agent()
      .post("/embeddings")
      .set("Authorization", authHeader())
      .send({ commit: COMMIT })
    expect(res.status).to.equal(400)
  })

  it("missing commit and branch resolves to latest indexed commit", async function () {
    const res = await agent()
      .post("/embeddings")
      .set("Authorization", authHeader())
      .send({ domain: DOMAIN, branch: BRANCH })
      .expect(200)

    expect(res.body).to.have.property("doc_embeddings").that.is.an("object")
    expect(res.body.served_commit).to.equal(COMMIT)
  })

  it("requires authentication", async function () {
    const res = await agent()
      .post("/embeddings")
      .send({ domain: DOMAIN, commit: COMMIT })
    expect(res.status).to.equal(401)
  })
})

describe("GET /embeddings with Accept: application/x-ndjson (direct)", function () {
  this.timeout(120000)

  const DOMAIN = "admin/embeddings_get_ndjson_test"
  const BRANCH = "main"
  const COMMIT = "eg_c0"

  before(async function () {
    await agent().delete("/domain").query({ domain: DOMAIN }).set("Authorization", authHeader())
    await pushAndWait(DOMAIN, BRANCH, COMMIT, [
      {
        op: "Inserted",
        id: "terminusdb:///eg/Doc/1",
        string: "First document about semantic search",
      },
      {
        op: "Inserted",
        id: "terminusdb:///eg/Doc/2",
        string: "Second document about vector embeddings",
      },
    ])
  })

  after(async function () {
    await agent().delete("/domain").query({ domain: DOMAIN }).set("Authorization", authHeader())
  })

  it("returns NDJSON stream with correct headers", async function () {
    const res = await rawNdjsonRequest(
      `/embeddings?domain=${encodeURIComponent(DOMAIN)}&commit=${COMMIT}`,
      "GET",
      null,
      { Accept: "application/x-ndjson" },
    )

    expect(res.status).to.equal(200)
    expect(res.headers["content-type"]).to.include("application/x-ndjson")
    expect(res.headers["x-served-commit"]).to.be.a("string")
    expect(res.headers).to.have.property("x-store-clustering")
    expect(res.headers).to.have.property("x-total-count")
    expect(res.lines.length).to.be.greaterThan(0)
    for (const line of res.lines) {
      expect(line).to.have.property("doc_id")
      expect(line).to.have.property("embedding")
      expect(line.embedding).to.be.an("array").with.length.at.least(1)
    }
  })

  it("returns JSON array when Accept header is application/json", async function () {
    const res = await agent()
      .get("/embeddings")
      .query({ domain: DOMAIN, commit: COMMIT })
      .set("Authorization", authHeader())
      .set("Accept", "application/json")
      .expect(200)

    expect(res.body).to.have.property("doc_embeddings").that.is.an("object")
  })
})
