/**
 * E2E tests for compress_ids parameter on vector search endpoints.
 *
 * These tests hit the TerminusDB server (port 7373) which proxies to vectorlink
 * (port 7372). A database is created on TerminusDB with store_indices enabled;
 * documents are inserted through the TerminusDB document API and the
 * auto-indexer pushes them to vectorlink — the same path production uses.
 *
 * vectorlink always stores and returns full IRIs — it does not understand
 * prefixes. TerminusDB handles compression/decompression of IDs in search
 * responses via maybe_compact_response/3 and maybe_prefixes/3.
 *
 * Requires:
 *   - TerminusDB running on port 7373 with vectorlink indexer backend
 *   - vectorlink running on port 7372
 *   - Ollama running with an embedding model
 */

const { expect } = require("chai")
const http = require("http")
const { agent, authHeader } = require("../lib/agent")

const TDB_AUTH = "Basic " + Buffer.from("admin:root").toString("base64")
const TDB_HOST = "127.0.0.1"
const TDB_PORT = 7373
const DB_NAME = "compress_ids_e2e"
const ORG = "admin"
const DOMAIN = `${ORG}/${DB_NAME}`

function tdbRequest (method, pathWithQuery, body) {
  return new Promise((resolve, reject) => {
    const data = body ? JSON.stringify(body) : null
    const opts = {
      hostname: TDB_HOST,
      port: TDB_PORT,
      path: pathWithQuery,
      method,
      headers: {
        Authorization: TDB_AUTH,
        "Content-Type": "application/json",
      },
    }
    if (data) opts.headers["Content-Length"] = Buffer.byteLength(data)
    const req = http.request(opts, (res) => {
      let raw = ""
      res.on("data", (c) => { raw += c })
      res.on("end", () => {
        let parsed
        try { parsed = JSON.parse(raw) } catch { parsed = raw }
        resolve({ status: res.statusCode, body: parsed, text: raw })
      })
    })
    req.on("error", reject)
    if (data) req.write(data)
    req.end()
  })
}

// Poll the TerminusDB indexing-status endpoint until the auto-indexer has
// caught up with every commit (or fail after 90s). The engine receives the
// rendered embedding text — we never push to vectorlink directly, because the
// auto-indexer claims every commit and a second push is a 409.
async function waitForIndexed (org, db) {
  const start = Date.now()
  while (Date.now() - start < 90000) {
    const res = await tdbRequest("GET", `/api/index/${org}/${db}`)
    if (res.status === 200 && res.body.status === "completed") {
      const bp = res.body.branch_processing || {}
      if (bp.commits_processed === bp.total_commits) {
        return res.body
      }
    }
    if (res.body?.error) {
      throw new Error(`indexer reported error: ${JSON.stringify(res.body.error)}`)
    }
    await new Promise((resolve) => setTimeout(resolve, 1000))
  }
  throw new Error("auto-indexer did not complete within 90000ms")
}

// Test documents inserted through the TerminusDB document API. The class's
// embedding template renders each doc to "{{title}}. {{content}}" — that
// string is what the engine embeds.
const testDocs = [
  {
    "@type": "Article",
    title: "WirelessHeadphones",
    content: "Premium wireless noise-cancelling headphones with 30-hour battery life and active noise reduction technology",
  },
  {
    "@type": "Article",
    title: "TeaCeremony",
    content: "The ancient art of Japanese tea ceremony emphasises mindfulness and precise ceremonial movements",
  },
  {
    "@type": "Article",
    title: "ConstructionGear",
    content: "Industrial grade concrete mixing equipment for large construction projects and infrastructure development",
  },
]

describe("compress_ids on vector search endpoints — E2E with live vectorlink", function () {
  this.timeout(120000)

  before(async function () {
    // Clean up any existing vectorlink domain and TerminusDB database
    await agent().delete("/domain").query({ domain: DOMAIN }).set("Authorization", authHeader())
    await tdbRequest("DELETE", `/api/db/${ORG}/${DB_NAME}`)

    // Create database on TerminusDB
    const createRes = await tdbRequest("POST", `/api/db/${ORG}/${DB_NAME}`, {
      label: "compress_ids E2E",
      comment: "Test database for compress_ids E2E tests",
    })
    expect(createRes.status).to.equal(200)

    // Insert a schema with store_indices enabled (context @metadata) and an
    // embedding query/template on Article — this is what turns the
    // auto-indexer on and decides which fields get embedded.
    const schemaRes = await tdbRequest(
      "POST",
      `/api/document/${ORG}/${DB_NAME}?graph_type=schema&author=test&message=insert+schema&full_replace=true`,
      [
        {
          "@type": "@context",
          "@base": "terminusdb:///data/",
          "@schema": "terminusdb:///schema#",
          "@metadata": { terminusdb: { options: ["store_indices"] } },
        },
        {
          "@type": "Class",
          "@id": "Article",
          "@key": { "@type": "Lexical", "@fields": ["title"] },
          "@metadata": {
            embedding: {
              query: "query($id: ID){ Article(id: $id) { title content } }",
              template: "{{title}}. {{content}}",
            },
          },
          title: "xsd:string",
          content: "xsd:string",
        },
      ],
    )
    expect(schemaRes.status).to.equal(200)

    // Insert the test documents — the post_commit_hook pushes the rendered
    // embedding strings to vectorlink automatically.
    const docsRes = await tdbRequest(
      "POST",
      `/api/document/${ORG}/${DB_NAME}?author=test&message=insert+articles`,
      testDocs,
    )
    expect(docsRes.status).to.equal(200)

    // Wait for the asynchronous indexer to catch up on every commit.
    await waitForIndexed(ORG, DB_NAME)
  })

  after(async function () {
    await agent().delete("/domain").query({ domain: DOMAIN }).set("Authorization", authHeader())
    await tdbRequest("DELETE", `/api/db/${ORG}/${DB_NAME}`)
  })

  it("GET /api/search returns compressed IDs when compress_ids=true", async function () {
    const res = await tdbRequest(
      "GET",
      `/api/search/${ORG}/${DB_NAME}?q=wireless+headphones&compress_ids=true`,
    )
    expect(res.status).to.equal(200)
    expect(res.body).to.be.an("array")
    expect(res.body.length).to.be.greaterThan(0)
    const firstId = res.body[0].id
    expect(firstId).to.match(/^Article\//)
    expect(firstId).to.not.include("terminusdb:///")
  })

  it("GET /api/search returns full IRIs when compress_ids=false", async function () {
    const res = await tdbRequest(
      "GET",
      `/api/search/${ORG}/${DB_NAME}?q=wireless+headphones&compress_ids=false`,
    )
    expect(res.status).to.equal(200)
    expect(res.body).to.be.an("array")
    expect(res.body.length).to.be.greaterThan(0)
    const firstId = res.body[0].id
    expect(firstId).to.include("terminusdb:///data/Article/")
    expect(firstId).to.not.match(/^Article\//)
  })

  it("GET /api/search returns compressed IDs when compress_ids defaults (no param)", async function () {
    const res = await tdbRequest(
      "GET",
      `/api/search/${ORG}/${DB_NAME}?q=wireless+headphones`,
    )
    expect(res.status).to.equal(200)
    expect(res.body).to.be.an("array")
    expect(res.body.length).to.be.greaterThan(0)
    const firstId = res.body[0].id
    expect(firstId).to.match(/^Article\//)
  })

  it("GET /api/search rejects invalid compress_ids with 400 BadParameterType", async function () {
    const res = await tdbRequest(
      "GET",
      `/api/search/${ORG}/${DB_NAME}?q=wireless+headphones&compress_ids=notabool`,
    )
    expect(res.status).to.equal(400)
    expect(res.body["api:error"]).to.have.property("@type", "api:BadParameterType")
  })

  it("POST /api/similar returns compressed IDs when compress_ids=true", async function () {
    const res = await tdbRequest(
      "POST",
      `/api/similar/${ORG}/${DB_NAME}?compress_ids=true`,
      { text: "wireless headphones" },
    )
    expect(res.status).to.equal(200)
    expect(res.body).to.be.an("array")
    expect(res.body.length).to.be.greaterThan(0)
    const firstId = res.body[0].id
    expect(firstId).to.match(/^Article\//)
    expect(firstId).to.not.include("terminusdb:///")
  })

  it("POST /api/similar returns full IRIs when compress_ids=false", async function () {
    const res = await tdbRequest(
      "POST",
      `/api/similar/${ORG}/${DB_NAME}?compress_ids=false`,
      { text: "wireless headphones" },
    )
    expect(res.status).to.equal(200)
    expect(res.body).to.be.an("array")
    expect(res.body.length).to.be.greaterThan(0)
    const firstId = res.body[0].id
    expect(firstId).to.include("terminusdb:///data/Article/")
    expect(firstId).to.not.match(/^Article\//)
  })

  it("POST /api/similar rejects invalid compress_ids with 400 BadParameterType", async function () {
    const res = await tdbRequest(
      "POST",
      `/api/similar/${ORG}/${DB_NAME}?compress_ids=notabool`,
      { text: "wireless headphones" },
    )
    expect(res.status).to.equal(400)
    expect(res.body["api:error"]).to.have.property("@type", "api:BadParameterType")
  })
})
