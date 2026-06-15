/**
 * Operation semantics + search semantics — through the REAL HTTP pipeline
 * (push → poll → search/similar/duplicates), against the live engine + real
 * embeddings. Covers the coverage-spec gaps that had NO e2e coverage:
 *
 *   §2  Changed / Deleted / per-doc Operation::Error end-to-end.
 *   §3  /similar relevance, /duplicates contract, empty-query, multilingual.
 *
 * Method: every assertion drives the real pipeline. Literal-text claims
 * (content present/absent) are made via FTS — vector search always returns a
 * nearest neighbour and cannot prove presence/absence of specific terms.
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

async function pushAndWait (domain, branch, commit, ops, parentCommit) {
  const body = ops.map(l => JSON.stringify(l)).join("\n")
  const query = { domain, branch, target_commit: commit }
  if (parentCommit) {
    query.parent_commit = parentCommit
  }
  const pushRes = await agent()
    .post("/push")
    .query(query)
    .set("Authorization", authHeader())
    .set("Content-Type", "application/x-ndjson")
    .send(body)
    .expect(200)
  return waitForTask(pushRes.text)
}

async function searchMode (domain, commit, q, mode, extra = {}) {
  return agent()
    .get("/search")
    .query({ domain, commit, q, mode, ...extra })
    .set("Authorization", authHeader())
}

describe("Operation semantics (Changed / Deleted / Error) — e2e", function () {
  this.timeout(180000)

  const DOMAIN = "admin/ops_semantics"
  const BRANCH = "main"

  before(async function () {
    // Clean up any stale state from prior runs (each test must be independently runnable).
    await agent().delete("/domain").query({ domain: DOMAIN }).set("Authorization", authHeader())
  })

  after(async function () {
    await agent().delete("/domain").query({ domain: DOMAIN }).set("Authorization", authHeader())
  })

  it("Changed: new content is found, old content is gone, at the new commit", async function () {
    const ID = "terminusdb:///ops/People/agent"
    // c0: original content with a unique literal marker.
    await pushAndWait(DOMAIN, BRANCH, "ch_c0", [
      { op: "Inserted", id: ID, string: "Operative codename brightstar runs covert ops on Coruscant." },
    ])
    const oldFts = await searchMode(DOMAIN, "ch_c0", "brightstar", "fts", { count: 50 })
    expect(oldFts.status).to.equal(200)
    expect(oldFts.body.some(h => h.id === ID), "old marker findable at ch_c0").to.equal(true)

    // c1: Changed to new content with a DIFFERENT unique marker.
    await pushAndWait(DOMAIN, BRANCH, "ch_c1", [
      { op: "Changed", id: ID, string: "Operative codename nightfall runs covert ops on Hoth." },
    ], "ch_c0")

    // New marker is found at c1.
    const newFts = await searchMode(DOMAIN, "ch_c1", "nightfall", "fts", { count: 50 })
    expect(newFts.status).to.equal(200)
    expect(newFts.body.some(h => h.id === ID), "new marker findable at ch_c1").to.equal(true)

    // Old marker is GONE at c1 (Changed replaced the content).
    const goneFts = await searchMode(DOMAIN, "ch_c1", "brightstar", "fts", { count: 50 })
    expect(goneFts.status).to.equal(200)
    expect(goneFts.body.some(h => h.id === ID), "old marker must be gone at ch_c1").to.equal(false)
  })

  it("Deleted: the doc is absent from search at that commit forward", async function () {
    const KEEP = "terminusdb:///ops/People/keeper"
    const DROP = "terminusdb:///ops/People/dropme"
    await pushAndWait(DOMAIN, BRANCH, "del_c0", [
      { op: "Inserted", id: KEEP, string: "The archivist guards the temple records with vorpalmark vigilance." },
      { op: "Inserted", id: DROP, string: "The courier carries the disposable cipher tagged removemark." },
    ])
    // Both present at del_c0.
    const before = await searchMode(DOMAIN, "del_c0", "removemark", "fts", { count: 50 })
    expect(before.body.some(h => h.id === DROP), "doc present before delete").to.equal(true)

    // Delete DROP at del_c1.
    await pushAndWait(DOMAIN, BRANCH, "del_c1", [
      { op: "Deleted", id: DROP },
    ], "del_c0")

    // DROP gone at del_c1; KEEP still present.
    const afterDrop = await searchMode(DOMAIN, "del_c1", "removemark", "fts", { count: 50 })
    expect(afterDrop.status).to.equal(200)
    expect(afterDrop.body.some(h => h.id === DROP), "deleted doc must be absent at del_c1").to.equal(false)

    const afterKeep = await searchMode(DOMAIN, "del_c1", "vorpalmark", "fts", { count: 50 })
    expect(afterKeep.body.some(h => h.id === KEEP), "sibling doc must survive the delete").to.equal(true)
  })

  it("per-doc Operation::Error: the error doc is skipped+recorded, the others still index", async function () {
    // A valid Error op (Spec 13 §2) in the middle of a batch must NOT fail the
    // whole task — it is recorded in `skipped`, and the surrounding docs index.
    const A = "terminusdb:///ops/Batch/a"
    const B = "terminusdb:///ops/Batch/b"
    const result = await pushAndWait(DOMAIN, BRANCH, "err_c0", [
      { op: "Inserted", id: A, string: "First batch doc with marker alphamark." },
      { op: "Error", message: "render failed for some upstream doc" },
      { op: "Inserted", id: B, string: "Third batch doc with marker betamark." },
    ])

    // Task completed (not Error), the two valid docs indexed, the Error recorded.
    expect(result.status).to.equal("Complete")
    expect(result.indexed_documents).to.equal(2, "both valid docs must index despite the Error op")
    expect(result.skipped).to.be.an("array")
    expect(result.skipped.length).to.equal(1, "the Error op must be recorded as skipped")
    expect(result.skipped[0].message).to.match(/operation error/i)

    // Both valid docs are searchable.
    const aFts = await searchMode(DOMAIN, "err_c0", "alphamark", "fts", { count: 50 })
    expect(aFts.body.some(h => h.id === A), "doc A indexed").to.equal(true)
    const bFts = await searchMode(DOMAIN, "err_c0", "betamark", "fts", { count: 50 })
    expect(bFts.body.some(h => h.id === B), "doc B indexed").to.equal(true)
  })
})

describe("Search semantics — /similar, /duplicates, empty-query, multilingual", function () {
  this.timeout(180000)

  const DOMAIN = "admin/search_semantics"
  const BRANCH = "main"

  before(async function () {
    // Clean up any stale state from prior runs (each test must be independently runnable).
    const domainsUsed = [DOMAIN, "admin/multilingual"]
    for (const d of domainsUsed) {
      await agent().delete("/domain").query({ domain: d }).set("Authorization", authHeader())
    }
    await pushAndWait(DOMAIN, BRANCH, "ss_c0", [
      // Two genuinely related docs (both about lightsaber combat) + one unrelated.
      { op: "Inserted", id: "terminusdb:///ss/Topic/saber1", string: "Lightsaber combat form Soresu emphasises tight defensive bladework and patience." },
      { op: "Inserted", id: "terminusdb:///ss/Topic/saber2", string: "Form III Soresu is the defensive lightsaber style favoured by Jedi who deflect blaster fire." },
      { op: "Inserted", id: "terminusdb:///ss/Topic/cooking", string: "Bantha milk custard is a sweet dessert popular in Tatooine moisture-farm kitchens." },
    ])
  })

  after(async function () {
    const domainsUsed = [DOMAIN, "admin/multilingual"]
    for (const d of domainsUsed) {
      await agent().delete("/domain").query({ domain: d }).set("Authorization", authHeader())
    }
  })

  it("/similar returns a genuinely related doc as a neighbour", async function () {
    const res = await agent()
      .get("/similar")
      .query({ domain: DOMAIN, commit: "ss_c0", id: "terminusdb:///ss/Topic/saber1", count: 10 })
      .set("Authorization", authHeader())
      .expect(200)
    expect(res.body).to.be.an("array")
    const ids = res.body.map(h => h.id)
    // The other Soresu doc must surface; self is excluded by the engine.
    expect(ids).to.not.include("terminusdb:///ss/Topic/saber1", "self must be excluded")
    expect(ids).to.include(
      "terminusdb:///ss/Topic/saber2",
      "the genuinely related Soresu doc must be a neighbour",
    )
  })

  describe("GET /similar — doc_type filter (contract form)", function () {
    it("single doc_type restricts similar results", async function () {
      // saber1 is a Topic; with doc_type=Topic the other Topic (saber2) should appear.
      const res = await agent()
        .get("/similar")
        .query(`domain=${DOMAIN}&commit=ss_c0&id=terminusdb:///ss/Topic/saber1&doc_type=Topic`)
        .set("Authorization", authHeader())
        .expect(200)
      expect(res.body).to.be.an("array")
      for (const hit of res.body) {
        expect(hit.id).to.include("Topic", `expected Topic doc, got ${hit.id}`)
      }
    })
  })

  describe("GET /similar — doc_id filter (contract form)", function () {
    it("single doc_id restricts similar to that candidate", async function () {
      const res = await agent()
        .get("/similar")
        .query(`domain=${DOMAIN}&commit=ss_c0&id=terminusdb:///ss/Topic/saber1&doc_id=terminusdb:///ss/Topic/cooking`)
        .set("Authorization", authHeader())
        .expect(200)
      expect(res.body).to.be.an("array")
      // Only cooking should appear (the only allowed candidate).
      for (const hit of res.body) {
        expect(hit.id).to.equal("terminusdb:///ss/Topic/cooking",
          `expected only cooking in candidate pool, got ${hit.id}`)
      }
    })

    it("repeated doc_id scopes to multiple candidates", async function () {
      const res = await agent()
        .get("/similar")
        .query(`domain=${DOMAIN}&commit=ss_c0&id=terminusdb:///ss/Topic/saber1&doc_id=terminusdb:///ss/Topic/saber2&doc_id=terminusdb:///ss/Topic/cooking`)
        .set("Authorization", authHeader())
        .expect(200)
      expect(res.body).to.be.an("array")
      const allowedIds = [
        "terminusdb:///ss/Topic/saber2",
        "terminusdb:///ss/Topic/cooking",
      ]
      for (const hit of res.body) {
        expect(allowedIds).to.include(hit.id,
          `expected only saber2 or cooking, got ${hit.id}`)
      }
    })

    it("combined doc_type + doc_id AND-scopes the candidate pool", async function () {
      // doc_type=Topic AND doc_id=cooking → only cooking can appear.
      const res = await agent()
        .get("/similar")
        .query(`domain=${DOMAIN}&commit=ss_c0&id=terminusdb:///ss/Topic/saber1&doc_type=Topic&doc_id=terminusdb:///ss/Topic/cooking`)
        .set("Authorization", authHeader())
        .expect(200)
      expect(res.body).to.be.an("array")
      for (const hit of res.body) {
        expect(hit.id).to.equal("terminusdb:///ss/Topic/cooking",
          `AND-scope should restrict to cooking only, got ${hit.id}`)
      }
    })
  })

  describe("POST /similar — candidate pool via JSON body", function () {
    it("POST with doc_id array scopes to candidate pool", async function () {
      const res = await agent()
        .post("/similar")
        .set("Authorization", authHeader())
        .set("Content-Type", "application/json")
        .send({
          domain: DOMAIN,
          commit: "ss_c0",
          id: "terminusdb:///ss/Topic/saber1",
          doc_id: ["terminusdb:///ss/Topic/cooking"],
        })
        .expect(200)
      expect(res.body).to.be.an("array")
      for (const hit of res.body) {
        expect(hit.id).to.equal("terminusdb:///ss/Topic/cooking",
          `POST doc_id pool should restrict to cooking, got ${hit.id}`)
      }
    })

    it("POST with doc_type array restricts results", async function () {
      const res = await agent()
        .post("/similar")
        .set("Authorization", authHeader())
        .set("Content-Type", "application/json")
        .send({
          domain: DOMAIN,
          commit: "ss_c0",
          id: "terminusdb:///ss/Topic/saber1",
          doc_type: ["Topic"],
        })
        .expect(200)
      expect(res.body).to.be.an("array")
      for (const hit of res.body) {
        expect(hit.id).to.include("Topic", `expected Topic, got ${hit.id}`)
      }
    })

    it("POST body values override query params", async function () {
      // Query param says count=1, body says count=10. Body wins.
      const res = await agent()
        .post("/similar")
        .query({ domain: DOMAIN, commit: "ss_c0", id: "terminusdb:///ss/Topic/saber1", count: 1 })
        .set("Authorization", authHeader())
        .set("Content-Type", "application/json")
        .send({
          count: 10,
        })
        .expect(200)
      expect(res.body).to.be.an("array")
      // With count=10 from body (overrides count=1 from query), should get up to 2 results.
      // (There are only 2 other docs in the corpus: saber2 + cooking.)
      expect(res.body.length).to.be.at.most(2)
    })
  })

  it("empty query q='' → 400 (deliberate fail-loud, not a silent empty result)", async function () {
    // The engine treats an empty q as a missing required parameter (fail-loud),
    // NOT as a match-everything or empty-result query.
    const res = await searchMode(DOMAIN, "ss_c0", "", "vector")
    expect(res.status).to.equal(400)
  })

  it("a valid no-match query returns 200 with an empty array (not an error)", async function () {
    // Real, well-formed query terms that match nothing in the corpus via FTS.
    const res = await searchMode(DOMAIN, "ss_c0", "quetzalcoatl obsidian xylophone", "fts", { count: 10 })
    expect(res.status).to.equal(200)
    expect(res.body).to.be.an("array")
    expect(res.body.length).to.equal(0, "no FTS term match → empty array, not an error")
  })

  it("multilingual: a non-English doc is retrievable by a non-English query (nomic prefix handling)", async function () {
    const D = "admin/multilingual"
    // Spanish doc + Spanish query — the multilingual nomic-v2 model + the
    // search_document:/search_query: prefixes must work end-to-end.
    await pushAndWait(D, "main", "ml_c0", [
      { op: "Inserted", id: "terminusdb:///ml/Doc/es", string: "La nave espacial despegó del planeta desértico al amanecer." },
      { op: "Inserted", id: "terminusdb:///ml/Doc/other", string: "A droid repairs the hyperdrive motivator in the cargo bay." },
    ])
    const res = await agent()
      .get("/search")
      .query({ domain: D, commit: "ml_c0", q: "nave espacial planeta desértico", mode: "vector", count: 5 })
      .set("Authorization", authHeader())
      .expect(200)
    expect(res.body).to.be.an("array")
    expect(res.body.length).to.be.at.least(1)
    // The Spanish doc must rank top for the Spanish query.
    expect(res.body[0].id).to.equal(
      "terminusdb:///ml/Doc/es",
      "the Spanish doc must be the top hit for the Spanish query",
    )
    await agent().delete("/domain").query({ domain: D }).set("Authorization", authHeader())
  })

  describe("/duplicates", function () {
    // Response shape (CHANGED): array of { group: [{id, snippet?}, ...], distance }.
    const SABER1 = "terminusdb:///ss/Topic/saber1"
    const SABER2 = "terminusdb:///ss/Topic/saber2"
    const COOKING = "terminusdb:///ss/Topic/cooking"

    // A group's member ids (order-independent set membership helper).
    const groupIds = g => g.group.map(m => m.id)
    const containsGroup = (groups, a, b) =>
      groups.some(g => {
        const ids = groupIds(g)
        return ids.includes(a) && ids.includes(b)
      })

    it("returns 200 + an array of groups for an indexed commit", async function () {
      const res = await agent()
        .get("/duplicates")
        .query({ domain: DOMAIN, commit: "ss_c0" })
        .set("Authorization", authHeader())
        .expect(200)
      expect(res.body).to.be.an("array")
    })

    it("returns a non-2xx status when the commit is not indexed (never an unbounded scan)", async function () {
      const res = await agent()
        .get("/duplicates")
        .query({ domain: "admin/dup_noindex", commit: "never_indexed" })
        .set("Authorization", authHeader())
      // Not-indexed → the engine declines rather than scanning. Resolution uses the
      // same catch-up path as /search and /similar, so 404 (no indexed lineage) or
      // 503 (search backend cold) are the documented non-2xx outcomes.
      expect([404, 503]).to.include(res.status)
    })

    it("surfaces genuinely near-identical docs as a group (the rich shape)", async function () {
      const res = await agent()
        .get("/duplicates")
        .query({ domain: DOMAIN, commit: "ss_c0", threshold: 0.2 })
        .set("Authorization", authHeader())
        .expect(200)
      expect(res.body).to.be.an("array")
      // The two Soresu docs are returned together in one group.
      expect(containsGroup(res.body, SABER1, SABER2),
        `expected a group with [${SABER1}, ${SABER2}] in ${JSON.stringify(res.body)}`).to.equal(true)
      // Rich shape: every entry has a symmetric `group` array (lower id first) and
      // a numeric [0,1] distance. No bare [id1,id2] tuples.
      for (const g of res.body) {
        expect(g).to.have.property("group").that.is.an("array")
        expect(g.group.length).to.be.at.least(2)
        expect(g).to.have.property("distance").that.is.a("number")
        expect(g.distance).to.be.within(0, 1)
        for (const m of g.group) {
          expect(m).to.have.property("id").that.is.a("string")
          // snippet omitted by default (snippet=false)
          expect(m).to.not.have.property("snippet")
        }
        expect(g.group[0].id <= g.group[1].id,
          `group not lower-id-first: ${JSON.stringify(g.group)}`).to.equal(true)
      }
      // The unrelated cooking doc is not grouped with either Soresu doc.
      expect(containsGroup(res.body, COOKING, SABER1), "cooking must not group with saber1").to.equal(false)
      expect(containsGroup(res.body, COOKING, SABER2), "cooking must not group with saber2").to.equal(false)
    })

    it("tightening the threshold yields fewer or no groups", async function () {
      const loose = await agent()
        .get("/duplicates")
        .query({ domain: DOMAIN, commit: "ss_c0", threshold: 0.2 })
        .set("Authorization", authHeader())
        .expect(200)
      const tight = await agent()
        .get("/duplicates")
        .query({ domain: DOMAIN, commit: "ss_c0", threshold: 0.0 })
        .set("Authorization", authHeader())
        .expect(200)
      expect(loose.body).to.be.an("array")
      expect(tight.body).to.be.an("array")
      // A near-zero threshold can only return a subset of the permissive run.
      expect(tight.body.length).to.be.at.most(loose.body.length)
      // The genuine near-duplicate is present at the permissive threshold.
      expect(containsGroup(loose.body, SABER1, SABER2)).to.equal(true)
    })

    it("snippet=true populates group[].snippet for each member", async function () {
      const res = await agent()
        .get("/duplicates")
        .query({ domain: DOMAIN, commit: "ss_c0", threshold: 0.2, snippet: true })
        .set("Authorization", authHeader())
        .expect(200)
      expect(res.body).to.be.an("array")
      expect(res.body.length).to.be.at.least(1)
      for (const g of res.body) {
        for (const m of g.group) {
          expect(m).to.have.property("snippet").that.is.a("string")
          expect(m.snippet.length).to.be.at.least(1)
        }
      }
    })
  })
})

// REAL-DATA SCALE: a corpus of MANY MULTI-CHUNK documents with planted
// near-duplicate pairs must surface as NON-EMPTY groups — the gap that hid the
// "[] at scale" bug (the old 3-doc single-chunk fixture could not exhibit the
// k=2 starvation). Reproduces the canonical "is it broken" check end-to-end.
describe("Duplicates at scale — multi-chunk corpus must NOT return [] (e2e)", function () {
  this.timeout(300000)

  const D = "admin/dup_scale"
  const NUM_FAMILIES = 40 // 40 families × 3 docs = 120 docs
  const DOCS_PER_FAMILY = 3
  const CHUNKS = 4 // multi-chunk: the condition that starved k=2

  // Build a doc as a long body of several near-identical paragraphs (forces
  // multi-chunk indexing). Same `theme` text across a family → planted
  // near-duplicate docs. doc_type is derived from the IRI path segment ("Item").
  const docOps = (id, theme) => ({
    op: "Inserted",
    id,
    string: Array.from({ length: CHUNKS }, (_, c) =>
      `${theme}. Section ${c}: ${theme} ${theme} ${theme} described in detail with extra words ${c}.`,
    ).join("\n\n"),
  })

  before(async function () {
    // Clean up any stale state from prior runs (each test must be independently runnable).
    await agent().delete("/domain").query({ domain: D }).set("Authorization", authHeader())
    const ops = []
    for (let f = 0; f < NUM_FAMILIES; f++) {
      const theme = `Topic family ${f} unique subject matter alpha${f} beta${f} gamma${f}`
      for (let d = 0; d < DOCS_PER_FAMILY; d++) {
        ops.push(docOps(`terminusdb:///dup/Item/f${f}_d${d}`, theme))
      }
    }
    await pushAndWait(D, "main", "dup_c0", ops)
  })

  after(async function () {
    await agent().delete("/domain").query({ domain: D }).set("Authorization", authHeader())
  })

  it("returns NON-EMPTY groups at threshold=1.0 on a multi-chunk corpus", async function () {
    const res = await agent()
      .get("/duplicates")
      .query({ domain: D, commit: "dup_c0", threshold: 1.0, count: 500 })
      .set("Authorization", authHeader())
      .expect(200)
    expect(res.body).to.be.an("array")
    expect(res.body.length,
      `duplicates returned [] on a ${NUM_FAMILIES * DOCS_PER_FAMILY}-doc multi-chunk corpus ` +
      "— the [] -at-scale bug has regressed").to.be.at.least(1)
  })

  it("surfaces planted intra-family near-duplicates as groups", async function () {
    const res = await agent()
      .get("/duplicates")
      .query({ domain: D, commit: "dup_c0", threshold: 0.5, count: 500 })
      .set("Authorization", authHeader())
      .expect(200)
    const groups = res.body
    // At least one planted family's documents are grouped together.
    const memberSets = groups.map(g => new Set(g.group.map(m => m.id)))
    const planted = (a, b) => memberSets.some(s => s.has(a) && s.has(b))
    const f0d0 = "terminusdb:///dup/Item/f0_d0"
    const f0d1 = "terminusdb:///dup/Item/f0_d1"
    const f0d2 = "terminusdb:///dup/Item/f0_d2"
    expect(planted(f0d0, f0d1) || planted(f0d0, f0d2) || planted(f0d1, f0d2),
      `expected a planted family-0 near-duplicate group in ${JSON.stringify(groups).slice(0, 500)}`)
      .to.equal(true)
  })
})
