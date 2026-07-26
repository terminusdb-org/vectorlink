/**
 * TEST-SPEC #42: Integration test — GET /integrity returns a clean report
 * after pushing data.
 *
 * Verifies that the HTTP endpoint exposes the integrity check and that
 * after pushing data, the store is in a clean state:
 *   - ok === true
 *   - no stale index dirs
 *   - no dangling index refs
 *   - no orphaned tags
 *   - no stale rebuild branches
 *   - at most 1 rebuild branch
 *
 * Prerequisite: TDB_SEARCH_DISABLE_AUTO_COMPACTION=1 in the engine env
 * (eliminates non-determinism from the 5% probabilistic trigger).
 */

const { expect } = require("chai")
const { agent, authHeader } = require("../lib/agent")

async function waitForTask (taskId, timeoutMs = 30000) {
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

async function pushAndWait (domain, branch, commit, ops) {
  const body = ops.map(l => JSON.stringify(l)).join("\n")
  const pushRes = await agent()
    .post("/push")
    .query({ domain, branch, target_commit: commit })
    .set("Authorization", authHeader())
    .set("Content-Type", "application/x-ndjson")
    .send(body)
    .expect(200)
  return waitForTask(pushRes.text)
}

describe("Integrity check endpoint (GET /integrity)", function () {
  this.timeout(120000)

  const DOMAIN = "admin/integrity_test"
  const BRANCH = "main"

  before(async function () {
    await agent().delete("/domain").query({ domain: DOMAIN }).set("Authorization", authHeader())

    // Push two commits to create enough data for compaction.
    const docs0 = []
    for (let i = 0; i < 10; i++) {
      docs0.push({
        op: "Inserted",
        id: `terminusdb:///integrity/Things/d${i}`,
        string: `Document number ${i} for integrity testing with sufficient content.`,
      })
    }
    await pushAndWait(DOMAIN, BRANCH, "c0", docs0)

    const docs1 = []
    for (let i = 10; i < 20; i++) {
      docs1.push({
        op: "Inserted",
        id: `terminusdb:///integrity/Things/d${i}`,
        string: `Document number ${i} for integrity testing with sufficient content.`,
      })
    }
    await pushAndWait(DOMAIN, BRANCH, "c1", docs1)
  })

  after(async function () {
    await agent().delete("/domain").query({ domain: DOMAIN }).set("Authorization", authHeader())
  })

  it("GET /integrity returns 200 with a well-formed report", async function () {
    const res = await agent()
      .get("/integrity")
      .query({ domain: DOMAIN })
      .set("Authorization", authHeader())
      .expect(200)

    expect(res.body).to.have.property("domain", DOMAIN)
    expect(res.body).to.have.property("ok")
    expect(res.body).to.have.property("tagged_versions")
    expect(res.body).to.have.property("on_disk_data_files")
    expect(res.body).to.have.property("on_disk_manifests")
    expect(res.body).to.have.property("stale_index_dirs")
    expect(res.body).to.have.property("dangling_index_refs")
    expect(res.body).to.have.property("orphaned_tags")
    expect(res.body).to.have.property("stale_rebuild_branches")
    expect(res.body).to.have.property("rebuild_branches")
    expect(res.body).to.have.property("warnings")
  })

  it("integrity report is clean (ok=true, no stale state)", async function () {
    const res = await agent()
      .get("/integrity")
      .query({ domain: DOMAIN })
      .set("Authorization", authHeader())
      .expect(200)

    const report = res.body

    expect(
      report.ok,
      `integrity check must report ok=true. Warnings: ${JSON.stringify(report.warnings)}`,
    ).to.equal(true)

    expect(
      report.stale_index_dirs.length,
      `stale index dirs must be empty, got: ${JSON.stringify(report.stale_index_dirs)}`,
    ).to.equal(0)

    expect(
      report.orphaned_tags.length,
      `orphaned tags must be empty, got: ${JSON.stringify(report.orphaned_tags)}`,
    ).to.equal(0)

    expect(
      report.stale_rebuild_branches.length,
      `stale rebuild branches must be empty, got: ${JSON.stringify(report.stale_rebuild_branches)}`,
    ).to.equal(0)

    expect(
      report.rebuild_branches.length,
      `at most 1 rebuild branch expected, got: ${JSON.stringify(report.rebuild_branches)}`,
    ).to.be.at.most(1)
  })

  it("GET /integrity without domain returns 400", async function () {
    await agent()
      .get("/integrity")
      .set("Authorization", authHeader())
      .expect(400)
  })

  it("GET /integrity for non-existent domain returns 200 with empty report", async function () {
    const res = await agent()
      .get("/integrity")
      .query({ domain: "admin/nonexistent_domain_12345" })
      .set("Authorization", authHeader())
      .expect(200)

    expect(res.body.ok).to.equal(true)
    expect(res.body.tagged_versions).to.equal(0)
    expect(res.body.warnings).to.include("dataset does not exist on disk")
  })
})
