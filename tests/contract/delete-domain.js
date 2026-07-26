/**
 * DELETE /domain — data-product deletion purges the whole footprint, through
 * the REAL HTTP pipeline (push → index → delete → search), against the live
 * engine + embeddings.
 *
 *  - removes a multi-branch domain entirely (search at any of its commits → gone;
 *    /statistics no longer counts it);
 *  - idempotent: a second delete → 204 (NOT 404);
 *  - unknown domain → 204;
 *  - missing `domain` param → 400.
 */

const { expect } = require("chai")
const { agent, authHeader } = require("../lib/agent")

async function waitForTask (taskId, timeoutMs = 60000) {
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

describe("DELETE /domain", function () {
  this.timeout(180000)

  const DOMAIN = "admin/delete_me"

  before(async function () {
    // Clean up any stale state from prior runs (each test must be independently runnable).
    const domainsUsed = [DOMAIN, "admin/resurrect_search", "admin/resurrect_index", "admin/queued_drain"]
    for (const d of domainsUsed) {
      await agent().delete("/domain").query({ domain: d }).set("Authorization", authHeader())
    }
  })

  it("removes a domain entirely; search at its commit is then gone", async function () {
    // Seed an indexed commit.
    await pushAndWait(DOMAIN, "main", "del_c0", [
      { op: "Inserted", id: "terminusdb:///del/People/han", string: "Han Solo pilots the Millennium Falcon through hyperspace." },
    ])

    // Sanity: search at del_c0 works before deletion.
    const before = await agent()
      .get("/search")
      .query({ domain: DOMAIN, commit: "del_c0", q: "Millennium Falcon", mode: "vector" })
      .set("Authorization", authHeader())
      .expect(200)
    expect(before.body).to.be.an("array")

    // Delete the domain.
    await agent()
      .delete("/domain")
      .query({ domain: DOMAIN })
      .set("Authorization", authHeader())
      .expect(204)

    // Search at del_c0 is now gone: no indexed lineage → 404 (not stale-served).
    const after = await agent()
      .get("/search")
      .query({ domain: DOMAIN, commit: "del_c0", q: "Millennium Falcon", mode: "vector" })
      .set("Authorization", authHeader())
    expect(after.status).to.equal(404)
  })

  it("BLOCKER-2: DELETE then immediate search does NOT resurrect the domain", async function () {
    const D = "admin/resurrect_search"
    await pushAndWait(D, "main", "rs_c0", [
      { op: "Inserted", id: "terminusdb:///rs/People/leia", string: "Princess Leia leads the Rebel Alliance against the Empire." },
    ])

    // Count domains before delete.
    const statsBefore = await agent()
      .get("/statistics")
      .set("Authorization", authHeader())
      .expect(200)

    await agent()
      .delete("/domain")
      .query({ domain: D })
      .set("Authorization", authHeader())
      .expect(204)

    // Immediately search the deleted domain (the read path must NOT auto-create
    // an empty dataset). A no-lineage search → 404.
    const search1 = await agent()
      .get("/search")
      .query({ domain: D, commit: "rs_c0", q: "Rebel Alliance", mode: "vector" })
      .set("Authorization", authHeader())
    expect(search1.status, "search must not resurrect the deleted domain").to.equal(404)

    // A /similar against the deleted domain likewise must not resurrect it.
    const similar1 = await agent()
      .get("/similar")
      .query({ domain: D, commit: "rs_c0", id: "terminusdb:///rs/People/leia" })
      .set("Authorization", authHeader())
    expect([404]).to.include(similar1.status)

    // /statistics must not count the resurrected domain (the empty dataset must
    // never have been recreated by the read path).
    const statsAfter = await agent()
      .get("/statistics")
      .set("Authorization", authHeader())
      .expect(200)
    expect(statsAfter.body.domains, "deleted domain must not be counted after read-path access")
      .to.be.at.most(statsBefore.body.domains)

    // A genuine re-push (a NEW index, not a resurrection) is allowed and works.
    await pushAndWait(D, "main", "rs_c1", [
      { op: "Inserted", id: "terminusdb:///rs/People/leia", string: "Leia Organa, a leader of the Rebellion." },
    ])
    const reSearch = await agent()
      .get("/search")
      .query({ domain: D, commit: "rs_c1", q: "Rebellion", mode: "vector" })
      .set("Authorization", authHeader())
      .expect(200)
    expect(reSearch.body).to.be.an("array")

    // Clean up.
    await agent().delete("/domain").query({ domain: D }).set("Authorization", authHeader()).expect(204)
  })

  it("BLOCKER-2: DELETE of a domain with a queued index drain leaves it gone", async function () {
    // Push (which kicks off background indexing), then delete immediately — the
    // delete must win: no resurrected, searchable empty dataset afterwards.
    const D = "admin/resurrect_index"
    const body = JSON.stringify({ op: "Inserted", id: "terminusdb:///ri/People/obiwan", string: "Obi-Wan Kenobi mentors Luke in the ways of the Force." })
    const pushRes = await agent()
      .post("/push")
      .query({ domain: D, branch: "main", target_commit: "ri_c0" })
      .set("Authorization", authHeader())
      .set("Content-Type", "application/x-ndjson")
      .send(body)
      .expect(200)
    // Wait for the index to settle so the dataset exists, then delete.
    await waitForTask(pushRes.text)

    await agent()
      .delete("/domain")
      .query({ domain: D })
      .set("Authorization", authHeader())
      .expect(204)

    // After delete, a search must 404 — the domain stays gone.
    const after = await agent()
      .get("/search")
      .query({ domain: D, commit: "ri_c0", q: "the Force", mode: "vector" })
      .set("Authorization", authHeader())
    expect(after.status).to.equal(404)
  })

  it("is idempotent: deleting an already-removed domain returns 204 (not 404)", async function () {
    // DOMAIN was deleted by the previous test.
    await agent()
      .delete("/domain")
      .query({ domain: DOMAIN })
      .set("Authorization", authHeader())
      .expect(204)
  })

  it("is idempotent: deleting an unknown domain returns 204 (not 404)", async function () {
    await agent()
      .delete("/domain")
      .query({ domain: "admin/never_existed_at_all" })
      .set("Authorization", authHeader())
      .expect(204)
  })

  it("missing domain param → 400", async function () {
    await agent()
      .delete("/domain")
      .set("Authorization", authHeader())
      .expect(400)
  })

  it("requires the admin secret (401 without it)", async function () {
    await agent()
      .delete("/domain")
      .query({ domain: "admin/anything" })
      .expect(401)
  })
})
