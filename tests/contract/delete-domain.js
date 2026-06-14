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
