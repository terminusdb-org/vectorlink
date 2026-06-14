/**
 * P3-LAG — lag, catch-up and the 404 negative cache, through the REAL HTTP
 * pipeline (push → index → search), against the live engine + embeddings.
 *
 *  P3-LAG-1  search a not-yet-indexed commit → 200 from the nearest indexed
 *            PROVEN ancestor (supplied in the `ancestor` window);
 *            `TerminusDB-Data-Version` reports the SERVED commit (≠ requested ⇒
 *            stale). Never blocks.
 *  P3-LAG-1b BLOCKER-1: requesting an OLDER un-indexed commit when the indexed
 *            tip is NEWER must NOT serve the newer tip's data — only a proven
 *            ancestor in the supplied window, else 404. (Snapshot-isolation
 *            leak guard: the prior tip-serving behaviour returned newer docs.)
 *  P3-LAG-2  branch with NO indexed ancestor → 404, negatively cached (the
 *            second search returns 404 too, without re-walking history).
 *  P3-LAG-3  negative cache invalidated: after a 404, a push to that branch's
 *            ancestry → the next search (with the ancestor window) no longer 404s.
 *
 * The `ancestor` repeated query param carries the nearest-first ancestor window
 * TerminusDB supplies (Spec 10 §5). Without it, a non-exact commit cannot be
 * PROVEN to descend from any indexed commit, so the engine 404s rather than risk
 * serving a descendant snapshot.
 *
 * P3-LAG-4 (outage/restart durability) is covered at the store level via the
 * on-disk layer index (Lance tags + last-indexed survive a restart) and is not
 * re-exercised here, which would require stopping the live engine mid-suite.
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

describe("P3-LAG — lag, catch-up, negative cache", function () {
  this.timeout(180000)

  const DOMAIN = "admin/lag_test"
  const BRANCH = "main"

  before(async function () {
    // Seed a single indexed commit c0 on main.
    await pushAndWait(DOMAIN, BRANCH, "lag_c0", [
      { op: "Inserted", id: "terminusdb:///lag/People/yoda", string: "Yoda is a wise old Jedi master who trains young Padawans." },
    ])
  })

  it("P3-LAG-1: search a not-yet-indexed commit serves the nearest PROVEN ancestor and flags stale", async function () {
    // Request a commit that was NEVER pushed (lag_c1), supplying it as a
    // descendant of the indexed lag_c0 via the ancestor window. The engine must
    // serve c0 (the nearest PROVEN ancestor) and report it.
    const res = await agent()
      .get("/search")
      .query({ domain: DOMAIN, commit: "lag_c1_never_pushed", q: "wise old man", mode: "vector", ancestor: ["lag_c0"] })
      .set("Authorization", authHeader())
      .expect(200)

    // Body is the bare array (never blocks, never empty-as-error).
    expect(res.body).to.be.an("array")

    // The data-version header reports the SERVED commit (c0), NOT the requested
    // one — so the caller detects staleness (served ≠ requested).
    const served = res.headers["terminusdb-data-version"]
    expect(served, "served data-version header must be present").to.be.a("string")
    expect(served).to.equal("commit:lag_c0")
    expect(served).to.not.equal("commit:lag_c1_never_pushed")
  })

  it("P3-LAG-1b (BLOCKER-1): an OLDER un-indexed commit must NOT be served the NEWER tip's data", async function () {
    // Build a lineage where an OLDER commit (blk_old) is NOT indexed, and a
    // NEWER commit (blk_new) IS — with a doc (anakin) that exists ONLY at the
    // newer commit. A search at blk_old must NEVER return anakin: the indexed
    // tip blk_new is a DESCENDANT, not an ancestor, of blk_old.
    const D = "admin/blocker1"

    // Index blk_new (the NEWER tip) with the unique doc 'anakin'.
    await pushAndWait(D, "main", "blk_new", [
      { op: "Inserted", id: "terminusdb:///b1/People/anakin", string: "Anakin Skywalker became Darth Vader, a powerful Sith Lord." },
    ])

    // Request the OLDER, never-indexed blk_old. The ONLY honest ancestor window
    // for an older commit does NOT contain the newer tip blk_new (blk_new is a
    // descendant). With no proven indexed ancestor, the engine MUST 404 — it
    // must NOT fall back to serving blk_new's snapshot.
    const res = await agent()
      .get("/search")
      .query({ domain: D, commit: "blk_old", q: "Sith Lord Darth Vader", mode: "vector", ancestor: ["blk_older_root"] })
      .set("Authorization", authHeader())

    // Either a clean 404 (no proven ancestor) — the correct outcome — OR, if any
    // result is returned, it must categorically NOT be the newer tip's snapshot.
    if (res.status === 200) {
      const served = res.headers["terminusdb-data-version"]
      expect(served, "must never serve the newer tip blk_new for an older request")
        .to.not.equal("commit:blk_new")
      const ids = res.body.map(h => h.id)
      expect(ids, "the newer-commit-only doc 'anakin' must be ABSENT at the older commit")
        .to.not.include("terminusdb:///b1/People/anakin")
    } else {
      expect(res.status, "no proven ancestor → 404 (never serve newer data)").to.equal(404)
    }
  })

  it("P3-LAG-1c: a non-exact commit with NO ancestor window 404s (cannot prove ancestry)", async function () {
    // lag_c0 is indexed on DOMAIN/main. A request for an un-indexed commit with
    // NO ancestor window cannot be proven to descend from lag_c0 → 404 (never
    // silently serve the tip).
    const res = await agent()
      .get("/search")
      .query({ domain: DOMAIN, commit: "lag_unknown_nowindow", q: "wise old man", mode: "vector" })
      .set("Authorization", authHeader())
    expect(res.status).to.equal(404)
  })

  it("P3-LAG-2: a branch with no indexed ancestor returns 404, negatively cached", async function () {
    const NO_INDEX_DOMAIN = "admin/lag_noindex"
    // No push has ever happened for this domain/branch → no indexed lineage.
    const first = await agent()
      .get("/search")
      .query({ domain: NO_INDEX_DOMAIN, commit: "whatever_commit", q: "anything", mode: "vector" })
      .set("Authorization", authHeader())
    expect(first.status).to.equal(404)

    // A second search must also 404 (served from the negative cache, no re-walk).
    const second = await agent()
      .get("/search")
      .query({ domain: NO_INDEX_DOMAIN, commit: "whatever_commit", q: "anything", mode: "vector" })
      .set("Authorization", authHeader())
    expect(second.status).to.equal(404)
  })

  it("P3-LAG-3: a push to the branch ancestry invalidates the 404 negative cache", async function () {
    const RECOVER_DOMAIN = "admin/lag_recover"

    // First search 404s (no indexed lineage) and negatively caches.
    const before = await agent()
      .get("/search")
      .query({ domain: RECOVER_DOMAIN, commit: "rc_head", q: "anything", mode: "vector" })
      .set("Authorization", authHeader())
    expect(before.status).to.equal(404)

    // Push an indexed commit to that branch's ancestry — this directly busts
    // the branch's negative cache.
    await pushAndWait(RECOVER_DOMAIN, "main", "rc_c0", [
      { op: "Inserted", id: "terminusdb:///rc/People/luke", string: "Luke Skywalker trains as a Jedi under Yoda on Dagobah." },
    ])

    // Now a search at the (still-unindexed) head, supplying rc_c0 as a proven
    // ancestor in the window, resolves via rc_c0 — no longer 404.
    const after = await agent()
      .get("/search")
      .query({ domain: RECOVER_DOMAIN, commit: "rc_head", q: "Jedi", mode: "vector", ancestor: ["rc_c0"] })
      .set("Authorization", authHeader())
    expect(after.status).to.equal(200)
    expect(after.headers["terminusdb-data-version"]).to.equal("commit:rc_c0")
  })
})
