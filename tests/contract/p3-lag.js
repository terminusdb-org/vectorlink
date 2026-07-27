/**
 * P3-LAG — lag and catch-up resolution, through the REAL HTTP pipeline
 * (push → index → search), against the live engine + embeddings.
 *
 *  P3-LAG-1  search a not-yet-indexed commit → 200 from the nearest indexed
 *            PROVEN ancestor (supplied in the `ancestor` window);
 *            `TerminusDB-Data-Version` reports the SERVED commit (≠ requested ⇒
 *            stale). Never blocks. The lag (served ≠ requested) is the catch-up
 *            nudge signal.
 *  P3-LAG-1b BLOCKER-1: requesting an OLDER un-indexed commit when the indexed
 *            tip is NEWER must NOT serve the newer tip's data — only a proven
 *            ancestor in the supplied window, else 404. (Snapshot-isolation
 *            leak guard: the prior tip-serving behaviour returned newer docs.)
 *  P3-LAG-2  branch with NO indexed lineage → 404 (every search; the lineage
 *            check is a cheap durable on-disk tag lookup, not a cached negative).
 *  P3-LAG-3  recovery: after a 404, a push to that branch's ancestry → the next
 *            search (with the ancestor window) no longer 404s — resolved purely
 *            from the durable tag the push wrote.
 *
 * The `ancestor` repeated query param carries the nearest-first ancestor window
 * TerminusDB supplies (Spec 10 §5). Without it, a non-exact commit cannot be
 * PROVEN to descend from any indexed commit, so the engine 404s rather than risk
 * serving a descendant snapshot.
 *
 * NOTE (task-durable-index-state §6): the former 404 NEGATIVE CACHE has been
 * REMOVED. Index state — "is this commit indexed", "does this branch have
 * indexed lineage", "what is last-indexed" — is derived from the on-disk Lance
 * tags, which survive a restart. The negative cache was the source of the
 * restart-loses-state bug and guarded nothing worth keeping on the now-fast
 * durable-lookup path. The observable behaviour below (404 then 404; 404 then
 * 200 after a push) is unchanged — only the internal cache is gone.
 *
 * P3-LAG-4 (restart durability — the headline regression) is exercised here when
 * the runner passes the engine container name via VECTORLINK_ITEST_CONTAINER:
 * index → restart the container → the same commit is STILL searchable / its
 * last-indexed STILL reported / duplicates STILL work — never a post-restart 404.
 * It is also covered exhaustively at the store level (a fresh LanceStore over the
 * same on-disk dir — the exact restart effect — in src/store/lance.rs).
 */

const { expect } = require("chai")
const { execFileSync, execSync } = require("child_process")
const { agent, authHeader, BASE_URL } = require("../lib/agent")

// The engine restart mechanism, injected by the integration runner.
// VECTORLINK_ITEST_RESTART_CMD: a shell command that restarts the engine
//   (e.g. "kill <pid> && <binary>" for local, "docker restart <container>" for Docker).
// VECTORLINK_ITEST_CONTAINER: (legacy) Docker container name — when present without
//   RESTART_CMD, uses `docker restart <container>`.
// When neither is set, the restart test is skipped — the same invariant is
// covered exhaustively at the store level (fresh LanceStore over the same
// on-disk dir, in src/store/lance.rs).
const ITEST_CONTAINER = process.env.VECTORLINK_ITEST_CONTAINER
const ITEST_RESTART_CMD = process.env.VECTORLINK_ITEST_RESTART_CMD
const restartAvailable = ITEST_RESTART_CMD || ITEST_CONTAINER

async function waitForLive (timeoutMs = 30000) {
  const start = Date.now()
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await agent().get("/health/live")
      if (res.status === 200) {
        return
      }
    } catch (_e) {
      // WHY: the engine is mid-restart — connection refused is expected here.
      // INVARIANT: we loop until /health/live answers 200 or the timeout fires.
      // CONSEQUENCE: a genuinely dead engine fails the test loudly at timeout.
    }
    await new Promise(resolve => setTimeout(resolve, 300))
  }
  throw new Error(`engine at ${BASE_URL} did not become live within ${timeoutMs}ms after restart`)
}

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

describe("P3-LAG — lag, catch-up, durable resolution", function () {
  this.timeout(180000)

  const DOMAIN = "admin/lag_test"
  const BRANCH = "main"

  before(async function () {
    // Clean up any stale state from prior runs (each test must be independently runnable).
    // DELETE /domain is idempotent (204 for non-existent domains).
    const domainsUsed = [DOMAIN, "admin/blocker1", "admin/lag_recover", "admin/lag_noindex", "admin/restart_invariant"]
    for (const d of domainsUsed) {
      await agent().delete("/domain").query({ domain: d }).set("Authorization", authHeader())
    }
    // Seed a single indexed commit c0 on main.
    await pushAndWait(DOMAIN, BRANCH, "lag_c0", [
      { op: "Inserted", id: "terminusdb:///lag/People/yoda", string: "Yoda is a wise old Jedi master who trains young Padawans." },
    ])
  })

  after(async function () {
    // Clean up all domains used by this test suite.
    const domainsUsed = [DOMAIN, "admin/blocker1", "admin/lag_recover", "admin/lag_noindex", "admin/restart_invariant"]
    for (const d of domainsUsed) {
      await agent().delete("/domain").query({ domain: d }).set("Authorization", authHeader())
    }
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

  it("P3-LAG-2: a branch with no indexed lineage returns 404 on every search (durable check)", async function () {
    const NO_INDEX_DOMAIN = "admin/lag_noindex"
    // No push has ever happened for this domain/branch → no indexed lineage on
    // disk. The lineage gate is a durable tag lookup (no negative cache).
    const first = await agent()
      .get("/search")
      .query({ domain: NO_INDEX_DOMAIN, commit: "whatever_commit", q: "anything", mode: "vector" })
      .set("Authorization", authHeader())
    expect(first.status).to.equal(404)

    // A second search must also 404 — still no indexed lineage on disk.
    const second = await agent()
      .get("/search")
      .query({ domain: NO_INDEX_DOMAIN, commit: "whatever_commit", q: "anything", mode: "vector" })
      .set("Authorization", authHeader())
    expect(second.status).to.equal(404)
  })

  it("P3-LAG-3: a push to the branch ancestry makes the next search resolve (durable tag)", async function () {
    const RECOVER_DOMAIN = "admin/lag_recover"

    // First search 404s (no indexed lineage on disk).
    const before = await agent()
      .get("/search")
      .query({ domain: RECOVER_DOMAIN, commit: "rc_head", q: "anything", mode: "vector" })
      .set("Authorization", authHeader())
    expect(before.status).to.equal(404)

    // Push an indexed commit to that branch's ancestry — this writes the durable
    // tag the next search resolves against (no cache to bust).
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

  // P3-LAG-4 — THE HEADLINE REGRESSION (task-durable-index-state).
  // Index a branch → RESTART the engine container → the same commit is STILL
  // searchable, its /last-indexed STILL reports it, /duplicates STILL works, and
  // it is NEVER a post-restart 404. This is the exact failure observed live
  // (1091 docs on disk, /last-indexed=null, search 404 after a rebuild).
  const restartIt = restartAvailable ? it : it.skip
  restartIt("P3-LAG-4: index survives a real engine container restart (no lost corpus)", async function () {
    this.timeout(120000)
    const RESTART_DOMAIN = "admin/restart_invariant"
    const COMMIT = "ri_c0"

    // 1. Index a commit (push → wait for the durable tag).
    await pushAndWait(RESTART_DOMAIN, "main", COMMIT, [
      { op: "Inserted", id: "terminusdb:///ri/People/leia", string: "Leia Organa is a leader of the Rebel Alliance and a Jedi." },
    ])

    // 2. Confirm the PRE-restart state: last-indexed reports the commit; search
    //    and duplicates resolve it.
    const liBefore = await agent()
      .get("/last-indexed")
      .query({ domain: RESTART_DOMAIN, branch: "main" })
      .set("Authorization", authHeader())
      .expect(200)
    expect(liBefore.body.commit, "pre-restart last-indexed must report the commit").to.equal(COMMIT)

    const searchBefore = await agent()
      .get("/search")
      .query({ domain: RESTART_DOMAIN, commit: COMMIT, q: "rebel leader", mode: "vector" })
      .set("Authorization", authHeader())
      .expect(200)
    expect(searchBefore.headers["terminusdb-data-version"]).to.equal(`commit:${COMMIT}`)

    // 3. RESTART the engine. Process state evaporates; on-disk Lance tags persist.
    if (ITEST_RESTART_CMD) {
      execSync(ITEST_RESTART_CMD, { stdio: "ignore" })
    } else {
      execFileSync("docker", ["restart", ITEST_CONTAINER], { stdio: "ignore" })
    }
    await waitForLive()

    // 4. THE INVARIANT: everything still works, derived from disk. NOT a 404.
    const liAfter = await agent()
      .get("/last-indexed")
      .query({ domain: RESTART_DOMAIN, branch: "main" })
      .set("Authorization", authHeader())
      .expect(200)
    expect(liAfter.body.commit, "after restart, last-indexed MUST still report the commit (from disk)")
      .to.equal(COMMIT)

    const searchAfter = await agent()
      .get("/search")
      .query({ domain: RESTART_DOMAIN, commit: COMMIT, q: "rebel leader", mode: "vector" })
      .set("Authorization", authHeader())
    expect(searchAfter.status, "search at the indexed commit must NOT 404 after a restart").to.equal(200)
    expect(searchAfter.headers["terminusdb-data-version"]).to.equal(`commit:${COMMIT}`)

    const dupAfter = await agent()
      .get("/duplicates")
      .query({ domain: RESTART_DOMAIN, commit: COMMIT })
      .set("Authorization", authHeader())
    expect(dupAfter.status, "duplicates at the indexed commit must NOT 404 after a restart").to.equal(200)
    expect(dupAfter.body).to.be.an("array")
  })
})
