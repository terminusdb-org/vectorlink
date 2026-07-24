/**
 * Security integration tests — filter injection prevention.
 *
 * Exercises the full HTTP pipeline (push → poll → search → similar → delete)
 * with doc_ids containing malicious characters that could break out of SQL
 * string literals if the code used string interpolation instead of DataFusion
 * Expr values.
 *
 * Vectors tested:
 *   - Backslash-quote:   x\' OR 1=1
 *   - Single quote:      it's a doc
 *   - Newline + comment: doc1\n-- OR 1=1
 *   - Null byte:         doc\0admin
 *   - SQL keywords:      '; DROP TABLE--
 *   - Unicode tricks:    ＇ (fullwidth apostrophe U+FF07)
 *
 * Each test pushes a document with a malicious id, then verifies:
 *   1. The push completes without error (no parser crash)
 *   2. The document is searchable by its exact id
 *   3. A Changed op on the same id updates (not duplicates) the document
 *   4. A Deleted op on the same id removes the document
 *   5. Search with doc_id filter using the malicious id returns only that doc
 *   6. /similar with the malicious id returns results without injection
 *
 * The key invariant: a malicious doc_id must never affect OTHER documents.
 * If injection were possible, a filter like doc_id = 'x\'' OR 1=1 --' would
 * match all rows, causing cross-document contamination.
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
    await new Promise((resolve) => setTimeout(resolve, 500))
  }
  throw new Error(`task ${taskId} did not complete within ${timeoutMs}ms`)
}

async function pushAndWait (domain, branch, commit, ndjsonLines) {
  const body = ndjsonLines.map((l) => JSON.stringify(l)).join("\n")
  const pushRes = await agent()
    .post("/push")
    .query({ domain, branch, target_commit: commit })
    .set("Authorization", authHeader())
    .set("Content-Type", "application/x-ndjson")
    .send(body)
    .expect(200)
  expect(pushRes.text).to.match(/^task-/)
  return waitForTask(pushRes.text)
}

async function searchIds (domain, commit, query, mode) {
  const res = await agent()
    .get("/search")
    .query({ domain, commit, q: query, mode })
    .set("Authorization", authHeader())
  expect(res.status).to.equal(200)
  return res.body.map((h) => h.id)
}

async function searchWithDocIdFilter (domain, commit, query, mode, docIds) {
  // Build query string manually to allow repeated doc_id params
  let qs = `domain=${encodeURIComponent(domain)}&commit=${encodeURIComponent(commit)}&q=${encodeURIComponent(query)}&mode=${mode}`
  for (const id of docIds) {
    qs += `&doc_id=${encodeURIComponent(id)}`
  }
  const res = await agent()
    .get("/search")
    .query(qs)
    .set("Authorization", authHeader())
  return res
}

async function similarById (domain, commit, id) {
  const res = await agent()
    .get("/similar")
    .query({ domain, commit, id, count: 10 })
    .set("Authorization", authHeader())
  return res
}

// ─── Malicious doc_ids ───────────────────────────────────────────────────

const MALICIOUS_IDS = [
  {
    label: "backslash-quote injection",
    id: "evil\\' OR 1=1 --",
    text: "Document with backslash quote injection attempt in its id",
  },
  {
    label: "single quote",
    id: "it's a doc with apostrophe",
    text: "Document whose id contains a single quote apostrophe character",
  },
  {
    label: "newline + SQL comment",
    id: "doc1\n-- OR 1=1",
    text: "Document whose id contains a newline followed by a SQL comment",
  },
  {
    label: "null byte",
    id: "doc\x00admin",
    text: "Document whose id contains a null byte character",
  },
  {
    label: "SQL DROP TABLE",
    id: "'; DROP TABLE doc_id;--",
    text: "Document whose id contains a SQL DROP TABLE injection attempt",
  },
  {
    label: "UNION SELECT",
    id: "' UNION SELECT doc_id FROM chunks--",
    text: "Document whose id contains a SQL UNION SELECT injection attempt",
  },
  {
    label: "fullwidth apostrophe",
    id: "doc\uFF07 OR 1=1",
    text: "Document whose id contains a fullwidth apostrophe unicode character",
  },
  {
    label: "backslash alone",
    id: "back\\slash\\doc",
    text: "Document whose id contains backslash characters without quotes",
  },
]

// ─── Tests ───────────────────────────────────────────────────────────────

describe("Security: filter injection prevention", function () {
  this.timeout(180000)

  const DOMAIN = "admin/security_injection"
  const BRANCH = "main"

  before(async function () {
    await agent()
      .delete("/domain")
      .query({ domain: DOMAIN })
      .set("Authorization", authHeader())
  })

  after(async function () {
    await agent()
      .delete("/domain")
      .query({ domain: DOMAIN })
      .set("Authorization", authHeader())
  })

  // ── Phase 1: Push all malicious-id documents in one commit ──────────

  it("should push all malicious-id documents without error", async function () {
    const ops = MALICIOUS_IDS.map((m) => ({
      op: "Inserted",
      id: m.id,
      string: m.text,
    }))
    // Also push a benign doc to verify no cross-contamination
    ops.push({
      op: "Inserted",
      id: "benign_doc_001",
      string: "A completely benign document about cats and dogs",
    })

    const result = await pushAndWait(DOMAIN, BRANCH, "sec-commit-001", ops)
    expect(result.status).to.equal("Complete")
    expect(result.indexed_documents).to.be.at.least(MALICIOUS_IDS.length)
  })

  // ── Phase 2: Each malicious doc is searchable by content ────────────

  for (const m of MALICIOUS_IDS) {
    it(`should find "${m.label}" document via vector search`, async function () {
      const ids = await searchIds(DOMAIN, "sec-commit-001", m.text, "vector")
      // The malicious-id doc should appear in results (it matches its own text)
      expect(ids).to.include(m.id)
    })
  }

  // ── Phase 3: doc_id filter with malicious id returns only that doc ──

  for (const m of MALICIOUS_IDS) {
    it(`should filter to only "${m.label}" when using its id as doc_id filter`, async function () {
      const res = await searchWithDocIdFilter(
        DOMAIN,
        "sec-commit-001",
        m.text,
        "vector",
        [m.id],
      )
      expect(res.status).to.equal(200)
      const ids = res.body.map((h) => h.id)
      // Must contain the malicious-id doc
      expect(ids).to.include(m.id)
      // Must NOT contain the benign doc (injection would match all rows)
      expect(ids).to.not.include("benign_doc_001")
      // Must NOT contain any other malicious-id doc
      for (const other of MALICIOUS_IDS) {
        if (other.id !== m.id) {
          expect(ids).to.not.include(other.id)
        }
      }
    })
  }

  // ── Phase 4: /similar with malicious id does not contaminate ────────

  it("should return /similar results for a malicious-id doc without cross-contamination", async function () {
    const target = MALICIOUS_IDS[0] // backslash-quote
    const res = await similarById(DOMAIN, "sec-commit-001", target.id)
    // /similar may return 200 (found neighbours) or 404 (no neighbours)
    // but must NOT return 500 (injection crash)
    expect([200, 404]).to.include(res.status)
    if (res.status === 200) {
      const ids = res.body.map((h) => h.id)
      // The source doc itself should not appear in its own similar results
      // (self-exclusion filter), but other docs may appear
      for (const id of ids) {
        expect(id).to.not.equal(target.id)
      }
    }
  })

  // ── Phase 5: Changed op on malicious id updates, not duplicates ─────

  it("should update a malicious-id doc via Changed without creating duplicates", async function () {
    const target = MALICIOUS_IDS[0] // backslash-quote
    const newText = "Updated content for the backslash quote injection doc about quantum computing"

    const result = await pushAndWait(DOMAIN, BRANCH, "sec-commit-002", [
      { op: "Changed", id: target.id, string: newText },
    ])
    expect(result.status).to.equal("Complete")

    // Search at the new commit — should find the updated content
    const ids = await searchIds(DOMAIN, "sec-commit-002", newText, "vector")
    expect(ids).to.include(target.id)

    // The old content should no longer appear in any result at the new commit.
    // Check both the ID and the actual content — the old text must not be
    // present in any returned snippet (Changed = replace, not append).
    const debugRes = await agent()
      .get("/search")
      .query({ domain: DOMAIN, commit: "sec-commit-002", q: target.text, mode: "vector", snippet: true, count: 10 })
      .set("Authorization", authHeader())

    const oldHits = debugRes.body
    const oldSnippets = oldHits.map((h) => h.chunk.snippet).filter(Boolean)
    const oldTextPresent = oldSnippets.some((s) => s.includes(target.text))

    // The doc's old content must not appear in any result
    expect(oldTextPresent).to.equal(false)

    // The doc must not appear with its old content — check by ID + content match
    const targetWithOldContent = oldHits.find((h) => h.id === target.id && h.chunk.snippet && h.chunk.snippet.includes(target.text))
    expect(targetWithOldContent).to.equal(undefined)
  })

  // ── Phase 5b: Changed op on a benign id — same behaviour, no malice ──

  it("should update a benign-id doc via Changed without creating duplicates", async function () {
    const benignId = "benign_doc_001"
    const benignOldText = "A completely benign document about cats and dogs"
    const benignNewText = "A completely benign document about quantum physics and relativity"

    const result = await pushAndWait(DOMAIN, BRANCH, "sec-commit-002b", [
      { op: "Changed", id: benignId, string: benignNewText },
    ])
    expect(result.status).to.equal("Complete")

    // Search at the new commit — should find the updated content
    const ids = await searchIds(DOMAIN, "sec-commit-002b", benignNewText, "vector")
    expect(ids).to.include(benignId)

    // The old content should not appear in any snippet at the new commit
    const debugRes = await agent()
      .get("/search")
      .query({ domain: DOMAIN, commit: "sec-commit-002b", q: benignOldText, mode: "vector", snippet: true, count: 10 })
      .set("Authorization", authHeader())
    const oldSnippets = debugRes.body.map((h) => h.chunk.snippet).filter(Boolean)
    const oldTextPresent = oldSnippets.some((s) => s.includes(benignOldText))
    expect(oldTextPresent).to.equal(false)

    // The doc must not appear with its old content
    const targetWithOldContent = debugRes.body.find((h) => h.id === benignId && h.chunk.snippet && h.chunk.snippet.includes(benignOldText))
    expect(targetWithOldContent).to.equal(undefined)
  })

  // ── Phase 6: Deleted op on malicious id removes it ──────────────────

  it("should delete a malicious-id doc via Deleted op", async function () {
    const target = MALICIOUS_IDS[1] // single quote

    const result = await pushAndWait(DOMAIN, BRANCH, "sec-commit-003", [
      { op: "Deleted", id: target.id },
    ])
    expect(result.status).to.equal("Complete")

    // At the new commit, the deleted doc should not appear in search
    const ids = await searchIds(DOMAIN, "sec-commit-003", target.text, "vector")
    expect(ids).to.not.include(target.id)

    // But it should still exist at the older commit (snapshot isolation)
    const oldIds = await searchIds(DOMAIN, "sec-commit-001", target.text, "vector")
    expect(oldIds).to.include(target.id)
  })

  // ── Phase 7: Benign doc is unaffected by any malicious operations ───

  it("should still find the benign doc after all malicious operations", async function () {
    // Phase 5b updated the benign doc to quantum physics content
    const ids = await searchIds(DOMAIN, "sec-commit-003", "quantum physics and relativity", "vector")
    expect(ids).to.include("benign_doc_001")
  })

  // ── Phase 8: doc_id filter with multiple malicious ids ──────────────

  it("should filter to multiple malicious ids without cross-contamination", async function () {
    const targets = [MALICIOUS_IDS[0].id, MALICIOUS_IDS[2].id] // backslash-quote + newline
    const res = await searchWithDocIdFilter(
      DOMAIN,
      "sec-commit-001",
      "document",
      "vector",
      targets,
    )
    expect(res.status).to.equal(200)
    const ids = res.body.map((h) => h.id)
    // Should only return docs whose id is in the targets list
    for (const id of ids) {
      expect(targets).to.include(id)
    }
    // Must NOT contain the benign doc
    expect(ids).to.not.include("benign_doc_001")
  })

  // ── Phase 9: /similar with SQL DROP TABLE id does not crash ─────────

  it("should handle /similar for a SQL DROP TABLE id without 500", async function () {
    const target = MALICIOUS_IDS[4] // SQL DROP TABLE
    const res = await similarById(DOMAIN, "sec-commit-001", target.id)
    expect([200, 404]).to.include(res.status)
  })

  // ── Phase 10: Statistics still work after all malicious operations ──

  it("should return valid statistics after all malicious operations", async function () {
    const res = await agent()
      .get("/statistics")
      .query({ domain: DOMAIN })
      .set("Authorization", authHeader())
    expect(res.status).to.equal(200)
    expect(res.body).to.be.an("object")
  })
})
