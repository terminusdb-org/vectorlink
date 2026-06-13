/**
 * P1-NDJSON-* — NDJSON push parse tests.
 */

const { expect } = require("chai")
const { agent, authHeader } = require("../lib/agent")

describe("P1-NDJSON: Push parse", function () {
  // P1-NDJSON-1: Incremental parse (large stream processed without buffering whole).
  // Verify that a multi-line NDJSON body is accepted and
  // processed (returns a task id). The bounded-memory assertion requires a
  // much larger payload in integration tests; here we verify the contract holds.
  describe("P1-NDJSON-1: multi-line NDJSON parsed", function () {
    it("a 100-line NDJSON body is accepted and returns a task id", async function () {
      const lines = []
      for (let i = 0; i < 100; i++) {
        lines.push(JSON.stringify({
          op: "Inserted",
          id: `terminusdb:///test/Doc/${i}`,
          string: `Document ${i} content for indexing`,
        }))
      }
      const ndjson = lines.join("\n")

      const res = await agent()
        .post("/push")
        .query({ domain: "admin/db", branch: "main", target_commit: "large-push" })
        .set("Authorization", authHeader())
        .set("Content-Type", "application/x-ndjson")
        .send(ndjson)
        .expect(200)

      expect(res.text).to.be.a("string")
      expect(res.text.length).to.be.greaterThan(0)
    })
  })

  // P1-NDJSON-2: Malformed JSON line fails loudly (task Error).
  describe("P1-NDJSON-2: malformed line fails loudly", function () {
    it("a malformed JSON line creates a task with Error status", async function () {
      const ndjson = [
        JSON.stringify({ op: "Inserted", id: "terminusdb:///test/Doc/1", string: "ok" }),
        "this is not valid json {{{",
        JSON.stringify({ op: "Deleted", id: "terminusdb:///test/Doc/2" }),
      ].join("\n")

      const pushRes = await agent()
        .post("/push")
        .query({ domain: "admin/malformed", branch: "main", target_commit: "bad-push" })
        .set("Authorization", authHeader())
        .set("Content-Type", "application/x-ndjson")
        .send(ndjson)
        .expect(200)

      const taskId = pushRes.text
      expect(taskId).to.be.a("string")
      expect(taskId.length).to.be.greaterThan(0)

      // Check the task — should be Error (500 with text/plain body per contract).
      const checkRes = await agent()
        .get("/check")
        .query({ task_id: taskId })
        .set("Authorization", authHeader())
        .expect(500)

      // Error response is text/plain with the error message.
      expect(checkRes.text).to.match(/malformed.*line.*2/i)
    })
  })
})
