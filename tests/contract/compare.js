/**
 * Contract test for POST /compare?method=embedding — stateless text distance.
 *
 * Tests:
 *   1. Identical texts → distance ≈ 0.
 *   2. Unrelated texts → distance ≈ 0.5.
 *   3. Missing method query param → 400.
 *   4. Unknown method query param → 400.
 *   5. Missing source in body → 400.
 *   6. Missing target in body → 400.
 *   7. Response contains distance, source_role, and target_role fields.
 *   8. Auth required — no credentials → 401.
 */

const { expect } = require("chai")
const { agent, authHeader, wrongAuthHeader } = require("../lib/agent")

describe("POST /compare — stateless semantic distance", function () {
  this.timeout(30000)

  it("identical texts → distance = 0 (same-string shortcircuit)", async function () {
    const text = "Sony 5-disc CD changer with carousel mechanism"
    const res = await agent()
      .post("/compare")
      .query({ method: "embedding" })
      .set("Authorization", authHeader())
      .send({ source: text, target: text })
      .expect(200)

    expect(res.body).to.have.property("distance")
    expect(res.body.distance).to.be.a("number")
    // FIX 3 (#48): Identical input strings return distance 0 without embedding.
    // The engine short-circuits on exact string equality BEFORE embedding,
    // avoiding the asymmetric-prefix artefact (~0.16 for identical content).
    expect(res.body.distance).to.equal(0)
  })

  it("near-identical texts (minor variation) → distance < 0.20", async function () {
    const source = "Sony 5-disc CD changer with carousel mechanism"
    const target = "Sony 5 disc CD player carousel mechanism"
    const res = await agent()
      .post("/compare")
      .query({ method: "embedding" })
      .set("Authorization", authHeader())
      .send({ source, target })
      .expect(200)

    expect(res.body).to.have.property("distance")
    expect(res.body.distance).to.be.a("number")
    // Near-identical texts have asymmetric-prefix distance floor of ~0.16;
    // bound at 0.20 to allow for minor embedding variance.
    expect(res.body.distance).to.be.below(0.20)
  })

  it("unrelated texts → distance ≈ 0.3–0.6", async function () {
    const res = await agent()
      .post("/compare")
      .query({ method: "embedding" })
      .set("Authorization", authHeader())
      .send({
        source: "The Eiffel Tower is a wrought-iron lattice tower in Paris",
        target: "Quantum chromodynamics describes the strong interaction between quarks",
      })
      .expect(200)

    expect(res.body).to.have.property("distance")
    expect(res.body.distance).to.be.a("number")
    // Unrelated texts should have distance roughly around 0.3–0.6.
    expect(res.body.distance).to.be.above(0.2)
    expect(res.body.distance).to.be.below(0.8)
  })

  it("response includes role transparency fields", async function () {
    const res = await agent()
      .post("/compare")
      .query({ method: "embedding" })
      .set("Authorization", authHeader())
      .send({ source: "hello world", target: "goodbye world" })
      .expect(200)

    expect(res.body).to.have.property("distance")
    expect(res.body).to.have.property("source_role", "query")
    expect(res.body).to.have.property("target_role", "document")
  })

  it("missing method query param → 400", async function () {
    const res = await agent()
      .post("/compare")
      .set("Authorization", authHeader())
      .send({ source: "hello", target: "world" })
      .expect(400)

    expect(res.body).to.have.property("error")
    expect(res.body.error).to.include("method")
  })

  it("unknown method query param → 400", async function () {
    const res = await agent()
      .post("/compare")
      .query({ method: "lexical" })
      .set("Authorization", authHeader())
      .send({ source: "hello", target: "world" })
      .expect(400)

    expect(res.body).to.have.property("error")
    expect(res.body.error).to.include("unsupported")
  })

  it("missing source in body → 400", async function () {
    const res = await agent()
      .post("/compare")
      .query({ method: "embedding" })
      .set("Authorization", authHeader())
      .send({ target: "world" })
      .expect(400)

    expect(res.body).to.have.property("error")
    expect(res.body.error).to.include("source")
  })

  it("missing target in body → 400", async function () {
    const res = await agent()
      .post("/compare")
      .query({ method: "embedding" })
      .set("Authorization", authHeader())
      .send({ source: "hello" })
      .expect(400)

    expect(res.body).to.have.property("error")
    expect(res.body.error).to.include("target")
  })

  it("empty source string → 400", async function () {
    const res = await agent()
      .post("/compare")
      .query({ method: "embedding" })
      .set("Authorization", authHeader())
      .send({ source: "", target: "world" })
      .expect(400)

    expect(res.body).to.have.property("error")
    expect(res.body.error).to.include("source")
  })

  it("empty target string → 400", async function () {
    const res = await agent()
      .post("/compare")
      .query({ method: "embedding" })
      .set("Authorization", authHeader())
      .send({ source: "hello", target: "" })
      .expect(400)

    expect(res.body).to.have.property("error")
    expect(res.body.error).to.include("target")
  })

  it("no auth → 401", async function () {
    await agent()
      .post("/compare")
      .query({ method: "embedding" })
      .send({ source: "hello", target: "world" })
      .expect(401)
  })

  it("wrong auth → 401", async function () {
    await agent()
      .post("/compare")
      .query({ method: "embedding" })
      .set("Authorization", wrongAuthHeader())
      .send({ source: "hello", target: "world" })
      .expect(401)
  })
})
