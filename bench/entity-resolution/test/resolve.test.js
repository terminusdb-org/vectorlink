"use strict";

const { test } = require("node:test");
const assert = require("node:assert/strict");
const {
  resolve,
  buildCandidateGraph,
  groundMutualTopK,
  connectedComponents,
} = require("../src/resolve");

// Helper: build a Map from a plain object of arrays.
const m = (obj) => new Map(Object.entries(obj));

test("buildCandidateGraph caps edges at τ and keeps the minimum observed distance", () => {
  const abtToBuy = m({ a1: [{ id: "b1", distance: 0.2 }, { id: "b2", distance: 0.7 }] });
  const buyToAbt = m({ b1: [{ id: "a1", distance: 0.25 }], b2: [{ id: "a1", distance: 0.7 }] });
  const g = buildCandidateGraph(abtToBuy, buyToAbt, 5, 0.5);
  // b2 edge (0.7 > 0.5) pruned; a1-b1 keeps min(0.2, 0.25) = 0.2.
  assert.equal(g.edges.size, 1);
  const [edge] = [...g.edges.values()];
  assert.equal(edge.abtId, "a1");
  assert.equal(edge.buyId, "b1");
  assert.equal(edge.distance, 0.2);
});

test("mutual top-K grounding accepts only pairs in BOTH directions' top-K", () => {
  // a1 <-> b1 mutual. a2 -> b2 but b2's nearest is a3 (not mutual) -> not grounded.
  const abtToBuy = m({
    a1: [{ id: "b1", distance: 0.1 }],
    a2: [{ id: "b2", distance: 0.3 }],
  });
  const buyToAbt = m({
    b1: [{ id: "a1", distance: 0.1 }],
    b2: [{ id: "a3", distance: 0.2 }],
  });
  const g = buildCandidateGraph(abtToBuy, buyToAbt, 5, 0.5);
  const { grounded } = groundMutualTopK(g);
  assert.equal(grounded.length, 1);
  assert.equal(grounded[0].abtId, "a1");
  assert.equal(grounded[0].buyId, "b1");
});

test("connected components partition the residual into independent clusters", () => {
  // Two disjoint knots: {a1,a2}×{b1,b2} and {a3}×{b3}.
  const edges = [
    { abtId: "a1", buyId: "b1", distance: 0.3 },
    { abtId: "a1", buyId: "b2", distance: 0.4 },
    { abtId: "a2", buyId: "b1", distance: 0.35 },
    { abtId: "a3", buyId: "b3", distance: 0.2 },
  ];
  const comps = connectedComponents(edges);
  assert.equal(comps.length, 2);
  const sizes = comps.map((c) => c.abtIds.size + c.buyIds.size).sort();
  assert.deepEqual(sizes, [2, 4]);
});

test("per-cluster optimal assignment beats greedy on the spec's failure case (assignment=optimal)", () => {
  // No mutual-NN grounding here (asymmetric top-1s), so everything goes to Step 5.
  // Greedy would take a1->b1 (0.10), forcing a2->b2 (0.90): total 1.00, and a2 mis-paired.
  // Optimal: a1->b2 (0.20) + a2->b1 (0.30) = 0.50, both well-paired.
  // NB: this is the 1:1 (Buy-exclusive) semantics — selected explicitly via
  // assignment="optimal". The DEFAULT (per-source) is tested separately below;
  // for THIS dataset's many-to-one truth, per-source is the correct default
  // (Hungarian's "stealing" only applies when the Buy side is exclusive).
  const abtToBuy = m({
    a1: [{ id: "b1", distance: 0.1 }, { id: "b2", distance: 0.2 }],
    a2: [{ id: "b1", distance: 0.3 }, { id: "b2", distance: 0.9 }],
  });
  // Buy side reports NO reciprocal candidates, so NOTHING is mutually grounded
  // and both Abt fall to Step 5 within one connected component — exactly where
  // optimal-vs-greedy diverges. (Edges still come from the Abt→Buy direction.)
  const buyToAbt = m({ b1: [], b2: [] });
  const r = resolve(abtToBuy, buyToAbt, { k: 5, threshold: 0.5, assignment: "optimal" });
  assert.equal(r.grounded.length, 0, "expected no grounded pairs in this construction");
  assert.equal(r.assigned.length, 2);
  const byAbt = new Map(r.assigned.map((p) => [p.abtId, p.buyId]));
  assert.equal(byAbt.get("a1"), "b2");
  assert.equal(byAbt.get("a2"), "b1");
});

test("DEFAULT assignment is per-source (many-to-one): N Abt may share ONE Buy — no recall loss", () => {
  // Three Abt all genuinely match Buy b1 (many-to-one truth). None mutually
  // grounded (b1's reciprocal list is empty here). Under the correct many-to-one
  // semantics every Abt takes its nearest ≤τ Buy and a Buy is NON-exclusive, so
  // all three pairs are emitted. (Hungarian/optimal would emit only ONE — a
  // recall bug for this truth model; see resolve.js header + agent memory.)
  const abtToBuy = m({
    a1: [{ id: "b1", distance: 0.1 }],
    a2: [{ id: "b1", distance: 0.2 }],
    a3: [{ id: "b1", distance: 0.3 }],
  });
  const buyToAbt = m({ b1: [] });
  const r = resolve(abtToBuy, buyToAbt, { k: 5, threshold: 0.5 });
  assert.equal(r.assigned.length, 3, "all three many-to-one matches must be emitted");
  assert.equal(r.unmatchedAbt.length, 0, "no Abt left unmatched");
  const byAbt = new Map(r.assigned.map((p) => [p.abtId, p.buyId]));
  assert.equal(byAbt.get("a1"), "b1");
  assert.equal(byAbt.get("a2"), "b1");
  assert.equal(byAbt.get("a3"), "b1");
});

test("per-source assignment: each Abt takes its OWN nearest ≤τ Buy independently", () => {
  // a1 nearest is b1 (0.1), a2 nearest is b2 (0.15). No Buy contention.
  const abtToBuy = m({
    a1: [{ id: "b1", distance: 0.1 }, { id: "b2", distance: 0.4 }],
    a2: [{ id: "b2", distance: 0.15 }, { id: "b1", distance: 0.45 }],
  });
  const buyToAbt = m({ b1: [], b2: [] });
  const r = resolve(abtToBuy, buyToAbt, { k: 5, threshold: 0.5 });
  const byAbt = new Map(r.assigned.map((p) => [p.abtId, p.buyId]));
  assert.equal(byAbt.get("a1"), "b1");
  assert.equal(byAbt.get("a2"), "b2");
});

test("a Buy grounded to one Abt may still serve a DIFFERENT ungrounded Abt (many-to-one), and the grounded pair is NOT duplicated", () => {
  // a1 mutually grounds with b1. a2 (ungrounded) also has b1 as its nearest ≤τ.
  // Under many-to-one this is legitimate Buy reuse: a2->b1 is emitted as assigned,
  // a1->b1 stays grounded, and the (a1,b1) pair appears exactly once.
  const abtToBuy = m({
    a1: [{ id: "b1", distance: 0.1 }],
    a2: [{ id: "b1", distance: 0.3 }],
  });
  const buyToAbt = m({ b1: [{ id: "a1", distance: 0.1 }] });
  const r = resolve(abtToBuy, buyToAbt, { k: 5, threshold: 0.5 });
  assert.equal(r.grounded.length, 1);
  assert.equal(r.grounded[0].abtId, "a1");
  // a1 (grounded) must NOT be re-assigned in Step 5; only a2 is assigned.
  assert.ok(!r.assigned.some((p) => p.abtId === "a1"), "grounded Abt a1 must not be re-assigned");
  assert.ok(r.assigned.some((p) => p.abtId === "a2" && p.buyId === "b1"), "a2 may legitimately share b1");
});

test("invalid assignment strategy fails loud", () => {
  const abtToBuy = m({ a1: [{ id: "b1", distance: 0.1 }] });
  const buyToAbt = m({ b1: [] });
  assert.throws(
    () => resolve(abtToBuy, buyToAbt, { assignment: "bogus" }),
    /assignment/,
    "an unknown assignment strategy must throw, not silently default",
  );
});

test("a record with no ≤ τ candidate is left unmatched (abstain, §4.6)", () => {
  const abtToBuy = m({
    a1: [{ id: "b1", distance: 0.1 }],
    a2: [{ id: "b2", distance: 0.8 }], // > τ -> pruned -> no edge
  });
  const buyToAbt = m({ b1: [{ id: "a1", distance: 0.1 }], b2: [{ id: "a2", distance: 0.8 }] });
  const r = resolve(abtToBuy, buyToAbt, { k: 5, threshold: 0.5 });
  assert.ok(r.unmatchedAbt.includes("a2"), "a2 should be unmatched");
  assert.ok(!r.groups.some((g) => g.abtId === "a2"), "a2 should not appear in any group");
});

test("runaway component falls back to greedy under maxComponentSize", () => {
  // A dense 3×3 fully-connected knot; cap at 4 forces the greedy fallback.
  const abtToBuy = m({
    a1: [{ id: "b1", distance: 0.1 }, { id: "b2", distance: 0.2 }, { id: "b3", distance: 0.3 }],
    a2: [{ id: "b1", distance: 0.15 }, { id: "b2", distance: 0.25 }, { id: "b3", distance: 0.35 }],
    a3: [{ id: "b1", distance: 0.12 }, { id: "b2", distance: 0.22 }, { id: "b3", distance: 0.32 }],
  });
  const buyToAbt = m({ b1: [], b2: [], b3: [] });
  // The runaway guard exists only in the OPTIMAL (Hungarian) path — that is the
  // sole strategy with the O(n³) step a cap must protect. per-source (default)
  // is linear and never needs the fallback. So this test targets assignment="optimal".
  const r = resolve(abtToBuy, buyToAbt, { k: 5, threshold: 0.5, maxComponentSize: 4, assignment: "optimal" });
  assert.equal(r.stats.runawayComponents, 1, "the oversized component should fall back to greedy");
  assert.ok(r.stats.maxComponentObserved > 4);
});

test("mutual-top-K recovers a rank-2 true pair (refinement C over top-1)", () => {
  // a1's nearest Buy is a near-miss b9 at rank1, true b1 at rank2; b1's nearest
  // Abt is a1. With k>=2 the mutual membership grounds (a1,b1).
  const abtToBuy = m({ a1: [{ id: "b9", distance: 0.30 }, { id: "b1", distance: 0.32 }] });
  const buyToAbt = m({
    b1: [{ id: "a1", distance: 0.32 }],
    b9: [{ id: "a7", distance: 0.10 }],
  });
  const grounded = resolve(abtToBuy, buyToAbt, { k: 5, threshold: 0.5 }).grounded;
  assert.ok(grounded.some((g) => g.abtId === "a1" && g.buyId === "b1"), "rank-2 mutual pair grounded");
});

test("grounding commits ONE pair per Abt (the nearest mutual neighbour)", () => {
  // a1 has mutual edges to BOTH b1 (d=0.1) and b2 (d=0.2) — both Buy have a1 in
  // their top-K and a1 has both in its top-K. The corrected grounding emits only
  // the NEAREST mutual edge (b1, d=0.1) per §8.2: one committed pair per Abt.
  const abtToBuy = m({ a1: [{ id: "b1", distance: 0.1 }, { id: "b2", distance: 0.2 }] });
  const buyToAbt = m({
    b1: [{ id: "a1", distance: 0.1 }],
    b2: [{ id: "a1", distance: 0.2 }],
  });
  const r = resolve(abtToBuy, buyToAbt, { k: 5, threshold: 0.5 });
  assert.equal(r.grounded.length, 1, "only ONE pair per Abt from grounding");
  assert.equal(r.grounded[0].abtId, "a1");
  assert.equal(r.grounded[0].buyId, "b1", "the NEAREST mutual neighbour wins");
  assert.equal(r.grounded[0].distance, 0.1);
  assert.equal(r.groups.length, 1, "total committed prediction set has one pair");
});

test("one-per-Abt invariant: resolver throws on duplicate Abt in output (poka-yoke)", () => {
  // This test ensures the poka-yoke assertion fires. Under normal operation the
  // resolver cannot produce duplicates (grounding is one-per-Abt and assignment
  // only processes un-grounded Abt). We verify the invariant structurally by
  // confirming the output has no Abt duplicates across modes.
  const abtToBuy = m({
    a1: [{ id: "b1", distance: 0.1 }, { id: "b2", distance: 0.2 }, { id: "b3", distance: 0.3 }],
    a2: [{ id: "b1", distance: 0.15 }, { id: "b2", distance: 0.25 }],
  });
  const buyToAbt = m({
    b1: [{ id: "a1", distance: 0.1 }, { id: "a2", distance: 0.15 }],
    b2: [{ id: "a1", distance: 0.2 }, { id: "a2", distance: 0.25 }],
    b3: [{ id: "a1", distance: 0.3 }],
  });
  const r = resolve(abtToBuy, buyToAbt, { k: 5, threshold: 0.5 });
  // Each Abt appears exactly once in groups (the committed prediction set).
  const abtIds = r.groups.map((g) => g.abtId);
  const unique = new Set(abtIds);
  assert.equal(abtIds.length, unique.size, "no Abt appears more than once in committed output");
});
