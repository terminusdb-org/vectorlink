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

test("per-cluster optimal assignment beats greedy on the spec's failure case", () => {
  // No mutual-NN grounding here (asymmetric top-1s), so everything goes to Step 5.
  // Greedy would take a1->b1 (0.10), forcing a2->b2 (0.90): total 1.00, and a2 mis-paired.
  // Optimal: a1->b2 (0.20) + a2->b1 (0.30) = 0.50, both well-paired.
  const abtToBuy = m({
    a1: [{ id: "b1", distance: 0.1 }, { id: "b2", distance: 0.2 }],
    a2: [{ id: "b1", distance: 0.3 }, { id: "b2", distance: 0.9 }],
  });
  // Buy side reports NO reciprocal candidates, so NOTHING is mutually grounded
  // and both Abt fall to Step 5 within one connected component — exactly where
  // optimal-vs-greedy diverges. (Edges still come from the Abt→Buy direction.)
  const buyToAbt = m({ b1: [], b2: [] });
  const r = resolve(abtToBuy, buyToAbt, { k: 5, threshold: 0.5 });
  assert.equal(r.grounded.length, 0, "expected no grounded pairs in this construction");
  assert.equal(r.assigned.length, 2);
  const byAbt = new Map(r.assigned.map((p) => [p.abtId, p.buyId]));
  assert.equal(byAbt.get("a1"), "b2");
  assert.equal(byAbt.get("a2"), "b1");
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
  const r = resolve(abtToBuy, buyToAbt, { k: 5, threshold: 0.5, maxComponentSize: 4 });
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
