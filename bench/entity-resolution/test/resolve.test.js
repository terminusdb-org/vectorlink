"use strict";

const { test } = require("node:test");
const assert = require("node:assert/strict");
const {
  resolve,
  buildCandidateGraph,
  groundCore,
  setExtras,
  targetExtras,
  resolveThresholds,
  maxActiveTau,
  DEFAULTS,
  CARDINALITIES,
  CARDINALITY_PRESETS,
} = require("../src/resolve");

// Suppress unused-import lint for exports verified by require() above.
void CARDINALITIES;
void CARDINALITY_PRESETS;
void DEFAULTS;

// Helper: build a Map from a plain object of arrays.
const m = (obj) => new Map(Object.entries(obj));

// ── Threshold resolution + validation ────────────────────────────────────────

test("resolveThresholds: many-to-many preset activates all three tau", () => {
  const t = resolveThresholds({ cardinality: "many-to-many" });
  assert.equal(t.tauOneToOne, 0.45);
  assert.equal(t.tauOneToMany, 0.2);
  assert.equal(t.tauManyToOne, 0.2);
});

test("resolveThresholds: one-to-many preset disables target-side extras", () => {
  const t = resolveThresholds({ cardinality: "one-to-many" });
  assert.equal(t.tauOneToOne, 0.45);
  assert.equal(t.tauOneToMany, 0.2);
  assert.equal(t.tauManyToOne, null);
});

test("resolveThresholds: one-to-one preset disables both extras", () => {
  const t = resolveThresholds({ cardinality: "one-to-one" });
  assert.equal(t.tauOneToOne, 0.45);
  assert.equal(t.tauOneToMany, null);
  assert.equal(t.tauManyToOne, null);
});

test("resolveThresholds: explicit overrides take precedence over preset", () => {
  const t = resolveThresholds({ cardinality: "one-to-one", tauOneToMany: 0.3, tauManyToOne: 0.15 });
  assert.equal(t.tauOneToOne, 0.45); // from preset
  assert.equal(t.tauOneToMany, 0.3); // override
  assert.equal(t.tauManyToOne, 0.15); // override
});

test("resolveThresholds: extras-looser-than-core is ALLOWED (no error)", () => {
  // Independent knobs — no hard-enforced relationship.
  assert.doesNotThrow(() => resolveThresholds({ tauOneToOne: 0.2, tauOneToMany: 0.5, tauManyToOne: 0.8 }));
});

test("resolveThresholds: out-of-[0,1] fails loud", () => {
  assert.throws(() => resolveThresholds({ tauOneToOne: -0.1 }), /tauOneToOne/);
  assert.throws(() => resolveThresholds({ tauOneToOne: 1.5 }), /tauOneToOne/);
  assert.throws(() => resolveThresholds({ tauOneToMany: 2 }), /tauOneToMany/);
  assert.throws(() => resolveThresholds({ tauManyToOne: -1 }), /tauManyToOne/);
});

test("resolveThresholds: unknown cardinality fails loud", () => {
  assert.throws(() => resolveThresholds({ cardinality: "bogus" }), /cardinality/);
});

test("maxActiveTau returns the loosest active tau", () => {
  assert.equal(maxActiveTau({ tauOneToOne: 0.4, tauOneToMany: 0.2, tauManyToOne: 0.3 }), 0.4);
  assert.equal(maxActiveTau({ tauOneToOne: 0.2, tauOneToMany: 0.5, tauManyToOne: null }), 0.5);
  assert.equal(maxActiveTau({ tauOneToOne: 0.3, tauOneToMany: null, tauManyToOne: null }), 0.3);
});

// ── Candidate graph ──────────────────────────────────────────────────────────

test("buildCandidateGraph caps edges at tau and keeps the minimum observed distance", () => {
  const setToTarget = m({ a1: [{ id: "b1", distance: 0.2 }, { id: "b2", distance: 0.7 }] });
  const targetToSet = m({ b1: [{ id: "a1", distance: 0.25 }], b2: [{ id: "a1", distance: 0.7 }] });
  const g = buildCandidateGraph(setToTarget, targetToSet, 5, 0.5);
  assert.equal(g.edges.size, 1);
  const [edge] = [...g.edges.values()];
  assert.equal(edge.setId, "a1");
  assert.equal(edge.targetId, "b1");
  assert.equal(edge.distance, 0.2);
});

// ── Core grounding (tauOneToOne) ─────────────────────────────────────────────

test("groundCore: only mutual pairs passing tauOneToOne are grounded", () => {
  const setToTarget = m({
    a1: [{ id: "b1", distance: 0.1 }],
    a2: [{ id: "b2", distance: 0.3 }],
  });
  const targetToSet = m({
    b1: [{ id: "a1", distance: 0.1 }],
    b2: [{ id: "a3", distance: 0.2 }], // not mutual with a2
  });
  const g = buildCandidateGraph(setToTarget, targetToSet, 5, 0.5);
  const core = groundCore(g, 0.5);
  assert.equal(core.length, 1);
  assert.equal(core[0].setId, "a1");
  assert.equal(core[0].targetId, "b1");
  assert.equal(core[0].stage, "core");
});

test("groundCore: tighter tauOneToOne excludes a mutual pair that is too distant", () => {
  const setToTarget = m({ a1: [{ id: "b1", distance: 0.3 }] });
  const targetToSet = m({ b1: [{ id: "a1", distance: 0.3 }] });
  const g = buildCandidateGraph(setToTarget, targetToSet, 5, 0.5);
  const core = groundCore(g, 0.2);
  assert.equal(core.length, 0, "pair at d=0.3 excluded by tauOneToOne=0.2");
});

test("groundCore: per-set-record nearest mutual (one core pair per set record)", () => {
  const setToTarget = m({ a1: [{ id: "b1", distance: 0.1 }, { id: "b2", distance: 0.2 }] });
  const targetToSet = m({
    b1: [{ id: "a1", distance: 0.1 }],
    b2: [{ id: "a1", distance: 0.2 }],
  });
  const g = buildCandidateGraph(setToTarget, targetToSet, 5, 0.5);
  const core = groundCore(g, 0.5);
  assert.equal(core.length, 1);
  assert.equal(core[0].targetId, "b1", "nearest mutual wins");
});

// ── Set-side extras (tauOneToMany) ───────────────────────────────────────────

test("setExtras: emits additional targets beyond the core pair, passing tauOneToMany", () => {
  const setToTarget = m({ a1: [{ id: "b1", distance: 0.1 }, { id: "b2", distance: 0.15 }, { id: "b3", distance: 0.4 }] });
  const targetToSet = m({
    b1: [{ id: "a1", distance: 0.1 }],
    b2: [],
    b3: [],
  });
  const g = buildCandidateGraph(setToTarget, targetToSet, 5, 0.5);
  const core = groundCore(g, 0.5); // a1::b1 grounded
  assert.equal(core.length, 1);
  // tauOneToMany = 0.2: b2 at 0.15 passes, b3 at 0.4 does not.
  const extras = setExtras(g, core, 0.2);
  const extraTargets = extras.map((e) => e.targetId);
  assert.ok(extraTargets.includes("b2"), "b2 within tauOneToMany");
  assert.ok(!extraTargets.includes("b3"), "b3 excluded by tauOneToMany");
  assert.ok(extras.every((e) => e.stage === "set_extra"));
});

test("setExtras: disabled when null", () => {
  const g = buildCandidateGraph(m({ a1: [{ id: "b1", distance: 0.1 }] }), m({ b1: [] }), 5, 0.5);
  const extras = setExtras(g, [], null);
  assert.equal(extras.length, 0);
});

// ── Target-side extras (tauManyToOne) ────────────────────────────────────────

test("targetExtras: emits additional set records for a target, passing tauManyToOne", () => {
  // a2::b1 is a target-directed-only edge: b1 has a2 in its top-K (the target
  // "found" a2), but a2 does NOT have b1 in its top-K (a2's top-K points
  // elsewhere). This makes it a pure target extra (not core, not set_extra).
  const setToTarget = m({
    a1: [{ id: "b1", distance: 0.1 }],
    a2: [{ id: "b2", distance: 0.1 }], // a2's top-K points to b2, not b1
  });
  const targetToSet = m({
    b1: [{ id: "a1", distance: 0.1 }, { id: "a2", distance: 0.15 }],
    b2: [],
  });
  const g = buildCandidateGraph(setToTarget, targetToSet, 5, 0.5);
  const core = groundCore(g, 0.5); // a1::b1 is mutual (a1 has b1, b1 has a1)
  assert.equal(core.length, 1, "only a1::b1 is core");
  const extras = targetExtras(g, core, 0.2);
  assert.ok(extras.some((e) => e.setId === "a2" && e.targetId === "b1"));
  assert.ok(extras.every((e) => e.stage === "target_extra"));
});

test("targetExtras: disabled when null", () => {
  const g = buildCandidateGraph(m({ a1: [{ id: "b1", distance: 0.1 }] }), m({ b1: [] }), 5, 0.5);
  const extras = targetExtras(g, [], null);
  assert.equal(extras.length, 0);
});

// ── Full resolve: many-to-many (DEFAULT) ─────────────────────────────────────

test("many-to-many: core + set_extra + target_extra all present when edges qualify", () => {
  // Topology with all 3 stages:
  //   - a1::b1 mutual core (both in each other's top-K, d=0.05)
  //   - a1::b2 set_extra: a1 has b2 in its top-K at d=0.08 (set-directed, d <= tauOneToMany=0.1)
  //   - a2::b1 target_extra: b1 has a2 in its top-K at d=0.15 (target-directed),
  //     but a2 does NOT have b1 in its top-K (a2's neighbours go elsewhere).
  //     d=0.15 > tauOneToMany=0.1 (so not a set_extra even if it were set-directed)
  //     d=0.15 <= tauManyToOne=0.2 (qualifies as target_extra).
  const setToTarget = m({
    a1: [{ id: "b1", distance: 0.05 }, { id: "b2", distance: 0.08 }],
    a2: [{ id: "b3", distance: 0.05 }], // a2's top-K points to b3, NOT b1
  });
  const targetToSet = m({
    b1: [{ id: "a1", distance: 0.05 }, { id: "a2", distance: 0.15 }],
    b2: [],
    b3: [],
  });
  const r = resolve(setToTarget, targetToSet, {
    k: 5, tauOneToOne: 0.45, tauOneToMany: 0.1, tauManyToOne: 0.2,
  });
  const stages = new Set(r.matched.map((p) => p.stage));
  assert.ok(stages.has("core"), "core pairs present (a1::b1 mutual)");
  assert.ok(stages.has("set_extra"), "set extras present (a1::b2 at d=0.08, set-directed)");
  assert.ok(stages.has("target_extra"), "target extras present (a2::b1 at d=0.15, target-directed only)");
  assert.equal(r.set_only.length, 0);
});

test("many-to-many: tightening extras collapses toward core-only", () => {
  const setToTarget = m({
    a1: [{ id: "b1", distance: 0.1 }, { id: "b2", distance: 0.3 }],
    a2: [{ id: "b1", distance: 0.25 }],
  });
  const targetToSet = m({
    b1: [{ id: "a1", distance: 0.1 }],
    b2: [],
  });
  const r = resolve(setToTarget, targetToSet, {
    k: 5, tauOneToOne: 0.45, tauOneToMany: 0.05, tauManyToOne: 0.05,
  });
  assert.ok(r.matched.length >= 1);
  assert.ok(r.matched.every((p) => p.stage === "core"), "only core when extras tau is very tight");
});

test("many-to-many: multiple targets per set record AND multiple set records per target", () => {
  const setToTarget = m({
    a1: [{ id: "b1", distance: 0.05 }, { id: "b2", distance: 0.08 }],
    a2: [{ id: "b1", distance: 0.08 }],
  });
  const targetToSet = m({
    b1: [{ id: "a1", distance: 0.05 }],
    b2: [],
  });
  const r = resolve(setToTarget, targetToSet, {
    k: 5, tauOneToOne: 0.45, tauOneToMany: 0.2, tauManyToOne: 0.2,
  });
  const keys = r.matched.map((p) => `${p.setId}::${p.targetId}`).sort();
  assert.ok(keys.includes("a1::b1"), "core pair");
  assert.ok(keys.includes("a1::b2"), "set extra");
  assert.ok(keys.includes("a2::b1"), "target extra");
});

// ── Full resolve: one-to-many ────────────────────────────────────────────────

test("one-to-many: no target-side extras (tauManyToOne disabled by preset)", () => {
  const setToTarget = m({
    a1: [{ id: "b1", distance: 0.05 }, { id: "b2", distance: 0.1 }],
    a2: [{ id: "b1", distance: 0.1 }],
  });
  const targetToSet = m({
    b1: [{ id: "a1", distance: 0.05 }],
    b2: [],
  });
  const r = resolve(setToTarget, targetToSet, { k: 5, cardinality: "one-to-many" });
  const stages = new Set(r.matched.map((p) => p.stage));
  assert.ok(!stages.has("target_extra"), "target extras disabled under one-to-many");
  assert.ok(stages.has("core"));
});

// ── Full resolve: one-to-one ─────────────────────────────────────────────────

test("one-to-one: only core pairs (both extras disabled by preset)", () => {
  const setToTarget = m({
    a1: [{ id: "b1", distance: 0.1 }, { id: "b2", distance: 0.15 }],
    a2: [{ id: "b1", distance: 0.12 }],
  });
  const targetToSet = m({
    b1: [{ id: "a1", distance: 0.1 }, { id: "a2", distance: 0.12 }],
    b2: [{ id: "a1", distance: 0.15 }],
  });
  const r = resolve(setToTarget, targetToSet, { k: 5, cardinality: "one-to-one" });
  assert.ok(r.matched.every((p) => p.stage === "core"), "only core under one-to-one preset");
});

// ── 3-PARTITION OUTPUT ───────────────────────────────────────────────────────

test("3-partition: set record with no edge under any active tau -> set_only", () => {
  const setToTarget = m({
    a1: [{ id: "b1", distance: 0.1 }],
    a2: [{ id: "b2", distance: 0.8 }], // beyond all tau
  });
  const targetToSet = m({ b1: [{ id: "a1", distance: 0.1 }], b2: [] });
  const r = resolve(setToTarget, targetToSet, {
    k: 5, tauOneToOne: 0.45, tauOneToMany: 0.2, tauManyToOne: 0.2,
  });
  assert.ok(r.set_only.includes("a2"), "a2 in set_only");
  assert.ok(!r.matched.some((p) => p.setId === "a2"));
});

test("3-partition: target record with no edge under any active tau -> target_only", () => {
  const setToTarget = m({ a1: [{ id: "b1", distance: 0.1 }] });
  const targetToSet = m({
    b1: [{ id: "a1", distance: 0.1 }],
    b2: [{ id: "a1", distance: 0.8 }], // beyond all tau
  });
  const r = resolve(setToTarget, targetToSet, {
    k: 5, tauOneToOne: 0.45, tauOneToMany: 0.2, tauManyToOne: 0.2,
  });
  assert.ok(r.target_only.includes("b2"), "b2 in target_only");
});

test("3-partition: counts are consistent", () => {
  const setToTarget = m({
    a1: [{ id: "b1", distance: 0.1 }],
    a2: [{ id: "b2", distance: 0.8 }],
    a3: [{ id: "b1", distance: 0.15 }],
  });
  const targetToSet = m({
    b1: [{ id: "a1", distance: 0.1 }, { id: "a3", distance: 0.15 }],
    b2: [],
    b3: [],
  });
  const r = resolve(setToTarget, targetToSet, {
    k: 5, tauOneToOne: 0.45, tauOneToMany: 0.2, tauManyToOne: 0.2,
  });
  const matchedSetIds = new Set(r.matched.map((p) => p.setId));
  const matchedTargetIds = new Set(r.matched.map((p) => p.targetId));
  const allSet = [...setToTarget.keys()];
  const allTarget = [...targetToSet.keys()];
  for (const id of allSet) {
    assert.ok(matchedSetIds.has(id) || r.set_only.includes(id), `set id ${id} in matched or set_only`);
  }
  for (const id of allTarget) {
    assert.ok(matchedTargetIds.has(id) || r.target_only.includes(id), `target id ${id} in matched or target_only`);
  }
});

// ── Deduplication ────────────────────────────────────────────────────────────

test("deduplication: same pair from set_extra and target_extra is kept once (higher-confidence stage)", () => {
  const setToTarget = m({
    a1: [{ id: "b1", distance: 0.05 }],
    a2: [{ id: "b1", distance: 0.1 }],
  });
  const targetToSet = m({ b1: [{ id: "a1", distance: 0.05 }] });
  const r = resolve(setToTarget, targetToSet, {
    k: 5, tauOneToOne: 0.45, tauOneToMany: 0.2, tauManyToOne: 0.2,
  });
  const a2b1 = r.matched.filter((p) => p.setId === "a2" && p.targetId === "b1");
  assert.equal(a2b1.length, 1, "deduplicated to one entry");
  assert.equal(a2b1[0].stage, "set_extra");
});

// ── Default cardinality ──────────────────────────────────────────────────────

test("default cardinality is many-to-many", () => {
  const setToTarget = m({ a1: [{ id: "b1", distance: 0.1 }] });
  const targetToSet = m({ b1: [{ id: "a1", distance: 0.1 }] });
  const r = resolve(setToTarget, targetToSet, { k: 5 });
  assert.equal(r.stats.cardinality, "many-to-many");
});

// ── Preset + override interaction ────────────────────────────────────────────

test("one-to-one preset with explicit tauOneToMany override activates set extras", () => {
  const setToTarget = m({
    a1: [{ id: "b1", distance: 0.05 }, { id: "b2", distance: 0.1 }],
  });
  const targetToSet = m({ b1: [{ id: "a1", distance: 0.05 }], b2: [] });
  const r = resolve(setToTarget, targetToSet, {
    k: 5, cardinality: "one-to-one", tauOneToMany: 0.2,
  });
  const hasSetExtra = r.matched.some((p) => p.stage === "set_extra");
  assert.ok(hasSetExtra, "explicit tauOneToMany override activates set extras even on one-to-one preset");
});

// ── INDEPENDENCE ACCEPTANCE TEST (the PO's bug symptom) ──────────────────────
// Proves that changing tauOneToMany ALONE (tauManyToOne fixed) changes the
// matched set, AND changing tauManyToOne alone changes it — i.e. the three
// thresholds are genuinely independent.

test("INDEPENDENCE: changing tauOneToMany alone changes matched count (tauManyToOne fixed)", () => {
  // A graph with set-directed edges at various distances so tauOneToMany sweep
  // produces different matched counts.
  const setToTarget = m({
    a1: [{ id: "b1", distance: 0.05 }, { id: "b2", distance: 0.12 }, { id: "b3", distance: 0.18 }],
  });
  const targetToSet = m({
    b1: [{ id: "a1", distance: 0.05 }],
    b2: [],
    b3: [],
  });
  // tauManyToOne fixed at null (disabled) → only set-extras vary.
  const rTight = resolve(setToTarget, targetToSet, {
    k: 5, tauOneToOne: 0.45, tauOneToMany: 0.1, tauManyToOne: null,
  });
  const rLoose = resolve(setToTarget, targetToSet, {
    k: 5, tauOneToOne: 0.45, tauOneToMany: 0.2, tauManyToOne: null,
  });
  assert.notEqual(rTight.matched.length, rLoose.matched.length,
    "different tauOneToMany must produce different matched counts (independence)");
  assert.ok(rLoose.matched.length > rTight.matched.length,
    "looser tauOneToMany admits more matches");
});

test("INDEPENDENCE: changing tauManyToOne alone changes matched count (tauOneToMany fixed)", () => {
  // A graph with target-directed-only edges at various distances.
  // a2 and a3 are NOT in a1's setTopK — they come from b1's targetTopK only.
  const setToTarget = m({
    a1: [{ id: "b1", distance: 0.05 }],
    a2: [{ id: "b2", distance: 0.05 }], // a2's top-K is b2, not b1
    a3: [{ id: "b3", distance: 0.05 }], // a3's top-K is b3, not b1
  });
  const targetToSet = m({
    b1: [{ id: "a1", distance: 0.05 }, { id: "a2", distance: 0.12 }, { id: "a3", distance: 0.18 }],
    b2: [],
    b3: [],
  });
  // tauOneToMany fixed at null (disabled) → only target-extras vary.
  const rTight = resolve(setToTarget, targetToSet, {
    k: 5, tauOneToOne: 0.45, tauOneToMany: null, tauManyToOne: 0.1,
  });
  const rLoose = resolve(setToTarget, targetToSet, {
    k: 5, tauOneToOne: 0.45, tauOneToMany: null, tauManyToOne: 0.2,
  });
  assert.notEqual(rTight.matched.length, rLoose.matched.length,
    "different tauManyToOne must produce different matched counts (independence)");
  assert.ok(rLoose.matched.length > rTight.matched.length,
    "looser tauManyToOne admits more matches");
});

test("INDEPENDENCE: one-to-many produces FEWER matches than many-to-many (distinct modes)", () => {
  // Graph with both set-directed and target-directed-only edges.
  const setToTarget = m({
    a1: [{ id: "b1", distance: 0.05 }, { id: "b2", distance: 0.1 }],
    a2: [{ id: "b3", distance: 0.05 }], // a2's top-K is b3, not b1
  });
  const targetToSet = m({
    b1: [{ id: "a1", distance: 0.05 }, { id: "a2", distance: 0.12 }], // b1 found a2
    b2: [],
    b3: [],
  });
  const rMM = resolve(setToTarget, targetToSet, {
    k: 5, tauOneToOne: 0.45, tauOneToMany: 0.2, tauManyToOne: 0.2,
  });
  const r1M = resolve(setToTarget, targetToSet, {
    k: 5, cardinality: "one-to-many",
  });
  assert.ok(rMM.matched.length > r1M.matched.length,
    "many-to-many includes target-directed extras that one-to-many excludes");
});
