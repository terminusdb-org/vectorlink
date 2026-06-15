"use strict";

const { test } = require("node:test");
const assert = require("node:assert/strict");
const { parseArgs, cacheReusable, validateGatherCeiling } = require("../src/bench-v2");

// argv shim: parseArgs reads from index 2 (node + script are argv[0..1]).
const argv = (...args) => ["node", "bench-v2.js", ...args];

test("reuse is the DEFAULT (no flag = reload false)", () => {
  const c = parseArgs(argv());
  assert.equal(c.reload, false, "default must reuse the indexed snapshot, not re-index");
});

test("--reload and --force opt into a conscious re-index", () => {
  assert.equal(parseArgs(argv("--reload")).reload, true);
  assert.equal(parseArgs(argv("--force")).reload, true);
});

test("--no-load is a reuse alias (reload stays false)", () => {
  assert.equal(parseArgs(argv("--no-load")).reload, false);
});

test("--cardinality defaults to many-to-many and accepts all three modes", () => {
  assert.equal(parseArgs(argv()).cardinality, "many-to-many");
  assert.equal(parseArgs(argv("--cardinality", "one-to-many")).cardinality, "one-to-many");
  assert.equal(parseArgs(argv("--cardinality", "one-to-one")).cardinality, "one-to-one");
});

test("--cardinality rejects an unknown value (fail loud)", () => {
  assert.throws(() => parseArgs(argv("--cardinality", "bogus")), /cardinality/);
});

test("an unknown flag fails loud (poka-yoke against silently mis-measuring)", () => {
  assert.throws(() => parseArgs(argv("--definitely-not-a-flag")), /Unknown flag/);
});

// ── cacheReusable ────────────────────────────────────────────────────────────
const COMMIT = "bench-abt-buy-v2-c1";
const cache = (over = {}) => ({
  commit: COMMIT,
  gatherK: 10,
  gatherThreshold: 0.5,
  ...over,
});

test("null cache is never reusable", () => {
  assert.equal(cacheReusable(null, "search", 5, 0.5, COMMIT), false);
});

test("reusable when commit matches and gatherK >= k (search)", () => {
  assert.equal(cacheReusable(cache(), "search", 5, 0.5, COMMIT), true);
});

test("NOT reusable when the snapshot commit differs (a re-index invalidates ids)", () => {
  assert.equal(cacheReusable(cache(), "search", 5, 0.5, "other-commit"), false);
});

test("NOT reusable when cached gatherK < requested k (cannot slice UP)", () => {
  assert.equal(cacheReusable(cache({ gatherK: 3 }), "search", 5, 0.5, COMMIT), false);
});

test("duplicates: NOT reusable when cached gather τ is TIGHTER than requested τ", () => {
  // Cache gathered at τ=0.3 is missing edges a τ=0.5 run needs (engine pruned them).
  assert.equal(cacheReusable(cache({ gatherThreshold: 0.3 }), "duplicates", 5, 0.5, COMMIT), false);
});

test("duplicates: reusable when cached gather τ is >= requested τ (can prune down)", () => {
  assert.equal(cacheReusable(cache({ gatherThreshold: 0.5 }), "duplicates", 5, 0.3, COMMIT), true);
});

test("search/similar: τ does NOT gate reuse (gather is τ-independent for these modes)", () => {
  // search applies τ in the resolver, not at gather; a tighter cached τ is irrelevant.
  assert.equal(cacheReusable(cache({ gatherThreshold: 0.3 }), "search", 5, 0.5, COMMIT), true);
  assert.equal(cacheReusable(cache({ gatherThreshold: 0.3 }), "similar", 5, 0.5, COMMIT), true);
});

// ── --tau-* flags ───────────────────────────────────────────────────────────

test("--tau-one-to-one parses to tauOneToOne (number)", () => {
  const c = parseArgs(argv("--tau-one-to-one", "0.35"));
  assert.equal(c.tauOneToOne, 0.35);
});

test("--tau-one-to-many parses to tauOneToMany (number)", () => {
  const c = parseArgs(argv("--tau-one-to-many", "0.15"));
  assert.equal(c.tauOneToMany, 0.15);
});

test("--tau-many-to-one parses to tauManyToOne (number)", () => {
  const c = parseArgs(argv("--tau-many-to-one", "0.25"));
  assert.equal(c.tauManyToOne, 0.25);
});

test("all three --tau-* flags can be combined with --cardinality", () => {
  const c = parseArgs(argv(
    "--cardinality", "one-to-one",
    "--tau-one-to-one", "0.4",
    "--tau-one-to-many", "0.3",
    "--tau-many-to-one", "0.1",
  ));
  assert.equal(c.cardinality, "one-to-one");
  assert.equal(c.tauOneToOne, 0.4);
  assert.equal(c.tauOneToMany, 0.3);
  assert.equal(c.tauManyToOne, 0.1);
});

test("tau flags default to undefined (resolved later by resolveThresholds)", () => {
  const c = parseArgs(argv());
  assert.equal(c.tauOneToOne, undefined);
  assert.equal(c.tauOneToMany, undefined);
  assert.equal(c.tauManyToOne, undefined);
});

// ── --search-mode flag ──────────────────────────────────────────────────────

test("--search-mode defaults to vector (backward-compatible)", () => {
  assert.equal(parseArgs(argv()).searchMode, "vector");
});

test("--search-mode accepts vector, fts, hybrid", () => {
  assert.equal(parseArgs(argv("--search-mode", "vector")).searchMode, "vector");
  assert.equal(parseArgs(argv("--search-mode", "fts")).searchMode, "fts");
  assert.equal(parseArgs(argv("--search-mode", "hybrid")).searchMode, "hybrid");
});

test("--search-mode rejects an unknown value (fail loud)", () => {
  assert.throws(() => parseArgs(argv("--search-mode", "bogus")), /search-mode/);
});

// ── --threshold flag ────────────────────────────────────────────────────────

test("--threshold defaults to undefined (derived from maxActiveTau at runtime)", () => {
  assert.equal(parseArgs(argv()).threshold, undefined);
});

test("--threshold parses to a number in [0, 1]", () => {
  assert.equal(parseArgs(argv("--threshold", "0.7")).threshold, 0.7);
});

test("--threshold rejects out-of-range values (fail loud)", () => {
  assert.throws(() => parseArgs(argv("--threshold", "-0.1")), /threshold/);
  assert.throws(() => parseArgs(argv("--threshold", "1.5")), /threshold/);
});

test("--threshold can be combined with --tau-* flags (decouple gather from resolve)", () => {
  const c = parseArgs(argv(
    "--threshold", "0.7",
    "--tau-one-to-one", "0.4",
    "--tau-one-to-many", "0.2",
  ));
  assert.equal(c.threshold, 0.7);
  assert.equal(c.tauOneToOne, 0.4);
  assert.equal(c.tauOneToMany, 0.2);
});

// ── validateGatherCeiling (poka-yoke) ──────────────────────────────────────

test("validateGatherCeiling: no error when gatherTau >= derivedGatherTau", () => {
  assert.doesNotThrow(() => validateGatherCeiling(0.7, 0.45));
  assert.doesNotThrow(() => validateGatherCeiling(0.45, 0.45)); // equal is fine
});

test("validateGatherCeiling: FAIL LOUD when tau exceeds gather threshold (recall ceiling)", () => {
  assert.throws(
    () => validateGatherCeiling(0.3, 0.45),
    /RECALL CEILING VIOLATION/,
  );
});

test("validateGatherCeiling: includes both values in error message for diagnostics", () => {
  assert.throws(
    () => validateGatherCeiling(0.2, 0.5),
    /0\.500.*0\.200|0\.200.*0\.500/,
  );
});
