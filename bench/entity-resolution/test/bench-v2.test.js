"use strict";

const { test } = require("node:test");
const assert = require("node:assert/strict");
const { parseArgs, cacheReusable } = require("../src/bench-v2");

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

test("--assignment defaults to per-source and accepts optimal", () => {
  assert.equal(parseArgs(argv()).assignment, "per-source");
  assert.equal(parseArgs(argv("--assignment", "optimal")).assignment, "optimal");
});

test("--assignment rejects an unknown strategy (fail loud)", () => {
  assert.throws(() => parseArgs(argv("--assignment", "bogus")), /assignment/);
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
