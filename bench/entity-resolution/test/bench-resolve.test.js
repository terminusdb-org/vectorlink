"use strict";

const { test } = require("node:test");
const assert = require("node:assert/strict");
const { parseArgs, mapMatchedToScorerFormat, shortDomain, fullDomain } = require("../src/bench-resolve");

// argv shim: parseArgs reads from index 2 (node + script are argv[0..1]).
const argv = (...args) => ["node", "bench-resolve.js", ...args];

// ── parseArgs ────────────────────────────────────────────────────────────────

test("parseArgs: defaults are sensible for Abt-Buy core-only run", () => {
  const c = parseArgs(argv());
  assert.equal(c.threshold, 0.5);
  assert.equal(c.tauOneToOne, 0.45);
  assert.equal(c.tauOneToMany, undefined);
  assert.equal(c.tauManyToOne, undefined);
  assert.equal(c.k, 5);
  assert.deepEqual(c.setDocTypes, ["Abt"]);
  assert.deepEqual(c.targetDocTypes, ["Buy"]);
  assert.equal(c.dataset, "abt-buy-v2");
});

test("parseArgs: all flags parse correctly", () => {
  const c = parseArgs(argv(
    "--threshold", "0.6",
    "--tau-one-to-one", "0.4",
    "--tau-one-to-many", "0.2",
    "--tau-many-to-one", "0.15",
    "--k", "10",
    "--domain", "admin/my_domain",
    "--commit", "abc123",
    "--set-doc-types", "Product,Item",
    "--target-doc-types", "Catalogue",
  ));
  assert.equal(c.threshold, 0.6);
  assert.equal(c.tauOneToOne, 0.4);
  assert.equal(c.tauOneToMany, 0.2);
  assert.equal(c.tauManyToOne, 0.15);
  assert.equal(c.k, 10);
  assert.equal(c.domain, "admin/my_domain");
  assert.equal(c.commit, "abc123");
  assert.deepEqual(c.setDocTypes, ["Product", "Item"]);
  assert.deepEqual(c.targetDocTypes, ["Catalogue"]);
});

test("parseArgs: positional argument sets dataset", () => {
  const c = parseArgs(argv("abt-buy-v2"));
  assert.equal(c.dataset, "abt-buy-v2");
});

test("parseArgs: unknown flag fails loud", () => {
  assert.throws(() => parseArgs(argv("--bogus")), /Unknown flag/);
});

test("parseArgs: threshold out of range fails loud", () => {
  assert.throws(() => parseArgs(argv("--threshold", "1.5")), /threshold/);
  assert.throws(() => parseArgs(argv("--threshold", "-0.1")), /threshold/);
});

test("parseArgs: tau_one_to_one out of range fails loud", () => {
  assert.throws(() => parseArgs(argv("--tau-one-to-one", "2.0")), /tau-one-to-one/);
});

test("parseArgs: tau > threshold is caught client-side (silent-recall trap)", () => {
  assert.throws(
    () => parseArgs(argv("--threshold", "0.3", "--tau-one-to-one", "0.5")),
    /silent-recall trap/,
  );
});

test("parseArgs: tau_one_to_many > threshold is caught", () => {
  assert.throws(
    () => parseArgs(argv("--threshold", "0.3", "--tau-one-to-one", "0.2", "--tau-one-to-many", "0.4")),
    /silent-recall trap/,
  );
});

test("parseArgs: tau_many_to_one > threshold is caught", () => {
  assert.throws(
    () => parseArgs(argv("--threshold", "0.3", "--tau-one-to-one", "0.2", "--tau-many-to-one", "0.4")),
    /silent-recall trap/,
  );
});

test("parseArgs: k < 1 fails loud", () => {
  assert.throws(() => parseArgs(argv("--k", "0")), /positive integer/);
});

// ── mapMatchedToScorerFormat ────────────────────────────────────────────────

test("mapMatchedToScorerFormat: strips IRIs to raw ids", () => {
  const input = [
    { set_id: "terminusdb:///bench/abt_buy_e2e/Abt/12345", target_id: "terminusdb:///bench/abt_buy_e2e/Buy/67890", distance: 0.12, stage: "core" },
    { set_id: "terminusdb:///bench/abt_buy_e2e/Abt/11111", target_id: "terminusdb:///bench/abt_buy_e2e/Buy/22222", distance: 0.3, stage: "set_extra" },
  ];
  const result = mapMatchedToScorerFormat(input);
  assert.equal(result.length, 2);
  assert.equal(result[0].setId, "12345");
  assert.equal(result[0].targetId, "67890");
  assert.equal(result[0].distance, 0.12);
  assert.equal(result[0].stage, "core");
  assert.equal(result[1].setId, "11111");
  assert.equal(result[1].targetId, "22222");
  assert.equal(result[1].stage, "set_extra");
});

test("mapMatchedToScorerFormat: empty array produces empty array", () => {
  assert.deepEqual(mapMatchedToScorerFormat([]), []);
});

// ── shortDomain ────────────────────────────────────────────────────────────

test("shortDomain: strips /local/branch/main suffix", () => {
  assert.equal(shortDomain("admin/abt_buy_e2e/local/branch/main"), "admin/abt_buy_e2e");
});

test("shortDomain: returns short domain unchanged", () => {
  assert.equal(shortDomain("admin/abt_buy_e2e"), "admin/abt_buy_e2e");
});

test("shortDomain: handles non-main branch path", () => {
  assert.equal(shortDomain("admin/db/local/branch/dev"), "admin/db");
});

// ── fullDomain ─────────────────────────────────────────────────────────────

test("fullDomain: appends /local/branch/main to short domain", () => {
  assert.equal(fullDomain("admin/abt_buy_e2e"), "admin/abt_buy_e2e/local/branch/main");
});

test("fullDomain: returns full domain unchanged", () => {
  assert.equal(fullDomain("admin/abt_buy_e2e/local/branch/main"), "admin/abt_buy_e2e/local/branch/main");
});
