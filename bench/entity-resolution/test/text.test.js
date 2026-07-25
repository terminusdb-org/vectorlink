"use strict";

const { test } = require("node:test");
const assert = require("node:assert/strict");
const { sentenceCase, normaliseToken } = require("../src/text");

test("de-allcaps a brand word: LINKSYS -> Linksys", () => {
  assert.equal(sentenceCase("LINKSYS"), "Linksys");
});

test("leaves an already mixed-case brand untouched", () => {
  assert.equal(sentenceCase("Linksys"), "Linksys");
  assert.equal(sentenceCase("EtherFast"), "EtherFast");
});

test("PRESERVES alphanumeric model numbers verbatim (identity-bearing)", () => {
  // EZXS88W and PSLX350H are model codes — their casing must survive.
  assert.equal(normaliseToken("EZXS88W"), "EZXS88W");
  assert.equal(normaliseToken("PSLX350H"), "PSLX350H");
  assert.equal(sentenceCase("LINKSYS EZXS88W"), "Linksys EZXS88W");
});

test("normalises only the all-caps brand within a mixed phrase", () => {
  assert.equal(sentenceCase("SONY Turntable PSLX350H"), "Sony Turntable PSLX350H");
});

test("collapses internal whitespace to single spaces and trims", () => {
  assert.equal(sentenceCase("  LINKSYS   switch  "), "Linksys switch");
});

test("empty / null / undefined -> empty string", () => {
  assert.equal(sentenceCase(""), "");
  assert.equal(sentenceCase(null), "");
  assert.equal(sentenceCase(undefined), "");
});

test("single-letter all-caps token is de-capped safely", () => {
  assert.equal(normaliseToken("A"), "A");
});

test("lowercase token is left as-is", () => {
  assert.equal(normaliseToken("ethernet"), "ethernet");
});
