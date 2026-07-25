"use strict";

const { test } = require("node:test");
const assert = require("node:assert/strict");
const { idFromIri, isSide } = require("../src/iri");

test("idFromIri strips an IRI to its last segment", () => {
  assert.equal(idFromIri("terminusdb:///bench/abt_buy/Buy/10011646"), "10011646");
  assert.equal(idFromIri("terminusdb:///bench/abt_buy_v2/Abt/552"), "552");
});

test("isSide detects the namespace segment", () => {
  const iri = "terminusdb:///bench/abt_buy_v2/Buy/123";
  assert.equal(isSide(iri, "Buy"), true);
  assert.equal(isSide(iri, "Abt"), false);
});

test("modes and verify share the SAME idFromIri (no divergence)", () => {
  // Regression guard: both modules must resolve ids identically.
  const { idFromIri: modesId } = require("../src/modes");
  const { corpusIdFromIri } = require("../src/verify");
  const iri = "terminusdb:///bench/abt_buy/Buy/999";
  assert.equal(modesId(iri), idFromIri(iri));
  assert.equal(corpusIdFromIri(iri), idFromIri(iri));
});
