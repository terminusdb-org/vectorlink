"use strict";

const { test } = require("node:test");
const assert = require("node:assert/strict");
const { scoreV2 } = require("../src/score-v2");

const truth = new Map([
  ["a1", new Set(["b1"])],
  ["a2", new Set(["b2"])],
  ["a3", new Set(["b3"])], // a3 has truth but is never predicted
]);

test("splits precision by stage and computes overall + recall", () => {
  const groups = [
    { abtId: "a1", buyId: "b1", distance: 0.1, stage: "grounded" }, // correct
    { abtId: "a2", buyId: "b9", distance: 0.4, stage: "assigned" }, // wrong
  ];
  const s = scoreV2(groups, truth);
  assert.equal(s.grounded.fraction, 1); // 1/1 correct
  assert.equal(s.assigned.fraction, 0); // 0/1 correct
  assert.equal(s.overall.correct, 1);
  assert.equal(s.overall.total, 2);
  // recall over the 3 mapped Abt ids: only a1 was hit correctly.
  assert.equal(s.recall.hits, 1);
  assert.equal(s.recall.total, 3);
});

test("a predicted pair whose Abt has no truth is excluded from precision", () => {
  const groups = [{ abtId: "aX", buyId: "bX", distance: 0.2, stage: "assigned" }];
  const s = scoreV2(groups, truth);
  assert.equal(s.overall.total, 0); // aX not scoreable
  assert.equal(s.assigned.total, 0);
});

test("perfect prediction -> precision 1 and recall over mapped ids", () => {
  const groups = [
    { abtId: "a1", buyId: "b1", distance: 0.1, stage: "grounded" },
    { abtId: "a2", buyId: "b2", distance: 0.1, stage: "grounded" },
    { abtId: "a3", buyId: "b3", distance: 0.1, stage: "assigned" },
  ];
  const s = scoreV2(groups, truth);
  assert.equal(s.overall.fraction, 1);
  assert.equal(s.recall.fraction, 1);
});
