"use strict";

const { test } = require("node:test");
const assert = require("node:assert/strict");
const { scoreV2, uniquePairs, totalTruePairs, precisionAll } = require("../src/score-v2");

void precisionAll; // exported for callers; exercised indirectly via scoreV2 below

const truth = new Map([
  ["a1", new Set(["b1"])],
  ["a2", new Set(["b2"])],
  ["a3", new Set(["b3"])], // a3 has truth but is never predicted
]);
// total true pairs = 3

test("totalTruePairs sums the truth set sizes (the perfect-pairs count)", () => {
  assert.equal(totalTruePairs(truth), 3);
  const many = new Map([["a1", new Set(["b1", "b2"])], ["a2", new Set(["b3"])]]);
  assert.equal(totalTruePairs(many), 3);
});

test("uniquePairs deduplicates a pair emitted twice (no double-counting)", () => {
  const groups = [
    { abtId: "a1", buyId: "b1", distance: 0.2, stage: "assigned" },
    { abtId: "a1", buyId: "b1", distance: 0.1, stage: "grounded" }, // same pair
  ];
  const u = uniquePairs(groups);
  assert.equal(u.length, 1);
  assert.equal(u[0].distance, 0.1); // kept min distance
  assert.equal(u[0].stage, "grounded"); // preferred higher-confidence origin
});

test("precision = TP/unique-predicted, recall = TP/true-pairs (consistent pair denominators)", () => {
  const groups = [
    { abtId: "a1", buyId: "b1", distance: 0.1, stage: "grounded" }, // correct
    { abtId: "a2", buyId: "b9", distance: 0.4, stage: "assigned" }, // wrong
  ];
  const s = scoreV2(groups, truth);
  assert.equal(s.counts.predictedPairsUnique, 2);
  assert.equal(s.counts.truePairs, 3);
  assert.equal(s.counts.truePositives, 1);
  assert.equal(s.overall.fraction, 0.5); // 1 correct / 2 predicted
  assert.equal(s.recall.fraction, 1 / 3); // 1 TP / 3 true pairs
  assert.equal(s.grounded.fraction, 1); // 1/1
  assert.equal(s.assigned.fraction, 0); // 0/1
});

test("double-counted input does not inflate the totals (the bug being fixed)", () => {
  const groups = [
    { abtId: "a1", buyId: "b1", distance: 0.1, stage: "grounded" },
    { abtId: "a1", buyId: "b1", distance: 0.1, stage: "grounded" }, // dupe
    { abtId: "a2", buyId: "b2", distance: 0.1, stage: "grounded" },
  ];
  const s = scoreV2(groups, truth);
  assert.equal(s.counts.predictedPairsRaw, 3);
  assert.equal(s.counts.predictedPairsUnique, 2); // dupe collapsed
  assert.equal(s.overall.fraction, 1); // 2/2, not 2/3
  assert.equal(s.recall.fraction, 2 / 3); // 2 TP / 3 true pairs
});

test("a predicted pair whose Abt has no truth COUNTS as a false positive in precision", () => {
  // aX is not in the mapping (it matches nothing in the truth). Predicting a pair
  // for it is a genuine false positive: the algorithm asserted a match where the
  // ground truth says there is none. It must lower precision, not be invisible.
  const groups = [{ abtId: "aX", buyId: "bX", distance: 0.2, stage: "assigned" }];
  const s = scoreV2(groups, truth);
  assert.equal(s.overall.total, 1, "the unmapped-Abt prediction is in the precision denominator");
  assert.equal(s.overall.correct, 0, "it is wrong (aX has no truth pair)");
  assert.equal(s.overall.fraction, 0, "precision is 0/1, not 0/0");
  assert.equal(s.recall.hits, 0);
});

test("mapped-only precision view (v1-comparable) still excludes unmapped-Abt predictions", () => {
  // The headline precision counts unmapped-Abt predictions as FPs (above), but we
  // ALSO expose a mapped-only view for apples-to-apples comparison with v1, which
  // scored only mapped queries.
  const groups = [
    { abtId: "a1", buyId: "b1", distance: 0.1, stage: "grounded" }, // correct, mapped
    { abtId: "aX", buyId: "bX", distance: 0.2, stage: "assigned" }, // FP, unmapped
  ];
  const s = scoreV2(groups, truth);
  assert.equal(s.overall.fraction, 0.5, "headline precision: 1 correct / 2 predicted (FP counts)");
  assert.equal(s.mappedOnly.fraction, 1, "mapped-only view: 1/1 (unmapped FP excluded)");
  assert.equal(s.mappedOnly.total, 1);
});

test("perfect prediction -> precision 1, recall 1, F1 1", () => {
  const groups = [
    { abtId: "a1", buyId: "b1", distance: 0.1, stage: "grounded" },
    { abtId: "a2", buyId: "b2", distance: 0.1, stage: "grounded" },
    { abtId: "a3", buyId: "b3", distance: 0.1, stage: "assigned" },
  ];
  const s = scoreV2(groups, truth);
  assert.equal(s.overall.fraction, 1);
  assert.equal(s.recall.fraction, 1);
  assert.equal(s.f1, 1);
});

test("over-grounding many Buy per Abt: precision drops, bestPerAbt recovers the p@1 view", () => {
  // a1 grounded to its truth b1 AND two wrong Buy — the over-grounding case.
  // NOTE: the corrected resolver no longer produces this output (it emits one per
  // Abt). This test validates the scorer handles such input gracefully as a safety
  // net — the bestPerAbt view still recovers the correct precision@1.
  const groups = [
    { abtId: "a1", buyId: "b1", distance: 0.10, stage: "grounded" }, // correct, nearest
    { abtId: "a1", buyId: "bX", distance: 0.30, stage: "grounded" }, // wrong
    { abtId: "a1", buyId: "bY", distance: 0.40, stage: "grounded" }, // wrong
  ];
  const s = scoreV2(groups, truth);
  assert.equal(s.overall.fraction, 1 / 3); // 1 correct of 3 predicted pairs
  // bestPerAbt collapses a1 to its nearest (b1, correct) -> 1/1.
  assert.equal(s.bestPerAbt.fraction, 1);
});

test("committed one-per-Abt output: TP/FP/FN counts are correct and F1 is meaningful", () => {
  // Simulate the corrected resolver output: exactly one pair per Abt.
  // a1->b1 (correct), a2->b9 (wrong — truth is a2->b2). a3 unmatched (FN).
  const groups = [
    { abtId: "a1", buyId: "b1", distance: 0.1, stage: "grounded" },
    { abtId: "a2", buyId: "b9", distance: 0.3, stage: "assigned" },
  ];
  const s = scoreV2(groups, truth);
  // |P|=2, |G|=3, TP=1, FP=1, FN=2
  assert.equal(s.counts.predictedPairsUnique, 2);
  assert.equal(s.counts.truePositives, 1);
  assert.equal(s.counts.falsePositives, 1);
  assert.equal(s.counts.falseNegatives, 2);
  // precision = 1/2, recall = 1/3
  assert.equal(s.overall.fraction, 0.5);
  assert.equal(s.recall.fraction, 1 / 3);
  // F1 = 2*(0.5*0.333)/(0.5+0.333) = 0.4
  const expectedF1 = 2 * 0.5 * (1 / 3) / (0.5 + 1 / 3);
  assert.ok(Math.abs(s.f1 - expectedF1) < 1e-10, "headline F1 is the standard harmonic mean");
});
