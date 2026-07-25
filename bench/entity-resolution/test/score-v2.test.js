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
    { setId: "a1", targetId: "b1", distance: 0.2, stage: "set_extra" },
    { setId: "a1", targetId: "b1", distance: 0.1, stage: "core" }, // same pair
  ];
  const u = uniquePairs(groups);
  assert.equal(u.length, 1);
  assert.equal(u[0].distance, 0.1); // kept min distance
  assert.equal(u[0].stage, "core"); // preferred higher-confidence origin
});

test("uniquePairs deduplicates across all 3 stages (core > set_extra > target_extra)", () => {
  const groups = [
    { setId: "a1", targetId: "b1", distance: 0.15, stage: "target_extra" },
    { setId: "a1", targetId: "b1", distance: 0.12, stage: "set_extra" },
    { setId: "a1", targetId: "b1", distance: 0.10, stage: "core" },
  ];
  const u = uniquePairs(groups);
  assert.equal(u.length, 1);
  assert.equal(u[0].stage, "core");
  assert.equal(u[0].distance, 0.10);
});

test("precision = TP/unique-predicted, recall = TP/true-pairs (consistent pair denominators)", () => {
  const groups = [
    { setId: "a1", targetId: "b1", distance: 0.1, stage: "core" }, // correct
    { setId: "a2", targetId: "b9", distance: 0.4, stage: "set_extra" }, // wrong
  ];
  const s = scoreV2(groups, truth);
  assert.equal(s.counts.predictedPairsUnique, 2);
  assert.equal(s.counts.truePairs, 3);
  assert.equal(s.counts.truePositives, 1);
  assert.equal(s.overall.fraction, 0.5); // 1 correct / 2 predicted
  assert.equal(s.recall.fraction, 1 / 3); // 1 TP / 3 true pairs
  assert.equal(s.core.fraction, 1); // 1/1
  assert.equal(s.setExtra.fraction, 0); // 0/1
  // Legacy compat aliases still work.
  assert.equal(s.grounded.fraction, 1);
  assert.equal(s.assigned.fraction, 0);
});

test("double-counted input does not inflate the totals (the bug being fixed)", () => {
  const groups = [
    { setId: "a1", targetId: "b1", distance: 0.1, stage: "core" },
    { setId: "a1", targetId: "b1", distance: 0.1, stage: "core" }, // dupe
    { setId: "a2", targetId: "b2", distance: 0.1, stage: "core" },
  ];
  const s = scoreV2(groups, truth);
  assert.equal(s.counts.predictedPairsRaw, 3);
  assert.equal(s.counts.predictedPairsUnique, 2); // dupe collapsed
  assert.equal(s.overall.fraction, 1); // 2/2, not 2/3
  assert.equal(s.recall.fraction, 2 / 3); // 2 TP / 3 true pairs
});

test("a predicted pair whose set record has no truth COUNTS as a false positive in precision", () => {
  // aX is not in the mapping (it matches nothing in the truth). Predicting a pair
  // for it is a genuine false positive.
  const groups = [{ setId: "aX", targetId: "bX", distance: 0.2, stage: "set_extra" }];
  const s = scoreV2(groups, truth);
  assert.equal(s.overall.total, 1, "the unmapped-set prediction is in the precision denominator");
  assert.equal(s.overall.correct, 0, "it is wrong (aX has no truth pair)");
  assert.equal(s.overall.fraction, 0, "precision is 0/1, not 0/0");
  assert.equal(s.recall.hits, 0);
});

test("mapped-only precision view excludes unmapped-set predictions", () => {
  const groups = [
    { setId: "a1", targetId: "b1", distance: 0.1, stage: "core" }, // correct, mapped
    { setId: "aX", targetId: "bX", distance: 0.2, stage: "set_extra" }, // FP, unmapped
  ];
  const s = scoreV2(groups, truth);
  assert.equal(s.overall.fraction, 0.5, "headline precision: 1 correct / 2 predicted (FP counts)");
  assert.equal(s.mappedOnly.fraction, 1, "mapped-only view: 1/1 (unmapped FP excluded)");
  assert.equal(s.mappedOnly.total, 1);
});

test("perfect prediction -> precision 1, recall 1, F1 1", () => {
  const groups = [
    { setId: "a1", targetId: "b1", distance: 0.1, stage: "core" },
    { setId: "a2", targetId: "b2", distance: 0.1, stage: "core" },
    { setId: "a3", targetId: "b3", distance: 0.1, stage: "set_extra" },
  ];
  const s = scoreV2(groups, truth);
  assert.equal(s.overall.fraction, 1);
  assert.equal(s.recall.fraction, 1);
  assert.equal(s.f1, 1);
});

test("many-to-many: multiple pairs per set record scored correctly", () => {
  // a1 predicted with b1 (correct) and bX (wrong). Both count in precision.
  const groups = [
    { setId: "a1", targetId: "b1", distance: 0.10, stage: "core" }, // correct
    { setId: "a1", targetId: "bX", distance: 0.30, stage: "set_extra" }, // wrong
  ];
  const s = scoreV2(groups, truth);
  assert.equal(s.counts.predictedPairsUnique, 2);
  assert.equal(s.overall.fraction, 0.5); // 1 correct of 2 predicted
  assert.equal(s.counts.truePositives, 1);
  assert.equal(s.recall.fraction, 1 / 3); // 1 of 3 gold pairs found
});

test("many-to-many truth: multiple gold pairs per set record, all countable", () => {
  // Truth says a1 matches both b1 AND b2.
  const multiTruth = new Map([
    ["a1", new Set(["b1", "b2"])],
    ["a2", new Set(["b3"])],
  ]);
  // Predict all 3 truth pairs.
  const groups = [
    { setId: "a1", targetId: "b1", distance: 0.1, stage: "core" },
    { setId: "a1", targetId: "b2", distance: 0.2, stage: "set_extra" },
    { setId: "a2", targetId: "b3", distance: 0.15, stage: "target_extra" },
  ];
  const s = scoreV2(groups, multiTruth);
  assert.equal(s.counts.truePairs, 3);
  assert.equal(s.counts.truePositives, 3);
  assert.equal(s.overall.fraction, 1);
  assert.equal(s.recall.fraction, 1);
  assert.equal(s.f1, 1);
});

test("committed prediction: TP/FP/FN counts are correct and F1 is meaningful", () => {
  // a1->b1 (correct), a2->b9 (wrong -- truth is a2->b2). a3 unmatched (FN).
  const groups = [
    { setId: "a1", targetId: "b1", distance: 0.1, stage: "core" },
    { setId: "a2", targetId: "b9", distance: 0.3, stage: "set_extra" },
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

test("per-stage counts: corePairs, setExtraPairs, targetExtraPairs are correct", () => {
  const groups = [
    { setId: "a1", targetId: "b1", distance: 0.1, stage: "core" },
    { setId: "a1", targetId: "b2", distance: 0.2, stage: "set_extra" },
    { setId: "a2", targetId: "b2", distance: 0.15, stage: "set_extra" },
    { setId: "a3", targetId: "b3", distance: 0.1, stage: "target_extra" },
  ];
  const s = scoreV2(groups, truth);
  assert.equal(s.counts.corePairs, 1);
  assert.equal(s.counts.setExtraPairs, 2);
  assert.equal(s.counts.targetExtraPairs, 1);
});
