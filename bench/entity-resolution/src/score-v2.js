"use strict";

// Pure v2 scoring (spec section 8) -- PAIR-BASED entity-resolution evaluation
// against the perfect mapping.
//
// The standard ER metric: compare the SET of predicted (set, target) pairs to the
// SET of ground-truth pairs.
//
//   truth     = perfectMapping as Map<setId, Set<targetId>>; the total number of
//               TRUE PAIRS is the sum of |truth(set)| (e.g. 1097 for Abt-Buy).
//   predicted = the MATCHED partition from the resolver -- the denormalised set of
//               predicted pairs.
//
//   TP        = |P intersect G|  (predicted pairs that are in the gold set)
//   FP        = |P \ G|          (predicted pairs not in gold)
//   FN        = |G \ P|          (gold pairs not predicted)
//   precision = TP / (TP + FP)   = TP / |P|
//   recall    = TP / (TP + FN)   = TP / |G|
//   F1        = 2*P*R / (P + R)

function pairKey(setId, targetId) {
  return `${setId}::${targetId}`;
}

// Total ground-truth pairs (sum of set sizes) -- the recall denominator |G|.
function totalTruePairs(truth) {
  let total = 0;
  for (const set of truth.values()) total += set.size;
  return total;
}

// Deduplicate predicted pairs by (setId, targetId) key. The resolver already
// deduplicates, but this is a safety net. Pure.
function uniquePairs(groups) {
  const stageRank = Object.freeze({ core: 0, set_extra: 1, target_extra: 2 });
  const byKey = new Map();
  for (const g of groups) {
    const key = pairKey(g.setId, g.targetId);
    const current = byKey.get(key);
    if (current === undefined) {
      byKey.set(key, { ...g });
    } else {
      const existingRank = stageRank[current.stage] ?? 99;
      const newRank = stageRank[g.stage] ?? 99;
      if (newRank < existingRank || (newRank === existingRank && g.distance < current.distance)) {
        byKey.set(key, { ...g });
      }
    }
  }
  return [...byKey.values()];
}

function isCorrect(pair, truth) {
  const set = truth.get(pair.setId);
  return set !== undefined && set.has(pair.targetId);
}

// Mapped-only precision: only pairs whose set record HAS a truth mapping.
function precisionOf(pairs, truth) {
  const scoreable = pairs.filter((p) => truth.has(p.setId));
  const correct = scoreable.reduce((acc, p) => acc + (isCorrect(p, truth) ? 1 : 0), 0);
  const total = scoreable.length;
  return { correct, total, fraction: total === 0 ? 0 : correct / total };
}

// Headline precision: EVERY predicted pair in the denominator.
function precisionAll(pairs, truth) {
  const correct = pairs.reduce((acc, p) => acc + (isCorrect(p, truth) ? 1 : 0), 0);
  const total = pairs.length;
  return { correct, total, fraction: total === 0 ? 0 : correct / total };
}

function f1(precisionFraction, recallFraction) {
  const denom = precisionFraction + recallFraction;
  return denom === 0 ? 0 : (2 * precisionFraction * recallFraction) / denom;
}

// matched: Array<{ setId, targetId, distance, stage }>
function scoreV2(matched, truth) {
  const predicted = uniquePairs(matched);
  const core = predicted.filter((p) => p.stage === "core");
  const setExtra = predicted.filter((p) => p.stage === "set_extra");
  const targetExtra = predicted.filter((p) => p.stage === "target_extra");

  const truePairCount = totalTruePairs(truth);

  const overall = precisionAll(predicted, truth);
  const mappedOnly = precisionOf(predicted, truth);

  const truePositives = overall.correct;
  const recall = {
    hits: truePositives,
    total: truePairCount,
    fraction: truePairCount === 0 ? 0 : truePositives / truePairCount,
  };

  const headlineF1 = f1(overall.fraction, recall.fraction);

  // Per-stage precision.
  const coreScore = precisionOf(core, truth);
  const setExtraScore = precisionOf(setExtra, truth);
  const targetExtraScore = precisionOf(targetExtra, truth);

  const falsePositives = predicted.length - truePositives;
  const falseNegatives = truePairCount - truePositives;

  const wrongExamples = predicted
    .filter((p) => truth.has(p.setId) && !isCorrect(p, truth))
    .slice(0, 8)
    .map((p) => ({ setId: p.setId, predictedTarget: p.targetId, stage: p.stage, truth: [...(truth.get(p.setId) || [])] }));

  return {
    overall,
    mappedOnly,
    recall,
    f1: headlineF1,
    core: coreScore,
    setExtra: setExtraScore,
    targetExtra: targetExtraScore,
    // Legacy compat aliases.
    grounded: coreScore,
    assigned: precisionOf([...setExtra, ...targetExtra], truth),
    counts: {
      predictedPairsRaw: matched.length,
      predictedPairsUnique: predicted.length,
      corePairs: core.length,
      setExtraPairs: setExtra.length,
      targetExtraPairs: targetExtra.length,
      truePairs: truePairCount,
      mappedSet: [...truth.keys()].length,
      truePositives,
      falsePositives,
      falseNegatives,
    },
    wrongExamples,
  };
}

module.exports = { scoreV2, precisionOf, precisionAll, isCorrect, uniquePairs, totalTruePairs, f1 };
