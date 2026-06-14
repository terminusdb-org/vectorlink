"use strict";

// Pure v2 scoring (spec §8) — PAIR-BASED entity-resolution evaluation against the
// perfect mapping. This is the standard ER metric and the one the perfect-pairs
// file defines: compare the SET of predicted (Abt, Buy) pairs to the SET of
// ground-truth pairs.
//
//   truth     = abt_buy_perfectMapping.csv as Map<abtId, Set<buyId>>; the total
//               number of TRUE PAIRS is Σ |truth(abt)| (the 1097 perfect pairs,
//               many-to-one: most Abt map to one Buy, a few to several).
//   predicted = the UNIQUE (abtId, buyId) pairs the algorithm output. A pair
//               emitted twice (e.g. by both grounding and assignment, or as two
//               mutual edges) is counted ONCE — no double-counting.
//
//   TP        = |predicted ∩ truth-pairs|
//   precision = TP / |predicted|                 (of what we predicted, how much is right)
//   recall    = TP / |truth-pairs|               (of the true pairs, how many we found)
//   F1        = 2·P·R / (P + R)
//
// Every denominator is a PAIR count on the SAME pair universe — precision and
// recall are consistent (no mixing pair-precision with id-recall), and a pair is
// never counted more than once.
//
// False positives on UNMAPPED Abt: a predicted pair whose Abt has NO ground-truth
// mapping is a genuine FALSE POSITIVE — the dataset asserts that Abt as a real
// record, and the perfect mapping says it matches nothing, so predicting a pair
// for it is wrong. The HEADLINE precision (`overall`) therefore counts EVERY
// unique predicted pair in its denominator and such a pair as incorrect (it
// lowers precision; it is not invisible). This is the sound pair-based metric.
//
// We ALSO expose a `mappedOnly` view that excludes unmapped-Abt predictions —
// the v1-comparable universe (v1 scored only mapped queries) — for apples-to-
// apples comparison with the 83% baseline. Use `overall` as the truth; use
// `mappedOnly` only when comparing against v1.

function pairKey(abtId, buyId) {
  return `${abtId}::${buyId}`;
}

// Total ground-truth pairs (Σ set sizes) — the recall denominator.
function totalTruePairs(truth) {
  let total = 0;
  for (const set of truth.values()) total += set.size;
  return total;
}

// Unique predicted pairs (deduplicated). Each retains its nearest distance and a
// stage; if the same pair came from two stages we keep "grounded" (the
// higher-confidence origin) and the lower distance. Pure.
function uniquePairs(groups) {
  const byKey = new Map();
  for (const g of groups) {
    const key = pairKey(g.abtId, g.buyId);
    const current = byKey.get(key);
    if (current === undefined) {
      byKey.set(key, { ...g });
    } else {
      // Dedup: keep min distance; prefer a "grounded" origin over "assigned".
      const stage = current.stage === "grounded" || g.stage === "grounded" ? "grounded" : current.stage;
      byKey.set(key, { ...current, stage, distance: Math.min(current.distance, g.distance) });
    }
  }
  return [...byKey.values()];
}

function isCorrect(pair, truth) {
  const set = truth.get(pair.abtId);
  return set !== undefined && set.has(pair.buyId);
}

// Mapped-only precision (the v1-comparable universe): only pairs whose Abt HAS a
// truth mapping are counted. A pair on an unmapped Abt is excluded entirely.
function precisionOf(pairs, truth) {
  const scoreable = pairs.filter((p) => truth.has(p.abtId));
  const correct = scoreable.reduce((acc, p) => acc + (isCorrect(p, truth) ? 1 : 0), 0);
  const total = scoreable.length;
  return { correct, total, fraction: total === 0 ? 0 : correct / total };
}

// Headline precision (the sound pair-based metric): EVERY unique predicted pair
// is in the denominator; a pair on an unmapped Abt is a false positive (wrong).
function precisionAll(pairs, truth) {
  const correct = pairs.reduce((acc, p) => acc + (isCorrect(p, truth) ? 1 : 0), 0);
  const total = pairs.length;
  return { correct, total, fraction: total === 0 ? 0 : correct / total };
}

// Collapse each Abt's predicted pairs to its SINGLE nearest Buy — the
// one-Buy-per-Abt view directly comparable to v1's precision@1 (one pick per
// Abt). Pure.
function bestPerAbt(pairs) {
  const best = new Map();
  for (const p of pairs) {
    const current = best.get(p.abtId);
    if (current === undefined || p.distance < current.distance) best.set(p.abtId, p);
  }
  return [...best.values()];
}

function f1(precisionFraction, recallFraction) {
  const denom = precisionFraction + recallFraction;
  return denom === 0 ? 0 : (2 * precisionFraction * recallFraction) / denom;
}

// results.groups: Array<{ abtId, buyId, distance, stage }>
function scoreV2(groups, truth) {
  // 1. Deduplicate the predicted pairs FIRST — this is the fix for the
  //    double-counting: the totals below are over UNIQUE pairs.
  const predicted = uniquePairs(groups);
  const grounded = predicted.filter((p) => p.stage === "grounded");
  const assigned = predicted.filter((p) => p.stage === "assigned");

  const truePairCount = totalTruePairs(truth);

  // 2. HEADLINE precision over ALL unique predicted pairs — an unmapped-Abt pair
  //    is a false positive (counts against precision, not excluded).
  const overall = precisionAll(predicted, truth);
  //    Mapped-only view (v1-comparable): excludes unmapped-Abt predictions.
  const mappedOnly = precisionOf(predicted, truth);

  // 3. Recall against the PERFECT-PAIRS denominator: TP / total true pairs.
  //    TP = unique predicted pairs that are in the truth set (a correct pair's
  //    Abt is necessarily mapped, so overall.correct === mappedOnly.correct).
  const truePositives = overall.correct;
  const recall = {
    hits: truePositives,
    total: truePairCount,
    fraction: truePairCount === 0 ? 0 : truePositives / truePairCount,
  };

  const headlineF1 = f1(overall.fraction, recall.fraction);

  // 4. Per-stage precision (over unique pairs) — measures each refinement.
  const groundedScore = precisionOf(grounded, truth);
  const assignedScore = precisionOf(assigned, truth);

  // 5. One-Buy-per-Abt view (comparable to v1 precision@1).
  const best = bestPerAbt(predicted);
  const bestScore = precisionOf(best, truth);
  const mappedAbtIds = [...truth.keys()];
  const bestRecall = {
    // Of mapped Abt, how many had their single best pick land on a truth pair.
    hits: bestScore.correct,
    total: mappedAbtIds.length,
    fraction: mappedAbtIds.length === 0 ? 0 : bestScore.correct / mappedAbtIds.length,
  };

  const wrongExamples = predicted
    .filter((p) => truth.has(p.abtId) && !isCorrect(p, truth))
    .slice(0, 8)
    .map((p) => ({ abtId: p.abtId, predictedBuy: p.buyId, stage: p.stage, truth: [...(truth.get(p.abtId) || [])] }));

  return {
    overall,
    mappedOnly,
    recall,
    f1: headlineF1,
    grounded: groundedScore,
    assigned: assignedScore,
    bestPerAbt: bestScore,
    bestRecall,
    counts: {
      predictedPairsRaw: groups.length,
      predictedPairsUnique: predicted.length,
      groundedPairs: grounded.length,
      assignedPairs: assigned.length,
      truePairs: truePairCount,
      mappedAbt: mappedAbtIds.length,
      truePositives,
    },
    wrongExamples,
  };
}

module.exports = { scoreV2, precisionOf, precisionAll, isCorrect, uniquePairs, totalTruePairs, bestPerAbt, f1 };
