"use strict";

// Pure v2 scoring (spec §8) — PAIR-BASED entity-resolution evaluation against the
// perfect mapping. See docs/abt-buy-evaluation-methodology.md for the full
// methodology.
//
// The standard ER metric: compare the SET of predicted (Abt, Buy) pairs to the
// SET of ground-truth pairs.
//
//   truth     = abt_buy_perfectMapping.csv as Map<abtId, Set<buyId>>; the total
//               number of TRUE PAIRS is Σ |truth(abt)| (the 1097 perfect pairs).
//   predicted = the COMMITTED prediction set P — one pair per source Abt at most.
//               The resolver guarantees this contract (see resolve.js invariant).
//
//   TP        = |P ∩ G|        (predicted pairs that are in the gold set)
//   FP        = |P \ G|        (predicted pairs not in gold)
//   FN        = |G \ P|        (gold pairs not predicted)
//   precision = TP / (TP + FP) = TP / |P|
//   recall    = TP / (TP + FN) = TP / |G|
//   F1        = 2·P·R / (P + R)
//
// Every denominator is a PAIR count on the SAME pair universe — precision and
// recall are consistent, and the F1 is meaningful and comparable across modes.
//
// False positives on UNMAPPED Abt: a predicted pair whose Abt has NO ground-truth
// mapping is a genuine FALSE POSITIVE — the dataset asserts that Abt as a real
// record, and the perfect mapping says it matches nothing, so predicting a pair
// for it is wrong. The HEADLINE precision counts EVERY predicted pair in its
// denominator and such a pair as incorrect (it lowers precision; it is not
// invisible). This is the sound pair-based metric.
//
// We ALSO expose a `mappedOnly` view that excludes unmapped-Abt predictions —
// the v1-comparable universe (v1 scored only mapped queries) — for apples-to-
// apples comparison with the 83% baseline.

function pairKey(abtId, buyId) {
  return `${abtId}::${buyId}`;
}

// Total ground-truth pairs (Σ set sizes) — the recall denominator |G|.
function totalTruePairs(truth) {
  let total = 0;
  for (const set of truth.values()) total += set.size;
  return total;
}

// Deduplicate predicted pairs by (abtId, buyId) key. With the corrected resolver
// (one-per-Abt), this is a safety net — it should produce the same array back.
// Retained for defensive correctness: if the same pair were somehow emitted twice
// (e.g. through a future code path), it must not be double-counted. Pure.
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

// Headline precision: EVERY predicted pair is in the denominator; a pair on an
// unmapped Abt is a false positive (wrong). This is P = TP / |P|.
function precisionAll(pairs, truth) {
  const correct = pairs.reduce((acc, p) => acc + (isCorrect(p, truth) ? 1 : 0), 0);
  const total = pairs.length;
  return { correct, total, fraction: total === 0 ? 0 : correct / total };
}

// Collapse each Abt's predicted pairs to its SINGLE nearest Buy. With the
// corrected resolver this is a no-op (resolver already guarantees one-per-Abt).
// Retained as a safety net and for backward-compatible access to the view. Pure.
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
// The resolver guarantees at most one pair per Abt (the committed prediction set P).
function scoreV2(groups, truth) {
  // 1. Deduplicate (safety net — resolver already guarantees one-per-Abt).
  const predicted = uniquePairs(groups);
  const grounded = predicted.filter((p) => p.stage === "grounded");
  const assigned = predicted.filter((p) => p.stage === "assigned");

  const truePairCount = totalTruePairs(truth);

  // 2. HEADLINE precision over the committed prediction set P.
  //    An unmapped-Abt pair is a false positive (counts against precision).
  const overall = precisionAll(predicted, truth);
  //    Mapped-only view (v1-comparable): excludes unmapped-Abt predictions.
  const mappedOnly = precisionOf(predicted, truth);

  // 3. Recall: TP / |G| (gold pairs found / total gold pairs).
  const truePositives = overall.correct;
  const recall = {
    hits: truePositives,
    total: truePairCount,
    fraction: truePairCount === 0 ? 0 : truePositives / truePairCount,
  };

  // 4. Headline F1 — the single comparable metric across all modes.
  //    Precision denominator = |P| (committed pairs, one-per-Abt).
  //    Recall denominator = |G| (1097 gold pairs).
  //    Both are pair-based with consistent universes.
  const headlineF1 = f1(overall.fraction, recall.fraction);

  // 5. Per-stage precision — measures each refinement's contribution.
  const groundedScore = precisionOf(grounded, truth);
  const assignedScore = precisionOf(assigned, truth);

  // 6. False-positive and false-negative counts for clarity.
  const falsePositives = predicted.length - truePositives;
  const falseNegatives = truePairCount - truePositives;

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
    // bestPerAbt retained for backward compat — now identical to overall since
    // the resolver guarantees one-per-Abt.
    bestPerAbt: precisionOf(bestPerAbt(predicted), truth),
    counts: {
      predictedPairsRaw: groups.length,
      predictedPairsUnique: predicted.length,
      groundedPairs: grounded.length,
      assignedPairs: assigned.length,
      truePairs: truePairCount,
      mappedAbt: [...truth.keys()].length,
      truePositives,
      falsePositives,
      falseNegatives,
    },
    wrongExamples,
  };
}

module.exports = { scoreV2, precisionOf, precisionAll, isCorrect, uniquePairs, totalTruePairs, bestPerAbt, f1 };
