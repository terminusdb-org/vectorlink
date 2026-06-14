"use strict";

// Pure v2 scoring (spec §8). The v2 algorithm emits resolved PAIRS (Abt↔Buy),
// each tagged with the stage that decided it (grounded in Step 4 vs assigned in
// Step 5). We score those pairs against the perfect mapping and split accuracy by
// stage, so each refinement's contribution is measurable.
//
// Definitions (truth = abt_buy_perfectMapping.csv, as Map<abtId, Set<buyId>>):
//   - A predicted pair (abtId, buyId) is CORRECT iff buyId ∈ truth(abtId).
//   - precision = correct predicted pairs / all predicted pairs.
//   - recall    = Abt ids whose truth was hit by ANY predicted pair / Abt ids
//                 that HAVE a truth mapping. (Recall is over mapped Abt ids: an
//                 Abt with no truth cannot contribute to recall.)
//   - per-stage precision = correct pairs from that stage / pairs from that stage.

function isCorrect(pair, truth) {
  const set = truth.get(pair.abtId);
  return set !== undefined && set.has(pair.buyId);
}

function precisionOf(pairs, truth) {
  // Only Abt ids that HAVE a truth mapping are scoreable for precision — a pair
  // whose Abt has no ground truth is neither right nor wrong, so it is excluded
  // (consistent with v1, which scored only mapped queries).
  const scoreable = pairs.filter((p) => truth.has(p.abtId));
  const correct = scoreable.reduce((acc, p) => acc + (isCorrect(p, truth) ? 1 : 0), 0);
  const total = scoreable.length;
  return { correct, total, fraction: total === 0 ? 0 : correct / total };
}

// results.groups: Array<{ abtId, buyId, distance, stage }>
function scoreV2(groups, truth) {
  const grounded = groups.filter((g) => g.stage === "grounded");
  const assigned = groups.filter((g) => g.stage === "assigned");

  const overall = precisionOf(groups, truth);
  const groundedScore = precisionOf(grounded, truth);
  const assignedScore = precisionOf(assigned, truth);

  // Recall over mapped Abt ids: which mapped Abt had at least one CORRECT pair.
  const mappedAbtIds = [...truth.keys()];
  const correctAbt = new Set(
    groups.filter((g) => isCorrect(g, truth)).map((g) => g.abtId)
  );
  const recallHits = mappedAbtIds.reduce((acc, id) => acc + (correctAbt.has(id) ? 1 : 0), 0);
  const recall = {
    hits: recallHits,
    total: mappedAbtIds.length,
    fraction: mappedAbtIds.length === 0 ? 0 : recallHits / mappedAbtIds.length,
  };

  const wrongExamples = groups
    .filter((g) => truth.has(g.abtId) && !isCorrect(g, truth))
    .slice(0, 8)
    .map((g) => ({ abtId: g.abtId, predictedBuy: g.buyId, stage: g.stage, truth: [...(truth.get(g.abtId) || [])] }));

  return {
    overall,
    grounded: groundedScore,
    assigned: assignedScore,
    recall,
    counts: {
      predictedPairs: groups.length,
      groundedPairs: grounded.length,
      assignedPairs: assigned.length,
      mappedAbt: mappedAbtIds.length,
    },
    wrongExamples,
  };
}

module.exports = { scoreV2, precisionOf, isCorrect };
