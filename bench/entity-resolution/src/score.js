"use strict";

// Pure scoring. Given, per query record, the ranked corpus ids the engine
// returned (nearest first) and the ground-truth corpus id set, compute
// precision@1 and recall@K.
//
// Definitions (corpus = Buy, query = Abt):
//   precision@1 : fraction of scored queries whose TOP-1 corpus hit is in the
//                 ground-truth set for that query.
//   recall@K    : fraction of scored queries for which AT LEAST ONE ground-truth
//                 corpus id appears within the top-K hits.
// A query maps to a set of valid corpus ids (the mapping is many-to-many), so a
// hit counts if the returned id is ANY of the ground-truth ids.
//
// Only queries that (a) have a ground-truth mapping are scored — a query with no
// mapping has no correct answer to measure against.

function isHitAtRank(rankedCorpusIds, truthSet, k) {
  const topK = rankedCorpusIds.slice(0, k);
  return topK.some((id) => truthSet.has(id));
}

// results: Array<{ queryId, rankedCorpusIds: string[], truthSet: Set<string> }>
function scoreResults(results, ks = [1, 5, 10]) {
  const scored = results.filter((r) => r.truthSet && r.truthSet.size > 0);
  const total = scored.length;

  const precisionAt1Hits = scored.reduce(
    (acc, r) => acc + (isHitAtRank(r.rankedCorpusIds, r.truthSet, 1) ? 1 : 0),
    0
  );

  const recallAtK = ks.reduce((acc, k) => {
    const hits = scored.reduce(
      (sum, r) => sum + (isHitAtRank(r.rankedCorpusIds, r.truthSet, k) ? 1 : 0),
      0
    );
    acc[k] = { hits, total, fraction: total === 0 ? 0 : hits / total };
    return acc;
  }, {});

  const misses = scored
    .filter((r) => !isHitAtRank(r.rankedCorpusIds, r.truthSet, 1))
    .map((r) => ({
      queryId: r.queryId,
      truth: [...r.truthSet],
      top: r.rankedCorpusIds.slice(0, 3),
    }));

  return {
    total,
    precisionAt1: { hits: precisionAt1Hits, total, fraction: total === 0 ? 0 : precisionAt1Hits / total },
    recallAtK,
    misses,
  };
}

module.exports = { scoreResults, isHitAtRank };
