"use strict";

// Verifier: for each query (Abt) record, search the corpus (Buy) and compare the
// top-K to the ground-truth mapping. Prints precision@1 + recall@{1,5,10}.
//
// The MATCHING STEP is isolated in `ioMatchPerQuery` so it can later be swapped
// for the bulk `/duplicates` path without touching loading or scoring.

const { getDataset } = require("./datasets");
const { ioWaitReady, ioSearch } = require("./engine");
const { ioLoadSide, ioLoadMapping } = require("./load-records");
const { scoreResults } = require("./score");

const TOP_K = 10; // request enough to score recall@10
const KS = [1, 5, 10];
// Optional cap on the number of query records (for a partial sweep when the
// engine has limited headroom). 0 = no cap (full sweep). Set via BENCH_LIMIT.
const QUERY_LIMIT = Number(process.env.BENCH_LIMIT || 0);
// Small inter-query pause: cooperative pacing on a SHARED engine (:8081 used by
// sibling agents) and to keep the Lance store's file-descriptor use bounded
// under a long sequential sweep. Tunable via BENCH_QUERY_DELAY_MS.
const QUERY_DELAY_MS = Number(process.env.BENCH_QUERY_DELAY_MS || 25);

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function log(msg) {
  process.stdout.write(`[verify] ${msg}\n`);
}

// Strip the corpus IRI back to its raw id, so scoring compares ids to ids.
function corpusIdFromIri(iri) {
  const parts = iri.split("/");
  return parts[parts.length - 1];
}

// One /search per query record (the current path). Returns ranked corpus ids.
// Swap point: a future bulk implementation can produce the same shape from
// /duplicates instead of this per-record loop.
async function ioMatchPerQuery(ds, queryRecords) {
  const out = [];
  let done = 0;
  // @allowloop: sequential HTTP queries against a shared engine; one at a time
  // by design (avoid hammering :8081 shared with sibling agents). No FP equiv.
  for (const rec of queryRecords) {
    const hits = await ioSearch({
      domain: ds.domain,
      commit: ds.commit,
      q: rec.text,
      mode: "vector",
      count: TOP_K,
    });
    const rankedCorpusIds = hits.map((h) => corpusIdFromIri(h.id));
    out.push({ queryId: rec.id, rankedCorpusIds });
    done += 1;
    if (done % 100 === 0) log(`  queried ${done}/${queryRecords.length}`);
    if (QUERY_DELAY_MS > 0) await sleep(QUERY_DELAY_MS);
  }
  return out;
}

async function ioVerify(datasetKey) {
  const ds = getDataset(datasetKey);
  log(`dataset=${ds.name} domain=${ds.domain} commit=${ds.commit}`);

  await ioWaitReady();

  const allQueryRecords = ioLoadSide(ds, ds.query);
  const mapping = ioLoadMapping(ds);
  const queryRecords = QUERY_LIMIT > 0 ? allQueryRecords.slice(0, QUERY_LIMIT) : allQueryRecords;
  log(`loaded ${allQueryRecords.length} ${ds.query.side} query records; ${mapping.size} mapped query ids`);
  if (QUERY_LIMIT > 0) log(`PARTIAL SWEEP: capped to first ${queryRecords.length} query records (BENCH_LIMIT)`);

  log(`running ${queryRecords.length} per-query searches (top-${TOP_K})…`);
  const matched = await ioMatchPerQuery(ds, queryRecords);

  const results = matched.map((m) => ({
    queryId: m.queryId,
    rankedCorpusIds: m.rankedCorpusIds,
    truthSet: mapping.get(m.queryId) || new Set(),
  }));

  const score = scoreResults(results, KS);
  printScorecard(ds, queryRecords.length, mapping.size, score);
  return score;
}

function pct(fraction) {
  return (fraction * 100).toFixed(2) + "%";
}

function printScorecard(ds, queryCount, mappedCount, score) {
  const lines = [];
  lines.push("");
  lines.push("==================== ENTITY-RESOLUTION SCORECARD ====================");
  lines.push(`dataset            : ${ds.name}`);
  lines.push(`corpus side        : ${ds.corpus.side}  (indexed population)`);
  lines.push(`query side         : ${ds.query.side}  (one /search each)`);
  lines.push(`query records      : ${queryCount}`);
  lines.push(`query ids w/ truth : ${mappedCount} (scored: ${score.total})`);
  lines.push("---------------------------------------------------------------------");
  lines.push(`precision@1        : ${pct(score.precisionAt1.fraction)}  (${score.precisionAt1.hits}/${score.precisionAt1.total})`);
  for (const k of KS) {
    const r = score.recallAtK[k];
    lines.push(`recall@${String(k).padEnd(2)}         : ${pct(r.fraction)}  (${r.hits}/${r.total})`);
  }
  lines.push("---------------------------------------------------------------------");
  lines.push(`example misses (top-1 wrong), showing up to 8 of ${score.misses.length}:`);
  for (const m of score.misses.slice(0, 8)) {
    lines.push(`  query ${m.queryId}: truth=[${m.truth.join(",")}] top3=[${m.top.join(",")}]`);
  }
  lines.push("=====================================================================");
  lines.push("");
  process.stdout.write(lines.join("\n"));
}

if (require.main === module) {
  const key = process.argv[2] || "abt-buy";
  ioVerify(key).catch((err) => {
    process.stderr.write(`[verify] FAILED: ${err.stack || err.message}\n`);
    process.exit(1);
  });
}

module.exports = { ioVerify, ioMatchPerQuery, corpusIdFromIri };
