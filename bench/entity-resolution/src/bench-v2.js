"use strict";

// v2 entrypoint — the reciprocal cross-NN entity-resolution benchmark (spec 17).
//
// Usage:
//   node src/bench-v2.js [--mode search|duplicates|similar] [--k N]
//                        [--threshold T] [--max-component N] [--no-load]
//                        [--query-delay-ms N] [dataset-key]
//
// Pipeline: (load both catalogues into one snapshot) → gather candidates via the
// chosen MODE → run the pure §4 resolver → score vs the perfect mapping → print
// the full config + scorecard (grounded vs assigned split, cluster-size stats,
// assignment time, wall-clock).

const { getDataset } = require("./datasets");
const { ioWaitReady } = require("./engine");
const { ioLoadV2, ioRenderSide } = require("./load-v2");
const { ioLoadMapping } = require("./load-records");
const { ioGatherCandidates } = require("./modes");
const { resolve, DEFAULTS } = require("./resolve");
const { scoreV2 } = require("./score-v2");

function log(msg) {
  process.stdout.write(`[bench-v2] ${msg}\n`);
}

// Tiny, dependency-free flag parser. Unknown flags fail loud (poka-yoke: a typo'd
// knob must not be silently ignored — it would mis-measure a run).
function parseArgs(argv) {
  const out = { mode: "search", k: DEFAULTS.k, threshold: DEFAULTS.threshold, maxComponentSize: DEFAULTS.maxComponentSize, load: true, queryDelayMs: 25, dataset: "abt-buy-v2" };
  const flags = {
    "--mode": (v) => { out.mode = v; },
    "--k": (v) => { out.k = Number(v); },
    "--threshold": (v) => { out.threshold = Number(v); },
    "--max-component": (v) => { out.maxComponentSize = Number(v); },
    "--query-delay-ms": (v) => { out.queryDelayMs = Number(v); },
  };
  // @allowloop: argv is a positional/flag stream; index-coupled consumption of
  // value-flags has no clean map/filter form. Bounded by argv.length.
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--no-load") { out.load = false; continue; }
    if (arg in flags) { flags[arg](argv[++i]); continue; }
    if (arg.startsWith("--")) throw new Error(`Unknown flag: ${arg}`);
    out.dataset = arg;
  }
  if (!["search", "duplicates", "similar"].includes(out.mode)) {
    throw new Error(`--mode must be one of search|duplicates|similar (got "${out.mode}")`);
  }
  if (!Number.isFinite(out.k) || out.k < 1) throw new Error(`--k must be a positive integer (got ${out.k})`);
  if (!Number.isFinite(out.threshold) || out.threshold < 0 || out.threshold > 1) {
    throw new Error(`--threshold must be in [0,1] (got ${out.threshold})`);
  }
  return out;
}

async function ioBenchV2(config) {
  const ds = getDataset(config.dataset);
  await ioWaitReady();

  let abt;
  let buy;
  if (config.load) {
    const loaded = await ioLoadV2(config.dataset);
    abt = loaded.abt;
    buy = loaded.buy;
  } else {
    // Re-use an already-indexed snapshot (A/B knob sweeps over the same corpus —
    // spec §8: cheap re-runs, no re-embedding). Render locally for the ids+text.
    log("--no-load: assuming the snapshot is already indexed; rendering sides locally");
    abt = ioRenderSide(ds, ds.sides.abt);
    buy = ioRenderSide(ds, ds.sides.buy);
  }

  const mapping = ioLoadMapping(ds);
  log(`loaded ${abt.length} Abt + ${buy.length} Buy; ${mapping.size} mapped Abt ids`);

  log(`gathering candidates via mode=${config.mode} (k=${config.k}, τ=${config.threshold})…`);
  const gatherStart = process.hrtime.bigint();
  const { abtToBuy, buyToAbt } = await ioGatherCandidates(config.mode, ds, abt, buy, {
    k: config.k,
    threshold: config.threshold,
    queryDelayMs: config.queryDelayMs,
  });
  const gatherMs = Number(process.hrtime.bigint() - gatherStart) / 1e6;

  log("running §4 resolver (ground → components → per-cluster assignment)…");
  const resolveStart = process.hrtime.bigint();
  const result = resolve(abtToBuy, buyToAbt, {
    k: config.k,
    threshold: config.threshold,
    maxComponentSize: config.maxComponentSize,
  });
  const resolveMs = Number(process.hrtime.bigint() - resolveStart) / 1e6;

  const score = scoreV2(result.groups, mapping);
  printScorecard(ds, config, result, score, { gatherMs, resolveMs });
  return { result, score, timings: { gatherMs, resolveMs } };
}

function pct(fraction) {
  return (fraction * 100).toFixed(2) + "%";
}

function clusterHistogram(sizes) {
  const buckets = sizes.reduce((acc, s) => {
    const key = s <= 2 ? "2" : s <= 4 ? "3-4" : s <= 8 ? "5-8" : s <= 16 ? "9-16" : ">16";
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});
  return Object.entries(buckets).map(([range, n]) => `${range}:${n}`).join("  ");
}

function printScorecard(ds, config, result, score, timings) {
  const s = result.stats;
  const lines = [];
  lines.push("");
  lines.push("================= ENTITY-RESOLUTION v2 SCORECARD (spec 17 §4) =================");
  lines.push(`dataset            : ${ds.name}   snapshot=${ds.domain}@${ds.commit}`);
  lines.push("--- config (every knob, printed per run) ---");
  lines.push(`mode               : ${config.mode}`);
  lines.push(`k (fan-out)        : ${config.k}`);
  lines.push(`threshold τ        : ${config.threshold}`);
  lines.push(`max component size : ${config.maxComponentSize}`);
  lines.push(`grounding strategy : mutual top-K membership (refinement C)`);
  lines.push(`assignment strategy: per-component optimal (Hungarian); greedy fallback on runaway`);
  lines.push("--- candidate graph ---");
  lines.push(`edges (≤ τ)        : ${s.edgeCount}`);
  lines.push("--- decisions (grounded Step 4 vs assigned Step 5) ---");
  lines.push(`grounded pairs     : ${s.groundedCount}   precision ${pct(score.grounded.fraction)} (${score.grounded.correct}/${score.grounded.total})`);
  lines.push(`assigned pairs     : ${s.assignedCount}   precision ${pct(score.assigned.fraction)} (${score.assigned.correct}/${score.assigned.total})`);
  lines.push(`unmatched Abt      : ${s.unmatchedCount}`);
  lines.push("--- overall vs perfect mapping ---");
  lines.push(`precision          : ${pct(score.overall.fraction)} (${score.overall.correct}/${score.overall.total} scoreable pairs)`);
  lines.push(`recall (mapped Abt): ${pct(score.recall.fraction)} (${score.recall.hits}/${score.recall.total})`);
  lines.push("--- §6 performance contract ---");
  lines.push(`components         : ${s.componentCount}`);
  lines.push(`max component      : ${s.maxComponentObserved} (guard cap ${config.maxComponentSize})`);
  lines.push(`runaway components : ${s.runawayComponents} (fell back to greedy)`);
  lines.push(`cluster sizes      : ${clusterHistogram(s.componentSizes)}`);
  lines.push(`assignment time    : ${s.assignmentMs.toFixed(1)} ms (Σ per-cluster)`);
  lines.push("--- wall-clock ---");
  lines.push(`candidate gather   : ${timings.gatherMs.toFixed(0)} ms (mode=${config.mode})`);
  lines.push(`resolve total      : ${timings.resolveMs.toFixed(0)} ms`);
  lines.push("--- example wrong pairs (predicted, scoreable, up to 8) ---");
  for (const w of score.wrongExamples) {
    lines.push(`  Abt ${w.abtId} -> Buy ${w.predictedBuy} [${w.stage}]  truth=[${w.truth.join(",")}]`);
  }
  lines.push("=============================================================================");
  lines.push("v1 baseline (for comparison): precision@1 83.16% (899/1081), recall@10 98.06%");
  lines.push("");
  process.stdout.write(lines.join("\n"));
}

if (require.main === module) {
  let config;
  try {
    config = parseArgs(process.argv);
  } catch (e) {
    process.stderr.write(`[bench-v2] ${e.message}\n`);
    process.exit(2);
  }
  ioBenchV2(config).catch((err) => {
    process.stderr.write(`[bench-v2] FAILED: ${err.stack || err.message}\n`);
    process.exit(1);
  });
}

module.exports = { ioBenchV2, parseArgs };
