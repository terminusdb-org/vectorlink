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

const fs = require("fs");
const path = require("path");
const { getDataset } = require("./datasets");
const { ioWaitReady, ioStatistics } = require("./engine");
const { ioLoadV2, ioRenderSide } = require("./load-v2");
const { ioLoadMapping } = require("./load-records");
const { ioGatherCandidates } = require("./modes");
const { resolve, DEFAULTS, ASSIGNMENT_STRATEGIES } = require("./resolve");
const { scoreV2 } = require("./score-v2");

// Candidate cache (kaizen — never re-pay the expensive per-query gather). The
// gather step (especially search/similar, which embed every query) dominates
// wall-clock; the resolve+score is milliseconds. Caching the gathered cross-NN
// lists lets us sweep k / τ and re-score for free, and compare modes without
// re-querying. Cache is keyed by mode + the gather k (the list length); a resolve
// at any k' ≤ cached-k just slices the lists, so we gather ONCE at a generous k.
const CACHE_DIR = path.join(__dirname, "..", ".candidate-cache");

function cachePath(ds, mode) {
  return path.join(CACHE_DIR, `${ds.name}.${mode}.json`);
}

function ioWriteCache(ds, mode, gatherK, gatherThreshold, abtToBuy, buyToAbt, gatherMs) {
  fs.mkdirSync(CACHE_DIR, { recursive: true });
  const payload = {
    dataset: ds.name,
    commit: ds.commit, // bind the cache to the exact snapshot it was gathered from
    mode,
    gatherK,
    // The τ used AT GATHER TIME. For duplicates mode the engine prunes by τ
    // server-side, so candidates are already τ-filtered; reusing them at a LOWER
    // τ would be wrong (fewer edges than a fresh gather would yield). search/
    // similar do not apply τ at gather (resolve does), so this is recorded for
    // all modes but only gates the τ-dependent ones (see ioReadCache caller).
    gatherThreshold,
    gatherMs,
    abtToBuy: Object.fromEntries(abtToBuy),
    buyToAbt: Object.fromEntries(buyToAbt),
  };
  fs.writeFileSync(cachePath(ds, mode), JSON.stringify(payload));
}

function ioReadCache(ds, mode) {
  const file = cachePath(ds, mode);
  if (!fs.existsSync(file)) return null;
  const payload = JSON.parse(fs.readFileSync(file, "utf-8"));
  return {
    commit: payload.commit,
    gatherK: payload.gatherK,
    gatherThreshold: payload.gatherThreshold,
    gatherMs: payload.gatherMs,
    abtToBuy: new Map(Object.entries(payload.abtToBuy)),
    buyToAbt: new Map(Object.entries(payload.buyToAbt)),
  };
}

// Decide whether a cached gather can be reused for this run. Reuse requires:
//   - same snapshot commit (a re-index invalidates the gathered ids);
//   - cached gatherK ≥ requested k (we can always slice DOWN, never up);
//   - for τ-dependent gather modes (duplicates), the cached gather τ must be ≥
//     the requested τ (a cache gathered at a TIGHTER τ is missing edges a looser
//     τ would include — refusing prevents a silently understated recall).
function cacheReusable(cached, mode, k, threshold, commit) {
  if (cached === null) return false;
  if (cached.commit !== undefined && cached.commit !== commit) return false;
  if (!(cached.gatherK >= k)) return false;
  const tauDependentGather = mode === "duplicates";
  if (tauDependentGather && !(cached.gatherThreshold >= threshold)) return false;
  return true;
}

function log(msg) {
  process.stdout.write(`[bench-v2] ${msg}\n`);
}

// Tiny, dependency-free flag parser. Unknown flags fail loud (poka-yoke: a typo'd
// knob must not be silently ignored — it would mis-measure a run).
function parseArgs(argv) {
  // REUSE-BY-DEFAULT: the indexed vectors do not change between runs (only the
  // matching algorithm/mode/knobs do), so re-embedding ~2173 docs every run is
  // waste. Default = reuse the already-indexed snapshot if it is present AND
  // complete (verified before gather). `--reload`/`--force` is the conscious
  // re-index (DELETE+push). `--no-load` is kept as an alias for reuse.
  const out = {
    mode: "search",
    k: DEFAULTS.k,
    threshold: DEFAULTS.threshold,
    maxComponentSize: DEFAULTS.maxComponentSize,
    assignment: DEFAULTS.assignment,
    reload: false,
    queryDelayMs: 25,
    dataset: "abt-buy-v2",
    useCache: true,
    gatherK: 10,
  };
  const flags = {
    "--mode": (v) => { out.mode = v; },
    "--k": (v) => { out.k = Number(v); },
    "--threshold": (v) => { out.threshold = Number(v); },
    "--max-component": (v) => { out.maxComponentSize = Number(v); },
    "--assignment": (v) => { out.assignment = v; },
    "--query-delay-ms": (v) => { out.queryDelayMs = Number(v); },
    "--gather-k": (v) => { out.gatherK = Number(v); },
  };
  // @allowloop: argv is a positional/flag stream; index-coupled consumption of
  // value-flags has no clean map/filter form. Bounded by argv.length.
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--reload" || arg === "--force") { out.reload = true; continue; }
    if (arg === "--no-load") { out.reload = false; continue; } // alias: reuse
    if (arg === "--no-cache") { out.useCache = false; continue; }
    if (arg in flags) { flags[arg](argv[++i]); continue; }
    if (arg.startsWith("--")) throw new Error(`Unknown flag: ${arg}`);
    out.dataset = arg;
  }
  if (!["search", "duplicates", "similar"].includes(out.mode)) {
    throw new Error(`--mode must be one of search|duplicates|similar (got "${out.mode}")`);
  }
  if (!ASSIGNMENT_STRATEGIES.includes(out.assignment)) {
    throw new Error(`--assignment must be one of ${ASSIGNMENT_STRATEGIES.join("|")} (got "${out.assignment}")`);
  }
  if (!Number.isFinite(out.k) || out.k < 1) throw new Error(`--k must be a positive integer (got ${out.k})`);
  if (!Number.isFinite(out.threshold) || out.threshold < 0 || out.threshold > 1) {
    throw new Error(`--threshold must be in [0,1] (got ${out.threshold})`);
  }
  return out;
}

// Verify the bench snapshot is present AND complete before reusing it, so we
// never score against a stale/partial index (refinement G — fail loud, never
// silently score an incomplete snapshot). Caveat: /statistics reports a GLOBAL
// document count across ALL domains, not per-domain, so it is a NECESSARY check
// (global ≥ expected) but not a sufficient per-domain one — we also require the
// indexing backlog to be drained (pending_index_fragments === 0). If either
// fails, we refuse to reuse and instruct --reload rather than guess.
async function ioVerifySnapshotReusable(expectedCount) {
  const stats = await ioStatistics();
  const pending = stats.pending_index_fragments ?? 0;
  if (stats.documents < expectedCount) {
    throw new Error(
      `Refusing to reuse snapshot: engine reports ${stats.documents} indexed documents ` +
      `(global), fewer than the ${expectedCount} this dataset needs — the snapshot is ` +
      "absent or partial. Re-index with --reload.",
    );
  }
  if (pending > 0) {
    throw new Error(
      `Refusing to reuse snapshot: ${pending} index fragments still pending ` +
      "(indexing not settled). Wait for indexing to complete, or re-index with --reload.",
    );
  }
  return stats;
}

async function ioBenchV2(config) {
  const ds = getDataset(config.dataset);
  await ioWaitReady();

  let abt;
  let buy;
  if (config.reload) {
    // Conscious re-index: DELETE the domain + push + poll to Complete.
    log("--reload: re-indexing the snapshot (DELETE + push + wait Complete)…");
    const loaded = await ioLoadV2(config.dataset);
    abt = loaded.abt;
    buy = loaded.buy;
  } else {
    // DEFAULT = reuse the already-indexed vectors (they don't change between
    // runs; only the algorithm/mode/knobs do). Render locally for ids+text, then
    // VERIFY the snapshot is present + complete before trusting it.
    log("reuse (default): rendering sides locally and verifying the indexed snapshot…");
    abt = ioRenderSide(ds, ds.sides.abt);
    buy = ioRenderSide(ds, ds.sides.buy);
    const stats = await ioVerifySnapshotReusable(abt.length + buy.length);
    log(`snapshot reuse OK: engine documents=${stats.documents} (global), pending_index_fragments=${stats.pending_index_fragments ?? 0}`);
  }

  const mapping = ioLoadMapping(ds);
  log(`loaded ${abt.length} Abt + ${buy.length} Buy; ${mapping.size} mapped Abt ids`);

  // Gather at a generous k so the cached lists support resolving at any smaller
  // k without re-querying (the resolver slices top-k). Default gatherK ≥ 10.
  const gatherK = Math.max(config.gatherK, config.k);

  let abtToBuy;
  let buyToAbt;
  let gatherMs;
  let gatherFromCache = false;
  const cached = config.useCache ? ioReadCache(ds, config.mode) : null;
  if (cacheReusable(cached, config.mode, config.k, config.threshold, ds.commit)) {
    log(`using cached candidates (mode=${config.mode}, gatherK=${cached.gatherK}, gatherτ=${cached.gatherThreshold}, commit=${cached.commit}); resolve at k=${config.k}`);
    abtToBuy = cached.abtToBuy;
    buyToAbt = cached.buyToAbt;
    gatherMs = cached.gatherMs; // report the ORIGINAL gather wall-clock (replayed, not measured this run)
    gatherFromCache = true;
  } else {
    log(`gathering candidates via mode=${config.mode} (gatherK=${gatherK}, τ=${config.threshold})…`);
    const gatherStart = process.hrtime.bigint();
    const gathered = await ioGatherCandidates(config.mode, ds, abt, buy, {
      k: gatherK,
      threshold: config.threshold,
      queryDelayMs: config.queryDelayMs,
    });
    gatherMs = Number(process.hrtime.bigint() - gatherStart) / 1e6;
    abtToBuy = gathered.abtToBuy;
    buyToAbt = gathered.buyToAbt;
    ioWriteCache(ds, config.mode, gatherK, config.threshold, abtToBuy, buyToAbt, gatherMs);
    log(`cached candidates to ${cachePath(ds, config.mode)}`);
  }

  log("running §4 resolver (ground → components → per-cluster assignment)…");
  const resolveStart = process.hrtime.bigint();
  const result = resolve(abtToBuy, buyToAbt, {
    k: config.k,
    threshold: config.threshold,
    maxComponentSize: config.maxComponentSize,
    assignment: config.assignment,
  });
  const resolveMs = Number(process.hrtime.bigint() - resolveStart) / 1e6;

  const score = scoreV2(result.groups, mapping);
  printScorecard(ds, config, result, score, { gatherMs, resolveMs, gatherFromCache });
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
  lines.push(`grounding strategy : mutual top-K nearest (one committed pair per Abt)`);
  lines.push(`assignment strategy: ${s.assignment}${s.assignment === "per-source" ? " (argmin Buy per Abt; Buy non-exclusive — many-to-one correct)" : " (per-component Hungarian 1:1; greedy fallback on runaway — for 1:1 truth)"}`);
  lines.push("--- candidate graph ---");
  lines.push(`edges (≤ τ)        : ${s.edgeCount}`);
  lines.push("--- committed prediction set P (one pair per Abt; §8.2 methodology) ---");
  lines.push(`|P| predicted      : ${score.counts.predictedPairsUnique}`);
  lines.push(`|G| gold pairs     : ${score.counts.truePairs} (perfectMapping)`);
  lines.push("--- decisions (grounded Step 4 vs assigned Step 5) ---");
  lines.push(`grounded pairs     : ${score.counts.groundedPairs}   precision ${pct(score.grounded.fraction)} (${score.grounded.correct}/${score.grounded.total})`);
  lines.push(`assigned pairs     : ${score.counts.assignedPairs}   precision ${pct(score.assigned.fraction)} (${score.assigned.correct}/${score.assigned.total})`);
  lines.push(`unmatched Abt      : ${s.unmatchedCount}`);
  lines.push("--- HEADLINE: pair-based F1 (comparable across all modes) ---");
  lines.push(`precision          : ${pct(score.overall.fraction)} (TP ${score.counts.truePositives} / |P| ${score.overall.total})`);
  lines.push(`recall             : ${pct(score.recall.fraction)} (TP ${score.counts.truePositives} / |G| ${score.recall.total})`);
  lines.push(`F1                 : ${pct(score.f1)}`);
  lines.push(`TP / FP / FN       : ${score.counts.truePositives} / ${score.counts.falsePositives} / ${score.counts.falseNegatives}`);
  lines.push("--- detail ---");
  lines.push(`precision (mapped) : ${pct(score.mappedOnly.fraction)} (${score.mappedOnly.correct}/${score.mappedOnly.total}; v1-comparable, excludes unmapped-Abt)`);
  lines.push("--- §6 performance contract ---");
  lines.push(`components         : ${s.componentCount}`);
  lines.push(`max component      : ${s.maxComponentObserved} (guard cap ${config.maxComponentSize})`);
  lines.push(`runaway components : ${s.runawayComponents} (fell back to greedy)`);
  lines.push(`cluster sizes      : ${clusterHistogram(s.componentSizes)}`);
  lines.push(`assignment time    : ${s.assignmentMs.toFixed(1)} ms (Σ per-cluster)`);
  lines.push("--- wall-clock ---");
  lines.push(`candidate gather   : ${timings.gatherMs.toFixed(0)} ms (mode=${config.mode})${timings.gatherFromCache ? " [REPLAYED from cache — NOT measured this run]" : ""}`);
  lines.push(`resolve total      : ${timings.resolveMs.toFixed(0)} ms`);
  lines.push("--- example wrong pairs (up to 8) ---");
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

module.exports = { ioBenchV2, parseArgs, cacheReusable };
