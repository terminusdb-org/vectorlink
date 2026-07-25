"use strict";

// v2 entrypoint -- the reciprocal cross-NN entity-resolution benchmark (spec 17).
//
// Usage:
//   node src/bench-v2.js [--mode search|duplicates|similar] [--search-mode vector|fts|hybrid]
//                        [--k N] [--threshold T]
//                        [--cardinality many-to-many|one-to-many|one-to-one]
//                        [--tau-one-to-one T] [--tau-one-to-many T] [--tau-many-to-one T]
//                        [--max-component N] [--no-load] [--query-delay-ms N]
//                        [dataset-key]
//
// --threshold T  The ENGINE/GATHER-side distance cap. Controls what the engine
//   returns and what gets cached. Gather once at a wide threshold (e.g. 0.7),
//   then sweep different --tau-* values against it for free (instant cache reuse).
//   Default: maxActiveTau (the loosest active tau) — backward-compatible.
//   INVARIANT: --threshold must be >= every active tau (fail-loud if violated).
//
// The THREE INDEPENDENT THRESHOLDS are the RESOLVE-side precision interface
// (in-memory filters applied downstream to the gathered/cached candidate set):
//   --tau-one-to-one   closeness for the 1:1 mutual-best CORE (reciprocal pairs)
//   --tau-one-to-many  closeness for ADDITIONAL set-side matches
//   --tau-many-to-one  closeness for ADDITIONAL target-side matches
// --cardinality selects a PRESET (convenience defaults for the three tau).
// Explicit --tau-* overrides take precedence over the preset.
//
// --search-mode controls the ENGINE's retrieval mode for the /search endpoint:
//   vector  — pure cosine ANN (default, backward-compatible)
//   fts     — full-text search only (BM25-like)
//   hybrid  — Reciprocal Rank Fusion (RRF) over vector + FTS ranked lists
//
// NOTE on duplicates mode: /duplicates returns TOP-1 per set point, so widening
// --threshold recovers distant 1:1-core matches (recall of the core) but CANNOT
// recover 2nd+ matches (many-to-many extras) — those only surface via search/similar
// with the directional-extras fix.

const fs = require("fs");
const path = require("path");
const { getDataset } = require("./datasets");
const { ioWaitReady, ioStatistics } = require("./engine");
const { ioLoadV2, ioRenderSide } = require("./load-v2");
const { ioLoadMapping } = require("./load-records");
const { ioGatherCandidates } = require("./modes");
const { resolve, DEFAULTS, CARDINALITIES, maxActiveTau } = require("./resolve");
const { scoreV2 } = require("./score-v2");

// Candidate cache.
const CACHE_DIR = path.join(__dirname, "..", ".candidate-cache");

function cachePath(ds, mode, searchMode) {
  // searchMode differentiates caches: vector candidates differ from hybrid.
  const suffix = (searchMode && searchMode !== "vector") ? `.${searchMode}` : "";
  return path.join(CACHE_DIR, `${ds.name}.${mode}${suffix}.json`);
}

function ioWriteCache(ds, mode, searchMode, gatherK, gatherThreshold, setToTarget, targetToSet, gatherMs) {
  fs.mkdirSync(CACHE_DIR, { recursive: true });
  const payload = {
    dataset: ds.name,
    commit: ds.commit,
    mode,
    searchMode,
    gatherK,
    gatherThreshold,
    gatherMs,
    abtToBuy: Object.fromEntries(setToTarget),
    buyToAbt: Object.fromEntries(targetToSet),
  };
  fs.writeFileSync(cachePath(ds, mode, searchMode), JSON.stringify(payload));
}

function ioReadCache(ds, mode, searchMode) {
  const file = cachePath(ds, mode, searchMode);
  if (!fs.existsSync(file)) return null;
  const payload = JSON.parse(fs.readFileSync(file, "utf-8"));
  return {
    commit: payload.commit,
    gatherK: payload.gatherK,
    gatherThreshold: payload.gatherThreshold,
    gatherMs: payload.gatherMs,
    setToTarget: new Map(Object.entries(payload.abtToBuy)),
    targetToSet: new Map(Object.entries(payload.buyToAbt)),
  };
}

function cacheReusable(cached, mode, k, threshold, commit) {
  if (cached === null) return false;
  if (cached.commit !== undefined && cached.commit !== commit) return false;
  if (!(cached.gatherK >= k)) return false;
  // Duplicates mode: the engine applies the threshold server-side (only returns
  // pairs within that distance). A cache gathered at a tighter threshold is missing
  // edges. Search/similar modes: the engine returns top-K regardless of distance;
  // the threshold is applied in the resolver, so cached data is complete at any K.
  const gatherIsDistanceCapped = mode === "duplicates";
  if (gatherIsDistanceCapped && cached.gatherThreshold !== undefined &&
      !(cached.gatherThreshold >= threshold)) return false;
  return true;
}

function log(msg) {
  process.stdout.write(`[bench-v2] ${msg}\n`);
}

const SEARCH_MODES = Object.freeze(["vector", "fts", "hybrid"]);

function parseArgs(argv) {
  const out = {
    mode: "search",
    searchMode: "vector",     // vector|fts|hybrid — the engine's retrieval mode
    k: DEFAULTS.k,
    threshold: undefined,     // undefined = derive from maxActiveTau (backwards-compat)
    tauOneToOne: undefined,   // undefined = use preset default
    tauOneToMany: undefined,
    tauManyToOne: undefined,
    maxComponentSize: DEFAULTS.maxComponentSize,
    cardinality: DEFAULTS.cardinality,
    reload: false,
    queryDelayMs: 25,
    dataset: "abt-buy-v2",
    useCache: true,
    gatherK: 10,
  };
  const flags = {
    "--mode": (v) => { out.mode = v; },
    "--search-mode": (v) => { out.searchMode = v; },
    "--k": (v) => { out.k = Number(v); },
    "--threshold": (v) => { out.threshold = Number(v); },
    "--tau-one-to-one": (v) => { out.tauOneToOne = Number(v); },
    "--tau-one-to-many": (v) => { out.tauOneToMany = Number(v); },
    "--tau-many-to-one": (v) => { out.tauManyToOne = Number(v); },
    "--max-component": (v) => { out.maxComponentSize = Number(v); },
    "--cardinality": (v) => { out.cardinality = v; },
    "--query-delay-ms": (v) => { out.queryDelayMs = Number(v); },
    "--gather-k": (v) => { out.gatherK = Number(v); },
  };
  // @allowloop: argv is a positional/flag stream; index-coupled consumption of
  // value-flags has no clean map/filter form. Bounded by argv.length.
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--reload" || arg === "--force") { out.reload = true; continue; }
    if (arg === "--no-load") { out.reload = false; continue; }
    if (arg === "--no-cache") { out.useCache = false; continue; }
    if (arg in flags) { flags[arg](argv[++i]); continue; }
    if (arg.startsWith("--")) throw new Error(`Unknown flag: ${arg}`);
    out.dataset = arg;
  }
  if (!["search", "duplicates", "similar"].includes(out.mode)) {
    throw new Error(`--mode must be one of search|duplicates|similar (got "${out.mode}")`);
  }
  if (!SEARCH_MODES.includes(out.searchMode)) {
    throw new Error(`--search-mode must be one of ${SEARCH_MODES.join("|")} (got "${out.searchMode}")`);
  }
  if (!CARDINALITIES.includes(out.cardinality)) {
    throw new Error(`--cardinality must be one of ${CARDINALITIES.join("|")} (got "${out.cardinality}")`);
  }
  if (!Number.isFinite(out.k) || out.k < 1) throw new Error(`--k must be a positive integer (got ${out.k})`);
  if (out.threshold !== undefined) {
    if (!Number.isFinite(out.threshold) || out.threshold < 0 || out.threshold > 1) {
      throw new Error(`--threshold must be a number in [0, 1] (got ${out.threshold})`);
    }
  }
  return out;
}

// POKA-YOKE: any active tau > gatherTau means the resolve would silently miss
// pairs the engine never returned. Fail loud — never produce understated recall.
function validateGatherCeiling(gatherTau, derivedGatherTau) {
  if (derivedGatherTau > gatherTau) {
    throw new Error(
      `RECALL CEILING VIOLATION: the loosest active tau (${derivedGatherTau.toFixed(3)}) ` +
      `exceeds --threshold (${gatherTau.toFixed(3)}). The gather would miss edges in ` +
      `[${gatherTau.toFixed(3)}, ${derivedGatherTau.toFixed(3)}], silently understating ` +
      "recall. Either widen --threshold or tighten the tau values.",
    );
  }
}

async function ioVerifySnapshotReusable(expectedCount) {
  const stats = await ioStatistics();
  const pending = stats.pending_index_fragments ?? 0;
  if (stats.documents < expectedCount) {
    throw new Error(
      `Refusing to reuse snapshot: engine reports ${stats.documents} indexed documents ` +
      `(global), fewer than the ${expectedCount} this dataset needs -- the snapshot is ` +
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
    log("--reload: re-indexing the snapshot (DELETE + push + wait Complete)...");
    const loaded = await ioLoadV2(config.dataset);
    abt = loaded.abt;
    buy = loaded.buy;
  } else {
    log("reuse (default): rendering sides locally and verifying the indexed snapshot...");
    abt = ioRenderSide(ds, ds.sides.abt);
    buy = ioRenderSide(ds, ds.sides.buy);
    const stats = await ioVerifySnapshotReusable(abt.length + buy.length);
    log(`snapshot reuse OK: engine documents=${stats.documents} (global), pending_index_fragments=${stats.pending_index_fragments ?? 0}`);
  }

  const mapping = ioLoadMapping(ds);
  log(`loaded ${abt.length} Abt + ${buy.length} Buy; ${mapping.size} mapped Abt ids`);

  const gatherK = Math.max(config.gatherK, config.k);

  // Build resolve options: preset + explicit tau overrides.
  const resolveOpts = {
    k: config.k,
    maxComponentSize: config.maxComponentSize,
    cardinality: config.cardinality,
  };
  if (config.tauOneToOne !== undefined) resolveOpts.tauOneToOne = config.tauOneToOne;
  if (config.tauOneToMany !== undefined) resolveOpts.tauOneToMany = config.tauOneToMany;
  if (config.tauManyToOne !== undefined) resolveOpts.tauManyToOne = config.tauManyToOne;

  // Compute the effective tau values (for the poka-yoke ceiling check).
  const effectiveThresholds = { tauOneToOne: resolveOpts.tauOneToOne ?? DEFAULTS.tauOneToOne };
  if (resolveOpts.tauOneToMany !== undefined) effectiveThresholds.tauOneToMany = resolveOpts.tauOneToMany;
  else if (config.cardinality !== "one-to-one") effectiveThresholds.tauOneToMany = DEFAULTS.tauOneToMany;
  if (resolveOpts.tauManyToOne !== undefined) effectiveThresholds.tauManyToOne = resolveOpts.tauManyToOne;
  else if (config.cardinality === "many-to-many") effectiveThresholds.tauManyToOne = DEFAULTS.tauManyToOne;

  // --threshold decouples the GATHER (engine-side cap) from the RESOLVE (in-memory
  // tau filters). If explicit, it drives the engine/cache threshold directly.
  // If not given, derive from the loosest active tau (backward-compatible default).
  const derivedGatherTau = maxActiveTau(effectiveThresholds);
  const gatherTau = config.threshold ?? derivedGatherTau;
  validateGatherCeiling(gatherTau, derivedGatherTau);

  let setToTarget;
  let targetToSet;
  let gatherMs;
  let gatherFromCache = false;
  const cached = config.useCache ? ioReadCache(ds, config.mode, config.searchMode) : null;
  if (cacheReusable(cached, config.mode, config.k, gatherTau, ds.commit)) {
    log(`using cached candidates (mode=${config.mode}, searchMode=${config.searchMode}, gatherK=${cached.gatherK}, gather-tau=${cached.gatherThreshold}, commit=${cached.commit}); resolve at k=${config.k}`);
    setToTarget = cached.setToTarget;
    targetToSet = cached.targetToSet;
    gatherMs = cached.gatherMs;
    gatherFromCache = true;
  } else {
    log(`gathering candidates via mode=${config.mode} (gatherK=${gatherK}, tau=${gatherTau}, searchMode=${config.searchMode})...`);
    const gatherStart = process.hrtime.bigint();
    const gathered = await ioGatherCandidates(config.mode, ds, abt, buy, {
      k: gatherK,
      threshold: gatherTau,
      queryDelayMs: config.queryDelayMs,
      searchMode: config.searchMode,
    });
    gatherMs = Number(process.hrtime.bigint() - gatherStart) / 1e6;
    setToTarget = gathered.abtToBuy;
    targetToSet = gathered.buyToAbt;
    ioWriteCache(ds, config.mode, config.searchMode, gatherK, gatherTau, setToTarget, targetToSet, gatherMs);
    log(`cached candidates to ${cachePath(ds, config.mode, config.searchMode)}`);
  }

  log(`running 3-threshold resolver (cardinality=${config.cardinality})...`);
  const resolveStart = process.hrtime.bigint();
  const result = resolve(setToTarget, targetToSet, resolveOpts);
  const resolveMs = Number(process.hrtime.bigint() - resolveStart) / 1e6;

  const score = scoreV2(result.matched, mapping);
  printScorecard(ds, config, result, score, { gatherMs, resolveMs, gatherFromCache, gatherTau });
  return { result, score, timings: { gatherMs, resolveMs } };
}

function pct(fraction) {
  return (fraction * 100).toFixed(2) + "%";
}

function tauDisplay(value) {
  if (value === null || value === undefined) return "disabled";
  return value.toFixed(3);
}

function printScorecard(ds, config, result, score, timings) {
  const s = result.stats;
  const lines = [];
  lines.push("");
  lines.push("================= ENTITY-RESOLUTION v2 SCORECARD (3-threshold model) ============");
  lines.push(`dataset            : ${ds.name}   snapshot=${ds.domain}@${ds.commit}`);
  lines.push("--- config ---");
  lines.push(`mode               : ${config.mode}`);
  lines.push(`search mode        : ${config.searchMode} (engine retrieval: vector|fts|hybrid)`);
  lines.push(`k (fan-out)        : ${s.k}`);
  lines.push(`cardinality preset : ${s.cardinality}`);
  lines.push(`tau_one_to_one     : ${tauDisplay(s.tauOneToOne)} (core: mutual-best reciprocal pairs)`);
  lines.push(`tau_one_to_many    : ${tauDisplay(s.tauOneToMany)} (set-side extras: one set -> many targets)`);
  lines.push(`tau_many_to_one    : ${tauDisplay(s.tauManyToOne)} (target-side extras: one target -> many set)`);
  lines.push(`gather threshold   : ${timings.gatherTau.toFixed(3)} (--threshold: engine-side cap, ceiling for tau)`);
  lines.push(`graph tau (max)    : ${s.graphTau.toFixed(3)} (loosest active tau, used for candidate graph filter)`);
  lines.push(`max component size : ${s.maxComponentSize}`);
  lines.push("--- candidate graph ---");
  lines.push(`edges (<= graph tau): ${s.edgeCount}`);
  lines.push("--- 3-PARTITION OUTPUT ---");
  lines.push(`matched pairs      : ${s.matchedCount}`);
  lines.push(`set_only (no match): ${s.setOnlyCount}`);
  lines.push(`target_only        : ${s.targetOnlyCount}`);
  lines.push("--- per-stage precision ---");
  lines.push(`core pairs         : ${score.counts.corePairs}   precision ${pct(score.core.fraction)} (${score.core.correct}/${score.core.total})`);
  lines.push(`set_extra pairs    : ${score.counts.setExtraPairs}   precision ${pct(score.setExtra.fraction)} (${score.setExtra.correct}/${score.setExtra.total})`);
  lines.push(`target_extra pairs : ${score.counts.targetExtraPairs}   precision ${pct(score.targetExtra.fraction)} (${score.targetExtra.correct}/${score.targetExtra.total})`);
  lines.push("--- HEADLINE: pair-based F1 ---");
  lines.push(`|P| predicted      : ${score.counts.predictedPairsUnique}`);
  lines.push(`|G| gold pairs     : ${score.counts.truePairs} (perfectMapping)`);
  lines.push(`precision          : ${pct(score.overall.fraction)} (TP ${score.counts.truePositives} / |P| ${score.overall.total})`);
  lines.push(`recall             : ${pct(score.recall.fraction)} (TP ${score.counts.truePositives} / |G| ${score.recall.total})`);
  lines.push(`F1                 : ${pct(score.f1)}`);
  lines.push(`TP / FP / FN       : ${score.counts.truePositives} / ${score.counts.falsePositives} / ${score.counts.falseNegatives}`);
  lines.push("--- detail ---");
  lines.push(`precision (mapped) : ${pct(score.mappedOnly.fraction)} (${score.mappedOnly.correct}/${score.mappedOnly.total}; excludes unmapped-set predictions)`);
  lines.push("--- wall-clock ---");
  lines.push(`candidate gather   : ${timings.gatherMs.toFixed(0)} ms (mode=${config.mode})${timings.gatherFromCache ? " [REPLAYED from cache]" : ""}`);
  lines.push(`resolve total      : ${timings.resolveMs.toFixed(0)} ms`);
  lines.push("--- example wrong pairs (up to 8) ---");
  for (const w of score.wrongExamples) {
    lines.push(`  Set ${w.setId} -> Target ${w.predictedTarget} [${w.stage}]  truth=[${w.truth.join(",")}]`);
  }
  lines.push("=============================================================================");
  lines.push("NOTE: The 3-threshold model replaces both the one-per-Abt model (88.8% F1)");
  lines.push("and the single-tau model. Precision is tuned via independent tau per match");
  lines.push("class, not by clamping cardinality.");
  lines.push("");
  lines.push("FUTURE: auto-fit model to calculate optimal tau from target distribution.");
  lines.push("");
  lines.push("v1 baseline: precision@1 83.16% (899/1081), recall@10 98.06%");
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

module.exports = { ioBenchV2, parseArgs, cacheReusable, validateGatherCeiling };
