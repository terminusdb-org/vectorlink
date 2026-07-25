"use strict";

// Endpoint-driven entity-resolution bench — replaces the per-record HTTP approach
// (bench-v2.js + modes.js + resolve.js) with a SINGLE POST /resolve call.
//
// Usage:
//   node src/bench-resolve.js [--threshold T] [--tau-one-to-one T]
//                             [--tau-one-to-many T] [--tau-many-to-one T]
//                             [--k N] [--domain D] [--commit C]
//                             [--set-doc-types Abt] [--target-doc-types Buy]
//                             [dataset-key]
//
// The engine handles BOTH the gather (cross-NN retrieval) AND the matching
// (3-threshold algorithm) in a single in-process batch call. The bench:
//   1. Calls POST /resolve once.
//   2. Maps the endpoint's {matched, set_only, target_only} to the scorer.
//   3. Reports F1/precision/recall against ground truth.
//   4. Reports wall-clock (end-to-end + endpoint's stats.elapsed_ms).

const { getDataset } = require("./datasets");
const { ioWaitReady, ioStatistics } = require("./engine");
const { ioLoadMapping } = require("./load-records");
const { ioRenderSide } = require("./load-v2");
const { idFromIri, isSide } = require("./iri");
const { scoreV2 } = require("./score-v2");

const ENGINE_URL = process.env.ENGINE_URL || "http://localhost:8081";
const ENGINE_CRED = process.env.ENGINE_CRED || "admin:root";
const AUTH_HEADER = "Basic " + Buffer.from(ENGINE_CRED).toString("base64");

// ── POST /resolve client ────────────────────────────────────────────────────

async function ioResolve({ domain, commit, setDocTypes, targetDocTypes,
  setDocIds, targetDocIds, threshold, tauOneToOne, tauOneToMany,
  tauManyToOne, k, ancestors }) {
  const body = {
    domain,
    commit,
    set_doc_types: setDocTypes,
    set_doc_ids: setDocIds || [],
    target_doc_types: targetDocTypes,
    target_doc_ids: targetDocIds || [],
    threshold,
    tau_one_to_one: tauOneToOne,
    k,
    ancestors: ancestors || [],
  };
  // Only include optional tau if defined (null = disabled at the engine level).
  if (tauOneToMany !== undefined && tauOneToMany !== null) {
    body.tau_one_to_many = tauOneToMany;
  }
  if (tauManyToOne !== undefined && tauManyToOne !== null) {
    body.tau_many_to_one = tauManyToOne;
  }

  const url = `${ENGINE_URL}/resolve`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": AUTH_HEADER,
    },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  if (!res.ok) {
    throw new Error(`POST /resolve -> HTTP ${res.status}: ${text.slice(0, 800)}`);
  }
  return JSON.parse(text);
}

// ── CLI arg parsing ─────────────────────────────────────────────────────────

function parseArgs(argv) {
  const out = {
    threshold: 0.5,
    tauOneToOne: 0.45,
    tauOneToMany: undefined,    // undefined = omit (engine disables)
    tauManyToOne: undefined,    // undefined = omit (engine disables)
    k: 5,
    domain: undefined,          // undefined = derive from dataset config
    commit: undefined,          // undefined = fetch from TerminusDB
    setDocTypes: ["Abt"],
    targetDocTypes: ["Buy"],
    dataset: "abt-buy-v2",
  };
  const flags = {
    "--threshold": (v) => { out.threshold = Number(v); },
    "--tau-one-to-one": (v) => { out.tauOneToOne = Number(v); },
    "--tau-one-to-many": (v) => { out.tauOneToMany = Number(v); },
    "--tau-many-to-one": (v) => { out.tauManyToOne = Number(v); },
    "--k": (v) => { out.k = Number(v); },
    "--domain": (v) => { out.domain = v; },
    "--commit": (v) => { out.commit = v; },
    "--set-doc-types": (v) => { out.setDocTypes = v.split(","); },
    "--target-doc-types": (v) => { out.targetDocTypes = v.split(","); },
  };
  // @allowloop: argv is a positional/flag stream; index-coupled consumption of
  // value-flags has no clean map/filter form. Bounded by argv.length.
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg in flags) { flags[arg](argv[++i]); continue; }
    if (arg.startsWith("--")) throw new Error(`Unknown flag: ${arg}`);
    out.dataset = arg;
  }
  // Validation: fail loud on obviously wrong values.
  if (!Number.isFinite(out.threshold) || out.threshold < 0 || out.threshold > 1) {
    throw new Error(`--threshold must be in [0, 1] (got ${out.threshold})`);
  }
  if (!Number.isFinite(out.tauOneToOne) || out.tauOneToOne < 0 || out.tauOneToOne > 1) {
    throw new Error(`--tau-one-to-one must be in [0, 1] (got ${out.tauOneToOne})`);
  }
  if (out.tauOneToMany !== undefined) {
    if (!Number.isFinite(out.tauOneToMany) || out.tauOneToMany < 0 || out.tauOneToMany > 1) {
      throw new Error(`--tau-one-to-many must be in [0, 1] (got ${out.tauOneToMany})`);
    }
  }
  if (out.tauManyToOne !== undefined) {
    if (!Number.isFinite(out.tauManyToOne) || out.tauManyToOne < 0 || out.tauManyToOne > 1) {
      throw new Error(`--tau-many-to-one must be in [0, 1] (got ${out.tauManyToOne})`);
    }
  }
  if (!Number.isFinite(out.k) || out.k < 1) {
    throw new Error(`--k must be a positive integer (got ${out.k})`);
  }
  // POKA-YOKE: tau > threshold = the silent-recall trap (engine rejects this too,
  // but catch it client-side for immediate diagnostics before the HTTP round-trip).
  if (out.tauOneToOne > out.threshold) {
    throw new Error(
      `tau_one_to_one (${out.tauOneToOne}) > threshold (${out.threshold}) — ` +
      "the engine would reject this (silent-recall trap). Widen --threshold or tighten tau.",
    );
  }
  if (out.tauOneToMany !== undefined && out.tauOneToMany > out.threshold) {
    throw new Error(
      `tau_one_to_many (${out.tauOneToMany}) > threshold (${out.threshold}) — silent-recall trap.`,
    );
  }
  if (out.tauManyToOne !== undefined && out.tauManyToOne > out.threshold) {
    throw new Error(
      `tau_many_to_one (${out.tauManyToOne}) > threshold (${out.threshold}) — silent-recall trap.`,
    );
  }
  return out;
}

// ── Fetch live commit from TerminusDB ───────────────────────────────────────

// Extract the short org/db form from a full domain graphspec.
// "admin/abt_buy_e2e/local/branch/main" → "admin/abt_buy_e2e"
// "admin/abt_buy_e2e" → "admin/abt_buy_e2e" (already short)
function shortDomain(domain) {
  const localIdx = domain.indexOf("/local/");
  if (localIdx > 0) return domain.slice(0, localIdx);
  return domain;
}

async function ioFetchHeadCommit(domain) {
  // TerminusDB log API: GET /api/log/<org>/<db>/local/branch/main?count=1
  const tdbUrl = process.env.TDB_URL || "http://localhost:6365";
  const tdbCred = process.env.TDB_CRED || "admin:root";
  const tdbAuth = "Basic " + Buffer.from(tdbCred).toString("base64");
  const dbPath = shortDomain(domain);
  const logUrl = `${tdbUrl}/api/log/${dbPath}/local/branch/main?count=1`;
  const res = await fetch(logUrl, {
    headers: { "Authorization": tdbAuth },
  });
  if (!res.ok) {
    throw new Error(
      `Failed to fetch head commit for ${dbPath}: HTTP ${res.status} (${await res.text()})`,
    );
  }
  const log = await res.json();
  if (!Array.isArray(log) || log.length === 0) {
    throw new Error(`No commits found for domain ${dbPath}`);
  }
  const commitId = log[0].identifier;
  if (typeof commitId !== "string" || commitId === "") {
    throw new Error(`Commit identifier is empty or invalid in log response: ${JSON.stringify(log[0])}`);
  }
  return commitId;
}

// ── Map endpoint response to scorer input ───────────────────────────────────
//
// The endpoint returns full IRIs (e.g. terminusdb:///bench/abt_buy_e2e/Abt/12345).
// The scorer expects raw ids (e.g. "12345"). We strip using idFromIri.
// We also verify side membership as a safety check.

function mapMatchedToScorerFormat(matched) {
  return matched.map((m) => {
    const setId = idFromIri(m.set_id);
    const targetId = idFromIri(m.target_id);
    return {
      setId,
      targetId,
      distance: m.distance,
      stage: m.stage,
    };
  });
}

// ── Main bench function ─────────────────────────────────────────────────────

function log(msg) {
  process.stdout.write(`[bench-resolve] ${msg}\n`);
}

// Ensure domain is in full graphspec form for the engine (org/db/local/branch/main).
function fullDomain(domain) {
  if (domain.includes("/local/")) return domain;
  return `${domain}/local/branch/main`;
}

async function ioBenchResolve(config) {
  const ds = getDataset(config.dataset);
  await ioWaitReady();

  // Determine domain. Prefer config override, else use the E2E domain.
  // The E2E domain "admin/abt_buy_e2e" is the live indexed data product.
  // Normalise to full graphspec (engine requires org/db/local/branch/main).
  const domain = fullDomain(config.domain || process.env.RESOLVE_DOMAIN || "admin/abt_buy_e2e");

  // Determine commit. Prefer config override, else fetch from TerminusDB.
  let commit;
  if (config.commit) {
    commit = config.commit;
    log(`using explicit commit: ${commit}`);
  } else {
    log(`fetching head commit for ${domain} from TerminusDB...`);
    commit = await ioFetchHeadCommit(domain);
    log(`resolved head commit: ${commit}`);
  }

  // Load ground truth (mapping file from the dataset).
  const mapping = ioLoadMapping(ds);
  log(`loaded ground truth: ${mapping.size} mapped set ids`);

  // Verify the snapshot is settled (no pending index fragments).
  const stats = await ioStatistics();
  const pending = stats.pending_index_fragments ?? 0;
  if (pending > 0) {
    throw new Error(
      `Refusing to bench: ${pending} index fragments still pending. ` +
      "Wait for indexing to settle before running the bench.",
    );
  }
  log(`engine snapshot OK: documents=${stats.documents}, pending=${pending}`);

  // The resolve endpoint takes the full domain spec.
  // For the E2E domain, the engine indexes under "admin/abt_buy_e2e" directly.
  const resolveArgs = {
    domain,
    commit,
    setDocTypes: config.setDocTypes,
    targetDocTypes: config.targetDocTypes,
    threshold: config.threshold,
    tauOneToOne: config.tauOneToOne,
    tauOneToMany: config.tauOneToMany,
    tauManyToOne: config.tauManyToOne,
    k: config.k,
  };

  log("calling POST /resolve (single batch call)...");
  log(`  params: threshold=${config.threshold}, tau_1:1=${config.tauOneToOne}, ` +
      `tau_1:M=${config.tauOneToMany ?? "disabled"}, tau_M:1=${config.tauManyToOne ?? "disabled"}, k=${config.k}`);

  const benchStart = process.hrtime.bigint();
  const result = await ioResolve(resolveArgs);
  const benchMs = Number(process.hrtime.bigint() - benchStart) / 1e6;

  log(`/resolve returned: matched=${result.matched.length}, set_only=${result.set_only.length}, target_only=${result.target_only.length}`);
  log(`engine elapsed_ms=${result.stats.elapsed_ms}, bench end-to-end=${benchMs.toFixed(0)}ms`);

  // Map to scorer format (strip IRIs to raw ids).
  const scorerMatched = mapMatchedToScorerFormat(result.matched);

  // Score against ground truth.
  const score = scoreV2(scorerMatched, mapping);

  printScorecard(domain, commit, config, result, score, benchMs);
  return { result, score, benchMs };
}

// ── Scorecard display ───────────────────────────────────────────────────────

function pct(fraction) {
  return (fraction * 100).toFixed(2) + "%";
}

function tauDisplay(value) {
  if (value === null || value === undefined) return "disabled";
  return value.toFixed(3);
}

function printScorecard(domain, commit, config, result, score, benchMs) {
  const s = result.stats;
  const lines = [];
  lines.push("");
  lines.push("============= ENTITY-RESOLUTION — ENGINE /resolve ENDPOINT =============");
  lines.push(`domain             : ${domain}`);
  lines.push(`commit             : ${commit}`);
  lines.push("--- config (sent to engine) ---");
  lines.push(`k (fan-out)        : ${s.k}`);
  lines.push(`threshold (gather) : ${s.threshold}`);
  lines.push(`tau_one_to_one     : ${tauDisplay(s.tau_one_to_one)} (core: mutual-best reciprocal pairs)`);
  lines.push(`tau_one_to_many    : ${tauDisplay(s.tau_one_to_many)} (set-side extras)`);
  lines.push(`tau_many_to_one    : ${tauDisplay(s.tau_many_to_one)} (target-side extras)`);
  lines.push("--- engine statistics ---");
  lines.push(`set_points         : ${s.set_points}`);
  lines.push(`target_points      : ${s.target_points}`);
  lines.push(`edge_count         : ${s.edge_count}`);
  lines.push(`core_count         : ${s.core_count}`);
  lines.push(`set_extra_count    : ${s.set_extra_count}`);
  lines.push(`target_extra_count : ${s.target_extra_count}`);
  lines.push("--- 3-PARTITION OUTPUT ---");
  lines.push(`matched pairs      : ${s.matched_count}`);
  lines.push(`set_only (no match): ${s.set_only_count}`);
  lines.push(`target_only        : ${s.target_only_count}`);
  lines.push("--- per-stage precision ---");
  lines.push(`core pairs         : ${score.counts.corePairs}   precision ${pct(score.core.fraction)} (${score.core.correct}/${score.core.total})`);
  lines.push(`set_extra pairs    : ${score.counts.setExtraPairs}   precision ${pct(score.setExtra.fraction)} (${score.setExtra.correct}/${score.setExtra.total})`);
  lines.push(`target_extra pairs : ${score.counts.targetExtraPairs}   precision ${pct(score.targetExtra.fraction)} (${score.targetExtra.correct}/${score.targetExtra.total})`);
  lines.push("--- HEADLINE: pair-based F1 ---");
  lines.push(`|P| predicted      : ${score.counts.predictedPairsUnique}`);
  lines.push(`|G| gold pairs     : ${score.counts.truePairs}`);
  lines.push(`precision          : ${pct(score.overall.fraction)} (TP ${score.counts.truePositives} / |P| ${score.overall.total})`);
  lines.push(`recall             : ${pct(score.recall.fraction)} (TP ${score.counts.truePositives} / |G| ${score.recall.total})`);
  lines.push(`F1                 : ${pct(score.f1)}`);
  lines.push(`TP / FP / FN       : ${score.counts.truePositives} / ${score.counts.falsePositives} / ${score.counts.falseNegatives}`);
  lines.push("--- wall-clock ---");
  lines.push(`engine elapsed_ms  : ${s.elapsed_ms} ms (server-side gather + resolve)`);
  lines.push(`bench end-to-end   : ${benchMs.toFixed(0)} ms (includes HTTP + JSON parse)`);
  lines.push(`baseline (per-rec) : ~35 min (2,100,000 ms) — sequential /search x 4346 records`);
  lines.push(`speedup factor     : ~${(2100000 / benchMs).toFixed(0)}x`);
  lines.push("--- example wrong pairs (up to 8) ---");
  for (const w of score.wrongExamples) {
    lines.push(`  Set ${w.setId} -> Target ${w.predictedTarget} [${w.stage}]  truth=[${w.truth.join(",")}]`);
  }
  lines.push("==========================================================================");
  lines.push("");
  process.stdout.write(lines.join("\n"));
}

// ── Entry point ─────────────────────────────────────────────────────────────

if (require.main === module) {
  let config;
  try {
    config = parseArgs(process.argv);
  } catch (e) {
    process.stderr.write(`[bench-resolve] ${e.message}\n`);
    process.exit(2);
  }
  ioBenchResolve(config).catch((err) => {
    process.stderr.write(`[bench-resolve] FAILED: ${err.stack || err.message}\n`);
    process.exit(1);
  });
}

module.exports = { ioBenchResolve, ioResolve, parseArgs, mapMatchedToScorerFormat, shortDomain, fullDomain };
