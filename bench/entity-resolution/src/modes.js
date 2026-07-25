"use strict";

// Retrieval-mode adapters: the io* shell that runs the chosen retrieval PRIMITIVE
// against the engine and produces the mode-agnostic candidate maps the pure
// resolver (resolve.js) consumes:
//   abtToBuy : Map<abtId, Array<{ id: buyId, distance }>>  (each Abt's top-K Buy)
//   buyToAbt : Map<buyId, Array<{ id: abtId, distance }>>  (each Buy's top-K Abt)
//
// THREE modes (spec §4 runs the SAME algorithm on whichever primitive supplies
// the cross-NN candidates):
//   search     — per-record /search (engine embeds the query text). v1-style
//                retrieval feeding the v2 algorithm. No engine dependency beyond
//                what is live. True top-K both directions. RUN FIRST.
//   duplicates — bulk cross-NN via /duplicates set/target over STORED vectors.
//                Fast bulk path. TOP-1 per set point (the endpoint takes the
//                single nearest), so it grounds at effective k=1 mutual-NN; the
//                residual is isolated pairs. RUN SECOND.
//   similar    — per-record /similar anchored on the STORED vector, pool scoped to
//                the opposite catalogue. True top-K. The engine still RE-EMBEDS
//                the anchor text (fix pending), so its speed-up is deferred — the
//                mode is wired and correct; its RUN waits on the engine fix. LAST.

const { ioSearch, ioSimilar, ioDuplicates } = require("./engine");
const { idFromIri, isSide } = require("./iri");

function log(msg) {
  process.stdout.write(`[modes] ${msg}\n`);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ── search mode ─────────────────────────────────────────────────────────────
// One /search per record. The query text is the record's RENDERED text; the
// candidate pool is scoped to the opposite side's doc_type so hits straddle.
async function ioGatherSearch(ds, abt, buy, { k, queryDelayMs, searchMode = "vector" }) {
  // The snapshot holds BOTH catalogues; we scope the query SERVER-SIDE to the
  // target catalogue's doc_type so the engine returns exactly the k nearest
  // opposite-side hits — no client over-fetch, smaller per-query reader footprint
  // (this materially reduces the transient Lance FD spike under a long sweep).
  const direction = async (sources, targetDocType) => {
    const out = new Map();
    let done = 0;
    // @allowloop: sequential HTTP queries on a shared engine; one at a time by
    // design (FD pressure + politeness to sibling agents). No FP equivalent.
    for (const rec of sources) {
      const hits = await ioSearch({
        domain: ds.domain,
        commit: ds.commit,
        q: rec.text,
        mode: searchMode,
        count: k,
        docTypes: [targetDocType],
      });
      // POKA-YOKE: we rely on the engine's server-side doc_type scope to return
      // ONLY the target side (the client over-fetch+filter was removed for FD
      // economy). Assert that invariant rather than trust it silently — if the
      // engine scope is ever ignored/regressed, an own-side hit would corrupt
      // precision/recall invisibly. Fail loud the moment a wrong-side id appears.
      const wrongSide = hits.find((h) => !isSide(h.id, targetDocType));
      if (wrongSide !== undefined) {
        throw new Error(
          `search mode: engine doc_type scope leaked a non-${targetDocType} hit ` +
          `(${wrongSide.id}) for query ${rec.id}. The server-side scope is not being ` +
          "honoured — refusing to gather wrong-side candidates. Check the engine /search doc_type filter.",
        );
      }
      const scoped = hits
        .slice(0, k)
        .map((h) => ({ id: idFromIri(h.id), distance: h.distance }));
      out.set(rec.id, scoped);
      done += 1;
      if (done % 100 === 0) log(`  ${targetDocType}-pool queried ${done}/${sources.length}`);
      if (queryDelayMs > 0) await sleep(queryDelayMs);
    }
    return out;
  };

  log(`search mode (${searchMode}): Abt→Buy (${abt.length} queries) then Buy→Abt (${buy.length} queries), top-${k}`);
  const abtToBuy = await direction(abt, "Buy");
  const buyToAbt = await direction(buy, "Abt");
  return { abtToBuy, buyToAbt };
}

// ── similar mode (GATED on engine reuse-stored-vector fix) ──────────────────
// One /similar per record, anchored on the record's STORED vector, pool scoped to
// the opposite doc_type. Identical candidate shape to search; the difference is
// the engine reuses the stored vector instead of re-embedding query text — that
// is the (pending) speed-up. Implemented and correct now; run deferred.
async function ioGatherSimilar(ds, abt, buy, { k, queryDelayMs }) {
  const direction = async (sources, targetDocType) => {
    const out = new Map();
    let done = 0;
    // @allowloop: sequential anchored similarity queries; same rationale as search.
    for (const rec of sources) {
      const hits = await ioSimilar({
        domain: ds.domain,
        commit: ds.commit,
        id: rec.iri,
        count: k,
        docTypes: [targetDocType],
      });
      const scoped = hits
        .filter((h) => isSide(h.id, targetDocType))
        .slice(0, k)
        .map((h) => ({ id: idFromIri(h.id), distance: h.distance }));
      out.set(rec.id, scoped);
      done += 1;
      if (done % 100 === 0) log(`  ${targetDocType}-pool anchored ${done}/${sources.length}`);
      if (queryDelayMs > 0) await sleep(queryDelayMs);
    }
    return out;
  };

  log(`similar mode: Abt→Buy then Buy→Abt, top-${k} (NOTE: engine re-embeds anchor — speed-up pending fix)`);
  const abtToBuy = await direction(abt, "Buy");
  const buyToAbt = await direction(buy, "Abt");
  return { abtToBuy, buyToAbt };
}

// ── duplicates mode ─────────────────────────────────────────────────────────
// Two bulk /duplicates calls — set=Abt target=Buy, then set=Buy target=Abt —
// over STORED vectors. The endpoint emits ONE nearest neighbour per set point, so
// each direction yields a TOP-1 list per record (length 0 or 1). The resolver
// runs unchanged; with top-1 lists it grounds the mutual-nearest pairs (k=1
// behaviour) and the residual is isolated pairs.
async function ioGatherDuplicates(ds, abt, buy, { threshold }) {
  // group shape: { group: [{id}, {id}], distance }. set=Abt target=Buy means each
  // group straddles Abt↔Buy; identify which member is which by namespace.
  const toMap = (groups, sourceSide, targetSide) => {
    const out = new Map();
    for (const g of groups) {
      const members = g.group || [];
      const source = members.find((m) => isSide(m.id, sourceSide));
      const target = members.find((m) => isSide(m.id, targetSide));
      if (source === undefined || target === undefined) continue; // not a straddling pair
      const sId = idFromIri(source.id);
      const tId = idFromIri(target.id);
      if (!out.has(sId)) out.set(sId, []);
      out.get(sId).push({ id: tId, distance: g.distance });
    }
    return out;
  };

  // BATCH the set into chunks so each /duplicates scan keeps a bounded FD
  // footprint. The engine's duplicates scan opens Lance readers per set-point and
  // does NOT release them within a single scan (FD count spikes 20→200+ then the
  // scan aborts with "Too many open files" at ~2173 set points under nofile=1024).
  // Batching the set via doc_id IN-lists confines each scan to `batchSize` points,
  // well under the ceiling, and FDs drain between calls. This is a client-side
  // adaptation to a real engine-side leak in the duplicates scan path (flagged to
  // the engine team) — NOT masking: the scan results are identical, just chunked.
  const batchSize = 100;
  const gatherDirection = async (sources, setDocType, targetDocType) => {
    const allGroups = [];
    let done = 0;
    // @allowloop: chunked sequential bulk scans; index-stride batching has no
    // clean FP form and each call is an io round-trip. Bounded by sources.length.
    for (let i = 0; i < sources.length; i += batchSize) {
      const batchIds = sources.slice(i, i + batchSize).map((r) => r.iri);
      const groups = await ioDuplicates({
        domain: ds.domain,
        commit: ds.commit,
        threshold,
        setDocTypes: [setDocType],
        setDocIds: batchIds,
        targetDocTypes: [targetDocType],
      });
      for (const g of groups) allGroups.push(g);
      done += batchIds.length;
      if (done % 500 === 0 || done >= sources.length) log(`  ${setDocType}→${targetDocType} scanned ${done}/${sources.length}`);
    }
    return allGroups;
  };

  log(`duplicates mode: set=Abt target=Buy then set=Buy target=Abt (bulk, stored vectors, top-1/dir, batched ${batchSize})`);
  const abtGroups = await gatherDirection(abt, "Abt", "Buy");
  const buyGroups = await gatherDirection(buy, "Buy", "Abt");

  const abtToBuy = toMap(abtGroups, "Abt", "Buy");
  const buyToAbt = toMap(buyGroups, "Buy", "Abt");
  // Ensure every source record has an entry (empty list = no ≤τ candidate) so the
  // resolver's unmatched accounting is complete.
  for (const rec of abt) if (!abtToBuy.has(rec.id)) abtToBuy.set(rec.id, []);
  for (const rec of buy) if (!buyToAbt.has(rec.id)) buyToAbt.set(rec.id, []);
  return { abtToBuy, buyToAbt };
}

const MODES = {
  search: ioGatherSearch,
  similar: ioGatherSimilar,
  duplicates: ioGatherDuplicates,
};

async function ioGatherCandidates(mode, ds, abt, buy, opts) {
  const gather = MODES[mode];
  switch (mode) {
    case "search":
    case "similar":
    case "duplicates":
      return gather(ds, abt, buy, opts);
    default:
      // Fail loud on an unknown mode (Coding-best-practices §8 — exhaustive match).
      throw new Error(`Unknown matching mode "${mode}". Known modes: ${Object.keys(MODES).join(", ")}`);
  }
}

module.exports = { ioGatherCandidates, idFromIri, isSide, MODES };
