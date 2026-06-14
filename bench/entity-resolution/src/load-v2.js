"use strict";

// v2 loader (spec §4 Step 1): index BOTH catalogues into ONE snapshot, in
// distinct id namespaces (.../Abt/<id>, .../Buy/<id>). Idempotent: DELETE the
// domain first so a re-run never 409s, then push one NDJSON stream of all Abt +
// all Buy Inserted ops, then poll to Complete. Fails loud if the indexed count
// does not equal the pushed count (never score against an incomplete snapshot).

const path = require("path");
const { getDataset } = require("./datasets");
const {
  ioWaitReady,
  ioDeleteDomain,
  ioPushNdjson,
  ioWaitTaskComplete,
  ioStatistics,
} = require("./engine");
const { ioReadCsvObjects } = require("./csv");
const { makeRenderer } = require("./render");

function log(msg) {
  process.stdout.write(`[load-v2] ${msg}\n`);
}

// IRI for a v2 record: terminusdb:///bench/abt_buy_v2/<Side>/<id>.
function sideIri(ds, side, id) {
  return `${ds.iriBase}/${side.side}/${id}`;
}

// Render one side's CSV into { id, raw, text, iri } records. Fails loud on a
// missing id (a dropped/blank id would corrupt provenance + scoring).
function ioRenderSide(ds, side) {
  const file = path.join(ds.dataDir, side.file);
  const rows = ioReadCsvObjects(file, side.encoding);
  const render = makeRenderer(side.template);
  return rows.map((raw) => {
    const id = raw[side.idField];
    if (id === undefined || String(id).trim() === "") {
      throw new Error(`Record in ${side.file} is missing id field "${side.idField}": ${JSON.stringify(raw)}`);
    }
    const cleanId = String(id).trim();
    return { id: cleanId, raw, text: render(raw), iri: sideIri(ds, side, cleanId) };
  });
}

function buildNdjson(records) {
  return records
    .map((rec) => JSON.stringify({ op: "Inserted", id: rec.iri, string: rec.text }))
    .join("\n");
}

async function ioLoadV2(datasetKey) {
  const ds = getDataset(datasetKey);
  if (!ds.sides) {
    throw new Error(`Dataset "${datasetKey}" has no v2 "sides" config; v2 needs both populations.`);
  }

  log(`dataset=${ds.name} domain=${ds.domain} commit=${ds.commit}`);
  log("waiting for engine readiness (index+search)…");
  const ready = await ioWaitReady();
  log(`engine ready: ${JSON.stringify(ready)}`);

  const abt = ioRenderSide(ds, ds.sides.abt);
  const buy = ioRenderSide(ds, ds.sides.buy);
  log(`rendered ${abt.length} Abt + ${buy.length} Buy = ${abt.length + buy.length} records (v2 templates, price-free)`);
  log(`Abt[0]: ${abt[0].iri} -> "${abt[0].text}"`);
  log(`Buy[0]: ${buy[0].iri} -> "${buy[0].text}"`);

  log(`DELETE /domain?domain=${ds.domain} (reset)…`);
  const delStatus = await ioDeleteDomain(ds.domain);
  log(`delete returned HTTP ${delStatus}`);

  const all = [...abt, ...buy];
  const ndjson = buildNdjson(all);
  log(`pushing ${all.length} records as NDJSON Inserted ops…`);
  const taskId = await ioPushNdjson({
    domain: ds.domain,
    branch: ds.branch,
    targetCommit: ds.commit,
    ndjson,
  });
  log(`push accepted, task=${taskId}; polling /check…`);

  const result = await ioWaitTaskComplete(taskId, {
    onProgress: (s) => log(`  indexing ${s.percentage?.toFixed?.(1) ?? "?"}%`),
  });
  log(`indexing Complete: indexed_documents=${result.indexed_documents}`);
  if (result.skipped && result.skipped.length > 0) {
    log(`WARNING: ${result.skipped.length} documents skipped during push:`);
    for (const s of result.skipped) log(`  skipped ${s.id}: ${s.message}`);
  }
  if (result.indexed_documents !== all.length) {
    throw new Error(
      `Indexed ${result.indexed_documents} documents but pushed ${all.length}. ` +
        `Refusing to resolve against an incomplete snapshot.`
    );
  }

  const stats = await ioStatistics();
  log(`engine statistics: ${JSON.stringify(stats)}`);
  return { ds, abt, buy };
}

module.exports = { ioLoadV2, ioRenderSide, buildNdjson, sideIri };
