"use strict";

// Loader: idempotent push of the corpus side into the engine.
//   1. DELETE /domain (idempotent 204) — reset so re-runs never 409.
//   2. Render every corpus record and push as one NDJSON Inserted stream.
//   3. Poll /check until Complete; surface skipped docs (never hidden).

const { getDataset } = require("./datasets");
const { ioWaitReady, ioDeleteDomain, ioPushNdjson, ioWaitTaskComplete, ioStatistics } = require("./engine");
const { ioLoadSide, corpusIri } = require("./load-records");

function log(msg) {
  process.stdout.write(`[load] ${msg}\n`);
}

function buildNdjson(ds, corpusRecords) {
  return corpusRecords
    .map((rec) => JSON.stringify({ op: "Inserted", id: corpusIri(ds, rec.id), string: rec.text }))
    .join("\n");
}

async function ioLoad(datasetKey) {
  const ds = getDataset(datasetKey);

  log(`dataset=${ds.name} domain=${ds.domain} commit=${ds.commit}`);
  log("waiting for engine readiness (index+search)…");
  const ready = await ioWaitReady();
  log(`engine ready: ${JSON.stringify(ready)}`);

  const corpusRecords = ioLoadSide(ds, ds.corpus);
  log(`rendered ${corpusRecords.length} ${ds.corpus.side} corpus records`);

  // Reset — idempotent. A repeat DELETE returns 204, so re-running never 409s.
  log(`DELETE /domain?domain=${ds.domain} (reset)…`);
  const delStatus = await ioDeleteDomain(ds.domain);
  log(`delete returned HTTP ${delStatus}`);

  const ndjson = buildNdjson(ds, corpusRecords);
  log(`pushing ${corpusRecords.length} records as NDJSON Inserted ops…`);
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

  // Genchi genbutsu: confirm the corpus count landed.
  if (result.indexed_documents !== corpusRecords.length) {
    throw new Error(
      `Indexed ${result.indexed_documents} documents but pushed ${corpusRecords.length}. ` +
        `Refusing to score against an incomplete corpus.`
    );
  }

  const stats = await ioStatistics();
  log(`engine statistics: ${JSON.stringify(stats)}`);
  return { ds, corpusCount: corpusRecords.length };
}

if (require.main === module) {
  const key = process.argv[2] || "abt-buy";
  ioLoad(key).catch((err) => {
    process.stderr.write(`[load] FAILED: ${err.stack || err.message}\n`);
    process.exit(1);
  });
}

module.exports = { ioLoad, buildNdjson };
