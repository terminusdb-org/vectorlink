"use strict";

// Load + render the two dataset sides and the ground-truth mapping.
// Pure-ish: reads files (io) then renders (pure). Fails loud on parse problems.

const path = require("path");
const { ioReadCsvObjects } = require("./csv");
const { makeRenderer } = require("./render");

// Build the corpus IRI for a record id, e.g.
// terminusdb:///bench/abt_buy/Buy/10011646
function corpusIri(ds, id) {
  return `${ds.iriBase}/${ds.corpus.side}/${id}`;
}

function ioLoadSide(ds, sideConfig) {
  const file = path.join(ds.dataDir, sideConfig.file);
  const rows = ioReadCsvObjects(file, sideConfig.encoding);
  const render = makeRenderer(sideConfig.template);
  return rows.map((raw) => {
    const id = raw[sideConfig.idField];
    if (id === undefined || String(id).trim() === "") {
      throw new Error(`Record in ${sideConfig.file} is missing id field "${sideConfig.idField}": ${JSON.stringify(raw)}`);
    }
    return { id: String(id).trim(), raw, text: render(raw) };
  });
}

// Ground truth as Map<queryId, Set<corpusId>> (handles many-to-many mappings).
function ioLoadMapping(ds) {
  const file = path.join(ds.dataDir, ds.mapping.file);
  const rows = ioReadCsvObjects(file, ds.mapping.encoding);
  const map = new Map();
  for (const row of rows) {
    const qId = String(row[ds.mapping.queryIdColumn]).trim();
    const cId = String(row[ds.mapping.corpusIdColumn]).trim();
    if (qId === "" || cId === "") {
      throw new Error(`Mapping row has empty id: ${JSON.stringify(row)}`);
    }
    if (!map.has(qId)) map.set(qId, new Set());
    map.get(qId).add(cId);
  }
  return map;
}

module.exports = { ioLoadSide, ioLoadMapping, corpusIri };
