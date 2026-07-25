"use strict";

// Dataset registry — the single extensibility point.
//
// To add another Leipzig ER dataset (e.g. Amazon-Google, DBLP-ACM):
//   1. Add a new entry here with its zip URL, file names, encodings, the
//      template files for each side, and the mapping column names.
//   2. Drop matching `.hbs` templates in ../templates/.
// No other code changes are required — loader/verifier are dataset-agnostic
// and driven entirely by this config.

const path = require("path");

const TEMPLATES_DIR = path.join(__dirname, "..", "templates");
const DATA_DIR = path.join(__dirname, "..", "data");

// Leipzig CSVs are frequently latin-1 (iso-8859-1), not utf-8. We read every
// file as latin-1 by default: latin-1 maps every byte 1:1 so it never throws on
// odd bytes, and pure-ASCII content (Buy/mapping here) round-trips identically.
// This avoids silent row drops from a utf-8 decode error on a stray byte.
const DEFAULT_ENCODING = "latin1";

const datasets = {
  "abt-buy": {
    name: "abt-buy",
    // Domain in the engine. A dedicated, disposable test domain.
    domain: "admin/bench_abt_buy",
    branch: "main",
    commit: "bench-abt-buy-c1",
    // IRI namespace for pushed corpus documents.
    iriBase: "terminusdb:///bench/abt_buy",
    download: {
      url: "https://dbs.uni-leipzig.de/files/datasets/Abt-Buy.zip",
      zipName: "Abt-Buy.zip",
    },
    dataDir: DATA_DIR,
    // The indexed corpus side (pushed to the engine).
    corpus: {
      side: "Buy",
      file: "Buy.csv",
      encoding: DEFAULT_ENCODING,
      idField: "id",
      template: path.join(TEMPLATES_DIR, "buy.hbs"),
    },
    // The query side (one /search per record).
    query: {
      side: "Abt",
      file: "Abt.csv",
      encoding: DEFAULT_ENCODING,
      idField: "id",
      template: path.join(TEMPLATES_DIR, "abt.hbs"),
    },
    // Ground-truth perfect mapping: query-id -> corpus-id (may be many-to-many).
    mapping: {
      file: "abt_buy_perfectMapping.csv",
      encoding: DEFAULT_ENCODING,
      queryIdColumn: "idAbt",
      corpusIdColumn: "idBuy",
    },
  },
};

// ── v2 reciprocal cross-NN dataset (spec 17 §4) ─────────────────────────────
// Both catalogues are indexed into ONE snapshot in DISTINCT id namespaces
// (.../Abt/<id> and .../Buy/<id>) so a pair's provenance is recoverable from its
// ids alone, and so the engine's set/target scoping (doc_type IN-list) can scope
// each cross-NN direction to the opposite catalogue. The doc_type the engine
// derives from an IRI is its second-to-last path segment (ingest::extract_doc_type),
// i.e. "Abt" / "Buy" here — used directly as the set/target filter value.
datasets["abt-buy-v2"] = {
  name: "abt-buy-v2",
  domain: "admin/bench_abt_buy_v2", // distinct from v1's domain — no collision
  branch: "main",
  commit: "bench-abt-buy-v2-c1",
  iriBase: "terminusdb:///bench/abt_buy_v2",
  download: datasets["abt-buy"].download, // same Leipzig zip + CSVs
  dataDir: DATA_DIR,
  // Both populations are indexed (v1 indexed only Buy). Each side: which CSV,
  // its id field, its v2 (price-free, brand-aligned) template, and the doc_type
  // its IRIs resolve to (= the side label).
  sides: {
    abt: {
      side: "Abt",
      docType: "Abt",
      file: "Abt.csv",
      encoding: DEFAULT_ENCODING,
      idField: "id",
      template: path.join(TEMPLATES_DIR, "abt.v2.hbs"),
    },
    buy: {
      side: "Buy",
      docType: "Buy",
      file: "Buy.csv",
      encoding: DEFAULT_ENCODING,
      idField: "id",
      template: path.join(TEMPLATES_DIR, "buy.v2.hbs"),
    },
  },
  mapping: {
    file: "abt_buy_perfectMapping.csv",
    encoding: DEFAULT_ENCODING,
    queryIdColumn: "idAbt",
    corpusIdColumn: "idBuy",
  },
};

function getDataset(key) {
  const ds = datasets[key];
  if (!ds) {
    const known = Object.keys(datasets).join(", ");
    throw new Error(`Unknown dataset "${key}". Known datasets: ${known}`);
  }
  return ds;
}

module.exports = { datasets, getDataset, DATA_DIR };
