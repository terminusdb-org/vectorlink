"use strict";

// CSV -> TerminusDB JSON document converter for the Abt/Buy E2E fixture.
//
// Reads ../data/Abt.csv and ../data/Buy.csv, emits abt-documents.json and
// buy-documents.json suitable for POST to TerminusDB's /api/document endpoint.
//
// Design decision (sentenceCase): TerminusDB's Rust Handlebars renderer
// (src/rust/terminusdb-community/src/template.rs) registers NO custom helpers.
// The bench's `sentenceCase` helper would throw "helper not found" server-side.
// Therefore we PRE-NORMALISE the relevant fields in the converter so that the
// plain Handlebars template (no helpers) produces output matching the bench's v2
// rendering. See README.md section "Helper Decision" for full rationale.

const fs = require("fs");
const path = require("path");

// ─── Reuse the bench's strict CSV parser ───────────────────────────────────
const { ioReadCsvObjects } = require("../src/csv");

// ─── Pure text helpers (ported from ../src/text.js) ────────────────────────
// We inline the logic rather than requiring the bench module, because the e2e
// fixture must be self-documenting and its dependencies explicit.

const ALL_CAPS_ALPHA = /^[A-Z][A-Z]*$/;

function deUppercaseWord(word) {
  return word.charAt(0) + word.slice(1).toLowerCase();
}

function normaliseToken(token) {
  return ALL_CAPS_ALPHA.test(token) ? deUppercaseWord(token) : token;
}

// Sentence-case: de-allcaps purely-alphabetic uppercase words, leave mixed-case
// and alphanumeric model numbers (EZXS88W, PSLX350H) verbatim.
function sentenceCase(value) {
  if (value === undefined || value === null) return "";
  const text = String(value).trim();
  if (text === "") return "";
  return text
    .split(/\s+/)
    .map(normaliseToken)
    .join(" ");
}

// ─── Price normalisation (from ../src/render.js) ───────────────────────────
// Strip "$", ",", and a trailing ".00"/".0". Kept as a stored field for
// reporting but EXCLUDED from the embedding template (per spec 17 section 3).
function normalisePrice(raw) {
  if (raw === undefined || raw === null) return "";
  const trimmed = String(raw).trim();
  if (trimmed === "") return "";
  const noCurrency = trimmed.replace(/[$,]/g, "");
  const asNumber = Number(noCurrency);
  if (!Number.isFinite(asNumber)) {
    throw new Error(`Unparseable price value: ${JSON.stringify(raw)}`);
  }
  return noCurrency.replace(/\.0+$/, "");
}

// ─── Document builders (pure) ──────────────────────────────────────────────

function buildAbtDocument(row) {
  const recordId = String(row.id).trim();
  if (recordId === "") {
    throw new Error(`Abt row missing id: ${JSON.stringify(row)}`);
  }
  const name = sentenceCase(row.name);
  const description = String(row.description || "").trim();
  const price = normalisePrice(row.price);

  const doc = {
    "@type": "Abt",
    "record_id": recordId,
    "name": name,
    "description": description
  };
  if (price !== "") {
    doc.price = price;
  }
  return doc;
}

function buildBuyDocument(row) {
  const recordId = String(row.id).trim();
  if (recordId === "") {
    throw new Error(`Buy row missing id: ${JSON.stringify(row)}`);
  }
  const manufacturer = sentenceCase(row.manufacturer);
  const name = sentenceCase(row.name);
  const description = String(row.description || "").trim();
  const price = normalisePrice(row.price);

  const doc = {
    "@type": "Buy",
    "record_id": recordId,
    "name": name,
    "description": description
  };
  if (manufacturer !== "") {
    doc.manufacturer = manufacturer;
  }
  if (price !== "") {
    doc.price = price;
  }
  return doc;
}

// ─── IO edge: read CSVs, write JSON ───────────────────────────────────────

function ioConvert() {
  const dataDir = path.resolve(__dirname, "..", "data");
  const outDir = __dirname;

  const abtPath = path.join(dataDir, "Abt.csv");
  const buyPath = path.join(dataDir, "Buy.csv");

  if (!fs.existsSync(abtPath)) {
    throw new Error(`Abt.csv not found at ${abtPath}`);
  }
  if (!fs.existsSync(buyPath)) {
    throw new Error(`Buy.csv not found at ${buyPath}`);
  }

  // Leipzig CSVs are latin-1 encoded.
  const abtRows = ioReadCsvObjects(abtPath, "latin1");
  const buyRows = ioReadCsvObjects(buyPath, "latin1");

  const abtDocs = abtRows.map(buildAbtDocument);
  const buyDocs = buyRows.map(buildBuyDocument);

  const abtOut = path.join(outDir, "abt-documents.json");
  const buyOut = path.join(outDir, "buy-documents.json");

  fs.writeFileSync(abtOut, JSON.stringify(abtDocs, null, 2), "utf-8");
  fs.writeFileSync(buyOut, JSON.stringify(buyDocs, null, 2), "utf-8");

  process.stdout.write(`[convert] Wrote ${abtDocs.length} Abt documents to ${abtOut}\n`);
  process.stdout.write(`[convert] Wrote ${buyDocs.length} Buy documents to ${buyOut}\n`);

  // Print samples for verification.
  if (abtDocs.length > 0) {
    process.stdout.write(`[convert] Abt sample: ${JSON.stringify(abtDocs[0])}\n`);
  }
  if (buyDocs.length > 0) {
    process.stdout.write(`[convert] Buy sample: ${JSON.stringify(buyDocs[0])}\n`);
  }
}

ioConvert();
