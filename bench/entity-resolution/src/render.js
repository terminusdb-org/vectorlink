"use strict";

// Handlebars renderer for record -> embedding text.
//
// Uses the real `handlebars` npm package (see package.json). Templates are
// editable `.hbs` files under ../templates/. Optional clauses are guarded by
// `{{#if field}}` so an empty field omits its whole clause, exactly per spec.

const fs = require("fs");
const Handlebars = require("handlebars");
const { sentenceCase } = require("./text");

// Register the v2 brand-alignment helper (refinement E). Idempotent — registering
// the same name twice just overwrites with the same function. Pure helper: it
// performs no I/O, only string normalisation (see src/text.js).
Handlebars.registerHelper("sentenceCase", (value) => sentenceCase(value));

// The raw Leipzig price fields carry a leading "$" and trailing ".00"
// (e.g. "$399.00"). The spec templates render `${{price}}` and the spec's
// worked example shows "$399" — i.e. a single dollar sign and no trailing
// zeros. So we normalise the raw price into the bare number the template
// expects: strip "$" and "," and a trailing ".00"/".0". A price that is empty
// or non-numeric after stripping is treated as ABSENT (the {{#if}} guard then
// omits the clause / renders the "no price" branch). Fail-soft only on the
// price FORMAT — never on a row.
function normalisePrice(raw) {
  if (raw === undefined || raw === null) return "";
  const trimmed = String(raw).trim();
  if (trimmed === "") return "";
  const noCurrency = trimmed.replace(/[$,]/g, "");
  const asNumber = Number(noCurrency);
  if (!Number.isFinite(asNumber)) {
    // Unexpected price shape — surface it rather than silently embed garbage.
    throw new Error(`Unparseable price value: ${JSON.stringify(raw)}`);
  }
  // Drop a pure ".00"/".0" cents tail to match the spec example ("$399").
  return noCurrency.replace(/\.0+$/, "");
}

// Normalise a raw CSV record into the template's field shape. Empty strings
// become "" so Handlebars `{{#if}}` treats them as falsy and omits the clause.
function normaliseRecord(raw) {
  const fields = {};
  for (const key of Object.keys(raw)) {
    const value = raw[key] === undefined || raw[key] === null ? "" : String(raw[key]).trim();
    fields[key] = value;
  }
  if ("price" in fields) {
    fields.price = normalisePrice(fields.price);
  }
  return fields;
}

function ioCompileTemplate(templatePath) {
  const source = fs.readFileSync(templatePath, "utf-8");
  return Handlebars.compile(source, { noEscape: true, strict: false });
}

// Build a render function for one dataset side from its compiled template.
function makeRenderer(templatePath) {
  const template = ioCompileTemplate(templatePath);
  return (rawRecord) => template(normaliseRecord(rawRecord)).trim();
}

module.exports = { normalisePrice, normaliseRecord, ioCompileTemplate, makeRenderer };
