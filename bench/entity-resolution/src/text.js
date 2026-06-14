"use strict";

// Pure text-normalisation helpers for the v2 rendering (spec §3, refinement E).
//
// The goal of `sentenceCase` is brand alignment across the two catalogues: Buy
// carries the vendor in a separate field, frequently ALL-CAPS (e.g. "LINKSYS"),
// while Abt embeds the brand mixed-case at the front of `name`
// (e.g. "Linksys EtherFast..."). De-uppercasing an all-caps BRAND word makes the
// shared identity token ("Linksys") match lexically across both renderings.
//
// CRITICAL design choice — we de-allcaps ONLY purely-alphabetic all-caps tokens
// (brand words). We deliberately DO NOT touch:
//   - mixed-case tokens ("EtherFast")            — already aligned, leave as-is.
//   - alphanumeric tokens ("EZXS88W", "PSLX350H") — MODEL NUMBERS; their exact
//     casing is identity-bearing and must survive verbatim. Lowercasing them
//     would destroy the strongest discriminative token (spec §3: "lead with
//     brand + model number ... consistently across populations").
// This is the "capitalize-first / de-uppercase" interpretation the task locked
// (NOT full-lowercase-everything), chosen to match the mixed-case name field.

const ALL_CAPS_ALPHA = /^[A-Z][A-Z]*$/; // one-or-more letters, all uppercase, NO digits

// Title-case a single all-caps alphabetic word: "LINKSYS" -> "Linksys".
function deUppercaseWord(word) {
  return word.charAt(0) + word.slice(1).toLowerCase();
}

// Normalise a token: de-allcaps a brand word, leave everything else verbatim.
function normaliseToken(token) {
  return ALL_CAPS_ALPHA.test(token) ? deUppercaseWord(token) : token;
}

// Sentence-case a string for brand alignment: split on whitespace, de-allcaps
// each purely-alphabetic all-caps word, preserve all other tokens (mixed case,
// model numbers, punctuation-bearing tokens) and the original single-spacing.
// Pure: string in, string out. An empty / non-string input yields "".
function sentenceCase(value) {
  if (value === undefined || value === null) return "";
  const text = String(value).trim();
  if (text === "") return "";
  return text
    .split(/\s+/)
    .map((token) => normaliseToken(token))
    .join(" ");
}

module.exports = { sentenceCase, normaliseToken, deUppercaseWord };
