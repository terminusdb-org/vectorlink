"use strict";

// Shared IRI helpers — the single source of truth for stripping an engine IRI
// back to its raw record id and for testing namespace membership. Used by both
// the v2 modes (modes.js) and the v1 verifier (verify.js) so the two paths can
// never diverge on how an id is parsed out of an IRI (a divergence would surface
// only as a silent accuracy drop in whichever path was missed). Pure.

// Strip an IRI back to its raw id (last "/"-delimited segment), e.g.
//   terminusdb:///bench/abt_buy/Buy/10011646 -> "10011646".
function idFromIri(iri) {
  const parts = iri.split("/");
  return parts[parts.length - 1];
}

// True iff the IRI belongs to the given side namespace (.../<Side>/<id>).
function isSide(iri, sideLabel) {
  return iri.includes(`/${sideLabel}/`);
}

module.exports = { idFromIri, isSide };
