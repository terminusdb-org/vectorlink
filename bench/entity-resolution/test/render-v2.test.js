"use strict";

const { test } = require("node:test");
const assert = require("node:assert/strict");
const path = require("path");
const { makeRenderer } = require("../src/render");

const TEMPLATES = path.join(__dirname, "..", "templates");
const renderAbt = makeRenderer(path.join(TEMPLATES, "abt.v2.hbs"));
const renderBuy = makeRenderer(path.join(TEMPLATES, "buy.v2.hbs"));

test("Buy: vendor leads, sentence-cased; price absent (refinements A + E)", () => {
  const out = renderBuy({
    name: "Linksys EtherFast EZXS88W Ethernet Switch - EZXS88W",
    description: "Linksys EtherFast 8-Port 10/100 Switch (New/Workgroup)",
    manufacturer: "LINKSYS",
    price: "$99.00",
  });
  // manufacturer de-allcapsed to "Linksys" and at the FRONT; model code intact;
  // price nowhere in the string.
  assert.ok(out.startsWith("Linksys."), `expected vendor-front, got: ${out}`);
  assert.ok(out.includes("EZXS88W"), "model number preserved");
  assert.ok(!out.includes("99"), "price must NOT appear in the embedding");
  assert.ok(!/manufacturer is/i.test(out), "no mid-string manufacturer clause");
});

test("Buy: omits the manufacturer prefix when manufacturer is empty", () => {
  const out = renderBuy({ name: "Generic Cable", description: "", manufacturer: "" });
  assert.equal(out, "Generic Cable.");
});

test("Buy: omits description clause when empty", () => {
  const out = renderBuy({ name: "Sony PSLX350H Turntable", manufacturer: "Sony", description: "" });
  assert.equal(out, "Sony. Sony PSLX350H Turntable.");
});

test("Abt: name sentence-cased, no price, description appended", () => {
  const out = renderAbt({
    name: "Sony Turntable - PSLX350H",
    description: "Belt Drive System 33-1/3 and 45 RPM",
    price: "$399.00",
  });
  assert.ok(out.includes("Sony Turntable - PSLX350H"), "name preserved/sentence-cased");
  assert.ok(out.includes("Belt Drive System"), "description present");
  assert.ok(!out.includes("399"), "price must NOT appear");
});

test("Abt: all-caps brand in name is de-allcapsed for cross-catalogue alignment", () => {
  const out = renderAbt({ name: "LINKSYS EZXS88W Switch", description: "8 port" });
  assert.ok(out.startsWith("Linksys EZXS88W Switch."), `got: ${out}`);
});
