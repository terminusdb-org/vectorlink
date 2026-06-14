"use strict";

// Single entrypoint: load (idempotent push + wait indexed) then verify (score).
// Usage: node src/bench.js [dataset-key]   (default: abt-buy)

const { ioLoad } = require("./load");
const { ioVerify } = require("./verify");

async function ioBench(datasetKey) {
  await ioLoad(datasetKey);
  await ioVerify(datasetKey);
}

if (require.main === module) {
  const key = process.argv[2] || "abt-buy";
  ioBench(key).catch((err) => {
    process.stderr.write(`[bench] FAILED: ${err.stack || err.message}\n`);
    process.exit(1);
  });
}

module.exports = { ioBench };
