"use strict";

// Fetch + extract a dataset zip into ./data. Idempotent: skips download if the
// expected CSVs are already present. Fails loud on network or extraction error.

const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");
const { getDataset } = require("./datasets");

function log(msg) {
  process.stdout.write(`[fetch] ${msg}\n`);
}

function expectedFiles(ds) {
  return [ds.corpus.file, ds.query.file, ds.mapping.file];
}

function allPresent(ds) {
  return expectedFiles(ds).every((f) => fs.existsSync(path.join(ds.dataDir, f)));
}

async function ioDownload(url, dest) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Download failed: GET ${url} -> HTTP ${res.status}`);
  }
  const buf = Buffer.from(await res.arrayBuffer());
  fs.writeFileSync(dest, buf);
  return buf.length;
}

// Extract via python3 zipfile (no `unzip` binary on this host). Fail loud.
function ioExtractZip(zipPath, destDir) {
  execFileSync(
    "python3",
    ["-c", `import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])`, zipPath, destDir],
    { stdio: "inherit" }
  );
}

async function ioFetch(datasetKey) {
  const ds = getDataset(datasetKey);
  fs.mkdirSync(ds.dataDir, { recursive: true });

  if (allPresent(ds)) {
    log(`all CSVs already present in ${ds.dataDir} — skipping download`);
    return;
  }

  const zipPath = path.join(ds.dataDir, ds.download.zipName);
  log(`downloading ${ds.download.url} -> ${zipPath}`);
  const bytes = await ioDownload(ds.download.url, zipPath);
  log(`downloaded ${bytes} bytes; extracting…`);
  ioExtractZip(zipPath, ds.dataDir);

  const missing = expectedFiles(ds).filter((f) => !fs.existsSync(path.join(ds.dataDir, f)));
  if (missing.length > 0) {
    throw new Error(`After extraction, missing expected files: ${missing.join(", ")}`);
  }
  log(`extracted: ${expectedFiles(ds).join(", ")}`);
}

if (require.main === module) {
  const key = process.argv[2] || "abt-buy";
  ioFetch(key).catch((err) => {
    process.stderr.write(`[fetch] FAILED: ${err.stack || err.message}\n`);
    process.exit(1);
  });
}

module.exports = { ioFetch };
