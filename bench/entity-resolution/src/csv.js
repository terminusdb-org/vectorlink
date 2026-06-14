"use strict";

// Minimal, strict RFC-4180-ish CSV parser.
//
// Why hand-rolled: the bench must FAIL LOUD on a malformed row rather than
// silently drop it (a dropped row skews the accuracy number). A dependency that
// "best-efforts" past bad rows is the wrong tool here. This parser throws on any
// structural inconsistency (unterminated quote, wrong column count).

const fs = require("fs");

// Parse one CSV document (already decoded to a JS string) into rows of fields.
// Pure: string in, array-of-arrays out. Throws on structural error.
function parseCsv(text) {
  const rows = [];
  let field = "";
  let row = [];
  let inQuotes = false;
  let i = 0;
  const n = text.length;
  let started = false; // whether the current row has any content yet

  const pushField = () => {
    row.push(field);
    field = "";
  };
  const pushRow = () => {
    pushField();
    rows.push(row);
    row = [];
    started = false;
  };

  while (i < n) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i += 2;
          continue;
        }
        inQuotes = false;
        i += 1;
        continue;
      }
      field += c;
      i += 1;
      continue;
    }
    if (c === '"') {
      if (field.length !== 0) {
        throw new Error(`CSV parse error at offset ${i}: quote in the middle of an unquoted field`);
      }
      inQuotes = true;
      started = true;
      i += 1;
      continue;
    }
    if (c === ",") {
      pushField();
      started = true;
      i += 1;
      continue;
    }
    if (c === "\r") {
      // Normalise CRLF / lone CR to a single row break.
      pushRow();
      i += text[i + 1] === "\n" ? 2 : 1;
      continue;
    }
    if (c === "\n") {
      pushRow();
      i += 1;
      continue;
    }
    field += c;
    started = true;
    i += 1;
  }

  if (inQuotes) {
    throw new Error("CSV parse error: unterminated quoted field at end of file");
  }
  // Flush a trailing row that did not end with a newline.
  if (started || field.length !== 0 || row.length !== 0) {
    pushRow();
  }
  return rows;
}

// Read a CSV file from disk with the given encoding and return an array of
// objects keyed by the header row. Fails loud on column-count mismatch.
function ioReadCsvObjects(filePath, encoding) {
  const buf = fs.readFileSync(filePath);
  // latin1 decode never throws and round-trips ASCII; explicit so odd bytes in
  // Leipzig files don't blow up or get replaced silently.
  const text = buf.toString(encoding);
  const rows = parseCsv(text);
  if (rows.length === 0) {
    throw new Error(`CSV file ${filePath} is empty (no header row)`);
  }
  const header = rows[0];
  const dataRows = rows.slice(1);

  return dataRows.map((cells, idx) => {
    // A single empty trailing cell from a final newline is the empty document — skip it.
    if (cells.length === 1 && cells[0] === "") {
      return null;
    }
    if (cells.length !== header.length) {
      throw new Error(
        `CSV parse error in ${filePath}: data row ${idx + 2} has ${cells.length} ` +
          `columns, expected ${header.length} (header: ${header.join(",")}). ` +
          `Row: ${JSON.stringify(cells)}`
      );
    }
    return header.reduce((obj, key, col) => {
      obj[key] = cells[col];
      return obj;
    }, {});
  }).filter((r) => r !== null);
}

module.exports = { parseCsv, ioReadCsvObjects };
