"use strict";

// Thin HTTP client for the tdb-search engine. All functions are io* — they talk
// to the network. They fail loud: a non-2xx response throws with the body.

const ENGINE_URL = process.env.ENGINE_URL || "http://localhost:8081";
const ENGINE_CRED = process.env.ENGINE_CRED || "admin:root";

const authHeader = "Basic " + Buffer.from(ENGINE_CRED).toString("base64");

async function ioRequest(method, pathAndQuery, { body, contentType, accept } = {}) {
  const url = `${ENGINE_URL}${pathAndQuery}`;
  const headers = { Authorization: authHeader };
  if (contentType) headers["Content-Type"] = contentType;
  if (accept) headers["Accept"] = accept;
  const res = await fetch(url, { method, headers, body });
  const text = await res.text();
  if (!res.ok) {
    throw new Error(`${method} ${pathAndQuery} -> HTTP ${res.status}: ${text.slice(0, 500)}`);
  }
  return { status: res.status, text };
}

async function ioHealthReady() {
  const { text } = await ioRequest("GET", "/health/ready");
  return JSON.parse(text);
}

async function ioWaitReady({ timeoutMs = 120000, intervalMs = 2000 } = {}) {
  const deadline = Date.now() + timeoutMs;
  // @allowloop: a readiness poll has no functional equivalent; bounded by deadline.
  while (Date.now() < deadline) {
    try {
      const r = await ioHealthReady();
      if (r.index && r.search) return r;
    } catch (e) {
      // WHY: the engine may be mid-rebuild and briefly unreachable (shared :8081).
      // INVARIANT: the deadline below bounds the wait; we never loop forever.
      // CONSEQUENCE: if it never becomes ready, we throw after timeout (fail loud).
      void e;
    }
    await sleep(intervalMs);
  }
  throw new Error(`Engine not ready (index+search) within ${timeoutMs}ms at ${ENGINE_URL}`);
}

async function ioDeleteDomain(domain) {
  const { status } = await ioRequest("DELETE", `/domain?domain=${encodeURIComponent(domain)}`);
  return status; // 204 expected; idempotent
}

async function ioPushNdjson({ domain, branch, targetCommit, ndjson }) {
  const q = `domain=${encodeURIComponent(domain)}&branch=${encodeURIComponent(branch)}&target_commit=${encodeURIComponent(targetCommit)}`;
  const { text } = await ioRequest("POST", `/push?${q}`, {
    body: ndjson,
    contentType: "application/x-ndjson",
  });
  return text.trim(); // task id
}

async function ioCheck(taskId) {
  const { text } = await ioRequest("GET", `/check?task_id=${encodeURIComponent(taskId)}`);
  return JSON.parse(text);
}

async function ioWaitTaskComplete(taskId, { timeoutMs = 600000, intervalMs = 2000, onProgress } = {}) {
  const deadline = Date.now() + timeoutMs;
  // @allowloop: async task poll; bounded by deadline.
  while (Date.now() < deadline) {
    const status = await ioCheck(taskId);
    if (status.status === "Complete") return status;
    if (onProgress) onProgress(status);
    await sleep(intervalMs);
  }
  throw new Error(`Index task ${taskId} did not complete within ${timeoutMs}ms`);
}

// A transient engine-side error under a tight query loop: the Lance store can
// hit the process file-descriptor limit ("Too many open files", os error 24)
// faster than it releases FDs. It is NOT a query defect — the same query
// succeeds moments later (verified: engine stays ready, FDs drain). We retry it
// with backoff so the bench can complete a full 1081-query sweep on a shared
// engine, while any OTHER 500 (a real systemic error) still fails loud.
function isTransientFdPressure(message) {
  return /Too many open files|os error 24/.test(message);
}

async function ioSearch({ domain, commit, q, mode, count, start, docTypes = [] }, { maxRetries = 8 } = {}) {
  const body = JSON.stringify({
    domain,
    commit,
    q,
    ...(mode ? { mode } : {}),
    ...(count !== undefined ? { count } : {}),
    ...(start !== undefined ? { start } : {}),
    // Server-side scope to a target population (e.g. ["Buy"]) so we fetch exactly
    // `count` opposite-side hits — no client over-fetch, smaller per-query reader
    // footprint (reduces the transient Lance FD spike under a long sweep).
    ...(docTypes.length > 0 ? { doc_type: docTypes } : {}),
  });
  let attempt = 0;
  // @allowloop: bounded retry of a transient engine back-pressure error.
  for (;;) {
    try {
      const { text } = await ioRequest("POST", "/search", {
        body,
        contentType: "application/json",
        accept: "application/json",
      });
      return JSON.parse(text);
    } catch (e) {
      // WHY: retry ONLY the transient FD-pressure 500 (Lance "too many open
      //   files") that a tight sequential search loop induces on a shared engine.
      // INVARIANT: the same query succeeds after the engine drains FDs (observed
      //   directly); maxRetries bounds the attempts so we never loop forever.
      // CONSEQUENCE: if it is any OTHER error, or persists past maxRetries, it is
      //   re-thrown unchanged — a genuine defect still fails loud, no masking.
      if (!isTransientFdPressure(e.message) || attempt >= maxRetries) throw e;
      attempt += 1;
      await sleep(500 * attempt); // linear backoff: 0.5s, 1s, 1.5s, …
    }
  }
}

// Per-record anchored similarity over a STORED vector (the `similar` mode).
// Scopes the candidate pool to a target population via doc_type (e.g. "Buy").
// Returns ranked [{ id, distance }] nearest-first.
// NOTE: the engine currently RE-EMBEDS the anchor's source text rather than
// reusing its stored vector (service/mod.rs) — so this mode is correct but its
// headline speed-up is pending the engine fix. Wired now, run deferred.
async function ioSimilar({ domain, commit, id, count, docTypes = [] }, { maxRetries = 5 } = {}) {
  const body = JSON.stringify({
    domain,
    commit,
    id,
    ...(count !== undefined ? { count } : {}),
    ...(docTypes.length > 0 ? { doc_type: docTypes } : {}),
  });
  let attempt = 0;
  // @allowloop: bounded retry of the same transient FD-pressure 500 as /search.
  for (;;) {
    try {
      const { text } = await ioRequest("POST", "/similar", {
        body,
        contentType: "application/json",
        accept: "application/json",
      });
      return JSON.parse(text);
    } catch (e) {
      // WHY/INVARIANT/CONSEQUENCE: identical to ioSearch's transient-FD retry —
      // only the Lance "too many open files" 500 is retried, bounded by
      // maxRetries; any other error or exhaustion re-throws unchanged (fail loud).
      if (!isTransientFdPressure(e.message) || attempt >= maxRetries) throw e;
      attempt += 1;
      await sleep(500 * attempt);
    }
  }
}

// Bulk cross-set near-duplicates over STORED vectors (the `duplicates` mode).
// set = one catalogue (doc_type), target = the other. Returns the engine's group
// shape [{ group: [{id}, ...], distance }]. ONE nearest neighbour per set point
// (the endpoint takes the single nearest of an over-fetched k=8), so this is a
// TOP-1-per-direction cross-NN — not top-K (see README v2 "mode caveats").
async function ioDuplicates({ domain, commit, threshold, setDocTypes = [], setDocIds = [], targetDocTypes = [], count }) {
  const params = new URLSearchParams();
  params.set("domain", domain);
  params.set("commit", commit);
  if (threshold !== undefined) params.set("threshold", String(threshold));
  if (count !== undefined) params.set("count", String(count));
  for (const t of setDocTypes) params.append("doc_type", t);
  // setDocIds narrows the SET population to specific ids — used to BATCH a large
  // set across several calls (the duplicates scan opens FDs per set-point and
  // leaks them within one scan at scale; small batches keep each scan's FD
  // footprint bounded — see README v2 "duplicates engine FD finding").
  for (const id of setDocIds) params.append("doc_id", id);
  for (const t of targetDocTypes) params.append("target_doc_type", t);
  const { text } = await ioRequest("GET", `/duplicates?${params.toString()}`, {
    accept: "application/json",
  });
  return JSON.parse(text);
}

async function ioStatistics() {
  const { text } = await ioRequest("GET", "/statistics");
  return JSON.parse(text);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

module.exports = {
  ENGINE_URL,
  ioHealthReady,
  ioWaitReady,
  ioDeleteDomain,
  ioPushNdjson,
  ioCheck,
  ioWaitTaskComplete,
  ioSearch,
  ioSimilar,
  ioDuplicates,
  ioStatistics,
};
