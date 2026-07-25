# Testing — tdb-search

This project adopts the TerminusDB repo's test infrastructure so the two codebases verify the same way and the integration suite can drive a real TerminusDB without reinventing the harness.

---

## What we adopt from TerminusDB (`terminusdb.test/tests/`)

| TerminusDB asset | What it is | How tdb-search uses it |
|------------------|-----------|------------------------|
| **mocha** test runner | `tests/test/*.js`, `.mocharc.js` | The HTTP-contract and integration suites run under mocha — the same runner, conventions, and reporters. |
| **eslint (standard)** | `.eslintrc.js`, `npm run check` (`eslint --ext .js,.json`) | The JS test sources are linted the same way; wired into `make lint`. |
| **`lib/` API helpers** | `agent.js` (HTTP agent), `params.js`, `document.js`, `test-infrastructure.js` | The fixture/push driver and search-assertion helpers are built in the same agent/params style, so a reader of TerminusDB's tests reads ours. |
| **`start-server-and-test`** | boots a server, waits on a health URL, runs the suite | The integration suite boots the `docker compose` stack and waits on readiness (`/health/ready`, the embeddings backend) before asserting — no fixed sleeps. |
| **`.bundle` fixtures** | committed TerminusDB DB bundles | The end-to-end suite seeds a real TerminusDB from a pinned bundle, exactly as TerminusDB's own tests do. |
| **`pr` aggregate target** | one green gate before a PR | `make pr` mirrors this — `lint test docs`. |

## Test layers

- **HTTP contract:** mocha + the `agent` helper against the running engine; asserts the `openapi.yaml` contract (every endpoint, the admin-secret `401` gate, NDJSON `/push` parse).
- **Indexing, history, search modes:** mocha against the real store and real embeddings, using a fake push driver (frozen NDJSON operation streams) for focused control.
- **End-to-end:** `start-server-and-test` boots `docker compose` (terminusdb + tdb-search + embeddings), seeds a `.bundle`, and asserts known search results.
- **Determinism:** runs the suite twice and diffs rankings; golden vectors are enforced strictly only once cross-restart reproducibility is confirmed for the pinned embedding backend.

Engine-internal unit tests (`cargo test`) live in the crate and run via `make test`; the adopted JS infrastructure covers the HTTP-contract and integration layers, where driving a real TerminusDB matters.

## Gates

- `make lint` — must pass before any commit (OpenAPI via Redocly; clippy and eslint as code lands).
- `make test` — unit and integration.
- `make pr` — full pre-PR gate: `lint test docs`. Must be green to open a PR.
