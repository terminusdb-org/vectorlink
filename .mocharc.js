// Mocha config — integration tests only (vectorlink direct).
// E2e tests (tests/e2e/) require TerminusDB running and are run separately
// via `make test-e2e` or .mocharc.e2e.js.
module.exports = {
  spec: "tests/contract/**/*.js",
  timeout: 10000,
  recursive: true,
}
