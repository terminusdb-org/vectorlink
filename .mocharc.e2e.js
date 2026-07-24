// Mocha config for e2e tests — requires TerminusDB running on port 7373
// and tdb-search running on port 7372. Run via `make test-e2e`.
module.exports = {
  spec: "tests/e2e/**/*.js",
  timeout: 60000,
  recursive: true,
}
