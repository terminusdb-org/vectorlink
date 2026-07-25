// ESLint config — matches TerminusDB test conventions (standard style).
module.exports = {
  extends: "standard",
  env: {
    mocha: true,
    node: true,
  },
  rules: {
    // Allow double quotes (project convention).
    quotes: ["error", "double"],
    // Comma dangle for cleaner diffs.
    "comma-dangle": ["error", "always-multiline"],
    // Allow unused vars prefixed with underscore (common in test assertions).
    "no-unused-vars": ["error", { argsIgnorePattern: "^_" }],
  },
}
