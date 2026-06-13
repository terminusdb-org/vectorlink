/**
 * HTTP test agent — mirrors the TerminusDB test infrastructure pattern.
 * Wraps supertest with the admin secret and base URL.
 */

const supertest = require("supertest")

const BASE_URL = process.env.TDB_SEARCH_URL || "http://localhost:8080"
const ADMIN_USER = process.env.TDB_SEARCH_ADMIN_USER || "admin"
const ADMIN_SECRET = process.env.TDB_SEARCH_ADMIN_SECRET || "root"

/**
 * Create an authenticated supertest agent for tdb-search.
 */
function agent () {
  return supertest(BASE_URL)
}

/**
 * Return the Basic auth header value for the default admin credentials.
 */
function authHeader () {
  const encoded = Buffer.from(`${ADMIN_USER}:${ADMIN_SECRET}`).toString("base64")
  return `Basic ${encoded}`
}

/**
 * Return a wrong auth header (for negative tests).
 */
function wrongAuthHeader () {
  const encoded = Buffer.from("admin:wrongsecret").toString("base64")
  return `Basic ${encoded}`
}

module.exports = {
  agent,
  authHeader,
  wrongAuthHeader,
  BASE_URL,
  ADMIN_USER,
  ADMIN_SECRET,
}
