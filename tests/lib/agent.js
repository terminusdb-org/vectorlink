'use strict'

const superagent = require('superagent')
const crypto = require('node:crypto')
const { expect } = require('chai')

/**
 * Random hex string for unique resource names.
 */
function randomString () {
  return crypto.randomBytes(8).toString('hex')
}

/**
 * TerminusDB API client — wraps superagent with auth and base URL,
 * mirroring the pattern from terminusdb/tests/lib/agent.js.
 */
class TerminusDBAgent {
  constructor (params) {
    params = params || {}
    this.baseUrl = params.baseUrl || process.env.TERMINUSDB_BASE_URL || 'http://localhost:6363'
    this.orgName = params.orgName || process.env.TERMINUSDB_ORG || 'admin'
    this.dbName = params.dbName || 'vl-test-' + randomString()
    this.user = params.user || process.env.TERMINUSDB_USER || 'admin'
    this.password = params.password || process.env.TERMINUSDB_PASSWORD || 'root'

    this.agent = superagent
      .agent()
      .ok((response) => response.status < 500)
      .use((request) => {
        request.url = this.baseUrl + request.url
      })
      .use((request) => {
        request.auth(this.user, this.password)
      })
  }

  createDatabase (params) {
    params = params || {}
    const body = {
      label: params.label || 'Vectorlink Test DB',
      comment: params.comment || 'Test database for vectorlink integration tests',
      schema: params.schema !== undefined ? params.schema : true,
      prefixes: params.prefixes || {
        '@base': 'terminusdb:///data/',
        '@schema': 'terminusdb:///schema#',
      },
    }
    return this.agent
      .post(`/api/db/${this.orgName}/${this.dbName}`)
      .send(body)
  }

  deleteDatabase () {
    return this.agent.delete(`/api/db/${this.orgName}/${this.dbName}`).send({})
  }

  insertSchema (schema) {
    return this.agent
      .post(`/api/document/${this.orgName}/${this.dbName}`)
      .query({ graph_type: 'schema', author: 'test', message: 'insert schema' })
      .send(schema)
  }

  insertDocuments (docs, params) {
    params = params || {}
    const req = this.agent
      .post(`/api/document/${this.orgName}/${this.dbName}`)
      .query({ graph_type: 'instance', author: params.author || 'test', message: params.message || 'insert docs' })
    if (params.fullReplace) {
      req.query({ full_replace: true })
    }
    return req.send(docs)
  }

  replaceDocuments (docs, params) {
    params = params || {}
    return this.agent
      .put(`/api/document/${this.orgName}/${this.dbName}`)
      .query({ graph_type: 'instance', author: params.author || 'test', message: params.message || 'replace docs', create: true })
      .send(docs)
  }

  deleteDocument (id) {
    return this.agent
      .delete(`/api/document/${this.orgName}/${this.dbName}`)
      .query({ graph_type: 'instance', author: 'test', message: 'delete doc', id })
      .send({})
  }

  getDocuments (params) {
    params = params || {}
    const req = this.agent.get(`/api/document/${this.orgName}/${this.dbName}`)
    if (params.type) {
      req.query({ type: params.type })
    }
    if (params.asList) {
      req.query({ as_list: true })
    }
    return req
  }

  getCommitId () {
    return this.agent
      .get(`/api/document/${this.orgName}/${this.dbName}`)
      .query({ as_list: true, count: 0 })
      .then((res) => {
        // The commit ID is in the response headers
        const dataVersion = res.headers['terminusdb-data-version']
        if (dataVersion) {
          // Format: "branch:commit_hash"
          const parts = dataVersion.split(':')
          return parts.length > 1 ? parts[1] : null
        }
        return null
      })
  }

  getBranchCommitId (branch) {
    branch = branch || 'main'
    return this.agent
      .get(`/api/log/${this.orgName}/${this.dbName}/local/branch/${branch}`)
      .query({ count: 1 })
      .then((res) => {
        const log = res.body
        if (Array.isArray(log) && log.length > 0) {
          // The commit ID is in the 'identifier' field
          const identifier = log[0].identifier
          if (identifier) {
            return identifier
          }
        }
        return null
      })
  }

  // Get the full descriptor path for vectorlink domain parameter
  getDescriptorPath () {
    return `${this.orgName}/${this.dbName}/local/branch/main`
  }
}

/**
 * Vectorlink API client — wraps superagent for the vectorlink server.
 */
class VectorlinkAgent {
  constructor (params) {
    params = params || {}
    this.baseUrl = params.baseUrl || process.env.VECTORLINK_BASE_URL || 'http://localhost:8080'
    this.ollamaUrl = params.ollamaUrl || process.env.VECTORLINK_OLLAMA_URL || 'http://localhost:11434'
    this.ollamaModel = params.ollamaModel || process.env.VECTORLINK_OLLAMA_MODEL || 'qwen3-embedding:4b'
    this.ollamaDimensions = params.ollamaDimensions || Number.parseInt(process.env.VECTORLINK_OLLAMA_DIMENSIONS || '1536', 10)
    this.useOllama = params.useOllama || process.env.VECTORLINK_EMBEDDING_PROVIDER === 'ollama'
    this.tdbUser = params.tdbUser || process.env.TERMINUSDB_USER || 'admin'
    this.tdbPassword = params.tdbPassword || process.env.TERMINUSDB_PASSWORD || 'root'
    this.authHeader = 'Basic ' + Buffer.from(this.tdbUser + ':' + this.tdbPassword).toString('base64')

    this.agent = superagent
      .agent()
      .ok((response) => response.status < 500)
      .use((request) => {
        request.url = this.baseUrl + request.url
      })
  }

  _setEmbeddingHeaders (req) {
    req.set('authorization', this.authHeader)
    if (this.useOllama) {
      req.set('VECTORLINK_OLLAMA_URL', this.ollamaUrl)
      req.set('VECTORLINK_OLLAMA_MODEL', this.ollamaModel)
      req.set('VECTORLINK_OLLAMA_DIMENSIONS', String(this.ollamaDimensions))
    } else {
      const apiKey = process.env.OPENAI_KEY || process.env.VECTORLINK_EMBEDDING_API_KEY || 'test-key'
      req.set('VECTORLINK_EMBEDDING_API_KEY', apiKey)
    }
    return req
  }

  startIndex (domain, commit, previousCommit) {
    const params = { domain, commit }
    if (previousCommit) {
      params.previous = previousCommit
    }
    const req = this.agent.get('/index').query(params)
    return this._setEmbeddingHeaders(req)
  }

  _parseJsonBody (res) {
    if (res.text && typeof res.body === 'object' && Object.keys(res.body).length === 0) {
      try { res.body = JSON.parse(res.text) } catch (e) { /* keep as-is */ }
    }
    return res
  }

  checkTask (taskId) {
    return this.agent
      .get('/check')
      .query({ task_id: taskId })
      .then((res) => this._parseJsonBody(res))
  }

  assignIndex (domain, sourceCommit, targetCommit) {
    return this.agent
      .get('/assign')
      .query({ domain, source_commit: sourceCommit, target_commit: targetCommit })
  }

  search (domain, commit, query, count) {
    count = count || 10
    const req = this.agent
      .post('/search')
      .query({ domain, commit, count })
      .type('text/plain')
      .send(query)
    return this._setEmbeddingHeaders(req).then((res) => this._parseJsonBody(res))
  }

  similar (domain, commit, id, count) {
    count = count || 10
    return this.agent.get('/similar').query({ domain, commit, id, count }).then((res) => this._parseJsonBody(res))
  }

  duplicates (domain, commit, threshold) {
    threshold = threshold || 0.01
    return this.agent.get('/duplicates').query({ domain, commit, threshold }).then((res) => this._parseJsonBody(res))
  }

  statistics () {
    return this.agent.get('/statistics')
  }

  /**
   * Poll /check until the task is Complete or Error.
   * Returns a promise that resolves with the final status.
   */
  async waitForTask (taskId, opts) {
    opts = opts || {}
    const maxRetries = opts.maxRetries || 60
    const interval = opts.interval || 1000
    for (let i = 0; i < maxRetries; i++) {
      const res = await this.checkTask(taskId)
      if (res.status === 200) {
        const body = res.body
        if (body.status === 'Complete') {
          return body
        }
        if (body.status === 'Pending') {
          await new Promise((resolve) => setTimeout(resolve, interval))
          continue
        }
      }
      if (res.status >= 400) {
        throw new Error(`Task ${taskId} failed with status ${res.status}: ${res.text}`)
      }
      await new Promise((resolve) => setTimeout(resolve, interval))
    }
    throw new Error(`Task ${taskId} timed out after ${maxRetries} retries`)
  }
}

module.exports = {
  TerminusDBAgent,
  VectorlinkAgent,
  randomString,
}
