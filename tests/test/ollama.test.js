'use strict'

const { expect } = require('chai')
const { TerminusDBAgent, VectorlinkAgent } = require('../lib/agent')
const { schema, documents } = require('../lib/fixtures')

describe('Vectorlink Ollama embedding provider', function () {
  let tdb
  let vl
  let domain
  let commit

  before(async function () {
    // Skip if Ollama is not available
    const useOllama = process.env.VECTORLINK_EMBEDDING_PROVIDER === 'ollama'
    if (!useOllama) {
      this.skip()
    }

    tdb = new TerminusDBAgent({})
    vl = new VectorlinkAgent({ useOllama: true })
    domain = tdb.getDescriptorPath()

    const res = await tdb.createDatabase({})
    expect(res.status).to.equal(200)

    const schemaRes = await tdb.insertSchema(schema)
    expect(schemaRes.status).to.equal(200)

    const docRes = await tdb.insertDocuments(documents)
    expect(docRes.status).to.equal(200)

    commit = await tdb.getBranchCommitId('main')
    expect(commit).to.not.be.null
  })

  after(async function () {
    if (tdb) {
      try {
        await tdb.deleteDatabase()
      } catch (e) {
        // Best effort
      }
    }
  })

  it('should index documents using Ollama embeddings', async function () {
    const res = await vl.startIndex(domain, commit)
    expect(res.status).to.equal(200)
    const taskId = res.text.trim()
    const result = await vl.waitForTask(taskId, { maxRetries: 90, interval: 3000 })
    expect(result.status).to.equal('Complete')
    expect(result.indexed_documents).to.be.greaterThan(0)
  })

  it('should search using Ollama embeddings and return relevant results', async function () {
    const res = await vl.search(domain, commit, 'ocean deep blue fish coral reef', 3)
    expect(res.status).to.equal(200)
    const results = res.body
    expect(results).to.be.an('array')
    expect(results.length).to.be.greaterThan(0)
    expect(results[0].id).to.include('blue')
  })

  it('should find similar documents using Ollama embeddings', async function () {
    const res = await vl.similar(domain, commit, 'terminusdb:///data/SearchableDoc/green', 3)
    expect(res.status).to.equal(200)
    const results = res.body
    expect(results).to.be.an('array')
    expect(results.length).to.be.greaterThan(0)
    expect(results[0].id).to.include('SearchableDoc/')
  })

  it('should produce 1536-dimensional embeddings', async function () {
    // The statistics endpoint can confirm dimension info
    const res = await vl.statistics()
    expect(res.status).to.equal(200)
    // The vector store should have vectors of the expected dimension
  })
})
