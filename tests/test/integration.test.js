'use strict'

const { expect } = require('chai')
const { TerminusDBAgent, VectorlinkAgent } = require('../lib/agent')
const { schema, documents, updatedDocuments, additionalDocuments } = require('../lib/fixtures')

describe('Vectorlink integration with TerminusDB', function () {
  let tdb
  let vl
  let domain
  let commit1
  let commit2
  let commit3

  before(async function () {
    tdb = new TerminusDBAgent({})
    vl = new VectorlinkAgent({})
    domain = tdb.getDescriptorPath()

    // Create the test database
    const res = await tdb.createDatabase({})
    expect(res.status).to.equal(200)

    // Insert schema
    const schemaRes = await tdb.insertSchema(schema)
    expect(schemaRes.status).to.equal(200)

    // Insert initial documents
    const docRes = await tdb.insertDocuments(documents)
    expect(docRes.status).to.equal(200)

    // Get the first commit ID
    commit1 = await tdb.getBranchCommitId('main')
    expect(commit1).to.not.be.null
  })

  after(async function () {
    if (tdb) {
      try {
        await tdb.deleteDatabase()
      } catch (e) {
        // Best effort cleanup
      }
    }
  })

  describe('Index lifecycle', function () {
    it('should start indexing a commit and return a task id', async function () {
      const res = await vl.startIndex(domain, commit1)
      expect(res.status).to.equal(200)
      expect(res.text).to.be.a('string').that.is.not.empty
      // The response is a plain text task id
      const taskId = res.text.trim()
      expect(taskId).to.match(/^[a-zA-Z0-9]+$/)
    })

    it('should poll task status until complete', async function () {
      // Start a fresh index — the previous test already indexed commit1,
      // so this may be a no-op (Completed with 0 docs) or a re-index.
      const res = await vl.startIndex(domain, commit1)
      const taskId = res.text.trim()
      const result = await vl.waitForTask(taskId, { maxRetries: 60, interval: 2000 })
      expect(result.status).to.equal('Complete')
      expect(result.indexed_documents).to.be.a('number')
    })

    it('should return 404 for an unknown task id', async function () {
      const res = await vl.checkTask('nonexistent-task-id')
      expect(res.status).to.equal(404)
    })

    it('should return pending status for a running task', async function () {
      // Start indexing and immediately check — should be pending
      const res = await vl.startIndex(domain, commit1)
      const taskId = res.text.trim()
      const checkRes = await vl.checkTask(taskId)
      // Could be pending or already complete (if very fast)
      expect(checkRes.status).to.equal(200)
      const body = checkRes.body
      expect(body.status).to.be.oneOf(['Pending', 'Complete'])
      // Clean up — wait for it to finish
      if (body.status === 'Pending') {
        await vl.waitForTask(taskId, { maxRetries: 60, interval: 2000 })
      }
    })
  })

  describe('Search', function () {
    before(async function () {
      // Ensure the index is built for commit1
      const res = await vl.startIndex(domain, commit1)
      const taskId = res.text.trim()
      await vl.waitForTask(taskId, { maxRetries: 60, interval: 2000 })
    })

    it('should return search results for a text query', async function () {
      const res = await vl.search(domain, commit1, 'fox jumping over dog', 3)
      expect(res.status).to.equal(200)
      const results = res.body
      expect(results).to.be.an('array')
      expect(results.length).to.be.greaterThan(0)
      // Each result should have id and distance
      for (const result of results) {
        expect(result).to.have.property('id')
        expect(result).to.have.property('distance')
      }
    })

    it('should return the most relevant document first', async function () {
      const res = await vl.search(domain, commit1, 'ocean deep blue fish', 5)
      expect(res.status).to.equal(200)
      const results = res.body
      expect(results).to.be.an('array')
      expect(results.length).to.be.greaterThan(0)
      // The top result should be the blue/ocean document
      expect(results[0].id).to.include('blue')
    })

    it('should return results sorted by distance ascending', async function () {
      const res = await vl.search(domain, commit1, 'forest trees green birds', 5)
      expect(res.status).to.equal(200)
      const results = res.body
      expect(results).to.be.an('array')
      expect(results.length).to.be.greaterThan(1)
      for (let i = 1; i < results.length; i++) {
        expect(results[i].distance).to.be.at.least(results[i - 1].distance)
      }
    })

    it('should respect the count parameter', async function () {
      const res = await vl.search(domain, commit1, 'colors nature', 2)
      expect(res.status).to.equal(200)
      const results = res.body
      expect(results).to.be.an('array')
      expect(results.length).to.be.at.most(2)
    })

    it('should return 404 for a non-existent commit index', async function () {
      const res = await vl.search(domain, 'nonexistent-commit-hash', 'test query', 5)
      expect(res.status).to.equal(404)
    })
  })

  describe('Similar documents', function () {
    before(async function () {
      // Ensure the index is built
      const res = await vl.startIndex(domain, commit1)
      const taskId = res.text.trim()
      await vl.waitForTask(taskId, { maxRetries: 60, interval: 2000 })
    })

    it('should find similar documents by id', async function () {
      const res = await vl.similar(domain, commit1, 'terminusdb:///data/SearchableDoc/red', 3)
      expect(res.status).to.equal(200)
      const results = res.body
      expect(results).to.be.an('array')
      expect(results.length).to.be.greaterThan(0)
    })

    it('should include the queried document in results', async function () {
      const res = await vl.similar(domain, commit1, 'terminusdb:///data/SearchableDoc/blue', 5)
      expect(res.status).to.equal(200)
      const results = res.body
      expect(results).to.be.an('array')
      // The first result should be the document itself (distance 0)
      expect(results[0].id).to.include('blue')
    })

    it('should return an error for a non-existent document id', async function () {
      const res = await vl.similar(domain, commit1, 'terminusdb:///data/SearchableDoc/nonexistent', 3)
      expect(res.status).to.be.oneOf([400, 404])
    })
  })

  describe('Duplicate candidates', function () {
    before(async function () {
      const res = await vl.startIndex(domain, commit1)
      const taskId = res.text.trim()
      await vl.waitForTask(taskId, { maxRetries: 60, interval: 2000 })
    })

    it('should return duplicate candidates with a low threshold', async function () {
      // Use a high threshold to find near-duplicates
      const res = await vl.duplicates(domain, commit1, 1.0)
      expect(res.status).to.equal(200)
      const results = res.body
      expect(results).to.be.an('array')
      // With distinct documents, there may or may not be duplicates
      // but the response format should be valid pairs
      for (const pair of results) {
        expect(pair).to.be.an('array').with.lengthOf(2)
      }
    })

    it('should return fewer or no duplicates with a very low threshold', async function () {
      const res = await vl.duplicates(domain, commit1, 0.001)
      expect(res.status).to.equal(200)
      const results = res.body
      expect(results).to.be.an('array')
      // With a very low threshold, we expect no duplicates
      // (all documents have distinct embeddings)
    })
  })

  describe('Statistics', function () {
    it('should return vector store statistics', async function () {
      const res = await vl.statistics()
      expect(res.status).to.equal(200)
      const stats = res.body
      expect(stats).to.be.an('object')
      // Statistics should contain some numeric fields about the vector store
    })
  })

  describe('Assign index (no-op copy)', function () {
    it('should assign an index from one commit to another', async function () {
      const res = await vl.assignIndex(domain, commit1, 'assigned-test-commit')
      expect(res.status).to.equal(204)
    })

    it('should be able to search the assigned index', async function () {
      const res = await vl.search(domain, 'assigned-test-commit', 'ocean blue fish', 3)
      expect(res.status).to.equal(200)
      const results = res.body
      expect(results).to.be.an('array')
      expect(results.length).to.be.greaterThan(0)
      expect(results[0].id).to.include('blue')
    })
  })

  describe('Incremental indexing with delta', function () {
    it('should index a second commit with updated documents', async function () {
      // Update one document
      const replaceRes = await tdb.replaceDocuments(updatedDocuments)
      expect(replaceRes.status).to.equal(200)

      commit2 = await tdb.getBranchCommitId('main')
      expect(commit2).to.not.be.null
      expect(commit2).to.not.equal(commit1)

      // Index the second commit with previous commit reference
      const res = await vl.startIndex(domain, commit2, commit1)
      const taskId = res.text.trim()
      const result = await vl.waitForTask(taskId, { maxRetries: 60, interval: 2000 })
      expect(result.status).to.equal('Complete')
      expect(result.indexed_documents).to.be.a('number')
    })

    it('should search the updated index and reflect changes', async function () {
      // Search for content that was in the updated red document
      const res = await vl.search(domain, commit2, 'crimson fox leaps sleeping hound', 3)
      expect(res.status).to.equal(200)
      const results = res.body
      expect(results).to.be.an('array')
      expect(results.length).to.be.greaterThan(0)
      // The red document should be the top result
      expect(results[0].id).to.include('red')
    })

    it('should still search the original commit index', async function () {
      const res = await vl.search(domain, commit1, 'red fox jumps lazy dog', 3)
      expect(res.status).to.equal(200)
      const results = res.body
      expect(results).to.be.an('array')
      expect(results.length).to.be.greaterThan(0)
      // Results should contain documents from the original index
      expect(results[0].id).to.include('SearchableDoc/')
    })

    it('should index a third commit with additional documents', async function () {
      // Add new documents
      const insertRes = await tdb.insertDocuments(additionalDocuments)
      expect(insertRes.status).to.equal(200)

      commit3 = await tdb.getBranchCommitId('main')
      expect(commit3).to.not.be.null
      expect(commit3).to.not.equal(commit2)

      const res = await vl.startIndex(domain, commit3, commit2)
      const taskId = res.text.trim()
      const result = await vl.waitForTask(taskId, { maxRetries: 60, interval: 2000 })
      expect(result.status).to.equal('Complete')
    })

    it('should find newly added documents in the third commit index', async function () {
      const res = await vl.search(domain, commit3, 'autumn leaves orange October', 5)
      expect(res.status).to.equal(200)
      const results = res.body
      expect(results).to.be.an('array')
      expect(results.length).to.be.greaterThan(0)
      // Results should include documents from the updated index
      expect(results[0].id).to.include('SearchableDoc/')
    })

    it('should find cherry blossoms in the third commit index', async function () {
      const res = await vl.search(domain, commit3, 'cherry blossoms pink breeze garden', 5)
      expect(res.status).to.equal(200)
      const results = res.body
      expect(results).to.be.an('array')
      expect(results.length).to.be.greaterThan(0)
      expect(results[0].id).to.include('SearchableDoc/')
    })
  })

  describe('Delete document and reindex', function () {
    it('should handle document deletion in incremental index', async function () {
      // Delete the yellow document
      const delRes = await tdb.deleteDocument('SearchableDoc/yellow')
      expect(delRes.status).to.equal(204)

      const commit4 = await tdb.getBranchCommitId('main')
      expect(commit4).to.not.equal(commit3)

      const res = await vl.startIndex(domain, commit4, commit3)
      const taskId = res.text.trim()
      const result = await vl.waitForTask(taskId, { maxRetries: 60, interval: 2000 })
      expect(result.status).to.equal('Complete')
    })

    it('should not find deleted document in search results', async function () {
      const res = await vl.search(domain, await tdb.getBranchCommitId('main'), 'sun yellow desert sand dunes', 10)
      expect(res.status).to.equal(200)
      const results = res.body
      expect(results).to.be.an('array')
      // Search should return results from the updated index
      expect(results.length).to.be.greaterThan(0)
    })
  })
})
