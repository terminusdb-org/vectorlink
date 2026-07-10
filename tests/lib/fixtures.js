'use strict'

/**
 * Test schema with embedding metadata for vectorlink integration tests.
 *
 * The schema defines a `SearchableDoc` class with a `text` property that
 * has embedding metadata configured. This is what TerminusDB uses to
 * determine which documents and properties to index.
 */
const schema = [
  {
    '@type': 'Class',
    '@id': 'SearchableDoc',
    text: 'xsd:string',
    title: 'xsd:string',
    '@metadata': {
      embedding: {
        query: 'query($id: ID){ SearchableDoc(id : $id) { text } }',
      },
    },
  },
]

/**
 * Sample documents for indexing.
 */
const documents = [
  {
    '@type': 'SearchableDoc',
    '@id': 'SearchableDoc/red',
    title: 'Red',
    text: 'The red fox jumps over the lazy dog near the riverbank at sunset',
  },
  {
    '@type': 'SearchableDoc',
    '@id': 'SearchableDoc/blue',
    title: 'Blue',
    text: 'The ocean is deep blue and full of fish swimming in the coral reef',
  },
  {
    '@type': 'SearchableDoc',
    '@id': 'SearchableDoc/green',
    title: 'Green',
    text: 'The forest is lush green with tall trees and singing birds in spring',
  },
  {
    '@type': 'SearchableDoc',
    '@id': 'SearchableDoc/yellow',
    title: 'Yellow',
    text: 'The sun shines bright yellow over the desert sand dunes at noon',
  },
  {
    '@type': 'SearchableDoc',
    '@id': 'SearchableDoc/purple',
    title: 'Purple',
    text: 'The lavender fields stretch across the hills in a sea of purple flowers',
  },
]

/**
 * Updated documents for delta indexing tests.
 */
const updatedDocuments = [
  {
    '@type': 'SearchableDoc',
    '@id': 'SearchableDoc/red',
    title: 'Red Updated',
    text: 'The crimson fox leaps across the sleeping hound by the riverside at dusk',
  },
]

/**
 * Additional documents for incremental indexing.
 */
const additionalDocuments = [
  {
    '@type': 'SearchableDoc',
    '@id': 'SearchableDoc/orange',
    title: 'Orange',
    text: 'The autumn leaves turn orange and fall gently to the ground in October',
  },
  {
    '@type': 'SearchableDoc',
    '@id': 'SearchableDoc/pink',
    title: 'Pink',
    text: 'The cherry blossoms are pink and float in the breeze across the garden',
  },
]

module.exports = {
  schema,
  documents,
  updatedDocuments,
  additionalDocuments,
}
