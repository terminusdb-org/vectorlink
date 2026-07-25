#![forbid(unsafe_code)]

//! Error taxonomy — explicit, layered error enums.
//! No `anyhow` in library APIs. Each module owns its error; service aggregates.
//! Fail-loud: no variant is silently ignored.

use thiserror::Error;

/// Service-level errors (aggregates domain errors for the transport layer).
#[derive(Debug, Error)]
pub enum ServiceError {
    #[error("validation error: {0}")]
    Validation(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("conflict: {0}")]
    Conflict(String),

    #[error("service unavailable: {0}")]
    Unavailable(String),

    #[error("internal error: {0}")]
    Internal(String),
}

/// Ingest (NDJSON parse) errors.
#[derive(Debug, Error)]
pub enum IngestError {
    #[error("malformed NDJSON at line {line}: {detail}")]
    MalformedLine { line: usize, detail: String },

    #[error("I/O error during ingest: {0}")]
    Io(String),
}

/// Store errors.
#[derive(Debug, Error)]
pub enum StoreError {
    #[error("store error: {0}")]
    Internal(String),
}
