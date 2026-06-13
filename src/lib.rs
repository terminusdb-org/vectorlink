#![forbid(unsafe_code)]

//! tdb-search library — shared modules for server and CLI binaries.

pub mod chunk;
pub mod config;
pub mod embed;
pub mod http_api;
pub mod ingest;
pub mod kernel;
pub mod layeridx;
pub mod service;
pub mod store;

// WHY: Phase-3 placeholder — async queue/worker for cross-branch parallel indexing.
//   Currently unused; Phase 2.5 uses inline optimize-then-tag under the pipeline lock.
//   Retained for re-activation when per-commit version forking lifts the serialisation constraint.
// INVARIANT: indexqueue has no callers and no side effects; allowing dead_code on it
//   cannot mask a real bug in the active codebase.
// CONSEQUENCE: if Phase 3 is cancelled, this module and the allow are deleted together.
#[allow(dead_code)]
mod indexqueue;
