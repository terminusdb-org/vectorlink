#![forbid(unsafe_code)]
#![allow(dead_code)]

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
