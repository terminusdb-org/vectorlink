// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 DFRNT AB

#![forbid(unsafe_code)]

//! HTTP API — thin transport adapter mapping openapi.yaml to the service layer.
//! Owns the admin-secret gate (HTTP Basic, constant-time compare).
//! No business logic beyond wire translation and auth enforcement.

mod auth;
mod handlers;

use axum::Router;

use crate::config::Config;
use crate::service::SearchService;

/// Application state shared across all handlers.
#[derive(Debug, Clone)]
pub struct AppState {
    pub service: SearchService,
    pub config: Config,
}

/// Build the axum router with all endpoints wired, admin-secret enforced.
pub fn router(service: SearchService, config: Config) -> Router {
    let state = AppState { service, config };
    handlers::build_router(state, true)
}

/// Build the axum router for a trusted in-process caller (no auth layer).
///
/// Only reachable through a library embedding where requests are already
/// authorized by the host — never expose this on a socket.
pub fn trusted_router(service: SearchService, config: Config) -> Router {
    let state = AppState { service, config };
    handlers::build_router(state, false)
}
