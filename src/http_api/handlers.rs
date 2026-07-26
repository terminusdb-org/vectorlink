// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 DFRNT AB

#![forbid(unsafe_code)]

//! HTTP handlers — map wire requests to service calls and serialise responses.
//! No business logic; thin translation only.

use axum::extract::{Query, RawQuery, State};
use axum::http::{HeaderMap, StatusCode};
use axum::middleware;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use axum::body::Body;
use serde::{Deserialize, Serialize};

use super::auth::auth_middleware;
use super::AppState;
use crate::embed::EmbeddingRole;
use crate::kernel::error::ServiceError;
use crate::kernel::model::{Operation, ProgressUpdate, TaskStatus};
use crate::kernel::model::{parse_domain, Domain};

// ─────────────────────────── Query param structs ──────────────────────────

#[derive(Debug, Deserialize)]
pub struct LastIndexedParams {
    pub domain: Option<String>,
    pub branch: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct PushParams {
    pub domain: Option<String>,
    pub branch: Option<String>,
    pub target_commit: Option<String>,
    pub parent_commit: Option<String>,
    pub stream: Option<bool>,
    /// When `true`, enables clustering embedding generation at index time.
    /// Updates the domain's setting (for both new and existing domains).
    /// Defaults to `false` (no param sent).
    pub store_clustering: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct CheckParams {
    pub task_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AssignParams {
    pub domain: Option<String>,
    pub source_commit: Option<String>,
    pub target_commit: Option<String>,
}

// WHY: These query-param structs are deserialized by Serde from the HTTP request.
//   Fields declared in openapi.yaml are present for wire-contract compliance even
//   when the handler does not yet consume them (Phase-3 features: doc_type filtering,
//   snippet mode, similarity threshold).
// INVARIANT: All fields match the OpenAPI spec and are accepted/parsed; unused ones
//   are silently ignored by the handler — no user-facing error, no silent data loss.
// CONSEQUENCE: Removing a field would reject valid API requests that include it.
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct SearchGetParams {
    pub domain: Option<String>,
    pub commit: Option<String>,
    pub branch: Option<String>,
    pub q: Option<String>,
    pub start: Option<i64>,
    pub count: Option<i64>,
    pub mode: Option<String>,
    pub snippet: Option<bool>,
    // NOTE: repeated/multi-value query params (ancestor, doc_type, doc_id) are
    // NOT declared here. axum's `Query` (serde_urlencoded) cannot deserialize
    // repeated keys into a `Vec` and would 400 the whole request. They are read
    // from the raw query string via `extract_repeated_param` instead.
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct SimilarParams {
    pub domain: Option<String>,
    pub commit: Option<String>,
    pub branch: Option<String>,
    pub id: Option<String>,
    pub start: Option<i64>,
    pub count: Option<i64>,
    pub snippet: Option<bool>,
    // Repeated params (ancestor, doc_type) read from raw query via
    // `extract_repeated_param` (see SearchGetParams note).
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct DuplicatesParams {
    pub domain: Option<String>,
    pub commit: Option<String>,
    pub branch: Option<String>,
    pub threshold: Option<f32>,
    pub start: Option<i64>,
    pub count: Option<i64>,
    pub snippet: Option<bool>,
    // Repeated scope params (doc_type, doc_id for the set; target_doc_type,
    // target_doc_id for the target) read from the raw query via
    // `extract_repeated_param` (see SearchGetParams note — serde_urlencoded
    // cannot deserialize repeated keys into a Vec).
}

// ─────────────────────────── Compare query params ────────────────────────

#[derive(Debug, Deserialize)]
pub struct CompareParams {
    pub method: Option<String>,
    pub role: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct SuggestParams {
    pub domain: Option<String>,
    pub commit: Option<String>,
    pub branch: Option<String>,
    pub q: Option<String>,
    pub count: Option<i64>,
    // Repeated params (ancestor, doc_type, doc_id) read from raw query via
    // `extract_repeated_param` (see SearchGetParams note).
}

// ─────────────────────────── Request body structs ─────────────────────────

/// Request body for POST /compare (stateless text distance).
#[derive(Debug, Deserialize)]
pub struct CompareRequestBody {
    pub source: Option<String>,
    pub target: Option<String>,
}

/// Request body for POST /candidates (raw KNN gather).
#[derive(Debug, Deserialize)]
pub struct CandidatesRequestBody {
    pub domain: Option<String>,
    pub commit: Option<String>,
    pub set_doc_types: Option<Vec<String>>,
    pub set_doc_ids: Option<Vec<String>>,
    pub target_doc_types: Option<Vec<String>>,
    pub target_doc_ids: Option<Vec<String>>,
    pub k: Option<usize>,
    pub threshold_set: Option<f32>,
    pub threshold_target: Option<f32>,
    /// Comma-separated list of extra fields to include: "embeddings", "content".
    pub include: Option<String>,
    /// Nearest-first ancestor window (same contract as /search POST).
    pub ancestors: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct SearchRequestBody {
    pub domain: Option<String>,
    pub commit: Option<String>,
    pub branch: Option<String>,
    pub q: Option<String>,
    pub mode: Option<String>,
    pub start: Option<i64>,
    pub count: Option<i64>,
    pub doc_type: Option<Vec<String>>,
    pub doc_id: Option<Vec<String>>,
    pub snippet: Option<bool>,
    /// Nearest-first ancestor window supplied by TerminusDB (Spec 10 §5) — drives
    /// catch-up resolution. Body value wins over the query param, like other fields.
    pub ancestors: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct SimilarRequestBody {
    pub domain: Option<String>,
    pub commit: Option<String>,
    pub branch: Option<String>,
    pub id: Option<String>,
    pub text: Option<String>,
    pub start: Option<i64>,
    pub count: Option<i64>,
    pub doc_type: Option<Vec<String>>,
    pub doc_id: Option<Vec<String>>,
    pub snippet: Option<bool>,
    /// Nearest-first ancestor window (same as /search POST).
    pub ancestors: Option<Vec<String>>,
}

// ─────────────────────────── Response helpers ─────────────────────────────

#[derive(Serialize)]
struct ErrorBody {
    error: String,
}

fn error_response(status: StatusCode, message: String) -> Response {
    let body = ErrorBody { error: message };
    (status, Json(body)).into_response()
}

fn service_error_to_response(err: ServiceError) -> Response {
    match err {
        ServiceError::Validation(msg) => error_response(StatusCode::BAD_REQUEST, msg),
        ServiceError::NotFound(msg) => error_response(StatusCode::NOT_FOUND, msg),
        ServiceError::Conflict(msg) => error_response(StatusCode::CONFLICT, msg),
        ServiceError::Unavailable(msg) => {
            let mut response = error_response(StatusCode::SERVICE_UNAVAILABLE, msg);
            response
                .headers_mut()
                .insert("Retry-After", "30".parse().expect("valid header value"));
            response
        }
        ServiceError::Internal(msg) => {
            error_response(StatusCode::INTERNAL_SERVER_ERROR, msg)
        }
        ServiceError::Abort(msg) => {
            error_response(StatusCode::UNPROCESSABLE_ENTITY, msg)
        }
    }
}

/// Validate the TerminusDB-Data-Version header format if present.
/// Pattern: `^[^:]{4,}:[^:]{4,}$`
/// Returns Some(error_response) on invalid header, None if OK.
#[allow(clippy::result_large_err)]
fn validate_data_version_header(headers: &HeaderMap) -> Result<(), Response> {
    if let Some(value) = headers.get("terminusdb-data-version") {
        let s = value.to_str().map_err(|_| {
            error_response(
                StatusCode::BAD_REQUEST,
                "malformed TerminusDB-Data-Version header: not valid UTF-8".to_owned(),
            )
        })?;
        if !is_valid_data_version(s) {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "malformed TerminusDB-Data-Version header: must match pattern label:value (each part >=4 chars, no colons)".to_owned(),
            ));
        }
    }
    Ok(())
}

/// Check if a data version string matches the expected pattern.
fn is_valid_data_version(s: &str) -> bool {
    let parts: Vec<&str> = s.splitn(3, ':').collect();
    if parts.len() != 2 {
        return false;
    }
    parts[0].len() >= 4 && parts[1].len() >= 4
}

/// Validate search mode if provided.
#[allow(clippy::result_large_err)]
fn validate_mode(mode: Option<&str>) -> Result<(), Response> {
    if let Some(m) = mode {
        if crate::kernel::model::SearchMode::parse(m).is_none() {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                format!("invalid mode: {}", m),
            ));
        }
    }
    Ok(())
}

/// Extract repeated query parameters by name from a raw query string.
/// Handles both `?key=A&key=B` (OpenAPI explode:true) and `?key[]=A&key[]=B` formats.
fn extract_repeated_param(raw_query: Option<&str>, name: &str) -> Vec<String> {
    let query = match raw_query {
        Some(q) => q,
        None => return Vec::new(),
    };
    let bracket_name = format!("{}[]", name);
    form_urlencoded::parse(query.as_bytes())
        .filter_map(|(k, v)| {
            if k == name || k == bracket_name {
                Some(v.into_owned())
            } else {
                None
            }
        })
        .collect()
}

/// Maximum number of results a single search/similar request can return.
/// Prevents overflow in `k = (start + count) * 3` and bounds scan cost.
const MAX_RESULT_COUNT: i64 = 1000;

/// Maximum number of duplicate groups a /duplicates request can return.
/// Same as MAX_RESULT_COUNT — the scan is bounded by `max_points` separately.
const MAX_DUPLICATE_COUNT: i64 = 1000;

/// Maximum `start` offset. Prevents integer overflow in `k = (start + count) * 3`
/// when `start` is cast to `usize` and added to `count`.
const MAX_START: i64 = 1_000_000;

/// Validate pagination parameters.
/// `max_count` is the endpoint-specific upper bound for `count`.
#[allow(clippy::result_large_err)]
fn validate_pagination_with(start: Option<i64>, count: Option<i64>, max_count: i64) -> Result<(), Response> {
    if let Some(s) = start {
        if s < 0 {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "invalid start: must be >= 0".to_owned(),
            ));
        }
        if s > MAX_START {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                format!("invalid start: must be <= {}", MAX_START),
            ));
        }
    }
    if let Some(c) = count {
        if c < 1 {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "invalid count: must be >= 1".to_owned(),
            ));
        }
        if c > max_count {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                format!("invalid count: must be <= {}", max_count),
            ));
        }
    }
    Ok(())
}

/// Validate pagination parameters for search endpoints (MAX_RESULT_COUNT).
#[allow(clippy::result_large_err)]
fn validate_pagination(start: Option<i64>, count: Option<i64>) -> Result<(), Response> {
    validate_pagination_with(start, count, MAX_RESULT_COUNT)
}

/// Validate that a branch name is not a reserved internal compaction branch.
/// Branch names starting with `.-compact_rebuild_` are internal to vectorlink
/// (used by delta-fork retagging during compaction) and must not be accepted
/// from external requests. Accepting them would allow a malicious user to
/// create branches that collide with internal compaction branches, causing
/// data loss during cleanup.
#[allow(clippy::result_large_err)]
fn validate_branch_name(branch: &str) -> Result<(), Response> {
    if crate::store::lance::is_compact_rebuild_branch(branch) {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "branch name '{}' is a reserved internal compaction branch and cannot be used externally",
                branch
            ),
        ));
    }
    Ok(())
}

// ─────────────────────────── Handlers ─────────────────────────────────────

async fn handle_last_indexed(
    State(state): State<AppState>,
    Query(params): Query<LastIndexedParams>,
) -> Response {
    let domain = match params.domain {
        Some(d) if !d.is_empty() => d,
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "missing required query parameter: domain".to_owned(),
            );
        }
    };
    let branch = params.branch.unwrap_or_else(|| "main".to_owned());
    if let Err(r) = validate_branch_name(&branch) {
        return r;
    }

    match state.service.last_indexed(&domain, &branch).await {
        Ok(result) => (StatusCode::OK, Json(result)).into_response(),
        Err(e) => service_error_to_response(e),
    }
}

async fn handle_push(
    State(state): State<AppState>,
    Query(params): Query<PushParams>,
    body: Body,
) -> Response {
    let domain = match params.domain {
        Some(d) if !d.is_empty() => d,
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "missing required query parameter: domain".to_owned(),
            );
        }
    };
    let branch = params.branch.unwrap_or_else(|| "main".to_owned());
    if let Err(r) = validate_branch_name(&branch) {
        return r;
    }
    let target_commit = match params.target_commit {
        Some(c) if !c.is_empty() => c,
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "missing required query parameter: target_commit".to_owned(),
            );
        }
    };

    let is_stream = params.stream.unwrap_or(false);

    // Per-domain clustering setting: update domain_settings when
    // store_clustering=true is passed. For new domains, this creates
    // the setting. For existing domains, this updates it (so schema
    // changes that enable clustering after initial indexing take effect
    // on the next reindex).
    if let Some(true) = params.store_clustering {
        let rp = parse_domain(&domain)
            .map_err(|e| {
                error_response(
                    StatusCode::BAD_REQUEST,
                    format!("invalid domain: {}", e),
                )
            })
            .ok();
        if let Some(rp) = rp {
            let domain_obj = Domain::from_resource_path(&rp);
            let domain_str = domain_obj.as_str().to_owned();
            state
                .service
                .store
                .domain_settings
                .set(
                    &domain_str,
                    crate::config::DomainSettings { store_clustering: true },
                )
                .await;
        }
    }

    // Stream the NDJSON body incrementally, sending each parsed operation
    // through a channel to the indexing pipeline. This enables true
    // document-by-document streaming: the pipeline starts chunking and
    // embedding operations as they arrive, rather than buffering the entire
    // body before processing begins.
    use futures::StreamExt;
    use tokio::sync::mpsc;

    // Channel for sending operations to the pipeline.
    // Buffer size 128 provides backpressure without excessive memory.
    let (op_tx, op_rx) = mpsc::channel::<Operation>(128);

    // Progress channel for streaming push (stream=true).
    // Buffer 1024 items (~128KB at ~128 bytes per ProgressUpdate).
    // Backpressure: pipeline blocks on send().await when full (levels the stream).
    // On send() Err (receiver dropped = client disconnected): pipeline aborts (R1).
    let (progress_tx, progress_rx) = mpsc::channel::<ProgressUpdate>(1024);
    let progress_tx_opt = if is_stream { Some(progress_tx) } else { None };

    // Start the streaming pipeline — it will read operations from op_rx.
    let pipeline_result = state
        .service
        .push_stream(
            &domain,
            &branch,
            &target_commit,
            params.parent_commit.as_deref(),
            op_rx,
            progress_tx_opt,
        )
        .await;

    // If the pipeline couldn't start (e.g. 409 conflict), return early.
    let task_id = match pipeline_result {
        Ok(id) => id,
        Err(e) => return service_error_to_response(e),
    };

    // ── Stream mode: build streaming NDJSON response ──
    if is_stream {
        // Clone task_id for the header — the spawned task captures its own clone.
        let task_id_for_header = task_id.clone();

        // Spawn the body-reader task: reads NDJSON from the request body,
        // parses each line, and sends operations to op_tx.
        // In stream mode, abort is NOT a 422 — it's sent to the pipeline
        // which sends ProgressUpdate::Aborted through the progress channel.
        tokio::spawn(async move {
            let mut stream = body.into_data_stream();
            let mut buf = String::new();
            let mut line_num = 0usize;

            loop {
                // Try to extract complete lines from the buffer.
                while let Some(nl_pos) = buf.find('\n') {
                    let line = buf[..nl_pos].to_owned();
                    buf.drain(..=nl_pos);
                    line_num += 1;
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<Operation>(trimmed) {
                        Ok(Operation::Abort) => {
                            // Send Abort to the pipeline — it will send
                            // ProgressUpdate::Aborted through the progress channel.
                            let _ = op_tx.send(Operation::Abort).await;
                            return;
                        }
                        Ok(op) => {
                            if op_tx.send(op).await.is_err() {
                                // Pipeline closed early. Stop sending.
                                return;
                            }
                        }
                        Err(e) => {
                            let error_msg = format!("malformed NDJSON at line {}: {}", line_num, e);
                            state.service.record_error_task(&task_id, error_msg).await;
                            // Drop op_tx to signal the pipeline to stop.
                            drop(op_tx);
                            return;
                        }
                    }
                }

                // Read the next chunk from the stream.
                // FAIL-LOUD: invalid UTF-8 in the push body is a data corruption
                // issue — reject it rather than silently replacing bytes with
                // U+FFFD, which would corrupt doc_ids and text content.
                match stream.next().await {
                    Some(Ok(chunk)) => {
                        match std::str::from_utf8(&chunk) {
                            Ok(s) => buf.push_str(s),
                            Err(e) => {
                                let error_msg = format!(
                                    "invalid UTF-8 in push body at byte offset {}: {}",
                                    e.valid_up_to(),
                                    e
                                );
                                state.service.record_error_task(&task_id, error_msg).await;
                                drop(op_tx);
                                return;
                            }
                        }
                    }
                    Some(Err(_)) => {
                        drop(op_tx);
                        return;
                    }
                    None => break,
                }
            }

            // Process any remaining data in the buffer (last line without trailing newline).
            let remaining = buf.trim();
            if !remaining.is_empty() {
                line_num += 1;
                match serde_json::from_str::<Operation>(remaining) {
                    Ok(Operation::Abort) => {
                        let _ = op_tx.send(Operation::Abort).await;
                    }
                    Ok(op) => {
                        let _ = op_tx.send(op).await;
                    }
                    Err(e) => {
                        let error_msg = format!("malformed NDJSON at line {}: {}", line_num, e);
                        state.service.record_error_task(&task_id, error_msg).await;
                    }
                }
            }

            // Drop op_tx to signal the pipeline that all operations have been sent.
            drop(op_tx);
        });

        // Build the streaming response: NDJSON lines from progress_rx.
        // Each ProgressUpdate is serialized as JSON + newline.
        let progress_stream = tokio_stream::wrappers::ReceiverStream::new(progress_rx);
        let ndjson_stream = progress_stream.map(|update| {
            let json = serde_json::to_string(&update).unwrap_or_else(|_| {
                r#"{"status":"error","error":"failed to serialize progress update"}"#.to_owned()
            });
            Ok::<_, std::io::Error>(format!("{}\n", json))
        });

        // Build response with X-Task-Id header and streaming body.
        let mut response = Response::new(Body::from_stream(ndjson_stream));
        *response.status_mut() = StatusCode::OK;
        response.headers_mut().insert(
            "Content-Type",
            axum::http::HeaderValue::from_static("application/x-ndjson"),
        );
        response.headers_mut().insert(
            "X-Task-Id",
            axum::http::HeaderValue::from_str(&task_id_for_header).unwrap_or_else(|_| {
                axum::http::HeaderValue::from_static("task-unknown")
            }),
        );
        return response;
    }

    // ── Non-stream mode: current behavior (text/plain with task-id) ──
    // Drop the unused progress_rx so progress_tx (held by the pipeline as None)
    // doesn't block. Since progress_tx_opt is None, the pipeline won't send
    // anything, but we still need to drop progress_rx to avoid a resource leak.
    drop(progress_rx);

    // Now stream the NDJSON body, parsing each line and forwarding to the channel.
    let mut stream = body.into_data_stream();
    let mut buf = String::new();
    let mut line_num = 0usize;
    let mut stream_error: Option<String> = None;

    loop {
        // Try to extract complete lines from the buffer.
        while let Some(nl_pos) = buf.find('\n') {
            let line = buf[..nl_pos].to_owned();
            buf.drain(..=nl_pos);
            line_num += 1;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            match serde_json::from_str::<Operation>(trimmed) {
                Ok(Operation::Abort) => {
                    // Send Abort to the pipeline so it can stop processing.
                    let _ = op_tx.send(Operation::Abort).await;
                    return error_response(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        format!("abort received at line {}", line_num),
                    );
                }
                Ok(op) => {
                    if op_tx.send(op).await.is_err() {
                        // Pipeline closed early (e.g. it aborted). Stop sending.
                        break;
                    }
                }
                Err(e) => {
                    let error_msg = format!("malformed NDJSON at line {}: {}", line_num, e);
                    state.service.record_error_task(&task_id, error_msg).await;
                    // Drop op_tx to signal the pipeline to stop.
                    drop(op_tx);
                    return (StatusCode::OK, task_id).into_response();
                }
            }
        }

        // Read the next chunk from the stream.
        // FAIL-LOUD: invalid UTF-8 in the push body is a data corruption
        // issue — reject it rather than silently replacing bytes with
        // U+FFFD, which would corrupt doc_ids and text content.
        match stream.next().await {
            Some(Ok(chunk)) => {
                match std::str::from_utf8(&chunk) {
                    Ok(s) => buf.push_str(s),
                    Err(e) => {
                        stream_error = Some(format!(
                            "invalid UTF-8 in push body at byte offset {}: {}",
                            e.valid_up_to(),
                            e
                        ));
                        break;
                    }
                }
            }
            Some(Err(e)) => {
                stream_error = Some(format!("broken stream: {}", e));
                break;
            }
            None => break,
        }
    }

    // Process any remaining data in the buffer (last line without trailing newline).
    let remaining = buf.trim();
    if !remaining.is_empty() && stream_error.is_none() {
        line_num += 1;
        match serde_json::from_str::<Operation>(remaining) {
            Ok(Operation::Abort) => {
                let _ = op_tx.send(Operation::Abort).await;
                return error_response(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    format!("abort received at line {}", line_num),
                );
            }
            Ok(op) => {
                let _ = op_tx.send(op).await;
            }
            Err(e) => {
                let error_msg = format!("malformed NDJSON at line {}: {}", line_num, e);
                state.service.record_error_task(&task_id, error_msg).await;
                drop(op_tx);
                return (StatusCode::OK, task_id).into_response();
            }
        }
    }

    // Drop the sender to signal the pipeline that all operations have been sent.
    drop(op_tx);

    if let Some(err) = stream_error {
        // Record the error on the task so TerminusDB polling /check sees it
        // (the pipeline task itself has no way to know why the channel closed).
        state.service.record_error_task(&task_id, err).await;
        return (StatusCode::OK, task_id).into_response();
    }

    (StatusCode::OK, task_id).into_response()
}

async fn handle_check(
    State(state): State<AppState>,
    Query(params): Query<CheckParams>,
) -> Response {
    let task_id = match params.task_id {
        Some(id) if !id.is_empty() => id,
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "missing required query parameter: task_id".to_owned(),
            );
        }
    };

    match state.service.check_task(&task_id).await {
        Ok(status) => match &status {
            TaskStatus::Error { error } => {
                // Per contract: 500 with text/plain body for failed tasks.
                (StatusCode::INTERNAL_SERVER_ERROR, error.clone()).into_response()
            }
            _ => (StatusCode::OK, Json(status)).into_response(),
        },
        Err(e) => service_error_to_response(e),
    }
}

async fn handle_assign(
    State(state): State<AppState>,
    Query(params): Query<AssignParams>,
) -> Response {
    let domain = match params.domain {
        Some(d) if !d.is_empty() => d,
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "missing required query parameter: domain".to_owned(),
            );
        }
    };
    let source_commit = match params.source_commit {
        Some(c) if !c.is_empty() => c,
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "missing required query parameter: source_commit".to_owned(),
            );
        }
    };
    let target_commit = match params.target_commit {
        Some(c) if !c.is_empty() => c,
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "missing required query parameter: target_commit".to_owned(),
            );
        }
    };

    match state.service.assign(&domain, &source_commit, &target_commit).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => service_error_to_response(e),
    }
}

async fn handle_search_get(
    State(state): State<AppState>,
    headers: HeaderMap,
    RawQuery(raw_query): RawQuery,
    Query(params): Query<SearchGetParams>,
) -> Response {
    if let Err(r) = validate_data_version_header(&headers) {
        return r;
    }

    let domain = match params.domain {
        Some(d) if !d.is_empty() => d,
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "missing required query parameter: domain".to_owned(),
            );
        }
    };
    let commit_raw = params.commit.as_deref().unwrap_or("");
    let branch = params.branch.as_deref();
    if let Some(b) = branch {
        if let Err(r) = validate_branch_name(b) {
            return r;
        }
    }
    let commit = match state.service.resolve_commit_or_branch(&domain, commit_raw, branch).await {
        Ok(c) => c,
        Err(e) => return service_error_to_response(e),
    };
    let q = match params.q {
        Some(q) if !q.is_empty() => q,
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "missing required query parameter: q".to_owned(),
            );
        }
    };

    if let Err(r) = validate_mode(params.mode.as_deref()) {
        return r;
    }
    if let Err(r) = validate_pagination(params.start, params.count) {
        return r;
    }

    let mode = params
        .mode
        .as_deref()
        .and_then(crate::kernel::model::SearchMode::parse)
        .unwrap_or(crate::kernel::model::SearchMode::Hybrid);
    let start = params.start.unwrap_or(0).max(0) as usize;
    let count = params.count.unwrap_or(10).max(1) as usize;
    let doc_type_filter = extract_repeated_param(raw_query.as_deref(), "doc_type");
    let doc_id_filter = extract_repeated_param(raw_query.as_deref(), "doc_id");
    let ancestors = extract_repeated_param(raw_query.as_deref(), "ancestor");
    let snippet = params.snippet.unwrap_or(false);

    match state
        .service
        .search_with_options(
            &domain,
            &commit,
            &q,
            mode,
            start,
            count,
            &doc_type_filter,
            &doc_id_filter,
            snippet,
            &ancestors,
        )
        .await
    {
        Ok(outcome) => search_response(outcome),
        Err(e) => service_error_to_response(e),
    }
}

async fn handle_search_post(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<SearchGetParams>,
    Json(body): Json<SearchRequestBody>,
) -> Response {
    if let Err(r) = validate_data_version_header(&headers) {
        return r;
    }

    // Body takes precedence over query params (per openapi contract).
    let domain = body.domain.or(params.domain).unwrap_or_default();
    let commit_raw = body.commit.or(params.commit).unwrap_or_default();
    let branch = body.branch.or(params.branch);
    if let Some(ref b) = branch {
        if let Err(r) = validate_branch_name(b) {
            return r;
        }
    }
    let q = body.q.or(params.q).unwrap_or_default();
    let mode = body.mode.or(params.mode);
    let start = body.start.or(params.start);
    let count = body.count.or(params.count);

    if domain.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "missing required parameter: domain".to_owned(),
        );
    }
    let commit = match state.service.resolve_commit_or_branch(&domain, &commit_raw, branch.as_deref()).await {
        Ok(c) => c,
        Err(e) => return service_error_to_response(e),
    };
    if q.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "missing required parameter: q".to_owned(),
        );
    }

    if let Err(r) = validate_mode(mode.as_deref()) {
        return r;
    }
    if let Err(r) = validate_pagination(start, count) {
        return r;
    }

    let parsed_mode = mode
        .as_deref()
        .and_then(crate::kernel::model::SearchMode::parse)
        .unwrap_or(crate::kernel::model::SearchMode::Hybrid);
    let start_val = start.unwrap_or(0).max(0) as usize;
    let count_val = count.unwrap_or(10).max(1) as usize;
    let doc_type_filter = body.doc_type.unwrap_or_default();
    let doc_id_filter = body.doc_id.unwrap_or_default();
    let snippet = body.snippet.unwrap_or(false);
    // The ancestor window is taken from the JSON body (`ancestors` array) on
    // POST. (The repeated `ancestor` query param is a GET convenience; on POST
    // the structured body is the canonical source — see the contract.)
    let ancestors = body.ancestors.unwrap_or_default();

    match state
        .service
        .search_with_options(
            &domain,
            &commit,
            &q,
            parsed_mode,
            start_val,
            count_val,
            &doc_type_filter,
            &doc_id_filter,
            snippet,
            &ancestors,
        )
        .await
    {
        Ok(outcome) => search_response(outcome),
        Err(e) => service_error_to_response(e),
    }
}

/// Attach the `TerminusDB-Data-Version` header (value `commit:<served>`) to a
/// JSON body, reporting the commit ACTUALLY served. Under lag this is the
/// nearest PROVEN ancestor (≠ requested ⇒ caller detects staleness) — never
/// hidden (RISK-15, P3-LAG-1).
///
/// GRACEFUL (#F): a served commit that cannot form a valid header value (e.g.
/// a malformed/control-char commit id) yields a clean 500 — NEVER a panic. This
/// shared helper is the single sanctioned place the data-version header is set,
/// so neither `/search` nor `/similar` can panic the handler on bad input.
fn response_with_served_commit<T: Serialize>(body: T, served_commit: &str) -> Response {
    let mut response = Json(body).into_response();
    match format!("commit:{}", served_commit).parse() {
        Ok(value) => {
            response
                .headers_mut()
                .insert("terminusdb-data-version", value);
            response
        }
        Err(_) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "served commit '{}' cannot be encoded as a data-version header",
                served_commit
            ),
        ),
    }
}

/// Build a search response: the bare `[{id,distance}]` array body plus the
/// served-commit staleness header.
fn search_response(outcome: crate::service::SearchOutcome) -> Response {
    response_with_served_commit(outcome.hits, &outcome.served_commit)
}

async fn handle_similar(
    State(state): State<AppState>,
    headers: HeaderMap,
    RawQuery(raw_query): RawQuery,
    Query(params): Query<SimilarParams>,
) -> Response {
    if let Err(r) = validate_data_version_header(&headers) {
        return r;
    }

    let domain = match params.domain {
        Some(d) if !d.is_empty() => d,
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "missing required query parameter: domain".to_owned(),
            );
        }
    };
    let commit_raw = params.commit.as_deref().unwrap_or("");
    let branch = params.branch.as_deref();
    if let Some(b) = branch {
        if let Err(r) = validate_branch_name(b) {
            return r;
        }
    }
    let commit = match state.service.resolve_commit_or_branch(&domain, commit_raw, branch).await {
        Ok(c) => c,
        Err(e) => return service_error_to_response(e),
    };
    let id = match params.id {
        Some(id) if !id.is_empty() => id,
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "missing required query parameter: id".to_owned(),
            );
        }
    };

    if let Err(r) = validate_pagination(params.start, params.count) {
        return r;
    }

    let start = params.start.unwrap_or(0).max(0) as usize;
    let count = params.count.unwrap_or(10).max(1) as usize;
    let doc_type_filter = extract_repeated_param(raw_query.as_deref(), "doc_type");
    let doc_id_filter = extract_repeated_param(raw_query.as_deref(), "doc_id");
    let ancestors = extract_repeated_param(raw_query.as_deref(), "ancestor");
    let snippet = params.snippet.unwrap_or(false);

    // Route through the SAME catch-up resolution as /search (#A): the served
    // commit (exact or proven ancestor) is reported via the data-version header
    // using the shared GRACEFUL helper (#F — never panics on a bad commit id).
    match state
        .service
        .similar_with_options(
            &domain,
            &commit,
            &id,
            start,
            count,
            &doc_type_filter,
            &doc_id_filter,
            snippet,
            &ancestors,
        )
        .await
    {
        Ok(outcome) => response_with_served_commit(outcome.hits, &outcome.served_commit),
        Err(e) => service_error_to_response(e),
    }
}

async fn handle_similar_post(
    State(state): State<AppState>,
    headers: HeaderMap,
    RawQuery(raw_query): RawQuery,
    Query(params): Query<SimilarParams>,
    Json(body): Json<SimilarRequestBody>,
) -> Response {
    if let Err(r) = validate_data_version_header(&headers) {
        return r;
    }

    // Body takes precedence over query params (same contract as POST /search).
    let domain = body.domain.or(params.domain).unwrap_or_default();
    let commit_raw = body.commit.or(params.commit).unwrap_or_default();
    let branch = body.branch.or(params.branch);
    if let Some(ref b) = branch {
        if let Err(r) = validate_branch_name(b) {
            return r;
        }
    }
    let id = body.id.or(params.id).unwrap_or_default();
    let start = body.start.or(params.start);
    let count = body.count.or(params.count);

    if domain.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "missing required parameter: domain".to_owned(),
        );
    }
    let commit = match state.service.resolve_commit_or_branch(&domain, &commit_raw, branch.as_deref()).await {
        Ok(c) => c,
        Err(e) => return service_error_to_response(e),
    };

    // Text-based similarity: when id is empty but text is present, embed the
    // text with Document role (search_document: prefix) and run vector search.
    let text = body.text.or(None).unwrap_or_default();
    if id.is_empty() && !text.is_empty() {
        if let Err(r) = validate_pagination(start, count) {
            return r;
        }

        let start_val = start.unwrap_or(0).max(0) as usize;
        let count_val = count.unwrap_or(10).max(1) as usize;
        let doc_type_filter = body.doc_type.unwrap_or_default();
        let doc_id_filter = body.doc_id.unwrap_or_default();
        let snippet = body.snippet.unwrap_or(false);
        let ancestors = body.ancestors.clone().unwrap_or_else(|| {
            extract_repeated_param(raw_query.as_deref(), "ancestor")
        });

        match state
            .service
            .similar_with_text(
                &domain,
                &commit,
                &text,
                start_val,
                count_val,
                &doc_type_filter,
                &doc_id_filter,
                snippet,
                &ancestors,
            )
            .await
        {
            Ok(outcome) => response_with_served_commit(outcome.hits, &outcome.served_commit),
            Err(e) => service_error_to_response(e),
        }
    } else {
        if id.is_empty() {
            return error_response(
                StatusCode::BAD_REQUEST,
                "missing required parameter: id or text".to_owned(),
            );
        }

        if let Err(r) = validate_pagination(start, count) {
            return r;
        }

        let start_val = start.unwrap_or(0).max(0) as usize;
        let count_val = count.unwrap_or(10).max(1) as usize;
        let doc_type_filter = body.doc_type.unwrap_or_default();
        let doc_id_filter = body.doc_id.unwrap_or_default();
        let snippet = body.snippet.unwrap_or(false);
        let ancestors = body.ancestors.clone().unwrap_or_else(|| {
            extract_repeated_param(raw_query.as_deref(), "ancestor")
        });

        match state
            .service
            .similar_with_options(
                &domain,
                &commit,
                &id,
                start_val,
                count_val,
                &doc_type_filter,
                &doc_id_filter,
                snippet,
                &ancestors,
            )
            .await
        {
            Ok(outcome) => response_with_served_commit(outcome.hits, &outcome.served_commit),
            Err(e) => service_error_to_response(e),
        }
    }
}

async fn handle_duplicates(
    State(state): State<AppState>,
    RawQuery(raw_query): RawQuery,
    Query(params): Query<DuplicatesParams>,
) -> Response {
    let domain = match params.domain {
        Some(d) if !d.is_empty() => d,
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "missing required query parameter: domain".to_owned(),
            );
        }
    };
    let commit_raw = params.commit.as_deref().unwrap_or("");
    let branch = params.branch.as_deref();
    if let Some(b) = branch {
        if let Err(r) = validate_branch_name(b) {
            return r;
        }
    }
    let commit = match state.service.resolve_commit_or_branch(&domain, commit_raw, branch).await {
        Ok(c) => c,
        Err(e) => return service_error_to_response(e),
    };

    if let Err(r) = validate_pagination_with(params.start, params.count, MAX_DUPLICATE_COUNT) {
        return r;
    }

    let threshold = params.threshold.unwrap_or(0.0);
    if !threshold.is_finite() || threshold < 0.0 {
        return error_response(
            StatusCode::BAD_REQUEST,
            "threshold must be a finite, non-negative number".to_owned(),
        );
    }
    // validate_pagination above guarantees start >= 0 and count >= 1 when present.
    let start = params.start.unwrap_or(0) as usize;
    let count = params.count.map(|c| c as usize).unwrap_or(usize::MAX);
    let snippet = params.snippet.unwrap_or(false);

    // Repeated scope params: the `set` population (doc_type/doc_id) and the
    // optional `target` population (target_doc_type/target_doc_id). Read from the
    // raw query — serde_urlencoded cannot deserialize repeated keys into a Vec.
    let scope = crate::store::lance::DuplicateScope {
        set_doc_types: extract_repeated_param(raw_query.as_deref(), "doc_type"),
        set_doc_ids: extract_repeated_param(raw_query.as_deref(), "doc_id"),
        target_doc_types: extract_repeated_param(raw_query.as_deref(), "target_doc_type"),
        target_doc_ids: extract_repeated_param(raw_query.as_deref(), "target_doc_id"),
    };
    let ancestors = extract_repeated_param(raw_query.as_deref(), "ancestor");

    match state
        .service
        .duplicates_with_options(
            &domain, &commit, threshold, &scope, snippet, start, count, &ancestors,
        )
        .await
    {
        // Wire format: array of { "group": [{id, snippet?}, ...], "distance" }.
        // `DuplicateGroup` serialises directly to this shape.
        Ok(groups) => Json(groups).into_response(),
        Err(e) => service_error_to_response(e),
    }
}

// handle_resolve removed — replaced by handle_candidates (/candidates endpoint).

#[derive(Debug, Deserialize)]
pub struct StatisticsParams {
    pub domain: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct IntegrityParams {
    pub domain: Option<String>,
}

async fn handle_statistics(
    State(state): State<AppState>,
    Query(params): Query<StatisticsParams>,
) -> Response {
    match params.domain {
        Some(ref d) if !d.is_empty() => {
            match state.service.statistics_for_domain(d).await {
                Ok(stats) => (StatusCode::OK, Json(stats)).into_response(),
                Err(e) => service_error_to_response(e),
            }
        }
        _ => {
            // No domain filter — global statistics (admin/internal use only).
            // Includes internal instrumentation (data structure sizes) for
            // memory leak monitoring.
            match state.service.statistics().await {
                Ok(stats) => {
                    let internal = state.service.internal_stats().await;
                    let mut body = serde_json::to_value(&stats).unwrap_or(serde_json::json!({}));
                    if let Some(obj) = body.as_object_mut() {
                        obj.insert("internal".to_owned(), serde_json::to_value(&internal).unwrap_or(serde_json::json!({})));
                    }
                    (StatusCode::OK, Json(body)).into_response()
                }
                Err(e) => service_error_to_response(e),
            }
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingsParams {
    pub domain: String,
    /// Commit ID (same contract as /search).
    pub commit: Option<String>,
    /// Branch name — when commit is absent, serves the latest indexed commit.
    pub branch: Option<String>,
    /// Comma-separated list of doc IDs to fetch embeddings for.
    /// Now optional — when empty, doc_types or all-embeddings mode is used.
    pub doc_ids: Option<String>,
    /// Comma-separated list of doc types to filter by.
    /// When empty along with doc_ids, all embeddings are returned.
    pub doc_types: Option<String>,
    /// Nearest-first ancestor window (same contract as /search).
    pub ancestors: Option<Vec<String>>,
}

async fn handle_embeddings(
    State(state): State<AppState>,
    Query(params): Query<EmbeddingsParams>,
    headers: HeaderMap,
) -> Response {
    let accept = headers
        .get("accept")
        .map(|v| v.to_str().unwrap_or(""))
        .unwrap_or("");

    let commit_raw = params.commit.as_deref().unwrap_or("");
    let branch = params.branch.as_deref();
    if let Some(b) = branch {
        if let Err(r) = validate_branch_name(b) {
            return r;
        }
    }
    let commit = match state.service.resolve_commit_or_branch(&params.domain, commit_raw, branch).await {
        Ok(c) => c,
        Err(e) => return service_error_to_response(e),
    };

    if accept.contains("application/x-ndjson") {
        return handle_embeddings_stream(state, &commit, params).await;
    }

    let doc_ids: Vec<String> = params
        .doc_ids
        .as_deref()
        .unwrap_or("")
        .split(',')
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .collect();

    let doc_types: Vec<String> = params
        .doc_types
        .as_deref()
        .unwrap_or("")
        .split(',')
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .collect();

    let ancestors = params.ancestors.unwrap_or_default();

    match state.service.fetch_embeddings(&params.domain, &commit, &doc_ids, &doc_types, &ancestors).await {
        Ok(result) => (StatusCode::OK, Json(result)).into_response(),
        Err(e) => service_error_to_response(e),
    }
}

/// Streaming NDJSON handler for GET /embeddings.
/// Returns headers: X-Served-Commit, X-Store-Clustering, X-Total-Count.
/// Body is NDJSON: one JSON object per line, each containing doc_id, embedding,
/// and optionally clustering_embedding.
async fn handle_embeddings_stream(
    state: AppState,
    commit: &str,
    params: EmbeddingsParams,
) -> Response {
    let doc_ids: Vec<String> = params
        .doc_ids
        .as_deref()
        .unwrap_or("")
        .split(',')
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .collect();

    let doc_types: Vec<String> = params
        .doc_types
        .as_deref()
        .unwrap_or("")
        .split(',')
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .collect();

    let ancestors = params.ancestors.unwrap_or_default();

    match state.service.fetch_embeddings_stream(&params.domain, commit, &doc_ids, &doc_types, &ancestors).await {
        Ok(stream_result) => {
            let served_commit = stream_result.served_commit;
            let store_clustering = stream_result.store_clustering;
            let total_count = stream_result.total_count;
            let receiver = stream_result.receiver;

            // Convert the mpsc receiver into an async stream of NDJSON lines.
            let mapped = futures::StreamExt::map(
                tokio_stream::wrappers::ReceiverStream::new(receiver),
                |record| {
                    serde_json::to_string(&record)
                        .map(|json| format!("{}\n", json))
                        .unwrap_or_default()
                },
            );
            let ndjson_stream = futures::StreamExt::map(mapped, Ok::<_, std::convert::Infallible>);

            let body = Body::from_stream(ndjson_stream);

            let mut headers = axum::http::HeaderMap::new();
            headers.insert("content-type", "application/x-ndjson".parse().unwrap());
            headers.insert("x-served-commit", served_commit.parse().unwrap());
            headers.insert(
                "x-store-clustering",
                if store_clustering { "true" } else { "false" }.parse().unwrap(),
            );
            headers.insert(
                "x-total-count",
                total_count.to_string().parse().unwrap(),
            );

            (StatusCode::OK, headers, body).into_response()
        }
        Err(e) => service_error_to_response(e),
    }
}

// ──────────────────── POST /embeddings (batch + streaming) ────────────────────

/// JSON body for POST /embeddings (batch and streaming-response modes).
#[derive(Debug, Deserialize)]
pub struct EmbeddingsBody {
    pub domain: String,
    /// Commit ID (same contract as /search).
    pub commit: Option<String>,
    /// Branch name — when commit is absent, serves the latest indexed commit.
    pub branch: Option<String>,
    /// List of doc IDs to fetch embeddings for.
    /// When empty along with doc_types, all embeddings are returned.
    #[serde(default)]
    pub doc_ids: Vec<String>,
    /// List of doc types to filter by.
    #[serde(default)]
    pub doc_types: Vec<String>,
    /// Nearest-first ancestor window (same contract as /search).
    #[serde(default)]
    pub ancestors: Vec<String>,
    /// When true, response is NDJSON stream; when false, single JSON response.
    #[serde(default)]
    pub stream: bool,
}

/// NDJSON line for bidirectional streaming mode (Content-Type: application/x-ndjson).
/// Each line carries domain, commit, and a single doc_id (or doc_types for batch start).
#[derive(Debug, Deserialize)]
pub struct EmbeddingsNdjsonLine {
    pub domain: String,
    pub commit: String,
    pub doc_id: Option<String>,
    #[serde(default)]
    pub doc_types: Vec<String>,
    #[serde(default)]
    pub ancestors: Vec<String>,
}

/// POST /embeddings handler.
///
/// Three modes based on Content-Type and `stream` flag:
/// 1. JSON body with stream=false → single JSON response (batch)
/// 2. JSON body with stream=true → NDJSON streaming response
/// 3. NDJSON request body → bidirectional streaming (one doc_id per line in, one embedding per line out)
async fn handle_embeddings_post(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let content_type = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if content_type.contains("application/x-ndjson") {
        return handle_embeddings_post_ndjson(state, body).await;
    }

    // Parse JSON body
    let params: EmbeddingsBody = match serde_json::from_slice(&body) {
        Ok(p) => p,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": format!("Invalid JSON body: {}", e)})),
            ).into_response();
        }
    };

    let commit_raw = params.commit.as_deref().unwrap_or("");
    let branch = params.branch.as_deref();
    if let Some(b) = branch {
        if let Err(r) = validate_branch_name(b) {
            return r;
        }
    }
    let commit = match state.service.resolve_commit_or_branch(&params.domain, commit_raw, branch).await {
        Ok(c) => c,
        Err(e) => return service_error_to_response(e),
    };

    if params.stream {
        // Streaming response mode: JSON body in, NDJSON stream out
        return embeddings_stream_response(
            &state,
            &params.domain,
            &commit,
            &params.doc_ids,
            &params.doc_types,
            &params.ancestors,
        ).await;
    }

    // Batch mode: JSON body in, JSON response out
    match state.service.fetch_embeddings(&params.domain, &commit, &params.doc_ids, &params.doc_types, &params.ancestors).await {
        Ok(result) => (StatusCode::OK, Json(result)).into_response(),
        Err(e) => service_error_to_response(e),
    }
}

/// Bidirectional NDJSON streaming: reads doc_id lines from request body,
/// fetches embeddings one at a time (or in small batches), and streams
/// NDJSON embedding records back.
///
/// The first line may include `doc_types` to pre-filter. Subsequent lines
/// each carry a single `doc_id`. Domain and commit must be present on every line
/// (or at least the first line — subsequent lines inherit from the first).
async fn handle_embeddings_post_ndjson(
    state: AppState,
    body: axum::body::Bytes,
) -> Response {
    let body_str = match std::str::from_utf8(&body) {
        Ok(s) => s,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Invalid UTF-8"}))).into_response(),
    };

    // Parse all NDJSON lines, collecting doc_ids.
    // The first line provides domain/commit/ancestors/doc_types context.
    let mut domain = String::new();
    let mut commit = String::new();
    let mut ancestors: Vec<String> = Vec::new();
    let mut doc_types: Vec<String> = Vec::new();
    let mut doc_ids: Vec<String> = Vec::new();

    for (i, line) in body_str.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }

        let parsed: EmbeddingsNdjsonLine = match serde_json::from_str(trimmed) {
            Ok(p) => p,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({"error": format!("Invalid NDJSON line {}: {}", i, e)})),
                ).into_response();
            }
        };

        if i == 0 || domain.is_empty() {
            domain = parsed.domain;
            commit = parsed.commit;
            if !parsed.ancestors.is_empty() {
                ancestors = parsed.ancestors;
            }
            if !parsed.doc_types.is_empty() {
                doc_types = parsed.doc_types;
            }
        }

        if let Some(id) = parsed.doc_id {
            doc_ids.push(id);
        }
    }

    if domain.is_empty() || commit.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Missing domain or commit in NDJSON request"})),
        ).into_response();
    }

    // Stream all collected doc_ids as NDJSON
    embeddings_stream_response(&state, &domain, &commit, &doc_ids, &doc_types, &ancestors).await
}

/// Build an NDJSON streaming response from the service layer.
/// Shared by both JSON-body-with-stream and NDJSON-body modes.
async fn embeddings_stream_response(
    state: &AppState,
    domain: &str,
    commit: &str,
    doc_ids: &[String],
    doc_types: &[String],
    ancestors: &[String],
) -> Response {
    match state.service.fetch_embeddings_stream(domain, commit, doc_ids, doc_types, ancestors).await {
        Ok(stream_result) => {
            let served_commit = stream_result.served_commit;
            let store_clustering = stream_result.store_clustering;
            let total_count = stream_result.total_count;
            let receiver = stream_result.receiver;

            let mapped = futures::StreamExt::map(
                tokio_stream::wrappers::ReceiverStream::new(receiver),
                |record| {
                    serde_json::to_string(&record)
                        .map(|json| format!("{}\n", json))
                        .unwrap_or_default()
                },
            );
            let ndjson_stream = futures::StreamExt::map(mapped, Ok::<_, std::convert::Infallible>);

            let body = Body::from_stream(ndjson_stream);

            let mut headers = axum::http::HeaderMap::new();
            headers.insert("content-type", "application/x-ndjson".parse().unwrap());
            headers.insert("x-served-commit", served_commit.parse().unwrap());
            headers.insert(
                "x-store-clustering",
                if store_clustering { "true" } else { "false" }.parse().unwrap(),
            );
            headers.insert(
                "x-total-count",
                total_count.to_string().parse().unwrap(),
            );

            (StatusCode::OK, headers, body).into_response()
        }
        Err(e) => service_error_to_response(e),
    }
}

#[derive(Debug, Deserialize)]
pub struct DeleteDomainParams {
    pub domain: Option<String>,
}

/// DELETE /domain?domain=<org/db> — remove a domain's entire search footprint.
/// Admin-secret gated. Idempotent: an unknown/already-removed domain returns
/// 204 (NOT 404) — TerminusDB may retry. Fails loud (500) only on a genuine I/O
/// error (e.g. the dataset dir exists but can't be removed).
async fn handle_delete_domain(
    State(state): State<AppState>,
    Query(params): Query<DeleteDomainParams>,
) -> Response {
    let domain = match params.domain {
        Some(d) if !d.is_empty() => d,
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "missing required query parameter: domain".to_owned(),
            );
        }
    };

    match state.service.delete_domain(&domain).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => service_error_to_response(e),
    }
}

#[derive(Debug, Deserialize)]
pub struct DeleteBranchIndexParams {
    pub domain: Option<String>,
    pub branch: Option<String>,
}

/// DELETE /branch-index?domain=<org/db>&branch=<branch> — remove a single
/// branch's index tags and in-memory caches without affecting other branches.
/// Admin-secret gated. Idempotent.
async fn handle_delete_branch_index(
    State(state): State<AppState>,
    Query(params): Query<DeleteBranchIndexParams>,
) -> Response {
    let domain = match params.domain {
        Some(d) if !d.is_empty() => d,
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "missing required query parameter: domain".to_owned(),
            );
        }
    };
    let branch = match params.branch {
        Some(b) if !b.is_empty() => b,
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "missing required query parameter: branch".to_owned(),
            );
        }
    };
    if let Err(r) = validate_branch_name(&branch) {
        return r;
    }

    match state.service.delete_branch_index(&domain, &branch).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => service_error_to_response(e),
    }
}

async fn handle_health_live() -> Response {
    (StatusCode::OK, Json(serde_json::json!({"status": "ok"}))).into_response()
}

async fn handle_health_ready(State(state): State<AppState>) -> Response {
    let index_ready = state.service.is_index_ready();
    let search_ready = state.service.is_search_ready();
    let ready = index_ready || search_ready;

    let body = serde_json::json!({
        "ready": ready,
        "index": index_ready,
        "search": search_ready,
    });

    if ready {
        (StatusCode::OK, Json(body)).into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response()
    }
}

/// GET /integrity?domain=<org/db> — run a store integrity check on a domain.
///
/// Read-only. Compares on-disk state (index dirs, data files, manifests)
/// against live references from all tagged manifests across all branches.
/// Reports stale index dirs, dangling index references, and rebuild branch
/// count. Does not modify the store.
async fn handle_integrity(
    State(state): State<AppState>,
    Query(params): Query<IntegrityParams>,
) -> Response {
    let domain = match params.domain {
        Some(ref d) if !d.is_empty() => d.clone(),
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "missing required query parameter: domain".to_owned(),
            );
        }
    };

    match state.service.integrity_check(&domain).await {
        Ok(report) => (StatusCode::OK, Json(report)).into_response(),
        Err(e) => service_error_to_response(e),
    }
}

// ─────────────────────────── Compare ────────────────────────────────────

/**
 * POST /compare?method=embedding&role=<role> — stateless semantic distance.
 *
 * Optional `role` query param selects the nomic task prefix for BOTH texts:
 *   - `query` (default) — source uses search_query:, target uses search_document:
 *   - `clustering` — both texts use clustering: prefix (symmetric)
 *   - `classification` — both texts use classification: prefix (symmetric)
 *   - `document` — both texts use search_document: prefix (symmetric)
 *
 * When role is `query` (default), the asymmetric query→document embedding is used.
 * For any other role, both sides use the same prefix (symmetric comparison),
 * which is the correct mode for clustering and classification tasks.
 */
async fn handle_compare(
    State(state): State<AppState>,
    Query(params): Query<CompareParams>,
    Json(body): Json<CompareRequestBody>,
) -> Response {
    // Validate method query param — must be present and "embedding".
    match params.method.as_deref() {
        Some("embedding") => { /* valid — proceed */ }
        Some(unknown) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "unsupported compare method: '{}' (supported: embedding)",
                    unknown
                ),
            );
        }
        None => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "missing required query parameter: method (supported: embedding)".to_owned(),
            );
        }
    }

    // Parse optional role parameter (default: query → asymmetric query/document).
    let (source_role, target_role) = match params.role.as_deref() {
        Some("query") | None => (EmbeddingRole::Query, EmbeddingRole::Document),
        Some("clustering") => (EmbeddingRole::Clustering, EmbeddingRole::Clustering),
        Some("classification") => (EmbeddingRole::Classification, EmbeddingRole::Classification),
        Some("document") => (EmbeddingRole::Document, EmbeddingRole::Document),
        Some(unknown) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "unsupported role: '{}' (supported: query, document, clustering, classification)",
                    unknown
                ),
            );
        }
    };

    // Validate body fields.
    let source = match body.source {
        Some(ref s) if !s.is_empty() => s.clone(),
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "missing required body field: source".to_owned(),
            );
        }
    };
    let target = match body.target {
        Some(ref t) if !t.is_empty() => t.clone(),
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "missing required body field: target".to_owned(),
            );
        }
    };

    match state.service.compare_with_roles(&source, &target, source_role, target_role).await {
        Ok(result) => {
            let body = serde_json::json!({
                "distance": result.distance,
                "source_role": result.source_role,
                "target_role": result.target_role,
            });
            (StatusCode::OK, Json(body)).into_response()
        }
        Err(e) => service_error_to_response(e),
    }
}

// ─────────────────────────── Suggest ─────────────────────────────────────

/// GET /suggest?domain=<org/db>&commit=<hash>&q=<partial>&count=11
///
/// Typeahead assist: FTS-only (no embedding), returns approximate match count,
/// completion suggestions, and the first N document IDs for UI typeahead.
async fn handle_suggest(
    State(state): State<AppState>,
    headers: HeaderMap,
    RawQuery(raw_query): RawQuery,
    Query(params): Query<SuggestParams>,
) -> Response {
    if let Err(r) = validate_data_version_header(&headers) {
        return r;
    }

    let domain = match params.domain {
        Some(d) if !d.is_empty() => d,
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "missing required query parameter: domain".to_owned(),
            );
        }
    };
    let commit_raw = params.commit.as_deref().unwrap_or("");
    let branch = params.branch.as_deref();
    if let Some(b) = branch {
        if let Err(r) = validate_branch_name(b) {
            return r;
        }
    }
    let commit = match state.service.resolve_commit_or_branch(&domain, commit_raw, branch).await {
        Ok(c) => c,
        Err(e) => return service_error_to_response(e),
    };
    let q = match params.q {
        Some(q) if !q.is_empty() => q,
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "missing required query parameter: q".to_owned(),
            );
        }
    };

    let count = params.count.unwrap_or(11).max(1) as usize;
    let doc_type_filter = extract_repeated_param(raw_query.as_deref(), "doc_type");
    let doc_id_filter = extract_repeated_param(raw_query.as_deref(), "doc_id");
    let ancestors = extract_repeated_param(raw_query.as_deref(), "ancestor");

    match state
        .service
        .suggest(&domain, &commit, &q, count, &doc_type_filter, &doc_id_filter, &ancestors)
        .await
    {
        Ok(outcome) => response_with_served_commit(&outcome, &outcome.served_commit),
        Err(e) => service_error_to_response(e),
    }
}

async fn handle_candidates(
    State(state): State<AppState>,
    Json(body): Json<CandidatesRequestBody>,
) -> Response {
    let domain = body.domain.unwrap_or_default();
    let commit = body.commit.unwrap_or_default();

    if domain.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "missing required parameter: domain".to_owned(),
        );
    }
    if commit.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "missing required parameter: commit".to_owned(),
        );
    }

    let threshold_set = match body.threshold_set {
        Some(t) => t,
        None => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "missing required parameter: threshold_set".to_owned(),
            );
        }
    };
    if !(0.0..=1.0).contains(&threshold_set) {
        return error_response(
            StatusCode::BAD_REQUEST,
            "threshold_set must be in [0, 1]".to_owned(),
        );
    }

    let threshold_target = match body.threshold_target {
        Some(t) => t,
        None => threshold_set,
    };
    if !(0.0..=1.0).contains(&threshold_target) {
        return error_response(
            StatusCode::BAD_REQUEST,
            "threshold_target must be in [0, 1]".to_owned(),
        );
    }

    let k = body.k.unwrap_or(5);
    if k < 1 {
        return error_response(
            StatusCode::BAD_REQUEST,
            "k must be >= 1".to_owned(),
        );
    }

    let include_str = body.include.unwrap_or_default();
    let include_embeddings = include_str.split(',').any(|s| s.trim() == "embeddings");
    let include_content = include_str.split(',').any(|s| s.trim() == "content");

    let scope = crate::store::lance::DuplicateScope {
        set_doc_types: body.set_doc_types.unwrap_or_default(),
        set_doc_ids: body.set_doc_ids.unwrap_or_default(),
        target_doc_types: body.target_doc_types.unwrap_or_default(),
        target_doc_ids: body.target_doc_ids.unwrap_or_default(),
    };
    let ancestors = body.ancestors.unwrap_or_default();

    match state
        .service
        .candidates_gather(
            &domain,
            &commit,
            &scope,
            k,
            threshold_set,
            threshold_target,
            include_embeddings,
            include_content,
            &ancestors,
        )
        .await
    {
        Ok(result) => Json(result).into_response(),
        Err(e) => service_error_to_response(e),
    }
}

// ─────────────────────────── Router construction ──────────────────────────

pub fn build_router(state: AppState) -> Router {
    // Authenticated routes (admin-secret required).
    let authed_routes = Router::new()
        .route("/last-indexed", get(handle_last_indexed))
        .route("/push", post(handle_push))
        .route("/check", get(handle_check))
        .route("/assign", post(handle_assign))
        .route("/search", get(handle_search_get).post(handle_search_post))
        .route("/suggest", get(handle_suggest))
        .route("/similar", get(handle_similar).post(handle_similar_post))
        .route("/duplicates", get(handle_duplicates))
        .route("/candidates", post(handle_candidates))
        .route("/compare", post(handle_compare))
        .route("/statistics", get(handle_statistics))
        .route("/integrity", get(handle_integrity))
        .route("/embeddings", get(handle_embeddings).post(handle_embeddings_post))
        .route("/domain", delete(handle_delete_domain))
        .route("/branch-index", delete(handle_delete_branch_index))
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware));

    // Unauthenticated routes (health probes).
    let public_routes = Router::new()
        .route("/health/live", get(handle_health_live))
        .route("/health/ready", get(handle_health_ready));

    Router::new()
        .merge(authed_routes)
        .merge(public_routes)
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_data_version_passes() {
        assert!(is_valid_data_version("commit:abc123def456"));
        assert!(is_valid_data_version("branch:mainbranch"));
    }

    #[test]
    fn short_label_fails() {
        assert!(!is_valid_data_version("co:abc123def456"));
    }

    #[test]
    fn short_value_fails() {
        assert!(!is_valid_data_version("commit:ab"));
    }

    #[test]
    fn no_colon_fails() {
        assert!(!is_valid_data_version("commitabc123"));
    }

    #[test]
    fn multiple_colons_fails() {
        assert!(!is_valid_data_version("commit:abc:def"));
    }

    // ─────────────────── Streaming /push tests ───────────────────

    #[test]
    fn service_error_abort_maps_to_422() {
        let err = ServiceError::Abort("test abort".to_owned());
        let response = service_error_to_response(err);
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[test]
    fn service_error_internal_maps_to_500() {
        let err = ServiceError::Internal("test internal".to_owned());
        let response = service_error_to_response(err);
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn service_error_conflict_maps_to_409() {
        let err = ServiceError::Conflict("test conflict".to_owned());
        let response = service_error_to_response(err);
        assert_eq!(response.status(), StatusCode::CONFLICT);
    }

    #[test]
    fn operation_abort_deserializes_correctly() {
        let json = r#"{"op":"Abort"}"#;
        let op: Operation = serde_json::from_str(json).unwrap();
        assert!(matches!(op, Operation::Abort));
    }

    #[test]
    fn operation_error_deserializes_correctly() {
        let json = r#"{"op":"Error","message":"something went wrong"}"#;
        let op: Operation = serde_json::from_str(json).unwrap();
        assert!(matches!(op, Operation::Error { .. }));
    }

    #[test]
    fn operation_inserted_deserializes_correctly() {
        let json = r#"{"op":"Inserted","id":"doc1","string":"hello world"}"#;
        let op: Operation = serde_json::from_str(json).unwrap();
        match op {
            Operation::Inserted { id, string } => {
                assert_eq!(id, "doc1");
                assert_eq!(string, "hello world");
            }
            _ => panic!("expected Inserted"),
        }
    }

    #[test]
    fn ndjson_with_abort_line_parses_abort() {
        let lines = vec![
            r#"{"op":"Inserted","id":"doc1","string":"text"}"#,
            r#"{"op":"Abort"}"#,
        ];
        let mut found_abort = false;
        for line in lines {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let op: Operation = serde_json::from_str(trimmed).unwrap();
            if matches!(op, Operation::Abort) {
                found_abort = true;
                break;
            }
        }
        assert!(found_abort, "should have found Abort operation");
    }

    #[test]
    fn ndjson_streaming_simulates_chunked_read() {
        // Simulate a chunked stream where a newline arrives in a later chunk.
        let chunk1 = r#"{"op":"Inserted","id":"d"#;
        let chunk2 = r#"oc1","string":"hello"}"#;
        let chunk3 = "\n";
        let full = format!("{}{}{}", chunk1, chunk2, chunk3);
        let mut buf = String::new();
        let mut operations = Vec::new();
        for chunk in [chunk1, chunk2, chunk3] {
            buf.push_str(chunk);
            while let Some(nl_pos) = buf.find('\n') {
                let line = buf[..nl_pos].to_owned();
                buf.drain(..=nl_pos);
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let op: Operation = serde_json::from_str(trimmed).unwrap();
                operations.push(op);
            }
        }
        assert_eq!(operations.len(), 1);
        assert!(matches!(operations[0], Operation::Inserted { .. }));
        assert_eq!(full.trim(), r#"{"op":"Inserted","id":"doc1","string":"hello"}"#);
    }

    #[test]
    fn ndjson_streaming_handles_empty_body() {
        let buf = String::new();
        let remaining = buf.trim();
        assert!(remaining.is_empty(), "empty body should produce no operations");
    }

    #[test]
    fn ndjson_streaming_handles_trailing_line_without_newline() {
        let mut buf = String::from(r#"{"op":"Inserted","id":"doc1","string":"hello"}"#);
        // No trailing newline — process remaining buffer.
        let remaining = buf.trim();
        assert!(!remaining.is_empty());
        let op: Operation = serde_json::from_str(remaining).unwrap();
        assert!(matches!(op, Operation::Inserted { .. }));
        buf.clear();
    }

    // ─────────────────── Reserved branch name validation ───────────────────

    #[test]
    fn validate_branch_name_rejects_compact_rebuild_prefix() {
        // A user must not be able to push to or search on a .-compact_rebuild_
        // branch. If allowed, the branch would collide with internal compaction
        // branches and could be silently deleted by startup or post-compaction
        // cleanup (io_cleanup_compaction_branches deletes unreferenced
        // .-compact_rebuild_* branches).
        let result = validate_branch_name(".-compact_rebuild_1234567890");
        assert!(result.is_err(), "reserved .-compact_rebuild_ prefix must be rejected");
        assert_eq!(result.unwrap_err().status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn validate_branch_name_accepts_other_dot_dash_prefixes() {
        // Only .-compact_rebuild_ is reserved, not the entire .- namespace.
        // Other .- prefixed branch names are allowed from external requests.
        assert!(validate_branch_name(".-anything").is_ok());
        assert!(validate_branch_name(".-my-branch").is_ok());
        assert!(validate_branch_name(".-").is_ok());
    }

    #[test]
    fn validate_branch_name_accepts_normal_branch_names() {
        assert!(validate_branch_name("main").is_ok());
        assert!(validate_branch_name("feature/foo").is_ok());
        assert!(validate_branch_name("dev").is_ok());
        assert!(validate_branch_name("branch_with_underscores").is_ok());
    }

    #[test]
    fn validate_branch_name_accepts_double_underscore_prefix() {
        // The old __ prefix is NOT reserved — only .-compact_rebuild_ is.
        assert!(validate_branch_name("__compact_rebuild_123").is_ok());
        assert!(validate_branch_name("__my_branch").is_ok());
    }
}
