#![forbid(unsafe_code)]

//! HTTP handlers — map wire requests to service calls and serialise responses.
//! No business logic; thin translation only.

use axum::extract::{Query, RawQuery, State};
use axum::http::{HeaderMap, StatusCode};
use axum::middleware;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use super::auth::auth_middleware;
use super::AppState;
use crate::kernel::error::ServiceError;
use crate::kernel::model::{Operation, TaskStatus};

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
    pub q: Option<String>,
    pub start: Option<i64>,
    pub count: Option<i64>,
    pub mode: Option<String>,
    pub doc_type: Option<Vec<String>>,
    pub doc_id: Option<Vec<String>>,
    pub snippet: Option<bool>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct SimilarParams {
    pub domain: Option<String>,
    pub commit: Option<String>,
    pub id: Option<String>,
    pub start: Option<i64>,
    pub count: Option<i64>,
    pub doc_type: Option<Vec<String>>,
    pub snippet: Option<bool>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct DuplicatesParams {
    pub domain: Option<String>,
    pub commit: Option<String>,
    pub threshold: Option<f32>,
    pub start: Option<i64>,
    pub count: Option<i64>,
}

// ─────────────────────────── Request body structs ─────────────────────────

#[derive(Debug, Deserialize)]
pub struct SearchRequestBody {
    pub domain: Option<String>,
    pub commit: Option<String>,
    pub q: Option<String>,
    pub mode: Option<String>,
    pub start: Option<i64>,
    pub count: Option<i64>,
    pub doc_type: Option<Vec<String>>,
    pub doc_id: Option<Vec<String>>,
    pub snippet: Option<bool>,
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

/// Validate pagination parameters.
#[allow(clippy::result_large_err)]
fn validate_pagination(start: Option<i64>, count: Option<i64>) -> Result<(), Response> {
    if let Some(s) = start {
        if s < 0 {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "invalid start: must be >= 0".to_owned(),
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

    match state.service.last_indexed(&domain, &branch).await {
        Ok(result) => (StatusCode::OK, Json(result)).into_response(),
        Err(e) => service_error_to_response(e),
    }
}

async fn handle_push(
    State(state): State<AppState>,
    Query(params): Query<PushParams>,
    body: String,
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
    let target_commit = match params.target_commit {
        Some(c) if !c.is_empty() => c,
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "missing required query parameter: target_commit".to_owned(),
            );
        }
    };

    // Parse NDJSON body incrementally (line by line).
    let mut operations = Vec::new();
    let mut line_num = 0usize;
    for line in body.lines() {
        line_num += 1;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match serde_json::from_str::<Operation>(trimmed) {
            Ok(op) => operations.push(op),
            Err(e) => {
                // Loud failure: malformed NDJSON line fails the push.
                // Per contract: push returns 200 with a task_id; the error surfaces
                // via /check (task status = Error with detail).
                let task_id = format!("task-{}", uuid::Uuid::new_v4().as_simple());
                let error_msg = format!("malformed NDJSON at line {}: {}", line_num, e);
                state.service.record_error_task(&task_id, error_msg).await;
                return (StatusCode::OK, task_id).into_response();
            }
        }
    }

    match state
        .service
        .push(&domain, &branch, &target_commit, params.parent_commit.as_deref(), operations)
        .await
    {
        Ok(task_id) => (StatusCode::OK, task_id).into_response(),
        Err(e) => service_error_to_response(e),
    }
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
    let commit = match params.commit {
        Some(c) if !c.is_empty() => c,
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "missing required query parameter: commit".to_owned(),
            );
        }
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
    let commit = body.commit.or(params.commit).unwrap_or_default();
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
    if commit.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "missing required parameter: commit".to_owned(),
        );
    }
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
        )
        .await
    {
        Ok(outcome) => search_response(outcome),
        Err(e) => service_error_to_response(e),
    }
}

/// Build a search response: the bare `[{id,distance}]` array body plus the
/// `TerminusDB-Data-Version` header reporting the commit ACTUALLY served. Under
/// lag this is the nearest indexed ancestor (≠ requested ⇒ the caller detects
/// staleness) — staleness is never hidden (RISK-15, P3-LAG-1).
fn search_response(outcome: crate::service::SearchOutcome) -> Response {
    let mut response = Json(outcome.hits).into_response();
    // The served commit is an opaque id; sanitise to a valid header value. If it
    // somehow cannot form a header (non-ASCII control chars), that is an internal
    // invariant violation — fail loud rather than drop the staleness signal.
    match format!("commit:{}", outcome.served_commit).parse() {
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
                outcome.served_commit
            ),
        ),
    }
}

async fn handle_similar(
    State(state): State<AppState>,
    headers: HeaderMap,
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
    let commit = match params.commit {
        Some(c) if !c.is_empty() => c,
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "missing required query parameter: commit".to_owned(),
            );
        }
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

    match state.service.similar(&domain, &commit, &id).await {
        Ok(results) => {
            let mut response = Json(results).into_response();
            response.headers_mut().insert(
                "terminusdb-data-version",
                format!("commit:{}", commit).parse().expect("valid header"),
            );
            response
        }
        Err(e) => service_error_to_response(e),
    }
}

async fn handle_duplicates(
    State(state): State<AppState>,
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
    let commit = match params.commit {
        Some(c) if !c.is_empty() => c,
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "missing required query parameter: commit".to_owned(),
            );
        }
    };

    if let Err(r) = validate_pagination(params.start, params.count) {
        return r;
    }

    match state.service.duplicates(&domain, &commit).await {
        Ok(pairs) => {
            // Wire format: array of [id1, id2] pairs.
            let wire: Vec<[String; 2]> = pairs
                .into_iter()
                .map(|(a, b)| [a, b])
                .collect();
            Json(wire).into_response()
        }
        Err(e) => service_error_to_response(e),
    }
}

async fn handle_statistics(State(state): State<AppState>) -> Response {
    match state.service.statistics().await {
        Ok(stats) => (StatusCode::OK, Json(stats)).into_response(),
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

// ─────────────────────────── Router construction ──────────────────────────

pub fn build_router(state: AppState) -> Router {
    // Authenticated routes (admin-secret required).
    let authed_routes = Router::new()
        .route("/last-indexed", get(handle_last_indexed))
        .route("/push", post(handle_push))
        .route("/check", get(handle_check))
        .route("/assign", post(handle_assign))
        .route("/search", get(handle_search_get).post(handle_search_post))
        .route("/similar", get(handle_similar))
        .route("/duplicates", get(handle_duplicates))
        .route("/statistics", get(handle_statistics))
        .route("/domain", delete(handle_delete_domain))
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
}
