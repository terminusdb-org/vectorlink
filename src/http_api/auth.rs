// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 DFRNT AB

#![forbid(unsafe_code)]

//! Admin-secret gate — HTTP Basic authentication with constant-time comparison.
//! Every functional endpoint is checked; health probes are exempt.

use axum::extract::Request;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use base64::Engine;
use subtle::ConstantTimeEq;

use super::AppState;

/// Extract and validate HTTP Basic credentials from the Authorization header.
/// Returns true if the credentials match the configured admin user/secret.
/// Uses constant-time comparison to avoid timing side-channels.
pub fn verify_basic_auth(
    auth_header: Option<&str>,
    expected_user: &str,
    expected_secret: &str,
) -> bool {
    let header_value = match auth_header {
        Some(v) => v,
        None => return false,
    };

    let encoded = match header_value.strip_prefix("Basic ") {
        Some(e) => e,
        None => return false,
    };

    let decoded_bytes = match base64::engine::general_purpose::STANDARD.decode(encoded) {
        Ok(b) => b,
        Err(_) => return false,
    };

    let decoded = match String::from_utf8(decoded_bytes) {
        Ok(s) => s,
        Err(_) => return false,
    };

    let (user, secret) = match decoded.split_once(':') {
        Some(parts) => parts,
        None => return false,
    };

    // Constant-time comparison for both user and secret.
    let user_match = user.as_bytes().ct_eq(expected_user.as_bytes());
    let secret_match = secret.as_bytes().ct_eq(expected_secret.as_bytes());

    // Both must match; combine in constant time.
    (user_match & secret_match).into()
}

/// Axum middleware that enforces admin-secret on every request passing through it.
pub async fn auth_middleware(
    axum::extract::State(state): axum::extract::State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    let auth_header = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok());

    if !verify_basic_auth(auth_header, &state.config.admin_user, &state.config.admin_secret) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "missing or wrong admin secret"})),
        )
            .into_response();
    }

    next.run(request).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_basic(user: &str, secret: &str) -> String {
        let encoded = base64::engine::general_purpose::STANDARD
            .encode(format!("{}:{}", user, secret));
        format!("Basic {}", encoded)
    }

    #[test]
    fn valid_credentials_pass() {
        let header = encode_basic("admin", "root");
        assert!(verify_basic_auth(Some(&header), "admin", "root"));
    }

    #[test]
    fn wrong_secret_fails() {
        let header = encode_basic("admin", "wrong");
        assert!(!verify_basic_auth(Some(&header), "admin", "root"));
    }

    #[test]
    fn wrong_user_fails() {
        let header = encode_basic("hacker", "root");
        assert!(!verify_basic_auth(Some(&header), "admin", "root"));
    }

    #[test]
    fn missing_header_fails() {
        assert!(!verify_basic_auth(None, "admin", "root"));
    }

    #[test]
    fn malformed_header_fails() {
        assert!(!verify_basic_auth(Some("Bearer token"), "admin", "root"));
    }

    #[test]
    fn no_colon_in_decoded_fails() {
        let encoded = base64::engine::general_purpose::STANDARD.encode("nocolon");
        let header = format!("Basic {}", encoded);
        assert!(!verify_basic_auth(Some(&header), "admin", "root"));
    }

    #[test]
    fn constant_time_comparison_rejects_different_lengths() {
        // Different length secrets should still fail (ct_eq handles differing lengths).
        let header = encode_basic("admin", "short");
        assert!(!verify_basic_auth(Some(&header), "admin", "longersecret"));
    }

    #[test]
    fn empty_secret_with_different_expected_fails() {
        let header = encode_basic("admin", "");
        assert!(!verify_basic_auth(Some(&header), "admin", "root"));
    }
}
