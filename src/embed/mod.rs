#![forbid(unsafe_code)]

//! Embed — configurable embedding provider with role-based task prefixes.
//!
//! Pure core (prefix lookup, request building, response parsing) with a single
//! io entry point for the actual HTTP call. Fail-loud on dimension mismatch
//! and provider errors.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Embedding role determines which task prefix is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingRole {
    Document,
    Query,
}

/// Task prefix pair for a model (document prefix, query prefix).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskPrefix {
    pub document: &'static str,
    pub query: &'static str,
}

/// Hard-coded model → prefix table. Returns None for unknown models (no prefix).
pub fn prefixes_for_model(model: &str) -> Option<TaskPrefix> {
    match model {
        "nomic-ai/nomic-embed-text-v2-moe"
        | "nomic-embed-v2"
        | "nomic-ai/nomic-embed-text-v1.5" => Some(TaskPrefix {
            document: "search_document: ",
            query: "search_query: ",
        }),
        _ => None,
    }
}

/// Select the appropriate prefix for the given role.
pub fn prefix_for_role(prefix: &TaskPrefix, role: EmbeddingRole) -> &'static str {
    match role {
        EmbeddingRole::Document => prefix.document,
        EmbeddingRole::Query => prefix.query,
    }
}

/// Apply the task prefix to each input text.
pub fn apply_prefix(texts: &[String], model: &str, role: EmbeddingRole) -> Vec<String> {
    match prefixes_for_model(model) {
        Some(prefix) => {
            let p = prefix_for_role(&prefix, role);
            texts.iter().map(|t| format!("{}{}", p, t)).collect()
        }
        None => texts.to_vec(),
    }
}

/// Embedding provider configuration.
#[derive(Debug, Clone)]
pub enum Provider {
    /// OpenAI-compatible endpoint (default — Ollama sidecar).
    OpenAiCompatible {
        base_url: String,
        model: String,
        dim: usize,
    },
    /// Real OpenAI API.
    OpenAi {
        base_url: String,
        model: String,
        dim: usize,
    },
    /// Generic HTTP endpoint with custom request shape.
    GenericHttp {
        base_url: String,
        model: String,
        dim: usize,
    },
}

impl Provider {
    pub fn model_name(&self) -> &str {
        match self {
            Provider::OpenAiCompatible { model, .. } => model,
            Provider::OpenAi { model, .. } => model,
            Provider::GenericHttp { model, .. } => model,
        }
    }

    pub fn expected_dim(&self) -> usize {
        match self {
            Provider::OpenAiCompatible { dim, .. } => *dim,
            Provider::OpenAi { dim, .. } => *dim,
            Provider::GenericHttp { dim, .. } => *dim,
        }
    }

    pub fn base_url(&self) -> &str {
        match self {
            Provider::OpenAiCompatible { base_url, .. } => base_url,
            Provider::OpenAi { base_url, .. } => base_url,
            Provider::GenericHttp { base_url, .. } => base_url,
        }
    }
}

/// Errors from the embedding module.
#[derive(Debug, Error)]
pub enum EmbedError {
    #[error("embedding provider unreachable: {0}")]
    Unreachable(String),
    #[error("embedding provider returned non-2xx: status={status}, body={body}")]
    ProviderError { status: u16, body: String },
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("response parse error: {0}")]
    ParseError(String),
}

/// OpenAI-compatible embedding request body.
#[derive(Debug, Serialize)]
struct EmbedRequest {
    model: String,
    input: Vec<String>,
}

/// OpenAI-compatible embedding response.
#[derive(Debug, Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedResponseItem>,
}

#[derive(Debug, Deserialize)]
struct EmbedResponseItem {
    embedding: Vec<f32>,
}

/// Parse an OpenAI-compatible response body. Validates dimensions.
pub fn parse_embedding_response(
    body: &[u8],
    expected_dim: usize,
) -> Result<Vec<Vec<f32>>, EmbedError> {
    let response: EmbedResponse =
        serde_json::from_slice(body).map_err(|e| EmbedError::ParseError(e.to_string()))?;

    let embeddings: Vec<Vec<f32>> = response.data.into_iter().map(|item| item.embedding).collect();

    // Validate dimensions on every vector.
    for (i, emb) in embeddings.iter().enumerate() {
        if emb.len() != expected_dim {
            return Err(EmbedError::DimensionMismatch {
                expected: expected_dim,
                actual: emb.len(),
            });
        }
        // Validate no NaN/Inf (would corrupt the index silently).
        if emb.iter().any(|v| v.is_nan() || v.is_infinite()) {
            return Err(EmbedError::ParseError(format!(
                "embedding {} contains NaN or Inf",
                i
            )));
        }
    }

    Ok(embeddings)
}

/// Fetch embeddings from the provider (the single io entry point).
/// Applies the task prefix per role before sending.
pub async fn io_embed(
    provider: &Provider,
    texts: &[String],
    role: EmbeddingRole,
    client: &reqwest::Client,
) -> Result<Vec<Vec<f32>>, EmbedError> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let prefixed = apply_prefix(texts, provider.model_name(), role);
    let url = format!("{}/v1/embeddings", provider.base_url().trim_end_matches('/'));

    let request_body = EmbedRequest {
        model: provider.model_name().to_owned(),
        input: prefixed,
    };

    let response = client
        .post(&url)
        .json(&request_body)
        .send()
        .await
        .map_err(|e| EmbedError::Unreachable(e.to_string()))?;

    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "failed to read body".to_owned());
        return Err(EmbedError::ProviderError { status, body });
    }

    let body_bytes = response
        .bytes()
        .await
        .map_err(|e| EmbedError::ParseError(e.to_string()))?;

    let embeddings = parse_embedding_response(&body_bytes, provider.expected_dim())?;

    // Count-parity guard: the provider MUST return exactly one embedding per input.
    // A short response would silently corrupt the scatter mapping in the batched
    // pipeline (wrong embeddings assigned to wrong docs). Fail loud here so the
    // io_embed_batched retry path isolates the offending batch.
    if embeddings.len() != texts.len() {
        return Err(EmbedError::ParseError(format!(
            "count mismatch: sent {} texts, received {} embeddings",
            texts.len(),
            embeddings.len()
        )));
    }

    Ok(embeddings)
}

/// Result of batched embedding: either a successful embedding vector or an error
/// for that specific text position. Used to scatter results back with per-doc
/// failure isolation.
#[derive(Debug, Clone)]
pub enum EmbedResult {
    /// Embedding succeeded for this text position.
    Ok(Vec<f32>),
    /// Embedding failed for this text position (the text was toxic or the
    /// provider rejected it individually after a batch retry).
    Failed(String),
}

/// Batch-embed a flat list of texts in groups of `batch_size`, calling `io_embed`
/// per batch. Returns one `EmbedResult` per input text, preserving order.
///
/// Per-doc failure isolation: if a batch call fails, the batch is retried as
/// INDIVIDUAL per-text calls. Only the actually toxic text(s) get `EmbedResult::Failed`;
/// the rest succeed. This ensures a single bad document does not fail the whole commit.
///
/// Panics if `batch_size == 0` (configuration error — fail-loud).
pub async fn io_embed_batched(
    provider: &Provider,
    texts: &[String],
    batch_size: usize,
    role: EmbeddingRole,
    client: &reqwest::Client,
) -> Vec<EmbedResult> {
    assert!(batch_size > 0, "embed_batch_size must be > 0 (configuration error)");

    if texts.is_empty() {
        return Vec::new();
    }

    let mut results: Vec<EmbedResult> = Vec::with_capacity(texts.len());

    for batch in texts.chunks(batch_size) {
        match io_embed(provider, batch, role, client).await {
            Ok(embeddings) => {
                // Happy path: batch succeeded. Push all embeddings.
                for emb in embeddings {
                    results.push(EmbedResult::Ok(emb));
                }
            }
            Err(batch_err) => {
                // WHY: A batch failure does not mean ALL texts in it are bad — often a
                // single toxic input (empty, over-long, provider-rejected) poisons the
                // whole request. We retry individually to isolate the toxic text(s).
                // INVARIANT: io_embed_batched guarantees exactly one EmbedResult per input
                // text. The individual retries below fulfil that for every text in the
                // failed batch — successes get Ok, failures get Failed.
                // CONSEQUENCE: Only the actually toxic text(s) are marked Failed; all
                // other texts in the batch get their correct embeddings. The caller
                // (pipeline scatter phase) skips only the affected doc(s).
                eprintln!(
                    "[embed] batch of {} texts failed, retrying individually: {}",
                    batch.len(),
                    batch_err
                );
                for text in batch {
                    let single = std::slice::from_ref(text);
                    match io_embed(provider, single, role, client).await {
                        Ok(mut embeddings) => {
                            if let Some(emb) = embeddings.pop() {
                                results.push(EmbedResult::Ok(emb));
                            } else {
                                results.push(EmbedResult::Failed(
                                    "individual retry returned empty".to_owned(),
                                ));
                            }
                        }
                        Err(individual_err) => {
                            results.push(EmbedResult::Failed(format!(
                                "embedding failed after individual retry: {}",
                                individual_err
                            )));
                        }
                    }
                }
            }
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- prefix lookup for known and unknown models ---
    #[test]
    fn nomic_v2_has_search_prefixes() {
        let prefix = prefixes_for_model("nomic-ai/nomic-embed-text-v2-moe");
        assert!(prefix.is_some());
        let p = prefix.unwrap();
        assert_eq!(p.document, "search_document: ");
        assert_eq!(p.query, "search_query: ");
    }

    #[test]
    fn nomic_v2_short_name_has_prefixes() {
        let prefix = prefixes_for_model("nomic-embed-v2");
        assert!(prefix.is_some());
    }

    #[test]
    fn unknown_model_returns_none() {
        let prefix = prefixes_for_model("text-embedding-ada-002");
        assert!(prefix.is_none());
    }

    // --- role-based prefix application ---
    #[test]
    fn apply_prefix_document_role() {
        let texts = vec!["hello world".to_owned()];
        let result = apply_prefix(&texts, "nomic-ai/nomic-embed-text-v2-moe", EmbeddingRole::Document);
        assert_eq!(result, vec!["search_document: hello world"]);
    }

    #[test]
    fn apply_prefix_query_role() {
        let texts = vec!["hello world".to_owned()];
        let result = apply_prefix(&texts, "nomic-ai/nomic-embed-text-v2-moe", EmbeddingRole::Query);
        assert_eq!(result, vec!["search_query: hello world"]);
    }

    #[test]
    fn apply_prefix_unknown_model_no_prefix() {
        let texts = vec!["hello world".to_owned()];
        let result = apply_prefix(&texts, "unknown-model", EmbeddingRole::Document);
        assert_eq!(result, vec!["hello world"]);
    }

    // --- dimension mismatch in parse ---
    #[test]
    fn parse_response_dimension_mismatch_errors() {
        let body = serde_json::json!({
            "data": [{"embedding": [1.0, 2.0, 3.0], "index": 0}]
        });
        let result = parse_embedding_response(
            serde_json::to_vec(&body).unwrap().as_slice(),
            768,
        );
        assert!(matches!(result, Err(EmbedError::DimensionMismatch { expected: 768, actual: 3 })));
    }

    // --- Parse valid response ---
    #[test]
    fn parse_response_valid() {
        let emb = vec![0.1f32; 768];
        let body = serde_json::json!({
            "data": [{"embedding": emb, "index": 0}]
        });
        let result = parse_embedding_response(
            serde_json::to_vec(&body).unwrap().as_slice(),
            768,
        );
        assert!(result.is_ok());
        let vecs = result.unwrap();
        assert_eq!(vecs.len(), 1);
        assert_eq!(vecs[0].len(), 768);
    }

    // --- NaN in embedding fails ---
    #[test]
    fn parse_response_nan_errors() {
        let mut emb = vec![0.1f32; 768];
        emb[100] = f32::NAN;
        let body = serde_json::json!({
            "data": [{"embedding": emb, "index": 0}]
        });
        let result = parse_embedding_response(
            serde_json::to_vec(&body).unwrap().as_slice(),
            768,
        );
        assert!(matches!(result, Err(EmbedError::ParseError(_))));
    }
}
