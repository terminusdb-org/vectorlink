use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};

use crate::vecmath::Embedding;

#[derive(Serialize)]
struct OllamaEmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
    dimensions: usize,
}

#[derive(Deserialize, Debug)]
struct OllamaEmbeddingResponse {
    data: Vec<OllamaEmbeddingData>,
}

#[derive(Deserialize, Debug)]
struct OllamaEmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Debug, thiserror::Error)]
pub enum OllamaError {
    #[error("ollama request error: {0:?}")]
    ReqwestError(#[from] reqwest::Error),
    #[error("ollama returned bad status: {0} {1}")]
    BadStatus(StatusCode, String),
    #[error("ollama returned {0} embeddings, expected {1}")]
    WrongCount(usize, usize),
    #[error("ollama returned embedding of dimension {0}, expected {1}")]
    WrongDimension(usize, usize),
    #[error("json parse error: {0:?}")]
    BadJson(#[from] serde_json::Error),
}

pub async fn embeddings_for(
    base_url: &str,
    model: &str,
    dimensions: usize,
    strings: &[String],
) -> Result<Vec<Embedding>, OllamaError> {
    let client = Client::new();
    let url = format!("{}/v1/embeddings", base_url.trim_end_matches('/'));

    let request_body = OllamaEmbeddingRequest {
        model,
        input: strings,
        dimensions,
    };

    let response = client.post(&url).json(&request_body).send().await?;
    let status = response.status();
    let response_bytes = response.bytes().await?;
    if status != StatusCode::OK {
        let body = String::from_utf8_lossy(&response_bytes).to_string();
        return Err(OllamaError::BadStatus(status, body));
    }

    let response: OllamaEmbeddingResponse = serde_json::from_slice(&response_bytes)?;
    if response.data.len() != strings.len() {
        return Err(OllamaError::WrongCount(
            response.data.len(),
            strings.len(),
        ));
    }

    let mut result = Vec::with_capacity(strings.len());
    for item in response.data {
        if item.embedding.len() != dimensions {
            return Err(OllamaError::WrongDimension(
                item.embedding.len(),
                dimensions,
            ));
        }
        let mut embedding = [0.0f32; 1536];
        let len = dimensions.min(1536);
        embedding[..len].copy_from_slice(&item.embedding[..len]);
        result.push(embedding);
    }

    Ok(result)
}
