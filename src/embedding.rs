use crate::vecmath::Embedding;

#[derive(Clone, Debug)]
pub enum EmbeddingProvider {
    OpenAI { api_key: String },
    Ollama {
        base_url: String,
        model: String,
        dimensions: usize,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("openai error: {0:?}")]
    OpenAI(#[from] crate::openai::EmbeddingError),
    #[error("ollama error: {0:?}")]
    Ollama(#[from] crate::ollama::OllamaError),
}

impl EmbeddingProvider {
    pub async fn embeddings_for(
        &self,
        strings: &[String],
    ) -> Result<Vec<Embedding>, EmbeddingError> {
        match self {
            EmbeddingProvider::OpenAI { api_key } => {
                let result = crate::openai::embeddings_for(api_key, strings).await?;
                Ok(result)
            }
            EmbeddingProvider::Ollama {
                base_url,
                model,
                dimensions,
            } => {
                let result =
                    crate::ollama::embeddings_for(base_url, model, *dimensions, strings).await?;
                Ok(result)
            }
        }
    }
}
