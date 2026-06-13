#![forbid(unsafe_code)]

//! Chunk — token-based document chunking with overlap.
//!
//! Splits a rendered string into model-sized chunks using the model's own
//! tokenizer. Guarantees full token coverage (no silent truncation) or fails
//! loud. Pure logic given a loaded tokenizer handle.

use thiserror::Error;
use tokenizers::Tokenizer;

/// Chunking parameters derived from the embedding model configuration.
#[derive(Debug, Clone)]
pub struct ChunkParams {
    /// Maximum tokens per chunk (model window minus prefix token cost).
    pub max_tokens: usize,
    /// Number of overlapping tokens between adjacent chunks (~15% of max_tokens).
    pub overlap: usize,
}

/// A single chunk produced from a document.
#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    /// 0-based index of this chunk within the document.
    pub index: u32,
    /// Total number of chunks the document was split into.
    pub count: u32,
    /// Token offset of this chunk's start within the full document.
    pub token_start: u32,
    /// Total token length of the full document.
    pub doc_token_len: u32,
    /// The chunk's text content.
    pub text: String,
}

/// Errors from the chunking module.
#[derive(Debug, Error)]
pub enum ChunkError {
    #[error("tokenizer encoding failed: {0}")]
    TokenizerError(String),
    #[error("chunk params invalid: max_tokens must be > 0, got {0}")]
    InvalidParams(usize),
}

/// Chunk a document's text into model-sized pieces with overlap.
///
/// Guarantees:
/// - Every token in the document appears in at least one chunk.
/// - Each chunk is at most `params.max_tokens` tokens.
/// - Adjacent chunks overlap by `params.overlap` tokens.
/// - A short document (fits in one chunk) yields exactly one chunk.
///
/// The tokenizer must be the model's own tokenizer (for accurate token counts).
pub fn chunk_text(
    tokenizer: &Tokenizer,
    text: &str,
    params: &ChunkParams,
) -> Result<Vec<Chunk>, ChunkError> {
    if params.max_tokens == 0 {
        return Err(ChunkError::InvalidParams(params.max_tokens));
    }

    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| ChunkError::TokenizerError(e.to_string()))?;

    let token_ids = encoding.get_ids();
    let offsets = encoding.get_offsets();
    let doc_token_len = token_ids.len() as u32;

    // If the document fits in a single chunk, return it directly.
    if token_ids.len() <= params.max_tokens {
        return Ok(vec![Chunk {
            index: 0,
            count: 1,
            token_start: 0,
            doc_token_len,
            text: text.to_owned(),
        }]);
    }

    let step = if params.overlap >= params.max_tokens {
        // Overlap must be less than max_tokens; clamp to at least 1 step forward.
        1
    } else {
        params.max_tokens - params.overlap
    };

    // First pass: determine chunk boundaries (token index ranges).
    let mut boundaries: Vec<(usize, usize)> = Vec::new();
    let mut start = 0usize;
    while start < token_ids.len() {
        let end = (start + params.max_tokens).min(token_ids.len());
        boundaries.push((start, end));
        if end == token_ids.len() {
            break;
        }
        start += step;
    }

    let count = boundaries.len() as u32;

    // Second pass: extract text for each chunk using byte offsets from the tokenizer.
    // Safety: tokenizer offsets may not land on char boundaries (normalising tokenizers).
    // We clamp to the nearest valid char boundary to prevent panics.
    let chunks = boundaries
        .iter()
        .enumerate()
        .map(|(i, &(tok_start, tok_end))| {
            // Get the byte range from the tokenizer's offset mapping.
            let raw_start = offsets[tok_start].0;
            let raw_end = if tok_end > 0 && tok_end <= offsets.len() {
                offsets[tok_end - 1].1
            } else {
                text.len()
            };

            // Clamp to char boundaries (round start down, end up).
            let byte_start = snap_to_char_boundary_down(text, raw_start);
            let byte_end = snap_to_char_boundary_up(text, raw_end);
            let chunk_text = &text[byte_start..byte_end];

            Chunk {
                index: i as u32,
                count,
                token_start: tok_start as u32,
                doc_token_len,
                text: chunk_text.to_owned(),
            }
        })
        .collect();

    Ok(chunks)
}

/// Snap a byte offset DOWN to the nearest char boundary (or 0).
fn snap_to_char_boundary_down(text: &str, offset: usize) -> usize {
    let clamped = offset.min(text.len());
    // Walk backwards until we hit a char boundary.
    let mut pos = clamped;
    while pos > 0 && !text.is_char_boundary(pos) {
        pos -= 1;
    }
    pos
}

/// Snap a byte offset UP to the nearest char boundary (or text.len()).
fn snap_to_char_boundary_up(text: &str, offset: usize) -> usize {
    let clamped = offset.min(text.len());
    // Walk forwards until we hit a char boundary.
    let mut pos = clamped;
    while pos < text.len() && !text.is_char_boundary(pos) {
        pos += 1;
    }
    pos
}

/// Compute chunk location as a fraction [0, 1].
/// Returns 0.0 when doc_token_len is 0.
pub fn chunk_location(token_start: u32, doc_token_len: u32) -> f32 {
    if doc_token_len == 0 {
        0.0
    } else {
        token_start as f32 / doc_token_len as f32
    }
}

/// Compute the default ChunkParams for the nomic model (512 window, prefix-budgeted).
pub fn params_for_nomic(tokenizer: &Tokenizer, prefix: &str) -> Result<ChunkParams, ChunkError> {
    const WINDOW: usize = 512;

    let prefix_encoding = tokenizer
        .encode(prefix, false)
        .map_err(|e| ChunkError::TokenizerError(e.to_string()))?;
    let prefix_tokens = prefix_encoding.get_ids().len();
    let max_tokens = WINDOW.saturating_sub(prefix_tokens);
    let overlap = max_tokens / 7; // ~15%

    Ok(ChunkParams { max_tokens, overlap })
}

/// Load a tokenizer from a JSON file path.
pub fn io_load_tokenizer(path: &std::path::Path) -> Result<Tokenizer, ChunkError> {
    Tokenizer::from_file(path)
        .map_err(|e| ChunkError::TokenizerError(format!("failed to load tokenizer: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_tokenizer() -> Tokenizer {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("spikes")
            .join("tokenizer")
            .join("tokenizer.json");
        io_load_tokenizer(&path).expect("test tokenizer must load")
    }

    // --- large doc splits into multiple chunks with full token coverage ---
    #[test]
    fn large_doc_splits_into_multiple_chunks_with_full_coverage() {
        let tok = test_tokenizer();
        let params = params_for_nomic(&tok, "search_document: ")
            .expect("params must compute");

        // Build a document well over 512 tokens.
        let long_doc: String = (0..4000).map(|i| format!("word{} ", i)).collect();
        let chunks = chunk_text(&tok, &long_doc, &params).expect("chunking must succeed");

        // Must produce multiple chunks.
        assert!(chunks.len() > 1, "expected multiple chunks, got {}", chunks.len());

        // Each chunk must be within budget.
        for chunk in &chunks {
            let enc = tok.encode(chunk.text.as_str(), false).expect("encode chunk");
            assert!(
                enc.get_ids().len() <= params.max_tokens,
                "chunk {} has {} tokens, exceeds max {}",
                chunk.index,
                enc.get_ids().len(),
                params.max_tokens
            );
        }

        // Full coverage: every token in the original doc must appear in at least one chunk.
        let full_enc = tok.encode(long_doc.as_str(), false).expect("encode full");
        let total_tokens = full_enc.get_ids().len();
        let mut covered = vec![false; total_tokens];
        for chunk in &chunks {
            let start = chunk.token_start as usize;
            let enc = tok.encode(chunk.text.as_str(), false).expect("encode chunk");
            let chunk_len = enc.get_ids().len();
            for item in covered.iter_mut().take((start + chunk_len).min(total_tokens)).skip(start) {
                *item = true;
            }
        }
        assert!(
            covered.iter().all(|&c| c),
            "not all tokens covered — silent truncation detected"
        );
    }

    // --- single-chunk doc has correct metadata ---
    #[test]
    fn single_chunk_doc_has_correct_metadata() {
        let tok = test_tokenizer();
        let params = ChunkParams { max_tokens: 512, overlap: 64 };

        let short_text = "The person's name is Yoda.";
        let chunks = chunk_text(&tok, short_text, &params).expect("chunk short");

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].index, 0);
        assert_eq!(chunks[0].count, 1);
        assert_eq!(chunks[0].token_start, 0);
        assert!(chunks[0].doc_token_len > 0);
        assert_eq!(chunks[0].text, short_text);
    }

    // --- location calculation ---
    #[test]
    fn chunk_location_single_chunk_is_zero() {
        assert_eq!(chunk_location(0, 41), 0.0);
    }

    #[test]
    fn chunk_location_later_chunk_is_positive() {
        let loc = chunk_location(256, 1024);
        assert!((loc - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn chunk_location_zero_len_is_zero() {
        assert_eq!(chunk_location(0, 0), 0.0);
    }

    // --- overlap is present between adjacent chunks ---
    #[test]
    fn adjacent_chunks_overlap() {
        let tok = test_tokenizer();
        let params = ChunkParams { max_tokens: 100, overlap: 15 };

        let text: String = (0..500).map(|i| format!("token{} ", i)).collect();
        let chunks = chunk_text(&tok, &text, &params).expect("chunk");

        assert!(chunks.len() > 2, "need multiple chunks to test overlap");

        // Adjacent chunks should share tokens in the overlap region.
        for i in 0..chunks.len() - 1 {
            let this_end = chunks[i].token_start + params.max_tokens as u32;
            let next_start = chunks[i + 1].token_start;
            assert!(
                next_start < this_end,
                "no overlap between chunk {} and {}: end={}, next_start={}",
                i, i + 1, this_end, next_start
            );
        }
    }

    // --- Edge case: empty text ---
    #[test]
    fn empty_text_produces_single_empty_chunk() {
        let tok = test_tokenizer();
        let params = ChunkParams { max_tokens: 512, overlap: 64 };

        let chunks = chunk_text(&tok, "", &params).expect("chunk empty");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].index, 0);
        assert_eq!(chunks[0].count, 1);
        assert_eq!(chunks[0].doc_token_len, 0);
    }

    // --- Invalid params ---
    #[test]
    fn zero_max_tokens_errors() {
        let tok = test_tokenizer();
        let params = ChunkParams { max_tokens: 0, overlap: 0 };
        let result = chunk_text(&tok, "hello", &params);
        assert!(result.is_err());
    }

    // --- #4: multi-byte chars don't panic ---
    #[test]
    fn multibyte_text_does_not_panic() {
        let tok = test_tokenizer();
        // Force multi-chunk by using small max_tokens.
        let params = ChunkParams { max_tokens: 10, overlap: 2 };

        // Text with multi-byte characters (3-byte UTF-8).
        let text = "你好世界 这是一个测试句子 包含很多中文字符 用来测试分块是否正确处理多字节字符";
        let result = chunk_text(&tok, text, &params);
        assert!(result.is_ok(), "must not panic on multi-byte text: {:?}", result.err());

        let chunks = result.unwrap();
        assert!(!chunks.is_empty());
        // All chunks must be valid UTF-8 (guaranteed by &str, but check non-empty).
        for chunk in &chunks {
            assert!(!chunk.text.is_empty() || chunk.doc_token_len == 0);
        }
    }

    // --- char boundary helpers ---
    #[test]
    fn snap_to_char_boundary_ascii() {
        let text = "hello world";
        assert_eq!(snap_to_char_boundary_down(text, 5), 5);
        assert_eq!(snap_to_char_boundary_up(text, 5), 5);
    }

    #[test]
    fn snap_to_char_boundary_multibyte() {
        // "你好" = 6 bytes (3 per char). Offset 1 is in the middle of '你'.
        let text = "你好";
        // Down: snap from middle of first char → 0.
        assert_eq!(snap_to_char_boundary_down(text, 1), 0);
        assert_eq!(snap_to_char_boundary_down(text, 2), 0);
        assert_eq!(snap_to_char_boundary_down(text, 3), 3); // Start of '好'.
        // Up: snap from middle of first char → 3.
        assert_eq!(snap_to_char_boundary_up(text, 1), 3);
        assert_eq!(snap_to_char_boundary_up(text, 2), 3);
        assert_eq!(snap_to_char_boundary_up(text, 3), 3);
    }
}
