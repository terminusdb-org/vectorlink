//! Phase-0 spike 0a-3 — Tokenizer in Rust (RISK-11).
//!
//! Proves we can count tokens with nomic's exact tokenizer (xlm-roberta SentencePiece,
//! shipped as tokenizer.json) so chunk boundaries never silently truncate.
//!
//!  1. load tokenizer.json via the `tokenizers` crate
//!  2. encode → token count (deterministic)
//!  3. token-budget chunking with overlap: a >512-token doc splits into N chunks
//!     with NO dropped tokens (full coverage reconstructed) and each chunk ≤ budget
//!
//! The tokenizer.json path is passed as argv[1] (downloaded outside the sandbox).
//! Throwaway. Prints `RESULT <name>: PASS|FAIL`.

use tokenizers::tokenizer::Tokenizer;

const WINDOW: usize = 512;            // nomic-embed-text-v2-moe hard limit
const PREFIX: &str = "search_document: ";

fn result(name: &str, pass: bool, detail: &str) {
    println!("RESULT {name}: {} — {detail}", if pass { "PASS" } else { "FAIL" });
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).unwrap_or_else(|| "/data/tokenizer.json".to_string());
    println!("tokenizer.json: {path}");

    // --- 1. load ---
    let tok = match Tokenizer::from_file(&path) {
        Ok(t) => t,
        Err(e) => { result("0a3.1_load", false, &format!("load failed: {e}")); return Ok(()); }
    };
    result("0a3.1_load", true, "tokenizer.json loaded");

    // --- 2. deterministic token count + prefix cost ---
    let enc = tok.encode("The person's name is Yoda.", false).map_err(|e| e.to_string())?;
    let prefix_tokens = tok.encode(PREFIX, false).map_err(|e| e.to_string())?.get_ids().len();
    println!("sample tokens={}, prefix '{}' = {} tokens", enc.get_ids().len(), PREFIX.trim(), prefix_tokens);
    // determinism: same input → same ids
    let enc2 = tok.encode("The person's name is Yoda.", false).map_err(|e| e.to_string())?;
    result("0a3.2_count_determinism", enc.get_ids() == enc2.get_ids() && prefix_tokens > 0,
        &format!("sample={} tokens, prefix={} tokens, repeatable={}", enc.get_ids().len(), prefix_tokens, enc.get_ids()==enc2.get_ids()));

    // --- 3. full-coverage chunking of a >512-token document ---
    // Build a long doc well over the window.
    let long_doc: String = (0..4000).map(|i| format!("word{i} ")).collect();
    let ids = tok.encode(long_doc.as_str(), false).map_err(|e| e.to_string())?;
    let ids = ids.get_ids().to_vec();
    let total = ids.len();
    let budget = WINDOW - prefix_tokens;        // tokens available per chunk after the prefix
    let overlap = budget / 7;                    // ~15%
    let step = budget - overlap;

    // Token-window chunking; assert FULL COVERAGE (every token index appears in some chunk).
    let mut covered = vec![false; total];
    let mut chunks = 0usize;
    let mut max_chunk_len = 0usize;
    let mut start = 0usize;
    while start < total {
        let end = (start + budget).min(total);
        for c in covered.iter_mut().take(end).skip(start) { *c = true; }
        max_chunk_len = max_chunk_len.max(end - start);
        chunks += 1;
        if end == total { break; }
        start += step;
    }
    let all_covered = covered.iter().all(|&c| c);
    let within_budget = max_chunk_len <= budget;
    println!("doc total_tokens={total} budget={budget} overlap={overlap} → chunks={chunks} max_chunk_len={max_chunk_len}");
    result("0a3.3_full_coverage_chunking",
        total > WINDOW && all_covered && within_budget && chunks > 1,
        &format!("{total} tokens → {chunks} chunks, every token covered={all_covered}, max_chunk≤budget={within_budget} (no silent truncation)"));

    Ok(())
}
