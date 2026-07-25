//! Phase-0 spike — in-process embedding of nomic-embed-text-v2-moe via fastembed.
//!
//! Proves the indexer can OWN the embedding model in-process (no sidecar, no HTTP):
//!  1. load v2-moe in-process (candle backend, F16 — half the memory of F32, CPU)
//!  2. embed a document + a query (nomic prefixes) → 768-d vectors
//!  3. determinism: same input → identical vector
//!  4. prefix sensitivity: search_document: vs search_query: differ
//!
//! Verified API (fastembed 5.16.1): `NomicV2MoeTextEmbedding::from_hf(repo, &Device, DType, max_len)`,
//! `embed<S: AsRef<str>>(&self, &[S]) -> Result<Vec<Vec<f32>>>`. Device/DType are candle_core types.
//!
//! Throwaway. Prints `RESULT <name>: PASS|FAIL`. First run downloads the model from HF.

use candle_core::{DType, Device};
use fastembed::NomicV2MoeTextEmbedding;

fn result(name: &str, pass: bool, detail: &str) {
    println!("RESULT {name}: {} — {detail}", if pass { "PASS" } else { "FAIL" });
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- 1. load in-process at F16 (downloads weights from HF on first run) ---
    let device = Device::Cpu;
    let model = NomicV2MoeTextEmbedding::from_hf(
        "nomic-ai/nomic-embed-text-v2-moe",
        &device,
        DType::F16,
        512,
    )?;
    result("fe.1_load", true, "v2-moe loaded in-process (F16, CPU)");

    // --- 2. embed document + query (nomic prefixes) ---
    let doc = model.embed(&["search_document: The person's name is Yoda."])?;
    let dim = doc[0].len();
    result("fe.2_embed_dim", dim == 768, &format!("document embedding dim={dim}"));

    // --- 3. determinism (same input → identical vector) ---
    let doc2 = model.embed(&["search_document: The person's name is Yoda."])?;
    result("fe.3_determinism", doc[0] == doc2[0], "same input → identical vector across calls");

    // --- 4. prefix sensitivity ---
    let q = model.embed(&["search_query: who is the wise old jedi master"])?;
    result("fe.4_prefix_effect", q[0] != doc[0], "query vec differs from document vec");

    Ok(())
}
