# Evidence — embedding stack (Ollama + nomic v2-moe) up & verified early

**Verdict: PASS.** The self-contained embedding runtime works end-to-end on this aarch64 host with no emulation. Captured 2026-06-13.

## Stack (docker-compose.yml)
- `embeddings` = **Ollama** (`ollama/ollama:latest`, arm64-native), OpenAI-compatible at `:11434/v1/embeddings`.
- Model = `nomic-embed-v2`, created from the official GGUF `hf.co/nomic-ai/nomic-embed-text-v2-moe-GGUF:Q8_0` (512 MB) via a one-line Modelfile (`embeddings-init` one-shot).
- `terminusdb` = `terminusdb/terminusdb-server:latest` (v12.0.5) on host **:6365** (6363/6364 taken by an existing dev TerminusDB).

## Verified (live)
- **MoE embedding support works** (the main uncertainty): `POST /v1/embeddings {model:"nomic-embed-v2", input:"search_document: ..."}` → **768-dim** vector. (Resolves the "does Ollama/llama.cpp serve a MoE embedder" risk.)
- **Determinism (same-process):** identical input → **bit-identical** embedding across two calls.
- **Prefix effect:** `search_document:` vs `search_query:` produce **different** vectors — confirms the model is prefix-sensitive (RISK-08 is real behaviour, not theoretical).
- **TerminusDB** healthy (`/api/info` → v12.0.5).

## Why Ollama (not TEI)
TEI ships **amd64-only**; on aarch64 it needs `tonistiigi/binfmt --install amd64` emulation (slow, ugly prereq). Ollama is arm64-native → reproducible, no emulation. v2-moe is kept via Nomic's official GGUF at **Q8_0** (stable memory footprint ~512 MB, near-F16 fidelity).

## Determinism caveats to carry into Phase 2
- Baseline is "**this Ollama image digest + this GGUF digest + Q8_0**" — pin all three; re-baseline goldens if any change. (Quantized, not full-precision F32.)
- Verified only **same-process repeatability** here. Cross-restart / cross-host bit-identity and any batching effects must be confirmed in the determinism battery.
- Host port for terminusdb is 6365 in this environment (6363/6364 occupied) — environment-specific, not a design fact.
