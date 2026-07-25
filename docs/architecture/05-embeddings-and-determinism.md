# 05 — Embeddings & determinism

### Learning objectives
After this document you will be able to:
- **Configure** the embedding provider (local, OpenAI, generic HTTP).
- **Explain** the task-prefix requirement of the default model and why omitting it silently degrades results.
- **State precisely** what makes embedding output deterministic, and what can break it.
- **Set up** the embedding service so integration-test output is reproducible bit-for-bit.

### Prerequisites
[01 — System overview](./01-system-overview.md). Helpful: [04 — Search](./04-search.md).

---

## 1. The reference, and what 🆕 changes

The reference (`openai.rs`) is hardwired: `https://api.openai.com/v1/embeddings`, model `text-embedding-ada-002`, 1536 dimensions, tiktoken token arrays. It needs an OpenAI key and a network.

tdb-search makes the provider **configurable** (a discriminated union, not flags):

| Provider | Endpoint | Use |
|----------|----------|-----|
| **OpenAiCompatible** (default → local Ollama) | `{base_url}/v1/embeddings` | Local model via **Ollama** (default), or TEI/vLLM |
| **OpenAi** | `api.openai.com` | Parity with the reference |
| **GenericHttp** | configurable | Any non-OpenAI shape |

> **Default local runtime:** a **local Ollama sidecar** (arm64-native, OpenAI-compatible). An in-process (no-sidecar) embedding variant is a possible future option; TEI is not used (amd64-only, needs emulation on arm64).

Embedding **dimension is configuration**, fixed per index at creation and recorded in metadata. Default is **768** (the local model). A query embedding whose dimension doesn't match the index fails loudly — never silently padded.

---

## 2. The default model and its prefix requirement

Default: **`nomic-ai/nomic-embed-text-v2-moe`** — 768-d, multilingual (~100 languages), mixture-of-experts (475M params, 305M active), served on **CPU** by a local **Ollama** sidecar (from Nomic's official GGUF, Q8_0) exposing an OpenAI-compatible `/v1/embeddings`.

**⚠️ Task prefixes are mandatory for this model.** It was trained with instruction prefixes and produces *different, worse* results without them — silently (no error). But prefixes are **model-specific**: applying nomic's prefixes to, say, an OpenAI model would corrupt the input. So prefixes are **a property of the model, not the deployment**.

tdb-search keeps a **hard-coded table of prefixes keyed by model name**. On every embed it looks up the configured model:

| Model | Document prefix | Query prefix |
|-------|-----------------|--------------|
| `nomic-ai/nomic-embed-text-v2-moe` (default) | `search_document: ` | `search_query: ` |
| `nomic-ai/nomic-embed-text-v1.5` | `search_document: ` | `search_query: ` |
| OpenAI / any unknown model | _(none)_ | _(none)_ |

- **match** → apply that model's prefix per role (Document = indexed content, Query = search text);
- **no match** → **no prefix** (safe default).

This is poka-yoke: the correct prefixes ship with the binary, so no deployment can misconfigure them. Adding a model means extending the table in code (reviewed), not changing config. A dedicated test (doc 06) asserts: the default model gets nomic prefixes, an unknown model gets none, and document-vs-query embeddings of the same text differ.

---

## 3. What "deterministic" means here

> **Goal:** running the integration suite twice against the same docker-compose stack yields **identical** vectors and therefore **identical** search rankings and distances.

An embedding model is a pure function of its inputs **given fixed weights and a fixed numerical path**. Determinism holds when *all* of the following are pinned:

| Factor | Requirement | How we pin it |
|--------|-------------|---------------|
| **Model weights** | Exact same weights every run | Pin the **GGUF digest** + the **Q8_0 quantization**. Weights load from the immutable, pinned GGUF. |
| **Inference mode** | No training-time randomness (dropout off) | Ollama/llama.cpp serves models in **inference mode** by construction — no dropout, no stochastic layers active. |
| **Quantization** | Same numeric path each run | Pin **Q8_0** (not F32 — the GGUF is quantized). Goldens are baseline-specific to this quantization; a quant change re-baselines them. |
| **Hardware/EP** | Same compute path | CPU-only in the test stack (no GPU nondeterminism). One pinned **Ollama image** digest. |
| **Pooling & normalization** | Deterministic reduction | Mean/CLS pooling + L2 normalize are deterministic given the above. |
| **Batching** ⚠️ | Must not change per-input output | **A hazard to confirm — see §4.** |
| **Input text** | Byte-identical, prefix included | Fixtures are fixed strings; prefixes injected deterministically. |

When the GGUF digest, quantization, device, and numerical path are fixed and the model runs in inference mode, the transformer forward pass is deterministic. There is no sampling step in embedding (unlike generative decoding), so there is no temperature/seed to worry about — **the output is a deterministic function of the (prefixed) input string.**

> **Determinism status:** **same-process** bit-identical embeddings hold. **Cross-restart / cross-host** bit-identity and any batching effects must be confirmed in the determinism battery (§4, doc 06) before golden vectors are enforced strictly.

---

## 4. The batching hazard (and how we neutralise it)

Any embedding server that **batches** requests can, with some kernels/precisions, let *padding perturb the low-order bits* of an item's output depending on what it was batched with — making a single input's vector depend on its neighbours and breaking determinism. (This is a general risk for Ollama as for TEI; Ollama's exact batching behaviour for embeddings is not yet characterised here.)

We neutralise this in the test configuration by making batch composition fixed:
1. **Pin the Ollama image digest, the GGUF digest, and Q8_0** (no weight or kernel drift between runs).
2. **CPU-only** (no GPU accumulation-order variance).
3. **Make batching reproducible** for tests: send embedding requests **one input per request** (no padding partners). Throughput is irrelevant for tests.
4. **Embed offline once, snapshot, and assert against the snapshot** for the strictest tests (golden vectors), so even an environment change is caught as a diff rather than silently accepted.

> The combination "pinned Ollama image + pinned GGUF + Q8_0 + CPU + one-input-per-request" is the intended foundation for bit-identical embeddings across runs. **Same-process** repeatability is proven; **cross-restart / cross-host** identity is a Phase-2 confirmation item, after which golden-vector assertions become safe to enforce strictly.

---

## 5. Failure behaviour (fail loud, no silent fallback)

- Provider returns non-2xx → the indexing task fails with the upstream status + body; `/check` reports `Error`. No "succeeded with 0 documents."
- Dimension mismatch → hard error.
- Embedding service unreachable → task errors; never a degraded success.
- Missing API key where the provider requires one → request rejected (400). For the local Ollama sidecar the key is ignored but a non-empty placeholder is still accepted (interface parity).

---

## 7. Large documents: chunking (no silent truncation)

The default model has a **512-token hard limit**. A rendered embedding string longer than that would be **silently truncated** by the tokenizer — discarding most of a large document. That is a silent-degradation failure and is forbidden.

tdb-search therefore **chunks** long inputs using the de-facto RAG standard (full design in `specs/08-document-chunking.md`):

- **Token-based chunking with overlap.** Split the string into chunks that fit the model window, with ~15% overlap so context isn't lost at boundaries.
- **One vector per chunk.** Each chunk is its own LanceDB row, keyed by `(doc_id, chunk_index)`. A short document is simply one chunk.
- **The prefix counts against the budget.** `search_document: ` consumes tokens, so chunk size = window − prefix tokens; the prefix is applied to each chunk.
- **The model's own tokenizer** sets boundaries (nomic → xlm-roberta SentencePiece) — counting with the wrong tokenizer would reintroduce truncation.
- **Search dedups chunks back to documents.** Vector/FTS/hybrid search runs over chunk rows, then groups by `doc_id` keeping the best chunk, so `/search` still returns documents (`[{id,distance}]`) — chunk ids never leak.
- **`Changed` replaces the whole chunk set; `Deleted` removes all chunks** for a `doc_id`.

This guarantees arbitrarily large documents are fully embedded and a query matching a passage deep in a document still finds it. Chunking is deterministic (a pure function of text + tokenizer + size + overlap), so it preserves the reproducibility guarantees of §3–§4.

> Why one-vector-per-chunk and not a single mean-pooled document vector? Pooling blurs passage-level meaning ("lost in the middle") and isn't the standard retrieval pattern. One-vector-per-chunk is what mainstream RAG stacks do and preserves recall. (A pooled mode may be offered later as an option, not the default.)

---

## 8. The default test stack (preview of doc 06)

`docker-compose.yml` runs three CPU-only services with no external network:
- `embeddings` — **Ollama** serving `nomic-embed-text-v2-moe` (official GGUF, Q8_0), pinned image + GGUF digests.
- `terminusdb` — a real TerminusDB for end-to-end fidelity.
- `tdb-search` — configured with `OpenAiCompatible { base_url: http://embeddings:11434 }` and the nomic prefixes.

Because the model is local, pinned, and run in inference mode with one-input-per-request batching, the vectors — and therefore every search ranking and distance — are reproducible within a run, which is what makes assertion-level integration testing possible (doc 06). (Cross-restart bit-identity is confirmed in the determinism battery before goldens are enforced strictly.)

---

## Check your understanding
1. Why must `search_document:` / `search_query:` prefixes be injected, and why is forgetting them dangerous? *(The model needs them for correct embeddings; omission degrades results silently — no error.)*
2. List the factors that must be pinned for deterministic embeddings. *(GGUF digest, quantization (Q8_0), inference mode, device/compute path, pooling, batch composition, input bytes.)*
3. What is the single biggest threat to determinism in a batching embedding server, and how is it removed for tests? *(Dynamic batching with padding; removed via one-input-per-request + pinned image/GGUF, or golden-vector snapshots.)*
4. Embedding has no temperature/seed to set — why not? *(There is no sampling step; the forward pass is deterministic given fixed inputs and weights.)*
