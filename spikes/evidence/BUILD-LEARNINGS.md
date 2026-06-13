# Spike build learnings (feed the implementation Dockerfile + CI)

Reusable facts discovered while building the Phase-0 spikes against the real crates. These carry forward into `tdb-search`'s Dockerfile, CI, and `Cargo.toml` even though the spike code is throwaway.

## Toolchain / environment
- **No host Rust** in this environment; build inside a `rust:1-bookworm` container (Rust 1.96, aarch64). crates.io reachable from the container.
- Use a **persistent cargo registry volume** (`-v tdb-spike-cargo:/usr/local/cargo/registry`) so the heavy `lance` dependency tree downloads once and is shared across the three spikes.

## Verified versions (pinned)
- `lancedb 0.30.0` pins **`lance =7.0.0`** (exact) and the whole `lance-*` family at `=7.0.0`.
- `lance 7.0.0` requires **`arrow ^58`** (NOT 54 — earlier research referenced an older lancedb). Use `arrow-array = "58"`, `arrow-schema = "58"`.

## Build-time system dependency: protoc (LEARNING)
- **`lance-encoding` (a transitive dep of lance 7.0) needs `protoc`** at build time — its build script fails with `Could not find protoc` (exit 101) without it. This is a build failure, not a runtime one.
- **Fix:** install **BOTH `protobuf-compiler` AND `libprotobuf-dev`**, or set `PROTOC`.
- **⚠️ `protobuf-compiler` ALONE is insufficient (subtle).** lance-encoding imports the protobuf *well-known types* (`google/protobuf/empty.proto`); those `.proto` includes ship in **`libprotobuf-dev`**, not the compiler package. With only the compiler you get a *different* failure that looks like a code bug: `protoc failed: google/protobuf/empty.proto: File not found`. Verify includes exist: `test -f /usr/include/google/protobuf/empty.proto`.
- **Implication for implementation:** `tdb-search`'s **Dockerfile build stage and CI must install `protobuf-compiler` + `libprotobuf-dev`**. (Two debugging rounds confirmed this — and both were only visible because build output was not suppressed.)

## Verified API surface (lance 7.0.0 — from cargo-cached source)
- **Branching:** `Dataset::create_branch(&mut self, &str, impl Into<Ref>, Option<ObjectStoreParams>) -> Result<Self>` — doc: "two-phase: shallow clone." `checkout_branch`, `list_branches`, `delete_branch`, `force_delete_branch`.
- **Tags (the layer index):** `Dataset::tags() -> Tags`; `Tags::{create(tag, impl Into<Ref>), get_version(tag)->u64, get(tag)->TagContents, list()->HashMap, update, delete, list_tags_ordered}`. (defined in `src/dataset/refs.rs`)
- **Compaction:** `lance::dataset::optimize::compact_files(&mut Dataset, CompactionOptions, Option<...>) -> CompactionMetrics`. Consolidates small fragments.
- **Cleanup:** `Dataset::cleanup_old_versions(...)`; `cleanup_cascade_branch` is **branch/tag-aware** — `clean_referenced_branches=false`, `error_if_tagged_old_versions`. Auto-cleanup configurable via `lance.auto_cleanup.interval` dataset config. → **Tagging every indexed commit protects it from reclamation essentially for free** (RISK-18).
- **Fragment files:** `Dataset::get_fragments()[].metadata().files[].path` — used to prove physical block sharing.
- **Arrow:** `FixedSizeListArray::from_iter_primitive::<Float32Type,_,_>(iter, dim)` builds embedding columns; `RecordBatchIterator::new(vec![Ok(batch)], schema)` feeds `Dataset::write`.

## Embedding runtime (TEI) platform reality on aarch64 (LEARNING)
- **HuggingFace `text-embeddings-inference` ships amd64-only** (verified: `ghcr.io/huggingface/text-embeddings-inference:cpu-1.8` and all `cpu*`/`1.8`/`latest` tags expose only `amd64/linux` + attestation; no arm64). The default dev host here is **aarch64**.
- **amd64 emulation was NOT pre-enabled** (`docker run --platform linux/amd64 …` → `exec format error`). **Fix:** `docker run --rm --privileged tonistiigi/binfmt --install amd64` registers `qemu-x86_64`; after that `--platform linux/amd64` works. This is a **one-time host setup** step.
- **Consequence for `docker-compose.yml`:** the `embeddings` (TEI) service must declare `platform: linux/amd64` and the host needs binfmt/qemu. It runs **emulated** on aarch64 — correct for functional + determinism tests (what we need), **slow** for throughput. Document the binfmt prerequisite in the compose/quickstart.
- **arm64-native alternative recorded (NOT adopted):** `ollama/ollama` has an arm64 image and an OpenAI-compatible `/v1/embeddings`. But determinism (arch doc 05) is pinned to **TEI + nomic specifically**; swapping the server changes the determinism baseline, so TEI-under-emulation is the chosen path. Revisit only if emulation proves unworkable.
- The model `nomic-ai/nomic-embed-text-v2-moe` (NomicBertMoE arch) requires `trust_remote_code` / TEI support — verify the running model loads in TEI as an early e2e check (not yet done; flagged).

## fastembed (in-process embedding) build deps (LEARNING)
- `fastembed` (even for the **candle** nomic-v2-moe model) hard-depends on **`ort` (ONNX Runtime, C++)**, which **statically links a C++ runtime**. Linking fails on a compiler-only image with `undefined reference to __cxa_call_terminate` / `__isoc23_strtoll` and `cannot find -lssl/-lcrypto/-lstdc++`.
- **Fix:** the build image needs **`g++` (libstdc++), `libssl-dev`, `pkg-config`** in addition to the Rust toolchain. (`g++` is the critical one — provides the C++ stdlib the ONNX static lib needs.)
- **Implication for implementation:** if the indexer embeds in-process via fastembed, its **build stage must install `g++ libssl-dev pkg-config`** (plus protoc+libprotobuf-dev for lance). The *runtime* image may also need `libstdc++`/`libssl` shared libs unless statically linked — confirm when building the real Dockerfile.
- API (verified, fastembed 5.16.1): `fastembed::NomicV2MoeTextEmbedding::from_hf(repo_id: &str, device: &candle_core::Device, dtype: candle_core::DType, max_length: usize) -> candle_core::Result<Self>`; `embed<S: AsRef<str>>(&self, texts: &[S]) -> Result<Vec<Vec<f32>>>`. `Device`/`DType` are **candle_core** types (add `candle-core = "0.10"`); fastembed does NOT re-export them. candle has **no INT8 dtype** — in-process quant tops out at **F16/BF16** (true INT8/Q8 only via the ONNX-quantized or GGUF/Ollama path).

## Environment quirk: apt GPG "invalid signature" in fresh containers
- This host intermittently fails `apt-get update` in fresh Debian containers with `At least one invalid signature was encountered ... is not signed` for **all** repos (main/updates/security). Likely cause: the **system clock is in the future (2026)** vs the repo signature validity window, or a qemu/VM gpg quirk. It is **not** repo- or sources.list-specific.
- **Practical mitigation used:** the `tdb-spikes` image was built earlier while apt was healthy (protoc + libprotobuf-dev baked in), and the C++/ssl libs (`g++`, `libssl-dev`, `pkg-config`) are **already present in the `rust:1-bookworm` base**. So fastembed builds need **no apt at run time** — run against the existing image rather than rebuilding.
- **Implication for real CI/Dockerfile:** don't assume `apt-get update` always succeeds here. Options: fix the host clock; pin a base image with deps pre-installed; or use `Acquire::Check-Valid-Until=false` / a date-correct base. Flag for the implementation Dockerfile.

## Process discipline (per product owner)
- **Never** `>/dev/null 2>&1` — it hides real failures (this protoc gap would have been invisible).
- **Route full output to a log file** (`spikes/evidence/logs/*.log`) and surface only the relevant tail; the full log stays for debugging + evidence.
