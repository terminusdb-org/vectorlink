#!/usr/bin/env bash
# Build multilingual-e5-base Q8_0 GGUF from first-party intfloat weights.
#
# No reputable quantizer publishes this model as GGUF (checked: bartowski,
# second-state, QuantFactory, mradermacher, qllama, Ollama library), so we
# build it ourselves from intfloat/multilingual-e5-base safetensors and
# publish it under our own org. Everything below is pinned: the HF commit,
# the safetensors SHA-256, and the llama.cpp release.
#
# Produces:
#   dist/multilingual-e5-base-Q8_0.gguf   (303 MiB, mean pooling, 768-dim)
#   dist/multilingual-e5-base-f16.gguf    (conversion intermediate, kept for
#                                        audit — delete before publishing)
#
# Publish the Q8_0 file to a Hugging Face repo named
# <org>/multilingual-e5-base-GGUF; `ollama pull hf.co/<org>/multilingual-e5-base-GGUF:Q8_0`
# then resolves it. Tag the local copy `multilingual-e5-base` — that is the
# name VECTORLINK_MODEL defaults to.
#
# Requirements: python3 (venv is created internally), curl, llama-quantize
# (`brew install llama.cpp`), ~4 GB disk.

set -euo pipefail

HF_REPO="intfloat/multilingual-e5-base"
HF_REV="d128750597153bb5987e10b1c3493a34e5a4502a"          # intfloat commit, 2026-04
SAFETENSORS_SHA256="a18a44fad1d0b46ded15928144138cff1135d5cc8233bdd90be5f18822de09a7"
LLAMACPP_TAG="b6987"                                       # llama.cpp release tag
FILES="config.json model.safetensors tokenizer.json tokenizer_config.json sentencepiece.bpe.model special_tokens_map.json modules.json 1_Pooling/config.json"

WORK_DIR="${1:-/tmp/multilingual-e5-base-build}"
DIST_DIR="$(cd "$(dirname "$0")/.." && pwd)/dist"
VENV="$WORK_DIR/venv"

mkdir -p "$WORK_DIR/model/1_Pooling" "$DIST_DIR"
cd "$WORK_DIR/model"

echo "==> Downloading first-party weights ($HF_REPO @ ${HF_REV:0:12})"
for f in $FILES; do
    curl -sfL "https://huggingface.co/$HF_REPO/resolve/$HF_REV/$f" -o "$f"
done

echo "==> Verifying safetensors integrity"
echo "$SAFETENSORS_SHA256  model.safetensors" | shasum -a 256 -c -

echo "==> Preparing converter (llama.cpp $LLAMACPP_TAG)"
curl -sfL "https://raw.githubusercontent.com/ggml-org/llama.cpp/$LLAMACPP_TAG/convert_hf_to_gguf.py" -o "$WORK_DIR/convert_hf_to_gguf.py"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q torch transformers sentencepiece protobuf \
    "gguf @ git+https://github.com/ggml-org/llama.cpp@$LLAMACPP_TAG#subdirectory=gguf-py"

echo "==> Converting safetensors → GGUF (f16)"
"$VENV/bin/python3" "$WORK_DIR/convert_hf_to_gguf.py" "$WORK_DIR/model" \
    --outfile "$DIST_DIR/multilingual-e5-base-f16.gguf" --outtype f16

echo "==> Verifying GGUF metadata (arch=bert, pooling=mean, dim=768)"
"$VENV/bin/python3" - "$DIST_DIR/multilingual-e5-base-f16.gguf" <<'PY'
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
def val(k):
    f = r.get_field(k)
    return f.parts[f.data[0]].tolist() if f and len(f.data) else None
assert bytes(val('general.architecture')).decode() == 'bert', 'wrong architecture'
assert val('bert.pooling_type') == 1, 'pooling must be mean (1), got %r' % val('bert.pooling_type')
assert val('bert.embedding_length') == 768, 'dim must be 768'
print('metadata ok')
PY

echo "==> Quantizing f16 → Q8_0"
llama-quantize "$DIST_DIR/multilingual-e5-base-f16.gguf" \
    "$DIST_DIR/multilingual-e5-base-Q8_0.gguf" Q8_0

echo "==> Done"
shasum -a 256 "$DIST_DIR/multilingual-e5-base-Q8_0.gguf"
echo "Artifact: $DIST_DIR/multilingual-e5-base-Q8_0.gguf"
echo "Smoke-test: ollama create e5-check -f <(echo 'FROM $DIST_DIR/multilingual-e5-base-Q8_0.gguf') && curl -s localhost:11434/api/embed -d '{\"model\":\"e5-check\",\"input\":\"query: test\"}' | python3 -c 'import json,sys; print(len(json.load(sys.stdin)[\"embeddings\"][0]))'  # expect 768"
