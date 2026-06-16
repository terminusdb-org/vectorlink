# syntax=docker/dockerfile:1
# Multi-stage Dockerfile for tdb-search.
#
# Build stage: rust:1-trixie with protobuf + future deps pre-installed.
# Runtime stage: debian:trixie-slim for minimal image size.
#
# Build-time system deps (per BUILD-LEARNINGS.md):
# - protobuf-compiler + libprotobuf-dev: required by lance-encoding (Phase 2+)
# - g++ libssl-dev pkg-config: required by fastembed/ort (Phase 7)

# ──────────────────────────── Build stage ──────────────────────────────────
FROM rust:1-trixie AS builder

# Install build-time system dependencies for Phase 2+ forward compatibility.
# protoc + libprotobuf-dev: lance-encoding needs protoc + well-known .proto includes.
# g++ libssl-dev pkg-config: fastembed/ort static C++ link (Phase 7).
RUN apt-get update && apt-get install -y --no-install-recommends \
    protobuf-compiler \
    libprotobuf-dev \
    g++ \
    libssl-dev \
    pkg-config \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy manifests and the service source. The `benches/` directory holds
# throwaway measurement bins (scale-evidence, crossover-measurement) that are
# NOT part of the shipped service, so the production image neither copies nor
# builds them — the build is scoped to the service binary below.
COPY Cargo.toml Cargo.lock* ./
COPY src/ src/
COPY assets/tokenizer.json.bz2 /build/tokenizer.json.bz2

# Build with BuildKit cache mounts for persistent cargo registry + target dir.
# The cache survives across builds so incremental recompilation is fast (~seconds
# for a source-only change vs ~10 min cold). The binary is copied OUT of the
# cache mount in the same RUN (cache mounts are not part of the image layer).
# --bin tdb-search builds ONLY the shipped service binary, not the benches/*
# measurement bins (which would otherwise require the benches/ source).
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo build --release --bin tdb-search && \
    cp /build/target/release/tdb-search /build/tdb-search

# ──────────────────────────── Runtime stage ────────────────────────────────
FROM debian:trixie-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
  && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/tdb-search /usr/local/bin/tdb-search
COPY --from=builder /build/tokenizer.json.bz2 /opt/tdb-search/tokenizer.json.bz2

# Non-root user for security.
RUN useradd -r -s /bin/false tdb-search

# Data directory for LanceDB datasets (owned by tdb-search user).
RUN mkdir -p /data && chown tdb-search:tdb-search /data

USER tdb-search

ENV TDB_SEARCH_TOKENIZER_PATH=/opt/tdb-search/tokenizer.json.bz2
ENV TDB_SEARCH_DATA_DIR=/data

EXPOSE 8080

ENTRYPOINT ["tdb-search"]
