# syntax=docker/dockerfile:1
# Multi-stage Dockerfile for vectorlink.
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

# Copy manifests and the service source.
COPY Cargo.toml Cargo.lock* ./
COPY src/ src/
COPY assets/tokenizer.json.bz2 /build/tokenizer.json.bz2

# Build with BuildKit cache mounts for persistent cargo registry + target dir.
# The cache survives across builds so incremental recompilation is fast (~seconds
# for a source-only change vs ~10 min cold). The binary is copied OUT of the
# cache mount in the same RUN (cache mounts are not part of the image layer).
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo build --release && \
    cp /build/target/release/vectorlink /build/vectorlink

# ──────────────────────────── Runtime stage ────────────────────────────────
FROM debian:trixie-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
  && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/vectorlink /usr/local/bin/vectorlink
COPY --from=builder /build/tokenizer.json.bz2 /opt/vectorlink/tokenizer.json.bz2

# Non-root user for security.
RUN useradd -r -s /bin/false vectorlink

# Data directory for LanceDB datasets (owned by vectorlink user).
RUN mkdir -p /data && chown vectorlink:vectorlink /data

USER vectorlink

ENV VECTORLINK_TOKENIZER_PATH=/opt/vectorlink/tokenizer.json.bz2
ENV VECTORLINK_DATA_DIR=/data

EXPOSE 8080

ENTRYPOINT ["vectorlink"]
