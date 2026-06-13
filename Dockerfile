# Multi-stage Dockerfile for tdb-search.
#
# Build stage: rust:1-bookworm with protobuf + future deps pre-installed.
# Runtime stage: debian:bookworm-slim for minimal image size.
#
# Build-time system deps (per BUILD-LEARNINGS.md):
# - protobuf-compiler + libprotobuf-dev: required by lance-encoding (Phase 2+)
# - g++ libssl-dev pkg-config: required by fastembed/ort (Phase 7)

# ──────────────────────────── Build stage ──────────────────────────────────
FROM rust:1-bookworm AS builder

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

# Cache dependency compilation by copying manifests first.
COPY Cargo.toml Cargo.lock* ./
RUN mkdir src && echo "fn main() {}" > src/main.rs \
  && cargo build --release 2>/dev/null || true \
  && rm -rf src

# Copy full source and build for real.
COPY src/ src/
RUN cargo build --release

# ──────────────────────────── Runtime stage ────────────────────────────────
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
  && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/tdb-search /usr/local/bin/tdb-search

# Non-root user for security.
RUN useradd -r -s /bin/false tdb-search
USER tdb-search

EXPOSE 8080

ENTRYPOINT ["tdb-search"]
