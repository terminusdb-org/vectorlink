# Phase-0 spike build/run environment (throwaway — not the tdb-search image).
# Declarative + reproducible: pins the toolchain and the protoc build-dep that
# lance-encoding requires (see evidence/BUILD-LEARNINGS.md), instead of ad-hoc
# `apt-get` inside `docker run`.
#
# Caching strategy (so deps compile once, not per-crate / per-run):
#   - the cargo registry + git caches and the workspace target/ are mounted as
#     named volumes at run time (see run.sh), so incremental rebuilds are fast.
#   - this image itself only provides the toolchain + protoc; source is mounted.
#
# Build:  docker build -t tdb-spikes spikes/
# Run:    spikes/run.sh <crate> [args...]

FROM rust:1-bookworm

# protoc — required at build time by lance-encoding (transitive dep of lance 7.0).
# NB: `protobuf-compiler` alone is NOT enough — lance-encoding imports the
# well-known types (google/protobuf/empty.proto), which live in `libprotobuf-dev`.
# Installing only the compiler fails with "google/protobuf/empty.proto: File not found".
# Build deps:
#  - protobuf-compiler + libprotobuf-dev : for lance-encoding (lance 7.0).
#  - g++ / libssl-dev / pkg-config       : for fastembed → ort (ONNX Runtime C++) static link
#                                          (these are ALREADY in rust:1-bookworm; listed for clarity).
# NOTE: this environment intermittently hits apt GPG "invalid signature" errors
# (future system clock). protoc/libprotobuf install when apt is healthy; the C++/ssl
# libs are pre-provided by the base image, so a fastembed-only build needs no apt.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      protobuf-compiler libprotobuf-dev \
      g++ libssl-dev pkg-config \
 && rm -rf /var/lib/apt/lists/* \
 && protoc --version \
 && test -f /usr/include/google/protobuf/empty.proto

WORKDIR /spikes

# Source is bind-mounted at run time; cargo registry + target are volume-mounted.
# CARGO_HOME defaults to /usr/local/cargo (registry volume mounts there);
# target/ is redirected to a volume via CARGO_TARGET_DIR in run.sh.
ENTRYPOINT ["cargo"]
CMD ["--version"]
