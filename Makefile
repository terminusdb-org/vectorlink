# vectorlink — verification & build orchestration.
#
# Conventions adopted from the TerminusDB repo Makefile (lint / clippy /
# lint-openapi / test / pr aggregate gate) so the two projects feel the same.
#
# HARD GATE (product owner): `make lint` MUST pass before any code is committed.
# `make pr` is the full pre-PR verification gate and must be green to open a PR.
#
# Tooling is invoked via `npx` (no global installs); Redocly is pinned. The Rust
# targets (clippy/test) guard on the crate's existence, so they no-op cleanly
# until the crate is present, then run for real with no Makefile change.

REDOCLY_VERSION := 1.34.7
OPENAPI         := openapi.yaml
DIST            := dist
DOCS_OUT        := $(DIST)/api/index.html

# Rust crate manifest. Targets guard on its existence so they are safe to run
# before any Cargo.toml exists.
CARGO_MANIFEST  := Cargo.toml

.DEFAULT_GOAL := help

# ─────────────────────────────── help ────────────────────────────────────
.PHONY: help
help:
	@echo "vectorlink — make targets"
	@echo ""
	@echo "  make dev              Incremental DEBUG build (fast iteration)"
	@echo "  make dev-image        Assemble dev container from debug binary (seconds)"
	@echo "  make dev-up           dev + dev-image + docker compose up (full edit-run)"
	@echo "  make dev-up-release   RELEASE build + compose up (80x faster runtime)"
	@echo "  make build            Production RELEASE build (lto=thin, slow)"
	@echo "  make lint             Run ALL component linters (must pass before commit)"
	@echo "  make lint-openapi     Strict OpenAPI 3.1 lint (Redocly $(REDOCLY_VERSION))"
	@echo "  make clippy           Rust lint (-D warnings) — runs once the crate exists"
	@echo "  make test             Rust unit tests — run once the crate exists"
	@echo "  make test-integration Run mocha integration suite against DEBUG engine"
	@echo "  make docs             Generate reviewable API docs into $(DOCS_OUT)"
	@echo "  make docs-rebuild     Clean rebuild of the dist/api API docs"
	@echo "  make verify           lint + test (no side effects)"
	@echo "  make pr               Full pre-PR gate: lint + test + release build + docs"
	@echo "  make pr-light         Pre-PR gate without release build (fast, no Rust recompile)"
	@echo "  make server-start     Start vectorlink (7372) + TerminusDB (7373)"
	@echo "  make server-stop      Stop both test servers"
	@echo "  make server-restart   Restart both test servers"
	@echo "  make server-clean     Stop, wipe storage, and start fresh"
	@echo "  make server-status    Show status of both servers"
	@echo "  make build-image      Build the dev/CI container image (deps pre-baked)"
	@echo "  make fix-volumes      Remediate named volume ownership for non-root user"
	@echo "  make clean            Remove generated artifacts ($(DIST)/)"

# ─────────────────────────────── lint ────────────────────────────────────
# Aggregate linter — every component. Add new component linters as prereqs here.
# This is the gate that MUST pass before committing code.
.PHONY: lint
lint: lint-openapi clippy lint-tests
	@echo "✓ all linters passed"

# Strict OpenAPI lint. redocly.yaml promotes the recommended ruleset to errors
# and justifies the only two relaxations (localhost server, probe 4xx). Any
# error fails the build (exit non-zero) — this is the immediate hard gate.
.PHONY: lint-openapi
lint-openapi:
	npx -y @redocly/cli@$(REDOCLY_VERSION) lint $(OPENAPI)

# Rust lint. -D warnings makes every clippy/compiler warning a hard error.
# Safe Rust only: each crate carries `#![forbid(unsafe_code)]`, so any `unsafe`
# fails the build here; -D warnings also denies clippy's own unsafe/correctness
# lints. Introducing `unsafe` is a human decision (remove the forbid in a
# reviewed, signed commit), never a silent change.
# Guarded: no-ops with a notice until the crate exists.
# Build runs inside a pinned container image (Dockerfile.build) with all deps
# pre-baked. No apt-at-runtime, no GPG workarounds.
# TARGET_VOLUME: Docker volume for target/ (avoids virtiofs execute-bit issues on Lima,
# and provides fast overlay-backed I/O for incremental builds).
CARGO_VOLUME := vectorlink-cargo
TARGET_VOLUME := vectorlink-target
BUILD_IMAGE  := vectorlink-build:local

# Run containers as the host user so bind-mounted files are owned correctly.
# CARGO_HOME=/cargo-registry maps to the named cargo volume mount point.
# HOME=/tmp/build-home — the entrypoint script (baked into the image) ensures
# this directory exists before exec'ing the user's command.
DOCKER_RUN := docker run --rm \
	--user "$$(id -u):$$(id -g)" \
	-e HOME=/tmp/build-home \
	-e CARGO_HOME=/cargo-registry \
	-v "$$(pwd)":/work \
	-v $(CARGO_VOLUME):/cargo-registry \
	-v $(TARGET_VOLUME):/work/target \
	-w /work \
	$(BUILD_IMAGE)

# Detect Docker availability. When Docker (or the build image) is absent,
# cargo targets fall back to running on the host directly.
DOCKER_AVAILABLE := $(shell command -v docker >/dev/null 2>&1 && docker image inspect $(BUILD_IMAGE) >/dev/null 2>&1 && echo yes || echo no)

# CARGO_RUN: expands to the Docker wrapper when available, or plain cargo when not.
ifeq ($(DOCKER_AVAILABLE),yes)
CARGO_RUN := $(DOCKER_RUN)
else
CARGO_RUN :=
endif

.PHONY: clippy
clippy:
	@if [ -f $(CARGO_MANIFEST) ]; then \
		$(CARGO_RUN) cargo clippy --all-targets --all-features -- -D warnings ; \
	else \
		echo "• clippy skipped — no $(CARGO_MANIFEST) yet" ; \
	fi

# ─────────────────────────────── build ───────────────────────────────────
# BINARIES: the crate's two bin targets, copied to the host target/ after a
# build so artifacts land where developers expect (the build itself runs
# against the named volume for Lima speed; see copy-binaries below).
BINARIES := vectorlink vectorlink-load

# copy-binaries: copy built bin artifacts from the named target VOLUME out to
# the host ./target/$(PROFILE_DIR)/ so they are visible on the host. The build
# runs against the volume (fast, avoids Lima virtiofs exec-bit issues), which
# shadows host ./target inside the container — so without this step the host
# target/ never receives the binary. Mounts the volume read-only at /vol and
# the host target/ read-write at /host, then copies each bin if it exists.
# PROFILE_DIR is "debug", "release", or "production".
.PHONY: copy-binaries
copy-binaries:
	@mkdir -p target/$(PROFILE_DIR)
	@docker run --rm \
		--user "$$(id -u):$$(id -g)" \
		-v $(TARGET_VOLUME):/vol:ro \
		-v "$$(pwd)/target":/host \
		$(BUILD_IMAGE) \
		bash -c 'set -e; for b in $(BINARIES); do \
			if [ -f "/vol/$(PROFILE_DIR)/$$b" ]; then \
				mkdir -p "/host/$(PROFILE_DIR)"; \
				cp -f "/vol/$(PROFILE_DIR)/$$b" "/host/$(PROFILE_DIR)/$$b"; \
				echo "✓ target/$(PROFILE_DIR)/$$b"; \
			fi; \
		done'

# dev: incremental DEBUG build. Fast (seconds after first build). Use during
# development for edit-build-test cycles. Binary copied to host target/debug/.
.PHONY: dev
dev:
	@if [ -f $(CARGO_MANIFEST) ]; then \
		$(DOCKER_RUN) \
			cargo build ; \
		$(MAKE) copy-binaries PROFILE_DIR=debug ; \
	else \
		echo "• dev build skipped — no $(CARGO_MANIFEST) yet" ; \
	fi

# dev-image: assemble the dev/E2E container image from the pre-built debug
# binary. Strips debug symbols for the container copy (1.2GB → ~30MB) while
# keeping the full debug binary on host for local debugging/backtraces.
# This is a simple COPY into debian:trixie-slim — no cargo build runs inside
# the image. Requires target/debug/vectorlink to exist (run `make dev` first,
# or use `make dev-up` which chains both). Assembles in ~5 seconds.
.PHONY: dev-image
dev-image:
	@if [ ! -f target/debug/vectorlink ]; then \
		echo "ERROR: target/debug/vectorlink not found. Run 'make dev' first." >&2 ; \
		exit 1 ; \
	fi
	@echo "→ stripping debug binary for container (host copy unchanged)"
	@docker run --rm \
		--user "$$(id -u):$$(id -g)" \
		-v "$$(pwd)/target/debug":/host \
		$(BUILD_IMAGE) \
		bash -c 'cp /host/vectorlink /host/vectorlink-stripped && strip /host/vectorlink-stripped'
	docker build -f Dockerfile.dev -t vectorlink:dev .

# dev-up: the full edit-run cycle in one command. Builds the debug binary
# (incremental, seconds), assembles the dev container image, and brings up
# the compose stack. The compose override (docker-compose.override.yml) points
# vectorlink at Dockerfile.dev so `docker compose up --build` also works after
# `make dev` has produced the binary.
#
# WARNING: debug builds are ~80x SLOWER for compute-heavy paths (flat-KNN in
# /resolve, /duplicates) due to missing SIMD optimisation. For ANY performance
# testing or benchmarking, use `make dev-up-release` instead.
.PHONY: dev-up
dev-up: dev dev-image
	docker compose up -d vectorlink

# dev-up-release: same as dev-up but uses the RELEASE binary (optimised SIMD).
# Slower to build (~45s incremental vs ~5s) but 80x faster at runtime for
# vector compute paths. Use for: benchmarking, /resolve testing, E2E timing.
.PHONY: dev-up-release
dev-up-release: build
	@cp target/release/vectorlink target/debug/vectorlink-stripped
	docker build -f Dockerfile.dev -t vectorlink:dev .
	docker compose up -d vectorlink

# build: RELEASE build (no LTO). Fast enough for `make pr` gate
# (incremental relinks). LTO only in `make release-image` (production publish).
# Binary lands in host target/release/.
.PHONY: build
build:
	@if [ -f $(CARGO_MANIFEST) ]; then \
		cargo build --release ; \
	else \
		echo "• release build skipped — no $(CARGO_MANIFEST) yet" ; \
	fi

# release-image: PRODUCTION build with LTO + codegen-units=1 + opt-level=s.
# Slow (15-20 min). Run ONLY when cutting a shippable image, NOT per-commit.
# Binary copied to host target/production/.
.PHONY: release-image
release-image:
	@if [ -f $(CARGO_MANIFEST) ]; then \
		$(DOCKER_RUN) \
			cargo build --profile production ; \
		$(MAKE) copy-binaries PROFILE_DIR=production ; \
	else \
		echo "• release-image build skipped — no $(CARGO_MANIFEST) yet" ; \
	fi

# ─────────────────────────────── test ────────────────────────────────────
# Unit + integration tests. Guarded until the crate exists. The integration
# suite adopts the TerminusDB test infrastructure (see TESTING.md).
.PHONY: test
test:
	@if [ -f $(CARGO_MANIFEST) ]; then \
		$(CARGO_RUN) cargo test --all-features ; \
	else \
		echo "• test skipped — no $(CARGO_MANIFEST) yet" ; \
	fi

# Integration tests (mocha) against a LIVE vectorlink server. The server and
# Ollama must already be running — this target only runs the test suite.
# Set TDB_SEARCH_URL to point at a non-default endpoint (default: localhost:7372).
# Part of the `pr` gate.
.PHONY: test-integration
test-integration:
	@if [ -f $(CARGO_MANIFEST) ]; then \
		TDB_SEARCH_URL="${TDB_SEARCH_URL:-http://localhost:7372}" \
		TDB_SEARCH_ADMIN_USER="${TDB_SEARCH_ADMIN_USER:-admin}" \
		TDB_SEARCH_ADMIN_SECRET="${TDB_SEARCH_ADMIN_SECRET:-root}" \
		npx mocha --timeout 60000 ; \
	else \
		echo "• integration tests skipped — no $(CARGO_MANIFEST) yet" ; \
	fi

# E2e tests (mocha) — require BOTH vectorlink (7372) and TerminusDB (7373)
# to be running. These tests exercise the TerminusDB plugin endpoints that
# proxy to vectorlink. NOT part of the `pr` gate (run separately).
# Use `make server-start` to start both servers.
.PHONY: test-e2e
test-e2e:
	@TDB_SEARCH_URL="${TDB_SEARCH_URL:-http://localhost:7372}" \
		TDB_SEARCH_ADMIN_USER="${TDB_SEARCH_ADMIN_USER:-admin}" \
		TDB_SEARCH_ADMIN_SECRET="${TDB_SEARCH_ADMIN_SECRET:-root}" \
		npx mocha --config .mocharc.e2e.js

# Lint the JS test sources (eslint, TerminusDB-style). Part of `lint`.
.PHONY: lint-tests
lint-tests:
	npx eslint --ext .js tests/

# ────────────────────────────── docs ─────────────────────────────────────
# Generate the human-reviewable API reference from the OpenAPI contract into
# dist/ (gitignored). Open $(DOCS_OUT) in a browser to review it.
.PHONY: docs
docs:
	@mkdir -p $(DIST)/api
	npx -y @redocly/cli@$(REDOCLY_VERSION) build-docs $(OPENAPI) -o $(DOCS_OUT)
	@echo "✓ API docs → $(DOCS_OUT)"

# docs-rebuild: remove the previous output first, then regenerate — a clean
# rebuild of dist/api so no stale artifact can survive.
.PHONY: docs-rebuild
docs-rebuild:
	rm -rf $(DIST)/api
	@$(MAKE) docs

# ────────────────────────── aggregate gates ──────────────────────────────
# verify: lint + test, no artifacts written. Quick local check.
.PHONY: verify
verify: lint test
	@echo "✓ verify passed"

# pr-light: lint + unit tests + integration + e2e + docs. No release build,
# so no expensive Rust recompile. Use for rapid iteration on JS/tests.
.PHONY: pr-light
pr-light: lint test test-integration test-e2e docs
	@echo "✓ PR-light gate passed (no release build) — review $(DOCS_OUT), then commit (GPG-signed)"

# pr: the full pre-PR gate. Mirrors TerminusDB's `pr` target. Must be green
# before opening a PR. Delegates to pr-light, then adds the release build
# which proves the production binary compiles cleanly (but is NOT used for
# tests — debug binary is used).
.PHONY: pr
pr: pr-light build
	@echo "✓ PR gate passed — review $(DOCS_OUT), then commit (GPG-signed)"

# ──────────────────────── test server ────────────────────────────────────
# Manage the local test stack: vectorlink (7372) + TerminusDB (7373).
# Delegates to tests/vectorlink-server.sh which handles both servers.
.PHONY: server-start
server-start:
	tests/vectorlink-server.sh start

.PHONY: server-stop
server-stop:
	tests/vectorlink-server.sh stop

.PHONY: server-restart
server-restart:
	tests/vectorlink-server.sh restart

.PHONY: server-status
server-status:
	tests/vectorlink-server.sh status

.PHONY: server-clean
server-clean:
	tests/vectorlink-server.sh stop
	rm -rf /tmp/vectorlink-data
	@if [ -n "$$(tests/vectorlink-server.sh status 2>/dev/null | grep -o 'TerminusDB repo not found')" ]; then \
		echo "• TerminusDB storage clean skipped — repo not found" ; \
	else \
		TDB_ROOT="$$(cd "$$(pwd)/../terminusdb" 2>/dev/null && pwd)" ; \
		if [ -n "$$TDB_ROOT" ] && [ -f "$$TDB_ROOT/tests/terminusdb-test-server.sh" ]; then \
			"$$TDB_ROOT/tests/terminusdb-test-server.sh" clean ; \
		fi ; \
	fi
	tests/vectorlink-server.sh start

# ──────────────────────── container image ────────────────────────────────
# Build the dev/CI container image with all deps pre-baked. No apt-at-runtime.
# Safe to run any time — does NOT touch the shared cargo/target volumes.
.PHONY: build-image
build-image:
	docker build -f Dockerfile.build -t $(BUILD_IMAGE) .

# ──────────────────────── volume remediation ────────────────────────────
# One-time fix: chown the named volumes so the non-root build user can write.
# Idempotent — safe to run repeatedly. Uses a throwaway root container (--user
# root) to set ownership, then the non-root --user containers can do
# incremental builds.
.PHONY: fix-volumes
fix-volumes:
	@echo "→ remediating volume ownership for uid $$(id -u):$$(id -g)"
	docker run --rm --user root \
		-v $(CARGO_VOLUME):/cargo-registry \
		-v $(TARGET_VOLUME):/work/target \
		$(BUILD_IMAGE) \
		bash -c "chown -R $$(id -u):$$(id -g) /cargo-registry /work/target"
	@echo "✓ volumes owned by $$(id -u):$$(id -g)"

# ────────────────────────────── clean ────────────────────────────────────
.PHONY: clean
clean:
	rm -rf $(DIST)/api
	@echo "✓ cleaned $(DIST)/api"
