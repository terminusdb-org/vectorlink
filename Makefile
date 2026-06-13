# tdb-search — verification & build orchestration.
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
	@echo "tdb-search — make targets"
	@echo ""
	@echo "  make dev              Incremental DEBUG build (fast iteration)"
	@echo "  make build-release    Production RELEASE build (lto=thin, slow)"
	@echo "  make lint             Run ALL component linters (must pass before commit)"
	@echo "  make lint-openapi     Strict OpenAPI 3.1 lint (Redocly $(REDOCLY_VERSION))"
	@echo "  make clippy           Rust lint (-D warnings) — runs once the crate exists"
	@echo "  make test             Rust unit tests — run once the crate exists"
	@echo "  make test-integration Run mocha integration suite against DEBUG engine"
	@echo "  make docs             Generate reviewable API docs into $(DOCS_OUT)"
	@echo "  make docs-rebuild     Clean rebuild of the dist/api API docs"
	@echo "  make verify           lint + test (no side effects)"
	@echo "  make pr               Full pre-PR gate: lint + test + release build + docs"
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
# Build runs inside a rust:1-bookworm container (no host toolchain).
CARGO_VOLUME := tdb-search-cargo

.PHONY: clippy
clippy:
	@if [ -f $(CARGO_MANIFEST) ]; then \
		docker run --rm \
			-v "$$(pwd)":/work \
			-v $(CARGO_VOLUME):/usr/local/cargo/registry \
			-w /work \
			rust:1-bookworm \
			bash -c "apt-get update -qq && apt-get install -yqq protobuf-compiler libprotobuf-dev >/dev/null 2>&1 && rustup component add clippy 2>/dev/null && cargo clippy --all-targets --all-features -- -D warnings" ; \
	else \
		echo "• clippy skipped — no $(CARGO_MANIFEST) yet" ; \
	fi

# ─────────────────────────────── build ───────────────────────────────────
# dev: incremental DEBUG build. Fast (seconds after first build). Use during
# development for edit-build-test cycles.
.PHONY: dev
dev:
	@if [ -f $(CARGO_MANIFEST) ]; then \
		docker run --rm \
			-v "$$(pwd)":/work \
			-v $(CARGO_VOLUME):/usr/local/cargo/registry \
			-w /work \
			rust:1-bookworm \
			bash -c "apt-get update -qq && apt-get install -yqq protobuf-compiler libprotobuf-dev >/dev/null 2>&1 && cargo build" ; \
	else \
		echo "• dev build skipped — no $(CARGO_MANIFEST) yet" ; \
	fi

# build-release: production RELEASE build with lto=thin. Slow (minutes). Only
# run at the end of an iteration cycle via `make pr`.
.PHONY: build-release
build-release:
	@if [ -f $(CARGO_MANIFEST) ]; then \
		docker run --rm \
			-v "$$(pwd)":/work \
			-v $(CARGO_VOLUME):/usr/local/cargo/registry \
			-w /work \
			rust:1-bookworm \
			bash -c "apt-get update -qq && apt-get install -yqq protobuf-compiler libprotobuf-dev >/dev/null 2>&1 && cargo build --release" ; \
	else \
		echo "• release build skipped — no $(CARGO_MANIFEST) yet" ; \
	fi

# ─────────────────────────────── test ────────────────────────────────────
# Unit + integration tests. Guarded until the crate exists. The integration
# suite adopts the TerminusDB test infrastructure (see TESTING.md).
.PHONY: test
test:
	@if [ -f $(CARGO_MANIFEST) ]; then \
		docker run --rm \
			-v "$$(pwd)":/work \
			-v $(CARGO_VOLUME):/usr/local/cargo/registry \
			-w /work \
			rust:1-bookworm \
			bash -c "apt-get update -qq && apt-get install -yqq protobuf-compiler libprotobuf-dev >/dev/null 2>&1 && cargo test --all-features" ; \
	else \
		echo "• test skipped — no $(CARGO_MANIFEST) yet" ; \
	fi

# Integration tests (mocha) against a LIVE server. Self-contained: builds the
# engine, starts it in a container on a free host port, waits for readiness,
# runs the suite, and tears down. Part of the `pr` gate.
.PHONY: test-integration
test-integration:
	@if [ -f $(CARGO_MANIFEST) ]; then \
		CARGO_VOLUME=$(CARGO_VOLUME) ./scripts/run-integration-tests.sh ; \
	else \
		echo "• integration tests skipped — no $(CARGO_MANIFEST) yet" ; \
	fi

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

# pr: the full pre-PR gate. Mirrors TerminusDB's `pr` target. Must be green
# before opening a PR. Runs lint + unit tests + integration tests (debug) +
# production release build + docs. The release build proves the production
# binary compiles cleanly but is NOT used for tests (debug binary is used).
.PHONY: pr
pr: lint test test-integration build-release docs
	@echo "✓ PR gate passed — review $(DOCS_OUT), then commit (GPG-signed)"

# ────────────────────────────── clean ────────────────────────────────────
.PHONY: clean
clean:
	rm -rf $(DIST)/api
	@echo "✓ cleaned $(DIST)/api"
