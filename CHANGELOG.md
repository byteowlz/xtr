# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Build

- Conformance to byteowlz workspace standards: added `CONTEXT.md`, `docs/adr/`
  (template + README), `clippy.toml`, `.ast-grep/rules/`, and
  `scripts/drift-check.sh`; rewrote `AGENTS.md` with source-of-truth,
  strict-lint, and JSON/TOML-preference guidance; added workspace lints and the
  `just check-all` gate.
- Added `[workspace.dependencies]`/`[workspace.package]` and per-crate
  `[lints] workspace = true`; crates inherit version/license/repository from
  the workspace root.

### Fixes

- Removed production `unwrap`/`expect` in `xtr-cli` and `xtr-core`
  (`strip_think_tags` now uses a lazily-initialized static regex, `log_dir`
  writes are guarded, `file_name` uses a safe fallback).
- Replaced the `unsafe` libc `getrlimit`/`setrlimit` calls in `increase_fd_limit`
  with the safe `rustix` API.
- Allowed `unsafe`/`unwrap`/`expect` only in `#[cfg(test)]` builds (test code
  isolates env state via `env::set_var` and unwraps fixtures).

## 2026.1.25

### Highlights

- CLI: Add JSONL format support for examples with automatic generation from existing TOML files.
- Build: Add `install-all` recipe for streamlined multi-crate installation.
- CI: Add release.toml for cargo-release automation.

### Changes

- Examples: Migrate title examples to JSONL format for better tooling compatibility.
- CLI: Add `--format jsonl` flag for example generation and processing.
- Build: Add `just install-all` recipe with cargo install script for all workspace crates.

### Fixes

- Build: Clippy fixes and general code cleanup.

## 2025.12.22

### Changes

- Scripts: Add schema sync script for pulling schemas from central repository.

## 2025.11.17

### Highlights

- Core: Add verbose JSON logging for debugging extraction pipelines.
- Core: Update to new rig/dspy-rs API for improved model compatibility.

### Changes

- Logging: Add verbose mode with detailed JSON output for inference debugging.

### Fixes

- Core: Fix compatibility with new rig/dspy-rs API changes.
- Optimization: Fix optimization metric calculations and convergence issues.

## 2025.11.2

### Highlights

- CLI: Add automatic JSON schema generation from example files (`xtr create schema`).
- Optimization: Make optimization metrics configurable via config.toml.

### Changes

- CLI: Add `xtr create schema` command to generate JSON schemas from example JSON files.
- Config: Add `[optimization.defaults]` and `[optimization.tasks.*]` configuration sections.
- Config: Support per-task optimization parameter overrides (iterations, batch_size, temperature, etc.).

## 2025.11.1

### Highlights

- CLI: Add schema validation with configurable modes (none, warn, error).
- CLI: Add retry logic with independent attempts or LLM learning from errors (shots mode).

### Changes

- CLI: Add `--validate` flag with modes: none, warn, error.
- CLI: Add `--retry` flag for automatic retries on validation failure.
- CLI: Add `--shots` flag for LLM error learning via chat history context.

## 2025.10.24

### Highlights

- Initial release of XTR - structured data extraction engine powered by GEPA.

### Changes

- Core: Schema-driven extraction with JSON schema support.
- Core: Multi-model architecture with teacher/student fallback.
- Core: GEPA optimization algorithm for automatic prompt refinement.
- Core: Support for OpenAI, Anthropic, and local LLMs (LM Studio, Ollama).
- CLI: `xtr get` command for extraction from stdin.
- CLI: `xtr optimize` command for running GEPA optimization.
- CLI: `xtr history`, `xtr activate`, `xtr compare`, `xtr clean` for optimization management.
- Config: XDG-compliant configuration and storage paths.
- Config: Task definitions with schema, examples, and description.
- Logging: MLflow integration for experiment tracking.
- Examples: Bundled schemas for contact_details, event, invoice, resume, and more.
