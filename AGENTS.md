# AGENTS.md

Short guidance for coding agents in this Rust workspace. Machine-checked
config is the source of truth; this file only carries the enforceable rules.

## Source of truth

- Crate list, deps, lints → root `Cargo.toml` (`cargo metadata --no-deps`).
- Task commands → `just` (run `just` to list). Tests run via `cargo nextest`.
- Issues → `trx`. Drift guard → `scripts/drift-check.sh`.
If they disagree with anything here, the command is right.

## Domain & architecture

- Read `CONTEXT.md` before domain work (XTR = schema-driven structured-data
  extraction, optimized by GEPA). Add ADRs in `docs/adr/` only for
  hard-to-reverse trade-offs.
- Layout: `xtr-core` is the library dependency root; `xtr-cli`/`xtr-api` depend
  on it but never on each other. Crates are named `xtr-*`.
- Never publish to a public registry (all `publish = false`).
- No backwards compatibility: break and replace in place; no "Modern"/"V2"
  parallel versions. Prefer enums/generics over type erasure.
- Target Windows 11, Linux, macOS 14+.

## Lints

`[workspace.lints.clippy]` is the source of truth. `unsafe_code` and
`unwrap`/`expect`/`panic`/`todo`/`dbg_macro` are denied; propagate with `?` /
`anyhow::Result` / `.context(...)`.

## Formats

Prefer **JSON/TOML** for config and machine output. Don't add `serde_yaml` as a
new dependency; leave existing YAML alone.

## Issue tracking

Use `trx`, never markdown TODOs. Commit `.trx/` changes with the code.

## House rules

- Do exactly what's asked; no unsolicited files.
- Keep README updates concise, emoji-free. Omit author headers/dates.
- Never commit secrets. `Cargo.lock` is committed (bump with manifests).
- Run `just check-all` (fmt + clippy + drift + tests) before committing.