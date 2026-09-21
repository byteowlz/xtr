#!/usr/bin/env bash
#
# Drift check (tmpl-8adx): verify that the facts this repo documents
# (commands, crate/version references, JSON/TOML-only constraint) match the
# machine-checked configuration (the actual Cargo.toml manifests and PATH).
#
# Anything this script checks is "source of truth" in AGENTS.md — if it fails,
# either the code or the documentation drift must be fixed.

set -u
FAILED=0

echo "==> Required commands on PATH"
for cmd in cargo just trx python3; do
  if command -v "$cmd" >/dev/null 2>&1; then
    echo "  ok: $cmd -> $(command -v "$cmd")"
  else
    echo "  FAIL: required command not found on PATH: $cmd"
    FAILED=1
  fi
done

echo "==> No new serde_yaml dependency (prefer JSON/TOML)"
if rg -l 'serde_yaml|serde-yaml' --glob 'Cargo.toml' . >/dev/null 2>&1; then
  echo "  FAIL: found 'serde_yaml' in a Cargo.toml (prefer JSON/TOML; leave existing YAML alone)"
  rg -n 'serde_yaml|serde-yaml' --glob 'Cargo.toml' . || true
  FAILED=1
else
  echo "  ok: no serde_yaml dependency in any manifest"
fi

echo "==> No external 'config' crate (use plain toml + XDG paths)"
if rg -n '^config ' --glob 'Cargo.toml' . >/dev/null 2>&1; then
  echo "  FAIL: found external 'config' crate dependency (prefer toml + hand-rolled XDG)"
  rg -n '^config ' --glob 'Cargo.toml' . || true
  FAILED=1
else
  echo "  ok: no external config crate dependency"
fi

echo "==> toml crate present for config parsing"
if rg -n '^toml ' --glob 'Cargo.toml' . >/dev/null 2>&1; then
  echo "  ok: toml crate present"
else
  echo "  FAIL: no toml crate dependency for config parsing"
  FAILED=1
fi

echo "==> Workspace resolves (cargo metadata)"
if cargo metadata --no-deps --format-version 1 >/dev/null 2>&1; then
  echo "  ok: cargo metadata resolves"
else
  echo "  FAIL: cargo metadata --no-deps does not resolve"
  FAILED=1
fi

echo
if [[ "$FAILED" -eq 0 ]]; then
  echo "drift-check: PASS"
else
  echo "drift-check: FAIL"
  exit 1
fi