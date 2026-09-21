set positional-arguments

# Display help
help:
    just -l

# format code
fmt:
    cargo fmt -- --config imports_granularity=Item

fix *args:
    cargo clippy --fix --all-features --tests --allow-dirty "$@"

clippy:
    cargo clippy --all-features --tests -- -D warnings "$@"

# Full gate: format + clippy (deny warnings) + drift + tests
check-all:
    cargo fmt --check -- --config imports_granularity=Item
    cargo clippy --workspace --all-features --tests -- -D warnings
    ./scripts/drift-check.sh
    cargo test --workspace --all-features

# Drift check: verify documented facts match manifests/PATH
drift-check:
    ./scripts/drift-check.sh

install:
    rustup show active-toolchain
    cargo fetch

# Install the xtr binary
install-all:
    ./scripts/install-xtr.sh

# Run `cargo nextest` since it's faster than `cargo test`, though including
# --no-fail-fast is important to ensure all tests are run.
#
# Run `cargo install cargo-nextest` if you don't have it installed.
test:
    cargo nextest run --no-fail-fast
