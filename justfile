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

# Run the workspace test suite with default test's fail-fast on the first
# failing binary. Matches the `check-all` gate.
test:
    cargo test --workspace --all-features
