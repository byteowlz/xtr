# XTR

Structured data extraction engine powered by GEPA (Guided Evolution of Prompts and Adapters).

## Overview

XTR extracts structured data from unstructured text using language models and automatically optimizes extraction quality through iterative prompt refinement.

## Features

- **Schema-driven extraction**: Define JSON schemas for your data structures
- **Schema validation**: Validate outputs during inference with configurable modes (none, warn, error)
- **Smart retries**: Automatic retry with independent attempts or LLM learning from errors
- **Auditable inference**: Typed receipts, termination checks, request-local logging and explicit fallback grants
- **Vision inputs**: Lossless file/base64/stdin images in multimodal messages
- **Auto-optimization**: GEPA algorithm improves extraction quality over time
- **Compatible endpoints**: OpenAI-compatible cloud/local LM and VLM APIs (including LM Studio, Ollama and gateways)
- **MLflow integration**: Track optimization experiments and results

## Quick Start

```bash
# Generate a schema from a JSON file
xtr create schema example.json --name my_schema
# Or let it use the filename automatically
xtr create schema example.json

# Extract contact details
echo "Contact John at john@example.com" | xtr get contact_details

# Extract event information
cat event_announcement.txt | xtr get event

# Extract with schema validation (error on invalid output)
echo "Contact John" | xtr get contact_details --validate error

# Extract with retries (up to 3 attempts if validation fails)
cat input.txt | xtr get event --validate error --retry 3

# Extract with shots mode (LLM learns from errors via chat history)
cat input.txt | xtr get event --validate error --retry 3 --shots

# Override max_tokens for this request
cat large_input.txt | xtr get event --max-tokens 4096

# Inspect private provenance (raw completion, attempts, validation and usage)
xtr get event --text "Synthetic event" --receipt

# Explicitly permit the configured teacher destination after student failure
xtr get event --text "Synthetic event" --allow-fallback teacher

# Image plus text, or binary image stdin
xtr get screen_context --text "Describe this screen" --image screen.png
ctx capture | xtr get screen_context --image stdin

# Optimize extraction for a task
xtr optimize contact_details
```

## Configuration

Configuration is stored in `~/.config/xtr/config.toml` (or `$XDG_CONFIG_HOME/xtr/config.toml`).

### Model Configuration

```toml
[models.defaults.teacher]
name = "gpt-4o-mini"
base_url = "https://api.openai.com/v1"
api_key = "sk-..."
adapter = "chat"
max_tokens = 4096  # Optional: set max completion tokens

[models.defaults.student]
name = "qwen3-coder-30b"
base_url = "http://localhost:1234/v1"
api_key = "not_needed"
adapter = "chat"
max_tokens = 2048  # Optional: can be overridden per-request with --max-tokens
```

### Using Local Models (LM Studio, Ollama)

Inference sends the exact configured model name to the configured OpenAI-compatible
`base_url`; prefixes/slashes do not change the destination. Provider-native SDK
compatibility is not implied by an OpenAI-compatible endpoint. Optimization still
uses DSPy's provider client behavior.

Example for LM Studio:

```toml
[models.defaults.student]
name = "qwen3-coder-30b"  # Without qwen/ prefix
base_url = "http://localhost:1234/v1"
api_key = "not_needed"
adapter = "chat"
```

Example for Ollama (works perfectly):

```toml
[models.defaults.student]
name = "llama3.2:3b"  # No provider prefix needed
base_url = "http://localhost:11434/v1"
api_key = "not_needed"
adapter = "chat"
```

### Context Length Considerations

Large inputs (HTML pages, long documents) may exceed your model's context window. If you encounter errors like "Incorrect API key provided" when processing large files, this is often a **context overflow** error being misinterpreted.

**Solutions:**

1. **Increase context in LM Studio**:
   - Open LM Studio → Click the loaded model
   - Find "Context Length" slider
   - Increase to 32768 or higher
   - Reload the model

2. **Preprocess input** - Extract relevant text before piping to xtr:

```bash
# Extract only relevant sections
curl -s https://example.com/contact | grep -A 100 "contact" | head -500 | xtr get contact_details

# Convert HTML to text first (requires pandoc)
curl -s https://example.com/page.html | pandoc -f html -t plain | xtr get contact_details

# Use pup to extract specific HTML elements
curl -s https://example.com | pup '.contact-info text{}' | xtr get contact_details
```

3. **Use a model with larger context**: Switch to models supporting 32K+ tokens (e.g., Qwen2.5, Llama 3.1)

## JSON Schemas

This project uses JSON schemas from the [byteowlz/schemas](https://github.com/byteowlz/schemas) repository.

### Syncing Schemas

Pull the latest schemas from the central repository:

```bash
./scripts/sync-schemas.sh
```

This will fetch all extraction schemas (contact_details, event, invoice, etc.) to the `examples/schemas/` directory.

### Validating Examples

Ensure all examples match their schemas:

```bash
./scripts/validate-schemas.sh
```

### Schemas Used

All extraction schemas from the central repository:
- `contact_details` - Contact information extraction
- `event` - Event extraction
- `contract` - Contract information
- `customer_feedback` - Customer feedback
- `email_triage` - Email categorization
- `invoice` - Invoice data
- `meeting_notes` - Meeting notes
- `movie` - Movie information
- `org` - Organization details
- `resume` - Resume/CV parsing

See [SCHEMA_REGISTRY.md](https://github.com/byteowlz/schemas/blob/main/SCHEMA_REGISTRY.md) for the complete schema reference.

## Schema Generation

XTR can automatically generate JSON schemas from example JSON files:

```bash
# Generate schema from a JSON file (uses filename as schema name)
xtr create schema examples/data/contact.json

# Specify a custom schema name
xtr create schema examples/data/contact.json --name contact_details

# The generated schema will be saved to examples/schemas/<name>.json
```

Generated schemas include:
- Inferred types for all fields (string, number, boolean, array, object)
- Required field detection
- Nested object support
- Array item type inference

You can then reference the generated schema in your task configuration or edit it manually to add validation rules, descriptions, or constraints.

## Task Configuration

Define tasks in your config:

```toml
[tasks."contact_details"]
schema = "$XDG_DATA_HOME/xtr/schemas/contact_details.json"
examples = "$XDG_DATA_HOME/xtr/examples/contact_details"
description = "Extract contact information from the input."

[tasks."event"]
schema = "$XDG_DATA_HOME/xtr/schemas/event.json"
examples = "$XDG_DATA_HOME/xtr/examples/event"
description = "Extract event details from the input."
include_timestamp = true
```

## Automatic Example Collection

Enable data collection to capture successful inference calls as ready-to-review JSON examples:

```toml
[data_collection]
enabled = true
output_dir = "$XDG_STATE_HOME/xtr/collected_examples"
```

Successful calls save private, atomic **unreviewed candidates** with receipt provenance,
not `expected_json` training labels. Failed completion content and source media are
not collected. Existing records are not migrated. Training loaders reject candidate
records; independently review labels before explicitly approving a separate example:

```bash
xtr examples approve event --candidate candidate.json --expected reviewed-labels.json \
  --reviewer reviewer-id --output-dir private-approved-examples
```

New Unix directories/files are 0700/0600; existing non-private directories or
symlink paths are rejected rather than rewritten. Capture failures are reported
as warnings without misrepresenting successful extraction.

## Optimization

Run GEPA optimization to improve extraction:

```bash
# Basic optimization
xtr optimize contact_details

# Override parameters via CLI flags
xtr optimize contact_details --iterations 20 --batch-size 50 --temperature 0.9

# Quick testing (fast iterations)
xtr optimize event --iterations 3 --batch-size 3 --max-rollouts 100

# Production optimization (best results)
xtr optimize contact_details \
  --iterations 20 \
  --batch-size 50 \
  --rollouts-per-iteration 10 \
  --temperature 0.9 \
  --max-rollouts 1000

# Budget-constrained optimization
xtr optimize event --max-lm-calls 200 --max-rollouts 100
```

### Managing Optimization Runs

After running optimizations, use these commands to manage your results:

```bash
# View optimization history for a task
xtr history contact_details

# View history for all tasks with detailed metrics
xtr history --detailed

# Activate a specific optimization run
xtr activate contact_details 2025-10-28/contact_details_20251028_141425

# Compare two optimization runs
xtr compare contact_details \
  2025-10-28/contact_details_20251028_141425 \
  2025-10-29/contact_details_20251029_091234

# Clean old optimization runs (dry run)
xtr clean --older-than 30 --dry-run

# Actually delete old runs
xtr clean --older-than 30
```

The latest optimization automatically becomes the active prompt. Use `xtr activate` to switch between different optimization runs.

### Optimization Parameters

Configure defaults in `config.toml` or override via CLI flags:

```toml
[optimization.defaults]
# Number of evolutionary iterations (default: 4)
iterations = 4

# Number of trials per iteration for averaging stochastic LM behavior (default: 6)
rollouts_per_iteration = 6

# Maximum total language model API calls budget (default: 32)
max_lm_calls = 32

# Batch size for evaluation - larger = better signal but slower (default: 4)
batch_size = 4

# Maximum total rollouts budget - hard constraint on evaluations (optional)
# max_rollouts = 100

# Temperature for LLM-based mutations (default: 0.9)
# Higher values (>1.0) = more creative/exploratory mutations
# Lower values (<1.0) = more conservative mutations
# Range: 0.0-2.0
temperature = 0.9

# Track detailed optimization statistics (default: true)
# Set to false to reduce memory overhead
track_stats = true

# Track best outputs for inference-time search (default: false)
# Enable to store best outputs found during evolution
track_best_outputs = false

[optimization.tasks."contact_details"]
# Override any parameter for specific tasks
iterations = 6
rollouts_per_iteration = 8
temperature = 1.2
```

### Available CLI Flags

All optimization parameters can be overridden via command-line flags:

- `--iterations <N>` - Number of optimization iterations
- `--batch-size <N>` - Batch size for evaluation
- `--rollouts-per-iteration <N>` - Number of trials per iteration
- `--max-lm-calls <N>` - Maximum total LM calls budget
- `--max-rollouts <N>` - Maximum total rollouts budget
- `--temperature <F>` - Temperature for LLM mutations (0.0-2.0)
- `--track-stats <BOOL>` - Track detailed statistics (true/false)
- `--track-best-outputs <BOOL>` - Track best outputs (true/false)

### Parameter Selection Guidelines

**Development/Testing:**
```bash
xtr optimize task --iterations 3 --batch-size 3 --max-rollouts 50
```
Fast iterations for quick feedback during development.

**Production/Best Results:**
```bash
xtr optimize task --iterations 20 --batch-size 50 --temperature 0.9 --max-rollouts 1000
```
Larger batches and more iterations for optimal extraction quality.

**Budget-Constrained:**
```bash
xtr optimize task --max-lm-calls 200 --max-rollouts 100
```
Hard limits on API calls and evaluations to control costs.

**Creative Exploration:**
```bash
xtr optimize task --temperature 1.3 --iterations 15
```
Higher temperature for exploring creative mutations when stuck in local optima.

**Conservative Refinement:**
```bash
xtr optimize task --temperature 0.7 --iterations 10
```
Lower temperature for refining near-optimal solutions.

## Schema Validation & Retries

XTR supports runtime schema validation during inference with configurable retry behavior.

### Response Admission

Complete JSON is parsed before any envelope handling, preserving literals. Only
one strict outer `<think>...</think>` block and/or complete lowercase `json` fence
is supported; prose fragments, truncated or multiple payloads are rejected.
`length` and refusal responses fail even when their JSON validates. Missing finish
metadata is unknown and rejected unless `--allow-unknown-termination` explicitly
accepts an unverified receipt. Student-only is the default; fallbacks need explicit
`--allow-fallback teacher` or `--allow-fallback fallback:N` grants.

See [the extraction contract and regression evidence](docs/adr/0002-request-owned-extraction.md)
for library migration, privacy, retries, capture and vision boundaries.

### Validation Modes

**None (default)**: No schema validation; JSON/termination admission still applies
```bash
xtr get contact_details < input.txt
```

**Warn**: Validate and show warning if validation fails, but still return the output
```bash
xtr get contact_details --validate warn < input.txt
```

**Error**: Validate and return error if validation fails
```bash
xtr get contact_details --validate error < input.txt
```

### Retry Strategies

**Independent Retries (`--retry`)**: Each retry is independent, no chat history
```bash
# Retry up to 3 times without context
xtr get event --validate error --retry 3 < input.txt
```

Each attempt starts fresh without knowledge of previous failures. Useful when the model has stochastic behavior and might succeed on a different sample.

**Shots Mode (`--shots`)**: LLM learns from previous errors via chat history
```bash
# Retry up to 3 times with error feedback in context
xtr get event --validate error --retry 3 --shots < input.txt
```

Each retry includes the previous validation error in the context, allowing the LLM to learn from its mistakes and correct the output. Better for systematic errors where the model needs guidance.

### Examples

```bash
# Quick validation without retries
echo "Contact: john@example.com" | xtr get contact_details --validate error

# Robust extraction with independent retries
cat complex_input.txt | xtr get event --validate error --retry 5

# Learning from errors with shots mode
cat tricky_format.txt | xtr get contact_details --validate error --retry 3 --shots

# Warning mode for debugging (non-blocking)
xtr get event --validate warn --verbose < input.txt
```

## Storage Locations

XTR follows the XDG Base Directory Specification:

### Configuration (`~/.config/xtr/` or `$XDG_CONFIG_HOME/xtr/`)
- `config.toml` - Main configuration file (models, tasks, optimization defaults)

### Data (`~/.local/share/xtr/` or `$XDG_DATA_HOME/xtr/`)
- `schemas/` - JSON schemas for tasks
- `examples/` - Training examples (TOML files)

### State (`~/.local/state/xtr/` or `$XDG_STATE_HOME/xtr/`)
- `prompts/` - Active prompts for inference
  - `{task}/optimized_instruction.txt` - Current active prompt
  - `{task}/metadata.json` - Tracks which optimization run is active
- `optimizations/` - Historical optimization runs
  - `YYYY-MM-DD/{task}_YYYYMMDD_HHMMSS/` - Individual run directories containing:
    - `config.json` - Optimization parameters
    - `metrics.json` - Scores and improvements
    - `optimized_instruction.txt` - The optimized prompt
    - `result.json` - Full GEPA result with evolution history
    - `summary.txt` - Human-readable summary
- `optimization_logs/` - Detailed logs (optional, for debugging)
- `cache/` - LM response cache

## MLflow Tracking

Enable experiment tracking:

```toml
[mlflow]
local_logging = true
log_dir = "~/.local/state/xtr/optimization_logs"
tracking_uri = "http://localhost:5000"
experiment_name = "xtr-optimization"
```

## Troubleshooting

### Error: "Incorrect API key provided" with local models

This error from local models (LM Studio, Ollama) usually indicates:

- **Context length exceeded** - The input is too large for your model's context window (see Context Length Considerations above)
- **Model server not running** - Verify with `curl http://localhost:1234/v1/models`
- **Wrong base_url in config** - Check your config.toml has the correct endpoint

### Error: "model response returned empty 'output_json' string"

The model returned a response but didn't follow the expected JSON format. Try:

- Optimizing the task: `xtr optimize contact_details`
- Using a different or larger model
- Adding more examples to your task's examples directory

### Model names with provider prefixes (e.g., `qwen/`, `anthropic/`)

For inference, use the exact model ID accepted by the configured compatible
endpoint. XTR does not reinterpret its prefix or silently select another destination.

## Development

```bash
# Build
cargo build

# Run tests
cargo test

# Format code
just fmt

# Run linter
just fix -p xtr-cli
```

## License

MIT License
