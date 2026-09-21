# XTR

A glossary for the domain language used by this project. Add terms only after
their meaning has been resolved. `CONTEXT.md` is a glossary, not a spec,
implementation guide, or scratchpad — include only project-specific domain
concepts, not general programming terms.

## Language

**Extraction**:
Turning unstructured text (email, invoice, event copy) into structured data,
driven by a JSON schema. The core operation of XTR, exposed via `xtr get`.
_Avoid_: parse, scrape (these imply different pipelines)

**Schema**:
A JSON Schema describing the expected output shape (its fields, types, and
constraints). Defines the target structure for an extraction Task.
_Avoid_: spec (ambiguous), model (reserved for LLM models)

**Task**:
A named, configured extraction target — a schema plus its examples, model
routing, and optimization settings. Tasks are resolved from config by name
(`invoice_extraction`, `event`, ...). The unit of work across the CLI.
_Avoid_: job, project

**Signature**:
The dspy-rs declaration of a Task's input/output contract — the natural-language
instruction plus typed input/output fields that a predictor prompt is built
from.
_Avoid_: interface (ambiguous)

**GEPA (Guided Evolution of Prompts and Adapters)**:
The dspy-rs optimization algorithm XTR uses to iteratively refine the extraction
prompt/instruction, scoring candidate generations against examples to improve
quality over time.
_Avoid_: optimizer (generic), trainer

**Candidate**:
One generated instruction/signature variant produced during a GEPA run, scored
by an evaluator. The best candidate becomes the selected instruction.
_Avoid_: iteration (misleading), variant

**Generation**:
A batch of candidates produced in one GEPA evolution step.
_Avoid_: run (run = the whole optimization session)

**Rollout**:
One evaluation pass (prompt → model → prediction → scored against an example)
within a generation.
_Avoid_: attempt

**Instruction**:
The natural-language directive (prompt) that accompanies a Signature when
evaluating/extracting. Optimized by GEPA.
_Avoid_: prompt (colloquial)

**Teacher / Student**:
A two-tier model pairing. The teacher model (stronger) produces evaluations or
helpful signals during optimization; the student model performs the actual
extraction at inference time. Configurable independently.
_Avoid_: main, primary (ambiguous)

**Fallback Model**:
An additional model used when the primary (teacher/student) fails or is
unavailable, in the multi-model failure chain.
_Avoid_: backup

**Validation**:
Checking a prediction against the schema during inference with a configured
mode — `none`, `warn`, or `error`. `error` rejects invalid output and triggers
retries.
_Avoid_: check (ambiguous)

**Retry**:
Retrying an extraction when validation fails. Two modes: `independent` (fresh
attempts) and `chat`/`shots` (the model learns from prior errors via chat
history).
_Avoid_: re-run (ambiguous)

**Predictor**:
The dspy-rs module that executes a Signature against a model (LM) and returns a
Prediction. Wrapped by XTR's `ExtractionEngine`.
_Avoid_: extractor (ambiguous)

**Example**:
A labeled input/output pair (input text + expected structured output) used to
build demo messages and to evaluate candidates during GEPA. Stored in JSONL or
JSON-dir format.
_Avoid_: sample, fixture

**TaskExample / RawTaskExample**:
XTR's typed representations of a training/optimization example, distinct from a
raw dspy-rs `Example`.
_Avoid_: row (ambiguous)

**LM (Language Model)**:
A model handle wrapping a dspy-rs `LM`/`ChatAdapter`, plus caching and verbose
logging. Resolved from a `ResolvedModelConfig`.
_Avoid_: model (also a config field), client

**Adapter**:
The dspy-rs adapter used to format input/output for a model (chat-based, etc.).
XTR selects one per model via its `adapter` config field.
_Avoid_: connector

**ModelDescriptor**:
A resolved, validated model configuration (name, base URL, API key, adapter,
timeout, max tokens) ready to build a model handle.
_Avoid_: model config (that is the unresolved spec)

**Metrics**:
Per-run observation data (scores, token/rollout counts, candidate history)
logged to a JSON metrics file and optionally to MLflow.
_Avoid_: logs (ambiguous with file logs)

**Optimization Run**:
A single GEPA optimization session for a Task, tracked with a run id, metrics,
and a selected best instruction. Addressable by id from the CLI.
_Avoid_: optimization (the mechanism), session

**Dataset Generator**:
A strong model used to synthesize training examples for the built-in
`example_generator` task when no examples exist.
_Avoid_: generator

**Run ID**:
The unique identifier of an optimization run, used to activate/select the best
instruction for a task.
_Avoid_: run (ambiguous)