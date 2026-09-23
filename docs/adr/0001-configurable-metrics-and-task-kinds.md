# Configurable extraction metrics, task kinds, and judge-as-task

XTR currently scores every extraction with a single hardcoded
schema-field-match heuristic (`ExtractionProgram::feedback_metric`). Users
cannot express a different grading philosophy (exact-match, an LLM judge, or an
external scorer), and the judge itself is not something XTR can optimize. We
decided to make the metric configurable per task and to model a grading task as
just another optimizable task, so GEPA can refine both the extractor prompt and
the judge (rubric) prompt.

## Decision

Add a per-task `kind` and a configurable metric.

1. **Task kind** (`extract` default, `grade`): a `grade` task has the same
   `ExtractionProgram`/GEPA machinery as `extract`, but its signature is
   `(expected_output, predicted_output, additional_context) -> verdict`, and its
   output contract is a `verdict_schema`. It is optimized identically to any
   extraction task.

2. **`MetricKind`** (`schema_field` default, `exact`, `llm_judge`, `command`):
   selected per task and resolved into `ExtractionProgram`'s
   `feedback_metric`, which dispatches on the kind. `llm_judge` runs a reusable
   grade task (optionally its optimized instruction) as the scorer; `command`
   shells out to a subprocess and validates its stdout against
   `verdict_schema`. Both accept a `verdict_schema` (default
   `{score, feedback, reasons[]}`); the rubric/instruction is plain text
   (string or `@file`).

3. **Verdict schema** is a JSON Schema for the judge/command *output* (not the
   rubric, which is prose instructions and is never schema-validated).

## Consequences

- `schema_field` keeps the existing heuristic (backward compatible).
- A `grade` task is optimized with the existing `xtr optimize <task>` path, so
  a user can refine a rubric and then use that optimized instruction as the
  judge for a different extraction task — "run XTR with different judge rubrics
  to get different optimized extractor prompts".
- New metric kinds are additive; existing configs resolve as before.
- `command` metrics must degrade gracefully (non-zero exit / malformed stdout ->
  fallback score, never crash optimization).