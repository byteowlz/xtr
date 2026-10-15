# ADR 0002: Request-owned extraction and explicit admission

## Decision

Implement the hardening requested in xtr-38z4 and image inputs in trx-ev42 at
XTR's existing inference seam. `run_inference` returns `ExtractionReceipt` or
`ExtractionError`; `InferenceOptions` supplies validation, retry, destination,
termination and logging policy. The former string-returning overloads are
replaced, not maintained in parallel. CLI stdout still defaults to parsed JSON;
`--receipt` explicitly requests private audit evidence.

DSPy's process-global configure/forward transaction cannot isolate concurrent
requests, its text-only Chat cannot carry images, and its LMResponse drops
finish metadata. Inference therefore reuses XTR/DSPy prompt formatters but sends
request-owned OpenAI-compatible messages directly, with the existing configured
model/base URL. No new server, provider routing platform, or global network lock
is introduced. Redirects are disabled. DSPy optimization still installs globals
and is not qualified for concurrent optimization; inference never reads them.
Direct generator prediction binds its LM/adapter rather than configuring globals.

## Policy and migration

- Student-only is the default. Configuration alone does not grant teacher or
  fallback access. A grant matches role, exact configured name, and exact base
  URL. CLI `--allow-fallback teacher` / `--allow-fallback fallback:N` resolves that
  identity for this invocation (zero-based fallback index). Library callers
  obtain identities with `fallback_identity` and populate `allowed_fallbacks`.
- Retry/fallback classes are explicit in options. Defaults permit transport,
  malformed completion, schema rejection, and truncation recovery, but only
  granted destinations are called. Refusal, missing/unsupported termination,
  configuration failure and credential echoes are not recoverable by default.
- Only provider-reported `stop` is admitted by default. `length` is rejected
  even when JSON validates; refusal/content-filter and tool termination are
  rejected, including null/non-text content and malformed ancillary metadata.
  Missing/null metadata remains `unknown`; wrongly typed finish metadata is
  malformed, not an unknown-metadata opt-in. Explicit
  `allow_unknown_termination` / `--allow-unknown-termination` yields an
  `unverified` receipt, never an invented normal stop.
- JSON parsing always precedes envelope handling. Any complete JSON root is
  supported. Only one exact non-nested leading `<think>...</think>` block and
  then optionally one complete lowercase `json` fence are supported (LF/CRLF
  delimiters on separate lines). No prose/brace fragments, repeated/nested
  envelopes, multiple roots, truncation or literal repair. A wrapped
  `output_json` string must itself contain complete JSON. Legacy chat formatting
  remains explicitly selected by `chat-legacy`, with DSPy's field decoder; the
  extracted field then has to satisfy the same complete-JSON admission rule.
- Validation `none` does not compile a validator; `warn` accepts and reports
  `failed`; `error` rejects with owned diagnostics. Validators own their compiled
  state and are released per invocation; there is no schema cache or leak.
- Original text, context and images are preserved. History retries append actual
  assistant/error turns to the full initial request. History is reset at a
  destination change; denied destinations receive no request. Independent
  retries send the full original prompt again.

These are explicit implementation/default-policy changes, not authorization to
change a live configuration, send data to a new destination, or promote a label.
No existing dataset, provider configuration, or collection setting is migrated.

## Evidence, privacy and capture

A receipt contains the decoded value, exact successful assistant completion,
requested/used configured target identities, ordered attempts/outcomes,
validation state/diagnostics, termination and optional provider usage. Target
identity describes the actual requested route, not proof of a provider's
underlying deployment. Absent usage stays absent; individual unreported counters
stay null. Normal stop/schema compliance do not establish semantic correctness.
Errors retain received completion evidence in their attempt records; transport
or malformed wire envelopes without a selected completion have no invented raw
completion. Display errors expose classes/counts, not private payloads.

Credentials are excluded from identity/serialization; credential-bearing URL
userinfo/query/fragment is rejected. Each request snapshots known keys from the
complete configuration, including other tasks, generator and feedback models.
Transport errors never echo provider bodies or authentication. Raw/decoded JSON
values, property names, envelopes and validation diagnostics are checked: Unicode
escaping cannot materialize a known key in a public parsed value. Credential echoes
are rejected and sensitive evidence withheld, not rewritten into success. Sensitive
prompts are not logged and sensitive input/context is not collected, while the
original request remains unchanged. Arbitrary application secrets unknown to XTR
cannot be identified automatically: receipts, logs and captures belong in private
stores.

Logging is request-owned (`Config`, `Disabled`, or `Directory`); engine creation
has no process-global logging effect. Enabled logging captures full sent prompts
(including media) and admitted/rejected attempt evidence, excluding credential
echoes. Files are published atomically without clobbering; failures produce a
warning rather than a new provider call.

Collection remains opt-in. It stores accepted candidate evidence plus truthful
failed-attempt metadata, but not rejected completion/parsed/diagnostic content.
Unrecognized failed termination reason text is projected to a bounded
`unsupported` classification, not persisted as arbitrary provider text. Collection
does not copy source media. Capture records have `qualification: unreviewed`,
receipt provenance, and no `expected_json`; training loaders reject them rather
than automatically treating predictions as gold. Existing captures remain
byte-identical and are not automatically filtered/reclassified. Random names do
not depend on clock resolution. Temporary-file write/sync followed by atomic
non-clobber publication prevents partial final JSON; interruption may leave a
`.tmp` file, never a partial final `.json`. Power-loss directory-entry durability
is not promised.

Unix new directories/files are 0700/0600. Existing non-private capture/log
directories and symlink components are rejected, not permission-repaired.
Trusted directory ancestors are required; path-based publication is not a defense
against a hostile directory-renaming race. Windows requires a private location
and owner-managed ACLs; Windows ACL enforcement is not claimed.

`xtr examples approve TASK --candidate FILE --expected LABELS.json --reviewer ID
--output-dir PRIVATE_DIR` is an explicit independent-review path. The separately
supplied labels must validate against the task schema. It writes a new approved
example with reviewer/source/time audit metadata; it never modifies/promotes the
candidate in place or selects predicted values as labels. A reviewer string alone
is not proof that review occurred; the owner must independently review labels.

## Vision

`--image PATH` (repeatable), `--image-base64 BASE64`, or `--image stdin` sends
OpenAI-compatible text-plus-image content. Binary stdin belongs exclusively to
that image; optional `--text` supplies accompanying text, otherwise text is empty.
Clap argument positions preserve the original order across mixed file/base64/stdin
flags. File/base64 signatures determine PNG/JPEG/GIF/WebP MIME type, preserving bytes
without OCR, resizing or recompression. Signature recognition is not full codec
validation. Invalid image inputs fail before any call. Training examples encode
`image_paths` into `image_data_uris`; the adapter includes current/demo image
turns in wire messages instead of dropping them. DSPy's optimizer requires
numeric usage counters and uses zero when missing; typed inference receipts do
not make that substitution. No native provider SDK/media-quality qualification
or live vision experiment is claimed.

## Regression evidence

CPU-only synthetic fixtures and loopback servers; no real providers.

- Parser RED: original Rust literal assertion received `needs selection` instead
  of `needs thinking before response selection`; GREEN preserves decoded values.
- Isolation RED: controlled configure A/configure B/forward A returned
  `request-b`; GREEN runs concurrent JSON/legacy requests with their own model,
  prompts/context/instructions and log destinations, without global reads.
- Ownership RED: eight compilations of a 1 MiB description retained live bytes
  `[1049025, 2098050, 3147075, 4196100, 5245125, 6294150, 7343175, 8392200]`.
  GREEN: `[0, 0, 0, 0, 0, 0, 0, 0]` after each validator is dropped. The
  test-only opt-in thread-local allocator measures live allocations, not RSS or
  allocator high-water marks; its counter has a separate alloc/realloc/drop test.
- Termination RED: mock `length` plus valid JSON returned `Ok("{\"value\":7}")`.
  GREEN returns typed `truncated` failure with raw completion and usage.
- Capture RED: old record contained `expected_json` without receipt/qualification.
  GREEN round-trips unreviewed candidate evidence, survives simultaneous writes,
  and preserves prior bytes under injected write/publication failures.
- Independent review REDs: escaped credential values/property names were accepted,
  sensitive input/context was captured, and schema-const credentials escaped via
  diagnostics. All now fail/suppress persistence without rewriting source input.
  Additional REDs classified null terminal events as malformed (four backend calls
  instead of one), accepted malformed finish metadata as unknown, and retained
  arbitrary failed termination text in capture; all are locked down by regressions.
  Follow-up RED found unsupported finish strings downgraded to malformed by boolean
  refusal metadata; GREEN preserves unsupported rejection, one student call and
  zero granted-teacher calls under the default recovery policy.
- Mixed-image-order RED: base64 PNG, file JPEG, stdin GIF, base64 JPEG arrived as
  JPEG/GIF/PNG/JPEG. GREEN preserves PNG/JPEG/GIF/JPEG using Clap positions.
- Loopback tests explicitly reset accepted sockets to blocking mode on macOS,
  retaining bounded read/write deadlines; otherwise inherited nonblocking mode
  could cause intermittent `WouldBlock` instead of a product regression.

Final `cargo check --workspace`, `just check-all` (120 synthetic tests) and
`git diff --check` passed before committing. Verification used no real providers,
live configuration changes or existing dataset/label changes. Issue closure remains
a separate owner action.

Reproduce targeted proof:

```sh
cargo test -p xtr-core --lib lm::parser_tests
cargo test -p xtr-core --lib schema::tests -- --nocapture
cargo test -p xtr-core --lib capture::tests
cargo test -p xtr-core --test hardening_regressions
cargo test -p xtr-core --test inference_policy
cargo test -p xtr-core --test review_regressions
cargo test -p xtr-core --test multimodal_adapter
cargo test -p xtr-cli --test inference_cli
cargo check --workspace
just check-all
```
