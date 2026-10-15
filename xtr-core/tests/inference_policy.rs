//! Public inference-policy contract, exercised without providers or live config.
//! Scripted transport records the actual messages/descriptors at each attempt;
//! barriers make overlap and cancellation deterministic instead of timing-based.
#![allow(
    clippy::expect_used,
    clippy::panic,
    reason = "Synthetic integration fixtures and unexpected backend calls must fail immediately"
)]

use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;
use serde_json::json;
use tempfile::TempDir;
use tokio::sync::Barrier;
use xtr_core::config::AppConfig;
use xtr_core::config::AppPaths;
use xtr_core::config::ConfigBundle;
use xtr_core::config::ModelDescriptor;
use xtr_core::config::ModelSpec;
use xtr_core::config::TaskConfig;
use xtr_core::engine::ExtractionEngine;
use xtr_core::engine::RetryMode;
use xtr_core::engine::ValidationMode;
use xtr_core::inference::AttemptOutcome;
use xtr_core::inference::AttemptReceipt;
use xtr_core::inference::Completeness;
use xtr_core::inference::Completion;
use xtr_core::inference::ExtractionError;
use xtr_core::inference::ExtractionReceipt;
use xtr_core::inference::FailureClass;
use xtr_core::inference::InferenceBackend;
use xtr_core::inference::InferenceOptions;
use xtr_core::inference::LoggingPolicy;
use xtr_core::inference::ModelIdentity;
use xtr_core::inference::ModelRole;
use xtr_core::inference::ProviderUsage;
use xtr_core::inference::Termination;
use xtr_core::inference::ValidationStatus;
use xtr_core::media::ImageInput;

const TASK: &str = "synthetic";
const DEADLINE: Duration = Duration::from_secs(5);
const KEYS: [&str; 4] = [
    "SYNTHETIC-STUDENT-CREDENTIAL-SENTINEL",
    "SYNTHETIC-TEACHER-CREDENTIAL-SENTINEL",
    "SYNTHETIC-FALLBACK-ZERO-CREDENTIAL-SENTINEL",
    "SYNTHETIC-FALLBACK-ONE-CREDENTIAL-SENTINEL",
];
const NAMES: [&str; 4] = ["student", "teacher", "fallback-zero", "fallback-one"];

struct Fixture {
    _temporary: TempDir,
    root: PathBuf,
    schema: PathBuf,
    engine: Arc<ExtractionEngine>,
}

impl Fixture {
    fn new(configure: impl FnOnce(&mut AppConfig)) -> Self {
        let temporary = tempfile::tempdir().expect("create isolated fixture");
        // /var -> /private/var on macOS must not look like unsafe capture storage.
        let root = temporary
            .path()
            .canonicalize()
            .expect("canonical fixture root");
        let paths = AppPaths {
            app_name: "inference-policy-test".into(),
            config_dir: root.join("config"),
            config_file: root.join("config/config.toml"),
            data_dir: root.join("data"),
            state_dir: root.join("state"),
            cache_dir: root.join("cache"),
        };
        paths.ensure_config_dir().expect("fixture config directory");
        paths
            .ensure_runtime_dirs()
            .expect("fixture runtime directories");
        let schema = paths.config_dir.join("synthetic.schema.json");
        fs::write(&schema, schema_value("integer").to_string()).expect("write synthetic schema");
        let mut config = AppConfig::default();
        config.models.defaults.student = model_spec(0);
        config.models.defaults.teacher = model_spec(1);
        config.models.defaults.fallbacks = vec![model_spec(2), model_spec(3)];
        config.models.defaults.dataset_generator = None;
        config.tasks.insert(
            TASK.into(),
            TaskConfig {
                schema: Some(schema.to_string_lossy().into_owned()),
                description: Some("Extract the synthetic value; preserve literal notes.".into()),
                include_timestamp: Some(false),
                ..Default::default()
            },
        );
        config.logging.verbose_llm_logging = false;
        config.data_collection.enabled = false;
        configure(&mut config);
        Self {
            _temporary: temporary,
            root,
            schema,
            engine: Arc::new(ExtractionEngine::from_bundle(ConfigBundle {
                config,
                paths,
            })),
        }
    }

    fn identity(&self, role: ModelRole) -> ModelIdentity {
        self.engine
            .fallback_identity(TASK, role)
            .expect("resolve synthetic identity")
    }

    fn grants(&self) -> Vec<ModelIdentity> {
        vec![
            self.identity(ModelRole::Teacher),
            self.identity(ModelRole::Fallback(0)),
            self.identity(ModelRole::Fallback(1)),
        ]
    }

    fn options(&self) -> InferenceOptions {
        InferenceOptions {
            validation: ValidationMode::Error,
            allowed_fallbacks: self.grants(),
            logging: LoggingPolicy::Disabled,
            ..Default::default()
        }
    }

    async fn run(
        &self,
        options: &InferenceOptions,
        backend: &ScriptedBackend,
    ) -> Result<ExtractionReceipt, ExtractionError> {
        self.engine
            .run_inference_with_backend(TASK, "value is 7", None, options, backend)
            .await
    }

    fn captures(&self) -> PathBuf {
        self.engine.paths().state_dir.join("collected_examples")
    }
}

fn schema_value(value_type: &str) -> Value {
    json!({
        "type": "object",
        "properties": {"value": {"type": value_type}, "note": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": false
    })
}

fn model_spec(index: usize) -> ModelSpec {
    ModelSpec {
        name: Some(NAMES[index].into()),
        // Reserved .invalid destinations are never contacted by these tests.
        base_url: Some(format!("https://{}.invalid/v1", NAMES[index])),
        api_key: Some(KEYS[index].into()),
        adapter: Some("json".into()),
        request_timeout_secs: Some(5),
        max_tokens: Some(32 + index as u32),
    }
}

#[derive(Clone)]
struct Call {
    model: ModelDescriptor,
    messages: Value,
}

struct Gate {
    entered: Barrier,
    release: Barrier,
}

impl Gate {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            entered: Barrier::new(2),
            release: Barrier::new(2),
        })
    }
}

struct Step {
    model: &'static str,
    result: Result<Completion, FailureClass>,
    gate: Option<Arc<Gate>>,
}

impl Step {
    fn success(model: &'static str, raw: impl Into<String>) -> Self {
        Self::completion(model, raw, Termination::Stop, Some(usage()))
    }

    fn completion(
        model: &'static str,
        raw: impl Into<String>,
        termination: Termination,
        usage: Option<ProviderUsage>,
    ) -> Self {
        Self {
            model,
            result: Ok(Completion {
                raw: Some(raw.into()),
                termination,
                usage,
            }),
            gate: None,
        }
    }

    fn failure(model: &'static str, failure: FailureClass) -> Self {
        Self {
            model,
            result: Err(failure),
            gate: None,
        }
    }

    fn gated(mut self, gate: &Arc<Gate>) -> Self {
        self.gate = Some(Arc::clone(gate));
        self
    }
}

#[derive(Default)]
struct ScriptState {
    steps: VecDeque<Step>,
    calls: Vec<Call>,
}

struct ScriptedBackend(Mutex<ScriptState>);

impl ScriptedBackend {
    fn new(steps: impl IntoIterator<Item = Step>) -> Self {
        Self(Mutex::new(ScriptState {
            steps: steps.into_iter().collect(),
            calls: Vec::new(),
        }))
    }

    fn calls(&self) -> Vec<Call> {
        self.0.lock().expect("script state lock").calls.clone()
    }

    fn assert_finished(&self, models: &[&str]) {
        let state = self.0.lock().expect("script state lock");
        assert!(
            state.steps.is_empty(),
            "scripted responses were not consumed"
        );
        assert_eq!(
            state
                .calls
                .iter()
                .map(|call| call.model.name.as_str())
                .collect::<Vec<_>>(),
            models
        );
    }
}

#[async_trait]
impl InferenceBackend for ScriptedBackend {
    async fn complete(
        &self,
        model: &ModelDescriptor,
        messages: &Value,
    ) -> Result<Completion, FailureClass> {
        // Only this synchronous scope owns the std mutex; no guard crosses await.
        let step = {
            let mut state = self.0.lock().expect("script state lock");
            let step = state.steps.pop_front().expect("unexpected backend call");
            assert_eq!(model.name, step.model, "unexpected model/destination order");
            state.calls.push(Call {
                model: model.clone(),
                messages: messages.clone(),
            });
            step
        };
        if let Some(gate) = step.gate {
            gate.entered.wait().await;
            gate.release.wait().await;
        }
        step.result
    }
}

fn usage() -> ProviderUsage {
    ProviderUsage {
        prompt_tokens: Some(11),
        completion_tokens: Some(5),
        total_tokens: Some(16),
    }
}

fn assert_attempt(
    attempt: &AttemptReceipt,
    ordinal: usize,
    model: &ModelIdentity,
    retry: u32,
    outcome: AttemptOutcome,
) {
    assert_eq!(attempt.ordinal, ordinal);
    assert_eq!(&attempt.model, model);
    assert_eq!(attempt.retry, retry);
    assert_eq!(attempt.outcome, outcome);
}

fn assert_withheld(attempt: &AttemptReceipt) {
    assert!(attempt.raw_completion.is_none());
    assert!(attempt.parsed.is_none());
    assert!(attempt.termination.is_none());
    assert!(attempt.usage.is_none());
    assert_eq!(attempt.validation, ValidationStatus::NotChecked);
}

fn assert_no_credentials(text: &str) {
    for key in KEYS {
        assert!(
            !text.contains(key),
            "synthetic credential escaped into public evidence"
        );
    }
}

fn assert_safe_error(error: &ExtractionError) {
    assert_no_credentials(&error.to_string());
    assert_no_credentials(&format!("{error:?}"));
    assert_no_credentials(&serde_json::to_string(error).expect("serialize typed error"));
}

fn files_under(path: &Path) -> Vec<PathBuf> {
    if !path.exists() {
        return Vec::new();
    }
    let mut files = Vec::new();
    for entry in fs::read_dir(path).expect("read synthetic directory") {
        let entry = entry.expect("read synthetic directory entry");
        if entry.file_type().expect("fixture file type").is_dir() {
            files.extend(files_under(&entry.path()));
        } else {
            files.push(entry.path());
        }
    }
    files.sort();
    files
}

fn records(path: &Path) -> Vec<Value> {
    files_under(path)
        .into_iter()
        .map(|path| {
            assert_eq!(
                path.extension().and_then(|extension| extension.to_str()),
                Some("json")
            );
            let bytes = fs::read(path).expect("read private synthetic JSON");
            assert_no_credentials(std::str::from_utf8(&bytes).expect("JSON is UTF-8"));
            serde_json::from_slice(&bytes).expect("private record is complete JSON")
        })
        .collect()
}

fn user_text(messages: &Value) -> &str {
    let turns = messages.as_array().expect("actual messages are an array");
    let user = turns
        .iter()
        .rev()
        .find(|turn| turn["role"] == "user")
        .expect("user turn");
    if let Some(text) = user["content"].as_str() {
        text
    } else {
        user["content"][0]["text"]
            .as_str()
            .expect("multimodal text first")
    }
}

#[tokio::test]
async fn explicitly_granted_teacher_keeps_provenance_literals_and_provider_usage() {
    let fixture = Fixture::new(|_| {});
    let student = fixture.identity(ModelRole::Student);
    let teacher = fixture.identity(ModelRole::Teacher);
    let note = "  literal \\path\n```json <think>inside</think> {braces} 'quotes'  ";
    let parsed = json!({"value": 7, "note": note});
    let raw = format!("  {}\r\n", parsed);
    let backend = ScriptedBackend::new([
        Step::failure("student", FailureClass::Transport),
        Step::success("teacher", raw.clone()),
    ]);
    let options = InferenceOptions {
        allowed_fallbacks: vec![teacher.clone()],
        max_tokens: Some(91),
        ..fixture.options()
    };
    let receipt = fixture
        .run(&options, &backend)
        .await
        .expect("authorized teacher succeeds");
    backend.assert_finished(&["student", "teacher"]);
    assert_eq!(receipt.requested_model, student);
    assert_eq!(receipt.used_model, teacher);
    assert_eq!(receipt.raw_completion, raw);
    assert_eq!(receipt.parsed, parsed);
    assert_eq!(receipt.validation, ValidationStatus::Passed);
    assert_eq!(receipt.termination, Termination::Stop);
    assert!(matches!(receipt.completeness, Completeness::ReportedStop));
    assert_eq!(receipt.usage, Some(usage()));
    assert!(receipt.warnings.is_empty());
    assert_eq!(receipt.attempts.len(), 2);
    assert_attempt(
        &receipt.attempts[0],
        1,
        &student,
        0,
        AttemptOutcome::Failed(FailureClass::Transport),
    );
    assert_withheld(&receipt.attempts[0]);
    assert_attempt(
        &receipt.attempts[1],
        2,
        &teacher,
        0,
        AttemptOutcome::Accepted,
    );
    assert_eq!(
        receipt.attempts[1].raw_completion.as_deref(),
        Some(raw.as_str())
    );
    assert_eq!(receipt.attempts[1].parsed.as_ref(), Some(&parsed));
    assert_eq!(receipt.attempts[1].usage, Some(usage()));
    assert_eq!(receipt.attempts[1].termination, Some(Termination::Stop));
    for call in backend.calls() {
        assert_eq!(call.model.max_tokens, Some(91));
        assert_eq!(
            call.model.base_url.as_deref(),
            Some(if call.model.name == "student" {
                "https://student.invalid/v1"
            } else {
                "https://teacher.invalid/v1"
            })
        );
    }
    assert_no_credentials(&format!("{receipt:?}"));
    assert_no_credentials(&serde_json::to_string(&receipt).expect("serialize receipt"));
}

#[tokio::test]
async fn explicitly_granted_later_fallback_records_each_failed_destination_in_order() {
    let fixture = Fixture::new(|_| {});
    let invalid = "{\"value\":\"not an integer\"}";
    let raw = " \n{\"value\":8,\"note\":\"literal <think>note</think>\"}\r\n";
    let partial_usage = ProviderUsage {
        prompt_tokens: Some(20),
        ..Default::default()
    };
    let backend = ScriptedBackend::new([
        Step::failure("student", FailureClass::Transport),
        Step::success("teacher", "{broken"),
        Step::success("fallback-zero", invalid),
        Step::completion(
            "fallback-one",
            raw,
            Termination::Stop,
            Some(partial_usage.clone()),
        ),
    ]);
    let receipt = fixture
        .run(&fixture.options(), &backend)
        .await
        .expect("later fallback succeeds");
    backend.assert_finished(&NAMES);
    assert_eq!(
        receipt.requested_model,
        fixture.identity(ModelRole::Student)
    );
    assert_eq!(receipt.used_model, fixture.identity(ModelRole::Fallback(1)));
    assert_eq!(receipt.raw_completion, raw);
    assert_eq!(
        receipt.parsed,
        json!({"value": 8, "note": "literal <think>note</think>"})
    );
    assert_eq!(receipt.usage, Some(partial_usage));
    assert_eq!(receipt.validation, ValidationStatus::Passed);
    assert_eq!(receipt.attempts.len(), 4);
    for (index, (role, outcome)) in [
        (
            ModelRole::Student,
            AttemptOutcome::Failed(FailureClass::Transport),
        ),
        (
            ModelRole::Teacher,
            AttemptOutcome::Failed(FailureClass::Malformed),
        ),
        (
            ModelRole::Fallback(0),
            AttemptOutcome::Failed(FailureClass::SchemaInvalid),
        ),
        (ModelRole::Fallback(1), AttemptOutcome::Accepted),
    ]
    .into_iter()
    .enumerate()
    {
        assert_attempt(
            &receipt.attempts[index],
            index + 1,
            &fixture.identity(role),
            0,
            outcome,
        );
    }
    assert_eq!(
        receipt.attempts[1].raw_completion.as_deref(),
        Some("{broken")
    );
    assert!(receipt.attempts[1].parsed.is_none());
    assert_eq!(receipt.attempts[2].raw_completion.as_deref(), Some(invalid));
    assert_eq!(
        receipt.attempts[2].parsed,
        Some(json!({"value": "not an integer"}))
    );
    assert_eq!(receipt.attempts[2].validation, ValidationStatus::Failed);
    assert_eq!(receipt.attempts[3].parsed.as_ref(), Some(&receipt.parsed));
    assert_eq!(receipt.attempts[3].usage, receipt.usage);
    assert!(receipt.warnings.is_empty());
}

#[tokio::test]
async fn configured_but_denied_fallbacks_make_zero_backend_calls() {
    let fixture = Fixture::new(|_| {});
    let backend = ScriptedBackend::new([Step::failure("student", FailureClass::Transport)]);
    let options = InferenceOptions {
        logging: LoggingPolicy::Disabled,
        ..Default::default()
    };
    let error = fixture
        .run(&options, &backend)
        .await
        .expect_err("config is not a fallback grant");
    backend.assert_finished(&["student"]);
    assert_eq!(error.kind, FailureClass::Exhausted);
    assert_eq!(error.last_failure, Some(FailureClass::Transport));
    assert_eq!(error.attempts.len(), 4);
    for (index, role) in [
        ModelRole::Teacher,
        ModelRole::Fallback(0),
        ModelRole::Fallback(1),
    ]
    .into_iter()
    .enumerate()
    {
        let attempt = &error.attempts[index + 1];
        assert_attempt(
            attempt,
            index + 2,
            &fixture.identity(role),
            0,
            AttemptOutcome::Failed(FailureClass::Denied),
        );
        assert_withheld(attempt);
    }
    assert_safe_error(&error);
}

#[tokio::test]
async fn altered_name_destination_or_role_cannot_grant_the_teacher() {
    let fixture = Fixture::new(|_| {});
    let exact = fixture.identity(ModelRole::Teacher);
    let mut altered_name = exact.clone();
    altered_name.name = "other-model".into();
    let mut altered_destination = exact.clone();
    altered_destination.destination = "https://elsewhere.invalid/v1".into();
    let mut altered_role = exact;
    altered_role.role = ModelRole::Fallback(0);
    for grant in [altered_name, altered_destination, altered_role] {
        let backend = ScriptedBackend::new([Step::failure("student", FailureClass::Transport)]);
        let options = InferenceOptions {
            allowed_fallbacks: vec![grant],
            ..Default::default()
        };
        let error = fixture
            .run(&options, &backend)
            .await
            .expect_err("grant must match all identity fields");
        backend.assert_finished(&["student"]);
        assert_eq!(error.kind, FailureClass::Exhausted);
        assert_eq!(error.last_failure, Some(FailureClass::Transport));
        assert_eq!(error.attempts.len(), 4);
        assert!(
            error.attempts[1..]
                .iter()
                .all(|attempt| attempt.outcome == AttemptOutcome::Failed(FailureClass::Denied))
        );
    }
}

#[tokio::test]
async fn denied_intermediate_destinations_do_not_block_an_exact_later_grant() {
    let fixture = Fixture::new(|_| {});
    let fallback = fixture.identity(ModelRole::Fallback(1));
    let backend = ScriptedBackend::new([
        Step::failure("student", FailureClass::Transport),
        Step::success("fallback-one", "{\"value\":9}"),
    ]);
    let options = InferenceOptions {
        allowed_fallbacks: vec![fallback.clone()],
        ..fixture.options()
    };
    let receipt = fixture
        .run(&options, &backend)
        .await
        .expect("only explicitly granted fallback is contacted");
    backend.assert_finished(&["student", "fallback-one"]);
    assert_eq!(receipt.used_model, fallback);
    assert_eq!(receipt.attempts.len(), 4);
    for attempt in &receipt.attempts[1..3] {
        assert_eq!(
            attempt.outcome,
            AttemptOutcome::Failed(FailureClass::Denied)
        );
        assert_withheld(attempt);
    }
    assert_eq!(receipt.attempts[3].ordinal, 4);
    assert_eq!(receipt.parsed, json!({"value": 9}));
}

#[tokio::test]
async fn length_termination_rejects_even_schema_valid_json_and_keeps_wire_evidence() {
    let fixture = Fixture::new(|config| config.data_collection.enabled = true);
    let raw = " {\"value\":7}\n";
    let backend = ScriptedBackend::new([Step::completion(
        "student",
        raw,
        Termination::Length,
        Some(usage()),
    )]);
    let options = InferenceOptions {
        fallback_on: Vec::new(),
        ..fixture.options()
    };
    let error = fixture
        .run(&options, &backend)
        .await
        .expect_err("length does not prove completeness");
    backend.assert_finished(&["student"]);
    assert_eq!(error.kind, FailureClass::Exhausted);
    assert_eq!(error.last_failure, Some(FailureClass::Truncated));
    assert_eq!(error.attempts.len(), 1);
    let attempt = &error.attempts[0];
    assert_eq!(
        attempt.outcome,
        AttemptOutcome::Failed(FailureClass::Truncated)
    );
    assert_eq!(attempt.raw_completion.as_deref(), Some(raw));
    assert_eq!(attempt.termination, Some(Termination::Length));
    assert_eq!(attempt.usage, Some(usage()));
    assert!(
        attempt.parsed.is_none(),
        "termination admission precedes schema decoding"
    );
    assert_eq!(attempt.validation, ValidationStatus::NotChecked);
    assert!(!fixture.captures().exists());
}

#[tokio::test]
async fn unknown_termination_requires_explicit_opt_in_and_stays_unverified() {
    let fixture = Fixture::new(|_| {});
    for allow_unknown in [false, true] {
        let backend = ScriptedBackend::new([Step::completion(
            "student",
            "{\"value\":7}",
            Termination::Unknown,
            None,
        )]);
        let options = InferenceOptions {
            max_retries: 3,
            allow_unknown_termination: allow_unknown,
            ..fixture.options()
        };
        let result = fixture.run(&options, &backend).await;
        backend.assert_finished(&["student"]);
        if allow_unknown {
            let receipt = result.expect("explicit unknown admission");
            assert!(matches!(receipt.completeness, Completeness::Unverified));
            assert_eq!(receipt.termination, Termination::Unknown);
            assert_eq!(receipt.validation, ValidationStatus::Passed);
            assert_eq!(receipt.parsed, json!({"value": 7}));
            assert!(
                receipt.usage.is_none(),
                "never invent usage from text length"
            );
            assert!(receipt.attempts[0].usage.is_none());
        } else {
            let error = result.expect_err("strict unknown rejection");
            assert_eq!(error.kind, FailureClass::Exhausted);
            assert_eq!(error.last_failure, Some(FailureClass::UnknownTermination));
            assert_eq!(error.attempts.len(), 1);
            assert_eq!(error.attempts[0].termination, Some(Termination::Unknown));
            assert_eq!(
                error.attempts[0].raw_completion.as_deref(),
                Some("{\"value\":7}")
            );
            assert!(error.attempts[0].parsed.is_none());
        }
    }
}

#[tokio::test]
async fn refusal_and_unsupported_termination_do_not_retry_or_fallback_by_default() {
    let fixture = Fixture::new(|_| {});
    for (termination, failure) in [
        (Termination::Refusal, FailureClass::Refusal),
        (Termination::ToolCalls, FailureClass::UnsupportedTermination),
        (
            Termination::Other("synthetic-other".into()),
            FailureClass::UnsupportedTermination,
        ),
    ] {
        // Valid JSON must not override refusal/unsupported termination metadata.
        let backend = ScriptedBackend::new([Step::completion(
            "student",
            "{\"value\":7}",
            termination.clone(),
            Some(usage()),
        )]);
        let options = InferenceOptions {
            max_retries: 4,
            ..fixture.options()
        };
        let error = fixture
            .run(&options, &backend)
            .await
            .expect_err("terminal rejection");
        backend.assert_finished(&["student"]);
        assert_eq!(error.kind, FailureClass::Exhausted);
        assert_eq!(error.last_failure, Some(failure.clone()));
        assert_eq!(error.attempts.len(), 1);
        assert_eq!(error.attempts[0].outcome, AttemptOutcome::Failed(failure));
        assert_eq!(error.attempts[0].termination, Some(termination));
        assert_eq!(error.attempts[0].usage, Some(usage()));
        assert!(error.attempts[0].parsed.is_none());
    }
}

#[tokio::test]
async fn independent_retries_repeat_original_turns_while_chat_appends_actual_history() {
    let input = format!(
        "  original \"quoted\" \\path\r\n{}\nEND-OF-FULL-INPUT  ",
        "synthetic paragraph\n".repeat(250)
    );
    let context = " \tcaller context\n```json <think>literal</think> {braces}\r\nEND-CONTEXT  ";
    for adapter in ["json", "chat-legacy"] {
        let fixture =
            Fixture::new(|config| config.models.defaults.student.adapter = Some(adapter.into()));
        for mode in [RetryMode::Independent, RetryMode::WithHistory] {
            let raw = |payload: &str| {
                if adapter == "json" {
                    payload.to_owned()
                } else {
                    format!("[[ ## output_json ## ]]\n{payload}\n[[ ## completed ## ]]")
                }
            };
            let rejected = raw("{\"value\":\"wrong\"}");
            let accepted = raw("{\"value\":7}");
            let backend = ScriptedBackend::new([
                Step::success("student", rejected.clone()),
                Step::success("student", rejected.clone()),
                Step::success("student", accepted.clone()),
            ]);
            let options = InferenceOptions {
                max_retries: 2,
                retry_mode: mode,
                ..fixture.options()
            };
            let receipt = fixture
                .engine
                .run_inference_with_backend(TASK, &input, Some(context), &options, &backend)
                .await
                .expect("third attempt succeeds");
            backend.assert_finished(&["student", "student", "student"]);
            assert_eq!(receipt.parsed, json!({"value": 7}));
            assert_eq!(receipt.raw_completion, accepted);
            assert_eq!(receipt.attempts.len(), 3);
            assert_eq!(
                receipt.attempts[0].outcome,
                AttemptOutcome::Failed(FailureClass::SchemaInvalid)
            );
            assert_eq!(receipt.attempts[1].retry, 1);
            assert_eq!(receipt.attempts[2].retry, 2);
            let calls = backend.calls();
            let base = calls[0].messages.as_array().expect("base turns");
            assert!(
                user_text(&calls[0].messages).contains(&input),
                "full input, including whitespace, must survive formatting"
            );
            assert!(user_text(&calls[0].messages).contains(context));
            assert!(
                user_text(&calls[0].messages)
                    .contains(&fs::read_to_string(&fixture.schema).expect("schema text"))
            );
            for (retry, call) in calls.iter().enumerate() {
                let turns = call.messages.as_array().expect("retry turns");
                assert_eq!(
                    &turns[..base.len()],
                    base,
                    "original system/user turns are never reduced"
                );
                if mode == RetryMode::Independent {
                    assert_eq!(
                        &call.messages, &calls[0].messages,
                        "independent means no history at all"
                    );
                } else {
                    assert_eq!(turns.len(), base.len() + retry * 2);
                    for pair in turns[base.len()..].chunks_exact(2) {
                        assert_eq!(pair[0], json!({"role": "assistant", "content": rejected}));
                        assert_eq!(pair[1]["role"], "user");
                        let feedback = pair[1]["content"].as_str().expect("retry feedback text");
                        assert!(feedback.contains("SchemaInvalid"));
                        assert!(feedback.contains("matching the schema"));
                    }
                }
            }
        }
    }
}

#[tokio::test]
async fn retry_history_is_reset_when_moving_to_a_granted_destination() {
    let fixture = Fixture::new(|_| {});
    let backend = ScriptedBackend::new([
        Step::success("student", "{broken"),
        Step::success("student", "{still-broken"),
        Step::success("teacher", "{\"value\":7}"),
    ]);
    let options = InferenceOptions {
        max_retries: 1,
        retry_mode: RetryMode::WithHistory,
        ..fixture.options()
    };
    let receipt = fixture
        .run(&options, &backend)
        .await
        .expect("teacher has fresh history");
    backend.assert_finished(&["student", "student", "teacher"]);
    assert_eq!(receipt.attempts[2].retry, 0);
    let calls = backend.calls();
    assert_eq!(calls[2].messages, calls[0].messages);
    assert_ne!(calls[1].messages, calls[0].messages);
    assert!(!calls[2].messages.to_string().contains("{broken"));
    assert!(
        !calls[2]
            .messages
            .to_string()
            .contains("Previous attempt rejected")
    );
}

#[tokio::test]
async fn transport_malformed_and_schema_failures_exhaust_with_typed_attempt_evidence() {
    let fixture = Fixture::new(|_| {});
    for failure in [
        FailureClass::Transport,
        FailureClass::Malformed,
        FailureClass::SchemaInvalid,
    ] {
        let raw = match failure {
            FailureClass::Malformed => Some("{\"value\":7} trailing prose"),
            FailureClass::SchemaInvalid => Some("{\"value\":\"wrong\"}"),
            _ => None,
        };
        let backend = ScriptedBackend::new((0..3).map(|_| match raw {
            Some(raw) => Step::success("student", raw),
            None => Step::failure("student", FailureClass::Transport),
        }));
        let options = InferenceOptions {
            max_retries: 2,
            fallback_on: Vec::new(),
            ..fixture.options()
        };
        let error = fixture
            .run(&options, &backend)
            .await
            .expect_err("bounded retry exhaustion");
        backend.assert_finished(&["student", "student", "student"]);
        assert_eq!(error.kind, FailureClass::Exhausted);
        assert_eq!(error.last_failure, Some(failure.clone()));
        assert_eq!(error.attempts.len(), 3);
        for (index, attempt) in error.attempts.iter().enumerate() {
            assert_attempt(
                attempt,
                index + 1,
                &fixture.identity(ModelRole::Student),
                index as u32,
                AttemptOutcome::Failed(failure.clone()),
            );
            assert_eq!(attempt.raw_completion.as_deref(), raw);
            if raw.is_none() {
                assert_withheld(attempt);
            } else {
                assert_eq!(attempt.termination, Some(Termination::Stop));
                assert_eq!(attempt.usage, Some(usage()));
                if failure == FailureClass::SchemaInvalid {
                    assert_eq!(attempt.parsed, Some(json!({"value": "wrong"})));
                    assert_eq!(attempt.validation, ValidationStatus::Failed);
                } else {
                    assert!(attempt.parsed.is_none());
                    assert_eq!(attempt.validation, ValidationStatus::NotChecked);
                }
            }
        }
        assert_safe_error(&error);
        let calls = backend.calls();
        assert!(calls.iter().all(|call| call.messages == calls[0].messages));
    }
}

#[tokio::test]
async fn validation_none_warn_and_error_have_distinct_admission_and_retry_status() {
    let fixture = Fixture::new(|_| {});
    let raw = "  {\"value\":\"wrong\"}\r\n";
    for (mode, status) in [
        (ValidationMode::None, ValidationStatus::NotChecked),
        (ValidationMode::Warn, ValidationStatus::Failed),
        (ValidationMode::Error, ValidationStatus::Failed),
    ] {
        let count = if mode == ValidationMode::Error { 2 } else { 1 };
        let backend = ScriptedBackend::new((0..count).map(|_| Step::success("student", raw)));
        let options = InferenceOptions {
            validation: mode,
            max_retries: 1,
            fallback_on: Vec::new(),
            ..fixture.options()
        };
        let result = fixture.run(&options, &backend).await;
        backend.assert_finished(&vec!["student"; count]);
        if mode == ValidationMode::Error {
            let error = result.expect_err("error mode rejects and retries schema invalidity");
            assert_eq!(error.kind, FailureClass::Exhausted);
            assert_eq!(error.last_failure, Some(FailureClass::SchemaInvalid));
            assert_eq!(error.attempts.len(), 2);
            for attempt in error.attempts {
                assert_eq!(attempt.validation, status);
                assert_eq!(attempt.parsed, Some(json!({"value": "wrong"})));
                assert_eq!(attempt.raw_completion.as_deref(), Some(raw));
            }
        } else {
            let receipt = result.expect("none/warn admit parsed response");
            assert_eq!(receipt.validation, status);
            assert_eq!(receipt.attempts[0].validation, status);
            assert_eq!(receipt.attempts[0].outcome, AttemptOutcome::Accepted);
            assert_eq!(receipt.parsed, json!({"value": "wrong"}));
            assert_eq!(receipt.raw_completion, raw);
            assert_eq!(receipt.attempts.len(), 1);
        }
    }
}

#[tokio::test]
async fn changed_schema_at_same_path_is_reloaded_and_revalidated_per_request() {
    let fixture = Fixture::new(|_| {});
    let options = InferenceOptions {
        fallback_on: Vec::new(),
        ..fixture.options()
    };
    let backend = ScriptedBackend::new([
        Step::success("student", "{\"value\":7}"),
        Step::success("student", "{\"value\":7}"),
        Step::success("student", "{\"value\":\"seven\"}"),
    ]);
    let first = fixture
        .run(&options, &backend)
        .await
        .expect("original integer schema");
    assert_eq!(first.validation, ValidationStatus::Passed);
    let changed = schema_value("string").to_string();
    fs::write(&fixture.schema, &changed).expect("replace schema in place");
    let error = fixture
        .run(&options, &backend)
        .await
        .expect_err("changed schema rejects old shape");
    assert_eq!(error.last_failure, Some(FailureClass::SchemaInvalid));
    assert_eq!(error.attempts[0].validation, ValidationStatus::Failed);
    let third = fixture
        .run(&options, &backend)
        .await
        .expect("changed schema admits new shape");
    assert_eq!(third.validation, ValidationStatus::Passed);
    assert_eq!(third.parsed, json!({"value": "seven"}));
    backend.assert_finished(&["student", "student", "student"]);
    let calls = backend.calls();
    assert!(user_text(&calls[1].messages).contains(&changed));
    assert!(user_text(&calls[2].messages).contains(&changed));
    assert_ne!(calls[0].messages, calls[1].messages);

    fs::write(&fixture.schema, "{bad schema}").expect("replace with invalid schema");
    let unused = ScriptedBackend::new([]);
    let error = fixture
        .run(&options, &unused)
        .await
        .expect_err("invalid schema fails before transport");
    assert_eq!(error.kind, FailureClass::Configuration);
    assert!(error.last_failure.is_none());
    assert!(error.attempts.is_empty());
    unused.assert_finished(&[]);
}

#[tokio::test]
async fn cancellation_at_transport_barrier_leaves_other_and_next_request_settings_owned() {
    let fixture = Fixture::new(|config| config.logging.verbose_llm_logging = true);
    let gate = Gate::new();
    let backend = Arc::new(ScriptedBackend::new([
        Step::success("student", "{\"value\":100}").gated(&gate),
        // Other request owns none-mode admission, independently of cancelled A.
        Step::success("student", "{\"value\":\"other\"}"),
        Step::success("student", "{\"value\":\"wrong\"}"),
        Step::success("student", "{\"value\":7}"),
    ]));
    let engine_a = Arc::clone(&fixture.engine);
    let backend_a = Arc::clone(&backend);
    let options_a = InferenceOptions {
        max_tokens: Some(111),
        max_retries: 4,
        retry_mode: RetryMode::WithHistory,
        logging: LoggingPolicy::Directory(fixture.root.join("cancelled-logs")),
        ..fixture.options()
    };
    let cancelled = tokio::spawn(async move {
        engine_a
            .run_inference_with_backend(
                TASK,
                "cancelled-input",
                Some("cancelled-context"),
                &options_a,
                backend_a.as_ref(),
            )
            .await
    });
    tokio::time::timeout(DEADLINE, gate.entered.wait())
        .await
        .expect("cancelled request entered backend");
    let other_logs = fixture.root.join("other-logs");
    let options_b = InferenceOptions {
        validation: ValidationMode::None,
        max_tokens: Some(222),
        max_retries: 0,
        retry_mode: RetryMode::Independent,
        logging: LoggingPolicy::Directory(other_logs.clone()),
        ..Default::default()
    };
    let other = tokio::time::timeout(
        DEADLINE,
        fixture.engine.run_inference_with_backend(
            TASK,
            "other-input",
            Some("other-context"),
            &options_b,
            backend.as_ref(),
        ),
    )
    .await
    .expect("other request is not blocked behind cancelled request")
    .expect("other request owns none validation");
    assert_eq!(other.parsed, json!({"value": "other"}));
    assert_eq!(other.validation, ValidationStatus::NotChecked);
    assert_eq!(other.attempts.len(), 1);
    assert!(other.warnings.is_empty());
    cancelled.abort();
    let join_error = tokio::time::timeout(DEADLINE, cancelled)
        .await
        .expect("cancellation completes")
        .expect_err("task must be cancelled, not finish extraction");
    assert!(join_error.is_cancelled());

    // The next call resets to descriptor tokens, its own retry policy and no logs.
    let next_options = InferenceOptions {
        max_retries: 1,
        retry_mode: RetryMode::Independent,
        ..fixture.options()
    };
    let next = tokio::time::timeout(
        DEADLINE,
        fixture.engine.run_inference_with_backend(
            TASK,
            "next-input",
            Some("next-context"),
            &next_options,
            backend.as_ref(),
        ),
    )
    .await
    .expect("subsequent request not poisoned by cancellation")
    .expect("next request retries schema error");
    assert_eq!(next.validation, ValidationStatus::Passed);
    assert_eq!(next.parsed, json!({"value": 7}));
    assert_eq!(next.attempts.len(), 2);
    assert_eq!(
        next.attempts[0].outcome,
        AttemptOutcome::Failed(FailureClass::SchemaInvalid)
    );
    assert_eq!(next.attempts[1].retry, 1);
    backend.assert_finished(&["student", "student", "student", "student"]);
    let calls = backend.calls();
    for (call, tokens, own, foreign) in [
        (&calls[0], 111, "cancelled", "other"),
        (&calls[1], 222, "other", "cancelled"),
        (&calls[2], 32, "next", "cancelled"),
        (&calls[3], 32, "next", "other"),
    ] {
        assert_eq!(call.model.max_tokens, Some(tokens));
        let text = user_text(&call.messages);
        assert!(text.contains(&format!("{own}-input")));
        assert!(text.contains(&format!("{own}-context")));
        assert!(!text.contains(&format!("{foreign}-input")));
        assert!(!text.contains("Previous attempt rejected"));
    }
    assert_eq!(calls[2].messages, calls[3].messages);
    assert!(!fixture.root.join("cancelled-logs").exists());
    assert!(!fixture.engine.paths().state_dir.join("llm_logs").exists());
    let logs = records(&other_logs);
    assert_eq!(logs.len(), 1);
    assert_eq!(logs[0]["request"], calls[1].messages);
    assert_eq!(logs[0]["attempt"]["parsed"], other.parsed);
    assert!(!fixture.captures().exists());
}

#[tokio::test]
async fn overlapping_disabled_logging_does_not_disable_or_redirect_other_destination() {
    let fixture = Fixture::new(|config| config.logging.verbose_llm_logging = true);
    let gate = Gate::new();
    let backend = ScriptedBackend::new([
        Step::success("student", "{\"value\":7}").gated(&gate),
        Step::success("student", "{\"value\":8}").gated(&gate),
    ]);
    let disabled = fixture.options();
    let destination = fixture.root.join("request-b-logs");
    let logged = InferenceOptions {
        logging: LoggingPolicy::Directory(destination.clone()),
        ..fixture.options()
    };
    let (a, b) = tokio::time::timeout(DEADLINE, async {
        tokio::join!(
            fixture.engine.run_inference_with_backend(
                TASK,
                "disabled-input",
                Some("disabled-context"),
                &disabled,
                &backend
            ),
            fixture.engine.run_inference_with_backend(
                TASK,
                "logged-input",
                Some("logged-context"),
                &logged,
                &backend
            )
        )
    })
    .await
    .expect("both requests reach barrier without global serialization");
    let a = a.expect("disabled request succeeds");
    let b = b.expect("logged request succeeds");
    backend.assert_finished(&["student", "student"]);
    assert!(a.warnings.is_empty());
    assert!(b.warnings.is_empty());
    assert_eq!(a.parsed, json!({"value": 7}));
    assert_eq!(b.parsed, json!({"value": 8}));
    assert!(!fixture.engine.paths().state_dir.join("llm_logs").exists());
    let logs = records(&destination);
    assert_eq!(logs.len(), 1);
    assert_eq!(
        logs[0]["attempt"],
        serde_json::to_value(&b.attempts[0]).expect("attempt JSON")
    );
    assert_eq!(logs[0]["request"], backend.calls()[1].messages);
    assert!(!logs[0].to_string().contains("disabled-input"));
    assert!(!logs[0].to_string().contains("disabled-context"));
}

#[tokio::test]
async fn credential_echo_is_sensitive_withheld_and_never_logged_or_captured() {
    let fixture = Fixture::new(|config| config.data_collection.enabled = true);
    let logs = fixture.root.join("echo-logs");
    let options = InferenceOptions {
        max_retries: 3,
        logging: LoggingPolicy::Directory(logs.clone()),
        ..fixture.options()
    };
    // Even a credential for a model not yet contacted must not escape in output.
    for key in KEYS {
        let raw = json!({"value": 7, "note": key}).to_string();
        let backend = ScriptedBackend::new([Step::success("student", raw)]);
        let error = fixture
            .run(&options, &backend)
            .await
            .expect_err("credential echo is never repaired or published");
        backend.assert_finished(&["student"]);
        assert_eq!(error.kind, FailureClass::Exhausted);
        assert_eq!(error.last_failure, Some(FailureClass::SensitiveResponse));
        assert_eq!(
            error.attempts.len(),
            1,
            "sensitive response not retryable/fallbackable by default"
        );
        assert_eq!(
            error.attempts[0].outcome,
            AttemptOutcome::Failed(FailureClass::SensitiveResponse)
        );
        assert_withheld(&error.attempts[0]);
        assert_safe_error(&error);
        assert!(!logs.exists());
        assert!(!fixture.captures().exists());
    }
}

#[tokio::test]
async fn credential_in_other_termination_reason_is_sensitive_and_fully_withheld() {
    let fixture = Fixture::new(|config| config.data_collection.enabled = true);
    let logs = fixture.root.join("reason-logs");
    let options = InferenceOptions {
        max_retries: 3,
        logging: LoggingPolicy::Directory(logs.clone()),
        ..fixture.options()
    };
    for key in KEYS {
        let backend = ScriptedBackend::new([Step::completion(
            "student",
            "{\"value\":7}",
            Termination::Other(format!("reason-{key}")),
            Some(usage()),
        )]);
        let error = fixture
            .run(&options, &backend)
            .await
            .expect_err("termination metadata can also echo a credential");
        backend.assert_finished(&["student"]);
        assert_eq!(error.kind, FailureClass::Exhausted);
        assert_eq!(error.last_failure, Some(FailureClass::SensitiveResponse));
        assert_eq!(error.attempts.len(), 1);
        assert_eq!(
            error.attempts[0].outcome,
            AttemptOutcome::Failed(FailureClass::SensitiveResponse)
        );
        assert_withheld(&error.attempts[0]);
        assert_safe_error(&error);
        assert!(!logs.exists());
        assert!(!fixture.captures().exists());
    }
}

#[tokio::test]
async fn sensitive_input_or_context_skips_logs_without_mutating_the_request() {
    let fixture = Fixture::new(|_| {});
    let logs = fixture.root.join("sensitive-prompt-logs");
    let options = InferenceOptions {
        fallback_on: Vec::new(),
        logging: LoggingPolicy::Directory(logs.clone()),
        ..fixture.options()
    };
    for key in KEYS {
        for in_input in [false, true] {
            let sensitive = format!("  synthetic prompt literal {key} \\path\n  ");
            let (input, context) = if in_input {
                (sensitive.as_str(), "  ordinary context\n  ")
            } else {
                ("  ordinary input\n  ", sensitive.as_str())
            };
            let backend = ScriptedBackend::new([
                Step::success("student", "{\"value\":7}"),
                Step::failure("student", FailureClass::Transport),
                Step::success("student", "{broken"),
            ]);
            let receipt = fixture
                .engine
                .run_inference_with_backend(TASK, input, Some(context), &options, &backend)
                .await
                .expect("sensitive prompt does not reclassify safe response");
            assert_eq!(receipt.parsed, json!({"value": 7}));
            assert!(
                receipt.warnings.is_empty(),
                "intentional logging suppression is not a write failure"
            );
            assert_no_credentials(&format!("{receipt:?}"));
            assert_no_credentials(&serde_json::to_string(&receipt).expect("receipt JSON"));
            for expected in [FailureClass::Transport, FailureClass::Malformed] {
                let error = fixture
                    .engine
                    .run_inference_with_backend(TASK, input, Some(context), &options, &backend)
                    .await
                    .expect_err("safe failure evidence omits prompt credentials");
                assert_eq!(error.last_failure, Some(expected));
                assert_safe_error(&error);
            }
            backend.assert_finished(&["student", "student", "student"]);
            for call in backend.calls() {
                let text = user_text(&call.messages);
                assert!(
                    text.contains(input),
                    "credential-bearing input is not rewritten"
                );
                assert!(
                    text.contains(context),
                    "credential-bearing context is not rewritten"
                );
                assert!(text.contains(key));
            }
            assert!(!logs.exists());
        }
    }
}

#[test]
fn identity_construction_rejects_credentials_in_public_name_or_destination() {
    let descriptor = model_spec(0)
        .into_descriptor("student", TASK)
        .expect("synthetic descriptor");
    let identity = ModelIdentity::from_descriptor(ModelRole::Student, &descriptor)
        .expect("validated identity");
    assert_eq!(identity.name, "student");
    assert_eq!(identity.destination, "https://student.invalid/v1");
    assert_no_credentials(&serde_json::to_string(&identity).expect("identity JSON"));
    let mut secret_name = descriptor.clone();
    secret_name.name = format!("model-{}", KEYS[0]);
    let mut secret_destination = descriptor.clone();
    secret_destination.base_url = Some(format!("https://student.invalid/{}", KEYS[0]));
    let mut userinfo = descriptor.clone();
    userinfo.base_url = Some("https://user:synthetic-password@student.invalid/v1".into());
    let mut query = descriptor;
    query.base_url = Some("https://student.invalid/v1?synthetic=private".into());
    for invalid in [secret_name, secret_destination, userinfo, query] {
        let error = ModelIdentity::from_descriptor(ModelRole::Student, &invalid)
            .expect_err("identity is an audited grant, not arbitrary config text");
        assert_eq!(error.kind, FailureClass::Configuration);
        assert!(error.attempts.is_empty());
        assert!(error.last_failure.is_none());
        assert_safe_error(&error);
    }
}

#[tokio::test]
async fn safe_success_and_failure_evidence_and_private_logs_exclude_descriptor_credentials() {
    let fixture = Fixture::new(|_| {});
    let logs = fixture.root.join("safe-logs");
    let options = InferenceOptions {
        fallback_on: Vec::new(),
        logging: LoggingPolicy::Directory(logs.clone()),
        ..fixture.options()
    };
    let backend = ScriptedBackend::new([
        Step::failure("student", FailureClass::Transport),
        Step::success("student", "{\"value\":7}"),
    ]);
    let error = fixture
        .run(&options, &backend)
        .await
        .expect_err("synthetic transport error");
    let receipt = fixture.run(&options, &backend).await.expect("safe success");
    backend.assert_finished(&["student", "student"]);
    assert_safe_error(&error);
    assert_no_credentials(&format!("{receipt:?}"));
    assert_no_credentials(&serde_json::to_string(&receipt).expect("receipt serialization"));
    for call in backend.calls() {
        assert_eq!(
            call.model.api_key.as_deref(),
            Some(KEYS[0]),
            "credential belongs only to transport descriptor"
        );
        assert_no_credentials(&call.messages.to_string());
    }
    let logs = records(&logs);
    assert_eq!(logs.len(), 2);
    assert!(
        logs.iter()
            .any(|log| log["attempt"]["outcome"] == json!({"failed": "transport"}))
    );
    assert!(
        logs.iter()
            .any(|log| log["attempt"]["outcome"] == "accepted")
    );
    assert!(receipt.warnings.is_empty());
}

#[tokio::test]
async fn collection_disabled_creates_no_files_even_after_successful_fallback() {
    let fixture = Fixture::new(|_| {});
    let before = files_under(&fixture.root);
    let backend = ScriptedBackend::new([
        Step::failure("student", FailureClass::Transport),
        Step::success("teacher", "{\"value\":7}"),
    ]);
    let receipt = fixture
        .run(&fixture.options(), &backend)
        .await
        .expect("fallback success without collection");
    backend.assert_finished(&["student", "teacher"]);
    assert_eq!(receipt.used_model.role, ModelRole::Teacher);
    assert!(receipt.warnings.is_empty());
    assert!(!fixture.captures().exists());
    assert_eq!(
        files_under(&fixture.root),
        before,
        "no receipts, logs or capture files when disabled"
    );
}

#[tokio::test]
async fn rejected_requests_never_capture_even_when_collection_is_enabled() {
    let fixture = Fixture::new(|config| config.data_collection.enabled = true);
    let before = files_under(&fixture.root);
    let options = InferenceOptions {
        fallback_on: Vec::new(),
        ..fixture.options()
    };
    for step in [
        Step::failure("student", FailureClass::Transport),
        Step::success("student", "{broken"),
        Step::success("student", "{\"value\":\"wrong\"}"),
        Step::completion("student", "{\"value\":7}", Termination::Length, None),
        Step::completion("student", "{\"value\":7}", Termination::Unknown, None),
        Step::completion("student", "", Termination::Refusal, None),
    ] {
        let backend = ScriptedBackend::new([step]);
        let error = fixture
            .run(&options, &backend)
            .await
            .expect_err("rejected attempt is not a captured prediction");
        assert_eq!(error.kind, FailureClass::Exhausted);
        backend.assert_finished(&["student"]);
        assert!(!fixture.captures().exists());
        assert_eq!(files_under(&fixture.root), before);
    }
}

#[tokio::test]
async fn fallback_capture_keeps_accepted_payload_and_failed_metadata_but_not_failed_payloads() {
    let fixture = Fixture::new(|config| config.data_collection.enabled = true);
    let invalid_raw = "  {\"value\":\"wrong\",\"note\":\"rejected-schema-private-note\"}\r\n";
    let truncated_raw = "{\"value\":7,\"note\":\"rejected-length-private-note\"}";
    let accepted_raw = " \n{\"value\":8,\"note\":\"accepted literal <think>note</think>\"}\r\n";
    let backend = ScriptedBackend::new([
        Step::success("student", invalid_raw),
        Step::completion("student", truncated_raw, Termination::Length, Some(usage())),
        Step::success("fallback-one", accepted_raw),
    ]);
    let options = InferenceOptions {
        max_retries: 1,
        allowed_fallbacks: vec![fixture.identity(ModelRole::Fallback(1))],
        ..fixture.options()
    };
    let input = "  original capture input\n  ";
    let context = "  original capture context\r\n  ";
    let receipt = fixture
        .engine
        .run_inference_with_backend(TASK, input, Some(context), &options, &backend)
        .await
        .expect("accepted fallback is captured");
    backend.assert_finished(&["student", "student", "fallback-one"]);
    assert_eq!(
        receipt.requested_model,
        fixture.identity(ModelRole::Student)
    );
    assert_eq!(receipt.used_model, fixture.identity(ModelRole::Fallback(1)));
    assert!(receipt.warnings.is_empty());
    assert_eq!(receipt.raw_completion, accepted_raw);
    assert_eq!(receipt.attempts.len(), 5);
    // Caller retains complete failure evidence, including a rejected parsed value.
    assert_eq!(
        receipt.attempts[0].raw_completion.as_deref(),
        Some(invalid_raw)
    );
    assert_eq!(
        receipt.attempts[0].parsed,
        Some(json!({"value": "wrong", "note": "rejected-schema-private-note"}))
    );
    assert_eq!(receipt.attempts[0].validation, ValidationStatus::Failed);
    assert_eq!(
        receipt.attempts[1].raw_completion.as_deref(),
        Some(truncated_raw)
    );
    assert_eq!(receipt.attempts[1].termination, Some(Termination::Length));
    assert_eq!(receipt.attempts[1].usage, Some(usage()));
    assert_eq!(
        receipt.attempts[2].outcome,
        AttemptOutcome::Failed(FailureClass::Denied)
    );
    assert_eq!(
        receipt.attempts[3].outcome,
        AttemptOutcome::Failed(FailureClass::Denied)
    );

    let captures = records(&fixture.captures().join(TASK));
    assert_eq!(
        captures.len(),
        1,
        "only the accepted request, not each attempt, is captured"
    );
    let capture = &captures[0];
    assert_eq!(capture["input_text"], input);
    assert_eq!(capture["additional_context"], context);
    assert_eq!(capture["qualification"], "unreviewed");
    assert!(
        capture.get("expected_json").is_none(),
        "a prediction never becomes a gold label"
    );
    assert_eq!(capture["receipt"]["raw_completion"], accepted_raw);
    assert_eq!(capture["receipt"]["parsed"], receipt.parsed);
    let returned = serde_json::to_value(&receipt).expect("full caller receipt JSON");
    let persisted = &capture["receipt"];
    for (index, attempt) in receipt.attempts.iter().enumerate() {
        let captured_attempt = &persisted["attempts"][index];
        let returned_attempt = &returned["attempts"][index];
        for field in [
            "ordinal",
            "model",
            "retry",
            "outcome",
            "termination",
            "usage",
            "validation",
        ] {
            assert_eq!(
                captured_attempt[field], returned_attempt[field],
                "failed metadata must remain truthful: {field}"
            );
        }
        if matches!(attempt.outcome, AttemptOutcome::Failed(_)) {
            assert!(captured_attempt["raw_completion"].is_null());
            assert!(captured_attempt["parsed"].is_null());
        } else {
            assert_eq!(
                captured_attempt, returned_attempt,
                "accepted evidence is unchanged"
            );
        }
    }
    for field in [
        "parsed",
        "raw_completion",
        "requested_model",
        "used_model",
        "validation",
        "termination",
        "usage",
        "completeness",
        "warnings",
    ] {
        assert_eq!(
            persisted[field], returned[field],
            "top-level receipt fidelity: {field}"
        );
    }
    let capture_text = capture.to_string();
    assert!(!capture_text.contains("rejected-schema-private-note"));
    assert!(!capture_text.contains("rejected-length-private-note"));
    assert!(capture_text.contains("accepted literal <think>note</think>"));
}

#[tokio::test]
async fn persistence_failure_warns_without_retrying_or_changing_success() {
    let fixture = Fixture::new(|config| {
        config.data_collection.enabled = true;
        config.data_collection.output_dir = Some("capture-blocker".into());
    });
    let capture_blocker = fixture.engine.paths().config_dir.join("capture-blocker");
    let log_blocker = fixture.root.join("log-blocker");
    fs::write(&capture_blocker, b"capture sentinel").expect("block collection with regular file");
    fs::write(&log_blocker, b"logging sentinel").expect("block logging with regular file");
    let before = files_under(&fixture.root);
    let raw = "  {\"value\":7}\r\n";
    let backend = ScriptedBackend::new([Step::success("student", raw)]);
    let options = InferenceOptions {
        max_retries: 4,
        logging: LoggingPolicy::Directory(log_blocker.clone()),
        ..fixture.options()
    };
    let receipt = fixture
        .run(&options, &backend)
        .await
        .expect("persistence cannot reclassify accepted inference");
    backend.assert_finished(&["student"]);
    assert_eq!(receipt.parsed, json!({"value": 7}));
    assert_eq!(receipt.raw_completion, raw);
    assert_eq!(receipt.validation, ValidationStatus::Passed);
    assert_eq!(receipt.usage, Some(usage()));
    assert_eq!(receipt.used_model, fixture.identity(ModelRole::Student));
    assert_eq!(receipt.attempts.len(), 1);
    assert_eq!(receipt.attempts[0].outcome, AttemptOutcome::Accepted);
    assert_eq!(
        receipt.warnings,
        vec!["attempt logging failed", "candidate capture failed"]
    );
    assert_eq!(
        fs::read(&capture_blocker).expect("unchanged capture blocker"),
        b"capture sentinel"
    );
    assert_eq!(
        fs::read(&log_blocker).expect("unchanged log blocker"),
        b"logging sentinel"
    );
    assert_eq!(
        files_under(&fixture.root),
        before,
        "no partial persistence artifacts"
    );
    assert_no_credentials(&serde_json::to_string(&receipt).expect("receipt serialization"));
}

#[tokio::test]
async fn multimodal_attempts_preserve_full_original_text_data_uris_order_and_history() {
    let fixture = Fixture::new(|_| {});
    // Synthetic signatures test transport, not a provider's image decoding.
    let images = vec![
        ImageInput::from_bytes(b"\x89PNG\r\n\x1a\n\x00\xfffirst")
            .to_data_uri()
            .expect("synthetic PNG"),
        ImageInput::from_bytes(b"GIF89a\x00\xffsecond")
            .to_data_uri()
            .expect("synthetic GIF"),
    ];
    let input = format!(
        "  \"full\" input\r\n{}\nLAST-INPUT-BYTE  ",
        "synthetic multimodal paragraph\n".repeat(200)
    );
    let context = "  all context\n```json <think>literal</think> \\path\r\nLAST-CONTEXT-BYTE  ";
    let plain_backend = ScriptedBackend::new([Step::success("student", "{\"value\":7}")]);
    fixture
        .engine
        .run_inference_with_backend(
            TASK,
            &input,
            Some(context),
            &fixture.options(),
            &plain_backend,
        )
        .await
        .expect("text-only baseline");
    plain_backend.assert_finished(&["student"]);
    let plain = plain_backend.calls()[0].messages.clone();
    let backend = ScriptedBackend::new([
        Step::success("student", "{\"value\":\"wrong\"}"),
        Step::success("student", "{\"value\":7}"),
    ]);
    let options = InferenceOptions {
        images: images.clone(),
        max_retries: 1,
        retry_mode: RetryMode::WithHistory,
        ..fixture.options()
    };
    let receipt = fixture
        .engine
        .run_inference_with_backend(TASK, &input, Some(context), &options, &backend)
        .await
        .expect("multimodal retry succeeds");
    backend.assert_finished(&["student", "student"]);
    assert_eq!(receipt.parsed, json!({"value": 7}));
    let calls = backend.calls();
    let original = calls[0]
        .messages
        .as_array()
        .expect("multimodal original turns");
    let plain_turns = plain.as_array().expect("text-only turns");
    assert_eq!(
        &original[..original.len() - 1],
        &plain_turns[..plain_turns.len() - 1]
    );
    let content = &original.last().expect("initial user turn")["content"];
    assert_eq!(
        content,
        &json!([
            {"type": "text", "text": user_text(&plain)},
            {"type": "image_url", "image_url": {"url": images[0]}},
            {"type": "image_url", "image_url": {"url": images[1]}}
        ])
    );
    assert!(user_text(&calls[0].messages).contains(&input));
    assert!(user_text(&calls[0].messages).contains(context));
    let retry = calls[1]
        .messages
        .as_array()
        .expect("multimodal retry turns");
    assert_eq!(&retry[..original.len()], original);
    assert_eq!(retry.len(), original.len() + 2);
    assert_eq!(
        retry[original.len()],
        json!({"role": "assistant", "content": "{\"value\":\"wrong\"}"})
    );
}

#[tokio::test]
async fn any_invalid_image_rejects_whole_request_before_any_backend_call() {
    let fixture = Fixture::new(|config| config.data_collection.enabled = true);
    let good = ImageInput::from_bytes(b"GIF89a")
        .to_data_uri()
        .expect("synthetic GIF");
    let before = files_under(&fixture.root);
    for invalid in [
        "not-a-data-uri".to_owned(),
        "https://images.invalid/private.png".to_owned(),
        "data:image/gif;base64,***".to_owned(),
        "data:image/gif;base64,".to_owned(),
        "data:image/png;base64,R0lGODlh".to_owned(), // Correct bytes, wrong MIME.
        "data:image/gif;base64,R0lGODlh\n".to_owned(),
        "data:image/gif;base64,cGxhaW4gdGV4dA==".to_owned(), // Not image bytes.
    ] {
        // Invalid image in either position cannot yield a partially sent request.
        for images in [
            vec![good.clone(), invalid.clone()],
            vec![invalid.clone(), good.clone()],
        ] {
            let backend = ScriptedBackend::new([]);
            let options = InferenceOptions {
                images,
                logging: LoggingPolicy::Directory(fixture.root.join("invalid-image-logs")),
                ..fixture.options()
            };
            let error = fixture
                .run(&options, &backend)
                .await
                .expect_err("all-or-error image admission");
            backend.assert_finished(&[]);
            assert_eq!(error.kind, FailureClass::Configuration);
            assert!(error.last_failure.is_none());
            assert!(error.attempts.is_empty());
            assert_safe_error(&error);
            assert!(!error.to_string().contains(&invalid));
            assert!(
                !serde_json::to_string(&error)
                    .expect("error JSON")
                    .contains(&invalid)
            );
            assert!(!fixture.captures().exists());
            assert_eq!(files_under(&fixture.root), before);
        }
    }
}
