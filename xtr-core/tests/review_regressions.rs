//! Review-discovered privacy regressions through the public inference seam.
//! All credentials are synthetic; config and persistence are isolated per test.
#![allow(
    clippy::expect_used,
    clippy::panic,
    reason = "Synthetic regression fixtures must fail immediately on setup or contract errors"
)]

use std::collections::VecDeque;
use std::fs;
use std::io::Read;
use std::io::Write;
use std::net::TcpListener;
use std::net::TcpStream;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::Duration;
use std::time::Instant;

use async_trait::async_trait;
use serde_json::Value;
use serde_json::json;
use tempfile::TempDir;
use xtr_core::config::AppConfig;
use xtr_core::config::AppPaths;
use xtr_core::config::ConfigBundle;
use xtr_core::config::FeedbackModelOverrides;
use xtr_core::config::ModelDescriptor;
use xtr_core::config::ModelSpec;
use xtr_core::config::OptimizationSettings;
use xtr_core::config::TaskConfig;
use xtr_core::config::TaskModelOverrides;
use xtr_core::engine::ExtractionEngine;
use xtr_core::engine::ValidationMode;
use xtr_core::inference::AttemptOutcome;
use xtr_core::inference::Completion;
use xtr_core::inference::ExtractionError;
use xtr_core::inference::ExtractionReceipt;
use xtr_core::inference::FailureClass;
use xtr_core::inference::InferenceBackend;
use xtr_core::inference::InferenceOptions;
use xtr_core::inference::LoggingPolicy;
use xtr_core::inference::ModelRole;
use xtr_core::inference::OpenAiBackend;
use xtr_core::inference::ProviderUsage;
use xtr_core::inference::Termination;
use xtr_core::inference::ValidationStatus;

const TASK: &str = "review";
const KEY: &str = "SYNTHETIC-REVIEW-CREDENTIAL-SENTINEL";
const DATASET_KEY: &str = "SYNTHETIC-DATASET-CREDENTIAL-SENTINEL";
const DEFAULT_FEEDBACK_KEYS: [&str; 2] = [
    "SYNTHETIC-DEFAULT-FEEDBACK-TEACHER-KEY",
    "SYNTHETIC-DEFAULT-FEEDBACK-STUDENT-KEY",
];
const TASK_FEEDBACK_KEYS: [&str; 2] = [
    "SYNTHETIC-TASK-FEEDBACK-TEACHER-KEY",
    "SYNTHETIC-TASK-FEEDBACK-STUDENT-KEY",
];
const OTHER_TASK_KEYS: [&str; 5] = [
    "SYNTHETIC-OTHER-TASK-STUDENT-KEY",
    "SYNTHETIC-OTHER-TASK-TEACHER-KEY",
    "SYNTHETIC-OTHER-TASK-FALLBACK-KEY",
    "SYNTHETIC-OTHER-TASK-FEEDBACK-TEACHER-KEY",
    "SYNTHETIC-OTHER-TASK-FEEDBACK-STUDENT-KEY",
];
const DEADLINE: Duration = Duration::from_secs(5);

struct Fixture {
    _temporary: TempDir,
    root: PathBuf,
    schema: PathBuf,
    engine: ExtractionEngine,
}

impl Fixture {
    fn new(configure: impl FnOnce(&mut AppConfig)) -> Self {
        let temporary = tempfile::tempdir().expect("isolated review fixture");
        // macOS /var symlinks must not trip private persistence admission.
        let root = temporary
            .path()
            .canonicalize()
            .expect("canonical temp root");
        let paths = AppPaths {
            app_name: "review-regressions".into(),
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
        let schema = paths.config_dir.join("review.schema.json");
        fs::write(&schema, json!({
            "type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]
        }).to_string()).expect("synthetic schema");
        let mut config = AppConfig::default();
        config.models.defaults.student = spec("student", Some(KEY));
        config.models.defaults.teacher = spec("teacher", None);
        config.models.defaults.fallbacks.clear();
        config.models.defaults.dataset_generator = None;
        config.tasks.insert(
            TASK.into(),
            TaskConfig {
                schema: Some(schema.to_string_lossy().into_owned()),
                include_timestamp: Some(false),
                ..Default::default()
            },
        );
        config.data_collection.enabled = true;
        config.logging.verbose_llm_logging = false;
        configure(&mut config);
        Self {
            _temporary: temporary,
            root,
            schema,
            engine: ExtractionEngine::from_bundle(ConfigBundle { config, paths }),
        }
    }

    fn options(&self) -> InferenceOptions {
        InferenceOptions {
            validation: ValidationMode::Error,
            allowed_fallbacks: vec![
                self.engine
                    .fallback_identity(TASK, ModelRole::Teacher)
                    .expect("exact teacher identity"),
            ],
            fallback_on: Vec::new(),
            logging: LoggingPolicy::Directory(self.logs()),
            ..Default::default()
        }
    }

    fn logs(&self) -> PathBuf {
        self.root.join("logs")
    }
    fn captures(&self) -> PathBuf {
        self.engine.paths().state_dir.join("collected_examples")
    }

    async fn run<B: InferenceBackend>(
        &self,
        input: &str,
        context: Option<&str>,
        options: &InferenceOptions,
        backend: &B,
    ) -> Result<ExtractionReceipt, ExtractionError> {
        self.engine
            .run_inference_with_backend(TASK, input, context, options, backend)
            .await
    }
}

fn spec(name: &str, key: Option<&str>) -> ModelSpec {
    ModelSpec {
        name: Some(name.into()),
        base_url: Some(format!("https://{name}.invalid/v1")),
        api_key: key.map(str::to_owned),
        adapter: Some("json".into()),
        request_timeout_secs: Some(3),
        max_tokens: Some(32),
    }
}

fn completion(raw: impl Into<String>, termination: Termination) -> Completion {
    Completion {
        raw: Some(raw.into()),
        termination,
        usage: Some(ProviderUsage {
            prompt_tokens: Some(11),
            completion_tokens: Some(5),
            total_tokens: Some(16),
        }),
    }
}

struct FixedBackend {
    response: Completion,
    calls: Mutex<Vec<Value>>,
}

impl FixedBackend {
    fn new(response: Completion) -> Self {
        Self {
            response,
            calls: Mutex::new(Vec::new()),
        }
    }
    fn calls(&self) -> Vec<Value> {
        self.calls.lock().expect("mock call lock").clone()
    }
}

#[async_trait]
impl InferenceBackend for FixedBackend {
    async fn complete(
        &self,
        _model: &ModelDescriptor,
        messages: &Value,
    ) -> Result<Completion, FailureClass> {
        {
            self.calls
                .lock()
                .expect("mock call lock")
                .push(messages.clone());
        }
        Ok(self.response.clone())
    }
}

fn files_under(path: &Path) -> Vec<PathBuf> {
    if !path.exists() {
        return Vec::new();
    }
    let mut files = Vec::new();
    for entry in fs::read_dir(path).expect("read fixture directory") {
        let entry = entry.expect("fixture entry");
        if entry.file_type().expect("fixture entry type").is_dir() {
            files.extend(files_under(&entry.path()));
        } else {
            files.push(entry.path());
        }
    }
    files.sort();
    files
}

fn assert_no_secret(text: &str, key: &str) {
    assert!(
        !text.contains(key),
        "configured synthetic credential escaped into public or persisted evidence"
    );
    // Checking JSON after decoding also catches escaped credentials in member names.
    if let Ok(parsed) = serde_json::from_str::<Value>(text) {
        assert!(
            !parsed.to_string().contains(key),
            "JSON escaping must not hide a persisted credential"
        );
    }
}

fn assert_private_artifacts_safe(fixture: &Fixture, key: &str) {
    for directory in [fixture.logs(), fixture.captures()] {
        for path in files_under(&directory) {
            assert_no_secret(&fs::read_to_string(path).expect("private JSON record"), key);
        }
    }
}

fn assert_sensitive(
    result: Result<ExtractionReceipt, ExtractionError>,
    fixture: &Fixture,
    key: &str,
) {
    let error = result.expect_err("encoded/configured credential must be SensitiveResponse, not admitted or ordinary schema evidence");
    assert_eq!(error.kind, FailureClass::Exhausted);
    assert_eq!(error.last_failure, Some(FailureClass::SensitiveResponse));
    assert_eq!(error.attempts.len(), 1);
    let attempt = &error.attempts[0];
    assert_eq!(
        attempt.outcome,
        AttemptOutcome::Failed(FailureClass::SensitiveResponse)
    );
    assert!(attempt.raw_completion.is_none());
    assert!(attempt.parsed.is_none());
    assert!(attempt.termination.is_none());
    assert!(attempt.usage.is_none());
    assert_eq!(attempt.validation, ValidationStatus::NotChecked);
    assert!(attempt.validation_errors.is_empty());
    assert_no_secret(&format!("{error:?}"), key);
    assert_no_secret(&error.to_string(), key);
    assert_no_secret(&serde_json::to_string(&error).expect("error JSON"), key);
    assert!(
        !fixture.logs().exists(),
        "sensitive response must not be logged"
    );
    assert!(
        !fixture.captures().exists(),
        "sensitive response must not be captured"
    );
    assert_private_artifacts_safe(fixture, key);
}

fn escaped_key() -> String {
    // Independent known encoding: raw does not contain KEY, parsed JSON does.
    format!(r"\u0053{}", &KEY[1..])
}

#[tokio::test]
async fn review_escaped_credential_value_is_sensitive_and_withheld() {
    let fixture = Fixture::new(|_| {});
    let raw = format!(r#"{{"value":7,"note":"{}"}}"#, escaped_key());
    assert!(!raw.contains(KEY));
    assert_eq!(
        serde_json::from_str::<Value>(&raw).expect("escaped fixture JSON")["note"],
        KEY
    );
    let backend = FixedBackend::new(completion(raw, Termination::Stop));
    let result = fixture
        .run("safe input", None, &fixture.options(), &backend)
        .await;
    assert_eq!(backend.calls().len(), 1);
    assert_sensitive(result, &fixture, KEY);
}

#[tokio::test]
async fn review_escaped_credential_property_name_is_sensitive_and_withheld() {
    let fixture = Fixture::new(|_| {});
    let raw = format!(r#"{{"value":7,"{}":"ordinary"}}"#, escaped_key());
    assert!(!raw.contains(KEY));
    assert_eq!(
        serde_json::from_str::<Value>(&raw).expect("escaped fixture JSON")[KEY],
        "ordinary"
    );
    let backend = FixedBackend::new(completion(raw, Termination::Stop));
    let result = fixture
        .run("safe input", None, &fixture.options(), &backend)
        .await;
    assert_eq!(backend.calls().len(), 1);
    assert_sensitive(result, &fixture, KEY);
}

#[tokio::test]
async fn review_schema_const_credential_in_validation_diagnostics_is_withheld() {
    let fixture = Fixture::new(|_| {});
    fs::write(
        &fixture.schema,
        json!({
            "type": "object",
            "properties": {"value": {"type": "integer"}, "note": {"const": KEY}},
            "required": ["value", "note"]
        })
        .to_string(),
    )
    .expect("schema with synthetic credential const");
    for validation in [ValidationMode::Warn, ValidationMode::Error] {
        let backend = FixedBackend::new(completion(
            "{\"value\":7,\"note\":\"ordinary\"}",
            Termination::Stop,
        ));
        let options = InferenceOptions {
            validation,
            ..fixture.options()
        };
        let result = fixture.run("safe input", None, &options, &backend).await;
        assert_eq!(backend.calls().len(), 1);
        assert_sensitive(result, &fixture, KEY);
    }
}

async fn sensitive_prompt_does_not_capture(in_input: bool) {
    let fixture = Fixture::new(|_| {});
    let sensitive = format!("  original synthetic {KEY} \\path\r\n  ");
    let (input, context) = if in_input {
        (sensitive.as_str(), "  safe context\n  ")
    } else {
        ("  safe input\n  ", sensitive.as_str())
    };
    let backend = FixedBackend::new(completion(" {\"value\":7}\r\n", Termination::Stop));
    let receipt = fixture
        .run(input, Some(context), &fixture.options(), &backend)
        .await
        .expect("safe completion remains a success");
    assert_eq!(receipt.parsed, json!({"value": 7}));
    assert_eq!(receipt.raw_completion, " {\"value\":7}\r\n");
    assert_eq!(receipt.validation, ValidationStatus::Passed);
    let calls = backend.calls();
    assert_eq!(calls.len(), 1);
    let user = calls[0]
        .as_array()
        .expect("wire turns")
        .last()
        .expect("initial user turn")["content"]
        .as_str()
        .expect("text user turn");
    assert!(
        user.contains(input),
        "backend must receive unchanged full input"
    );
    assert!(
        user.contains(context),
        "backend must receive unchanged full context"
    );
    assert_no_secret(&serde_json::to_string(&receipt).expect("receipt JSON"), KEY);
    assert!(!fixture.logs().exists(), "sensitive prompts suppress logs");
    assert!(
        !fixture.captures().exists(),
        "collection must suppress credential-bearing input/context, not silently persist it"
    );
    assert_private_artifacts_safe(&fixture, KEY);
}

#[tokio::test]
async fn review_sensitive_input_is_preserved_for_backend_but_never_captured() {
    sensitive_prompt_does_not_capture(true).await;
}

#[tokio::test]
async fn review_sensitive_context_is_preserved_for_backend_but_never_captured() {
    sensitive_prompt_does_not_capture(false).await;
}

fn feedback(teacher_key: &str, student_key: &str) -> FeedbackModelOverrides {
    FeedbackModelOverrides {
        teacher: Some(spec("feedback-teacher", Some(teacher_key))),
        student: Some(spec("feedback-student", Some(student_key))),
    }
}

fn inventory_fixture() -> Fixture {
    Fixture::new(|config| {
        config.models.defaults.dataset_generator =
            Some(spec("dataset-generator", Some(DATASET_KEY)));
        config.optimization.defaults.feedback_models =
            feedback(DEFAULT_FEEDBACK_KEYS[0], DEFAULT_FEEDBACK_KEYS[1]);
        config.optimization.tasks.insert(
            TASK.into(),
            OptimizationSettings {
                feedback_models: feedback(TASK_FEEDBACK_KEYS[0], TASK_FEEDBACK_KEYS[1]),
                ..Default::default()
            },
        );
        config.models.tasks.insert(
            "other-task".into(),
            TaskModelOverrides {
                student: Some(spec("other-student", Some(OTHER_TASK_KEYS[0]))),
                teacher: Some(spec("other-teacher", Some(OTHER_TASK_KEYS[1]))),
                fallbacks: vec![spec("other-fallback", Some(OTHER_TASK_KEYS[2]))],
            },
        );
        config.optimization.tasks.insert(
            "other-task".into(),
            OptimizationSettings {
                feedback_models: feedback(OTHER_TASK_KEYS[3], OTHER_TASK_KEYS[4]),
                ..Default::default()
            },
        );
    })
}

async fn inventory_echo_is_sensitive(key: &str) {
    let fixture = inventory_fixture();
    let backend = FixedBackend::new(completion(
        json!({"value": 7, "note": key}).to_string(),
        Termination::Stop,
    ));
    let result = fixture
        .run("ordinary input", None, &fixture.options(), &backend)
        .await;
    assert_eq!(
        backend.calls().len(),
        1,
        "no non-inference model may be invoked"
    );
    assert_sensitive(result, &fixture, key);
}

#[tokio::test]
async fn review_dataset_generator_credential_echo_is_sensitive() {
    inventory_echo_is_sensitive(DATASET_KEY).await;
}

#[tokio::test]
async fn review_default_feedback_teacher_credential_echo_is_sensitive() {
    inventory_echo_is_sensitive(DEFAULT_FEEDBACK_KEYS[0]).await;
}

#[tokio::test]
async fn review_default_feedback_student_credential_echo_is_sensitive() {
    inventory_echo_is_sensitive(DEFAULT_FEEDBACK_KEYS[1]).await;
}

#[tokio::test]
async fn review_task_feedback_teacher_credential_echo_is_sensitive() {
    inventory_echo_is_sensitive(TASK_FEEDBACK_KEYS[0]).await;
}

#[tokio::test]
async fn review_task_feedback_student_credential_echo_is_sensitive() {
    inventory_echo_is_sensitive(TASK_FEEDBACK_KEYS[1]).await;
}

#[tokio::test]
async fn review_other_task_model_and_feedback_credentials_are_sensitive() {
    for key in OTHER_TASK_KEYS {
        inventory_echo_is_sensitive(key).await;
    }
}

async fn inventory_prompt_skips_logs(keys: &[&str]) {
    for key in keys {
        for in_context in [false, true] {
            let fixture = inventory_fixture();
            let sensitive = format!("  original inventory prompt {key}\r\n  ");
            let (input, context) = if in_context {
                ("ordinary input", sensitive.as_str())
            } else {
                (sensitive.as_str(), "ordinary context")
            };
            let backend = FixedBackend::new(completion("{\"value\":7}", Termination::Stop));
            let receipt = fixture
                .run(input, Some(context), &fixture.options(), &backend)
                .await
                .expect("safe completion does not mutate sensitive prompts");
            assert_eq!(receipt.parsed, json!({"value": 7}));
            let calls = backend.calls();
            assert_eq!(calls.len(), 1);
            let user = calls[0]
                .as_array()
                .expect("wire turns")
                .last()
                .expect("user turn")["content"]
                .as_str()
                .expect("text user content");
            assert!(user.contains(input));
            assert!(user.contains(context));
            assert_no_secret(&serde_json::to_string(&receipt).expect("receipt JSON"), key);
            assert!(
                !fixture.logs().exists(),
                "entire config credential inventory must protect prompt logs"
            );
            assert!(
                !fixture.captures().exists(),
                "entire config credential inventory must protect capture"
            );
            assert_private_artifacts_safe(&fixture, key);
        }
    }
}

#[tokio::test]
async fn review_dataset_generator_credential_in_prompt_skips_logs_and_capture() {
    inventory_prompt_skips_logs(&[DATASET_KEY]).await;
}

#[tokio::test]
async fn review_feedback_credentials_in_prompt_skip_logs_and_capture() {
    inventory_prompt_skips_logs(&DEFAULT_FEEDBACK_KEYS).await;
    inventory_prompt_skips_logs(&TASK_FEEDBACK_KEYS).await;
}

#[tokio::test]
async fn review_other_task_credentials_in_prompt_skip_logs_and_capture() {
    inventory_prompt_skips_logs(&OTHER_TASK_KEYS).await;
}

struct RecoveryBackend {
    state: Mutex<(VecDeque<(&'static str, Completion)>, Vec<String>)>,
}

impl RecoveryBackend {
    fn new(reason: &str) -> Self {
        Self {
            state: Mutex::new((
                VecDeque::from([
                    (
                        "student",
                        completion(
                            "{\"value\":7,\"note\":\"rejected-note\"}",
                            Termination::Other(reason.into()),
                        ),
                    ),
                    (
                        "teacher",
                        completion(" {\"value\":8}\r\n", Termination::Stop),
                    ),
                ]),
                Vec::new(),
            )),
        }
    }

    fn assert_finished(&self) {
        let state = self.state.lock().expect("recovery script lock");
        assert!(state.0.is_empty());
        assert_eq!(state.1, ["student", "teacher"]);
    }
}

#[async_trait]
impl InferenceBackend for RecoveryBackend {
    async fn complete(
        &self,
        model: &ModelDescriptor,
        _messages: &Value,
    ) -> Result<Completion, FailureClass> {
        let mut state = self.state.lock().expect("recovery script lock");
        let (expected, response) = state.0.pop_front().expect("unexpected recovery attempt");
        assert_eq!(model.name, expected);
        state.1.push(model.name.clone());
        Ok(response)
    }
}

#[tokio::test]
async fn review_recovered_other_termination_capture_uses_bounded_classification_not_private_reason()
{
    let fixture = Fixture::new(|_| {});
    let reasons = [
        "PRIVATE-REVIEW-TERMINATION-DETAIL-SENTINEL-alpha",
        "UNRELATED-PRIVATE-REVIEW-TERMINATION-DETAIL-SENTINEL-beta",
    ];
    let options = InferenceOptions {
        fallback_on: vec![FailureClass::UnsupportedTermination],
        logging: LoggingPolicy::Disabled,
        ..fixture.options()
    };
    for reason in reasons {
        let backend = RecoveryBackend::new(reason);
        let receipt = fixture
            .run("safe input", None, &options, &backend)
            .await
            .expect("explicit unsupported-termination recovery");
        backend.assert_finished();
        assert_eq!(receipt.used_model.role, ModelRole::Teacher);
        assert_eq!(receipt.parsed, json!({"value": 8}));
        assert_eq!(receipt.raw_completion, " {\"value\":8}\r\n");
        assert_eq!(receipt.attempts.len(), 2);
        assert_eq!(
            receipt.attempts[0].outcome,
            AttemptOutcome::Failed(FailureClass::UnsupportedTermination)
        );
        assert!(receipt.warnings.is_empty());
    }
    let files = files_under(&fixture.captures());
    assert_eq!(files.len(), 2);
    let mut classifications = Vec::new();
    for path in files {
        let text = fs::read_to_string(path).expect("capture JSON");
        for reason in reasons {
            assert!(
                !text.contains(reason),
                "capture must not retain provider-supplied private termination reasons"
            );
        }
        assert!(!text.contains("rejected-note"));
        let record: Value = serde_json::from_str(&text).expect("complete capture JSON");
        let failed = &record["receipt"]["attempts"][0];
        assert_eq!(
            failed["outcome"],
            json!({"failed": "unsupported_termination"})
        );
        assert_eq!(failed["model"]["role"], "student");
        assert_eq!(failed["ordinal"], 1);
        assert_eq!(failed["retry"], 0);
        assert!(failed["raw_completion"].is_null());
        assert!(failed["parsed"].is_null());
        assert_eq!(failed["usage"]["total_tokens"], 16);
        assert!(
            !failed["termination"].is_null(),
            "keep bounded classification, not false absence of metadata"
        );
        classifications.push(failed["termination"].clone());
        assert_eq!(record["receipt"]["raw_completion"], " {\"value\":8}\r\n");
        assert_eq!(record["receipt"]["parsed"], json!({"value": 8}));
    }
    // Distinct arbitrary provider strings must collapse to the same safe label.
    assert_eq!(
        classifications[0], classifications[1],
        "classification cannot be a truncated or transformed provider reason"
    );
    assert!(
        classifications[0].to_string().len() <= 64,
        "bounded metadata label"
    );
}

/// Repeat a synthetic wire response so erroneous retries/fallbacks are counted,
/// not disguised as transport failures after a one-shot server closes.
struct WireServer {
    endpoint: String,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<Vec<Value>>>,
}

impl WireServer {
    fn new(response: Value) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("loopback listener");
        listener.set_nonblocking(true).expect("bounded accept loop");
        let endpoint = format!(
            "http://{}/v1",
            listener.local_addr().expect("loopback address")
        );
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let worker = thread::spawn(move || {
            let deadline = Instant::now() + DEADLINE;
            let body = response.to_string();
            let mut requests = Vec::new();
            while !worker_stop.load(Ordering::Acquire) {
                assert!(
                    Instant::now() < deadline,
                    "loopback test exceeded its deadline"
                );
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        // Accepted sockets can inherit the listener's nonblocking mode.
                        stream
                            .set_nonblocking(false)
                            .expect("blocking bounded wire socket");
                        stream
                            .set_read_timeout(Some(Duration::from_secs(2)))
                            .expect("bounded request read");
                        stream
                            .set_write_timeout(Some(Duration::from_secs(2)))
                            .expect("bounded response write");
                        requests.push(read_wire_request(&mut stream));
                        write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).expect("synthetic wire response");
                        stream.flush().expect("flush wire response");
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2))
                    }
                    Err(error) => panic!("loopback accept failed: {error}"),
                }
            }
            requests
        });
        Self {
            endpoint,
            stop,
            worker: Some(worker),
        }
    }

    fn finish(&mut self) -> Vec<Value> {
        self.stop.store(true, Ordering::Release);
        self.worker
            .take()
            .expect("finish loopback worker once")
            .join()
            .expect("loopback server completed")
    }
}

impl Drop for WireServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            // Cleanup during assertion unwind must not trigger a double panic.
            let _ = worker.join();
        }
    }
}

fn read_wire_request(stream: &mut TcpStream) -> Value {
    let mut bytes = Vec::new();
    let (header_end, length) = loop {
        let mut chunk = [0_u8; 4096];
        let count = stream
            .read(&mut chunk)
            .expect("read synthetic request headers");
        assert!(count > 0, "request ended before headers");
        bytes.extend_from_slice(&chunk[..count]);
        assert!(bytes.len() < 1024 * 1024, "bounded synthetic request");
        if let Some(index) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&bytes[..index]);
            let length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().expect("request length"))
                })
                .expect("content-length header");
            assert!(length < 1024 * 1024);
            break (index + 4, length);
        }
    };
    while bytes.len() < header_end + length {
        let mut chunk = [0_u8; 4096];
        let count = stream
            .read(&mut chunk)
            .expect("read synthetic request body");
        assert!(count > 0, "request ended before body");
        bytes.extend_from_slice(&chunk[..count]);
    }
    serde_json::from_slice(&bytes[header_end..header_end + length]).expect("complete request JSON")
}

struct CountingWire {
    inner: OpenAiBackend,
    models: Mutex<Vec<String>>,
}

impl CountingWire {
    fn new() -> Self {
        Self {
            inner: OpenAiBackend::new().expect("local wire client"),
            models: Mutex::new(Vec::new()),
        }
    }
    fn models(&self) -> Vec<String> {
        self.models.lock().expect("wire call lock").clone()
    }
}

#[async_trait]
impl InferenceBackend for CountingWire {
    async fn complete(
        &self,
        model: &ModelDescriptor,
        messages: &Value,
    ) -> Result<Completion, FailureClass> {
        {
            self.models
                .lock()
                .expect("wire call lock")
                .push(model.name.clone());
        }
        self.inner.complete(model, messages).await
    }
}

fn wire_response(content: Value, finish_reason: Value) -> Value {
    json!({
        "choices": [{"message": {"role": "assistant", "content": content, "refusal": null}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}
    })
}

async fn terminal_wire(response: Value, expected: FailureClass, termination: Termination) {
    let mut server = WireServer::new(response);
    let fixture = Fixture::new(|config| {
        config.models.defaults.student.base_url = Some(server.endpoint.clone());
        config.models.defaults.teacher.base_url = Some(server.endpoint.clone());
    });
    let backend = CountingWire::new();
    let options = InferenceOptions {
        max_retries: 1,
        fallback_on: InferenceOptions::default().fallback_on,
        ..fixture.options()
    };
    let result = tokio::time::timeout(
        DEADLINE,
        fixture.run("safe input", None, &options, &backend),
    )
    .await
    .expect("bounded inference on loopback wire");
    let requests = server.finish();
    assert_eq!(
        backend.models(),
        ["student"],
        "terminal null-content response must not become malformed retry/fallback"
    );
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["model"], "student");
    let error = result.expect_err("terminal metadata rejects response");
    assert_eq!(error.kind, FailureClass::Exhausted);
    assert_eq!(error.last_failure, Some(expected.clone()));
    assert_eq!(error.attempts.len(), 1);
    let attempt = &error.attempts[0];
    assert_eq!(attempt.outcome, AttemptOutcome::Failed(expected));
    assert_eq!(attempt.termination, Some(termination));
    assert_eq!(
        attempt.usage.as_ref().expect("reported usage").total_tokens,
        Some(16)
    );
    assert!(attempt.parsed.is_none());
    assert!(!fixture.captures().exists());
}

#[tokio::test]
async fn review_null_content_content_filter_is_terminal_refusal_without_retry_or_fallback() {
    terminal_wire(
        wire_response(Value::Null, json!("content_filter")),
        FailureClass::Refusal,
        Termination::Refusal,
    )
    .await;
}

#[tokio::test]
async fn review_null_content_tool_calls_is_terminal_without_retry_or_fallback() {
    terminal_wire(
        wire_response(Value::Null, json!("tool_calls")),
        FailureClass::UnsupportedTermination,
        Termination::ToolCalls,
    )
    .await;
}

#[tokio::test]
async fn review_unsupported_finish_with_malformed_refusal_never_retries_or_falls_back() {
    for refusal in [json!(false), json!({"unexpected": "metadata"})] {
        let mut response = wire_response(json!("{\"value\":7}"), json!("vendor_end"));
        response["choices"][0]["message"]["refusal"] = refusal;
        terminal_wire(
            response,
            FailureClass::UnsupportedTermination,
            Termination::Other("vendor_end".into()),
        )
        .await;
    }
}

#[tokio::test]
async fn review_malformed_finish_reason_types_are_not_unknown_opt_in_successes() {
    for reason in [
        json!(42),
        json!(false),
        json!([]),
        json!({"private": "metadata"}),
    ] {
        let mut server = WireServer::new(wire_response(json!("{\"value\":7}"), reason.clone()));
        let fixture = Fixture::new(|config| {
            config.models.defaults.student.base_url = Some(server.endpoint.clone());
            config.models.defaults.teacher.base_url = Some(server.endpoint.clone());
        });
        let backend = CountingWire::new();
        let options = InferenceOptions {
            allow_unknown_termination: true,
            ..fixture.options()
        };
        let result = tokio::time::timeout(
            DEADLINE,
            fixture.run("safe input", None, &options, &backend),
        )
        .await
        .expect("bounded malformed metadata wire request");
        let requests = server.finish();
        assert_eq!(backend.models(), ["student"]);
        assert_eq!(requests.len(), 1);
        let error = result.expect_err("unknown opt-in cannot accept wrongly typed finish_reason");
        assert_eq!(error.kind, FailureClass::Exhausted, "reason: {reason}");
        assert_eq!(error.last_failure, Some(FailureClass::Malformed));
        assert_eq!(error.attempts.len(), 1);
        assert_eq!(
            error.attempts[0].outcome,
            AttemptOutcome::Failed(FailureClass::Malformed)
        );
        assert!(error.attempts[0].parsed.is_none());
        assert!(!fixture.captures().exists());
    }
}
