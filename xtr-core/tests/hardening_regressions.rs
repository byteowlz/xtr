//! Synthetic regressions for xtr-38z4.2, .4 and .5.
//! Separate old-code runs proved cross-request model routing, accepted truncation,
//! and unreviewed predictions mislabeled as expected_json. The permanent tests
//! exercise the request-owned engine rather than DSPy's global settings path.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::panic_in_result_fn,
    reason = "Bounded synthetic regression fixtures fail immediately on setup errors"
)]

use std::io::Read;
use std::io::Write;
use std::net::TcpListener;
use std::sync::Mutex;
use std::thread;
use std::time::Duration;
use std::time::Instant;

use async_trait::async_trait;
use serde_json::Value;
use serde_json::json;
use xtr_core::config::AppConfig;
use xtr_core::config::AppPaths;
use xtr_core::config::ConfigBundle;
use xtr_core::config::ModelDescriptor;
use xtr_core::config::ModelSpec;
use xtr_core::config::TaskConfig;
use xtr_core::engine::ExtractionEngine;
use xtr_core::engine::ValidationMode;
use xtr_core::inference::Completion;
use xtr_core::inference::FailureClass;
use xtr_core::inference::InferenceBackend;
use xtr_core::inference::InferenceOptions;
use xtr_core::inference::LoggingPolicy;
use xtr_core::inference::ProviderUsage;
use xtr_core::inference::Termination;
use xtr_core::inference::ValidationStatus;

struct RequestLocalBackend {
    rendezvous: tokio::sync::Barrier,
    calls: Mutex<Vec<(String, Value)>>,
}

#[async_trait]
impl InferenceBackend for RequestLocalBackend {
    async fn complete(
        &self,
        model: &ModelDescriptor,
        messages: &Value,
    ) -> Result<Completion, FailureClass> {
        self.calls
            .lock()
            .unwrap()
            .push((model.name.clone(), messages.clone()));
        // Both requests must enter transport before either may return.
        self.rendezvous.wait().await;
        Ok(Completion {
            raw: Some(if model.name == "request-a" {
                "{\"value\":7}".into()
            } else {
                "[[ ## output_json ## ]]\n{\"value\":8}\n[[ ## completed ## ]]".into()
            }),
            termination: Termination::Stop,
            usage: Some(ProviderUsage {
                prompt_tokens: Some(11),
                completion_tokens: Some(5),
                total_tokens: Some(16),
            }),
        })
    }
}

#[tokio::test]
async fn concurrent_requests_keep_own_model_adapter_prompt_and_logs() {
    let root_a = tempfile::tempdir().unwrap();
    let root_b = tempfile::tempdir().unwrap();
    let a = isolated_engine_named(
        root_a.path(),
        "http://127.0.0.1:1/v1".into(),
        false,
        "request-a",
        "json",
    );
    let b = isolated_engine_named(
        root_b.path(),
        "http://127.0.0.1:1/v1".into(),
        false,
        "request-b",
        "chat-legacy",
    );
    let backend = RequestLocalBackend {
        rendezvous: tokio::sync::Barrier::new(2),
        calls: Mutex::new(vec![]),
    };
    let options_a = InferenceOptions {
        validation: ValidationMode::Error,
        logging: LoggingPolicy::Directory(a.paths().state_dir.join("logs-a")),
        ..Default::default()
    };
    let options_b = InferenceOptions {
        validation: ValidationMode::Error,
        logging: LoggingPolicy::Directory(b.paths().state_dir.join("logs-b")),
        ..Default::default()
    };
    let (receipt_a, receipt_b) = tokio::time::timeout(Duration::from_secs(5), async {
        tokio::join!(
            a.run_inference_with_backend(
                "synthetic",
                "only-a value is 7",
                Some("context-a"),
                &options_a,
                &backend
            ),
            b.run_inference_with_backend(
                "synthetic",
                "only-b value is 8",
                Some("context-b"),
                &options_b,
                &backend
            )
        )
    })
    .await
    .expect("concurrent engine calls must not serialize behind global settings");
    let receipt_a = receipt_a.unwrap();
    let receipt_b = receipt_b.unwrap();
    assert_eq!(receipt_a.used_model.name, "request-a");
    assert_eq!(receipt_b.used_model.name, "request-b");
    assert_eq!(receipt_a.parsed, json!({"value": 7}));
    assert_eq!(receipt_b.parsed, json!({"value": 8}));
    assert_eq!(receipt_a.validation, ValidationStatus::Passed);
    assert_eq!(receipt_b.validation, ValidationStatus::Passed);
    assert!(receipt_a.warnings.is_empty(), "{:?}", receipt_a.warnings);
    assert!(receipt_b.warnings.is_empty(), "{:?}", receipt_b.warnings);
    let calls = backend.calls.lock().unwrap();
    assert_eq!(calls.len(), 2);
    for (name, messages) in calls.iter() {
        let own = if name == "request-a" { "a" } else { "b" };
        let other = if own == "a" { "b" } else { "a" };
        let prompt = messages.to_string();
        assert!(prompt.contains(&format!("only-{own}")));
        assert!(prompt.contains(&format!("context-{own}")));
        assert!(
            prompt.contains(name),
            "own task instruction missing: {prompt}"
        );
        assert!(!prompt.contains(&format!("only-{other}")));
        assert!(!prompt.contains(&format!("context-{other}")));
    }
    for (engine, own, other) in [(&a, "a", "b"), (&b, "b", "a")] {
        let logs: Vec<_> = std::fs::read_dir(engine.paths().state_dir.join(format!("logs-{own}")))
            .unwrap()
            .collect();
        assert_eq!(logs.len(), 1);
        let log = std::fs::read_to_string(logs[0].as_ref().unwrap().path()).unwrap();
        assert!(log.contains(&format!("request-{own}")));
        assert!(!log.contains(&format!("request-{other}")));
    }
}

/// Bounded loopback-only OpenAI-compatible response. Read the complete request
/// before replying and return its JSON to prove an actual engine transport call.
fn mock_completion(finish_reason: &str) -> (String, thread::JoinHandle<Value>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}/v1", listener.local_addr().unwrap());
    listener.set_nonblocking(true).unwrap();
    let body = json!({
        "id": "synthetic-completion", "object": "chat.completion", "created": 0,
        "model": "synthetic-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "{\"value\":7}"},
            "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}
    })
    .to_string();
    let server = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut stream = loop {
            match listener.accept() {
                Ok((stream, _)) => break stream,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(
                        Instant::now() < deadline,
                        "mock received no inference request"
                    );
                    thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("mock accept failed: {error}"),
            }
        };
        // Accepted sockets can inherit nonblocking mode on macOS.
        stream.set_nonblocking(false).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut request = Vec::new();
        let (header_end, length) = loop {
            let mut chunk = [0_u8; 4096];
            let count = stream.read(&mut chunk).unwrap();
            assert!(count > 0, "request ended before headers");
            request.extend_from_slice(&chunk[..count]);
            if let Some(start) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&request[..start]);
                let length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().unwrap())
                    })
                    .expect("content-length header");
                break (start + 4, length);
            }
        };
        while request.len() < header_end + length {
            let mut chunk = [0_u8; 4096];
            let count = stream.read(&mut chunk).unwrap();
            assert!(count > 0, "request ended before body");
            request.extend_from_slice(&chunk[..count]);
        }
        let input = serde_json::from_slice(&request[header_end..header_end + length]).unwrap();
        write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).unwrap();
        stream.flush().unwrap();
        input
    });
    (endpoint, server)
}

fn isolated_engine(root: &std::path::Path, endpoint: String, capture: bool) -> ExtractionEngine {
    isolated_engine_named(root, endpoint, capture, "synthetic-model", "json")
}

fn isolated_engine_named(
    root: &std::path::Path,
    endpoint: String,
    capture: bool,
    name: &str,
    adapter: &str,
) -> ExtractionEngine {
    // macOS /var is a symlink; production private storage rejects symlink paths.
    let root = root.canonicalize().unwrap();
    let paths = AppPaths {
        app_name: "hardening-regression".into(),
        config_dir: root.join("config"),
        config_file: root.join("config/config.toml"),
        data_dir: root.join("data"),
        state_dir: root.join("state"),
        cache_dir: root.join("cache"),
    };
    paths.ensure_config_dir().unwrap();
    paths.ensure_runtime_dirs().unwrap();
    let schema = paths.config_dir.join("synthetic.schema.json");
    std::fs::write(
        &schema,
        json!({
            "type": "object", "properties": {"value": {"type": "integer"}, "note": {"type": "string"}},
            "required": ["value"], "additionalProperties": false
        })
        .to_string(),
    )
    .unwrap();
    let model = ModelSpec {
        name: Some(name.into()),
        base_url: Some(endpoint),
        api_key: None,
        adapter: Some(adapter.into()),
        request_timeout_secs: Some(5),
        max_tokens: Some(32),
    };
    let mut config = AppConfig::default();
    config.models.defaults.student = model.clone();
    config.models.defaults.teacher = model;
    config.models.defaults.fallbacks.clear();
    config.models.defaults.dataset_generator = None;
    config.tasks.insert(
        "synthetic".into(),
        TaskConfig {
            schema: Some(schema.to_string_lossy().into_owned()),
            examples: None,
            description: Some(format!("Extract the synthetic integer value for {name}.")),
            include_timestamp: Some(false),
            kind: None,
            metric: None,
        },
    );
    config.data_collection.enabled = capture;
    config.logging.verbose_llm_logging = false;
    ExtractionEngine::from_bundle(ConfigBundle { config, paths })
}

#[tokio::test]
async fn valid_json_with_length_finish_reason_is_rejected() {
    let root = tempfile::tempdir().unwrap();
    let (endpoint, server) = mock_completion("length");
    let engine = isolated_engine(root.path(), endpoint, false);
    let result = engine
        .run_inference(
            "synthetic",
            "value is 7",
            None,
            &InferenceOptions::default(),
        )
        .await;
    let request = server.join().unwrap();
    assert_eq!(request["model"], "synthetic-model");
    let error =
        result.expect_err("length-terminated valid JSON must not count as complete success");
    assert_eq!(error.last_failure, Some(FailureClass::Truncated));
    let attempt = &error.attempts[0];
    assert_eq!(attempt.raw_completion.as_deref(), Some("{\"value\":7}"));
    assert_eq!(attempt.termination, Some(Termination::Length));
    assert_eq!(attempt.usage.as_ref().unwrap().total_tokens, Some(16));
    assert!(!engine.paths().state_dir.join("collected_examples").exists());
}

#[tokio::test]
async fn opt_in_capture_is_unreviewed_observation_with_receipt_not_gold_label() {
    let root = tempfile::tempdir().unwrap();
    let (endpoint, server) = mock_completion("stop");
    let engine = isolated_engine(root.path(), endpoint, true);
    let receipt = engine
        .run_inference(
            "synthetic",
            "value is 7",
            None,
            &InferenceOptions::default(),
        )
        .await
        .unwrap();
    let request = server.join().unwrap();
    assert_eq!(request["model"], "synthetic-model");
    assert!(receipt.warnings.is_empty(), "{:?}", receipt.warnings);
    let directory = engine
        .paths()
        .state_dir
        .join("collected_examples/synthetic");
    let files: Vec<_> = std::fs::read_dir(directory).unwrap().collect();
    assert_eq!(files.len(), 1);
    let record: Value =
        serde_json::from_slice(&std::fs::read(files[0].as_ref().unwrap().path()).unwrap()).unwrap();
    assert_eq!(record["qualification"], "unreviewed");
    assert!(record.get("expected_json").is_none());
    assert_eq!(record["input_text"], "value is 7");
    assert_eq!(record["receipt"], serde_json::to_value(&receipt).unwrap());
    assert_eq!(record["receipt"]["raw_completion"], "{\"value\":7}");
    assert_eq!(record["receipt"]["parsed"], json!({"value": 7}));
    assert_eq!(record["receipt"]["used_model"]["name"], "synthetic-model");
    assert_eq!(record["receipt"]["usage"]["total_tokens"], 16);
}

struct FixedCompletion(String);

#[async_trait]
impl InferenceBackend for FixedCompletion {
    async fn complete(
        &self,
        _model: &ModelDescriptor,
        _messages: &Value,
    ) -> Result<Completion, FailureClass> {
        Ok(Completion {
            raw: Some(self.0.clone()),
            termination: Termination::Stop,
            usage: None,
        })
    }
}

#[tokio::test]
async fn complete_parser_and_schema_validation_reject_without_capture() {
    let root = tempfile::tempdir().unwrap();
    let engine = isolated_engine(root.path(), "http://127.0.0.1:1/v1".into(), true);
    let options = InferenceOptions {
        validation: ValidationMode::Error,
        ..Default::default()
    };
    for (raw, failure) in [
        ("{\"value\":7} {\"value\":8}", FailureClass::Malformed),
        ("Prose before {\"value\":7}", FailureClass::Malformed),
        ("{\"value\":7", FailureClass::Malformed),
        (
            "{\"value\":\"not an integer\"}",
            FailureClass::SchemaInvalid,
        ),
    ] {
        let backend = FixedCompletion(raw.into());
        let error = engine
            .run_inference_with_backend("synthetic", "synthetic input", None, &options, &backend)
            .await
            .unwrap_err();
        assert_eq!(error.last_failure, Some(failure), "raw: {raw}");
        assert_eq!(error.attempts[0].raw_completion.as_deref(), Some(raw));
        assert!(!engine.paths().state_dir.join("collected_examples").exists());
    }
    // Envelope-like text inside literal strings must never be stripped/repaired.
    let note = "  literal \\path\n```json <think>inside</think> {braces} 'quotes'  ";
    let parsed = json!({"value": 7, "note": note});
    let raw = parsed.to_string();
    let receipt = engine
        .run_inference_with_backend(
            "synthetic",
            "synthetic input",
            None,
            &options,
            &FixedCompletion(raw.clone()),
        )
        .await
        .unwrap();
    assert_eq!(receipt.parsed, parsed);
    assert_eq!(receipt.raw_completion, raw);
    assert_eq!(receipt.validation, ValidationStatus::Passed);
    assert!(receipt.usage.is_none(), "missing usage must remain unknown");
    assert!(receipt.warnings.is_empty(), "{:?}", receipt.warnings);
}
