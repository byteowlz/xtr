//! Process-level CLI contract tests: synthetic loopback providers and private,
//! isolated HOME/XDG roots; no inherited provider credentials or live settings.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "Bounded synthetic subprocess fixtures must fail immediately on setup errors"
)]

use std::fs;
use std::io::Read;
use std::io::Write;
use std::net::TcpListener;
use std::net::TcpStream;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::process::Output;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::Duration;

use serde_json::Value;
use serde_json::json;

const SENTINEL: &str = "synthetic-header-only-sentinel";
const PNG: &[u8] = b"\x89PNG\r\n\x1a\n\x00\xff";
const PNG_BASE64: &str = "iVBORw0KGgoA/w==";
const PNG_URI: &str = "data:image/png;base64,iVBORw0KGgoA/w==";

#[derive(Clone, Debug)]
struct Request {
    headers: String,
    body: Value,
}

struct MockServer {
    endpoint: String,
    calls: Arc<Mutex<Vec<Request>>>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}

impl MockServer {
    fn new(raw: &str, finish: Option<&str>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}/v1", listener.local_addr().unwrap());
        let calls = Arc::new(Mutex::new(vec![]));
        let stop = Arc::new(AtomicBool::new(false));
        let seen = calls.clone();
        let stopping = stop.clone();
        let mut choice = json!({"index": 0, "message": {"role": "assistant", "content": raw}});
        if let Some(finish) = finish {
            choice["finish_reason"] = json!(finish);
        }
        let response = json!({
            "id": "synthetic-cli", "object": "chat.completion", "created": 0,
            "model": "synthetic-response-model", "choices": [choice],
            "usage": {"prompt_tokens": 13, "completion_tokens": 6, "total_tokens": 19}
        })
        .to_string();
        let worker = thread::spawn(move || {
            while !stopping.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let request = read_request(&mut stream);
                        seen.lock().unwrap().push(request);
                        write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", response.len(), response).unwrap();
                        stream.flush().unwrap();
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                    }
                    Err(error) => panic!("mock accept: {error}"),
                }
            }
        });
        Self {
            endpoint,
            calls,
            stop,
            worker: Some(worker),
        }
    }

    fn requests(&self) -> Vec<Request> {
        self.calls.lock().unwrap().clone()
    }
}

impl Drop for MockServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(worker) = self.worker.take() {
            let result = worker.join();
            if !thread::panicking() {
                assert!(result.is_ok(), "mock server thread failed");
            }
        }
    }
}

fn read_request(stream: &mut TcpStream) -> Request {
    // Accepted sockets can inherit O_NONBLOCK on macOS; timeouts need blocking I/O.
    stream.set_nonblocking(false).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    stream
        .set_write_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    let mut bytes = Vec::new();
    let (headers, offset, length) = loop {
        let mut chunk = [0; 4096];
        let count = stream.read(&mut chunk).unwrap();
        assert!(count > 0, "request ended before headers");
        bytes.extend_from_slice(&chunk[..count]);
        if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
            let headers = String::from_utf8(bytes[..end].to_vec()).unwrap();
            let length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .expect("content-length");
            break (headers, end + 4, length);
        }
    };
    while bytes.len() < offset + length {
        let mut chunk = [0; 4096];
        let count = stream.read(&mut chunk).unwrap();
        assert!(count > 0, "request ended before JSON body");
        bytes.extend_from_slice(&chunk[..count]);
    }
    Request {
        headers,
        body: serde_json::from_slice(&bytes[offset..offset + length]).unwrap(),
    }
}

struct Fixture {
    _temporary: tempfile::TempDir,
    root: PathBuf,
}

impl Fixture {
    fn new(student: &MockServer, teacher: &MockServer) -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().canonicalize().unwrap();
        let config = root.join("config/xtr");
        fs::create_dir_all(&config).unwrap();
        fs::write(
            config.join("schema.json"),
            json!({
                "type": "object", "properties": {"value": {"type": "integer"}},
                "required": ["value"], "additionalProperties": false
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            config.join("config.toml"),
            format!(
                r#"
[models.defaults.student]
name = "synthetic-student"
base_url = "{}"
api_key = "{SENTINEL}"
adapter = "json"
request_timeout_secs = 2
[models.defaults.teacher]
name = "synthetic-teacher"
base_url = "{}"
api_key = "{SENTINEL}"
adapter = "json"
request_timeout_secs = 2
[tasks.synthetic]
schema = "schema.json"
description = "Preserve all synthetic input."
include_timestamp = false
[data_collection]
enabled = false
[logging]
verbose_llm_logging = false
"#,
                student.endpoint, teacher.endpoint
            ),
        )
        .unwrap();
        Self {
            _temporary: temporary,
            root,
        }
    }

    fn run(&self, args: &[&str], stdin: &[u8]) -> Output {
        let mut child = Command::new(env!("CARGO_BIN_EXE_xtr"))
            .args(args)
            .env_clear()
            .env("HOME", self.root.join("home"))
            .env("XDG_CONFIG_HOME", self.root.join("config"))
            .env("XDG_DATA_HOME", self.root.join("data"))
            .env("XDG_STATE_HOME", self.root.join("state"))
            .env("XDG_CACHE_HOME", self.root.join("cache"))
            .current_dir(&self.root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.take().unwrap().write_all(stdin).unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(!String::from_utf8_lossy(&output.stdout).contains(SENTINEL));
        assert!(!String::from_utf8_lossy(&output.stderr).contains(SENTINEL));
        output
    }
}

fn stdout_json(output: &Output) -> Value {
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "stdout must be JSON: {error}; stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    })
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn user_content(request: &Request) -> &Vec<Value> {
    request.body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["role"] == "user")
        .unwrap()["content"]
        .as_array()
        .unwrap()
}

#[test]
fn binary_stdin_image_keeps_full_text_and_receipt_without_key_leak() {
    let raw = " \n{\"value\": 7}\n ";
    let student = MockServer::new(raw, Some("stop"));
    let teacher = MockServer::new("{\"value\":9}", Some("stop"));
    let fixture = Fixture::new(&student, &teacher);
    let text = "begin synthetic input\nAll lines stay intact: {braces}, <think>, Unicode ü.\nend";
    let output = fixture.run(
        &[
            "get",
            "synthetic",
            "--text",
            text,
            "--image",
            "stdin",
            "--receipt",
            "--validate",
            "error",
        ],
        PNG,
    );
    assert_success(&output);
    let receipt = stdout_json(&output);
    assert_eq!(receipt["raw_completion"], raw);
    assert_eq!(receipt["parsed"], json!({"value":7}));
    assert_eq!(receipt["requested_model"]["name"], "synthetic-student");
    assert_eq!(receipt["used_model"]["destination"], student.endpoint);
    assert_eq!(receipt["termination"], "stop");
    assert_eq!(receipt["validation"], "passed");
    assert_eq!(
        receipt["usage"],
        json!({"prompt_tokens":13,"completion_tokens":6,"total_tokens":19})
    );
    let calls = student.requests();
    assert_eq!(calls.len(), 1);
    assert!(
        calls[0]
            .headers
            .lines()
            .any(|line| line.eq_ignore_ascii_case(&format!("authorization: Bearer {SENTINEL}")))
    );
    assert!(!calls[0].body.to_string().contains(SENTINEL));
    let content = user_content(&calls[0]);
    assert!(content[0]["text"].as_str().unwrap().contains(text));
    assert_eq!(
        content[1],
        json!({"type":"image_url","image_url":{"url":PNG_URI}})
    );
    assert_eq!(content.len(), 2);
    assert!(teacher.requests().is_empty());
}

#[test]
fn text_stdin_and_repeated_file_and_base64_images_preserve_wire_order() {
    let student = MockServer::new("{\"value\":7}", Some("stop"));
    let teacher = MockServer::new("{\"value\":9}", Some("stop"));
    let fixture = Fixture::new(&student, &teacher);
    fs::write(fixture.root.join("first.wrong-extension"), PNG).unwrap();
    fs::write(fixture.root.join("second.png"), b"\xff\xd8\xff\x00").unwrap();
    let text = "  text from stdin\nFull second line\n  ";
    let output = fixture.run(
        &[
            "get",
            "synthetic",
            "--image",
            "first.wrong-extension",
            "--image",
            "second.png",
            "--image-base64",
            PNG_BASE64,
            "--image-base64",
            PNG_BASE64,
        ],
        text.as_bytes(),
    );
    assert_success(&output);
    assert_eq!(stdout_json(&output), json!({"value":7}));
    let calls = student.requests();
    assert_eq!(calls.len(), 1);
    let content = user_content(&calls[0]);
    assert!(
        content[0]["text"].as_str().unwrap().contains(text),
        "stdin must not be trimmed or replaced"
    );
    let images: Vec<_> = content[1..]
        .iter()
        .map(|part| part["image_url"]["url"].as_str().unwrap())
        .collect();
    assert_eq!(
        images,
        [PNG_URI, "data:image/jpeg;base64,/9j/AA==", PNG_URI, PNG_URI]
    );
    assert!(teacher.requests().is_empty());
}

#[test]
fn length_and_schema_invalid_exit_nonzero_with_typed_private_receipts() {
    for (raw, finish, failure) in [
        ("{\"value\":7}", "length", "truncated"),
        ("{\"value\":\"wrong-type\"}", "stop", "schema_invalid"),
    ] {
        let student = MockServer::new(raw, Some(finish));
        let teacher = MockServer::new("{\"value\":9}", Some("stop"));
        let fixture = Fixture::new(&student, &teacher);
        let output = fixture.run(
            &[
                "get",
                "synthetic",
                "--text",
                "value 7",
                "--receipt",
                "--validate",
                "error",
            ],
            b"",
        );
        assert!(!output.status.success());
        let error = stdout_json(&output);
        assert_eq!(error["kind"], "exhausted");
        assert_eq!(error["last_failure"], failure);
        assert_eq!(error["attempts"][0]["outcome"], json!({"failed":failure}));
        assert_eq!(error["attempts"][0]["raw_completion"], raw);
        assert_eq!(error["attempts"][0]["usage"]["total_tokens"], 19);
        assert_eq!(student.requests().len(), 1);
        assert!(
            teacher.requests().is_empty(),
            "configured teacher is not a caller grant"
        );
    }
}

#[test]
fn teacher_gets_zero_calls_until_explicit_fallback_grant() {
    let student = MockServer::new("{\"value\":7}", Some("length"));
    let teacher = MockServer::new("{\"value\":9}", Some("stop"));
    let fixture = Fixture::new(&student, &teacher);
    let denied = fixture.run(
        &["get", "synthetic", "--text", "synthetic", "--receipt"],
        b"",
    );
    assert!(!denied.status.success());
    assert!(teacher.requests().is_empty());
    let allowed = fixture.run(
        &[
            "get",
            "synthetic",
            "--text",
            "synthetic",
            "--receipt",
            "--allow-fallback",
            "teacher",
        ],
        b"",
    );
    assert_success(&allowed);
    let receipt = stdout_json(&allowed);
    assert_eq!(receipt["parsed"], json!({"value":9}));
    assert_eq!(receipt["requested_model"]["name"], "synthetic-student");
    assert_eq!(receipt["used_model"]["name"], "synthetic-teacher");
    assert_eq!(receipt["used_model"]["destination"], teacher.endpoint);
    assert_eq!(receipt["attempts"].as_array().unwrap().len(), 2);
    assert_eq!(
        receipt["attempts"][0]["outcome"],
        json!({"failed":"truncated"})
    );
    assert_eq!(receipt["attempts"][1]["outcome"], "accepted");
    assert_eq!(student.requests().len(), 2);
    assert_eq!(teacher.requests().len(), 1);
}

#[test]
fn unknown_termination_needs_explicit_cli_admission() {
    let student = MockServer::new("{\"value\":7}", None);
    let teacher = MockServer::new("{\"value\":9}", Some("stop"));
    let fixture = Fixture::new(&student, &teacher);
    let denied = fixture.run(
        &["get", "synthetic", "--text", "synthetic", "--receipt"],
        b"",
    );
    assert!(!denied.status.success());
    assert_eq!(stdout_json(&denied)["last_failure"], "unknown_termination");
    let allowed = fixture.run(
        &[
            "get",
            "synthetic",
            "--text",
            "synthetic",
            "--receipt",
            "--allow-unknown-termination",
        ],
        b"",
    );
    assert_success(&allowed);
    let receipt = stdout_json(&allowed);
    assert_eq!(receipt["termination"], "unknown");
    assert_eq!(receipt["completeness"], "unverified");
    assert!(teacher.requests().is_empty());
}

fn json_files(directory: &Path) -> Vec<PathBuf> {
    fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect()
}

#[test]
fn approval_uses_separate_validated_labels_and_private_nonclobbering_output() {
    let student = MockServer::new("{\"value\":7}", Some("stop"));
    let teacher = MockServer::new("{\"value\":9}", Some("stop"));
    let fixture = Fixture::new(&student, &teacher);
    let candidate = json!({
        "input_text":"synthetic source", "additional_context":"synthetic context",
        "receipt":{"parsed":{"value":7},"raw_completion":"{\"value\":7}"},
        "qualification":"unreviewed"
    })
    .to_string();
    fs::write(fixture.root.join("candidate.json"), &candidate).unwrap();
    fs::write(fixture.root.join("labels.json"), "{\"value\":\"invalid\"}").unwrap();
    let args = [
        "examples",
        "approve",
        "synthetic",
        "--candidate",
        "candidate.json",
        "--expected",
        "labels.json",
        "--reviewer",
        "synthetic-reviewer",
        "--output-dir",
        "approved",
    ];
    let invalid = fixture.run(&args, b"");
    assert!(
        !invalid.status.success(),
        "task schema must validate independently supplied labels"
    );
    assert!(!fixture.root.join("approved").exists());
    fs::write(fixture.root.join("labels.json"), "{\"value\":9}").unwrap();
    let first = fixture.run(&args, b"");
    assert_success(&first);
    let directory = fixture.root.join("approved");
    let files = json_files(&directory);
    assert_eq!(files.len(), 1);
    let original = fs::read(&files[0]).unwrap();
    let approved: Value = serde_json::from_slice(&original).unwrap();
    assert_eq!(
        approved["expected_json"],
        json!({"value":9}),
        "never promote candidate prediction 7"
    );
    assert_eq!(approved["input_text"], "synthetic source");
    assert_eq!(approved["additional_context"], "synthetic context");
    assert_eq!(approved["approval"]["reviewer"], "synthetic-reviewer");
    assert_eq!(approved["approval"]["policy"], "independent_labels");
    assert_success(&fixture.run(&args, b""));
    assert_eq!(json_files(&directory).len(), 2);
    assert_eq!(fs::read(&files[0]).unwrap(), original);
    assert_eq!(
        fs::read_to_string(fixture.root.join("candidate.json")).unwrap(),
        candidate
    );
    assert!(student.requests().is_empty());
    assert!(teacher.requests().is_empty());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            fs::metadata(&directory).unwrap().permissions().mode() & 0o777,
            0o700
        );
        for file in json_files(&directory) {
            assert_eq!(
                fs::metadata(file).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }
}

#[test]
fn mixed_base64_file_and_stdin_images_keep_cli_arrival_order() {
    let student = MockServer::new("{\"value\":7}", Some("stop"));
    let teacher = MockServer::new("{\"value\":9}", Some("stop"));
    let fixture = Fixture::new(&student, &teacher);
    fs::write(fixture.root.join("file.jpg"), b"\xff\xd8\xff\x00").unwrap();
    let output = fixture.run(
        &[
            "get",
            "synthetic",
            "--text",
            "first PNG, second JPEG, third GIF, fourth JPEG",
            "--image-base64",
            PNG_BASE64,
            "--image",
            "file.jpg",
            "--image=stdin",
            "--image-base64=/9j/AA==",
        ],
        b"GIF89a\x00\xff",
    );
    assert_success(&output);
    let calls = student.requests();
    assert_eq!(calls.len(), 1);
    let content = user_content(&calls[0]);
    let images: Vec<_> = content[1..]
        .iter()
        .map(|part| part["image_url"]["url"].as_str().unwrap())
        .collect();
    assert_eq!(
        images,
        [
            PNG_URI,
            "data:image/jpeg;base64,/9j/AA==",
            "data:image/gif;base64,R0lGODlhAP8=",
            "data:image/jpeg;base64,/9j/AA==",
        ],
        "ordinal image references must follow arrival order across both argument IDs"
    );
    assert!(teacher.requests().is_empty());
}
