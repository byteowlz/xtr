//! Real adapter dispatch against a bounded, synthetic loopback HTTP endpoint.
//! No DSPy globals, configuration files, optimization runs, or live providers.
#![allow(
    clippy::panic_in_result_fn,
    reason = "integration-test assertions report mismatches; Result propagates fixture errors"
)]

use std::collections::HashMap;
use std::io::Read;
use std::io::Write;
use std::io::{self};
use std::net::TcpListener;
use std::net::TcpStream;
use std::thread;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use dspy_rs::Example;
use dspy_rs::Optimizable;
use serde_json::Value;
use serde_json::json;
use xtr_core::config::ModelDescriptor;
use xtr_core::lm::build_model_handle;
use xtr_core::media::ImageInput;
use xtr_core::optimization::ExtractionProgram;

const DEADLINE: Duration = Duration::from_secs(5);
const MAX_REQUEST: usize = 64 * 1024;
const DEMO_SCHEMA: &str = r#"{"type":"object","properties":{"demo_value":{"type":"integer"}}}"#;
const FINAL_SCHEMA: &str = r#"{"type":"object","properties":{"value":{"type":"integer"}}}"#;
const DEMO_TEXT: &str = "DEMO text: quoted \"value\", 東京\nsecond line";
const FINAL_TEXT: &str = "FINAL text: quoted \"value\", Grüße\nsecond line";
const DEMO_CONTEXT: &str = "DEMO context: retain C:\\synthetic\\demo";
const FINAL_CONTEXT: &str = "FINAL context: retain C:\\synthetic\\final";
const DEMO_OUTPUT: &str = r#"{"demo_value":3}"#;
const FINAL_OUTPUT: &str = r#"{"value":7,"note":"quoted \"literal\" 東京"}"#;
const DEMO_REASONING: &str = "synthetic demo reasoning";
const FINAL_REASONING: &str = "synthetic final reasoning";

fn images(demo: bool) -> Result<Vec<String>> {
    let bytes = if demo {
        [
            b"\x89PNG\r\n\x1a\n\x00\xffdemo".as_slice(),
            b"GIF89a\x80demo".as_slice(),
        ]
    } else {
        [
            b"GIF87a\x00\xfffinal".as_slice(),
            b"\x89PNG\r\n\x1a\n\x80final".as_slice(),
        ]
    };
    bytes
        .into_iter()
        .map(|bytes| ImageInput::from_bytes(bytes).to_data_uri())
        .collect()
}

fn example(demo: bool, images: Vec<String>) -> Example {
    let (schema, text, context) = if demo {
        (DEMO_SCHEMA, DEMO_TEXT, DEMO_CONTEXT)
    } else {
        (FINAL_SCHEMA, FINAL_TEXT, FINAL_CONTEXT)
    };
    Example::new(
        HashMap::from([
            ("schema".into(), json!(schema)),
            ("input_text".into(), json!(text)),
            ("additional_context".into(), json!(context)),
            ("image_data_uris".into(), json!(images)),
            ("reasoning".into(), json!(DEMO_REASONING)),
            ("output_json".into(), json!(DEMO_OUTPUT)),
        ]),
        [
            "schema",
            "input_text",
            "additional_context",
            "image_data_uris",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
        vec!["reasoning".into(), "output_json".into()],
    )
}

fn descriptor(endpoint: String, adapter: &str) -> ModelDescriptor {
    ModelDescriptor {
        name: "synthetic-multimodal-model".into(),
        base_url: Some(endpoint),
        api_key: Some("synthetic-loopback-key".into()),
        adapter: Some(adapter.into()),
        request_timeout_secs: Some(5),
        max_tokens: Some(48),
    }
}

fn listener() -> Result<(TcpListener, String)> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    listener.set_nonblocking(true)?;
    let endpoint = format!("http://{}/v1", listener.local_addr()?);
    Ok((listener, endpoint))
}

/// Both total elapsed time and request size are bounded, including partial reads.
fn read_request(stream: &mut TcpStream) -> Result<(String, Value)> {
    // Accepted sockets can inherit nonblocking mode on macOS.
    stream.set_nonblocking(false)?;
    stream.set_read_timeout(Some(Duration::from_millis(100)))?;
    let deadline = Instant::now() + DEADLINE;
    let mut request = Vec::new();
    let mut body_range = None;
    loop {
        ensure!(Instant::now() < deadline, "HTTP request exceeded deadline");
        ensure!(
            request.len() <= MAX_REQUEST,
            "HTTP request exceeded size bound"
        );
        if let Some((start, length)) = body_range {
            if request.len() >= start + length {
                let headers = String::from_utf8(request[..start].to_vec())?;
                let body = serde_json::from_slice(&request[start..start + length])?;
                return Ok((headers, body));
            }
        } else if let Some(start) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n") {
            let headers = std::str::from_utf8(&request[..start])?;
            let length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then_some(value.trim())
                })
                .context("Expected Content-Length header")?
                .parse::<usize>()?;
            ensure!(
                length <= MAX_REQUEST - (start + 4),
                "HTTP body exceeded size bound"
            );
            body_range = Some((start + 4, length));
            continue;
        }
        let mut chunk = [0_u8; 4096];
        match stream.read(&mut chunk) {
            Ok(0) => return Err(anyhow!("HTTP request ended early")),
            Ok(count) => request.extend_from_slice(&chunk[..count]),
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) => {}
            Err(error) => return Err(error.into()),
        }
    }
}

fn mock_completion(
    adapter: &str,
    stop: Option<&str>,
) -> Result<(String, thread::JoinHandle<Result<Value>>)> {
    let (listener, endpoint) = listener()?;
    let content = if adapter == "json" {
        json!({"reasoning": FINAL_REASONING, "output_json": FINAL_OUTPUT}).to_string()
    } else {
        format!(
            "[[ ## reasoning ## ]]\n{FINAL_REASONING}\n\n[[ ## output_json ## ]]\n{FINAL_OUTPUT}\n\n[[ ## completed ## ]]\n"
        )
    };
    let mut choice = json!({"index": 0, "message": {"role": "assistant", "content": content}});
    if let Some(stop) = stop {
        choice["finish_reason"] = json!(stop);
    }
    let response = json!({
        "id": "synthetic-completion", "object": "chat.completion", "created": 0,
        "model": "synthetic-multimodal-model", "choices": [choice],
        "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}
    })
    .to_string();
    let server = thread::spawn(move || -> Result<Value> {
        let deadline = Instant::now() + DEADLINE;
        let mut stream = loop {
            ensure!(
                Instant::now() < deadline,
                "No loopback request before deadline"
            );
            match listener.accept() {
                Ok((stream, peer)) => {
                    ensure!(peer.ip().is_loopback(), "Non-loopback peer");
                    break stream;
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(5))
                }
                Err(error) => return Err(error.into()),
            }
        };
        let (headers, request) = read_request(&mut stream)?;
        ensure!(
            headers.starts_with("POST /v1/chat/completions HTTP/1.1\r\n"),
            "Wrong endpoint"
        );
        ensure!(
            headers
                .to_ascii_lowercase()
                .contains("authorization: bearer synthetic-loopback-key\r\n"),
            "Missing synthetic credential"
        );
        stream.set_write_timeout(Some(Duration::from_secs(1)))?;
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            response.len(),
            response
        )?;
        stream.flush()?;
        Ok(request)
    });
    Ok((endpoint, server))
}

fn assert_user_turn(turn: &Value, demo: bool, images: &[String], adapter: &str) -> Result<()> {
    let content = turn["content"]
        .as_array()
        .context("User turn must be multimodal")?;
    assert_eq!(content.len(), images.len() + 1);
    assert_eq!(content[0]["type"], "text");
    let text = content[0]["text"]
        .as_str()
        .context("First block must preserve text")?;
    let (schema, input, context, other_text) = if demo {
        (DEMO_SCHEMA, DEMO_TEXT, DEMO_CONTEXT, FINAL_TEXT)
    } else {
        (FINAL_SCHEMA, FINAL_TEXT, FINAL_CONTEXT, DEMO_TEXT)
    };
    for field in [schema, input, context] {
        assert!(
            text.contains(field),
            "{adapter}: missing field {field:?} in {text:?}"
        );
    }
    assert!(!text.contains(other_text), "Demo/current inputs were mixed");
    if adapter == "json" {
        assert!(text.contains("Schema:\n```json\n"));
        assert!(text.contains("input_text:\n"));
        assert!(text.contains("additional_context:\n"));
        assert!(text.contains("Return only the JSON object"));
    } else {
        for field in ["schema", "input_text", "additional_context"] {
            assert!(text.contains(&format!("[[ ## {field} ## ]]\n")));
        }
        assert!(text.contains("ending with the marker for `completed`"));
    }
    for (block, uri) in content.iter().skip(1).zip(images) {
        assert_eq!(
            block,
            &json!({"type": "image_url", "image_url": {"url": uri}})
        );
        assert!(
            !text.contains(uri),
            "Image must not replace or pollute text"
        );
    }
    Ok(())
}

async fn normal_stop(adapter: &str) -> Result<()> {
    let (endpoint, server) = mock_completion(adapter, Some("stop"))?;
    let model = build_model_handle(&descriptor(endpoint, adapter)).await?;
    let demo_images = images(true)?;
    let final_images = images(false)?;
    let mut program = ExtractionProgram::new();
    program
        .solver_mut()
        .signature
        .set_demos(vec![example(true, demo_images.clone())])?;
    let signature = Optimizable::get_signature(program.solver());
    let result = tokio::time::timeout(
        DEADLINE,
        model.predict(signature, example(false, final_images.clone())),
    )
    .await;
    let wire = server
        .join()
        .map_err(|_| anyhow!("Loopback thread panicked"))??;
    let prediction = result??;
    assert_eq!(wire["model"], "synthetic-multimodal-model");
    assert_eq!(wire["max_tokens"], 48);
    let messages = wire["messages"]
        .as_array()
        .context("Messages must be an array")?;
    assert_eq!(messages.len(), 4);
    assert_eq!(
        messages
            .iter()
            .map(|turn| turn["role"].clone())
            .collect::<Vec<_>>(),
        vec![
            json!("system"),
            json!("user"),
            json!("assistant"),
            json!("user")
        ]
    );
    let system = messages[0]["content"]
        .as_str()
        .context("System text missing")?;
    assert!(system.contains(&signature.instruction()));
    assert!(system.contains("output_json"));
    assert_user_turn(&messages[1], true, &demo_images, adapter)?;
    assert_user_turn(&messages[3], false, &final_images, adapter)?;
    let demo_response = messages[2]["content"]
        .as_str()
        .context("Demo assistant text missing")?;
    if adapter == "json" {
        let demo: Value = serde_json::from_str(demo_response)?;
        assert_eq!(demo["reasoning"], DEMO_REASONING);
        assert_eq!(demo["output_json"], DEMO_OUTPUT);
    } else {
        assert!(demo_response.contains(&format!("[[ ## reasoning ## ]]\n{DEMO_REASONING}")));
        assert!(demo_response.contains(&format!("[[ ## output_json ## ]]\n{DEMO_OUTPUT}")));
        assert!(demo_response.ends_with("[[ ## completed ## ]]\n"));
    }
    let output = prediction.get("output_json", None);
    assert_eq!(
        serde_json::from_str::<Value>(output.as_str().context("Missing parsed output")?)?,
        serde_json::from_str::<Value>(FINAL_OUTPUT)?
    );
    assert_eq!(prediction.get("reasoning", None), FINAL_REASONING);
    assert_eq!(prediction.lm_usage.prompt_tokens, 11);
    assert_eq!(prediction.lm_usage.completion_tokens, 5);
    assert_eq!(prediction.lm_usage.total_tokens, 16);
    Ok(())
}

#[tokio::test]
async fn json_multimodal_predict_preserves_demos_and_current_turn() -> Result<()> {
    normal_stop("json").await
}

#[tokio::test]
async fn legacy_multimodal_predict_preserves_demos_and_current_turn() -> Result<()> {
    normal_stop("chat-legacy").await
}

#[tokio::test]
async fn missing_or_length_stop_rejects_complete_output_in_both_modes() -> Result<()> {
    for adapter in ["json", "chat-legacy"] {
        for stop in [None, Some("length")] {
            let (endpoint, server) = mock_completion(adapter, stop)?;
            let model = build_model_handle(&descriptor(endpoint, adapter)).await?;
            let program = ExtractionProgram::new();
            let signature = Optimizable::get_signature(program.solver());
            let result = tokio::time::timeout(
                DEADLINE,
                model.predict(signature, example(false, images(false)?)),
            )
            .await;
            let wire = server
                .join()
                .map_err(|_| anyhow!("Loopback thread panicked"))??;
            let error = result?.err().context("Non-stop completion was accepted")?;
            assert!(
                error.to_string().contains("did not report a normal stop"),
                "{adapter}, {stop:?}: {error:#}"
            );
            assert_eq!(
                wire["messages"][1]["content"]
                    .as_array()
                    .context("Missing multimodal content")?
                    .len(),
                3
            );
        }
    }
    Ok(())
}

#[tokio::test]
async fn invalid_demo_or_current_image_sends_zero_http_calls() -> Result<()> {
    let valid = images(false)?;
    let mismatched = valid[0].replacen("image/gif", "image/png", 1);
    for adapter in ["json", "chat-legacy"] {
        for bad in [
            json!([valid[0], "data:image/png;base64,not-base64!"]),
            json!([mismatched]),
            json!(["data:image/png;base64,dGV4dA=="]),
            json!([42]),
            json!("not an array"),
        ] {
            for invalid_demo in [false, true] {
                let (listener, endpoint) = listener()?;
                let model = build_model_handle(&descriptor(endpoint, adapter)).await?;
                let mut demo = example(true, images(true)?);
                let mut current = example(false, valid.clone());
                if invalid_demo {
                    demo.data.insert("image_data_uris".into(), bad.clone());
                } else {
                    current.data.insert("image_data_uris".into(), bad.clone());
                }
                let mut program = ExtractionProgram::new();
                program.solver_mut().signature.set_demos(vec![demo])?;
                let signature = Optimizable::get_signature(program.solver());
                let result =
                    tokio::time::timeout(DEADLINE, model.predict(signature, current)).await?;
                assert!(
                    result.is_err(),
                    "{adapter}, invalid_demo={invalid_demo}: {bad}"
                );
                assert!(
                    matches!(listener.accept(), Err(error) if error.kind() == io::ErrorKind::WouldBlock),
                    "Invalid image opened an HTTP connection"
                );
            }
        }
    }
    Ok(())
}
