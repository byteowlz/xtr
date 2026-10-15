use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use dspy_rs::Cache;
use dspy_rs::CallResult;
use dspy_rs::Chat;
use dspy_rs::ChatAdapter;
use dspy_rs::Example;
use dspy_rs::LM;
use dspy_rs::Message;
use dspy_rs::MetaSignature;
use dspy_rs::Prediction;
use dspy_rs::adapter::Adapter;
use dspy_rs::serde_utils::get_iter_from_value;
use rig::tool::ToolDyn;
use serde_json::Value;

use crate::config::ModelDescriptor;
use crate::config::ResolvedModelConfig;

#[derive(Default, Clone)]
struct JsonAdapter;

impl JsonAdapter {
    fn format_system_message(signature: &dyn MetaSignature) -> String {
        let base_instruction = signature.instruction();
        let mut message = if base_instruction.is_empty() {
            "You convert unstructured text into structured JSON that strictly matches the caller-provided schema. Only include fields supported by the schema and leave everything else out.".to_string()
        } else {
            base_instruction.clone()
        };

        message.push_str(
            "\n\nRespond with a single JSON object containing:\n  - \"reasoning\": brief natural-language justification (string)\n  - \"output_json\": the extracted data as a JSON string that validates against the provided schema.\nDo not add any surrounding commentary, markdown fences, or additional keys. The JSON must be the entire response.",
        );
        message
    }

    fn value_to_string(value: Value) -> String {
        match value {
            Value::String(s) => s,
            Value::Null => String::new(),
            other => other.to_string(),
        }
    }

    fn format_user_message(signature: &dyn MetaSignature, inputs: &Example) -> String {
        let mut sections = Vec::new();

        for (field_name, _) in get_iter_from_value(&signature.input_fields()) {
            let value = inputs.get(field_name.as_str(), None);
            let as_string = Self::value_to_string(value);

            if field_name.eq_ignore_ascii_case("schema") {
                sections.push(format!("Schema:\n```json\n{}\n```", as_string.trim()));
            } else {
                sections.push(format!("{field_name}:\n{as_string}"));
            }
        }

        sections.push(
            "Return only the JSON object described in the system message. It must be valid JSON without markdown fences."
                .to_string(),
        );

        sections.join("\n\n")
    }

    fn format_demo_messages(signature: &dyn MetaSignature, demo: &Example) -> (String, String) {
        let user = Self::format_user_message(signature, demo);

        let mut response_object = serde_json::Map::new();
        for (field_name, _) in get_iter_from_value(&signature.output_fields()) {
            let value = demo.get(field_name.as_str(), None);
            let as_string = Self::value_to_string(value);

            response_object.insert(field_name.clone(), Value::String(as_string));
        }

        (user, Value::Object(response_object).to_string())
    }

    /// Accept one complete JSON value (any root type), without rewriting literals.
    /// Optional envelopes are exactly one leading, non-nested `<think>...</think>`
    /// block, then optionally one lowercase `json` code fence. Fence delimiters
    /// must occupy their own lines (LF or CRLF); outer whitespace is allowed.
    /// Prose, brace fragments, nested/repeated envelopes, multiple values and
    /// truncated payloads are rejected, never repaired or partially recovered.
    fn extract_json_payload(raw: &str) -> Option<Value> {
        if let Ok(value) = serde_json::from_str::<Value>(raw) {
            return Some(value);
        }

        let mut payload = raw.trim();
        if let Some(reasoning) = payload.strip_prefix("<think>") {
            let (body, remainder) = reasoning.split_once("</think>")?;
            if body.contains("<think>") {
                return None;
            }
            payload = remainder.trim();
        }

        if let Some(fenced) = Self::strip_code_fence(payload) {
            payload = fenced;
        }
        serde_json::from_str::<Value>(payload).ok()
    }

    fn strip_code_fence(text: &str) -> Option<&str> {
        text.strip_prefix("```json\n")
            .or_else(|| text.strip_prefix("```json\r\n"))?
            .strip_suffix("\n```")
    }

    fn parse_response_payload(content: &str) -> HashMap<String, Value> {
        let mut data = HashMap::new();
        if let Some(payload) = Self::extract_json_payload(content) {
            if let Some(map) = payload.as_object() {
                if let Some(reasoning) = map.get("reasoning") {
                    data.insert(
                        "reasoning".to_string(),
                        if reasoning.is_string() {
                            reasoning.clone()
                        } else {
                            Value::String(reasoning.to_string())
                        },
                    );
                }

                if let Some(output_json) = map.get("output_json") {
                    let normalized = if output_json.is_string() {
                        output_json.clone()
                    } else {
                        Value::String(output_json.to_string())
                    };
                    data.insert("output_json".to_string(), normalized);
                    return data;
                }
            }

            // If payload is already the structured object, treat it as the extraction.
            data.insert(
                "output_json".to_string(),
                Value::String(payload.to_string()),
            );
        } else {
            // Preserve rejected responses verbatim for downstream validation/evidence.
            data.insert(
                "output_json".to_string(),
                Value::String(content.to_string()),
            );
        }

        data
    }
}

#[cfg(test)]
#[expect(
    clippy::panic_in_result_fn,
    reason = "test assertions report mismatches; Result propagates fixture setup errors"
)]
mod parser_tests {
    use super::JsonAdapter;
    use anyhow::Result;
    use anyhow::anyhow;
    use serde_json::Value;
    use serde_json::json;

    #[test]
    fn literal_safe_preserves_thinking_response_note() -> Result<()> {
        let raw = r#"{"output_json":{"note":"needs thinking before response selection"}}"#;
        let expected = json!({"note": "needs thinking before response selection"});
        let payload = JsonAdapter::extract_json_payload(raw)
            .ok_or_else(|| anyhow!("expected complete JSON payload"))?;
        assert_eq!(payload["output_json"], expected);

        let parsed = JsonAdapter::parse_response_payload(raw);
        let output = parsed["output_json"]
            .as_str()
            .ok_or_else(|| anyhow!("expected output_json string"))?;
        assert_eq!(serde_json::from_str::<Value>(output)?, expected);
        Ok(())
    }

    #[test]
    fn literal_safe_complete_json_table() -> Result<()> {
        let cases = [
            (
                "marker words",
                json!({"note": "needs thinking before response selection"}),
            ),
            (
                "literal tags",
                json!({"note": "<think>thinking</think> response ```json ```"}),
            ),
            (
                "unicode",
                json!({"note": "Grüße 東京 Ω thinking before response"}),
            ),
            (
                "escapes",
                json!({"note": "quoted \"thinking before response\" and C:\\tmp\\file"}),
            ),
            ("braces", json!({"note": "} { [ ] \\"})),
            (
                "nested",
                json!({"items": [{"note": "} thinking before response {", "list": [null, true, 42]}]}),
            ),
            (
                "array root",
                json!([{"note": "thinking before response"}, [1, 2]]),
            ),
            (
                "string root",
                json!("<think>thinking before response</think> { }"),
            ),
            ("number root", json!(12.5)),
            ("boolean root", json!(true)),
            ("null root", Value::Null),
        ];
        for (name, expected) in cases {
            let raw = serde_json::to_string(&expected)?;
            assert_eq!(
                JsonAdapter::extract_json_payload(&raw),
                Some(expected.clone()),
                "{name}"
            );

            let parsed = JsonAdapter::parse_response_payload(&raw);
            let output = parsed["output_json"]
                .as_str()
                .ok_or_else(|| anyhow!("expected output_json string for {name}"))?;
            assert_eq!(serde_json::from_str::<Value>(output)?, expected, "{name}");
        }
        Ok(())
    }

    #[test]
    fn literal_safe_envelope_and_wrapper_table() -> Result<()> {
        let expected = json!({
            "note": "needs thinking before response selection <think>literal</think>",
            "items": [{"quoted": "\" } { \\"}, [true, null, "東京"]]
        });
        let output = serde_json::to_string(&expected)?;
        let payloads = [
            ("bare", expected.clone()),
            (
                "object output_json",
                json!({"reasoning": "thinking before response", "output_json": expected}),
            ),
            (
                "string output_json",
                json!({"reasoning": "<think>literal</think>", "output_json": output}),
            ),
        ];
        for (name, expected_payload) in payloads {
            let raw = serde_json::to_string(&expected_payload)?;
            let envelopes = [
                raw.clone(),
                format!(" \n{raw}\t "),
                format!("<think>thinking before response\nreasoning</think>\n{raw}"),
                format!("```json\n{raw}\n```"),
                format!(" \r\n```json\r\n{raw}\r\n```\r\n "),
                format!("<think>reasoning</think>\n```json\n{raw}\n```"),
            ];
            for response in envelopes {
                assert_eq!(
                    JsonAdapter::extract_json_payload(&response),
                    Some(expected_payload.clone()),
                    "{name}: {response}"
                );
                let parsed = JsonAdapter::parse_response_payload(&response);
                let parsed_output = parsed["output_json"]
                    .as_str()
                    .ok_or_else(|| anyhow!("expected output_json string for {name}"))?;
                assert_eq!(
                    serde_json::from_str::<Value>(parsed_output)?,
                    expected,
                    "{name}"
                );
                if let Some(reasoning) = expected_payload.get("reasoning") {
                    assert_eq!(parsed.get("reasoning"), Some(reasoning), "{name}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn literal_safe_rejects_recovery_table() {
        let cases = [
            ("empty", " \n\t "),
            (
                "prose prefix",
                "Here is JSON: {\"note\":\"thinking before response\"}",
            ),
            ("prose suffix", "{\"note\":\"unchanged\"} done"),
            (
                "quoted prose fragment",
                r#"Answer: {"note":"} thinking before response {"}"#,
            ),
            ("two objects", "{\"first\":1} {\"second\":2}"),
            ("two roots", "[] null"),
            ("truncated object", "{\"nested\":{\"valid\":true}"),
            ("truncated array", "[{\"valid\":true}"),
            ("truncated string", "{\"note\":\"unfinished}"),
            ("invalid escape", r#"{"note":"bad\q"}"#),
            ("unclosed think", "<think>reasoning {\"valid\":true}"),
            ("closing think only", "</think>{}"),
            ("nested think", "<think><think>nested</think></think>{}"),
            ("repeated think", "<think>one</think><think>two</think>{}"),
            ("uppercase think", "<THINK>reasoning</THINK>{}"),
            ("think after JSON", "{}<think>reasoning</think>"),
            ("prose after think", "<think>reasoning</think>answer: {}"),
            ("multiple after think", "<think>reasoning</think>{} {}"),
            ("truncated after think", "<think>reasoning</think>[{}"),
            ("unclosed fence", "```json\n{}"),
            ("unlabeled fence", "```\n{}\n```"),
            ("wrong fence language", "```python\n{}\n```"),
            ("uppercase fence language", "```JSON\n{}\n```"),
            ("inline opening fence", "```json {}\n```"),
            ("inline closing fence", "```json\n{}```"),
            ("fence suffix prose", "```json\n{}\n``` done"),
            ("two fences", "```json\n{}\n```\n```json\n{}\n```"),
            ("nested fences", "```json\n```json\n{}\n```\n```"),
            ("multiple fenced values", "```json\n{} {}\n```"),
            ("truncated fenced value", "```json\n[{\"valid\":true}\n```"),
            (
                "think inside fence",
                "```json\n<think>reasoning</think>{}\n```",
            ),
        ];
        for (name, raw) in cases {
            assert_eq!(JsonAdapter::extract_json_payload(raw), None, "{name}");
            let parsed = JsonAdapter::parse_response_payload(raw);
            assert_eq!(
                parsed.get("output_json"),
                Some(&Value::String(raw.to_string())),
                "raw evidence: {name}"
            );
            assert_eq!(parsed.len(), 1, "{name}");
        }
    }

    #[test]
    fn literal_safe_parsed_output_passes_exact_schema_validation() -> Result<()> {
        let note = "needs thinking before response selection";
        let schema = json!({
            "type": "object",
            "properties": {"note": {"type": "string", "const": note}},
            "required": ["note"],
            "additionalProperties": false
        });
        let validator = jsonschema::JSONSchema::compile(&schema)
            .map_err(|err| anyhow!("failed to compile synthetic schema: {err}"))?;
        let response = r#"{"output_json":{"note":"needs thinking before response selection"}}"#;
        let parsed = JsonAdapter::parse_response_payload(response);
        let output = parsed["output_json"]
            .as_str()
            .ok_or_else(|| anyhow!("expected output_json string"))?;
        let extracted = serde_json::from_str::<Value>(output)?;
        assert!(validator.is_valid(&extracted));
        assert_eq!(extracted["note"], note);
        Ok(())
    }
}

#[async_trait::async_trait]
impl Adapter for JsonAdapter {
    fn format(&self, signature: &dyn MetaSignature, inputs: Example) -> Chat {
        let mut chat = Chat::new(vec![]);
        chat.push("system", &Self::format_system_message(signature));

        let demos = signature.demos();
        for demo in demos {
            let (user, assistant) = Self::format_demo_messages(signature, &demo);
            chat.push("user", &user);
            chat.push("assistant", &assistant);
        }

        let user_message = Self::format_user_message(signature, &inputs);
        chat.push("user", &user_message);

        chat
    }

    fn parse_response(
        &self,
        _signature: &dyn MetaSignature,
        response: Message,
    ) -> HashMap<String, Value> {
        let content = response.content();
        Self::parse_response_payload(&content)
    }

    async fn call(
        &self,
        lm: Arc<LM>,
        signature: &dyn MetaSignature,
        inputs: Example,
        _tools: Vec<Arc<dyn ToolDyn>>,
    ) -> Result<Prediction> {
        if lm.cache
            && let Some(cache) = lm.cache_handler.as_ref()
        {
            let cache_key = inputs.clone();
            if let Some(cached) = cache.lock().await.get(cache_key).await? {
                return Ok(cached);
            }
        }

        let messages = self.format(signature, inputs.clone());
        let response = lm.call(messages.clone(), vec![]).await?;
        let prompt_str = response.chat.to_json().to_string();

        let data = self.parse_response(signature, response.output.clone());
        let prediction = Prediction {
            data,
            lm_usage: response.usage,
        };

        if lm.cache
            && let Some(cache) = lm.cache_handler.as_ref()
        {
            let (tx, rx) = tokio::sync::mpsc::channel(1);
            let cache_clone = cache.clone();
            let inputs_clone = inputs.clone();
            let prediction_clone = prediction.clone();

            tokio::spawn(async move {
                let _ = cache_clone.lock().await.insert(inputs_clone, rx).await;
            });

            tx.send(CallResult {
                prompt: prompt_str,
                prediction: prediction_clone,
            })
            .await
            .map_err(|_| anyhow!("Failed to send to cache"))?;
        }

        Ok(prediction)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterKind {
    Chat,
    Json,
}

impl AdapterKind {
    pub(crate) fn from_descriptor(descriptor: &ModelDescriptor) -> Result<Self> {
        match descriptor
            .adapter
            .as_deref()
            .unwrap_or("json")
            .to_ascii_lowercase()
            .as_str()
        {
            "chat-legacy" | "legacy_chat" | "legacy-chat" => Ok(Self::Chat),
            "chat" | "json" | "structured_json" | "structured-json" => Ok(Self::Json),
            other => Err(anyhow!(
                "unsupported adapter '{other}' for model '{}'",
                descriptor.name
            )),
        }
    }

    pub(crate) fn format_messages(self, signature: &dyn MetaSignature, inputs: Example) -> Chat {
        match self {
            Self::Json => JsonAdapter.format(signature, inputs),
            Self::Chat => ChatAdapter.format(signature, inputs),
        }
    }

    /// Wire formatting preserves every demo/current user turn and its images.
    pub(crate) fn format_wire_messages(
        self,
        signature: &dyn MetaSignature,
        inputs: Example,
    ) -> Result<Value> {
        let mut examples = signature.demos();
        examples.push(inputs.clone());
        let mut messages = self.format_messages(signature, inputs).to_json();
        let turns = messages
            .as_array_mut()
            .ok_or_else(|| anyhow!("invalid messages"))?;
        let mut users = turns.iter_mut().filter(|turn| turn["role"] == "user");
        for example in examples {
            let user = users.next().ok_or_else(|| anyhow!("missing user turn"))?;
            let images = example.get("image_data_uris", None);
            if images.is_null() {
                continue;
            }
            let images = images
                .as_array()
                .ok_or_else(|| anyhow!("image_data_uris must be an array"))?;
            if images.is_empty() {
                continue;
            }
            let text = user["content"]
                .as_str()
                .ok_or_else(|| anyhow!("invalid user content"))?;
            let mut image_inputs = Vec::with_capacity(images.len());
            for uri in images {
                let uri = uri
                    .as_str()
                    .ok_or_else(|| anyhow!("image URI must be a string"))?;
                let (_, encoded) = uri
                    .split_once(',')
                    .ok_or_else(|| anyhow!("invalid image URI"))?;
                let image = crate::media::ImageInput::from_base64(encoded);
                if image.to_data_uri()? != uri {
                    return Err(anyhow!("image URI MIME/content mismatch"));
                }
                image_inputs.push(image);
            }
            user["content"] = crate::media::multimodal_content(text, &image_inputs)?;
        }
        Ok(messages)
    }

    pub(crate) fn decoded_envelope(self, content: &str) -> Option<Value> {
        match self {
            Self::Json => JsonAdapter::extract_json_payload(content),
            Self::Chat => None,
        }
    }

    pub(crate) fn decode_extraction(self, content: &str) -> Result<Value> {
        match self {
            Self::Json => {
                let payload = JsonAdapter::extract_json_payload(content)
                    .ok_or_else(|| anyhow!("response is not one complete JSON payload"))?;
                if let Some(output) = payload.as_object().and_then(|map| map.get("output_json")) {
                    if let Some(encoded) = output.as_str() {
                        serde_json::from_str(encoded).context("output_json is not complete JSON")
                    } else {
                        Ok(output.clone())
                    }
                } else {
                    Ok(payload)
                }
            }
            Self::Chat => {
                let program = crate::optimization::ExtractionProgram::new();
                let signature = dspy_rs::core::Optimizable::get_signature(program.solver());
                let data = ChatAdapter.parse_response(signature, Message::assistant(content));
                let output = data
                    .get("output_json")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("legacy response is missing output_json"))?;
                JsonAdapter::extract_json_payload(output)
                    .ok_or_else(|| anyhow!("legacy output_json is not complete JSON"))
            }
        }
    }

    /// Only the non-concurrent DSPy optimization workflow installs globals.
    pub fn install(self, lm: LM) {
        dspy_rs::configure(lm, RequestAdapter { kind: self });
    }
}

/// DSPy's Chat is text-only, so multimodal calls use the same compatible wire
/// transport directly. Text-only optimization calls retain their existing cache.
#[derive(Clone)]
struct RequestAdapter {
    kind: AdapterKind,
}

#[async_trait::async_trait]
impl Adapter for RequestAdapter {
    fn format(&self, signature: &dyn MetaSignature, inputs: Example) -> Chat {
        self.kind.format_messages(signature, inputs)
    }

    fn parse_response(
        &self,
        signature: &dyn MetaSignature,
        response: Message,
    ) -> HashMap<String, Value> {
        match self.kind {
            AdapterKind::Json => JsonAdapter.parse_response(signature, response),
            AdapterKind::Chat => ChatAdapter.parse_response(signature, response),
        }
    }

    async fn call(
        &self,
        lm: Arc<LM>,
        signature: &dyn MetaSignature,
        inputs: Example,
        tools: Vec<Arc<dyn ToolDyn>>,
    ) -> Result<Prediction> {
        let mut examples = signature.demos();
        examples.push(inputs.clone());
        let has_images = examples.iter().any(|example| {
            let images = example.get("image_data_uris", None);
            !images.is_null() && images != serde_json::json!([])
        });
        if !has_images {
            return match self.kind {
                AdapterKind::Json => JsonAdapter.call(lm, signature, inputs, tools).await,
                AdapterKind::Chat => ChatAdapter.call(lm, signature, inputs, tools).await,
            };
        }
        if !tools.is_empty() {
            return Err(anyhow!("multimodal extraction does not execute tools"));
        }
        use crate::inference::InferenceBackend;
        let descriptor = ModelDescriptor {
            name: lm.model.clone(),
            base_url: lm.base_url.clone(),
            api_key: lm.api_key.clone(),
            adapter: None,
            request_timeout_secs: None,
            max_tokens: Some(lm.max_tokens),
        };
        let messages = self.kind.format_wire_messages(signature, inputs)?;
        let completion = crate::inference::OpenAiBackend::new()?
            .complete(&descriptor, &messages)
            .await
            .map_err(|failure| anyhow!("multimodal completion failed: {failure:?}"))?;
        if completion.termination != crate::inference::Termination::Stop {
            return Err(anyhow!(
                "multimodal completion did not report a normal stop"
            ));
        }
        let raw = completion
            .raw
            .ok_or_else(|| anyhow!("completion is missing text"))?;
        self.kind.decode_extraction(&raw)?;
        let usage = completion.usage.unwrap_or_default();
        Ok(Prediction {
            data: self.parse_response(signature, Message::assistant(raw)),
            // DSPy's optimizer requires numerical usage, unlike extraction receipts.
            lm_usage: dspy_rs::LmUsage {
                prompt_tokens: usage.prompt_tokens.unwrap_or(0),
                completion_tokens: usage.completion_tokens.unwrap_or(0),
                total_tokens: usage.total_tokens.unwrap_or(0),
            },
        })
    }
}

#[derive(Clone)]
pub struct ModelHandle {
    pub descriptor: ModelDescriptor,
    pub adapter: AdapterKind,
    pub lm: LM,
}

impl ModelHandle {
    /// Direct adapter dispatch avoids DSPy's configure/forward global transaction.
    pub async fn predict(
        &self,
        signature: &dyn MetaSignature,
        inputs: Example,
    ) -> Result<Prediction> {
        RequestAdapter { kind: self.adapter }
            .call(Arc::new(self.lm.clone()), signature, inputs, vec![])
            .await
    }

    pub fn configure_global(&self) {
        self.adapter.install(self.lm.clone());
    }
}

pub async fn build_model_handle(descriptor: &ModelDescriptor) -> Result<ModelHandle> {
    let adapter = AdapterKind::from_descriptor(descriptor)?;
    let api_key = descriptor
        .api_key
        .clone()
        .unwrap_or_else(|| "not_needed".to_string());

    let base_url = descriptor
        .base_url
        .clone()
        .unwrap_or_else(|| "https://api.openai.com/v1".to_string());

    let lm = if let Some(max_tokens) = descriptor.max_tokens {
        LM::builder()
            .api_key(api_key)
            .base_url(base_url)
            .model(descriptor.name.clone())
            .max_tokens(max_tokens)
            .build()
            .await?
    } else {
        LM::builder()
            .api_key(api_key)
            .base_url(base_url)
            .model(descriptor.name.clone())
            .build()
            .await?
    };

    Ok(ModelHandle {
        descriptor: descriptor.clone(),
        adapter,
        lm,
    })
}

#[derive(Clone)]
pub struct TaskModelHandles {
    pub teacher: ModelHandle,
    pub student: ModelHandle,
    pub fallbacks: Vec<ModelHandle>,
}

impl TaskModelHandles {
    pub async fn load(config: &ResolvedModelConfig) -> Result<Self> {
        let teacher = build_model_handle(&config.teacher)
            .await
            .context("failed to build teacher LM")?;
        let student = build_model_handle(&config.student)
            .await
            .context("failed to build student LM")?;

        let mut fallbacks = Vec::with_capacity(config.fallbacks.len());
        for descriptor in &config.fallbacks {
            fallbacks.push(
                build_model_handle(descriptor).await.with_context(|| {
                    format!("failed to build fallback LM '{}'", descriptor.name)
                })?,
            );
        }

        Ok(Self {
            teacher,
            student,
            fallbacks,
        })
    }
}
