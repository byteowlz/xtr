//! Request-owned extraction attempts, admission policy and audit receipts.
use std::path::PathBuf;

use async_trait::async_trait;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use serde_json::json;

use crate::config::ModelDescriptor;
use crate::engine::RetryMode;
use crate::engine::ValidationMode;
use crate::lm::AdapterKind;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelRole {
    Student,
    Teacher,
    Fallback(usize),
}

/// A grant binds both model and destination. Credentials are never part of identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelIdentity {
    pub role: ModelRole,
    pub name: String,
    pub destination: String,
}

impl ModelIdentity {
    pub fn from_descriptor(
        role: ModelRole,
        descriptor: &ModelDescriptor,
    ) -> Result<Self, ExtractionError> {
        let destination = descriptor
            .base_url
            .clone()
            .unwrap_or_else(|| "https://api.openai.com/v1".into());
        validate_destination(&destination).map_err(|_| ExtractionError::configuration())?;
        if descriptor.api_key.as_deref().is_some_and(|key| {
            !key.is_empty() && (destination.contains(key) || descriptor.name.contains(key))
        }) {
            return Err(ExtractionError::configuration());
        }
        Ok(Self {
            role,
            name: descriptor.name.clone(),
            destination,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Termination {
    Stop,
    Length,
    Refusal,
    ToolCalls,
    Unknown,
    /// Bounded capture classification; untrusted provider reason text is omitted.
    Unsupported,
    Other(String),
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderUsage {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureClass {
    Configuration,
    Transport,
    Malformed,
    SchemaInvalid,
    Truncated,
    Refusal,
    UnknownTermination,
    UnsupportedTermination,
    SensitiveResponse,
    Denied,
    Exhausted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationStatus {
    NotChecked,
    Passed,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptOutcome {
    Accepted,
    Failed(FailureClass),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttemptReceipt {
    pub ordinal: usize,
    pub model: ModelIdentity,
    pub retry: u32,
    pub outcome: AttemptOutcome,
    pub raw_completion: Option<String>,
    pub parsed: Option<Value>,
    pub termination: Option<Termination>,
    pub usage: Option<ProviderUsage>,
    pub validation: ValidationStatus,
    pub validation_errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Completeness {
    /// The provider reports stop; this does not establish semantic correctness.
    ReportedStop,
    /// Accepted only by explicit caller policy; completeness cannot be established.
    Unverified,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionReceipt {
    pub parsed: Value,
    pub raw_completion: String,
    pub requested_model: ModelIdentity,
    pub used_model: ModelIdentity,
    pub attempts: Vec<AttemptReceipt>,
    pub validation: ValidationStatus,
    pub termination: Termination,
    pub usage: Option<ProviderUsage>,
    pub completeness: Completeness,
    /// Persistence failures do not turn a successful extraction into a failed call.
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionError {
    pub kind: FailureClass,
    pub last_failure: Option<FailureClass>,
    pub attempts: Vec<AttemptReceipt>,
}

impl std::fmt::Display for ExtractionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "extraction {:?} ({:?}; {} attempts)",
            self.kind,
            self.last_failure,
            self.attempts.len()
        )
    }
}
impl std::error::Error for ExtractionError {}

impl ExtractionError {
    pub(crate) fn configuration() -> Self {
        Self {
            kind: FailureClass::Configuration,
            last_failure: None,
            attempts: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub enum LoggingPolicy {
    #[default]
    Config,
    Disabled,
    Directory(PathBuf),
}

#[derive(Debug, Clone)]
pub struct InferenceOptions {
    pub validation: ValidationMode,
    pub max_retries: u32,
    pub retry_mode: RetryMode,
    pub max_tokens: Option<u32>,
    /// Exact, resolved destinations approved by the caller for this invocation.
    pub allowed_fallbacks: Vec<ModelIdentity>,
    pub retry_on: Vec<FailureClass>,
    pub fallback_on: Vec<FailureClass>,
    pub allow_unknown_termination: bool,
    pub logging: LoggingPolicy,
    /// Validated image data URIs, attached to the full initial user turn.
    pub images: Vec<String>,
}

impl Default for InferenceOptions {
    fn default() -> Self {
        let recoverable = vec![
            FailureClass::Transport,
            FailureClass::Malformed,
            FailureClass::SchemaInvalid,
            FailureClass::Truncated,
        ];
        Self {
            validation: ValidationMode::None,
            max_retries: 0,
            retry_mode: RetryMode::Independent,
            max_tokens: None,
            allowed_fallbacks: Vec::new(),
            retry_on: recoverable.clone(),
            fallback_on: recoverable,
            allow_unknown_termination: false,
            logging: LoggingPolicy::Config,
            images: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Completion {
    /// Exact assistant text if reported; missing/null content is not an empty string.
    pub raw: Option<String>,
    pub termination: Termination,
    pub usage: Option<ProviderUsage>,
}

/// Injectable per-attempt transport: no process-global model or settings reads.
#[async_trait]
pub trait InferenceBackend: Sync {
    async fn complete(
        &self,
        model: &ModelDescriptor,
        messages: &Value,
    ) -> Result<Completion, FailureClass>;
}

/// Uses the existing OpenAI-compatible wire contract; no separate runtime/server.
pub struct OpenAiBackend {
    client: reqwest::Client,
}

impl OpenAiBackend {
    pub fn new() -> Result<Self, ExtractionError> {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|_| ExtractionError::configuration())?;
        Ok(Self { client })
    }
}

#[async_trait]
impl InferenceBackend for OpenAiBackend {
    async fn complete(
        &self,
        model: &ModelDescriptor,
        messages: &Value,
    ) -> Result<Completion, FailureClass> {
        let destination = model
            .base_url
            .as_deref()
            .unwrap_or("https://api.openai.com/v1");
        validate_destination(destination)?;
        let endpoint = format!("{}/chat/completions", destination.trim_end_matches('/'));
        let mut request = self.client.post(endpoint)
            .timeout(std::time::Duration::from_secs(model.request_timeout_secs.unwrap_or(120)))
            .json(&json!({"model": model.name, "messages": messages, "max_tokens": model.max_tokens.unwrap_or(512), "temperature": 0.7}));
        if let Some(key) = &model.api_key {
            request = request.bearer_auth(key);
        }
        let response = request.send().await.map_err(|_| FailureClass::Transport)?;
        if !response.status().is_success() {
            return Err(FailureClass::Transport);
        }
        let wire: Value = response.json().await.map_err(|_| FailureClass::Malformed)?;
        decode_completion(&wire)
    }
}

fn validate_destination(destination: &str) -> Result<(), FailureClass> {
    let url = reqwest::Url::parse(destination).map_err(|_| FailureClass::Configuration)?;
    if !matches!(url.scheme(), "http" | "https")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(FailureClass::Configuration);
    }
    Ok(())
}

fn decode_completion(wire: &Value) -> Result<Completion, FailureClass> {
    let choices = wire
        .get("choices")
        .and_then(Value::as_array)
        .ok_or(FailureClass::Malformed)?;
    if choices.len() != 1 {
        return Err(FailureClass::Malformed);
    }
    let choice = &choices[0];
    let message = &choice["message"];
    let refused = match message.get("refusal") {
        None | Some(Value::Null) => Ok(false),
        Some(Value::String(_)) => Ok(true),
        _ => Err(FailureClass::Malformed),
    };
    let reported_termination = match choice.get("finish_reason") {
        None | Some(Value::Null) => Ok(Termination::Unknown),
        Some(Value::String(reason)) => Ok(match reason.as_str() {
            "stop" => Termination::Stop,
            "length" => Termination::Length,
            "content_filter" => Termination::Refusal,
            "tool_calls" | "function_call" => Termination::ToolCalls,
            _ => Termination::Other(reason.clone()),
        }),
        _ => Err(FailureClass::Malformed),
    };
    // A known rejection cannot be downgraded into recovery by malformed ancillary
    // content/metadata. Missing metadata is unknown; wrongly typed metadata isn't.
    let termination = match (refused, reported_termination) {
        (Ok(true), _) | (_, Ok(Termination::Refusal)) => Termination::Refusal,
        (_, Ok(Termination::Length)) => Termination::Length,
        (_, Ok(Termination::ToolCalls)) => Termination::ToolCalls,
        (_, Ok(Termination::Other(reason))) => Termination::Other(reason),
        (Ok(false), reported) => reported?,
        (Err(failure), _) => return Err(failure),
    };
    let raw = match message.get("content") {
        Some(Value::String(raw)) => Some(raw.clone()),
        Some(Value::Null) | None => None,
        _ if matches!(
            termination,
            Termination::Refusal
                | Termination::Length
                | Termination::ToolCalls
                | Termination::Other(_)
        ) =>
        {
            None
        }
        _ => return Err(FailureClass::Malformed),
    };
    let usage = wire
        .get("usage")
        .filter(|v| v.is_object())
        .map(|v| ProviderUsage {
            prompt_tokens: v.get("prompt_tokens").and_then(Value::as_u64),
            completion_tokens: v.get("completion_tokens").and_then(Value::as_u64),
            total_tokens: v.get("total_tokens").and_then(Value::as_u64),
        });
    Ok(Completion {
        raw,
        termination,
        usage,
    })
}

pub(crate) struct CandidateModel {
    pub descriptor: ModelDescriptor,
    pub identity: ModelIdentity,
    pub adapter: AdapterKind,
    pub messages: Value,
}

fn rejected_termination(termination: &Termination, allow_unknown: bool) -> Option<FailureClass> {
    match termination {
        Termination::Stop => None,
        Termination::Unknown if allow_unknown => None,
        Termination::Unknown => Some(FailureClass::UnknownTermination),
        Termination::Length => Some(FailureClass::Truncated),
        Termination::Refusal => Some(FailureClass::Refusal),
        _ => Some(FailureClass::UnsupportedTermination),
    }
}

/// Preserve all original turns. History is scoped to retries on ONE destination.
fn retry_messages(base: &Value, history: &[Value]) -> Value {
    let mut messages = base.as_array().cloned().unwrap_or_default();
    messages.extend_from_slice(history);
    Value::Array(messages)
}

/// One request's owned validator and credential/termination admission policy.
struct Admission<'a> {
    options: &'a InferenceOptions,
    validator: Option<jsonschema::JSONSchema>,
    credentials: &'a crate::privacy::CredentialInventory,
}

impl Admission<'_> {
    fn contains_credential(&self, text: &str) -> bool {
        self.credentials.contains_text(text)
    }

    fn assess(
        &self,
        adapter: AdapterKind,
        completion: Completion,
        attempt: &mut AttemptReceipt,
    ) -> Result<(), FailureClass> {
        // Never redact a literal into a successful extraction: reject/withhold
        // known credential echoes before adding evidence to results or logs.
        let sensitive_envelope = completion
            .raw
            .as_deref()
            .and_then(|raw| adapter.decoded_envelope(raw))
            .is_some_and(|payload| self.credentials.contains_json(&payload));
        if completion
            .raw
            .as_deref()
            .is_some_and(|raw| self.contains_credential(raw))
            || sensitive_envelope
            || matches!(&completion.termination, Termination::Other(reason) if self.contains_credential(reason))
        {
            return Err(FailureClass::SensitiveResponse);
        }
        attempt.raw_completion = completion.raw.clone();
        attempt.termination = Some(completion.termination.clone());
        attempt.usage = completion.usage;
        if let Some(failure) = rejected_termination(
            &completion.termination,
            self.options.allow_unknown_termination,
        ) {
            return Err(failure);
        }
        let raw = completion.raw.as_deref().ok_or(FailureClass::Malformed)?;
        let parsed = adapter
            .decode_extraction(raw)
            .map_err(|_| FailureClass::Malformed)?;
        if self.credentials.contains_json(&parsed) {
            attempt.raw_completion = None;
            attempt.termination = None;
            attempt.usage = None;
            return Err(FailureClass::SensitiveResponse);
        }
        attempt.validation = match &self.validator {
            None => ValidationStatus::NotChecked,
            Some(validator) => match validator.validate(&parsed) {
                Ok(()) => ValidationStatus::Passed,
                Err(errors) => {
                    attempt.validation_errors = errors
                        .map(|error| format!("{}: {error}", error.instance_path))
                        .collect();
                    ValidationStatus::Failed
                }
            },
        };
        if attempt
            .validation_errors
            .iter()
            .any(|error| self.contains_credential(error))
        {
            attempt.raw_completion = None;
            attempt.termination = None;
            attempt.usage = None;
            attempt.validation_errors.clear();
            attempt.validation = ValidationStatus::NotChecked;
            return Err(FailureClass::SensitiveResponse);
        }
        attempt.parsed = Some(parsed);
        if attempt.validation == ValidationStatus::Failed
            && self.options.validation == ValidationMode::Error
        {
            return Err(FailureClass::SchemaInvalid);
        }
        Ok(())
    }
}

pub(crate) async fn execute<B: InferenceBackend>(
    models: &[CandidateModel],
    schema: &str,
    options: &InferenceOptions,
    backend: &B,
    credentials: &crate::privacy::CredentialInventory,
    log_dir: Option<&std::path::Path>,
) -> Result<ExtractionReceipt, ExtractionError> {
    let validator = if options.validation != ValidationMode::None {
        Some(crate::schema::compile_schema(schema).map_err(|_| ExtractionError::configuration())?)
    } else {
        None
    };
    let admission = Admission {
        options,
        validator,
        credentials,
    };
    let requested = models
        .first()
        .ok_or_else(ExtractionError::configuration)?
        .identity
        .clone();
    let mut attempts = Vec::new();
    let mut last_failure = None;
    let mut warnings = Vec::new();
    for (index, model) in models.iter().enumerate() {
        if index > 0 {
            if !options.allowed_fallbacks.contains(&model.identity) {
                attempts.push(AttemptReceipt {
                    ordinal: attempts.len() + 1,
                    model: model.identity.clone(),
                    retry: 0,
                    outcome: AttemptOutcome::Failed(FailureClass::Denied),
                    raw_completion: None,
                    parsed: None,
                    termination: None,
                    usage: None,
                    validation: ValidationStatus::NotChecked,
                    validation_errors: Vec::new(),
                });
                continue;
            }
            if !last_failure
                .as_ref()
                .is_some_and(|f| options.fallback_on.contains(f))
            {
                break;
            }
        }
        let mut history = Vec::new();
        for retry in 0..=options.max_retries {
            let messages = retry_messages(&model.messages, &history);
            let mut attempt = AttemptReceipt {
                ordinal: attempts.len() + 1,
                model: model.identity.clone(),
                retry,
                outcome: AttemptOutcome::Accepted,
                raw_completion: None,
                parsed: None,
                termination: None,
                usage: None,
                validation: ValidationStatus::NotChecked,
                validation_errors: Vec::new(),
            };
            let completion = backend.complete(&model.descriptor, &messages).await;
            let failure = completion
                .and_then(|completion| admission.assess(model.adapter, completion, &mut attempt))
                .err();
            if let Some(failure) = &failure {
                attempt.outcome = AttemptOutcome::Failed(failure.clone());
            }
            if let Some(dir) = log_dir {
                // Response logging is explicitly per-request and private; failures are
                // warnings, not network retries. Sensitive completions are not logged.
                let sensitive_prompt = credentials.contains_json(&messages);
                if !sensitive_prompt
                    && failure != Some(FailureClass::SensitiveResponse)
                    && log_attempt(dir, &messages, &attempt).is_err()
                {
                    warnings.push("attempt logging failed".into());
                }
            }
            if failure.is_none() {
                let parsed = attempt
                    .parsed
                    .clone()
                    .ok_or_else(ExtractionError::configuration)?;
                let raw_completion = attempt
                    .raw_completion
                    .clone()
                    .ok_or_else(ExtractionError::configuration)?;
                let termination = attempt
                    .termination
                    .clone()
                    .ok_or_else(ExtractionError::configuration)?;
                let validation = attempt.validation.clone();
                let usage = attempt.usage.clone();
                let completeness = if termination == Termination::Stop {
                    Completeness::ReportedStop
                } else {
                    Completeness::Unverified
                };
                attempts.push(attempt);
                return Ok(ExtractionReceipt {
                    parsed,
                    raw_completion,
                    requested_model: requested,
                    used_model: model.identity.clone(),
                    attempts,
                    validation,
                    termination,
                    usage,
                    completeness,
                    warnings,
                });
            }
            let failure = failure.ok_or_else(ExtractionError::configuration)?;
            if options.retry_mode == RetryMode::WithHistory {
                if let Some(raw) = &attempt.raw_completion {
                    history.push(json!({"role": "assistant", "content": raw}));
                }
                history.push(json!({"role": "user", "content": format!("Previous attempt rejected: {failure:?}. Return a complete response matching the schema.")}));
            }
            attempts.push(attempt);
            last_failure = Some(failure.clone());
            if !options.retry_on.contains(&failure) {
                break;
            }
        }
    }
    Err(ExtractionError {
        kind: FailureClass::Exhausted,
        last_failure,
        attempts,
    })
}

fn log_attempt(
    dir: &std::path::Path,
    messages: &Value,
    attempt: &AttemptReceipt,
) -> anyhow::Result<()> {
    crate::capture::write_private_json(dir, &json!({"request": messages, "attempt": attempt}))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_metadata_cannot_be_downgraded_by_ancillary_shape_errors() {
        let cases = [
            (
                json!("stop"),
                json!("declined"),
                json!([]),
                Termination::Refusal,
            ),
            (
                json!(true),
                json!("declined"),
                Value::Null,
                Termination::Refusal,
            ),
            (
                json!("content_filter"),
                json!(false),
                Value::Null,
                Termination::Refusal,
            ),
            (
                json!("tool_calls"),
                Value::Null,
                json!([]),
                Termination::ToolCalls,
            ),
            (
                json!("length"),
                json!(false),
                json!({}),
                Termination::Length,
            ),
            (
                json!("vendor_end"),
                json!(false),
                json!({}),
                Termination::Other("vendor_end".into()),
            ),
        ];
        for (finish, refusal, content, expected) in cases {
            let completion = decode_completion(&json!({"choices": [{"finish_reason": finish, "message": {"content": content, "refusal": refusal}}]})).expect("known terminal event");
            assert_eq!(completion.termination, expected);
            assert_eq!(completion.raw, None);
        }
    }

    #[test]
    fn wire_metadata_is_truthful() {
        for (finish, expected) in [
            ("stop", Termination::Stop),
            ("length", Termination::Length),
            ("content_filter", Termination::Refusal),
            ("tool_calls", Termination::ToolCalls),
        ] {
            let completion = decode_completion(
                &json!({"choices": [{"message": {"content": "{}"}, "finish_reason": finish}]}),
            )
            .expect("synthetic response");
            assert_eq!(completion.termination, expected);
            assert_eq!(completion.usage, None);
        }
        let completion = decode_completion(
            &json!({"choices": [{"message": {"content": "{}"}}], "usage": {"prompt_tokens": 4}}),
        )
        .expect("missing metadata is valid wire input");
        assert_eq!(completion.termination, Termination::Unknown);
        assert_eq!(
            completion
                .usage
                .expect("reported partial usage")
                .total_tokens,
            None
        );
        let refusal = decode_completion(&json!({"choices": [{"message": {"content": null, "refusal": "declined"}, "finish_reason": "stop"}]})).expect("refusal");
        assert_eq!(refusal.termination, Termination::Refusal);
    }
}
