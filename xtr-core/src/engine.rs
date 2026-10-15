use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use dspy_rs::core::Optimizable as OptimizableTrait;
use dspy_rs::example;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;

use crate::capture::CaptureQualification;
use crate::capture::CaptureRecord;
use crate::config::AppConfig;
use crate::config::AppPaths;
use crate::config::ConfigBundle;
use crate::config::OptimizationSettings;
use crate::config::ResolvedTaskConfig;
use crate::config::load_or_initialize_config;
use crate::inference::CandidateModel;
use crate::inference::ExtractionError;
use crate::inference::ExtractionReceipt;
use crate::inference::InferenceBackend;
use crate::inference::InferenceOptions;
use crate::inference::LoggingPolicy;
use crate::inference::ModelIdentity;
use crate::inference::ModelRole;
use crate::inference::OpenAiBackend;
use crate::lm::AdapterKind;
use crate::lm::TaskModelHandles;
use crate::optimization::ExtractionProgram;
use crate::optimization::GepaOutcome;
use crate::optimization::GepaRunner;
use crate::optimization::load_best_instruction;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationMode {
    None,
    Warn,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryMode {
    Independent,
    WithHistory,
}

/// Engines own configuration only: construction never mutates model/log globals.
#[derive(Debug)]
pub struct ExtractionEngine {
    bundle: ConfigBundle,
}

impl ExtractionEngine {
    pub fn load(app_name: impl AsRef<str>) -> Result<Self> {
        Ok(Self {
            bundle: load_or_initialize_config(app_name)?,
        })
    }

    pub fn from_bundle(bundle: ConfigBundle) -> Self {
        Self { bundle }
    }

    fn resolve_llm_log_dir(bundle: &ConfigBundle) -> Result<std::path::PathBuf> {
        if let Some(custom_dir) = bundle.config.logging.llm_log_dir.as_ref() {
            crate::config::resolve_path_value(custom_dir, &bundle.paths.config_dir)
        } else {
            Ok(bundle.paths.state_dir.canonicalize()?.join("llm_logs"))
        }
    }

    pub fn config(&self) -> &AppConfig {
        &self.bundle.config
    }
    pub fn paths(&self) -> &AppPaths {
        &self.bundle.paths
    }

    pub fn resolve_task(&self, task_name: &str) -> Result<ResolvedTaskConfig> {
        self.bundle
            .config
            .resolve_task(task_name, &self.bundle.paths)
    }

    pub async fn task_models(&self, task_name: &str) -> Result<TaskModelHandles> {
        TaskModelHandles::load(&self.resolve_task(task_name)?.models).await
    }

    pub async fn optimize_task(&self, task_name: &str) -> Result<GepaOutcome> {
        self.optimize_task_with_overrides(task_name, None).await
    }

    pub async fn optimize_task_with_overrides(
        &self,
        task_name: &str,
        overrides: Option<OptimizationSettings>,
    ) -> Result<GepaOutcome> {
        let mut task = self.resolve_task(task_name)?;
        if let Some(override_settings) = overrides {
            let task_overrides = self.bundle.config.optimization.tasks.get(task_name);
            task.optimization = self
                .bundle
                .config
                .optimization
                .resolve(Some(&override_settings), task_name)?;
            if let Some(existing_overrides) = task_overrides {
                let merged = crate::config::merge_optimization_settings_public(
                    existing_overrides,
                    Some(&override_settings),
                );
                task.optimization = self
                    .bundle
                    .config
                    .optimization
                    .resolve(Some(&merged), task_name)?;
            }
        }
        let models = TaskModelHandles::load(&task.models).await?;
        let optimization = task.optimization.clone();
        let log_dir = self.bundle.config.mlflow.log_dir.as_ref().map(|s| {
            let expanded = shellexpand::tilde(s);
            std::path::PathBuf::from(expanded.as_ref())
        });
        let runner = GepaRunner {
            task,
            optimization,
            models,
            metrics_config: self.bundle.config.metrics.clone(),
            #[cfg(feature = "mlflow")]
            mlflow_tracking_uri: self.bundle.config.mlflow.tracking_uri.clone(),
            #[cfg(feature = "mlflow")]
            mlflow_experiment: self.bundle.config.mlflow.experiment_name.clone(),
            local_logging: self.bundle.config.mlflow.local_logging,
            log_dir,
        };
        runner.run(&self.bundle.paths.state_dir).await
    }

    /// Resolve an exact destination before explicitly granting it in request options.
    pub fn fallback_identity(
        &self,
        task: &str,
        role: ModelRole,
    ) -> Result<ModelIdentity, ExtractionError> {
        let task = self
            .resolve_task(task)
            .map_err(|_| ExtractionError::configuration())?;
        let descriptor = match &role {
            ModelRole::Student => &task.models.student,
            ModelRole::Teacher => &task.models.teacher,
            ModelRole::Fallback(index) => task
                .models
                .fallbacks
                .get(*index)
                .ok_or_else(ExtractionError::configuration)?,
        };
        let identity = ModelIdentity::from_descriptor(role, descriptor)?;
        let credentials = crate::privacy::CredentialInventory::from_config(self.config());
        if credentials.contains_text(&identity.name)
            || credentials.contains_text(&identity.destination)
        {
            return Err(ExtractionError::configuration());
        }
        Ok(identity)
    }

    /// One typed inference seam. Configuration is not a grant to send to fallbacks.
    pub async fn run_inference(
        &self,
        task: &str,
        input: &str,
        context: Option<&str>,
        options: &InferenceOptions,
    ) -> Result<ExtractionReceipt, ExtractionError> {
        let backend = OpenAiBackend::new()?;
        self.run_inference_with_backend(task, input, context, options, &backend)
            .await
    }

    /// CPU-only tests and embedders can inject a request-local completion backend.
    pub async fn run_inference_with_backend<B: InferenceBackend>(
        &self,
        task_name: &str,
        input_text: &str,
        additional_context: Option<&str>,
        options: &InferenceOptions,
        backend: &B,
    ) -> Result<ExtractionReceipt, ExtractionError> {
        let credentials = crate::privacy::CredentialInventory::from_config(self.config());
        let (schema, models, final_context) = self
            .prepare_request(task_name, input_text, additional_context, options)
            .map_err(|_| ExtractionError::configuration())?;
        let log_dir = match &options.logging {
            LoggingPolicy::Disabled => None,
            LoggingPolicy::Directory(path) => Some(path.clone()),
            LoggingPolicy::Config if self.bundle.config.logging.verbose_llm_logging => Some(
                Self::resolve_llm_log_dir(&self.bundle)
                    .map_err(|_| ExtractionError::configuration())?,
            ),
            LoggingPolicy::Config => None,
        };
        let mut receipt = crate::inference::execute(
            &models,
            &schema,
            options,
            backend,
            &credentials,
            log_dir.as_deref(),
        )
        .await?;
        let sensitive_source =
            credentials.contains_text(input_text) || credentials.contains_text(&final_context);
        if self.bundle.config.data_collection.enabled && sensitive_source {
            receipt
                .warnings
                .push("candidate capture suppressed for sensitive source".into());
        }
        if self.bundle.config.data_collection.enabled && !sensitive_source {
            let capture = (|| -> Result<()> {
                let configured = self
                    .config()
                    .resolved_collection_dir(self.paths())?
                    .ok_or_else(|| anyhow!("collection policy is disabled"))?;
                let base = if self.config().data_collection.output_dir.is_some() {
                    configured
                } else {
                    self.paths()
                        .state_dir
                        .canonicalize()?
                        .join("collected_examples")
                };
                // Default capture retains accepted evidence and truthful failed-attempt
                // metadata, not rejected completions or rejected parsed candidates.
                let mut captured_receipt = receipt.clone();
                for attempt in &mut captured_receipt.attempts {
                    if matches!(attempt.outcome, crate::inference::AttemptOutcome::Failed(_)) {
                        attempt.raw_completion = None;
                        attempt.parsed = None;
                        attempt.validation_errors.clear();
                        if matches!(
                            attempt.termination,
                            Some(crate::inference::Termination::Other(_))
                        ) {
                            attempt.termination = Some(crate::inference::Termination::Unsupported);
                        }
                    }
                }
                let record = CaptureRecord {
                    input_text: input_text.to_string(),
                    additional_context: (!final_context.is_empty()).then_some(final_context),
                    receipt: serde_json::to_value(&captured_receipt)?,
                    qualification: CaptureQualification::Unreviewed,
                };
                crate::capture::capture_candidate(&base, task_name, &record)?;
                Ok(())
            })();
            if capture.is_err() {
                receipt.warnings.push("candidate capture failed".into());
            }
        }
        Ok(receipt)
    }

    fn prepare_request(
        &self,
        task_name: &str,
        input_text: &str,
        additional_context: Option<&str>,
        options: &InferenceOptions,
    ) -> Result<(String, Vec<CandidateModel>, String)> {
        let task = self.resolve_task(task_name)?;
        let schema = std::fs::read_to_string(
            task.schema_path
                .as_ref()
                .ok_or_else(|| anyhow!("task is missing schema"))?,
        )?;
        let mut program = ExtractionProgram::new();
        if let Some(best) = load_best_instruction(&self.bundle.paths, task_name) {
            OptimizableTrait::update_signature_instruction(program.solver_mut(), best)?;
        } else if let Some(description) = &task.description {
            let base = OptimizableTrait::get_signature(program.solver()).instruction();
            let combined = if base.trim().is_empty() {
                description.clone()
            } else {
                format!("{base}\n\nTask context: {description}")
            };
            OptimizableTrait::update_signature_instruction(program.solver_mut(), combined)?;
        }
        let mut context_parts = Vec::new();
        // Preserve caller context, including whitespace. Never silently reduce input.
        if let Some(context) = additional_context {
            context_parts.push(context.to_string());
        }
        if task.include_timestamp {
            let timestamp = chrono::Local::now()
                .format("%A, %B %d, %Y at %I:%M %p %Z")
                .to_string();
            context_parts.push(format!("Current date and time: {timestamp}"));
        }
        let final_context = context_parts.join("\n\n");
        let signature = OptimizableTrait::get_signature(program.solver());
        let mut descriptors = vec![
            (ModelRole::Student, task.models.student),
            (ModelRole::Teacher, task.models.teacher),
        ];
        descriptors.extend(
            task.models
                .fallbacks
                .into_iter()
                .enumerate()
                .map(|(index, model)| (ModelRole::Fallback(index), model)),
        );
        let mut models = Vec::new();
        for (role, mut descriptor) in descriptors {
            if let Some(tokens) = options.max_tokens {
                descriptor.max_tokens = Some(tokens);
            }
            let identity = ModelIdentity::from_descriptor(role, &descriptor)?;
            let credentials = crate::privacy::CredentialInventory::from_config(self.config());
            if credentials.contains_text(&identity.name)
                || credentials.contains_text(&identity.destination)
            {
                return Err(anyhow!("credential-bearing model identity"));
            }
            let adapter = AdapterKind::from_descriptor(&descriptor)?;
            let inputs = example! {
                "schema": "input" => &schema,
                "input_text": "input" => input_text,
                "additional_context": "input" => final_context.as_str(),
                "image_data_uris": "input" => options.images.clone()
            };
            let messages = adapter.format_wire_messages(signature, inputs)?;
            models.push(CandidateModel {
                descriptor,
                identity,
                adapter,
                messages,
            });
        }
        Ok((schema, models, final_context))
    }

    /// Generate examples using direct, local adapter dispatch rather than DSPy globals.
    pub async fn generate_examples(
        &self,
        task_name: &str,
        count: u32,
        additional_context: Option<&str>,
    ) -> Result<Vec<crate::examples::RawTaskExample>> {
        let task = self.resolve_task(task_name)?;
        let generator_descriptor = task
            .models
            .dataset_generator
            .as_ref()
            .ok_or_else(|| anyhow!("no dataset_generator model configured"))?;
        let generator_model = crate::lm::build_model_handle(generator_descriptor)
            .await
            .context("failed to build dataset_generator model")?;
        let schema_content = std::fs::read_to_string(
            task.schema_path
                .as_ref()
                .ok_or_else(|| anyhow!("task is missing schema"))?,
        )?;
        let mut context_parts = vec![format!("Generate {count} diverse training examples.")];
        if let Some(ctx) = additional_context {
            context_parts.push(ctx.to_string());
        }
        if let Some(desc) = &task.description {
            context_parts.push(format!("Task description: {desc}"));
        }
        let context = context_parts.join("\n\n");
        let mut program = ExtractionProgram::new();
        OptimizableTrait::update_signature_instruction(program.solver_mut(), r#"You are an expert at generating high-quality training examples for structured data extraction tasks.

Given a target JSON schema, generate diverse and realistic examples that:
1. Cover different input formats (emails, documents, forms, web pages, etc.)
2. Include varying difficulty levels (easy, medium, hard, edge cases)
3. Represent realistic data with appropriate noise and variations
4. Accurately extract the correct structured output

Generate examples that would help train a model to perform the extraction task well."#.to_string())?;
        let input_text =
            format!("Target Schema for extraction task '{task_name}':\n\n{schema_content}");
        let signature = OptimizableTrait::get_signature(program.solver());
        let response = generator_model.predict(signature, example! {
            "schema": "input" => include_str!("../../examples/schemas/example_generator.json"),
            "input_text": "input" => &input_text,
            "additional_context": "input" => &context
        }).await?;
        let output = response.get("output_json", None);
        let parsed: Value = serde_json::from_str(
            output
                .as_str()
                .ok_or_else(|| anyhow!("missing generator output_json"))?,
        )?;
        let examples = parsed
            .get("examples")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("generator output missing examples"))?;
        examples
            .iter()
            .map(|value| {
                Ok(crate::examples::RawTaskExample {
                    input_text: value
                        .get("input_text")
                        .and_then(Value::as_str)
                        .ok_or_else(|| anyhow!("generated example missing input_text"))?
                        .to_string(),
                    expected_json: value
                        .get("expected_json")
                        .cloned()
                        .ok_or_else(|| anyhow!("generated example missing expected_json"))?,
                    additional_context: value
                        .get("additional_context")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    images: Vec::new(),
                })
            })
            .collect()
    }
}
