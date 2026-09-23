use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use serde::Deserialize;
use serde::Serialize;

/// Embedded template used to bootstrap the on-disk configuration when the user
/// runs the tool for the first time.
pub const DEFAULT_CONFIG_TEMPLATE: &str = include_str!("../../examples/config.toml");

/// Container returned after loading configuration data and resolving runtime
/// paths.
#[derive(Debug, Clone)]
pub struct ConfigBundle {
    pub config: AppConfig,
    pub paths: AppPaths,
}

/// Resolve and load the configuration for the provided application name. If no
/// config file exists yet, a default file is created from
/// [`DEFAULT_CONFIG_TEMPLATE`].
pub fn load_or_initialize_config(app_name: impl AsRef<str>) -> Result<ConfigBundle> {
    let app_name = app_name.as_ref();
    let mut paths = AppPaths::discover(app_name)?;
    paths.ensure_config_dir()?;

    if !paths.config_file.exists() {
        if let Some(parent) = paths.config_file.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create config directory {}", parent.display())
            })?;
        }

        fs::write(&paths.config_file, DEFAULT_CONFIG_TEMPLATE).with_context(|| {
            format!(
                "failed to write default config to {}",
                paths.config_file.display()
            )
        })?;
    }

    let config_toml = fs::read_to_string(&paths.config_file).with_context(|| {
        format!(
            "failed to read configuration at {}",
            paths.config_file.display()
        )
    })?;

    let mut config: AppConfig = toml::from_str(&config_toml).with_context(|| {
        format!(
            "failed to parse configuration at {}",
            paths.config_file.display()
        )
    })?;

    paths = paths.apply_storage_overrides(&config.storage)?;
    paths.ensure_runtime_dirs()?;

    config.normalize()?;

    Ok(ConfigBundle { config, paths })
}

/// Persistent runtime paths derived from XDG environment variables or sensible
/// fallbacks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppPaths {
    pub app_name: String,
    pub config_dir: PathBuf,
    pub config_file: PathBuf,
    pub data_dir: PathBuf,
    pub state_dir: PathBuf,
    pub cache_dir: PathBuf,
}

impl AppPaths {
    pub fn discover(app_name: impl Into<String>) -> Result<Self> {
        let app_name = app_name.into();

        let config_dir = base_dir("XDG_CONFIG_HOME", ".config", "APPDATA")?.join(&app_name);
        let data_dir = base_dir("XDG_DATA_HOME", ".local/share", "APPDATA")?.join(&app_name);
        let state_dir = base_dir("XDG_STATE_HOME", ".local/state", "LOCALAPPDATA")?.join(&app_name);
        // Cache lives under an explicit XDG_CACHE_HOME if set, else under state/cache.
        let cache_dir = match base_dir("XDG_CACHE_HOME", ".cache", "LOCALAPPDATA") {
            Ok(base) => base.join(&app_name),
            Err(_) => state_dir.join("cache"),
        };
        let config_file = config_dir.join("config.toml");

        Ok(Self {
            app_name,
            config_dir,
            config_file,
            data_dir,
            state_dir,
            cache_dir,
        })
    }

    pub fn ensure_config_dir(&self) -> Result<()> {
        fs::create_dir_all(&self.config_dir).with_context(|| {
            format!(
                "failed to create config directory {}",
                self.config_dir.display()
            )
        })
    }

    pub fn ensure_runtime_dirs(&self) -> Result<()> {
        for dir in [&self.data_dir, &self.state_dir, &self.cache_dir] {
            fs::create_dir_all(dir)
                .with_context(|| format!("failed to create runtime directory {}", dir.display()))?;
        }
        Ok(())
    }

    pub fn apply_storage_overrides(&self, storage: &StorageSettings) -> Result<Self> {
        let mut next = self.clone();

        if let Some(data_dir) = storage.data_dir.as_ref() {
            next.data_dir = resolve_path_value(data_dir, &self.config_dir)?;
        }

        if let Some(state_dir) = storage.state_dir.as_ref() {
            next.state_dir = resolve_path_value(state_dir, &self.config_dir)?;
        }

        if let Some(cache_dir) = storage.cache_dir.as_ref() {
            next.cache_dir = resolve_path_value(cache_dir, &self.config_dir)?;
        } else {
            // Ensure cache lives under the state directory by default.
            next.cache_dir = next.state_dir.join("cache");
        }

        Ok(next)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct AppConfig {
    pub models: ModelSection,
    pub tasks: HashMap<String, TaskConfig>,
    pub storage: StorageSettings,
    pub data_collection: DataCollectionSettings,
    pub optimization: OptimizationSection,
    pub mlflow: MlflowSettings,
    pub metrics: MetricsConfig,
    pub logging: LoggingSettings,
}

impl AppConfig {
    pub fn normalize(&mut self) -> Result<()> {
        // Ensure defaults always have named teacher/student models.
        if self.models.defaults.teacher.name.is_none()
            || self.models.defaults.student.name.is_none()
        {
            bail!("default teacher and student models must specify a name");
        }
        Ok(())
    }

    pub fn resolve_task(&self, task_name: &str, paths: &AppPaths) -> Result<ResolvedTaskConfig> {
        let task_config = self.tasks.get(task_name);
        let model_overrides = self.models.tasks.get(task_name);
        let optimization_overrides = self.optimization.tasks.get(task_name);

        let models = self
            .models
            .resolve(task_name, model_overrides)
            .context("failed to resolve model configuration")?;

        let optimization = self
            .optimization
            .resolve(optimization_overrides, task_name)?;

        let schema_path = task_config
            .and_then(|task| task.schema.as_ref())
            .map(|path| resolve_path_value(path, &paths.config_dir))
            .transpose()
            .with_context(|| format!("failed to resolve schema path for task {task_name}"))?;

        let examples_dir = task_config
            .and_then(|task| task.examples.as_ref())
            .map(|path| resolve_path_value(path, &paths.config_dir))
            .transpose()
            .with_context(|| format!("failed to resolve examples path for task {task_name}"))?;

        // Resolve `@file` references (rubric, verdict_schema) inside the task's
        // metric config relative to the config directory.
        let metric_config = task_config
            .and_then(|task| task.metric.as_ref())
            .map(|metric| resolve_metric_files(metric, &paths.config_dir))
            .transpose()
            .with_context(|| format!("failed to resolve metric config for task {task_name}"))?;

        Ok(ResolvedTaskConfig {
            name: task_name.to_string(),
            description: task_config.and_then(|task| task.description.clone()),
            schema_path,
            examples_dir,
            models,
            optimization,
            include_timestamp: task_config
                .and_then(|task| task.include_timestamp)
                .unwrap_or(false),
            kind: task_config
                .and_then(|task| task.kind)
                .unwrap_or(TaskKind::Extract),
            metric: metric_config,
        })
    }

    pub fn resolved_collection_dir(&self, paths: &AppPaths) -> Result<Option<PathBuf>> {
        if !self.data_collection.enabled {
            return Ok(None);
        }

        let dir = if let Some(custom_dir) = self.data_collection.output_dir.as_ref() {
            resolve_path_value(custom_dir, &paths.config_dir)?
        } else {
            paths.state_dir.join("collected_examples")
        };

        Ok(Some(dir))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct StorageSettings {
    pub data_dir: Option<String>,
    pub state_dir: Option<String>,
    pub cache_dir: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct DataCollectionSettings {
    pub enabled: bool,
    pub output_dir: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MlflowSettings {
    pub tracking_uri: Option<String>,
    pub experiment_name: Option<String>,
    pub local_logging: bool,
    pub log_dir: Option<String>,
}

impl Default for MlflowSettings {
    fn default() -> Self {
        Self {
            tracking_uri: None,
            experiment_name: Some("xtr-optimization".to_string()),
            local_logging: true,
            log_dir: None,
        }
    }
}

/// Configuration for evaluation metrics used during optimization.
/// These settings control how extraction quality is scored.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MetricsConfig {
    /// Weight for penalizing extra/hallucinated fields (0.0-1.0)
    /// Lower values = lighter penalty for hallucinations
    pub extra_field_weight: f32,

    /// Beta parameter for F-beta score (balances precision vs recall)
    /// Values > 1.0 favor recall, < 1.0 favor precision
    pub beta: f32,

    /// Base score awarded for valid JSON parsing
    pub base_parse_score: f32,

    /// Additional score awarded for schema validation
    pub base_schema_score: f32,

    /// Weight for field-level quality score (precision/recall)
    pub field_weight: f32,

    /// Weight for coverage bonus (encourages completeness)
    pub coverage_weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct LoggingSettings {
    /// Enable verbose logging of complete request-response cycle with the LLM
    pub verbose_llm_logging: bool,
    /// Directory where LLM request-response logs will be written (as JSON files)
    pub llm_log_dir: Option<String>,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            extra_field_weight: 0.5,
            beta: 1.5,
            base_parse_score: 0.2,
            base_schema_score: 0.2,
            field_weight: 0.5,
            coverage_weight: 0.1,
        }
    }
}

/// The kind of work a task performs. `extract` is the default (unstructured
/// input -> structured JSON). `grade` is a judge task: it rates a predicted
/// extraction against an expected output and emits a verdict conforming to a
/// `verdict_schema`. Both use the same optimization machinery.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    #[default]
    Extract,
    Grade,
}

/// How an extraction is scored during optimization.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricKind {
    /// Schema + field-level match against an expected output (default).
    #[default]
    SchemaField,
    /// Exact JSON equality against the expected output.
    Exact,
    /// Use a reusable grade task as an LLM judge; the rubric is that task's
    /// instruction, optionally optimized by GEPA.
    LlmJudge,
    /// Shell out to a subprocess that scores a predicted output and writes a
    /// verdict (conforming to `verdict_schema`) to stdout.
    Command,
}

/// Per-task metric configuration, resolved into an `ExtractionProgram`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MetricConfig {
    /// Which scoring mechanism to use.
    pub kind: MetricKind,
    /// JSON Schema the judge/command verdict must conform to. A sensible
    /// default (`{score, feedback, reasons[]}`) is used when unset.
    pub verdict_schema: Option<String>,
    /// Name of the reusable `grade` task to use as the `llm_judge` scorer.
    pub judge_task: Option<String>,
    /// Prose rubric for `llm_judge`, or `@path` to read it from a file. When
    /// unset, the grade task's (possibly optimized) instruction is used.
    pub rubric: Option<String>,
    /// Subprocess command for `command` metrics (receives JSON on stdin).
    pub command: Option<String>,
    /// Timeout for a `command` metric.
    pub timeout_secs: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct TaskConfig {
    pub schema: Option<String>,
    pub examples: Option<String>,
    pub description: Option<String>,
    pub include_timestamp: Option<bool>,
    pub kind: Option<TaskKind>,
    pub metric: Option<MetricConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct ModelSection {
    pub defaults: ModelDefaults,
    pub tasks: HashMap<String, TaskModelOverrides>,
}

impl ModelSection {
    pub fn resolve(
        &self,
        task_name: &str,
        overrides: Option<&TaskModelOverrides>,
    ) -> Result<ResolvedModelConfig> {
        let merged_teacher = merge_model_specs(
            &self.defaults.teacher,
            overrides.and_then(|o| o.teacher.as_ref()),
        );
        let merged_student = merge_model_specs(
            &self.defaults.student,
            overrides.and_then(|o| o.student.as_ref()),
        );

        let teacher = merged_teacher.into_descriptor("teacher", task_name)?;
        let student = merged_student.into_descriptor("student", task_name)?;

        let mut fallback_specs = Vec::new();

        if let Some(override_list) = overrides {
            fallback_specs.extend(override_list.fallbacks.clone());
        }
        fallback_specs.extend(self.defaults.fallbacks.clone());

        let mut fallbacks = Vec::with_capacity(fallback_specs.len());
        for (index, spec) in fallback_specs.into_iter().enumerate() {
            let descriptor = spec
                .into_descriptor("fallback", task_name)
                .with_context(|| {
                    format!("failed to resolve fallback model {index} for task {task_name}")
                })?;
            fallbacks.push(descriptor);
        }

        // Resolve dataset_generator if configured
        let dataset_generator = if let Some(spec) = &self.defaults.dataset_generator {
            Some(
                spec.clone()
                    .into_descriptor("dataset_generator", task_name)?,
            )
        } else {
            None
        };

        Ok(ResolvedModelConfig {
            teacher,
            student,
            fallbacks,
            dataset_generator,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelDefaults {
    pub teacher: ModelSpec,
    pub student: ModelSpec,
    pub fallbacks: Vec<ModelSpec>,
    /// Strong model for generating synthetic training examples.
    /// Used by the built-in example_generator task.
    pub dataset_generator: Option<ModelSpec>,
}

impl Default for ModelDefaults {
    fn default() -> Self {
        Self {
            teacher: ModelSpec {
                name: Some("llama-teacher".to_string()),
                base_url: Some("http://localhost:8080/v1".to_string()),
                api_key: None,
                adapter: Some("chat".to_string()),
                request_timeout_secs: Some(60),
                max_tokens: None,
            },
            student: ModelSpec {
                name: Some("llama-student".to_string()),
                base_url: Some("http://localhost:8081/v1".to_string()),
                api_key: None,
                max_tokens: None,
                adapter: Some("chat".to_string()),
                request_timeout_secs: Some(60),
            },
            fallbacks: Vec::new(),
            dataset_generator: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct TaskModelOverrides {
    pub teacher: Option<ModelSpec>,
    pub student: Option<ModelSpec>,
    pub fallbacks: Vec<ModelSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct ModelSpec {
    pub name: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub adapter: Option<String>,
    pub request_timeout_secs: Option<u64>,
    pub max_tokens: Option<u32>,
}

impl ModelSpec {
    pub fn into_descriptor(self, role: &str, task_name: &str) -> Result<ModelDescriptor> {
        let name = self
            .name
            .ok_or_else(|| anyhow!("model '{role}' for task '{task_name}' is missing a name"))?;

        Ok(ModelDescriptor {
            name,
            base_url: self.base_url,
            api_key: self.api_key,
            adapter: self.adapter,
            request_timeout_secs: self.request_timeout_secs,
            max_tokens: self.max_tokens,
        })
    }
}

fn merge_model_specs(base: &ModelSpec, overrides: Option<&ModelSpec>) -> ModelSpec {
    let mut merged = base.clone();

    if let Some(override_spec) = overrides {
        if override_spec.name.is_some() {
            merged.name = override_spec.name.clone();
        }
        if override_spec.base_url.is_some() {
            merged.base_url = override_spec.base_url.clone();
        }
        if override_spec.api_key.is_some() {
            merged.api_key = override_spec.api_key.clone();
        }
        if override_spec.adapter.is_some() {
            merged.adapter = override_spec.adapter.clone();
        }
        if override_spec.request_timeout_secs.is_some() {
            merged.request_timeout_secs = override_spec.request_timeout_secs;
        }
    }

    merged
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct OptimizationSection {
    pub defaults: OptimizationSettings,
    pub tasks: HashMap<String, OptimizationSettings>,
}

impl OptimizationSection {
    pub fn resolve(
        &self,
        overrides: Option<&OptimizationSettings>,
        task_name: &str,
    ) -> Result<ResolvedOptimizationSettings> {
        let merged = merge_optimization_settings(&self.defaults, overrides);

        let iterations = merged.iterations.unwrap_or(DEFAULT_ITERATIONS);
        let rollouts = merged
            .rollouts_per_iteration
            .unwrap_or(DEFAULT_ROLLOUTS_PER_ITERATION);
        let max_lm_calls = merged.max_lm_calls.unwrap_or(DEFAULT_MAX_LM_CALLS);
        let batch_size = merged.batch_size.unwrap_or(DEFAULT_BATCH_SIZE);
        let max_rollouts = merged.max_rollouts;
        let temperature = merged.temperature.unwrap_or(DEFAULT_TEMPERATURE);
        let track_stats = merged.track_stats.unwrap_or(DEFAULT_TRACK_STATS);
        let track_best_outputs = merged
            .track_best_outputs
            .unwrap_or(DEFAULT_TRACK_BEST_OUTPUTS);

        let feedback_teacher = merge_optional_models(
            self.defaults.feedback_models.teacher.as_ref(),
            overrides.and_then(|opt| opt.feedback_models.teacher.as_ref()),
        );
        let feedback_student = merge_optional_models(
            self.defaults.feedback_models.student.as_ref(),
            overrides.and_then(|opt| opt.feedback_models.student.as_ref()),
        );

        let teacher_feedback = feedback_teacher
            .map(|spec| spec.into_descriptor("feedback_teacher", task_name))
            .transpose()?;

        let student_feedback = feedback_student
            .map(|spec| spec.into_descriptor("feedback_student", task_name))
            .transpose()?;

        Ok(ResolvedOptimizationSettings {
            iterations,
            rollouts_per_iteration: rollouts,
            max_lm_calls,
            batch_size,
            max_rollouts,
            temperature,
            track_stats,
            track_best_outputs,
            feedback_models: ResolvedFeedbackModels {
                teacher: teacher_feedback,
                student: student_feedback,
            },
        })
    }
}

const DEFAULT_ITERATIONS: u32 = 4;
const DEFAULT_ROLLOUTS_PER_ITERATION: u32 = 6;
const DEFAULT_MAX_LM_CALLS: u32 = 32;
const DEFAULT_BATCH_SIZE: u32 = 4;
const DEFAULT_TEMPERATURE: f32 = 0.9;
const DEFAULT_TRACK_STATS: bool = true;
const DEFAULT_TRACK_BEST_OUTPUTS: bool = false;

pub fn merge_optimization_settings_public(
    base: &OptimizationSettings,
    overrides: Option<&OptimizationSettings>,
) -> OptimizationSettings {
    merge_optimization_settings(base, overrides)
}

fn merge_optimization_settings(
    base: &OptimizationSettings,
    overrides: Option<&OptimizationSettings>,
) -> OptimizationSettings {
    let mut merged = base.clone();
    if let Some(override_settings) = overrides {
        if override_settings.iterations.is_some() {
            merged.iterations = override_settings.iterations;
        }
        if override_settings.rollouts_per_iteration.is_some() {
            merged.rollouts_per_iteration = override_settings.rollouts_per_iteration;
        }
        if override_settings.max_lm_calls.is_some() {
            merged.max_lm_calls = override_settings.max_lm_calls;
        }
        if override_settings.batch_size.is_some() {
            merged.batch_size = override_settings.batch_size;
        }
        if override_settings.max_rollouts.is_some() {
            merged.max_rollouts = override_settings.max_rollouts;
        }
        if override_settings.temperature.is_some() {
            merged.temperature = override_settings.temperature;
        }
        if override_settings.track_stats.is_some() {
            merged.track_stats = override_settings.track_stats;
        }
        if override_settings.track_best_outputs.is_some() {
            merged.track_best_outputs = override_settings.track_best_outputs;
        }
        if override_settings.feedback_models.teacher.is_some() {
            merged.feedback_models.teacher = override_settings.feedback_models.teacher.clone();
        }
        if override_settings.feedback_models.student.is_some() {
            merged.feedback_models.student = override_settings.feedback_models.student.clone();
        }
    }
    merged
}

fn merge_optional_models(
    base: Option<&ModelSpec>,
    overrides: Option<&ModelSpec>,
) -> Option<ModelSpec> {
    match (base, overrides) {
        (None, None) => None,
        (Some(base_spec), None) => Some(base_spec.clone()),
        (None, Some(override_spec)) => Some(override_spec.clone()),
        (Some(base_spec), Some(override_spec)) => {
            Some(merge_model_specs(base_spec, Some(override_spec)))
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OptimizationSettings {
    pub iterations: Option<u32>,
    pub rollouts_per_iteration: Option<u32>,
    pub max_lm_calls: Option<u32>,
    pub batch_size: Option<u32>,
    pub max_rollouts: Option<u32>,
    pub temperature: Option<f32>,
    pub track_stats: Option<bool>,
    pub track_best_outputs: Option<bool>,
    pub feedback_models: FeedbackModelOverrides,
}

impl Default for OptimizationSettings {
    fn default() -> Self {
        Self {
            iterations: Some(DEFAULT_ITERATIONS),
            rollouts_per_iteration: Some(DEFAULT_ROLLOUTS_PER_ITERATION),
            max_lm_calls: Some(DEFAULT_MAX_LM_CALLS),
            batch_size: Some(DEFAULT_BATCH_SIZE),
            max_rollouts: None,
            temperature: Some(DEFAULT_TEMPERATURE),
            track_stats: Some(DEFAULT_TRACK_STATS),
            track_best_outputs: Some(DEFAULT_TRACK_BEST_OUTPUTS),
            feedback_models: FeedbackModelOverrides::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct FeedbackModelOverrides {
    pub teacher: Option<ModelSpec>,
    pub student: Option<ModelSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDescriptor {
    pub name: String,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub adapter: Option<String>,
    pub request_timeout_secs: Option<u64>,
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModelConfig {
    pub teacher: ModelDescriptor,
    pub student: ModelDescriptor,
    pub fallbacks: Vec<ModelDescriptor>,
    /// Strong model for generating synthetic training examples.
    pub dataset_generator: Option<ModelDescriptor>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedFeedbackModels {
    pub teacher: Option<ModelDescriptor>,
    pub student: Option<ModelDescriptor>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedOptimizationSettings {
    pub iterations: u32,
    pub rollouts_per_iteration: u32,
    pub max_lm_calls: u32,
    pub batch_size: u32,
    pub max_rollouts: Option<u32>,
    pub temperature: f32,
    pub track_stats: bool,
    pub track_best_outputs: bool,
    pub feedback_models: ResolvedFeedbackModels,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedTaskConfig {
    pub name: String,
    pub description: Option<String>,
    pub schema_path: Option<PathBuf>,
    pub examples_dir: Option<PathBuf>,
    pub models: ResolvedModelConfig,
    pub optimization: ResolvedOptimizationSettings,
    pub include_timestamp: bool,
    pub kind: TaskKind,
    pub metric: Option<MetricConfig>,
}

/// Resolve a base directory deterministically (pure; unit-tested below).
///
/// An explicit, absolute `XDG_*` path wins on any OS; otherwise `~/<unix_rel>`
/// on all unix **including macOS** (deliberately NOT `~/Library/Application
/// Support` for CLI tools), or the Windows known directory.
fn resolve_base(
    xdg: Option<PathBuf>,
    home: Option<PathBuf>,
    win_dir: Option<PathBuf>,
    is_windows: bool,
    unix_rel: &str,
) -> Option<PathBuf> {
    if let Some(path) = xdg.filter(|p| p.is_absolute()) {
        return Some(path);
    }
    if is_windows {
        win_dir
    } else {
        home.map(|home| home.join(unix_rel))
    }
}

/// Read the relevant env vars and resolve a base dir via [`resolve_base`].
/// Zero-dependency: no `dirs` crate (whose macOS default we deliberately avoid).
fn base_dir(xdg_var: &str, unix_rel: &str, win_var: &str) -> Result<PathBuf> {
    resolve_base(
        env::var_os(xdg_var).map(PathBuf::from),
        env::var_os("HOME").map(PathBuf::from),
        env::var_os(win_var).map(PathBuf::from),
        cfg!(windows),
        unix_rel,
    )
    .ok_or_else(|| anyhow!("unable to determine base directory ({xdg_var})"))
}

/// Resolve `@file` references in a metric config (`rubric`, `verdict_schema`)
/// to their file contents, relative to `base_dir`. Values not starting with
/// `@` are left as-is.
fn resolve_metric_files(metric: &MetricConfig, base_dir: &Path) -> Result<MetricConfig> {
    let mut next = metric.clone();
    if let Some(value) = &next.rubric {
        next.rubric = Some(resolve_file_ref(value, base_dir)?);
    }
    if let Some(value) = &next.verdict_schema {
        next.verdict_schema = Some(resolve_file_ref(value, base_dir)?);
    }
    Ok(next)
}

/// Read the contents of a `@path` reference, or return the value unchanged if
/// it does not start with `@`.
fn resolve_file_ref(value: &str, base_dir: &Path) -> Result<String> {
    let Some(ref_path) = value.strip_prefix('@') else {
        return Ok(value.to_string());
    };
    let path = resolve_path_value(ref_path, base_dir)?;
    fs::read_to_string(&path)
        .with_context(|| format!("failed to read referenced file {}", path.display()))
}

pub fn resolve_path_value(value: &str, base_dir: &Path) -> Result<PathBuf> {
    let expanded = expand_path(value)?;
    let mut path = PathBuf::from(&expanded);
    if path.is_absolute() {
        path = path.components().collect();
        Ok(path)
    } else {
        Ok(base_dir.join(path))
    }
}

fn expand_path(value: &str) -> Result<String> {
    let home = home_dir();
    let home_utf8 = match home.as_ref() {
        Some(path) => Some(
            path.to_str()
                .ok_or_else(|| anyhow!("home directory contains invalid UTF-8"))?
                .to_string(),
        ),
        None => None,
    };

    let expanded = shellexpand::full_with_context(
        value,
        || home_utf8.as_deref(),
        |var| Ok(env::var(var).ok()),
    )
    .map_err(|error: shellexpand::LookupError<std::env::VarError>| {
        anyhow!("failed to expand '{value}': {error}")
    })?;
    Ok(expanded.into_owned())
}

fn home_dir() -> Option<PathBuf> {
    env::var_os("HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("USERPROFILE").map(PathBuf::from))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::Path;
    use std::sync::Mutex;
    use std::sync::OnceLock;
    use tempfile::TempDir;

    fn set_env_path(var: &str, value: &Path) {
        // `std::env::set_var` is marked unsafe in Rust 1.88 because it mutates
        // global process state. Tests run in isolation, so we gate the call in a
        // single helper.
        unsafe { env::set_var(var, value) };
    }

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn resolve_base_xdg_absolute_path_wins_on_any_os() {
        let got = resolve_base(
            Some(PathBuf::from("/explicit/xdg")),
            Some(PathBuf::from("/home/u")),
            Some(PathBuf::from("C:/Users/u/AppData/Roaming")),
            true,
            ".config",
        );
        assert_eq!(got, Some(PathBuf::from("/explicit/xdg")));
    }

    #[test]
    fn resolve_base_unix_incl_macos_uses_home_dotpath_not_library() {
        let got = resolve_base(None, Some(PathBuf::from("/home/u")), None, false, ".config");
        assert_eq!(got, Some(PathBuf::from("/home/u/.config")));
    }

    #[test]
    fn resolve_base_windows_uses_known_dir() {
        let got = resolve_base(
            None,
            Some(PathBuf::from("C:/Users/u")),
            Some(PathBuf::from("C:/Users/u/AppData/Roaming")),
            true,
            ".config",
        );
        assert_eq!(got, Some(PathBuf::from("C:/Users/u/AppData/Roaming")));
    }

    #[test]
    fn resolve_base_relative_xdg_is_ignored() {
        let got = resolve_base(
            Some(PathBuf::from("relative/path")),
            Some(PathBuf::from("/home/u")),
            None,
            false,
            ".local/state",
        );
        assert_eq!(got, Some(PathBuf::from("/home/u/.local/state")));
    }

    #[test]
    fn creates_config_when_missing() {
        let _guard = env_lock().lock().unwrap();
        let tmp = TempDir::new().unwrap();
        let config_home = tmp.path().join("config");
        let data_home = tmp.path().join("data");
        let state_home = tmp.path().join("state");

        set_env_path("XDG_CONFIG_HOME", &config_home);
        set_env_path("XDG_DATA_HOME", &data_home);
        set_env_path("XDG_STATE_HOME", &state_home);

        let bundle = load_or_initialize_config("xtr-test").unwrap();

        assert!(
            bundle.paths.config_file.exists(),
            "config file not created at {}",
            bundle.paths.config_file.display()
        );
        assert!(bundle.paths.data_dir.exists());
        assert!(bundle.paths.state_dir.exists());
        assert!(bundle.paths.cache_dir.exists());

        let resolved = bundle
            .config
            .resolve_task("invoice_extraction", &bundle.paths)
            .unwrap();
        assert_eq!(resolved.name, "invoice_extraction");
    }

    #[test]
    fn respects_storage_overrides() {
        let _guard = env_lock().lock().unwrap();
        let tmp = TempDir::new().unwrap();
        let config_home = tmp.path().join("config");
        let data_home = tmp.path().join("data");
        let state_home = tmp.path().join("state");

        set_env_path("XDG_CONFIG_HOME", &config_home);
        set_env_path("XDG_DATA_HOME", &data_home);
        set_env_path("XDG_STATE_HOME", &state_home);

        let app_dir = config_home.join("xtr-override");
        fs::create_dir_all(&app_dir).unwrap();
        let config_file = app_dir.join("config.toml");
        let mut file = fs::File::create(&config_file).unwrap();
        writeln!(
            file,
            r#"
                [storage]
                data_dir = "~/custom/data"
                state_dir = "~/custom/state"
                cache_dir = "~/custom/state/cache"
            "#
        )
        .unwrap();

        let bundle = load_or_initialize_config("xtr-override").unwrap();

        let expanded_home = home_dir().unwrap();
        assert_eq!(bundle.paths.data_dir, expanded_home.join("custom/data"));
        assert_eq!(bundle.paths.state_dir, expanded_home.join("custom/state"));
        assert_eq!(
            bundle.paths.cache_dir,
            expanded_home.join("custom/state/cache")
        );
    }
}
