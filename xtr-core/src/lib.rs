//! Core library for the XTR structured data extraction engine.

// Tests isolate global env state (e.g. `XDG_CONFIG_HOME`) via `env::set_var`,
// which is `unsafe` in recent Rust, and idiomatically `unwrap`/`expect` on
// fixtures. These `unwrap`/`expect`/`unsafe` lints apply to production code;
// allow them only in test builds.
#![cfg_attr(
    test,
    allow(
        unsafe_code,
        clippy::unwrap_used,
        clippy::expect_used,
        reason = "test code isolates env state (unsafe env::set_var) and unwraps fixtures"
    )
)]

pub mod config;
pub mod engine;
pub mod examples;
pub mod lm;
pub mod mlflow_logger;
pub mod optimization;

pub use config::AppConfig;
pub use config::AppPaths;
pub use config::ConfigBundle;
pub use config::MetricsConfig;
pub use config::ModelDescriptor;
pub use config::OptimizationSettings;
pub use config::ResolvedModelConfig;
pub use config::ResolvedTaskConfig;
pub use config::StorageSettings;
pub use config::TaskConfig;
pub use config::load_or_initialize_config;
pub use engine::ExtractionEngine;
pub use engine::RetryMode;
pub use engine::ValidationMode;
pub use examples::RawTaskExample;
pub use examples::TaskExample;
pub use examples::convert_json_dir_to_jsonl;
pub use examples::convert_jsonl_to_json_dir;
pub use examples::generate_example_template;
pub use examples::load_task_examples;
pub use lm::AdapterKind;
pub use lm::ModelHandle;
pub use lm::TaskModelHandles;
pub use lm::build_model_handle;
pub use optimization::GepaOutcome;
pub use optimization::GepaRunner;
pub use optimization::load_best_instruction;
