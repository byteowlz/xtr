use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use clap::Subcommand;
use clap::ValueEnum;
use std::fs;
use std::io::Read;
use std::io::{self};
use std::path::PathBuf;
use xtr_core::ExtractionEngine;
use xtr_core::config::FeedbackModelOverrides;
use xtr_core::config::OptimizationSettings;

mod commands;
mod create;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliValidationMode {
    None,
    Warn,
    Error,
}

impl From<CliValidationMode> for xtr_core::ValidationMode {
    fn from(mode: CliValidationMode) -> Self {
        match mode {
            CliValidationMode::None => xtr_core::ValidationMode::None,
            CliValidationMode::Warn => xtr_core::ValidationMode::Warn,
            CliValidationMode::Error => xtr_core::ValidationMode::Error,
        }
    }
}

#[cfg(unix)]
fn increase_fd_limit() -> Result<()> {
    use rustix::process::Resource;
    use rustix::process::getrlimit;
    use rustix::process::setrlimit;

    let limits = getrlimit(Resource::Nofile);
    let target = limits.maximum.map_or(10240, |max| max.min(10240));
    let new = rustix::process::Rlimit {
        current: Some(target),
        maximum: limits.maximum,
    };

    if let Err(err) = setrlimit(Resource::Nofile, new) {
        eprintln!("Warning: failed to increase file descriptor limit: {err}");
    }

    Ok(())
}

#[cfg(not(unix))]
fn increase_fd_limit() -> Result<()> {
    Ok(())
}

#[derive(Parser)]
#[command(name = "xtr-cli")]
#[command(about = "Extraction and optimization CLI", long_about = None)]
#[command(arg_required_else_help = true)]
struct Cli {
    #[arg(long, short, global = true, help = "Show verbose debug output")]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    #[command(about = "Show configuration and task information")]
    Info,

    #[command(about = "Create resources (schemas, tasks, etc.)")]
    Create {
        #[command(subcommand)]
        resource: CreateCommands,
    },

    #[command(about = "Run GEPA optimization for a task")]
    Optimize {
        #[arg(help = "Task name to optimize (uses first configured task if omitted)")]
        task: Option<String>,

        #[arg(long, help = "Number of optimization iterations (default: 4)")]
        iterations: Option<u32>,

        #[arg(long, help = "Batch size for evaluation (default: 4)")]
        batch_size: Option<u32>,

        #[arg(long, help = "Number of trials per iteration (default: 6)")]
        rollouts_per_iteration: Option<u32>,

        #[arg(long, help = "Maximum total LM calls budget")]
        max_lm_calls: Option<u32>,

        #[arg(long, help = "Maximum total rollouts budget")]
        max_rollouts: Option<u32>,

        #[arg(
            long,
            help = "Temperature for LLM mutations (default: 0.9, range: 0.0-2.0)"
        )]
        temperature: Option<f32>,

        #[arg(long, help = "Track detailed optimization statistics (default: true)")]
        track_stats: Option<bool>,

        #[arg(
            long,
            help = "Track best outputs for inference-time search (default: false)"
        )]
        track_best_outputs: Option<bool>,
    },

    #[command(about = "Run inference on input text")]
    Get {
        #[arg(help = "Task name")]
        task: String,

        #[arg(long, help = "Input text to process (reads from stdin if omitted)")]
        text: Option<String>,

        #[arg(long, help = "Optional additional context")]
        context: Option<String>,

        #[arg(
            long,
            value_enum,
            default_value = "none",
            help = "Schema validation mode"
        )]
        validate: CliValidationMode,

        #[arg(
            long,
            short = 'r',
            default_value = "0",
            help = "Number of retry attempts on validation failure"
        )]
        retry: u32,

        #[arg(
            long,
            short = 's',
            help = "Use shots mode (with chat history) instead of independent retries"
        )]
        shots: bool,

        #[arg(long, short = 'm', help = "Override max_tokens for this request")]
        max_tokens: Option<u32>,
    },

    #[command(about = "Set the active optimization run for a task")]
    Activate {
        #[arg(help = "Task name")]
        task: String,

        #[arg(help = "Run path (e.g., 2025-10-28/contact_details_20251028_141425)")]
        run: String,
    },

    #[command(about = "List optimization history for a task")]
    History {
        #[arg(help = "Task name (shows all tasks if omitted)")]
        task: Option<String>,

        #[arg(long, help = "Show detailed metrics for each run")]
        detailed: bool,
    },

    #[command(about = "Compare two optimization runs")]
    Compare {
        #[arg(help = "Task name")]
        task: String,

        #[arg(help = "First run path")]
        run1: String,

        #[arg(help = "Second run path")]
        run2: String,
    },

    #[command(about = "Clean old optimization runs")]
    Clean {
        #[arg(long, help = "Delete runs older than N days (default: 30)")]
        older_than: Option<u32>,

        #[arg(long, help = "Dry run - show what would be deleted")]
        dry_run: bool,
    },

    #[command(about = "Manage task examples")]
    Examples {
        #[command(subcommand)]
        command: ExamplesCommands,
    },
}

#[derive(Subcommand)]
enum CreateCommands {
    #[command(about = "Generate a JSON schema from a JSON file")]
    Schema {
        #[arg(help = "Path to the JSON file")]
        input: String,

        #[arg(
            long,
            help = "Schema name (defaults to input filename without extension)"
        )]
        name: Option<String>,
    },
}

#[derive(Subcommand)]
enum ExamplesCommands {
    #[command(about = "Convert examples between JSON directory and JSONL formats")]
    Convert {
        #[arg(help = "Input path (directory of .json files or .jsonl file)")]
        input: PathBuf,

        #[arg(
            long,
            short,
            help = "Output path (defaults to input with changed format)"
        )]
        output: Option<PathBuf>,

        #[arg(
            long,
            help = "Output format: 'jsonl' or 'json' (auto-detected from input if omitted)"
        )]
        to: Option<String>,
    },

    #[command(about = "Generate an example template from a JSON schema")]
    Template {
        #[arg(help = "Task name or path to schema file")]
        task_or_schema: String,
    },

    #[command(about = "Generate synthetic training examples using the dataset_generator model")]
    Generate {
        #[arg(help = "Task name to generate examples for")]
        task: String,

        #[arg(
            long,
            short = 'n',
            default_value = "5",
            help = "Number of examples to generate"
        )]
        count: u32,

        #[arg(long, short, help = "Output file (JSONL format, defaults to stdout)")]
        output: Option<PathBuf>,

        #[arg(long, short, help = "Additional context/instructions for generation")]
        context: Option<String>,

        #[arg(long, help = "Append to output file instead of overwriting")]
        append: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    increase_fd_limit()?;

    let cli = Cli::parse();
    let engine = ExtractionEngine::load("xtr")?;

    match cli.command {
        Commands::Create { resource } => match resource {
            CreateCommands::Schema { input, name } => {
                create::handle_create_schema(&input, name.as_deref())?;
            }
        },
        Commands::Optimize {
            task,
            iterations,
            batch_size,
            rollouts_per_iteration,
            max_lm_calls,
            max_rollouts,
            temperature,
            track_stats,
            track_best_outputs,
        } => {
            let task_name = task
                .or_else(|| engine.config().tasks.keys().next().cloned())
                .context("no task provided and configuration has no tasks")?;

            let mut overrides = None;
            if iterations.is_some()
                || batch_size.is_some()
                || rollouts_per_iteration.is_some()
                || max_lm_calls.is_some()
                || max_rollouts.is_some()
                || temperature.is_some()
                || track_stats.is_some()
                || track_best_outputs.is_some()
            {
                overrides = Some(OptimizationSettings {
                    iterations,
                    rollouts_per_iteration,
                    max_lm_calls,
                    batch_size,
                    max_rollouts,
                    temperature,
                    track_stats,
                    track_best_outputs,
                    feedback_models: FeedbackModelOverrides::default(),
                });
            }

            println!("Running GEPA optimization for task '{task_name}'...");
            let outcome = engine
                .optimize_task_with_overrides(&task_name, overrides)
                .await?;
            println!(
                "Best score: {:.3} after {} generations ({} rollouts, {} LM calls)",
                outcome.best_score,
                outcome.total_iterations,
                outcome.total_rollouts,
                outcome.total_lm_calls
            );
            println!("Optimized instruction saved to state directory.");
        }
        Commands::Get {
            task,
            text,
            context,
            validate,
            retry,
            shots,
            max_tokens,
        } => {
            let input_text = match text {
                Some(t) => t,
                None => {
                    let mut buffer = String::new();
                    io::stdin()
                        .read_to_string(&mut buffer)
                        .context("failed to read from stdin")?;
                    buffer
                }
            };

            let validation_mode = validate.into();
            let retry_mode = if shots {
                xtr_core::RetryMode::WithHistory
            } else {
                xtr_core::RetryMode::Independent
            };

            let output = engine
                .run_inference_with_options(
                    &task,
                    &input_text,
                    context.as_deref(),
                    cli.verbose,
                    validation_mode,
                    retry,
                    retry_mode,
                    max_tokens,
                )
                .await?;

            if cli.verbose {
                println!("{output}");
            } else {
                match serde_json::from_str::<serde_json::Value>(&output) {
                    Ok(json) => println!("{}", serde_json::to_string_pretty(&json)?),
                    Err(_) => println!("{output}"),
                }
            }
        }
        Commands::Activate { task, run } => {
            commands::handle_activate(&engine, &task, &run)?;
        }
        Commands::History { task, detailed } => {
            commands::handle_history(&engine, task.as_deref(), detailed)?;
        }
        Commands::Compare { task, run1, run2 } => {
            commands::handle_compare(&engine, &task, &run1, &run2)?;
        }
        Commands::Clean {
            older_than,
            dry_run,
        } => {
            commands::handle_clean(&engine, older_than.unwrap_or(30), dry_run)?;
        }
        Commands::Info => {
            println!(
                "Loaded configuration from {}",
                engine.paths().config_file.display()
            );

            if let Some(first_task) = engine.config().tasks.keys().next() {
                println!("Configured task detected: {}", first_task);
                let task = engine.resolve_task(first_task)?;
                println!(
                    "Teacher model: {}, Student model: {} ({} fallbacks)",
                    task.models.teacher.name,
                    task.models.student.name,
                    task.models.fallbacks.len()
                );
            } else {
                println!("No tasks configured yet. Add one to run optimization.");
            }
        }
        Commands::Examples { command } => match command {
            ExamplesCommands::Convert { input, output, to } => {
                handle_examples_convert(&input, output.as_deref(), to.as_deref())?;
            }
            ExamplesCommands::Template { task_or_schema } => {
                handle_examples_template(&engine, &task_or_schema)?;
            }
            ExamplesCommands::Generate {
                task,
                count,
                output,
                context,
                append,
            } => {
                handle_examples_generate(
                    &engine,
                    &task,
                    count,
                    output.as_deref(),
                    context.as_deref(),
                    append,
                )
                .await?;
            }
        },
    }

    Ok(())
}

fn handle_examples_convert(
    input: &std::path::Path,
    output: Option<&std::path::Path>,
    to: Option<&str>,
) -> Result<()> {
    let is_jsonl_input = input.extension().and_then(|e| e.to_str()) == Some("jsonl");
    let is_dir_input = input.is_dir();

    // Determine output format
    let to_jsonl = match to {
        Some("jsonl") => true,
        Some("json") => false,
        Some(other) => anyhow::bail!("unknown format '{}', use 'json' or 'jsonl'", other),
        None => {
            // Auto-detect: if input is jsonl, output json; if input is dir, output jsonl
            if is_jsonl_input {
                false
            } else if is_dir_input {
                true
            } else {
                anyhow::bail!(
                    "cannot auto-detect output format for '{}', use --to",
                    input.display()
                );
            }
        }
    };

    if to_jsonl {
        // Convert directory -> JSONL
        if !is_dir_input {
            anyhow::bail!(
                "input '{}' is not a directory, cannot convert to JSONL",
                input.display()
            );
        }

        let output_path = output
            .map(PathBuf::from)
            .unwrap_or_else(|| input.with_extension("jsonl"));

        let count = xtr_core::convert_json_dir_to_jsonl(input, &output_path)?;
        println!(
            "Converted {} examples from '{}' to '{}'",
            count,
            input.display(),
            output_path.display()
        );
    } else {
        // Convert JSONL -> directory
        if !is_jsonl_input {
            anyhow::bail!(
                "input '{}' is not a .jsonl file, cannot convert to JSON directory",
                input.display()
            );
        }

        let output_path = output.map(PathBuf::from).unwrap_or_else(|| {
            input
                .file_stem()
                .map(|s| input.with_file_name(s))
                .unwrap_or_else(|| input.with_extension(""))
        });

        let count = xtr_core::convert_jsonl_to_json_dir(input, &output_path)?;
        println!(
            "Converted {} examples from '{}' to '{}'",
            count,
            input.display(),
            output_path.display()
        );
    }

    Ok(())
}

fn handle_examples_template(engine: &ExtractionEngine, task_or_schema: &str) -> Result<()> {
    // Check if it's a file path first
    let schema_path = PathBuf::from(task_or_schema);
    let schema: serde_json::Value = if schema_path.exists() {
        let content = fs::read_to_string(&schema_path)
            .with_context(|| format!("failed to read schema file '{}'", schema_path.display()))?;
        serde_json::from_str(&content)
            .with_context(|| format!("failed to parse schema '{}'", schema_path.display()))?
    } else {
        // Try as a task name
        let task_config = engine.resolve_task(task_or_schema)?;
        let schema_path = task_config
            .schema_path
            .as_ref()
            .context("task has no schema configured")?;
        let content = fs::read_to_string(schema_path)
            .with_context(|| format!("failed to read schema file '{}'", schema_path.display()))?;
        serde_json::from_str(&content)
            .with_context(|| format!("failed to parse schema '{}'", schema_path.display()))?
    };

    let template = xtr_core::generate_example_template(&schema);
    println!("{}", serde_json::to_string_pretty(&template)?);

    Ok(())
}

async fn handle_examples_generate(
    engine: &ExtractionEngine,
    task: &str,
    count: u32,
    output: Option<&std::path::Path>,
    context: Option<&str>,
    append: bool,
) -> Result<()> {
    use std::io::Write;

    eprintln!("Generating {count} examples for task '{task}'...");

    let examples = engine
        .generate_examples(task, count, context)
        .await
        .context("failed to generate examples")?;

    eprintln!("Generated {} examples", examples.len());

    // Serialize to JSONL
    let mut jsonl_output = String::new();
    for example in &examples {
        let line = serde_json::to_string(example)?;
        jsonl_output.push_str(&line);
        jsonl_output.push('\n');
    }

    if let Some(output_path) = output {
        let file = if append {
            fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(output_path)
                .with_context(|| {
                    format!("failed to open '{}' for appending", output_path.display())
                })?
        } else {
            fs::File::create(output_path)
                .with_context(|| format!("failed to create '{}'", output_path.display()))?
        };

        let mut writer = std::io::BufWriter::new(file);
        writer.write_all(jsonl_output.as_bytes())?;
        writer.flush()?;

        eprintln!(
            "{} {} examples to '{}'",
            if append { "Appended" } else { "Wrote" },
            examples.len(),
            output_path.display()
        );
    } else {
        // Write to stdout
        print!("{jsonl_output}");
    }

    Ok(())
}
