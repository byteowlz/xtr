use anyhow::Result;
use dspy_rs::optimizer::GEPAResult;
use std::fs;
use std::path::PathBuf;

#[cfg(feature = "mlflow")]
use mlflow_rs::experiment::Experiment;
#[cfg(feature = "mlflow")]
use mlflow_rs::run::Run;

pub struct GepaLogger {
    task_name: String,
    #[cfg(feature = "mlflow")]
    run: Option<Run>,
    local_logging: bool,
    log_dir: Option<PathBuf>,
    run_id: String,
}

impl GepaLogger {
    pub fn new(
        task_name: String,
        #[cfg(feature = "mlflow")] mlflow_tracking_uri: Option<String>,
        #[cfg(feature = "mlflow")] experiment_name: Option<String>,
        local_logging: bool,
        log_dir: Option<PathBuf>,
    ) -> Result<Self> {
        let run_id = Self::generate_run_id();

        let log_dir = if local_logging {
            Some(log_dir.unwrap_or_else(|| {
                PathBuf::from(
                    std::env::var("XDG_STATE_HOME")
                        .ok()
                        .filter(|v| !v.is_empty())
                        .unwrap_or_else(|| {
                            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
                            format!("{home}/.local/state")
                        }),
                )
                .join("xtr")
                .join("optimization_logs")
            }))
        } else {
            None
        };

        if let Some(ref dir) = log_dir {
            fs::create_dir_all(dir)?;
        }

        #[cfg(feature = "mlflow")]
        {
            let run = if let Some(uri) = mlflow_tracking_uri {
                let exp_name = experiment_name.unwrap_or_else(|| "xtr-optimization".to_string());
                match Experiment::new(&uri, &exp_name) {
                    Ok(experiment) => match experiment.create_run(Some(&task_name), vec![]) {
                        Ok(run) => Some(run),
                        Err(e) => {
                            eprintln!("Warning: Failed to create MLflow run: {e}");
                            eprintln!("Continuing with local logging only...");
                            None
                        }
                    },
                    Err(e) => {
                        eprintln!("Warning: Failed to connect to MLflow at {uri}: {e}");
                        eprintln!("Continuing with local logging only...");
                        None
                    }
                }
            } else {
                None
            };

            Ok(Self {
                task_name,
                run,
                local_logging,
                log_dir,
                run_id,
            })
        }

        #[cfg(not(feature = "mlflow"))]
        Ok(Self {
            task_name,
            local_logging,
            log_dir,
            run_id,
        })
    }

    fn generate_run_id() -> String {
        use std::time::SystemTime;
        use std::time::UNIX_EPOCH;
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        format!("{timestamp}")
    }

    fn log_to_file(&self, name: &str, content: &str) -> Result<()> {
        if !self.local_logging {
            return Ok(());
        }

        if let Some(ref log_dir) = self.log_dir {
            let run_dir = log_dir.join(&self.task_name).join(&self.run_id);
            fs::create_dir_all(&run_dir)?;

            let file_path = run_dir.join(name);
            fs::write(&file_path, content)?;
        }

        Ok(())
    }

    pub fn log_config(&self, config: &serde_json::Value) -> Result<()> {
        #[cfg(feature = "mlflow")]
        if let Some(run) = &self.run
            && let Some(obj) = config.as_object()
        {
            for (key, value) in obj {
                let value_str = match value {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                let _ = run.log_parameter(key, &value_str);
            }
        }

        if self.local_logging {
            let config_str = serde_json::to_string_pretty(config)?;
            self.log_to_file("config.json", &config_str)?;
        }

        Ok(())
    }

    pub fn log_final_result(&self, result: &GEPAResult) -> Result<()> {
        #[cfg(feature = "mlflow")]
        if let Some(run) = &self.run {
            let _ = run.log_parameter("best_instruction", &result.best_candidate.instruction);
            let _ = run.log_metric(
                "final_best_score",
                result.best_candidate.average_score(),
                None,
            );
            let _ = run.log_metric("total_rollouts", result.total_rollouts as f32, None);
            let _ = run.log_metric("total_lm_calls", result.total_lm_calls as f32, None);
            let _ = run.log_metric("num_candidates", result.all_candidates.len() as f32, None);

            for (generation, score) in &result.evolution_history {
                let _ = run.log_metric("best_score", *score, Some(*generation as u64));
            }

            for (idx, candidate) in result.all_candidates.iter().enumerate() {
                let _ = run.log_parameter(
                    &format!("candidate_{idx}_instruction"),
                    &candidate.instruction,
                );

                let _ = run.log_metric(
                    &format!("candidate_{idx}_avg_score"),
                    candidate.average_score(),
                    None,
                );

                if let Some(parent_id) = candidate.parent_id {
                    let _ = run
                        .log_parameter(&format!("candidate_{idx}_parent"), &parent_id.to_string());
                }
            }

            let result_json = serde_json::to_string_pretty(result)?;
            let _ = run.log_artifact_bytes(result_json.into_bytes(), "gepa_result.json");
        }

        if self.local_logging {
            let result_json = serde_json::to_string_pretty(result)?;
            self.log_to_file("gepa_result.json", &result_json)?;

            let summary = format!(
                "Task: {}\n\
                 Run ID: {}\n\
                 Best Score: {:.4}\n\
                 Total Rollouts: {}\n\
                 Total LM Calls: {}\n\
                 Candidates Evaluated: {}\n\
                 \n\
                 Best Instruction:\n\
                 {}\n\
                 \n\
                 Evolution History:\n",
                self.task_name,
                self.run_id,
                result.best_candidate.average_score(),
                result.total_rollouts,
                result.total_lm_calls,
                result.all_candidates.len(),
                result.best_candidate.instruction,
            );

            let mut evolution_table = String::new();
            for (generation, score) in &result.evolution_history {
                evolution_table.push_str(&format!("  Generation {generation}: {score:.4}\n"));
            }

            self.log_to_file("summary.txt", &format!("{summary}{evolution_table}"))?;

            let mut candidates_md = String::from("# Candidates\n\n");
            for (idx, candidate) in result.all_candidates.iter().enumerate() {
                candidates_md.push_str(&format!(
                    "## Candidate {} (Generation {})\n\n\
                     Average Score: {:.4}\n\n\
                     Parent: {}\n\n\
                     Instruction:\n```\n{}\n```\n\n",
                    idx,
                    candidate.generation,
                    candidate.average_score(),
                    candidate
                        .parent_id
                        .map(|id| id.to_string())
                        .unwrap_or_else(|| "None".to_string()),
                    candidate.instruction,
                ));
            }
            self.log_to_file("candidates.md", &candidates_md)?;

            if let Some(dir) = &self.log_dir {
                eprintln!(
                    "\n✓ Local logs saved to: {}",
                    dir.join(&self.task_name).join(&self.run_id).display()
                );
            }
        }

        #[cfg(not(feature = "mlflow"))]
        if !self.local_logging {
            eprintln!(
                "[Logging Disabled] GEPA result for task '{}':",
                self.task_name
            );
            eprintln!("  Best score: {:.4}", result.best_candidate.average_score());
            eprintln!("  Total rollouts: {}", result.total_rollouts);
            eprintln!("  Total LM calls: {}", result.total_lm_calls);
            eprintln!("  Candidates evaluated: {}", result.all_candidates.len());
        }

        Ok(())
    }

    pub fn finish(&self) -> Result<()> {
        Ok(())
    }
}
