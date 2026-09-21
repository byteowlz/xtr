use anyhow::Result;
use std::fs;
use std::time::Duration;
use std::time::SystemTime;
use xtr_core::ExtractionEngine;

pub fn handle_activate(engine: &ExtractionEngine, task: &str, run: &str) -> Result<()> {
    let state_dir = &engine.paths().state_dir;
    let run_dir = state_dir.join("optimizations").join(run);

    if !run_dir.exists() {
        anyhow::bail!("Run directory does not exist: {}", run_dir.display());
    }

    let instruction_file = run_dir.join("optimized_instruction.txt");
    if !instruction_file.exists() {
        anyhow::bail!(
            "No optimized_instruction.txt found in {}",
            run_dir.display()
        );
    }

    let metrics_file = run_dir.join("metrics.json");
    let metrics: serde_json::Value = if metrics_file.exists() {
        serde_json::from_str(&fs::read_to_string(&metrics_file)?)?
    } else {
        serde_json::json!({})
    };

    // Copy instruction to active location
    let prompts_dir = state_dir.join("prompts").join(task);
    fs::create_dir_all(&prompts_dir)?;

    let instruction = fs::read_to_string(&instruction_file)?;
    fs::write(prompts_dir.join("optimized_instruction.txt"), &instruction)?;

    // Update metadata
    let metadata = serde_json::json!({
        "active_run": run,
        "timestamp": chrono::Local::now().to_rfc3339(),
        "baseline_score": metrics.get("baseline_score"),
        "optimized_score": metrics.get("optimized_score"),
        "improvement": metrics.get("improvement"),
    });
    fs::write(
        prompts_dir.join("metadata.json"),
        serde_json::to_string_pretty(&metadata)?,
    )?;

    println!("✓ Activated run for task '{}':", task);
    println!("  Run: {}", run);
    if let Some(score) = metrics.get("optimized_score") {
        println!("  Score: {}", score);
    }
    println!(
        "  Active prompt: {}",
        prompts_dir.join("optimized_instruction.txt").display()
    );

    Ok(())
}

pub fn handle_history(engine: &ExtractionEngine, task: Option<&str>, detailed: bool) -> Result<()> {
    let state_dir = &engine.paths().state_dir;
    let optimizations_dir = state_dir.join("optimizations");

    if !optimizations_dir.exists() {
        println!("No optimization history found.");
        return Ok(());
    }

    let mut runs = Vec::new();

    // Walk through dated directories
    for date_entry in fs::read_dir(&optimizations_dir)? {
        let date_entry = date_entry?;
        let date_path = date_entry.path();
        if !date_path.is_dir() {
            continue;
        }

        for run_entry in fs::read_dir(&date_path)? {
            let run_entry = run_entry?;
            let run_path = run_entry.path();
            if !run_path.is_dir() {
                continue;
            }

            let metrics_file = run_path.join("metrics.json");
            if !metrics_file.exists() {
                continue;
            }

            let metrics: serde_json::Value =
                serde_json::from_str(&fs::read_to_string(&metrics_file)?)?;
            let run_task = metrics
                .get("task")
                .and_then(|t| t.as_str())
                .unwrap_or("unknown");

            // Filter by task if specified
            if let Some(filter_task) = task
                && run_task != filter_task
            {
                continue;
            }

            let relative_path = format!(
                "{}/{}",
                date_path
                    .file_name()
                    .map_or_else(|| "".into(), |n| n.to_string_lossy().into_owned()),
                run_path
                    .file_name()
                    .map_or_else(|| "".into(), |n| n.to_string_lossy().into_owned())
            );

            runs.push((relative_path, run_task.to_string(), metrics));
        }
    }

    if runs.is_empty() {
        println!(
            "No optimization runs found{}",
            if task.is_some() {
                " for the specified task"
            } else {
                ""
            }
        );
        return Ok(());
    }

    // Sort by date (newest first)
    runs.sort_by(|a, b| b.0.cmp(&a.0));

    println!("\n{}", "=".repeat(80));
    println!("Optimization History");
    println!("{}", "=".repeat(80));

    for (run, run_task, metrics) in &runs {
        let baseline = metrics
            .get("baseline_score")
            .and_then(|s| s.as_f64())
            .unwrap_or(0.0);
        let optimized = metrics
            .get("optimized_score")
            .and_then(|s| s.as_f64())
            .unwrap_or(0.0);
        let improvement = metrics
            .get("improvement")
            .and_then(|s| s.as_f64())
            .unwrap_or(0.0);

        println!("\n{}", run);
        println!("  Task: {}", run_task);
        println!("  Baseline:  {:.3} ({:.1}%)", baseline, baseline * 100.0);
        println!("  Optimized: {:.3} ({:.1}%)", optimized, optimized * 100.0);
        println!(
            "  Improvement: {:+.3} ({:+.1}%)",
            improvement,
            improvement * 100.0
        );

        if detailed {
            if let Some(rollouts) = metrics.get("total_rollouts") {
                println!("  Rollouts: {}", rollouts);
            }
            if let Some(calls) = metrics.get("total_lm_calls") {
                println!("  LM calls: {}", calls);
            }
            if let Some(timestamp) = metrics.get("timestamp").and_then(|t| t.as_str()) {
                println!("  Timestamp: {}", timestamp);
            }
        }
    }

    println!("\n{}", "=".repeat(80));
    println!("Total runs: {}", runs.len());

    Ok(())
}

pub fn handle_compare(engine: &ExtractionEngine, task: &str, run1: &str, run2: &str) -> Result<()> {
    let state_dir = &engine.paths().state_dir;

    let load_run = |run: &str| -> Result<(String, serde_json::Value)> {
        let run_dir = state_dir.join("optimizations").join(run);
        if !run_dir.exists() {
            anyhow::bail!("Run directory does not exist: {}", run_dir.display());
        }

        let metrics_file = run_dir.join("metrics.json");
        if !metrics_file.exists() {
            anyhow::bail!("No metrics.json found in {}", run_dir.display());
        }

        let metrics: serde_json::Value = serde_json::from_str(&fs::read_to_string(&metrics_file)?)?;

        let instruction_file = run_dir.join("optimized_instruction.txt");
        let instruction = if instruction_file.exists() {
            fs::read_to_string(&instruction_file)?
        } else {
            String::from("<not available>")
        };

        Ok((instruction, metrics))
    };

    let (inst1, metrics1) = load_run(run1)?;
    let (inst2, metrics2) = load_run(run2)?;

    println!("\n{}", "=".repeat(80));
    println!("Comparing Optimization Runs for Task: {}", task);
    println!("{}", "=".repeat(80));

    let get_metric = |m: &serde_json::Value, key: &str| -> f64 {
        m.get(key).and_then(|v| v.as_f64()).unwrap_or(0.0)
    };

    println!("\nRun 1: {}", run1);
    println!(
        "  Baseline:  {:.3}",
        get_metric(&metrics1, "baseline_score")
    );
    println!(
        "  Optimized: {:.3}",
        get_metric(&metrics1, "optimized_score")
    );
    println!(
        "  Improvement: {:+.3}",
        get_metric(&metrics1, "improvement")
    );

    println!("\nRun 2: {}", run2);
    println!(
        "  Baseline:  {:.3}",
        get_metric(&metrics2, "baseline_score")
    );
    println!(
        "  Optimized: {:.3}",
        get_metric(&metrics2, "optimized_score")
    );
    println!(
        "  Improvement: {:+.3}",
        get_metric(&metrics2, "improvement")
    );

    let diff = get_metric(&metrics2, "optimized_score") - get_metric(&metrics1, "optimized_score");
    println!("\nDifference (Run 2 - Run 1): {:+.3}", diff);

    if diff > 0.0 {
        println!("→ Run 2 performs better");
    } else if diff < 0.0 {
        println!("→ Run 1 performs better");
    } else {
        println!("→ Runs perform equally");
    }

    println!("\n{}", "-".repeat(80));
    println!("Instruction Comparison:");
    println!("{}", "-".repeat(80));
    println!("\nRun 1 Instruction:");
    println!("{}", inst1);
    println!("\nRun 2 Instruction:");
    println!("{}", inst2);
    println!("\n{}", "=".repeat(80));

    Ok(())
}

pub fn handle_clean(engine: &ExtractionEngine, older_than_days: u32, dry_run: bool) -> Result<()> {
    let state_dir = &engine.paths().state_dir;
    let optimizations_dir = state_dir.join("optimizations");

    if !optimizations_dir.exists() {
        println!("No optimization directory found.");
        return Ok(());
    }

    let cutoff_time =
        SystemTime::now() - Duration::from_secs(older_than_days as u64 * 24 * 60 * 60);

    let mut to_delete = Vec::new();
    let mut total_size = 0u64;

    for date_entry in fs::read_dir(&optimizations_dir)? {
        let date_entry = date_entry?;
        let date_path = date_entry.path();
        if !date_path.is_dir() {
            continue;
        }

        for run_entry in fs::read_dir(&date_path)? {
            let run_entry = run_entry?;
            let run_path = run_entry.path();
            if !run_path.is_dir() {
                continue;
            }

            let metadata = fs::metadata(&run_path)?;
            if let Ok(modified) = metadata.modified()
                && modified < cutoff_time
            {
                let size = calculate_dir_size(&run_path)?;
                total_size += size;
                to_delete.push((run_path, size));
            }
        }
    }

    if to_delete.is_empty() {
        println!(
            "No optimization runs older than {} days found.",
            older_than_days
        );
        return Ok(());
    }

    println!("\n{}", "=".repeat(80));
    if dry_run {
        println!("DRY RUN - Would delete the following:");
    } else {
        println!(
            "Deleting optimization runs older than {} days:",
            older_than_days
        );
    }
    println!("{}", "=".repeat(80));

    for (path, size) in &to_delete {
        println!(
            "  {} ({:.2} MB)",
            path.display(),
            *size as f64 / 1_000_000.0
        );
    }

    println!(
        "\nTotal: {} runs, {:.2} MB",
        to_delete.len(),
        total_size as f64 / 1_000_000.0
    );

    if !dry_run {
        for (path, _) in to_delete {
            fs::remove_dir_all(&path)?;
        }
        println!("\n✓ Cleanup complete");
    } else {
        println!("\nRun without --dry-run to actually delete these runs.");
    }

    Ok(())
}

fn calculate_dir_size(path: &std::path::Path) -> Result<u64> {
    let mut size = 0;
    if path.is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                size += calculate_dir_size(&path)?;
            } else {
                size += fs::metadata(&path)?.len();
            }
        }
    }
    Ok(size)
}
