use std::fs;
use std::io;
use std::io::BufRead;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use serde::Deserialize;
use serde::Serialize;

use crate::media::ImageInput;

/// Representation of a single training example loaded from a `.json` file.
#[derive(Debug, Clone)]
pub struct TaskExample {
    pub name: String,
    pub input_text: String,
    pub expected_json: serde_json::Value,
    pub additional_context: Option<String>,
    pub image_paths: Vec<PathBuf>,
}

impl TaskExample {
    /// Preserve text and context while encoding every image as an input data URI.
    /// Unreadable or unsupported images fail conversion rather than being omitted.
    pub fn to_training_example(
        &self,
        schema: &str,
        include_timestamp: bool,
    ) -> Result<dspy_rs::Example> {
        let expected_str = serde_json::to_string(&self.expected_json)?;
        let image_data_uris: Vec<String> = self
            .image_paths
            .iter()
            .map(|path| ImageInput::File(path.clone()).to_data_uri())
            .collect::<Result<_>>()?;

        let mut context_parts = Vec::new();
        if let Some(ctx) = &self.additional_context
            && !ctx.trim().is_empty()
        {
            context_parts.push(ctx.clone());
        }

        if include_timestamp {
            let now = chrono::Local::now();
            let timestamp = now.format("%A, %B %d, %Y at %I:%M %p %Z").to_string();
            context_parts.push(format!("Current date and time: {timestamp}"));
        }

        let final_context = context_parts.join("\n\n");

        let example = dspy_rs::example! {
            "schema": "input" => schema,
            "input_text": "input" => &self.input_text,
            "additional_context": "input" => final_context.as_str(),
            "image_data_uris": "input" => image_data_uris,
            "expected_output": "input" => expected_str,
            "example_name": "input" => &self.name
        };
        Ok(example)
    }
}

/// Raw example format for serialization/deserialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawTaskExample {
    pub input_text: String,
    pub expected_json: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub additional_context: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<String>,
}

/// Load task examples from a path.
///
/// The path can be either:
/// - A directory containing `.json` files (one example per file)
/// - A `.jsonl` file (one example per line)
pub fn load_task_examples(path: &Path) -> Result<Vec<TaskExample>> {
    if !path.exists() {
        return Err(anyhow::anyhow!(
            "examples path '{}' does not exist",
            path.display()
        ));
    }

    if path.is_file() {
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("jsonl") => load_examples_from_jsonl(path),
            Some("json") => {
                // Single JSON file - could be an array of examples or a single example
                load_examples_from_json_file(path)
            }
            _ => Err(anyhow::anyhow!(
                "unsupported example file format: {}",
                path.display()
            )),
        }
    } else {
        load_examples_from_directory(path)
    }
}

/// Load examples from a directory of `.json` files.
fn load_examples_from_directory(dir: &Path) -> Result<Vec<TaskExample>> {
    let mut entries = Vec::new();

    for entry in fs::read_dir(dir)
        .with_context(|| format!("unable to read examples directory {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }

        let content = fs::read_to_string(&path)
            .with_context(|| format!("failed to read example file '{}'", path.display()))?;

        let raw: RawTaskExample = serde_json::from_str(&content)
            .with_context(|| format!("failed to parse example json '{}'", path.display()))?;

        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("example")
            .to_string();

        entries.push(raw_to_task_example(raw, name, &path)?);
    }

    entries.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(entries)
}

/// Load examples from a JSONL file (one JSON object per line).
fn load_examples_from_jsonl(path: &Path) -> Result<Vec<TaskExample>> {
    let file = fs::File::open(path)
        .with_context(|| format!("failed to open JSONL file '{}'", path.display()))?;
    let reader = io::BufReader::new(file);

    let mut entries = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.with_context(|| {
            format!(
                "failed to read line {} of '{}'",
                line_num + 1,
                path.display()
            )
        })?;

        // Skip empty lines
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let raw: RawTaskExample = serde_json::from_str(trimmed).with_context(|| {
            format!(
                "failed to parse JSON at line {} of '{}'",
                line_num + 1,
                path.display()
            )
        })?;

        let name = format!("example_{}", entries.len() + 1);
        entries.push(raw_to_task_example(raw, name, path)?);
    }

    Ok(entries)
}

/// Load examples from a single JSON file (array of examples).
fn load_examples_from_json_file(path: &Path) -> Result<Vec<TaskExample>> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read example file '{}'", path.display()))?;

    // Try parsing as an array first
    if let Ok(examples) = serde_json::from_str::<Vec<RawTaskExample>>(&content) {
        let mut entries = Vec::new();
        for (i, raw) in examples.into_iter().enumerate() {
            let name = format!("example_{}", i + 1);
            entries.push(raw_to_task_example(raw, name, path)?);
        }
        return Ok(entries);
    }

    // Fall back to single example
    let raw: RawTaskExample = serde_json::from_str(&content)
        .with_context(|| format!("failed to parse example json '{}'", path.display()))?;

    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("example")
        .to_string();

    Ok(vec![raw_to_task_example(raw, name, path)?])
}

/// Convert a raw example to a TaskExample, resolving image paths.
fn raw_to_task_example(
    raw: RawTaskExample,
    name: String,
    source_path: &Path,
) -> Result<TaskExample> {
    let mut image_paths = Vec::with_capacity(raw.images.len());
    for relative in &raw.images {
        let resolved = resolve_example_path(source_path, relative)?;
        image_paths.push(resolved);
    }

    Ok(TaskExample {
        name,
        input_text: raw.input_text,
        expected_json: raw.expected_json,
        additional_context: raw.additional_context,
        image_paths,
    })
}

fn resolve_example_path(example_file: &Path, relative: &str) -> Result<PathBuf> {
    let base = example_file
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "example file has no parent"))?;
    let expanded = shellexpand::full_with_context(
        relative,
        || None::<&str>,
        |var| Ok::<Option<String>, std::convert::Infallible>(std::env::var(var).ok()),
    )
    .with_context(|| format!("failed to expand path '{relative}'"))?;
    let candidate = PathBuf::from(expanded.as_ref());
    Ok(if candidate.is_absolute() {
        candidate
    } else {
        base.join(candidate)
    })
}

/// Convert a directory of JSON examples to a JSONL file.
pub fn convert_json_dir_to_jsonl(dir: &Path, output: &Path) -> Result<usize> {
    let examples = load_examples_from_directory(dir)?;
    let file = fs::File::create(output)
        .with_context(|| format!("failed to create JSONL file '{}'", output.display()))?;
    let mut writer = io::BufWriter::new(file);

    for example in &examples {
        let raw = RawTaskExample {
            input_text: example.input_text.clone(),
            expected_json: example.expected_json.clone(),
            additional_context: example.additional_context.clone(),
            images: example
                .image_paths
                .iter()
                .map(|p| p.to_string_lossy().to_string())
                .collect(),
        };
        serde_json::to_writer(&mut writer, &raw)?;
        writeln!(writer)?;
    }

    Ok(examples.len())
}

/// Convert a JSONL file to a directory of JSON files.
pub fn convert_jsonl_to_json_dir(jsonl: &Path, output_dir: &Path) -> Result<usize> {
    let examples = load_examples_from_jsonl(jsonl)?;

    fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create directory '{}'", output_dir.display()))?;

    for (i, example) in examples.iter().enumerate() {
        let raw = RawTaskExample {
            input_text: example.input_text.clone(),
            expected_json: example.expected_json.clone(),
            additional_context: example.additional_context.clone(),
            images: example
                .image_paths
                .iter()
                .map(|p| p.to_string_lossy().to_string())
                .collect(),
        };

        let filename = format!("example_{:03}.json", i + 1);
        let filepath = output_dir.join(&filename);
        let content = serde_json::to_string_pretty(&raw)?;
        fs::write(&filepath, content)
            .with_context(|| format!("failed to write '{}'", filepath.display()))?;
    }

    Ok(examples.len())
}

/// Generate an empty example template from a JSON schema.
pub fn generate_example_template(schema: &serde_json::Value) -> RawTaskExample {
    let expected_json = generate_template_from_schema(schema);
    RawTaskExample {
        input_text: "<input text here>".to_string(),
        expected_json,
        additional_context: None,
        images: Vec::new(),
    }
}

/// Recursively generate a template JSON value from a schema.
fn generate_template_from_schema(schema: &serde_json::Value) -> serde_json::Value {
    use serde_json::Value;

    let schema_type = schema.get("type").and_then(|t| t.as_str());

    match schema_type {
        Some("object") => {
            let mut obj = serde_json::Map::new();
            if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
                for (key, prop_schema) in props {
                    obj.insert(key.clone(), generate_template_from_schema(prop_schema));
                }
            }
            Value::Object(obj)
        }
        Some("array") => {
            let items_schema = schema.get("items").unwrap_or(&Value::Null);
            Value::Array(vec![generate_template_from_schema(items_schema)])
        }
        Some("string") => {
            // Check for enum values
            if let Some(enum_values) = schema.get("enum").and_then(|e| e.as_array())
                && let Some(first) = enum_values.first()
            {
                return first.clone();
            }
            Value::String("<string>".to_string())
        }
        Some("number") | Some("integer") => Value::Number(0.into()),
        Some("boolean") => Value::Bool(false),
        Some("null") => Value::Null,
        _ => Value::Null,
    }
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests use assertions while propagating fixture and conversion errors"
)]
mod training_tests {
    use super::*;
    use base64::Engine as _;
    use base64::engine::general_purpose::STANDARD;
    use serde_json::Value;
    use serde_json::json;

    fn task_example(image_paths: Vec<PathBuf>) -> TaskExample {
        TaskExample {
            name: "synthetic-example".to_string(),
            input_text: "Read the images without losing this text.".to_string(),
            expected_json: json!({ "answer": "synthetic" }),
            additional_context: Some("Keep this context too.".to_string()),
            image_paths,
        }
    }

    #[test]
    fn training_images_preserve_bytes_order_and_text_fields() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let fixtures = [
            ("image/png", b"\x89PNG\r\n\x1a\n\x00\xfffirst".as_slice()),
            ("image/gif", b"GIF89a\x00\x80second".as_slice()),
        ];
        let mut paths = Vec::new();
        for (index, (_, bytes)) in fixtures.iter().enumerate() {
            let path = dir.path().join(format!("image-{index}.bin"));
            fs::write(&path, bytes)?;
            paths.push(path);
        }
        let task = task_example(paths);
        let schema = r#"{"type":"object"}"#;
        let example = task.to_training_example(schema, false)?;
        let uris = example.get("image_data_uris", None);
        let uris = uris.as_array().context("Images must be a JSON array")?;
        assert_eq!(uris.len(), fixtures.len());
        for (uri, (mime, bytes)) in uris.iter().zip(fixtures) {
            let uri = uri.as_str().context("Image URI must be a string")?;
            let prefix = format!("data:{mime};base64,");
            let payload = uri.strip_prefix(&prefix).context("Incorrect image MIME")?;
            assert_eq!(STANDARD.decode(payload)?, bytes);
        }
        assert!(
            example
                .input_keys
                .iter()
                .any(|key| key == "image_data_uris")
        );
        assert_eq!(example.get("schema", None), json!(schema));
        assert_eq!(example.get("input_text", None), json!(task.input_text));
        assert_eq!(
            example.get("additional_context", None),
            json!(task.additional_context)
        );
        assert_eq!(example.get("example_name", None), json!(task.name));
        let expected = example.get("expected_output", None);
        let expected = expected
            .as_str()
            .context("Expected output must be a string")?;
        assert_eq!(serde_json::from_str::<Value>(expected)?, task.expected_json);
        Ok(())
    }

    #[test]
    fn text_only_training_keeps_empty_image_array() -> Result<()> {
        let mut task = task_example(Vec::new());
        task.additional_context = None;
        let example = task.to_training_example("schema", false)?;
        assert_eq!(example.get("image_data_uris", None), json!([]));
        assert_eq!(example.get("input_text", None), json!(task.input_text));
        assert_eq!(example.get("additional_context", None), json!(""));
        Ok(())
    }

    #[test]
    fn timestamp_does_not_replace_context_or_images() -> Result<()> {
        let file = tempfile::NamedTempFile::new()?;
        fs::write(file.path(), b"GIF89a")?;
        let task = task_example(vec![file.path().to_path_buf()]);
        let example = task.to_training_example("schema", true)?;
        let context = example.get("additional_context", None);
        let context = context.as_str().context("Context must be a string")?;
        assert!(context.starts_with("Keep this context too.\n\nCurrent date and time: "));
        assert_eq!(example.get("input_text", None), json!(task.input_text));
        assert_eq!(
            example.get("image_data_uris", None),
            json!(["data:image/gif;base64,R0lGODlh"])
        );
        Ok(())
    }

    #[test]
    fn missing_image_fails_even_after_valid_image() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let valid = dir.path().join("valid.gif");
        fs::write(&valid, b"GIF89a")?;
        let missing = dir.path().join("missing.png");
        let task = task_example(vec![valid, missing]);
        assert!(task.to_training_example("schema", false).is_err());
        Ok(())
    }

    #[test]
    fn unsupported_or_empty_image_is_not_silently_omitted() -> Result<()> {
        let file = tempfile::NamedTempFile::new()?;
        for bytes in [b"not an image".as_slice(), b""] {
            fs::write(file.path(), bytes)?;
            let task = task_example(vec![file.path().to_path_buf()]);
            assert!(task.to_training_example("schema", false).is_err());
        }
        Ok(())
    }
}
