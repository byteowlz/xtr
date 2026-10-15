//! Owned JSON Schema compilation shared by extraction callers.

use anyhow::Context;
use anyhow::Result;
use jsonschema::JSONSchema;
use serde_json::Value;

/// Compile with jsonschema's default draft detection and validation options.
pub(crate) fn compile_schema(schema: &str) -> Result<JSONSchema> {
    let schema_value: Value = serde_json::from_str(schema).context("failed to parse schema")?;
    // JSONSchema owns its compiled representation; only compilation errors
    // borrow the input. Own the diagnostic before schema_value is dropped.
    JSONSchema::compile(&schema_value)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .context("failed to compile schema")
}

/// Validate independently supplied labels against the schema.
///
/// This checks schema conformance only; it does not establish label provenance
/// or grant approval. Errors own their diagnostics and outlive both inputs.
pub fn validate_labels(schema: &str, labels: &Value) -> Result<()> {
    let validator = compile_schema(schema)?;
    validator.validate(labels).map_err(|errors| {
        let diagnostics = errors
            .map(|error| {
                format!(
                    "{error} (instance path: '{}', schema path: '{}')",
                    error.instance_path, error.schema_path
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        anyhow::anyhow!("labels do not match schema: {diagnostics}")
    })
}

#[cfg(test)]
#[allow(
    unsafe_code,
    reason = "test-only GlobalAlloc instrumentation delegates unchanged pointers/layouts to System"
)]
mod tests {
    use super::compile_schema;
    use super::validate_labels;
    use std::alloc::GlobalAlloc;
    use std::alloc::Layout;
    use std::alloc::System;
    use std::cell::Cell;

    // This is the lib test binary's sole global allocator, not a production
    // allocator. Another #[global_allocator] in that binary would conflict.
    // Const TLS initialization cannot allocate; try_with tolerates TLS teardown.
    // Only the measuring thread opts in, so parallel tests are not counted.
    thread_local! {
        static LIVE_BYTES: Cell<Option<isize>> = const { Cell::new(None) };
    }

    struct CountingAllocator;

    fn record(delta: isize) {
        let _ = LIVE_BYTES.try_with(|counter| {
            if let Some(live) = counter.get() {
                counter.set(Some(live + delta));
            }
        });
    }

    // SAFETY: every operation forwards the original allocation contract to
    // System; instrumentation neither dereferences nor changes the pointers.
    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            // SAFETY: the caller supplies the GlobalAlloc layout contract.
            let ptr = unsafe { System.alloc(layout) };
            if !ptr.is_null() {
                record(layout.size() as isize);
            }
            ptr
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            // SAFETY: the caller supplies the GlobalAlloc layout contract.
            let ptr = unsafe { System.alloc_zeroed(layout) };
            if !ptr.is_null() {
                record(layout.size() as isize);
            }
            ptr
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            record(-(layout.size() as isize));
            // SAFETY: ptr/layout are forwarded unchanged to their allocator.
            unsafe { System.dealloc(ptr, layout) };
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            // SAFETY: the caller supplies a valid pointer/layout/new-size tuple.
            let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
            if !new_ptr.is_null() {
                record(new_size as isize - layout.size() as isize);
            }
            new_ptr
        }
    }

    #[global_allocator]
    static ALLOCATOR: CountingAllocator = CountingAllocator;

    struct Measurement;

    impl Measurement {
        fn start() -> Self {
            LIVE_BYTES.with(|counter| {
                assert!(
                    counter.get().is_none(),
                    "allocation measurements cannot nest"
                );
                counter.set(Some(0));
            });
            Self
        }

        fn live_bytes(&self) -> isize {
            LIVE_BYTES.with(|counter| counter.get().unwrap())
        }
    }

    impl Drop for Measurement {
        fn drop(&mut self) {
            LIVE_BYTES.with(|counter| counter.set(None));
        }
    }

    #[test]
    fn label_validation_accepts_conforming_values_and_rejects_missing_fields() {
        let schema = r#"{
            "type": "object",
            "properties": {"count": {"type": "integer", "minimum": 2}},
            "required": ["count"],
            "additionalProperties": false
        }"#;
        assert!(validate_labels(schema, &serde_json::json!({"count": 2})).is_ok());
        assert!(validate_labels(schema, &serde_json::json!({"count": 1})).is_err());
        assert!(validate_labels(schema, &serde_json::json!({})).is_err());
        assert!(validate_labels(schema, &serde_json::json!({"count": 2, "extra": true})).is_err());
        // Conformance must follow the supplied schema, not assume an object.
        assert!(validate_labels(r#"{"type":"integer"}"#, &serde_json::json!(2)).is_ok());
    }

    #[test]
    fn label_validation_propagates_schema_errors() {
        for (schema, context) in [
            ("{not json}", "failed to parse schema"),
            (r#"{"type":"invalid"}"#, "failed to compile schema"),
        ] {
            let error = validate_labels(schema, &serde_json::json!({})).unwrap_err();
            assert!(format!("{error:#}").contains(context));
        }
    }

    // Rust red/green command:
    // cargo test -p xtr-core --lib schema::tests::invalid_labels_return_all_owned_diagnostics_with_paths -- --nocapture
    // RED: compile-only stub returned Ok(()) for invalid labels; unwrap_err failed.
    // GREEN: validate_labels returns owned diagnostics for both failing fields.
    #[test]
    fn invalid_labels_return_all_owned_diagnostics_with_paths() {
        let error = {
            let schema = serde_json::json!({
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                    "name": {"type": "string", "minLength": 3},
                },
                "required": ["count", "name"],
            })
            .to_string();
            let labels = serde_json::json!({"count": "two", "name": "a"});
            validate_labels(&schema, &labels).unwrap_err()
        };

        let diagnostic = format!("{error:#}");
        assert!(diagnostic.contains("labels do not match schema"));
        assert!(diagnostic.contains("/count"));
        assert!(diagnostic.contains("/properties/count/type"));
        assert!(diagnostic.contains("integer"));
        assert!(diagnostic.contains("/name"));
        assert!(diagnostic.contains("/properties/name/minLength"));
    }

    #[test]
    fn allocation_counter_tracks_live_bytes_not_allocation_volume() {
        let measurement = Measurement::start();
        let mut bytes = std::hint::black_box(vec![0_u8; 4096]);
        assert_eq!(measurement.live_bytes(), 4096);
        bytes.reserve_exact(4096);
        assert_eq!(measurement.live_bytes(), bytes.capacity() as isize);
        drop(bytes);
        assert_eq!(measurement.live_bytes(), 0);
    }

    #[test]
    fn validator_and_local_references_outlive_source() {
        let validator = {
            let source = serde_json::json!({
                "type": "object",
                "properties": {"count": {"$ref": "#/definitions/count"}},
                "required": ["count"],
                "definitions": {"count": {"type": "integer", "minimum": 2}},
            })
            .to_string();
            compile_schema(&source).unwrap()
        };

        assert!(validator.is_valid(&serde_json::json!({"count": 2})));
        assert!(!validator.is_valid(&serde_json::json!({"count": 1})));
        assert!(!validator.is_valid(&serde_json::json!({"count": "2"})));
        assert!(!validator.is_valid(&serde_json::json!({})));
    }

    #[test]
    fn invalid_json_returns_parse_context() {
        let error = compile_schema("{not json}").unwrap_err();
        let diagnostic = format!("{error:#}");
        assert!(diagnostic.contains("failed to parse schema"));
        assert!(diagnostic.contains("key must be a string"));
    }

    #[test]
    fn invalid_schema_error_outlives_source() {
        let error = {
            let source = String::from(r#"{"type":"not-a-json-schema-type"}"#);
            compile_schema(&source).unwrap_err()
        };
        let diagnostic = format!("{error:#}");
        assert!(diagnostic.contains("failed to compile schema"));
        assert!(diagnostic.contains("not-a-json-schema-type"));
    }

    #[test]
    fn changed_schema_compiles_independently() {
        let mut source = String::from(r#"{"type":"integer","minimum":2}"#);
        let original = compile_schema(&source).unwrap();
        source.clear();
        source.push_str(r#"{"type":"string","minLength":3}"#);
        let changed = compile_schema(&source).unwrap();
        drop(source);

        assert!(original.is_valid(&serde_json::json!(2)));
        assert!(!changed.is_valid(&serde_json::json!(2)));
        assert!(changed.is_valid(&serde_json::json!("abc")));
        assert!(!changed.is_valid(&serde_json::json!("ab")));
        assert!(!original.is_valid(&serde_json::json!("abc")));
    }

    #[test]
    fn default_options_detect_declared_draft_and_validate_schema() {
        // Draft 4 uses a boolean exclusiveMinimum; Draft 7 uses a number.
        // Selecting a fixed draft instead of the default would change this.
        let validator = compile_schema(
            r#"{
                "$schema": "http://json-schema.org/draft-04/schema#",
                "type": "number",
                "minimum": 2,
                "exclusiveMinimum": true
            }"#,
        )
        .unwrap();
        assert!(!validator.is_valid(&serde_json::json!(2)));
        assert!(validator.is_valid(&serde_json::json!(3)));
        assert!(compile_schema(r#"{"type":"number","minimum":"invalid"}"#).is_err());
    }

    #[test]
    fn repeated_large_schema_compilation_releases_live_allocations() {
        const DESCRIPTION_BYTES: usize = 1024 * 1024;
        const REPETITIONS: usize = 8;
        const RETAINED_BUDGET: isize = 64 * 1024;

        // Rust red/green evidence, measured with the command below (not RSS):
        // cargo test -p xtr-core --lib schema::tests -- --nocapture
        // RED: exact engine Box::leak(Box::new(schema_value)) compile path:
        // [1049025, 2098050, 3147075, 4196100, 5245125, 6294150, 7343175, 8392200]
        // GREEN: owned compile_schema: [0, 0, 0, 0, 0, 0, 0, 0].
        // Counts are observed bytes on this platform, not portable assertions.
        //
        // Synthetic annotation: large enough to distinguish a leaked input tree
        // from small one-time caches. No providers, files, or remote references.
        let schema = serde_json::json!({
            "type": "object",
            "description": "x".repeat(DESCRIPTION_BYTES),
        })
        .to_string();

        // Initialize jsonschema's lazy draft/meta-schema caches outside the
        // window. The source string also stays alive throughout the window, so
        // its pre-existing allocation is neither added nor subtracted.
        drop(compile_schema(&schema).unwrap());
        let mut retained = [0; REPETITIONS];
        {
            let measurement = Measurement::start();
            for live in &mut retained {
                drop(compile_schema(&schema).unwrap());
                *live = measurement.live_bytes();
            }
        }
        // Print/assert only after disabling tracking (including on unwind).
        eprintln!(
            "description_bytes={DESCRIPTION_BYTES}, repetitions={REPETITIONS}, retained_live_bytes={retained:?}"
        );
        for live in retained {
            assert!(
                (0..=RETAINED_BUDGET).contains(&live),
                "retained {live} live bytes after dropping validators; budget {RETAINED_BUDGET}"
            );
        }
    }
}
