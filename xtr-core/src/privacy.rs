//! Request-owned known-credential checks. Never mutate extraction literals.
use crate::config::AppConfig;
use serde_json::Value;

/// Deliberately neither Debug nor Serialize: inventories themselves are private.
pub(crate) struct CredentialInventory {
    values: Vec<String>,
}

impl CredentialInventory {
    pub(crate) fn from_config(config: &AppConfig) -> Self {
        let defaults = &config.models.defaults;
        let mut specs = vec![&defaults.student, &defaults.teacher];
        specs.extend(defaults.fallbacks.iter());
        specs.extend(defaults.dataset_generator.iter());
        for overrides in config.models.tasks.values() {
            specs.extend(overrides.student.iter());
            specs.extend(overrides.teacher.iter());
            specs.extend(overrides.fallbacks.iter());
        }
        let optimization = std::iter::once(&config.optimization.defaults)
            .chain(config.optimization.tasks.values());
        for settings in optimization {
            specs.extend(settings.feedback_models.student.iter());
            specs.extend(settings.feedback_models.teacher.iter());
        }
        let mut values: Vec<String> = specs
            .into_iter()
            .filter_map(|spec| spec.api_key.as_ref())
            .filter(|key| !key.is_empty())
            .cloned()
            .collect();
        values.sort();
        values.dedup();
        Self { values }
    }

    pub(crate) fn contains_text(&self, text: &str) -> bool {
        self.values.iter().any(|key| text.contains(key))
    }

    /// Include decoded values AND object keys: JSON escaping must not bypass
    /// a guard and later materialize a credential in parsed output/diagnostics.
    pub(crate) fn contains_json(&self, value: &Value) -> bool {
        match value {
            Value::String(text) => self.contains_text(text),
            Value::Array(values) => values.iter().any(|value| self.contains_json(value)),
            Value::Object(map) => map
                .iter()
                .any(|(key, value)| self.contains_text(key) || self.contains_json(value)),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelSpec;
    use serde_json::json;

    #[test]
    fn decoded_keys_values_and_all_configured_locations_are_checked() {
        let mut config = AppConfig::default();
        config.models.defaults.student.api_key = Some("SYNTHETIC-student-key".into());
        config.models.defaults.dataset_generator = Some(ModelSpec {
            api_key: Some("SYNTHETIC-generator-key".into()),
            ..Default::default()
        });
        config.optimization.defaults.feedback_models.teacher = Some(ModelSpec {
            api_key: Some("SYNTHETIC-feedback-key".into()),
            ..Default::default()
        });
        let inventory = CredentialInventory::from_config(&config);
        let escaped: Value = serde_json::from_str(r#"{"\u0053YNTHETIC-student-key":"safe"}"#)
            .expect("synthetic JSON");
        assert!(inventory.contains_json(&escaped));
        for key in [
            "SYNTHETIC-student-key",
            "SYNTHETIC-generator-key",
            "SYNTHETIC-feedback-key",
        ] {
            assert!(inventory.contains_text(key));
            assert!(inventory.contains_json(&json!([{"nested": key}])));
        }
        assert!(!inventory.contains_json(&json!({"safe": [true, null, 1]})));
    }
}
