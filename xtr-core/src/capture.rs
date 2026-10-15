//! Private candidate observations, deliberately separate from supervised examples.
//!
//! Callers must gate capture on an explicit collection policy and serialize the
//! extraction receipt without normalizing its raw completion or provenance. This
//! module does not invoke providers, collect source media, or select attempts.
//! Failures are returned to the caller, which owns the extraction/capture failure
//! policy. Existing records are never rewritten or permission-repaired.
//!
//! Publication is an atomic, non-clobbering link/rename of a fully written and
//! synced temporary file in the destination directory. Ordinary failures clean
//! up the temporary file; process termination can leave a `.tmp` file, never a
//! partial final `.json` record. Directory-entry durability across power loss is
//! not promised. Unix directories/files are private (0700/0600, subject to umask);
//! other platforms need an owner-approved private location/ACL policy.
//!
//! Paths must not contain symlink components. Existing capture base/task
//! directories must already be private on Unix. Ancestors outside the capture
//! base may be shared. As with `tempfile` path-based persistence, callers must
//! trust the directory tree against concurrent replacement, including through
//! writable ancestors; rejecting existing symlinks is not a defense against
//! malicious directory-renaming races.

use std::fs;
use std::io::Write;
use std::path::Component;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use tempfile::NamedTempFile;

/// Qualification of a captured observation, not a training label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureQualification {
    Unreviewed,
}

/// An observation with opaque, lossless extraction-receipt JSON.
///
/// There is intentionally no `expected_json`: capture cannot create gold labels.
/// Keeping the receipt opaque avoids coupling publication to inference types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureRecord {
    pub input_text: String,
    pub additional_context: Option<String>,
    pub receipt: Value,
    pub qualification: CaptureQualification,
}

/// Publish a candidate under `base_dir/task` without overwriting any record.
///
/// Task names are single ASCII components containing letters, digits, `_`, `-`
/// or `.` (but not `.` or `..`). Random filenames do not depend on the clock.
/// Calling this function opts into capture; it has no implicit enabled/default
/// policy. No directory is created until the task name has been checked.
pub fn capture_candidate(base_dir: &Path, task: &str, record: &CaptureRecord) -> Result<PathBuf> {
    ensure!(
        !task.is_empty()
            && task != "."
            && task != ".."
            && task
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"_-.".contains(&byte)),
        "capture task must be a single safe ASCII path component"
    );
    let value = serde_json::to_value(record).context("serialize capture record")?;
    prepare_base(base_dir)?;
    write_private_json(&base_dir.join(task), &value)
}

/// Atomically publish JSON in a private directory without overwriting any file.
///
/// Shared storage primitive for candidate capture and request-local attempt logs.
/// Creates missing directories privately, rejects symlink path components and a
/// non-private existing `dir`, and returns only after syncing the complete file
/// and publishing it under a unique `.json` name. The module-level platform,
/// cleanup, durability and trusted-directory limitations apply here as well.
/// Collection policy and the supplied JSON's meaning remain caller-owned.
pub fn write_private_json(dir: &Path, value: &Value) -> Result<PathBuf> {
    prepare_base(dir)?;
    let temporary = new_temporary(dir)?;
    let destination = temporary.path().with_extension("json");
    write_and_publish(temporary, &destination, |file| {
        serde_json::to_writer_pretty(&mut *file, value).context("serialize private JSON record")?;
        file.write_all(b"\n")
            .context("finish private JSON record")?;
        Ok(())
    })
}

/// Explicitly construct a supervised example using separately supplied labels.
///
/// A nonempty reviewer is required, but an identity string is not proof of
/// independent review. The caller must enforce owner-approved independent review,
/// schema validation of the supplied labels, and storage in a separate approved
/// output. This function neither reads predictions as gold nor modifies/promotes
/// the candidate in place. Labels may legitimately equal predictions after review.
/// Approval provenance/audit storage remains the caller's responsibility because
/// `RawTaskExample` has no reviewer field.
pub fn approve_candidate(
    record: &CaptureRecord,
    expected_json: Value,
    reviewer: &str,
) -> Result<crate::examples::RawTaskExample> {
    ensure!(
        !reviewer.trim().is_empty(),
        "candidate approval requires a reviewer"
    );
    Ok(crate::examples::RawTaskExample {
        input_text: record.input_text.clone(),
        expected_json,
        additional_context: record.additional_context.clone(),
        images: Vec::new(),
    })
}

fn prepare_base(base_dir: &Path) -> Result<()> {
    let absolute = if base_dir.is_absolute() {
        base_dir.to_path_buf()
    } else {
        std::env::current_dir()
            .context("resolve capture base")?
            .join(base_dir)
    };
    let mut path = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::ParentDir => {
                anyhow::bail!("capture base must not contain parent components")
            }
            Component::CurDir => continue,
            Component::Prefix(_) | Component::RootDir => path.push(component.as_os_str()),
            Component::Normal(_) => {
                path.push(component.as_os_str());
                ensure_directory(&path, false)?;
            }
        }
    }
    ensure_directory(&absolute, true)
}

fn ensure_directory(path: &Path, require_private: bool) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let mut builder = fs::DirBuilder::new();
            #[cfg(unix)]
            {
                use std::os::unix::fs::DirBuilderExt;
                builder.mode(0o700);
            }
            match builder.create(path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error).context("create private capture directory"),
            }
        }
        Err(error) => return Err(error).context("inspect capture directory"),
    }
    let metadata =
        fs::symlink_metadata(path).context("inspect capture directory after creation")?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "capture path must be a real directory: {}",
        path.display()
    );
    if require_private {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            ensure!(
                metadata.permissions().mode() & 0o077 == 0,
                "capture directory is not private: {}",
                path.display()
            );
        }
    }
    Ok(())
}

fn new_temporary(directory: &Path) -> Result<NamedTempFile> {
    let temporary = tempfile::Builder::new()
        .prefix("candidate-")
        .suffix(".tmp")
        .rand_bytes(16)
        .tempfile_in(directory)
        .context("create capture temporary file")?;
    // NamedTempFile uses 0600 on Unix. Check rather than chmod even the new file.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        ensure!(
            temporary.as_file().metadata()?.permissions().mode() & 0o077 == 0,
            "capture temporary file is not private"
        );
    }
    Ok(temporary)
}

fn write_and_publish(
    mut temporary: NamedTempFile,
    destination: &Path,
    write: impl FnOnce(&mut fs::File) -> Result<()>,
) -> Result<PathBuf> {
    write(temporary.as_file_mut())?;
    temporary
        .as_file()
        .sync_all()
        .context("sync capture record")?;
    temporary
        .persist_noclobber(destination)
        .map_err(|error| error.error)
        .context("publish capture record without clobbering")?;
    Ok(destination.to_path_buf())
}

#[cfg(test)]
#[expect(
    clippy::panic_in_result_fn,
    reason = "test assertions report mismatches while fixture IO errors propagate"
)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Arc;
    use std::sync::Barrier;

    fn fixture(id: usize) -> CaptureRecord {
        CaptureRecord {
            input_text: format!("synthetic input {id}"),
            additional_context: Some("synthetic context".to_owned()),
            receipt: json!({
                "parsed": {"id": id},
                "raw_completion": format!("  {{\"id\":{id}}}\r\n"),
                "attempts": [
                    {"model": "synthetic-primary", "termination": "failed"},
                    {"model": "synthetic-fallback", "termination": "complete"}
                ],
                "validation": "valid"
            }),
            qualification: CaptureQualification::Unreviewed,
        }
    }

    // macOS's system temporary path may itself contain /var -> /private/var.
    fn private_base() -> Result<(tempfile::TempDir, PathBuf)> {
        let temporary = tempfile::tempdir()?;
        let base = temporary.path().canonicalize()?.join("captures");
        Ok((temporary, base))
    }

    fn entries(directory: &Path) -> Result<Vec<PathBuf>> {
        fs::read_dir(directory)?
            .map(|entry| entry.map(|entry| entry.path()).map_err(anyhow::Error::from))
            .collect()
    }

    #[test]
    fn private_json_preserves_arbitrary_values_without_clobbering() -> Result<()> {
        let (_temporary, base) = private_base()?;
        let values = [
            json!({"attempt": 1, "raw_completion": "  synthetic\r\n", "validation": "rejected"}),
            json!([null, "synthetic", {"nested": true}]),
            Value::Null,
        ];
        let mut paths = std::collections::HashSet::new();
        for value in &values {
            let path = write_private_json(&base, value)?;
            assert!(paths.insert(path.clone()));
            assert_eq!(
                path.extension().and_then(|extension| extension.to_str()),
                Some("json")
            );
            let decoded: Value = serde_json::from_slice(&fs::read(path)?)?;
            assert_eq!(&decoded, value);
        }
        assert_eq!(entries(&base)?.len(), values.len());
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn private_json_directly_enforces_directory_and_file_privacy() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;
        use std::os::unix::fs::symlink;
        let (_temporary, base) = private_base()?;
        let path = write_private_json(&base, &json!({"synthetic": true}))?;
        let original = fs::read(&path)?;
        assert_eq!(fs::metadata(&base)?.permissions().mode() & 0o777, 0o700);
        assert_eq!(fs::metadata(&path)?.permissions().mode() & 0o777, 0o600);
        fs::set_permissions(&base, fs::Permissions::from_mode(0o755))?;
        assert!(write_private_json(&base, &Value::Null).is_err());
        assert_eq!(fs::metadata(&base)?.permissions().mode() & 0o777, 0o755);
        assert_eq!(fs::read(path)?, original);
        assert_eq!(entries(&base)?.len(), 1);
        let linked = base.with_extension("linked");
        symlink(&base, &linked)?;
        assert!(write_private_json(&linked, &Value::Null).is_err());
        assert_eq!(entries(&base)?.len(), 1);
        Ok(())
    }

    #[test]
    fn receipt_round_trips_without_gold_or_normalization() -> Result<()> {
        let (_temporary, base) = private_base()?;
        let record = fixture(7);
        let path = capture_candidate(&base, "synthetic", &record)?;
        let bytes = fs::read(path)?;
        let decoded: CaptureRecord = serde_json::from_slice(&bytes)?;
        assert_eq!(decoded, record);
        let value: Value = serde_json::from_slice(&bytes)?;
        assert_eq!(value["qualification"], "unreviewed");
        assert!(value.get("expected_json").is_none());
        assert!(serde_json::from_slice::<crate::examples::RawTaskExample>(&bytes).is_err());
        assert_eq!(entries(&base.join("synthetic"))?.len(), 1);
        Ok(())
    }

    #[test]
    fn simultaneous_captures_keep_every_input_receipt_pair() -> Result<()> {
        let (_temporary, base) = private_base()?;
        let count = 24;
        let barrier = Arc::new(Barrier::new(count));
        let mut workers = Vec::new();
        for id in 0..count {
            let barrier = Arc::clone(&barrier);
            let base = base.clone();
            workers.push(std::thread::spawn(move || {
                barrier.wait();
                capture_candidate(&base, "synthetic", &fixture(id))
            }));
        }
        let mut paths = std::collections::HashSet::new();
        for (id, worker) in workers.into_iter().enumerate() {
            let path = worker
                .join()
                .map_err(|_| anyhow::anyhow!("capture thread failed"))??;
            assert!(paths.insert(path.clone()));
            let decoded: CaptureRecord = serde_json::from_slice(&fs::read(path)?)?;
            assert_eq!(decoded, fixture(id));
        }
        assert_eq!(entries(&base.join("synthetic"))?.len(), count);
        Ok(())
    }

    #[test]
    fn partial_write_failure_leaves_no_final_or_temporary_record() -> Result<()> {
        let (_temporary, base) = private_base()?;
        let existing = capture_candidate(&base, "synthetic", &fixture(1))?;
        let original = fs::read(&existing)?;
        let directory = base.join("synthetic");
        let temporary = new_temporary(&directory)?;
        let destination = temporary.path().with_extension("json");
        let result = write_and_publish(temporary, &destination, |file| {
            file.write_all(b"{\"input_text\":\"partial")?;
            anyhow::bail!("synthetic interrupted write")
        });
        assert!(result.is_err());
        assert!(!destination.exists());
        assert_eq!(fs::read(existing)?, original);
        assert_eq!(entries(&directory)?.len(), 1);
        Ok(())
    }

    #[test]
    fn publication_collision_preserves_existing_bytes_and_cleans_temp() -> Result<()> {
        let (_temporary, base) = private_base()?;
        let existing = capture_candidate(&base, "synthetic", &fixture(1))?;
        let original = fs::read(&existing)?;
        let directory = base.join("synthetic");
        let temporary = new_temporary(&directory)?;
        assert!(
            write_and_publish(temporary, &existing, |file| {
                file.write_all(b"different complete content")?;
                Ok(())
            })
            .is_err()
        );
        assert_eq!(fs::read(existing)?, original);
        assert_eq!(entries(&directory)?.len(), 1);
        Ok(())
    }

    #[test]
    fn failed_publication_to_directory_cleans_temp() -> Result<()> {
        let (_temporary, base) = private_base()?;
        capture_candidate(&base, "synthetic", &fixture(1))?;
        let directory = base.join("synthetic");
        let blocker = directory.join("blocked.json");
        fs::create_dir(&blocker)?;
        let temporary = new_temporary(&directory)?;
        assert!(
            write_and_publish(temporary, &blocker, |file| {
                file.write_all(b"{}")?;
                Ok(())
            })
            .is_err()
        );
        assert!(blocker.is_dir());
        assert_eq!(entries(&directory)?.len(), 2);
        Ok(())
    }

    #[test]
    fn unsafe_task_names_do_not_create_capture_directories() -> Result<()> {
        let (_temporary, base) = private_base()?;
        for task in [
            "",
            ".",
            "..",
            "../escape",
            "/absolute",
            "a/b",
            "a\\b",
            "C:drive",
            "has space",
        ] {
            assert!(capture_candidate(&base, task, &fixture(1)).is_err());
            assert!(!base.exists());
        }
        Ok(())
    }

    #[test]
    fn approval_requires_a_reviewer_and_separately_supplied_labels() -> Result<()> {
        let mut record = fixture(3);
        record.receipt["validation"] = json!("rejected");
        record.receipt["parsed"] = json!({"id": "syntactically-valid-but-wrong-type"});
        for reviewer in ["", " ", "\t\r\n"] {
            assert!(approve_candidate(&record, json!({"id": 3}), reviewer).is_err());
        }
        let approved = approve_candidate(&record, json!({"id": 3}), "synthetic-reviewer")?;
        assert_eq!(approved.expected_json, json!({"id": 3}));
        assert_ne!(approved.expected_json, record.receipt["parsed"]);
        assert_eq!(approved.input_text, record.input_text);
        assert_eq!(approved.additional_context, record.additional_context);
        assert!(approved.images.is_empty());
        assert_eq!(record.qualification, CaptureQualification::Unreviewed);
        assert!(
            serde_json::from_value::<crate::examples::RawTaskExample>(serde_json::to_value(
                &record
            )?)
            .is_err()
        );
        assert!(
            serde_json::from_value::<CaptureRecord>(json!({
                "input_text": "synthetic", "additional_context": null,
                "receipt": {}, "qualification": "approved"
            }))
            .is_err()
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn unix_permissions_are_private_without_changing_existing_records() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;
        let (_temporary, base) = private_base()?;
        let path = capture_candidate(&base, "synthetic", &fixture(1))?;
        assert_eq!(fs::metadata(&base)?.permissions().mode() & 0o777, 0o700);
        assert_eq!(
            fs::metadata(base.join("synthetic"))?.permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(fs::metadata(&path)?.permissions().mode() & 0o777, 0o600);
        fs::set_permissions(&path, fs::Permissions::from_mode(0o640))?;
        let original = fs::read(&path)?;
        capture_candidate(&base, "synthetic", &fixture(2))?;
        assert_eq!(fs::read(&path)?, original);
        assert_eq!(fs::metadata(path)?.permissions().mode() & 0o777, 0o640);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn nonprivate_existing_base_and_task_are_rejected_not_repaired() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;
        let (_temporary, base) = private_base()?;
        let path = capture_candidate(&base, "synthetic", &fixture(1))?;
        let original = fs::read(&path)?;
        for directory in [&base, &base.join("synthetic")] {
            fs::set_permissions(directory, fs::Permissions::from_mode(0o755))?;
            assert!(capture_candidate(&base, "synthetic", &fixture(2)).is_err());
            assert_eq!(fs::metadata(directory)?.permissions().mode() & 0o777, 0o755);
            assert_eq!(fs::read(&path)?, original);
            assert_eq!(entries(&base.join("synthetic"))?.len(), 1);
            // Restore only this test's synthetic fixture, never user records.
            fs::set_permissions(directory, fs::Permissions::from_mode(0o700))?;
        }
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn symlink_base_task_ancestor_and_destination_are_rejected() -> Result<()> {
        use std::os::unix::fs::symlink;
        let (temporary, base) = private_base()?;
        let root = temporary.path().canonicalize()?;
        let outside = root.join("outside");
        fs::create_dir(&outside)?;
        symlink(&outside, &base)?;
        assert!(capture_candidate(&base, "synthetic", &fixture(1)).is_err());
        assert!(capture_candidate(&base.join("nested"), "synthetic", &fixture(1)).is_err());
        assert!(entries(&outside)?.is_empty());
        fs::remove_file(&base)?;
        let existing = capture_candidate(&base, "synthetic", &fixture(1))?;
        let original = fs::read(&existing)?;
        symlink(&outside, base.join("linked-task"))?;
        assert!(capture_candidate(&base, "linked-task", &fixture(1)).is_err());
        let directory = base.join("synthetic");
        let destination = directory.join("linked.json");
        symlink(&existing, &destination)?;
        let temp = new_temporary(&directory)?;
        assert!(
            write_and_publish(temp, &destination, |file| {
                file.write_all(b"{}")?;
                Ok(())
            })
            .is_err()
        );
        assert!(fs::symlink_metadata(destination)?.file_type().is_symlink());
        assert_eq!(fs::read(existing)?, original);
        assert_eq!(entries(&directory)?.len(), 2);
        Ok(())
    }
}
