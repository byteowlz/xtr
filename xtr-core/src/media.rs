//! Image inputs for multimodal messages, independent of providers and the engine.
//!
//! MIME types are inferred from PNG, JPEG, GIF, or WebP magic bytes, never from
//! file extensions or caller-supplied metadata. This is signature sniffing, not
//! full image decoding: a recognized header does not prove an image is complete.
//! Files are read synchronously when encoded; no network URLs are fetched.

use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use serde_json::Value;
use serde_json::json;

/// An image source, validated only when [`Self::to_data_uri`] is called.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageInput {
    /// A local file. Its extension is ignored; its bytes determine the MIME type.
    File(PathBuf),
    /// Raw image bytes (including bytes read from stdin by a caller).
    Bytes(Vec<u8>),
    /// Strict, standard padded base64, without a data-URI prefix or whitespace.
    Base64(String),
}

impl ImageInput {
    /// Construct a local file input without reading it yet.
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        Self::File(path.into())
    }

    /// Construct an input from raw image bytes.
    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> Self {
        Self::Bytes(bytes.into())
    }

    /// Construct an input from raw standard base64 (not an existing data URI).
    pub fn from_base64(encoded: impl Into<String>) -> Self {
        Self::Base64(encoded.into())
    }

    /// Encode an image as `data:<sniffed-mime>;base64,<standard-base64>`.
    ///
    /// Returns an error for unreadable files, invalid base64, empty bytes, or
    /// unrecognized signatures. Errors never include image contents, encoded
    /// payloads, or local paths. No image transformation or re-compression occurs.
    pub fn to_data_uri(&self) -> Result<String> {
        match self {
            Self::File(path) => {
                let bytes = std::fs::read(path).context("Could not read image file")?;
                encode_data_uri(&bytes)
            }
            Self::Bytes(bytes) => encode_data_uri(bytes),
            Self::Base64(encoded) => {
                let bytes = STANDARD
                    .decode(encoded)
                    .context("Invalid standard base64 image input")?;
                encode_data_uri(&bytes)
            }
        }
    }
}

/// Build an OpenAI-compatible content array: text first, then images in order.
///
/// Returns an array even when there are no images, and preserves empty text.
/// Image conversion is all-or-error; no partial message is returned on failure.
/// This only constructs JSON and does not send a request or select a provider.
pub fn multimodal_content(text: &str, images: &[ImageInput]) -> Result<Value> {
    let mut content = Vec::with_capacity(images.len() + 1);
    content.push(json!({ "type": "text", "text": text }));
    for image in images {
        content.push(json!({
            "type": "image_url",
            "image_url": { "url": image.to_data_uri()? }
        }));
    }
    Ok(Value::Array(content))
}

fn encode_data_uri(bytes: &[u8]) -> Result<String> {
    let mime = sniff_mime(bytes)?;
    Ok(format!("data:{mime};base64,{}", STANDARD.encode(bytes)))
}

fn sniff_mime(bytes: &[u8]) -> Result<&'static str> {
    if bytes.is_empty() {
        bail!("Image input is empty");
    }
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Ok("image/png");
    }
    if bytes.starts_with(b"\xff\xd8\xff") {
        return Ok("image/jpeg");
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return Ok("image/gif");
    }
    if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP") {
        return Ok("image/webp");
    }
    bail!("Unsupported image signature; expected PNG, JPEG, GIF, or WebP")
}

#[cfg(test)]
#[expect(
    clippy::panic_in_result_fn,
    reason = "test assertions report mismatches while fixture IO errors propagate"
)]
mod tests {
    use super::*;

    // Synthetic signatures plus binary payloads test sniffing and lossless
    // transport, not codec validity. No real screenshots or image files needed.
    fn fixtures() -> Vec<(&'static str, Vec<u8>)> {
        vec![
            (
                "image/png",
                b"\x89PNG\r\n\x1a\n\x00\xff\x80payload".to_vec(),
            ),
            (
                "image/jpeg",
                b"\xff\xd8\xff\xe0\x00\xff\x80payload".to_vec(),
            ),
            ("image/gif", b"GIF87a\x00\xff\x80payload".to_vec()),
            ("image/gif", b"GIF89a\x00\xff\x80payload".to_vec()),
            (
                "image/webp",
                b"RIFF\x10\x00\x00\x00WEBP\x00\xff\x80payload".to_vec(),
            ),
        ]
    }

    fn assert_fidelity(input: &ImageInput, mime: &str, bytes: &[u8]) -> Result<()> {
        let uri = input.to_data_uri()?;
        let prefix = format!("data:{mime};base64,");
        let payload = uri.strip_prefix(&prefix).context("Wrong MIME prefix")?;
        assert_eq!(STANDARD.decode(payload)?, bytes);
        assert_eq!(payload, STANDARD.encode(bytes));
        Ok(())
    }

    #[test]
    fn raw_bytes_preserve_all_supported_formats() -> Result<()> {
        for (mime, bytes) in fixtures() {
            assert_fidelity(&ImageInput::from_bytes(bytes.clone()), mime, &bytes)?;
        }
        Ok(())
    }

    #[test]
    fn base64_preserves_all_supported_formats() -> Result<()> {
        for (mime, bytes) in fixtures() {
            assert_fidelity(
                &ImageInput::from_base64(STANDARD.encode(&bytes)),
                mime,
                &bytes,
            )?;
        }
        Ok(())
    }

    #[test]
    fn file_magic_overrides_extension_and_preserves_bytes() -> Result<()> {
        let dir = tempfile::tempdir()?;
        for (index, (mime, bytes)) in fixtures().into_iter().enumerate() {
            let path = dir.path().join(format!("image-{index}.txt"));
            std::fs::write(&path, &bytes)?;
            assert_fidelity(&ImageInput::from_path(path), mime, &bytes)?;
        }
        Ok(())
    }

    #[test]
    fn known_encoding_is_canonical() -> Result<()> {
        assert_eq!(
            ImageInput::from_bytes(b"GIF89a".to_vec()).to_data_uri()?,
            "data:image/gif;base64,R0lGODlh"
        );
        Ok(())
    }

    #[test]
    fn empty_inputs_are_rejected() -> Result<()> {
        assert!(ImageInput::from_bytes(Vec::new()).to_data_uri().is_err());
        assert!(ImageInput::from_base64("").to_data_uri().is_err());
        let file = tempfile::NamedTempFile::new()?;
        assert!(ImageInput::from_path(file.path()).to_data_uri().is_err());
        Ok(())
    }

    #[test]
    fn unsupported_and_truncated_signatures_are_rejected() {
        for bytes in [
            b"plain text".as_slice(),
            b"<svg></svg>",
            b"BM\x00\x00",
            b"\x89PNG\r\n\x1a",
            b"\xff\xd8",
            b"GIF89",
            b"RIFF",
            b"RIFF\x00\x00\x00\x00WEB",
            b"RIFF\x00\x00\x00\x00WAVE",
            b"xxxx\x00\x00\x00\x00WEBP",
        ] {
            assert!(ImageInput::from_bytes(bytes).to_data_uri().is_err());
            assert!(
                ImageInput::from_base64(STANDARD.encode(bytes))
                    .to_data_uri()
                    .is_err()
            );
        }
    }

    #[test]
    fn malformed_base64_is_rejected_without_echoing_payload() -> Result<()> {
        for encoded in [
            "private-image-payload!",
            "R0lGODlh$",
            "R0lGODlh\n",
            "R0lGODlhYQ",   // Missing required padding.
            "R0lGODlhYR==", // Nonzero trailing bits.
            "data:image/gif;base64,R0lGODlh",
        ] {
            let error = ImageInput::from_base64(encoded)
                .to_data_uri()
                .err()
                .context("Malformed base64 was accepted")?;
            assert!(!format!("{error:#}").contains(encoded));
        }
        Ok(())
    }

    #[test]
    fn missing_file_and_directory_return_safe_errors() -> Result<()> {
        let dir = tempfile::tempdir()?;
        for path in [
            dir.path().join("private-missing-image.png"),
            dir.path().to_path_buf(),
        ] {
            let error = ImageInput::from_path(path.clone())
                .to_data_uri()
                .err()
                .context("Unreadable file was accepted")?;
            assert!(!format!("{error:#}").contains(path.to_string_lossy().as_ref()));
        }
        Ok(())
    }

    #[test]
    fn unsupported_file_does_not_trust_png_extension() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("not-an-image.png");
        std::fs::write(&path, b"private non-image payload")?;
        let error = ImageInput::from_path(path)
            .to_data_uri()
            .err()
            .context("Unsupported file was accepted")?;
        assert!(!format!("{error:#}").contains("private non-image payload"));
        Ok(())
    }

    #[test]
    fn multimodal_content_preserves_text_and_image_order() -> Result<()> {
        let images = [
            ImageInput::from_bytes(b"GIF89a".to_vec()),
            ImageInput::from_bytes(b"GIF87a".to_vec()),
        ];
        assert_eq!(
            multimodal_content("quoted \"text\"\n", &images)?,
            json!([
                { "type": "text", "text": "quoted \"text\"\n" },
                { "type": "image_url", "image_url": { "url": images[0].to_data_uri()? } },
                { "type": "image_url", "image_url": { "url": images[1].to_data_uri()? } }
            ])
        );
        Ok(())
    }

    #[test]
    fn multimodal_content_supports_text_only_and_empty_text() -> Result<()> {
        for text in ["text only", ""] {
            assert_eq!(
                multimodal_content(text, &[])?,
                json!([{ "type": "text", "text": text }])
            );
        }
        Ok(())
    }

    #[test]
    fn multimodal_content_rejects_any_invalid_image() {
        let images = [
            ImageInput::from_bytes(b"GIF89a".to_vec()),
            ImageInput::from_bytes(b"invalid".to_vec()),
        ];
        assert!(multimodal_content("text", &images).is_err());
    }
}
