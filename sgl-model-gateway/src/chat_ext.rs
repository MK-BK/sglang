//! Wrapper that preserves all non-standard fields through the gateway's
//! deserialise → re-serialise round-trip **and** normalises non-standard
//! content-part shapes.
//!
//! `openai_protocol::chat::ChatCompletionRequest` only models the OpenAI Chat
//! Completions schema.  Two things get lost during a strict deserialise:
//!
//! 1. **Top-level fields** not in the schema (e.g. `thinking`,
//!    `reasoning_effort`) — captured by the `extra` catch-all `HashMap`.
//! 2. **Content-part shapes** that differ from the OpenAI spec — e.g.
//!    Moonshot/Kimi sends `"image_url": "https://..."` (a bare string)
//!    while `openai_protocol::ImageUrl` expects `{"url": ..., "detail": ...}`,
//!    causing `#[serde(untagged)] MessageContent` to reject the entire
//!    request with a 400.
//!
//! `ChatCompletionRequestExt` solves both problems:
//!
//! - `extra` captures unknown top-level keys via `#[serde(flatten)]`.
//! - When strict deserialisation of `inner` fails, a custom `Deserialize`
//!   implementation normalises the problematic content parts (e.g. wrapping
//!   a bare-string `image_url` into `{"url": ...}`) so the typed struct can
//!   be built and forwarded successfully.
use std::collections::HashMap;

use serde::de::Error as _;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use validator::Validate;

use crate::protocols::{
    chat::ChatCompletionRequest,
    common::GenerationRequest,
    validated::Normalizable,
};

/// Walk every message in a JSON request body and normalise content parts
/// so that `openai_protocol` can deserialize them.  Currently handles:
///
/// - **String `image_url`** — Moonshot/Kimi sends
///   `{"type": "image_url", "image_url": "https://..."}` but
///   `openai_protocol::ImageUrl` expects `{"url": ..., "detail": ...}`.
///   We rewrite to `{"type": "image_url", "image_url": {"url": "https://..."}}`.
///
/// Returns `true` if at least one field was modified.
fn normalize_message_contents(body: &mut Value) -> bool {
    let messages = match body.get_mut("messages").and_then(|m| m.as_array_mut()) {
        Some(arr) => arr,
        None => return false,
    };

    let mut modified = false;
    for msg in messages.iter_mut() {
        let Some(parts) = msg.get_mut("content").and_then(|c| c.as_array_mut()) else {
            continue;
        };

        for part in parts.iter_mut() {
            // Normalise string-form `image_url` → object form
            if part.get("type").and_then(|t| t.as_str()) == Some("image_url") {
                if let Some(iu) = part.get_mut("image_url") {
                    if iu.is_string() {
                        let url = iu.take();
                        *iu = serde_json::json!({"url": url});
                        modified = true;
                    }
                }
            }
        }
    }
    modified
}

/// Wrapper around [`ChatCompletionRequest`] that captures **all** fields not
/// recognised by the OpenAI schema.
///
/// Both `inner` and `extra` use `#[serde(flatten)]`, so on deserialise every
/// known key populates `inner` and every remaining key lands in `extra`; on
/// serialise both sets are spread back to a single flat JSON object.
#[derive(Debug, Clone, Serialize, Default)]
pub struct ChatCompletionRequestExt {
    /// The standard OpenAI Chat Completions fields.
    #[serde(flatten)]
    pub inner: ChatCompletionRequest,

    /// Catch-all for every field not in the OpenAI schema (e.g. `thinking`,
    /// `reasoning_effort`, vendor-specific extensions, …).
    ///
    /// Stored as raw [`Value`]s so any shape is passed through verbatim.
    /// Empty on serialise to avoid emitting `"extra": {}`.
    #[serde(flatten)]
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub extra: HashMap<String, Value>,
}

impl ChatCompletionRequestExt {
    /// Convenience accessor for a single extra field by name.
    pub fn get_extra(&self, key: &str) -> Option<&Value> {
        self.extra.get(key)
    }
}

// --- Custom Deserialize ----------------------------------------------------
//
// Two-phase strategy:
//
// 1. **Fast path** — Try the derived `Deserialize` directly.  This is
//    zero-allocation for the common case (standard OpenAI format).
//
// 2. **Slow path** — On failure, buffer the input into a `serde_json::Value`,
//    normalise known-incompatible content shapes (e.g. bare-string
//    `image_url` → object form), and retry deserialisation.

impl<'de> Deserialize<'de> for ChatCompletionRequestExt {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Fast path: try derived deserialisation first.
        // We use `serde_json::value::Value` as an intermediate only when the
        // fast path fails.  Unfortunately we cannot "peek" at the deserializer
        // without consuming it, so we buffer the full input upfront.
        //
        // For the overwhelming majority of requests (standard format), the
        // overhead of buffering to Value first is modest compared to the
        // safety of catching non-standard content shapes.
        let raw: Value = Value::deserialize(deserializer)?;

        // Try strict deserialisation first.
        match serde_json::from_value::<ChatCompletionRequestExtInner>(raw.clone()) {
            Ok(inner_result) => Ok(ChatCompletionRequestExt {
                inner: inner_result.inner,
                extra: inner_result.extra,
            }),
            Err(strict_err) => {
                // Slow path: normalise known-incompatible content shapes.
                let mut normalised = raw;
                let was_modified = normalize_message_contents(&mut normalised);

                if was_modified {
                    match serde_json::from_value::<ChatCompletionRequestExtInner>(normalised) {
                        Ok(inner_result) => {
                            tracing::debug!(
                                "Normalised content-part shapes in chat request"
                            );
                            Ok(ChatCompletionRequestExt {
                                inner: inner_result.inner,
                                extra: inner_result.extra,
                            })
                        }
                        Err(e) => Err(D::Error::custom(format!(
                            "Invalid JSON data: {}",
                            e
                        ))),
                    }
                } else {
                    // Normalisation didn't help — surface the original error.
                    Err(D::Error::custom(format!(
                        "Invalid JSON data: {}",
                        strict_err
                    )))
                }
            }
        }
    }
}

/// Helper struct that mirrors the derived `Deserialize` impl of
/// `ChatCompletionRequestExt` (used internally by the custom Deserialize).
#[derive(Debug, Deserialize)]
struct ChatCompletionRequestExtInner {
    #[serde(flatten)]
    inner: ChatCompletionRequest,
    #[serde(flatten)]
    extra: HashMap<String, Value>,
}

// --- Deref / DerefMut ------------------------------------------------------

impl std::ops::Deref for ChatCompletionRequestExt {
    type Target = ChatCompletionRequest;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for ChatCompletionRequestExt {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

// --- GenerationRequest -----------------------------------------------------

impl GenerationRequest for ChatCompletionRequestExt {
    fn is_stream(&self) -> bool {
        self.inner.is_stream()
    }

    fn get_model(&self) -> Option<&str> {
        self.inner.get_model()
    }

    fn extract_text_for_routing(&self) -> String {
        self.inner.extract_text_for_routing()
    }
}

// --- Validate --------------------------------------------------------------

impl Validate for ChatCompletionRequestExt {
    fn validate(&self) -> Result<(), validator::ValidationErrors> {
        self.inner.validate()
    }
}

// --- Normalizable ----------------------------------------------------------

impl Normalizable for ChatCompletionRequestExt {
    fn normalize(&mut self) {
        self.inner.normalize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // --- normalize_message_contents tests ---

    #[test]
    fn test_normalize_string_image_url() {
        let mut body = json!({
            "model": "test",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": "https://blobproxy.moonshot.cn/blobs/k3.abc?sig=123&exp=456"}
                ]
            }]
        });

        let modified = normalize_message_contents(&mut body);
        assert!(modified);

        let iu = &body["messages"][0]["content"][1]["image_url"];
        assert!(iu.is_object());
        assert_eq!(iu["url"], "https://blobproxy.moonshot.cn/blobs/k3.abc?sig=123&exp=456");
    }

    #[test]
    fn test_normalize_preserves_object_image_url() {
        let mut body = json!({
            "model": "test",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png", "detail": "auto"}}
                ]
            }]
        });

        let modified = normalize_message_contents(&mut body);
        assert!(!modified);
    }

    #[test]
    fn test_normalize_mixed_image_url_formats() {
        let mut body = json!({
            "model": "test",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "compare these"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/a.png", "detail": "high"}},
                    {"type": "image_url", "image_url": "https://blobproxy.moonshot.cn/blobs/k3.xyz?sig=abc"}
                ]
            }]
        });

        let modified = normalize_message_contents(&mut body);
        assert!(modified);

        // First image_url (object) unchanged
        let iu1 = &body["messages"][0]["content"][1]["image_url"];
        assert!(iu1.is_object());
        assert_eq!(iu1["detail"], "high");

        // Second image_url (was string) normalised
        let iu2 = &body["messages"][0]["content"][2]["image_url"];
        assert!(iu2.is_object());
        assert_eq!(iu2["url"], "https://blobproxy.moonshot.cn/blobs/k3.xyz?sig=abc");
    }

    // --- Full deserialize / serialize round-trip tests ---

    #[test]
    fn test_deserialize_standard_image_url() {
        let body = json!({
            "model": "test-model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png", "detail": "auto"}}
                ]
            }]
        });

        let ext: ChatCompletionRequestExt = serde_json::from_value(body).unwrap();
        assert_eq!(ext.inner.model, "test-model");
    }

    #[test]
    fn test_deserialize_moonshot_string_image_url() {
        let body = json!({
            "model": "moonshot-v1",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this image"},
                    {"type": "image_url", "image_url": "https://blobproxy.moonshot.cn/blobs/k3.FihmOmRuFRQ5VdlWQqparicJzq0S?sig=abc&exp=123"}
                ],
                "name": "resource"
            }]
        });

        let ext: ChatCompletionRequestExt = serde_json::from_value(body).unwrap();

        // inner is populated for routing
        assert_eq!(ext.inner.model, "moonshot-v1");
        assert!(!ext.inner.messages.is_empty());

        // Serialising should produce normalised (object-form) image_url
        let serialised = serde_json::to_value(&ext).unwrap();
        let serialised_iu = &serialised["messages"][0]["content"][1]["image_url"];
        assert!(serialised_iu.is_object());
        assert_eq!(serialised_iu["url"], "https://blobproxy.moonshot.cn/blobs/k3.FihmOmRuFRQ5VdlWQqparicJzq0S?sig=abc&exp=123");
    }

    #[test]
    fn test_deserialize_moonshot_real_world_request() {
        // Simulates the actual request from the bug report
        let body = json!({
            "model": "moonshot-v1",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "<uploaded_files>\n1. /upload/粘贴图片_1772796257362.png\n</uploaded_files>"},
                    {"type": "text", "text": "<image path=\"/upload/粘贴图片_1772796257362.png\" content_type=\"image/png\">"},
                    {"type": "image_url", "image_url": {"url": "https://blobproxy.moonshot.cn/blobs/k3.FpaohAb4c4X9VZXxTsoHbiayWO1C?sig=qyvr7KoukAp1EQSzEAMVDw6Y41Ls4NCBisWAGntkxY0&exp=1804926322", "detail": "auto"}},
                    {"type": "text", "text": "</image>"}
                ],
                "name": "resource"
            }]
        });

        let ext: ChatCompletionRequestExt = serde_json::from_value(body).unwrap();

        // This request uses the standard object format for image_url,
        // so no normalisation is needed — fast path
        assert_eq!(ext.inner.model, "moonshot-v1");
    }

    #[test]
    fn test_deserialize_with_extra_fields_and_string_image_url() {
        let body = json!({
            "model": "test-model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": "https://example.com/img.png"}
                ]
            }],
            "thinking": {"type": "enabled", "budget_tokens": 5000}
        });

        let ext: ChatCompletionRequestExt = serde_json::from_value(body).unwrap();

        // Extra fields should still be captured
        assert!(ext.extra.contains_key("thinking"));

        // Serialised image_url should be normalised to object form
        let serialised = serde_json::to_value(&ext).unwrap();
        assert!(serialised["thinking"].is_object());
        let iu = &serialised["messages"][0]["content"][0]["image_url"];
        assert!(iu.is_object());
        assert_eq!(iu["url"], "https://example.com/img.png");
    }
}