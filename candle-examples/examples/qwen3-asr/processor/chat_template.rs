//! Chat template utilities.
//!
//! The official Qwen3-ASR stack formats prompts via HuggingFace chat templates,
//! but the public model repo does not always ship a `chat_template`. For parity,
//! we implement the minimal prompt builder used in our golden exporter script.

use anyhow::Result;

/// `<|im_start|>` token string.
pub const IM_START: &str = "<|im_start|>";
/// `<|im_end|>` token string.
pub const IM_END: &str = "<|im_end|>";

/// `<|audio_start|>` token string.
pub const AUDIO_BOS: &str = "<|audio_start|>";
/// `<|audio_pad|>` token string.
pub const AUDIO_PAD: &str = "<|audio_pad|>";
/// `<|audio_end|>` token string.
pub const AUDIO_EOS: &str = "<|audio_end|>";

/// Build the ASR prompt string for one request.
///
/// This mirrors the output shape of:
/// `processor.apply_chat_template([system, user(audio)], add_generation_prompt=True, tokenize=False)`
///
/// If `force_language` is provided, appends: `language {X}<asr_text>` after the assistant prefix.
pub fn build_prompt(context: &str, force_language: Option<&str>) -> String {
    let mut s = String::new();
    s.push_str(IM_START);
    s.push_str("system\n");
    s.push_str(context);
    s.push_str(IM_END);
    s.push('\n');

    s.push_str(IM_START);
    s.push_str("user\n");
    s.push_str(AUDIO_BOS);
    s.push_str(AUDIO_PAD);
    s.push_str(AUDIO_EOS);
    s.push_str(IM_END);
    s.push('\n');

    s.push_str(IM_START);
    s.push_str("assistant\n");
    if let Some(lang) = force_language {
        s.push_str("language ");
        s.push_str(lang);
        s.push_str("<asr_text>");
    }
    s
}

/// Extract the forced language name from a prompt suffix.
///
/// This is a small helper for tests and debugging.
pub fn parse_forced_language_suffix(prompt: &str) -> Result<Option<&str>> {
    const PREFIX: &str = "language ";
    const SUFFIX: &str = "<asr_text>";

    let idx = prompt.find(PREFIX);
    let Some(idx) = idx else {
        return Ok(None);
    };

    let rest = &prompt[idx + PREFIX.len()..];
    let Some(end) = rest.find(SUFFIX) else {
        return Ok(None);
    };

    Ok(Some(&rest[..end]))
}

#[cfg(test)]
mod tests {
    use super::build_prompt;

    #[test]
    fn test_build_prompt_empty_context_no_language() -> anyhow::Result<()> {
        let expected = concat!(
            "<|im_start|>system\n",
            "<|im_end|>\n",
            "<|im_start|>user\n",
            "<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n",
            "<|im_start|>assistant\n",
        );
        let got = build_prompt("", None);
        if got != expected {
            anyhow::bail!("prompt mismatch:\nexpected={expected:?}\n     got={got:?}");
        }
        Ok(())
    }

    #[test]
    fn test_build_prompt_with_language_suffix() -> anyhow::Result<()> {
        let got = build_prompt("ctx", Some("English"));
        if !got.ends_with("language English<asr_text>") {
            anyhow::bail!("missing language suffix: {got:?}");
        }
        Ok(())
    }
}
