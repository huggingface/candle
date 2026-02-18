//! Output parsing and text post-processing helpers.
//!
//! Faithful port of:
//! `../../../Qwen3-ASR/qwen_asr/inference/utils.py`

use anyhow::{bail, Result};

/// Canonical list of languages supported by the official Qwen3-ASR Python stack.
pub const SUPPORTED_LANGUAGES: &[&str] = &[
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian",
];

/// Maximum audio duration (seconds) per ASR request in the official stack.
pub const MAX_ASR_INPUT_SECONDS: f32 = 1200.0;

/// Maximum audio duration (seconds) per forced-aligner request in the official stack.
pub const MAX_FORCE_ALIGN_INPUT_SECONDS: f32 = 180.0;

const ASR_TEXT_TAG: &str = "<asr_text>";
const LANG_PREFIX: &str = "language ";
const DEFAULT_REPETITION_THRESHOLD: usize = 20;
const MAX_PATTERN_LEN: usize = 20;

/// Normalize a language name to the canonical "Qwen3-ASR" form:
/// first letter uppercase, the rest lowercase (e.g., `cHINese` -> `Chinese`).
pub fn normalize_language_name(language: &str) -> Result<String> {
    let s = language.trim();
    if s.is_empty() {
        bail!("language is empty");
    }
    let mut chars = s.chars();
    let first = chars
        .next()
        .ok_or_else(|| anyhow::anyhow!("language is empty"))?;
    let mut out = String::new();
    out.extend(first.to_uppercase());
    for c in chars {
        out.extend(c.to_lowercase());
    }
    Ok(out)
}

/// Validate that `language` is supported by the official Qwen3-ASR implementation.
pub fn validate_language(language: &str) -> Result<()> {
    if SUPPORTED_LANGUAGES.contains(&language) {
        Ok(())
    } else {
        bail!(
            "Unsupported language: {language}. Supported: {:?}",
            SUPPORTED_LANGUAGES
        )
    }
}

fn fix_char_repeats(s: &str, thresh: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let mut i = 0usize;
    let mut out = String::new();

    while i < n {
        let mut count = 1usize;
        while i + count < n && chars[i + count] == chars[i] {
            count += 1;
        }

        if count > thresh {
            out.push(chars[i]);
        } else {
            for j in 0..count {
                out.push(chars[i + j]);
            }
        }

        i += count;
    }

    out
}

fn fix_pattern_repeats(chars: &[char], thresh: usize, max_len: usize) -> Vec<char> {
    let n = chars.len();
    let min_repeat_chars = thresh.saturating_mul(2);
    if n < min_repeat_chars {
        return chars.to_vec();
    }

    let mut i = 0usize;
    let mut result: Vec<char> = Vec::new();
    let mut found = false;

    while i <= n - min_repeat_chars {
        found = false;
        for k in 1..=max_len {
            let needed = k.saturating_mul(thresh);
            if i + needed > n {
                break;
            }

            let pattern = &chars[i..i + k];
            let mut valid = true;
            for rep in 1..thresh {
                let start_idx = i + rep * k;
                let end_idx = start_idx + k;
                let slice = chars.get(start_idx..end_idx);
                if slice != Some(pattern) {
                    valid = false;
                    break;
                }
            }

            if valid {
                let mut end_index = i + thresh * k;
                while end_index + k <= n && chars[end_index..end_index + k] == *pattern {
                    end_index += k;
                }

                result.extend_from_slice(pattern);
                result.extend(fix_pattern_repeats(&chars[end_index..], thresh, max_len));
                i = n;
                found = true;
                break;
            }
        }

        if found {
            break;
        }

        result.push(chars[i]);
        i += 1;
    }

    if !found {
        result.extend_from_slice(&chars[i..]);
    }

    result
}

/// Detect and compress pathological repetitions in decoded text.
///
/// This matches `detect_and_fix_repetitions` in the official Python stack.
pub fn detect_and_fix_repetitions(text: &str, threshold: usize) -> String {
    let text = fix_char_repeats(text, threshold);
    let chars: Vec<char> = text.chars().collect();
    fix_pattern_repeats(&chars, threshold, MAX_PATTERN_LEN)
        .into_iter()
        .collect()
}

/// Parse Qwen3-ASR raw model output into `(language, text)`.
///
/// Supported forms (from the official Python stack):
/// - With tag: `language Chinese<asr_text>...`
/// - With newlines in the metadata: `language Chinese\n...\n<asr_text>...`
/// - Without tag: treat the full string as pure text
/// - `language None<asr_text>`: treat as empty audio => `("", "")` (unless text is non-empty)
///
/// If `user_language` is provided, language is forced to `user_language` and `raw`
/// is treated as text-only (model output expected to omit metadata).
pub fn parse_asr_output(raw: Option<&str>, user_language: Option<&str>) -> (String, String) {
    let Some(raw) = raw else {
        return (String::new(), String::new());
    };
    let s0 = raw.trim();
    if s0.is_empty() {
        return (String::new(), String::new());
    }

    let s = detect_and_fix_repetitions(s0, DEFAULT_REPETITION_THRESHOLD);

    if let Some(lang) = user_language {
        return (lang.to_string(), s);
    }

    let Some((meta_part, text_part)) = s.split_once(ASR_TEXT_TAG) else {
        return (String::new(), s.trim().to_string());
    };

    let meta_lower = meta_part.to_ascii_lowercase();
    if meta_lower.contains("language none") {
        let t = text_part.trim();
        if t.is_empty() {
            return (String::new(), String::new());
        }
        return (String::new(), t.to_string());
    }

    let mut lang = String::new();
    for line in meta_part.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let low = line.to_ascii_lowercase();
        if low.starts_with(LANG_PREFIX) {
            let val = line.get(LANG_PREFIX.len()..).unwrap_or_default().trim();
            if !val.is_empty() {
                if let Ok(normalized) = normalize_language_name(val) {
                    lang = normalized;
                }
            }
            break;
        }
    }

    (lang, text_part.trim().to_string())
}

/// Merge per-chunk languages into a compact comma-separated string,
/// keeping order and removing consecutive duplicates and empty entries.
pub fn merge_languages<I, S>(langs: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut out: Vec<String> = Vec::new();
    let mut prev: Option<String> = None;

    for x in langs {
        let x = x.as_ref().trim();
        if x.is_empty() {
            continue;
        }

        if prev.as_deref() == Some(x) {
            continue;
        }

        out.push(x.to_string());
        prev = Some(x.to_string());
    }

    out.join(",")
}

#[cfg(test)]
mod tests {
    use super::{detect_and_fix_repetitions, merge_languages, parse_asr_output};

    #[test]
    fn test_detect_and_fix_repetitions_char_runs() -> anyhow::Result<()> {
        let got = detect_and_fix_repetitions("aaaaab", 3);
        if got != "ab" {
            anyhow::bail!("expected 'ab', got {got:?}");
        }

        let got = detect_and_fix_repetitions("aaab", 3);
        if got != "aaab" {
            anyhow::bail!("expected 'aaab', got {got:?}");
        }

        Ok(())
    }

    #[test]
    fn test_detect_and_fix_repetitions_patterns() -> anyhow::Result<()> {
        // "ab" repeated 3 times should compress to one "ab".
        let got = detect_and_fix_repetitions("ababab", 3);
        if got != "ab" {
            anyhow::bail!("expected 'ab', got {got:?}");
        }
        Ok(())
    }

    #[test]
    fn test_parse_asr_output_none_and_empty() -> anyhow::Result<()> {
        let (lang, text) = parse_asr_output(None, None);
        if !(lang.is_empty() && text.is_empty()) {
            anyhow::bail!("expected empty output for None");
        }

        let (lang, text) = parse_asr_output(Some("   \n "), None);
        if !(lang.is_empty() && text.is_empty()) {
            anyhow::bail!("expected empty output for whitespace");
        }

        Ok(())
    }

    #[test]
    fn test_parse_asr_output_no_tag() -> anyhow::Result<()> {
        let (lang, text) = parse_asr_output(Some("hello"), None);
        if !lang.is_empty() || text != "hello" {
            anyhow::bail!("unexpected parse: lang={lang:?} text={text:?}");
        }
        Ok(())
    }

    #[test]
    fn test_parse_asr_output_with_tag_and_language() -> anyhow::Result<()> {
        let raw = "language Chinese<asr_text> hello";
        let (lang, text) = parse_asr_output(Some(raw), None);
        if lang != "Chinese" || text != "hello" {
            anyhow::bail!("unexpected parse: lang={lang:?} text={text:?}");
        }
        Ok(())
    }

    #[test]
    fn test_parse_asr_output_with_tag_and_newlines() -> anyhow::Result<()> {
        let raw = "language Chinese\nfoo\n<asr_text>bar";
        let (lang, text) = parse_asr_output(Some(raw), None);
        if lang != "Chinese" || text != "bar" {
            anyhow::bail!("unexpected parse: lang={lang:?} text={text:?}");
        }
        Ok(())
    }

    #[test]
    fn test_parse_asr_output_language_none() -> anyhow::Result<()> {
        let raw = "language None<asr_text>";
        let (lang, text) = parse_asr_output(Some(raw), None);
        if !(lang.is_empty() && text.is_empty()) {
            anyhow::bail!("unexpected parse: lang={lang:?} text={text:?}");
        }

        let raw = "language None<asr_text>hi";
        let (lang, text) = parse_asr_output(Some(raw), None);
        if !lang.is_empty() || text != "hi" {
            anyhow::bail!("unexpected parse: lang={lang:?} text={text:?}");
        }

        Ok(())
    }

    #[test]
    fn test_parse_asr_output_forced_language_treats_raw_as_text() -> anyhow::Result<()> {
        let raw = "language Chinese<asr_text>hi";
        let (lang, text) = parse_asr_output(Some(raw), Some("English"));
        if lang != "English" || text.trim() != raw {
            anyhow::bail!("unexpected parse: lang={lang:?} text={text:?}");
        }
        Ok(())
    }

    #[test]
    fn test_merge_languages() -> anyhow::Result<()> {
        let langs = vec![
            "Chinese", "English", "English", "", "French", "French", "English",
        ];
        let got = merge_languages(langs);
        if got != "Chinese,English,French,English" {
            anyhow::bail!("expected merged languages, got {got:?}");
        }
        Ok(())
    }
}
