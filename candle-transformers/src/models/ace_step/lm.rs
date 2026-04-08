//! Language Model pipeline for ACE-Step audio code generation.
//!
//! Implements two-phase autoregressive generation using a fine-tuned Qwen3 model:
//! 1. **CoT phase**: generates chain-of-thought metadata (BPM, duration, etc.)
//!    inside `<think>...</think>` tags
//! 2. **Codes phase**: generates audio code tokens `<|audio_code_N|>` constrained
//!    to valid codebook entries, with duration-based EOS control
//!
//! The generated audio codes are converted to latents via
//! `ResidualFSQ::get_output_from_indices → AudioTokenDetokenizer` and fed to the
//! DiT as `precomputed_lm_hints_25hz`.
//!
//! Reference: `acestep/llm_inference.py`, `acestep/constrained_logits_processor.py`

use std::collections::BTreeMap;

use crate::generation::{LogitsProcessor, Sampling};
use crate::models::qwen3;
use candle::{Device, Result, Tensor};

/// Default system instruction for the LM.
pub const DEFAULT_LM_INSTRUCTION: &str =
    "Generate audio semantic tokens based on the given conditions:";

/// Number of audio codes generated per second of audio (5 Hz).
pub const CODES_PER_SECOND: f64 = 5.0;

/// Maximum valid audio code value (codebook size = 64000).
pub const MAX_AUDIO_CODE: u32 = 63999;

/// Generation parameters for the LM pipeline.
pub struct LmConfig {
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    /// Maximum tokens for CoT (metadata) generation.
    pub max_cot_tokens: usize,
    /// Maximum tokens for audio code generation.
    pub max_code_tokens: usize,
    pub repetition_penalty: f64,
    pub seed: u64,
}

impl Default for LmConfig {
    fn default() -> Self {
        Self {
            temperature: 0.85,
            top_p: None,
            top_k: None,
            max_cot_tokens: 512,
            max_code_tokens: 3072,
            repetition_penalty: 1.0,
            seed: 42,
        }
    }
}

/// Result of LM generation.
pub struct LmOutput {
    /// Parsed metadata from CoT (bpm, caption, duration, genres, keyscale, language, timesignature).
    pub metadata: BTreeMap<String, String>,
    /// Raw audio code values in `[0, 63999]`.
    pub audio_codes: Vec<i64>,
    /// Full generated text (CoT + codes) for debugging.
    pub raw_text: String,
}

/// Token IDs discovered from the tokenizer at runtime.
pub struct TokenIds {
    pub eos: u32,
    pub think_start: u32,
    pub think_end: u32,
    pub audio_code_start: u32,
    pub audio_code_end: u32,
}

impl TokenIds {
    /// Discover special token IDs by encoding known strings.
    ///
    /// `encode_fn` should encode a string without special tokens and return
    /// the token IDs. It must produce exactly one token for each special string.
    pub fn discover(encode_fn: impl Fn(&str) -> Result<Vec<u32>>) -> Result<Self> {
        let get_single = |s: &str| -> Result<u32> {
            let ids = encode_fn(s)?;
            if ids.len() != 1 {
                candle::bail!("expected single token for {s:?}, got {ids:?}");
            }
            Ok(ids[0])
        };

        let eos = get_single("<|im_end|>")?;
        let think_start = get_single("<think>")?;
        let think_end = get_single("</think>")?;
        let audio_code_start = get_single("<|audio_code_0|>")?;
        let audio_code_end = audio_code_start + MAX_AUDIO_CODE;

        Ok(Self {
            eos,
            think_start,
            think_end,
            audio_code_start,
            audio_code_end,
        })
    }
}

/// Maximum tokens for the caption value before forcing transition.
/// Python's FSM allows up to 512 tokens.
const MAX_CAPTION_TOKENS: usize = 512;

/// Maximum tokens for free-text numeric fields.
const MAX_NUMERIC_TOKENS: usize = 16;

/// Field types with their constraint strategies.
#[derive(Debug, Clone, Copy, PartialEq)]
enum FieldKind {
    /// Digits only. Transition on argmax=`\n` or max tokens.
    Numeric,
    /// Free text. Block `\n`, transition on argmax=`\n` or max tokens.
    Caption,
    /// Pick the best match from a pre-tokenized set of valid values,
    /// then force the entire value + `\n` (greedy selection).
    Enumerated,
}

/// One metadata field: its forced prefix and value constraint.
struct FieldSpec {
    /// Field name prefix to force (e.g. `"bpm: "`). Tokenized at init.
    prefix_tokens: Vec<u32>,
    kind: FieldKind,
    /// Pre-tokenized valid values (only for `Enumerated`).
    /// Each entry: complete token sequence for `" value\n"` (space + value + newline).
    valid_values: Vec<Vec<u32>>,
}

/// Valid language codes (matches Python `VALID_LANGUAGES`).
const VALID_LANGUAGES: &[&str] = &[
    "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en", "es", "fa", "fi", "fr", "he", "hi",
    "hr", "ht", "hu", "id", "is", "it", "ja", "ko", "la", "lt", "ms", "ne", "nl", "no", "pa", "pl",
    "pt", "ro", "ru", "sa", "sk", "sr", "sv", "sw", "ta", "te", "th", "tl", "tr", "uk", "ur", "vi",
    "yue", "zh",
];

/// Valid time signatures.
const VALID_TIMESIGS: &[&str] = &["2", "3", "4", "6"];

/// Valid keyscale values (7 notes x 5 accidentals x 2 modes = 70).
fn valid_keyscales() -> Vec<String> {
    let notes = ["A", "B", "C", "D", "E", "F", "G"];
    let accidentals = ["", "#", "b"];
    let modes = ["major", "minor"];
    let mut values = Vec::with_capacity(notes.len() * accidentals.len() * modes.len());
    for note in &notes {
        for acc in &accidentals {
            for mode in &modes {
                values.push(format!("{note}{acc} {mode}"));
            }
        }
    }
    values
}

/// CoT field-order constraint for metadata generation.
///
/// Matches Python's `MetadataConstrainedLogitsProcessor` field order:
/// `bpm → caption → duration → keyscale → language → timesignature`.
///
/// For each field:
/// 1. **Force** the field-name prefix token-by-token
/// 2. **Constrain** the value according to the field kind:
///    - `Numeric` (bpm, duration): allow only digit tokens; transition on argmax=`\n`
///    - `Caption`: allow all tokens except `\n`; transition on argmax=`\n` or 512 tokens
///    - `Enumerated` (keyscale, language, timesig): pick the best-scoring valid value
///      from logits and force it entirely (greedy selection, matching Python's approach)
struct CotFieldForcer {
    fields: Vec<FieldSpec>,
    /// Index of the current field (0..fields.len()).
    field_idx: usize,
    /// Queue of tokens to force.
    force_queue: Vec<u32>,
    newline_token: u32,
    /// Token IDs for ASCII digits 0-9.
    digit_tokens: Vec<u32>,
    active: bool,
    done: bool,
    value_token_count: usize,
}

impl CotFieldForcer {
    fn new(encode_fn: &dyn Fn(&str) -> Result<Vec<u32>>) -> Result<Self> {
        let newline_tokens = encode_fn("\n")?;
        let newline_token = *newline_tokens.last().ok_or_else(|| {
            candle::Error::Msg("newline encodes to empty token sequence".to_string())
        })?;

        // Discover digit token IDs + space (for leading space after ":")
        let mut digit_tokens: Vec<u32> = (0..=9u32)
            .filter_map(|d| {
                let ids = encode_fn(&d.to_string()).ok()?;
                ids.last().copied()
            })
            .collect();
        if let Ok(sp) = encode_fn(" ") {
            if let Some(&t) = sp.last() {
                digit_tokens.push(t);
            }
        }

        // Pre-tokenize valid values for enumerated fields.
        // Tokenize as " value\n" to capture the space-after-colon + newline.
        let tokenize_values = |values: &[&str]| -> Result<Vec<Vec<u32>>> {
            let mut result = Vec::with_capacity(values.len());
            for &v in values {
                let tokens = encode_fn(&format!(" {v}\n"))?;
                if !tokens.is_empty() {
                    result.push(tokens);
                }
            }
            Ok(result)
        };
        let keyscale_strings = valid_keyscales();
        let keyscale_strs: Vec<&str> = keyscale_strings.iter().map(|s| s.as_str()).collect();

        let fields = vec![
            FieldSpec {
                prefix_tokens: encode_fn("bpm:")?,
                kind: FieldKind::Numeric,
                valid_values: vec![],
            },
            FieldSpec {
                prefix_tokens: encode_fn("caption:")?,
                kind: FieldKind::Caption,
                valid_values: vec![],
            },
            FieldSpec {
                prefix_tokens: encode_fn("duration:")?,
                kind: FieldKind::Numeric,
                valid_values: vec![],
            },
            FieldSpec {
                prefix_tokens: encode_fn("keyscale:")?,
                kind: FieldKind::Enumerated,
                valid_values: tokenize_values(&keyscale_strs)?,
            },
            FieldSpec {
                prefix_tokens: encode_fn("language:")?,
                kind: FieldKind::Enumerated,
                valid_values: tokenize_values(VALID_LANGUAGES)?,
            },
            FieldSpec {
                prefix_tokens: encode_fn("timesignature:")?,
                kind: FieldKind::Enumerated,
                valid_values: tokenize_values(VALID_TIMESIGS)?,
            },
        ];

        Ok(Self {
            fields,
            field_idx: 0,
            force_queue: Vec::new(),
            newline_token,
            digit_tokens,
            active: false,
            done: false,
            value_token_count: 0,
        })
    }

    /// Call after `<think>` has been emitted.
    fn activate(&mut self) {
        self.active = true;
        self.seed_next_field_prefix();
    }

    /// Load `\n{field_name}:` tokens into the force queue.
    fn seed_next_field_prefix(&mut self) {
        if self.field_idx < self.fields.len() {
            self.force_queue.clear();
            self.force_queue.push(self.newline_token);
            self.force_queue
                .extend_from_slice(&self.fields[self.field_idx].prefix_tokens);
            self.value_token_count = 0;
        } else {
            // All fields done — inject final newline, then let LM emit </think>.
            self.force_queue.clear();
            self.force_queue.push(self.newline_token);
            self.done = true;
        }
    }

    /// Returns the next forced token, or `None` for free generation.
    fn next_forced_token(&mut self) -> Option<u32> {
        if !self.active || self.force_queue.is_empty() {
            return None;
        }
        Some(self.force_queue.remove(0))
    }

    /// Apply per-field value constraints to logits.
    ///
    /// For `Enumerated` fields, picks the best valid value from logits and
    /// queues the entire value for forcing (returns `true` always).
    ///
    /// For `Numeric` and `Caption` fields, masks invalid tokens and returns
    /// `true` when the LM wants to transition (argmax = `\n` or limit hit).
    fn constrain_value_logits(&mut self, logits: &mut [f32]) -> bool {
        if !self.active || self.done || !self.force_queue.is_empty() {
            return false;
        }

        let field = &self.fields[self.field_idx];
        match field.kind {
            FieldKind::Enumerated => {
                // Greedy selection: pick the valid value whose first token
                // has the highest logit, then force the entire value + next
                // field prefix in one go.
                //
                // Valid values are pre-tokenized as " value\n", so the newline
                // is included. After the value we append the next field prefix.
                let best = field
                    .valid_values
                    .iter()
                    .filter(|v| !v.is_empty())
                    .max_by(|a, b| {
                        let sa = logits
                            .get(a[0] as usize)
                            .copied()
                            .unwrap_or(f32::NEG_INFINITY);
                        let sb = logits
                            .get(b[0] as usize)
                            .copied()
                            .unwrap_or(f32::NEG_INFINITY);
                        sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .cloned();

                self.field_idx += 1;

                // Build force_queue: value tokens + next field prefix (or final \n)
                self.force_queue.clear();
                if let Some(value_tokens) = best {
                    // value_tokens = " value\n" — already includes newline
                    self.force_queue.extend_from_slice(&value_tokens);
                }
                // Append next field prefix (or final newline if all fields done)
                if self.field_idx < self.fields.len() {
                    // Don't push extra newline — value_tokens already ends with \n
                    self.force_queue
                        .extend_from_slice(&self.fields[self.field_idx].prefix_tokens);
                    self.value_token_count = 0;
                } else {
                    // All fields done — the value's trailing \n is enough
                    self.done = true;
                }
                true
            }
            FieldKind::Numeric => {
                self.value_token_count += 1;
                let argmax = Self::argmax(logits);
                if argmax == self.newline_token || self.value_token_count >= MAX_NUMERIC_TOKENS {
                    self.field_idx += 1;
                    self.seed_next_field_prefix();
                    true
                } else {
                    // Allow only digits + space (for leading space after colon)
                    Self::whitelist(logits, &self.digit_tokens, self.newline_token);
                    false
                }
            }
            FieldKind::Caption => {
                self.value_token_count += 1;
                let argmax = Self::argmax(logits);
                if argmax == self.newline_token || self.value_token_count >= MAX_CAPTION_TOKENS {
                    self.field_idx += 1;
                    self.seed_next_field_prefix();
                    true
                } else {
                    // Block newline only
                    if (self.newline_token as usize) < logits.len() {
                        logits[self.newline_token as usize] = f32::NEG_INFINITY;
                    }
                    false
                }
            }
        }
    }

    fn argmax(logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }

    /// Keep only `allowed` tokens + space in logits, set everything else to -inf.
    fn whitelist(logits: &mut [f32], allowed: &[u32], _newline: u32) {
        // Build a quick bitset for allowed tokens
        let mut allowed_set = vec![false; logits.len()];
        for &t in allowed {
            if (t as usize) < logits.len() {
                allowed_set[t as usize] = true;
            }
        }
        // Also allow space (U+0020 = token for " ")
        // Space is often merged with the next token, but allowing the standalone
        // space token handles the leading space after ": ".
        for i in 0..logits.len() {
            if !allowed_set[i] {
                logits[i] = f32::NEG_INFINITY;
            }
        }
    }
}

/// Constrained decoding for audio code generation.
///
/// During the codes phase, only audio code tokens and EOS are allowed.
/// Duration is enforced by blocking EOS until the target number of codes
/// is reached, then forcing EOS.
struct AudioCodeConstraint {
    audio_code_start: u32,
    audio_code_end: u32,
    eos_token_id: u32,
    target_codes: usize,
    codes_count: usize,
}

impl AudioCodeConstraint {
    fn new(token_ids: &TokenIds, target_codes: usize) -> Self {
        Self {
            audio_code_start: token_ids.audio_code_start,
            audio_code_end: token_ids.audio_code_end,
            eos_token_id: token_ids.eos,
            target_codes,
            codes_count: 0,
        }
    }

    /// Apply constraints to logits. Called via `LogitsProcessor::sample_f`.
    fn apply(&self, logits: &mut [f32]) {
        // Only allow audio code tokens + EOS (after target is reached)
        for (i, v) in logits.iter_mut().enumerate() {
            let i = i as u32;
            let is_audio_code = i >= self.audio_code_start && i <= self.audio_code_end;
            let is_eos = i == self.eos_token_id;
            if is_audio_code {
                // Always allow audio codes
            } else if is_eos && self.codes_count >= self.target_codes {
                // Allow EOS only after target reached — model decides when to stop
            } else {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    /// Record that an audio code token was generated.
    fn record_token(&mut self, token_id: u32) {
        if token_id >= self.audio_code_start && token_id <= self.audio_code_end {
            self.codes_count += 1;
        }
    }
}

/// LM pipeline wrapping a Qwen3 causal LM for audio code generation.
pub struct LmPipeline {
    model: qwen3::ModelForCausalLM,
    token_ids: TokenIds,
    device: Device,
}

impl LmPipeline {
    /// Create from a pre-loaded model and discovered token IDs.
    pub fn new(model: qwen3::ModelForCausalLM, token_ids: TokenIds, device: Device) -> Self {
        Self {
            model,
            token_ids,
            device,
        }
    }

    /// Format the chat prompt for the LM.
    ///
    /// Returns the formatted string. The caller must tokenize it.
    ///
    /// Format: `<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n`
    pub fn format_prompt(caption: &str, lyrics: &str) -> String {
        let system = format!("# Instruction\n{DEFAULT_LM_INSTRUCTION}\n\n");
        let lyrics_section = if lyrics.is_empty() {
            "[Instrumental]".to_string()
        } else {
            lyrics.to_string()
        };
        let user = format!("# Caption\n{caption}\n\n# Lyric\n{lyrics_section}\n");
        format!(
            "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        )
    }

    /// Format the codes-phase prompt: original prompt + CoT output + newline.
    pub fn format_codes_prompt(caption: &str, lyrics: &str, cot_text: &str) -> String {
        let system = format!("# Instruction\n{DEFAULT_LM_INSTRUCTION}\n\n");
        let lyrics_section = if lyrics.is_empty() {
            "[Instrumental]".to_string()
        } else {
            lyrics.to_string()
        };
        let user = format!("# Caption\n{caption}\n\n# Lyric\n{lyrics_section}\n");
        format!(
            "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{cot_text}\n"
        )
    }

    /// Run two-phase generation: CoT (metadata) + audio codes.
    ///
    /// `prompt_tokens`: tokenized CoT prompt (from `format_prompt`)
    /// `decode_fn`: converts a single token ID to its string representation
    /// `encode_fn`: tokenizes a string without special tokens
    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        target_duration_secs: f64,
        config: &LmConfig,
        decode_fn: &dyn Fn(u32) -> Result<String>,
        encode_fn: &dyn Fn(&str) -> Result<Vec<u32>>,
    ) -> Result<LmOutput> {
        let target_codes = (target_duration_secs * CODES_PER_SECOND).ceil() as usize;
        // Allow LM to generate up to 2x target (it decides when to stop),
        // but cap at max_code_tokens to avoid runaway generation
        let max_codes = config.max_code_tokens.min(target_codes * 2 + 50);

        // ---- Phase 1: CoT generation with field-order constraints ----
        self.model.clear_kv_cache();
        let mut logits_processor = self.make_logits_processor(config);
        let mut field_forcer = CotFieldForcer::new(encode_fn)?;

        let mut all_tokens: Vec<u32> = prompt_tokens.to_vec();
        let mut generated_cot_tokens: Vec<u32> = Vec::new();

        // Process prompt token-by-token to build KV cache incrementally
        // (avoids large attention matrix issues on Metal)
        let mut next_token = 0u32;
        for (pos, &token) in prompt_tokens.iter().enumerate() {
            let input = Tensor::new(&[token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, pos)?;
            let logits = Self::sanitize_logits(logits)?;
            next_token = logits_processor.sample(&logits)?;
        }

        // Force <think> as first generated token
        if next_token != self.token_ids.think_start {
            next_token = self.token_ids.think_start;
        }
        all_tokens.push(next_token);
        generated_cot_tokens.push(next_token);
        field_forcer.activate();

        // Generate until </think> or max tokens.
        //
        // The field forcer alternates between two modes:
        // 1. **Forced**: inject pre-tokenized field prefixes (e.g. "\nbpm:")
        //    and enumerated values (keyscale, language, timesig)
        // 2. **Free**: LM generates the field value with per-field constraints:
        //    - Numeric (bpm, duration): only digit tokens allowed
        //    - Caption: all tokens except `\n`
        //    - Enumerated: greedy pick from pre-tokenized valid values
        //    Transition when argmax=`\n` or token limit reached.
        for _ in 1..config.max_cot_tokens {
            // Always feed the previous token to advance KV cache
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, all_tokens.len() - 1)?;

            if let Some(forced) = field_forcer.next_forced_token() {
                // Forced mode: discard logits, use the predetermined token
                next_token = forced;
            } else {
                // Free mode: apply sanitize + field constraint, then sample
                let logits = Self::sanitize_and_constrain(logits, |l| {
                    field_forcer.constrain_value_logits(l);
                })?;

                // If constrain_value_logits triggered a transition, a forced
                // token is now available — use it instead of sampling.
                if let Some(forced) = field_forcer.next_forced_token() {
                    next_token = forced;
                } else {
                    next_token = logits_processor.sample(&logits)?;
                }
            }

            all_tokens.push(next_token);
            generated_cot_tokens.push(next_token);

            if next_token == self.token_ids.think_end {
                break;
            }
        }

        // Decode CoT text and parse metadata
        let cot_text = self.decode_tokens(&generated_cot_tokens, decode_fn)?;
        let metadata = Self::parse_metadata(&cot_text);

        // ---- Phase 2: Audio codes generation ----
        // Re-feed all tokens (prompt + CoT + newline) to build fresh KV cache
        self.model.clear_kv_cache();

        let newline_tokens = encode_fn("\n")?;
        let mut full_tokens = all_tokens.clone();
        full_tokens.extend_from_slice(&newline_tokens);
        let prompt_len = full_tokens.len();

        let mut constraint = AudioCodeConstraint::new(&self.token_ids, target_codes);
        let mut logits_processor = self.make_logits_processor(config);

        // Process all-but-last to build KV cache, then last token to get logits
        for (pos, &token) in full_tokens[..prompt_len - 1].iter().enumerate() {
            let input = Tensor::new(&[token], &self.device)?.unsqueeze(0)?;
            let _ = self.model.forward(&input, pos)?;
        }
        let last_input = Tensor::new(&[full_tokens[prompt_len - 1]], &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&last_input, prompt_len - 1)?;
        let logits = Self::sanitize_and_constrain(logits, |l| constraint.apply(l))?;

        let mut next_token = logits_processor.sample(&logits)?;
        constraint.record_token(next_token);

        let mut code_tokens: Vec<u32> = vec![next_token];
        let mut pos = prompt_len;

        for _ in 1..max_codes {
            if next_token == self.token_ids.eos {
                break;
            }

            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, pos)?;
            let logits = Self::sanitize_and_constrain(logits, |l| constraint.apply(l))?;

            next_token = logits_processor.sample(&logits)?;
            constraint.record_token(next_token);
            code_tokens.push(next_token);
            pos += 1;
        }

        // Extract audio code values from token IDs
        let audio_codes: Vec<i64> = code_tokens
            .iter()
            .filter(|&&t| {
                t >= self.token_ids.audio_code_start && t <= self.token_ids.audio_code_end
            })
            .map(|&t| (t - self.token_ids.audio_code_start) as i64)
            .collect();

        let codes_text = self.decode_tokens(&code_tokens, decode_fn)?;

        Ok(LmOutput {
            metadata,
            audio_codes,
            raw_text: format!("{cot_text}\n{codes_text}"),
        })
    }

    /// Parse metadata from CoT text between `<think>` and `</think>`.
    pub fn parse_metadata(text: &str) -> BTreeMap<String, String> {
        let mut metadata = BTreeMap::new();

        // Extract content between <think> and </think>
        let inner = if let Some(start) = text.find("<think>") {
            let after_tag = &text[start + 7..];
            if let Some(end) = after_tag.find("</think>") {
                &after_tag[..end]
            } else {
                after_tag
            }
        } else {
            text
        };

        let mut current_key: Option<String> = None;
        let mut current_value_lines: Vec<String> = Vec::new();

        let save_field =
            |key: &Option<String>, lines: &[String], map: &mut BTreeMap<String, String>| {
                if let Some(k) = key {
                    let value = lines.join("\n").trim().to_string();
                    if !value.is_empty() {
                        map.insert(k.clone(), value);
                    }
                }
            };

        for line in inner.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('<') {
                continue;
            }

            // New field: non-indented line with ':'
            if !line.starts_with(' ') && !line.starts_with('\t') && line.contains(':') {
                save_field(&current_key, &current_value_lines, &mut metadata);
                if let Some((k, v)) = line.split_once(':') {
                    current_key = Some(k.trim().to_lowercase());
                    current_value_lines = vec![v.to_string()];
                }
            } else if current_key.is_some() {
                // Continuation line
                current_value_lines.push(line.to_string());
            }
        }
        save_field(&current_key, &current_value_lines, &mut metadata);

        metadata
    }

    /// Squeeze model output to 1D, pull to CPU as F32, and sanitize NaN/Inf.
    ///
    /// Metal can produce NaN/Inf for large-vocab matmuls. We pull to CPU,
    /// replace bad values with -inf (zero probability after softmax), and
    /// return a clean F32 tensor on CPU for sampling.
    fn sanitize_logits(logits: Tensor) -> Result<Tensor> {
        let logits = logits.squeeze(0)?.squeeze(0)?;
        let n = logits.dim(0)?;
        let mut data = logits.to_dtype(candle::DType::F32)?.to_vec1::<f32>()?;
        for v in data.iter_mut() {
            if !v.is_finite() {
                *v = f32::NEG_INFINITY;
            }
        }
        Tensor::from_vec(data, n, &candle::Device::Cpu)
    }

    /// Like `sanitize_logits` but also applies a constraint function to the
    /// logit values before creating the tensor. This ensures constraints
    /// operate on logits (before softmax), not probabilities.
    fn sanitize_and_constrain(
        logits: Tensor,
        constrain: impl FnOnce(&mut [f32]),
    ) -> Result<Tensor> {
        let logits = logits.squeeze(0)?.squeeze(0)?;
        let n = logits.dim(0)?;
        let mut data = logits.to_dtype(candle::DType::F32)?.to_vec1::<f32>()?;
        for v in data.iter_mut() {
            if !v.is_finite() {
                *v = f32::NEG_INFINITY;
            }
        }
        constrain(&mut data);
        Tensor::from_vec(data, n, &candle::Device::Cpu)
    }

    fn make_logits_processor(&self, config: &LmConfig) -> LogitsProcessor {
        let sampling = if config.temperature <= 0.0 {
            Sampling::ArgMax
        } else {
            match (config.top_k, config.top_p) {
                (Some(k), Some(p)) => Sampling::TopKThenTopP {
                    k,
                    p,
                    temperature: config.temperature,
                },
                (Some(k), None) => Sampling::TopK {
                    k,
                    temperature: config.temperature,
                },
                (None, Some(p)) => Sampling::TopP {
                    p,
                    temperature: config.temperature,
                },
                (None, None) => Sampling::All {
                    temperature: config.temperature,
                },
            }
        };
        LogitsProcessor::from_sampling(config.seed, sampling)
    }

    fn decode_tokens(
        &self,
        tokens: &[u32],
        decode_fn: &dyn Fn(u32) -> Result<String>,
    ) -> Result<String> {
        let mut text = String::new();
        for &t in tokens {
            text.push_str(&decode_fn(t)?);
        }
        Ok(text)
    }
}

/// Parse audio code values from a text string containing `<|audio_code_N|>` tokens.
///
/// Returns code values in `[0, 63999]`, clamped to valid range.
pub fn parse_audio_codes_from_text(text: &str) -> Vec<i64> {
    let mut codes = Vec::new();
    let mut remaining = text;
    while let Some(start) = remaining.find("<|audio_code_") {
        let after_prefix = &remaining[start + 13..]; // skip "<|audio_code_"
        if let Some(end) = after_prefix.find("|>") {
            let num_str = &after_prefix[..end];
            if let Ok(val) = num_str.parse::<i64>() {
                codes.push(val.clamp(0, MAX_AUDIO_CODE as i64));
            }
            remaining = &after_prefix[end + 2..];
        } else {
            break;
        }
    }
    codes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_metadata() {
        let text = "<think>\nbpm: 120\ncaption: A calm piano melody\nduration: 30\ngenres: Classical\nkeyscale: C major\nlanguage: en\ntimesignature: 4\n</think>";
        let meta = LmPipeline::parse_metadata(text);
        assert_eq!(meta.get("bpm").unwrap().trim(), "120");
        assert_eq!(meta.get("caption").unwrap().trim(), "A calm piano melody");
        assert_eq!(meta.get("duration").unwrap().trim(), "30");
        assert_eq!(meta.get("keyscale").unwrap().trim(), "C major");
        assert_eq!(meta.get("language").unwrap().trim(), "en");
        assert_eq!(meta.get("timesignature").unwrap().trim(), "4");
    }

    #[test]
    fn test_parse_metadata_multiline_caption() {
        let text = "<think>\nbpm: 90\ncaption: A long description\n  that spans multiple lines\nduration: 60\n</think>";
        let meta = LmPipeline::parse_metadata(text);
        assert!(meta.get("caption").unwrap().contains("multiple lines"));
        assert_eq!(meta.get("bpm").unwrap().trim(), "90");
        assert_eq!(meta.get("duration").unwrap().trim(), "60");
    }

    #[test]
    fn test_parse_audio_codes_from_text() {
        let text = "<|audio_code_100|><|audio_code_200|><|audio_code_63999|>";
        let codes = parse_audio_codes_from_text(text);
        assert_eq!(codes, vec![100, 200, 63999]);
    }

    #[test]
    fn test_parse_audio_codes_clamp() {
        let text = "<|audio_code_99999|><|audio_code_0|>";
        let codes = parse_audio_codes_from_text(text);
        assert_eq!(codes, vec![63999, 0]);
    }

    #[test]
    fn test_parse_audio_codes_empty() {
        assert!(parse_audio_codes_from_text("no codes here").is_empty());
        assert!(parse_audio_codes_from_text("").is_empty());
    }

    #[test]
    fn test_format_prompt() {
        let prompt = LmPipeline::format_prompt("calm piano", "hello world");
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("Generate audio semantic tokens"));
        assert!(prompt.contains("calm piano"));
        assert!(prompt.contains("hello world"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_format_prompt_no_lyrics() {
        let prompt = LmPipeline::format_prompt("calm piano", "");
        assert!(prompt.contains("[Instrumental]"));
    }

    #[test]
    fn test_audio_code_constraint() {
        let token_ids = TokenIds {
            eos: 10,
            think_start: 20,
            think_end: 21,
            audio_code_start: 100,
            audio_code_end: 199,
        };
        let mut constraint = AudioCodeConstraint::new(&token_ids, 3);

        // Before target: only audio codes allowed
        let mut logits = vec![1.0f32; 200];
        constraint.apply(&mut logits);
        assert_eq!(logits[0], f32::NEG_INFINITY); // non-audio blocked
        assert_eq!(logits[10], f32::NEG_INFINITY); // EOS blocked
        assert_eq!(logits[100], 1.0); // audio code allowed
        assert_eq!(logits[150], 1.0); // audio code allowed

        // Generate 3 codes
        constraint.record_token(100);
        constraint.record_token(101);
        constraint.record_token(102);

        // At target: EOS now allowed alongside audio codes (model decides)
        let mut logits = vec![1.0f32; 200];
        constraint.apply(&mut logits);
        assert_eq!(logits[10], 1.0); // EOS allowed
        assert_eq!(logits[100], 1.0); // audio codes still allowed
        assert_eq!(logits[0], f32::NEG_INFINITY); // non-audio still blocked
    }

    #[test]
    fn test_cot_field_forcer_numeric_field() -> Result<()> {
        let encode_fn = |s: &str| -> Result<Vec<u32>> { Ok(s.bytes().map(|b| b as u32).collect()) };
        let mut forcer = CotFieldForcer::new(&encode_fn)?;
        let newline = b'\n' as u32;

        assert!(forcer.next_forced_token().is_none());
        forcer.activate();

        // Drain forced "\nbpm:"
        let mut forced = Vec::new();
        while let Some(t) = forcer.next_forced_token() {
            forced.push(t);
        }
        let forced_str: String = forced.iter().map(|&t| t as u8 as char).collect();
        assert_eq!(forced_str, "\nbpm:");

        // bpm is Numeric: digits should be allowed, non-digits blocked
        let mut logits = vec![1.0f32; 256];
        logits[b'1' as usize] = 10.0; // argmax = '1'
        let transitioned = forcer.constrain_value_logits(&mut logits);
        assert!(!transitioned);
        // 'A' should be blocked (not a digit or space)
        assert_eq!(logits[b'A' as usize], f32::NEG_INFINITY);
        // '1' should still be allowed
        assert!(logits[b'1' as usize] > f32::NEG_INFINITY);

        // argmax = newline → transition
        let mut logits = vec![0.0f32; 256];
        logits[newline as usize] = 10.0;
        assert!(forcer.constrain_value_logits(&mut logits));

        // Next should be "\ncaption:" prefix
        let mut forced = Vec::new();
        while let Some(t) = forcer.next_forced_token() {
            forced.push(t);
        }
        let forced_str: String = forced.iter().map(|&t| t as u8 as char).collect();
        assert_eq!(forced_str, "\ncaption:");

        Ok(())
    }

    #[test]
    fn test_cot_field_forcer_enumerated_field() -> Result<()> {
        let encode_fn = |s: &str| -> Result<Vec<u32>> { Ok(s.bytes().map(|b| b as u32).collect()) };
        let mut forcer = CotFieldForcer::new(&encode_fn)?;
        let newline = b'\n' as u32;
        forcer.activate();

        // Skip bpm, caption, duration (Numeric/Caption fields)
        for _ in 0..3 {
            while forcer.next_forced_token().is_some() {} // drain prefix
            let mut logits = vec![0.0f32; 256];
            logits[newline as usize] = 10.0;
            forcer.constrain_value_logits(&mut logits); // transition
        }

        // Now at keyscale (Enumerated). Drain prefix "\nkeyscale:"
        let mut forced = Vec::new();
        while let Some(t) = forcer.next_forced_token() {
            forced.push(t);
        }
        let forced_str: String = forced.iter().map(|&t| t as u8 as char).collect();
        assert_eq!(forced_str, "\nkeyscale:");

        // Enumerated: constrain_value_logits picks the best valid value.
        // Set logit for space+C tokens high (to bias toward "C major" or similar)
        let mut logits = vec![0.0f32; 256];
        logits[b' ' as usize] = 10.0; // " C major\n" starts with space
        let transitioned = forcer.constrain_value_logits(&mut logits);
        assert!(transitioned);

        // force_queue should now contain " <value>\n" + next field prefix
        let mut forced = Vec::new();
        while let Some(t) = forcer.next_forced_token() {
            forced.push(t);
        }
        let forced_str: String = forced.iter().map(|&t| t as u8 as char).collect();
        // Should contain a valid keyscale value + newline + "language:" prefix
        assert!(
            forced_str.contains("major") || forced_str.contains("minor"),
            "expected keyscale value, got: {forced_str:?}"
        );
        assert!(
            forced_str.contains("language:"),
            "expected next field prefix, got: {forced_str:?}"
        );

        Ok(())
    }

    #[test]
    fn test_cot_field_forcer_completes_all_fields() -> Result<()> {
        let encode_fn = |s: &str| -> Result<Vec<u32>> { Ok(s.bytes().map(|b| b as u32).collect()) };
        let mut forcer = CotFieldForcer::new(&encode_fn)?;
        let newline = b'\n' as u32;
        forcer.activate();

        // Run through all fields until done
        let mut total_forced = Vec::new();
        for _ in 0..2000 {
            if forcer.done {
                break;
            }
            if let Some(t) = forcer.next_forced_token() {
                total_forced.push(t);
            } else {
                // Free generation — simulate argmax = newline to transition
                let mut logits = vec![0.0f32; 256];
                logits[newline as usize] = 10.0;
                forcer.constrain_value_logits(&mut logits);
            }
        }

        assert!(forcer.done, "forcer should be done after all fields");
        let text: String = total_forced.iter().map(|&t| t as u8 as char).collect();
        // All field names should appear in order
        assert!(text.contains("bpm:"), "missing bpm in: {text:?}");
        assert!(text.contains("caption:"), "missing caption in: {text:?}");
        assert!(text.contains("duration:"), "missing duration in: {text:?}");
        assert!(text.contains("keyscale:"), "missing keyscale in: {text:?}");
        assert!(text.contains("language:"), "missing language in: {text:?}");
        assert!(
            text.contains("timesignature:"),
            "missing timesignature in: {text:?}"
        );

        Ok(())
    }
}
