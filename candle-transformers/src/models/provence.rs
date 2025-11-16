use candle::{Device, Module, Result, Tensor};
use candle_nn::{Dropout, VarBuilder};

use crate::models::debertav2::{Config, DebertaV2ContextPooler, DebertaV2Model, StableDropout};

// https://huggingface.co/naver/provence-reranker-debertav3-v1/blob/421f9139ad3f5ed919d9b04dd4ff02c10301dac9/modeling_provence.py
#[derive(Debug, Clone)]
pub struct ProvenceOutput {
    pub compression_logits: Tensor,
    pub ranking_scores: Tensor,
    pub hidden_states: Option<Vec<Tensor>>,
    pub attentions: Option<Vec<Tensor>>,
}

pub struct ProvenceModel {
    pub device: Device,
    deberta: DebertaV2Model,
    pooler: DebertaV2ContextPooler,
    dropout: StableDropout,
    classifier: candle_nn::Linear,
    token_dropout: Dropout,
    token_classifier: candle_nn::Linear,
}

pub mod config {
    pub const SEPARATOR_TOKEN: &str = "[SEP]";
}

impl ProvenceModel {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        // TODO: okay to hardcode?
        // For ranking (single score)
        let num_labels = 1;

        let deberta = DebertaV2Model::load(vb.clone(), config)?;
        let pooler = DebertaV2ContextPooler::load(vb.clone(), config)?;
        let output_dim = pooler.output_dim()?;

        let base_dropout = config.cls_dropout.unwrap_or(config.hidden_dropout_prob);

        // RANKING LAYER (on pooled output)
        let dropout = StableDropout::new(base_dropout);
        let classifier = candle_nn::linear(output_dim, num_labels, vb.root().pp("classifier"))?;

        // COMPRESSION LAYER (on token embeddings)
        let token_dropout = Dropout::new(base_dropout as f32);
        let token_classifier =
            candle_nn::linear(config.hidden_size, 1, vb.root().pp("classifier"))?;

        Ok(Self {
            device: vb.device().clone(),
            deberta,
            pooler,
            dropout,
            classifier,
            token_dropout,
            token_classifier,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<Tensor>,
    ) -> Result<ProvenceOutput> {
        let encoder_layer = self.deberta.forward(input_ids, None, attention_mask)?;

        // Ranking
        let pooled_output = self.pooler.forward(&encoder_layer)?;
        let pooled_output = self.dropout.forward(&pooled_output)?;
        let ranking_logits = self.classifier.forward(&pooled_output)?;
        let ranking_scores = ranking_logits.squeeze(1)?;

        // Compression
        let token_output = self.token_dropout.forward(&encoder_layer, false)?;
        let compression_logits = self.token_classifier.forward(&token_output)?;

        Ok(ProvenceOutput {
            compression_logits,
            ranking_scores,
            // TODO: implement
            hidden_states: None,
            attentions: None,
        })
    }

    pub fn format_input(&self, question: &str, context: &str) -> String {
        format!("{} {} {}", question, config::SEPARATOR_TOKEN, context)
    }
}

#[cfg(feature = "provence-process")]
pub mod process {
    use std::fmt;

    use candle::{Context, Error, Result, Tensor};
    use candle_nn::ops::sigmoid;
    use tokenizers::{Encoding, Tokenizer};

    use super::{
        config,
        sentence_rounding::{sentence_rounding, split_sentences_and_track_from_encoding},
        ProvenceModel, ProvenceOutput,
    };

    pub type InputEncodingResult = (Encoding, Tensor, Tensor);

    #[derive(Debug, Clone)]
    pub struct ProcessedResult {
        pub pruned_context: String,
        pub reranking_score: f32,
        pub compression_rate: f32,
        pub token_details: Option<Vec<TokenDetail>>,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum TokenStatus {
        QuestionOrSpecial,
        Kept,
        Dropped,
    }

    impl fmt::Display for TokenStatus {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let s = match self {
                TokenStatus::QuestionOrSpecial => "KEEP (Q/SPECIAL)",
                TokenStatus::Kept => "KEEP",
                TokenStatus::Dropped => "DROP",
            };
            write!(f, "{}", s)
        }
    }

    #[derive(Debug, Clone)]
    pub struct TokenDetail {
        pub index: usize,
        pub token: String,
        pub probability: f32,
        pub status: TokenStatus,
    }

    impl ProvenceModel {
        /// Process a single query-context pair with sentence-level rounding
        pub fn process_single(
            &self,
            tokenizer: &Tokenizer,
            question: &str,
            context: &str,
            threshold: f32,
            always_select_first: bool,
            include_token_details: bool,
        ) -> Result<ProcessedResult> {
            let input_text = self.format_input(question, context);

            let (encoding, input_ids, attention_mask) =
                self.encode_input(tokenizer, &input_text)?;

            let tokens = encoding.get_ids();

            let separator_index = self
                .get_separator_index(&encoding)
                .context("separator token missing")?;

            let output = self.forward(&input_ids, Some(attention_mask))?;

            let ranking_scores = output.ranking_scores.to_vec1::<f32>()?;
            let reranking_score = ranking_scores
                .first()
                .copied()
                .context("ranking_scores was empty")?;

            let context_start_byte = question.len() + 1 + config::SEPARATOR_TOKEN.len() + 1;
            let (_sentence_texts, sentence_coords) =
                split_sentences_and_track_from_encoding(context, &encoding, context_start_byte)?;

            let keep_probs = self.get_compression_probabilities(&output)?;
            let keep_mask = sentence_rounding(
                &keep_probs,
                &sentence_coords,
                threshold,
                always_select_first,
            )?;

            let (kept_token_ids, _removed_token_ids) =
                self.group_context_tokens(tokens, separator_index, &keep_mask);

            let pruned_context = tokenizer
                .decode(&kept_token_ids, true)
                .map_err(|e| Error::msg(format!("Decoding failed: {}", e)))?;

            let compression_rate = self.calculate_compression_rate(context, &pruned_context);

            let token_details = if include_token_details {
                let mut token_details = Vec::new();
                for (i, token_id) in tokens.iter().enumerate() {
                    let token_str = tokenizer.decode(&[*token_id], false).unwrap_or_default();
                    let prob = keep_probs[i];

                    let status = if i <= separator_index {
                        TokenStatus::QuestionOrSpecial
                    } else if prob > threshold {
                        TokenStatus::Kept
                    } else {
                        TokenStatus::Dropped
                    };

                    token_details.push(TokenDetail {
                        index: i,
                        token: token_str,
                        probability: prob,
                        status,
                    });
                }

                Some(token_details)
            } else {
                None
            };

            Ok(ProcessedResult {
                pruned_context,
                reranking_score,
                compression_rate,
                token_details,
            })
        }

        pub fn encode_input(
            &self,
            tokenizer: &Tokenizer,
            input_text: &str,
        ) -> Result<InputEncodingResult> {
            let encoding = tokenizer
                .encode(input_text, true)
                .map_err(|e| Error::msg(format!("Tokenization failed: {}", e)))?;

            let input_ids = Tensor::new(encoding.get_ids(), &self.device)?.unsqueeze(0)?;

            let attention_mask =
                Tensor::new(encoding.get_attention_mask(), &self.device)?.unsqueeze(0)?;

            Ok((encoding, input_ids, attention_mask))
        }

        pub fn get_compression_probabilities(&self, output: &ProvenceOutput) -> Result<Vec<f32>> {
            let compression_logits = output.compression_logits.squeeze(2)?;
            let compression_probs = sigmoid(&compression_logits)?;
            let keep_probs_vec = compression_probs.to_vec2::<f32>()?[0].clone();

            Ok(keep_probs_vec)
        }

        pub fn get_separator_index(&self, encoding: &Encoding) -> Option<usize> {
            encoding
                .get_tokens()
                .iter()
                .position(|t| t == config::SEPARATOR_TOKEN)
        }

        pub fn group_context_tokens(
            &self,
            tokens: &[u32],
            separator_index: usize,
            keep_mask: &[bool],
        ) -> (Vec<u32>, Vec<u32>) {
            let mut kept_token_ids = Vec::new();
            let mut removed_token_ids = Vec::new();

            for (i, &token_id) in tokens.iter().enumerate() {
                if i <= separator_index {
                    // Always keep question and separator
                    continue;
                } else if keep_mask.get(i).copied().unwrap_or(false) {
                    kept_token_ids.push(token_id);
                } else {
                    removed_token_ids.push(token_id);
                }
            }

            (kept_token_ids, removed_token_ids)
        }

        pub fn calculate_compression_rate(&self, context: &str, pruned_context: &str) -> f32 {
            if !context.is_empty() {
                (1.0 - pruned_context.len() as f32 / context.len() as f32) * 100.0
            } else {
                0.0
            }
        }
    }
}

#[cfg(feature = "provence-process")]
pub mod sentence_rounding {
    use candle::{bail, Result};
    use tokenizers::Encoding;

    pub type SplitAndTrimResult = Result<(Vec<String>, Vec<(usize, usize)>)>;
    pub type TrimRangeResult = Result<Option<(usize, usize, String)>>;

    pub mod config {
        pub const SENTENCE_ENDING: &[char] = &['.', '!', '?'];
    }

    /// Represents a sentence boundary in the token sequence
    #[derive(Debug, Copy, Clone)]
    pub struct SentenceCoord {
        pub start: usize,
        pub end: usize,
    }

    /// Apply sentence-level rounding to token predictions.
    ///
    /// A sentence is kept if the mean probability of its tokens exceeds `threshold`
    /// If always_select_first is true, the first sentence is kept (only when
    /// another sentence would be kept).
    pub fn sentence_rounding(
        token_predictions: &[f32],
        sentence_coords: &[SentenceCoord],
        threshold: f32,
        always_select_first: bool,
    ) -> Result<Vec<bool>> {
        let n_tokens = token_predictions.len();

        // Strict validation: require at least one sentence and valid coords
        if sentence_coords.is_empty() {
            bail!("sentence_coords is empty");
        }

        for coord in sentence_coords {
            if coord.start >= coord.end {
                bail!(
                    "invalid SentenceCoord: start ({}) >= end ({})",
                    coord.start,
                    coord.end
                );
            }

            if coord.end > n_tokens {
                bail!(
                    "invalid SentenceCoord: end ({}) > token_predictions.len() ({})",
                    coord.end,
                    n_tokens
                );
            }
        }

        let mut sentence_means: Vec<f32> = Vec::with_capacity(sentence_coords.len());

        for coord in sentence_coords {
            let slice = &token_predictions[coord.start..coord.end];

            let (sum, count) = slice.iter().copied().fold((0.0f32, 0usize), |(s, c), v| {
                if v.is_nan() {
                    (s, c)
                } else {
                    (s + v, c + 1)
                }
            });

            let mean = if count == 0 {
                0.0
            } else {
                sum / (count as f32)
            };
            sentence_means.push(mean);
        }

        if always_select_first
            && sentence_means.len() > 1
            && sentence_means
                .iter()
                .enumerate()
                .any(|(i, &m)| i != 0 && m > threshold)
        {
            sentence_means[0] = 1.0;
        }

        let mut keep_mask = vec![false; n_tokens];
        for (coord, &mean) in sentence_coords.iter().zip(sentence_means.iter()) {
            if mean > threshold {
                for v in &mut keep_mask[coord.start..coord.end] {
                    *v = true;
                }
            }
        }

        Ok(keep_mask)
    }

    /// Split context into sentences (simple splitter) but return token index coords
    /// relative to the full encoding token indices by using encoding offsets.
    ///
    /// Kept the same signature, but split into helpers for clarity and performance.
    pub fn split_sentences_and_track_from_encoding(
        context: &str,
        encoding: &Encoding,
        context_start_byte: usize,
    ) -> Result<(Vec<String>, Vec<SentenceCoord>)> {
        let (sentences, sentence_ranges_rel) = split_and_trim_sentences(context)?;

        let offsets = encoding.get_offsets();
        let coords = map_ranges_to_token_coords(offsets, &sentence_ranges_rel, context_start_byte);

        Ok((sentences, coords))
    }

    /// Split `context` into trimmed sentences and return
    /// trimmed sentences and byte ranges relative to context
    fn split_and_trim_sentences(context: &str) -> SplitAndTrimResult {
        let mut sentences: Vec<String> = Vec::new();
        let mut ranges: Vec<(usize, usize)> = Vec::new();

        let mut current_start = 0usize;

        for (i, ch) in context.char_indices() {
            let is_sentence_ending = config::SENTENCE_ENDING.contains(&ch);

            if is_sentence_ending {
                let next_i = i + ch.len_utf8();

                // if next char is whitespace or we are at EOF, treat as sentence boundary
                if next_i >= context.len()
                    || context[next_i..]
                        .chars()
                        .next()
                        .map(|c| c.is_whitespace())
                        .unwrap_or(false)
                {
                    if let Some((trim_s, trim_e, s)) = trim_range(context, current_start, next_i)? {
                        if !s.is_empty() {
                            sentences.push(s);
                            ranges.push((trim_s, trim_e));
                        }
                    }
                    current_start = next_i;
                }
            }
        }

        if current_start < context.len() {
            if let Some((trim_s, trim_e, s)) = trim_range(context, current_start, context.len())? {
                if !s.is_empty() {
                    sentences.push(s);
                    ranges.push((trim_s, trim_e));
                }
            }
        }

        Ok((sentences, ranges))
    }

    /// Trim a byte range `[start, end)` within `context` (both byte indices relative to `context`) and
    /// return trim_start_rel, trim_end_rel, trimmed_string or None if trimmed string is empty.
    fn trim_range(context: &str, start: usize, end: usize) -> TrimRangeResult {
        // Validate bounds and char boundaries to avoid panics from slicing
        if start > end || end > context.len() {
            bail!(
                "trim_range: invalid range [{}, {}) for context len {}",
                start,
                end,
                context.len()
            );
        }
        if !context.is_char_boundary(start) || !context.is_char_boundary(end) {
            bail!(
                "trim_range: start or end not on char boundary: start={} end={}",
                start,
                end
            );
        }

        let slice = &context[start..end];

        let (lead_idx, _) = match slice.char_indices().find(|(_, c)| !c.is_whitespace()) {
            Some(v) => v,
            None => return Ok(None),
        };

        let (trail_idx, trail_ch) = slice
            .char_indices()
            .rev()
            .find(|(_, c)| !c.is_whitespace())
            .expect("we already found a non-whitespace above; qed");

        let trail_end = trail_idx + trail_ch.len_utf8();

        let trim_start = start + lead_idx;
        let trim_end = start + trail_end;

        Ok(Some((
            trim_start,
            trim_end,
            context[trim_start..trim_end].to_string(),
        )))
    }

    /// Map trimmed byte ranges (relative to `context`) into token coordinate ranges using `offsets`.
    /// offsets: slice of (token_byte_start, token_byte_end) for the full input.
    fn map_ranges_to_token_coords(
        offsets: &[(usize, usize)],
        sentence_ranges_rel: &[(usize, usize)],
        context_start_byte: usize,
    ) -> Vec<SentenceCoord> {
        let mut coords: Vec<SentenceCoord> = Vec::with_capacity(sentence_ranges_rel.len());
        let mut token_idx = 0usize;
        let n_tokens = offsets.len();

        for &(start_rel, end_rel) in sentence_ranges_rel {
            let start_abs = context_start_byte + start_rel;
            let end_abs = context_start_byte + end_rel;

            // advance token_idx to the first token that could overlap (skip tokens entirely before sentence)
            while token_idx < n_tokens {
                let (tok_s, tok_e) = offsets[token_idx];
                // skip tokens with no offsets (common for special tokens)
                if tok_s == 0 && tok_e == 0 {
                    token_idx += 1;
                    continue;
                }

                // if token ends before or at sentence start, keep advancing
                if tok_e <= start_abs {
                    token_idx += 1;
                    continue;
                }

                // otherwise this token may overlap; stop advancing
                break;
            }

            // collect overlapping tokens starting from token_idx
            let mut first_token: Option<usize> = None;
            let mut last_token: Option<usize> = None;
            let mut j = token_idx;

            while j < n_tokens {
                let (tok_s, tok_e) = offsets[j];
                if tok_s == 0 && tok_e == 0 {
                    j += 1;
                    continue;
                }

                // if this token starts at or after sentence end, we're done
                if tok_s >= end_abs {
                    break;
                }

                // overlap test: token end > start && token start < end  (half-open intervals)
                if tok_e > start_abs && tok_s < end_abs {
                    if first_token.is_none() {
                        first_token = Some(j);
                    }
                    last_token = Some(j);
                }
                j += 1;
            }

            // advance token_idx for next sentence to j (no backtracking)
            token_idx = j;

            if let (Some(first), Some(last)) = (first_token, last_token) {
                coords.push(SentenceCoord {
                    start: first,
                    end: last + 1, // end exclusive
                });
            } else {
                // no overlapping tokens found — skip; keeping behavior similar to original
            }
        }

        coords
    }
}

// Re-export for convenience
#[cfg(feature = "provence-process")]
pub use process::ProcessedResult;
