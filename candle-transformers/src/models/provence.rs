use candle::{Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Dropout, VarBuilder};

use crate::models::debertav2::{
    id2label_len, Config, DebertaV2ContextPooler, DebertaV2Model, Id2Label, StableDropout,
};

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
    pub fn load(vb: VarBuilder, config: &Config, id2label: Option<Id2Label>) -> Result<Self> {
        let id2label_len = id2label_len(config, id2label)?;

        let deberta = DebertaV2Model::load(vb.clone(), config)?;
        let pooler = DebertaV2ContextPooler::load(vb.clone(), config)?;
        let output_dim = pooler.output_dim()?;

        let base_dropout = config.cls_dropout.unwrap_or(config.hidden_dropout_prob);

        // RANKING LAYER (on pooled output)
        let dropout = StableDropout::new(base_dropout);
        let classifier = candle_nn::linear(output_dim, id2label_len, vb.root().pp("classifier"))?;

        // COMPRESSION LAYER (on token embeddings)
        let token_dropout = Dropout::new(base_dropout as f32);
        let token_classifier =
            candle_nn::linear(config.hidden_size, 2, vb.root().pp("token_classifier"))?;

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
        let ranking_scores = ranking_logits.i((.., 0))?;

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

    pub fn format_input(question: &str, context: &str) -> String {
        format!("{} {} {}", question, config::SEPARATOR_TOKEN, context)
    }
}

#[cfg(feature = "provence-process")]
pub mod process {
    use std::fmt;

    use candle::{Context, Error, IndexOp, Result, Tensor};
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
            // TODO: check python implementation
            let normalize_question = true;

            let input_text;
            let context_start_byte;

            if normalize_question {
                let normalized_question = Self::normalize_string(question);

                input_text = Self::format_input(&normalized_question, context);
                context_start_byte =
                    normalized_question.len() + 1 + config::SEPARATOR_TOKEN.len() + 1;
            } else {
                input_text = Self::format_input(question, context);
                context_start_byte = question.len() + 1 + config::SEPARATOR_TOKEN.len() + 1;
            };

            let (encoding, input_ids, attention_mask) =
                self.encode_input(tokenizer, &input_text)?;

            let tokens = encoding.get_ids();

            let separator_index =
                Self::get_separator_index(&encoding).context("separator token missing")?;

            let output = self.forward(&input_ids, Some(attention_mask))?;

            let ranking_scores = output.ranking_scores.to_vec1::<f32>()?;
            let reranking_score = ranking_scores
                .first()
                .copied()
                .context("ranking_scores was empty")?;

            let (_sentence_texts, sentences_token_coords) =
                split_sentences_and_track_from_encoding(context, &encoding, context_start_byte)?;

            let keep_probs = Self::get_keep_probabilities(&output)?;
            let keep_mask = sentence_rounding(
                &keep_probs,
                &sentences_token_coords,
                threshold,
                always_select_first,
            )?;

            let (kept_token_ids, _removed_token_ids) =
                Self::group_context_tokens(tokens, separator_index, &keep_mask);

            let pruned_context = tokenizer
                .decode(&kept_token_ids, true)
                .map_err(|e| Error::msg(format!("Decoding failed: {}", e)))?;

            let compression_rate = Self::calculate_compression_rate(context, &pruned_context);

            let token_details = if include_token_details {
                let token_details = Self::build_token_details(
                    encoding.get_tokens(),
                    separator_index,
                    &keep_mask,
                    &keep_probs,
                );

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

        fn normalize_string(text: &str) -> String {
            let lower_no_punctuation: String = text
                .to_lowercase()
                .chars()
                .filter(|c| !c.is_ascii_punctuation())
                .collect();

            let normalized_white_space = lower_no_punctuation
                .split_whitespace()
                .collect::<Vec<&str>>()
                .join(" ");

            normalized_white_space
        }

        pub fn get_separator_index(encoding: &Encoding) -> Option<usize> {
            encoding
                .get_tokens()
                .iter()
                .position(|t| t == config::SEPARATOR_TOKEN)
        }

        pub fn get_keep_probabilities(output: &ProvenceOutput) -> Result<Vec<f32>> {
            let compression_logits = output.compression_logits.squeeze(0)?;
            let compression_probs = candle_nn::ops::softmax(&compression_logits, 1)?;

            let keep_probs = compression_probs.i((.., 1))?;
            let keep_probs_vec = keep_probs.to_vec1::<f32>()?;

            Ok(keep_probs_vec)
        }

        pub fn group_context_tokens(
            tokens: &[u32],
            separator_index: usize,
            keep_mask: &[bool],
        ) -> (Vec<u32>, Vec<u32>) {
            let mut kept_token_ids = Vec::new();
            let mut removed_token_ids = Vec::new();

            for (i, &token_id) in tokens.iter().enumerate().skip(separator_index + 1) {
                if keep_mask.get(i).copied().unwrap_or(false) {
                    kept_token_ids.push(token_id);
                } else {
                    removed_token_ids.push(token_id);
                }
            }

            (kept_token_ids, removed_token_ids)
        }

        pub fn calculate_compression_rate(context: &str, pruned_context: &str) -> f32 {
            if !context.is_empty() {
                (1.0 - pruned_context.len() as f32 / context.len() as f32) * 100.0
            } else {
                0.0
            }
        }

        pub fn build_token_details(
            token_strings: &[String],
            separator_index: usize,
            keep_mask: &[bool],
            keep_probs: &[f32],
        ) -> Vec<TokenDetail> {
            let mut token_details = Vec::with_capacity(token_strings.len());

            for (i, token_string) in token_strings.iter().enumerate() {
                let prob = keep_probs[i];

                let status = if i <= separator_index {
                    TokenStatus::QuestionOrSpecial
                } else if keep_mask.get(i).copied().unwrap_or(false) {
                    TokenStatus::Kept
                } else {
                    TokenStatus::Dropped
                };

                token_details.push(TokenDetail {
                    index: i,
                    token: token_string.clone(),
                    probability: prob,
                    status,
                });
            }

            token_details
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

    /// Represents a boundary in a sequence
    #[derive(Debug, Copy, Clone)]
    pub struct Coordinate {
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
        sentences_token_coords: &[Coordinate],
        threshold: f32,
        always_select_first: bool,
    ) -> Result<Vec<bool>> {
        let n_tokens = token_predictions.len();

        if sentences_token_coords.is_empty() {
            bail!("sentences_token_coords is empty");
        }

        let mut sentence_means: Vec<f32> = Vec::with_capacity(sentences_token_coords.len());

        for coord in sentences_token_coords {
            if coord.start >= coord.end {
                bail!(
                    "invalid Coordinate: start ({}) >= end ({})",
                    coord.start,
                    coord.end
                );
            }

            if coord.end > n_tokens {
                bail!(
                    "invalid Coordinate: end ({}) > token_predictions.len() ({})",
                    coord.end,
                    n_tokens
                );
            }

            let token_coords = &token_predictions[coord.start..coord.end];

            let (tokens_sum, tokens_count) =
                token_coords
                    .iter()
                    .copied()
                    .fold((0.0, 0), |(sum, count), token_keep_prob| {
                        if token_keep_prob.is_nan() {
                            (sum, count)
                        } else {
                            (sum + token_keep_prob, count + 1)
                        }
                    });

            let mean = if tokens_count == 0 {
                0.0
            } else {
                tokens_sum / (tokens_count as f32)
            };

            sentence_means.push(mean);
        }

        if always_select_first
            && sentence_means.len() > 1
            && sentence_means
                .iter()
                .enumerate()
                .any(|(index, &mean)| index != 0 && mean > threshold)
        {
            sentence_means[0] = 1.0;
        }

        let mut keep_mask = vec![false; n_tokens];

        for (coord, &mean) in sentences_token_coords.iter().zip(sentence_means.iter()) {
            if mean > threshold {
                for keep_token in &mut keep_mask[coord.start..coord.end] {
                    *keep_token = true;
                }
            }
        }

        Ok(keep_mask)
    }

    /// Split context into sentences (simple splitter) but return token index coords
    /// relative to the full encoding token indices by using encoding offsets.
    pub fn split_sentences_and_track_from_encoding(
        context: &str,
        encoding: &Encoding,
        context_start_byte: usize,
    ) -> Result<(Vec<String>, Vec<Coordinate>)> {
        let offsets = encoding.get_offsets();

        let (sentences, sentence_ranges_rel_to_context) = split_and_trim_sentences(context)?;

        let sentences_token_coords = map_ranges_to_token_coords(
            offsets,
            &sentence_ranges_rel_to_context,
            context_start_byte,
        );

        Ok((sentences, sentences_token_coords))
    }

    /// Split `context` into trimmed sentences and return
    /// trimmed sentences and byte ranges relative to context
    fn split_and_trim_sentences(context: &str) -> SplitAndTrimResult {
        let mut sentences: Vec<String> = Vec::new();
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        let mut current_start = 0;

        // TODO use something like https://crates.io/crates/punkt?
        for (i, ch) in context.char_indices() {
            let is_sentence_ending = config::SENTENCE_ENDING.contains(&ch);

            if is_sentence_ending {
                let next_i = i + ch.len_utf8();

                // if next char is whitespace or we are at EOF, treat as sentence boundary
                if next_i >= context.len()
                    || context[next_i..]
                        .chars()
                        .next()
                        .map(|next_ch| next_ch.is_whitespace())
                        .unwrap_or(false)
                {
                    if let Some((trim_start, trim_end, sentence)) =
                        trim_range(context, current_start, next_i)?
                    {
                        if !sentence.is_empty() {
                            sentences.push(sentence);
                            ranges.push((trim_start, trim_end));
                        }
                    }

                    current_start = next_i;
                }
            }
        }

        // Handle any leftovers
        // Last part of the text that doesn't end with a punctuation mark
        if current_start < context.len() {
            if let Some((trim_start, trim_end, sentence)) =
                trim_range(context, current_start, context.len())?
            {
                if !sentence.is_empty() {
                    sentences.push(sentence);
                    ranges.push((trim_start, trim_end));
                }
            }
        }

        Ok((sentences, ranges))
    }

    /// Trim a byte range `[start, end)` within `context` (both byte indices relative to `context`) and
    /// return trim_start_rel, trim_end_rel, trimmed_string or None if trimmed string is empty.
    fn trim_range(context: &str, start: usize, end: usize) -> TrimRangeResult {
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

        let context_slice = &context[start..end];

        let (leading_index, _) = match context_slice
            .char_indices()
            .find(|(_, ch)| !ch.is_whitespace())
        {
            Some(leading_index_ch) => leading_index_ch,
            None => return Ok(None),
        };

        let (trailing_index, trailing_ch) = match context_slice
            .char_indices()
            .rev()
            .find(|(_, ch)| !ch.is_whitespace())
        {
            Some(trail_index_ch) => trail_index_ch,
            None => return Ok(None),
        };

        let trailing_end = trailing_index + trailing_ch.len_utf8();

        let trim_start = start + leading_index;
        let trim_end = start + trailing_end;

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
        sentence_ranges_rel_to_context: &[(usize, usize)],
        context_start_byte: usize,
    ) -> Vec<Coordinate> {
        let mut sentences_token_coords: Vec<Coordinate> =
            Vec::with_capacity(sentence_ranges_rel_to_context.len());

        let mut token_index_sentence_start = 0;

        let n_tokens = offsets.len();

        for &(sentence_start, sentence_end) in sentence_ranges_rel_to_context {
            let sentence_start_abs = context_start_byte + sentence_start;
            let sentence_end_abs = context_start_byte + sentence_end;

            // advance token_index_sentence_start to the first token that could be in the sentence range
            // (skip tokens entirely before sentence)
            while token_index_sentence_start < n_tokens {
                let (token_start, token_end) = offsets[token_index_sentence_start];
                // skip tokens with no offsets (common for special tokens)
                if token_start == 0 && token_end == 0 {
                    token_index_sentence_start += 1;
                    continue;
                }

                // if token ends before or at sentence start, keep advancing
                if token_end <= sentence_start_abs {
                    token_index_sentence_start += 1;
                    continue;
                }

                // otherwise this token may be in range; stop advancing
                break;
            }

            // collect in_range tokens starting from token_index
            let mut first_token_index: Option<usize> = None;
            let mut last_token_index: Option<usize> = None;
            let mut token_index_in_sentence = token_index_sentence_start;

            while token_index_in_sentence < n_tokens {
                let (token_start, token_end) = offsets[token_index_in_sentence];
                // skip tokens with no offsets (common for special tokens)
                if token_start == 0 && token_end == 0 {
                    token_index_in_sentence += 1;
                    continue;
                }

                // if this token starts at or after sentence end, break
                if token_start >= sentence_end_abs {
                    break;
                }

                if overlaps(
                    (sentence_start_abs, sentence_end_abs),
                    (token_start, token_end),
                ) {
                    if first_token_index.is_none() {
                        first_token_index = Some(token_index_in_sentence);
                    }

                    last_token_index = Some(token_index_in_sentence);
                }

                token_index_in_sentence += 1;
            }

            // advance token_index for next sentence (no backtracking)
            token_index_sentence_start = token_index_in_sentence;

            if let (Some(first), Some(last)) = (first_token_index, last_token_index) {
                sentences_token_coords.push(Coordinate {
                    start: first,
                    end: last + 1, // end exclusive
                });
            } else {
                // no overlapping tokens found — skip; keeping behavior similar to original
            }
        }

        sentences_token_coords
    }

    // A and B overlap if
    // B doesn't end before A starts, AND
    // B doesn't start after A ends.
    fn overlaps(a: (usize, usize), b: (usize, usize)) -> bool {
        b.1 > a.0 && b.0 < a.1
    }
}

// Re-export for convenience
#[cfg(feature = "provence-process")]
pub use process::ProcessedResult;
