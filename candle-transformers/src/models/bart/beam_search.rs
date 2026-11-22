use candle::{Device, IndexOp, Result, Tensor};
use std::collections::HashSet;

use crate::models::bart::{config::BartConfig, generation::BartForConditionalGeneration};

/// A hypothesis with its tokens and score.
pub type ScoredHypothesis = (Vec<u32>, f32);

/// Results from batched beam search: for each input in the batch, a list of scored hypotheses.
pub type BatchedBeamSearchResult = Vec<Vec<ScoredHypothesis>>;

/// Single hypothesis in the beam.
#[derive(Clone, Debug)]
struct Hypothesis {
    /// Token sequence including start tokens.
    tokens: Vec<u32>,
    /// Cumulative log probability.
    log_prob: f32,
    /// Is sequence finished (generated EOS)?
    finished: bool,
}

impl Hypothesis {
    fn new(start_tokens: Vec<u32>) -> Self {
        Self {
            tokens: start_tokens,
            log_prob: 0.0,
            finished: false,
        }
    }

    /// Length-normalized score for ranking beams.
    /// Formula from Google NMT (Wu et al. 2016):
    /// score = log_prob / ((5 + length)^α / 6^α)
    /// Simplified: log_prob / length^α when α is the main parameter
    fn score(&self, length_penalty: f64, num_start_tokens: usize) -> f32 {
        let length = (self.tokens.len() - num_start_tokens).max(1) as f64;
        self.log_prob / length.powf(length_penalty) as f32
    }

    /// Check if n-gram would repeat.
    fn would_repeat_ngram(&self, new_token: u32, n: usize) -> bool {
        if n == 0 || self.tokens.len() < n {
            return false;
        }

        // Build new sequence with candidate token
        let mut test_seq = self.tokens.clone();
        test_seq.push(new_token);

        // Check if last n-gram appears earlier
        let last_ngram = &test_seq[test_seq.len() - n..];
        for i in 0..test_seq.len() - n {
            if &test_seq[i..i + n] == last_ngram {
                return true;
            }
        }
        false
    }
}

pub struct BeamSearchConfig {
    /// Beam size for beam search decoding. When > 1, uses beam search instead of sampling.
    beam_size: usize,
    /// Length penalty α for beam search. Higher values (1.5-2.0) favor longer outputs.
    /// Wu et al. 2016 formula: score = log_prob / (length^α)
    length_penalty: f64,
    /// Minimum generation length before EOS is allowed (beam search only).
    min_length: usize,
    /// Block n-gram repetition (0 = disabled, beam search only).
    no_repeat_ngram_size: usize,
    /// The number of tokens to generate.
    sample_len: usize,
}

impl BeamSearchConfig {
    pub fn new(
        beam_size: usize,
        length_penalty: f64,
        min_length: usize,
        no_repeat_ngram_size: usize,
        sample_len: usize,
    ) -> Self {
        BeamSearchConfig {
            beam_size,
            length_penalty,
            min_length,
            no_repeat_ngram_size,
            sample_len,
        }
    }
}

// ============================================================================
// Batched Beam Search Support
// ============================================================================

/// External KV cache for batched decoding.
/// Manages KV states for all layers, supporting beam reordering.
/// Shape convention: (batch_size * beam_size, num_heads, seq_len, head_dim)
#[derive(Debug, Clone)]
pub struct BatchedKVCache {
    /// Self-attention KV cache per layer.
    /// Shape: (batch_beams, num_heads, seq_len, head_dim)
    self_attn_cache: Vec<Option<(Tensor, Tensor)>>,
    /// Cross-attention KV cache per layer (computed once from encoder, then reordered).
    /// Shape: (batch_beams, num_heads, encoder_seq_len, head_dim)
    cross_attn_cache: Vec<Option<(Tensor, Tensor)>>,
    /// Number of decoder layers.
    num_layers: usize,
}

impl BatchedKVCache {
    /// Create new empty cache for given number of layers.
    pub fn new(num_layers: usize) -> Self {
        Self {
            self_attn_cache: vec![None; num_layers],
            cross_attn_cache: vec![None; num_layers],
            num_layers,
        }
    }

    /// Reset all caches (for new batch).
    pub fn reset(&mut self) {
        self.self_attn_cache = vec![None; self.num_layers];
        self.cross_attn_cache = vec![None; self.num_layers];
    }

    /// Reorder cache entries according to beam indices.
    /// Called after beam selection to align cache with surviving beams.
    ///
    /// # Arguments
    /// * `beam_indices` - Tensor of shape (batch_size * beam_size,) containing
    ///   source beam indices for each position
    pub fn reorder_cache(&mut self, beam_indices: &Tensor) -> Result<()> {
        // Reorder self-attention cache
        for (k, v) in self.self_attn_cache.iter_mut().flatten() {
            // index_select on batch dimension (dim 0), then contiguous for efficient layout
            *k = k.index_select(beam_indices, 0)?.contiguous()?;
            *v = v.index_select(beam_indices, 0)?.contiguous()?;
        }

        // Reorder cross-attention cache
        for (k, v) in self.cross_attn_cache.iter_mut().flatten() {
            *k = k.index_select(beam_indices, 0)?.contiguous()?;
            *v = v.index_select(beam_indices, 0)?.contiguous()?;
        }

        Ok(())
    }

    /// Get current sequence length from self-attention cache.
    pub fn get_past_kv_len(&self) -> usize {
        self.self_attn_cache
            .first()
            .and_then(|c| c.as_ref())
            .map(|(k, _)| k.dim(2).unwrap_or(0))
            .unwrap_or(0)
    }

    /// Get self-attention cache for a specific layer.
    pub fn get_self_attn(&self, layer_idx: usize) -> Option<&(Tensor, Tensor)> {
        self.self_attn_cache.get(layer_idx).and_then(|c| c.as_ref())
    }

    /// Set self-attention cache for a specific layer.
    pub fn set_self_attn(&mut self, layer_idx: usize, kv: (Tensor, Tensor)) {
        if layer_idx < self.num_layers {
            self.self_attn_cache[layer_idx] = Some(kv);
        }
    }

    /// Get cross-attention cache for a specific layer.
    pub fn get_cross_attn(&self, layer_idx: usize) -> Option<&(Tensor, Tensor)> {
        self.cross_attn_cache
            .get(layer_idx)
            .and_then(|c| c.as_ref())
    }

    /// Set cross-attention cache for a specific layer.
    pub fn set_cross_attn(&mut self, layer_idx: usize, kv: (Tensor, Tensor)) {
        if layer_idx < self.num_layers {
            self.cross_attn_cache[layer_idx] = Some(kv);
        }
    }

    /// Take self-attention cache for a specific layer (removes from cache).
    pub fn take_self_attn(&mut self, layer_idx: usize) -> Option<(Tensor, Tensor)> {
        if layer_idx < self.num_layers {
            self.self_attn_cache[layer_idx].take()
        } else {
            None
        }
    }

    /// Take cross-attention cache for a specific layer (removes from cache).
    pub fn take_cross_attn(&mut self, layer_idx: usize) -> Option<(Tensor, Tensor)> {
        if layer_idx < self.num_layers {
            self.cross_attn_cache[layer_idx].take()
        } else {
            None
        }
    }
}

/// Early stopping behavior for batched beam search.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum EarlyStoppingMode {
    /// Never stop early, always generate to max_length.
    Never,
    /// Stop batch item when beam_size hypotheses finish.
    #[default]
    WhenComplete,
}

/// Configuration for batched beam search.
#[derive(Debug, Clone)]
pub struct BatchedBeamSearchConfig {
    /// Beam size for beam search decoding.
    pub beam_size: usize,
    /// Length penalty α for beam search. Higher values favor longer outputs.
    /// Wu et al. 2016 formula: score = log_prob / (length^α)
    pub length_penalty: f64,
    /// Minimum generation length before EOS is allowed.
    pub min_length: usize,
    /// Maximum generation length.
    pub max_length: usize,
    /// Block n-gram repetition (0 = disabled).
    pub no_repeat_ngram_size: usize,
    /// Early stopping behavior.
    pub early_stopping: EarlyStoppingMode,
    /// Number of hypotheses to return per batch item.
    pub num_return_sequences: usize,
}

impl BatchedBeamSearchConfig {
    pub fn new(
        beam_size: usize,
        length_penalty: f64,
        min_length: usize,
        max_length: usize,
        no_repeat_ngram_size: usize,
    ) -> Self {
        Self {
            beam_size,
            length_penalty,
            min_length,
            max_length,
            no_repeat_ngram_size,
            early_stopping: EarlyStoppingMode::default(),
            num_return_sequences: 1,
        }
    }

    pub fn with_early_stopping(mut self, mode: EarlyStoppingMode) -> Self {
        self.early_stopping = mode;
        self
    }

    pub fn with_num_return_sequences(mut self, n: usize) -> Self {
        self.num_return_sequences = n;
        self
    }
}

pub fn beam_search(
    model: &mut BartForConditionalGeneration,
    encoder_output: &Tensor,
    config: &BartConfig,
    beam_search_config: &BeamSearchConfig,
    device: &candle::Device,
) -> Result<Vec<u32>> {
    let beam_size = beam_search_config.beam_size;
    let mut beams = vec![Hypothesis::new(config.initial_decoder_tokens())];
    let num_start_tokens = beams[0].tokens.len();

    let mut finished_beams: Vec<Hypothesis> = Vec::new();

    for _step in 0..beam_search_config.sample_len {
        let mut candidates: Vec<Hypothesis> = Vec::new();

        for beam in &beams {
            if beam.finished {
                finished_beams.push(beam.clone());
                continue;
            }

            // For beam search, we must process the full sequence each time
            // because beam indices change and we can't reuse the KV cache
            model.reset_kv_cache();
            let decoder_input = &beam.tokens[..];
            let input_tensor = Tensor::new(decoder_input, device)?.unsqueeze(0)?;

            // Get logits for next token (past_kv_len=0 since we reset cache)
            let logits = model.decode(&input_tensor, encoder_output, 0)?;
            let logits = logits.squeeze(0)?.get(logits.dim(1)? - 1)?;

            // Convert to log probabilities
            let log_probs = candle_nn::ops::softmax_last_dim(&logits)?;
            let log_probs = log_probs.log()?;
            let log_probs_vec = log_probs.to_vec1::<f32>()?;

            // Get top-k candidates (k = beam_size for expansion)
            let mut token_scores: Vec<(usize, f32)> = log_probs_vec
                .iter()
                .enumerate()
                .map(|(i, &score)| (i, score))
                .collect();
            token_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Expand beam with top candidates
            for &(token_id, log_prob_token) in token_scores.iter().take(beam_size) {
                let token = token_id as u32;

                // Enforce minimum length (block EOS)
                let current_len = beam.tokens.len() - num_start_tokens;
                let is_eos =
                    token == config.eos_token_id || config.forced_eos_token_id == Some(token);

                if is_eos && current_len < beam_search_config.min_length {
                    continue; // Skip EOS tokens before minimum length
                }

                // Block n-gram repetition
                if beam_search_config.no_repeat_ngram_size > 0
                    && beam.would_repeat_ngram(token, beam_search_config.no_repeat_ngram_size)
                {
                    continue;
                }

                let mut new_hyp = beam.clone();
                new_hyp.tokens.push(token);
                new_hyp.log_prob += log_prob_token;
                new_hyp.finished = is_eos;

                candidates.push(new_hyp);
            }
        }

        // Handle edge case: all candidates filtered out
        if candidates.is_empty() {
            break;
        }

        // Select top beam_size candidates by score
        candidates.sort_by(|a, b| {
            b.score(beam_search_config.length_penalty, num_start_tokens)
                .partial_cmp(&a.score(beam_search_config.length_penalty, num_start_tokens))
                .unwrap()
        });
        beams = candidates.into_iter().take(beam_size).collect();

        // Early stopping: all beams finished
        if beams.iter().all(|b| b.finished) {
            break;
        }
    }

    // Combine finished and unfinished beams
    finished_beams.extend(beams.into_iter().filter(|b| !b.finished));

    // Return best hypothesis
    finished_beams.sort_by(|a, b| {
        b.score(beam_search_config.length_penalty, num_start_tokens)
            .partial_cmp(&a.score(beam_search_config.length_penalty, num_start_tokens))
            .unwrap()
    });

    Ok(finished_beams
        .first()
        .ok_or_else(|| candle::Error::Msg("No beams generated".to_string()))?
        .tokens
        .clone())
}

// ============================================================================
// Batched Beam Search Implementation
// ============================================================================

/// Expand encoder output for beam search.
/// Uses expand() for memory efficiency (view, not copy).
fn expand_encoder_output(encoder_output: &Tensor, beam_size: usize) -> Result<Tensor> {
    // encoder_output: (batch_size, seq_len, hidden_dim)
    // Target: (batch_size * beam_size, seq_len, hidden_dim)
    let (batch_size, seq_len, hidden_dim) = encoder_output.dims3()?;

    // Reshape to (batch_size, 1, seq_len, hidden_dim)
    let expanded = encoder_output.unsqueeze(1)?;

    // Expand to (batch_size, beam_size, seq_len, hidden_dim) - uses view, no copy
    let expanded = expanded.expand((batch_size, beam_size, seq_len, hidden_dim))?;

    // Reshape to (batch_size * beam_size, seq_len, hidden_dim)
    expanded.reshape((batch_size * beam_size, seq_len, hidden_dim))
}

/// Get tokens that would create a repeated n-gram if appended.
/// Returns a HashSet of blocked token IDs.
fn get_blocked_tokens(tokens: &[u32], n: usize) -> HashSet<u32> {
    if n == 0 || tokens.len() < n {
        return HashSet::new();
    }

    let mut blocked = HashSet::new();
    let prefix_len = n - 1;

    // Get current prefix (last n-1 tokens)
    if tokens.len() < prefix_len {
        return blocked;
    }
    let current_prefix = &tokens[tokens.len() - prefix_len..];

    // Find all n-grams that start with this prefix
    for i in 0..=tokens.len().saturating_sub(n) {
        if i + prefix_len <= tokens.len() && &tokens[i..i + prefix_len] == current_prefix {
            // If there's a token after this prefix, block it
            if i + n - 1 < tokens.len() {
                blocked.insert(tokens[i + n - 1]);
            }
        }
    }

    blocked
}

/// Compute length-normalized score for beam search.
fn compute_score(log_prob: f32, length: usize, length_penalty: f64) -> f32 {
    if length_penalty == 0.0 {
        return log_prob;
    }
    let length_factor = (length as f64).powf(length_penalty) as f32;
    log_prob / length_factor.max(1e-8)
}

/// Batched hypothesis tracking for beam search.
#[derive(Debug, Clone)]
struct BatchedHypotheses {
    /// Token IDs: [batch][beam][seq]
    tokens: Vec<Vec<Vec<u32>>>,
    /// Log probabilities: [batch][beam]
    log_probs: Vec<Vec<f32>>,
    /// Finished flags: [batch][beam]
    finished: Vec<Vec<bool>>,
    /// Number of initial tokens (for length penalty calculation)
    num_start_tokens: usize,
}

impl BatchedHypotheses {
    fn new(batch_size: usize, beam_size: usize, initial_tokens: &[u32]) -> Self {
        let num_start_tokens = initial_tokens.len();
        let tokens = vec![vec![initial_tokens.to_vec(); beam_size]; batch_size];
        let log_probs = vec![vec![0.0f32; beam_size]; batch_size];
        let finished = vec![vec![false; beam_size]; batch_size];

        Self {
            tokens,
            log_probs,
            finished,
            num_start_tokens,
        }
    }

    /// Get length (excluding start tokens) for a specific hypothesis.
    fn get_length(&self, batch_idx: usize, beam_idx: usize) -> usize {
        self.tokens[batch_idx][beam_idx]
            .len()
            .saturating_sub(self.num_start_tokens)
    }

    /// Compute score for a specific hypothesis.
    fn score(&self, batch_idx: usize, beam_idx: usize, length_penalty: f64) -> f32 {
        let length = self.get_length(batch_idx, beam_idx).max(1);
        compute_score(self.log_probs[batch_idx][beam_idx], length, length_penalty)
    }
}

/// Batched beam search for BART models.
///
/// Processes all batch_size × beam_size hypotheses in parallel with efficient
/// KV cache reuse and reordering.
///
/// # Arguments
/// * `model` - The BART model (immutable reference)
/// * `encoder_output` - Encoder hidden states, shape (batch_size, enc_seq_len, hidden_dim)
/// * `config` - BART model configuration
/// * `beam_config` - Beam search configuration
/// * `device` - Device for tensor operations
///
/// # Returns
/// Vec of hypotheses per batch item. Each hypothesis is (tokens, score).
/// Sorted by score (best first).
pub fn batched_beam_search(
    model: &BartForConditionalGeneration,
    encoder_output: &Tensor,
    config: &BartConfig,
    beam_config: &BatchedBeamSearchConfig,
    device: &Device,
) -> Result<BatchedBeamSearchResult> {
    let batch_size = encoder_output.dim(0)?;
    let beam_size = beam_config.beam_size;
    let batch_beams = batch_size * beam_size;

    // Initialize cache
    let num_layers = config.decoder_layers;
    let mut cache = BatchedKVCache::new(num_layers);

    // Expand encoder output for beam search (memory-efficient view)
    let expanded_encoder = expand_encoder_output(encoder_output, beam_size)?;

    // Initialize hypotheses
    let initial_tokens = config.initial_decoder_tokens();
    let mut hypotheses = BatchedHypotheses::new(batch_size, beam_size, &initial_tokens);

    // Track finished beams per batch item: Vec of (tokens, score)
    let mut finished_hypotheses: Vec<Vec<(Vec<u32>, f32)>> = vec![Vec::new(); batch_size];

    // Beam search loop
    for step in 0..beam_config.max_length {
        // Check early stopping
        let all_done = match beam_config.early_stopping {
            EarlyStoppingMode::WhenComplete => finished_hypotheses
                .iter()
                .all(|finished| finished.len() >= beam_config.num_return_sequences),
            EarlyStoppingMode::Never => false,
        };
        if all_done {
            break;
        }

        // Collect input tokens for this step
        let input_tensor = if step == 0 {
            // First step: process all initial tokens
            let tokens: Vec<u32> = (0..batch_beams)
                .flat_map(|i| {
                    let batch_idx = i / beam_size;
                    let beam_idx = i % beam_size;
                    hypotheses.tokens[batch_idx][beam_idx].clone()
                })
                .collect();
            let seq_len = initial_tokens.len();
            Tensor::new(tokens.as_slice(), device)?.reshape((batch_beams, seq_len))?
        } else {
            // Subsequent steps: just the last token
            let tokens: Vec<u32> = (0..batch_beams)
                .map(|i| {
                    let batch_idx = i / beam_size;
                    let beam_idx = i % beam_size;
                    *hypotheses.tokens[batch_idx][beam_idx].last().unwrap()
                })
                .collect();
            Tensor::new(tokens.as_slice(), device)?.reshape((batch_beams, 1))?
        };

        // Forward pass through decoder with external cache
        let logits = model.decode_with_cache(&input_tensor, &expanded_encoder, &mut cache)?;

        // Get logits for last position: (batch_beams, vocab_size)
        let last_pos = logits.dim(1)? - 1;
        let logits = logits.i((.., last_pos, ..))?;

        // Convert to log probabilities
        let log_probs = candle_nn::ops::log_softmax(&logits, candle::D::Minus1)?;
        let log_probs_data: Vec<f32> = log_probs.flatten_all()?.to_vec1()?;
        let vocab_size = log_probs.dim(1)?;

        // Beam selection per batch item
        let mut new_beam_indices: Vec<usize> = Vec::with_capacity(batch_beams);
        let mut new_tokens_to_add: Vec<u32> = Vec::with_capacity(batch_beams);
        let mut new_log_probs: Vec<f32> = Vec::with_capacity(batch_beams);
        let mut new_finished: Vec<bool> = Vec::with_capacity(batch_beams);

        for batch_idx in 0..batch_size {
            // Collect candidates for this batch item
            let mut candidates: Vec<(usize, u32, f32, bool)> = Vec::new(); // (src_beam, token, new_log_prob, is_finished)

            for beam_idx in 0..beam_size {
                let global_idx = batch_idx * beam_size + beam_idx;

                if hypotheses.finished[batch_idx][beam_idx] {
                    // Keep finished beam with pad token
                    candidates.push((
                        beam_idx,
                        config.pad_token_id,
                        hypotheses.log_probs[batch_idx][beam_idx],
                        true,
                    ));
                    continue;
                }

                let current_log_prob = hypotheses.log_probs[batch_idx][beam_idx];
                let current_len = hypotheses.get_length(batch_idx, beam_idx);
                let tokens = &hypotheses.tokens[batch_idx][beam_idx];

                // Get blocked tokens for n-gram repetition
                let blocked = if beam_config.no_repeat_ngram_size > 0 {
                    get_blocked_tokens(tokens, beam_config.no_repeat_ngram_size)
                } else {
                    HashSet::new()
                };

                // Get log probs for this beam
                let beam_start = global_idx * vocab_size;
                let beam_log_probs = &log_probs_data[beam_start..beam_start + vocab_size];

                // Get top-2*beam_size tokens
                let mut token_scores: Vec<(usize, f32)> = beam_log_probs
                    .iter()
                    .enumerate()
                    .filter(|(token_id, _)| !blocked.contains(&(*token_id as u32)))
                    .map(|(i, &score)| (i, score))
                    .collect();
                token_scores
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                for &(token_id, token_log_prob) in token_scores.iter().take(2 * beam_size) {
                    let token = token_id as u32;
                    let new_log_prob = current_log_prob + token_log_prob;

                    // Check if EOS
                    let is_eos =
                        token == config.eos_token_id || config.forced_eos_token_id == Some(token);

                    // Enforce minimum length
                    if is_eos && current_len < beam_config.min_length {
                        continue;
                    }

                    candidates.push((beam_idx, token, new_log_prob, is_eos));
                }
            }

            // Sort candidates by score (considering new token)
            candidates.sort_by(|a, b| {
                let len_a = hypotheses.get_length(batch_idx, a.0) + 1;
                let len_b = hypotheses.get_length(batch_idx, b.0) + 1;
                let score_a = compute_score(a.2, len_a, beam_config.length_penalty);
                let score_b = compute_score(b.2, len_b, beam_config.length_penalty);
                score_b
                    .partial_cmp(&score_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Select top beam_size candidates
            let selected: Vec<_> = candidates.into_iter().take(beam_size).collect();

            // Ensure we have beam_size candidates (pad with first if needed)
            for i in 0..beam_size {
                let (src_beam, token, log_prob, is_finished) = if i < selected.len() {
                    selected[i]
                } else if !selected.is_empty() {
                    selected[0]
                } else {
                    // Fallback: keep current beam with pad
                    (
                        0,
                        config.pad_token_id,
                        hypotheses.log_probs[batch_idx][0],
                        true,
                    )
                };

                new_beam_indices.push(batch_idx * beam_size + src_beam);
                new_tokens_to_add.push(token);
                new_log_probs.push(log_prob);
                new_finished.push(is_finished);
            }
        }

        // Reorder cache according to new beam indices
        let beam_indices_i64: Vec<i64> = new_beam_indices.iter().map(|&i| i as i64).collect();
        let beam_indices_tensor = Tensor::new(beam_indices_i64.as_slice(), device)?;
        cache.reorder_cache(&beam_indices_tensor)?;

        // Update hypotheses
        let mut new_hypotheses = BatchedHypotheses::new(batch_size, beam_size, &initial_tokens);

        for i in 0..batch_beams {
            let batch_idx = i / beam_size;
            let beam_idx = i % beam_size;
            let src_idx = new_beam_indices[i];
            let src_batch = src_idx / beam_size;
            let src_beam = src_idx % beam_size;

            // Copy tokens from source beam
            new_hypotheses.tokens[batch_idx][beam_idx] =
                hypotheses.tokens[src_batch][src_beam].clone();
            new_hypotheses.tokens[batch_idx][beam_idx].push(new_tokens_to_add[i]);

            // Update log prob and finished state
            new_hypotheses.log_probs[batch_idx][beam_idx] = new_log_probs[i];
            new_hypotheses.finished[batch_idx][beam_idx] = new_finished[i];

            // Store finished hypothesis
            if new_finished[i] && !hypotheses.finished[src_batch][src_beam] {
                let length = new_hypotheses.get_length(batch_idx, beam_idx);
                let score = compute_score(new_log_probs[i], length, beam_config.length_penalty);
                finished_hypotheses[batch_idx]
                    .push((new_hypotheses.tokens[batch_idx][beam_idx].clone(), score));
            }
        }

        hypotheses = new_hypotheses;

        // Check if all beams finished
        let all_finished = (0..batch_size).all(|batch_idx| {
            (0..beam_size).all(|beam_idx| hypotheses.finished[batch_idx][beam_idx])
        });
        if all_finished {
            break;
        }
    }

    // Collect results per batch item
    let mut results = Vec::with_capacity(batch_size);

    for (batch_idx, batch_finished) in finished_hypotheses.into_iter().enumerate() {
        // Combine finished hypotheses with any remaining active beams
        let mut all_candidates: Vec<ScoredHypothesis> = batch_finished;

        for beam_idx in 0..beam_size {
            if !hypotheses.finished[batch_idx][beam_idx] {
                let score = hypotheses.score(batch_idx, beam_idx, beam_config.length_penalty);
                all_candidates.push((hypotheses.tokens[batch_idx][beam_idx].clone(), score));
            }
        }

        // Sort by score (best first)
        all_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top num_return_sequences
        let selected: Vec<_> = all_candidates
            .into_iter()
            .take(beam_config.num_return_sequences)
            .collect();

        // Ensure at least one result
        if selected.is_empty() {
            results.push(vec![(initial_tokens.to_vec(), 0.0)]);
        } else {
            results.push(selected);
        }
    }

    Ok(results)
}
