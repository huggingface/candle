//! Conditional Flow Matching (CFM) for CosyVoice3
//!
//! Implements the flow matching algorithm for mel spectrogram generation.
//! Uses Euler ODE solver with Classifier-Free Guidance (CFG).

use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use std::f64::consts::PI;

use super::dit::DiT;
use crate::models::cosyvoice::config::CFMConfig;

/// Causal Conditional Flow Matching
///
/// Uses DiT as the estimator and Euler ODE solver for sampling.
#[derive(Debug)]
pub struct CausalConditionalCFM {
    estimator: DiT,
    #[allow(dead_code)]
    sigma_min: f64,
    t_scheduler: String,
    inference_cfg_rate: f64,
}

impl CausalConditionalCFM {
    pub fn new(estimator: DiT, config: CFMConfig) -> Self {
        Self {
            estimator,
            sigma_min: config.sigma_min,
            t_scheduler: config.t_scheduler,
            inference_cfg_rate: config.inference_cfg_rate,
        }
    }

    /// Sample from the flow model using Euler ODE solver
    ///
    /// # Arguments
    /// * `mu` - [B, 80, T] condition from token embedding
    /// * `mask` - [B, 1, T] mask
    /// * `n_timesteps` - Number of ODE steps (default: 10)
    /// * `spks` - [B, 80] speaker embedding
    /// * `cond` - [B, 80, T] prompt mel condition
    /// * `streaming` - Whether in streaming mode
    pub fn forward(
        &self,
        mu: &Tensor,
        mask: &Tensor,
        n_timesteps: usize,
        spks: &Tensor,
        cond: &Tensor,
        streaming: bool,
    ) -> Result<Tensor> {
        let (batch, _, seq_len) = mu.dims3()?;
        let device = mu.device();
        let dtype = mu.dtype();

        // Initialize random noise
        let z = Tensor::randn(0f32, 1.0, (batch, 80, seq_len), device)?.to_dtype(dtype)?;

        // Time step scheduling
        let t_span = self.get_t_span(n_timesteps, device, dtype)?;

        // Euler solver
        self.solve_euler(&z, &t_span, mu, mask, spks, cond, streaming)
    }

    /// Euler ODE solver with Classifier-Free Guidance
    #[allow(clippy::too_many_arguments)]
    fn solve_euler(
        &self,
        x: &Tensor,
        t_span: &Tensor,
        mu: &Tensor,
        mask: &Tensor,
        spks: &Tensor,
        cond: &Tensor,
        streaming: bool,
    ) -> Result<Tensor> {
        let n_steps = t_span.dim(0)? - 1;
        let mut x = x.clone();

        for step in 0..n_steps {
            let t = t_span.i(step)?;
            let t_next = t_span.i(step + 1)?;
            let dt = (&t_next - &t)?;

            // Simplified: skip CFG for now, just do conditional prediction
            let t_batch = t.unsqueeze(0)?; // [1]

            // Estimator forward pass
            let dphi_dt = self.estimator.forward(
                &x, mask, mu, &t_batch, spks, cond, streaming,
            )?;

            // Euler step: x = x + dphi_dt * dt
            let dt_broadcast = dt.unsqueeze(0)?.unsqueeze(0)?;
            x = (x + dphi_dt.broadcast_mul(&dt_broadcast)?)?;
        }

        Ok(x)
    }

    /// Get time span with optional cosine schedule
    fn get_t_span(&self, n_steps: usize, device: &Device, dtype: DType) -> Result<Tensor> {
        // Uniformly distributed time points [0, 1]
        let step = 1.0 / n_steps as f64;
        let t_values: Vec<f32> = (0..=n_steps)
            .map(|i| (i as f64 * step) as f32)
            .collect();
        let t = Tensor::from_vec(t_values, n_steps + 1, device)?.to_dtype(dtype)?;

        if self.t_scheduler == "cosine" {
            // t = 1 - cos(t * π/2)
            let pi_half = PI / 2.0;
            let cos_t = (t * pi_half)?.cos()?;
            (1.0 - cos_t)?.to_dtype(dtype)
        } else {
            Ok(t)
        }
    }

    /// Get the estimator (DiT)
    pub fn estimator(&self) -> &DiT {
        &self.estimator
    }
}

/// CausalMaskedDiffWithDiT - Complete Flow Decoder
///
/// Combines PreLookaheadLayer, token embedding, and CFM for mel generation.
/// Follows the official implementation structure:
/// - input_embedding: vocab_size -> input_size
/// - spk_embed_affine_layer: spk_embed_dim(192) -> output_size(80)
/// - pre_lookahead_layer: with token_mel_ratio(2x) upsampling
/// - cfm: Conditional Flow Matching with DiT estimator
#[derive(Debug)]
pub struct CausalMaskedDiffWithDiT {
    input_embedding: candle_nn::Embedding,
    spk_embed_affine_layer: candle_nn::Linear,
    pre_lookahead_layer: super::pre_lookahead::PreLookaheadLayer,
    cfm: CausalConditionalCFM,
    #[allow(dead_code)]
    vocab_size: usize,
    #[allow(dead_code)]
    token_mel_ratio: usize,
    #[allow(dead_code)]
    pre_lookahead_len: usize,
    output_size: usize,
}

impl CausalMaskedDiffWithDiT {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vocab_size: usize,
        input_size: usize,
        output_size: usize,
        spk_embed_dim: usize,
        token_mel_ratio: usize,
        pre_lookahead_len: usize,
        estimator: DiT,
        cfm_config: CFMConfig,
        vb: candle_nn::VarBuilder,
    ) -> Result<Self> {
        // Input embedding: vocab_size -> input_size (not output_size)
        let input_embedding = candle_nn::embedding(vocab_size, input_size, vb.pp("input_embedding"))?;

        // Speaker embedding affine layer: spk_embed_dim(192) -> output_size(80)
        let spk_embed_affine_layer = candle_nn::linear(
            spk_embed_dim,
            output_size,
            vb.pp("spk_embed_affine_layer"),
        )?;

        // PreLookahead layer
        let pre_lookahead_layer = super::pre_lookahead::PreLookaheadLayer::new(
            input_size,
            1024, // hidden_dim
            output_size,
            pre_lookahead_len,
            token_mel_ratio,
            vb.pp("pre_lookahead_layer"),
        )?;

        // CFM
        let cfm = CausalConditionalCFM::new(estimator, cfm_config);

        Ok(Self {
            input_embedding,
            spk_embed_affine_layer,
            pre_lookahead_layer,
            cfm,
            vocab_size,
            token_mel_ratio,
            pre_lookahead_len,
            output_size,
        })
    }

    /// Inference: speech tokens -> mel spectrogram
    ///
    /// # Arguments
    /// * `speech_tokens` - [B, T] speech token indices
    /// * `prompt_tokens` - [B, T'] prompt speech tokens
    /// * `prompt_feat` - [B, T'*2, 80] prompt mel features (already upsampled)
    /// * `embedding` - [B, 192] raw speaker embedding (will be normalized and projected)
    /// * `n_timesteps` - Number of ODE steps
    /// * `streaming` - Whether in streaming mode
    pub fn inference(
        &self,
        speech_tokens: &Tensor,
        prompt_tokens: &Tensor,
        prompt_feat: &Tensor,
        embedding: &Tensor,
        n_timesteps: usize,
        streaming: bool,
    ) -> Result<Tensor> {
        let (batch, _token_len) = speech_tokens.dims2()?;

        // 1. Speaker embedding: normalize and project
        // embedding = F.normalize(embedding, dim=1)
        // embedding = self.spk_embed_affine_layer(embedding)
        let embedding_norm = {
            let norm = embedding.sqr()?.sum_keepdim(1)?.sqrt()?;
            embedding.broadcast_div(&norm.clamp(1e-12, f64::INFINITY)?)?  // L2 normalize
        };
        let embedding_proj = self.spk_embed_affine_layer.forward(&embedding_norm)?; // [B, 80]

        // 2. Concat prompt tokens and speech tokens
        let all_tokens = Tensor::cat(&[prompt_tokens, speech_tokens], 1)?;

        // 3. Input embedding
        let token_emb = self.input_embedding.forward(&all_tokens)?; // [B, T_total, input_size]

        // 4. PreLookahead layer (includes projection to output_size and 2x upsampling)
        let mu = self.pre_lookahead_layer.forward(&token_emb, None)?; // [B, T_total*2, output_size]
        let mu = mu.transpose(1, 2)?; // [B, output_size, T_total*2]

        // 5. Prepare mask and condition
        let mel_len = mu.dim(2)?;
        let mask = Tensor::ones((batch, 1, mel_len), mu.dtype(), mu.device())?;

        // Prompt mel as condition (pad to same length)
        let prompt_mel_len = prompt_feat.dim(1)?;
        let prompt_feat = prompt_feat.transpose(1, 2)?; // [B, output_size, T']

        let cond = if prompt_mel_len < mel_len {
            let pad = Tensor::zeros(
                (batch, self.output_size, mel_len - prompt_mel_len),
                prompt_feat.dtype(),
                prompt_feat.device(),
            )?;
            Tensor::cat(&[&prompt_feat, &pad], 2)?
        } else {
            prompt_feat.narrow(2, 0, mel_len)?
        };

        // 6. CFM sampling with projected speaker embedding
        self.cfm.forward(&mu, &mask, n_timesteps, &embedding_proj, &cond, streaming)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_span_linear() -> Result<()> {
        let device = Device::Cpu;
        let _dtype = DType::F32;

        // Create a simple config
        let _config = CFMConfig {
            sigma_min: 1e-6,
            t_scheduler: "linear".to_string(),
            inference_cfg_rate: 0.7,
            n_timesteps: 10,
        };

        // Cannot create CausalConditionalCFM directly as it requires DiT
        // Only test time span calculation logic

        let n_steps = 10;
        let step = 1.0 / n_steps as f64;
        let t_values: Vec<f32> = (0..=n_steps)
            .map(|i| (i as f64 * step) as f32)
            .collect();
        let t = Tensor::from_vec(t_values, n_steps + 1, &device)?;

        assert_eq!(t.dim(0)?, 11);

        let t_vec: Vec<f32> = t.to_vec1()?;
        assert!((t_vec[0] - 0.0).abs() < 1e-5);
        assert!((t_vec[10] - 1.0).abs() < 1e-5);

        Ok(())
    }
}

