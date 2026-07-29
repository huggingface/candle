//! 12 Hz Mimi-based audio decoder for Qwen3-TTS.
//!
//! Converts 16-codebook codec tokens to a 24 kHz mono audio waveform.
//! The architecture is:
//!
//! 1. Dual-path RVQ decode (rvq_first + rvq_rest) → sum → pre_conv
//! 2. Mini transformer (8 layers, 512 hidden) with RoPE
//! 3. 2-stage pre-upsampling (CausalTransConv + ConvNeXtBlock, ratios 2×2)
//! 4. BigVGAN decoder blocks (upsample rates 8, 5, 4, 3)
//! 5. SnakeBeta + final causal conv → clamp to [−1, 1]

use std::collections::HashMap;

use candle::{DType, IndexOp, Module, Result, Tensor, D};
use super::{CausalConv1d, CausalTransConv1d, ConvNeXtBlock, DecoderBlock, SnakeBeta};

/// Configuration for the 12 Hz decoder (defaults match all released variants).
#[derive(Debug, Clone)]
pub struct Decoder12HzConfig {
    pub codebook_dim: usize,
    pub latent_dim: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub num_quantizers: usize,
    pub codebook_size: usize,
    pub upsampling_ratios: Vec<usize>,
    pub decoder_dim: usize,
    pub upsample_rates: Vec<usize>,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub layer_scale: f64,
}

impl Default for Decoder12HzConfig {
    fn default() -> Self {
        Self {
            codebook_dim: 512,
            latent_dim: 1024,
            hidden_size: 512,
            num_layers: 8,
            num_heads: 16,
            head_dim: 64,
            intermediate_size: 1024,
            num_quantizers: 16,
            codebook_size: 2048,
            upsampling_ratios: vec![2, 2],
            decoder_dim: 1536,
            upsample_rates: vec![8, 5, 4, 3],
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            layer_scale: 0.01,
        }
    }
}

// ── Transformer layer weights (stored as raw tensors) ──────────────────────

struct TxLayerW {
    input_ln: Tensor,
    q: Tensor, k: Tensor, v: Tensor, o: Tensor,
    attn_scale: Tensor,
    post_ln: Tensor,
    gate: Tensor, up: Tensor, down: Tensor,
    mlp_scale: Tensor,
}

// ── UpsampleStage ──────────────────────────────────────────────────────────

struct UpsampleStage {
    trans_conv: CausalTransConv1d,
    convnext: ConvNeXtBlock,
}

impl UpsampleStage {
    fn from_weights(
        tc_w: Tensor, tc_b: Tensor,
        cn_dw_w: Tensor, cn_dw_b: Tensor,
        cn_norm_w: Tensor, cn_norm_b: Tensor,
        cn_pw1_w: Tensor, cn_pw1_b: Tensor,
        cn_pw2_w: Tensor, cn_pw2_b: Tensor,
        cn_gamma: Tensor,
        stride: usize,
    ) -> Result<Self> {
        Ok(Self {
            trans_conv: CausalTransConv1d::from_weights(tc_w, Some(tc_b), stride)?,
            convnext: ConvNeXtBlock::from_weights(
                cn_dw_w, Some(cn_dw_b),
                cn_norm_w, cn_norm_b,
                cn_pw1_w, cn_pw1_b,
                cn_pw2_w, cn_pw2_b,
                cn_gamma,
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.convnext.forward(&self.trans_conv.forward(x)?)
    }
}

// ── Decoder12Hz ────────────────────────────────────────────────────────────

/// Full 12 Hz audio decoder.
pub struct Decoder12Hz {
    config: Decoder12HzConfig,
    first_codebook: Tensor,
    rest_codebooks: Vec<Tensor>,
    first_output_proj: Tensor,
    rest_output_proj: Tensor,
    pre_conv: CausalConv1d,
    input_proj_w: Tensor, input_proj_b: Tensor,
    tx_layers: Vec<TxLayerW>,
    final_norm_w: Tensor,
    output_proj_w: Tensor, output_proj_b: Tensor,
    upsample_stages: Vec<UpsampleStage>,
    decoder_init_conv: CausalConv1d,
    decoder_blocks: Vec<DecoderBlock>,
    final_snake: SnakeBeta,
    final_conv: CausalConv1d,
}

fn w(weights: &HashMap<String, Tensor>, key: &str) -> Result<Tensor> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| candle::Error::Msg(format!("Missing weight: {key}")))
}

impl Decoder12Hz {
    pub fn from_weights(weights: &HashMap<String, Tensor>, config: Decoder12HzConfig) -> Result<Self> {
        // ── Codebooks (normalize by cluster_usage) ─────────────────────────
        let eps = 1e-7f32;
        let norm_cb = |sum: Tensor, usage: Tensor| -> Result<Tensor> {
            let usage_c = usage.clamp(eps, f32::MAX)?;
            sum.broadcast_div(&usage_c.unsqueeze(1)?)
        };

        let first_codebook = norm_cb(
            w(weights, "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum")?,
            w(weights, "decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage")?,
        )?;

        let mut rest_codebooks = Vec::with_capacity(15);
        for i in 0..15 {
            let cb = norm_cb(
                w(weights, &format!("decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum"))?,
                w(weights, &format!("decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.cluster_usage"))?,
            )?;
            rest_codebooks.push(cb);
        }

        let first_output_proj = w(weights, "decoder.quantizer.rvq_first.output_proj.weight")?;
        let rest_output_proj  = w(weights, "decoder.quantizer.rvq_rest.output_proj.weight")?;

        // ── Pre-conv ────────────────────────────────────────────────────────
        let pre_conv = CausalConv1d::from_weights(
            w(weights, "decoder.pre_conv.conv.weight")?,
            Some(w(weights, "decoder.pre_conv.conv.bias")?),
            1,
        )?;

        // ── Pre-transformer projections ─────────────────────────────────────
        let input_proj_w = w(weights, "decoder.pre_transformer.input_proj.weight")?;
        let input_proj_b = w(weights, "decoder.pre_transformer.input_proj.bias")?;
        let output_proj_w = w(weights, "decoder.pre_transformer.output_proj.weight")?;
        let output_proj_b = w(weights, "decoder.pre_transformer.output_proj.bias")?;

        // ── Transformer layers ──────────────────────────────────────────────
        let mut tx_layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let p = format!("decoder.pre_transformer.layers.{i}");
            tx_layers.push(TxLayerW {
                input_ln: w(weights, &format!("{p}.input_layernorm.weight"))?,
                q: w(weights, &format!("{p}.self_attn.q_proj.weight"))?,
                k: w(weights, &format!("{p}.self_attn.k_proj.weight"))?,
                v: w(weights, &format!("{p}.self_attn.v_proj.weight"))?,
                o: w(weights, &format!("{p}.self_attn.o_proj.weight"))?,
                attn_scale: w(weights, &format!("{p}.self_attn_layer_scale.scale"))?,
                post_ln: w(weights, &format!("{p}.post_attention_layernorm.weight"))?,
                gate: w(weights, &format!("{p}.mlp.gate_proj.weight"))?,
                up:   w(weights, &format!("{p}.mlp.up_proj.weight"))?,
                down: w(weights, &format!("{p}.mlp.down_proj.weight"))?,
                mlp_scale: w(weights, &format!("{p}.mlp_layer_scale.scale"))?,
            });
        }
        let final_norm_w = w(weights, "decoder.pre_transformer.norm.weight")?;

        // ── Upsample stages ─────────────────────────────────────────────────
        let mut upsample_stages = Vec::with_capacity(config.upsampling_ratios.len());
        for (i, &stride) in config.upsampling_ratios.iter().enumerate() {
            let p = format!("decoder.upsample.{i}");
            upsample_stages.push(UpsampleStage::from_weights(
                w(weights, &format!("{p}.0.conv.weight"))?,
                w(weights, &format!("{p}.0.conv.bias"))?,
                w(weights, &format!("{p}.1.dwconv.conv.weight"))?,
                w(weights, &format!("{p}.1.dwconv.conv.bias"))?,
                w(weights, &format!("{p}.1.norm.weight"))?,
                w(weights, &format!("{p}.1.norm.bias"))?,
                w(weights, &format!("{p}.1.pwconv1.weight"))?,
                w(weights, &format!("{p}.1.pwconv1.bias"))?,
                w(weights, &format!("{p}.1.pwconv2.weight"))?,
                w(weights, &format!("{p}.1.pwconv2.bias"))?,
                w(weights, &format!("{p}.1.gamma"))?,
                stride,
            )?);
        }

        // ── Decoder init conv ────────────────────────────────────────────────
        let decoder_init_conv = CausalConv1d::from_weights(
            w(weights, "decoder.decoder.0.conv.weight")?,
            Some(w(weights, "decoder.decoder.0.conv.bias")?),
            1,
        )?;

        // ── Decoder blocks ───────────────────────────────────────────────────
        let mut decoder_blocks = Vec::with_capacity(config.upsample_rates.len());
        for (i, &rate) in config.upsample_rates.iter().enumerate() {
            let block_idx = i + 1;
            let bp = format!("decoder.decoder.{block_idx}.block");
            let load_res = |u: usize| -> Result<(Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor)> {
                let p = format!("{bp}.{u}");
                Ok((
                    w(weights, &format!("{p}.act1.alpha"))?,
                    w(weights, &format!("{p}.act1.beta"))?,
                    w(weights, &format!("{p}.conv1.conv.weight"))?,
                    w(weights, &format!("{p}.conv1.conv.bias"))?,
                    w(weights, &format!("{p}.act2.alpha"))?,
                    w(weights, &format!("{p}.act2.beta"))?,
                    w(weights, &format!("{p}.conv2.conv.weight"))?,
                    w(weights, &format!("{p}.conv2.conv.bias"))?,
                ))
            };
            let (r1a1a,r1a1b,r1c1w,r1c1b,r1a2a,r1a2b,r1c2w,r1c2b) = load_res(2)?;
            let (r2a1a,r2a1b,r2c1w,r2c1b,r2a2a,r2a2b,r2c2w,r2c2b) = load_res(3)?;
            let (r3a1a,r3a1b,r3c1w,r3c1b,r3a2a,r3a2b,r3c2w,r3c2b) = load_res(4)?;

            decoder_blocks.push(DecoderBlock::from_weights(
                w(weights, &format!("{bp}.0.alpha"))?,
                w(weights, &format!("{bp}.0.beta"))?,
                w(weights, &format!("{bp}.1.conv.weight"))?,
                w(weights, &format!("{bp}.1.conv.bias"))?,
                r1a1a,r1a1b,r1c1w,r1c1b,r1a2a,r1a2b,r1c2w,r1c2b,
                r2a1a,r2a1b,r2c1w,r2c1b,r2a2a,r2a2b,r2c2w,r2c2b,
                r3a1a,r3a1b,r3c1w,r3c1b,r3a2a,r3a2b,r3c2w,r3c2b,
                rate,
            )?);
        }

        let final_snake = SnakeBeta::from_weights(
            w(weights, "decoder.decoder.5.alpha")?,
            w(weights, "decoder.decoder.5.beta")?,
        )?;
        let final_conv = CausalConv1d::from_weights(
            w(weights, "decoder.decoder.6.conv.weight")?,
            Some(w(weights, "decoder.decoder.6.conv.bias")?),
            1,
        )?;

        Ok(Self {
            config,
            first_codebook,
            rest_codebooks,
            first_output_proj,
            rest_output_proj,
            pre_conv,
            input_proj_w, input_proj_b,
            tx_layers,
            final_norm_w,
            output_proj_w, output_proj_b,
            upsample_stages,
            decoder_init_conv,
            decoder_blocks,
            final_snake,
            final_conv,
        })
    }

    /// Decode codec tokens `[batch, 16, T]` → audio `[batch, 1, samples]`.
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let device = codes.device();
        let (batch, _nq, seq) = codes.dims3()?;
        let cb_size = self.config.codebook_size as i64;

        // ── RVQ decode ────────────────────────────────────────────────────
        // First quantizer — apply modulo (3072 → 2048)
        let first_codes = codes.i((.., 0, ..))?.flatten_all()?;
        let codes_f = first_codes.to_dtype(DType::F32)?;
        let cb_f = cb_size as f64;
        let quot = codes_f.affine(1.0 / cb_f, 0.0)?.floor()?;
        let rem = (codes_f - quot.affine(cb_f, 0.0)?)?.to_dtype(DType::I64)?;
        let first_embed = self.first_codebook.index_select(&rem, 0)?
            .reshape((batch, seq, 256))?;
        let first_proj = self.conv1d_1x1(&first_embed.transpose(1, 2)?, &self.first_output_proj)?;

        // Rest quantizers
        let mut rest_embed = Tensor::zeros((batch, seq, 256), DType::F32, device)?;
        for i in 0..15 {
            let c = codes.i((.., i + 1, ..))?.flatten_all()?;
            let e = self.rest_codebooks[i].index_select(&c, 0)?
                .reshape((batch, seq, 256))?;
            rest_embed = (rest_embed + e)?;
        }
        let rest_proj = self.conv1d_1x1(&rest_embed.transpose(1, 2)?, &self.rest_output_proj)?;

        let quantized = (first_proj + rest_proj)?; // [B, 512, T]

        // ── Pre-conv ──────────────────────────────────────────────────────
        use candle::Module;
        let hidden = self.pre_conv.forward(&quantized)?; // [B, 1024, T]

        // ── Pre-transformer ───────────────────────────────────────────────
        let hidden = hidden.transpose(1, 2)?; // [B, T, 1024]
        let hidden = self.linear_3d(&hidden, &self.input_proj_w, Some(&self.input_proj_b))?;
        let hidden = self.run_transformer(hidden, seq)?;
        let hidden = self.rms_norm(&hidden, &self.final_norm_w)?;
        let hidden = self.linear_3d(&hidden, &self.output_proj_w, Some(&self.output_proj_b))?;
        let mut hidden = hidden.transpose(1, 2)?; // [B, latent_dim, T]

        // ── Pre-upsample ──────────────────────────────────────────────────
        for stage in &self.upsample_stages {
            hidden = stage.forward(&hidden)?;
        }

        // ── Decoder ───────────────────────────────────────────────────────
        hidden = self.decoder_init_conv.forward(&hidden)?;
        for block in &self.decoder_blocks {
            hidden = block.forward(&hidden)?;
        }
        hidden = self.final_snake.forward(&hidden)?;
        hidden = self.final_conv.forward(&hidden)?;
        hidden.clamp(-1.0f32, 1.0f32)
    }

    // ── Private helpers ────────────────────────────────────────────────────

    fn conv1d_1x1(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let w2 = weight.squeeze(2)?;
        let (b, in_ch, t) = x.dims3()?;
        let x_t = x.transpose(1, 2)?.reshape((b * t, in_ch))?;
        let out = x_t.matmul(&w2.t()?)?;
        let out_ch = out.dim(1)?;
        Ok(out.reshape((b, t, out_ch))?.transpose(1, 2)?)
    }

    fn linear_3d(&self, x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        let (b, t, f) = x.dims3()?;
        let out = x.reshape((b * t, f))?.matmul(&weight.t()?)?;
        let out = out.reshape((b, t, out.dim(1)?))?;
        match bias {
            Some(b) => Ok(out.broadcast_add(b)?),
            None => Ok(out),
        }
    }

    fn rms_norm(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let var = x.sqr()?.mean_keepdim(D::Minus1)?;
        let xn = x.broadcast_div(&(var + self.config.rms_norm_eps)?.sqrt()?)?;
        Ok(xn.broadcast_mul(weight)?)
    }

    fn apply_rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let d = self.config.head_dim;
        let x1 = x.narrow(D::Minus1, 0, d / 2)?;
        let x2 = x.narrow(D::Minus1, d / 2, d / 2)?;
        let rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
        Ok((x.broadcast_mul(cos)? + rotated.broadcast_mul(sin)?)?)
    }

    fn run_transformer(&self, mut hidden: Tensor, seq: usize) -> Result<Tensor> {
        let device = hidden.device();
        let (b, _, _) = hidden.dims3()?;

        // RoPE
        let inv: Vec<f32> = (0..self.config.head_dim)
            .step_by(2)
            .map(|i| {
                1.0 / (self.config.rope_theta as f32)
                    .powf(i as f32 / self.config.head_dim as f32)
            })
            .collect();
        let inv_t = Tensor::from_vec(inv, (self.config.head_dim / 2,), device)?;
        let pos: Vec<f32> = (0..seq).map(|i| i as f32).collect();
        let pos_t = Tensor::new(pos.as_slice(), device)?.unsqueeze(1)?;
        let freqs = pos_t.matmul(&inv_t.unsqueeze(0)?)?;
        let cos = freqs.cos()?.repeat((1, 2))?.unsqueeze(0)?.unsqueeze(0)?; // [1,1,T,D]
        let sin = freqs.sin()?.repeat((1, 2))?.unsqueeze(0)?.unsqueeze(0)?;

        // Causal mask
        let mut md = vec![0.0f32; seq * seq];
        for i in 0..seq {
            for j in i + 1..seq {
                md[i * seq + j] = f32::NEG_INFINITY;
            }
        }
        let mask = Tensor::from_vec(md, (seq, seq), device)?
            .unsqueeze(0)?
            .unsqueeze(0)?;

        for layer in &self.tx_layers {
            hidden = self.run_tx_layer(&hidden, layer, &cos, &sin, &mask, b, seq)?;
        }
        Ok(hidden)
    }

    fn run_tx_layer(
        &self,
        hidden: &Tensor,
        l: &TxLayerW,
        cos: &Tensor, sin: &Tensor,
        mask: &Tensor,
        b: usize, seq: usize,
    ) -> Result<Tensor> {
        let _h = self.config.hidden_size;
        let nh = self.config.num_heads;
        let d = self.config.head_dim;

        let normed = self.rms_norm(hidden, &l.input_ln)?;
        let q = self.linear_3d(&normed, &l.q, None)?.reshape((b, seq, nh, d))?.transpose(1, 2)?;
        let k = self.linear_3d(&normed, &l.k, None)?.reshape((b, seq, nh, d))?.transpose(1, 2)?;
        let v = self.linear_3d(&normed, &l.v, None)?.reshape((b, seq, nh, d))?.transpose(1, 2)?;
        let q = self.apply_rope(&q, cos, sin)?;
        let k = self.apply_rope(&k, cos, sin)?;
        let scale = (d as f64).powf(-0.5);
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let attn = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?
            .broadcast_add(&mask.to_dtype(q.dtype())?)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, seq, nh * d))?;
        let out = self.linear_3d(&out, &l.o, None)?;
        let out = out.broadcast_mul(&l.attn_scale)?;
        let hidden = (hidden + out)?;

        let normed = self.rms_norm(&hidden, &l.post_ln)?;
        let gate = self.linear_3d(&normed, &l.gate, None)?;
        let up   = self.linear_3d(&normed, &l.up, None)?;
        let mlp  = self.linear_3d(
            &(candle_nn::ops::silu(&gate)? * up)?,
            &l.down,
            None,
        )?;
        let mlp = mlp.broadcast_mul(&l.mlp_scale)?;
        Ok((hidden + mlp)?)
    }
}
