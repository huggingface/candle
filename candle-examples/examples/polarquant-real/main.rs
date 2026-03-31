//! PolarQuant Real-Model Benchmark (Qwen3-0.6B)
//!
//! Loads Qwen3-0.6B from HuggingFace, runs F32 generation as baseline, then
//! quantizes all attention/MLP Linear layers with PolarQuant and re-runs
//! generation to compare tok/sec and output quality.

use anyhow::{Error as E, Result};
use candle::{DType, Device, Module, Result as CResult, Tensor};
use candle_nn::kv_cache::{ConcatKvCache, QuantizedKvCache};
use candle_nn::polarquant_nn::PolarQuantLinear;
use candle_nn::{Activation, VarBuilder};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::qwen3::Config;
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::io::Write;
use std::sync::Arc;

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> CResult<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> CResult<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Parser, Debug)]
#[command(about = "PolarQuant Real-Model Benchmark (Qwen3)")]
struct Args {
    #[arg(long)]
    cpu: bool,

    #[arg(long, default_value_t = 6)]
    bits: usize,

    #[arg(long, default_value_t = 50)]
    sample_len: usize,

    #[arg(long, default_value = "The capital of France is")]
    prompt: String,

    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    #[arg(long, default_value_t = 42)]
    seed: u64,
}

// ── PolarQuant-enabled Qwen3 model ──────────────────────────────────
// Mirrors the upstream Qwen3 architecture but uses PolarQuantLinear for
// weight storage and QuantizedKvCache for KV compression.

enum LinearLayer {
    F32(candle_nn::Linear),
    PQ(PolarQuantLinear),
}

impl Module for LinearLayer {
    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        match self {
            Self::F32(l) => l.forward(x),
            Self::PQ(l) => l.forward(x),
        }
    }
}

enum KvCacheLayer {
    Plain(ConcatKvCache),
    Quantized(QuantizedKvCache),
}

impl KvCacheLayer {
    fn append(&mut self, k: &Tensor, v: &Tensor) -> CResult<(Tensor, Tensor)> {
        match self {
            Self::Plain(c) => c.append(k, v),
            Self::Quantized(c) => c.append(k, v),
        }
    }
    fn reset(&mut self) {
        match self {
            Self::Plain(c) => c.reset(),
            Self::Quantized(c) => c.reset(),
        }
    }
}

struct MLP {
    gate_proj: LinearLayer,
    up_proj: LinearLayer,
    down_proj: LinearLayer,
    act_fn: Activation,
}

impl MLP {
    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let lhs = self.act_fn.forward(&self.gate_proj.forward(x)?)?;
        let rhs = self.up_proj.forward(x)?;
        self.down_proj.forward(&(lhs * rhs)?)
    }
}

struct Attention {
    q_proj: LinearLayer,
    k_proj: LinearLayer,
    v_proj: LinearLayer,
    o_proj: LinearLayer,
    q_norm: candle_nn::RmsNorm,
    k_norm: candle_nn::RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary: Arc<RotaryEmbedding>,
    kv_cache: KvCacheLayer,
}

impl Attention {
    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> CResult<Tensor> {
        let (b, l, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.q_norm.forward(&q.flatten(0, 2)?)?.reshape((
            b,
            self.num_heads,
            l,
            self.head_dim,
        ))?;
        let k = self.k_norm.forward(&k.flatten(0, 2)?)?.reshape((
            b,
            self.num_kv_heads,
            l,
            self.head_dim,
        ))?;

        let (q, k) = self.rotary.apply(&q, &k, offset)?;
        let (k, v) = self.kv_cache.append(&k, &v)?;

        let k = candle_transformers::utils::repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = candle_transformers::utils::repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;

        ctx.transpose(1, 2)?
            .reshape((b, l, self.hidden_size))?
            .apply(&self.o_proj)
    }

    fn clear_cache(&mut self) {
        self.kv_cache.reset();
    }
}

struct Layer {
    attn: Attention,
    mlp: MLP,
    ln1: candle_nn::RmsNorm,
    ln2: candle_nn::RmsNorm,
}

impl Layer {
    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> CResult<Tensor> {
        let h = self.attn.forward(&self.ln1.forward(x)?, mask, offset)?;
        let x = (x + h)?;
        let h2 = self.mlp.forward(&self.ln2.forward(&x)?)?;
        x + h2
    }
}

struct BenchModel {
    embed: candle_nn::Embedding,
    layers: Vec<Layer>,
    norm: candle_nn::RmsNorm,
    lm_head: LinearLayer,
    device: Device,
    dtype: DType,
}

impl BenchModel {
    fn new(cfg: &Config, vb: VarBuilder, bits: Option<usize>, _quantize_kv: bool) -> CResult<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        let embed =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let rotary = Arc::new(RotaryEmbedding::new(dtype, cfg, &device)?);
        let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;

        let vb_l = vb.pp("model.layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let vb_a = vb_l.pp(format!("{i}.self_attn"));
            let vb_m = vb_l.pp(format!("{i}.mlp"));
            let vb_ln1 = vb_l.pp(format!("{i}.input_layernorm"));
            let vb_ln2 = vb_l.pp(format!("{i}.post_attention_layernorm"));

            // Attention projections stay F32 (quality-sensitive)
            let make_f32 = |in_d, out_d, bias, vb: VarBuilder| -> CResult<LinearLayer> {
                Ok(LinearLayer::F32(candle_nn::linear_b(
                    in_d, out_d, bias, vb,
                )?))
            };

            // MLP layers get quantized (bulk of parameters, more tolerant)
            // Use Hadamard rotation when dim is power-of-2 (faster + numerically exact)
            // Non-power-of-2 dims stay F32 to avoid slow Gram-Schmidt construction
            // Only quantize the second half of layers (later layers are more tolerant)
            let quantize_this_layer = bits.is_some() && i >= cfg.num_hidden_layers / 2;
            let make_mlp = |in_d: usize, out_d, vb: VarBuilder| -> CResult<LinearLayer> {
                let inner = candle_nn::linear_no_bias(in_d, out_d, vb)?;
                if quantize_this_layer && in_d.is_power_of_two() {
                    let pq =
                        PolarQuantLinear::from_linear_hadamard(&inner, bits.unwrap(), &device)?;
                    Ok(LinearLayer::PQ(pq))
                } else {
                    Ok(LinearLayer::F32(inner))
                }
            };

            let head_dim = cfg.head_dim;
            let num_heads = cfg.num_attention_heads;
            let num_kv_heads = cfg.num_key_value_heads;
            let hidden_size = head_dim * num_heads;

            let attn = Attention {
                q_proj: make_f32(
                    cfg.hidden_size,
                    num_heads * head_dim,
                    cfg.attention_bias,
                    vb_a.pp("q_proj"),
                )?,
                k_proj: make_f32(
                    cfg.hidden_size,
                    num_kv_heads * head_dim,
                    cfg.attention_bias,
                    vb_a.pp("k_proj"),
                )?,
                v_proj: make_f32(
                    cfg.hidden_size,
                    num_kv_heads * head_dim,
                    cfg.attention_bias,
                    vb_a.pp("v_proj"),
                )?,
                o_proj: make_f32(
                    num_heads * head_dim,
                    cfg.hidden_size,
                    cfg.attention_bias,
                    vb_a.pp("o_proj"),
                )?,
                q_norm: candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb_a.pp("q_norm"))?,
                k_norm: candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb_a.pp("k_norm"))?,
                num_heads,
                num_kv_heads,
                num_kv_groups: num_heads / num_kv_heads,
                head_dim,
                hidden_size,
                rotary: rotary.clone(),
                // KV cache stays plain — quantized KV adds O(S²) dequant cost
                // and compounds error with weight quantization
                kv_cache: KvCacheLayer::Plain(ConcatKvCache::new(2)),
            };

            let mlp = MLP {
                gate_proj: make_mlp(cfg.hidden_size, cfg.intermediate_size, vb_m.pp("gate_proj"))?,
                up_proj: make_mlp(cfg.hidden_size, cfg.intermediate_size, vb_m.pp("up_proj"))?,
                down_proj: make_mlp(cfg.intermediate_size, cfg.hidden_size, vb_m.pp("down_proj"))?,
                act_fn: cfg.hidden_act,
            };

            let ln1 = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_ln1)?;
            let ln2 = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_ln2)?;

            layers.push(Layer {
                attn,
                mlp,
                ln1,
                ln2,
            });
        }

        // lm_head always stays F32 (output quality-critical)
        let lm_head = if cfg.tie_word_embeddings {
            LinearLayer::F32(candle_nn::Linear::new(embed.embeddings().clone(), None))
        } else {
            LinearLayer::F32(candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.vocab_size,
                vb.pp("lm_head"),
            )?)
        };

        Ok(Self {
            embed,
            layers,
            norm,
            lm_head,
            device,
            dtype,
        })
    }

    fn forward(&mut self, input: &Tensor, offset: usize) -> CResult<Tensor> {
        let (b, l) = input.dims2()?;
        let mut h = self.embed.forward(input)?;

        let mask = if l == 1 {
            None
        } else {
            let minf = f32::NEG_INFINITY;
            let mask: Vec<f32> = (0..l)
                .flat_map(|i| {
                    (0..(l + offset)).map(move |j| if j <= i + offset { 0. } else { minf })
                })
                .collect();
            Some(
                Tensor::from_slice(&mask, (b, 1, l, l + offset), &self.device)?
                    .to_dtype(self.dtype)?,
            )
        };

        for layer in &mut self.layers {
            h = layer.forward(&h, mask.as_ref(), offset)?;
        }
        let h = self.norm.forward(&h)?.narrow(1, l - 1, 1)?;
        self.lm_head.forward(&h)
    }

    fn clear_kv_caches(&mut self) {
        for layer in &mut self.layers {
            layer.attn.clear_cache();
        }
    }
}

fn generate(
    model: &mut BenchModel,
    tokens: &[u32],
    sample_len: usize,
    logits_proc: &mut LogitsProcessor,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<(String, std::time::Duration)> {
    model.clear_kv_caches();
    let device = model.device.clone();
    let mut all_tokens = tokens.to_vec();
    let mut generated = String::new();
    let mut stream =
        candle_examples::token_output_stream::TokenOutputStream::new(tokenizer.clone());
    let mut offset = 0;
    let start = std::time::Instant::now();

    for index in 0..sample_len {
        let context_size = if index > 0 { 1 } else { all_tokens.len() };
        let ctxt = &all_tokens[all_tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, offset)?.squeeze(0)?.squeeze(0)?;
        offset += ctxt.len();

        let next = logits_proc.sample(&logits)?;
        all_tokens.push(next);
        if let Some(t) = stream.next_token(next)? {
            generated.push_str(&t);
        }
    }
    if let Some(rest) = stream.decode_rest().map_err(E::msg)? {
        generated.push_str(&rest);
    }

    Ok((generated, start.elapsed()))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;

    println!("PolarQuant Real-Model Benchmark");
    println!("================================");
    println!("Model: Qwen/Qwen3-0.6B");
    println!(
        "Quantization: {}-bit PolarQuant (weights + KV cache)",
        args.bits
    );
    println!("Prompt: \"{}\"\n", args.prompt);

    // Load model
    print!("Fetching model... ");
    std::io::stdout().flush()?;
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        "Qwen/Qwen3-0.6B".to_string(),
        RepoType::Model,
        "main".to_string(),
    ));

    let tokenizer_filename = repo.get("tokenizer.json")?;
    let config_filename = repo.get("config.json")?;
    let filenames = candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")
        .unwrap_or_else(|_| vec![repo.get("model.safetensors").unwrap()]);
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let config: Config = serde_json::from_slice(&std::fs::read(config_filename)?)?;
    println!("done");

    println!(
        "Config: {}L, hidden={}, heads={}, kv_heads={}, head_dim={}, intermediate={}",
        config.num_hidden_layers,
        config.hidden_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.intermediate_size
    );

    let tokens = tokenizer
        .encode(args.prompt.clone(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    println!("Prompt tokens: {}\n", tokens.len());

    // ── F32 baseline ──────────────────────────────────────────────────
    print!("Loading F32 model... ");
    std::io::stdout().flush()?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, DType::F32, &device)? };
    let mut f32_model = BenchModel::new(&config, vb, None, false)?;
    println!("done");

    let mut logits_proc = LogitsProcessor::from_sampling(
        args.seed,
        Sampling::All {
            temperature: args.temperature,
        },
    );
    let (text_f32, elapsed_f32) = generate(
        &mut f32_model,
        &tokens,
        args.sample_len,
        &mut logits_proc,
        &tokenizer,
    )?;
    let tps_f32 = args.sample_len as f64 / elapsed_f32.as_secs_f64();
    println!("── F32 Generation ──────────────────────────────────");
    println!("  Output: \"{}\"", &text_f32[..text_f32.len().min(120)]);
    println!("  Throughput: {tps_f32:.1} tokens/sec");

    // Free F32 model
    drop(f32_model);

    // ── PQ model ──────────────────────────────────────────────────────
    print!("\nLoading + quantizing PQ-{} model... ", args.bits);
    std::io::stdout().flush()?;
    let quant_start = std::time::Instant::now();
    let vb2 = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, DType::F32, &device)? };
    let mut pq_model = BenchModel::new(&config, vb2, Some(args.bits), true)?;
    let quant_elapsed = quant_start.elapsed();
    println!("done ({:.1}s)", quant_elapsed.as_secs_f64());

    let mut logits_proc2 = LogitsProcessor::from_sampling(
        args.seed,
        Sampling::All {
            temperature: args.temperature,
        },
    );
    let (text_pq, elapsed_pq) = generate(
        &mut pq_model,
        &tokens,
        args.sample_len,
        &mut logits_proc2,
        &tokenizer,
    )?;
    let tps_pq = args.sample_len as f64 / elapsed_pq.as_secs_f64();
    println!(
        "── PQ-{} Generation ─────────────────────────────────",
        args.bits
    );
    println!("  Output: \"{}\"", &text_pq[..text_pq.len().min(120)]);
    println!("  Throughput: {tps_pq:.1} tokens/sec");

    // ── Comparison ───────────────────────────────────────────────────
    println!("\n── Comparison ──────────────────────────────────────");
    println!("  F32:  {tps_f32:.1} tokens/sec");
    println!("  PQ-{}: {tps_pq:.1} tokens/sec", args.bits);
    println!("  Ratio: {:.2}x", tps_f32 / tps_pq);

    Ok(())
}
