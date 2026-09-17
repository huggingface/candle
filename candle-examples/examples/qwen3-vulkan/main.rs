//! Qwen3 on the Vulkan device.
//!
//! Loads a Qwen3 GGUF file (any quantization), dequantizes all weights to
//! f32 on the CPU at load time and uploads them to the GPU, then runs the
//! full forward pass on the Vulkan backend. Greedy decoding, with the
//! argmax done on the CPU to keep the sample self-contained.
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use candle::{quantized::gguf_file, DType, Device, Tensor};
use candle_nn::{Activation, VarBuilder};
use candle_transformers::models::qwen3::{Config, ModelForCausalLM};
use clap::Parser;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to a Qwen3 GGUF file.
    #[arg(long)]
    model: String,

    /// Path to the tokenizer JSON file.
    #[arg(long)]
    tokenizer: String,

    /// The initial prompt.
    #[arg(long)]
    prompt: String,

    /// Number of tokens to generate.
    #[arg(short = 'n', long, default_value_t = 64)]
    sample_len: usize,
    /// The Vulkan device ordinal.
    #[arg(short = 'g', long, default_value_t = 0)]
    gpu: usize,
}

fn md_u32(md: &HashMap<String, gguf_file::Value>, key: &str) -> Result<usize> {
    match md.get(key) {
        Some(gguf_file::Value::U32(v)) => Ok(*v as usize),
        Some(gguf_file::Value::F32(v)) => Ok(*v as usize),
        _ => bail!("cannot find {key} in the gguf metadata"),
    }
}

fn md_f32(md: &HashMap<String, gguf_file::Value>, key: &str) -> Result<f64> {
    match md.get(key) {
        Some(gguf_file::Value::F32(v)) => Ok(*v as f64),
        Some(gguf_file::Value::U32(v)) => Ok(*v as f64),
        _ => bail!("cannot find {key} in the gguf metadata"),
    }
}

/// Maps a gguf tensor name to the Hugging Face name expected by
/// `candle_transformers::models::qwen3`.
fn hf_name(name: &str) -> Option<String> {
    if let Some(rest) = name.strip_prefix("blk.") {
        let (idx, sub) = rest.split_once('.')?;
        let idx: usize = idx.parse().ok()?;
        let l = format!("model.layers.{idx}");
        let mapped = match sub {
            "attn_q.weight" => format!("{l}.self_attn.q_proj.weight"),
            "attn_k.weight" => format!("{l}.self_attn.k_proj.weight"),
            "attn_v.weight" => format!("{l}.self_attn.v_proj.weight"),
            "attn_output.weight" => format!("{l}.self_attn.o_proj.weight"),
            "attn_q_norm.weight" => format!("{l}.self_attn.q_norm.weight"),
            "attn_k_norm.weight" => format!("{l}.self_attn.k_norm.weight"),
            "attn_norm.weight" => format!("{l}.input_layernorm.weight"),
            "ffn_norm.weight" => format!("{l}.post_attention_layernorm.weight"),
            "ffn_gate.weight" => format!("{l}.mlp.gate_proj.weight"),
            "ffn_up.weight" => format!("{l}.mlp.up_proj.weight"),
            "ffn_down.weight" => format!("{l}.mlp.down_proj.weight"),
            _ => return None,
        };
        Some(mapped)
    } else {
        match name {
            "token_embd.weight" => Some("model.embed_tokens.weight".to_string()),
            "norm.weight" => Some("model.norm.weight".to_string()),
            "output.weight" => Some("lm_head.weight".to_string()),
            "output_norm.weight" => Some("model.norm.weight".to_string()),
            _ => None,
        }
    }
}

fn next_token(logits: &Tensor) -> Result<u32> {
    let vals = logits
        .to_device(&Device::Cpu)
        .context("moving logits to the CPU")?
        .flatten_all()?
        .to_vec1::<f32>()?;
    vals.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .ok_or_else(|| anyhow!("empty logits"))
}

fn main() -> Result<()> {
    let args = Args::parse();

    let device = Device::new_vulkan(args.gpu).context("creating the Vulkan device")?;
    let ct = gguf_file::Content::read(&mut BufReader::new(File::open(&args.model)?))
        .with_context(|| format!("reading gguf metadata from {}", args.model))?;

    // Dequantize every model weight to f32 on the GPU.
    let mut file = BufReader::new(File::open(&args.model)?);
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    let mut skipped = 0usize;
    for name in ct.tensor_infos.keys() {
        let Some(hf) = hf_name(name) else {
            skipped += 1;
            continue;
        };
        let qt = ct.tensor(&mut file, name, &Device::Cpu)?;
        let t = qt
            .dequantize(&device)
            .with_context(|| format!("dequantizing {name}"))?;
        weights.insert(hf, t);
    }
    if skipped > 0 {
        eprintln!("skipped {skipped} non-weight gguf tensors");
    }

    let md = &ct.metadata;
    let cfg = Config {
        // This gguf has no vocab_size key; take it from the embedding shape.
        vocab_size: ct
            .tensor_infos
            .get("token_embd.weight")
            .and_then(|t| t.shape.dims().first().copied())
            .expect("token_embd.weight with a vocab dim"),
        hidden_size: md_u32(md, "qwen3.embedding_length")?,
        intermediate_size: md_u32(md, "qwen3.feed_forward_length")?,
        num_hidden_layers: md_u32(md, "qwen3.block_count")?,
        num_attention_heads: md_u32(md, "qwen3.attention.head_count")?,
        head_dim: md_u32(md, "qwen3.attention.key_length")?,
        num_key_value_heads: md_u32(md, "qwen3.attention.head_count_kv")?,
        max_position_embeddings: md_u32(md, "qwen3.context_length")?,
        rope_theta: md_f32(md, "qwen3.rope.freq_base")?,
        rms_norm_eps: md_f32(md, "qwen3.attention.layer_norm_rms_epsilon")?,
        attention_bias: false,
        sliding_window: None,
        max_window_layers: 0,
        tie_word_embeddings: !ct.tensor_infos.contains_key("output.weight"),
        use_sliding_window: false,
        hidden_act: Activation::Silu,
    };
    eprintln!(
        "loaded {} weights; {} layers, hidden {}, heads {} (kv {})",
        weights.len(),
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
    );

    let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
    let mut model = ModelForCausalLM::new(&cfg, vb)?;

    let tokenizer = Tokenizer::from_file(&args.tokenizer).map_err(anyhow::Error::msg)?;
    let prompt_ids: Vec<u32> = tokenizer
        .encode(args.prompt.as_str(), true)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();
    print!(
        "{}",
        tokenizer
            .decode(&prompt_ids, true)
            .map_err(anyhow::Error::msg)?
    );

    // Prefill with the whole prompt.
    let mut offset = 0;
    let t0 = Instant::now();
    let mut logits = model.forward(
        &Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), &device)?,
        offset,
    )?;
    offset += prompt_ids.len();
    let prefill_s = t0.elapsed().as_secs_f64();

    // Greedy decode.
    let mut all = prompt_ids.clone();
    let t0 = Instant::now();
    for _ in 0..args.sample_len {
        let next = next_token(&logits)?;
        let input = Tensor::from_vec(vec![next], (1, 1), &device)?;
        logits = model.forward(&input, offset)?;
        offset += 1;
        print!(
            "{}",
            tokenizer
                .decode(&[next], false)
                .map_err(anyhow::Error::msg)?
        );
        std::io::Write::flush(&mut std::io::stdout())?;
        all.push(next);
    }
    let decode_s = t0.elapsed().as_secs_f64();
    println!();
    eprintln!(
        "prompt tokens: {}, prefill: {:.1} tok/s, decode: {:.1} tok/s, generated: {}",
        prompt_ids.len(),
        prompt_ids.len() as f64 / prefill_s,
        args.sample_len as f64 / decode_s,
        all.len(),
    );
    Ok(())
}
