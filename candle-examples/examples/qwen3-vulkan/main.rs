//! Qwen3 on the Vulkan device (quantized weights).
//!
//! Loads a Qwen3 GGUF file and keeps the quantized weights (Q4_K) in VRAM:
//! the decode (m == 1) matmul runs a Q4_K GEMV kernel directly on the raw
//! bytes, and the prefill dequantizes in VRAM and runs the f32 GEMM.
//! Greedy decoding, with the argmax done on the CPU to keep the sample
//! self-contained.
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;
use anyhow::{anyhow, Context, Result};
use candle::{quantized::gguf_file, Device, Tensor};
use candle_transformers::models::quantized_qwen3::ModelWeights as Qwen3;
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
    let md = &ct.metadata;
    eprintln!(
        "layers {}, hidden {}, heads {} (kv {})",
        md.get("qwen3.block_count")
            .and_then(|v| match v {
                gguf_file::Value::U32(v) => Some(*v as usize),
                _ => None,
            })
            .unwrap_or_default(),
        md.get("qwen3.embedding_length")
            .and_then(|v| match v {
                gguf_file::Value::U32(v) => Some(*v as usize),
                _ => None,
            })
            .unwrap_or_default(),
        md.get("qwen3.attention.head_count")
            .and_then(|v| match v {
                gguf_file::Value::U32(v) => Some(*v as usize),
                _ => None,
            })
            .unwrap_or_default(),
        md.get("qwen3.attention.head_count_kv")
            .and_then(|v| match v {
                gguf_file::Value::U32(v) => Some(*v as usize),
                _ => None,
            })
            .unwrap_or_default(),
    );
    // Quantized model: weights stay quantized in VRAM.
    let t0 = Instant::now();
    let mut file = BufReader::new(File::open(&args.model)?);
    let mut model = Qwen3::from_gguf(ct, &mut file, &device)
        .context("loading the quantized model")?;
    eprintln!("model loaded in {:.1} s", t0.elapsed().as_secs_f64());
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
    if std::env::var("QWV_DEBUG").is_ok() {
        let lv = logits.flatten_all()?.to_vec1::<f32>()?;
        let mut top: Vec<_> = lv.iter().enumerate().collect();
        top.sort_by(|a,b| b.1.partial_cmp(a.1).unwrap());
        top.truncate(5);
        eprintln!("PREFILL top5: {:?}", top.iter().map(|(i,v)| (*i, *v)).collect::<Vec<_>>());
        eprintln!("PREFILL logits[0..4]: {:?} max={}", &lv[0..4], lv.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    }
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
        if std::env::var("QWV_DEBUG").is_ok() {
            eprintln!("tok {} id={}", all.len() - 1, next);
        }
    }
    if std::env::var("QWV_DEBUG").is_ok() {
        let lv = logits.flatten_all()?.to_vec1::<f32>()?;
        let mut top: Vec<_> = lv.iter().enumerate().collect();
        top.sort_by(|a,b| b.1.partial_cmp(a.1).unwrap());
        top.truncate(5);
        eprintln!("top5 logits: {:?}", top.iter().map(|(i,v)| (*i, *v)).collect::<Vec<_>>());
        eprintln!("logits[0..4]: {:?}", &lv[0..4.min(lv.len())]);
        eprintln!("logit max={} min={}", lv.iter().cloned().fold(f32::NEG_INFINITY, f32::max), lv.iter().cloned().fold(f32::INFINITY, f32::min));
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
