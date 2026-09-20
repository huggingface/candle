//! TEMPORARY DEBUG: compare prefill logits for the Qwen3 quantized model
//! on the CPU device (reference) and the Vulkan device. Delete after use.
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;

use anyhow::{Context, Result};
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
    /// The prompt.
    #[arg(long)]
    prompt: String,
    /// Number of decode steps after the prefill (per device).
    #[arg(short = 'n', long, default_value_t = 3)]
    decode: usize,
    /// The Vulkan device ordinal.
    #[arg(short = 'g', long, default_value_t = 1)]
    gpu: usize,
}

fn top5(lv: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut top: Vec<_> = lv.iter().enumerate().collect();
    top.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    top.truncate(k);
    top.into_iter().map(|(i, v)| (i, *v)).collect()
}

fn run_one(
    tag: &str,
    device: &Device,
    model_path: &str,
    tok: &Tokenizer,
    prompt_ids: &[u32],
    decode: usize,
) -> Result<(Vec<f32>, Vec<u32>)> {
    if let Some(dir) = std::env::var("QWV_LAYER_DUMP_DIR").ok() {
        let sub = std::path::Path::new(&dir).join(tag);
        std::fs::create_dir_all(&sub).ok();
        std::env::set_var("QWV_LAYER_DUMP", sub);
    }
    let mut file = BufReader::new(File::open(model_path)?);
    let ct = gguf_file::Content::read(&mut file)
        .with_context(|| format!("reading gguf metadata from {model_path}"))?;
    let t0 = Instant::now();
    let mut model = Qwen3::from_gguf(ct, &mut file, device).context("loading model")?;
    eprintln!("[{tag}] loaded in {:.1} s", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let mut logits = model.forward(
        &Tensor::from_vec(prompt_ids.to_vec(), (1, prompt_ids.len()), device)?,
        0,
    )?;
    let prefill_s = t0.elapsed().as_secs_f64();
    {
        let lv = logits
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?;
        if let Ok(dir) = std::env::var("QWV_LAYER_DUMP") {
            let bytes: Vec<u8> = lv.iter().flat_map(|f| f.to_le_bytes()).collect();
            let _ = std::fs::write(format!("{dir}/logits.bin"), bytes);
        }
    }
    let mut offset = prompt_ids.len();

    let mut ids: Vec<u32> = Vec::new();
    let t0 = Instant::now();
    for _ in 0..decode {
        let lv = logits
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?;
        let best = lv
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        ids.push(best.0 as u32);
        let input = Tensor::from_vec(vec![best.0 as u32], (1, 1), device)?;
        logits = model.forward(&input, offset)?;
        offset += 1;
    }
    let decode_s = t0.elapsed().as_secs_f64();

    let final_logits = logits
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1::<f32>()?;
    eprintln!("[{tag}] prefill {prefill_s:.2} s, decode {decode_s:.2} s, first decode ids {ids:?}");
    eprintln!("[{tag}] prefill top5: {:?}", top5(&final_logits, 5));
    eprintln!(
        "[{tag}] prefill decoded: {:?}",
        tok.decode(&ids, true).map_err(|e| anyhow::anyhow!("{e}"))?
    );
    Ok((final_logits, ids))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let tok = Tokenizer::from_file(&args.tokenizer).map_err(anyhow::Error::msg)?;
    let prompt_ids: Vec<u32> = tok
        .encode(args.prompt.as_str(), true)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();
    eprintln!("prompt ids: {prompt_ids:?}");

    let (cpu_logits, cpu_ids) = run_one(
        "cpu",
        &Device::Cpu,
        &args.model,
        &tok,
        &prompt_ids,
        args.decode,
    )?;
    let (vk_logits, vk_ids) = run_one(
        "vulkan",
        &Device::new_vulkan(args.gpu)?,
        &args.model,
        &tok,
        &prompt_ids,
        args.decode,
    )?;

    assert_eq!(cpu_logits.len(), vk_logits.len());
    let max_abs: f32 = cpu_logits
        .iter()
        .zip(vk_logits.iter())
        .map(|(a, b)| (*a - *b).abs())
        .fold(0f32, f32::max);
    let bad = cpu_logits
        .iter()
        .zip(vk_logits.iter())
        .filter(|(a, b)| (**a - **b).abs() > 1e-3)
        .count();
    let bad_frac = bad as f64 / cpu_logits.len() as f64;
    eprintln!("logits max abs diff: {max_abs:.6}  frac >1e-3: {bad} ({bad_frac:.4})");
    eprintln!("cpu  first ids: {cpu_ids:?}");
    eprintln!("vulkan first ids: {vk_ids:?}");
    if cpu_ids == vk_ids {
        eprintln!("MATCH: first decode ids identical");
    } else {
        eprintln!("MISMATCH: first decode ids differ");
    }
    if let Some(dir) = std::env::var("QWV_LAYER_DUMP_DIR").ok() {
        let mut files: Vec<String> = std::fs::read_dir(format!("{dir}/cpu"))
            .ok()
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .filter(|n| n.ends_with(".bin"))
                    .collect()
            })
            .unwrap_or_default();
        files.sort();
        let mut worst = 0.0f32;
        let mut first_bad = None;
        for name in &files {
            let Ok(a) = std::fs::read(format!("{dir}/cpu/{name}")) else {
                continue;
            };
            let Ok(b) = std::fs::read(format!("{dir}/vulkan/{name}")) else {
                continue;
            };
            if a.len() != b.len() {
                eprintln!(
                    "{name}: size mismatch {} vs {} (FIRST DIVERGING)",
                    a.len(),
                    b.len()
                );
                first_bad = Some(name.clone());
                break;
            }
            let av: Vec<f32> = a
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let bv: Vec<f32> = b
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let d: f32 = av
                .iter()
                .zip(bv.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            let mark = if d > 1e-3 { "  <<< DIVERGES" } else { "" };
            eprintln!("{name}: max abs diff {d:.6}{mark}");
            if d > worst {
                worst = d;
            }
            if first_bad.is_none() && d > 1e-3 {
                first_bad = Some(name.clone());
            }
        }
        match first_bad {
            Some(n) => eprintln!("FIRST DIVERGING DUMP: {n} (worst {worst:.6})"),
            None => eprintln!("all dumped tensors match (worst {worst:.6})"),
        }
    }
    Ok(())
}
