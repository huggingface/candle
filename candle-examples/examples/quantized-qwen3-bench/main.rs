//! Throughput benchmark for the quantized Qwen3 model, matched to `llama-bench`
//! methodology so candle and llama.cpp numbers are comparable:
//!   - prefill (pp): one forward over a dummy prompt of `--pp` tokens.
//!   - decode  (tg): `--tg` greedy single-token forwards.
//!
//! No tokenizer, no sampling, no repeat penalty - raw model throughput only.
//! Reports median tokens/s over `--reps` measured runs (after `--warmup`).
//!
//! Pair with thread pinning to simulate a Lambda tier, e.g.:
//!   CANDLE_QMATMUL_DECODE_THREADS=2 CANDLE_QMATMUL_PREFILL_THREADS=2 \
//!     taskset -c 0-1 cargo run --release --example quantized-qwen3-bench -- \
//!     --model model.gguf --json

use anyhow::Result;
use candle::quantized::gguf_file;
use candle::{Device, Tensor, D};
use candle_transformers::models::quantized_qwen3::ModelWeights as Qwen3;
use clap::Parser;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// GGUF file to load.
    #[arg(long)]
    model: String,

    /// Prompt length in tokens for the prefill (pp) measurement.
    #[arg(long, default_value_t = 512)]
    pp: usize,

    /// Number of tokens to generate for the decode (tg) measurement.
    #[arg(long, default_value_t = 128)]
    tg: usize,

    /// Measured repetitions (median is reported).
    #[arg(long, default_value_t = 5)]
    reps: usize,

    /// Warmup repetitions discarded before measuring.
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Dummy token id used to fill the synthetic prompt (must be < vocab size).
    #[arg(long, default_value_t = 100)]
    token_id: u32,

    /// Emit a single JSON line on stdout instead of a human table.
    #[arg(long)]
    json: bool,
}

/// Greedy next token from a 1-D logits tensor, computed on-device so the host
/// copy stays out of the decode loop.
fn argmax(logits: &Tensor) -> Result<u32> {
    Ok(logits.argmax(D::Minus1)?.to_scalar::<u32>()?)
}

/// (median, min, max) of a sample. Returns zeros for an empty slice.
fn stats(xs: &[f64]) -> (f64, f64, f64) {
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    let median = if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    };
    (median, s[0], s[n - 1])
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::Cpu;

    let mut file = std::fs::File::open(&args.model)?;
    let content = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(&args.model))?;
    let mut model = Qwen3::from_gguf(content, &mut file, &device)?;

    let prompt: Vec<u32> = vec![args.token_id; args.pp];

    let mut pp_rates = Vec::new();
    let mut tg_rates = Vec::new();

    let total = args.warmup + args.reps;
    for rep in 0..total {
        model.clear_kv_cache();

        // Prefill: one forward over the whole synthetic prompt.
        let input = Tensor::new(prompt.as_slice(), &device)?.unsqueeze(0)?;
        let t = Instant::now();
        let logits = model.forward(&input, 0)?.squeeze(0)?;
        let pp_dt = t.elapsed().as_secs_f64();
        let mut next = argmax(&logits)?;

        // Decode: tg greedy single-token forwards.
        let t = Instant::now();
        for i in 0..args.tg {
            let input = Tensor::new(&[next], &device)?.unsqueeze(0)?;
            let logits = model.forward(&input, args.pp + i)?.squeeze(0)?;
            next = argmax(&logits)?;
        }
        let tg_dt = t.elapsed().as_secs_f64();

        let pp_rate = args.pp as f64 / pp_dt;
        let tg_rate = args.tg as f64 / tg_dt;
        if rep >= args.warmup {
            pp_rates.push(pp_rate);
            tg_rates.push(tg_rate);
        }
        if !args.json {
            let tag = if rep < args.warmup { "warmup" } else { "run" };
            eprintln!("{tag} {rep}: pp {pp_rate:.2} t/s   tg {tg_rate:.2} t/s");
        }
    }

    let (pp_med, pp_min, pp_max) = stats(&pp_rates);
    let (tg_med, tg_min, tg_max) = stats(&tg_rates);

    if args.json {
        println!(
            "{{\"engine\":\"candle\",\"pp\":{},\"tg\":{},\"reps\":{},\
\"pp_tok_s_median\":{:.3},\"pp_tok_s_min\":{:.3},\"pp_tok_s_max\":{:.3},\
\"tg_tok_s_median\":{:.3},\"tg_tok_s_min\":{:.3},\"tg_tok_s_max\":{:.3}}}",
            args.pp, args.tg, args.reps, pp_med, pp_min, pp_max, tg_med, tg_min, tg_max
        );
    } else {
        println!("\ncandle  pp{}  tg{}  reps={}", args.pp, args.tg, args.reps);
        println!("  prefill (pp): {pp_med:.2} t/s  [{pp_min:.2}..{pp_max:.2}]");
        println!("  decode  (tg): {tg_med:.2} t/s  [{tg_min:.2}..{tg_max:.2}]");
    }
    Ok(())
}
