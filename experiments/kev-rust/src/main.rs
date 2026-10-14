use anyhow::{ensure, Result};
use candle::Device;
use clap::{Parser, ValueEnum};
use kev_rust::Runner;
use std::{
    io::{self, BufRead},
    path::PathBuf,
    time::Instant,
};

#[derive(Clone, Copy, ValueEnum)]
enum Backend {
    Cpu,
    Cuda,
}

#[derive(Parser)]
struct Args {
    /// Inference device. CUDA requires a binary built with --features cuda.
    #[arg(long, value_enum, default_value = "cpu")]
    device: Backend,
    /// CUDA GPU ordinal.
    #[arg(long, default_value_t = 0)]
    device_index: usize,
    #[arg(long, default_value = "weights")]
    weights: PathBuf,
    /// One System One request per line. Omit to read stdin.
    #[arg(long)]
    input: Option<PathBuf>,
    /// Disable shared-state optimization to measure the full-row baseline.
    #[arg(long)]
    rows: bool,
    /// Use the allocating tensor scan instead of the fused CPU/CUDA scan.
    #[arg(long)]
    tensor: bool,
    /// Explicitly request batching (the default for fused kernels).
    #[arg(long)]
    batch: bool,
    /// Disable question batching for the comparison baseline.
    #[arg(long, conflicts_with = "batch")]
    unbatched: bool,
    #[arg(long, default_value_t = 1)]
    repeats: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.repeats > 0, "repeats must be positive");
    let start = Instant::now();
    let device = match args.device {
        Backend::Cpu => Device::Cpu,
        Backend::Cuda => {
            #[cfg(not(feature = "cuda"))]
            anyhow::bail!("CUDA requires a build with cargo build --release --features cuda");
            #[cfg(feature = "cuda")]
            Device::new_cuda(args.device_index)?
        }
    };
    let runner = Runner::load_on_device(&args.weights, !args.tensor, device)?;
    eprintln!("Loaded Kev-0.8B in {:.2}s", start.elapsed().as_secs_f64());
    let input: Box<dyn BufRead> = match args.input {
        Some(p) => Box::new(io::BufReader::new(std::fs::File::open(p)?)),
        None => Box::new(io::BufReader::new(io::stdin())),
    };
    for line in input.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let req = serde_json::from_str(&line)?;
        for iteration in 0..args.repeats {
            let mut result = if args.batch || (!args.unbatched && !args.tensor) {
                if args.rows {
                    runner.predict_batched(&req, false)?
                } else {
                    runner.infer(&req)?
                }
            } else {
                runner.predict(&req, !args.rows)?
            };
            result["iteration"] = iteration.into();
            println!("{}", serde_json::to_string(&result)?);
        }
    }
    Ok(())
}
