// https://github.com/karpathy/llama2.c
#![allow(dead_code)]
#![allow(unused)]

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use clap::Parser;

use anyhow::Result;
use candle::{DType, Device, Error, Layout, Shape, Tensor};

#[derive(Debug, Clone)]
struct Config {
    dim: i32,        // transformer dimension
    hidden_dim: i32, // for ffn layers
    n_layers: i32,   // number of layers
    n_heads: i32,    // number of query heads
    n_kv_heads: i32, // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: i32, // vocabulary size, usually 256 (byte-level)
    seq_len: i32,    // max sequence length
}

impl Config {
    fn read_i32<R: std::io::Read>(r: &mut R) -> Result<i32> {
        let mut buf = [0u8; 4];
        r.read_exact(&mut buf)?;
        Ok(i32::from_le_bytes(buf))
    }

    fn from_reader<R: std::io::Read>(r: &mut R) -> Result<Self> {
        let dim = Self::read_i32(r)?;
        let hidden_dim = Self::read_i32(r)?;
        let n_layers = Self::read_i32(r)?;
        let n_heads = Self::read_i32(r)?;
        let n_kv_heads = Self::read_i32(r)?;
        let vocab_size = Self::read_i32(r)?;
        let seq_len = Self::read_i32(r)?;
        Ok(Self {
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            seq_len,
        })
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Config file in binary format.
    #[arg(long)]
    config: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    let t = Tensor::arange(0f32, 14f32, &device)?.reshape((2, 7))?;
    println!("{t}");
    let mut file = std::fs::File::open(&args.config)?;
    let config = Config::from_reader(&mut file);
    println!("config: {config:?}");

    Ok(())
}
