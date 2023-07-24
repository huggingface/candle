// https://github.com/karpathy/llama2.c
#![allow(dead_code)]
#![allow(unused)]

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use clap::Parser;

use anyhow::Result;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use candle::{DType, Device, Error, Layout, Shape, Tensor};

#[derive(Debug, Clone)]
struct Config {
    dim: usize,        // transformer dimension
    hidden_dim: usize, // for ffn layers
    n_layers: usize,   // number of layers
    n_heads: usize,    // number of query heads
    n_kv_heads: usize, // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: usize, // vocabulary size, usually 256 (byte-level)
    seq_len: usize,    // max sequence length
}

struct TransformerWeights {
    // token embedding table
    token_embedding_table: Tensor, // (vocab_size, dim)
    // weights for rmsnorms
    rms_att_weight: Tensor, // (layer, dim) rmsnorm weights
    rms_ffn_weight: Tensor, // (layer, dim)
    // weights for matmuls
    wq: Tensor, // (layer, dim, dim)
    wk: Tensor, // (layer, dim, dim)
    wv: Tensor, // (layer, dim, dim)
    wo: Tensor, // (layer, dim, dim)
    // weights for ffn
    w1: Tensor, // (layer, hidden_dim, dim)
    w2: Tensor, // (layer, dim, hidden_dim)
    w3: Tensor, // (layer, hidden_dim, dim)
    // final rmsnorm
    rms_final_weight: Tensor, // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    freq_cis_real: Tensor, // (seq_len, dim/2)
    freq_cis_imag: Tensor, // (seq_len, dim/2)
}

struct RunState {
    // current wave of activations
    x: Tensor,      // activation at current time stamp (dim,)
    xb: Tensor,     // same, but inside a residual branch (dim,)
    xb2: Tensor,    // an additional buffer just for convenience (dim,)
    hb: Tensor,     // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: Tensor,    // buffer for hidden dimension in the ffn (hidden_dim,)
    q: Tensor,      // query (dim,)
    k: Tensor,      // key (dim,)
    v: Tensor,      // value (dim,)
    att: Tensor,    // buffer for scores/attention values (seq_len,)
    logits: Tensor, // output logits
    // kv cache
    key_cache: Tensor,   // (layer, seq_len, dim)
    value_cache: Tensor, // (layer, seq_len, dim)
}

impl Config {
    fn read_i32<R: std::io::Read>(r: &mut R) -> Result<i32> {
        let mut buf = [0u8; 4];
        r.read_exact(&mut buf)?;
        Ok(i32::from_le_bytes(buf))
    }

    fn from_reader<R: std::io::Read>(r: &mut R) -> Result<Self> {
        let dim = Self::read_i32(r)? as usize;
        let hidden_dim = Self::read_i32(r)? as usize;
        let n_layers = Self::read_i32(r)? as usize;
        let n_heads = Self::read_i32(r)? as usize;
        let n_kv_heads = Self::read_i32(r)? as usize;
        let vocab_size = Self::read_i32(r)? as usize;
        let seq_len = Self::read_i32(r)? as usize;
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

    fn head_size(&self) -> usize {
        self.dim / self.n_heads
    }
}

impl TransformerWeights {
    fn read_tensor<R: std::io::Read, S: Into<Shape>>(
        r: &mut R,
        shape: S,
        dev: &Device,
    ) -> Result<Tensor> {
        let shape = shape.into();
        let mut data_t = vec![0f32; shape.elem_count()];
        r.read_f32_into::<LittleEndian>(&mut data_t)?;
        let tensor = Tensor::from_vec(data_t, shape, dev)?;
        Ok(tensor)
    }

    fn from_reader<R: std::io::Read>(r: &mut R, c: &Config, dev: &Device) -> Result<Self> {
        let token_embedding_table = Self::read_tensor(r, (c.vocab_size, c.dim), dev)?;
        let rms_att_weight = Self::read_tensor(r, (c.n_layers, c.dim), dev)?;
        let wq = Self::read_tensor(r, (c.n_layers, c.dim, c.dim), dev)?;
        let wk = Self::read_tensor(r, (c.n_layers, c.dim, c.dim), dev)?;
        let wv = Self::read_tensor(r, (c.n_layers, c.dim, c.dim), dev)?;
        let wo = Self::read_tensor(r, (c.n_layers, c.dim, c.dim), dev)?;
        let rms_ffn_weight = Self::read_tensor(r, (c.n_layers, c.dim), dev)?;
        let w1 = Self::read_tensor(r, (c.n_layers, c.dim, c.hidden_dim), dev)?;
        let w2 = Self::read_tensor(r, (c.n_layers, c.hidden_dim, c.dim), dev)?;
        let w3 = Self::read_tensor(r, (c.n_layers, c.dim, c.hidden_dim), dev)?;
        let rms_final_weight = Self::read_tensor(r, c.dim, dev)?;
        let head_size = c.head_size();
        let freq_cis_real = Self::read_tensor(r, (c.seq_len, head_size / 2), dev)?;
        let freq_cis_imag = Self::read_tensor(r, (c.seq_len, head_size / 2), dev)?;
        Ok(Self {
            token_embedding_table,
            rms_att_weight,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_weight,
            w1,
            w2,
            w3,
            rms_final_weight,
            freq_cis_real,
            freq_cis_imag,
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
    let config = Config::from_reader(&mut file)?;
    println!("config: {config:?}");
    let _weights = TransformerWeights::from_reader(&mut file, &config, &device)?;

    Ok(())
}
