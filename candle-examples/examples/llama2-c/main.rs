// https://github.com/karpathy/llama2.c
#![allow(dead_code)]
#![allow(unused)]

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

mod model;
use clap::Parser;

use anyhow::Result;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use candle::{DType, Device, Error, IndexOp, Layout, Shape, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder};
use candle_transformers::generation::LogitsProcessor;

use model::{Config, Llama};

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
    freq_cis_real: Tensor, // (seq_len, head_size/2)
    freq_cis_imag: Tensor, // (seq_len, head_size/2)
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
            norm_eps: 1e-5,
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

    fn var_builder(&self, device: &Device) -> Result<VarBuilder> {
        let mut ws = std::collections::HashMap::new();
        let mut insert = |name: &str, t: Tensor| {
            ws.insert(name.to_string(), t);
        };
        insert("rot.freq_cis_real", self.freq_cis_real.clone());
        insert("rot.freq_cis_imag", self.freq_cis_imag.clone());
        insert(
            "model.embed_tokens.weight",
            self.token_embedding_table.clone(),
        );
        insert("lm_head.weight", self.token_embedding_table.clone());
        insert("model.norm.weight", self.rms_final_weight.clone());
        let vb = VarBuilder::from_tensors(ws, DType::F32, device);
        Ok(vb)
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
    let weights = TransformerWeights::from_reader(&mut file, &config, &device)?;
    let vb = weights.var_builder(&device)?;
    let cache = model::Cache::new(true, &config, vb.pp("rot"))?;
    let model = Llama::load(vb, &cache, &config)?;

    println!("starting the inference loop");
    let mut logits_processor = LogitsProcessor::new(299792458, None);
    let mut new_tokens: Vec<u32> = vec![];
    let start_gen = std::time::Instant::now();
    let mut index_pos = 0;
    let mut tokens = vec![1u32];

    for index in 0..config.seq_len {
        let start_gen = std::time::Instant::now();
        let context_size = if cache.use_kv_cache && index > 0 {
            1
        } else {
            tokens.len()
        };
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, index_pos)?;
        let logits = logits.squeeze(0)?;
        index_pos += ctxt.len();

        let next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);
        new_tokens.push(next_token);
        println!("> {:?}", start_gen.elapsed());
        println!(
            "{} token: {} '{}'",
            index + 1,
            next_token,
            0,
            // tokenizer.decode(vec![next_token], true).map_err(E::msg)?
        );
    }
    let dt = start_gen.elapsed();
    println!(
        "{} tokens generated ({} token/s)\n----\n{}\n----",
        config.seq_len,
        config.seq_len as f64 / dt.as_secs_f64(),
        0,
        // tokenizer.decode(new_tokens, true).map_err(E::msg)?
    );
    Ok(())
}
