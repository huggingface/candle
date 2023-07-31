// https://github.com/karpathy/llama2.c
#![allow(dead_code)]
#![allow(unused)]

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

mod model;
use clap::{Parser, ValueEnum};

use anyhow::{Error as E, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use candle::{DType, Device, Error, IndexOp, Layout, Shape, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder};
use candle_transformers::generation::LogitsProcessor;
use std::io::Write;
use tokenizers::Tokenizer;

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

fn read_i32<R: std::io::Read>(r: &mut R) -> Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

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

impl Config {
    fn from_reader<R: std::io::Read>(r: &mut R) -> Result<Self> {
        let dim = read_i32(r)? as usize;
        let hidden_dim = read_i32(r)? as usize;
        let n_layers = read_i32(r)? as usize;
        let n_heads = read_i32(r)? as usize;
        let n_kv_heads = read_i32(r)? as usize;
        let vocab_size = read_i32(r)? as usize;
        let seq_len = read_i32(r)? as usize;
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
    fn from_reader<R: std::io::Read>(r: &mut R, c: &Config, dev: &Device) -> Result<Self> {
        let token_embedding_table = read_tensor(r, (c.vocab_size, c.dim), dev)?;
        let rms_att_weight = read_tensor(r, (c.n_layers, c.dim), dev)?;
        let wq = read_tensor(r, (c.n_layers, c.dim, c.dim), dev)?;
        let wk = read_tensor(r, (c.n_layers, c.dim, c.dim), dev)?;
        let wv = read_tensor(r, (c.n_layers, c.dim, c.dim), dev)?;
        let wo = read_tensor(r, (c.n_layers, c.dim, c.dim), dev)?;
        let rms_ffn_weight = read_tensor(r, (c.n_layers, c.dim), dev)?;
        let w1 = read_tensor(r, (c.n_layers, c.hidden_dim, c.dim), dev)?;
        let w2 = read_tensor(r, (c.n_layers, c.dim, c.hidden_dim), dev)?;
        let w3 = read_tensor(r, (c.n_layers, c.hidden_dim, c.dim), dev)?;
        let rms_final_weight = read_tensor(r, c.dim, dev)?;
        let head_size = c.head_size();
        let freq_cis_real = read_tensor(r, (c.seq_len, head_size / 2), dev)?;
        let freq_cis_imag = read_tensor(r, (c.seq_len, head_size / 2), dev)?;
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

    fn var_builder(&self, cfg: &Config, device: &Device) -> Result<VarBuilder> {
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
        for layer in 0..cfg.n_layers {
            ws.insert(
                format!("model.layers.{layer}.self_attn.q_proj.weight"),
                self.wq.i(layer)?,
            );
            ws.insert(
                format!("model.layers.{layer}.self_attn.k_proj.weight"),
                self.wk.i(layer)?,
            );
            ws.insert(
                format!("model.layers.{layer}.self_attn.v_proj.weight"),
                self.wv.i(layer)?,
            );
            ws.insert(
                format!("model.layers.{layer}.self_attn.o_proj.weight"),
                self.wo.i(layer)?,
            );
            ws.insert(
                format!("model.layers.{layer}.mlp.gate_proj.weight"),
                self.w1.i(layer)?,
            );
            ws.insert(
                format!("model.layers.{layer}.mlp.down_proj.weight"),
                self.w2.i(layer)?,
            );
            ws.insert(
                format!("model.layers.{layer}.mlp.up_proj.weight"),
                self.w3.i(layer)?,
            );
            ws.insert(
                format!("model.layers.{layer}.input_layernorm.weight"),
                self.rms_att_weight.i(layer)?,
            );
            ws.insert(
                format!("model.layers.{layer}.post_attention_layernorm.weight"),
                self.rms_ffn_weight.i(layer)?,
            );
        }
        let vb = VarBuilder::from_tensors(ws, DType::F32, device);
        Ok(vb)
    }
}

#[derive(ValueEnum, Debug, Clone)]
enum Task {
    Inference,
    Evaluation,
    Training,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The task to be performed, inference, training or evaluation.
    #[clap(value_enum, default_value_t = Task::Inference)]
    task: Task,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Config file in binary format.
    #[arg(long)]
    config: Option<String>,

    /// Tokenizer config file.
    #[arg(long)]
    tokenizer: Option<String>,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    #[arg(long, default_value = "karpathy/tinyllamas")]
    model_id: String,

    /// The model to be used when getting it from the hub. Possible
    /// values are 'stories15M.bin', 'stories42M.bin', see more at:
    /// https://huggingface.co/karpathy/tinyllamas/tree/main
    #[arg(long, default_value = "stories15M.bin")]
    which_model: String,

    #[arg(long, default_value = "")]
    prompt: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let config_path = match &args.config {
        Some(config) => std::path::PathBuf::from(config),
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            println!("loading the model weights from {}", args.model_id);
            let api = api.model(args.model_id.clone());
            api.get(&args.which_model)?
        }
    };

    let tokenizer_path = match &args.tokenizer {
        Some(config) => std::path::PathBuf::from(config),
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model("hf-internal-testing/llama-tokenizer".to_string());
            api.get("tokenizer.json")?
        }
    };
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;

    match args.task {
        Task::Inference => run_inference(tokenizer, &config_path, args)?,
        Task::Evaluation => run_eval(tokenizer, &config_path, args)?,
        Task::Training => todo!(),
    }
    Ok(())
}

fn run_eval(tokenizer: Tokenizer, config_path: &std::path::PathBuf, args: Args) -> Result<()> {
    use std::io::BufRead;

    let device = candle_examples::device(args.cpu)?;
    let mut file = std::fs::File::open(config_path)?;
    let config = Config::from_reader(&mut file)?;
    let weights = TransformerWeights::from_reader(&mut file, &config, &device)?;
    let vb = weights.var_builder(&config, &device)?;
    let cache = model::Cache::new(false, &config, vb.pp("rot"))?;
    let model = Llama::load(vb, &cache, config)?;

    let api = hf_hub::api::sync::Api::new()?;
    let model_id = "roneneldan/TinyStories"; // TODO: Make this configurable.
    println!("loading the evaluation dataset from {}", model_id);
    let api = api.dataset(model_id.to_string());
    let dataset_path = api.get("TinyStories-valid.txt")?;
    let file = std::fs::File::open(dataset_path)?;
    let file = std::io::BufReader::new(file);
    let mut tokens = vec![];
    for line in file.lines() {
        let line = tokenizer.encode(line?, false).map_err(E::msg)?;
        tokens.push(line.get_ids().to_vec())
    }
    let tokens = tokens.concat();
    println!("dataset loaded and encoded: {} tokens", tokens.len());
    let seq_len = 256;
    for start_idx in (0..tokens.len()).step_by(seq_len) {
        if start_idx + seq_len + 1 > tokens.len() {
            break;
        }
        let tokens = &tokens[start_idx..start_idx + seq_len + 1];
        let inputs = Tensor::new(&tokens[..seq_len], &device)?.unsqueeze(0)?;
        let targets = Tensor::new(&tokens[1..], &device)?;
        let logits = model.forward(&inputs, 0)?.squeeze(0)?;
        let loss = candle_nn::loss::cross_entropy(&logits, &targets)?;
        println!("{start_idx} {}", loss.to_vec0::<f32>()?);
    }
    Ok(())
}

fn run_inference(tokenizer: Tokenizer, config_path: &std::path::PathBuf, args: Args) -> Result<()> {
    let device = candle_examples::device(args.cpu)?;

    let mut file = std::fs::File::open(config_path)?;
    let config = Config::from_reader(&mut file)?;
    let weights = TransformerWeights::from_reader(&mut file, &config, &device)?;
    let vb = weights.var_builder(&config, &device)?;
    let cache = model::Cache::new(true, &config, vb.pp("rot"))?;
    let model = Llama::load(vb, &cache, config)?;

    println!("starting the inference loop");
    let mut logits_processor = LogitsProcessor::new(299792458, args.temperature);
    let mut index_pos = 0;

    print!("{}", args.prompt);
    let mut tokens = tokenizer
        .encode(args.prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    let start_gen = std::time::Instant::now();
    for index in 0.. {
        if tokens.len() >= model.config.seq_len {
            break;
        }
        let start_gen = std::time::Instant::now();
        let context_size = if index > 0 { 1 } else { tokens.len() };
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, index_pos)?;
        let logits = logits.i((0, logits.dim(1)? - 1))?;
        index_pos += ctxt.len();

        let next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);
        // Extracting the last token as a string is complicated, here we just apply some simple
        // heuristics as it seems to work well enough for this example. See the following for more
        // details:
        // https://github.com/huggingface/tokenizers/issues/1141#issuecomment-1562644141
        if let Some(text) = tokenizer.id_to_token(next_token) {
            let text = text.replace('▁', " ").replace("<0x0A>", "\n");
            print!("{text}");
            std::io::stdout().flush()?;
        }
    }
    let dt = start_gen.elapsed();
    println!(
        "\n{} tokens generated ({:.2} token/s)\n",
        tokens.len(),
        tokens.len() as f64 / dt.as_secs_f64(),
    );
    Ok(())
}
