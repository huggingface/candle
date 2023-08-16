#![allow(dead_code)]
use clap::Parser;
use std::collections::HashMap;
use std::io::Write;
use tokenizers::Tokenizer;

use candle::quantized::ggml_file::Content;
use candle::quantized::{QMatMul, QTensor};
use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::Embedding;
use candle_transformers::generation::LogitsProcessor;

const MAX_SEQ_LEN: usize = 4096;
const DEFAULT_PROMPT: &str = "My favorite theorem is ";

struct RmsNorm {
    scale: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(scale: QTensor) -> Result<Self> {
        let scale = scale.dequantize(&Device::Cpu)?;
        Ok(Self { scale, eps: 1e-5 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_size) = x.dims3()?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let norm_x = norm_x.broadcast_as((b_sz, seq_len, hidden_size))?;
        let x_normed = (x / (norm_x + self.eps)?.sqrt()?)?;
        let size = self.scale.dims1()?;
        let scale = self
            .scale
            .to_dtype(DType::F32)?
            .broadcast_as((b_sz, seq_len, size))?;
        let x = (scale * x_normed)?;
        Ok(x)
    }
}

struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_wo: QMatMul,
    attention_norm: RmsNorm,
    feed_forward_w1: QMatMul,
    feed_forward_w2: QMatMul,
    feed_forward_w3: QMatMul,
    ffn_norm: RmsNorm,
    n_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

impl LayerWeights {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (b_sz, _, seq_len, n_embd) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        let cos = cos.broadcast_as((b_sz, 1, seq_len, n_embd))?;
        let sin = sin.broadcast_as((b_sz, 1, seq_len, n_embd))?;
        let x1 = x.narrow(D::Minus1, 0, n_embd / 2)?;
        let x2 = x.narrow(D::Minus1, n_embd / 2, n_embd / 2)?;
        let rotate_x = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
        let rope = (x.broadcast_mul(&cos)? + rotate_x.broadcast_mul(&sin)?)?;
        Ok(rope)
    }

    fn forward_attn(&mut self, x: &Tensor, mask: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (b_sz, seq_len, n_embd) = x.dims3()?;
        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                let k = Tensor::cat(&[k_cache, &k], 2)?.contiguous()?;
                let v = Tensor::cat(&[v_cache, &v], 2)?.contiguous()?;
                (k, v)
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // If we start supporting MQA, we need to repeat the k and v tensors here.

        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let mask = mask.broadcast_as(att.shape())?;
        let att = masked_fill(&att, &mask, f32::NEG_INFINITY)?;
        let att = candle_nn::ops::softmax(&att, D::Minus1)?;
        // Convert to contiguous as matmul doesn't support strided vs for now.
        let y = att.matmul(&v.contiguous()?)?;
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        let y = self.attention_wo.forward(&y)?;
        Ok(y)
    }
}

struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    // TODO: Switch to using QMatMul instead of linear once we have support for Q6K/Q8K.
    output: candle_nn::Linear,
    masks: HashMap<usize, Tensor>,
}

struct WeightMap(HashMap<String, QTensor>);
impl WeightMap {
    fn get(&mut self, name: &str) -> Result<QTensor> {
        match self.0.remove(name) {
            None => candle::bail!("cannot find tensor with name '{name}'"),
            Some(tensor) => Ok(tensor),
        }
    }
}

impl ModelWeights {
    fn new(mut ct: Content) -> Result<Self> {
        let cpu = &Device::Cpu;
        let head_dim = (ct.hparams.n_embd / ct.hparams.n_head) as usize;

        // precompute freqs_cis
        let theta: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / 10000f32.powf(i as f32 / head_dim as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), &Device::Cpu)?;
        let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, &Device::Cpu)?
            .to_dtype(DType::F32)?
            .reshape((MAX_SEQ_LEN, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        // This is different from the paper, see:
        // https://github.com/huggingface/transformers/blob/6112b1c6442aaf7affd2b0676a1cd4eee30c45cf/src/transformers/models/llama/modeling_llama.py#L112
        let idx_theta = Tensor::cat(&[&idx_theta, &idx_theta], D::Minus1)?;
        let cos = idx_theta.cos()?;
        let sin = idx_theta.sin()?;

        let tok_embeddings = ct.remove("tok_embeddings.weight")?;
        let tok_embeddings = tok_embeddings.dequantize(cpu)?;
        let norm = RmsNorm::new(ct.remove("norm.weight")?)?;
        let output = ct.remove("output.weight")?;
        let output = candle_nn::Linear::new(output.dequantize(cpu)?, None);
        let mut layers = Vec::with_capacity(ct.hparams.n_layer as usize);
        for layer_idx in 0..ct.hparams.n_layer {
            let prefix = format!("layers.{layer_idx}");
            let attention_wq = ct.remove(&format!("layers.{layer_idx}.attention.wq.weight"))?;
            let attention_wk = ct.remove(&format!("{prefix}.attention.wk.weight"))?;
            let attention_wv = ct.remove(&format!("{prefix}.attention.wv.weight"))?;
            let attention_wo = ct.remove(&format!("{prefix}.attention.wo.weight"))?;
            let feed_forward_w1 = ct.remove(&format!("{prefix}.feed_forward.w1.weight"))?;
            let feed_forward_w2 = ct.remove(&format!("{prefix}.feed_forward.w2.weight"))?;
            let feed_forward_w3 = ct.remove(&format!("{prefix}.feed_forward.w3.weight"))?;
            let attention_norm = ct.remove(&format!("{prefix}.attention_norm.weight"))?;
            let ffn_norm = ct.remove(&format!("{prefix}.ffn_norm.weight"))?;
            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq),
                attention_wk: QMatMul::from_qtensor(attention_wk),
                attention_wv: QMatMul::from_qtensor(attention_wv),
                attention_wo: QMatMul::from_qtensor(attention_wo),
                attention_norm: RmsNorm::new(attention_norm)?,
                feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1),
                feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2),
                feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3),
                ffn_norm: RmsNorm::new(ffn_norm)?,
                n_head: ct.hparams.n_head as usize,
                head_dim: (ct.hparams.n_embd / ct.hparams.n_head) as usize,
                cos: cos.clone(),
                sin: sin.clone(),
                kv_cache: None,
            })
        }
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, ct.hparams.n_embd as usize),
            layers,
            norm,
            output,
            masks: HashMap::new(),
        })
    }

    fn mask(&mut self, t: usize) -> Result<Tensor> {
        if let Some(mask) = self.masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (t, t), &Device::Cpu)?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }

    fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mask = self.mask(seq_len)?;
        let mut layer_in = self.tok_embeddings.forward(x)?;
        for layer in self.layers.iter_mut() {
            let x = layer_in;
            let residual = &x;
            let x = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn(&x, &mask, index_pos)?;
            let x = (attn + residual)?;

            // MLP
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let w1 = layer.feed_forward_w1.forward(&x)?;
            let w3 = layer.feed_forward_w3.forward(&x)?;
            let mlp = layer
                .feed_forward_w2
                .forward(&(candle_nn::ops::silu(&w1)? * w3)?)?;
            layer_in = (mlp + residual)?;
        }
        let x = self.norm.forward(&layer_in)?;
        let x = x.i((.., seq_len - 1, ..))?;
        self.output.forward(&x)
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// GGML file to load, typically a .bin file generated by the quantize command from llama.cpp
    #[arg(long)]
    model: Option<String>,

    /// The initial prompt.
    #[arg(long)]
    prompt: Option<String>,

    /// The length of the sample to generate (in tokens).
    #[arg(long, default_value_t = 100)]
    sample_len: usize,

    /// The tokenizer config in json format.
    #[arg(long)]
    tokenizer: Option<String>,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,
}

impl Args {
    fn tokenizer(&self) -> anyhow::Result<Tokenizer> {
        let tokenizer_path = match &self.tokenizer {
            Some(config) => std::path::PathBuf::from(config),
            None => {
                let api = hf_hub::api::sync::Api::new()?;
                let api = api.model("hf-internal-testing/llama-tokenizer".to_string());
                api.get("tokenizer.json")?
            }
        };
        Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)
    }

    fn model(&self) -> anyhow::Result<std::path::PathBuf> {
        let model_path = match &self.model {
            Some(config) => std::path::PathBuf::from(config),
            None => {
                let api = hf_hub::api::sync::Api::new()?;
                let api = api.model("TheBloke/Llama-2-7B-GGML".to_string());
                api.get("llama-2-7b.ggmlv3.q4_0.bin")?
            }
        };
        Ok(model_path)
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let mut file = std::fs::File::open(&args.model()?)?;
    let start = std::time::Instant::now();
    let model = Content::read(&mut file)?;

    let mut total_size_in_bytes = 0;
    for (_, tensor) in model.tensors.iter() {
        let elem_count = tensor.shape().elem_count();
        total_size_in_bytes += elem_count * tensor.dtype().type_size() / tensor.dtype().blck_size();
    }
    let total_size = if total_size_in_bytes < 1_000 {
        format!("{}B", total_size_in_bytes)
    } else if total_size_in_bytes < 1_000_000 {
        format!("{:.2}KB", total_size_in_bytes as f64 / 1e3)
    } else if total_size_in_bytes < 1_000_000_000 {
        format!("{:.2}MB", total_size_in_bytes as f64 / 1e6)
    } else {
        format!("{:.2}GB", total_size_in_bytes as f64 / 1e9)
    };

    println!(
        "loaded {:?} tensors ({}) in {:.2}s",
        model.tensors.len(),
        total_size,
        start.elapsed().as_secs_f32(),
    );
    println!("params: {:?}", model.hparams);
    let mut model = ModelWeights::new(model)?;
    println!("model built");

    let tokenizer = args.tokenizer()?;
    let prompt = args.prompt.as_ref().map_or(DEFAULT_PROMPT, |p| p.as_str());
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();
    let mut index_pos = 0;
    let mut logits_processor = LogitsProcessor::new(args.seed, args.temperature);
    let start_gen = std::time::Instant::now();
    let mut token_generated = 0;
    print!("{prompt}");
    for index in 0..args.sample_len {
        let context_size = if index == 0 { tokens.len() } else { 1 };
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &Device::Cpu)?.unsqueeze(0)?;
        let logits = model.forward(&input, index_pos)?;
        let logits = logits.squeeze(0)?;
        index_pos += ctxt.len();

        let next_token = logits_processor.sample(&logits)?;
        token_generated += 1;
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
        "\n\n{} tokens generated ({} token/s)\n",
        token_generated,
        token_generated as f64 / dt.as_secs_f64(),
    );
    Ok(())
}
