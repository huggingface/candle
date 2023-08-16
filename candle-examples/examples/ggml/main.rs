#![allow(dead_code)]
use clap::Parser;
use std::collections::HashMap;

use candle::quantized::ggml_file::Content;
use candle::quantized::{QMatMul, QTensor};
use candle::{DType, Device, Result, Tensor, D};
use candle_nn::Embedding;

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
}

struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
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
    fn new(content: Content) -> Result<Self> {
        let cpu = &Device::Cpu;
        let p = content.hparams;
        let mut wm = WeightMap(content.tensors.into_iter().collect::<HashMap<_, _>>());
        let tok_embeddings = wm.get("tok_embeddings.weight")?;
        let tok_embeddings = tok_embeddings.dequantize(cpu)?;
        let norm = RmsNorm::new(wm.get("norm.weight")?)?;
        let output = QMatMul::from_qtensor(wm.get("output.weight")?);
        let mut layers = Vec::with_capacity(p.n_layer as usize);
        for layer_idx in 0..p.n_layer {
            let prefix = format!("layers.{layer_idx}");
            let attention_wq = wm.get(&format!("layers.{layer_idx}.attention.wq.weight"))?;
            let attention_wk = wm.get(&format!("{prefix}.attention.wk.weight"))?;
            let attention_wv = wm.get(&format!("{prefix}.attention.wv.weight"))?;
            let attention_wo = wm.get(&format!("{prefix}.attention.wo.weight"))?;
            let feed_forward_w1 = wm.get(&format!("{prefix}.feed_forward.w1.weight"))?;
            let feed_forward_w2 = wm.get(&format!("{prefix}.feed_forward.w2.weight"))?;
            let feed_forward_w3 = wm.get(&format!("{prefix}.feed_forward.w3.weight"))?;
            let attention_norm = wm.get(&format!("{prefix}.attention_norm.weight"))?;
            let ffn_norm = wm.get(&format!("{prefix}.ffn_norm.weight"))?;
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
            })
        }
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, p.n_vocab as usize),
            layers,
            norm,
            output,
        })
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// GGML file to load, typically a .bin file generated by the quantize command from llama.cpp
    #[arg(long)]
    model: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut file = std::fs::File::open(args.model)?;
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
    let _model = ModelWeights::new(model);
    Ok(())
}
