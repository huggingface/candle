#![allow(dead_code)]
// https://huggingface.co/facebook/musicgen-small/tree/main
// https://github.com/huggingface/transformers/blob/cd4584e3c809bb9e1392ccd3fe38b40daba5519a/src/transformers/models/musicgen/modeling_musicgen.py

use anyhow::{Error as E, Result};
use candle::{safetensors::SafeTensors, DType, Device, Shape, Tensor};
use clap::Parser;
use std::collections::HashMap;

const DTYPE: DType = DType::F32;

struct VarBuilder<'a> {
    safetensors: Option<(HashMap<String, usize>, Vec<SafeTensors<'a>>)>,
    dtype: DType,
    device: Device,
}

impl<'a> VarBuilder<'a> {
    pub fn from_safetensors(
        safetensors: Vec<SafeTensors<'a>>,
        dtype: DType,
        device: Device,
    ) -> Self {
        let mut routing = HashMap::new();
        for (index, sf) in safetensors.iter().enumerate() {
            for k in sf.names() {
                routing.insert(k.to_string(), index);
            }
        }
        Self {
            safetensors: Some((routing, safetensors)),
            device,
            dtype,
        }
    }

    pub fn zeros(dtype: DType, device: Device) -> Self {
        Self {
            safetensors: None,
            device,
            dtype,
        }
    }

    pub fn get<S: Into<Shape>>(&self, s: S, tensor_name: &str) -> candle::Result<Tensor> {
        let s: Shape = s.into();
        match &self.safetensors {
            None => Tensor::zeros(s, self.dtype, &self.device),
            Some((routing, safetensors)) => {
                // Unwrap or 0  just to let the proper error flow.
                let index = routing.get(tensor_name).unwrap_or(&0);
                let tensor = safetensors[*index]
                    .tensor(tensor_name, &self.device)?
                    .to_dtype(self.dtype)?;
                if *tensor.shape() != s {
                    let msg = format!("shape mismatch for {tensor_name}");
                    Err(candle::Error::UnexpectedShape {
                        msg,
                        expected: s,
                        got: tensor.shape().clone(),
                    })?
                }
                Ok(tensor)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HiddenAct {
    Gelu,
    Relu,
}

impl HiddenAct {
    fn forward(&self, xs: &Tensor) -> candle::Result<Tensor> {
        match self {
            Self::Gelu => xs.gelu(),
            Self::Relu => xs.relu(),
        }
    }
}

// https://github.com/huggingface/transformers/blob/cd4584e3c809bb9e1392ccd3fe38b40daba5519a/src/transformers/models/musicgen/configuration_musicgen.py#L83
#[derive(Debug, Clone, PartialEq)]
struct Config {
    vocab_size: usize,
    max_position_embeddings: usize,
    num_hidden_layers: usize,
    ffn_dim: usize,
    num_attention_heads: usize,
    layerdrop: f64,
    use_cache: bool,
    activation_function: HiddenAct,
    hidden_size: usize,
    dropout: f64,
    attention_dropout: f64,
    activation_dropout: f64,
    initializer_factor: f64,
    scale_embedding: bool,
    num_codebooks: usize,
    pad_token_id: usize,
    bos_token_id: usize,
    eos_token_id: Option<usize>,
    tie_word_embeddings: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 2048,
            max_position_embeddings: 2048,
            num_hidden_layers: 24,
            ffn_dim: 4096,
            num_attention_heads: 16,
            layerdrop: 0.0,
            use_cache: true,
            activation_function: HiddenAct::Gelu, // TODO: Handle old style gelu.
            hidden_size: 1024,
            dropout: 0.1,
            attention_dropout: 0.0,
            activation_dropout: 0.0,
            initializer_factor: 0.02,
            scale_embedding: false,
            num_codebooks: 4,
            pad_token_id: 2048,
            bos_token_id: 2048,
            eos_token_id: None,
            tie_word_embeddings: false,
        }
    }
}

struct Embedding {
    embeddings: Tensor,
    hidden_size: usize,
}

impl Embedding {
    fn new(embeddings: Tensor, hidden_size: usize) -> Self {
        Self {
            embeddings,
            hidden_size,
        }
    }

    fn load(vocab_size: usize, hidden_size: usize, p: &str, vb: &VarBuilder) -> Result<Self> {
        let embeddings = vb.get((vocab_size, hidden_size), &format!("{p}.weight"))?;
        Ok(Self::new(embeddings, hidden_size))
    }

    fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        let mut final_dims = indexes.dims().to_vec();
        final_dims.push(self.hidden_size);
        let indexes = indexes.flatten_all()?;
        let values = Tensor::embedding(&indexes, &self.embeddings)?;
        let values = values.reshape(final_dims)?;
        Ok(values)
    }
}

struct Linear {
    weight: Tensor,
    bias: Tensor,
}

impl Linear {
    fn new(weight: Tensor, bias: Tensor) -> Self {
        Self { weight, bias }
    }

    fn load(size1: usize, size2: usize, p: &str, vb: &VarBuilder) -> Result<Self> {
        let weight = vb.get((size2, size1), &format!("{p}.weight"))?;
        let bias = vb.get(size2, &format!("{p}.bias"))?;
        Ok(Self::new(weight, bias))
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (bsize, _, _) = x.shape().r3()?;
        let w = self.weight.broadcast_left(bsize)?.t()?;
        let x = x.matmul(&w)?;
        let x = x.broadcast_add(&self.bias)?;
        Ok(x)
    }
}

struct Dropout {
    pr: f64,
}

impl Dropout {
    fn new(pr: f64) -> Self {
        Self { pr }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // TODO
        Ok(x.clone())
    }
}

// This layer norm version handles both weight and bias so removes the mean.
struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        Self { weight, bias, eps }
    }

    fn load(size: usize, eps: f64, p: &str, vb: &VarBuilder) -> Result<Self> {
        let weight = vb.get(size, &format!("{p}.weight"))?;
        let bias = vb.get(size, &format!("{p}.bias"))?;
        Ok(Self::new(weight, bias, eps))
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_bsize, _seq_len, hidden_size) = x.shape().r3()?;
        let mean_x = (x.sum(&[2])? / hidden_size as f64)?;
        let x = x.broadcast_sub(&mean_x)?;
        let norm_x = ((&x * &x)?.sum(&[2])? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        let x = x_normed
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)?;
        Ok(x)
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    tokenizer_config: String,

    #[arg(long)]
    weights: String,
}

fn main() -> Result<()> {
    use tokenizers::Tokenizer;

    let args = Args::parse();
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0)?
    };

    let mut tokenizer = Tokenizer::from_file(args.tokenizer_config).map_err(E::msg)?;
    let _tokenizer = tokenizer.with_padding(None).with_truncation(None);

    let weights = unsafe { candle::safetensors::MmapedFile::new(args.weights)? };
    let weights = weights.deserialize()?;
    let _vb = VarBuilder::from_safetensors(vec![weights], DTYPE, device);
    Ok(())
}
