#![allow(dead_code)]
// https://github.com/openai/whisper/blob/main/whisper/model.py

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

#[derive(Debug, Clone, PartialEq)]
struct Config {
    n_mels: usize,
    n_audio_ctx: usize,
    n_audio_state: usize,
    n_audio_head: usize,
    n_audio_layer: usize,
    n_vocab: usize,
    n_text_ctx: usize,
    n_text_state: usize,
    n_text_head: usize,
    n_text_layer: usize,
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
    bias: Option<Tensor>,
}

impl Linear {
    fn load(size1: usize, size2: usize, p: &str, vb: &VarBuilder) -> Result<Self> {
        let weight = vb.get((size2, size1), &format!("{p}.weight"))?;
        let bias = vb.get(size2, &format!("{p}.bias"))?;
        Ok(Self {
            weight,
            bias: Some(bias),
        })
    }

    fn load_no_bias(size1: usize, size2: usize, p: &str, vb: &VarBuilder) -> Result<Self> {
        let weight = vb.get((size2, size1), &format!("{p}.weight"))?;
        Ok(Self { weight, bias: None })
    }

    fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        let (bsize, _, _) = x.shape().r3()?;
        let w = self.weight.broadcast_left(bsize)?.t()?;
        let x = x.matmul(&w)?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
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
    fn load(size: usize, p: &str, vb: &VarBuilder) -> Result<Self> {
        let weight = vb.get(size, &format!("{p}.weight"))?;
        let bias = vb.get(size, &format!("{p}.bias"))?;
        Ok(Self {
            weight,
            bias,
            eps: 1e-5,
        })
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

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L62
struct MultiHeadAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    n_head: usize,
}

impl MultiHeadAttention {
    fn load(n_state: usize, n_head: usize, p: &str, vb: &VarBuilder) -> Result<Self> {
        let query = Linear::load(n_state, n_state, &format!("{p}.query"), vb)?;
        let value = Linear::load_no_bias(n_state, n_state, &format!("{p}.value"), vb)?;
        let key = Linear::load(n_state, n_state, &format!("{p}.key"), vb)?;
        let out = Linear::load(n_state, n_state, &format!("{p}.out"), vb)?;
        Ok(Self {
            query,
            key,
            value,
            out,
            n_head,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let q = self.query.forward(x)?;
        let k = self.key.forward(x)?;
        let v = self.value.forward(x)?;
        let wv = self.qkv_attention(&q, &k, &v)?;
        let out = self.out.forward(&wv)?;
        Ok(out)
    }

    fn qkv_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (n_batch, n_ctx, n_state) = q.shape().r3()?;
        let target_dims = &[n_batch, n_ctx, self.n_head, n_state / self.n_head];
        let scale = ((n_state / self.n_head) as f64).powf(-0.25);
        let q = (q.reshape(target_dims)?.transpose(1, 2)? * scale)?;
        let k = (k.reshape(target_dims)?.transpose(1, 2)?.transpose(2, 3)? * scale)?;
        let v = v.reshape(target_dims)?.transpose(1, 2)?;
        let qk = q.matmul(&k)?;
        let w = qk.softmax(qk.rank() - 1)?;
        let wv = w.matmul(&v)?.transpose(1, 2)?.flatten(Some(2), None)?;
        Ok(wv)
    }
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L111
struct ResidualAttentionBlock {
    attn: MultiHeadAttention,
    attn_ln: LayerNorm,
    cross_attn: Option<MultiHeadAttention>,
    cross_attn_ln: Option<LayerNorm>,
    mlp_linear1: Linear,
    mlp_linear2: Linear,
    mlp_ln: LayerNorm,
}

impl ResidualAttentionBlock {
    fn load(n_state: usize, n_head: usize, ca: bool, p: &str, vb: &VarBuilder) -> Result<Self> {
        let attn = MultiHeadAttention::load(n_state, n_head, &format!("{p}.attn"), vb)?;
        let attn_ln = LayerNorm::load(n_state, &format!("{p}.attn_ln"), vb)?;
        let (cross_attn, cross_attn_ln) = if ca {
            let cross_attn =
                MultiHeadAttention::load(n_state, n_head, &format!("{p}.cross_attn"), vb)?;
            let cross_attn_ln = LayerNorm::load(n_state, &format!("{p}.cross_attn_ln"), vb)?;
            (Some(cross_attn), Some(cross_attn_ln))
        } else {
            (None, None)
        };
        let n_mlp = n_state * 4;
        let mlp_linear1 = Linear::load(n_state, n_mlp, &format!("{p}.mlp.0"), vb)?;
        let mlp_linear2 = Linear::load(n_mlp, n_state, &format!("{p}.mlp.2"), vb)?;
        let mlp_ln = LayerNorm::load(n_state, &format!("{p}.mlp_ln"), vb)?;
        Ok(Self {
            attn,
            attn_ln,
            cross_attn,
            cross_attn_ln,
            mlp_linear1,
            mlp_linear2,
            mlp_ln,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let attn = self.attn.forward(&self.attn_ln.forward(x)?)?;
        let mut x = (x + attn)?;
        // Cross-Attn
        if let Some(cross_attn) = &self.cross_attn {
            x = cross_attn.forward(&x)?
        }
        if let Some(cross_attn_ln) = &self.cross_attn_ln {
            x = cross_attn_ln.forward(&x)?
        }
        // Mlp
        let mlp = self.mlp_linear2.forward(
            &self
                .mlp_linear1
                .forward(&self.mlp_ln.forward(&x)?)?
                .gelu()?,
        )?;
        Ok((x + mlp)?)
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
