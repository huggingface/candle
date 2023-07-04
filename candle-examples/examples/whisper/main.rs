#![allow(dead_code)]
// https://github.com/openai/whisper/blob/main/whisper/model.py
// TODO:
// - kv-cache support?

use anyhow::Result;
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

impl Config {
    fn tiny_en() -> Self {
        Self {
            n_mels: 80,
            n_vocab: 51864,
            n_audio_ctx: 1500,
            n_audio_state: 384,
            n_audio_head: 6,
            n_audio_layer: 4,
            n_text_ctx: 448,
            n_text_state: 384,
            n_text_head: 6,
            n_text_layer: 4,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ConvConfig {
    padding: usize,
    stride: usize,
}

impl Default for ConvConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            stride: 1,
        }
    }
}

struct Conv1D {
    weight: Tensor,
    bias: Option<Tensor>,
    config: ConvConfig,
}

impl Conv1D {
    fn load(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        config: ConvConfig,
        p: &str,
        vb: &VarBuilder,
    ) -> Result<Self> {
        let weight = vb.get(
            (out_channels, in_channels, kernel_size),
            &format!("{p}.weight"),
        )?;
        let bias = vb.get(out_channels, &format!("{p}.bias"))?;
        Ok(Self {
            weight,
            bias: Some(bias),
            config,
        })
    }

    fn load_no_bias(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        config: ConvConfig,
        p: &str,
        vb: &VarBuilder,
    ) -> Result<Self> {
        let weight = vb.get(
            (out_channels, in_channels, kernel_size),
            &format!("{p}.weight"),
        )?;
        Ok(Self {
            weight,
            bias: None,
            config,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.conv1d(&self.weight, self.config.padding, self.config.stride)?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => {
                let b = bias.shape().r1()?;
                let bias = bias.reshape((1, b, 1))?;
                Ok(x.broadcast_add(&bias)?)
            }
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
        let value = Linear::load(n_state, n_state, &format!("{p}.value"), vb)?;
        let key = Linear::load_no_bias(n_state, n_state, &format!("{p}.key"), vb)?;
        let out = Linear::load(n_state, n_state, &format!("{p}.out"), vb)?;
        Ok(Self {
            query,
            key,
            value,
            out,
            n_head,
        })
    }

    fn forward(&self, x: &Tensor, xa: Option<&Tensor>, mask: Option<&Tensor>) -> Result<Tensor> {
        let q = self.query.forward(x)?;
        let k = self.key.forward(xa.unwrap_or(x))?;
        let v = self.value.forward(xa.unwrap_or(x))?;
        let wv = self.qkv_attention(&q, &k, &v, mask)?;
        let out = self.out.forward(&wv)?;
        Ok(out)
    }

    fn reshape_head(&self, x: &Tensor) -> Result<Tensor> {
        let (n_batch, n_ctx, n_state) = x.shape().r3()?;
        let target_dims = &[n_batch, n_ctx, self.n_head, n_state / self.n_head];
        Ok(x.reshape(target_dims)?.transpose(1, 2)?)
    }

    fn qkv_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (_, n_ctx, n_state) = q.shape().r3()?;
        let scale = ((n_state / self.n_head) as f64).powf(-0.25);
        let q = (self.reshape_head(q)? * scale)?;
        let k = (self.reshape_head(k)?.transpose(2, 3)? * scale)?;
        let v = self.reshape_head(v)?.contiguous()?;
        let mut qk = q.matmul(&k)?;
        if let Some(mask) = mask {
            let mask = mask.narrow(0, 0, n_ctx)?.narrow(1, 0, n_ctx)?;
            qk = qk.broadcast_add(&mask)?
        }
        let w = qk.softmax(qk.rank() - 1)?;
        let wv = w.matmul(&v)?.transpose(1, 2)?.flatten(Some(2), None)?;
        Ok(wv)
    }
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L111
struct ResidualAttentionBlock {
    attn: MultiHeadAttention,
    attn_ln: LayerNorm,
    cross_attn: Option<(MultiHeadAttention, LayerNorm)>,
    mlp_linear1: Linear,
    mlp_linear2: Linear,
    mlp_ln: LayerNorm,
}

impl ResidualAttentionBlock {
    fn load(n_state: usize, n_head: usize, ca: bool, p: &str, vb: &VarBuilder) -> Result<Self> {
        let attn = MultiHeadAttention::load(n_state, n_head, &format!("{p}.attn"), vb)?;
        let attn_ln = LayerNorm::load(n_state, &format!("{p}.attn_ln"), vb)?;
        let cross_attn = if ca {
            let cross_attn =
                MultiHeadAttention::load(n_state, n_head, &format!("{p}.cross_attn"), vb)?;
            let cross_attn_ln = LayerNorm::load(n_state, &format!("{p}.cross_attn_ln"), vb)?;
            Some((cross_attn, cross_attn_ln))
        } else {
            None
        };
        let n_mlp = n_state * 4;
        let mlp_linear1 = Linear::load(n_state, n_mlp, &format!("{p}.mlp.0"), vb)?;
        let mlp_linear2 = Linear::load(n_mlp, n_state, &format!("{p}.mlp.2"), vb)?;
        let mlp_ln = LayerNorm::load(n_state, &format!("{p}.mlp_ln"), vb)?;
        Ok(Self {
            attn,
            attn_ln,
            cross_attn,
            mlp_linear1,
            mlp_linear2,
            mlp_ln,
        })
    }

    fn forward(&self, x: &Tensor, xa: Option<&Tensor>, mask: Option<&Tensor>) -> Result<Tensor> {
        let attn = self.attn.forward(&self.attn_ln.forward(x)?, None, mask)?;
        let mut x = (x + attn)?;
        if let Some((attn, ln)) = &self.cross_attn {
            x = (&x + attn.forward(&ln.forward(&x)?, xa, None)?)?;
        }
        let mlp = self.mlp_linear2.forward(
            &self
                .mlp_linear1
                .forward(&self.mlp_ln.forward(&x)?)?
                .gelu()?,
        )?;
        Ok((x + mlp)?)
    }
}

fn sinusoids(length: usize, channels: usize) -> Result<Tensor> {
    let max_timescale = 10000f32;
    let log_timescale_increment = max_timescale.ln() / (channels / 2 - 1) as f32;
    let inv_timescales: Vec<_> = (0..channels / 2)
        .map(|i| (i as f32 * (-log_timescale_increment)).exp())
        .collect();
    let arange: Vec<_> = (0..length).map(|c| c as f32).collect();
    let inv_timescales = Tensor::new(inv_timescales.as_slice(), &Device::Cpu)?.unsqueeze(0)?;
    let arange = Tensor::new(arange.as_slice(), &Device::Cpu)?.unsqueeze(1)?;
    let sh = (length, channels / 2);
    let scaled_time = (arange.broadcast_as(sh)? * inv_timescales.broadcast_as(sh)?)?;
    let sincos = Tensor::cat(&[scaled_time.sin()?, scaled_time.cos()?], 1)?;
    Ok(sincos)
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L143
struct AudioEncoder {
    conv1: Conv1D,
    conv2: Conv1D,
    positional_embedding: Tensor,
    blocks: Vec<ResidualAttentionBlock>,
    ln_post: LayerNorm,
}

impl AudioEncoder {
    fn load(p: &str, vb: &VarBuilder, cfg: &Config) -> Result<Self> {
        let n_state = cfg.n_audio_state;
        let n_head = cfg.n_audio_head;
        let n_ctx = cfg.n_audio_ctx;
        let cfg1 = ConvConfig {
            padding: 1,
            stride: 1,
        };
        let cfg2 = ConvConfig {
            padding: 1,
            stride: 2,
        };
        let conv1 = Conv1D::load(cfg.n_mels, n_state, 3, cfg1, &format!("{p}.conv1"), vb)?;
        let conv2 = Conv1D::load(n_state, n_state, 3, cfg2, &format!("{p}.conv2"), vb)?;
        let positional_embedding = sinusoids(n_ctx, n_state)?.to_device(&vb.device)?;
        let blocks = (0..cfg.n_audio_layer)
            .map(|i| {
                ResidualAttentionBlock::load(n_state, n_head, false, &format!("{p}.blocks.{i}"), vb)
            })
            .collect::<Result<Vec<_>>>()?;
        let ln_post = LayerNorm::load(n_state, &format!("{p}.ln_post"), vb)?;
        Ok(Self {
            conv1,
            conv2,
            positional_embedding,
            blocks,
            ln_post,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?.gelu()?;
        let x = self.conv2.forward(&x)?.gelu()?;
        let x = x.transpose(1, 2)?;
        let mut x = x.broadcast_add(&self.positional_embedding)?;
        for block in self.blocks.iter() {
            x = block.forward(&x, None, None)?
        }
        let x = self.ln_post.forward(&x)?;
        Ok(x)
    }
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L176
struct TextDecoder {
    token_embedding: Embedding,
    positional_embedding: Tensor,
    blocks: Vec<ResidualAttentionBlock>,
    ln: LayerNorm,
    mask: Tensor,
}

impl TextDecoder {
    fn load(p: &str, vb: &VarBuilder, cfg: &Config) -> Result<Self> {
        let n_state = cfg.n_text_state;
        let n_head = cfg.n_text_head;
        let n_ctx = cfg.n_text_ctx;
        let token_embedding =
            Embedding::load(cfg.n_vocab, n_state, &format!("{p}.token_embedding"), vb)?;
        let positional_embedding =
            vb.get((n_ctx, n_state), &format!("{p}.positional_embedding"))?;
        let blocks = (0..cfg.n_text_layer)
            .map(|i| {
                ResidualAttentionBlock::load(n_state, n_head, true, &format!("{p}.blocks.{i}"), vb)
            })
            .collect::<Result<Vec<_>>>()?;
        let ln = LayerNorm::load(n_state, &format!("{p}.ln"), vb)?;
        let mask: Vec<_> = (0..n_ctx)
            .flat_map(|i| (0..n_ctx).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 }))
            .collect();
        let mask = Tensor::from_vec(mask, (n_ctx, n_ctx), &vb.device)?;

        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
            mask,
        })
    }
    fn forward(&self, x: &Tensor, xa: &Tensor) -> Result<Tensor> {
        let x_dims = x.dims();
        let last = x_dims[x_dims.len() - 1];
        let token_embedding = self.token_embedding.forward(x)?;
        let positional_embedding = self.positional_embedding.narrow(0, 0, last)?;
        let mut x = token_embedding.broadcast_add(&positional_embedding)?;
        for block in self.blocks.iter() {
            x = block.forward(&x, Some(xa), Some(&self.mask))?;
        }
        let x = self.ln.forward(&x)?;
        let w = self.token_embedding.embeddings.broadcast_left(x_dims[0])?;
        let logits = x.matmul(&w.t()?)?;
        Ok(logits)
    }
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L221
struct Whisper {
    encoder: AudioEncoder,
    decoder: TextDecoder,
}

impl Whisper {
    fn load(vb: &VarBuilder, cfg: &Config) -> Result<Self> {
        let encoder = AudioEncoder::load("encoder", vb, cfg)?;
        let decoder = TextDecoder::load("decoder", vb, cfg)?;
        Ok(Self { encoder, decoder })
    }
    fn forward(&self, mel: &Tensor, tokens: &Tensor) -> Result<Tensor> {
        let enc = self.encoder.forward(mel)?;
        let dec = self.decoder.forward(tokens, &enc)?;
        Ok(dec)
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    weights: String,

    #[arg(long)]
    input: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0)?
    };

    let input = unsafe { candle::safetensors::MmapedFile::new(args.input)? };
    let input = input.deserialize()?;
    let tokens = input.tensor("tokens", &device)?.to_dtype(DType::U32)?;
    let mel = input.tensor("mel", &device)?;

    let weights = unsafe { candle::safetensors::MmapedFile::new(args.weights)? };
    let weights = weights.deserialize()?;
    let vb = VarBuilder::from_safetensors(vec![weights], DTYPE, device.clone());
    let cfg = Config::tiny_en();

    let model = Whisper::load(&vb, &cfg)?;
    let logits = model.forward(&mel, &tokens)?;
    println!("tokens\n{tokens}");
    println!("logits:\n{logits}");
    println!("python logits: {}", input.tensor("dec", &device)?);
    let enc = model.encoder.forward(&mel)?;
    println!("encoder:\n{enc}");
    println!("python enc: {}", input.tensor("enc", &device)?);
    Ok(())
}
