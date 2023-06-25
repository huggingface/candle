// An implementation of LLaMA https://github.com/facebookresearch/llama
//
// This is based on nanoGPT in a similar way to:
// https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
//
// The tokenizer config can be retrieved from:
// https://huggingface.co/hf-internal-testing/llama-tokenizer/raw/main/tokenizer.json
//
// In order to convert the llama weights to a .npz file, run:
// python examples/llama/convert_checkpoint.py ..../LLaMA/7B/consolidated.00.pth
use anyhow::{Error as E, Result};
use clap::Parser;

use candle::{Device, Tensor};

mod var_store;
use var_store::VarBuilder;

const CONTEXT_SIZE: usize = 512;
const START_PROMPT: &str = r"
EDWARD:
I wonder how our princely father 'scaped,
Or whether he be 'scaped away or no
From Clifford's and Northumberland's pursuit:
Had he been ta'en, we should have heard the news;
Had he been slain, we should have heard the news;
Or had he 'scaped, methinks we should have heard
The happy tidings of his good escape.
How fares my brother? why is he so sad?

RICHARD:
I cannot joy, until I be resolved
Where our right valiant father is become.
I saw him in the battle range about;
And watch'd him how he singled Clifford forth.
Methought he bore him in the thickest troop
As doth a lion in a herd of neat;
Or as a bear, encompass'd round with dogs,
Who having pinch'd a few and made them cry,
The rest stand all aloof, and bark at him.
So fared our father with his enemies;
So fled his enemies my warlike father:
Methinks, 'tis prize enough to be his son.
See how the morning opes her golden gates,
And takes her farewell of the glorious sun!
How well resembles it the prime of youth,
Trimm'd like a younker prancing to his love!

EDWARD:
Dazzle mine eyes, or do I see three suns?

RICHARD:
Three glorious suns, each one a perfect sun;
Not separated with the racking clouds,
But sever'd in a pale clear-shining sky.
See, see! they join, embrace, and seem to kiss,
As if they vow'd some league inviolable:
Now are they but one lamp, one light, one sun.
In this the heaven figures some event.

EDWARD:
'Tis wondrous strange, the like yet never heard of.
I think it cites us, brother, to the field,
That we, the sons of brave Plantagenet,
Each one already blazing by our meeds,
Should notwithstanding join our lights together
And over-shine the earth as this the world.
Whate'er it bodes, henceforward will I bear
Upon my target three fair-shining suns.
";

#[allow(dead_code)]
struct Config {
    block_size: usize,
    vocab_size: usize,
    n_layer: usize,
    n_head: usize,
    n_embd: usize,
}

#[allow(dead_code)]
impl Config {
    fn config_7b() -> Self {
        Self {
            block_size: 4096,
            vocab_size: 32000,
            n_layer: 32,
            n_head: 32,
            n_embd: 4096,
        }
    }

    fn config_13b() -> Self {
        Self {
            block_size: 4096,
            vocab_size: 32000,
            n_layer: 40,
            n_head: 40,
            n_embd: 5120,
        }
    }

    fn config_30b() -> Self {
        Self {
            block_size: 4096,
            vocab_size: 32000,
            n_layer: 60,
            n_head: 52,
            n_embd: 6656,
        }
    }

    fn config_65b() -> Self {
        Self {
            block_size: 4096,
            vocab_size: 32000,
            n_layer: 80,
            n_head: 64,
            n_embd: 8192,
        }
    }
}

struct Embedding {
    embeddings: Tensor,
}

impl Embedding {
    fn new(mut vb: VarBuilder, vocab_size: usize, n_embd: usize) -> Result<Self> {
        let embeddings = vb.var("weight", (vocab_size, n_embd))?;
        Ok(Self { embeddings })
    }

    fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        Ok(Tensor::embedding(indexes, &self.embeddings)?)
    }
}

struct Linear {
    ws: Tensor,
    bs: Option<Tensor>,
}

impl Linear {
    #[allow(dead_code)]
    fn new(mut vb: VarBuilder, in_size: usize, out_size: usize) -> Result<Self> {
        let ws = vb.var("weight", (in_size, out_size))?;
        let bs = vb.var("bias", out_size)?;
        Ok(Self { ws, bs: Some(bs) })
    }

    fn new_no_bias(mut vb: VarBuilder, in_size: usize, out_size: usize) -> Result<Self> {
        let ws = vb.var("weight", (in_size, out_size))?;
        Ok(Self { ws, bs: None })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.matmul(&self.ws)?;
        let y = match &self.bs {
            None => x,
            Some(bs) => x.broadcast_add(bs)?,
        };
        Ok(y)
    }
}

struct RmsNorm {
    scale: Tensor,
    size: usize,
}

impl RmsNorm {
    fn new(mut vb: VarBuilder, size: usize) -> Result<Self> {
        let scale = vb.var("scale", &[size])?;
        Ok(Self { scale, size })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let last_dim = x.dims().last().unwrap();
        let norm_x = ((x * x)?.sum(&[x.rank() - 1])? / *last_dim as f64)?;
        let x_normed = (x / (norm_x + 1e-5)?.sqrt()?)?;
        let scale = self.scale.reshape(&[1, 1, self.size])?;
        Ok((scale * x_normed)?)
    }
}

struct Mlp {
    c_fc1: Linear,
    c_fc2: Linear,
    c_proj: Linear,
}

fn silu(xs: &Tensor) -> Result<Tensor> {
    Ok((xs / (xs.neg()?.exp()? + 1.0)?)?)
}

impl Mlp {
    fn new(vb: VarBuilder, n_embd: usize) -> Result<Self> {
        let n_hidden = 8 * n_embd / 3;
        let n_hidden = (n_hidden - 1) / 256 * 256 + 256;
        let c_fc1 = Linear::new_no_bias(&vb / "c_fc1", n_embd, n_hidden)?;
        let c_fc2 = Linear::new_no_bias(&vb / "c_fc2", n_embd, n_hidden)?;
        let c_proj = Linear::new_no_bias(&vb / "c_proj", n_hidden, n_embd)?;
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        self.c_proj.forward(&x)
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let _on_true = Tensor::new(on_true, &on_false.device())?.broadcast_as(shape.dims())?;
    // TODO: add an equivalent to where (or xla's select) so that we can use the following:
    // let m = mask.where_cond(&on_true, on_false)?;
    let m = on_false.clone();
    Ok(m)
}

struct CausalSelfAttention {
    c_attn: Linear,
    c_proj: Linear,
    n_head: usize,
    n_embd: usize,
}

impl CausalSelfAttention {
    fn new(vb: VarBuilder, n_head: usize, n_embd: usize) -> Result<Self> {
        let c_attn = Linear::new_no_bias(&vb / "c_attn", n_embd, 3 * n_embd)?;
        let c_proj = Linear::new_no_bias(&vb / "c_proj", n_embd, n_embd)?;
        Ok(Self {
            c_attn,
            c_proj,
            n_head,
            n_embd,
        })
    }

    fn apply_rotary_emb(&self, x: &Tensor, freqs_cis: &Tensor) -> Result<Tensor> {
        let mut dims = x.dims().to_vec();
        let v = dims.pop().unwrap();
        dims.push(v / 2);
        dims.push(2);
        let x = x.reshape(dims)?;
        let rank = x.rank();
        let re_x = x.narrow(rank - 1, 0, 1)?;
        let im_x = x.narrow(rank - 1, 1, 2)?;
        let re_f = freqs_cis.narrow(rank - 1, 0, 1)?;
        let im_f = freqs_cis.narrow(rank - 1, 1, 2)?;
        let re = ((&re_x * &re_f)? - (&im_x * &im_f)?)?;
        let im = ((&re_x * &im_f)? + (&im_x * &re_f)?)?;
        let rope = Tensor::cat(&[&re, &im], rank - 1)?;
        // TODO: Add the flatten op.
        let mut dims = rope.dims().to_vec();
        let v1 = dims.pop().unwrap();
        let v2 = dims.pop().unwrap();
        dims.push(v1 * v2);
        let rope = rope.reshape(dims)?;
        Ok(rope)
    }

    fn forward(&self, x: &Tensor, freqs_cis: &Tensor) -> Result<Tensor> {
        let (b, t, c) = x.shape().r3()?;
        let qkv = self.c_attn.forward(x)?;
        let n_embd = self.n_embd;
        let q = qkv.narrow(2, 0, n_embd)?;
        let k = qkv.narrow(2, n_embd, 2 * n_embd)?;
        let v = qkv.narrow(2, 2 * n_embd, 3 * n_embd)?;
        let target_dim = [b, t, self.n_head, c / self.n_head];
        let k = k.reshape(target_dim.as_slice())?.transpose(1, 2)?;
        let q = q.reshape(target_dim.as_slice())?.transpose(1, 2)?;
        let v = v.reshape(target_dim.as_slice())?.transpose(1, 2)?;
        let q = self.apply_rotary_emb(&q, freqs_cis)?;
        let k = self.apply_rotary_emb(&k, freqs_cis)?;
        let k_shape = k.shape();
        let att = (q.matmul(&k.t()?)? / (*k_shape.dims().last().unwrap() as f64).sqrt())?;
        let device = x.device();
        // TODO: If we support bool or u8 tensors, this would be better.
        let mask = Tensor::new(1u32, &device)?
            .broadcast_as(&[t, t])?
            // TODO: .lower_triangle()?
            .reshape(&[1, 1, t, t])?;
        let att = masked_fill(&att, &mask, f32::NEG_INFINITY)?;
        let y = att.softmax(att.rank() - 1)?.matmul(&v)?;
        let y = y.transpose(1, 2)?.reshape(&[b, t, c])?;
        let y = self.c_proj.forward(&y)?;
        Ok(y)
    }
}

struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn new(vb: VarBuilder, config: &Config) -> Result<Self> {
        let rms_1 = RmsNorm::new(&vb / "rms_1", config.n_embd)?;
        let attn = CausalSelfAttention::new(&vb / "attn", config.n_head, config.n_embd)?;
        let rms_2 = RmsNorm::new(&vb / "rms_2", config.n_embd)?;
        let mlp = Mlp::new(&vb / "mlp", config.n_embd)?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
        })
    }

    fn forward(&self, x: &Tensor, freqs_cis: &Tensor) -> Result<Tensor> {
        let x = (self.attn.forward(&self.rms_1.forward(x)?, freqs_cis)? + x)?;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + x)?;
        Ok(x)
    }
}

struct Llama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,
}

impl Llama {
    fn new(vb: VarBuilder, config: &Config) -> Result<Self> {
        let lm_head = Linear::new_no_bias(&vb / "lm_head", config.n_embd, config.vocab_size)?;
        let wte = Embedding::new(
            &vb / "transformer" / "wte",
            config.vocab_size,
            config.n_embd,
        )?;
        let blocks = (0..config.n_layer)
            .map(|i| Block::new(&vb / "transformer" / "h" / i, config))
            .collect::<Result<Vec<_>>>()?;
        let ln_f = RmsNorm::new(&vb / "transformer" / "ln_f", config.n_embd)?;
        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
        })
    }

    fn forward(&self, x: &Tensor, freqs_cis: &Tensor) -> Result<Tensor> {
        let (_, t) = x.shape().r2()?;
        let mut x = self.wte.forward(x)?;
        for block in self.blocks.iter() {
            x = block.forward(&x, freqs_cis)?;
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.narrow(1, t - 1, t)?;
        let logits = self.lm_head.forward(&x)?;
        Ok(logits)
    }
}

fn precompute_freqs_cis(config: &Config) -> Result<Tensor> {
    let seq_len = CONTEXT_SIZE;
    let n_elem = config.n_embd / config.n_head;
    let theta: Vec<_> = (0..n_elem)
        .step_by(2)
        .map(|i| 1f32 / 10000f32.powf(i as f32 / n_elem as f32))
        .collect();
    let arange: Vec<_> = (0..seq_len).map(|c| c as f32).collect();
    let theta = Tensor::new(theta.as_slice(), &candle::Device::Cpu)?;
    let arange = Tensor::new(arange.as_slice(), &candle::Device::Cpu)?;
    let idx_theta = arange
        .reshape((arange.elem_count(), 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let shape = [1, 1, seq_len, n_elem / 2, 1];
    let idx_theta_cos = idx_theta.cos()?.reshape(&shape)?;
    let idx_theta_sin = idx_theta.sin()?.reshape(&shape)?;
    let last_dim = idx_theta_cos.rank() - 1;
    Ok(Tensor::cat(&[&idx_theta_cos, &idx_theta_sin], last_dim)?)
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 1.0)]
    temperature: f64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, default_value_t = 100)]
    sample_len: usize,
}

fn main() -> Result<()> {
    use rand::prelude::*;
    use tokenizers::Tokenizer;

    let args = Args::parse();
    println!("loading tokenizer config");
    let tokenizer = Tokenizer::from_file("llama-tokenizer.json").map_err(E::msg)?;
    let mut tokens = tokenizer
        .encode(START_PROMPT, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    println!("loading weights");
    let start_load = std::time::Instant::now();
    let vb = VarBuilder::new::<f32>(); // TODO: load the weights from llama.npz
    println!("loaded weights in {:?}", start_load.elapsed());

    println!("building the model");
    let config = Config::config_7b();
    let llama = Llama::new(vb, &config)?;

    println!("pre-computing the positional embeddings");
    let freqs_cis = precompute_freqs_cis(&config)?;
    println!("starting the inference loop");
    let mut new_tokens = vec![];
    let mut rng = thread_rng();
    for index in 0..args.sample_len {
        let ctxt = &tokens[tokens.len().saturating_sub(CONTEXT_SIZE)..];
        let input = Tensor::new(ctxt, &Device::Cpu)?.reshape((1, ctxt.len()))?;
        let logits = llama.forward(&input, &freqs_cis)?;
        let prs = (&logits / args.temperature)?.softmax(logits.rank() - 1)?;
        let logits_v: Vec<f32> = prs.to_vec1()?;
        let distr = rand::distributions::WeightedIndex::new(&logits_v)?;
        let next_token = distr.sample(&mut rng) as u32;
        tokens.push(next_token);
        new_tokens.push(next_token);
        println!(
            "{} token: {} '{}'",
            index + 1,
            next_token,
            tokenizer.decode(vec![next_token], true).map_err(E::msg)?
        );
    }
    println!(
        "----\n{}\n----",
        tokenizer.decode(new_tokens, true).map_err(E::msg)?
    );
    Ok(())
}
