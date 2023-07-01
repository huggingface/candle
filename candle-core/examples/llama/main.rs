extern crate blas;
extern crate openblas_src;
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

// TODO: This does not use a batch dimension. If adding it back, be cautious about the
// transposition operations.
use anyhow::{Error as E, Result};
use clap::Parser;
use rand::{distributions::Distribution, SeedableRng};

use candle::{DType, Device, Tensor};
use candle_hub::{api::Api, Repo, RepoType};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

mod var_store;
mod weights;

const MAX_SEQ_LEN: usize = 4096;
const DTYPE: DType = DType::F32;
const DEFAULT_PROMPT: &str = r"
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
    fn new(embeddings: Tensor) -> Self {
        Self { embeddings }
    }

    fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        let embeddings = self.embeddings.to_dtype(DTYPE)?;
        Ok(Tensor::embedding(indexes, &embeddings)?)
    }
}

struct Linear {
    weight: Tensor,
}

impl Linear {
    fn new(weight: Tensor) -> Self {
        Self { weight }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let weight = self.weight.to_dtype(DTYPE)?;
        let x = x.matmul(&weight.t()?)?;
        Ok(x)
    }
}

struct RmsNorm {
    scale: Tensor,
}

impl RmsNorm {
    fn new(scale: Tensor) -> Self {
        Self { scale }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // This is a no-op if x's dtype is already f32.
        let x = x.to_dtype(DType::F32)?;
        let (seq_len, hidden_size) = x.shape().r2()?;
        let norm_x = ((&x * &x)?.sum(&[1])? / hidden_size as f64)?;
        let norm_x = norm_x.broadcast_as((seq_len, hidden_size))?;
        let x_normed = (x / (norm_x + 1e-5)?.sqrt()?)?;
        let size = self.scale.shape().r1()?;
        let scale = self
            .scale
            .to_dtype(DType::F32)?
            .broadcast_as((seq_len, size))?;
        let x = (scale * x_normed)?;
        let x = x.to_dtype(DTYPE)?;
        Ok(x)
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
    fn new(c_fc1: Linear, c_fc2: Linear, c_proj: Linear) -> Self {
        Self {
            c_fc1,
            c_fc2,
            c_proj,
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        self.c_proj.forward(&x)
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, &on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

#[derive(Clone)]
struct Cache {
    masks: Arc<Mutex<HashMap<usize, Tensor>>>,
    use_kv_cache: bool,
    #[allow(clippy::type_complexity)]
    kvs: Arc<Mutex<Vec<Option<(Tensor, Tensor)>>>>,
    device: Device,
}

impl Cache {
    fn new(use_kv_cache: bool, config: &Config, device: &Device) -> Self {
        Self {
            masks: Arc::new(Mutex::new(HashMap::new())),
            use_kv_cache,
            kvs: Arc::new(Mutex::new(vec![None; config.n_layer])),
            device: device.clone(),
        }
    }

    fn mask(&self, t: usize) -> Result<Tensor> {
        let mut masks = self.masks.lock().unwrap();
        if let Some(mask) = masks.get(&t) {
            Ok(mask.clone())
        } else {
            // TODO: If we support bool or u8 tensors, this would be better.
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u32::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (t, t), &self.device)?;
            masks.insert(t, mask.clone());
            Ok(mask)
        }
    }
}

struct CausalSelfAttention {
    c_attn: Linear,
    c_proj: Linear,
    n_head: usize,
    cache: Cache,
}

impl CausalSelfAttention {
    fn new(c_attn: Linear, c_proj: Linear, n_head: usize, cache: &Cache) -> Self {
        Self {
            c_attn,
            c_proj,
            n_head,
            cache: cache.clone(),
        }
    }

    fn apply_rotary_emb(&self, x: &Tensor, freqs_cis: &Tensor) -> Result<Tensor> {
        let mut dims = x.dims().to_vec();
        let fcis_dims = freqs_cis.dims();
        let freqs_cis = if dims[1] < fcis_dims[1] {
            freqs_cis.narrow(1, 0, dims[1])?
        } else {
            freqs_cis.clone()
        };
        let v = dims.pop().unwrap();
        dims.push(v / 2);
        dims.push(2);
        let x = x.reshape(dims)?;
        let rank = x.rank();
        let re_x = x.narrow(rank - 1, 0, 1)?;
        let im_x = x.narrow(rank - 1, 1, 1)?;
        let re_f = freqs_cis
            .narrow(rank - 1, 0, 1)?
            .broadcast_as(re_x.shape())?;
        let im_f = freqs_cis
            .narrow(rank - 1, 1, 1)?
            .broadcast_as(im_x.shape())?;
        let re = ((&re_x * &re_f)? - (&im_x * &im_f)?)?;
        let im = ((&re_x * &im_f)? + (&im_x * &re_f)?)?;
        let rope = Tensor::cat(&[&re, &im], rank - 1)?;
        let rope = rope.flatten(Some(rope.rank() - 2), None)?;
        Ok(rope)
    }

    fn forward(&self, x: &Tensor, freqs_cis: &Tensor, block_idx: usize) -> Result<Tensor> {
        let (t, c) = x.shape().r2()?;
        let qkv = self.c_attn.forward(x)?;
        let qkv = qkv.to_dtype(DType::F32)?;
        let n_embd = c;
        let q = qkv.narrow(1, 0, n_embd)?;
        let k = qkv.narrow(1, n_embd, n_embd)?;
        let v = qkv.narrow(1, 2 * n_embd, n_embd)?;
        let target_dim = [t, self.n_head, c / self.n_head];
        let k = k.reshape(target_dim.as_slice())?.transpose(0, 1)?;
        let q = q.reshape(target_dim.as_slice())?.transpose(0, 1)?;
        let mut v = v.reshape(target_dim.as_slice())?.transpose(0, 1)?;
        let q = self.apply_rotary_emb(&q, freqs_cis)?;
        let mut k = self.apply_rotary_emb(&k, freqs_cis)?;

        if self.cache.use_kv_cache {
            let mut cache = self.cache.kvs.lock().unwrap();
            if let Some((cache_k, cache_v)) = &cache[block_idx] {
                k = Tensor::cat(&[cache_k, &k], 1)?.contiguous()?;
                v = Tensor::cat(&[cache_v, &v], 1)?.contiguous()?;
                let k_seq_len = k.dims()[1];
                if k_seq_len > MAX_SEQ_LEN {
                    k = k
                        .narrow(1, k_seq_len - MAX_SEQ_LEN, MAX_SEQ_LEN)?
                        .contiguous()?
                }
                let v_seq_len = v.dims()[1];
                if v_seq_len > 2 * MAX_SEQ_LEN {
                    v = v
                        .narrow(1, v_seq_len - MAX_SEQ_LEN, MAX_SEQ_LEN)?
                        .contiguous()?
                }
            }
            cache[block_idx] = Some((k.clone(), v.clone()))
        }

        let k_shape = k.shape();
        let att = (q.matmul(&k.t()?)? / (*k_shape.dims().last().unwrap() as f64).sqrt())?;
        let mask = self.cache.mask(t)?.broadcast_as(att.shape())?;
        let att = masked_fill(&att, &mask, f32::NEG_INFINITY)?;
        let att = att.softmax(att.rank() - 1)?;
        // Convert to contiguous as matmul doesn't support strided vs for now.
        let y = att.matmul(&v.contiguous()?)?;
        let y = y.transpose(0, 1)?.reshape(&[t, c])?;
        let y = y.to_dtype(DTYPE)?;
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
    fn new(rms_1: RmsNorm, attn: CausalSelfAttention, rms_2: RmsNorm, mlp: Mlp) -> Self {
        Self {
            rms_1,
            attn,
            rms_2,
            mlp,
        }
    }

    fn forward(&self, x: &Tensor, freqs_cis: &Tensor, block_idx: usize) -> Result<Tensor> {
        let x = (self
            .attn
            .forward(&self.rms_1.forward(x)?, freqs_cis, block_idx)?
            + x)?;
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
    fn new(wte: Embedding, blocks: Vec<Block>, ln_f: RmsNorm, lm_head: Linear) -> Self {
        Self {
            wte,
            blocks,
            ln_f,
            lm_head,
        }
    }

    fn forward(&self, x: &Tensor, freqs_cis: &Tensor) -> Result<Tensor> {
        // TODO: Support for mini-batches? (i.e. r2)
        let t = x.shape().r1()?;
        let mut x = self.wte.forward(x)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, freqs_cis, block_idx)?;
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.narrow(0, t - 1, 1)?;
        let logits = self.lm_head.forward(&x)?;
        let logits = logits.to_dtype(DType::F32)?;
        let (b, vocab_size) = logits.shape().r2()?;
        assert_eq!(b, 1);
        Ok(logits.reshape(vocab_size)?)
    }
}

fn precompute_freqs_cis(config: &Config, device: &Device) -> Result<Tensor> {
    let n_elem = config.n_embd / config.n_head;
    let theta: Vec<_> = (0..n_elem)
        .step_by(2)
        .map(|i| 1f32 / 10000f32.powf(i as f32 / n_elem as f32))
        .collect();
    let arange: Vec<_> = (0..MAX_SEQ_LEN).map(|c| c as f32).collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let arange = Tensor::new(arange.as_slice(), device)?;
    let idx_theta = arange
        .reshape((arange.elem_count(), 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let shape = [1, MAX_SEQ_LEN, n_elem / 2, 1];
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

    /// Use npy instead of safetensors
    #[arg(long)]
    npy: Option<String>,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, default_value_t = 100)]
    sample_len: usize,

    /// Disable the key-value cache.
    #[arg(long)]
    no_kv_cache: bool,

    /// The initial prompt.
    #[arg(long)]
    prompt: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    use tokenizers::Tokenizer;

    let args = Args::parse();
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0)?
    };
    let config = Config::config_7b();
    let cache = Cache::new(!args.no_kv_cache, &config, &device);
    let start = std::time::Instant::now();
    let (llama, tokenizer_filename) = match args.npy {
        Some(npy) => {
            println!("building the model (NPY)");
            let weights = Llama::load_npy(&device, &npy, &cache, &config)?;
            let token_path = std::path::Path::new("llama-tokenizer.json").to_path_buf();
            (weights, token_path)
        }
        None => {
            let api = Api::new()?;
            let repo = Repo::new("Narsil/amall-7b".to_string(), RepoType::Model);
            println!("building the model");
            let tokenizer_filename = api.get(&repo, "tokenizer.json").await?;
            let mut filenames = vec![];
            for rfilename in [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ] {
                let filename = api.get(&repo, rfilename).await?;
                filenames.push(filename);
            }

            println!("building the model (SF)");
            (
                Llama::load(&device, &filenames, &cache, &config)?,
                tokenizer_filename,
            )
        }
    };
    println!("Loaded in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let prompt = args.prompt.as_ref().map_or(DEFAULT_PROMPT, |p| p.as_str());
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    println!("pre-computing the positional embeddings");
    let freqs_cis = precompute_freqs_cis(&config, &device)?;
    println!("starting the inference loop");
    let mut new_tokens = vec![];
    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);
    let start_gen = std::time::Instant::now();
    let mut index_pos = 0;
    for index in 0..args.sample_len {
        let start_gen = std::time::Instant::now();
        let context_size = if cache.use_kv_cache && index > 0 {
            1
        } else {
            tokens.len()
        };
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &device)?;
        let freqs_cis = if cache.use_kv_cache {
            freqs_cis.narrow(1, index_pos, ctxt.len())?
        } else {
            freqs_cis.clone()
        };
        let logits = llama.forward(&input, &freqs_cis)?;
        index_pos += ctxt.len();

        let next_token = if let Some(temperature) = args.temperature {
            println!("Sampling with temperature {temperature:?}");
            let prs = (&logits / temperature)?.softmax(logits.rank() - 1)?;
            let logits_v: Vec<f32> = prs.to_vec1()?;
            let distr = rand::distributions::WeightedIndex::new(&logits_v)?;

            distr.sample(&mut rng) as u32
        } else {
            let logits_v: Vec<f32> = logits.to_vec1()?;
            logits_v
                .iter()
                .enumerate()
                .max_by(|(_, u), (_, v)| u.total_cmp(v))
                .map(|(i, _)| i as u32)
                .unwrap()
        };
        tokens.push(next_token);
        new_tokens.push(next_token);
        println!("> {:?}", start_gen.elapsed());
        println!(
            "{} token: {} '{}'",
            index + 1,
            next_token,
            tokenizer.decode(vec![next_token], true).map_err(E::msg)?
        );
    }
    let dt = start_gen.elapsed();
    println!(
        "{} tokens generated ({} token/s)\n----\n{}\n----",
        args.sample_len,
        args.sample_len as f64 / dt.as_secs_f64(),
        tokenizer.decode(new_tokens, true).map_err(E::msg)?
    );
    Ok(())
}
