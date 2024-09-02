use candle::backend::BackendStorage;
use candle::{CpuStorage, CustomOp1, DType, Device, IndexOp, Layout, Result, Shape, Tensor, D};
use candle_nn::var_builder::ShardedVarBuilder as VarBuilder;
use candle_nn::{Embedding, Linear, Module, RmsNorm};
use cudarc::nccl::safe::{Comm, ReduceOp};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use super::MAX_SEQ_LEN;

pub type Config = candle_transformers::models::llama::LlamaConfig;

struct TensorParallelColumnLinear {
    linear: Linear,
}

impl TensorParallelColumnLinear {
    fn new(linear: Linear) -> Self {
        Self { linear }
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear.forward(x)
    }
}

struct TensorParallelRowLinear {
    linear: Linear,
    all_reduce: AllReduce,
}

struct AllReduce {
    comm: Rc<Comm>,
}

/// This is actually not safe: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/threadsafety.html
/// But for this example purposes, this will work
unsafe impl Sync for AllReduce {}
unsafe impl Send for AllReduce {}

impl CustomOp1 for AllReduce {
    fn name(&self) -> &'static str {
        "allreduce"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        candle::bail!("AllReduce is never used on cpu")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s: &candle::CudaStorage,
        l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::cuda_backend::WrapErr;
        use cudarc::driver::DeviceSlice;
        use half::{bf16, f16};

        let elem_count = l.shape().elem_count();
        let dev = s.device().clone();
        let dst = match s.dtype() {
            DType::BF16 => {
                let s = s.as_cuda_slice::<bf16>()?;
                let s = match l.contiguous_offsets() {
                    Some((0, l)) if l == s.len() => s,
                    Some(_) | None => candle::bail!("input has to be contiguous"),
                };
                let mut dst = unsafe { dev.alloc::<bf16>(elem_count) }.w()?;
                self.comm
                    .all_reduce(s, &mut dst, &ReduceOp::Sum)
                    .map_err(candle::Error::debug)?;
                candle::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            DType::F16 => {
                let s = s.as_cuda_slice::<f16>()?;
                let s = match l.contiguous_offsets() {
                    Some((0, l)) if l == s.len() => s,
                    Some(_) | None => candle::bail!("input has to be contiguous"),
                };
                let mut dst = unsafe { dev.alloc::<f16>(elem_count) }.w()?;
                self.comm
                    .all_reduce(s, &mut dst, &ReduceOp::Sum)
                    .map_err(candle::Error::debug)?;
                candle::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            dtype => candle::bail!("unsupported dtype {dtype:?}"),
        };
        Ok((dst, l.shape().clone()))
    }
}

impl TensorParallelRowLinear {
    fn new(linear: Linear, comm: Rc<Comm>) -> Self {
        let all_reduce = AllReduce { comm };
        Self { linear, all_reduce }
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear.forward(x)?.apply_op1_no_bwd(&self.all_reduce)
    }
}

fn shard(dim: usize, rank: usize, world_size: usize) -> candle_nn::var_builder::Shard {
    candle_nn::var_builder::Shard {
        dim,
        rank,
        world_size,
    }
}

impl TensorParallelColumnLinear {
    fn load(vb: VarBuilder, comm: Rc<Comm>) -> Result<Self> {
        let rank = comm.rank();
        let size = comm.world_size();
        let weight = vb.get_with_hints((), "weight", shard(0, rank, size))?;
        Ok(Self::new(Linear::new(weight, None)))
    }

    fn load_multi(vb: VarBuilder, prefixes: &[&str], comm: Rc<Comm>) -> Result<Self> {
        let rank = comm.rank();
        let size = comm.world_size();
        let weights: Vec<_> = prefixes
            .iter()
            .map(|p| vb.pp(p).get_with_hints((), "weight", shard(0, rank, size)))
            .collect::<Result<Vec<_>>>()?;
        let weight = Tensor::cat(&weights, 0)?;
        Ok(Self::new(Linear::new(weight, None)))
    }
}

impl TensorParallelRowLinear {
    fn load(vb: VarBuilder, comm: Rc<Comm>) -> Result<Self> {
        let rank = comm.rank();
        let size = comm.world_size();
        let weight = vb.get_with_hints((), "weight", shard(1, rank, size))?;
        Ok(Self::new(Linear::new(weight, None), comm))
    }
}

#[derive(Clone)]
pub struct Cache {
    #[allow(clippy::type_complexity)]
    kvs: Arc<Mutex<Vec<Option<(Tensor, Tensor)>>>>,
    cos: Tensor,
    sin: Tensor,
}

impl Cache {
    pub fn new(dtype: DType, config: &Config, device: &Device) -> Result<Self> {
        // precompute freqs_cis
        let n_elem = config.hidden_size / config.num_attention_heads;
        let theta: Vec<_> = (0..n_elem)
            .step_by(2)
            .map(|i| 1f32 / config.rope_theta.powf(i as f32 / n_elem as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_SEQ_LEN, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        // This is different from the paper, see:
        // https://github.com/huggingface/transformers/blob/6112b1c6442aaf7affd2b0676a1cd4eee30c45cf/src/transformers/models/llama/modeling_llama.py#L112
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;
        Ok(Self {
            kvs: Arc::new(Mutex::new(vec![None; config.num_hidden_layers])),
            cos,
            sin,
        })
    }
}

fn silu(xs: &Tensor) -> Result<Tensor> {
    xs / (xs.neg()?.exp()? + 1.0)?
}

fn linear(size1: usize, size2: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((size2, size1), "weight")?;
    Ok(Linear::new(weight, None))
}

fn embedding(cfg: &Config, vb: VarBuilder) -> Result<Embedding> {
    let embeddings = vb.get((cfg.vocab_size, cfg.hidden_size), "weight")?;
    Ok(Embedding::new(embeddings, cfg.hidden_size))
}

struct CausalSelfAttention {
    qkv_proj: TensorParallelColumnLinear,
    o_proj: TensorParallelRowLinear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    cache: Cache,
}

impl CausalSelfAttention {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, _, seq_len, _hidden_size) = x.shape().dims4()?;
        let cos = self.cache.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.cache.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }

    fn forward(&self, x: &Tensor, index_pos: usize, block_idx: usize) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.shape().dims3()?;

        let qkv = self.qkv_proj.forward(x)?;
        let hidden_size = self.num_attention_heads * self.head_dim;

        let q = qkv.i((.., .., ..self.num_attention_heads * self.head_dim))?;
        let k = qkv.i((
            ..,
            ..,
            self.num_attention_heads * self.head_dim
                ..self.num_attention_heads * self.head_dim
                    + self.num_key_value_heads * self.head_dim,
        ))?;
        let v = qkv.i((
            ..,
            ..,
            self.num_attention_heads * self.head_dim + self.num_key_value_heads * self.head_dim..,
        ))?;
        // todo!("Q {:?} K {:?} V {:?} - x {:?}", q.shape(), k.shape(), v.shape(), x.shape());

        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let mut k = self.apply_rotary_emb(&k, index_pos)?;

        let mut cache = self.cache.kvs.lock().unwrap();
        if let Some((cache_k, cache_v)) = &cache[block_idx] {
            k = Tensor::cat(&[cache_k, &k], 2)?.contiguous()?;
            v = Tensor::cat(&[cache_v, &v], 2)?.contiguous()?;
            let k_seq_len = k.dims()[1];
            if k_seq_len > MAX_SEQ_LEN {
                k = k
                    .narrow(D::Minus1, k_seq_len - MAX_SEQ_LEN, MAX_SEQ_LEN)?
                    .contiguous()?
            }
            let v_seq_len = v.dims()[1];
            if v_seq_len > 2 * MAX_SEQ_LEN {
                v = v
                    .narrow(D::Minus1, v_seq_len - MAX_SEQ_LEN, MAX_SEQ_LEN)?
                    .contiguous()?
            }
        }
        cache[block_idx] = Some((k.clone(), v.clone()));

        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
        let y = candle_flash_attn::flash_attn(&q, &k, &v, softmax_scale, seq_len > 1)?
            .reshape((b_sz, seq_len, hidden_size))?;
        let y = self.o_proj.forward(&y)?;
        Ok(y)
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        let n_rep = self.num_attention_heads / self.num_key_value_heads;
        candle_transformers::utils::repeat_kv(x, n_rep)
    }

    fn load(vb: VarBuilder, cache: &Cache, cfg: &Config, comm: Rc<Comm>) -> Result<Self> {
        let qkv_proj = TensorParallelColumnLinear::load_multi(
            vb.clone(),
            &["q_proj", "k_proj", "v_proj"],
            comm.clone(),
        )?;
        let o_proj = TensorParallelRowLinear::load(vb.pp("o_proj"), comm.clone())?;
        Ok(Self {
            qkv_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads / comm.world_size(),
            num_key_value_heads: cfg.num_key_value_heads() / comm.world_size(),
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            cache: cache.clone(),
        })
    }
}

struct Mlp {
    c_fc1: TensorParallelColumnLinear,
    c_fc2: TensorParallelColumnLinear,
    c_proj: TensorParallelRowLinear,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        self.c_proj.forward(&x)
    }

    fn load(vb: VarBuilder, _cfg: &Config, comm: Rc<Comm>) -> Result<Self> {
        let c_fc1 = TensorParallelColumnLinear::load(vb.pp("gate_proj"), comm.clone())?;
        let c_fc2 = TensorParallelColumnLinear::load(vb.pp("up_proj"), comm.clone())?;
        let c_proj = TensorParallelRowLinear::load(vb.pp("down_proj"), comm)?;
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
        })
    }
}

struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
}

fn rms_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm> {
    let weight = vb.get_with_hints(size, "weight", shard(0, 0, 1))?;
    Ok(RmsNorm::new(weight, eps))
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

    fn forward(&self, x: &Tensor, index_pos: usize, block_idx: usize) -> Result<Tensor> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self.attn.forward(&x, index_pos, block_idx)? + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }

    fn load(vb: VarBuilder, cache: &Cache, cfg: &Config, comm: Rc<Comm>) -> Result<Self> {
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cache, cfg, comm.clone())?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg, comm)?;
        let input_layernorm = rms_norm(cfg.hidden_size, 1e-5, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            rms_norm(cfg.hidden_size, 1e-5, vb.pp("post_attention_layernorm"))?;
        Ok(Self::new(
            input_layernorm,
            attn,
            post_attention_layernorm,
            mlp,
        ))
    }
}

pub struct Llama {
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

    pub fn forward(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.shape().dims2()?;
        let mut x = self.wte.forward(x)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx)?;
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.i((.., seq_len - 1, ..))?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)
    }

    pub fn load(vb: VarBuilder, cache: &Cache, cfg: &Config, comm: Rc<Comm>) -> Result<Self> {
        let wte = embedding(cfg, vb.pp("model.embed_tokens"))?;
        let lm_head = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        let norm = rms_norm(cfg.hidden_size, 1e-5, vb.pp("model.norm"))?;
        let blocks: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|i| {
                Block::load(
                    vb.pp(&format!("model.layers.{i}")),
                    cache,
                    cfg,
                    comm.clone(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::new(wte, blocks, norm, lm_head))
    }
}
