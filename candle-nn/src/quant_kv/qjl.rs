//! QJL: Asymmetric 1-Bit Quantized Johnson-Lindenstrauss Transform for KV Cache.
//!
//! ## Paper Reference
//!
//! "Asymmetric 1-Bit Quantized Johnson-Lindenstrauss Transformations for KV Cache"
//! Zandieh et al. (2024) — arxiv:2406.03482

use candle::{D, DType, Result, Tensor};
use half::f16;
use super::prng::Prng;

/// Configuration for QJL quantization.
#[derive(Debug, Clone, PartialEq)]
pub struct QjlConfig {
    /// Dimension of the key/query vectors (head dimension)
    pub dim: usize,
    /// Seed for the random projection matrix S.
    pub seed: u64,
}

impl QjlConfig {
    pub fn new(dim: usize, seed: u64) -> Self {
        Self { dim, seed }
    }
}

/// A quantized representation of a single key vector.
#[derive(Debug, Clone)]
pub struct QjlQuantizedKey {
    pub sign_bits: Vec<u8>,
    pub norm: f16,
    pub dim: usize,
}

/// GPU-native storage for a batch of QJL-quantized keys.
#[derive(Debug, Clone)]
pub struct QjlTensors {
    pub sign_bits: Vec<Tensor>,
    pub norms: Vec<Tensor>,
    pub cur_len: usize,
}

impl QjlTensors {
    pub fn new(_num_heads: usize, _max_seq_len: usize, _packed_dim: usize, _device: &candle::Device) -> Result<Self> {
        Ok(Self { sign_bits: Vec::new(), norms: Vec::new(), cur_len: 0 })
    }

    pub fn cat_sign_bits(&self) -> Result<Tensor> {
        let cat_dim = self.sign_bits[0].dims().len() - 2;
        Tensor::cat(&self.sign_bits, cat_dim)
    }

    pub fn cat_norms(&self) -> Result<Tensor> {
        let cat_dim = self.norms[0].dims().len() - 2;
        Tensor::cat(&self.norms, cat_dim)
    }
}

/// Quantize a key tensor using QJL.
pub fn qjl_quantize_tensor(
    k: &Tensor,
    config: &QjlConfig,
) -> Result<(Tensor, Tensor)> {
    let device = k.device();
    let dims = k.dims();
    let (num_heads, seq_len, d) = match dims.len() {
        4 => (dims[1], dims[2], dims[3]),
        3 => (dims[0], dims[1], dims[2]),
        _ => candle::bail!("qjl_quantize_tensor: expected 3D or 4D tensor"),
    };

    let mut rng = Prng::new(config.seed);
    let mut s_vec = vec![0.0f32; config.dim * d];
    rng.fill_normal(&mut s_vec);
    let s = Tensor::from_vec(s_vec, (config.dim, d), device)?.to_dtype(k.dtype())?;

    // Project: [H, S, D] @ [Projected_D, D] -> [H, S, Projected_D]
    let original_flat_shape = vec![num_heads * seq_len, d];
    let k_flat = k.contiguous()?.reshape(original_flat_shape)?;
    let projected = k_flat.matmul(&s.t()?)?;
    let projected = projected.reshape((num_heads, seq_len, config.dim))?;
    
    // Sign bits: bit j of byte i = sign(projected[i*8 + j])
    let packed_dim = config.dim / 8;
    let mut sign_bits_vec = Vec::with_capacity(num_heads * seq_len * packed_dim);
    
    // We do this part on CPU for now as bit-packing on GPU in Candle is complex
    let proj_data = projected.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    
    for i in 0..(num_heads * seq_len * packed_dim) {
        let mut byte = 0u8;
        for j in 0..8 {
            if proj_data[i * 8 + j] > 0.0 {
                byte |= 1 << j;
            }
        }
        sign_bits_vec.push(byte);
    }
    
    let sign_bits_tensor = Tensor::from_vec(sign_bits_vec, (num_heads, seq_len, packed_dim), device)?;
    
    // Norms: L2 norm along head dimension
    let norms = k.to_dtype(DType::F32)?.sqr()?.sum(candle::D::Minus1)?.sqrt()?;
    let norms = norms.reshape((num_heads, seq_len, 1))?;

    Ok((sign_bits_tensor, norms))
}

/// Compute attention scores using vectorized QJL estimation on GPU.
pub fn qjl_attention_scores_vectorized(
    q: &Tensor,
    k_tensors: &QjlTensors,
    config: &QjlConfig,
) -> Result<Tensor> {
    let device = q.device();
    let dims = q.dims();
    let rank = q.dims().len();
    let num_heads = if rank == 4 { dims[1] } else { dims[0] };
    let q_len = dims[rank - 2];
    let d_final = dims[rank - 1];
    let kv_len = k_tensors.cur_len;

    if kv_len == 0 {
        let batch = if rank == 4 { dims[0] } else { 1 };
        return Tensor::zeros((batch, num_heads, q_len, 0), q.dtype(), device);
    }
    
    // 1. Project Query: q_proj = S * q
    let mut rng = Prng::new(config.seed);
    let mut s_vec = vec![0.0f32; config.dim * d_final];
    rng.fill_normal(&mut s_vec);
    let s = Tensor::from_vec(s_vec, (config.dim, d_final), device)?.to_dtype(q.dtype())?;
    
    let original_shape = q.shape().clone();
    let q_flat = q.contiguous()?.reshape((num_heads * q_len, d_final))?;
    let q_proj = q_flat.matmul(&s.t()?)?;
    let q_proj = q_proj.reshape((num_heads, q_len, config.dim))?;

    // 2. Decode Sign Bits: List of Tensors -> Single Concat Tensor -> {-1, 1}
    let k_bits = k_tensors.cat_sign_bits()?.to_dtype(DType::F32)?;
    let norms = k_tensors.cat_norms()?;
    
    let mut sigmas = Vec::with_capacity(8);
    for i in 0..8 {
        let shift = (1 << i) as f64;
        let k_bits_scaled = (k_bits.clone() / shift)?;
        let sigma = k_bits_scaled.clone()
            .floor()?
            .affine(0.5, 0.0)?
            .floor()?
            .affine(-2.0, 0.0)?
            .add(&k_bits_scaled.clone().floor()?)?
            .to_dtype(q.dtype())?
            .affine(2.0, -1.0)?; // 0/1 -> -1/1
        sigmas.push(sigma);
    }
    
    let sigma_tensor = Tensor::stack(&sigmas, candle::D::Minus1)? 
        .reshape((num_heads, kv_len, config.dim))?
        .unsqueeze(0)?; // [Batch=1, Heads, Seq_K, Dim_Proj]

    // 3. Inner Product Estimation
    // Using a robust dimension-blind approach with flatten_to(rank - 3)
    let original_q_shape = q.shape().clone();  // 4D [1, num_heads, q_len, head_dim]
    let q_reshape = q_proj.unsqueeze(0)?;
    let k_reshape = sigma_tensor;

    let scores = q_reshape.matmul(&k_reshape.transpose(2, 3)?)?;
    let mut scores_shape = original_q_shape.dims().to_vec();
    scores_shape[rank - 1] = kv_len;
    let scores = scores.reshape(scores_shape)?; // [Batch, Heads, Q_Len, KV_Len]

    // 4. Apply Scaling and Norms
    let norms = norms.to_dtype(q.dtype())?
        .reshape((1, num_heads, 1, kv_len))?;

    // Correct QJL scale from Stein's lemma: sqrt(π/2) / m
    let scale = (std::f32::consts::PI / 2.0_f32).sqrt() / config.dim as f32;
    scores.broadcast_mul(&norms)?.affine(scale as f64, 0.0)
}
