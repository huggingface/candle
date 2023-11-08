use byteorder::{LittleEndian, ReadBytesExt};
use candle::{DType, Device, IndexOp, Result, Shape, Tensor};
use candle_nn::VarBuilder;

use super::llama2_c::Config;

pub struct TransformerWeights {
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
    pub fn from_reader<R: std::io::Read>(r: &mut R) -> Result<Self> {
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

    pub fn head_size(&self) -> usize {
        self.dim / self.n_heads
    }
}

impl TransformerWeights {
    pub fn from_reader<R: std::io::Read>(r: &mut R, c: &Config, dev: &Device) -> Result<Self> {
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

    pub fn var_builder(&self, cfg: &Config, device: &Device) -> Result<VarBuilder<'static>> {
        // TODO: As of 2023-08-04, gemm is slower than expected when multiplying a matrix of
        // size (1, k) with the transpose of a matrix of size (k, n) as it ends up transposing the
        // second matrix back. We detect this case here and as a temporary hack make the weight
        // matrix column major rather than row major. This ends up speeding up text generation from
        // 120 token/s to 220 token/s on a Ryzen 2600X.
        let tr = device.is_cpu() && !candle::utils::has_mkl();
        let tr = |x: Tensor| if tr { x.t()?.contiguous()?.t() } else { Ok(x) };
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
        insert("lm_head.weight", tr(self.token_embedding_table.clone())?);
        insert("model.norm.weight", self.rms_final_weight.clone());
        for layer in 0..cfg.n_layers {
            ws.insert(
                format!("model.layers.{layer}.self_attn.q_proj.weight"),
                tr(self.wq.i(layer)?)?,
            );
            ws.insert(
                format!("model.layers.{layer}.self_attn.k_proj.weight"),
                tr(self.wk.i(layer)?)?,
            );
            ws.insert(
                format!("model.layers.{layer}.self_attn.v_proj.weight"),
                tr(self.wv.i(layer)?)?,
            );
            ws.insert(
                format!("model.layers.{layer}.self_attn.o_proj.weight"),
                tr(self.wo.i(layer)?)?,
            );
            ws.insert(
                format!("model.layers.{layer}.mlp.gate_proj.weight"),
                tr(self.w1.i(layer)?)?,
            );
            ws.insert(
                format!("model.layers.{layer}.mlp.down_proj.weight"),
                tr(self.w2.i(layer)?)?,
            );
            ws.insert(
                format!("model.layers.{layer}.mlp.up_proj.weight"),
                tr(self.w3.i(layer)?)?,
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
