use super::model::{attention, timestep_embedding, Config, EmbedNd};
use crate::quantized_nn::{linear, linear_b, Linear};
use crate::quantized_var_builder::VarBuilder;
use candle::quantized::QuantizedBackend;
use candle::{BackendStorage, DType, IndexOp, Result, Tensor, D};
use candle_nn::{LayerNorm, RmsNorm};

fn layer_norm<QB: QuantizedBackend>(
    dim: usize,
    vb: VarBuilder<QB>,
) -> Result<LayerNorm<QB::Storage>> {
    let ws = Tensor::ones(dim, DType::F32, vb.device())?;
    Ok(LayerNorm::new_no_bias(ws, 1e-6))
}

#[derive(Debug, Clone)]
pub struct MlpEmbedder<QB: QuantizedBackend> {
    in_layer: Linear<QB>,
    out_layer: Linear<QB>,
}

impl<QB: QuantizedBackend> MlpEmbedder<QB> {
    fn new(in_sz: usize, h_sz: usize, vb: VarBuilder<QB>) -> Result<Self> {
        let in_layer = linear(in_sz, h_sz, vb.pp("in_layer"))?;
        let out_layer = linear(h_sz, h_sz, vb.pp("out_layer"))?;
        Ok(Self {
            in_layer,
            out_layer,
        })
    }
}

impl<QB: QuantizedBackend> candle::Module<QB::Storage> for MlpEmbedder<QB>
where
    Linear<QB>: candle::Module<QB::Storage>,
{
    fn forward(&self, xs: &Tensor<QB::Storage>) -> Result<Tensor<QB::Storage>> {
        xs.apply(&self.in_layer)?.silu()?.apply(&self.out_layer)
    }
}

#[derive(Debug, Clone)]
pub struct QkNorm<QB: QuantizedBackend> {
    query_norm: RmsNorm<QB::Storage>,
    key_norm: RmsNorm<QB::Storage>,
}

impl<QB: QuantizedBackend> QkNorm<QB> {
    fn new(dim: usize, vb: VarBuilder<QB>) -> Result<Self> {
        let query_norm = vb.get(dim, "query_norm.scale")?.dequantize(vb.device())?;
        let query_norm = RmsNorm::new(query_norm, 1e-6);
        let key_norm = vb.get(dim, "key_norm.scale")?.dequantize(vb.device())?;
        let key_norm = RmsNorm::new(key_norm, 1e-6);
        Ok(Self {
            query_norm,
            key_norm,
        })
    }
}

struct ModulationOut<B: BackendStorage> {
    shift: Tensor<B>,
    scale: Tensor<B>,
    gate: Tensor<B>,
}

impl<B: BackendStorage> ModulationOut<B> {
    fn scale_shift(&self, xs: &Tensor<B>) -> Result<Tensor<B>> {
        xs.broadcast_mul(&(&self.scale + 1.)?)?
            .broadcast_add(&self.shift)
    }

    fn gate(&self, xs: &Tensor<B>) -> Result<Tensor<B>> {
        self.gate.broadcast_mul(xs)
    }
}

#[derive(Debug, Clone)]
struct Modulation1<QB: QuantizedBackend> {
    lin: Linear<QB>,
}

impl<QB: QuantizedBackend> Modulation1<QB>
where
    Linear<QB>: candle::Module<QB::Storage>,
{
    fn new(dim: usize, vb: VarBuilder<QB>) -> Result<Self> {
        let lin = linear(dim, 3 * dim, vb.pp("lin"))?;
        Ok(Self { lin })
    }

    fn forward(&self, vec_: &Tensor<QB::Storage>) -> Result<ModulationOut<QB::Storage>> {
        let ys = vec_
            .silu()?
            .apply(&self.lin)?
            .unsqueeze(1)?
            .chunk(3, D::Minus1)?;
        if ys.len() != 3 {
            candle::bail!("unexpected len from chunk {ys:?}")
        }
        Ok(ModulationOut {
            shift: ys[0].clone(),
            scale: ys[1].clone(),
            gate: ys[2].clone(),
        })
    }
}

#[derive(Debug, Clone)]
struct Modulation2<QB: QuantizedBackend> {
    lin: Linear<QB>,
}

impl<QB: QuantizedBackend> Modulation2<QB>
where
    Linear<QB>: candle::Module<QB::Storage>,
{
    fn new(dim: usize, vb: VarBuilder<QB>) -> Result<Self> {
        let lin = linear(dim, 6 * dim, vb.pp("lin"))?;
        Ok(Self { lin })
    }

    fn forward(
        &self,
        vec_: &Tensor<QB::Storage>,
    ) -> Result<(ModulationOut<QB::Storage>, ModulationOut<QB::Storage>)> {
        let ys = vec_
            .silu()?
            .apply(&self.lin)?
            .unsqueeze(1)?
            .chunk(6, D::Minus1)?;
        if ys.len() != 6 {
            candle::bail!("unexpected len from chunk {ys:?}")
        }
        let mod1 = ModulationOut {
            shift: ys[0].clone(),
            scale: ys[1].clone(),
            gate: ys[2].clone(),
        };
        let mod2 = ModulationOut {
            shift: ys[3].clone(),
            scale: ys[4].clone(),
            gate: ys[5].clone(),
        };
        Ok((mod1, mod2))
    }
}

#[derive(Debug, Clone)]
pub struct SelfAttention<QB: QuantizedBackend> {
    qkv: Linear<QB>,
    norm: QkNorm<QB>,
    proj: Linear<QB>,
    num_heads: usize,
}

impl<QB: QuantizedBackend> SelfAttention<QB>
where
    Linear<QB>: candle::Module<QB::Storage>,
{
    fn new(dim: usize, num_heads: usize, qkv_bias: bool, vb: VarBuilder<QB>) -> Result<Self> {
        let head_dim = dim / num_heads;
        let qkv = linear_b(dim, dim * 3, qkv_bias, vb.pp("qkv"))?;
        let norm = QkNorm::new(head_dim, vb.pp("norm"))?;
        let proj = linear(dim, dim, vb.pp("proj"))?;
        Ok(Self {
            qkv,
            norm,
            proj,
            num_heads,
        })
    }

    fn qkv(
        &self,
        xs: &Tensor<QB::Storage>,
    ) -> Result<(
        Tensor<QB::Storage>,
        Tensor<QB::Storage>,
        Tensor<QB::Storage>,
    )> {
        let qkv = xs.apply(&self.qkv)?;
        let (b, l, _khd) = qkv.dims3()?;
        let qkv = qkv.reshape((b, l, 3, self.num_heads, ()))?;
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?;
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?;
        let q = q.apply(&self.norm.query_norm)?;
        let k = k.apply(&self.norm.key_norm)?;
        Ok((q, k, v))
    }

    #[allow(unused)]
    fn forward(
        &self,
        xs: &Tensor<QB::Storage>,
        pe: &Tensor<QB::Storage>,
    ) -> Result<Tensor<QB::Storage>> {
        let (q, k, v) = self.qkv(xs)?;
        attention(&q, &k, &v, pe)?.apply(&self.proj)
    }
}

#[derive(Debug, Clone)]
struct Mlp<QB: QuantizedBackend> {
    lin1: Linear<QB>,
    lin2: Linear<QB>,
}

impl<QB: QuantizedBackend> Mlp<QB> {
    fn new(in_sz: usize, mlp_sz: usize, vb: VarBuilder<QB>) -> Result<Self> {
        let lin1 = linear(in_sz, mlp_sz, vb.pp("0"))?;
        let lin2 = linear(mlp_sz, in_sz, vb.pp("2"))?;
        Ok(Self { lin1, lin2 })
    }
}

impl<QB: QuantizedBackend> candle::Module<QB::Storage> for Mlp<QB>
where
    Linear<QB>: candle::Module<QB::Storage>,
{
    fn forward(&self, xs: &Tensor<QB::Storage>) -> Result<Tensor<QB::Storage>> {
        xs.apply(&self.lin1)?.gelu()?.apply(&self.lin2)
    }
}

#[derive(Debug, Clone)]
pub struct DoubleStreamBlock<QB: QuantizedBackend> {
    img_mod: Modulation2<QB>,
    img_norm1: LayerNorm<QB::Storage>,
    img_attn: SelfAttention<QB>,
    img_norm2: LayerNorm<QB::Storage>,
    img_mlp: Mlp<QB>,
    txt_mod: Modulation2<QB>,
    txt_norm1: LayerNorm<QB::Storage>,
    txt_attn: SelfAttention<QB>,
    txt_norm2: LayerNorm<QB::Storage>,
    txt_mlp: Mlp<QB>,
}

impl<QB: QuantizedBackend> DoubleStreamBlock<QB>
where
    Linear<QB>: candle::Module<QB::Storage>,
{
    fn new(cfg: &Config, vb: VarBuilder<QB>) -> Result<Self> {
        let h_sz = cfg.hidden_size;
        let mlp_sz = (h_sz as f64 * cfg.mlp_ratio) as usize;
        let img_mod = Modulation2::new(h_sz, vb.pp("img_mod"))?;
        let img_norm1 = layer_norm(h_sz, vb.pp("img_norm1"))?;
        let img_attn = SelfAttention::new(h_sz, cfg.num_heads, cfg.qkv_bias, vb.pp("img_attn"))?;
        let img_norm2 = layer_norm(h_sz, vb.pp("img_norm2"))?;
        let img_mlp = Mlp::new(h_sz, mlp_sz, vb.pp("img_mlp"))?;
        let txt_mod = Modulation2::new(h_sz, vb.pp("txt_mod"))?;
        let txt_norm1 = layer_norm(h_sz, vb.pp("txt_norm1"))?;
        let txt_attn = SelfAttention::new(h_sz, cfg.num_heads, cfg.qkv_bias, vb.pp("txt_attn"))?;
        let txt_norm2 = layer_norm(h_sz, vb.pp("txt_norm2"))?;
        let txt_mlp = Mlp::new(h_sz, mlp_sz, vb.pp("txt_mlp"))?;
        Ok(Self {
            img_mod,
            img_norm1,
            img_attn,
            img_norm2,
            img_mlp,
            txt_mod,
            txt_norm1,
            txt_attn,
            txt_norm2,
            txt_mlp,
        })
    }

    fn forward(
        &self,
        img: &Tensor<QB::Storage>,
        txt: &Tensor<QB::Storage>,
        vec_: &Tensor<QB::Storage>,
        pe: &Tensor<QB::Storage>,
    ) -> Result<(Tensor<QB::Storage>, Tensor<QB::Storage>)> {
        let (img_mod1, img_mod2) = self.img_mod.forward(vec_)?; // shift, scale, gate
        let (txt_mod1, txt_mod2) = self.txt_mod.forward(vec_)?; // shift, scale, gate
        let img_modulated = img.apply(&self.img_norm1)?;
        let img_modulated = img_mod1.scale_shift(&img_modulated)?;
        let (img_q, img_k, img_v) = self.img_attn.qkv(&img_modulated)?;

        let txt_modulated = txt.apply(&self.txt_norm1)?;
        let txt_modulated = txt_mod1.scale_shift(&txt_modulated)?;
        let (txt_q, txt_k, txt_v) = self.txt_attn.qkv(&txt_modulated)?;

        let q = Tensor::cat(&[txt_q, img_q], 2)?;
        let k = Tensor::cat(&[txt_k, img_k], 2)?;
        let v = Tensor::cat(&[txt_v, img_v], 2)?;

        let attn = attention(&q, &k, &v, pe)?;
        let txt_attn = attn.narrow(1, 0, txt.dim(1)?)?;
        let img_attn = attn.narrow(1, txt.dim(1)?, attn.dim(1)? - txt.dim(1)?)?;

        let img = (img + img_mod1.gate(&img_attn.apply(&self.img_attn.proj)?))?;
        let img = (&img
            + img_mod2.gate(
                &img_mod2
                    .scale_shift(&img.apply(&self.img_norm2)?)?
                    .apply(&self.img_mlp)?,
            )?)?;

        let txt = (txt + txt_mod1.gate(&txt_attn.apply(&self.txt_attn.proj)?))?;
        let txt = (&txt
            + txt_mod2.gate(
                &txt_mod2
                    .scale_shift(&txt.apply(&self.txt_norm2)?)?
                    .apply(&self.txt_mlp)?,
            )?)?;

        Ok((img, txt))
    }
}

#[derive(Debug, Clone)]
pub struct SingleStreamBlock<QB: QuantizedBackend> {
    linear1: Linear<QB>,
    linear2: Linear<QB>,
    norm: QkNorm<QB>,
    pre_norm: LayerNorm<QB::Storage>,
    modulation: Modulation1<QB>,
    h_sz: usize,
    mlp_sz: usize,
    num_heads: usize,
}

impl<QB: QuantizedBackend> SingleStreamBlock<QB>
where
    Linear<QB>: candle::Module<QB::Storage>,
{
    fn new(cfg: &Config, vb: VarBuilder<QB>) -> Result<Self> {
        let h_sz = cfg.hidden_size;
        let mlp_sz = (h_sz as f64 * cfg.mlp_ratio) as usize;
        let head_dim = h_sz / cfg.num_heads;
        let linear1 = linear(h_sz, h_sz * 3 + mlp_sz, vb.pp("linear1"))?;
        let linear2 = linear(h_sz + mlp_sz, h_sz, vb.pp("linear2"))?;
        let norm = QkNorm::new(head_dim, vb.pp("norm"))?;
        let pre_norm = layer_norm(h_sz, vb.pp("pre_norm"))?;
        let modulation = Modulation1::new(h_sz, vb.pp("modulation"))?;
        Ok(Self {
            linear1,
            linear2,
            norm,
            pre_norm,
            modulation,
            h_sz,
            mlp_sz,
            num_heads: cfg.num_heads,
        })
    }

    fn forward(
        &self,
        xs: &Tensor<QB::Storage>,
        vec_: &Tensor<QB::Storage>,
        pe: &Tensor<QB::Storage>,
    ) -> Result<Tensor<QB::Storage>> {
        let mod_ = self.modulation.forward(vec_)?;
        let x_mod = mod_.scale_shift(&xs.apply(&self.pre_norm)?)?;
        let x_mod = x_mod.apply(&self.linear1)?;
        let qkv = x_mod.narrow(D::Minus1, 0, 3 * self.h_sz)?;
        let (b, l, _khd) = qkv.dims3()?;
        let qkv = qkv.reshape((b, l, 3, self.num_heads, ()))?;
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?;
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?;
        let mlp = x_mod.narrow(D::Minus1, 3 * self.h_sz, self.mlp_sz)?;
        let q = q.apply(&self.norm.query_norm)?;
        let k = k.apply(&self.norm.key_norm)?;
        let attn = attention(&q, &k, &v, pe)?;
        let output = Tensor::cat(&[attn, mlp.gelu()?], 2)?.apply(&self.linear2)?;
        xs + mod_.gate(&output)
    }
}

#[derive(Debug, Clone)]
pub struct LastLayer<QB: QuantizedBackend> {
    norm_final: LayerNorm<QB::Storage>,
    linear: Linear<QB>,
    ada_ln_modulation: Linear<QB>,
}

impl<QB: QuantizedBackend> LastLayer<QB>
where
    Linear<QB>: candle::Module<QB::Storage>,
{
    fn new(h_sz: usize, p_sz: usize, out_c: usize, vb: VarBuilder<QB>) -> Result<Self> {
        let norm_final = layer_norm(h_sz, vb.pp("norm_final"))?;
        let linear_ = linear(h_sz, p_sz * p_sz * out_c, vb.pp("linear"))?;
        let ada_ln_modulation = linear(h_sz, 2 * h_sz, vb.pp("adaLN_modulation.1"))?;
        Ok(Self {
            norm_final,
            linear: linear_,
            ada_ln_modulation,
        })
    }

    fn forward(
        &self,
        xs: &Tensor<QB::Storage>,
        vec: &Tensor<QB::Storage>,
    ) -> Result<Tensor<QB::Storage>> {
        let chunks = vec.silu()?.apply(&self.ada_ln_modulation)?.chunk(2, 1)?;
        let (shift, scale) = (&chunks[0], &chunks[1]);
        let xs = xs
            .apply(&self.norm_final)?
            .broadcast_mul(&(scale.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift.unsqueeze(1)?)?;
        xs.apply(&self.linear)
    }
}

#[derive(Debug, Clone)]
pub struct Flux<QB: QuantizedBackend> {
    img_in: Linear<QB>,
    txt_in: Linear<QB>,
    time_in: MlpEmbedder<QB>,
    vector_in: MlpEmbedder<QB>,
    guidance_in: Option<MlpEmbedder<QB>>,
    pe_embedder: EmbedNd,
    double_blocks: Vec<DoubleStreamBlock<QB>>,
    single_blocks: Vec<SingleStreamBlock<QB>>,
    final_layer: LastLayer<QB>,
}

impl<QB: QuantizedBackend> Flux<QB>
where
    Linear<QB>: candle::Module<QB::Storage>,
{
    pub fn new(cfg: &Config, vb: VarBuilder<QB>) -> Result<Self> {
        let img_in = linear(cfg.in_channels, cfg.hidden_size, vb.pp("img_in"))?;
        let txt_in = linear(cfg.context_in_dim, cfg.hidden_size, vb.pp("txt_in"))?;
        let mut double_blocks = Vec::with_capacity(cfg.depth);
        let vb_d = vb.pp("double_blocks");
        for idx in 0..cfg.depth {
            let db = DoubleStreamBlock::new(cfg, vb_d.pp(idx))?;
            double_blocks.push(db)
        }
        let mut single_blocks = Vec::with_capacity(cfg.depth_single_blocks);
        let vb_s = vb.pp("single_blocks");
        for idx in 0..cfg.depth_single_blocks {
            let sb = SingleStreamBlock::new(cfg, vb_s.pp(idx))?;
            single_blocks.push(sb)
        }
        let time_in = MlpEmbedder::new(256, cfg.hidden_size, vb.pp("time_in"))?;
        let vector_in = MlpEmbedder::new(cfg.vec_in_dim, cfg.hidden_size, vb.pp("vector_in"))?;
        let guidance_in = if cfg.guidance_embed {
            let mlp = MlpEmbedder::new(256, cfg.hidden_size, vb.pp("guidance_in"))?;
            Some(mlp)
        } else {
            None
        };
        let final_layer =
            LastLayer::new(cfg.hidden_size, 1, cfg.in_channels, vb.pp("final_layer"))?;
        let pe_dim = cfg.hidden_size / cfg.num_heads;
        let pe_embedder = EmbedNd::new(pe_dim, cfg.theta, cfg.axes_dim.to_vec());
        Ok(Self {
            img_in,
            txt_in,
            time_in,
            vector_in,
            guidance_in,
            pe_embedder,
            double_blocks,
            single_blocks,
            final_layer,
        })
    }
}

impl<QB: QuantizedBackend> super::WithForward<QB::Storage> for Flux<QB>
where
    Linear<QB>: candle::Module<QB::Storage>,
{
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        img: &Tensor<QB::Storage>,
        img_ids: &Tensor<QB::Storage>,
        txt: &Tensor<QB::Storage>,
        txt_ids: &Tensor<QB::Storage>,
        timesteps: &Tensor<QB::Storage>,
        y: &Tensor<QB::Storage>,
        guidance: Option<&Tensor<QB::Storage>>,
    ) -> Result<Tensor<QB::Storage>> {
        if txt.rank() != 3 {
            candle::bail!("unexpected shape for txt {:?}", txt.shape())
        }
        if img.rank() != 3 {
            candle::bail!("unexpected shape for img {:?}", img.shape())
        }
        let dtype = img.dtype();
        let pe = {
            let ids = Tensor::cat(&[txt_ids, img_ids], 1)?;
            ids.apply(&self.pe_embedder)?
        };
        let mut txt = txt.apply(&self.txt_in)?;
        let mut img = img.apply(&self.img_in)?;
        let vec_ = timestep_embedding(timesteps, 256, dtype)?.apply(&self.time_in)?;
        let vec_ = match (self.guidance_in.as_ref(), guidance) {
            (Some(g_in), Some(guidance)) => {
                (vec_ + timestep_embedding(guidance, 256, dtype)?.apply(g_in))?
            }
            _ => vec_,
        };
        let vec_ = (vec_ + y.apply(&self.vector_in))?;

        // Double blocks
        for block in self.double_blocks.iter() {
            (img, txt) = block.forward(&img, &txt, &vec_, &pe)?
        }
        // Single blocks
        let mut img = Tensor::cat(&[&txt, &img], 1)?;
        for block in self.single_blocks.iter() {
            img = block.forward(&img, &vec_, &pe)?;
        }
        let img = img.i((.., txt.dim(1)?..))?;
        self.final_layer.forward(&img, &vec_)
    }
}
