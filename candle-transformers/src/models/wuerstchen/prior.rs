use super::common::{AttnBlock, ResBlock, TimestepBlock};
use candle::{DType, Result, Tensor, D};
use candle_nn::VarBuilder;

fn wuerstchen_prior_freqs_cached(device: &candle::Device, half_dim: usize, neg_emb_scale: f64) -> Result<Tensor> {
    use std::sync::{Mutex, OnceLock};
    use std::collections::HashMap;
    type Key = (candle::DeviceLocation, usize, u64);
    static CACHE: OnceLock<Mutex<HashMap<Key, Tensor>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (device.location(), half_dim, neg_emb_scale.to_bits());
    if let Ok(g) = cache.lock() {
        if let Some(t) = g.get(&key) {
            return Ok(t.clone());
        }
    }
    let t = (Tensor::arange(0u32, half_dim as u32, device)?.to_dtype(DType::F32)? * -neg_emb_scale)?
        .exp()?
        .unsqueeze(0)?;
    if let Ok(mut g) = cache.lock() {
        g.insert(key, t.clone());
    }
    Ok(t)
}

#[derive(Debug)]
struct Block {
    res_block: ResBlock,
    ts_block: TimestepBlock,
    attn_block: AttnBlock,
}

#[derive(Debug)]
pub struct WPrior {
    projection: candle_nn::Conv2d,
    cond_mapper_lin1: candle_nn::Linear,
    cond_mapper_lin2: candle_nn::Linear,
    blocks: Vec<Block>,
    out_ln: super::common::WLayerNorm,
    out_conv: candle_nn::Conv2d,
    c_r: usize,
    // c_embed = c.apply(lin1).leaky_relu().apply(lin2) on constant CLIP
    // text embedding — cache by c.id() so the per-image gen invocations
    // share the result and downstream Attention caches also stay hot.
    c_emb_cache: std::sync::Mutex<Option<(candle::TensorId, Tensor)>>,
}

impl WPrior {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        c_in: usize,
        c: usize,
        c_cond: usize,
        c_r: usize,
        depth: usize,
        nhead: usize,
        use_flash_attn: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let projection = candle_nn::conv2d(c_in, c, 1, Default::default(), vb.pp("projection"))?;
        let cond_mapper_lin1 = candle_nn::linear(c_cond, c, vb.pp("cond_mapper.0"))?;
        let cond_mapper_lin2 = candle_nn::linear(c, c, vb.pp("cond_mapper.2"))?;
        let out_ln = super::common::WLayerNorm::new(c)?;
        let out_conv = candle_nn::conv2d(c, c_in * 2, 1, Default::default(), vb.pp("out.1"))?;
        let mut blocks = Vec::with_capacity(depth);
        for index in 0..depth {
            let res_block = ResBlock::new(c, 0, 3, vb.pp(format!("blocks.{}", 3 * index)))?;
            let ts_block = TimestepBlock::new(c, c_r, vb.pp(format!("blocks.{}", 3 * index + 1)))?;
            let attn_block = AttnBlock::new(
                c,
                c,
                nhead,
                true,
                use_flash_attn,
                vb.pp(format!("blocks.{}", 3 * index + 2)),
            )?;
            blocks.push(Block {
                res_block,
                ts_block,
                attn_block,
            })
        }
        Ok(Self {
            projection,
            cond_mapper_lin1,
            cond_mapper_lin2,
            blocks,
            out_ln,
            out_conv,
            c_r,
            c_emb_cache: std::sync::Mutex::new(None),
        })
    }

    pub fn gen_r_embedding(&self, r: &Tensor) -> Result<Tensor> {
        const MAX_POSITIONS: usize = 10000;
        let r = (r * MAX_POSITIONS as f64)?;
        let half_dim = self.c_r / 2;
        let emb_const = (MAX_POSITIONS as f64).ln() / (half_dim - 1) as f64;
        let emb = wuerstchen_prior_freqs_cached(r.device(), half_dim, emb_const)?;
        let emb = r.unsqueeze(1)?.broadcast_mul(&emb)?;
        let emb = Tensor::cat(&[emb.sin()?, emb.cos()?], 1)?;
        let emb = if self.c_r % 2 == 1 {
            emb.pad_with_zeros(D::Minus1, 0, 1)?
        } else {
            emb
        };
        emb.to_dtype(r.dtype())
    }

    pub fn forward(&self, xs: &Tensor, r: &Tensor, c: &Tensor) -> Result<Tensor> {
        let x_in = xs;
        let mut xs = xs.apply(&self.projection)?;
        let c_embed = {
            let id = c.id();
            let mut g = self.c_emb_cache.lock().unwrap_or_else(|e| e.into_inner());
            if let Some((cid, ref t)) = *g {
                if cid == id {
                    t.clone()
                } else {
                    let emb = c
                        .apply(&self.cond_mapper_lin1)?
                        .apply(&|xs: &_| candle_nn::ops::leaky_relu(xs, 0.2))?
                        .apply(&self.cond_mapper_lin2)?;
                    *g = Some((id, emb.clone()));
                    emb
                }
            } else {
                let emb = c
                    .apply(&self.cond_mapper_lin1)?
                    .apply(&|xs: &_| candle_nn::ops::leaky_relu(xs, 0.2))?
                    .apply(&self.cond_mapper_lin2)?;
                *g = Some((id, emb.clone()));
                emb
            }
        };
        let r_embed = self.gen_r_embedding(r)?;
        for block in self.blocks.iter() {
            xs = block.res_block.forward(&xs, None)?;
            xs = block.ts_block.forward(&xs, &r_embed)?;
            xs = block.attn_block.forward(&xs, &c_embed)?;
        }
        let ab = xs.apply(&self.out_ln)?.apply(&self.out_conv)?.chunk(2, 1)?;
        (x_in - &ab[0])? / ((&ab[1] - 1.)?.abs()? + 1e-5)
    }
}
