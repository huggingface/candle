use candle::{DType, IndexOp, Result, Tensor, D};
use candle_nn::{layer_norm, LayerNorm, Linear, Module, VarBuilder};

#[derive(Debug)]
struct MlpBlock {
    lin1: Linear,
    lin2: Linear,
}

impl MlpBlock {
    fn new(embedding_dim: usize, mlp_dim: usize, vb: VarBuilder) -> Result<Self> {
        let lin1 = candle_nn::linear(embedding_dim, mlp_dim, vb.pp("lin1"))?;
        let lin2 = candle_nn::linear(mlp_dim, embedding_dim, vb.pp("lin2"))?;
        Ok(Self { lin1, lin2 })
    }
}

impl Module for MlpBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.lin1)?.gelu()?.apply(&self.lin2)
    }
}

#[derive(Debug)]
struct PatchEmbed {
    proj: candle_nn::Conv2d,
}

impl PatchEmbed {
    fn new(
        in_chans: usize,
        embed_dim: usize,
        k_size: usize,
        stride: usize,
        padding: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = candle_nn::Conv2dConfig {
            stride,
            padding,
            ..Default::default()
        };
        let proj = candle_nn::conv2d(in_chans, embed_dim, k_size, cfg, vb.pp("proj"))?;
        Ok(Self { proj })
    }
}

impl Module for PatchEmbed {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.proj)?.permute((0, 2, 3, 1))
    }
}

#[derive(Debug)]
struct Attention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    scale: f64,
    use_rel_pos: bool,
    rel_pos_hw: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(
        dim: usize,
        num_heads: usize,
        qkv_bias: bool,
        use_rel_pos: bool,
        window_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let qkv = crate::linear(vb.pp("qkv"), dim, dim * 3, qkv_bias)?;
        let proj = crate::linear(vb.pp("proj"), dim, dim, true)?;
        let head_dim = dim / num_heads;
        let scale = 1. / (head_dim as f64).sqrt();
        let rel_pos_hw = if use_rel_pos {
            let h = vb.get((2 * window_size - 1, head_dim), "rel_pos_h")?;
            let w = vb.get((2 * window_size - 1, head_dim), "rel_pos_w")?;
            Some((h, w))
        } else {
            None
        };
        Ok(Self {
            qkv,
            proj,
            num_heads,
            scale,
            use_rel_pos,
            rel_pos_hw,
        })
    }
}

impl Module for Attention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, h, w, c) = xs.dims4()?;
        let qkv = self
            .qkv
            .forward(xs)?
            .reshape((b, h * w, 3, self.num_heads, c / self.num_heads))?
            .permute((2, 0, 3, 1, 4))?
            .reshape((3, b * self.num_heads, h * w, c / self.num_heads))?;
        let q = qkv.i(0)?;
        let k = qkv.i(1)?;
        let v = qkv.i(2)?;
        let attn = (q * self.scale)?.matmul(&k.t()?)?;
        if self.use_rel_pos {
            todo!()
        }
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let attn = attn
            .matmul(&v)?
            .reshape((b, self.num_heads, h, w, c / self.num_heads))?
            .permute((0, 2, 3, 1, 4))?
            .reshape((b, h, w, c / self.num_heads))?;
        self.proj.forward(&attn)
    }
}

#[derive(Debug)]
struct Block {
    norm1: LayerNorm,
    attn: Attention,
    norm2: LayerNorm,
    mlp: MlpBlock,
    window_size: usize,
}

impl Block {
    fn new(
        dim: usize,
        num_heads: usize,
        qkv_bias: bool,
        use_rel_pos: bool,
        window_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm1 = layer_norm(dim, 1e-5, vb.pp("norm1"))?;
        let norm2 = layer_norm(dim, 1e-5, vb.pp("norm2"))?;
        let attn = Attention::new(
            dim,
            num_heads,
            qkv_bias,
            use_rel_pos,
            window_size,
            vb.pp("attn"),
        )?;
        let mlp = MlpBlock::new(dim, dim * 4, vb.pp("mlp"))?;
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
            window_size,
        })
    }
}

impl Module for Block {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let shortcut = xs;
        let xs = self.norm1.forward(xs)?;
        if self.window_size > 0 {
            todo!()
        }
        let xs = self.attn.forward(&xs)?;
        if self.window_size > 0 {
            todo!()
        }
        let xs = (xs + shortcut)?;
        &xs + xs.apply(&self.norm2)?.apply(&self.mlp)?
    }
}

#[derive(Debug)]
struct ImageEncoderViT {
    img_size: usize,
    patch_embed: PatchEmbed,
    blocks: Vec<Block>,
    neck_conv1: candle_nn::Conv2d,
    neck_ln1: LayerNorm,
    neck_conv2: candle_nn::Conv2d,
    neck_ln2: LayerNorm,
    pos_embed: Option<Tensor>,
}

impl ImageEncoderViT {
    #[allow(clippy::too_many_arguments)]
    fn new(
        img_size: usize,
        patch_size: usize,
        in_chans: usize,
        embed_dim: usize,
        depth: usize,
        num_heads: usize,
        out_chans: usize,
        qkv_bias: bool,
        use_rel_pos: bool,
        use_abs_pos: bool,
        window_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let patch_embed = PatchEmbed::new(
            in_chans,
            embed_dim,
            patch_size,
            patch_size,
            0,
            vb.pp("patch_embed"),
        )?;
        let mut blocks = Vec::with_capacity(depth);
        let vb_b = vb.pp("blocks");
        for i in 0..depth {
            let block = Block::new(
                embed_dim,
                num_heads,
                qkv_bias,
                use_rel_pos,
                window_size,
                vb_b.pp(i),
            )?;
            blocks.push(block)
        }
        let neck_conv1 = candle_nn::conv2d_no_bias(
            embed_dim,
            out_chans,
            1,
            Default::default(),
            vb.pp("neck.0"),
        )?;
        let neck_ln1 = layer_norm(out_chans, 1e-6, vb.pp("neck.1"))?;
        let cfg = candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let neck_conv2 = candle_nn::conv2d_no_bias(out_chans, out_chans, 3, cfg, vb.pp("neck.2"))?;
        let neck_ln2 = layer_norm(out_chans, 1e-6, vb.pp("neck.3"))?;
        let pos_embed = if use_abs_pos {
            let p = vb.get(
                (1, img_size / patch_size, img_size / patch_size, embed_dim),
                "pos_embed",
            )?;
            Some(p)
        } else {
            None
        };
        Ok(Self {
            img_size,
            patch_embed,
            blocks,
            neck_conv1,
            neck_ln1,
            neck_conv2,
            neck_ln2,
            pos_embed,
        })
    }
}

impl Module for ImageEncoderViT {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.patch_embed.forward(xs)?;
        let mut xs = match &self.pos_embed {
            Some(pos_embed) => (xs + pos_embed)?,
            None => xs,
        };
        for block in self.blocks.iter() {
            xs = block.forward(&xs)?
        }
        xs.permute((0, 3, 1, 2))?
            .apply(&self.neck_conv1)?
            .apply(&self.neck_ln1)?
            .apply(&self.neck_conv2)?
            .apply(&self.neck_ln2)
    }
}
