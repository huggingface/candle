use candle::{Result, Tensor};
use candle_nn::VarBuilder;

// https://github.com/black-forest-labs/flux/blob/727e3a71faf37390f318cf9434f0939653302b60/src/flux/model.py#L12
#[derive(Debug, Clone)]
pub struct Config {
    pub in_channels: usize,
    pub vec_in_dim: usize,
    pub context_in_dim: usize,
    pub hidden_size: usize,
    pub mlp_ratio: f64,
    pub num_heads: usize,
    pub depth: usize,
    pub depth_single_blocks: usize,
    pub axes_dim: Vec<usize>,
    pub theta: usize,
    pub qkv_bias: bool,
    pub guidance_embed: bool,
}

impl Config {
    // https://github.com/black-forest-labs/flux/blob/727e3a71faf37390f318cf9434f0939653302b60/src/flux/util.py#L32
    pub fn dev() -> Self {
        Self {
            in_channels: 64,
            vec_in_dim: 768,
            context_in_dim: 4096,
            hidden_size: 3072,
            mlp_ratio: 4.0,
            num_heads: 24,
            depth: 19,
            depth_single_blocks: 38,
            axes_dim: vec![16, 56, 56],
            theta: 10_000,
            qkv_bias: true,
            guidance_embed: true,
        }
    }

    // https://github.com/black-forest-labs/flux/blob/727e3a71faf37390f318cf9434f0939653302b60/src/flux/util.py#L64
    pub fn schnell() -> Self {
        Self {
            in_channels: 64,
            vec_in_dim: 768,
            context_in_dim: 4096,
            hidden_size: 3072,
            mlp_ratio: 4.0,
            num_heads: 24,
            depth: 19,
            depth_single_blocks: 38,
            axes_dim: vec![16, 56, 56],
            theta: 10_000,
            qkv_bias: true,
            guidance_embed: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Flux {
    img_in: candle_nn::Linear,
    txt_in: candle_nn::Linear,
}

impl Flux {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let img_in = candle_nn::linear(cfg.in_channels, cfg.hidden_size, vb.pp("img_in"))?;
        let txt_in = candle_nn::linear(cfg.context_in_dim, cfg.hidden_size, vb.pp("txt_in"))?;
        Ok(Self { img_in, txt_in })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        img: &Tensor,
        _img_ids: &Tensor,
        txt: &Tensor,
        _txt_ids: &Tensor,
        _timesteps: &Tensor,
        _y: &Tensor,
        _guidance: Option<&Tensor>,
    ) -> Result<Tensor> {
        if txt.rank() != 3 {
            candle::bail!("unexpected shape for txt {:?}", txt.shape())
        }
        if img.rank() != 3 {
            candle::bail!("unexpected shape for img {:?}", img.shape())
        }
        let _txt = txt.apply(&self.txt_in)?;
        let _img = img.apply(&self.img_in)?;
        todo!()
    }
}
