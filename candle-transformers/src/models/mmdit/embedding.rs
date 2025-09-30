use candle::{bail, BackendStorage, DType, Module, Result, Tensor};
use candle_nn as nn;

pub struct PatchEmbedder<B: BackendStorage> {
    proj: nn::Conv2d<B>,
}

impl<B: BackendStorage> PatchEmbedder<B> {
    pub fn new(
        patch_size: usize,
        in_channels: usize,
        embed_dim: usize,
        vb: nn::VarBuilder<B>,
    ) -> Result<Self> {
        let proj = nn::conv2d(
            in_channels,
            embed_dim,
            patch_size,
            nn::Conv2dConfig {
                stride: patch_size,
                ..Default::default()
            },
            vb.pp("proj"),
        )?;

        Ok(Self { proj })
    }
}

impl<B: BackendStorage> Module<B> for PatchEmbedder<B> {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        let x = self.proj.forward(x)?;

        // flatten spatial dim and transpose to channels last
        let (b, c, h, w) = x.dims4()?;
        x.reshape((b, c, h * w))?.transpose(1, 2)
    }
}

pub struct Unpatchifier {
    patch_size: usize,
    out_channels: usize,
}

impl Unpatchifier {
    pub fn new(patch_size: usize, out_channels: usize) -> Result<Self> {
        Ok(Self {
            patch_size,
            out_channels,
        })
    }

    pub fn unpatchify<B: BackendStorage>(
        &self,
        x: &Tensor<B>,
        h: usize,
        w: usize,
    ) -> Result<Tensor<B>> {
        let h = (h + 1) / self.patch_size;
        let w = (w + 1) / self.patch_size;

        let x = x.reshape((
            x.dim(0)?,
            h,
            w,
            self.patch_size,
            self.patch_size,
            self.out_channels,
        ))?;
        let x = x.permute((0, 5, 1, 3, 2, 4))?; // "nhwpqc->nchpwq"
        x.reshape((
            x.dim(0)?,
            self.out_channels,
            self.patch_size * h,
            self.patch_size * w,
        ))
    }
}

pub struct PositionEmbedder<B: BackendStorage> {
    pos_embed: Tensor<B>,
    patch_size: usize,
    pos_embed_max_size: usize,
}

impl<B: BackendStorage> PositionEmbedder<B> {
    pub fn new(
        hidden_size: usize,
        patch_size: usize,
        pos_embed_max_size: usize,
        vb: nn::VarBuilder<B>,
    ) -> Result<Self> {
        let pos_embed = vb.get(
            (1, pos_embed_max_size * pos_embed_max_size, hidden_size),
            "pos_embed",
        )?;
        Ok(Self {
            pos_embed,
            patch_size,
            pos_embed_max_size,
        })
    }
    pub fn get_cropped_pos_embed(&self, h: usize, w: usize) -> Result<Tensor<B>> {
        let h = (h + 1) / self.patch_size;
        let w = (w + 1) / self.patch_size;

        if h > self.pos_embed_max_size || w > self.pos_embed_max_size {
            bail!("Input size is too large for the position embedding")
        }

        let top = (self.pos_embed_max_size - h) / 2;
        let left = (self.pos_embed_max_size - w) / 2;

        let pos_embed =
            self.pos_embed
                .reshape((1, self.pos_embed_max_size, self.pos_embed_max_size, ()))?;
        let pos_embed = pos_embed.narrow(1, top, h)?.narrow(2, left, w)?;
        pos_embed.reshape((1, h * w, ()))
    }
}

pub struct TimestepEmbedder<B: BackendStorage> {
    mlp: nn::Sequential<B>,
    frequency_embedding_size: usize,
}

impl<B: BackendStorage + 'static> TimestepEmbedder<B> {
    pub fn new(
        hidden_size: usize,
        frequency_embedding_size: usize,
        vb: nn::VarBuilder<B>,
    ) -> Result<Self> {
        let mlp = nn::seq()
            .add(nn::linear(
                frequency_embedding_size,
                hidden_size,
                vb.pp("mlp.0"),
            )?)
            .add(nn::Activation::Silu)
            .add(nn::linear(hidden_size, hidden_size, vb.pp("mlp.2"))?);

        Ok(Self {
            mlp,
            frequency_embedding_size,
        })
    }

    fn timestep_embedding(t: &Tensor<B>, dim: usize, max_period: f64) -> Result<Tensor<B>> {
        if dim % 2 != 0 {
            bail!("Embedding dimension must be even")
        }

        if t.dtype() != DType::F32 && t.dtype() != DType::F64 {
            bail!("Input tensor must be floating point")
        }

        let half = dim / 2;
        let freqs = Tensor::arange(0f32, half as f32, t.device())?
            .to_dtype(candle::DType::F32)?
            .mul(&Tensor::full(
                (-f64::ln(max_period) / half as f64) as f32,
                half,
                t.device(),
            )?)?
            .exp()?;

        let args = t
            .unsqueeze(1)?
            .to_dtype(candle::DType::F32)?
            .matmul(&freqs.unsqueeze(0)?)?;
        let embedding = Tensor::cat(&[args.cos()?, args.sin()?], 1)?;
        embedding.to_dtype(candle::DType::F16)
    }
}

impl<B: BackendStorage + 'static> Module<B> for TimestepEmbedder<B> {
    fn forward(&self, t: &Tensor<B>) -> Result<Tensor<B>> {
        let t_freq = Self::timestep_embedding(t, self.frequency_embedding_size, 10000.0)?;
        self.mlp.forward(&t_freq)
    }
}

pub struct VectorEmbedder<B: BackendStorage> {
    mlp: nn::Sequential<B>,
}

impl<B: BackendStorage + 'static> VectorEmbedder<B> {
    pub fn new(input_dim: usize, hidden_size: usize, vb: nn::VarBuilder<B>) -> Result<Self> {
        let mlp = nn::seq()
            .add(nn::linear(input_dim, hidden_size, vb.pp("mlp.0"))?)
            .add(nn::Activation::Silu)
            .add(nn::linear(hidden_size, hidden_size, vb.pp("mlp.2"))?);

        Ok(Self { mlp })
    }
}

impl<B: BackendStorage> Module<B> for VectorEmbedder<B> {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        self.mlp.forward(x)
    }
}
