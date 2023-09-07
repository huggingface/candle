use candle::{DType, IndexOp, Result, Tensor, D};
use candle_nn::{layer_norm, LayerNorm, Linear, Module, VarBuilder};

#[derive(Debug)]
struct PostionEmbeddingRandom {
    positional_encoding_gaussian_matrix: Tensor,
}

impl PostionEmbeddingRandom {
    fn new(num_pos_feats: usize, vb: VarBuilder) -> Result<Self> {
        let positional_encoding_gaussian_matrix =
            vb.get((2, num_pos_feats), "positional_encoding_gaussian_matrix")?;
        Ok(Self {
            positional_encoding_gaussian_matrix,
        })
    }

    fn pe_encoding(&self, coords: &Tensor) -> Result<Tensor> {
        let coords = coords.affine(2., -1.)?;
        let coords = coords.matmul(&self.positional_encoding_gaussian_matrix)?;
        let coords = (coords * (2. * std::f64::consts::PI))?;
        Tensor::cat(&[coords.sin()?, coords.cos()?], D::Minus1)
    }

    fn forward(&self, h: usize, w: usize) -> Result<Tensor> {
        let device = self.positional_encoding_gaussian_matrix.device();
        let grid = Tensor::ones((h, w), DType::F32, device)?;
        // TODO: cumsum
        let x_embed = (&grid - 0.5)?;
        // TODO: cumsum
        let y_embed = (&grid - 0.5)?;
        let x_embed = (x_embed / w as f64)?;
        let y_embed = (y_embed / h as f64)?;
        let coords = Tensor::stack(&[&x_embed, &y_embed], D::Minus1)?;
        self.pe_encoding(&coords)?.permute((2, 0, 1))
    }

    fn forward_with_coords(
        &self,
        coords_input: &Tensor,
        image_size: (usize, usize),
    ) -> Result<Tensor> {
        let coords0 = (coords_input.narrow(D::Minus1, 0, 1)? / image_size.1 as f64)?;
        let coords1 = (coords_input.narrow(D::Minus1, 1, 1)? / image_size.0 as f64)?;
        let c = coords_input.dim(D::Minus1)?;
        let coords_rest = coords_input.narrow(D::Minus1, 2, c - 2)?;
        let coords = Tensor::cat(&[&coords0, &coords1, &coords_rest], D::Minus1)?;
        self.pe_encoding(&coords)
    }
}

#[derive(Debug)]
struct PromptEncoder {
    pe_layer: PostionEmbeddingRandom,
    point_embeddings: Vec<candle_nn::Embedding>,
    not_a_point_embed: candle_nn::Embedding,
    mask_downscaling_conv1: candle_nn::Conv2d,
    mask_downscaling_ln1: LayerNorm,
    mask_downscaling_conv2: candle_nn::Conv2d,
    mask_downscaling_ln2: LayerNorm,
    mask_downscaling_conv3: candle_nn::Conv2d,
    no_mask_embed: candle_nn::Embedding,
}

impl PromptEncoder {
    fn new(
        embed_dim: usize,
        image_embedding_size: (usize, usize),
        input_image_size: (usize, usize),
        mask_in_chans: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let num_points_embeddings = 4;
        let pe_layer = PostionEmbeddingRandom::new(embed_dim / 2, vb.pp("pe_layer"))?;
        let not_a_point_embed = candle_nn::embedding(1, embed_dim, vb.pp("not_a_point_embed"))?;
        let no_mask_embed = candle_nn::embedding(1, embed_dim, vb.pp("no_mask_embed"))?;
        let cfg = candle_nn::Conv2dConfig {
            stride: 2,
            ..Default::default()
        };
        let mask_downscaling_conv1 =
            candle_nn::conv2d(1, mask_in_chans / 4, 2, cfg, vb.pp("mask_downscaling.0"))?;
        let mask_downscaling_conv2 = candle_nn::conv2d(
            mask_in_chans / 4,
            mask_in_chans,
            2,
            cfg,
            vb.pp("mask_downscaling.3"),
        )?;
        let mask_downscaling_conv3 = candle_nn::conv2d(
            mask_in_chans,
            embed_dim,
            1,
            Default::default(),
            vb.pp("mask_downscaling.6"),
        )?;
        let mask_downscaling_ln1 =
            layer_norm(mask_in_chans / 4, 1e-6, vb.pp("mask_downscaling.1"))?;
        let mask_downscaling_ln2 = layer_norm(mask_in_chans, 1e-6, vb.pp("mask_downscaling.4"))?;
        let mut point_embeddings = Vec::with_capacity(num_points_embeddings);
        let vb_e = vb.pp("points_embeddings");
        for i in 0..num_points_embeddings {
            let emb = candle_nn::embedding(1, embed_dim, vb_e.pp(i))?;
            point_embeddings.push(emb)
        }
        Ok(Self {
            pe_layer,
            point_embeddings,
            not_a_point_embed,
            mask_downscaling_conv1,
            mask_downscaling_ln1,
            mask_downscaling_conv2,
            mask_downscaling_ln2,
            mask_downscaling_conv3,
            no_mask_embed,
        })
    }
}
