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
struct PromptEncoder {}
