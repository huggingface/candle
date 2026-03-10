use candle::Tensor;
use candle_nn::{Linear, Module, RmsNorm, VarBuilder, rms_norm};
use candle::Result;

pub struct Projector{
    merging_layer: Linear,
    norm: RmsNorm,
    linear_1: Linear,
    linear_2: Linear,
}

impl Projector {
    pub fn new(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let merging_layer = candle_nn::linear_no_bias(
            hidden_size * 4, 
            hidden_size, 
            vb.pp("patch_merger").pp("merging_layer")
        )?;

        let norm = rms_norm(hidden_size, 1e-5, vb.pp("norm"))?;
        let linear_1 = candle_nn::linear_no_bias(
            hidden_size, 
            hidden_size, 
            vb.pp("linear_1")
        )?;

        let linear_2 = candle_nn::linear_no_bias(
            hidden_size, 
            hidden_size, 
            vb.pp("linear_2")
        )?;

        Ok(Self { merging_layer, norm, linear_1, linear_2 })
    }

    
    pub fn forward(&self, x: &Tensor, ph: usize, pw: usize) -> Result<Tensor> {
        let hidden = x.dim(candle::D::Minus1)?;
        let x = self.norm.forward(x)?;

        let x = x.reshape((ph, pw, hidden))?;
        let x = x.permute((2, 0 ,1))?.unsqueeze(0)?;
        let x = x.reshape((1, hidden, ph/2, 2, pw/2, 2))?.permute((0,2,4,1,3,5))?
        .reshape((ph/2*pw/2, hidden*4))?;
        let x = self.merging_layer.forward(&x)?;
        let x = self.linear_1.forward(&x)?;
        let x = x.gelu()?;
        let x = self.linear_2.forward(&x)?;

        Ok(x)
    }
}