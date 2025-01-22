use candle::{Result, Tensor};

pub trait WithForward {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
    ) -> Result<Tensor>;
}

pub mod autoencoder;
pub mod model;
pub mod quantized_model;
pub mod sampling;
