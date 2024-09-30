use candle::{Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

#[derive(Debug, Clone)]
pub struct Config {
    pub projector_hidden_act: candle_nn::Activation,
    pub text_config_hidden_size: usize,
    pub vision_config_hidden_size: usize,
}

#[derive(Debug, Clone)]
pub struct MultiModalProjector {
    linear_1: Linear,
    act: candle_nn::Activation,
    linear_2: Linear,
}

impl MultiModalProjector {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let (hidden_v, hidden_t) = (cfg.vision_config_hidden_size, cfg.text_config_hidden_size);
        let linear_1 = linear(hidden_v, hidden_t, vb.pp("linear_1"))?;
        let linear_2 = linear(hidden_t, hidden_t, vb.pp("linear_2"))?;
        Ok(Self {
            linear_1,
            act: cfg.projector_hidden_act,
            linear_2,
        })
    }
}

impl Module for MultiModalProjector {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.linear_1)?
            .apply(&self.act)?
            .apply(&self.linear_2)
    }
}
