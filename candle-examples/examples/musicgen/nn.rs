use anyhow::Result;
use candle::Tensor;

const MAX_SEQ_LEN: usize = 5000;

pub type VarBuilder<'a> = candle_nn::VarBuilder<'a>;
pub type Linear = candle_nn::Linear;

pub fn linear(size1: usize, size2: usize, bias: bool, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((size2, size1), "weight")?;
    let bias = if bias {
        Some(vb.get(size2, "bias")?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias))
}

pub type LayerNorm = candle_nn::LayerNorm;

pub fn layer_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<LayerNorm> {
    let (weight, bias) = match (vb.get(size, "weight"), vb.get(size, "bias")) {
        (Ok(weight), Ok(bias)) => (weight, bias),
        (Err(err), _) | (_, Err(err)) => {
            if let (Ok(weight), Ok(bias)) = (vb.get(size, "gamma"), vb.get(size, "beta")) {
                (weight, bias)
            } else {
                return Err(err.into());
            }
        }
    };
    Ok(LayerNorm::new(weight, bias, eps))
}

#[derive(Debug)]
pub struct Dropout {
    pr: f64,
}

impl Dropout {
    pub fn new(pr: f64) -> Self {
        Self { pr }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // TODO
        Ok(x.clone())
    }
}

pub type Embedding = candle_nn::Embedding;

pub fn embedding(vocab_size: usize, hidden_size: usize, vb: VarBuilder) -> Result<Embedding> {
    let embeddings = vb.get((vocab_size, hidden_size), "weight")?;
    Ok(Embedding::new(embeddings, hidden_size))
}

pub type Conv1d = candle_nn::Conv1d;
pub type Conv1dConfig = candle_nn::Conv1dConfig;

// Applies weight norm for inference by recomputing the weight tensor. This
// does not apply to training.
// https://pytorch.org/docs/stable/generated/torch.nn.utils.weight_norm.html
pub fn conv1d_weight_norm(
    in_c: usize,
    out_c: usize,
    kernel_size: usize,
    config: Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let weight_g = vb.get((out_c, 1, 1), "weight_g")?;
    let weight_v = vb.get((out_c, in_c, kernel_size), "weight_v")?;
    let norm_v = weight_v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    let weight = weight_v.broadcast_mul(&weight_g)?.broadcast_div(&norm_v)?;
    let bias = vb.get(out_c, "bias")?;
    Ok(Conv1d::new(weight, Some(bias), config))
}

pub fn conv1d(
    in_c: usize,
    out_c: usize,
    kernel_size: usize,
    config: Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let weight = vb.get((out_c, in_c, kernel_size), "weight")?;
    let bias = vb.get(out_c, "bias")?;
    Ok(Conv1d::new(weight, Some(bias), config))
}

pub type HiddenAct = candle_nn::Activation;
