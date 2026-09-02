use anyhow::{Ok, Result};
use candle::{DType, IndexOp, Tensor};

use candle_transformers::models::flux;
use candle_transformers::models::mmdit::model::MMDiT;

pub struct SkipLayerGuidanceConfig {
    pub scale: f64,
    pub start: f64,
    pub end: f64,
    pub layers: Vec<usize>,
}

#[allow(clippy::too_many_arguments)]
pub fn euler_sample(
    mmdit: &MMDiT,
    y: &Tensor,
    context: &Tensor,
    num_inference_steps: usize,
    cfg_scale: f64,
    time_shift: f64,
    height: usize,
    width: usize,
    slg_config: Option<SkipLayerGuidanceConfig>,
) -> Result<Tensor> {
    let mut x = flux::sampling::get_noise(1, height, width, y.device())?.to_dtype(DType::F16)?;
    let sigmas = (0..=num_inference_steps)
        .map(|x| x as f64 / num_inference_steps as f64)
        .rev()
        .map(|x| time_snr_shift(time_shift, x))
        .collect::<Vec<f64>>();

    for (step, window) in sigmas.windows(2).enumerate() {
        let (s_curr, s_prev) = match window {
            [a, b] => (a, b),
            _ => continue,
        };

        let timestep = (*s_curr) * 1000.0;
        let noise_pred = mmdit.forward(
            &Tensor::cat(&[&x, &x], 0)?,
            &Tensor::full(timestep as f32, (2,), x.device())?.contiguous()?,
            y,
            context,
            None,
        )?;

        let mut guidance = apply_cfg(cfg_scale, &noise_pred)?;

        if let Some(slg_config) = slg_config.as_ref() {
            if (num_inference_steps as f64) * slg_config.start < (step as f64)
                && (step as f64) < (num_inference_steps as f64) * slg_config.end
            {
                let slg_noise_pred = mmdit.forward(
                    &x,
                    &Tensor::full(timestep as f32, (1,), x.device())?.contiguous()?,
                    &y.i(..1)?,
                    &context.i(..1)?,
                    Some(&slg_config.layers),
                )?;
                guidance = (guidance
                    + (slg_config.scale * (noise_pred.i(..1)? - slg_noise_pred.i(..1))?)?)?;
            }
        }

        x = (x + (guidance * (*s_prev - *s_curr))?)?;
    }
    Ok(x)
}

// The "Resolution-dependent shifting of timestep schedules" recommended in the SD3 tech report paper
// https://arxiv.org/pdf/2403.03206
// Following the implementation in ComfyUI:
// https://github.com/comfyanonymous/ComfyUI/blob/3c60ecd7a83da43d694e26a77ca6b93106891251/
// comfy/model_sampling.py#L181
fn time_snr_shift(alpha: f64, t: f64) -> f64 {
    alpha * t / (1.0 + (alpha - 1.0) * t)
}

fn apply_cfg(cfg_scale: f64, noise_pred: &Tensor) -> Result<Tensor> {
    Ok(((cfg_scale * noise_pred.narrow(0, 0, 1)?)?
        - ((cfg_scale - 1.0) * noise_pred.narrow(0, 1, 1)?)?)?)
}
