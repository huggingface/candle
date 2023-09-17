use super::schedulers::{betas_for_alpha_bar, BetaSchedule, PredictionType};
use candle::{Result, Tensor};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DDPMVarianceType {
    FixedSmall,
    FixedSmallLog,
    FixedLarge,
    FixedLargeLog,
    Learned,
}

impl Default for DDPMVarianceType {
    fn default() -> Self {
        Self::FixedSmall
    }
}

#[derive(Debug, Clone)]
pub struct DDPMSchedulerConfig {
    /// The value of beta at the beginning of training.
    pub beta_start: f64,
    /// The value of beta at the end of training.
    pub beta_end: f64,
    /// How beta evolved during training.
    pub beta_schedule: BetaSchedule,
    /// Option to predicted sample between -1 and 1 for numerical stability.
    pub clip_sample: bool,
    /// Option to clip the variance used when adding noise to the denoised sample.
    pub variance_type: DDPMVarianceType,
    /// prediction type of the scheduler function
    pub prediction_type: PredictionType,
    /// number of diffusion steps used to train the model.
    pub train_timesteps: usize,
}

impl Default for DDPMSchedulerConfig {
    fn default() -> Self {
        Self {
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: BetaSchedule::ScaledLinear,
            clip_sample: false,
            variance_type: DDPMVarianceType::FixedSmall,
            prediction_type: PredictionType::Epsilon,
            train_timesteps: 1000,
        }
    }
}

pub struct DDPMScheduler {
    alphas_cumprod: Vec<f64>,
    init_noise_sigma: f64,
    timesteps: Vec<usize>,
    step_ratio: usize,
    pub config: DDPMSchedulerConfig,
}

impl DDPMScheduler {
    pub fn new(inference_steps: usize, config: DDPMSchedulerConfig) -> Result<Self> {
        let betas = match config.beta_schedule {
            BetaSchedule::ScaledLinear => super::utils::linspace(
                config.beta_start.sqrt(),
                config.beta_end.sqrt(),
                config.train_timesteps,
            )?
            .sqr()?,
            BetaSchedule::Linear => {
                super::utils::linspace(config.beta_start, config.beta_end, config.train_timesteps)?
            }
            BetaSchedule::SquaredcosCapV2 => betas_for_alpha_bar(config.train_timesteps, 0.999)?,
        };

        let betas = betas.to_vec1::<f64>()?;
        let mut alphas_cumprod = Vec::with_capacity(betas.len());
        for &beta in betas.iter() {
            let alpha = 1.0 - beta;
            alphas_cumprod.push(alpha * *alphas_cumprod.last().unwrap_or(&1f64))
        }

        // min(train_timesteps, inference_steps)
        // https://github.com/huggingface/diffusers/blob/8331da46837be40f96fbd24de6a6fb2da28acd11/src/diffusers/schedulers/scheduling_ddpm.py#L187
        let inference_steps = inference_steps.min(config.train_timesteps);
        // arange the number of the scheduler's timesteps
        let step_ratio = config.train_timesteps / inference_steps;
        let timesteps: Vec<usize> = (0..inference_steps).map(|s| s * step_ratio).rev().collect();

        Ok(Self {
            alphas_cumprod,
            init_noise_sigma: 1.0,
            timesteps,
            step_ratio,
            config,
        })
    }

    fn get_variance(&self, timestep: usize) -> f64 {
        let prev_t = timestep as isize - self.step_ratio as isize;
        let alpha_prod_t = self.alphas_cumprod[timestep];
        let alpha_prod_t_prev = if prev_t >= 0 {
            self.alphas_cumprod[prev_t as usize]
        } else {
            1.0
        };
        let current_beta_t = 1. - alpha_prod_t / alpha_prod_t_prev;

        // For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        // and sample from it to get previous sample
        // x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        let variance = (1. - alpha_prod_t_prev) / (1. - alpha_prod_t) * current_beta_t;

        // retrieve variance
        match self.config.variance_type {
            DDPMVarianceType::FixedSmall => variance.max(1e-20),
            // for rl-diffuser https://arxiv.org/abs/2205.09991
            DDPMVarianceType::FixedSmallLog => {
                let variance = variance.max(1e-20).ln();
                (variance * 0.5).exp()
            }
            DDPMVarianceType::FixedLarge => current_beta_t,
            DDPMVarianceType::FixedLargeLog => current_beta_t.ln(),
            DDPMVarianceType::Learned => variance,
        }
    }

    pub fn timesteps(&self) -> &[usize] {
        self.timesteps.as_slice()
    }

    ///  Ensures interchangeability with schedulers that need to scale the denoising model input
    /// depending on the current timestep.
    pub fn scale_model_input(&self, sample: Tensor, _timestep: usize) -> Tensor {
        sample
    }

    pub fn step(&self, model_output: &Tensor, timestep: usize, sample: &Tensor) -> Result<Tensor> {
        let prev_t = timestep as isize - self.step_ratio as isize;

        // https://github.com/huggingface/diffusers/blob/df2b548e893ccb8a888467c2508756680df22821/src/diffusers/schedulers/scheduling_ddpm.py#L272
        // 1. compute alphas, betas
        let alpha_prod_t = self.alphas_cumprod[timestep];
        let alpha_prod_t_prev = if prev_t >= 0 {
            self.alphas_cumprod[prev_t as usize]
        } else {
            1.0
        };
        let beta_prod_t = 1. - alpha_prod_t;
        let beta_prod_t_prev = 1. - alpha_prod_t_prev;
        let current_alpha_t = alpha_prod_t / alpha_prod_t_prev;
        let current_beta_t = 1. - current_alpha_t;

        // 2. compute predicted original sample from predicted noise also called "predicted x_0" of formula (15)
        let mut pred_original_sample = match self.config.prediction_type {
            PredictionType::Epsilon => {
                ((sample - model_output * beta_prod_t.sqrt())? / alpha_prod_t.sqrt())?
            }
            PredictionType::Sample => model_output.clone(),
            PredictionType::VPrediction => {
                ((sample * alpha_prod_t.sqrt())? - model_output * beta_prod_t.sqrt())?
            }
        };

        // 3. clip predicted x_0
        if self.config.clip_sample {
            pred_original_sample = pred_original_sample.clamp(-1f32, 1f32)?;
        }

        // 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        // See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        let pred_original_sample_coeff = (alpha_prod_t_prev.sqrt() * current_beta_t) / beta_prod_t;
        let current_sample_coeff = current_alpha_t.sqrt() * beta_prod_t_prev / beta_prod_t;

        // 5. Compute predicted previous sample µ_t
        // See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        let pred_prev_sample = ((&pred_original_sample * pred_original_sample_coeff)?
            + sample * current_sample_coeff)?;

        // https://github.com/huggingface/diffusers/blob/df2b548e893ccb8a888467c2508756680df22821/src/diffusers/schedulers/scheduling_ddpm.py#L305
        // 6. Add noise
        let mut variance = model_output.zeros_like()?;
        if timestep > 0 {
            let variance_noise = model_output.randn_like(0., 1.)?;
            if self.config.variance_type == DDPMVarianceType::FixedSmallLog {
                variance = (variance_noise * self.get_variance(timestep))?;
            } else {
                variance = (variance_noise * self.get_variance(timestep).sqrt())?;
            }
        }
        &pred_prev_sample + variance
    }

    pub fn add_noise(
        &self,
        original_samples: &Tensor,
        noise: Tensor,
        timestep: usize,
    ) -> Result<Tensor> {
        (original_samples * self.alphas_cumprod[timestep].sqrt())?
            + noise * (1. - self.alphas_cumprod[timestep]).sqrt()
    }

    pub fn init_noise_sigma(&self) -> f64 {
        self.init_noise_sigma
    }
}
