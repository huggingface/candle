//! Ancestral sampling with Euler method steps.
//!
//! Reference implementation in Rust:
//!
//! https://github.com/pykeio/diffusers/blob/250b9ad1898af41e76a74c0d8d4292652823338a/src/schedulers/euler_ancestral_discrete.rs
//!
//! Based on the original [`k-diffusion` implementation by Katherine Crowson][kd].
///
/// [kd]: https://github.com/crowsonkb/k-diffusion/blob/481677d114f6ea445aa009cf5bd7a9cdee909e47/k_diffusion/sampling.py#L72
use super::{
    schedulers::{
        betas_for_alpha_bar, BetaSchedule, PredictionType, Scheduler, SchedulerConfig,
        TimestepSpacing,
    },
    utils::interp,
};
use candle::{bail, Error, Result, Tensor};

/// The configuration for the EulerAncestral Discrete scheduler.
#[derive(Debug, Clone, Copy)]
pub struct EulerAncestralDiscreteSchedulerConfig {
    /// The value of beta at the beginning of training.n
    pub beta_start: f64,
    /// The value of beta at the end of training.
    pub beta_end: f64,
    /// How beta evolved during training.
    pub beta_schedule: BetaSchedule,
    /// Adjust the indexes of the inference schedule by this value.
    pub steps_offset: usize,
    /// prediction type of the scheduler function, one of `epsilon` (predicting
    /// the noise of the diffusion process), `sample` (directly predicting the noisy sample`)
    /// or `v_prediction` (see section 2.4 https://imagen.research.google/video/paper.pdf)
    pub prediction_type: PredictionType,
    /// number of diffusion steps used to train the model
    pub train_timesteps: usize,
    /// time step spacing for the diffusion process
    pub timestep_spacing: TimestepSpacing,
}

impl Default for EulerAncestralDiscreteSchedulerConfig {
    fn default() -> Self {
        Self {
            beta_start: 0.00085f64,
            beta_end: 0.012f64,
            beta_schedule: BetaSchedule::ScaledLinear,
            steps_offset: 1,
            prediction_type: PredictionType::Epsilon,
            train_timesteps: 1000,
            timestep_spacing: TimestepSpacing::Leading,
        }
    }
}

impl SchedulerConfig for EulerAncestralDiscreteSchedulerConfig {
    fn build(&self, inference_steps: usize) -> Result<Box<dyn Scheduler>> {
        Ok(Box::new(EulerAncestralDiscreteScheduler::new(
            inference_steps,
            *self,
        )?))
    }
}

/// The EulerAncestral Discrete scheduler.
#[derive(Debug, Clone)]
pub struct EulerAncestralDiscreteScheduler {
    timesteps: Vec<usize>,
    sigmas: Vec<f64>,
    init_noise_sigma: f64,
    pub config: EulerAncestralDiscreteSchedulerConfig,
}

// clip_sample: False, set_alpha_to_one: False
impl EulerAncestralDiscreteScheduler {
    /// Creates a new EulerAncestral Discrete scheduler given the number of steps to be
    /// used for inference as well as the number of steps that was used
    /// during training.
    pub fn new(
        inference_steps: usize,
        config: EulerAncestralDiscreteSchedulerConfig,
    ) -> Result<Self> {
        let step_ratio = config.train_timesteps / inference_steps;
        let timesteps: Vec<usize> = match config.timestep_spacing {
            TimestepSpacing::Leading => (0..(inference_steps))
                .map(|s| s * step_ratio + config.steps_offset)
                .rev()
                .collect(),
            TimestepSpacing::Trailing => std::iter::successors(Some(config.train_timesteps), |n| {
                if *n > step_ratio {
                    Some(n - step_ratio)
                } else {
                    None
                }
            })
            .map(|n| n - 1)
            .collect(),
            TimestepSpacing::Linspace => {
                super::utils::linspace(0.0, (config.train_timesteps - 1) as f64, inference_steps)?
                    .to_vec1::<f64>()?
                    .iter()
                    .map(|&f| f as usize)
                    .rev()
                    .collect()
            }
        };

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
        let sigmas: Vec<f64> = alphas_cumprod
            .iter()
            .map(|&f| ((1. - f) / f).sqrt())
            .collect();

        let sigmas_xa: Vec<_> = (0..sigmas.len()).map(|i| i as f64).collect();

        let mut sigmas_int = interp(
            &timesteps.iter().map(|&t| t as f64).collect::<Vec<_>>(),
            &sigmas_xa,
            &sigmas,
        );
        sigmas_int.push(0.0);

        // standard deviation of the initial noise distribution
        // f64 does not implement Ord such that there is no `max`, so we need to use this workaround
        let init_noise_sigma = *sigmas_int
            .iter()
            .chain(std::iter::once(&0.0))
            .reduce(|a, b| if a > b { a } else { b })
            .expect("init_noise_sigma could not be reduced from sigmas - this should never happen");

        Ok(Self {
            sigmas: sigmas_int,
            timesteps,
            init_noise_sigma,
            config,
        })
    }
}

impl Scheduler for EulerAncestralDiscreteScheduler {
    fn timesteps(&self) -> &[usize] {
        self.timesteps.as_slice()
    }

    /// Ensures interchangeability with schedulers that need to scale the denoising model input
    /// depending on the current timestep.
    ///
    /// Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the K-LMS algorithm
    fn scale_model_input(&self, sample: Tensor, timestep: usize) -> Result<Tensor> {
        let step_index = match self.timesteps.iter().position(|&t| t == timestep) {
            Some(i) => i,
            None => bail!("timestep out of this schedulers bounds: {timestep}"),
        };

        let sigma = self
            .sigmas
            .get(step_index)
            .expect("step_index out of sigma bounds - this shouldn't happen");

        sample / ((sigma.powi(2) + 1.).sqrt())
    }

    /// Performs a backward step during inference.
    fn step(&self, model_output: &Tensor, timestep: usize, sample: &Tensor) -> Result<Tensor> {
        let step_index = self
            .timesteps
            .iter()
            .position(|&p| p == timestep)
            .ok_or_else(|| Error::Msg("timestep out of this schedulers bounds".to_string()))?;

        let sigma_from = &self.sigmas[step_index];
        let sigma_to = &self.sigmas[step_index + 1];

        // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        let pred_original_sample = match self.config.prediction_type {
            PredictionType::Epsilon => (sample - (model_output * *sigma_from))?,
            PredictionType::VPrediction => {
                ((model_output * (-sigma_from / (sigma_from.powi(2) + 1.0).sqrt()))?
                    + (sample / (sigma_from.powi(2) + 1.0))?)?
            }
            PredictionType::Sample => bail!("prediction_type not implemented yet: sample"),
        };

        let sigma_up = (sigma_to.powi(2) * (sigma_from.powi(2) - sigma_to.powi(2))
            / sigma_from.powi(2))
        .sqrt();
        let sigma_down = (sigma_to.powi(2) - sigma_up.powi(2)).sqrt();

        // 2. convert to a ODE derivative
        let derivative = ((sample - pred_original_sample)? / *sigma_from)?;
        let dt = sigma_down - *sigma_from;
        let prev_sample = (sample + derivative * dt)?;

        let noise = prev_sample.randn_like(0.0, 1.0)?;

        prev_sample + noise * sigma_up
    }

    fn add_noise(&self, original: &Tensor, noise: Tensor, timestep: usize) -> Result<Tensor> {
        let step_index = self
            .timesteps
            .iter()
            .position(|&p| p == timestep)
            .ok_or_else(|| Error::Msg("timestep out of this schedulers bounds".to_string()))?;

        let sigma = self
            .sigmas
            .get(step_index)
            .expect("step_index out of sigma bounds - this shouldn't happen");

        original + (noise * *sigma)?
    }

    fn init_noise_sigma(&self) -> f64 {
        match self.config.timestep_spacing {
            TimestepSpacing::Trailing | TimestepSpacing::Linspace => self.init_noise_sigma,
            TimestepSpacing::Leading => (self.init_noise_sigma.powi(2) + 1.0).sqrt(),
        }
    }
}
