use candle::{Result, Tensor};

#[derive(Debug, Clone)]
pub struct DDPMWSchedulerConfig {
    scaler: f64,
    s: f64,
}

impl Default for DDPMWSchedulerConfig {
    fn default() -> Self {
        Self {
            scaler: 1f64,
            s: 0.008f64,
        }
    }
}

pub struct DDPMWScheduler {
    init_alpha_cumprod: f64,
    init_noise_sigma: f64,
    timesteps: Vec<f64>,
    pub config: DDPMWSchedulerConfig,
}

impl DDPMWScheduler {
    pub fn new(inference_steps: usize, config: DDPMWSchedulerConfig) -> Result<Self> {
        let init_alpha_cumprod = (config.s / (1. + config.s) * std::f64::consts::PI)
            .cos()
            .powi(2);
        let timesteps = (0..=inference_steps)
            .map(|i| 1. - i as f64 / inference_steps as f64)
            .collect::<Vec<_>>();
        Ok(Self {
            init_alpha_cumprod,
            init_noise_sigma: 1.0,
            timesteps,
            config,
        })
    }

    pub fn timesteps(&self) -> &[f64] {
        &self.timesteps
    }

    fn alpha_cumprod(&self, t: f64) -> f64 {
        let scaler = self.config.scaler;
        let s = self.config.s;
        let t = if scaler > 1. {
            1. - (1. - t).powf(scaler)
        } else if scaler < 1. {
            t.powf(scaler)
        } else {
            t
        };
        let alpha_cumprod = ((t + s) / (1. + s) * std::f64::consts::PI * 0.5)
            .cos()
            .powi(2)
            / self.init_alpha_cumprod;
        alpha_cumprod.clamp(0.0001, 0.9999)
    }

    fn previous_timestep(&self, ts: f64) -> f64 {
        let index = self
            .timesteps
            .iter()
            .enumerate()
            .map(|(idx, v)| (idx, (v - ts).abs()))
            .min_by(|x, y| x.1.total_cmp(&y.1))
            .unwrap()
            .0;
        self.timesteps[index + 1]
    }

    ///  Ensures interchangeability with schedulers that need to scale the denoising model input
    /// depending on the current timestep.
    pub fn scale_model_input(&self, sample: Tensor, _timestep: usize) -> Tensor {
        sample
    }

    pub fn step(&self, model_output: &Tensor, ts: f64, sample: &Tensor) -> Result<Tensor> {
        let prev_t = self.previous_timestep(ts);

        let alpha_cumprod = self.alpha_cumprod(ts);
        let alpha_cumprod_prev = self.alpha_cumprod(prev_t);
        let alpha = alpha_cumprod / alpha_cumprod_prev;

        let mu = (sample - model_output * ((1. - alpha) / (1. - alpha_cumprod).sqrt()))?;
        let mu = (mu * (1. / alpha).sqrt())?;

        let std_noise = mu.randn_like(0., 1.)?;
        let std =
            std_noise * ((1. - alpha) * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)).sqrt();
        if prev_t == 0. {
            Ok(mu)
        } else {
            mu + std
        }
    }

    pub fn init_noise_sigma(&self) -> f64 {
        self.init_noise_sigma
    }
}
