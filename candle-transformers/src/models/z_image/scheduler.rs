//! FlowMatch Euler Discrete Scheduler for Z-Image
//!
//! Implements the flow matching scheduler used in Z-Image generation.

use candle::{Result, Tensor};

/// FlowMatchEulerDiscreteScheduler configuration
#[derive(Debug, Clone, serde::Deserialize)]
pub struct SchedulerConfig {
    #[serde(default = "default_num_train_timesteps")]
    pub num_train_timesteps: usize,
    #[serde(default = "default_shift")]
    pub shift: f64,
    #[serde(default)]
    pub use_dynamic_shifting: bool,
}

fn default_num_train_timesteps() -> usize {
    1000
}
fn default_shift() -> f64 {
    3.0
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            num_train_timesteps: default_num_train_timesteps(),
            shift: default_shift(),
            use_dynamic_shifting: false,
        }
    }
}

impl SchedulerConfig {
    /// Create configuration for Z-Image Turbo
    pub fn z_image_turbo() -> Self {
        Self {
            num_train_timesteps: 1000,
            shift: 3.0,
            use_dynamic_shifting: false,
        }
    }
}

/// FlowMatch Euler Discrete Scheduler
#[derive(Debug, Clone)]
pub struct FlowMatchEulerDiscreteScheduler {
    /// Configuration
    pub config: SchedulerConfig,
    /// Timesteps for inference
    pub timesteps: Vec<f64>,
    /// Sigma values
    pub sigmas: Vec<f64>,
    /// Minimum sigma
    pub sigma_min: f64,
    /// Maximum sigma
    pub sigma_max: f64,
    /// Current step index
    step_index: usize,
}

impl FlowMatchEulerDiscreteScheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        let num_train_timesteps = config.num_train_timesteps;
        let shift = config.shift;

        // Generate initial sigmas
        let timesteps: Vec<f64> = (1..=num_train_timesteps).rev().map(|t| t as f64).collect();

        let sigmas: Vec<f64> = timesteps
            .iter()
            .map(|&t| t / num_train_timesteps as f64)
            .collect();

        // Apply shift
        let sigmas: Vec<f64> = if !config.use_dynamic_shifting {
            sigmas
                .iter()
                .map(|&s| shift * s / (1.0 + (shift - 1.0) * s))
                .collect()
        } else {
            sigmas
        };

        let timesteps: Vec<f64> = sigmas
            .iter()
            .map(|&s| s * num_train_timesteps as f64)
            .collect();

        let sigma_max = sigmas[0];
        let sigma_min = *sigmas.last().unwrap_or(&0.0);

        Self {
            config,
            timesteps,
            sigmas,
            sigma_min,
            sigma_max,
            step_index: 0,
        }
    }

    /// Set timesteps for inference
    ///
    /// # Arguments
    /// * `num_inference_steps` - Number of denoising steps
    /// * `mu` - Optional time shift parameter (from calculate_shift)
    pub fn set_timesteps(&mut self, num_inference_steps: usize, mu: Option<f64>) {
        let sigma_max = self.sigmas[0];
        let sigma_min = *self.sigmas.last().unwrap_or(&0.0);

        // Linear interpolation to generate timesteps
        let timesteps: Vec<f64> = (0..num_inference_steps)
            .map(|i| {
                let t = i as f64 / num_inference_steps as f64;
                sigma_max * (1.0 - t) + sigma_min * t
            })
            .map(|s| s * self.config.num_train_timesteps as f64)
            .collect();

        let mut sigmas: Vec<f64> = timesteps
            .iter()
            .map(|&t| t / self.config.num_train_timesteps as f64)
            .collect();

        // Apply shift
        if let Some(mu) = mu {
            if self.config.use_dynamic_shifting {
                // time_shift: exp(mu) / (exp(mu) + (1/t - 1))
                sigmas = sigmas
                    .iter()
                    .map(|&t| {
                        if t <= 0.0 {
                            0.0
                        } else {
                            let e_mu = mu.exp();
                            e_mu / (e_mu + (1.0 / t - 1.0))
                        }
                    })
                    .collect();
            }
        } else if !self.config.use_dynamic_shifting {
            let shift = self.config.shift;
            sigmas = sigmas
                .iter()
                .map(|&s| shift * s / (1.0 + (shift - 1.0) * s))
                .collect();
        }

        // Add terminal sigma = 0
        sigmas.push(0.0);

        self.timesteps = timesteps;
        self.sigmas = sigmas;
        self.step_index = 0;
    }

    /// Get current sigma value
    pub fn current_sigma(&self) -> f64 {
        self.sigmas[self.step_index]
    }

    /// Get current timestep (for model input)
    /// Converts scheduler timestep to model input format: (1000 - t) / 1000
    pub fn current_timestep_normalized(&self) -> f64 {
        let t = self.timesteps.get(self.step_index).copied().unwrap_or(0.0);
        (1000.0 - t) / 1000.0
    }

    /// Euler step
    ///
    /// # Arguments
    /// * `model_output` - Model predicted velocity field
    /// * `sample` - Current sample x_t
    ///
    /// # Returns
    /// Next sample x_{t-1}
    pub fn step(&mut self, model_output: &Tensor, sample: &Tensor) -> Result<Tensor> {
        let sigma = self.sigmas[self.step_index];
        let sigma_next = self.sigmas[self.step_index + 1];

        let dt = sigma_next - sigma;

        // prev_sample = sample + dt * model_output
        let prev_sample = (sample + (model_output * dt)?)?;

        self.step_index += 1;
        Ok(prev_sample)
    }

    /// Reset scheduler state
    pub fn reset(&mut self) {
        self.step_index = 0;
    }

    /// Get number of inference steps
    pub fn num_inference_steps(&self) -> usize {
        self.timesteps.len()
    }

    /// Get current step index
    pub fn step_index(&self) -> usize {
        self.step_index
    }

    /// Check if denoising is complete
    pub fn is_complete(&self) -> bool {
        self.step_index >= self.timesteps.len()
    }
}

/// Calculate timestep shift parameter mu
///
/// # Arguments
/// * `image_seq_len` - Image sequence length (after patchify)
/// * `base_seq_len` - Base sequence length (typically 256)
/// * `max_seq_len` - Maximum sequence length (typically 4096)
/// * `base_shift` - Base shift value (typically 0.5)
/// * `max_shift` - Maximum shift value (typically 1.15)
pub fn calculate_shift(
    image_seq_len: usize,
    base_seq_len: usize,
    max_seq_len: usize,
    base_shift: f64,
    max_shift: f64,
) -> f64 {
    let m = (max_shift - base_shift) / (max_seq_len - base_seq_len) as f64;
    let b = base_shift - m * base_seq_len as f64;
    image_seq_len as f64 * m + b
}

/// Constants for shift calculation
pub const BASE_IMAGE_SEQ_LEN: usize = 256;
pub const MAX_IMAGE_SEQ_LEN: usize = 4096;
pub const BASE_SHIFT: f64 = 0.5;
pub const MAX_SHIFT: f64 = 1.15;
