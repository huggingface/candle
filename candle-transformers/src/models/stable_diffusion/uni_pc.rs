//! # UniPC Scheduler
//!
//! UniPC is a training-free framework designed for the fast sampling of diffusion models, which consists of a
//! corrector (UniC) and a predictor (UniP) that share a unified analytical form and support arbitrary orders.
//!
//! UniPC is by design model-agnostic, supporting pixel-space/latent-space DPMs on unconditional/conditional
//! sampling. It can also be applied to both noise prediction and data prediction models. Compared with prior
//! methods, UniPC converges faster thanks to the increased order of accuracy. Both quantitative and qualitative
//! results show UniPC can improve sampling quality, especially at very low step counts (5~10).
//!
//! For more information, see the original publication:
//! UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models, W. Zhao et al, 2023.
//! https://arxiv.org/abs/2302.04867
//!
//! This work is based largely on UniPC implementation from the diffusers python package:
//! https://raw.githubusercontent.com/huggingface/diffusers/e8aacda762e311505ba05ae340af23b149e37af3/src/diffusers/schedulers/scheduling_unipc_multistep.py
use std::collections::HashSet;
use std::ops::Neg;

use super::schedulers::PredictionType;
use super::{
    schedulers::{Scheduler, SchedulerConfig},
    utils::{interp, linspace},
};
use candle::{Error, IndexOp, Result, Tensor};

#[derive(Debug, Clone, Copy)]
pub enum SigmaSchedule {
    Karras(KarrasSigmaSchedule),
    Exponential(ExponentialSigmaSchedule),
}

impl SigmaSchedule {
    fn sigma_t(&self, t: f64) -> f64 {
        match self {
            Self::Karras(x) => x.sigma_t(t),
            Self::Exponential(x) => x.sigma_t(t),
        }
    }
}

impl Default for SigmaSchedule {
    fn default() -> Self {
        Self::Karras(KarrasSigmaSchedule::default())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct KarrasSigmaSchedule {
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub rho: f64,
}

impl KarrasSigmaSchedule {
    fn sigma_t(&self, t: f64) -> f64 {
        let (min_inv_rho, max_inv_rho) = (
            self.sigma_min.powf(1.0 / self.rho),
            self.sigma_max.powf(1.0 / self.rho),
        );

        (max_inv_rho + ((1.0 - t) * (min_inv_rho - max_inv_rho))).powf(self.rho)
    }
}

impl Default for KarrasSigmaSchedule {
    fn default() -> Self {
        Self {
            sigma_max: 10.0,
            sigma_min: 0.1,
            rho: 4.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ExponentialSigmaSchedule {
    sigma_min: f64,
    sigma_max: f64,
}

impl ExponentialSigmaSchedule {
    fn sigma_t(&self, t: f64) -> f64 {
        (t * (self.sigma_max.ln() - self.sigma_min.ln()) + self.sigma_min.ln()).exp()
    }
}

impl Default for ExponentialSigmaSchedule {
    fn default() -> Self {
        Self {
            sigma_max: 80.0,
            sigma_min: 0.1,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub enum SolverType {
    #[default]
    Bh1,
    Bh2,
}

#[derive(Debug, Default, Clone, Copy)]
pub enum AlgorithmType {
    #[default]
    DpmSolverPlusPlus,
    SdeDpmSolverPlusPlus,
}

#[derive(Debug, Default, Clone, Copy)]
pub enum FinalSigmasType {
    #[default]
    Zero,
    SigmaMin,
}

#[derive(Debug, Clone)]
pub enum TimestepSchedule {
    /// Timesteps will be determined by interpolation of sigmas
    FromSigmas,
    /// Timesteps will be separated by regular intervals
    Linspace,
}

impl TimestepSchedule {
    fn timesteps(
        &self,
        sigma_schedule: &SigmaSchedule,
        num_inference_steps: usize,
        num_training_steps: usize,
    ) -> Result<Vec<usize>> {
        match self {
            Self::FromSigmas => {
                let sigmas: Tensor = linspace(1., 0., num_inference_steps)?
                    .to_vec1()?
                    .into_iter()
                    .map(|t| sigma_schedule.sigma_t(t))
                    .collect::<Vec<f64>>()
                    .try_into()?;
                let log_sigmas = sigmas.log()?.to_vec1::<f64>()?;
                let timesteps = interp(
                    &log_sigmas.iter().copied().rev().collect::<Vec<_>>(),
                    &linspace(
                        log_sigmas[log_sigmas.len() - 1] - 0.001,
                        log_sigmas[0] + 0.001,
                        num_inference_steps,
                    )?
                    .to_vec1::<f64>()?,
                    &linspace(0., num_training_steps as f64, num_inference_steps)?
                        .to_vec1::<f64>()?,
                )
                .into_iter()
                .map(|f| (num_training_steps - 1) - (f as usize))
                .collect::<Vec<_>>();

                Ok(timesteps)
            }

            Self::Linspace => {
                Ok(
                    linspace((num_training_steps - 1) as f64, 0., num_inference_steps)?
                        .to_vec1::<f64>()?
                        .into_iter()
                        .map(|f| f as usize)
                        .collect(),
                )
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum CorrectorConfiguration {
    Disabled,
    Enabled { skip_steps: HashSet<usize> },
}

impl Default for CorrectorConfiguration {
    fn default() -> Self {
        Self::Enabled {
            skip_steps: [0, 1, 2].into_iter().collect(),
        }
    }
}

impl CorrectorConfiguration {
    pub fn new(disabled_steps: impl IntoIterator<Item = usize>) -> Self {
        Self::Enabled {
            skip_steps: disabled_steps.into_iter().collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct UniPCSchedulerConfig {
    /// Configure the UNIC corrector. By default it is disabled
    pub corrector: CorrectorConfiguration,
    /// Determines how sigma relates to a given timestep
    pub sigma_schedule: SigmaSchedule,
    /// Determines the points
    pub timestep_schedule: TimestepSchedule,
    /// The solver order which can be `1` or higher. It is recommended to use `solver_order=2` for guided
    /// sampling, and `solver_order=3` for unconditional sampling.
    pub solver_order: usize,
    /// Prediction type of the scheduler function
    pub prediction_type: PredictionType,
    pub num_training_timesteps: usize,
    /// Whether to use the "dynamic thresholding" method. This is unsuitable for latent-space diffusion models such
    /// as Stable Diffusion.
    pub thresholding: bool,
    /// The ratio for the dynamic thresholding method. Valid only when `thresholding=True`.
    pub dynamic_thresholding_ratio: f64,
    /// The threshold value for dynamic thresholding.
    pub sample_max_value: f64,
    pub solver_type: SolverType,
    /// Whether to use lower-order solvers in the final steps.
    pub lower_order_final: bool,
}

impl Default for UniPCSchedulerConfig {
    fn default() -> Self {
        Self {
            corrector: Default::default(),
            timestep_schedule: TimestepSchedule::FromSigmas,
            sigma_schedule: SigmaSchedule::Karras(Default::default()),
            prediction_type: PredictionType::Epsilon,
            num_training_timesteps: 1000,
            solver_order: 2,
            thresholding: false,
            dynamic_thresholding_ratio: 0.995,
            sample_max_value: 1.0,
            solver_type: SolverType::Bh1,
            lower_order_final: true,
        }
    }
}

impl SchedulerConfig for UniPCSchedulerConfig {
    fn build(&self, inference_steps: usize) -> Result<Box<dyn Scheduler>> {
        Ok(Box::new(EdmDpmMultistepScheduler::new(
            self.clone(),
            inference_steps,
        )?))
    }
}

struct State {
    model_outputs: Vec<Option<Tensor>>,
    lower_order_nums: usize,
    order: usize,
    last_sample: Option<Tensor>,
}

impl State {
    fn new(solver_order: usize) -> Self {
        Self {
            model_outputs: vec![None; solver_order],
            lower_order_nums: 0,
            order: 0,
            last_sample: None,
        }
    }

    fn lower_order_nums(&self) -> usize {
        self.lower_order_nums
    }

    fn update_lower_order_nums(&mut self, n: usize) {
        self.lower_order_nums = n;
    }

    fn model_outputs(&self) -> &[Option<Tensor>] {
        self.model_outputs.as_slice()
    }

    fn update_model_output(&mut self, idx: usize, output: Option<Tensor>) {
        self.model_outputs[idx] = output;
    }

    fn last_sample(&self) -> Option<&Tensor> {
        self.last_sample.as_ref()
    }

    fn update_last_sample(&mut self, sample: Tensor) {
        let _ = self.last_sample.replace(sample);
    }

    fn order(&self) -> usize {
        self.order
    }

    fn update_order(&mut self, order: usize) {
        self.order = order;
    }
}

pub struct EdmDpmMultistepScheduler {
    schedule: Schedule,
    config: UniPCSchedulerConfig,
    state: State,
}

impl EdmDpmMultistepScheduler {
    pub fn new(config: UniPCSchedulerConfig, num_inference_steps: usize) -> Result<Self> {
        let schedule = Schedule::new(
            config.timestep_schedule.clone(),
            config.sigma_schedule,
            num_inference_steps,
            config.num_training_timesteps,
        )?;

        Ok(Self {
            schedule,
            state: State::new(config.solver_order),
            config,
        })
    }

    fn step_index(&self, timestep: usize) -> usize {
        let index_candidates = self
            .schedule
            .timesteps()
            .iter()
            .enumerate()
            .filter(|(_, t)| (*t == &timestep))
            .map(|(i, _)| i)
            .collect::<Vec<_>>();

        match index_candidates.len() {
            0 => 0,
            1 => index_candidates[0],
            _ => index_candidates[1],
        }
    }

    fn timestep(&self, step_idx: usize) -> usize {
        self.schedule
            .timesteps()
            .get(step_idx)
            .copied()
            .unwrap_or(0)
    }

    fn convert_model_output(
        &self,
        model_output: &Tensor,
        sample: &Tensor,
        timestep: usize,
    ) -> Result<Tensor> {
        let (alpha_t, sigma_t) = (
            self.schedule.alpha_t(timestep),
            self.schedule.sigma_t(timestep),
        );

        let x0_pred = match self.config.prediction_type {
            PredictionType::Epsilon => ((sample - (model_output * sigma_t))? / alpha_t)?,
            PredictionType::Sample => model_output.clone(),
            PredictionType::VPrediction => ((alpha_t * sample)? - (sigma_t * model_output)?)?,
        };

        if self.config.thresholding {
            self.threshold_sample(x0_pred)
        } else {
            Ok(x0_pred)
        }
    }

    fn threshold_sample(&self, sample: Tensor) -> Result<Tensor> {
        let shape = sample.shape().clone().into_dims();
        let v = sample
            .abs()?
            .reshape((shape[0], shape[1] * shape[2..].iter().product::<usize>()))?
            .to_dtype(candle::DType::F64)?
            .to_vec2::<f64>()?;
        let q = stats::Quantile::new(self.config.dynamic_thresholding_ratio)
            .with_samples(v.into_iter().flatten());
        let (threshold, max) = (q.quantile().max(self.config.sample_max_value), q.max());

        sample.clamp(-threshold, threshold)? / (threshold / max).sqrt().min(1.)
    }

    fn multistep_uni_p_bh_update(&self, sample: &Tensor, timestep: usize) -> Result<Tensor> {
        let step_index = self.step_index(timestep);
        let ns = &self.schedule;
        let model_outputs = self.state.model_outputs();
        let Some(m0) = &model_outputs[model_outputs.len() - 1] else {
            return Err(Error::Msg(
                "Expected model output for predictor update".to_string(),
            ));
        };

        let (t0, tt) = (timestep, self.timestep(self.step_index(timestep) + 1));
        let (sigma_t, sigma_s0) = (ns.sigma_t(tt), ns.sigma_t(t0));
        let (alpha_t, _alpha_s0) = (ns.alpha_t(tt), ns.alpha_t(t0));
        let (lambda_t, lambda_s0) = (ns.lambda_t(tt), ns.lambda_t(t0));

        let h = lambda_t - lambda_s0;
        let device = sample.device();

        let (mut rks, mut d1s) = (vec![], vec![]);
        for i in 1..self.state.order() {
            let ti = self.timestep(step_index.saturating_sub(i + 1));
            let Some(mi) = model_outputs
                .get(model_outputs.len().saturating_sub(i + 1))
                .into_iter()
                .flatten()
                .next()
            else {
                return Err(Error::Msg(
                    "Expected model output for predictor update".to_string(),
                ));
            };
            let (alpha_si, sigma_si) = (ns.alpha_t(ti), ns.sigma_t(ti));
            let lambda_si = alpha_si.ln() - sigma_si.ln();
            let rk = (lambda_si - lambda_s0) / h;
            rks.push(rk);
            d1s.push(((mi - m0)? / rk)?);
        }
        rks.push(1.0);
        let rks = Tensor::new(rks, device)?;
        let (mut r, mut b) = (vec![], vec![]);

        let hh = h.neg();
        let h_phi_1 = hh.exp_m1();
        let mut h_phi_k = h_phi_1 / hh - 1.;
        let mut factorial_i = 1.;

        let b_h = match self.config.solver_type {
            SolverType::Bh1 => hh,
            SolverType::Bh2 => hh.exp_m1(),
        };

        for i in 1..self.state.order() + 1 {
            r.push(rks.powf(i as f64 - 1.)?);
            b.push(h_phi_k * factorial_i / b_h);
            factorial_i = i as f64 + 1.;
            h_phi_k = h_phi_k / hh - 1. / factorial_i;
        }

        let (r, b) = (Tensor::stack(&r, 0)?, Tensor::new(b, device)?);
        let (d1s, rhos_p) = match d1s.len() {
            0 => (None, None),
            _ => {
                let rhos_p = match self.state.order() {
                    2 => Tensor::new(&[0.5f64], m0.device())?.to_dtype(m0.dtype())?,
                    _ => {
                        let ((r1, r2), b1) = (r.dims2()?, b.dims1()?);
                        let inverse = linalg::inverse(&r.i((..(r1 - 1), ..(r2 - 1)))?)?;
                        let b = b.i(..(b1 - 1))?;
                        b.broadcast_mul(&inverse)?.sum(1)?.to_dtype(m0.dtype())?
                    }
                };

                (Some(Tensor::stack(&d1s, 1)?), Some(rhos_p))
            }
        };

        let x_t_ = ((sigma_t / sigma_s0 * sample)? - (alpha_t * h_phi_1 * m0)?)?;
        if let (Some(d1s), Some(rhos_p)) = (d1s, rhos_p) {
            use linalg::{Permutation, TensordotFixedPosition, TensordotGeneral};
            let output_shape = m0.shape().clone();
            let pred_res = TensordotGeneral {
                lhs_permutation: Permutation { dims: vec![0] },
                rhs_permutation: Permutation {
                    dims: vec![1, 0, 2, 3, 4],
                },
                tensordot_fixed_position: TensordotFixedPosition {
                    len_uncontracted_lhs: 1,
                    len_uncontracted_rhs: output_shape.dims().iter().product::<usize>(),
                    len_contracted_axes: d1s.dim(1)?,
                    output_shape,
                },
                output_permutation: Permutation {
                    dims: vec![0, 1, 2, 3],
                },
            }
            .eval(&rhos_p, &d1s)?;
            x_t_ - (alpha_t * b_h * pred_res)?
        } else {
            Ok(x_t_)
        }
    }

    fn multistep_uni_c_bh_update(
        &self,
        model_output: &Tensor,
        model_outputs: &[Option<Tensor>],
        last_sample: &Tensor,
        sample: &Tensor,
        timestep: usize,
    ) -> Result<Tensor> {
        let step_index = self.step_index(timestep);
        let Some(m0) = model_outputs.last().into_iter().flatten().next() else {
            return Err(Error::Msg(
                "Expected model output for corrector update".to_string(),
            ));
        };
        let model_t = model_output;
        let (x, _xt) = (last_sample, sample);

        let (t0, tt, ns) = (
            self.timestep(self.step_index(timestep) - 1),
            timestep,
            &self.schedule,
        );
        let (sigma_t, sigma_s0) = (ns.sigma_t(tt), ns.sigma_t(t0));
        let (alpha_t, _alpha_s0) = (ns.alpha_t(tt), ns.alpha_t(t0));
        let (lambda_t, lambda_s0) = (ns.lambda_t(tt), ns.lambda_t(t0));

        let h = lambda_t - lambda_s0;
        let device = sample.device();

        let (mut rks, mut d1s) = (vec![], vec![]);
        for i in 1..self.state.order() {
            let ti = self.timestep(step_index.saturating_sub(i + 1));
            let Some(mi) = model_outputs
                .get(model_outputs.len().saturating_sub(i + 1))
                .into_iter()
                .flatten()
                .next()
            else {
                return Err(Error::Msg(
                    "Expected model output for corrector update".to_string(),
                ));
            };
            let (alpha_si, sigma_si) = (ns.alpha_t(ti), ns.sigma_t(ti));
            let lambda_si = alpha_si.ln() - sigma_si.ln();
            let rk = (lambda_si - lambda_s0) / h;
            rks.push(rk);
            d1s.push(((mi - m0)? / rk)?);
        }
        rks.push(1.0);
        let rks = Tensor::new(rks, device)?;
        let (mut r, mut b) = (vec![], vec![]);

        let hh = h.neg();
        let h_phi_1 = hh.exp_m1();
        let mut h_phi_k = h_phi_1 / hh - 1.;
        let mut factorial_i = 1.;

        let b_h = match self.config.solver_type {
            SolverType::Bh1 => hh,
            SolverType::Bh2 => hh.exp_m1(),
        };

        for i in 1..self.state.order() + 1 {
            r.push(rks.powf(i as f64 - 1.)?);
            b.push(h_phi_k * factorial_i / b_h);
            factorial_i = i as f64 + 1.;
            h_phi_k = h_phi_k / hh - 1. / factorial_i;
        }

        let (r, b) = (Tensor::stack(&r, 0)?, Tensor::new(b, device)?);
        let d1s = match d1s.len() {
            0 => None,
            _ => Some(Tensor::stack(&d1s, 1)?),
        };
        let rhos_c = match self.state.order() {
            1 => Tensor::new(&[0.5f64], m0.device())?.to_dtype(m0.dtype())?,
            _ => {
                let inverse = linalg::inverse(&r)?;
                b.broadcast_mul(&inverse)?.sum(1)?.to_dtype(m0.dtype())?
            }
        };

        let x_t_ = ((sigma_t / sigma_s0 * x)? - (alpha_t * h_phi_1 * m0)?)?;
        let corr_res = d1s
            .map(|d1s| {
                use linalg::{Permutation, TensordotFixedPosition, TensordotGeneral};
                let output_shape = x_t_.shape().clone();
                TensordotGeneral {
                    lhs_permutation: Permutation { dims: vec![0] },
                    rhs_permutation: Permutation {
                        dims: vec![1, 0, 2, 3, 4],
                    },
                    tensordot_fixed_position: TensordotFixedPosition {
                        len_uncontracted_lhs: 1,
                        len_uncontracted_rhs: output_shape.dims().iter().product::<usize>(),
                        len_contracted_axes: d1s.dim(1)?,
                        output_shape,
                    },
                    output_permutation: Permutation {
                        dims: vec![0, 1, 2, 3],
                    },
                }
                .eval(&rhos_c.i(..rhos_c.dims()[0] - 1)?, &d1s)
            })
            .unwrap_or_else(|| Tensor::zeros_like(m0))?;

        let d1_t = (model_t - m0)?;
        let x_t = (x_t_
            - (alpha_t
                * b_h
                * (corr_res + rhos_c.i(rhos_c.dims()[0] - 1)?.broadcast_mul(&d1_t)?)?)?)?;

        Ok(x_t)
    }
}

impl Scheduler for EdmDpmMultistepScheduler {
    fn step(&mut self, model_output: &Tensor, timestep: usize, sample: &Tensor) -> Result<Tensor> {
        let step_index = self.step_index(timestep);
        let model_output_converted = &self.convert_model_output(model_output, sample, timestep)?;
        let sample = match (&self.config.corrector, self.state.last_sample()) {
            (CorrectorConfiguration::Enabled { skip_steps: s }, Some(last_sample))
                if !s.contains(&step_index) && step_index > 0 =>
            {
                &self.multistep_uni_c_bh_update(
                    model_output_converted,
                    self.state.model_outputs(),
                    last_sample,
                    sample,
                    timestep,
                )?
            }
            (CorrectorConfiguration::Enabled { .. }, _) | (CorrectorConfiguration::Disabled, _) => {
                sample
            }
        };

        let mut model_outputs = self.state.model_outputs().to_vec();
        for i in 0..self.config.solver_order.saturating_sub(1) {
            self.state
                .update_model_output(i, model_outputs[i + 1].take());
        }
        self.state.update_model_output(
            model_outputs.len() - 1,
            Some(model_output_converted.clone()),
        );

        let mut this_order = self.config.solver_order;
        if self.config.lower_order_final {
            this_order = self
                .config
                .solver_order
                .min(self.schedule.timesteps.len() - step_index);
        }
        self.state
            .update_order(this_order.min(self.state.lower_order_nums() + 1));

        self.state.update_last_sample(sample.clone());
        let prev_sample = self.multistep_uni_p_bh_update(sample, timestep)?;

        let lower_order_nums = self.state.lower_order_nums();
        if lower_order_nums < self.config.solver_order {
            self.state.update_lower_order_nums(lower_order_nums + 1);
        }

        Ok(prev_sample)
    }

    fn scale_model_input(&self, sample: Tensor, _timestep: usize) -> Result<Tensor> {
        Ok(sample)
    }

    fn timesteps(&self) -> &[usize] {
        &self.schedule.timesteps
    }

    fn add_noise(&self, original: &Tensor, noise: Tensor, timestep: usize) -> Result<Tensor> {
        let (alpha_t, sigma_t) = (
            self.schedule.alpha_t(timestep),
            self.schedule.sigma_t(timestep),
        );

        (alpha_t * original)? + (sigma_t * noise)?
    }

    fn init_noise_sigma(&self) -> f64 {
        self.schedule.sigma_t(self.schedule.num_training_steps())
    }
}

#[derive(Debug, Clone)]
struct Schedule {
    timesteps: Vec<usize>,
    num_training_steps: usize,
    sigma_schedule: SigmaSchedule,
    #[allow(unused)]
    timestep_schedule: TimestepSchedule,
}

impl Schedule {
    fn new(
        timestep_schedule: TimestepSchedule,
        sigma_schedule: SigmaSchedule,
        num_inference_steps: usize,
        num_training_steps: usize,
    ) -> Result<Self> {
        Ok(Self {
            timesteps: timestep_schedule.timesteps(
                &sigma_schedule,
                num_inference_steps,
                num_training_steps,
            )?,
            timestep_schedule,
            sigma_schedule,
            num_training_steps,
        })
    }

    fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn num_training_steps(&self) -> usize {
        self.num_training_steps
    }

    fn t(&self, step: usize) -> f64 {
        (step as f64 + 1.) / self.num_training_steps as f64
    }

    fn alpha_t(&self, t: usize) -> f64 {
        (1. / (self.sigma_schedule.sigma_t(self.t(t)).powi(2) + 1.)).sqrt()
    }

    fn sigma_t(&self, t: usize) -> f64 {
        self.sigma_schedule.sigma_t(self.t(t)) * self.alpha_t(t)
    }

    fn lambda_t(&self, t: usize) -> f64 {
        self.alpha_t(t).ln() - self.sigma_t(t).ln()
    }
}

mod stats {
    //! This is a slightly modified form of the P² quantile implementation from https://github.com/vks/average.
    //! Also see: http://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf
    use num_traits::{Float, ToPrimitive};

    #[derive(Debug, Clone)]
    pub struct Quantile {
        q: [f64; 5],
        n: [i64; 5],
        m: [f64; 5],
        dm: [f64; 5],
        max: Option<f64>,
    }

    impl Quantile {
        pub fn new(p: f64) -> Quantile {
            assert!((0. ..=1.).contains(&p));
            Quantile {
                q: [0.; 5],
                n: [1, 2, 3, 4, 0],
                m: [1., 1. + 2. * p, 1. + 4. * p, 3. + 2. * p, 5.],
                dm: [0., p / 2., p, (1. + p) / 2., 1.],
                max: None,
            }
        }

        pub fn max(&self) -> f64 {
            self.max.unwrap_or(f64::NAN)
        }

        fn p(&self) -> f64 {
            self.dm[2]
        }

        fn parabolic(&self, i: usize, d: f64) -> f64 {
            let s = d.round() as i64;
            self.q[i]
                + d / (self.n[i + 1] - self.n[i - 1]).to_f64().unwrap()
                    * ((self.n[i] - self.n[i - 1] + s).to_f64().unwrap()
                        * (self.q[i + 1] - self.q[i])
                        / (self.n[i + 1] - self.n[i]).to_f64().unwrap()
                        + (self.n[i + 1] - self.n[i] - s).to_f64().unwrap()
                            * (self.q[i] - self.q[i - 1])
                            / (self.n[i] - self.n[i - 1]).to_f64().unwrap())
        }

        fn linear(&self, i: usize, d: f64) -> f64 {
            let sum = if d < 0. { i - 1 } else { i + 1 };
            self.q[i] + d * (self.q[sum] - self.q[i]) / (self.n[sum] - self.n[i]).to_f64().unwrap()
        }

        pub fn quantile(&self) -> f64 {
            if self.len() >= 5 {
                return self.q[2];
            }

            if self.is_empty() {
                return f64::NAN;
            }
            let mut heights: [f64; 4] = [self.q[0], self.q[1], self.q[2], self.q[3]];
            let len = self.len() as usize;
            debug_assert!(len < 5);
            sort_floats(&mut heights[..len]);
            let desired_index = (len as f64) * self.p() - 1.;
            let mut index = desired_index.ceil();
            if desired_index == index && index >= 0. {
                let index = index.round() as usize;
                debug_assert!(index < 5);
                if index < len - 1 {
                    return 0.5 * self.q[index] + 0.5 * self.q[index + 1];
                }
            }
            index = index.max(0.);
            let mut index = index.round() as usize;
            debug_assert!(index < 5);
            index = index.min(len - 1);
            self.q[index]
        }

        fn len(&self) -> u64 {
            self.n[4] as u64
        }

        fn is_empty(&self) -> bool {
            self.len() == 0
        }

        pub fn add(&mut self, x: f64) {
            self.max = self.max.map(|y| y.max(x)).or(Some(x));

            if self.n[4] < 5 {
                self.q[self.n[4] as usize] = x;
                self.n[4] += 1;
                if self.n[4] == 5 {
                    sort_floats(&mut self.q);
                }
                return;
            }

            let mut k: usize;
            if x < self.q[0] {
                self.q[0] = x;
                k = 0;
            } else {
                k = 4;
                for i in 1..5 {
                    if x < self.q[i] {
                        k = i;
                        break;
                    }
                }
                if self.q[4] < x {
                    self.q[4] = x;
                }
            };

            for i in k..5 {
                self.n[i] += 1;
            }
            for i in 0..5 {
                self.m[i] += self.dm[i];
            }

            for i in 1..4 {
                let d = self.m[i] - self.n[i].to_f64().unwrap();
                if d >= 1. && self.n[i + 1] - self.n[i] > 1
                    || d <= -1. && self.n[i - 1] - self.n[i] < -1
                {
                    let d = Float::signum(d);
                    let q_new = self.parabolic(i, d);
                    if self.q[i - 1] < q_new && q_new < self.q[i + 1] {
                        self.q[i] = q_new;
                    } else {
                        self.q[i] = self.linear(i, d);
                    }
                    let delta = d.round() as i64;
                    debug_assert_eq!(delta.abs(), 1);
                    self.n[i] += delta;
                }
            }
        }

        pub fn with_samples(mut self, samples: impl IntoIterator<Item = f64>) -> Self {
            for sample in samples {
                self.add(sample);
            }

            self
        }
    }

    fn sort_floats(v: &mut [f64]) {
        v.sort_unstable_by(|a, b| a.total_cmp(b));
    }
}

mod linalg {
    use candle::{IndexOp, Result, Shape, Tensor};

    pub fn inverse(m: &Tensor) -> Result<Tensor> {
        adjoint(m)? / determinant(m)?.to_scalar::<f64>()?
    }

    pub fn adjoint(m: &Tensor) -> Result<Tensor> {
        cofactor(m)?.transpose(0, 1)
    }

    pub fn cofactor(m: &Tensor) -> Result<Tensor> {
        let s = m.shape().dim(0)?;
        if s == 2 {
            let mut v = vec![];
            for i in 0..2 {
                let mut x = vec![];
                for j in 0..2 {
                    x.push((m.i((i, j))? * (-1.0f64).powi(i as i32 + j as i32))?)
                }
                v.push(Tensor::stack(&x, 0)?.unsqueeze(0)?);
            }
            return Tensor::stack(&v, 1)?.squeeze(0);
        }

        let minors = minors(m)?;
        let mut v = vec![];
        for i in 0..s {
            let mut x = vec![];
            for j in 0..s {
                let det = (determinant(&minors.i((i, j))?)?
                    * ((-1.0f64).powi(i as i32) * (-1.0f64).powi(j as i32)))?;
                x.push(det);
            }
            v.push(Tensor::stack(&x, 0)?.unsqueeze(0)?);
        }

        Tensor::stack(&v, 1)?.squeeze(0)
    }

    pub fn determinant(m: &Tensor) -> Result<Tensor> {
        let s = m.shape().dim(0)?;
        if s == 2 {
            return (m.i((0, 0))? * m.i((1, 1))?)? - (m.i((0, 1))? * m.i((1, 0))?);
        }

        let cofactor = cofactor(m)?;
        let m0 = m.i((0, 0))?;
        let det = (0..s)
            .map(|i| (m.i((0, i))? * cofactor.i((0, i))?))
            .try_fold(m0.zeros_like()?, |acc, cur| (acc + cur?))?;

        Ok(det)
    }

    pub fn minors(m: &Tensor) -> Result<Tensor> {
        let s = m.shape().dim(0)?;
        if s == 1 {
            return m.i((0, 0));
        }

        let mut v = vec![];
        for i in 0..s {
            let msub = Tensor::cat(&[m.i((..i, ..))?, m.i(((i + 1).., ..))?], 0)?;
            let mut x = vec![];
            for j in 0..s {
                let t = Tensor::cat(&[msub.i((.., ..j))?, msub.i((.., (j + 1)..))?], 1)?;
                x.push(t);
            }
            v.push(Tensor::stack(&x, 0)?.unsqueeze(0)?);
        }

        Tensor::stack(&v, 1)?.squeeze(0)
    }

    #[derive(Debug)]
    pub struct TensordotGeneral {
        pub lhs_permutation: Permutation,
        pub rhs_permutation: Permutation,
        pub tensordot_fixed_position: TensordotFixedPosition,
        pub output_permutation: Permutation,
    }

    impl TensordotGeneral {
        pub fn eval(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
            let permuted_lhs = self.lhs_permutation.eval(lhs)?;
            let permuted_rhs = self.rhs_permutation.eval(rhs)?;
            let tensordotted = self
                .tensordot_fixed_position
                .eval(&permuted_lhs, &permuted_rhs)?;
            self.output_permutation.eval(&tensordotted)
        }
    }

    #[derive(Debug)]
    pub struct TensordotFixedPosition {
        pub len_uncontracted_lhs: usize,
        pub len_uncontracted_rhs: usize,
        pub len_contracted_axes: usize,
        pub output_shape: Shape,
    }

    impl TensordotFixedPosition {
        fn eval(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
            let lhs_view = lhs.reshape((self.len_uncontracted_lhs, self.len_contracted_axes))?;
            let rhs_view = rhs.reshape((self.len_contracted_axes, self.len_uncontracted_rhs))?;

            lhs_view.matmul(&rhs_view)?.reshape(&self.output_shape)
        }
    }

    #[derive(Debug)]
    pub struct Permutation {
        pub dims: Vec<usize>,
    }

    impl Permutation {
        fn eval(&self, tensor: &Tensor) -> Result<Tensor> {
            tensor.permute(self.dims.as_slice())
        }
    }
}
