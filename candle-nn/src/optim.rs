//! Various optimization algorithms.
use candle::{Result, Tensor, Var};

/// The interface optimizers should implement.
pub trait Optimizer: Sized {
    type Config: Sized;

    fn new(vars: Vec<Var>, config: Self::Config) -> Result<Self>;

    fn step(&mut self, grads: &candle::backprop::GradStore) -> Result<()>;

    fn learning_rate(&self) -> f64;

    fn set_learning_rate(&mut self, lr: f64);

    fn empty(config: Self::Config) -> Result<Self> {
        Self::new(vec![], config)
    }

    fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        self.step(&grads)
    }

    fn from_slice(vars: &[&Var], config: Self::Config) -> Result<Self> {
        let vars: Vec<_> = vars.iter().map(|&v| v.clone()).collect();
        Self::new(vars, config)
    }
}

#[derive(Clone, Debug)]
pub struct ParamsSGD {
    pub lr: f64,
    pub momentum: Option<f64>,
    pub nesterov: bool,
}

impl Default for ParamsSGD {
    fn default() -> Self {
        Self {
            lr: 0.01,
            momentum: None,
            nesterov: false,
        }
    }
}

#[derive(Debug)]
struct VarSGD {
    var: Var,
    velocity: Option<Var>,
}

/// Optimizer for Stochastic Gradient Descent.
/// By Default,the update rule of SGD is
/// ```tex
/// \theta_{t+1} = \theta_{t} - \eta \cdot g_{t}
/// ```
/// # momentum
/// Momentum accumulates a moving average of past gradients to accelerate
/// updates in consistent directions and dampen oscillations.
///
/// you can specify the momentum by `momentum = Some(mu)`
/// ```tex
/// v_t = \mu * v_{t-1} + g_t
/// \theta_t = \theta_{t-1} - lr * v_t
/// ```
/// # Nesterov
/// Nesterov momentum improves upon standard momentum by computing the gradient
/// at a predicted future position, rather than the current position.
///
/// you can specify the momentum by `nesterov = true`
/// ```tex
/// v_t = \mu * v_{t-1} + g_t
/// \theta_t = \theta_{t-1} - lr * (g_t + \mu * v_{t-1})
/// ```
#[derive(Debug)]
pub struct SGD {
    vars: Vec<VarSGD>,
    params: ParamsSGD,
}

impl Optimizer for SGD {
    type Config = ParamsSGD;

    fn new(vars: Vec<Var>, params: ParamsSGD) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|v| {
                let velocity = params
                    .momentum
                    .map(|_| Var::zeros(v.shape(), v.dtype(), v.device()))
                    .transpose()?;
                Ok(VarSGD { var: v, velocity })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { vars, params })
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr
    }

    fn step(&mut self, grads: &candle::backprop::GradStore) -> Result<()> {
        let lr = self.params.lr;
        for var in self.vars.iter_mut() {
            let theta = &var.var;
            if let Some(g) = grads.get(theta) {
                match (&mut var.velocity, self.params.momentum) {
                    (None, None) => {
                        let next = theta.sub(&(g * lr)?)?;
                        theta.set(&next)?;
                    }
                    (Some(v), Some(mu)) => {
                        let buf = v.as_tensor();
                        let next_v = ((buf * mu)? + g)?;
                        v.set(&next_v)?;

                        let update = if self.params.nesterov {
                            (g + buf * mu)?
                        } else {
                            next_v
                        };

                        let next_theta = theta.sub(&(update * lr)?)?;
                        theta.set(&next_theta)?;
                    }
                    _ => {
                        unreachable!("velocity and momentum must be consistent")
                    }
                }
            }
        }
        Ok(())
    }
}

impl SGD {
    pub fn into_inner(self) -> Vec<Var> {
        self.vars.into_iter().map(|v| v.var).collect()
    }

    pub fn push(&mut self, var: &Var) -> Result<()> {
        let velocity = self
            .params
            .momentum
            .map(|_| Var::zeros(var.shape(), var.dtype(), var.device()))
            .transpose()?;

        self.vars.push(VarSGD {
            var: var.clone(),
            velocity,
        });
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ParamsAdamW {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl Default for ParamsAdamW {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

#[derive(Debug)]
struct VarAdamW {
    var: Var,
    first_moment: Var,
    second_moment: Var,
}

#[derive(Debug)]
pub struct AdamW {
    vars: Vec<VarAdamW>,
    step_t: usize,
    params: ParamsAdamW,
}

impl Optimizer for AdamW {
    type Config = ParamsAdamW;

    fn new(vars: Vec<Var>, params: ParamsAdamW) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                let dtype = var.dtype();
                let shape = var.shape();
                let device = var.device();
                let first_moment = Var::zeros(shape, dtype, device)?;
                let second_moment = Var::zeros(shape, dtype, device)?;
                Ok(VarAdamW {
                    var,
                    first_moment,
                    second_moment,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            vars,
            params,
            step_t: 0,
        })
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr
    }

    fn step(&mut self, grads: &candle::backprop::GradStore) -> Result<()> {
        self.step_t += 1;
        let lr = self.params.lr;
        let lambda = self.params.weight_decay;
        let lr_lambda = lr * lambda;
        let beta1 = self.params.beta1;
        let beta2 = self.params.beta2;
        let scale_m = 1f64 / (1f64 - beta1.powi(self.step_t as i32));
        let scale_v = 1f64 / (1f64 - beta2.powi(self.step_t as i32));
        for var in self.vars.iter() {
            let theta = &var.var;
            let m = &var.first_moment;
            let v = &var.second_moment;
            if let Some(g) = grads.get(theta) {
                // This involves locking 3 RWLocks per params, if the parameters are large this
                // should not be an issue but this may be problematic with models with lots of
                // small parameters.
                let next_m = ((m.as_tensor() * beta1)? + (g * (1.0 - beta1))?)?;
                let next_v = ((v.as_tensor() * beta2)? + (g.sqr()? * (1.0 - beta2))?)?;
                let m_hat = (&next_m * scale_m)?;
                let v_hat = (&next_v * scale_v)?;
                let next_theta = (theta.as_tensor() * (1f64 - lr_lambda))?;
                let adjusted_grad = (m_hat / (v_hat.sqrt()? + self.params.eps)?)?;
                let next_theta = (next_theta - (adjusted_grad * lr)?)?;
                m.set(&next_m)?;
                v.set(&next_v)?;
                theta.set(&next_theta)?;
            }
        }
        Ok(())
    }
}

impl AdamW {
    pub fn new_lr(vars: Vec<Var>, learning_rate: f64) -> Result<Self> {
        let params = ParamsAdamW {
            lr: learning_rate,
            ..ParamsAdamW::default()
        };
        Self::new(vars, params)
    }

    pub fn params(&self) -> &ParamsAdamW {
        &self.params
    }

    pub fn set_params(&mut self, params: ParamsAdamW) {
        self.params = params;
    }
}
