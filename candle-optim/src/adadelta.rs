use candle::{Result, Tensor, TensorId, Var};
use candle_nn::optim::Optimizer;
use std::collections::HashMap;

/// Adadelta optimizer
///
/// Described in <https://arxiv.org/abs/1212.5701>
///
/// For pseudocde see <https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html>

#[derive(Debug)]
pub struct Adadelta {
    vars: Vec<Var>,
    params: ParamsAdaDelta,
    avg_acc: HashMap<TensorId, (Tensor, Tensor)>,
}

#[derive(Debug)]
pub struct ParamsAdaDelta {
    pub lr: f64,
    pub rho: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl Default for ParamsAdaDelta {
    fn default() -> Self {
        Self {
            lr: 1.0,
            rho: 0.9,
            weight_decay: 0.0,
            eps: 1e-6,
        }
    }
}

impl Optimizer for Adadelta {
    type Config = ParamsAdaDelta;

    fn new(vars: Vec<Var>, params: ParamsAdaDelta) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .collect();
        // // Err(SGDError::NoMomentum)?;
        // let mut params = params;
        // params.t = 0;
        Ok(Self {
            vars,
            params,
            avg_acc: HashMap::new(),
        })
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn step(&mut self, grads: &candle::backprop::GradStore) -> Result<()> {
        for var in &self.vars {
            if let Some(grad) = grads.get(var) {
                if self.params.weight_decay == 0. {
                    if let Some((v, u)) = self.avg_acc.get(&var.id()) {
                        let v =
                            ((v * self.params.rho)? + (1. - self.params.rho) * grad.powf(2.)?)?;
                        let delta_x = (((u + self.params.eps)?.powf(0.5)?)
                            .div(&((&v + self.params.eps)?.powf(0.5)?))?
                            * grad)?;
                        let u = ((u * self.params.rho)?
                            + (1. - self.params.rho) * delta_x.powf(2.)?)?;
                        var.set(&var.sub(&(delta_x * self.params.lr)?)?)?;
                        self.avg_acc.insert(var.id(), (v, u));
                    } else {
                        // start  u and v as 0 tensors
                        let v = ((1. - self.params.rho) * grad.powf(2.)?)?;
                        let delta_x = ((self.params.eps.powf(0.5))
                            * (&((&v + self.params.eps)?.powf(-0.5)?))
                            * grad)?;
                        let u = ((1. - self.params.rho) * delta_x.powf(2.)?)?;
                        var.set(&var.sub(&(delta_x * self.params.lr)?)?)?;
                        self.avg_acc.insert(var.id(), (v, u));
                    };
                } else {
                    let grad = &(grad + (self.params.weight_decay * var.as_tensor())?)?;
                    if let Some((v, u)) = self.avg_acc.get(&var.id()) {
                        let v =
                            ((v * self.params.rho)? + (1. - self.params.rho) * grad.powf(2.)?)?;
                        let delta_x = (((u + self.params.eps)?.powf(0.5)?)
                            .div(&((&v + self.params.eps)?.powf(0.5)?))?
                            * grad)?;
                        let u = ((u * self.params.rho)?
                            + (1. - self.params.rho) * delta_x.powf(2.)?)?;
                        var.set(&var.sub(&(delta_x * self.params.lr)?)?)?;
                        self.avg_acc.insert(var.id(), (v, u));
                    } else {
                        // start  u and v as 0 tensors
                        let v = ((1. - self.params.rho) * grad.powf(2.)?)?;
                        let delta_x = ((self.params.eps.powf(0.5))
                            * (&((&v + self.params.eps)?.powf(-0.5)?))
                            * grad)?;
                        let u = ((1. - self.params.rho) * delta_x.powf(2.)?)?;
                        var.set(&var.sub(&(delta_x * self.params.lr)?)?)?;
                        self.avg_acc.insert(var.id(), (v, u));
                    };
                }
            }
        }
        Ok(())
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }
}

impl Adadelta {
    #[must_use]
    pub fn into_inner(self) -> Vec<Var> {
        self.vars
    }

    pub fn push(&mut self, var: &Var) {
        self.vars.push(var.clone());
    }
}
