use candle::{Result, Tensor, TensorId, Var};
use candle_nn::optim::Optimizer;
use std::collections::HashMap;

/// Adagrad optimizer
///
/// Described in <https://jmlr.org/papers/v12/duchi11a.html>
///
/// For pseudocde see <https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html>

#[derive(Debug)]
pub struct Adagrad {
    vars: Vec<Var>,
    params: ParamsAdaGrad,
    t: usize,
    state_sum: HashMap<TensorId, Tensor>,
}

#[derive(Debug)]
pub struct ParamsAdaGrad {
    pub lr: f64,
    pub lr_decay: f64,
    pub initial_acc: f64,
    pub dampening: f64,
    pub weight_decay: f64,
    pub eps: f64,
}

impl Default for ParamsAdaGrad {
    fn default() -> Self {
        Self {
            lr: 0.01,
            lr_decay: 0.0,
            initial_acc: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            eps: 1e-10,
        }
    }
}

impl Optimizer for Adagrad {
    type Config = ParamsAdaGrad;

    fn new(vars: Vec<Var>, params: ParamsAdaGrad) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .collect();
        // // Err(SGDError::NoMomentum)?;
        // let mut params = params;
        // params.t = 0;
        Ok(Self {
            vars,
            t: 0,
            params,
            state_sum: HashMap::new(),
        })
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn step(&mut self, grads: &candle::backprop::GradStore) -> Result<()> {
        for var in &self.vars {
            if let Some(grad) = grads.get(var) {
                #[allow(clippy::cast_precision_loss)]
                let gamma_tilde = self.params.lr / (1. + (self.t as f64 * self.params.lr_decay));
                if self.params.weight_decay == 0. {
                    // let gt = (grad + (self.params.weight_decay * var.as_tensor())?)?;
                    let current_sum = if let Some(sum) = self.state_sum.get(&var.id()) {
                        (sum + grad.powf(2.)?)?
                    } else {
                        (self.params.initial_acc + grad.powf(2.)?)?
                    };
                    let change =
                        (gamma_tilde * (grad.div(&(current_sum.powf(0.5)? + self.params.eps)?))?)?;
                    self.state_sum.insert(var.id(), current_sum);
                    var.set(&var.sub(&change)?)?;
                } else {
                    let grad = &(grad + (self.params.weight_decay * var.as_tensor())?)?;
                    let current_sum = if let Some(sum) = self.state_sum.get(&var.id()) {
                        (sum + grad.powf(2.)?)?
                    } else {
                        (self.params.initial_acc + grad.powf(2.)?)?
                    };
                    let change =
                        (gamma_tilde * (grad.div(&(current_sum.powf(0.5)? + self.params.eps)?))?)?;
                    self.state_sum.insert(var.id(), current_sum);
                    var.set(&var.sub(&change)?)?;
                }
            }
        }
        self.t += 1;
        Ok(())
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }
}

impl Adagrad {
    #[must_use]
    pub fn into_inner(self) -> Vec<Var> {
        self.vars
    }

    pub fn push(&mut self, var: &Var) {
        self.vars.push(var.clone());
    }
}
