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

/// Optimizer for Stochastic Gradient Descent.
///
/// Contrary to the PyTorch implementation of SGD, this version does not support momentum.
#[derive(Debug)]
pub struct SGD {
    vars: Vec<Var>,
    learning_rate: f64,
}

impl Optimizer for SGD {
    type Config = f64;

    fn new(vars: Vec<Var>, learning_rate: f64) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .collect();
        Ok(Self {
            vars,
            learning_rate,
        })
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn step(&mut self, grads: &candle::backprop::GradStore) -> Result<()> {
        for var in self.vars.iter() {
            if let Some(grad) = grads.get(var) {
                var.set(&var.sub(&(grad * self.learning_rate)?)?)?;
            }
        }
        Ok(())
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr
    }
}

impl SGD {
    pub fn into_inner(self) -> Vec<Var> {
        self.vars
    }

    pub fn push(&mut self, var: &Var) {
        self.vars.push(var.clone())
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

/// Specifies whether the metric should be minimized or maximized. Used for [`ReduceLROnPlateau`]
// Named with a prefix to avoid naming conflicts with similar types in future. Modules are an alternative,
// but this did not feel more ergonomic than a name prefix.
#[derive(Clone, Copy, Debug)]
pub enum LRPlateauMode {
    Min,
    Max,
}

/// Specifies how the threshold is applied: relative or absolute. Used for [`ReduceLROnPlateau`]
// Named with a prefix to avoid naming conflicts with similar types in future.
#[derive(Clone, Copy, Debug)]
pub enum LRPlateauThresholdMode {
    Rel,
    Abs,
}

/// A learning rate scheduler which reduces the learning rate when a quantity has stopped improving.
/// Analogous to the PyTorch `ReduceLROnPlateau`.
///
/// Models typically benefit from reducing the learning rate by a few orders of magnitude when learning stalls.
/// This scheduler observes some quantity/metric and will reduce the learning rate when no improvement has been seen
/// for a "patience" number of epochs.
///
/// # Example
///
/// ```
/// use candle_nn::{AdamW, ParamsAdamW, Optimizer};
/// use candle_nn::optim::{ReduceLROnPlateauBuilder, LRPlateauMode};
///
/// // Create an optimizer
/// let opt = AdamW::new(vec![], ParamsAdamW::default()).unwrap();
/// let initial_lr = opt.learning_rate();
///
/// // Create scheduler
/// let mut scheduler = ReduceLROnPlateauBuilder::new(opt)
///     .patience(0)  // Wait 5 epochs before reducing LR
///     .factor(0.5)  // Reduce LR by half
///     .build();
///
/// // Example losses
/// let losses = vec![1., 1., 0.75, 0.5, 0.5];
/// let expected_lrs = vec![
///     initial_lr,           // Epoch 1: First value, sets baseline
///     initial_lr * 0.5,     // Epoch 2: Loss didn't improve (1.0 == 1.0)
///     initial_lr * 0.5,     // Epoch 3: Loss improved (0.75 < 1.0)
///     initial_lr * 0.5,     // Epoch 4: Loss improved (0.5 < 0.75)
///     initial_lr * 0.25,    // Epoch 5: Loss didn't improve (0.5 == 0.5)
/// ];
/// // In your training loop:
/// for epoch in 1..=5 {
///     // ... training code ...
///     
///     // Get validation loss
///     let val_loss = losses[epoch - 1];
///     
///     // Update scheduler
///     scheduler.step(val_loss).unwrap();
///     
///     // Current learning rate after potential adjustment
///     let current_lr = scheduler.opt().learning_rate();
/// }
/// ```
#[derive(Debug)]
pub struct ReduceLROnPlateau<O> {
    /// The optimizer to schedule
    optimizer: O,
    /// One of min or max. In min mode, the lr will be reduced with the quantity has stopped decreasing.
    /// In max mode, the quantity will reduce when the quantity has stopped increasing.
    mode: LRPlateauMode,
    /// Factor the lr will be reduced by (new_lr = lr * factor)
    factor: f64,
    /// The number of epochs with no improvement in quantity (crossing a threshold) before reducing the lr
    patience: usize,
    /// The threshold for the quantity to stop reducing
    threshold: f64,
    /// The mode for the threshold. Determines how the threshold will calculate the quantity
    threshold_mode: LRPlateauThresholdMode,
    /// The number of epochs to wait before resuming normal operation after a lr reduction
    cooldown: usize,
    /// The minimum lr value
    min_lr: f64,
    /// Minimal decay applied to the lr. If the difference between the previous and the current lr is less than eps, the update is ignored.
    eps: f64,
    /// The best metric value seen so far
    best: Option<f64>,
    /// The number of epochs with no improvement in quantity (crossing a threshold) before reducing the lr
    num_bad_epochs: usize,
    /// The number of epochs to wait before resuming normal operation after a lr reduction
    cooldown_counter: usize,
}

impl<O: Optimizer> std::fmt::Display for ReduceLROnPlateau<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ReduceLROnPlateau {{")?;
        write!(f, "mode: {:?}, ", self.mode)?;
        write!(f, "factor: {}, ", self.factor)?;
        write!(f, "patience: {}, ", self.patience)?;
        write!(f, "threshold: {}, ", self.threshold)?;
        write!(f, "threshold_mode: {:?}, ", self.threshold_mode)?;
        write!(f, "cooldown: {}, ", self.cooldown)?;
        write!(f, "min_lr: {}, ", self.min_lr)?;
        write!(f, "eps: {}, ", self.eps)?;
        write!(f, "best: {:?}, ", self.best)?;
        write!(f, "num_bad_epochs: {}, ", self.num_bad_epochs)?;
        write!(f, "cooldown_counter: {}, ", self.cooldown_counter)?;
        // Instead of printing the entire optimizer, print the most relevant field
        // which is the current learning rate
        write!(f, "current_lr: {}", self.optimizer.learning_rate())?;
        write!(f, "}}")
    }
}

impl<O: Optimizer> ReduceLROnPlateau<O> {
    /// Perform a learning rate step given the currrent metric level. This learning rate scheduler will adjust the
    /// learning rate based on how this metric value compares to the previously best seen metric.
    pub fn step(&mut self, metric: f64) -> Result<()> {
        let improvement = match self.mode {
            LRPlateauMode::Min => match self.threshold_mode {
                LRPlateauThresholdMode::Rel => self
                    .best
                    .map_or(true, |best| metric < best * (1.0 - self.threshold)),
                LRPlateauThresholdMode::Abs => self
                    .best
                    .map_or(true, |best| metric < best - self.threshold),
            },
            LRPlateauMode::Max => match self.threshold_mode {
                LRPlateauThresholdMode::Rel => self
                    .best
                    .map_or(true, |best| metric > best * (1.0 + self.threshold)),
                LRPlateauThresholdMode::Abs => self
                    .best
                    .map_or(true, |best| metric > best + self.threshold),
            },
        };

        if improvement {
            self.best = Some(metric);
            self.num_bad_epochs = 0;
        } else {
            self.num_bad_epochs += 1;
        }

        if self.cooldown_counter > 0 {
            self.cooldown_counter -= 1;
            self.num_bad_epochs = 0;
        }

        if self.num_bad_epochs > self.patience {
            let current_lr = self.optimizer.learning_rate();
            let new_lr = current_lr * self.factor;
            if (current_lr - new_lr) > self.eps && new_lr >= self.min_lr {
                self.optimizer.set_learning_rate(new_lr);
                self.cooldown_counter = self.cooldown;
                self.num_bad_epochs = 0;
            }
        }

        Ok(())
    }

    /// Get the inner optimizer mutably
    pub fn opt_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }

    /// Get the inner optimizer immutably
    pub fn opt(&self) -> &O {
        &self.optimizer
    }

    /// Get the number of consecutive epochs for which the best metric has not decreased below the threshold
    pub fn num_bad_epochs(&self) -> usize {
        self.num_bad_epochs
    }

    /// Get the mode (Min or Max)
    pub fn mode(&self) -> LRPlateauMode {
        self.mode
    }

    /// Get the factor by which the learning rate is reduced
    pub fn factor(&self) -> f64 {
        self.factor
    }

    /// Get the number of epochs with no improvement before reducing the learning rate
    pub fn patience(&self) -> usize {
        self.patience
    }

    /// Get the threshold for determining improvement
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Get the threshold mode (Rel or Abs)
    pub fn threshold_mode(&self) -> LRPlateauThresholdMode {
        self.threshold_mode
    }

    /// Get the cooldown period after a learning rate reduction
    pub fn cooldown(&self) -> usize {
        self.cooldown
    }

    /// Get the minimum learning rate
    pub fn min_lr(&self) -> f64 {
        self.min_lr
    }

    /// Get the epsilon value for minimal learning rate changes
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// Get the best metric value seen so far
    pub fn best(&self) -> Option<f64> {
        self.best
    }

    /// Get the current cooldown counter
    pub fn cooldown_counter(&self) -> usize {
        self.cooldown_counter
    }
}

/// Builder for the [`ReduceLROnPlateau`] scheduler
// Uses a builder to provide sensible, overridable defaults and to allow extensible fields.
pub struct ReduceLROnPlateauBuilder<O> {
    /// The optimizer to schedule
    optimizer: O,
    /// One of min or max. In min mode, the lr will be reduced with the quantity has stopped decreasing.
    /// In max mode, the quantity will reduce when the quantity has stopped increasing.
    mode: LRPlateauMode,
    /// Factor the lr will be reduced by (new_lr = lr * factor)
    factor: f64,
    /// The number of epochs with no improvement in quantity (crossing a threshold) before reducing the lr
    patience: usize,
    /// The threshold for the quantity to stop reducing
    threshold: f64,
    /// The mode for the threshold. Determines how the threshold will calculate the quantity
    threshold_mode: LRPlateauThresholdMode,
    /// The number of epochs to wait before resuming normal operation after a lr reduction
    cooldown: usize,
    /// The minimum lr value
    min_lr: f64,
    /// Minimal decay applied to the lr. If the difference between the previous and the current lr is less than eps, the update is ignored.
    eps: f64,
}

impl<O: Optimizer> ReduceLROnPlateauBuilder<O> {
    /// Initializes a new builder for the ReduceLROnPlateau scheduler with sensible, overridable defaults.
    pub fn new(optimizer: O) -> Self {
        // Default values mirror PyTorch
        Self {
            optimizer,
            mode: LRPlateauMode::Min,
            factor: 0.1,
            patience: 10,
            threshold: 1e-4,
            threshold_mode: LRPlateauThresholdMode::Rel,
            cooldown: 0,
            min_lr: 0.0,
            eps: 1e-8,
        }
    }

    pub fn optimizer(mut self, optimizer: O) -> Self {
        self.optimizer = optimizer;
        self
    }

    pub fn mode(mut self, mode: LRPlateauMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn factor(mut self, factor: f64) -> Self {
        self.factor = factor;
        self
    }

    pub fn patience(mut self, patience: usize) -> Self {
        self.patience = patience;
        self
    }

    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn threshold_mode(mut self, threshold_mode: LRPlateauThresholdMode) -> Self {
        self.threshold_mode = threshold_mode;
        self
    }

    pub fn cooldown(mut self, cooldown: usize) -> Self {
        self.cooldown = cooldown;
        self
    }

    pub fn min_lr(mut self, min_lr: f64) -> Self {
        self.min_lr = min_lr;
        self
    }

    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    pub fn build(self) -> ReduceLROnPlateau<O> {
        ReduceLROnPlateau {
            optimizer: self.optimizer,
            mode: self.mode,
            factor: self.factor,
            patience: self.patience,
            threshold: self.threshold,
            threshold_mode: self.threshold_mode,
            cooldown: self.cooldown,
            min_lr: self.min_lr,
            eps: self.eps,
            best: None,
            num_bad_epochs: 0,
            cooldown_counter: 0,
        }
    }
}
