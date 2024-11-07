use candle::Result;

/// The interface LR Schedulers should implement.
pub trait LRScheduler<T> {
    /// Step the scheduler and return the new learning rate.
    fn step(&mut self, params: T) -> Result<f64>;

    /// Get the current learning rate.
    fn get_lr(&self) -> f64;
}

/// A learning rate scheduler that uses a function to determine the learning rate.
/// The function should take a parameter of type `T` and return a `f64`.
pub struct FnLRScheduler<T> {
    pub func: Box<dyn Fn(T) -> Result<f64>>,
    pub lr: f64,
}

impl<T> FnLRScheduler<T> {
    pub fn new(func: Box<dyn Fn(T) -> Result<f64>>) -> Self {
        Self { func, lr: 0.0 }
    }
}

impl<T> LRScheduler<T> for FnLRScheduler<T> {
    fn step(&mut self, params: T) -> Result<f64> {
        self.lr = (self.func)(params)?;
        Ok(self.lr)
    }

    fn get_lr(&self) -> f64 {
        self.lr
    }
}

/// Decays the learning rate of each parameter group by gamma every step_size epochs.
// https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR
pub struct StepLR {
    step_size: usize,
    last_epoch: usize,
    gamma: f64,
    lr: f64,
}

impl StepLR {
    pub fn new(step_size: usize, gamma: f64, lr: f64) -> Self {
        Self {
            step_size,
            last_epoch: 0,
            gamma,
            lr,
        }
    }
}

impl LRScheduler<()> for StepLR {
    fn step(&mut self, _params: ()) -> Result<f64> {
        self.last_epoch += 1;
        if self.last_epoch % self.step_size == 0 {
            self.lr *= self.gamma;
        }
        Ok(self.lr)
    }

    fn get_lr(&self) -> f64 {
        self.lr
    }
}
