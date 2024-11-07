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
        Self {
            func,
            lr: 0.0,
        }
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
