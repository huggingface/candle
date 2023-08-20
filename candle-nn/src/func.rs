//! Layers defined by closures.
use candle::{Result, Tensor};

/// A layer defined by a simple closure.
pub struct Func<'a> {
    #[allow(clippy::type_complexity)]
    f: Box<dyn 'a + Fn(&Tensor) -> Result<Tensor> + Send>,
}

impl<'a> std::fmt::Debug for Func<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "func")
    }
}

pub fn func<'a, F>(f: F) -> Func<'a>
where
    F: 'a + Fn(&Tensor) -> Result<Tensor> + Send,
{
    Func { f: Box::new(f) }
}

impl<'a> super::Module for Func<'a> {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        (*self.f)(xs)
    }
}
