//! Layers defined by closures.
use candle::{Result, Tensor};
use std::sync::Arc;

/// A layer defined by a simple closure.
#[derive(Clone)]
pub struct Func<'a> {
    #[allow(clippy::type_complexity)]
    f: Arc<dyn 'a + Fn(&Tensor) -> Result<Tensor> + Send + Sync>,
}

impl std::fmt::Debug for Func<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "func")
    }
}

pub fn func<'a, F>(f: F) -> Func<'a>
where
    F: 'a + Fn(&Tensor) -> Result<Tensor> + Send + Sync,
{
    Func { f: Arc::new(f) }
}

impl super::Module for Func<'_> {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        (*self.f)(xs)
    }
}

impl<'a> Func<'a> {
    pub fn new<F>(f: F) -> Self
    where
        F: 'a + Fn(&Tensor) -> Result<Tensor> + Send + Sync,
    {
        Self { f: Arc::new(f) }
    }
}

/// A layer defined by a simple closure.
#[derive(Clone)]
pub struct FuncT<'a> {
    #[allow(clippy::type_complexity)]
    f: Arc<dyn 'a + Fn(&Tensor, bool) -> Result<Tensor> + Send + Sync>,
}

impl std::fmt::Debug for FuncT<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "func")
    }
}

pub fn func_t<'a, F>(f: F) -> FuncT<'a>
where
    F: 'a + Fn(&Tensor, bool) -> Result<Tensor> + Send + Sync,
{
    FuncT { f: Arc::new(f) }
}

impl super::ModuleT for FuncT<'_> {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        (*self.f)(xs, train)
    }
}

impl<'a> FuncT<'a> {
    pub fn new<F>(f: F) -> Self
    where
        F: 'a + Fn(&Tensor, bool) -> Result<Tensor> + Send + Sync,
    {
        Self { f: Arc::new(f) }
    }
}
