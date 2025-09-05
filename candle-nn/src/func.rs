//! Layers defined by closures.
use candle::{BackendStorage, Result, Tensor};
use std::sync::Arc;

/// A layer defined by a simple closure.
#[derive(Clone)]
pub struct Func<'a, B: BackendStorage> {
    #[allow(clippy::type_complexity)]
    f: Arc<dyn 'a + Fn(&Tensor<B>) -> Result<Tensor<B>> + Send + Sync>,
}

impl<B: BackendStorage> std::fmt::Debug for Func<'_, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "func")
    }
}

pub fn func<'a, F, B: BackendStorage>(f: F) -> Func<'a, B>
where
    F: 'a + Fn(&Tensor<B>) -> Result<Tensor<B>> + Send + Sync,
{
    Func { f: Arc::new(f) }
}

impl<B: BackendStorage> super::Module<B> for Func<'_, B> {
    fn forward(&self, xs: &Tensor<B>) -> Result<Tensor<B>> {
        (*self.f)(xs)
    }
}

impl<'a, B: BackendStorage> Func<'a, B> {
    pub fn new<F>(f: F) -> Self
    where
        F: 'a + Fn(&Tensor<B>) -> Result<Tensor<B>> + Send + Sync,
    {
        Self { f: Arc::new(f) }
    }
}

/// A layer defined by a simple closure.
#[derive(Clone)]
pub struct FuncT<'a, B: BackendStorage> {
    #[allow(clippy::type_complexity)]
    f: Arc<dyn 'a + Fn(&Tensor<B>, bool) -> Result<Tensor<B>> + Send + Sync>,
}

impl<B: BackendStorage> std::fmt::Debug for FuncT<'_, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "func")
    }
}

pub fn func_t<'a, F, B: BackendStorage>(f: F) -> FuncT<'a, B>
where
    F: 'a + Fn(&Tensor<B>, bool) -> Result<Tensor<B>> + Send + Sync,
{
    FuncT { f: Arc::new(f) }
}

impl<B: BackendStorage> super::ModuleT<B> for FuncT<'_, B> {
    fn forward_t(&self, xs: &Tensor<B>, train: bool) -> Result<Tensor<B>> {
        (*self.f)(xs, train)
    }
}

impl<'a, B: BackendStorage> FuncT<'a, B> {
    pub fn new<F>(f: F) -> Self
    where
        F: 'a + Fn(&Tensor<B>, bool) -> Result<Tensor<B>> + Send + Sync,
    {
        Self { f: Arc::new(f) }
    }
}
