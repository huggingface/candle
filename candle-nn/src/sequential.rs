//! Sequential Layer
//!
//! A sequential layer used to chain multiple layers and closures.
use candle::{BackendStorage, Module, Result, Tensor};

/// A sequential layer combining multiple other layers.
pub struct Sequential<B: BackendStorage> {
    layers: Vec<Box<dyn Module<B>>>,
}

/// Creates a new empty sequential layer.
pub fn seq<B: BackendStorage>() -> Sequential<B> {
    Sequential { layers: vec![] }
}

impl<B: BackendStorage> Sequential<B> {
    /// The number of sub-layers embedded in this layer.
    pub fn len(&self) -> i64 {
        self.layers.len() as i64
    }

    /// Returns true if this layer does not have any sub-layer.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl<B: BackendStorage> Module<B> for Sequential<B> {
    fn forward(&self, xs: &Tensor<B>) -> Result<Tensor<B>> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs)?
        }
        Ok(xs)
    }
}

impl<B: BackendStorage + 'static> Sequential<B> {
    /// Appends a layer after all the current layers.
    #[allow(clippy::should_implement_trait)]
    pub fn add<M: Module<B> + 'static>(mut self, layer: M) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    /// Appends a closure after all the current layers.
    pub fn add_fn<F>(self, f: F) -> Self
    where
        F: 'static + Fn(&Tensor<B>) -> Result<Tensor<B>> + Send + Sync,
    {
        self.add(super::func(f))
    }

    /// Applies the forward pass and returns the output for each layer.
    pub fn forward_all(&self, xs: &Tensor<B>) -> Result<Vec<Tensor<B>>> {
        let mut vec = Vec::with_capacity(self.layers.len());
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs)?;
            vec.push(xs.clone())
        }
        Ok(vec)
    }
}
