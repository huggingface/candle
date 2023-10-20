//! A sequential layer used to chain multiple layers and closures.
use candle::{Module, Result, Tensor};

/// A sequential layer combining multiple other layers.
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

/// Creates a new empty sequential layer.
pub fn seq() -> Sequential {
    Sequential { layers: vec![] }
}

impl Sequential {
    /// The number of sub-layers embedded in this layer.
    pub fn len(&self) -> i64 {
        self.layers.len() as i64
    }

    /// Returns true if this layer does not have any sub-layer.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl Module for Sequential {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs)?
        }
        Ok(xs)
    }
}

impl Sequential {
    /// Appends a layer after all the current layers.
    #[allow(clippy::should_implement_trait)]
    pub fn add<M: Module + 'static>(mut self, layer: M) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    /// Appends a closure after all the current layers.
    pub fn add_fn<F>(self, f: F) -> Self
    where
        F: 'static + Fn(&Tensor) -> Result<Tensor> + Send + Sync,
    {
        self.add(super::func(f))
    }

    /// Applies the forward pass and returns the output for each layer.
    pub fn forward_all(&self, xs: &Tensor) -> Result<Vec<Tensor>> {
        let mut vec = Vec::with_capacity(self.layers.len());
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs)?;
            vec.push(xs.clone())
        }
        Ok(vec)
    }
}
