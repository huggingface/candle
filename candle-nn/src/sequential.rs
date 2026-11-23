//! Sequential Layer
//!
//! A sequential layer used to chain multiple layers and closures.
use candle::{ModuleT, Result, Tensor};

/// A sequential layer combining multiple other layers.
/// Internal modules must at least implement ModuleT.
/// NOTE: It is incompatible with dispatch based on 'Module'.
/// Recommend to switch to dispatch based on 'ModuleT'.
/// For example, change Box<dyn Module> to Box<dyn ModuleT>.
pub struct Sequential {
    layers: Vec<Box<dyn ModuleT>>,
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

impl ModuleT for Sequential {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward_t(&xs, train)?
        }
        Ok(xs)
    }
}

impl Sequential {
    /// Regular forward that conforms to Module Trait.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward_t(&xs, false)?
        }
        Ok(xs)
    }

    /// Appends a layer after all the current layers.
    #[allow(clippy::should_implement_trait)]
    pub fn add<M: ModuleT + 'static>(mut self, layer: M) -> Self {
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

    /// Appends a closure that implements at least ModuleT after all current layers.
    pub fn add_fn_t<F>(self, f: F) -> Self
    where
        F: 'static + Fn(&Tensor, bool) -> Result<Tensor> + Send + Sync,
    {
        self.add(super::func_t(f))
    }

    /// Applies the forward pass and returns the output for each layer.
    pub fn forward_all(&self, xs: &Tensor) -> Result<Vec<Tensor>> {
        let mut vec = Vec::with_capacity(self.layers.len());
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward_t(&xs, false)?;
            vec.push(xs.clone())
        }
        Ok(vec)
    }

    /// Apply the ModuleT::forward_t pass and returns the output of each layer.
    pub fn forward_all_t(&self, xs: &Tensor, train: bool) -> Result<Vec<Tensor>> {
        let mut vec = Vec::with_capacity(self.layers.len());
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward_t(&xs, train)?;
            vec.push(xs.clone())
        }
        Ok(vec)
    }
}
