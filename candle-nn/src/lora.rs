//! LoRA (Low-Rank Adaptation) layers.
//!
//! This module provides a `Linear` wrapper that adds one or more low-rank
//! adapters on top of a frozen base layer, computing
//! `y = W·x + (B·A)·x·(alpha/r)` for the currently active adapter.
//!
//! Adapters can be loaded from PEFT-style `.safetensors` checkpoints (tensors
//! named `lora_A.weight` of shape `[r, in_dim]` and `lora_B.weight` of shape
//! `[out_dim, r]`) via [`LoraLinear::from_peft`] / [`LoraLinear::load_adapter`],
//! and several adapters can coexist so that the active one can be swapped at
//! runtime with [`LoraLinear::set_active_adapter`].
//!
//! ```rust
//! use candle::{Device, Tensor};
//! use candle_nn::{linear, lora::LoraLinear, Module, VarBuilder};
//! # fn main() -> candle::Result<()> {
//! let dev = &Device::Cpu;
//! let varmap = candle_nn::VarMap::new();
//! let vb = VarBuilder::from_varmap(&varmap, candle::DType::F32, dev);
//! let base = linear(8, 4, vb.pp("q_proj"))?;
//! let mut lora = LoraLinear::new(base);
//! lora.add_adapter("default", vb.pp("lora.q_proj"), 2, 16.0)?;
//! let x = Tensor::zeros((1, 8), candle::DType::F32, dev)?;
//! let y = lora.forward(&x)?; // base + LoRA delta
//! # Ok(()) }
//! ```
use std::collections::HashMap;

use candle::{Result, Tensor};

use crate::{Init, Linear, Module, VarBuilder};

/// A single low-rank adapter: `B [out, r]`, `A [r, in]` and the `alpha/r` scaling.
#[derive(Clone, Debug)]
struct Adapter {
    lora_a: Tensor,
    lora_b: Tensor,
    scale: f64,
}

impl Adapter {
    /// `(B·A) * scale`, i.e. the delta that gets added to the base weight.
    fn delta_weight(&self) -> Result<Tensor> {
        self.lora_b.matmul(&self.lora_a)?.affine(self.scale, 0.)
    }

    /// `(x·A^T)·B^T * scale`, i.e. the LoRA contribution to the output.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let lora_a = self.lora_a.t()?;
        let lora_b = self.lora_b.t()?;
        x.broadcast_matmul(&lora_a)?
            .broadcast_matmul(&lora_b)?
            .affine(self.scale, 0.)
    }
}

/// A `Linear` layer augmented with zero or more named LoRA adapters.
///
/// At most one adapter is active at a time (see [`Self::set_active_adapter`]).
/// The active adapter can be folded into the base weight with [`Self::merge`]
/// for inference, and restored with [`Self::unmerge`].
#[derive(Clone, Debug)]
pub struct LoraLinear {
    base: Linear,
    adapters: HashMap<String, Adapter>,
    active: Option<String>,
    merged: Option<String>,
}

impl LoraLinear {
    /// Wrap a base `Linear` layer with no adapters attached.
    pub fn new(base: Linear) -> Self {
        Self {
            base,
            adapters: HashMap::new(),
            active: None,
            merged: None,
        }
    }

    /// Load a PEFT-style adapter (a freshly initialized `B` matrix and a
    /// Kaiming-initialized `A` matrix, as is standard for LoRA) under `name`,
    /// pulling `lora_A.weight` and `lora_B.weight` from `vb`, and make it the
    /// active adapter.
    pub fn from_peft(base: Linear, vb: VarBuilder, rank: usize, alpha: f64) -> Result<Self> {
        let mut lora = Self::new(base);
        lora.load_adapter("default", vb, rank, alpha)?;
        lora.set_active_adapter(Some("default"))?;
        Ok(lora)
    }

    /// Load (or create, if the tensors are not present in `vb`) an adapter
    /// named `name` with the given `rank` and `alpha`, without changing which
    /// adapter is currently active.
    pub fn load_adapter(
        &mut self,
        name: &str,
        vb: VarBuilder,
        rank: usize,
        alpha: f64,
    ) -> Result<()> {
        let (out_dim, in_dim) = self.base.weight().dims2()?;
        let lora_a = vb.get_with_hints(
            (rank, in_dim),
            "lora_A.weight",
            crate::init::DEFAULT_KAIMING_UNIFORM,
        )?;
        let lora_b = vb.get_with_hints((out_dim, rank), "lora_B.weight", Init::Const(0.))?;
        self.add_adapter_tensors(name, lora_a, lora_b, alpha)
    }

    /// Load an adapter (see [`Self::load_adapter`]) and make it the active
    /// adapter.
    pub fn add_adapter(
        &mut self,
        name: &str,
        vb: VarBuilder,
        rank: usize,
        alpha: f64,
    ) -> Result<()> {
        self.load_adapter(name, vb, rank, alpha)?;
        self.set_active_adapter(Some(name))
    }

    fn add_adapter_tensors(
        &mut self,
        name: &str,
        lora_a: Tensor,
        lora_b: Tensor,
        alpha: f64,
    ) -> Result<()> {
        let rank = lora_a.dim(0)?;
        let scale = alpha / rank as f64;
        self.adapters.insert(
            name.to_string(),
            Adapter {
                lora_a,
                lora_b,
                scale,
            },
        );
        Ok(())
    }

    /// Remove an adapter. If it was the active or merged one, it is
    /// unmerged/deactivated first.
    pub fn remove_adapter(&mut self, name: &str) -> Result<()> {
        if self.merged.as_deref() == Some(name) {
            self.unmerge()?;
        }
        if self.active.as_deref() == Some(name) {
            self.active = None;
        }
        self.adapters.remove(name);
        Ok(())
    }

    /// Select the active adapter (or `None` to bypass all adapters and use
    /// only the base layer). Fails if `name` is not a known adapter, or if a
    /// different adapter is currently merged into the base weight.
    pub fn set_active_adapter(&mut self, name: Option<&str>) -> Result<()> {
        if let Some(name) = name {
            if !self.adapters.contains_key(name) {
                candle::bail!("no such LoRA adapter: {name}");
            }
        }
        if let Some(merged) = &self.merged {
            if name != Some(merged.as_str()) {
                candle::bail!(
                    "adapter {merged} is merged into the base weight, unmerge it before switching"
                );
            }
        }
        self.active = name.map(|s| s.to_string());
        Ok(())
    }

    /// Name of the currently active adapter, if any.
    pub fn active_adapter(&self) -> Option<&str> {
        self.active.as_deref()
    }

    /// Names of all the adapters currently registered.
    pub fn adapter_names(&self) -> Vec<&str> {
        self.adapters.keys().map(|s| s.as_str()).collect()
    }

    /// Fold the active adapter's weights into the base layer, so that
    /// `forward` no longer needs to compute the low-rank update separately.
    /// This is a no-op if no adapter is active or one is already merged.
    pub fn merge(&mut self) -> Result<()> {
        if self.merged.is_some() {
            return Ok(());
        }
        let Some(active) = self.active.clone() else {
            return Ok(());
        };
        let adapter = &self.adapters[&active];
        let weight = self.base.weight().add(&adapter.delta_weight()?)?;
        self.base = Linear::new(weight, self.base.bias().cloned());
        self.merged = Some(active);
        Ok(())
    }

    /// Undo a previous [`Self::merge`], restoring the base layer's original
    /// weights. This is a no-op if no adapter is currently merged.
    pub fn unmerge(&mut self) -> Result<()> {
        let Some(merged) = self.merged.take() else {
            return Ok(());
        };
        let adapter = &self.adapters[&merged];
        let weight = self.base.weight().sub(&adapter.delta_weight()?)?;
        self.base = Linear::new(weight, self.base.bias().cloned());
        Ok(())
    }

    /// Whether an adapter is currently merged into the base weight.
    pub fn is_merged(&self) -> bool {
        self.merged.is_some()
    }

    /// The underlying base layer.
    pub fn base(&self) -> &Linear {
        &self.base
    }
}

impl Module for LoraLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let base_out = self.base.forward(x)?;
        if self.merged.is_some() {
            return Ok(base_out);
        }
        match &self.active {
            None => Ok(base_out),
            Some(name) => base_out.add(&self.adapters[name].forward(x)?),
        }
    }
}
