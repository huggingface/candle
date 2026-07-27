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
//! For serving scenarios where a single batch mixes sequences that target
//! different adapters (S-LoRA / Punica style multi-tenant inference),
//! [`LoraLinear::forward_with_adapters`] applies a per-row adapter selection
//! in one batched computation instead of one forward per adapter group.
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

    fn rank(&self) -> Result<usize> {
        self.lora_a.dim(0)
    }

    /// `A^T [in, r]` zero-padded along the rank dimension to `r_max`.
    fn a_t_padded(&self, r_max: usize) -> Result<Tensor> {
        let a_t = self.lora_a.t()?.contiguous()?;
        let (in_dim, r) = a_t.dims2()?;
        if r == r_max {
            return Ok(a_t);
        }
        let pad = Tensor::zeros((in_dim, r_max - r), a_t.dtype(), a_t.device())?;
        Tensor::cat(&[a_t, pad], 1)
    }

    /// `B^T * scale [r, out]` zero-padded along the rank dimension to `r_max`.
    fn b_t_scaled_padded(&self, r_max: usize) -> Result<Tensor> {
        let b_t = self.lora_b.t()?.affine(self.scale, 0.)?;
        let (r, out_dim) = b_t.dims2()?;
        if r == r_max {
            return Ok(b_t);
        }
        let pad = Tensor::zeros((r_max - r, out_dim), b_t.dtype(), b_t.device())?;
        Tensor::cat(&[b_t, pad], 0)
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
        // Overwriting the adapter that is currently merged into the base weight
        // would make `unmerge` subtract the new delta from a base that still
        // holds the old one, corrupting the weights. Refuse it, as
        // `set_active_adapter` does for switching.
        if self.merged.as_deref() == Some(name) {
            candle::bail!(
                "adapter {name} is merged into the base weight, unmerge it before overwriting"
            )
        }
        let rank = lora_a.dim(0)?;
        let scale = alpha / rank as f64;
        // Align the adapter tensors with the base weight's dtype so the plain
        // `forward` (and `merge`/`delta_weight`) never hit a mixed-dtype matmul
        // when a PEFT checkpoint is stored in a different dtype than the model.
        let dtype = self.base.weight().dtype();
        let lora_a = lora_a.to_dtype(dtype)?;
        let lora_b = lora_b.to_dtype(dtype)?;
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

    /// Forward pass over a heterogeneous batch where each row selects its own
    /// adapter (S-LoRA / Punica style multi-adapter batching).
    ///
    /// `x` has shape `[batch, in_dim]` or `[batch, seq, in_dim]` and
    /// `assignments` holds, for each batch row, either `Some(adapter_name)` or
    /// `None` for rows that should only go through the base layer. The
    /// currently active adapter is ignored by this method.
    ///
    /// Rather than splitting the batch and running one forward per adapter
    /// group, the selected `A`/`B` matrices are gathered into per-row stacks
    /// (zero-padded to the largest rank, with the `alpha/r` scaling folded in)
    /// and applied with two batched matmuls, SGMV-style:
    /// `y = W·x + gather(B_all)·gather(A_all)·x`. Rows with no adapter hit an
    /// all-zero slot, so the base-only path is preserved exactly and the
    /// result matches running each row separately with its own adapter.
    pub fn forward_with_adapters(
        &self,
        x: &Tensor,
        assignments: &[Option<&str>],
    ) -> Result<Tensor> {
        if let Some(merged) = &self.merged {
            candle::bail!(
                "cannot use forward_with_adapters while adapter {merged} is merged into the base \
                 weight, call unmerge first"
            )
        }
        let batch = x.dim(0)?;
        if assignments.len() != batch {
            candle::bail!(
                "forward_with_adapters: {} adapter assignments for a batch of {batch} rows",
                assignments.len()
            )
        }
        let base_out = self.base.forward(x)?;
        // Map each row to a slot in the gathered stacks; slot 0 is an all-zero
        // adapter used by the `None` rows.
        let mut involved: Vec<&str> = Vec::new();
        let mut row_slots: Vec<u32> = Vec::with_capacity(batch);
        for assignment in assignments {
            match assignment {
                None => row_slots.push(0),
                Some(name) => {
                    if !self.adapters.contains_key(*name) {
                        candle::bail!("no such LoRA adapter: {name}")
                    }
                    let slot = match involved.iter().position(|n| n == name) {
                        Some(i) => i + 1,
                        None => {
                            involved.push(name);
                            involved.len()
                        }
                    };
                    row_slots.push(slot as u32);
                }
            }
        }
        if involved.is_empty() {
            return Ok(base_out);
        }
        let (out_dim, in_dim) = self.base.weight().dims2()?;
        let r_max = involved
            .iter()
            .map(|name| self.adapters[*name].rank())
            .try_fold(0, |acc, r| r.map(|r| acc.max(r)))?;
        let (dtype, device) = (x.dtype(), x.device());
        let mut a_stack = vec![Tensor::zeros((in_dim, r_max), dtype, device)?];
        let mut b_stack = vec![Tensor::zeros((r_max, out_dim), dtype, device)?];
        // Adapter tensors are stored in the base weight's dtype (see
        // `add_adapter_tensors`), which equals `x`'s dtype here since
        // `base.forward(x)` above already ran, so the stacks are homogeneous
        // with the zero slot without a per-adapter cast.
        for name in involved {
            let adapter = &self.adapters[name];
            a_stack.push(adapter.a_t_padded(r_max)?);
            b_stack.push(adapter.b_t_scaled_padded(r_max)?);
        }
        let row_slots = Tensor::from_vec(row_slots, batch, device)?;
        let a_stack = Tensor::stack(&a_stack, 0)?;
        let b_stack = Tensor::stack(&b_stack, 0)?;
        let x3 = match x.rank() {
            2 => x.unsqueeze(1)?,
            3 => x.clone(),
            rank => candle::bail!(
                "forward_with_adapters expects a [batch, in] or [batch, seq, in] input, got a \
                 rank {rank} tensor"
            ),
        };
        let delta = batched_lora_delta(&x3, &a_stack, &b_stack, &row_slots)?;
        let delta = if x.rank() == 2 {
            delta.squeeze(1)?
        } else {
            delta
        };
        base_out.add(&delta)
    }
}

/// Computes the LoRA delta for a prepared heterogeneous batch:
/// `delta[i] = x[i] · a_stack[slot(i)] · b_stack[slot(i)]` for every batch
/// row `i`, where `x` is `[batch, seq, in]`, `a_stack` is
/// `[n_slots, in, r_max]` (`A^T`, rank-padded with zeros), `b_stack` is
/// `[n_slots, r_max, out]` (`B^T`, rank-padded, `alpha/r` scaling folded in,
/// slot 0 all-zero for no-adapter rows) and `row_slots` is a `[batch]` u32
/// tensor mapping every row to its slot.
///
/// This is the seam a fused SGMV/BGMV kernel can replace: the reference
/// implementation below gathers per-row copies of the adapter matrices and
/// applies them with two batched matmuls, whereas a dedicated kernel can read
/// the stacks in place through `row_slots` without materializing the gather.
///
/// With the `lora-cuda` feature, a decode-step batch (`seq == 1`) on a CUDA
/// device is routed through the [`candle_lora_kernels`] BGMV kernels, which
/// return the same delta; every other case uses the reference path below.
fn batched_lora_delta(
    x: &Tensor,
    a_stack: &Tensor,
    b_stack: &Tensor,
    row_slots: &Tensor,
) -> Result<Tensor> {
    #[cfg(feature = "lora-cuda")]
    {
        let (batch, seq, in_dim) = x.dims3()?;
        let supported = matches!(
            x.dtype(),
            candle::DType::F32 | candle::DType::F16 | candle::DType::BF16
        );
        if seq == 1 && x.device().is_cuda() && supported {
            let x2d = x.reshape((batch, in_dim))?;
            let delta = candle_lora_kernels::bgmv_delta(&x2d, a_stack, b_stack, row_slots)?;
            let out_dim = delta.dim(1)?;
            return delta.reshape((batch, 1, out_dim));
        }
    }
    // [batch, in, r_max] and [batch, r_max, out].
    let a_sel = a_stack.index_select(row_slots, 0)?;
    let b_sel = b_stack.index_select(row_slots, 0)?;
    x.contiguous()?.matmul(&a_sel)?.matmul(&b_sel)
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
