//! Various optimization algorithms.
use candle::{Result, Tensor, Var};

/// The interface optimizers should implement.
pub trait Optimizer: Sized {
    type Config: Sized;

    fn new(vars: Vec<Var>, config: Self::Config) -> Result<Self>;

    fn step(&mut self, grads: &candle::backprop::GradStore) -> Result<()>;

    fn learning_rate(&self) -> f64;

    fn set_learning_rate(&mut self, lr: f64);

    fn empty(config: Self::Config) -> Result<Self> {
        Self::new(vec![], config)
    }

    fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        self.step(&grads)
    }

    fn from_slice(vars: &[&Var], config: Self::Config) -> Result<Self> {
        let vars: Vec<_> = vars.iter().map(|&v| v.clone()).collect();
        Self::new(vars, config)
    }
}

/// Optimizer for Stochastic Gradient Descent.
///
/// Contrary to the PyTorch implementation of SGD, this version does not support momentum.
#[derive(Debug)]
pub struct SGD {
    vars: Vec<Var>,
    learning_rate: f64,
}

impl Optimizer for SGD {
    type Config = f64;

    fn new(vars: Vec<Var>, learning_rate: f64) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .collect();
        Ok(Self {
            vars,
            learning_rate,
        })
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn step(&mut self, grads: &candle::backprop::GradStore) -> Result<()> {
        for var in self.vars.iter() {
            if let Some(grad) = grads.get(var) {
                var.set(&var.sub(&(grad * self.learning_rate)?)?)?;
            }
        }
        Ok(())
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr
    }
}

impl SGD {
    pub fn into_inner(self) -> Vec<Var> {
        self.vars
    }

    pub fn push(&mut self, var: &Var) {
        self.vars.push(var.clone())
    }
}

#[derive(Clone, Debug)]
pub struct ParamsAdamW {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl Default for ParamsAdamW {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

#[derive(Debug)]
struct VarAdamW {
    var: Var,
    first_moment: Var,
    second_moment: Var,
}

#[derive(Debug)]
pub struct AdamW {
    vars: Vec<VarAdamW>,
    step_t: usize,
    params: ParamsAdamW,
}

impl Optimizer for AdamW {
    type Config = ParamsAdamW;

    fn new(vars: Vec<Var>, params: ParamsAdamW) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                let dtype = var.dtype();
                let shape = var.shape();
                let device = var.device();
                let first_moment = Var::zeros(shape, dtype, device)?;
                let second_moment = Var::zeros(shape, dtype, device)?;
                Ok(VarAdamW {
                    var,
                    first_moment,
                    second_moment,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            vars,
            params,
            step_t: 0,
        })
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr
    }

    fn step(&mut self, grads: &candle::backprop::GradStore) -> Result<()> {
        self.step_t += 1;
        let lr = self.params.lr;
        let lambda = self.params.weight_decay;
        let lr_lambda = lr * lambda;
        let beta1 = self.params.beta1;
        let beta2 = self.params.beta2;
        let scale_m = 1f64 / (1f64 - beta1.powi(self.step_t as i32));
        let scale_v = 1f64 / (1f64 - beta2.powi(self.step_t as i32));
        for var in self.vars.iter() {
            let theta = &var.var;
            let m = &var.first_moment;
            let v = &var.second_moment;
            if let Some(g) = grads.get(theta) {
                // This involves locking 3 RWLocks per params, if the parameters are large this
                // should not be an issue but this may be problematic with models with lots of
                // small parameters.
                let next_m = ((m.as_tensor() * beta1)? + (g * (1.0 - beta1))?)?;
                let next_v = ((v.as_tensor() * beta2)? + (g.sqr()? * (1.0 - beta2))?)?;
                let m_hat = (&next_m * scale_m)?;
                let v_hat = (&next_v * scale_v)?;
                let next_theta = (theta.as_tensor() * (1f64 - lr_lambda))?;
                let adjusted_grad = (m_hat / (v_hat.sqrt()? + self.params.eps)?)?;
                let next_theta = (next_theta - (adjusted_grad * lr)?)?;
                m.set(&next_m)?;
                v.set(&next_v)?;
                theta.set(&next_theta)?;
            }
        }
        Ok(())
    }
}

impl AdamW {
    pub fn new_lr(vars: Vec<Var>, learning_rate: f64) -> Result<Self> {
        let params = ParamsAdamW {
            lr: learning_rate,
            ..ParamsAdamW::default()
        };
        Self::new(vars, params)
    }

    pub fn params(&self) -> &ParamsAdamW {
        &self.params
    }

    pub fn set_params(&mut self, params: ParamsAdamW) {
        self.params = params;
    }
}

/// Wraps any [`Optimizer`] with gradient-accumulation semantics.
///
/// Call [`accumulate`](Self::accumulate) once per micro-batch to compute
/// and merge that batch's gradients into an internal
/// [`GradStore`](candle::backprop::GradStore), then call
/// [`step`](Self::step) once to average the K accumulated gradients and
/// apply a single optimizer update. This yields an effective batch size
/// of `K × micro_batch` without holding K computation graphs alive
/// simultaneously.
///
/// The accumulated gradients are divided by K at flush time, so the
/// resulting parameter update equals what a single backward pass on the
/// mean of the K per-micro-batch losses would produce. If each
/// micro-batch loss is already a mean over its samples, this matches the
/// gradient of a K× larger mean-reduced batch. If your losses are sums,
/// divide by the total sample count yourself.
///
/// Composition-based by design: works with [`SGD`], [`AdamW`], and any
/// future optimizer implementing [`Optimizer`] — no per-optimizer code.
///
/// ```no_run
/// use candle_nn::optim::{AdamW, GradAccumulator, Optimizer, ParamsAdamW};
/// # use candle::{Result, Tensor, Var};
/// # fn example(params: ParamsAdamW, vars: Vec<Var>, micro_batches: Vec<Tensor>) -> Result<()> {
/// let mut opt = GradAccumulator::new(AdamW::new(vars, params)?);
/// for loss in &micro_batches {
///     opt.accumulate(loss)?;
/// }
/// opt.step()?; // one update using the mean of all micro-batch gradients
/// # Ok(()) }
/// ```
pub struct GradAccumulator<O: Optimizer> {
    opt: O,
    accum: Option<candle::backprop::GradStore>,
    count: usize,
}

impl<O: Optimizer> GradAccumulator<O> {
    pub fn new(opt: O) -> Self {
        Self {
            opt,
            accum: None,
            count: 0,
        }
    }

    /// Number of micro-batches currently accumulated but not yet stepped.
    pub fn pending(&self) -> usize {
        self.count
    }

    pub fn optimizer(&self) -> &O {
        &self.opt
    }

    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.opt
    }

    pub fn into_inner(self) -> O {
        self.opt
    }

    /// Compute the gradient of `loss` and merge it into the accumulator.
    pub fn accumulate(&mut self, loss: &Tensor) -> Result<()> {
        let fresh = loss.backward()?;
        match &mut self.accum {
            // First micro-batch: adopt the fresh store as the persistent
            // accumulator. Detach each tensor defensively — GradStore
            // entries don't currently carry op history, but guarding here
            // keeps the invariant stable against future candle-core
            // changes that might introduce it, so the accumulator can
            // never chain graphs across K iterations.
            None => {
                let ids: Vec<_> = fresh.get_ids().copied().collect();
                let mut store = fresh;
                for id in ids {
                    let detached = store.get_id(id).unwrap().detach();
                    store.insert_id(id, detached);
                }
                self.accum = Some(store);
            }
            Some(existing) => {
                let ids: Vec<_> = fresh.get_ids().copied().collect();
                for id in ids {
                    let g = fresh.get_id(id).unwrap().clone();
                    let merged = match existing.get_id(id) {
                        Some(prev) => (prev + &g)?.detach(),
                        None => g.detach(),
                    };
                    existing.insert_id(id, merged);
                }
            }
        }
        self.count += 1;
        Ok(())
    }

    /// Scale the accumulated gradients by 1/K and apply a single
    /// optimizer step. Resets the accumulator on success. No-op if
    /// nothing is pending.
    ///
    /// On error the accumulator state is preserved so the caller may
    /// retry `step()`. Scaling is atomic — a mid-scale failure leaves
    /// the buffer untouched, and once scaling has committed the count
    /// collapses to 1 so a subsequent retry scales by `1/1` rather than
    /// re-applying `1/K` to already-averaged gradients.
    ///
    /// After a failed `step()`, prefer retrying `step()` over calling
    /// [`accumulate`](Self::accumulate) — adding new micro-batches on
    /// top of an already-averaged buffer mixes scales. Call
    /// [`reset`](Self::reset) if you want to discard the pending state
    /// and start over.
    pub fn step(&mut self) -> Result<()> {
        if self.count == 0 {
            return Ok(());
        }
        let Some(store) = self.accum.as_mut() else {
            return Ok(());
        };
        let scale = 1.0 / self.count as f64;
        // Compute all scaled tensors up-front. If any scaling op errors
        // the buffer is untouched, so retry is safe.
        let ids: Vec<_> = store.get_ids().copied().collect();
        let mut scaled: Vec<(_, Tensor)> = Vec::with_capacity(ids.len());
        for id in &ids {
            let g = store.get_id(*id).unwrap();
            scaled.push((*id, (g * scale)?));
        }
        // All scaling succeeded — commit. The buffer now holds one
        // averaged gradient; collapsing count to 1 means that if
        // opt.step errors below and the caller retries, scaling will be
        // a no-op (1/1) rather than wrongly re-applying 1/K.
        for (id, g) in scaled {
            store.insert_id(id, g);
        }
        self.count = 1;
        self.opt.step(store)?;
        self.accum = None;
        self.count = 0;
        Ok(())
    }

    /// Discard pending gradients without applying them.
    pub fn reset(&mut self) {
        self.accum = None;
        self.count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, Tensor};

    /// Accumulating K micro-batches through GradAccumulator<AdamW> then
    /// stepping once must match a single backward_step on the mean of
    /// those K losses.
    #[test]
    fn adamw_accumulate_matches_single_step() {
        let dev = Device::Cpu;
        let params = ParamsAdamW {
            lr: 0.01,
            weight_decay: 0.0,
            ..Default::default()
        };

        let w1 = Var::from_tensor(&Tensor::new(&[1.0f32, 2.0, 3.0], &dev).unwrap()).unwrap();
        let w2 = Var::from_tensor(&Tensor::new(&[1.0f32, 2.0, 3.0], &dev).unwrap()).unwrap();

        let x1 = Tensor::new(&[0.5f32, -0.5, 1.0], &dev).unwrap();
        let x2 = Tensor::new(&[-1.0f32, 0.3, 0.7], &dev).unwrap();

        // Path A: accumulate 2 micro-batches, step once.
        let mut acc = GradAccumulator::new(AdamW::new(vec![w1.clone()], params.clone()).unwrap());
        let l1a = (w1.as_tensor() * &x1).unwrap().sum_all().unwrap();
        let l2a = (w1.as_tensor() * &x2).unwrap().sum_all().unwrap();
        acc.accumulate(&l1a).unwrap();
        acc.accumulate(&l2a).unwrap();
        acc.step().unwrap();
        let result_a: Vec<f32> = w1.as_tensor().to_vec1().unwrap();

        // Path B: single backward_step on the mean of both losses.
        let mut opt_b = AdamW::new(vec![w2.clone()], params).unwrap();
        let l1b = (w2.as_tensor() * &x1).unwrap().sum_all().unwrap();
        let l2b = (w2.as_tensor() * &x2).unwrap().sum_all().unwrap();
        let combined = ((&l1b + &l2b).unwrap() * 0.5).unwrap();
        opt_b.backward_step(&combined).unwrap();
        let result_b: Vec<f32> = w2.as_tensor().to_vec1().unwrap();

        for (a, b) in result_a.iter().zip(result_b.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "accumulate diverged from single step: {a} vs {b}"
            );
        }
    }

    /// The whole selling point: GradAccumulator works unchanged over SGD.
    #[test]
    fn sgd_accumulate_matches_single_step() {
        let dev = Device::Cpu;
        let lr = 0.1;

        let w1 = Var::from_tensor(&Tensor::new(&[1.0f32, 2.0, 3.0], &dev).unwrap()).unwrap();
        let w2 = Var::from_tensor(&Tensor::new(&[1.0f32, 2.0, 3.0], &dev).unwrap()).unwrap();

        let x1 = Tensor::new(&[0.5f32, -0.5, 1.0], &dev).unwrap();
        let x2 = Tensor::new(&[-1.0f32, 0.3, 0.7], &dev).unwrap();

        let mut acc = GradAccumulator::new(SGD::new(vec![w1.clone()], lr).unwrap());
        let l1a = (w1.as_tensor() * &x1).unwrap().sum_all().unwrap();
        let l2a = (w1.as_tensor() * &x2).unwrap().sum_all().unwrap();
        acc.accumulate(&l1a).unwrap();
        acc.accumulate(&l2a).unwrap();
        acc.step().unwrap();
        let result_a: Vec<f32> = w1.as_tensor().to_vec1().unwrap();

        let mut sgd = SGD::new(vec![w2.clone()], lr).unwrap();
        let l1b = (w2.as_tensor() * &x1).unwrap().sum_all().unwrap();
        let l2b = (w2.as_tensor() * &x2).unwrap().sum_all().unwrap();
        let combined = ((&l1b + &l2b).unwrap() * 0.5).unwrap();
        sgd.backward_step(&combined).unwrap();
        let result_b: Vec<f32> = w2.as_tensor().to_vec1().unwrap();

        for (a, b) in result_a.iter().zip(result_b.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "SGD accumulate diverged from single step: {a} vs {b}"
            );
        }
    }

    #[test]
    fn step_with_no_accumulation_is_noop() {
        let dev = Device::Cpu;
        let w = Var::from_tensor(&Tensor::new(&[1.0f32, 2.0], &dev).unwrap()).unwrap();
        let mut acc =
            GradAccumulator::new(AdamW::new(vec![w.clone()], ParamsAdamW::default()).unwrap());
        let before: Vec<f32> = w.as_tensor().to_vec1().unwrap();
        acc.step().unwrap();
        let after: Vec<f32> = w.as_tensor().to_vec1().unwrap();
        assert_eq!(before, after);
        assert_eq!(acc.pending(), 0);
    }

    #[test]
    fn accumulate_resets_after_step() {
        let dev = Device::Cpu;
        let w = Var::from_tensor(&Tensor::new(&[1.0f32, 2.0], &dev).unwrap()).unwrap();
        let mut acc =
            GradAccumulator::new(AdamW::new(vec![w.clone()], ParamsAdamW::default()).unwrap());
        let x = Tensor::new(&[1.0f32, 1.0], &dev).unwrap();
        let loss = (w.as_tensor() * &x).unwrap().sum_all().unwrap();
        acc.accumulate(&loss).unwrap();
        assert_eq!(acc.pending(), 1);
        acc.step().unwrap();
        assert_eq!(acc.pending(), 0);
        assert!(acc.accum.is_none());
    }

    /// Multi-variable parity with weight_decay != 0, exercising the
    /// decoupled-decay path.
    #[test]
    fn accumulate_multi_var_with_weight_decay() {
        let dev = Device::Cpu;
        let params = ParamsAdamW {
            lr: 0.01,
            weight_decay: 0.1,
            ..Default::default()
        };

        let wa1 = Var::from_tensor(&Tensor::new(&[1.0f32, -2.0, 3.0], &dev).unwrap()).unwrap();
        let wa2 = Var::from_tensor(&Tensor::new(&[0.5f32, 0.5], &dev).unwrap()).unwrap();
        let wb1 = Var::from_tensor(&Tensor::new(&[1.0f32, -2.0, 3.0], &dev).unwrap()).unwrap();
        let wb2 = Var::from_tensor(&Tensor::new(&[0.5f32, 0.5], &dev).unwrap()).unwrap();

        let x1 = Tensor::new(&[0.5f32, -0.5, 1.0], &dev).unwrap();
        let x2 = Tensor::new(&[-1.0f32, 0.3, 0.7], &dev).unwrap();
        let y1 = Tensor::new(&[1.0f32, 0.5], &dev).unwrap();
        let y2 = Tensor::new(&[-0.5f32, 2.0], &dev).unwrap();

        let mut acc = GradAccumulator::new(
            AdamW::new(vec![wa1.clone(), wa2.clone()], params.clone()).unwrap(),
        );
        let l1a = ((wa1.as_tensor() * &x1).unwrap().sum_all().unwrap()
            + (wa2.as_tensor() * &y1).unwrap().sum_all().unwrap())
        .unwrap();
        let l2a = ((wa1.as_tensor() * &x2).unwrap().sum_all().unwrap()
            + (wa2.as_tensor() * &y2).unwrap().sum_all().unwrap())
        .unwrap();
        acc.accumulate(&l1a).unwrap();
        acc.accumulate(&l2a).unwrap();
        acc.step().unwrap();

        let mut opt_b = AdamW::new(vec![wb1.clone(), wb2.clone()], params).unwrap();
        let l1b = ((wb1.as_tensor() * &x1).unwrap().sum_all().unwrap()
            + (wb2.as_tensor() * &y1).unwrap().sum_all().unwrap())
        .unwrap();
        let l2b = ((wb1.as_tensor() * &x2).unwrap().sum_all().unwrap()
            + (wb2.as_tensor() * &y2).unwrap().sum_all().unwrap())
        .unwrap();
        let combined = ((&l1b + &l2b).unwrap() * 0.5).unwrap();
        opt_b.backward_step(&combined).unwrap();

        let a1: Vec<f32> = wa1.as_tensor().to_vec1().unwrap();
        let b1: Vec<f32> = wb1.as_tensor().to_vec1().unwrap();
        let a2: Vec<f32> = wa2.as_tensor().to_vec1().unwrap();
        let b2: Vec<f32> = wb2.as_tensor().to_vec1().unwrap();
        for (a, b) in a1.iter().zip(b1.iter()) {
            assert!((a - b).abs() < 1e-5, "var1 diverged: {a} vs {b}");
        }
        for (a, b) in a2.iter().zip(b2.iter()) {
            assert!((a - b).abs() < 1e-5, "var2 diverged: {a} vs {b}");
        }
    }

    /// A var that never appears in the loss graph must be left untouched.
    #[test]
    fn accumulate_skips_var_without_grad() {
        let dev = Device::Cpu;
        let used = Var::from_tensor(&Tensor::new(&[1.0f32, 2.0], &dev).unwrap()).unwrap();
        let unused = Var::from_tensor(&Tensor::new(&[7.0f32, 8.0], &dev).unwrap()).unwrap();
        let used_before: Vec<f32> = used.as_tensor().to_vec1().unwrap();
        let unused_before: Vec<f32> = unused.as_tensor().to_vec1().unwrap();

        let mut acc = GradAccumulator::new(
            AdamW::new(vec![used.clone(), unused.clone()], ParamsAdamW::default()).unwrap(),
        );
        let x = Tensor::new(&[0.5f32, 0.5], &dev).unwrap();
        let loss = (used.as_tensor() * &x).unwrap().sum_all().unwrap();
        acc.accumulate(&loss).unwrap();
        acc.accumulate(&loss).unwrap();
        acc.step().unwrap();

        let used_after: Vec<f32> = used.as_tensor().to_vec1().unwrap();
        let unused_after: Vec<f32> = unused.as_tensor().to_vec1().unwrap();
        assert_ne!(used_after, used_before, "used var should have moved");
        assert_eq!(unused_after, unused_before, "unused var must not move");
    }

    #[test]
    fn reset_discards_pending() {
        let dev = Device::Cpu;
        let w = Var::from_tensor(&Tensor::new(&[1.0f32, 2.0], &dev).unwrap()).unwrap();
        let mut acc =
            GradAccumulator::new(AdamW::new(vec![w.clone()], ParamsAdamW::default()).unwrap());
        let x = Tensor::new(&[1.0f32, 1.0], &dev).unwrap();
        let loss = (w.as_tensor() * &x).unwrap().sum_all().unwrap();
        acc.accumulate(&loss).unwrap();
        assert_eq!(acc.pending(), 1);
        acc.reset();
        assert_eq!(acc.pending(), 0);
        let before: Vec<f32> = w.as_tensor().to_vec1().unwrap();
        acc.step().unwrap();
        let after: Vec<f32> = w.as_tensor().to_vec1().unwrap();
        assert_eq!(before, after, "step after reset must not update");
    }
}
