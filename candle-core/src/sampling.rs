use crate::{CpuStorage, DType, Layout, Result, Shape, Tensor, D};

struct RepeatPenalty {
    penalty: f32,
}

impl crate::CustomOp2 for RepeatPenalty {
    fn name(&self) -> &'static str {
        "repeat-penalty"
    }

    fn cpu_fwd(
        &self,
        logits_s: &CpuStorage,
        logits_l: &Layout,
        ctx_s: &CpuStorage,
        ctx_l: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let logits = match logits_s {
            CpuStorage::F32(s) => s,
            _ => crate::bail!("repeat-penalty: expected f32 logits"),
        };
        let context = match ctx_s {
            CpuStorage::U32(s) => s,
            _ => crate::bail!("repeat-penalty: expected u32 context"),
        };
        let vocab_size = logits_l.shape().elem_count();
        let ctx_size = ctx_l.shape().elem_count();
        let logits_offset = logits_l.start_offset();
        let ctx_offset = ctx_l.start_offset();

        let mut result = logits[logits_offset..logits_offset + vocab_size].to_vec();
        let penalty = self.penalty;
        for &token_id in &context[ctx_offset..ctx_offset + ctx_size] {
            if let Some(logit) = result.get_mut(token_id as usize) {
                if *logit >= 0. {
                    *logit /= penalty;
                } else {
                    *logit *= penalty;
                }
            }
        }
        let storage = crate::WithDType::to_cpu_storage_owned(result);
        Ok((storage, logits_l.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        logits_s: &crate::MetalStorage,
        logits_l: &Layout,
        ctx_s: &crate::MetalStorage,
        ctx_l: &Layout,
    ) -> Result<(crate::MetalStorage, Shape)> {
        use crate::backend::BackendStorage;

        let name = match logits_s.dtype() {
            DType::F32 => "repeat_penalty_f32",
            DType::F16 => "repeat_penalty_f16",
            DType::BF16 => "repeat_penalty_bf16",
            dt => crate::bail!("repeat-penalty: unsupported dtype {dt:?}"),
        };
        let device = logits_s.device();
        let kernels = device.kernels();
        let command_encoder = device.command_encoder()?;
        let vocab_size = logits_l.shape().elem_count();
        let ctx_size = ctx_l.shape().elem_count();
        let output = device.new_buffer(vocab_size, logits_s.dtype(), "repeat-penalty")?;
        let input = crate::metal_backend::buffer_o(logits_s.buffer(), logits_l, logits_s.dtype());
        candle_metal_kernels::call_repeat_penalty(
            device.metal_device(),
            &command_encoder,
            kernels,
            name,
            vocab_size,
            input,
            &output,
            ctx_s.buffer(),
            ctx_size,
            self.penalty,
        )
        .map_err(crate::Error::wrap)?;
        let output = crate::MetalStorage::new(output, device.clone(), vocab_size, logits_s.dtype());
        Ok((output, logits_l.shape().clone()))
    }
}

impl Tensor {
    /// Apply repetition penalty to logits.
    ///
    /// `context` must be a 1D U32 tensor of deduped previously seen token ids.
    pub fn apply_repeat_penalty(&self, context: &Tensor, penalty: f32) -> Result<Tensor> {
        debug_assert_eq!(context.dims().len(), 1);
        self.apply_op2_no_bwd(context, &RepeatPenalty { penalty })
    }

    /// Zero out logits for all but the top `k` tokens (set to -inf).
    /// Returns the tensor unchanged when `k >= vocab_size`.
    pub fn apply_topk_mask(&self, k: usize) -> Result<Tensor> {
        let vocab_size = self.elem_count();
        if k >= vocab_size {
            return Ok(self.clone());
        }
        let (sorted, _) = self.sort_last_dim(false)?;
        // kth-largest value is at index k-1 in the desc sorted tensor.
        let threshold = sorted.narrow(0, k - 1, 1)?.broadcast_as(self.shape())?;
        let neg_inf =
            Tensor::full(f32::NEG_INFINITY, self.shape(), self.device())?.to_dtype(self.dtype())?;
        self.ge(&threshold)?.where_cond(self, &neg_inf)
    }

    /// Zero out logits for tokens outside the top-p nucleus (set to -inf).
    ///
    /// Computes softmax probabilities, sort desc, masks all tokens whose
    /// exclusive prefix-sum of probabilities is >= `p`.
    /// Ensures at least one token is always kept.
    pub fn apply_topp_mask(&self, p: f64) -> Result<Tensor> {
        let vocab_size = self.elem_count();
        // TODO: candle-nn is not available in candle-core. Should move to candle-transformers as fns.
        let max = self.max_keepdim(D::Minus1)?;
        let exp = self.broadcast_sub(&max)?.exp()?;
        let probs = exp.broadcast_div(&exp.sum_keepdim(D::Minus1)?)?;
        let (sorted_probs, sorted_idx) = probs.sort_last_dim(false)?;

        // Exclusive prefix-sum
        let cumsum = sorted_probs.cumsum(D::Minus1)?;
        let zero = Tensor::zeros(&[1], probs.dtype(), self.device())?;
        let shifted = Tensor::cat(&[&zero, &cumsum.narrow(0, 0, vocab_size - 1)?], 0)?;

        // Keep a token if the probability mass before it is still < p.
        let keep_sorted = shifted.lt(p as f32)?;

        // Scatter the sorted keep-mask back to the original vocabulary positions.
        let keep_orig = Tensor::zeros(&[vocab_size], DType::U8, self.device())?.scatter(
            &sorted_idx,
            &keep_sorted,
            D::Minus1,
        )?;

        let neg_inf =
            Tensor::full(f32::NEG_INFINITY, self.shape(), self.device())?.to_dtype(self.dtype())?;
        keep_orig.where_cond(self, &neg_inf)
    }
}
