//! Tests for variable-length flash attention implementations.
//!
//! Validates both fused (`flash_attn_varlen_cpu`) and unfused (`flash_attn_varlen_unfused`)
//! implementations against a padded reference attention.
//!
//! Ported from <https://github.com/huggingface/candle/pull/3250>.

use candle::Result;
use candle_nn::attention::flash_attn_varlen_cpu;
use candle_nn::varlen_attention::flash_attn_varlen_unfused;

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device, IndexOp, Tensor};
    use rand::prelude::*;

    // ── Implementation dispatch ─────────────────────────────────────────

    #[derive(Debug, Clone, Copy)]
    enum VarlenImpl {
        CpuFlash,
        Unfused,
    }

    impl VarlenImpl {
        fn forward(
            &self,
            q: &Tensor,
            k: &Tensor,
            v: &Tensor,
            alibi_slopes: Option<&Tensor>,
            seqlens_q: &Tensor,
            seqlens_k: &Tensor,
            max_q: usize,
            max_k: usize,
            softmax_scale: f32,
            causal: bool,
            window_left: Option<usize>,
            window_right: Option<usize>,
        ) -> Result<Tensor> {
            match self {
                VarlenImpl::CpuFlash => flash_attn_varlen_cpu(
                    q,
                    k,
                    v,
                    alibi_slopes,
                    seqlens_q,
                    seqlens_k,
                    max_q,
                    max_k,
                    softmax_scale,
                    causal,
                    window_left,
                    window_right,
                ),
                VarlenImpl::Unfused => flash_attn_varlen_unfused(
                    q,
                    k,
                    v,
                    alibi_slopes,
                    seqlens_q,
                    seqlens_k,
                    max_q,
                    max_k,
                    softmax_scale,
                    causal,
                    window_left,
                    window_right,
                ),
            }
        }
    }

    // ── Helpers ─────────────────────────────────────────────────────────

    fn convert_to_precision(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        precision: DType,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        match precision {
            DType::F32 => Ok((q.clone(), k.clone(), v.clone())),
            DType::F16 => Ok((
                q.to_dtype(DType::F16)?,
                k.to_dtype(DType::F16)?,
                v.to_dtype(DType::F16)?,
            )),
            _ => candle::bail!("Unsupported precision: {:?}", precision),
        }
    }

    fn get_tolerances(precision: DType) -> (f32, f32) {
        match precision {
            DType::F32 => (1e-5, 5e-5),
            DType::F16 => (2e-3, 1e-3),
            _ => (1e-4, 1e-4),
        }
    }

    fn rmse(a: &Tensor, reference: &Tensor) -> Result<f32> {
        let diff = a.sub(reference)?;
        let mse_tensor = diff.sqr()?.mean_all()?.to_dtype(DType::F32)?;
        let mse = mse_tensor.to_scalar::<f32>()?;
        if mse.is_nan() || mse < 0.0 {
            let a_vec = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let ref_vec = reference
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            if ref_vec.iter().any(|x| x.is_nan()) && !a_vec.iter().any(|x| x.is_nan()) {
                return Ok(0.0);
            }
            Ok(100.0)
        } else {
            Ok(mse.sqrt())
        }
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        let diff = a.sub(b)?.to_dtype(DType::F32)?;
        let v = diff.flatten_all()?.to_vec1::<f32>()?;
        Ok(v.into_iter().map(|x| x.abs()).fold(0.0f32, f32::max))
    }

    fn repeat_kv_for_gqa(k: &Tensor, v: &Tensor, num_heads: usize) -> Result<(Tensor, Tensor)> {
        let (total_k, num_kv_heads, head_dim) = k.dims3()?;
        if num_heads == num_kv_heads {
            return Ok((k.clone(), v.clone()));
        }
        if num_heads % num_kv_heads != 0 {
            candle::bail!(
                "Invalid GQA config: num_heads={} not divisible by num_kv_heads={}",
                num_heads,
                num_kv_heads
            );
        }
        let repeat_factor = num_heads / num_kv_heads;
        let k = k
            .reshape((total_k, num_kv_heads, 1, head_dim))?
            .broadcast_as((total_k, num_kv_heads, repeat_factor, head_dim))?
            .reshape((total_k, num_heads, head_dim))?;
        let v = v
            .reshape((total_k, num_kv_heads, 1, head_dim))?
            .broadcast_as((total_k, num_kv_heads, repeat_factor, head_dim))?
            .reshape((total_k, num_heads, head_dim))?;
        Ok((k, v))
    }

    // ── Input generators ────────────────────────────────────────────────

    #[allow(clippy::type_complexity)]
    fn make_varlen_inputs_prefill(
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, usize, usize)> {
        let mut rng = StdRng::seed_from_u64(456);

        let mut seqlens = Vec::<u32>::with_capacity(batch_size);
        let mut total = 0usize;
        let mut max_l = 0usize;

        for _ in 0..batch_size {
            let l = rng.random_range(4..=max_seq);
            seqlens.push(l as u32);
            total += l;
            max_l = max_l.max(l);
        }

        let q_data: Vec<f32> = (0..total * num_heads * head_dim)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let k_data: Vec<f32> = (0..total * num_kv_heads * head_dim)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let v_data: Vec<f32> = (0..total * num_kv_heads * head_dim)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();

        let q = Tensor::from_vec(q_data, (total, num_heads, head_dim), device)?;
        let k = Tensor::from_vec(k_data, (total, num_kv_heads, head_dim), device)?;
        let v = Tensor::from_vec(v_data, (total, num_kv_heads, head_dim), device)?;

        let seqlens_q = Tensor::from_vec(seqlens.clone(), batch_size, device)?;
        let seqlens_k = Tensor::from_vec(seqlens, batch_size, device)?;

        Ok((q, k, v, seqlens_q, seqlens_k, max_l, max_l))
    }

    // ── Reference implementation ────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    fn build_reference_bias(
        seqlens_q: &[u32],
        seqlens_k: &[u32],
        num_heads: usize,
        max_q: usize,
        max_k: usize,
        causal: bool,
        window_left: Option<usize>,
        window_right: Option<usize>,
        alibi_slopes: Option<&Tensor>,
        device: &Device,
    ) -> Result<Tensor> {
        let bsz = seqlens_q.len();
        let slopes = if let Some(s) = alibi_slopes {
            let v = s.to_vec1::<f32>()?;
            if v.len() != num_heads {
                candle::bail!("alibi_slopes has len {}, expected {}", v.len(), num_heads);
            }
            Some(v)
        } else {
            None
        };

        let mut per_batch = Vec::with_capacity(bsz);

        for b in 0..bsz {
            let lq = seqlens_q[b] as usize;
            let lk = seqlens_k[b] as usize;
            let offset = lk as isize - lq as isize;

            let mut bias = vec![0f32; num_heads * max_q * max_k];

            for h in 0..num_heads {
                let slope = slopes.as_ref().map(|s| s[h]).unwrap_or(0.0);

                for i in 0..max_q {
                    for j in 0..max_k {
                        let idx = h * (max_q * max_k) + i * max_k + j;

                        if i >= lq || j >= lk {
                            bias[idx] = -1e10;
                            continue;
                        }

                        if causal {
                            let ii = i as isize;
                            let jj = j as isize;
                            if jj > ii + offset {
                                bias[idx] = -1e10;
                                continue;
                            }
                        }

                        if window_left.is_some() || window_right.is_some() {
                            let i_k = i as isize + offset;
                            match (window_left, window_right) {
                                (Some(left), Some(right)) => {
                                    let left_dist = (i_k - j as isize).max(0) as usize;
                                    let right_dist = (j as isize - i_k).max(0) as usize;
                                    if left_dist > left || right_dist > right {
                                        bias[idx] = f32::NEG_INFINITY;
                                        continue;
                                    }
                                }
                                (Some(left), None) => {
                                    if (j as isize) > i_k {
                                        bias[idx] = f32::NEG_INFINITY;
                                        continue;
                                    }
                                    let dist = (i_k - j as isize) as usize;
                                    if dist > left {
                                        bias[idx] = f32::NEG_INFINITY;
                                        continue;
                                    }
                                }
                                (None, None) => {}
                                (None, Some(_)) => {
                                    candle::bail!("window_right without window_left")
                                }
                            }
                        }

                        if slopes.is_some() {
                            let i_k = i as isize + offset;
                            let dist = (i_k - j as isize).abs() as f32;
                            bias[idx] += -slope * dist;
                        }
                    }
                }
            }

            per_batch.push(Tensor::from_vec(bias, (num_heads, max_q, max_k), device)?);
        }

        Tensor::stack(&per_batch, 0) // [B,H,max_q,max_k]
    }

    #[allow(clippy::too_many_arguments)]
    fn reference_padded_attention(
        q_var: &Tensor,
        k_var: &Tensor,
        v_var: &Tensor,
        alibi_slopes: Option<&Tensor>,
        seqlens_q: &Tensor,
        seqlens_k: &Tensor,
        max_q: usize,
        max_k: usize,
        softmax_scale: f32,
        causal: bool,
        window_left: Option<usize>,
        window_right: Option<usize>,
    ) -> Result<Tensor> {
        let device = q_var.device();
        let (total_q, num_heads, head_dim) = q_var.dims3()?;
        let (_total_k, num_kv_heads, _hd2) = k_var.dims3()?;
        assert_eq!(head_dim, _hd2);

        let seqlens_q_vec = seqlens_q.to_vec1::<u32>()?;
        let seqlens_k_vec = seqlens_k.to_vec1::<u32>()?;
        let bsz = seqlens_q_vec.len();

        let mut cu_q = vec![0usize; bsz + 1];
        let mut cu_k = vec![0usize; bsz + 1];
        for i in 0..bsz {
            cu_q[i + 1] = cu_q[i] + seqlens_q_vec[i] as usize;
            cu_k[i + 1] = cu_k[i] + seqlens_k_vec[i] as usize;
        }
        assert_eq!(cu_q[bsz], total_q);

        let (k_var, v_var) = repeat_kv_for_gqa(k_var, v_var, num_heads)?;

        let mut q_padded = Vec::with_capacity(bsz);
        let mut k_padded = Vec::with_capacity(bsz);
        let mut v_padded = Vec::with_capacity(bsz);

        for i in 0..bsz {
            let lq = seqlens_q_vec[i] as usize;
            let lk = seqlens_k_vec[i] as usize;

            let q_i = q_var.narrow(0, cu_q[i], lq)?;
            let k_i = k_var.narrow(0, cu_k[i], lk)?;
            let v_i = v_var.narrow(0, cu_k[i], lk)?;

            q_padded.push(Tensor::cat(
                &[
                    &q_i,
                    &Tensor::zeros((max_q - lq, num_heads, head_dim), q_i.dtype(), device)?,
                ],
                0,
            )?);
            k_padded.push(Tensor::cat(
                &[
                    &k_i,
                    &Tensor::zeros((max_k - lk, num_heads, head_dim), k_i.dtype(), device)?,
                ],
                0,
            )?);
            v_padded.push(Tensor::cat(
                &[
                    &v_i,
                    &Tensor::zeros((max_k - lk, num_heads, head_dim), v_i.dtype(), device)?,
                ],
                0,
            )?);
        }

        let q = Tensor::stack(&q_padded, 0)?.transpose(1, 2)?.contiguous()?;
        let k = Tensor::stack(&k_padded, 0)?.transpose(1, 2)?.contiguous()?;
        let v = Tensor::stack(&v_padded, 0)?.transpose(1, 2)?.contiguous()?;

        let k_t = k.transpose(2, 3)?.contiguous()?;
        let mut scores = q.matmul(&k_t)?;
        scores = (scores * softmax_scale as f64)?;

        let bias = build_reference_bias(
            &seqlens_q_vec,
            &seqlens_k_vec,
            num_heads,
            max_q,
            max_k,
            causal,
            window_left,
            window_right,
            alibi_slopes,
            device,
        )?
        .to_dtype(scores.dtype())?;
        scores = scores.add(&bias)?;

        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?.transpose(1, 2)?; // [B,max_q,H,D]

        let mut outs = Vec::with_capacity(bsz);
        for i in 0..bsz {
            let lq = seqlens_q_vec[i] as usize;
            outs.push(ctx.i(i)?.narrow(0, 0, lq)?);
        }
        Tensor::cat(&outs, 0)
    }

    // ── Assertion helper ────────────────────────────────────────────────

    fn assert_close(
        out_var: &Tensor,
        out_ref: &Tensor,
        precision: DType,
        label: &str,
    ) -> Result<()> {
        let mae = max_abs_diff(out_var, out_ref)?;
        let e = rmse(out_var, out_ref)?;
        let (mae_tol, rmse_tol) = get_tolerances(precision);
        assert!(
            mae < mae_tol,
            "{label}: MAE too large: {mae:.6e} > {mae_tol:.6e}"
        );
        assert!(
            e < rmse_tol,
            "{label}: RMSE too large: {e:.6e} > {rmse_tol:.6e}"
        );
        Ok(())
    }

    // ── Test functions ──────────────────────────────────────────────────

    fn test_prefill_noncausal(impl_fn: VarlenImpl, precision: DType) -> Result<()> {
        let device = Device::Cpu;
        let (batch_size, num_heads, num_kv_heads, head_dim, max_seq) = (4, 8, 8, 64, 64);

        let (q, k, v, seqlens_q, seqlens_k, max_q, max_k) = make_varlen_inputs_prefill(
            batch_size,
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq,
            &device,
        )?;
        let (q, k, v) = convert_to_precision(&q, &k, &v, precision)?;
        let softmax_scale = 1.0 / (head_dim as f64).sqrt() as f32;

        let out_var = impl_fn.forward(
            &q,
            &k,
            &v,
            None,
            &seqlens_q,
            &seqlens_k,
            max_q,
            max_k,
            softmax_scale,
            false,
            None,
            None,
        )?;
        let out_ref = reference_padded_attention(
            &q,
            &k,
            &v,
            None,
            &seqlens_q,
            &seqlens_k,
            max_q,
            max_k,
            softmax_scale,
            false,
            None,
            None,
        )?;

        assert_close(&out_var, &out_ref, precision, "prefill_noncausal")
    }

    fn test_prefill_causal(impl_fn: VarlenImpl, precision: DType) -> Result<()> {
        let device = Device::Cpu;
        let (batch_size, num_heads, num_kv_heads, head_dim, max_seq) = (4, 8, 8, 64, 64);

        let (q, k, v, seqlens_q, seqlens_k, max_q, max_k) = make_varlen_inputs_prefill(
            batch_size,
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq,
            &device,
        )?;
        let (q, k, v) = convert_to_precision(&q, &k, &v, precision)?;
        let softmax_scale = 1.0 / (head_dim as f64).sqrt() as f32;

        let out_var = impl_fn.forward(
            &q,
            &k,
            &v,
            None,
            &seqlens_q,
            &seqlens_k,
            max_q,
            max_k,
            softmax_scale,
            true,
            None,
            None,
        )?;
        let out_ref = reference_padded_attention(
            &q,
            &k,
            &v,
            None,
            &seqlens_q,
            &seqlens_k,
            max_q,
            max_k,
            softmax_scale,
            true,
            None,
            None,
        )?;

        assert_close(&out_var, &out_ref, precision, "prefill_causal")
    }

    fn test_prefill_gqa(impl_fn: VarlenImpl, precision: DType) -> Result<()> {
        let device = Device::Cpu;
        let (batch_size, num_heads, num_kv_heads, head_dim, max_seq) = (3, 12, 4, 64, 64);

        let (q, k, v, seqlens_q, seqlens_k, max_q, max_k) = make_varlen_inputs_prefill(
            batch_size,
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq,
            &device,
        )?;
        let (q, k, v) = convert_to_precision(&q, &k, &v, precision)?;
        let softmax_scale = 1.0 / (head_dim as f64).sqrt() as f32;

        let out_var = impl_fn.forward(
            &q,
            &k,
            &v,
            None,
            &seqlens_q,
            &seqlens_k,
            max_q,
            max_k,
            softmax_scale,
            false,
            None,
            None,
        )?;
        let out_ref = reference_padded_attention(
            &q,
            &k,
            &v,
            None,
            &seqlens_q,
            &seqlens_k,
            max_q,
            max_k,
            softmax_scale,
            false,
            None,
            None,
        )?;

        assert_close(&out_var, &out_ref, precision, "prefill_gqa")
    }

    fn test_prefill_gqa_causal(impl_fn: VarlenImpl, precision: DType) -> Result<()> {
        let device = Device::Cpu;
        let (batch_size, num_heads, num_kv_heads, head_dim, max_seq) = (3, 12, 4, 64, 64);

        let (q, k, v, seqlens_q, seqlens_k, max_q, max_k) = make_varlen_inputs_prefill(
            batch_size,
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq,
            &device,
        )?;
        let (q, k, v) = convert_to_precision(&q, &k, &v, precision)?;
        let softmax_scale = 1.0 / (head_dim as f64).sqrt() as f32;

        let out_var = impl_fn.forward(
            &q,
            &k,
            &v,
            None,
            &seqlens_q,
            &seqlens_k,
            max_q,
            max_k,
            softmax_scale,
            true,
            None,
            None,
        )?;
        let out_ref = reference_padded_attention(
            &q,
            &k,
            &v,
            None,
            &seqlens_q,
            &seqlens_k,
            max_q,
            max_k,
            softmax_scale,
            true,
            None,
            None,
        )?;

        assert_close(&out_var, &out_ref, precision, "prefill_gqa_causal")
    }

    /// Test that fused and unfused implementations agree with each other.
    fn test_fused_vs_unfused(causal: bool, precision: DType) -> Result<()> {
        let device = Device::Cpu;
        let (batch_size, num_heads, num_kv_heads, head_dim, max_seq) = (4, 8, 8, 64, 64);

        let (q, k, v, seqlens_q, seqlens_k, max_q, max_k) = make_varlen_inputs_prefill(
            batch_size,
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq,
            &device,
        )?;
        let (q, k, v) = convert_to_precision(&q, &k, &v, precision)?;
        let softmax_scale = 1.0 / (head_dim as f64).sqrt() as f32;

        let out_fused = VarlenImpl::CpuFlash.forward(
            &q,
            &k,
            &v,
            None,
            &seqlens_q,
            &seqlens_k,
            max_q,
            max_k,
            softmax_scale,
            causal,
            None,
            None,
        )?;
        let out_unfused = VarlenImpl::Unfused.forward(
            &q,
            &k,
            &v,
            None,
            &seqlens_q,
            &seqlens_k,
            max_q,
            max_k,
            softmax_scale,
            causal,
            None,
            None,
        )?;

        let label = if causal {
            "fused_vs_unfused_causal"
        } else {
            "fused_vs_unfused_noncausal"
        };
        assert_close(&out_fused, &out_unfused, precision, label)
    }

    // ── Parameterized test entries ──────────────────────────────────────

    // Prefill noncausal
    #[test]
    fn test_prefill_noncausal_cpu_flash_f32() -> Result<()> {
        test_prefill_noncausal(VarlenImpl::CpuFlash, DType::F32)
    }

    #[test]
    fn test_prefill_noncausal_cpu_flash_f16() -> Result<()> {
        test_prefill_noncausal(VarlenImpl::CpuFlash, DType::F16)
    }

    #[test]
    fn test_prefill_noncausal_unfused_f32() -> Result<()> {
        test_prefill_noncausal(VarlenImpl::Unfused, DType::F32)
    }

    #[test]
    fn test_prefill_noncausal_unfused_f16() -> Result<()> {
        test_prefill_noncausal(VarlenImpl::Unfused, DType::F16)
    }

    // Prefill causal
    #[test]
    fn test_prefill_causal_cpu_flash_f32() -> Result<()> {
        test_prefill_causal(VarlenImpl::CpuFlash, DType::F32)
    }

    #[test]
    fn test_prefill_causal_cpu_flash_f16() -> Result<()> {
        test_prefill_causal(VarlenImpl::CpuFlash, DType::F16)
    }

    #[test]
    fn test_prefill_causal_unfused_f32() -> Result<()> {
        test_prefill_causal(VarlenImpl::Unfused, DType::F32)
    }

    #[test]
    fn test_prefill_causal_unfused_f16() -> Result<()> {
        test_prefill_causal(VarlenImpl::Unfused, DType::F16)
    }

    // GQA noncausal
    #[test]
    fn test_prefill_gqa_cpu_flash_f32() -> Result<()> {
        test_prefill_gqa(VarlenImpl::CpuFlash, DType::F32)
    }

    #[test]
    fn test_prefill_gqa_unfused_f32() -> Result<()> {
        test_prefill_gqa(VarlenImpl::Unfused, DType::F32)
    }

    // GQA causal
    #[test]
    fn test_prefill_gqa_causal_cpu_flash_f32() -> Result<()> {
        test_prefill_gqa_causal(VarlenImpl::CpuFlash, DType::F32)
    }

    #[test]
    fn test_prefill_gqa_causal_unfused_f32() -> Result<()> {
        test_prefill_gqa_causal(VarlenImpl::Unfused, DType::F32)
    }

    // Fused vs unfused agreement
    #[test]
    fn test_fused_vs_unfused_noncausal_f32() -> Result<()> {
        test_fused_vs_unfused(false, DType::F32)
    }

    #[test]
    fn test_fused_vs_unfused_causal_f32() -> Result<()> {
        test_fused_vs_unfused(true, DType::F32)
    }

    #[test]
    fn test_fused_vs_unfused_noncausal_f16() -> Result<()> {
        test_fused_vs_unfused(false, DType::F16)
    }

    #[test]
    fn test_fused_vs_unfused_causal_f16() -> Result<()> {
        test_fused_vs_unfused(true, DType::F16)
    }
}
