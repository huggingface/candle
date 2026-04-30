// Adapted from: https://github.com/guoqingbao/vllm.rs/blob/main/src/models/layers/moe.rs
use candle::Module;
use candle::{quantized::QTensor, DType, Result, Tensor, D};
use candle_nn::{linear_no_bias, moe, Activation, Linear, VarBuilder};
use std::sync::Arc;

pub struct MoeCfg {
    pub hidden_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub norm_topk_prob: bool,
    pub act: Activation,
    pub decoder_sparse_step: Option<usize>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct FusedMoe {
    gate: Linear,
    gate_up_w: Tensor,
    down_w: Tensor,
    w_size_n: usize,
    act: Activation,
    norm_topk_prob: bool,
    num_experts_per_tok: usize,
    // world_size: usize,
    dtype: DType,
}

impl FusedMoe {
    pub fn new(cfg: &MoeCfg, vb: VarBuilder, dtype: DType) -> Result<Self> {
        let num_experts = cfg.num_experts;

        let gate = linear_no_bias(cfg.hidden_size, num_experts, vb.pp("gate"))?;

        let experts_vb = vb.pp("experts");
        let mut gate_up_experts = Vec::with_capacity(num_experts);
        let mut down_experts = Vec::with_capacity(num_experts);

        //pack experts
        for i in 0..num_experts {
            let experts_vb = experts_vb.pp(format!("{i}").as_str());

            let (gate_up_expert, down_expert) = {
                // n x k format
                let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
                let gate_expert = experts_vb.pp("gate_proj").get_with_hints(
                    (cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                    init_ws,
                )?;
                let up_expert = experts_vb.pp("up_proj").get_with_hints(
                    (cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                    init_ws,
                )?;
                let down_expert = experts_vb.pp("down_proj").get_with_hints(
                    (cfg.hidden_size, cfg.moe_intermediate_size),
                    "weight",
                    init_ws,
                )?;
                //pack gate_proj and up_proj
                let gate_up_expert = Tensor::cat(&[&gate_expert, &up_expert], 0)?;

                (gate_up_expert, down_expert)
            };

            gate_up_experts.push(gate_up_expert);
            down_experts.push(down_expert);
        }

        let gate_up_w = Tensor::stack(&gate_up_experts, 0)?;
        let down_w = Tensor::stack(&down_experts, 0)?;
        // let world_size = comm.world_size();
        let w_size_n = gate_up_w.dim(1)? / 2;

        Ok(Self {
            gate,
            gate_up_w,
            down_w,
            w_size_n,
            act: cfg.act,
            norm_topk_prob: cfg.norm_topk_prob,
            num_experts_per_tok: cfg.num_experts_per_tok,
            // world_size,
            dtype,
        })
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let (batch, seq_len, hidden_dim) = xs.dims3()?;
        let xs = xs.reshape(((), hidden_dim))?;
        let (num_tokens, hidden_dim) = xs.dims2()?;

        let router_logits = self.gate.forward(&xs)?;

        let routing_weights =
            candle_nn::ops::softmax_last_dim(&router_logits.to_dtype(DType::F32)?)?;

        let topk_ids = routing_weights
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.num_experts_per_tok)?
            .contiguous()?;

        let mut topk_weights = routing_weights.gather(&topk_ids, D::Minus1)?;

        if self.norm_topk_prob {
            topk_weights = topk_weights.broadcast_div(&topk_weights.sum_keepdim(D::Minus1)?)?;
        }

        // Candle's on-GPU bitonic sort_last_dim caps out at ~4096 elements
        // per row — qwen3-coder hits that at seq ≈ 512 tokens
        // (seq * num_experts_per_tok = 4096) and returns
        // CUDA_ERROR_INVALID_VALUE at launch beyond it. The sort here is
        // a once-per-MoE-layer grouping of (expert, token) pairs; pulling
        // the ~n_tokens*topk u32 keys through CPU sorts them in a few ms
        // per layer even at 128K (~1M keys). Decode (seq=1) stays on
        // the GPU sort path to avoid the host hop at per-token cadence.
        let (expert_ids, sorted_token_ids) = {
            let topk_flat = topk_ids.flatten_all()?;
            let n = topk_flat.dim(0)?;
            if n > 4096 {
                let device = topk_flat.device().clone();
                let flat_cpu = topk_flat.to_device(&candle::Device::Cpu)?;
                let mut vals: Vec<u32> = flat_cpu.to_vec1()?;
                let mut idx: Vec<u32> = (0..n as u32).collect();
                idx.sort_by_key(|&i| vals[i as usize]);
                vals.sort();
                let vals_t = Tensor::from_vec(vals, (n,), &candle::Device::Cpu)?
                    .to_device(&device)?;
                let idx_t = Tensor::from_vec(idx, (n,), &candle::Device::Cpu)?
                    .to_device(&device)?;
                (vals_t, idx_t)
            } else {
                topk_flat.sort_last_dim(true)?
            }
        };
        let _ = is_prefill; // both paths now gate on size, not the hint

        //out (M, top_k, N)
        let gate_up = moe::moe_gemm(
            &xs,
            &self.gate_up_w,
            &None,
            &sorted_token_ids,
            &expert_ids,
            self.num_experts_per_tok,
            is_prefill,
        )?;

        let gate = gate_up
            .narrow(candle::D::Minus1, 0, self.w_size_n)?
            .contiguous()?;
        let up = gate_up
            .narrow(candle::D::Minus1, self.w_size_n, self.w_size_n)?
            .contiguous()?;

        //(M * top_k, N // 2)
        let down_inputs = (up * gate.apply(&self.act)?)?.reshape(((), self.w_size_n))?;

        //view(M, top_k, K) -> sum -> (M, K)
        let ys = moe::moe_gemm(
            &down_inputs,
            &self.down_w,
            &Some(topk_weights),
            &sorted_token_ids,
            &expert_ids,
            self.num_experts_per_tok,
            is_prefill,
        )?
        .reshape((num_tokens, (), hidden_dim))?
        .sum(D::Minus2)?;

        ys.reshape((batch, seq_len, hidden_dim))
    }
}

pub struct FusedMoeGGUF {
    pub gate: Linear,
    /// Separate gate weights — populated only when the load-time gate+up
    /// fusion failed (quant-type mismatch, shape mismatch, or build
    /// error). Mutually exclusive with `gate_up_experts`.
    pub gate_experts: Option<Arc<QTensor>>,
    /// Separate up weights — see `gate_experts`.
    pub up_experts: Option<Arc<QTensor>>,
    pub down_experts: Arc<QTensor>,
    /// Per-expert byte-concat of gate_experts and up_experts along the
    /// output dim. When present, the forward path runs ONE moe_gemm_gguf
    /// call against this fused tensor instead of two — saving one kernel
    /// launch and one quantize+input-load per MoE layer per token.
    /// Shape: [num_experts, 2 * moe_intermediate_size, hidden_size].
    /// Mutually exclusive with `gate_experts`/`up_experts`.
    pub gate_up_experts: Option<Arc<QTensor>>,
    pub act: Activation,
    pub norm_topk_prob: bool,
    pub num_experts_per_tok: usize,
    // all_reduce: AllReduce,
    // world_size: usize,
    pub dtype: DType,
}

/// Build a per-expert byte-concat of gate and up experts along the output
/// dim, producing a fused [E, 2N, K] QTensor. Returns None on shape or
/// quant-type mismatch — caller should fall back to separate matmuls.
///
/// Per-expert layout: for each expert e, the bytes are
///   gate_experts[e].bytes ++ up_experts[e].bytes
/// concatenated end-to-end. The kernel sees a single [E, 2N, K] tensor
/// indexed by (expert, output_row, k_block) — output rows 0..N belong to
/// gate, rows N..2N belong to up.
pub fn try_fuse_gate_up(
    gate: &QTensor,
    up: &QTensor,
    device: &candle::Device,
) -> Option<Arc<QTensor>> {
    use candle::quantized::QStorage;
    if gate.dtype() != up.dtype() {
        return None;
    }
    let g_shape = gate.shape().dims().to_vec();
    let u_shape = up.shape().dims().to_vec();
    if g_shape.len() != 3 || u_shape.len() != 3 {
        return None;
    }
    let (e, n_g, k_g) = (g_shape[0], g_shape[1], g_shape[2]);
    let (e2, n_u, k_u) = (u_shape[0], u_shape[1], u_shape[2]);
    if e != e2 || k_g != k_u || n_g != n_u {
        return None;
    }
    let dtype = gate.dtype();
    let g_bytes = gate.data().ok()?.into_owned();
    let u_bytes = up.data().ok()?.into_owned();
    if g_bytes.len() % e != 0 || u_bytes.len() % e != 0 {
        return None;
    }
    let g_per = g_bytes.len() / e;
    let u_per = u_bytes.len() / e;
    let total = g_bytes.len() + u_bytes.len();
    // u32-aligned scratch buffer; QStorage::from_data needs >=2-byte
    // alignment for K-quant block structs.
    let n_u32 = (total + 3) / 4;
    let mut buf = vec![0u32; n_u32];
    let buf_bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, n_u32 * 4)
    };
    let stride = g_per + u_per;
    for ei in 0..e {
        let dst = ei * stride;
        buf_bytes[dst..dst + g_per].copy_from_slice(&g_bytes[ei * g_per..(ei + 1) * g_per]);
        buf_bytes[dst + g_per..dst + stride]
            .copy_from_slice(&u_bytes[ei * u_per..(ei + 1) * u_per]);
    }
    let combined: &[u8] = unsafe {
        std::slice::from_raw_parts(buf.as_ptr() as *const u8, total)
    };
    let storage = QStorage::from_data(std::borrow::Cow::Borrowed(combined), device, dtype).ok()?;
    drop(buf);
    let shape = candle::Shape::from((e, 2 * n_g, k_g));
    QTensor::new(storage, shape).ok().map(Arc::new)
}

impl FusedMoeGGUF {
    pub fn new(
        cfg: &MoeCfg,
        vb: crate::quantized_var_builder::VarBuilder,
        dtype: DType,
    ) -> Result<Self> {
        let num_experts = cfg.num_experts;
        let gate_ws = vb
            .pp("ffn_gate_inp")
            .get((num_experts, cfg.hidden_size), "weight")?
            .dequantize(vb.device())?
            .to_dtype(DType::F32)?;

        let gate = Linear::new(gate_ws, None);

        let (gate_experts, up_experts, down_experts) = {
            (
                vb.pp("ffn_gate_exps").get(
                    (num_experts, cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                )?,
                vb.pp("ffn_up_exps").get(
                    (num_experts, cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                )?,
                vb.pp("ffn_down_exps").get(
                    (num_experts, cfg.hidden_size, cfg.moe_intermediate_size),
                    "weight",
                )?,
            )
        };

        // Try to fuse gate+up at load. When fusion succeeds, drop the
        // originals so we don't pay the VRAM cost of holding both a
        // packed and unpacked copy (the packed copy is ≈ gate + up
        // bytes; keeping all three doubles MoE FFN-input weight VRAM).
        let gate_up_experts = try_fuse_gate_up(&gate_experts, &up_experts, vb.device());
        let (gate_experts, up_experts) = if gate_up_experts.is_some() {
            (None, None)
        } else {
            (Some(gate_experts), Some(up_experts))
        };

        Ok(Self {
            gate,
            gate_experts,
            up_experts,
            down_experts,
            gate_up_experts,
            act: cfg.act,
            norm_topk_prob: cfg.norm_topk_prob,
            num_experts_per_tok: cfg.num_experts_per_tok,
            // all_reduce: AllReduce::new(comm),
            // world_size: 1,
            dtype,
        })
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        self.forward_inner(xs, is_prefill, None)
    }

    /// Forward + add residual fused into the down kernel's atomicAdd
    /// reduction. The residual must have the same shape as the FFN
    /// output (`[batch, seq_len, hidden]`) and the same dtype as the
    /// model's working dtype. When the `LLMSERVER_MOE_DOWN_REDUCE`
    /// fast path is taken, this fold is free; when it isn't, we fall
    /// back to a separate add at the end.
    pub fn forward_with_residual(&self, xs: &Tensor, residual: &Tensor, is_prefill: bool) -> Result<Tensor> {
        self.forward_inner(xs, is_prefill, Some(residual))
    }

    fn forward_inner(&self, xs: &Tensor, is_prefill: bool, residual: Option<&Tensor>) -> Result<Tensor> {
        let (batch, seq_len, hidden_dim) = xs.dims3()?;
        let xs = xs.reshape(((), hidden_dim))?;
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let original_dtype = xs.dtype();
        let xs = if xs.dtype() != DType::F32 {
            xs.to_dtype(DType::F32)?
        } else {
            xs.to_owned()
        };

        // Optional fused gate matmul + softmax + topk: ONE kernel launch
        // combining cublas SGEMV `gate.forward(xs)` with topk_softmax. In
        // practice the single-warp design under-saturates the GPU and is
        // SLOWER than the unfused (cublas SGEMV + topk_softmax) pair for
        // batch=1 decode. Default OFF; opt in via LLMSERVER_MOE_GATE_FUSE=1
        // for measurement/future tuning.
        // Set LLMSERVER_MOE_GATE_PROFILE=1 to log per-stage gate timings
        // (sync overhead serializes execution; use for relative comparison only).
        let profile_gate = !is_prefill
            && xs.device().is_cuda()
            && std::env::var("LLMSERVER_MOE_GATE_PROFILE").map_or(false, |v| v == "1");
        let want_gate_fuse = !is_prefill
            && std::env::var("LLMSERVER_MOE_GATE_FUSE")
                .map_or(false, |v| v == "1");
        let fused_gate_topk = if want_gate_fuse {
            #[cfg(feature = "cuda")]
            {
                candle_nn::moe::gate_topk_softmax(
                    &xs,
                    self.gate.weight(),
                    self.num_experts_per_tok,
                    self.norm_topk_prob,
                )?
            }
            #[cfg(not(feature = "cuda"))]
            { None }
        } else {
            None
        };
        let (mut topk_weights, topk_ids) = if let Some((w, i)) = fused_gate_topk {
            (w, i)
        } else {
            let t_gate0 = if profile_gate {
                let _ = xs.device().synchronize();
                Some(std::time::Instant::now())
            } else { None };
            // Optional fast path: multi-block F32 GEMV instead of cublas
            // SGEMV. Default ON; opt out via LLMSERVER_MOE_GATE_GEMV=0.
            // Falls back to gate.forward when shape isn't supported.
            let want_gate_gemv = std::env::var("LLMSERVER_MOE_GATE_GEMV")
                .map_or(true, |v| v == "1");
            let router_logits = if want_gate_gemv {
                #[cfg(feature = "cuda")]
                {
                    match candle_nn::moe::gate_gemv_f32(&xs, self.gate.weight())? {
                        Some(t) => t,
                        None => self.gate.forward(&xs)?,
                    }
                }
                #[cfg(not(feature = "cuda"))]
                { self.gate.forward(&xs)? }
            } else {
                self.gate.forward(&xs)?
            };
            let logits_f32 = router_logits.to_dtype(DType::F32)?;
            let t_gate_us = if let Some(t) = t_gate0 {
                let _ = xs.device().synchronize();
                t.elapsed().as_micros()
            } else { 0 };
            let t_topk0 = if profile_gate {
                Some(std::time::Instant::now())
            } else { None };
            let result = if logits_f32.device().is_cuda()
                && matches!(logits_f32.dim(D::Minus1)?, 32 | 64 | 128 | 256)
            {
                candle_nn::moe::topk_softmax(
                    &logits_f32,
                    self.num_experts_per_tok,
                    self.norm_topk_prob,
                )?
            } else {
                let routing_weights = candle_nn::ops::softmax_last_dim(&logits_f32)?;
                let topk_ids = routing_weights
                    .arg_sort_last_dim(false)?
                    .narrow(D::Minus1, 0, self.num_experts_per_tok)?
                    .contiguous()?;
                let mut topk_weights = routing_weights.gather(&topk_ids, D::Minus1)?;
                if self.norm_topk_prob {
                    topk_weights = topk_weights.broadcast_div(&topk_weights.sum_keepdim(D::Minus1)?)?;
                }
                (topk_weights, topk_ids)
            };
            if let Some(t) = t_topk0 {
                let _ = xs.device().synchronize();
                let t_topk_us = t.elapsed().as_micros();
                tracing::info!("🟦 GATE_PROF gate={}us topk={}us", t_gate_us, t_topk_us);
            }
            result
        };
        let _ = &mut topk_weights; // silence unused-mut on the always-mutated branch

        // Candle's on-GPU bitonic sort_last_dim caps out at ~4096 elements
        // per row — qwen3-coder hits that at seq ≈ 512 tokens
        // (seq * num_experts_per_tok = 4096) and returns
        // CUDA_ERROR_INVALID_VALUE at launch beyond it. The sort here is
        // a once-per-MoE-layer grouping of (expert, token) pairs; pulling
        // the ~n_tokens*topk u32 keys through CPU sorts them in a few ms
        // per layer even at 128K (~1M keys). Decode (seq=1) stays on
        // the GPU sort path to avoid the host hop at per-token cadence.
        let (expert_ids, sorted_token_ids) = {
            let topk_flat = topk_ids.flatten_all()?;
            let n = topk_flat.dim(0)?;
            if n > 4096 {
                let device = topk_flat.device().clone();
                let flat_cpu = topk_flat.to_device(&candle::Device::Cpu)?;
                let mut vals: Vec<u32> = flat_cpu.to_vec1()?;
                let mut idx: Vec<u32> = (0..n as u32).collect();
                idx.sort_by_key(|&i| vals[i as usize]);
                vals.sort();
                let vals_t = Tensor::from_vec(vals, (n,), &candle::Device::Cpu)?
                    .to_device(&device)?;
                let idx_t = Tensor::from_vec(idx, (n,), &candle::Device::Cpu)?
                    .to_device(&device)?;
                (vals_t, idx_t)
            } else {
                topk_flat.sort_last_dim(true)?
            }
        };
        let _ = is_prefill; // both paths now gate on size, not the hint

        let ys = {
            // Decide which gate+up path to use:
            //   1. Kernel-level fusion (preferred): one launch produces
            //      `silu(gate) * up` directly. Requires separate gate/up
            //      QTensors, SiLU activation, decode-shape (is_prefill=false),
            //      and CUDA. Saves 4 launches per MoE layer per token.
            //   2. Load-time concat fusion (opt-in via env): one matmul on
            //      packed [E, 2N, K] tensor, then narrow + silu + mul.
            //      Saves 1 matmul launch but adds 2 contiguous copies.
            //   3. Plain unfused: two matmuls + silu + mul.
            let act_is_silu = matches!(self.act, Activation::Silu);
            let on_cuda = xs.device().is_cuda();
            let can_kernel_fuse = !is_prefill
                && on_cuda
                && act_is_silu
                && self.gate_experts.is_some()
                && self.up_experts.is_some();

            let down_inputs = if can_kernel_fuse {
                // Single fused launch: silu(gate · x) * (up · x)
                let gate_w = self.gate_experts.as_ref().unwrap();
                let up_w = self.up_experts.as_ref().unwrap();
                moe::moe_gemm_gguf_gate_up_silu_mul(
                    &xs,
                    gate_w,
                    up_w,
                    &sorted_token_ids,
                    &expert_ids,
                    self.num_experts_per_tok,
                )?
            } else {
                let (gate, up) = if let Some(gu) = self.gate_up_experts.as_ref() {
                    let gu_out = moe::moe_gemm_gguf(
                        &xs, gu, &None,
                        &sorted_token_ids, &expert_ids,
                        self.num_experts_per_tok, is_prefill, self.dtype,
                    )?;
                    let n = gu_out.dim(D::Minus1)? / 2;
                    let gate = gu_out.narrow(D::Minus1, 0, n)?.contiguous()?;
                    let up = gu_out.narrow(D::Minus1, n, n)?.contiguous()?;
                    (gate, up)
                } else {
                    let gate_w = self.gate_experts.as_ref()
                        .ok_or_else(|| candle::Error::Msg(
                            "FusedMoeGGUF: neither gate_up_experts nor gate_experts is set".into()
                        ))?;
                    let up_w = self.up_experts.as_ref()
                        .ok_or_else(|| candle::Error::Msg(
                            "FusedMoeGGUF: up_experts is None but gate_up_experts is also None".into()
                        ))?;
                    let gate = moe::moe_gemm_gguf(
                        &xs, gate_w, &None,
                        &sorted_token_ids, &expert_ids,
                        self.num_experts_per_tok, is_prefill, self.dtype,
                    )?;
                    let up = moe::moe_gemm_gguf(
                        &xs, up_w, &None,
                        &sorted_token_ids, &expert_ids,
                        self.num_experts_per_tok, is_prefill, self.dtype,
                    )?;
                    (gate, up)
                };
                (up * gate.apply(&self.act)?)?
            };

            // Fused down + topk reduction (atomicAdd) when on CUDA and
            // not prefill — saves the per-(token,expert) intermediate
            // [M*topk, hidden] write and the explicit .sum(topk_dim)
            // launch. Output is already [num_tokens, hidden].
            let want_down_reduce = !is_prefill
                && xs.device().is_cuda()
                && std::env::var("LLMSERVER_MOE_DOWN_REDUCE")
                    .map_or(true, |v| v == "1");
            if want_down_reduce {
                // If the caller passed a residual, reshape it to
                // [num_tokens, hidden] and let the kernel fold the add
                // into its atomicAdd accumulation. Saves one downstream
                // launch + one elementwise pass.
                let res_reshaped = if let Some(r) = residual {
                    Some(r.reshape((num_tokens, hidden_dim))?)
                } else {
                    None
                };
                let mut ys = moe::moe_gemm_gguf_down_reduce(
                    &down_inputs,
                    &self.down_experts,
                    &sorted_token_ids,
                    &expert_ids,
                    &topk_weights,
                    self.num_experts_per_tok,
                    num_tokens,
                    res_reshaped.as_ref(),
                )?;
                if ys.dtype() != original_dtype {
                    ys = ys.to_dtype(original_dtype)?;
                }
                return ys.reshape((batch, seq_len, hidden_dim));
            }
            // Slow path: residual must be added by the caller after this
            // function returns; signal that via the unfused fallback.
            moe::moe_gemm_gguf(
                &down_inputs,
                &self.down_experts,
                &Some(topk_weights),
                &sorted_token_ids,
                &expert_ids,
                self.num_experts_per_tok,
                is_prefill,
                self.dtype,
            )?
        };
        let mut ys = ys.reshape((num_tokens, (), hidden_dim))?.sum(D::Minus2)?;
        if ys.dtype() != original_dtype {
            ys = ys.to_dtype(original_dtype)?;
        }
        let mut ys = ys.reshape((batch, seq_len, hidden_dim))?;
        // Slow-path residual add (when the down_reduce kernel wasn't taken).
        if let Some(r) = residual {
            ys = (ys + r)?;
        }
        Ok(ys)
    }
}
