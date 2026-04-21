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
    pub gate_experts: Arc<QTensor>,
    pub up_experts: Arc<QTensor>,
    pub down_experts: Arc<QTensor>,
    pub act: Activation,
    pub norm_topk_prob: bool,
    pub num_experts_per_tok: usize,
    // all_reduce: AllReduce,
    // world_size: usize,
    pub dtype: DType,
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

        Ok(Self {
            gate,
            gate_experts,
            up_experts,
            down_experts,
            act: cfg.act,
            norm_topk_prob: cfg.norm_topk_prob,
            num_experts_per_tok: cfg.num_experts_per_tok,
            // all_reduce: AllReduce::new(comm),
            // world_size: 1,
            dtype,
        })
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let (batch, seq_len, hidden_dim) = xs.dims3()?;
        let xs = xs.reshape(((), hidden_dim))?;
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let original_dtype = xs.dtype();
        let xs = if xs.dtype() != DType::F32 {
            xs.to_dtype(DType::F32)?
        } else {
            xs.to_owned()
        };

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

        let ys = {
            let gate = moe::moe_gemm_gguf(
                &xs,
                &self.gate_experts,
                &None,
                &sorted_token_ids,
                &expert_ids,
                self.num_experts_per_tok,
                is_prefill,
                self.dtype,
            )?;
            let up = moe::moe_gemm_gguf(
                &xs,
                &self.up_experts,
                &None,
                &sorted_token_ids,
                &expert_ids,
                self.num_experts_per_tok,
                is_prefill,
                self.dtype,
            )?;

            let down_inputs = (up * gate.apply(&self.act)?)?;
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
        ys.reshape((batch, seq_len, hidden_dim))
    }
}
