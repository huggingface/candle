//! Route-aware grouped matrix multiplication using cuTile.
#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

use candle::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, DeviceRepr};
use candle::cuda_backend::{CudaDType, CudaDevice};
use candle::cutile;
use candle::{CudaStorage, DType, Layout, Result, Storage, Tensor};
use candle_kernels::ffi;
use cutile::cuda_async::device_operation::DeviceOp;
use cutile::tile_kernel::TileKernel;
use half::bf16;

const MAX_EXPERTS: usize = 1023;

#[cutile::module]
mod kernels {
    use candle::cutile;
    use cutile::core::*;

    #[cutile::entry(
        unchecked_accesses = true,
        optimization_hints = (
            sm_120 = (num_cta_in_cga = 2,),
        )
    )]
    pub unsafe fn routed_grouped_matmul<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const GROUP_M: i32,
        const TOP_K: i32,
        const MUL_ROUTED_WEIGHT: i32,
    >(
        out_ptr: *mut bf16,
        input_ptr: *mut bf16,
        weight_ptr: *mut bf16,
        sorted_token_ids_ptr: *mut i32,
        expert_ids_ptr: *mut i32,
        num_tokens_post_padded_ptr: *mut i32,
        route_weights_ptr: *mut f32,
        n_size: i32,
        k_size: i32,
        em: i32,
        num_routes: i32,
    ) {
        let pid: i32 = get_tile_block_id().0;
        let num_pid_m: i32 = ceil_div(em, BM);
        let num_pid_n: i32 = ceil_div(n_size, BN);
        let num_pid_in_group: i32 = GROUP_M * num_pid_n;
        let group_id: i32 = pid / num_pid_in_group;
        let first_pid_m: i32 = group_id * GROUP_M;
        let group_size_m: i32 = {
            let remaining = num_pid_m - first_pid_m;
            if remaining < GROUP_M {
                remaining
            } else {
                GROUP_M
            }
        };
        let pid_m: i32 = first_pid_m + ((pid % num_pid_in_group) % group_size_m);
        let pid_n: i32 = (pid % num_pid_in_group) / group_size_m;

        let ntpp_p0: PointerTile<*mut i32, { [] }> = pointer_to_tile(num_tokens_post_padded_ptr);
        let ntpp_p1: PointerTile<*mut i32, { [1] }> = ntpp_p0.reshape(const_shape![1]);
        let (ntpp_t, _): (Tile<i32, { [1] }>, Token) = load_ptr_tko(
            ntpp_p1,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<0>,
        );
        let ntpp: i32 = tile_to_scalar(ntpp_t.reshape(const_shape![]));

        if pid_m * BM < ntpp {
            let iota_m: Tile<i32, { [BM] }> = iota(const_shape![BM]);
            let base_m: Tile<i32, { [BM] }> = broadcast_scalar(pid_m * BM, const_shape![BM]);
            let sorted_offsets: Tile<i32, { [BM] }> = iota_m + base_m;
            let em_tile: Tile<i32, { [BM] }> = broadcast_scalar(em, const_shape![BM]);
            let sorted_mask: Tile<bool, { [BM] }> = lt_tile(sorted_offsets, em_tile);

            let sorted_p0: PointerTile<*mut i32, { [] }> = pointer_to_tile(sorted_token_ids_ptr);
            let sorted_p1: PointerTile<*mut i32, { [1] }> = sorted_p0.reshape(const_shape![1]);
            let sorted_p2: PointerTile<*mut i32, { [BM] }> = sorted_p1.broadcast(const_shape![BM]);
            let sorted_ptrs: PointerTile<*mut i32, { [BM] }> =
                sorted_p2.offset_tile(sorted_offsets);
            let (route_ids, _): (Tile<i32, { [BM] }>, Token) = load_ptr_tko(
                sorted_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(sorted_mask),
                Some(num_routes),
                None,
                Latency::<0>,
            );
            let route_count: Tile<i32, { [BM] }> = broadcast_scalar(num_routes, const_shape![BM]);
            let route_mask: Tile<bool, { [BM] }> = lt_tile(route_ids, route_count);

            let expert_p0: PointerTile<*mut i32, { [] }> = pointer_to_tile(expert_ids_ptr);
            let expert_p1: PointerTile<*mut i32, { [1] }> = expert_p0.reshape(const_shape![1]);
            let pid_m_tile: Tile<i32, { [1] }> = broadcast_scalar(pid_m, const_shape![1]);
            let expert_ptr: PointerTile<*mut i32, { [1] }> = expert_p1.offset_tile(pid_m_tile);
            let (expert_t, _): (Tile<i32, { [1] }>, Token) = load_ptr_tko(
                expert_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let expert: i32 = tile_to_scalar(expert_t.reshape(const_shape![]));

            let iota_n: Tile<i32, { [BN] }> = iota(const_shape![BN]);
            let base_n: Tile<i32, { [BN] }> = broadcast_scalar(pid_n * BN, const_shape![BN]);
            let output_columns: Tile<i32, { [BN] }> = iota_n + base_n;
            let n_tile: Tile<i32, { [BN] }> = broadcast_scalar(n_size, const_shape![BN]);
            let column_mask: Tile<bool, { [BN] }> = lt_tile(output_columns, n_tile);

            let route_column: Tile<i32, { [BM, 1] }> = route_ids.reshape(const_shape![BM, 1]);
            let routes: Tile<i32, { [BM, BN] }> = route_column.broadcast(const_shape![BM, BN]);
            let output_stride: Tile<i32, { [BM, BN] }> =
                broadcast_scalar(n_size, const_shape![BM, BN]);
            let route_offsets: Tile<i32, { [BM, BN] }> =
                muli(routes, output_stride, overflow::NoSignedWrap);
            let column_row: Tile<i32, { [1, BN] }> = output_columns.reshape(const_shape![1, BN]);
            let columns: Tile<i32, { [BM, BN] }> = column_row.broadcast(const_shape![BM, BN]);
            let output_offsets: Tile<i32, { [BM, BN] }> = route_offsets + columns;
            let output_p0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(out_ptr);
            let output_p1: PointerTile<*mut bf16, { [1, 1] }> =
                output_p0.reshape(const_shape![1, 1]);
            let output_p2: PointerTile<*mut bf16, { [BM, BN] }> =
                output_p1.broadcast(const_shape![BM, BN]);
            let output_ptrs: PointerTile<*mut bf16, { [BM, BN] }> =
                output_p2.offset_tile(output_offsets);

            let route_mask_column: Tile<bool, { [BM, 1] }> =
                route_mask.reshape(const_shape![BM, 1]);
            let route_mask_2d: Tile<bool, { [BM, BN] }> =
                route_mask_column.broadcast(const_shape![BM, BN]);
            let column_mask_row: Tile<bool, { [1, BN] }> = column_mask.reshape(const_shape![1, BN]);
            let column_mask_2d: Tile<bool, { [BM, BN] }> =
                column_mask_row.broadcast(const_shape![BM, BN]);
            let output_mask: Tile<bool, { [BM, BN] }> = route_mask_2d & column_mask_2d;

            if expert == -1 {
                let zeros: Tile<bf16, { [BM, BN] }> = constant(bf16::ZERO, const_shape![BM, BN]);
                store_ptr_tko(
                    output_ptrs,
                    zeros,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    Some(output_mask),
                    None,
                    Latency::<0>,
                );
            } else {
                let quotient: Tile<i32, { [BN] }> = output_columns / n_tile;
                let quotient_n: Tile<i32, { [BN] }> =
                    muli(quotient, n_tile, overflow::NoSignedWrap);
                let weight_columns: Tile<i32, { [BN] }> =
                    subi(output_columns, quotient_n, overflow::NoSignedWrap);
                let top_k_tile: Tile<i32, { [BM] }> = broadcast_scalar(TOP_K, const_shape![BM]);
                let input_rows: Tile<i32, { [BM] }> = route_ids / top_k_tile;
                let zero_rows: Tile<i32, { [BM] }> = broadcast_scalar(0i32, const_shape![BM]);
                let safe_rows: Tile<i32, { [BM] }> = select(route_mask, input_rows, zero_rows);
                let k_tile_m: Tile<i32, { [BM] }> = broadcast_scalar(k_size, const_shape![BM]);
                let input_row_offsets: Tile<i32, { [BM] }> =
                    muli(safe_rows, k_tile_m, overflow::NoSignedWrap);
                let expert_offset: i32 = expert * (k_size * n_size);

                let input_p0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(input_ptr);
                let input_p1: PointerTile<*mut bf16, { [1, 1] }> =
                    input_p0.reshape(const_shape![1, 1]);
                let input_p2: PointerTile<*mut bf16, { [BM, BK] }> =
                    input_p1.broadcast(const_shape![BM, BK]);
                let weight_p0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(weight_ptr);
                let weight_p1: PointerTile<*mut bf16, { [1, 1] }> =
                    weight_p0.reshape(const_shape![1, 1]);
                let weight_p2: PointerTile<*mut bf16, { [BK, BN] }> =
                    weight_p1.broadcast(const_shape![BK, BN]);

                let iota_k: Tile<i32, { [BK] }> = iota(const_shape![BK]);
                let input_row: Tile<i32, { [BM, 1] }> =
                    input_row_offsets.reshape(const_shape![BM, 1]);
                let input_rows_2d: Tile<i32, { [BM, BK] }> =
                    input_row.broadcast(const_shape![BM, BK]);
                let k_row: Tile<i32, { [1, BK] }> = iota_k.reshape(const_shape![1, BK]);
                let k_input: Tile<i32, { [BM, BK] }> = k_row.broadcast(const_shape![BM, BK]);
                let input_offsets: Tile<i32, { [BM, BK] }> = input_rows_2d + k_input;
                let mut input_ptrs: PointerTile<*mut bf16, { [BM, BK] }> =
                    input_p2.offset_tile(input_offsets);

                let expert_offsets: Tile<i32, { [BK, BN] }> =
                    broadcast_scalar(expert_offset, const_shape![BK, BN]);
                let k_column: Tile<i32, { [BK, 1] }> = iota_k.reshape(const_shape![BK, 1]);
                let k_weight: Tile<i32, { [BK, BN] }> = k_column.broadcast(const_shape![BK, BN]);
                let weight_column_row: Tile<i32, { [1, BN] }> =
                    weight_columns.reshape(const_shape![1, BN]);
                let weight_columns_2d: Tile<i32, { [BK, BN] }> =
                    weight_column_row.broadcast(const_shape![BK, BN]);
                let k_stride: Tile<i32, { [BK, BN] }> =
                    broadcast_scalar(k_size, const_shape![BK, BN]);
                let weight_column_offsets: Tile<i32, { [BK, BN] }> =
                    muli(weight_columns_2d, k_stride, overflow::NoSignedWrap);
                let weight_offsets: Tile<i32, { [BK, BN] }> =
                    expert_offsets + weight_column_offsets + k_weight;
                let mut weight_ptrs: PointerTile<*mut bf16, { [BK, BN] }> =
                    weight_p2.offset_tile(weight_offsets);
                let input_step: Tile<i32, { [BM, BK] }> =
                    broadcast_scalar(BK, const_shape![BM, BK]);
                let weight_step: Tile<i32, { [BK, BN] }> =
                    broadcast_scalar(BK, const_shape![BK, BN]);

                let mut accumulator: Tile<f32, { [BM, BN] }> =
                    constant(0.0f32, const_shape![BM, BN]);
                let k_tiles: i32 = ceil_div(k_size, BK);
                for k_index in 0i32..k_tiles {
                    let k_base: Tile<i32, { [BK] }> =
                        broadcast_scalar(k_index * BK, const_shape![BK]);
                    let k_offsets: Tile<i32, { [BK] }> = iota_k + k_base;
                    let k_size_tile: Tile<i32, { [BK] }> =
                        broadcast_scalar(k_size, const_shape![BK]);
                    let k_mask: Tile<bool, { [BK] }> = lt_tile(k_offsets, k_size_tile);
                    let input_k_row: Tile<bool, { [1, BK] }> = k_mask.reshape(const_shape![1, BK]);
                    let input_mask: Tile<bool, { [BM, BK] }> =
                        input_k_row.broadcast(const_shape![BM, BK]);
                    let weight_k_column: Tile<bool, { [BK, 1] }> =
                        k_mask.reshape(const_shape![BK, 1]);
                    let weight_k_mask: Tile<bool, { [BK, BN] }> =
                        weight_k_column.broadcast(const_shape![BK, BN]);
                    let weight_n_row: Tile<bool, { [1, BN] }> =
                        column_mask.reshape(const_shape![1, BN]);
                    let weight_n_mask: Tile<bool, { [BK, BN] }> =
                        weight_n_row.broadcast(const_shape![BK, BN]);
                    let weight_mask: Tile<bool, { [BK, BN] }> = weight_k_mask & weight_n_mask;
                    let (input_load, _): (Tile<bf16, { [BM, BK] }>, Token) = load_ptr_tko(
                        input_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        Some(input_mask),
                        None,
                        None,
                        Latency::<0>,
                    );
                    let (weight_load, _): (Tile<bf16, { [BK, BN] }>, Token) = load_ptr_tko(
                        weight_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        Some(weight_mask),
                        None,
                        None,
                        Latency::<0>,
                    );
                    let input_zeros: Tile<bf16, { [BM, BK] }> =
                        constant(bf16::ZERO, const_shape![BM, BK]);
                    let weight_zeros: Tile<bf16, { [BK, BN] }> =
                        constant(bf16::ZERO, const_shape![BK, BN]);
                    let input_tile: Tile<bf16, { [BM, BK] }> =
                        select(input_mask, input_load, input_zeros);
                    let weight_tile: Tile<bf16, { [BK, BN] }> =
                        select(weight_mask, weight_load, weight_zeros);

                    accumulator = mmaf(input_tile, weight_tile, accumulator);
                    input_ptrs = input_ptrs.offset_tile(input_step);
                    weight_ptrs = weight_ptrs.offset_tile(weight_step);
                }

                if MUL_ROUTED_WEIGHT != 0 {
                    let route_weight_p0: PointerTile<*mut f32, { [] }> =
                        pointer_to_tile(route_weights_ptr);
                    let route_weight_p1: PointerTile<*mut f32, { [1] }> =
                        route_weight_p0.reshape(const_shape![1]);
                    let route_weight_p2: PointerTile<*mut f32, { [BM] }> =
                        route_weight_p1.broadcast(const_shape![BM]);
                    let route_weight_ptrs: PointerTile<*mut f32, { [BM] }> =
                        route_weight_p2.offset_tile(route_ids);
                    let (route_weights, _): (Tile<f32, { [BM] }>, Token) = load_ptr_tko(
                        route_weight_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        Some(route_mask),
                        Some(0.0f32),
                        None,
                        Latency::<0>,
                    );
                    let route_weight_column: Tile<f32, { [BM, 1] }> =
                        route_weights.reshape(const_shape![BM, 1]);
                    let route_weights_2d: Tile<f32, { [BM, BN] }> =
                        route_weight_column.broadcast(const_shape![BM, BN]);
                    accumulator = accumulator * route_weights_2d;
                }

                let output: Tile<bf16, { [BM, BN] }> = convert_tile(accumulator);
                store_ptr_tko(
                    output_ptrs,
                    output,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    Some(output_mask),
                    None,
                    Latency::<0>,
                );
            }
        }
    }
}

/// Selects how input rows map to flattened routes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeInputMode {
    /// Input has one row per token and each row is reused for all top-k routes.
    TokenRows,
    /// Input has one row per flattened token-expert route.
    RoutedRows,
}

#[derive(Clone, Copy)]
struct TileConfig {
    bm: i32,
    bn: i32,
    bk: i32,
    group_m: i32,
}

/// CUDA routing metadata shared by multiple grouped matrix multiplications.
pub struct MoeRouting {
    sorted_token_ids: CudaSlice<i32>,
    expert_ids: CudaSlice<i32>,
    num_tokens_post_padded: CudaSlice<i32>,
    route_weight_dummy: CudaSlice<f32>,
    device: CudaDevice,
    num_tokens: usize,
    top_k: usize,
    num_experts: usize,
    em: usize,
    config: TileConfig,
}

impl MoeRouting {
    /// Builds a route plan from contiguous `[tokens, top_k]` u32 or i32 expert ids.
    ///
    /// Expert ids outside `0..num_experts` are inactive routes and produce zero output rows.
    pub fn new(topk_ids: &Tensor, num_experts: usize) -> Result<Self> {
        let (num_tokens, top_k) = topk_ids.dims2()?;
        if num_tokens == 0 || top_k == 0 {
            candle::bail!("cuTile MoE routing requires at least one route")
        }
        if num_experts == 0 || num_experts > MAX_EXPERTS {
            candle::bail!("cuTile MoE supports between 1 and {MAX_EXPERTS} experts")
        }
        if !matches!(topk_ids.dtype(), DType::U32 | DType::I32) {
            candle::bail!("cuTile MoE top-k ids must be u32 or i32")
        }
        let device = topk_ids.device().as_cuda_device()?.clone();
        let num_routes = num_tokens
            .checked_mul(top_k)
            .ok_or_else(|| candle::Error::msg("cuTile MoE route count overflow"))?;
        checked_i32(num_routes, "route count")?;
        checked_i32(num_experts, "expert count")?;
        let config = default_config(num_tokens, num_experts);
        let block_size = config.bm as usize;
        let padded_bound = num_routes
            .checked_add(
                num_experts
                    .checked_mul(block_size - 1)
                    .ok_or_else(|| candle::Error::msg("cuTile MoE route padding overflow"))?,
            )
            .ok_or_else(|| candle::Error::msg("cuTile MoE route padding overflow"))?;
        let em = if num_routes < num_experts {
            num_routes
                .checked_mul(block_size)
                .ok_or_else(|| candle::Error::msg("cuTile MoE route padding overflow"))?
                .min(padded_bound)
        } else {
            padded_bound
        };
        checked_i32(em, "padded route count")?;

        let mut sorted_token_ids = device.alloc_zeros::<i32>(em)?;
        let mut expert_ids = device.alloc_zeros::<i32>(em.div_ceil(block_size))?;
        let mut num_tokens_post_padded = device.alloc_zeros::<i32>(1)?;
        let mut cumsum = device.alloc_zeros::<i32>(num_experts + 1)?;
        let route_weight_dummy = device.alloc_zeros::<f32>(1)?;
        let (storage, layout) = topk_ids.storage_and_layout();
        if !layout.is_contiguous() {
            candle::bail!("cuTile MoE top-k ids must be contiguous")
        }
        match topk_ids.dtype() {
            DType::U32 => launch_alignment::<u32>(
                &storage,
                layout,
                &mut sorted_token_ids,
                &mut expert_ids,
                &mut num_tokens_post_padded,
                &mut cumsum,
                num_experts,
                block_size,
                num_routes,
                em,
                &device,
            )?,
            DType::I32 => launch_alignment::<i32>(
                &storage,
                layout,
                &mut sorted_token_ids,
                &mut expert_ids,
                &mut num_tokens_post_padded,
                &mut cumsum,
                num_experts,
                block_size,
                num_routes,
                em,
                &device,
            )?,
            _ => unreachable!(),
        }

        Ok(Self {
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            route_weight_dummy,
            device,
            num_tokens,
            top_k,
            num_experts,
            em,
            config,
        })
    }

    /// Returns the number of source tokens.
    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    /// Returns the number of selected experts per token.
    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// Returns the number of experts represented by the route plan.
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }
}

/// Runs route gather, expert BF16 GEMM, route scatter, and optional route-weight scaling.
///
/// Expert weights use `[experts, out_features, in_features]` layout. The result is flattened as
/// `[tokens * top_k, out_features]`.
pub fn routed_grouped_matmul(
    input: &Tensor,
    expert_weights: &Tensor,
    routing: &MoeRouting,
    input_mode: MoeInputMode,
    route_weights: Option<&Tensor>,
) -> Result<Tensor> {
    routed_grouped_matmul_inner(
        input,
        expert_weights,
        routing,
        input_mode,
        route_weights,
        false,
    )
}

/// Compiles the exact routed grouped matmul specialization without launching it.
pub fn warmup_routed_grouped_matmul(
    input: &Tensor,
    expert_weights: &Tensor,
    routing: &MoeRouting,
    input_mode: MoeInputMode,
    route_weights: Option<&Tensor>,
) -> Result<()> {
    routed_grouped_matmul_inner(
        input,
        expert_weights,
        routing,
        input_mode,
        route_weights,
        true,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn routed_grouped_matmul_inner(
    input: &Tensor,
    expert_weights: &Tensor,
    routing: &MoeRouting,
    input_mode: MoeInputMode,
    route_weights: Option<&Tensor>,
    compile_only: bool,
) -> Result<Tensor> {
    if input.dtype() != DType::BF16 || expert_weights.dtype() != DType::BF16 {
        candle::bail!("cuTile routed grouped matmul requires bf16 input and weights")
    }
    let (input_rows, input_k) = input.dims2()?;
    let (num_experts, n_size, weight_k) = expert_weights.dims3()?;
    if n_size == 0 || weight_k == 0 {
        candle::bail!("cuTile routed grouped matmul requires nonzero N and K dimensions")
    }
    if num_experts != routing.num_experts {
        candle::bail!(
            "cuTile routed grouped matmul expected {} experts, got {num_experts}",
            routing.num_experts
        )
    }
    if input_k != weight_k {
        candle::bail!(
            "cuTile routed grouped matmul input K {input_k} does not match weight K {weight_k}"
        )
    }
    let input_len = input_rows
        .checked_mul(input_k)
        .ok_or_else(|| candle::Error::msg("cuTile routed grouped matmul input size overflow"))?;
    let weight_len = num_experts
        .checked_mul(n_size)
        .and_then(|size| size.checked_mul(weight_k))
        .ok_or_else(|| candle::Error::msg("cuTile routed grouped matmul weight size overflow"))?;
    checked_i32(input_len, "input element count")?;
    checked_i32(weight_len, "weight element count")?;
    let expected_rows = match input_mode {
        MoeInputMode::TokenRows => routing.num_tokens,
        MoeInputMode::RoutedRows => routing.num_tokens * routing.top_k,
    };
    if input_rows != expected_rows {
        candle::bail!(
            "cuTile routed grouped matmul expected {expected_rows} input rows, got {input_rows}"
        )
    }
    checked_i32(n_size, "output size")?;
    checked_i32(weight_k, "reduction size")?;
    same_device(input, &routing.device, "input")?;
    same_device(expert_weights, &routing.device, "expert weights")?;
    if let Some(route_weights) = route_weights {
        if route_weights.dtype() != DType::F32 {
            candle::bail!("cuTile MoE route weights must be f32")
        }
        if route_weights.dims2()? != (routing.num_tokens, routing.top_k) {
            candle::bail!(
                "cuTile MoE route weights must have shape [{}, {}]",
                routing.num_tokens,
                routing.top_k
            )
        }
        same_device(route_weights, &routing.device, "route weights")?;
    }

    let (input_storage, input_layout) = input.storage_and_layout();
    let (weight_storage, weight_layout) = expert_weights.storage_and_layout();
    require_contiguous(input_layout, "input")?;
    require_contiguous(weight_layout, "expert weights")?;
    let route_weight_storage = route_weights.map(Tensor::storage_and_layout);
    if let Some((_, layout)) = &route_weight_storage {
        require_contiguous(layout, "route weights")?;
    }

    let output_len = routing
        .num_tokens
        .checked_mul(routing.top_k)
        .and_then(|rows| rows.checked_mul(n_size))
        .ok_or_else(|| candle::Error::msg("cuTile routed grouped matmul output size overflow"))?;
    checked_i32(output_len, "output element count")?;
    let mut output = routing.device.alloc_zeros::<bf16>(output_len)?;
    let context = cutile::CutileContext::new(&routing.device)?;
    let input_cuda = cuda_storage(&input_storage, "input")?;
    let weight_cuda = cuda_storage(&weight_storage, "expert weights")?;
    let input_ptr = context.read_storage::<bf16>(input_cuda, input_layout)?;
    let weight_ptr = context.read_storage::<bf16>(weight_cuda, weight_layout)?;
    let output_ptr = context.write(&mut output, 0)?;
    let sorted_ptr = context.read(&routing.sorted_token_ids, 0)?;
    let expert_ptr = context.read(&routing.expert_ids, 0)?;
    let count_ptr = context.read(&routing.num_tokens_post_padded, 0)?;
    let route_weight_ptr = if let Some((storage, layout)) = &route_weight_storage {
        context.read_storage::<f32>(cuda_storage(storage, "route weights")?, layout)?
    } else {
        context.read(&routing.route_weight_dummy, 0)?
    };

    let config = routing.config;
    let grid_x = routing
        .em
        .div_ceil(config.bm as usize)
        .checked_mul(n_size.div_ceil(config.bn as usize))
        .ok_or_else(|| candle::Error::msg("cuTile routed grouped matmul launch grid overflow"))?;
    let generics = vec![
        config.bm.to_string(),
        config.bn.to_string(),
        config.bk.to_string(),
        config.group_m.to_string(),
        match input_mode {
            MoeInputMode::TokenRows => routing.top_k,
            MoeInputMode::RoutedRows => 1,
        }
        .to_string(),
        usize::from(route_weights.is_some()).to_string(),
    ];
    let launcher = unsafe {
        kernels::routed_grouped_matmul(
            output_ptr.device_pointer(),
            input_ptr.device_pointer(),
            weight_ptr.device_pointer(),
            sorted_ptr.device_pointer(),
            expert_ptr.device_pointer(),
            count_ptr.device_pointer(),
            route_weight_ptr.device_pointer(),
            n_size as i32,
            weight_k as i32,
            routing.em as i32,
            (routing.num_tokens * routing.top_k) as i32,
        )
    }
    .generics(generics)
    .grid((checked_u32(grid_x, "launch grid")?, 1, 1));

    if compile_only {
        cutile::kernel("routed grouped matmul compile", || {
            launcher.compile_on(context.stream())
        })?;
    } else {
        cutile::kernel("routed grouped matmul launch", || unsafe {
            launcher.async_on(context.stream())
        })?;
    }
    drop((
        route_weight_ptr,
        count_ptr,
        expert_ptr,
        sorted_ptr,
        output_ptr,
        weight_ptr,
        input_ptr,
    ));

    Ok(Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, routing.device.clone())),
        (routing.num_tokens * routing.top_k, n_size),
    )))
}

#[allow(clippy::too_many_arguments)]
fn launch_alignment<T: CudaDType + DeviceRepr>(
    storage: &Storage,
    layout: &Layout,
    sorted_token_ids: &mut CudaSlice<i32>,
    expert_ids: &mut CudaSlice<i32>,
    num_tokens_post_padded: &mut CudaSlice<i32>,
    cumsum: &mut CudaSlice<i32>,
    num_experts: usize,
    block_size: usize,
    num_routes: usize,
    em: usize,
    device: &CudaDevice,
) -> Result<()> {
    let storage = match storage {
        Storage::Cuda(storage) => storage,
        _ => candle::bail!("cuTile MoE top-k ids must be on CUDA"),
    };
    let topk_ids = storage.as_cuda_slice::<T>()?;
    let stream = device.cuda_stream();
    let (topk_ptr, topk_guard) = topk_ids.device_ptr(&stream);
    let topk_offset = layout
        .start_offset()
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| candle::Error::msg("cuTile MoE top-k id offset overflow"))?;
    let topk_ptr = topk_ptr
        .checked_add(topk_offset as u64)
        .ok_or_else(|| candle::Error::msg("cuTile MoE top-k id pointer overflow"))?;
    let (sorted_ptr, sorted_guard) = sorted_token_ids.device_ptr_mut(&stream);
    let (expert_ptr, expert_guard) = expert_ids.device_ptr_mut(&stream);
    let (count_ptr, count_guard) = num_tokens_post_padded.device_ptr_mut(&stream);
    let (cumsum_ptr, cumsum_guard) = cumsum.device_ptr_mut(&stream);
    let status = unsafe {
        ffi::candle_launch_moe_align(
            topk_ptr as *const i32,
            sorted_ptr as *mut i32,
            expert_ptr as *mut i32,
            count_ptr as *mut i32,
            cumsum_ptr as *mut i32,
            num_experts as i32,
            block_size as i32,
            num_routes as i32,
            em as i32,
            stream.cu_stream() as *mut std::ffi::c_void,
        )
    };
    if status != 0 {
        candle::bail!("cuTile MoE route alignment CUDA launch failed with error {status}")
    }
    drop((
        cumsum_guard,
        count_guard,
        expert_guard,
        sorted_guard,
        topk_guard,
    ));
    Ok(())
}

fn default_config(num_tokens: usize, num_experts: usize) -> TileConfig {
    let bm = if num_tokens <= 32 {
        16
    } else if num_tokens <= 96 {
        32
    } else if num_tokens <= 512 {
        64
    } else {
        128
    };
    TileConfig {
        bm,
        bn: if num_tokens <= 64 { 64 } else { 128 },
        bk: if num_tokens <= 64 { 128 } else { 64 },
        group_m: if num_tokens / num_experts > 128 {
            16
        } else {
            1
        },
    }
}

fn require_contiguous(layout: &Layout, name: &str) -> Result<()> {
    if !layout.is_contiguous() {
        candle::bail!("cuTile routed grouped matmul {name} must be contiguous")
    }
    Ok(())
}

fn cuda_storage<'a>(storage: &'a Storage, name: &str) -> Result<&'a CudaStorage> {
    match storage {
        Storage::Cuda(storage) => Ok(storage),
        _ => candle::bail!("cuTile routed grouped matmul {name} must be on CUDA"),
    }
}

fn same_device(tensor: &Tensor, device: &CudaDevice, name: &str) -> Result<()> {
    let tensor_device = tensor.device().as_cuda_device()?;
    if tensor_device.id() != device.id() {
        candle::bail!("cuTile routed grouped matmul {name} is on a different CUDA device")
    }
    Ok(())
}

fn checked_i32(value: impl TryInto<i32>, name: &str) -> Result<i32> {
    value
        .try_into()
        .map_err(|_| candle::Error::msg(format!("cuTile MoE {name} exceeds i32 range")))
}

fn checked_u32(value: impl TryInto<u32>, name: &str) -> Result<u32> {
    value
        .try_into()
        .map_err(|_| candle::Error::msg(format!("cuTile MoE {name} exceeds u32 range")))
}

#[cfg(test)]
mod tests {
    use super::default_config;

    #[test]
    fn tile_config_thresholds() {
        assert_eq!(default_config(32, 4).bm, 16);
        assert_eq!(default_config(33, 4).bm, 32);
        assert_eq!(default_config(97, 4).bm, 64);
        assert_eq!(default_config(513, 4).bm, 128);
        assert_eq!(default_config(64, 4).bn, 64);
        assert_eq!(default_config(65, 4).bn, 128);
        assert_eq!(default_config(64, 4).bk, 128);
        assert_eq!(default_config(65, 4).bk, 64);
        assert_eq!(default_config(128, 1).group_m, 1);
        assert_eq!(default_config(129, 1).group_m, 16);
    }
}
