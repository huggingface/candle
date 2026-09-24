use crate::utils::{BufferOffset, EncoderProvider};
use crate::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, EncoderParam, Kernels,
    MetalKernelError, Output, Source,
};
use objc2_metal::MTLSize;

/// Fused gated-DeltaNet single-token decode step -- one Metal dispatch
/// instead of the ~9 serialized elementwise/reduction ops the equivalent
/// candle-tensor-op sequence would need. See `metal_src/gdn.metal`'s own
/// doc comment for the exact math and the functional-state-write
/// correctness argument (`state_out` must be a fresh buffer, `state_in`
/// is read-only and untouched).
///
/// Shapes (all contiguous F32): `q`/`k`: `[b, h, hk]`; `v`: `[b, h, hv]`;
/// `g`/`beta`: `[b, h]` (`g` is the raw decay-gate log, **not**
/// exponentiated -- the kernel exponentiates it internally, one dispatch
/// fewer than passing the already-`exp`'d gate); `state_in`/`state_out`:
/// `[b, h, hk, hv]`; `out`: `[b, h, hv]`.
///
/// Every read input takes a `BufferOffset`, not a bare `Buffer`: callers
/// commonly derive `q`/`k`/`v` from slices of a shared QKV-projection
/// buffer via `narrow`, which can carry a genuine nonzero byte offset
/// even when the resulting tensor is itself contiguous -- binding at a
/// fixed offset 0 regardless of the tensor's real layout silently reads
/// the wrong region of the buffer. Every input takes a real offset, not
/// just the ones a particular call site happens to need it for, since
/// which inputs need a nonzero offset is caller-dependent.
///
/// Caller must bind `state_out` and `out` via the write path (this
/// function already does, via `Output::new`) -- binding them read-only
/// would leave a later dispatch's read of `state_out` without a
/// hazard-tracking barrier under `HazardTrackingModeUntracked`.
#[allow(clippy::too_many_arguments)]
pub fn call_gdn_decode_step_f32(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    b: usize,
    h: usize,
    hk: usize,
    hv: usize,
    q: &BufferOffset,
    k: &BufferOffset,
    v: &BufferOffset,
    g: &BufferOffset,
    beta: &BufferOffset,
    state_in: &BufferOffset,
    state_out: &Buffer,
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    #[derive(Debug)]
    #[repr(C)]
    struct GdnStepArgs {
        hk: u32,
        hv: u32,
        h: u32,
    }

    impl EncoderParam for GdnStepArgs {
        fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
            encoder.set_bytes(position, &data);
        }
    }

    let pipeline = kernels.load_pipeline(device, Source::Gdn, "kernel_gdn_decode_step_f32")?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "gdn_decode_step b={b} h={h} hk={hk} hv={hv}");

    let args = GdnStepArgs {
        hk: hk as u32,
        hv: hv as u32,
        h: h as u32,
    };
    set_params!(
        encoder,
        (
            q,
            k,
            v,
            g,
            beta,
            state_in,
            Output::new(state_out),
            Output::new(out),
            args
        )
    );

    let grid_dims = MTLSize {
        width: hv,
        height: h,
        depth: b,
    };
    let group_dims = MTLSize {
        width: hv.min(64),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_threads(grid_dims, group_dims);
    Ok(())
}

#[repr(C)]
struct GdnConv1dArgs {
    seq_len: u32,
    hist_len: u32,
    channels: u32,
    kernel_size: u32,
}

impl EncoderParam for GdnConv1dArgs {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_bytes(position, &data);
    }
}

/// Fused causal depthwise conv1d + silu -- the output half. See
/// `metal_src/gdn.metal`'s own doc comment for the exact math (operates
/// conceptually on `history ++ x` without ever materializing the
/// concatenation) and the dispatch-count arithmetic this replaces.
///
/// Shapes (all contiguous F32): `x`: `[b, seq_len, channels]`; `history`:
/// `[b, hist_len, channels]`; `weight`: `[channels, kernel_size]` --
/// **caller must canonicalize to this exact layout** (a raw checkpoint
/// tensor can arrive as `[channels, kernel]` or `[kernel, channels]`; do the
/// transpose once at load time, not per call, and never pass the
/// non-canonical layout here). `out`: `[b, seq_len, channels]`.
///
/// Every read input takes a `BufferOffset`, not a bare `Buffer` -- same
/// discipline as `call_gdn_decode_step_f32` above, after the real
/// nonzero-offset bug that kernel found live: `history` in particular is
/// exactly the kind of tensor (a restored/cloned cache view, or a
/// `narrow()`'d slice) most likely to carry a real nonzero byte offset in
/// production, not just in a synthetic test.
#[allow(clippy::too_many_arguments)]
pub fn call_gdn_causal_conv1d_output_f32(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    b: usize,
    seq_len: usize,
    hist_len: usize,
    channels: usize,
    kernel_size: usize,
    x: &BufferOffset,
    history: &BufferOffset,
    weight: &BufferOffset,
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline =
        kernels.load_pipeline(device, Source::Gdn, "kernel_gdn_causal_conv1d_output_f32")?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "gdn_causal_conv1d_output b={b} seq_len={seq_len} hist_len={hist_len} channels={channels} kernel_size={kernel_size}");

    let args = GdnConv1dArgs {
        seq_len: seq_len as u32,
        hist_len: hist_len as u32,
        channels: channels as u32,
        kernel_size: kernel_size as u32,
    };
    set_params!(encoder, (x, history, weight, Output::new(out), args));

    let grid_dims = MTLSize {
        width: channels,
        height: seq_len,
        depth: b,
    };
    let group_dims = MTLSize {
        width: channels.min(64),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_threads(grid_dims, group_dims);
    Ok(())
}

/// Fused causal depthwise conv1d -- the next-conv-state half, companion to
/// `call_gdn_causal_conv1d_output_f32` above (same inputs, no `weight`,
/// different output). `new_state` is written functionally (a fresh buffer,
/// `history` read-only and untouched) -- same correctness discipline as
/// `state_out` in `call_gdn_decode_step_f32`: a prior session-checkpoint
/// clone of `history` must survive this call unchanged, and the caller must
/// bind `new_state` via the write path (this function already does) so the
/// *next* call's read of it gets a barrier under this fork's
/// `HazardTrackingModeUntracked` convention.
///
/// Shapes: `x`: `[b, seq_len, channels]`; `history`: `[b, hist_len,
/// channels]`; `new_state`: `[b, hist_len, channels]`. Correct (a no-op
/// grid) when `hist_len == 0`.
#[allow(clippy::too_many_arguments)]
pub fn call_gdn_causal_conv1d_state_f32(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    b: usize,
    seq_len: usize,
    hist_len: usize,
    channels: usize,
    x: &BufferOffset,
    history: &BufferOffset,
    new_state: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline =
        kernels.load_pipeline(device, Source::Gdn, "kernel_gdn_causal_conv1d_state_f32")?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "gdn_causal_conv1d_state b={b} seq_len={seq_len} hist_len={hist_len} channels={channels}"
    );

    let args = GdnConv1dArgs {
        seq_len: seq_len as u32,
        hist_len: hist_len as u32,
        channels: channels as u32,
        kernel_size: 0,
    };
    set_params!(encoder, (x, history, Output::new(new_state), args));

    let grid_dims = MTLSize {
        width: channels,
        height: hist_len,
        depth: b,
    };
    let group_dims = MTLSize {
        width: channels.min(64),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_threads(grid_dims, group_dims);
    Ok(())
}

#[repr(C)]
struct GdnL2NormArgs {
    seq_len: u32,
    heads: u32,
    dim: u32,
    scale: f32,
    eps: f32,
}

impl EncoderParam for GdnL2NormArgs {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_bytes(position, &data);
    }
}

/// Fused L2-normalize + scale -- the q/k half of gated-DeltaNet's
/// preprocessing gating tail. See `metal_src/gdn.metal`'s own doc comment
/// for the exact math (eps is added to the sum of squares, not the mean --
/// do not substitute a generic RMS-norm kernel here, the epsilon placement
/// differs). Called once for q (with a query-side scale factor) and once
/// for k (`scale = 1.0`).
///
/// Shapes (contiguous F32): `x`/`out`: `[b, seq_len, heads, dim]`.
#[allow(clippy::too_many_arguments)]
pub fn call_gdn_l2_normalize_scale_f32(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    b: usize,
    seq_len: usize,
    heads: usize,
    dim: usize,
    scale: f32,
    eps: f32,
    x: &BufferOffset,
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline =
        kernels.load_pipeline(device, Source::Gdn, "kernel_gdn_l2_normalize_scale_f32")?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "gdn_l2_normalize_scale b={b} seq_len={seq_len} heads={heads} dim={dim} scale={scale}"
    );

    let args = GdnL2NormArgs {
        seq_len: seq_len as u32,
        heads: heads as u32,
        dim: dim as u32,
        scale,
        eps,
    };
    set_params!(encoder, (x, Output::new(out), args));

    let grid_dims = MTLSize {
        width: heads,
        height: seq_len,
        depth: b,
    };
    let group_dims = MTLSize {
        width: heads.min(64),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_threads(grid_dims, group_dims);
    Ok(())
}

#[repr(C)]
struct GdnDecayBetaArgs {
    seq_len: u32,
    heads: u32,
}

impl EncoderParam for GdnDecayBetaArgs {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_bytes(position, &data);
    }
}

/// Fused decay-gate softplus + beta sigmoid -- the gating half of
/// gated-DeltaNet's preprocessing gating tail. See `metal_src/gdn.metal`'s
/// own doc comment for the exact math. `dt_bias`/`a` are indexed directly
/// by head (`[heads]`), not pre-broadcast -- this eliminates the caller's
/// own broadcast step entirely, not just the elementwise math around it.
///
/// Shapes (contiguous F32): `alpha_logits`/`beta_logits`/`g_out`/`beta_out`:
/// `[b, seq_len, heads]`; `dt_bias`/`a`: `[heads]`.
#[allow(clippy::too_many_arguments)]
pub fn call_gdn_decay_beta_gate_f32(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    b: usize,
    seq_len: usize,
    heads: usize,
    alpha_logits: &BufferOffset,
    dt_bias: &BufferOffset,
    a: &BufferOffset,
    beta_logits: &BufferOffset,
    g_out: &Buffer,
    beta_out: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Gdn, "kernel_gdn_decay_beta_gate_f32")?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "gdn_decay_beta_gate b={b} seq_len={seq_len} heads={heads}"
    );

    let args = GdnDecayBetaArgs {
        seq_len: seq_len as u32,
        heads: heads as u32,
    };
    set_params!(
        encoder,
        (
            alpha_logits,
            dt_bias,
            a,
            beta_logits,
            Output::new(g_out),
            Output::new(beta_out),
            args
        )
    );

    let grid_dims = MTLSize {
        width: heads,
        height: seq_len,
        depth: b,
    };
    let group_dims = MTLSize {
        width: heads.min(64),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_threads(grid_dims, group_dims);
    Ok(())
}

/// Chunk size for `call_gdn_chunked_scan_solve_f32` -- must match
/// `metal_src/gdn.metal`'s `GDN_SCAN_CHUNK` and ratatoskr's own
/// `CHUNK_SIZE` (`qwen3_5_linear_attn_scan.rs`) exactly. The kernel
/// hardcodes this as a compile-time constant for the dispatch-grid/
/// buffer-stride arithmetic, not for register-array unrolling -- an
/// earlier version of this kernel *did* hold a compile-time-sized
/// per-thread array for exactly that reason, and was found to
/// miscompile at this size (see the kernel's own doc comment for the
/// real bug and the fix, which deliberately avoids a per-thread array
/// of this size entirely).
pub const GDN_SCAN_CHUNK: usize = 64;

#[repr(C)]
struct GdnScanSolveArgs {
    bhnc: u32,
}

impl EncoderParam for GdnScanSolveArgs {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_bytes(position, &data);
    }
}

/// Column-parallel forward-substitution solve for gated-DeltaNet's chunked
/// prefill scan (Phase 3.1a). See `metal_src/gdn.metal`'s own doc comment
/// for the derivation (`(I - A)X = I` decomposed by columns) and why this
/// layout -- one thread per column, zero threadgroup memory, zero barriers
/// -- was chosen over an in-place threadgroup-tile solve.
///
/// Shapes (contiguous F32): `a_mat`/`attn`: `[bhnc, GDN_SCAN_CHUNK,
/// GDN_SCAN_CHUNK]`, where `bhnc` is the caller's own flattened `batch *
/// n_heads * num_chunks` (matching the tensor path's own `bhnc` reshape in
/// `gated_delta_rule_chunked`). `a_mat` must be strictly lower-triangular
/// (zero diagonal and above) -- the kernel does not verify this.
///
/// `a_mat` takes a `BufferOffset`, not a bare `Buffer` -- same discipline
/// as every other kernel in this file, after the real nonzero-offset bug
/// `call_gdn_decode_step_f32` found live: `a_mat` is exactly the kind of
/// tensor (built via `narrow`/`contiguous` chains in the tensor path) that
/// could carry a real nonzero byte offset in a future caller.
///
/// Caller must bind `attn` via the write path (this function already does,
/// via `Output::new`) -- binding it read-only would leave any dependent
/// dispatch (a future 3.1b/c fold, or the tensor-side comparison this
/// stage is verified against) without a barrier under this fork's
/// `HazardTrackingModeUntracked` convention, the same bug class as the
/// mm_id-counts incident.
pub fn call_gdn_chunked_scan_solve_f32(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    bhnc: usize,
    a_mat: &BufferOffset,
    attn: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline =
        kernels.load_pipeline(device, Source::Gdn, "kernel_gdn_chunked_scan_solve_f32")?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "gdn_chunked_scan_solve bhnc={bhnc}");

    let args = GdnScanSolveArgs { bhnc: bhnc as u32 };
    set_params!(encoder, (a_mat, Output::new(attn), args));

    let grid_dims = MTLSize {
        width: GDN_SCAN_CHUNK,
        height: bhnc,
        depth: 1,
    };
    let group_dims = MTLSize {
        width: GDN_SCAN_CHUNK.min(64),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_threads(grid_dims, group_dims);
    Ok(())
}

#[repr(C)]
struct GdnScanBuildSolveArgs {
    hk: u32, // bhnc is deliberately not a field -- the kernel never reads it, see gdn.metal
}

impl EncoderParam for GdnScanBuildSolveArgs {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_bytes(position, &data);
    }
}

/// Phase 3.1b: folds `a_mat`'s construction (mask/`kk`/triangle) into
/// the solve, this fork's first kernel using threadgroup memory and a
/// barrier. See `metal_src/gdn.metal`'s own doc comment for the full
/// derivation, the numerics note (this kernel's own running-suffix-sum
/// `a_mat` differs from the tensor path's two-prefix-sum-subtraction
/// form by ordinary summation-order drift -- tolerance-close, not
/// bit-identical, same as every other reassociation in this design),
/// and why no thread ever needs a bounds-check `return` (dispatch is
/// sized to exactly `(GDN_SCAN_CHUNK, bhnc)` threads, so every thread
/// is always in range -- an early return before this kernel's barrier
/// would be undefined behavior).
///
/// Shapes (contiguous F32): `k_c`: `[bhnc, GDN_SCAN_CHUNK, hk]`;
/// `log_g_c`/`beta_c`: `[bhnc, GDN_SCAN_CHUNK]`; `attn` (output,
/// functional): `[bhnc, GDN_SCAN_CHUNK, GDN_SCAN_CHUNK]`. `q_c` is not
/// an input -- `a_mat`'s construction never reads it (verified against
/// `gated_delta_rule_chunked`'s own steps 3a-3c).
///
/// Every read input takes a `BufferOffset`, not a bare `Buffer` --
/// same discipline as every other kernel in this file.
#[allow(clippy::too_many_arguments)]
pub fn call_gdn_chunked_scan_build_and_solve_f32(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    bhnc: usize,
    hk: usize,
    k_c: &BufferOffset,
    log_g_c: &BufferOffset,
    beta_c: &BufferOffset,
    attn: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(
        device,
        Source::Gdn,
        "kernel_gdn_chunked_scan_build_and_solve_f32",
    )?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "gdn_chunked_scan_build_and_solve bhnc={bhnc} hk={hk}"
    );

    let args = GdnScanBuildSolveArgs { hk: hk as u32 };
    set_params!(encoder, (k_c, log_g_c, beta_c, Output::new(attn), args));

    let grid_dims = MTLSize {
        width: GDN_SCAN_CHUNK,
        height: bhnc,
        depth: 1,
    };
    let group_dims = MTLSize {
        width: GDN_SCAN_CHUNK,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_threads(grid_dims, group_dims);
    Ok(())
}
