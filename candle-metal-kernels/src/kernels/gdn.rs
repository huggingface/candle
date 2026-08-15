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
