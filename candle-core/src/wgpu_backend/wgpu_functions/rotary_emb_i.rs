use candle_wgpu_kernels::{Constants};

use super::*;

#[allow(clippy::too_many_arguments)]
pub fn queue_rotary_emb_i(
    dev: &WgpuDevice,
    buffer_src: (BufferReferenceId, u32),
    buffer_cos: (BufferReferenceId, u32),
    buffer_sin: (BufferReferenceId, u32),
    dtype: crate::DType,
    buffer_dest: BufferReferenceId,
    unbatched: bool,
    bhtd: (u32, u32, u32, u32),
) -> crate::Result<()> {
    let (b, h, t, d) = bhtd;
    let (buffer_src, src_offset) = buffer_src;
    let (buffer_cos, cos_offset) = buffer_cos;
    let (buffer_sin, sin_offset) = buffer_sin;

    // D must be even for interleaved rotary
    debug_assert!(d % 2 == 0, "RotaryEmbI requires even head_dim (d)");

    // ---- Workgroup layout must match WGSL ----
    // WGSL:
    //   @workgroup_size(8,8,1)
    //   global_id.x in [0 .. B*H)
    //   global_id.y in [0 .. T*(D/2))
    //
    // So we need:
    //   num_invocations_x = B * H
    //   num_invocations_y = T * (D/2)
    //
    // Dispatch uses workgroup counts:
    //   workgroups_x = ceil_div(num_invocations_x, 8)
    //   workgroups_y = ceil_div(num_invocations_y, 8)

    let workgroup_size_x: u32 = 8;
    let workgroup_size_y: u32 = 8;

    let num_invocations_x = b * h;
    let num_invocations_y = t * (d / 2);

    fn ceil_div(a: u32, b: u32) -> u32 {
        if a == 0 { 0 } else { (a + b - 1) / b }
    }

    let workgroup_count_x = ceil_div(num_invocations_x, workgroup_size_x);
    let workgroup_count_y = ceil_div(num_invocations_y, workgroup_size_y);

    let mut queue = dev.get_queue();

    // op_meta[0..4] = B, H, T, D, unbatched (as in WGSL)
    queue.add(b);
    queue.add(h);
    queue.add(t);
    queue.add(d);
    queue.add(src_offset);
    queue.add(cos_offset);
    queue.add(sin_offset);

    // op_meta[4] = unbatched flag (0/1)
    queue.add_const(Constants::Constv0, unbatched);

    let pipeline = queue.get_pipeline(Pipelines::RotaryEmbI(
        dev.get_dtype(dtype)?,
        candle_wgpu_kernels::rotary_emb_i::Functions::RotaryEmbI,
    ));

    // dest + 3 inputs (src, cos, sin)
    let bind_group =
        dev.create_bind_group_input3(buffer_dest, buffer_src, buffer_cos, buffer_sin, dtype.into());

    queue.enqueue_workgroups(
        pipeline,
        bind_group,
        workgroup_count_x,
        workgroup_count_y,
        1,
        // "num elements" hint; you already had b*h*t*d here which is fine
        (b * h * t * d) as usize,
    );

    Ok(())
}