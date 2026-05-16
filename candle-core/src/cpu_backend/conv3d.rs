use std::borrow::Cow;

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    conv::ParamsConv3D,
    cpu_backend::{Map2, MatMul},
    shape::dims5,
    DType, Layout, Result, WithDType,
};

pub(super) struct Conv3D<'a>(pub(super) &'a ParamsConv3D);
pub(super) struct Conv3DBackwardInput<'a>(pub(super) &'a ParamsConv3D);
pub(super) struct Conv3DBackwardWeight<'a>(pub(super) &'a ParamsConv3D);

#[allow(dead_code)]
enum Conv3dImpl {
    TiledIm2Col,
    Direct,
}

// Unlike conv2d, there is intentionally no full-im2col Conv3d variant here.
// The full 3D im2col buffer can become very large, so the default general
// implementation materializes one output tile at a time.
const DEFAULT_CONV3D_IMPL: Conv3dImpl = Conv3dImpl::TiledIm2Col;
const CONV3D_TILE_SIZE: usize = 256;

fn src_index(
    out: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    padding: usize,
    input: usize,
) -> Option<usize> {
    let src = out * stride + kernel * dilation;
    if src < padding {
        None
    } else {
        let src = src - padding;
        (src < input).then_some(src)
    }
}

impl Map2 for Conv3D<'_> {
    const OP: &'static str = "conv3d";

    fn f<T: WithDType + num_traits::Num + Copy + 'static>(
        &self,
        inp: &[T],
        inp_l: &Layout,
        k: &[T],
        k_l: &Layout,
    ) -> Result<Vec<T>> {
        let p = self.0;
        // The generic CPU MatMul path currently supports F16/F32/F64 here.
        // Keep direct Conv3d as a fallback so dtypes such as BF16 still work.
        if !matches!(T::DTYPE, DType::F16 | DType::F32 | DType::F64) {
            return conv3d_direct(p, inp, inp_l, k, k_l);
        }
        if p.k_d == 1
            && p.k_h == 1
            && p.k_w == 1
            && p.stride == [1, 1, 1]
            && p.padding == [0, 0, 0]
            && p.dilation == [1, 1, 1]
        {
            return conv3d_1x1(p, inp, inp_l, k, k_l);
        }

        match DEFAULT_CONV3D_IMPL {
            Conv3dImpl::TiledIm2Col => conv3d_tiled(p, inp, inp_l, k, k_l),
            Conv3dImpl::Direct => conv3d_direct(p, inp, inp_l, k, k_l),
        }
    }
}

fn conv3d_1x1<T: WithDType + num_traits::Num + Copy + 'static>(
    p: &ParamsConv3D,
    inp: &[T],
    inp_l: &Layout,
    k: &[T],
    k_l: &Layout,
) -> Result<Vec<T>> {
    let inp = &inp[inp_l.start_offset()..];
    let (inp_s0, inp_s1, inp_s2, inp_s3, inp_s4) = dims5(inp_l.stride())?;
    let k = &k[k_l.start_offset()..];
    let (k_s0, k_s1, _k_s2, _k_s3, _k_s4) = dims5(k_l.stride())?;
    let spatial_size = p.i_d * p.i_h * p.i_w;
    let dst = vec![T::zero(); p.b_size * p.c_out * spatial_size];
    let k_reshaped: Cow<[T]> = if k_s0 == p.c_in && k_s1 == 1 {
        Cow::Borrowed(&k[..p.c_out * p.c_in])
    } else {
        let mut k_reshaped = Vec::with_capacity(p.c_out * p.c_in);
        for dst_c_idx in 0..p.c_out {
            for src_c_idx in 0..p.c_in {
                let k_idx = dst_c_idx * k_s0 + src_c_idx * k_s1;
                k_reshaped.push(k[k_idx]);
            }
        }
        Cow::Owned(k_reshaped)
    };
    let k_layout = Layout::contiguous((p.c_out, p.c_in));

    (0..p.b_size).into_par_iter().try_for_each(|b_idx| {
        let mut inp_reshaped = Vec::with_capacity(p.c_in * spatial_size);
        for c_idx in 0..p.c_in {
            for d_idx in 0..p.i_d {
                for h_idx in 0..p.i_h {
                    for w_idx in 0..p.i_w {
                        let inp_idx = b_idx * inp_s0
                            + c_idx * inp_s1
                            + d_idx * inp_s2
                            + h_idx * inp_s3
                            + w_idx * inp_s4;
                        inp_reshaped.push(inp[inp_idx]);
                    }
                }
            }
        }
        let inp_layout = Layout::contiguous((p.c_in, spatial_size));
        let result = MatMul((1, p.c_out, spatial_size, p.c_in)).f(
            &k_reshaped,
            &k_layout,
            &inp_reshaped,
            &inp_layout,
        )?;
        let out_offset = b_idx * p.c_out * spatial_size;
        for (idx, value) in result.iter().enumerate() {
            unsafe {
                let ptr = dst.as_ptr().add(out_offset + idx) as *mut T;
                *ptr = *value;
            }
        }
        Ok::<(), crate::Error>(())
    })?;

    Ok(dst)
}

fn conv3d_tiled<T: WithDType + num_traits::Num + Copy + 'static>(
    p: &ParamsConv3D,
    inp: &[T],
    inp_l: &Layout,
    k: &[T],
    k_l: &Layout,
) -> Result<Vec<T>> {
    let inp = &inp[inp_l.start_offset()..];
    let (inp_s0, inp_s1, inp_s2, inp_s3, inp_s4) = dims5(inp_l.stride())?;
    let k = &k[k_l.start_offset()..];
    let (k_s0, k_s1, k_s2, k_s3, k_s4) = dims5(k_l.stride())?;
    let (out_d, out_h, out_w) = (p.out_d()?, p.out_h()?, p.out_w()?);

    let dst = vec![T::zero(); p.b_size * p.c_out * out_d * out_h * out_w];

    let cont_s0 = p.i_d * p.i_h * p.i_w * p.c_in;
    let cont_s1 = p.i_h * p.i_w * p.c_in;
    let cont_s2 = p.i_w * p.c_in;
    let cont_s3 = p.c_in;
    let mut inp_cont = vec![T::zero(); p.b_size * p.c_in * p.i_d * p.i_h * p.i_w];
    for b_idx in 0..p.b_size {
        for d_idx in 0..p.i_d {
            for h_idx in 0..p.i_h {
                for w_idx in 0..p.i_w {
                    for c_idx in 0..p.c_in {
                        let src_idx = b_idx * inp_s0
                            + c_idx * inp_s1
                            + d_idx * inp_s2
                            + h_idx * inp_s3
                            + w_idx * inp_s4;
                        let dst_idx = b_idx * cont_s0
                            + d_idx * cont_s1
                            + h_idx * cont_s2
                            + w_idx * cont_s3
                            + c_idx;
                        inp_cont[dst_idx] = inp[src_idx]
                    }
                }
            }
        }
    }

    // Flatten weights from [c_out, c_in, k_d, k_h, k_w] into
    // [c_out, k_d * k_h * k_w * c_in] for GEMM.
    let k_size = p.c_in * p.k_d * p.k_h * p.k_w;
    let mut k_flat = Vec::with_capacity(p.c_out * k_size);
    for dst_c_idx in 0..p.c_out {
        for kd in 0..p.k_d {
            for kh in 0..p.k_h {
                for kw in 0..p.k_w {
                    for src_c_idx in 0..p.c_in {
                        let k_idx =
                            dst_c_idx * k_s0 + src_c_idx * k_s1 + kd * k_s2 + kh * k_s3 + kw * k_s4;
                        k_flat.push(k[k_idx]);
                    }
                }
            }
        }
    }
    let k_layout = Layout::contiguous((p.c_out, k_size));
    let total_out_pixels = out_d * out_h * out_w;

    (0..p.b_size).into_par_iter().try_for_each(|b_idx| {
        let inp_offset = b_idx * cont_s0;
        let out_batch_offset = b_idx * p.c_out * total_out_pixels;
        let num_tiles = total_out_pixels.div_ceil(CONV3D_TILE_SIZE);
        (0..num_tiles).into_par_iter().try_for_each(|tile_idx| {
            let tile_start = tile_idx * CONV3D_TILE_SIZE;
            let tile_end = (tile_start + CONV3D_TILE_SIZE).min(total_out_pixels);
            let tile_size = tile_end - tile_start;
            // Tile layout is [k_size, tile_size], where each column is one
            // flattened 3D receptive field.
            let mut col_tile = vec![T::zero(); k_size * tile_size];

            for (local_idx, out_idx) in (tile_start..tile_end).enumerate() {
                let dst_d = out_idx / (out_h * out_w);
                let dst_h = (out_idx / out_w) % out_h;
                let dst_w = out_idx % out_w;
                for kd in 0..p.k_d {
                    let Some(src_d) =
                        src_index(dst_d, kd, p.stride[0], p.dilation[0], p.padding[0], p.i_d)
                    else {
                        continue;
                    };
                    for kh in 0..p.k_h {
                        let Some(src_h) =
                            src_index(dst_h, kh, p.stride[1], p.dilation[1], p.padding[1], p.i_h)
                        else {
                            continue;
                        };
                        for kw in 0..p.k_w {
                            let Some(src_w) = src_index(
                                dst_w,
                                kw,
                                p.stride[2],
                                p.dilation[2],
                                p.padding[2],
                                p.i_w,
                            ) else {
                                continue;
                            };
                            let patch_offset =
                                (((kd * p.k_h + kh) * p.k_w + kw) * p.c_in) * tile_size + local_idx;
                            let inp_idx =
                                inp_offset + src_d * cont_s1 + src_h * cont_s2 + src_w * cont_s3;
                            for src_c_idx in 0..p.c_in {
                                col_tile[patch_offset + src_c_idx * tile_size] =
                                    inp_cont[inp_idx + src_c_idx];
                            }
                        }
                    }
                }
            }

            let col_layout = Layout::contiguous((k_size, tile_size));
            let result = MatMul((1, p.c_out, tile_size, k_size)).f(
                &k_flat,
                &k_layout,
                &col_tile,
                &col_layout,
            )?;

            for (local_idx, out_idx) in (tile_start..tile_end).enumerate() {
                for dst_c_idx in 0..p.c_out {
                    let dst_idx = out_batch_offset + dst_c_idx * total_out_pixels + out_idx;
                    let result_idx = dst_c_idx * tile_size + local_idx;
                    unsafe {
                        let ptr = dst.as_ptr().add(dst_idx) as *mut T;
                        *ptr = result[result_idx];
                    }
                }
            }
            Ok::<(), crate::Error>(())
        })
    })?;

    Ok(dst)
}

fn conv3d_direct<T: WithDType + num_traits::Num + Copy + 'static>(
    p: &ParamsConv3D,
    inp: &[T],
    inp_l: &Layout,
    k: &[T],
    k_l: &Layout,
) -> Result<Vec<T>> {
    let inp = &inp[inp_l.start_offset()..];
    let k = &k[k_l.start_offset()..];
    let (inp_s0, inp_s1, inp_s2, inp_s3, inp_s4) = dims5(inp_l.stride())?;
    let (k_s0, k_s1, k_s2, k_s3, k_s4) = dims5(k_l.stride())?;
    let (out_d, out_h, out_w) = (p.out_d()?, p.out_h()?, p.out_w()?);

    let mut dst = vec![T::zero(); p.b_size * p.c_out * out_d * out_h * out_w];
    for b_idx in 0..p.b_size {
        for dst_c_idx in 0..p.c_out {
            for dst_d in 0..out_d {
                for dst_h in 0..out_h {
                    for dst_w in 0..out_w {
                        let mut acc = T::zero();
                        for src_c_idx in 0..p.c_in {
                            for kd in 0..p.k_d {
                                let Some(src_d) = src_index(
                                    dst_d,
                                    kd,
                                    p.stride[0],
                                    p.dilation[0],
                                    p.padding[0],
                                    p.i_d,
                                ) else {
                                    continue;
                                };
                                for kh in 0..p.k_h {
                                    let Some(src_h) = src_index(
                                        dst_h,
                                        kh,
                                        p.stride[1],
                                        p.dilation[1],
                                        p.padding[1],
                                        p.i_h,
                                    ) else {
                                        continue;
                                    };
                                    for kw in 0..p.k_w {
                                        let Some(src_w) = src_index(
                                            dst_w,
                                            kw,
                                            p.stride[2],
                                            p.dilation[2],
                                            p.padding[2],
                                            p.i_w,
                                        ) else {
                                            continue;
                                        };
                                        let inp_idx = b_idx * inp_s0
                                            + src_c_idx * inp_s1
                                            + src_d * inp_s2
                                            + src_h * inp_s3
                                            + src_w * inp_s4;
                                        let k_idx = dst_c_idx * k_s0
                                            + src_c_idx * k_s1
                                            + kd * k_s2
                                            + kh * k_s3
                                            + kw * k_s4;
                                        acc += inp[inp_idx] * k[k_idx];
                                    }
                                }
                            }
                        }
                        let dst_idx = ((((b_idx * p.c_out + dst_c_idx) * out_d + dst_d) * out_h
                            + dst_h)
                            * out_w)
                            + dst_w;
                        dst[dst_idx] = acc;
                    }
                }
            }
        }
    }
    Ok(dst)
}

impl Map2 for Conv3DBackwardInput<'_> {
    const OP: &'static str = "conv3d-backward-input";

    fn f<T: WithDType>(
        &self,
        grad: &[T],
        grad_l: &Layout,
        k: &[T],
        k_l: &Layout,
    ) -> Result<Vec<T>> {
        // Keep grad-input direct for now. A fast variant needs col2im-style
        // scatter-add with overlapping writes, which is a separate optimization.
        let p = self.0;
        let grad = &grad[grad_l.start_offset()..];
        let k = &k[k_l.start_offset()..];
        let (grad_s0, grad_s1, grad_s2, grad_s3, grad_s4) = dims5(grad_l.stride())?;
        let (k_s0, k_s1, k_s2, k_s3, k_s4) = dims5(k_l.stride())?;
        let (out_d, out_h, out_w) = (p.out_d()?, p.out_h()?, p.out_w()?);

        let mut dst = vec![T::zero(); p.b_size * p.c_in * p.i_d * p.i_h * p.i_w];
        for b_idx in 0..p.b_size {
            for dst_c_idx in 0..p.c_out {
                for dst_d in 0..out_d {
                    for dst_h in 0..out_h {
                        for dst_w in 0..out_w {
                            let grad_idx = b_idx * grad_s0
                                + dst_c_idx * grad_s1
                                + dst_d * grad_s2
                                + dst_h * grad_s3
                                + dst_w * grad_s4;
                            let grad_value = grad[grad_idx];
                            for src_c_idx in 0..p.c_in {
                                for kd in 0..p.k_d {
                                    let Some(src_d) = src_index(
                                        dst_d,
                                        kd,
                                        p.stride[0],
                                        p.dilation[0],
                                        p.padding[0],
                                        p.i_d,
                                    ) else {
                                        continue;
                                    };
                                    for kh in 0..p.k_h {
                                        let Some(src_h) = src_index(
                                            dst_h,
                                            kh,
                                            p.stride[1],
                                            p.dilation[1],
                                            p.padding[1],
                                            p.i_h,
                                        ) else {
                                            continue;
                                        };
                                        for kw in 0..p.k_w {
                                            let Some(src_w) = src_index(
                                                dst_w,
                                                kw,
                                                p.stride[2],
                                                p.dilation[2],
                                                p.padding[2],
                                                p.i_w,
                                            ) else {
                                                continue;
                                            };
                                            let k_idx = dst_c_idx * k_s0
                                                + src_c_idx * k_s1
                                                + kd * k_s2
                                                + kh * k_s3
                                                + kw * k_s4;
                                            let dst_idx =
                                                ((((b_idx * p.c_in + src_c_idx) * p.i_d + src_d)
                                                    * p.i_h
                                                    + src_h)
                                                    * p.i_w)
                                                    + src_w;
                                            dst[dst_idx] += grad_value * k[k_idx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(dst)
    }
}

impl Map2 for Conv3DBackwardWeight<'_> {
    const OP: &'static str = "conv3d-backward-weight";

    fn f<T: WithDType + num_traits::Num + Copy + 'static>(
        &self,
        inp: &[T],
        inp_l: &Layout,
        grad: &[T],
        grad_l: &Layout,
    ) -> Result<Vec<T>> {
        let p = self.0;
        // Reuse the tiled im2col + MatMul strategy for supported dtypes, but
        // preserve the direct path for BF16 and other dtypes.
        if matches!(T::DTYPE, DType::F16 | DType::F32 | DType::F64) {
            conv3d_backward_weight_tiled(p, inp, inp_l, grad, grad_l)
        } else {
            conv3d_backward_weight_direct(p, inp, inp_l, grad, grad_l)
        }
    }
}

fn conv3d_backward_weight_tiled<T: WithDType + num_traits::Num + Copy + 'static>(
    p: &ParamsConv3D,
    inp: &[T],
    inp_l: &Layout,
    grad: &[T],
    grad_l: &Layout,
) -> Result<Vec<T>> {
    let inp = &inp[inp_l.start_offset()..];
    let grad = &grad[grad_l.start_offset()..];
    let (inp_s0, inp_s1, inp_s2, inp_s3, inp_s4) = dims5(inp_l.stride())?;
    let (grad_s0, grad_s1, grad_s2, grad_s3, grad_s4) = dims5(grad_l.stride())?;
    let (out_d, out_h, out_w) = (p.out_d()?, p.out_h()?, p.out_w()?);
    let total_out_pixels = out_d * out_h * out_w;
    let k_size = p.c_in * p.k_d * p.k_h * p.k_w;

    let cont_s0 = p.i_d * p.i_h * p.i_w * p.c_in;
    let cont_s1 = p.i_h * p.i_w * p.c_in;
    let cont_s2 = p.i_w * p.c_in;
    let cont_s3 = p.c_in;
    let mut inp_cont = vec![T::zero(); p.b_size * p.c_in * p.i_d * p.i_h * p.i_w];
    for b_idx in 0..p.b_size {
        for d_idx in 0..p.i_d {
            for h_idx in 0..p.i_h {
                for w_idx in 0..p.i_w {
                    for c_idx in 0..p.c_in {
                        let src_idx = b_idx * inp_s0
                            + c_idx * inp_s1
                            + d_idx * inp_s2
                            + h_idx * inp_s3
                            + w_idx * inp_s4;
                        let dst_idx = b_idx * cont_s0
                            + d_idx * cont_s1
                            + h_idx * cont_s2
                            + w_idx * cont_s3
                            + c_idx;
                        inp_cont[dst_idx] = inp[src_idx]
                    }
                }
            }
        }
    }

    let mut dst = vec![T::zero(); p.c_out * k_size];
    let num_tiles = total_out_pixels.div_ceil(CONV3D_TILE_SIZE);
    for b_idx in 0..p.b_size {
        let inp_offset = b_idx * cont_s0;
        for tile_idx in 0..num_tiles {
            let tile_start = tile_idx * CONV3D_TILE_SIZE;
            let tile_end = (tile_start + CONV3D_TILE_SIZE).min(total_out_pixels);
            let tile_size = tile_end - tile_start;
            // grad_tile is [c_out, tile_size], col_tile is [k_size, tile_size].
            // Their product gives one tile contribution to flattened weights.
            let mut col_tile = vec![T::zero(); k_size * tile_size];
            let mut grad_tile = vec![T::zero(); p.c_out * tile_size];

            for (local_idx, out_idx) in (tile_start..tile_end).enumerate() {
                let dst_d = out_idx / (out_h * out_w);
                let dst_h = (out_idx / out_w) % out_h;
                let dst_w = out_idx % out_w;

                for dst_c_idx in 0..p.c_out {
                    let grad_idx = b_idx * grad_s0
                        + dst_c_idx * grad_s1
                        + dst_d * grad_s2
                        + dst_h * grad_s3
                        + dst_w * grad_s4;
                    grad_tile[dst_c_idx * tile_size + local_idx] = grad[grad_idx];
                }

                for kd in 0..p.k_d {
                    let Some(src_d) =
                        src_index(dst_d, kd, p.stride[0], p.dilation[0], p.padding[0], p.i_d)
                    else {
                        continue;
                    };
                    for kh in 0..p.k_h {
                        let Some(src_h) =
                            src_index(dst_h, kh, p.stride[1], p.dilation[1], p.padding[1], p.i_h)
                        else {
                            continue;
                        };
                        for kw in 0..p.k_w {
                            let Some(src_w) = src_index(
                                dst_w,
                                kw,
                                p.stride[2],
                                p.dilation[2],
                                p.padding[2],
                                p.i_w,
                            ) else {
                                continue;
                            };
                            let patch_offset =
                                (((kd * p.k_h + kh) * p.k_w + kw) * p.c_in) * tile_size + local_idx;
                            let inp_idx =
                                inp_offset + src_d * cont_s1 + src_h * cont_s2 + src_w * cont_s3;
                            for src_c_idx in 0..p.c_in {
                                col_tile[patch_offset + src_c_idx * tile_size] =
                                    inp_cont[inp_idx + src_c_idx];
                            }
                        }
                    }
                }
            }

            let grad_layout = Layout::contiguous((p.c_out, tile_size));
            let col_layout = Layout::contiguous((k_size, tile_size)).transpose(0, 1)?;
            let result = MatMul((1, p.c_out, k_size, tile_size)).f(
                &grad_tile,
                &grad_layout,
                &col_tile,
                &col_layout,
            )?;
            for (idx, value) in result.iter().enumerate() {
                dst[idx] += *value;
            }
        }
    }

    let mut dst_weight = vec![T::zero(); p.c_out * k_size];
    for dst_c_idx in 0..p.c_out {
        for kd in 0..p.k_d {
            for kh in 0..p.k_h {
                for kw in 0..p.k_w {
                    for src_c_idx in 0..p.c_in {
                        let flat_idx = dst_c_idx * k_size
                            + (((kd * p.k_h + kh) * p.k_w + kw) * p.c_in)
                            + src_c_idx;
                        let weight_idx =
                            ((((dst_c_idx * p.c_in + src_c_idx) * p.k_d + kd) * p.k_h + kh)
                                * p.k_w)
                                + kw;
                        dst_weight[weight_idx] = dst[flat_idx];
                    }
                }
            }
        }
    }
    Ok(dst_weight)
}

fn conv3d_backward_weight_direct<T: WithDType>(
    p: &ParamsConv3D,
    inp: &[T],
    inp_l: &Layout,
    grad: &[T],
    grad_l: &Layout,
) -> Result<Vec<T>> {
    let inp = &inp[inp_l.start_offset()..];
    let grad = &grad[grad_l.start_offset()..];
    let (inp_s0, inp_s1, inp_s2, inp_s3, inp_s4) = dims5(inp_l.stride())?;
    let (grad_s0, grad_s1, grad_s2, grad_s3, grad_s4) = dims5(grad_l.stride())?;
    let (out_d, out_h, out_w) = (p.out_d()?, p.out_h()?, p.out_w()?);

    let mut dst = vec![T::zero(); p.c_out * p.c_in * p.k_d * p.k_h * p.k_w];
    for dst_c_idx in 0..p.c_out {
        for src_c_idx in 0..p.c_in {
            for kd in 0..p.k_d {
                for kh in 0..p.k_h {
                    for kw in 0..p.k_w {
                        let mut acc = T::zero();
                        for b_idx in 0..p.b_size {
                            for dst_d in 0..out_d {
                                let Some(src_d) = src_index(
                                    dst_d,
                                    kd,
                                    p.stride[0],
                                    p.dilation[0],
                                    p.padding[0],
                                    p.i_d,
                                ) else {
                                    continue;
                                };
                                for dst_h in 0..out_h {
                                    let Some(src_h) = src_index(
                                        dst_h,
                                        kh,
                                        p.stride[1],
                                        p.dilation[1],
                                        p.padding[1],
                                        p.i_h,
                                    ) else {
                                        continue;
                                    };
                                    for dst_w in 0..out_w {
                                        let Some(src_w) = src_index(
                                            dst_w,
                                            kw,
                                            p.stride[2],
                                            p.dilation[2],
                                            p.padding[2],
                                            p.i_w,
                                        ) else {
                                            continue;
                                        };
                                        let inp_idx = b_idx * inp_s0
                                            + src_c_idx * inp_s1
                                            + src_d * inp_s2
                                            + src_h * inp_s3
                                            + src_w * inp_s4;
                                        let grad_idx = b_idx * grad_s0
                                            + dst_c_idx * grad_s1
                                            + dst_d * grad_s2
                                            + dst_h * grad_s3
                                            + dst_w * grad_s4;
                                        acc += inp[inp_idx] * grad[grad_idx];
                                    }
                                }
                            }
                        }
                        let dst_idx = ((((dst_c_idx * p.c_in + src_c_idx) * p.k_d + kd) * p.k_h
                            + kh)
                            * p.k_w)
                            + kw;
                        dst[dst_idx] = acc;
                    }
                }
            }
        }
    }
    Ok(dst)
}
