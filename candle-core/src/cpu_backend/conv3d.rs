use std::borrow::Cow;

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    conv::{checked_conv3d_numel, ParamsConv3D, ParamsConvTranspose3D},
    cpu_backend::{Map2, MatMul},
    shape::dims5,
    DType, Layout, Result, WithDType,
};

pub(super) struct Conv3D<'a>(pub(super) &'a ParamsConv3D);
pub(super) struct ConvTranspose3D<'a>(pub(super) &'a ParamsConvTranspose3D);

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

// Bound each im2col tile even when a gradient uses a large volume as its kernel.
const CONV3D_MAX_IM2COL_BYTES: usize = 1024 * 1024;

fn output_len<T>(dims: &[usize]) -> Result<usize> {
    let len = checked_conv3d_numel(dims)?;
    if len > isize::MAX as usize / std::mem::size_of::<T>() {
        crate::bail!("3d convolution allocation size overflow")
    }
    Ok(len)
}

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
        if p.b_size == 0 {
            return Ok(Vec::new());
        }
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
    let dst = vec![T::zero(); output_len::<T>(&p.out_dims())?];
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
    let (out_d, out_h, out_w) = (p.out_d(), p.out_h(), p.out_w());
    let k_size = checked_conv3d_numel(&[p.c_in, p.k_d, p.k_h, p.k_w])?;
    let max_tile_elements = CONV3D_MAX_IM2COL_BYTES / std::mem::size_of::<T>();
    if k_size == 0 || k_size > max_tile_elements {
        return conv3d_direct(p, inp, inp_l, k, k_l);
    }
    let output_tile_size = CONV3D_TILE_SIZE.min(max_tile_elements / k_size);

    let inp = &inp[inp_l.start_offset()..];
    let (inp_s0, inp_s1, inp_s2, inp_s3, inp_s4) = dims5(inp_l.stride())?;
    let k = &k[k_l.start_offset()..];
    let (k_s0, k_s1, k_s2, k_s3, k_s4) = dims5(k_l.stride())?;

    let dst = vec![T::zero(); output_len::<T>(&p.out_dims())?];

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
        let num_tiles = total_out_pixels.div_ceil(output_tile_size);
        (0..num_tiles).into_par_iter().try_for_each(|tile_idx| {
            let tile_start = tile_idx * output_tile_size;
            let tile_end = (tile_start + output_tile_size).min(total_out_pixels);
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
    let (out_d, out_h, out_w) = (p.out_d(), p.out_h(), p.out_w());

    let mut dst = vec![T::zero(); output_len::<T>(&p.out_dims())?];
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

impl Map2 for ConvTranspose3D<'_> {
    const OP: &'static str = "conv-transpose3d";

    fn f<T: WithDType>(&self, inp: &[T], inp_l: &Layout, k: &[T], k_l: &Layout) -> Result<Vec<T>> {
        // Scatter each input value through the kernel into the output volume.
        let p = self.0;
        if p.b_size == 0 {
            return Ok(Vec::new());
        }
        let inp = &inp[inp_l.start_offset()..];
        let k = &k[k_l.start_offset()..];
        let (inp_s0, inp_s1, inp_s2, inp_s3, inp_s4) = dims5(inp_l.stride())?;
        let (k_s0, k_s1, k_s2, k_s3, k_s4) = dims5(k_l.stride())?;
        let (out_d, out_h, out_w) = (p.out_d(), p.out_h(), p.out_w());

        let mut dst = vec![T::zero(); output_len::<T>(&p.out_dims())?];
        for b_idx in 0..p.b_size {
            for src_c_idx in 0..p.c_in {
                for inp_d in 0..p.i_d {
                    for inp_h in 0..p.i_h {
                        for inp_w in 0..p.i_w {
                            let inp_idx = b_idx * inp_s0
                                + src_c_idx * inp_s1
                                + inp_d * inp_s2
                                + inp_h * inp_s3
                                + inp_w * inp_s4;
                            let inp_value = inp[inp_idx];
                            for dst_c_idx in 0..p.c_out {
                                for kd in 0..p.k_d {
                                    let Some(dst_d) = src_index(
                                        inp_d,
                                        kd,
                                        p.stride[0],
                                        p.dilation[0],
                                        p.padding[0],
                                        out_d,
                                    ) else {
                                        continue;
                                    };
                                    for kh in 0..p.k_h {
                                        let Some(dst_h) = src_index(
                                            inp_h,
                                            kh,
                                            p.stride[1],
                                            p.dilation[1],
                                            p.padding[1],
                                            out_h,
                                        ) else {
                                            continue;
                                        };
                                        for kw in 0..p.k_w {
                                            let Some(dst_w) = src_index(
                                                inp_w,
                                                kw,
                                                p.stride[2],
                                                p.dilation[2],
                                                p.padding[2],
                                                out_w,
                                            ) else {
                                                continue;
                                            };
                                            let k_idx = src_c_idx * k_s0
                                                + dst_c_idx * k_s1
                                                + kd * k_s2
                                                + kh * k_s3
                                                + kw * k_s4;
                                            let dst_idx =
                                                ((((b_idx * p.c_out + dst_c_idx) * out_d + dst_d)
                                                    * out_h
                                                    + dst_h)
                                                    * out_w)
                                                    + dst_w;
                                            dst[dst_idx] += inp_value * k[k_idx];
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
