use crate::{conv::ParamsConv3D, cpu_backend::Map2, shape::dims5, Layout, Result, WithDType};

pub(super) struct Conv3D<'a>(pub(super) &'a crate::conv::ParamsConv3D);

impl Map2 for Conv3D<'_> {
    const OP: &'static str = "conv3d";

    fn f<T: WithDType + num_traits::Num + Copy + 'static>(
        &self,
        inp: &[T],
        inp_l: &Layout,
        k: &[T],
        k_l: &Layout,
    ) -> Result<Vec<T>> {
        conv3d_direct(self.0, inp, inp_l, k, k_l)
    }
}

fn conv3d_direct<T: WithDType + num_traits::Num + Copy + 'static>(
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
    let (out_d, out_h, out_w) = (p.out_d(), p.out_h(), p.out_w());

    let mut dst = vec![T::zero(); p.b_size * p.c_out * out_d * out_h * out_w];
    let out_s0 = p.c_out * out_d * out_h * out_w;
    let out_s1 = out_d * out_h * out_w;
    let out_s2 = out_h * out_w;
    let out_s3 = out_w;

    for b_idx in 0..p.b_size {
        for dst_c_idx in 0..p.c_out {
            for dst_d in 0..out_d {
                for dst_h in 0..out_h {
                    for dst_w in 0..out_w {
                        let mut acc = T::zero();
                        for src_c_idx in 0..p.c_in {
                            for offset_d in 0..p.k_d {
                                let src_d = p.stride[0] * dst_d + offset_d * p.dilation[0];
                                if src_d < p.padding[0] || src_d >= p.padding[0] + p.i_d {
                                    continue;
                                }
                                let src_d = src_d - p.padding[0];
                                for offset_h in 0..p.k_h {
                                    let src_h = p.stride[1] * dst_h + offset_h * p.dilation[1];
                                    if src_h < p.padding[1] || src_h >= p.padding[1] + p.i_h {
                                        continue;
                                    }
                                    let src_h = src_h - p.padding[1];
                                    for offset_w in 0..p.k_w {
                                        let src_w = p.stride[2] * dst_w + offset_w * p.dilation[2];
                                        if src_w < p.padding[2] || src_w >= p.padding[2] + p.i_w {
                                            continue;
                                        }
                                        let src_w = src_w - p.padding[2];
                                        let inp_idx = b_idx * inp_s0
                                            + src_c_idx * inp_s1
                                            + src_d * inp_s2
                                            + src_h * inp_s3
                                            + src_w * inp_s4;
                                        let k_idx = dst_c_idx * k_s0
                                            + src_c_idx * k_s1
                                            + offset_d * k_s2
                                            + offset_h * k_s3
                                            + offset_w * k_s4;
                                        acc = acc + inp[inp_idx] * k[k_idx];
                                    }
                                }
                            }
                        }
                        let dst_idx = b_idx * out_s0
                            + dst_c_idx * out_s1
                            + dst_d * out_s2
                            + dst_h * out_s3
                            + dst_w;
                        dst[dst_idx] = acc;
                    }
                }
            }
        }
    }

    Ok(dst)
}
