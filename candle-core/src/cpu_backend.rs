use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOp, UnaryOp};
use crate::{DType, Error, Layout, Result, Shape, WithDType};
use half::{bf16, f16};

// TODO: Maybe we should not implement [Clone] here and instead have an explicit allocator +
// intercept the oom errors to avoid panicking and provide a proper error.
#[derive(Debug, Clone)]
pub enum CpuStorage {
    U8(Vec<u8>),
    U32(Vec<u32>),
    BF16(Vec<bf16>),
    F16(Vec<f16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

#[derive(Debug, Clone)]
pub struct CpuDevice;

trait Map1 {
    fn f<T: WithDType>(&self, vs: &[T], layout: &Layout) -> Result<Vec<T>>;

    fn map(&self, vs: &CpuStorage, layout: &Layout) -> Result<CpuStorage> {
        match vs {
            CpuStorage::U8(vs) => Ok(CpuStorage::U8(self.f(vs, layout)?)),
            CpuStorage::U32(vs) => Ok(CpuStorage::U32(self.f(vs, layout)?)),
            CpuStorage::BF16(vs) => Ok(CpuStorage::BF16(self.f(vs, layout)?)),
            CpuStorage::F16(vs) => Ok(CpuStorage::F16(self.f(vs, layout)?)),
            CpuStorage::F32(vs) => Ok(CpuStorage::F32(self.f(vs, layout)?)),
            CpuStorage::F64(vs) => Ok(CpuStorage::F64(self.f(vs, layout)?)),
        }
    }
}

type C = CpuStorage;
trait Map2 {
    const OP: &'static str;
    fn f<T: WithDType>(&self, v1: &[T], l1: &Layout, v2: &[T], l2: &Layout) -> Result<Vec<T>>;

    fn map(
        &self,
        v1: &CpuStorage,
        l1: &Layout,
        v2: &CpuStorage,
        l2: &Layout,
    ) -> Result<CpuStorage> {
        match (v1, v2) {
            (C::U8(v1), C::U8(v2)) => Ok(C::U8(self.f(v1, l1, v2, l2)?)),
            (C::U32(v1), C::U32(v2)) => Ok(C::U32(self.f(v1, l1, v2, l2)?)),
            (C::BF16(v1), C::BF16(v2)) => Ok(C::BF16(self.f(v1, l1, v2, l2)?)),
            (C::F16(v1), C::F16(v2)) => Ok(C::F16(self.f(v1, l1, v2, l2)?)),
            (C::F32(v1), C::F32(v2)) => Ok(C::F32(self.f(v1, l1, v2, l2)?)),
            (C::F64(v1), C::F64(v2)) => Ok(C::F64(self.f(v1, l1, v2, l2)?)),
            _ => Err(Error::DTypeMismatchBinaryOp {
                lhs: v1.dtype(),
                rhs: v2.dtype(),
                op: Self::OP,
            }
            .bt()),
        }
    }
}

struct WCond<'a>(&'a [u32], &'a Layout);

impl<'a> Map2 for WCond<'a> {
    const OP: &'static str = "where";
    #[inline(always)]
    fn f<T: WithDType>(&self, t: &[T], t_l: &Layout, f: &[T], f_l: &Layout) -> Result<Vec<T>> {
        let vs = match (
            self.1.contiguous_offsets(),
            t_l.contiguous_offsets(),
            f_l.contiguous_offsets(),
        ) {
            (Some((o1, o2)), Some((o_t1, o_t2)), Some((o_f1, o_f2))) => {
                let pred = &self.0[o1..o2];
                let t = &t[o_t1..o_t2];
                let f = &f[o_f1..o_f2];
                pred.iter()
                    .zip(t.iter().zip(f.iter()))
                    .map(|(&p, (&t, &f))| if p > 0 { t } else { f })
                    .collect::<Vec<_>>()
            }
            _ => self
                .1
                .strided_index()
                .zip(t_l.strided_index().zip(f_l.strided_index()))
                .map(|(i_p, (i_t, i_f))| if self.0[i_p] > 0 { t[i_t] } else { f[i_f] })
                .collect::<Vec<_>>(),
        };
        Ok(vs)
    }
}

struct Sum<'a> {
    dst_shape: &'a Shape,
    sum_dims_and_stride: Vec<(usize, usize)>,
}

impl<'a> Map1 for Sum<'a> {
    #[inline(always)]
    fn f<T: WithDType>(&self, src: &[T], src_layout: &Layout) -> Result<Vec<T>> {
        let mut dst = vec![T::zero(); self.dst_shape.elem_count()];
        for (unstr_index, src_index) in src_layout.strided_index().enumerate() {
            let mut dst_index = unstr_index;
            // Set the sum_dims indexes to 0.
            for &(dim, stride) in self.sum_dims_and_stride.iter() {
                // The compiler is able to optimize the following in a single divmod op.
                let (pre, post) = (dst_index / stride, dst_index % stride);
                dst_index = (pre / dim) * stride + post;
            }
            dst[dst_index] += src[src_index];
        }
        Ok(dst)
    }
}

fn unary_map<T: Copy, U: Copy, F: FnMut(T) -> U>(vs: &[T], layout: &Layout, mut f: F) -> Vec<U> {
    match layout.strided_blocks() {
        crate::StridedBlocks::SingleBlock { start_offset, len } => vs
            [start_offset..start_offset + len]
            .iter()
            .map(|&v| f(v))
            .collect(),
        crate::StridedBlocks::MultipleBlocks {
            block_start_index,
            block_len,
        } => {
            let mut result = vec![];
            result.reserve(layout.shape().elem_count());
            // Specialize the case where block_len is one to avoid the second loop.
            if block_len == 1 {
                for index in block_start_index {
                    let v = unsafe { vs.get_unchecked(index) };
                    result.push(f(*v))
                }
            } else {
                for index in block_start_index {
                    for offset in 0..block_len {
                        let v = unsafe { vs.get_unchecked(index + offset) };
                        result.push(f(*v))
                    }
                }
            }
            result
        }
    }
}

// This function maps over two strided index sequences.
fn binary_map<T: Copy, F: FnMut(T, T) -> T>(
    lhs_l: &Layout,
    rhs_l: &Layout,
    lhs: &[T],
    rhs: &[T],
    mut f: F,
) -> Vec<T> {
    match (lhs_l.contiguous_offsets(), rhs_l.contiguous_offsets()) {
        (Some((o_l1, o_l2)), Some((o_r1, o_r2))) => lhs[o_l1..o_l2]
            .iter()
            .zip(rhs[o_r1..o_r2].iter())
            .map(|(&l, &r)| f(l, r))
            .collect(),
        (Some((o_l1, o_l2)), None) => {
            // TODO: Maybe we want to avoid going through the layout twice.
            match rhs_l.offsets_b() {
                Some(ob) => {
                    let mut i_in_block = 0;
                    let mut i_right_broadcast = 0;
                    lhs[o_l1..o_l2]
                        .iter()
                        .map(|&l| {
                            let r = unsafe { rhs.get_unchecked(i_in_block + ob.start) };
                            i_right_broadcast += 1;
                            if i_right_broadcast >= ob.right_broadcast {
                                i_in_block += 1;
                                i_right_broadcast = 0;
                            }
                            if i_in_block >= ob.len {
                                i_in_block = 0
                            }
                            f(l, *r)
                        })
                        .collect()
                }
                None => lhs_l
                    .strided_index()
                    .zip(rhs_l.strided_index())
                    .map(|(lhs_i, rhs_i)| f(lhs[lhs_i], rhs[rhs_i]))
                    .collect(),
            }
        }
        (None, Some((o_r1, o_r2))) => {
            // TODO: Maybe we want to avoid going through the layout twice.
            match lhs_l.offsets_b() {
                Some(ob) => {
                    let mut i_in_block = 0;
                    let mut i_right_broadcast = 0;
                    rhs[o_r1..o_r2]
                        .iter()
                        .map(|&r| {
                            let l = unsafe { lhs.get_unchecked(i_in_block + ob.start) };
                            i_right_broadcast += 1;
                            if i_right_broadcast >= ob.right_broadcast {
                                i_in_block += 1;
                                i_right_broadcast = 0;
                            }
                            if i_in_block >= ob.len {
                                i_in_block = 0
                            }
                            f(*l, r)
                        })
                        .collect()
                }
                None => lhs_l
                    .strided_index()
                    .zip(rhs_l.strided_index())
                    .map(|(lhs_i, rhs_i)| f(lhs[lhs_i], rhs[rhs_i]))
                    .collect(),
            }
        }
        _ => lhs_l
            .strided_index()
            .zip(rhs_l.strided_index())
            .map(|(lhs_i, rhs_i)| f(lhs[lhs_i], rhs[rhs_i]))
            .collect(),
    }
}

struct Affine(f64, f64);

impl Map1 for Affine {
    fn f<T: WithDType>(&self, vs: &[T], layout: &Layout) -> Result<Vec<T>> {
        let mul = T::from_f64(self.0);
        let add = T::from_f64(self.1);
        Ok(unary_map(vs, layout, |v| v * mul + add))
    }
}

struct Embedding<'a> {
    vocab_size: usize,
    hidden_size: usize,
    ids: &'a [u32],
    ids_l: &'a Layout,
}

impl<'a> Map1 for Embedding<'a> {
    fn f<T: WithDType>(&self, vs: &[T], layout: &Layout) -> Result<Vec<T>> {
        // TODO: We assume that vs is contiguous here.
        let vs = &vs[layout.start_offset()..];
        let mut values = Vec::with_capacity(self.ids_l.shape().elem_count() * self.hidden_size);
        // TODO: Optimize for the case where ids are contiguous.
        for index in self.ids_l.strided_index() {
            let index = self.ids[index].try_into()?;
            if index >= self.vocab_size {
                Err(Error::InvalidIndex {
                    index,
                    vocab_size: self.vocab_size,
                    op: "take",
                }
                .bt())?
            } else {
                let hidden_size = self.hidden_size;
                values.extend(&vs[hidden_size * index..hidden_size * (index + 1)]);
            }
        }
        Ok(values)
    }
}

fn copy_strided_src_<T: Copy + std::fmt::Display>(
    src: &[T],
    dst: &mut [T],
    dst_offset: usize,
    src_l: &Layout,
) {
    match src_l.strided_blocks() {
        crate::StridedBlocks::SingleBlock { start_offset, len } => {
            let to_copy = (dst.len() - dst_offset).min(len);
            dst[dst_offset..dst_offset + to_copy]
                .copy_from_slice(&src[start_offset..start_offset + to_copy])
        }
        crate::StridedBlocks::MultipleBlocks {
            block_start_index,
            block_len: 1,
        } => {
            for (dst_index, src_index) in block_start_index.enumerate() {
                let dst_index = dst_index + dst_offset;
                if dst_index >= dst.len() {
                    break;
                }
                dst[dst_index] = src[src_index]
            }
        }
        crate::StridedBlocks::MultipleBlocks {
            block_start_index,
            block_len,
        } => {
            let mut dst_index = dst_offset;
            for src_index in block_start_index {
                let next_dst_index = dst_index + block_len;
                if dst_index >= dst.len() {
                    break;
                }
                let to_copy = usize::min(block_len, dst.len() - dst_index);
                dst[dst_index..dst_index + to_copy]
                    .copy_from_slice(&src[src_index..src_index + to_copy]);
                dst_index = next_dst_index
            }
        }
    }
}

struct Conv1D<'a>(&'a crate::conv::ParamsConv1D);

impl<'a> Map2 for Conv1D<'a> {
    const OP: &'static str = "conv1d";
    fn f<T: 'static + num_traits::NumAssign + Copy>(
        &self,
        inp: &[T],
        inp_l: &Layout,
        k: &[T],
        k_l: &Layout,
    ) -> Result<Vec<T>> {
        // TODO: Optimize this (proper algorithm, simd, multithread, remove bound checks, etc).
        let p = self.0;
        let inp = &inp[inp_l.start_offset()..];
        let k = &k[k_l.start_offset()..];
        let inp_stride = inp_l.stride();
        let (inp_stride0, inp_stride) = if inp_stride.len() == 3 {
            (inp_stride[0], &inp_stride[1..])
        } else {
            (0, inp_stride) // This value never gets used anyway
        };
        let k_stride = k_l.stride();
        let k_over_2 = p.k_size / 2;
        let l_out = p.l_out();
        let dst_elems = p.c_out * l_out * p.b_size.unwrap_or(1);
        let mut dst = vec![T::zero(); dst_elems];
        // The output shape is [b_size, c_out, l_out]
        for b_idx in 0..p.b_size.unwrap_or(1) {
            let inp_idx = b_idx * inp_stride0;
            let dst_idx = b_idx * p.c_out * l_out;
            for dst_c_idx in 0..p.c_out {
                let dst_idx = dst_idx + dst_c_idx * l_out;
                for dst_l in 0..l_out {
                    let dst_idx = dst_idx + dst_l;
                    let mut d = T::zero();
                    for offset in 0..p.k_size {
                        let src_l_plus = p.stride * dst_l + offset;
                        // inp[bidx, src_c_idx, dst_l + offset - k//2] * k[dst_c_idx, src_c_idx, offset]
                        if k_over_2 <= src_l_plus && src_l_plus < k_over_2 + p.l_in {
                            let src_l = src_l_plus - k_over_2;
                            for src_c_idx in 0..p.c_in {
                                let inp_idx =
                                    inp_idx + src_c_idx * inp_stride[0] + src_l * inp_stride[1];
                                let k_idx = dst_c_idx * k_stride[0]
                                    + src_c_idx * k_stride[1]
                                    + offset * k_stride[2];
                                d += inp[inp_idx] * k[k_idx]
                            }
                        }
                    }
                    dst[dst_idx] = d
                }
            }
        }
        Ok(dst)
    }
}

struct MatMul((usize, usize, usize, usize));

impl MatMul {
    fn striding_error(&self, lhs_l: &Layout, rhs_l: &Layout, msg: &'static str) -> Error {
        Error::MatMulUnexpectedStriding(Box::new(crate::error::MatMulUnexpectedStriding {
            lhs_l: lhs_l.clone(),
            rhs_l: rhs_l.clone(),
            bmnk: self.0,
            msg,
        }))
        .bt()
    }
}

impl Map2 for MatMul {
    const OP: &'static str = "mat_mul";

    #[cfg(not(feature = "mkl"))]
    fn f<T: 'static + WithDType + num_traits::Num + Copy>(
        &self,
        lhs: &[T],
        lhs_l: &Layout,
        rhs: &[T],
        rhs_l: &Layout,
    ) -> Result<Vec<T>> {
        use gemm::{gemm, Parallelism};
        let (b, m, n, k) = self.0;
        let lhs = &lhs[lhs_l.start_offset()..];
        let rhs = &rhs[rhs_l.start_offset()..];

        let lhs_stride = lhs_l.stride();
        let rhs_stride = rhs_l.stride();
        let rank = lhs_stride.len();
        let lhs_cs = lhs_stride[rank - 1];
        let lhs_rs = lhs_stride[rank - 2];

        let rhs_cs = rhs_stride[rank - 1];
        let rhs_rs = rhs_stride[rank - 2];

        let a_skip: usize = match lhs_stride[..rank - 2] {
            [s1, stride] if s1 == stride * lhs_l.dims()[1] => stride,
            [stride] => stride,
            [] => m * k,
            _ => Err(self.striding_error(lhs_l, rhs_l, "non-contiguous lhs"))?,
        };
        let b_skip: usize = match rhs_stride[..rank - 2] {
            [s1, stride] if s1 == stride * rhs_l.dims()[1] => stride,
            [stride] => stride,
            [] => n * k,
            _ => Err(self.striding_error(lhs_l, rhs_l, "non-contiguous rhs"))?,
        };
        let c_skip: usize = m * n;

        let dst_shape: Shape = (m, n).into();
        let dst_strides = dst_shape.stride_contiguous();
        let dst_rs = dst_strides[0];
        let dst_cs = dst_strides[1];

        let mut dst = vec![T::zero(); b * m * n];
        let num_threads = crate::utils::get_num_threads();
        let parallelism = if num_threads > 1 {
            Parallelism::Rayon(num_threads)
        } else {
            Parallelism::None
        };
        for step in 0..b {
            let lhs_p = &lhs[step * a_skip..];
            let rhs_p = &rhs[step * b_skip..];
            let dst_p = &mut dst[step * c_skip..];
            unsafe {
                gemm(
                    /* m: usize = */ m,
                    /* n: usize = */ n,
                    /* k: usize = */ k,
                    /* dst: *mut T = */ dst_p.as_mut_ptr(),
                    /* dst_cs: isize = */ dst_cs as isize,
                    /* dst_rs: isize = */ dst_rs as isize,
                    /* read_dst: bool = */ false,
                    /* lhs: *const T = */ lhs_p.as_ptr(),
                    /* lhs_cs: isize = */ lhs_cs as isize,
                    /* lhs_rs: isize = */ lhs_rs as isize,
                    /* rhs: *const T = */ rhs_p.as_ptr(),
                    /* rhs_cs: isize = */ rhs_cs as isize,
                    /* rhs_rs: isize = */ rhs_rs as isize,
                    /* alpha: T = */ T::zero(),
                    /* beta: T = */ T::one(),
                    /* conj_dst: bool = */ false,
                    /* conj_lhs: bool = */ false,
                    /* conj_rhs: bool = */ false,
                    parallelism,
                )
            }
        }
        Ok(dst)
    }

    #[cfg(feature = "mkl")]
    fn f<T: 'static + WithDType + num_traits::Num + Copy>(
        &self,
        lhs: &[T],
        lhs_l: &Layout,
        rhs: &[T],
        rhs_l: &Layout,
    ) -> Result<Vec<T>> {
        let (b, m, n, k) = self.0;
        let lhs = &lhs[lhs_l.start_offset()..];
        let rhs = &rhs[rhs_l.start_offset()..];

        let lhs_stride = lhs_l.stride();
        let rhs_stride = rhs_l.stride();
        let rank = lhs_stride.len();

        let a_skip: usize = match lhs_stride[..rank - 2] {
            [s1, stride] if s1 == stride * lhs_l.dims()[1] => stride,
            [stride] => stride,
            [] => m * k,
            _ => Err(self.striding_error(lhs_l, rhs_l, "non-contiguous lhs"))?,
        };
        let b_skip: usize = match rhs_stride[..rank - 2] {
            [s1, stride] if s1 == stride * rhs_l.dims()[1] => stride,
            [stride] => stride,
            [] => n * k,
            _ => Err(self.striding_error(lhs_l, rhs_l, "non-contiguous rhs"))?,
        };
        let c_skip: usize = m * n;

        let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
        let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
        let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
        let lhs_m2 = lhs_stride[lhs_stride.len() - 2];

        let (lda, transa) = if rhs_m1 == 1 && rhs_m2 == n {
            (n as i32, b'N')
        } else if rhs_m1 == k && rhs_m2 == 1 {
            (k as i32, b'T')
        } else {
            Err(self.striding_error(lhs_l, rhs_l, "non-contiguous rhs"))?
        };
        // The b tensor has dims batching, m, k (lhs)
        let (ldb, transb) = if lhs_m1 == 1 && lhs_m2 == k {
            (k as i32, b'N')
        } else if lhs_m1 == m && lhs_m2 == 1 {
            (m as i32, b'T')
        } else {
            Err(self.striding_error(lhs_l, rhs_l, "non-contiguous lhs"))?
        };

        let mut dst = vec![T::zero(); b * m * n];
        match T::DTYPE {
            DType::F16 => {
                for step in 0..b {
                    let lhs_p = &lhs[step * a_skip..];
                    let rhs_p = &rhs[step * b_skip..];
                    let dst_p = &mut dst[step * c_skip..];
                    unsafe {
                        let a = rhs_p.as_ptr() as *const f16;
                        let b = lhs_p.as_ptr() as *const f16;
                        let c = dst_p.as_mut_ptr() as *mut f16;
                        let a = std::slice::from_raw_parts(a, a_skip);
                        let b = std::slice::from_raw_parts(b, b_skip);
                        let c = std::slice::from_raw_parts_mut(c, c_skip);
                        crate::mkl::hgemm(
                            transa,
                            transb,
                            /* m= */ n as i32,
                            /* n= */ m as i32,
                            /* k= */ k as i32,
                            /* alpha= */ f16::ONE,
                            /* a= */ a,
                            /* lda= */ lda,
                            /* b= */ b,
                            /* ldb= */ ldb,
                            /* beta= */ f16::ZERO,
                            /* c= */ c,
                            /* ldc= */ n as i32,
                        )
                    }
                }
            }
            DType::F32 => {
                for step in 0..b {
                    let lhs_p = &lhs[step * a_skip..];
                    let rhs_p = &rhs[step * b_skip..];
                    let dst_p = &mut dst[step * c_skip..];
                    unsafe {
                        let a = rhs_p.as_ptr() as *const f32;
                        let b = lhs_p.as_ptr() as *const f32;
                        let c = dst_p.as_mut_ptr() as *mut f32;
                        let a = std::slice::from_raw_parts(a, a_skip);
                        let b = std::slice::from_raw_parts(b, b_skip);
                        let c = std::slice::from_raw_parts_mut(c, c_skip);
                        crate::mkl::sgemm(
                            transa, transb, /* m= */ n as i32, /* n= */ m as i32,
                            /* k= */ k as i32, /* alpha= */ 1., /* a= */ a,
                            /* lda= */ lda, /* b= */ b, /* ldb= */ ldb,
                            /* beta= */ 0., /* c= */ c, /* ldc= */ n as i32,
                        )
                    }
                }
            }
            DType::F64 => {
                for step in 0..b {
                    let lhs_p = &lhs[step * a_skip..];
                    let rhs_p = &rhs[step * b_skip..];
                    let dst_p = &mut dst[step * c_skip..];
                    unsafe {
                        let a = rhs_p.as_ptr() as *const f64;
                        let b = lhs_p.as_ptr() as *const f64;
                        let c = dst_p.as_mut_ptr() as *mut f64;
                        let a = std::slice::from_raw_parts(a, a_skip);
                        let b = std::slice::from_raw_parts(b, b_skip);
                        let c = std::slice::from_raw_parts_mut(c, c_skip);
                        crate::mkl::dgemm(
                            transa, transb, /* m= */ n as i32, /* n= */ m as i32,
                            /* k= */ k as i32, /* alpha= */ 1., /* a= */ a,
                            /* lda= */ lda, /* b= */ b, /* ldb= */ ldb,
                            /* beta= */ 0., /* c= */ c, /* ldc= */ n as i32,
                        )
                    }
                }
            }
            dtype => Err(Error::UnsupportedDTypeForOp(dtype, "matmul").bt())?,
        }
        Ok(dst)
    }
}

fn divide_by_sum_over_dim<T: WithDType>(s: &mut [T], shape: &Shape, dim: usize) -> Result<()> {
    // [self] stores data in a contiguous way starting at offset 0.
    let dims = shape.dims();
    let elem_per_slice = dims[dim];
    let prod_pre_dim = dims[..dim].iter().product();
    let prod_post_dim = dims[dim + 1..].iter().product();
    for pre_idx in 0..prod_pre_dim {
        for post_idx in 0..prod_post_dim {
            let mut sum = 0f64;
            let mut idx = pre_idx * prod_post_dim * elem_per_slice + post_idx;
            for _ in 0..elem_per_slice {
                sum += s[idx].to_f64();
                idx += prod_post_dim
            }
            let sum = T::from_f64(sum);
            let mut idx = pre_idx * prod_post_dim * elem_per_slice + post_idx;
            for _ in 0..elem_per_slice {
                s[idx] /= sum;
                idx += prod_post_dim
            }
        }
    }
    Ok(())
}

fn elu<T: num_traits::Float>(v: T, alpha: T) -> T {
    if v.is_sign_positive() {
        v
    } else {
        (v.exp() - T::one()) * alpha
    }
}

impl CpuStorage {
    pub fn as_slice<D: WithDType>(&self) -> Result<&[D]> {
        D::cpu_storage_as_slice(self)
    }
}

impl BackendStorage for CpuStorage {
    type Device = CpuDevice;

    fn dtype(&self) -> DType {
        match self {
            Self::U8(_) => DType::U8,
            Self::U32(_) => DType::U32,
            Self::BF16(_) => DType::BF16,
            Self::F16(_) => DType::F16,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
        }
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        // TODO: find a way around the quadratic number of cases below.
        match (self, dtype) {
            (Self::U8(storage), DType::BF16) => {
                let data = unary_map(storage, layout, |v| bf16::from_f32(v as f32));
                Ok(Self::BF16(data))
            }
            (Self::U32(storage), DType::BF16) => {
                let data = unary_map(storage, layout, |v| bf16::from_f32(v as f32));
                Ok(Self::BF16(data))
            }
            (Self::BF16(storage), DType::BF16) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::BF16(data))
            }
            (Self::F16(storage), DType::BF16) => {
                let data = unary_map(storage, layout, |v| bf16::from_f32(v.to_f32()));
                Ok(Self::BF16(data))
            }
            (Self::F32(storage), DType::BF16) => {
                let data = unary_map(storage, layout, bf16::from_f32);
                Ok(Self::BF16(data))
            }
            (Self::F64(storage), DType::BF16) => {
                let data = unary_map(storage, layout, bf16::from_f64);
                Ok(Self::BF16(data))
            }
            (Self::U8(storage), DType::F16) => {
                let data = unary_map(storage, layout, |v| f16::from_f32(v as f32));
                Ok(Self::F16(data))
            }
            (Self::U32(storage), DType::F16) => {
                let data = unary_map(storage, layout, |v| f16::from_f32(v as f32));
                Ok(Self::F16(data))
            }
            (Self::BF16(storage), DType::F16) => {
                let data = unary_map(storage, layout, |v| f16::from_f32(v.to_f32()));
                Ok(Self::F16(data))
            }
            (Self::F16(storage), DType::F16) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::F16(data))
            }
            (Self::F32(storage), DType::F16) => {
                let data = unary_map(storage, layout, f16::from_f32);
                Ok(Self::F16(data))
            }
            (Self::F64(storage), DType::F16) => {
                let data = unary_map(storage, layout, f16::from_f64);
                Ok(Self::F16(data))
            }
            (Self::U8(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v as f32);
                Ok(Self::F32(data))
            }
            (Self::U32(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v as f32);
                Ok(Self::F32(data))
            }
            (Self::BF16(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v.to_f32());
                Ok(Self::F32(data))
            }
            (Self::F16(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v.to_f32());
                Ok(Self::F32(data))
            }
            (Self::F32(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::F32(data))
            }
            (Self::F64(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v as f32);
                Ok(Self::F32(data))
            }
            (Self::U8(storage), DType::U8) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::U8(data))
            }
            (Self::BF16(storage), DType::U8) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as u8);
                Ok(Self::U8(data))
            }
            (Self::F16(storage), DType::U8) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as u8);
                Ok(Self::U8(data))
            }
            (Self::F32(storage), DType::U8) => {
                let data = unary_map(storage, layout, |v| v as u8);
                Ok(Self::U8(data))
            }
            (Self::F64(storage), DType::U8) => {
                let data = unary_map(storage, layout, |v| v as u8);
                Ok(Self::U8(data))
            }
            (Self::U8(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v as u32);
                Ok(Self::U32(data))
            }
            (Self::U32(storage), DType::U8) => {
                let data = unary_map(storage, layout, |v| v as u8);
                Ok(Self::U8(data))
            }
            (Self::U32(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::U32(data))
            }
            (Self::BF16(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as u32);
                Ok(Self::U32(data))
            }
            (Self::F16(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as u32);
                Ok(Self::U32(data))
            }
            (Self::F32(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v as u32);
                Ok(Self::U32(data))
            }
            (Self::F64(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v as u32);
                Ok(Self::U32(data))
            }
            (Self::U8(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v as f64);
                Ok(Self::F64(data))
            }
            (Self::U32(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v as f64);
                Ok(Self::F64(data))
            }
            (Self::BF16(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v.to_f64());
                Ok(Self::F64(data))
            }
            (Self::F16(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v.to_f64());
                Ok(Self::F64(data))
            }
            (Self::F32(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v as f64);
                Ok(Self::F64(data))
            }
            (Self::F64(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::F64(data))
            }
        }
    }

    fn sum(&self, layout: &Layout, sum_dims: &[usize]) -> Result<Self> {
        let src_dims = layout.dims();
        let mut dst_dims = src_dims.to_vec();
        for &sum_dim in sum_dims.iter() {
            dst_dims[sum_dim] = 1;
        }
        let dst_shape = Shape::from(dst_dims);
        let mut sum_dims = sum_dims.to_vec();
        // Sort the sum_dims as they have to be processed from left to right when converting the
        // indexes.
        sum_dims.sort();
        let sum_dims_and_stride: Vec<_> = sum_dims
            .iter()
            .map(|&d| (src_dims[d], src_dims[d + 1..].iter().product::<usize>()))
            .collect();
        Sum {
            dst_shape: &dst_shape,
            sum_dims_and_stride,
        }
        .map(self, layout)
    }

    fn divide_by_sum_over_dim(&mut self, shape: &Shape, dim: usize) -> Result<()> {
        // [self] stores data in a contiguous way starting at offset 0.
        match self {
            Self::BF16(s) => divide_by_sum_over_dim(s, shape, dim),
            Self::F16(s) => divide_by_sum_over_dim(s, shape, dim),
            Self::F32(s) => divide_by_sum_over_dim(s, shape, dim),
            Self::F64(s) => divide_by_sum_over_dim(s, shape, dim),
            Self::U8(_) | Self::U32(_) => Ok(()),
        }
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        Affine(mul, add).map(self, layout)
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        // TODO: Have some generic map for functions that apply on num_traits::Float elements.
        match self {
            Self::BF16(storage) => {
                let data = unary_map(storage, layout, |v| elu(v, bf16::from_f64(alpha)));
                Ok(Self::BF16(data))
            }
            Self::F16(storage) => {
                let data = unary_map(storage, layout, |v| elu(v, f16::from_f64(alpha)));
                Ok(Self::F16(data))
            }
            Self::F32(storage) => {
                let data = unary_map(storage, layout, |v| elu(v, f32::from_f64(alpha)));
                Ok(Self::F32(data))
            }
            Self::F64(storage) => {
                let data = unary_map(storage, layout, |v| elu(v, alpha));
                Ok(Self::F64(data))
            }
            Self::U8(_) => Err(Error::UnsupportedDTypeForOp(DType::U8, "elu").bt()),
            Self::U32(_) => Err(Error::UnsupportedDTypeForOp(DType::U32, "elu").bt()),
        }
    }

    fn unary_impl<B: UnaryOp>(&self, layout: &Layout) -> Result<Self> {
        match self {
            Self::BF16(storage) => {
                let data = unary_map(storage, layout, B::bf16);
                Ok(Self::BF16(data))
            }
            Self::F16(storage) => {
                let data = unary_map(storage, layout, B::f16);
                Ok(Self::F16(data))
            }
            Self::F32(storage) => {
                let data = unary_map(storage, layout, B::f32);
                Ok(Self::F32(data))
            }
            Self::F64(storage) => {
                let data = unary_map(storage, layout, B::f64);
                Ok(Self::F64(data))
            }
            Self::U8(storage) => {
                let data = unary_map(storage, layout, B::u8);
                Ok(Self::U8(data))
            }
            Self::U32(storage) => {
                let data = unary_map(storage, layout, B::u32);
                Ok(Self::U32(data))
            }
        }
    }

    fn binary_impl<B: BinaryOp>(&self, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        match (self, rhs) {
            (Self::BF16(lhs), Self::BF16(rhs)) => {
                let data = binary_map(lhs_l, rhs_l, lhs, rhs, B::bf16);
                Ok(Self::BF16(data))
            }
            (Self::F16(lhs), Self::F16(rhs)) => {
                let data = binary_map(lhs_l, rhs_l, lhs, rhs, B::f16);
                Ok(Self::F16(data))
            }
            (Self::F32(lhs), Self::F32(rhs)) => {
                let data = binary_map(lhs_l, rhs_l, lhs, rhs, B::f32);
                Ok(Self::F32(data))
            }
            (Self::F64(lhs), Self::F64(rhs)) => {
                let data = binary_map(lhs_l, rhs_l, lhs, rhs, B::f64);
                Ok(Self::F64(data))
            }
            (Self::U32(lhs), Self::U32(rhs)) => {
                let data = binary_map(lhs_l, rhs_l, lhs, rhs, B::u32);
                Ok(Self::U32(data))
            }
            (Self::U8(lhs), Self::U8(rhs)) => {
                let data = binary_map(lhs_l, rhs_l, lhs, rhs, B::u8);
                Ok(Self::U8(data))
            }
            _ => {
                // This should be covered by the dtype check above.
                Err(Error::DTypeMismatchBinaryOp {
                    lhs: self.dtype(),
                    rhs: rhs.dtype(),
                    op: B::NAME,
                }
                .bt())
            }
        }
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        match (self, dst) {
            (Self::U8(src), Self::U8(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (Self::U32(src), Self::U32(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (Self::BF16(src), Self::BF16(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (Self::F16(src), Self::F16(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (Self::F32(src), Self::F32(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (Self::F64(src), Self::F64(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (_, dst) => {
                // This should be covered by the dtype check above.
                return Err(Error::DTypeMismatchBinaryOp {
                    lhs: self.dtype(),
                    rhs: dst.dtype(),
                    op: "copy_strided",
                }
                .bt());
            }
        }
        Ok(())
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        // TODO: Support types that could be casted to a boolean.
        let pred = self.as_slice::<u32>()?;
        WCond(pred, layout).map(t, t_l, f, f_l)
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        Conv1D(params).map(self, l, kernel, kernel_l)
    }

    fn embedding(&self, ids_l: &Layout, rhs: &Self, rhs_l: &Layout) -> Result<Self> {
        let ids = self.as_slice::<u32>()?;
        let (vocab_size, hidden_size) = rhs_l.shape().r2()?;
        Embedding {
            vocab_size,
            hidden_size,
            ids,
            ids_l,
        }
        .map(rhs, rhs_l)
    }

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        MatMul(bmnk).map(self, lhs_l, rhs, rhs_l)
    }

    fn device(&self) -> &Self::Device {
        &CpuDevice
    }

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        Ok(self.clone())
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        Ok(self.clone())
    }
}

impl BackendDevice for CpuDevice {
    type Storage = CpuStorage;

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Cpu
    }

    fn same_device(&self, _: &Self) -> bool {
        true
    }

    fn storage_from_cpu_storage(&self, s: &CpuStorage) -> Result<Self::Storage> {
        Ok(s.clone())
    }

    fn new(_: usize) -> Result<Self> {
        Ok(Self)
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, min: f64, max: f64) -> Result<CpuStorage> {
        use rand::prelude::*;

        let elem_count = shape.elem_count();
        let mut rng = rand::thread_rng();
        match dtype {
            DType::U8 | DType::U32 | DType::BF16 | DType::F16 => {
                Err(Error::UnsupportedDTypeForOp(dtype, "rand_normal").bt())
            }
            DType::F32 => {
                let mut data = Vec::new();
                data.reserve(elem_count);
                let uniform = rand::distributions::Uniform::new(min as f32, max as f32);
                for _i in 0..elem_count {
                    data.push(rng.sample::<f32, _>(uniform))
                }
                Ok(CpuStorage::F32(data))
            }
            DType::F64 => {
                let mut data = Vec::new();
                data.reserve(elem_count);
                let uniform = rand::distributions::Uniform::new(min, max);
                for _i in 0..elem_count {
                    data.push(rng.sample::<f64, _>(uniform))
                }
                Ok(CpuStorage::F64(data))
            }
        }
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<CpuStorage> {
        use rand::prelude::*;

        let elem_count = shape.elem_count();
        let mut rng = rand::thread_rng();
        match dtype {
            DType::U8 | DType::U32 | DType::BF16 | DType::F16 => {
                Err(Error::UnsupportedDTypeForOp(dtype, "rand_normal").bt())
            }
            DType::F32 => {
                let mut data = Vec::new();
                data.reserve(elem_count);
                let std = std as f32;
                let mean = mean as f32;
                for _i in 0..elem_count {
                    data.push(rng.sample::<f32, _>(rand::distributions::Standard) * std + mean)
                }
                Ok(CpuStorage::F32(data))
            }
            DType::F64 => {
                let mut data = Vec::new();
                data.reserve(elem_count);
                for _i in 0..elem_count {
                    data.push(rng.sample::<f64, _>(rand::distributions::Standard) * std + mean)
                }
                Ok(CpuStorage::F64(data))
            }
        }
    }

    fn ones_impl(&self, shape: &Shape, dtype: DType) -> Result<CpuStorage> {
        let elem_count = shape.elem_count();
        let storage = match dtype {
            DType::U8 => CpuStorage::U8(vec![1u8; elem_count]),
            DType::U32 => CpuStorage::U32(vec![1u32; elem_count]),
            DType::BF16 => CpuStorage::BF16(vec![bf16::ONE; elem_count]),
            DType::F16 => CpuStorage::F16(vec![f16::ONE; elem_count]),
            DType::F32 => CpuStorage::F32(vec![1f32; elem_count]),
            DType::F64 => CpuStorage::F64(vec![1f64; elem_count]),
        };
        Ok(storage)
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<CpuStorage> {
        let elem_count = shape.elem_count();
        let storage = match dtype {
            DType::U8 => CpuStorage::U8(vec![0u8; elem_count]),
            DType::U32 => CpuStorage::U32(vec![0u32; elem_count]),
            DType::BF16 => CpuStorage::BF16(vec![bf16::ZERO; elem_count]),
            DType::F16 => CpuStorage::F16(vec![f16::ZERO; elem_count]),
            DType::F32 => CpuStorage::F32(vec![0f32; elem_count]),
            DType::F64 => CpuStorage::F64(vec![0f64; elem_count]),
        };
        Ok(storage)
    }
}
