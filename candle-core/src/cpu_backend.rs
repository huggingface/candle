use crate::op::{BinaryOp, UnaryOp};
use crate::{DType, Error, Layout, Result, Shape, StridedIndex};
use gemm::{gemm, Parallelism};
use half::{bf16, f16};

// TODO: Think about whether we would be better off with a dtype and
// a buffer as an owned slice of bytes.
// TODO: Maybe we should not implement [Clone] here and instead have an explicit allocator +
// intercept the oom errors to avoid panicking and provide a proper error.
#[derive(Debug, Clone)]
pub enum CpuStorage {
    U32(Vec<u32>),
    BF16(Vec<bf16>),
    F16(Vec<f16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

fn wcond<T: Copy>(
    pred: &[u32],
    layout: &Layout,
    t: &[T],
    layout_t: &Layout,
    f: &[T],
    layout_f: &Layout,
) -> Vec<T> {
    match (
        layout.contiguous_offsets(),
        layout_t.contiguous_offsets(),
        layout_f.contiguous_offsets(),
    ) {
        (Some((o1, o2)), Some((o_t1, o_t2)), Some((o_f1, o_f2))) => {
            let pred = &pred[o1..o2];
            let t = &t[o_t1..o_t2];
            let f = &f[o_f1..o_f2];
            pred.iter()
                .zip(t.iter().zip(f.iter()))
                .map(|(&p, (&t, &f))| if p > 0 { t } else { f })
                .collect::<Vec<_>>()
        }
        _ => layout
            .strided_index()
            .zip(layout_t.strided_index().zip(layout_f.strided_index()))
            .map(|(i_p, (i_t, i_f))| if pred[i_p] > 0 { t[i_t] } else { f[i_f] })
            .collect::<Vec<_>>(),
    }
}

macro_rules! map1 {
    ($v: expr, $fn: ident, $( $args:expr ),*) => {{
        let v = match $v {
            CpuStorage::BF16(__s) => CpuStorage::BF16($fn::<bf16>(__s, $($args),*)?),
            CpuStorage::F16(__s) => CpuStorage::F16($fn::<f16>(__s, $($args),*)?),
            CpuStorage::F32(__s) => CpuStorage::F32($fn::<f32>(__s, $($args),*)?),
            CpuStorage::F64(__s) => CpuStorage::F64($fn::<f64>(__s, $($args),*)?),
            CpuStorage::U32(__s) => CpuStorage::U32($fn::<u32>(__s, $($args),*)?),
        };
        Ok(v)
    }};
}

fn sum_impl1<T: Copy + num_traits::NumAssign>(
    src: &[T],
    dst_shape: &Shape,
    src_layout: &Layout,
    to_dst_index: impl Fn(usize) -> usize,
) -> Result<Vec<T>> {
    let mut dst = vec![T::zero(); dst_shape.elem_count()];
    for (unstr_index, src_index) in src_layout.strided_index().enumerate() {
        dst[to_dst_index(unstr_index)] += src[src_index];
    }
    Ok(dst)
}

fn unary_map<T: Copy, U: Copy, F: FnMut(T) -> U>(vs: &[T], layout: &Layout, mut f: F) -> Vec<U> {
    match layout.contiguous_offsets() {
        Some((o1, o2)) => vs[o1..o2].iter().map(|&v| f(v)).collect(),
        None => layout.strided_index().map(|i| f(vs[i])).collect(),
    }
}

// This function maps over two strided index sequences.
fn binary_map<T: Copy, F: FnMut(T, T) -> T>(
    shape: &Shape,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    lhs: &[T],
    rhs: &[T],
    mut f: F,
) -> Vec<T> {
    let dims = shape.dims();
    if lhs_layout.is_contiguous() && rhs_layout.is_contiguous() {
        (0..shape.elem_count()).map(|i| f(lhs[i], rhs[i])).collect()
    } else {
        let lhs_index = lhs_layout.strided_index();
        let rhs_index = rhs_layout.strided_index();
        lhs_index
            .zip(rhs_index)
            .map(|(lhs_i, rhs_i)| f(lhs[lhs_i], rhs[rhs_i]))
            .collect()
    }
}

fn take_impl1<T: Copy>(
    vs: &[T],
    ids: &[u32],
    layout: &Layout,
    vocab_size: usize,
    hidden_size: usize,
) -> Result<Vec<T>> {
    // TODO: Optimize for the case where ids are contiguous.
    let mut values = Vec::with_capacity(layout.shape().elem_count() * hidden_size);
    for index in layout.strided_index() {
        let index = ids[index].try_into()?;
        if index >= vocab_size {
            return Err(Error::InvalidIndex {
                index,
                vocab_size,
                op: "take",
            });
        } else {
            values.extend(&vs[hidden_size * index..hidden_size * (index + 1)]);
        }
    }
    Ok(values)
}

fn copy_strided_src_<T: Copy + std::fmt::Display>(
    src: &[T],
    dst: &mut [T],
    dst_offset: usize,
    src_l: &Layout,
) {
    match src_l.contiguous_offsets() {
        Some((o_dst1, o_dst2)) => {
            let elem_to_copy = (dst.len() - dst_offset).min(o_dst2 - o_dst1);
            dst[dst_offset..dst_offset + elem_to_copy].copy_from_slice(&src[o_dst1..o_dst2])
        }
        None => {
            for (dst_index, src_index) in src_l.strided_index().enumerate() {
                let dst_index = dst_index + dst_offset;
                if dst_index >= dst.len() {
                    break;
                }
                dst[dst_index] = src[src_index]
            }
        }
    }
}

fn matmul_impl<T: 'static + num_traits::Num + Copy>(
    lhs: &[T],
    rhs: &[T],
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_stride: &[usize],
    rhs_stride: &[usize],
) -> Result<Vec<T>> {
    let a_skip: usize = m * k;
    let b_skip: usize = n * k;
    let c_skip: usize = m * n;

    let rank = lhs_stride.len();
    let lhs_cs = lhs_stride[rank - 1];
    let lhs_rs = lhs_stride[rank - 2];

    let rhs_cs = rhs_stride[rank - 1];
    let rhs_rs = rhs_stride[rank - 2];

    if lhs_stride.len() > 2 {
        let lhs_batch_stride = &lhs_stride[..rank - 2];
        let rhs_batch_stride = &rhs_stride[..rank - 2];

        if lhs_batch_stride != [a_skip] || rhs_batch_stride != [b_skip] {
            // Temporary error before we support abitrary striding.
            return Err(Error::UnexpectedStriding);
        }
    }

    let dst_shape: Shape = (m, n).into();
    let dst_strides = dst_shape.stride_contiguous();
    let dst_rs = dst_strides[0];
    let dst_cs = dst_strides[1];

    let mut dst = vec![T::zero(); b * m * n];
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
                Parallelism::Rayon(crate::utils::get_num_threads()),
            )
        }
    }
    Ok(dst)
}

impl CpuStorage {
    pub fn dtype(&self) -> DType {
        match self {
            Self::U32(_) => DType::U32,
            Self::BF16(_) => DType::BF16,
            Self::F16(_) => DType::F16,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
        }
    }

    pub fn as_slice<D: crate::WithDType>(&self) -> Result<&[D]> {
        D::cpu_storage_as_slice(self)
    }

    pub fn as_mut_slice<D: crate::WithDType>(&mut self) -> Result<&mut [D]> {
        D::cpu_storage_as_mut_slice(self)
    }

    pub(crate) fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        // TODO: find a way around the quadratic number of cases below.
        match (self, dtype) {
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

    pub(crate) fn sum(&self, layout: &Layout, sum_dims: &[usize]) -> Result<Self> {
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
        let to_dst_index = |unstr_index: usize| {
            // TODO: Optimize, the following does lots of slow division.
            let mut dst_index = unstr_index;
            // Set the sum_dims indexes to 0.
            for &(dim, stride) in sum_dims_and_stride.iter() {
                // The compiler is able to optimize the following in a single divmod op.
                let (pre, post) = (dst_index / stride, dst_index % stride);
                dst_index = (pre / dim) * stride + post;
            }
            dst_index
        };
        // TODO: Maybe provide an implementation with higher precision accumulators?
        map1!(self, sum_impl1, &dst_shape, layout, to_dst_index)
    }

    pub(crate) fn divide_by_sum_over_dim(&mut self, shape: &Shape, dim: usize) -> Result<()> {
        // [self] stores data in a contiguous way.
        let dims = shape.dims();
        let elem_per_slice = dims[dim];
        let prod_pre_dim = dims[..dim].iter().product();
        let prod_post_dim = dims[dim + 1..].iter().product();
        match self {
            Self::BF16(storage) => {
                for pre_idx in 0..prod_pre_dim {
                    for post_idx in 0..prod_post_dim {
                        let mut sum = 0f64;
                        let mut idx = pre_idx * prod_post_dim * elem_per_slice + post_idx;
                        for _ in 0..elem_per_slice {
                            sum += storage[idx].to_f64();
                            idx += prod_post_dim
                        }
                        let sum = bf16::from_f64(sum);
                        let mut idx = pre_idx * prod_post_dim * elem_per_slice + post_idx;
                        for _ in 0..elem_per_slice {
                            storage[idx] /= sum;
                            idx += prod_post_dim
                        }
                    }
                }
            }
            Self::F16(storage) => {
                for pre_idx in 0..prod_pre_dim {
                    for post_idx in 0..prod_post_dim {
                        let mut sum = 0f64;
                        let mut idx = pre_idx * prod_post_dim * elem_per_slice + post_idx;
                        for _ in 0..elem_per_slice {
                            sum += storage[idx].to_f64();
                            idx += prod_post_dim
                        }
                        let sum = f16::from_f64(sum);
                        let mut idx = pre_idx * prod_post_dim * elem_per_slice + post_idx;
                        for _ in 0..elem_per_slice {
                            storage[idx] /= sum;
                            idx += prod_post_dim
                        }
                    }
                }
            }
            Self::F32(storage) => {
                for pre_idx in 0..prod_pre_dim {
                    for post_idx in 0..prod_post_dim {
                        let mut sum = 0f64;
                        let mut idx = pre_idx * prod_post_dim * elem_per_slice + post_idx;
                        for _ in 0..elem_per_slice {
                            sum += storage[idx] as f64;
                            idx += prod_post_dim
                        }
                        let sum = sum as f32;
                        let mut idx = pre_idx * prod_post_dim * elem_per_slice + post_idx;
                        for _ in 0..elem_per_slice {
                            storage[idx] /= sum;
                            idx += prod_post_dim
                        }
                    }
                }
            }
            Self::F64(storage) => {
                for pre_idx in 0..prod_pre_dim {
                    for post_idx in 0..prod_post_dim {
                        let mut sum = 0f64;
                        let mut idx = pre_idx * prod_post_dim * elem_per_slice + post_idx;
                        for _ in 0..elem_per_slice {
                            sum += storage[idx];
                            idx += prod_post_dim
                        }
                        let mut idx = pre_idx * prod_post_dim * elem_per_slice + post_idx;
                        for _ in 0..elem_per_slice {
                            storage[idx] /= sum;
                            idx += prod_post_dim
                        }
                    }
                }
            }
            Self::U32(_) => {}
        }
        Ok(())
    }

    pub(crate) fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        match self {
            Self::U32(storage) => {
                let mul = mul as u32;
                let add = add as u32;
                let data = unary_map(storage, layout, |v| v * mul + add);
                Ok(Self::U32(data))
            }
            Self::BF16(storage) => {
                let mul = bf16::from_f64(mul);
                let add = bf16::from_f64(add);
                let data = unary_map(storage, layout, |v| v * mul + add);
                Ok(Self::BF16(data))
            }
            Self::F16(storage) => {
                let mul = f16::from_f64(mul);
                let add = f16::from_f64(add);
                let data = unary_map(storage, layout, |v| v * mul + add);
                Ok(Self::F16(data))
            }
            Self::F32(storage) => {
                let mul = mul as f32;
                let add = add as f32;
                let data = unary_map(storage, layout, |v| v * mul + add);
                Ok(Self::F32(data))
            }
            Self::F64(storage) => {
                let data = unary_map(storage, layout, |v| v * mul + add);
                Ok(Self::F64(data))
            }
        }
    }

    pub(crate) fn unary_impl<B: UnaryOp>(&self, layout: &Layout) -> Result<Self> {
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
            Self::U32(storage) => {
                let data = unary_map(storage, layout, B::u32);
                Ok(Self::U32(data))
            }
        }
    }

    pub(crate) fn binary_impl<B: BinaryOp>(
        &self,
        rhs: &Self,
        shape: &Shape,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        match (self, rhs) {
            (Self::BF16(lhs), Self::BF16(rhs)) => {
                let data = binary_map(shape, lhs_layout, rhs_layout, lhs, rhs, B::bf16);
                Ok(Self::BF16(data))
            }
            (Self::F16(lhs), Self::F16(rhs)) => {
                let data = binary_map(shape, lhs_layout, rhs_layout, lhs, rhs, B::f16);
                Ok(Self::F16(data))
            }
            (Self::F32(lhs), Self::F32(rhs)) => {
                let data = binary_map(shape, lhs_layout, rhs_layout, lhs, rhs, B::f32);
                Ok(Self::F32(data))
            }
            (Self::F64(lhs), Self::F64(rhs)) => {
                let data = binary_map(shape, lhs_layout, rhs_layout, lhs, rhs, B::f64);
                Ok(Self::F64(data))
            }
            (Self::U32(lhs), Self::U32(rhs)) => {
                let data = binary_map(shape, lhs_layout, rhs_layout, lhs, rhs, B::u32);
                Ok(Self::U32(data))
            }
            _ => {
                // This should be covered by the dtype check above.
                Err(Error::DTypeMismatchBinaryOp {
                    lhs: self.dtype(),
                    rhs: rhs.dtype(),
                    op: B::NAME,
                })
            }
        }
    }

    pub(crate) fn copy_strided_src(
        &self,
        dst: &mut Self,
        dst_offset: usize,
        src_l: &Layout,
    ) -> Result<()> {
        match (self, dst) {
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
                });
            }
        }
        Ok(())
    }

    pub(crate) fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        layout_t: &Layout,
        f: &Self,
        layout_f: &Layout,
    ) -> Result<Self> {
        // TODO: Support types that could be casted to a boolean.
        let pred = self.as_slice::<u32>()?;
        match (t, f) {
            (Self::BF16(t), Self::BF16(f)) => {
                let data = wcond(pred, layout, t, layout_t, f, layout_f);
                Ok(Self::BF16(data))
            }
            (Self::F16(t), Self::F16(f)) => {
                let data = wcond(pred, layout, t, layout_t, f, layout_f);
                Ok(Self::F16(data))
            }
            (Self::F32(t), Self::F32(f)) => {
                let data = wcond(pred, layout, t, layout_t, f, layout_f);
                Ok(Self::F32(data))
            }
            (Self::F64(t), Self::F64(f)) => {
                let data = wcond(pred, layout, t, layout_t, f, layout_f);
                Ok(Self::F64(data))
            }
            (Self::U32(t), Self::U32(f)) => {
                let data = wcond(pred, layout, t, layout_t, f, layout_f);
                Ok(Self::U32(data))
            }
            _ => Err(Error::DTypeMismatchBinaryOp {
                lhs: t.dtype(),
                rhs: f.dtype(),
                op: "where_cond",
            }),
        }
    }

    pub(crate) fn embedding(
        &self,
        layout: &Layout,
        vs: &Self,
        hidden_size: usize,
        vocab_size: usize,
    ) -> Result<Self> {
        let ids = self.as_slice::<u32>()?;
        map1!(vs, take_impl1, ids, layout, vocab_size, hidden_size)
    }

    pub(crate) fn matmul_impl(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_stride: &[usize],
        rhs_stride: &[usize],
    ) -> Result<Self> {
        match (self, rhs) {
            (CpuStorage::F16(lhs), CpuStorage::F16(rhs)) => {
                let dst = matmul_impl(lhs, rhs, bmnk, lhs_stride, rhs_stride)?;
                Ok(Self::F16(dst))
            }
            (CpuStorage::F32(lhs), CpuStorage::F32(rhs)) => {
                let dst = matmul_impl(lhs, rhs, bmnk, lhs_stride, rhs_stride)?;
                Ok(Self::F32(dst))
            }
            (CpuStorage::F64(lhs), CpuStorage::F64(rhs)) => {
                let dst = matmul_impl(lhs, rhs, bmnk, lhs_stride, rhs_stride)?;
                Ok(Self::F64(dst))
            }
            _ => Err(Error::DTypeMismatchBinaryOp {
                lhs: self.dtype(),
                rhs: rhs.dtype(),
                op: "matmul",
            }),
        }
    }

    pub(crate) fn ones_impl(shape: &Shape, dtype: DType) -> Self {
        let elem_count = shape.elem_count();
        match dtype {
            DType::U32 => {
                let data = vec![1u32; elem_count];
                Self::U32(data)
            }
            DType::BF16 => {
                let data = vec![bf16::ONE; elem_count];
                Self::BF16(data)
            }
            DType::F16 => {
                let data = vec![f16::ONE; elem_count];
                Self::F16(data)
            }
            DType::F32 => {
                let data = vec![1f32; elem_count];
                Self::F32(data)
            }
            DType::F64 => {
                let data = vec![1f64; elem_count];
                Self::F64(data)
            }
        }
    }

    pub(crate) fn zeros_impl(shape: &Shape, dtype: DType) -> Self {
        let elem_count = shape.elem_count();
        match dtype {
            DType::U32 => {
                let data = vec![0u32; elem_count];
                Self::U32(data)
            }
            DType::BF16 => {
                let data = vec![bf16::ZERO; elem_count];
                Self::BF16(data)
            }
            DType::F16 => {
                let data = vec![f16::ZERO; elem_count];
                Self::F16(data)
            }
            DType::F32 => {
                let data = vec![0f32; elem_count];
                Self::F32(data)
            }
            DType::F64 => {
                let data = vec![0f64; elem_count];
                Self::F64(data)
            }
        }
    }
}
