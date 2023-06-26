use crate::op::{BinaryOp, UnaryOp};
use crate::{DType, Error, Result, Shape, StridedIndex};
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
    shape: &Shape,
    stride: &[usize],
    t: &[T],
    stride_t: &[usize],
    f: &[T],
    stride_f: &[usize],
) -> Vec<T> {
    if shape.is_contiguous(stride) && shape.is_contiguous(stride_t) && shape.is_contiguous(stride_f)
    {
        let elem_count = shape.elem_count();
        let pred = &pred[..elem_count];
        let t = &t[..elem_count];
        let f = &f[..elem_count];
        pred.iter()
            .zip(t.iter().zip(f.iter()))
            .map(|(&p, (&t, &f))| if p > 0 { t } else { f })
            .collect::<Vec<_>>()
    } else {
        let dims = shape.dims();
        let it_p = StridedIndex::new(dims, stride);
        let it_t = StridedIndex::new(dims, stride_t);
        let it_f = StridedIndex::new(dims, stride_f);
        it_p.zip(it_t.zip(it_f))
            .map(|(i_p, (i_t, i_f))| if pred[i_p] > 0 { t[i_t] } else { f[i_f] })
            .collect::<Vec<_>>()
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
    src_dims: &[usize],
    stride: &[usize],
    to_dst_index: impl Fn(usize) -> usize,
) -> Result<Vec<T>> {
    let mut dst = vec![T::zero(); dst_shape.elem_count()];
    for (unstr_index, src_index) in StridedIndex::new(src_dims, stride).enumerate() {
        dst[to_dst_index(unstr_index)] += src[src_index];
    }
    Ok(dst)
}

fn unary_map<T: Copy, U: Copy, F: FnMut(T) -> U>(
    vs: &[T],
    shape: &Shape,
    stride: &[usize],
    mut f: F,
) -> Vec<U> {
    if shape.is_contiguous(stride) {
        vs[..shape.elem_count()].iter().map(|&v| f(v)).collect()
    } else {
        StridedIndex::new(shape.dims(), stride)
            .map(|i| f(vs[i]))
            .collect()
    }
}

// This function maps over two strided index sequences.
fn binary_map<T: Copy, F: FnMut(T, T) -> T>(
    shape: &Shape,
    lhs_stride: &[usize],
    rhs_stride: &[usize],
    lhs: &[T],
    rhs: &[T],
    mut f: F,
) -> Vec<T> {
    let dims = shape.dims();
    if shape.is_contiguous(lhs_stride) && shape.is_contiguous(rhs_stride) {
        (0..shape.elem_count()).map(|i| f(lhs[i], rhs[i])).collect()
    } else {
        let lhs_index = StridedIndex::new(dims, lhs_stride);
        let rhs_index = StridedIndex::new(dims, rhs_stride);
        lhs_index
            .zip(rhs_index)
            .map(|(lhs_i, rhs_i)| f(lhs[lhs_i], rhs[rhs_i]))
            .collect()
    }
}

fn take_impl1<T: Copy>(
    vs: &[T],
    ids: &[u32],
    shape: &Shape,
    stride: &[usize],
    vocab_size: usize,
    hidden_size: usize,
) -> Result<Vec<T>> {
    let mut values = Vec::with_capacity(shape.elem_count() * hidden_size);
    for index in StridedIndex::new(shape.dims(), stride) {
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
    src_shape: &Shape,
    src_stride: &[usize],
    src_offset: usize,
) {
    let src = &src[src_offset..];
    if src_shape.is_contiguous(src_stride) {
        let elem_to_copy = (dst.len() - dst_offset).min(src.len());
        dst[dst_offset..dst_offset + elem_to_copy].copy_from_slice(&src[..elem_to_copy])
    } else {
        let src_indexes = StridedIndex::new(src_shape.dims(), src_stride);
        for (dst_index, src_index) in src_indexes.enumerate() {
            let dst_index = dst_index + dst_offset;
            if dst_index >= dst.len() {
                break;
            }
            dst[dst_index] = src[src_index]
        }
    }
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

    pub(crate) fn to_dtype(&self, shape: &Shape, stride: &[usize], dtype: DType) -> Result<Self> {
        // TODO: find a way around the quadratic number of cases below.
        match (self, dtype) {
            (Self::U32(storage), DType::BF16) => {
                let data = unary_map(storage, shape, stride, |v| bf16::from_f32(v as f32));
                Ok(Self::BF16(data))
            }
            (Self::BF16(storage), DType::BF16) => {
                let data = unary_map(storage, shape, stride, |v| v);
                Ok(Self::BF16(data))
            }
            (Self::F16(storage), DType::BF16) => {
                let data = unary_map(storage, shape, stride, |v| bf16::from_f32(v.to_f32()));
                Ok(Self::BF16(data))
            }
            (Self::F32(storage), DType::BF16) => {
                let data = unary_map(storage, shape, stride, bf16::from_f32);
                Ok(Self::BF16(data))
            }
            (Self::F64(storage), DType::BF16) => {
                let data = unary_map(storage, shape, stride, bf16::from_f64);
                Ok(Self::BF16(data))
            }
            (Self::U32(storage), DType::F16) => {
                let data = unary_map(storage, shape, stride, |v| f16::from_f32(v as f32));
                Ok(Self::F16(data))
            }
            (Self::BF16(storage), DType::F16) => {
                let data = unary_map(storage, shape, stride, |v| f16::from_f32(v.to_f32()));
                Ok(Self::F16(data))
            }
            (Self::F16(storage), DType::F16) => {
                let data = unary_map(storage, shape, stride, |v| v);
                Ok(Self::F16(data))
            }
            (Self::F32(storage), DType::F16) => {
                let data = unary_map(storage, shape, stride, f16::from_f32);
                Ok(Self::F16(data))
            }
            (Self::F64(storage), DType::F16) => {
                let data = unary_map(storage, shape, stride, f16::from_f64);
                Ok(Self::F16(data))
            }
            (Self::U32(storage), DType::F32) => {
                let data = unary_map(storage, shape, stride, |v| v as f32);
                Ok(Self::F32(data))
            }
            (Self::BF16(storage), DType::F32) => {
                let data = unary_map(storage, shape, stride, |v| v.to_f32());
                Ok(Self::F32(data))
            }
            (Self::F16(storage), DType::F32) => {
                let data = unary_map(storage, shape, stride, |v| v.to_f32());
                Ok(Self::F32(data))
            }
            (Self::F32(storage), DType::F32) => {
                let data = unary_map(storage, shape, stride, |v| v);
                Ok(Self::F32(data))
            }
            (Self::F64(storage), DType::F32) => {
                let data = unary_map(storage, shape, stride, |v| v as f32);
                Ok(Self::F32(data))
            }
            (Self::U32(storage), DType::U32) => {
                let data = unary_map(storage, shape, stride, |v| v);
                Ok(Self::U32(data))
            }
            (Self::BF16(storage), DType::U32) => {
                let data = unary_map(storage, shape, stride, |v| v.to_f32() as u32);
                Ok(Self::U32(data))
            }
            (Self::F16(storage), DType::U32) => {
                let data = unary_map(storage, shape, stride, |v| v.to_f32() as u32);
                Ok(Self::U32(data))
            }
            (Self::F32(storage), DType::U32) => {
                let data = unary_map(storage, shape, stride, |v| v as u32);
                Ok(Self::U32(data))
            }
            (Self::F64(storage), DType::U32) => {
                let data = unary_map(storage, shape, stride, |v| v as u32);
                Ok(Self::U32(data))
            }
            (Self::U32(storage), DType::F64) => {
                let data = unary_map(storage, shape, stride, |v| v as f64);
                Ok(Self::F64(data))
            }
            (Self::BF16(storage), DType::F64) => {
                let data = unary_map(storage, shape, stride, |v| v.to_f64());
                Ok(Self::F64(data))
            }
            (Self::F16(storage), DType::F64) => {
                let data = unary_map(storage, shape, stride, |v| v.to_f64());
                Ok(Self::F64(data))
            }
            (Self::F32(storage), DType::F64) => {
                let data = unary_map(storage, shape, stride, |v| v as f64);
                Ok(Self::F64(data))
            }
            (Self::F64(storage), DType::F64) => {
                let data = unary_map(storage, shape, stride, |v| v);
                Ok(Self::F64(data))
            }
        }
    }

    pub(crate) fn sum(&self, shape: &Shape, stride: &[usize], sum_dims: &[usize]) -> Result<Self> {
        let src_dims = shape.dims();
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
        map1!(self, sum_impl1, &dst_shape, src_dims, stride, to_dst_index)
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

    pub(crate) fn affine_impl(
        &self,
        shape: &Shape,
        stride: &[usize],
        mul: f64,
        add: f64,
    ) -> Result<Self> {
        match self {
            Self::U32(storage) => {
                let mul = mul as u32;
                let add = add as u32;
                let data = unary_map(storage, shape, stride, |v| v * mul + add);
                Ok(Self::U32(data))
            }
            Self::BF16(storage) => {
                let mul = bf16::from_f64(mul);
                let add = bf16::from_f64(add);
                let data = unary_map(storage, shape, stride, |v| v * mul + add);
                Ok(Self::BF16(data))
            }
            Self::F16(storage) => {
                let mul = f16::from_f64(mul);
                let add = f16::from_f64(add);
                let data = unary_map(storage, shape, stride, |v| v * mul + add);
                Ok(Self::F16(data))
            }
            Self::F32(storage) => {
                let mul = mul as f32;
                let add = add as f32;
                let data = unary_map(storage, shape, stride, |v| v * mul + add);
                Ok(Self::F32(data))
            }
            Self::F64(storage) => {
                let data = unary_map(storage, shape, stride, |v| v * mul + add);
                Ok(Self::F64(data))
            }
        }
    }

    pub(crate) fn unary_impl<B: UnaryOp>(&self, shape: &Shape, stride: &[usize]) -> Result<Self> {
        match self {
            Self::BF16(storage) => {
                let data = unary_map(storage, shape, stride, B::bf16);
                Ok(Self::BF16(data))
            }
            Self::F16(storage) => {
                let data = unary_map(storage, shape, stride, B::f16);
                Ok(Self::F16(data))
            }
            Self::F32(storage) => {
                let data = unary_map(storage, shape, stride, B::f32);
                Ok(Self::F32(data))
            }
            Self::F64(storage) => {
                let data = unary_map(storage, shape, stride, B::f64);
                Ok(Self::F64(data))
            }
            Self::U32(storage) => {
                let data = unary_map(storage, shape, stride, B::u32);
                Ok(Self::U32(data))
            }
        }
    }

    pub(crate) fn binary_impl<B: BinaryOp>(
        &self,
        rhs: &Self,
        shape: &Shape,
        lhs_stride: &[usize],
        rhs_stride: &[usize],
    ) -> Result<Self> {
        match (self, rhs) {
            (Self::BF16(lhs), Self::BF16(rhs)) => {
                let data = binary_map(shape, lhs_stride, rhs_stride, lhs, rhs, B::bf16);
                Ok(Self::BF16(data))
            }
            (Self::F16(lhs), Self::F16(rhs)) => {
                let data = binary_map(shape, lhs_stride, rhs_stride, lhs, rhs, B::f16);
                Ok(Self::F16(data))
            }
            (Self::F32(lhs), Self::F32(rhs)) => {
                let data = binary_map(shape, lhs_stride, rhs_stride, lhs, rhs, B::f32);
                Ok(Self::F32(data))
            }
            (Self::F64(lhs), Self::F64(rhs)) => {
                let data = binary_map(shape, lhs_stride, rhs_stride, lhs, rhs, B::f64);
                Ok(Self::F64(data))
            }
            (Self::U32(lhs), Self::U32(rhs)) => {
                let data = binary_map(shape, lhs_stride, rhs_stride, lhs, rhs, B::u32);
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
        src_shape: &Shape,
        src_stride: &[usize],
        src_offset: usize,
    ) -> Result<()> {
        if src_shape.rank() != src_stride.len() {
            panic!("incoherent shape and strides {src_shape:?} {src_stride:?}")
        }
        match (self, dst) {
            (Self::U32(src), Self::U32(dst)) => {
                copy_strided_src_(src, dst, dst_offset, src_shape, src_stride, src_offset)
            }
            (Self::BF16(src), Self::BF16(dst)) => {
                copy_strided_src_(src, dst, dst_offset, src_shape, src_stride, src_offset)
            }
            (Self::F16(src), Self::F16(dst)) => {
                copy_strided_src_(src, dst, dst_offset, src_shape, src_stride, src_offset)
            }
            (Self::F32(src), Self::F32(dst)) => {
                copy_strided_src_(src, dst, dst_offset, src_shape, src_stride, src_offset)
            }
            (Self::F64(src), Self::F64(dst)) => {
                copy_strided_src_(src, dst, dst_offset, src_shape, src_stride, src_offset)
            }
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
        shape: &Shape,
        stride: &[usize],
        t: &Self,
        stride_t: &[usize],
        f: &Self,
        stride_f: &[usize],
    ) -> Result<Self> {
        // TODO: Support types that could be casted to a boolean.
        let pred = self.as_slice::<u32>()?;
        match (t, f) {
            (Self::BF16(t), Self::BF16(f)) => {
                let data = wcond(pred, shape, stride, t, stride_t, f, stride_f);
                Ok(Self::BF16(data))
            }
            (Self::F16(t), Self::F16(f)) => {
                let data = wcond(pred, shape, stride, t, stride_t, f, stride_f);
                Ok(Self::F16(data))
            }
            (Self::F32(t), Self::F32(f)) => {
                let data = wcond(pred, shape, stride, t, stride_t, f, stride_f);
                Ok(Self::F32(data))
            }
            (Self::F64(t), Self::F64(f)) => {
                let data = wcond(pred, shape, stride, t, stride_t, f, stride_f);
                Ok(Self::F64(data))
            }
            (Self::U32(t), Self::U32(f)) => {
                let data = wcond(pred, shape, stride, t, stride_t, f, stride_f);
                Ok(Self::U32(data))
            }
            _ => Err(Error::DTypeMismatchBinaryOp {
                lhs: t.dtype(),
                rhs: f.dtype(),
                op: "where_cond",
            }),
        }
    }

    pub(crate) fn embedding_impl(
        &self,
        shape: &Shape,
        stride: &[usize],
        vs: &Self,
        hidden_size: usize,
        vocab_size: usize,
    ) -> Result<Self> {
        let ids = self.as_slice::<u32>()?;
        map1!(vs, take_impl1, ids, shape, stride, vocab_size, hidden_size)
    }

    pub(crate) fn matmul_impl(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_stride: &[usize],
        rhs_stride: &[usize],
    ) -> Result<Self> {
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

        let mut dst = vec![0.0; b * m * n];

        let dst_shape: Shape = (m, n).into();
        let dst_strides = dst_shape.stride_contiguous();
        let dst_rs = dst_strides[0];
        let dst_cs = dst_strides[1];

        for step in 0..b {
            let lhs_p = &self.as_slice::<f32>()?[step * a_skip..];
            let rhs_p = &rhs.as_slice::<f32>()?[step * b_skip..];
            let dst_p = &mut dst[step * c_skip..];
            unsafe {
                gemm(
                    // m: usize,
                    m,
                    // n: usize,
                    n,
                    // k: usize,
                    k,
                    // dst: *mut T,
                    dst_p.as_mut_ptr(),
                    // dst_cs: isize,
                    dst_cs as isize,
                    // dst_rs: isize,
                    dst_rs as isize,
                    // read_dst: bool,
                    false,
                    // lhs: *const T,
                    lhs_p.as_ptr(),
                    // lhs_cs: isize,
                    lhs_cs as isize,
                    // lhs_rs: isize,
                    lhs_rs as isize,
                    // rhs: *const T,
                    rhs_p.as_ptr(),
                    // rhs_cs: isize,
                    rhs_cs as isize,
                    // rhs_rs: isize,
                    rhs_rs as isize,
                    // alpha: T,
                    1.0,
                    // beta: T,
                    1.0,
                    // conj_dst: bool,
                    false,
                    // conj_lhs: bool,
                    false,
                    // conj_rhs: bool,
                    true,
                    // parallelism: Parallelism
                    Parallelism::None,
                )
            }
        }

        let c = Self::F32(dst);
        Ok(c)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Device, Tensor};

    #[test]
    fn simple_matmul() -> Result<()> {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let a = Tensor::from_slice(&data, (2, 2), &Device::Cpu)?;
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = Tensor::from_slice(&data, (2, 2), &Device::Cpu)?;

        let c = a.matmul(&b)?;
        assert_eq!(c.to_vec2::<f32>()?, &[&[7.0f32, 10.0], &[15.0, 22.0]]);

        let data = vec![1.0f32, 2.0];
        let a = Tensor::from_slice(&data, (2, 1), &Device::Cpu)?;
        let data = vec![3.0f32, 4.0];
        let b = Tensor::from_slice(&data, (1, 2), &Device::Cpu)?;
        let c = a.matmul(&b)?;
        assert_eq!(c.to_vec2::<f32>()?, &[&[3.0, 4.0], &[6.0, 8.0]]);

        let data: Vec<_> = (0..6).map(|i| i as f32).collect();
        let a = Tensor::from_slice(&data, (2, 3), &Device::Cpu)?;
        let data: Vec<_> = (0..6).map(|i| (i + 2) as f32).collect();
        let b = Tensor::from_slice(&data, (3, 2), &Device::Cpu)?;
        let c = a.matmul(&b)?;
        assert_eq!(c.to_vec2::<f32>()?, &[&[16., 19.], &[52., 64.]]);

        let data: Vec<_> = (0..12).map(|i| i as f32).collect();
        let a = Tensor::from_slice(&data, (2, 2, 3), &Device::Cpu)?;
        let data: Vec<_> = (0..12).map(|i| (i + 2) as f32).collect();
        let b = Tensor::from_slice(&data, (2, 3, 2), &Device::Cpu)?;
        let c = a.matmul(&b)?;
        assert_eq!(
            c.to_vec3::<f32>()?,
            &[&[&[16., 19.], &[52., 64.]], &[&[214., 235.], &[304., 334.]]]
        );
        Ok(())
    }
}
