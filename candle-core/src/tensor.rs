use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{
    BackpropOp, BinaryOp, CmpOp, CustomOp1, CustomOp2, CustomOp3, Op, ReduceOp, UnaryOp,
};
use crate::shape::{Dim, Dims};
use crate::{storage::Storage, DType, Device, Error, Layout, Result, Shape};
use std::sync::{Arc, RwLock};

/// Unique identifier for tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(usize);

impl TensorId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

pub struct Tensor_ {
    id: TensorId,
    // As we provide inner mutability on the tensor content, the alternatives are:
    // - Using a mutex, this would have the highest cost when retrieving the storage but would
    //   prevent errors when concurrent access takes place. Mutex would also be subject to
    //   deadlocks for example using the current code if the same tensor is used twice by a single
    //   binary op.
    // - Using a refcell unsafe cell would have some intermediary cost, borrow checking would be
    //   verified dynamically, but the resulting tensors would not be send or sync.
    // - Using an unsafe cell would have the lowest cost but undefined behavior on concurrent
    //   accesses.
    // Ideally, we would use Arc<Storage> for tensors on which we don't plan on modifying the data
    // and Arc<Mutex<Storage>> for tensors where the data could be modified, e.g. variables but
    // that's tricky to encode in the current setup.
    storage: Arc<RwLock<Storage>>,
    layout: Layout,
    op: BackpropOp,
    is_variable: bool,
    dtype: DType,
    device: Device,
}

impl AsRef<Self> for Tensor {
    fn as_ref(&self) -> &Self {
        self
    }
}

// Tensors are refcounted so that cloning is cheap when building the op graph.
// Storages are also refcounted independently so that its possible to avoid
// copying the storage for operations that only modify the shape or stride.
#[derive(Clone)]
/// The core struct for manipulating tensors.
///
/// ```rust
/// use candle_core::{Tensor, DType, Device};
///
/// let a = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
/// let b = Tensor::arange(0f32, 12f32, &Device::Cpu)?.reshape((3, 4))?;
///
/// let c = a.matmul(&b)?;
/// # Ok::<(), candle_core::Error>(())
/// ```
///
/// Tensors are reference counted with [`Arc`] so cloning them is cheap.
pub struct Tensor(Arc<Tensor_>);

impl std::ops::Deref for Tensor {
    type Target = Tensor_;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

macro_rules! unary_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self) -> Result<Self> {
            let shape = self.shape();
            let storage = self
                .storage()
                .unary_impl::<crate::op::$op_name>(self.layout())?;
            let op = BackpropOp::new1(self, |s| Op::Unary(s, UnaryOp::$op_name));
            Ok(from_storage_op(storage, shape.clone(), op))
        }
    };
}

macro_rules! binary_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self, rhs: &Self) -> Result<Self> {
            let shape = self.same_shape_binary_op(rhs, stringify!($fn_name))?;
            let storage = self.storage().binary_impl::<crate::op::$op_name>(
                &*rhs.storage(),
                self.layout(),
                rhs.layout(),
            )?;
            let op = BackpropOp::new2(self, rhs, |t1, t2| Op::Binary(t1, t2, BinaryOp::$op_name));
            Ok(from_storage_op(storage, shape.clone(), op))
        }
    };
}

macro_rules! broadcast_binary_op {
    ($fn_name:ident, $inner_fn_name:ident) => {
        pub fn $fn_name(&self, rhs: &Self) -> Result<Self> {
            let lhs = self;
            let shape = lhs.broadcast_shape_binary_op(rhs, stringify!($fn_name))?;
            let l_broadcast = shape != *lhs.shape();
            let r_broadcast = shape != *rhs.shape();
            match (l_broadcast, r_broadcast) {
                (true, true) => lhs
                    .broadcast_as(&shape)?
                    .$inner_fn_name(&rhs.broadcast_as(&shape)?),
                (false, true) => lhs.$inner_fn_name(&rhs.broadcast_as(&shape)?),
                (true, false) => lhs.broadcast_as(&shape)?.$inner_fn_name(rhs),
                (false, false) => lhs.$inner_fn_name(rhs),
            }
        }
    };
}

/// Creates a fresh tensor structure based on a storage and a shape, this uses contiguous strides.
fn from_storage<S: Into<Shape>>(
    storage: Storage,
    shape: S,
    op: impl Into<Option<BackpropOp>>,
    is_variable: bool,
) -> Tensor {
    let op = op.into().unwrap_or_default();
    let dtype = storage.dtype();
    let device = storage.device();
    Tensor_ {
        id: TensorId::new(),
        storage: RwLock::new(storage).into(),
        layout: Layout::contiguous(shape),
        op,
        is_variable,
        dtype,
        device,
    }.into()
}

fn from_storage_op<S: Into<Shape>>(
    storage: Storage,
    shape: S,
    op: impl Into<Option<BackpropOp>>,
) -> Tensor {
    from_storage(storage, shape, op, false)
}

impl From<Tensor_> for Tensor {
    #[inline]
    fn from(value: Tensor_) -> Self {
        Self(value.into())
    }
}

fn dim_match(dim: usize, a: &[usize], b: &[usize]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for (i, (&d1, &d2)) in a.iter().zip(b.iter()).enumerate() {
        if i != dim && d1 != d2 {
            return false;
        }
    }
    true
}

impl Tensor {
    pub(crate) fn ones_impl<S: Into<Shape>>(
        shape: S,
        dtype: DType,
        device: &Device,
        is_variable: bool,
    ) -> Result<Self> {
        if is_variable {
            let shape = shape.into();
            let storage = device.ones(&shape, dtype)?;
            Ok(from_storage(storage, shape, None, is_variable))
        } else {
            let storage = device.ones(&crate::shape::SCALAR, dtype)?;
            from_storage(storage, crate::shape::SCALAR, None, is_variable).broadcast_as(shape)
        }
    }

    /// Creates a new tensor filled with ones.
    ///
    /// ```rust
    /// use candle_core::{Tensor, DType, Device};
    /// let a = Tensor::ones((2, 3), DType::F32, &Device::Cpu)?;
    /// let b = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0], (2, 3), &Device::Cpu)?;
    /// // a == b
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn ones<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        Self::ones_impl(shape, dtype, device, false)
    }

    /// Creates a new tensor filled with ones with same shape, dtype, and device as the other tensor.
    ///
    /// ```rust
    /// use candle_core::{Tensor, DType, Device};
    /// let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    /// let b = a.ones_like()?;
    /// // b == a + 1
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn ones_like(&self) -> Result<Self> {
        Self::ones(self.shape(), self.dtype(), self.device())
    }

    // Do not expose outside of the crate, the `is_variable=true` case should only be accessed from
    // the variable module.
    pub(crate) fn zeros_impl<S: Into<Shape>>(
        shape: S,
        dtype: DType,
        device: &Device,
        is_variable: bool,
    ) -> Result<Self> {
        if is_variable {
            let shape = shape.into();
            let storage = device.zeros(&shape, dtype)?;
            Ok(from_storage(storage, shape, None, is_variable))
        } else {
            let storage = device.zeros(&crate::shape::SCALAR, dtype)?;
            from_storage(storage, crate::shape::SCALAR, None, is_variable).broadcast_as(shape)
        }
    }

    /// Creates a new tensor filled with zeros.
    ///
    /// ```rust
    /// use candle_core::{Tensor, DType, Device};
    /// let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    /// let b = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0], (2, 3), &Device::Cpu)?;
    /// // a == b
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn zeros<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        Self::zeros_impl(shape, dtype, device, false)
    }

    /// Creates a new tensor filled with ones with same shape, dtype, and device as the other
    /// tensor.
    ///
    /// ```rust
    /// use candle_core::{Tensor, DType, Device};
    /// let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    /// let b = a.zeros_like()?;
    /// // b is on CPU f32.
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn zeros_like(&self) -> Result<Self> {
        Self::zeros(self.shape(), self.dtype(), self.device())
    }

    pub(crate) fn rand_impl<S: Into<Shape>, T: crate::FloatDType>(
        lo: T,
        up: T,
        s: S,
        device: &Device,
        is_variable: bool,
    ) -> Result<Self> {
        let s = s.into();
        let storage = device.rand_uniform(lo, up, &s)?;

        Ok(from_storage(storage, s, None, is_variable))
    }

    pub(crate) fn rand_f64_impl<S: Into<Shape>>(
        lo: f64,
        up: f64,
        s: S,
        dtype: DType,
        device: &Device,
        is_variable: bool,
    ) -> Result<Self> {
        let s = s.into();
        let storage = device.rand_uniform_f64(lo, up, &s, dtype)?;
        Ok(from_storage(storage, s, None, is_variable))
    }

    /// Creates a new tensor initialized with values sampled uniformly between `lo` and `up`.
    pub fn rand<S: Into<Shape>, T: crate::FloatDType>(
        lo: T,
        up: T,
        s: S,
        device: &Device,
    ) -> Result<Self> {
        Self::rand_impl(lo, up, s, device, false)
    }

    pub fn rand_like(&self, lo: f64, up: f64) -> Result<Self> {
        Self::rand_f64_impl(lo, up, self.shape(), self.dtype(), self.device(), false)
    }

    pub(crate) fn randn_impl<S: Into<Shape>, T: crate::FloatDType>(
        mean: T,
        std: T,
        s: S,
        device: &Device,
        is_variable: bool,
    ) -> Result<Self> {
        let s = s.into();
        let storage = device.rand_normal(mean, std, &s)?;
        Ok(from_storage(storage, s, None, is_variable))
    }

    pub(crate) fn randn_f64_impl<S: Into<Shape>>(
        mean: f64,
        std: f64,
        s: S,
        dtype: DType,
        device: &Device,
        is_variable: bool,
    ) -> Result<Self> {
        let s = s.into();
        let storage = device.rand_normal_f64(mean, std, &s, dtype)?;
        Ok(from_storage(storage, s, None, is_variable))
    }

    pub fn randn_like(&self, mean: f64, stdev: f64) -> Result<Self> {
        Self::randn_f64_impl(
            mean,
            stdev,
            self.shape(),
            self.dtype(),
            self.device(),
            false,
        )
    }

    /// Creates a new tensor initialized with values sampled from a normal distribution with the
    /// specified `mean` and standard deviation `std`.
    pub fn randn<S: Into<Shape>, T: crate::FloatDType>(
        mean: T,
        std: T,
        s: S,
        device: &Device,
    ) -> Result<Self> {
        Self::randn_impl(mean, std, s, device, false)
    }

    pub(crate) fn new_impl<A: crate::device::NdArray>(
        array: A,
        shape: Shape,
        device: &Device,
        is_variable: bool,
    ) -> Result<Self> {
        let n: usize = shape.elem_count();
        let buffer_size: usize = array.shape()?.elem_count();
        if buffer_size != n {
            return Err(Error::ShapeMismatch { buffer_size, shape }.bt());
        }
        let storage = device.storage(array)?;
        Ok(from_storage(storage, shape, None, is_variable))
    }

    /// Creates a new tensor on the specified device using the content and shape of the input.
    pub fn new<A: crate::device::NdArray>(array: A, device: &Device) -> Result<Self> {
        let shape = array.shape()?;
        Self::new_impl(array, shape, device, false)
    }

    /// Creates a new 1D tensor from an iterator.
    pub fn from_iter<D: crate::WithDType>(
        iter: impl IntoIterator<Item = D>,
        device: &Device,
    ) -> Result<Self> {
        let data = iter.into_iter().collect::<Vec<_>>();
        let len = data.len();
        Self::from_vec_impl(data, len, device, false)
    }

    /// Creates a new 1D tensor with values from the interval `[start, end)` taken with a common
    /// difference `1` from `start`.
    pub fn arange<D: crate::WithDType>(start: D, end: D, device: &Device) -> Result<Self> {
        Self::arange_step(start, end, D::one(), device)
    }

    /// Creates a new 1D tensor with values from the interval `[start, end)` taken with a common
    /// difference `step` from `start`.
    pub fn arange_step<D: crate::WithDType>(
        start: D,
        end: D,
        step: D,
        device: &Device,
    ) -> Result<Self> {
        let mut data = vec![];
        let mut current = start;
        while current < end {
            data.push(current);
            current += step;
        }
        let len = data.len();
        Self::from_vec_impl(data, len, device, false)
    }

    pub(crate) fn from_vec_impl<S: Into<Shape>, D: crate::WithDType>(
        data: Vec<D>,
        shape: S,
        device: &Device,
        is_variable: bool,
    ) -> Result<Self> {
        let shape = shape.into();
        let buffer_size = data.len();
        if buffer_size != shape.elem_count() {
            return Err(Error::ShapeMismatch { buffer_size, shape }.bt());
        }
        let storage = device.storage_owned(data)?;
        Ok(from_storage(storage, shape, None, is_variable))
    }

    /// Creates a new tensor initialized with values from the input vector. The number of elements
    /// in this vector must be the same as the number of elements defined by the shape.
    /// If the device is cpu, no data copy is made.
    pub fn from_vec<S: Into<Shape>, D: crate::WithDType>(
        data: Vec<D>,
        shape: S,
        device: &Device,
    ) -> Result<Self> {
        Self::from_vec_impl(data, shape, device, false)
    }

    /// Creates a new tensor initialized with values from the input slice. The number of elements
    /// in this vector must be the same as the number of elements defined by the shape.
    pub fn from_slice<S: Into<Shape>, D: crate::WithDType>(
        array: &[D],
        shape: S,
        device: &Device,
    ) -> Result<Self> {
        Self::new_impl(array, shape.into(), device, false)
    }

    pub(crate) fn broadcast_shape_binary_op<'a>(
        &'a self,
        rhs: &'a Self,
        op: &'static str,
    ) -> Result<Shape> {
        let lhs = self;
        let lhs_dims = lhs.shape().dims();
        let rhs_dims = rhs.shape().dims();
        let lhs_ndims = lhs_dims.len();
        let rhs_ndims = rhs_dims.len();
        let bcast_ndims = usize::max(lhs_ndims, rhs_ndims);
        let mut bcast_dims = vec![0; bcast_ndims];
        for (idx, bcast_value) in bcast_dims.iter_mut().enumerate() {
            let rev_idx = bcast_ndims - idx;
            let l_value = if lhs_ndims < rev_idx {
                1
            } else {
                lhs_dims[lhs_ndims - rev_idx]
            };
            let r_value = if rhs_ndims < rev_idx {
                1
            } else {
                rhs_dims[rhs_ndims - rev_idx]
            };
            *bcast_value = if l_value == r_value {
                l_value
            } else if l_value == 1 {
                r_value
            } else if r_value == 1 {
                l_value
            } else {
                Err(Error::ShapeMismatchBinaryOp {
                    lhs: self.shape().clone(),
                    rhs: rhs.shape().clone(),
                    op,
                }
                .bt())?
            }
        }
        Ok(Shape::from(bcast_dims))
    }

    pub(crate) fn same_shape_binary_op(&self, rhs: &Self, op: &'static str) -> Result<&Shape> {
        let lhs = self.shape();
        let rhs = rhs.shape();
        if lhs != rhs {
            Err(Error::ShapeMismatchBinaryOp {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                op,
            }
            .bt())
        } else {
            Ok(lhs)
        }
    }

    /// Returns true if the computation graph should track this op, that is if it is
    /// a variable or if it has some variable as dependencies.
    pub(crate) fn track_op(&self) -> bool {
        self.is_variable || self.op.is_some()
    }

    // TODO: Also make an inplace version or a pre-allocated? This could be tricky
    // if this can create cycles in the compute graph.
    binary_op!(add, Add);
    binary_op!(mul, Mul);
    binary_op!(sub, Sub);
    binary_op!(div, Div);
    broadcast_binary_op!(broadcast_add, add);
    broadcast_binary_op!(broadcast_mul, mul);
    broadcast_binary_op!(broadcast_sub, sub);
    broadcast_binary_op!(broadcast_div, div);

    unary_op!(recip, Recip);
    unary_op!(neg, Neg);
    unary_op!(exp, Exp);
    unary_op!(log, Log);
    unary_op!(sin, Sin);
    unary_op!(cos, Cos);
    unary_op!(abs, Abs);
    unary_op!(sqr, Sqr);
    unary_op!(sqrt, Sqrt);
    unary_op!(gelu, Gelu);
    unary_op!(relu, Relu);

    /// Retrieves the single scalar value hold in the tensor. If the tensor contains multiple
    /// dimensions, an error is returned instead.
    pub fn to_scalar<S: crate::WithDType>(&self) -> Result<S> {
        if self.rank() != 0 {
            Err(Error::UnexpectedNumberOfDims {
                expected: 0,
                got: self.rank(),
                shape: self.shape().clone(),
            }
            .bt())?
        }
        let from_cpu_storage = |cpu_storage: &crate::CpuStorage| {
            let data = S::cpu_storage_as_slice(cpu_storage)?;
            Ok::<_, Error>(data[self.layout().start_offset()])
        };
        match &*self.storage() {
            Storage::Cpu(cpu_storage) => from_cpu_storage(cpu_storage),
            Storage::Cuda(storage) => from_cpu_storage(&storage.to_cpu_storage()?),
        }
    }

    /// An alias for `to_scalar`.
    pub fn to_vec0<S: crate::WithDType>(&self) -> Result<S> {
        self.to_scalar::<S>()
    }

    /// This operation multiplies the input tensor by `mul` then adds `add` and return the result.
    /// The input values `mul` and `add` are casted to the appropriate type so some rounding might
    /// be performed.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let a = Tensor::new(&[[0f32, 1.], [2., 3.]], &Device::Cpu)?;
    /// let a = a.affine(4., -2.)?;
    /// assert_eq!(a.to_vec2::<f32>()?, &[[-2.0, 2.0], [6.0, 10.0]]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn affine(&self, mul: f64, add: f64) -> Result<Self> {
        let storage = self.storage().affine(self.layout(), mul, add)?;
        let op = BackpropOp::new1(self, |arg| Op::Affine { arg, mul, add });
        Ok(from_storage_op(storage, self.shape(), op))
    }

    /// Applies the Exponential Linear Unit (ELU) function on each element of the input tensor.
    pub fn elu(&self, alpha: f64) -> Result<Self> {
        let storage = self.storage().elu(self.layout(), alpha)?;
        let op = BackpropOp::new1(self, |t| Op::Elu(t, alpha));
        Ok(from_storage_op(storage, self.shape(), op))
    }

    fn check_dim(&self, dim: usize, op: &'static str) -> Result<()> {
        if dim >= self.dims().len() {
            Err(Error::DimOutOfRange {
                shape: self.shape().clone(),
                dim: dim as i32,
                op,
            }
            .bt())?
        } else {
            Ok(())
        }
    }

    /// Split a tensor into the specified number of chunks, this may return less chunks than
    /// specificed.
    pub fn chunk<D: Dim>(&self, chunks: usize, dim: D) -> Result<Vec<Self>> {
        let dim = dim.to_index(self.shape(), "chunk")?;
        let size = self.dim(dim)?;
        if size < chunks {
            (0..size).map(|i| self.narrow(dim, i, 1)).collect()
        } else {
            let chunk_size = size / chunks;
            let cnt_additional = size % chunks;
            let mut tensors = vec![];
            let mut sum_chunk_size = 0;
            for i in 0..chunks {
                let chunk_size = if i < cnt_additional {
                    chunk_size + 1
                } else {
                    chunk_size
                };
                let tensor = self.narrow(dim, sum_chunk_size, chunk_size)?;
                tensors.push(tensor);
                sum_chunk_size += chunk_size
            }
            Ok(tensors)
        }
    }

    /// Returns a new tensor that is a narrowed version of the input, the dimension `dim`
    /// ranges from `start` to `start + len`.
    pub fn narrow<D: Dim>(&self, dim: D, start: usize, len: usize) -> Result<Self> {
        let dims = self.dims();
        let dim = dim.to_index(self.shape(), "narrow")?;
        if start + len > dims[dim] {
            Err(Error::NarrowInvalidArgs {
                shape: self.shape().clone(),
                dim,
                start,
                len,
                msg: "start + len > dim_len",
            }
            .bt())?
        }
        if start == 0 && dims[dim] == len {
            Ok(self.clone())
        } else {
            let op = BackpropOp::new1(self, |t| Op::Narrow(t, dim, start, len));
            let layout = self.layout().narrow(dim, start, len)?;
            Ok(Tensor_ {
                id: TensorId::new(),
                storage: self.storage.clone(),
                layout,
                op,
                is_variable: false,
                dtype: self.dtype,
                device: self.device.clone(),
            }.into())
        }
    }

    fn squeeze_dims(self, dims: &[usize]) -> Result<Self> {
        match dims {
            [] => Ok(self),
            [i] => self.squeeze(*i),
            dims => {
                let dims = self
                    .dims()
                    .iter()
                    .enumerate()
                    .filter_map(|(dim_idx, &v)| {
                        if dims.contains(&dim_idx) {
                            None
                        } else {
                            Some(v)
                        }
                    })
                    .collect::<Vec<_>>();
                self.reshape(dims)
            }
        }
    }

    fn reduce_impl<D: Dim>(&self, dim: D, keepdim: bool, op: ReduceOp) -> Result<Self> {
        let dim = dim.to_index(self.shape(), op.name())?;
        let storage = self.storage().reduce_op(op, self.layout(), &[dim])?;
        let mut dims = self.dims().to_vec();
        dims[dim] = 1;
        let op = BackpropOp::new1(self, |arg| Op::Reduce(arg, op, dims.to_vec()));
        let res = from_storage_op(storage, dims, op);
        if keepdim {
            Ok(res)
        } else {
            res.squeeze_dims(&[dim])
        }
    }

    fn sum_impl<D: Dims>(&self, sum_dims: D, keepdim: bool) -> Result<Self> {
        let sum_dims = sum_dims.to_indexes(self.shape(), "sum")?;
        let storage = self
            .storage()
            .reduce_op(ReduceOp::Sum, self.layout(), &sum_dims)?;
        let mut dims = self.dims().to_vec();
        for &sum_dim in sum_dims.iter() {
            dims[sum_dim] = 1
        }
        let op = BackpropOp::new1(self, |a| Op::Reduce(a, ReduceOp::Sum, dims.to_vec()));
        let sum = from_storage_op(storage, dims, op);
        if keepdim {
            Ok(sum)
        } else {
            sum.squeeze_dims(&sum_dims)
        }
    }

    /// Returns the sum of all elements in the input tensor. The sum is performed over all the
    /// input dimensions.
    ///
    /// The resulting tensor has a shape that is similar to the shape of the input tensor, except
    /// that the number of elements for each dimension index in `sum_dims` is 1.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let a = Tensor::new(&[[0f32, 1.], [2., 3.]], &Device::Cpu)?;
    /// let s = a.sum_keepdim(0)?;
    /// assert_eq!(s.to_vec2::<f32>()?, &[[2., 4.]]);
    /// let s = a.sum_keepdim(1)?;
    /// assert_eq!(s.to_vec2::<f32>()?, &[[1.], [5.]]);
    /// let s = a.sum_keepdim((0, 1))?;
    /// assert_eq!(s.to_vec2::<f32>()?, &[[6.]]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn sum_keepdim<D: Dims>(&self, sum_dims: D) -> Result<Self> {
        self.sum_impl(sum_dims, true)
    }

    /// Returns the sum of all elements in the input tensor. The sum is performed over all the
    /// input dimensions and compared to `sum_keepdim` these dimensions are squeezed rather than
    /// kept.
    pub fn sum<D: Dims>(&self, sum_dims: D) -> Result<Self> {
        self.sum_impl(sum_dims, false)
    }

    pub fn max_keepdim<D: Dim>(&self, dim: D) -> Result<Self> {
        self.reduce_impl(dim, true, ReduceOp::Max)
    }

    pub fn max<D: Dim>(&self, dim: D) -> Result<Self> {
        self.reduce_impl(dim, false, ReduceOp::Max)
    }

    pub fn min_keepdim<D: Dim>(&self, dim: D) -> Result<Self> {
        self.reduce_impl(dim, true, ReduceOp::Min)
    }

    pub fn min<D: Dim>(&self, dim: D) -> Result<Self> {
        self.reduce_impl(dim, false, ReduceOp::Min)
    }

    pub fn argmax_keepdim<D: Dim>(&self, dim: D) -> Result<Self> {
        self.reduce_impl(dim, true, ReduceOp::ArgMax)
    }

    pub fn argmax<D: Dim>(&self, dim: D) -> Result<Self> {
        self.reduce_impl(dim, false, ReduceOp::ArgMax)
    }

    pub fn argmin_keepdim<D: Dim>(&self, dim: D) -> Result<Self> {
        self.reduce_impl(dim, true, ReduceOp::ArgMin)
    }

    pub fn argmin<D: Dim>(&self, dim: D) -> Result<Self> {
        self.reduce_impl(dim, false, ReduceOp::ArgMin)
    }

    pub fn cmp(&self, rhs: &Self, op: CmpOp) -> Result<Self> {
        let shape = self.same_shape_binary_op(rhs, "cmp")?;
        let storage = self
            .storage()
            .cmp(op, &rhs.storage(), self.layout(), rhs.layout())?;
        let op = BackpropOp::new1(self, |a| Op::Cmp(a, op));
        Ok(from_storage_op(storage, shape.dims(), op))
    }

    pub fn eq(&self, rhs: &Self) -> Result<Self> {
        self.cmp(rhs, CmpOp::Eq)
    }

    pub fn ne(&self, rhs: &Self) -> Result<Self> {
        self.cmp(rhs, CmpOp::Ne)
    }

    pub fn lt(&self, rhs: &Self) -> Result<Self> {
        self.cmp(rhs, CmpOp::Lt)
    }

    pub fn gt(&self, rhs: &Self) -> Result<Self> {
        self.cmp(rhs, CmpOp::Gt)
    }

    pub fn ge(&self, rhs: &Self) -> Result<Self> {
        self.cmp(rhs, CmpOp::Ge)
    }

    pub fn le(&self, rhs: &Self) -> Result<Self> {
        self.cmp(rhs, CmpOp::Le)
    }

    /// Applies a 1D convolution over the input tensor.
    pub fn conv1d(&self, kernel: &Self, padding: usize, stride: usize) -> Result<Self> {
        let (c_out, c_in_k, k_size) = kernel.dims3()?;
        let (b_size, c_in, l_in) = match *self.dims() {
            [b_size, c_in, l_in] => (Some(b_size), c_in, l_in),
            [c_in, l_in] => (None, c_in, l_in),
            _ => Err(Error::Conv1dInvalidArgs {
                inp_shape: self.shape().clone(),
                k_shape: kernel.shape().clone(),
                padding,
                stride,
                msg: "input rank is not 2 or 3",
            }
            .bt())?,
        };
        if c_in != c_in_k {
            Err(Error::Conv1dInvalidArgs {
                inp_shape: self.shape().clone(),
                k_shape: kernel.shape().clone(),
                padding,
                stride,
                msg: "the number of in-channels on the input doesn't match the kernel size",
            }
            .bt())?
        }
        let params = crate::conv::ParamsConv1D {
            b_size,
            l_in,
            c_out,
            c_in,
            k_size,
            padding,
            stride,
        };
        let storage =
            self.storage()
                .conv1d(self.layout(), &kernel.storage(), kernel.layout(), &params)?;
        let op = BackpropOp::new2(self, kernel, |arg, kernel| Op::Conv1D {
            arg,
            kernel,
            padding,
            stride,
        });
        let out_dims = params.out_dims();
        Ok(from_storage_op(storage, out_dims, op))
    }

    pub fn conv2d(&self, kernel: &Self, padding: usize, stride: usize) -> Result<Self> {
        let (b_size, c_in, i_h, i_w) = self.dims4()?;
        let (c_out, c_in_k, k_h, k_w) = kernel.dims4()?;
        if c_in != c_in_k {
            crate::bail!("in_channel mismatch between input ({c_in}) and kernel ({c_in_k})")
        }
        let params = crate::conv::ParamsConv2D {
            b_size,
            i_h,
            i_w,
            k_h,
            k_w,
            c_out,
            c_in,
            padding,
            stride,
        };
        let storage =
            self.storage()
                .conv2d(self.layout(), &kernel.storage(), kernel.layout(), &params)?;
        let op = BackpropOp::new2(self, kernel, |arg, kernel| Op::Conv2D {
            arg,
            kernel,
            padding,
            stride,
        });
        let out_dims = params.out_dims();
        Ok(from_storage_op(storage, out_dims, op))
    }

    pub fn upsample_nearest2d(&self, target_h: usize, target_w: usize) -> Result<Self> {
        let (n, c, _h, _w) = self.dims4()?;
        let op = BackpropOp::new1(self, Op::UpsampleNearest2D);
        let storage = self
            .storage()
            .upsample_nearest2d(self.layout(), target_h, target_w)?;
        Ok(from_storage_op(storage, (n, c, target_h, target_w), op))
    }

    pub fn avg_pool2d(&self, kernel_size: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        let (n, c, h, w) = self.dims4()?;
        // https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html#torch.nn.AvgPool2d
        let h_out = (h - kernel_size.0) / stride.0 + 1;
        let w_out = (w - kernel_size.1) / stride.1 + 1;
        let op = BackpropOp::new1(self, |arg| Op::AvgPool2D {
            arg,
            kernel_size,
            stride,
        });
        let storage = self
            .storage()
            .avg_pool2d(self.layout(), kernel_size, stride)?;
        Ok(from_storage_op(storage, (n, c, h_out, w_out), op))
    }

    /// Returns the matrix-multiplication of the input tensor with the other provided tensor.
    ///
    /// # Arguments
    ///
    /// * `self` - A tensor with dimensions `b1, b2, ..., bi, m, k`.
    /// * `rhs` - A tensor with dimensions `b1, b2, ..., bi, k, n`.
    ///
    /// The resulting tensor has dimensions `b1, b2, ..., bi, m, n`.
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        let a_dims = self.shape().dims();
        let b_dims = rhs.shape().dims();

        let dim = a_dims.len();

        if dim < 2 || b_dims.len() != dim {
            Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: rhs.shape().clone(),
                op: "matmul",
            }
            .bt())?
        }

        let m = a_dims[dim - 2];
        let k = a_dims[dim - 1];
        let k2 = b_dims[dim - 2];
        let n = b_dims[dim - 1];

        let c_shape = Shape::from(&a_dims[..dim - 2]).extend(&[m, n]);
        let batching: usize = a_dims[..dim - 2].iter().product();
        let batching_b: usize = b_dims[..dim - 2].iter().product();
        if k != k2 || batching != batching_b {
            Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: rhs.shape().clone(),
                op: "matmul",
            }
            .bt())?
        }

        let storage = self.storage().matmul(
            &rhs.storage(),
            (batching, m, n, k),
            self.layout(),
            rhs.layout(),
        )?;
        let op = BackpropOp::new2(self, rhs, Op::Matmul);
        Ok(from_storage_op(storage, c_shape, op))
    }

    /// Returns a tensor with the same shape as the input tensor, the values are taken from
    /// `on_true` if the input tensor value is not zero, and `on_false` at the positions where the
    /// input tensor is equal to zero.
    pub fn where_cond(&self, on_true: &Self, on_false: &Self) -> Result<Self> {
        let _shap = self.same_shape_binary_op(on_true, "where_cond")?;
        let shape = self.same_shape_binary_op(on_false, "where_cond")?;
        let storage = self.storage().where_cond(
            self.layout(),
            &on_true.storage(),
            on_true.layout(),
            &on_false.storage(),
            on_false.layout(),
        )?;
        let op = BackpropOp::new3(self, on_true, on_false, Op::WhereCond);
        Ok(from_storage_op(storage, shape, op))
    }

    /// Returns a tensor with the values from the `self` tensor at the index corresponding to the
    /// values hold in the `ids` tensor.
    ///
    /// # Arguments
    ///
    /// * `self` - A tensor with dimensions `v, h`.
    /// * `ids` - A tensor with dimensions `s` and with integer values between 0 and v (exclusive).
    ///
    /// The resulting tensor has dimensions `s, h`. `s` is called the sequence length, `v` the
    /// vocabulary size, and `h` the hidden size.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let values = Tensor::new(&[[0f32, 1.], [2., 3.], [4., 5.]], &Device::Cpu)?;
    /// let ids = Tensor::new(&[2u32, 1u32, 2u32], &Device::Cpu)?;
    /// let emb = values.embedding(&ids)?;
    /// assert_eq!(emb.to_vec2::<f32>()?, &[[4., 5.], [2., 3.], [4., 5.]]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn embedding(&self, ids: &Self) -> Result<Self> {
        if self.rank() != 2 || ids.rank() != 1 {
            Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: ids.shape().clone(),
                op: "embedding",
            }
            .bt())?
        }
        self.index_select(ids, 0)
    }

    pub fn scatter_add<D: Dim>(&self, indexes: &Self, source: &Self, dim: D) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "scatter-add")?;
        let source_dims = source.dims();
        let self_dims = self.dims();

        let dim_match = dim_match(dim, self_dims, source_dims);
        if !dim_match {
            Err(Error::ShapeMismatchBinaryOp {
                op: "scatter-add (self, src)",
                lhs: self.shape().clone(),
                rhs: source.shape().clone(),
            })?
        }
        if indexes.dims() != source.dims() {
            Err(Error::ShapeMismatchBinaryOp {
                op: "scatter-add (indexes, src)",
                lhs: indexes.shape().clone(),
                rhs: source.shape().clone(),
            })?
        }
        let storage = self.storage().scatter_add(
            self.layout(),
            &indexes.storage(),
            indexes.layout(),
            &source.storage(),
            source.layout(),
            dim,
        )?;
        let op = BackpropOp::new3(self, indexes, source, |t1, t2, t3| {
            Op::ScatterAdd(t1, t2, t3, dim)
        });
        Ok(from_storage_op(storage, self.shape(), op))
    }

    pub fn index_add<D: Dim>(&self, indexes: &Self, source: &Self, dim: D) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "index-add")?;
        let source_dims = source.dims();
        let self_dims = self.dims();
        let dim_match = dim_match(dim, self_dims, source_dims);
        if !dim_match {
            Err(Error::ShapeMismatchBinaryOp {
                op: "index-add (self, source)",
                lhs: self.shape().clone(),
                rhs: source.shape().clone(),
            })?
        }
        // The number of element in indexes must match the dimension on which the add is
        // performed on the source tensor (and the index values from `indexes` are taken from
        // the target tensor self)
        let indexes_len = indexes.dims1()?;
        if source_dims[dim] != indexes_len {
            Err(Error::ShapeMismatchBinaryOp {
                op: "index-add (ids, source))",
                lhs: indexes.shape().clone(),
                rhs: source.shape().clone(),
            })?
        }
        let storage = self.storage().index_add(
            self.layout(),
            &indexes.storage(),
            indexes.layout(),
            &source.storage(),
            source.layout(),
            dim,
        )?;
        let op = BackpropOp::new3(self, indexes, source, |t1, t2, t3| {
            Op::IndexAdd(t1, t2, t3, dim)
        });
        Ok(from_storage_op(storage, self.shape(), op))
    }

    pub fn gather<D: Dim>(&self, indexes: &Self, dim: D) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "gather")?;
        let self_dims = self.dims();
        let indexes_dims = indexes.dims();

        let dim_match = dim_match(dim, self_dims, indexes_dims);
        if !dim_match {
            Err(Error::ShapeMismatchBinaryOp {
                op: "gather",
                lhs: self.shape().clone(),
                rhs: indexes.shape().clone(),
            })?
        }
        let storage =
            self.storage()
                .gather(self.layout(), &indexes.storage(), indexes.layout(), dim)?;
        let op = BackpropOp::new2(self, indexes, |t1, t2| Op::Gather(t1, t2, dim));
        Ok(from_storage_op(storage, indexes.shape(), op))
    }

    pub fn index_select<D: Dim>(&self, indexes: &Self, dim: D) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "index-select")?;
        let indexes_len = match indexes.dims() {
            [l] => *l,
            _ => Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: indexes.shape().clone(),
                op: "index-select",
            }
            .bt())?,
        };
        let storage = self.storage().index_select(
            &indexes.storage(),
            self.layout(),
            indexes.layout(),
            dim,
        )?;
        let mut dims = self.dims().to_vec();
        dims[dim] = indexes_len;
        let op = BackpropOp::new2(self, indexes, |t1, t2| Op::IndexSelect(t1, t2, dim));
        Ok(from_storage_op(storage, dims, op))
    }

    /// Returns an iterator over position of the elements in the storage when ranging over the
    /// index tuples in lexicographic order.
    pub fn strided_index(&self) -> crate::StridedIndex {
        self.layout.strided_index()
    }

    /// Similar to `strided_index` but returns the position of the start of each contiguous block
    /// as well as the length of the contiguous blocks. For a contiguous tensor, the index iterator
    /// will only return the start offset and the size would be the number of elements in the
    /// tensor.
    pub fn strided_blocks(&self) -> crate::StridedBlocks {
        self.layout.strided_blocks()
    }

    /// Returns the data contained in a 1D tensor as a vector of scalar values.
    pub fn to_vec1<S: crate::WithDType>(&self) -> Result<Vec<S>> {
        if self.rank() != 1 {
            Err(Error::UnexpectedNumberOfDims {
                expected: 1,
                got: self.rank(),
                shape: self.shape().clone(),
            }
            .bt())?
        }
        let from_cpu_storage = |cpu_storage: &crate::CpuStorage| {
            let data = S::cpu_storage_as_slice(cpu_storage)?;
            let data = match self.layout.contiguous_offsets() {
                Some((o1, o2)) => data[o1..o2].to_vec(),
                None => self.strided_index().map(|i| data[i]).collect(),
            };
            Ok::<Vec<_>, Error>(data)
        };
        match &*self.storage() {
            Storage::Cpu(storage) => from_cpu_storage(storage),
            Storage::Cuda(storage) => from_cpu_storage(&storage.to_cpu_storage()?),
        }
    }

    /// Returns the data contained in a 2D tensor as a vector of vector of scalar values.
    pub fn to_vec2<S: crate::WithDType>(&self) -> Result<Vec<Vec<S>>> {
        let (dim1, dim2) = self.dims2()?;
        let from_cpu_storage = |cpu_storage: &crate::CpuStorage| {
            let data = S::cpu_storage_as_slice(cpu_storage)?;
            let mut rows = vec![];
            match self.layout.contiguous_offsets() {
                Some((o1, o2)) => {
                    let data = &data[o1..o2];
                    for idx_row in 0..dim1 {
                        rows.push(data[idx_row * dim2..(idx_row + 1) * dim2].to_vec())
                    }
                }
                None => {
                    let mut src_index = self.strided_index();
                    for _idx_row in 0..dim1 {
                        let row = (0..dim2).map(|_| data[src_index.next().unwrap()]).collect();
                        rows.push(row)
                    }
                    assert!(src_index.next().is_none());
                }
            }
            Ok(rows)
        };
        match &*self.storage() {
            Storage::Cpu(storage) => from_cpu_storage(storage),
            Storage::Cuda(storage) => from_cpu_storage(&storage.to_cpu_storage()?),
        }
    }

    /// Returns the data contained in a 3D tensor.
    pub fn to_vec3<S: crate::WithDType>(&self) -> Result<Vec<Vec<Vec<S>>>> {
        let (dim1, dim2, dim3) = self.dims3()?;
        let from_cpu_storage = |cpu_storage: &crate::CpuStorage| {
            let data = S::cpu_storage_as_slice(cpu_storage)?;
            let mut top_rows = vec![];
            match self.layout.contiguous_offsets() {
                Some((o1, o2)) => {
                    let data = &data[o1..o2];
                    let dim23 = dim2 * dim3;
                    for idx1 in 0..dim1 {
                        let data = &data[idx1 * dim23..(idx1 + 1) * dim23];
                        let mut rows = vec![];
                        for idx2 in 0..dim2 {
                            rows.push(data[idx2 * dim3..(idx2 + 1) * dim3].to_vec())
                        }
                        top_rows.push(rows);
                    }
                }
                None => {
                    let mut src_index = self.strided_index();
                    for _idx in 0..dim1 {
                        let mut rows = vec![];
                        for _jdx in 0..dim2 {
                            let row = (0..dim3).map(|_| data[src_index.next().unwrap()]).collect();
                            rows.push(row)
                        }
                        top_rows.push(rows);
                    }
                    assert!(src_index.next().is_none());
                }
            }
            Ok(top_rows)
        };
        match &*self.storage() {
            Storage::Cpu(storage) => from_cpu_storage(storage),
            Storage::Cuda(storage) => from_cpu_storage(&storage.to_cpu_storage()?),
        }
    }

    /// The dtype for the elements stored in the input tensor.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The device on which the input tensor is located.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The tensor shape, i.e. dimension sizes on each axis.
    pub fn shape(&self) -> &Shape {
        self.layout().shape()
    }

    /// The dimension size for this tensor on each axis.
    pub fn dims(&self) -> &[usize] {
        self.shape().dims()
    }

    /// The dimension size for a specified dimension index.
    pub fn dim<D: Dim>(&self, dim: D) -> Result<usize> {
        let dim = dim.to_index(self.shape(), "dim")?;
        Ok(self.dims()[dim])
    }

    /// The layout of the input tensor, this stores both the shape of the tensor as well as the
    /// strides and the start offset to apply to the underlying storage.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn stride(&self) -> &[usize] {
        self.layout.stride()
    }

    /// The number of dimensions for this tensor, 0 for a scalar tensor, 1 for a 1D tensor, etc.
    pub fn rank(&self) -> usize {
        self.shape().rank()
    }

    /// The number of elements stored in this tensor.
    pub fn elem_count(&self) -> usize {
        self.shape().elem_count()
    }

    /// The unique identifier for this tensor.
    pub fn id(&self) -> TensorId {
        self.id
    }

    /// Whether this tensor is a variable or not. A variable is a tensor for which gradient is
    /// tracked and on which backpropagation can be performed.
    pub fn is_variable(&self) -> bool {
        self.is_variable
    }

    pub(crate) fn op(&self) -> &Option<Op> {
        &self.op
    }

    /// Computes the sum of all the elements in this tensor and returns a tensor holding this
    /// scalar with zero dimensions.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::new(&[[0f32, 1.], [2., 3.], [4., 5.]], &Device::Cpu)?;
    /// let tensor = tensor.sum_all()?;
    /// assert_eq!(tensor.to_scalar::<f32>()?, 15.);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn sum_all(&self) -> Result<Self> {
        let dims: Vec<_> = (0..self.rank()).collect();
        self.sum(dims)
    }

    fn flatten_<D1: Dim, D2: Dim>(
        &self,
        start_dim: Option<D1>,
        end_dim: Option<D2>,
    ) -> Result<Self> {
        if self.rank() == 0 {
            self.reshape(1)
        } else {
            let start_dim = match start_dim {
                None => 0,
                Some(dim) => dim.to_index(self.shape(), "flatten")?,
            };
            let end_dim = match end_dim {
                None => self.rank() - 1,
                Some(dim) => dim.to_index(self.shape(), "flatten")?,
            };
            if start_dim < end_dim {
                let dims = self.dims();
                let mut dst_dims = dims[..start_dim].to_vec();
                dst_dims.push(dims[start_dim..end_dim + 1].iter().product::<usize>());
                if end_dim + 1 < dims.len() {
                    dst_dims.extend(&dims[end_dim + 1..]);
                }
                self.reshape(dst_dims)
            } else {
                Ok(self.clone())
            }
        }
    }

    /// Flattens the input tensor on the dimension indexes from `start_dim` to `end_dim` (both
    /// inclusive).
    pub fn flatten<D1: Dim, D2: Dim>(&self, start_dim: D1, end_dim: D2) -> Result<Self> {
        self.flatten_(Some(start_dim), Some(end_dim))
    }

    /// Flattens the input tensor on the dimension indexes from `0` to `end_dim` (inclusive).
    pub fn flatten_to<D: Dim>(&self, end_dim: D) -> Result<Self> {
        self.flatten_(None::<usize>, Some(end_dim))
    }

    /// Flattens the input tensor on the dimension indexes from `start_dim` (inclusive) to the last
    /// dimension.
    pub fn flatten_from<D: Dim>(&self, start_dim: D) -> Result<Self> {
        self.flatten_(Some(start_dim), None::<usize>)
    }

    /// Flattens the input tensor by reshaping it into a one dimension tensor.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::new(&[[0f32, 1.], [2., 3.], [4., 5.]], &Device::Cpu)?;
    /// let tensor = tensor.flatten_all()?;
    /// assert_eq!(tensor.to_vec1::<f32>()?, &[0., 1., 2., 3., 4., 5.]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn flatten_all(&self) -> Result<Self> {
        self.flatten_(None::<usize>, None::<usize>)
    }

    /// Returns the sub-tensor fixing the index at `i` on the first dimension.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::new(&[[0f32, 1.], [2., 3.], [4., 5.]], &Device::Cpu)?;
    /// let t = tensor.get(0)?;
    /// assert_eq!(t.to_vec1::<f32>()?, &[0., 1.]);
    /// let t = tensor.get(1)?;
    /// assert_eq!(t.to_vec1::<f32>()?, &[2., 3.]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn get(&self, i: usize) -> Result<Self> {
        let dims = self.dims();
        if dims.is_empty() {
            Ok(self.clone())
        } else {
            self.narrow(0, i, 1)?.reshape(&dims[1..])
        }
    }

    /// Returns a tensor that is a transposed version of the input, the two last dimensions of the
    /// input are swapped.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::new(&[[0f32, 1.], [2., 3.], [4., 5.]], &Device::Cpu)?;
    /// let tensor = tensor.t()?;
    /// assert_eq!(tensor.to_vec2::<f32>()?, &[[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn t(&self) -> Result<Self> {
        let rank = self.rank();
        if rank < 2 {
            Err(Error::UnexpectedNumberOfDims {
                expected: 2,
                got: rank,
                shape: self.shape().clone(),
            }
            .bt())?
        }
        self.transpose(rank - 2, rank - 1)
    }

    /// Returns a tensor that is a transposed version of the input, the given dimensions are
    /// swapped.
    pub fn transpose<D1: Dim, D2: Dim>(&self, dim1: D1, dim2: D2) -> Result<Self> {
        let dim1 = dim1.to_index(self.shape(), "transpose")?;
        let dim2 = dim2.to_index(self.shape(), "transpose")?;
        let op = BackpropOp::new1(self, |t| Op::Transpose(t, dim1, dim2));
        Ok(Tensor_ {
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: self.layout.transpose(dim1, dim2)?,
            op,
            is_variable: false,
            dtype: self.dtype,
            device: self.device.clone(),
        }.into())
    }

    /// Returns true if the data is stored in a C contiguous (aka row major) way.
    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    /// Returns true if the data is stored in a Fortran contiguous (aka column major) way.
    pub fn is_fortran_contiguous(&self) -> bool {
        self.layout.is_fortran_contiguous()
    }

    /// Compared to clone, this copies the actual storage but may fail because of running out of
    /// memory.
    pub fn copy(&self) -> Result<Self> {
        let op = BackpropOp::new1(self, Op::Copy);
        Ok(Tensor_ {
            id: TensorId::new(),
            storage: RwLock::new(self.storage().try_clone(self.layout())?).into(),
            layout: self.layout.clone(),
            op,
            is_variable: false,
            dtype: self.dtype,
            device: self.device.clone(),
        }.into())
    }

    /// Returns a new tensor detached from the current graph, gradient are not propagated through
    /// this new node. The storage of this tensor is shared with the initial tensor.
    pub fn detach(&self) -> Result<Self> {
        Ok(Tensor_ {
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: self.layout.clone(),
            op: BackpropOp::none(),
            is_variable: false,
            dtype: self.dtype,
            device: self.device.clone(),
        }.into())
    }

    /// If the target device is the same as the tensor device, only a shallow copy is performed.
    pub fn to_device(&self, device: &Device) -> Result<Self> {
        if self.device().same_device(device) {
            Ok(self.clone())
        } else {
            let storage = match (&*self.storage(), device) {
                (Storage::Cpu(storage), Device::Cuda(cuda)) => {
                    Storage::Cuda(cuda.storage_from_cpu_storage(storage)?)
                }
                (Storage::Cuda(storage), Device::Cpu) => Storage::Cpu(storage.to_cpu_storage()?),
                (Storage::Cuda(storage), Device::Cuda(cuda)) => {
                    // TODO: Avoid passing through the cpu storage here, especially if the gpu ids
                    // are the same.
                    let cpu_storage = storage.to_cpu_storage()?;
                    Storage::Cuda(cuda.storage_from_cpu_storage(&cpu_storage)?)
                }
                (Storage::Cpu(storage), Device::Cpu) => Storage::Cpu(storage.clone()),
            };
            let op = BackpropOp::new1(self, Op::ToDevice);
            Ok(Tensor_ {
                id: TensorId::new(),
                storage: RwLock::new(storage).into(),
                layout: self.layout.clone(),
                op,
                is_variable: false,
                dtype: self.dtype,
                device: device.clone(),
            }.into())
        }
    }

    /// Returns a new tensor duplicating data from the original tensor. New dimensions are inserted
    /// on the left.
    pub fn broadcast_left<S: Into<Shape>>(&self, left_shape: S) -> Result<Self> {
        let left_shape = left_shape.into();
        let mut dims = left_shape.into_dims();
        dims.extend(self.dims());
        self.broadcast_as(dims)
    }

    /// Broadcast the input tensor to the target shape. This returns an error if the input shape is
    /// not compatible with the target shape.
    ///
    /// If the input shape is `i_1, i_2, ... i_k`, the target shape has to have `k` dimensions or
    /// more and shape `j_1, ..., j_l, t_1, t_2, ..., t_k`. The dimensions `j_1` to `j_l` can have
    /// any value, the dimension `t_a` must be equal to `i_a` if `i_a` is different from 1. If
    /// `i_a` is equal to 1, any value can be used.
    pub fn broadcast_as<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        Ok(Tensor_ {
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: self.layout.broadcast_as(shape)?,
            op: BackpropOp::new1(self, Op::Broadcast),
            is_variable: false,
            dtype: self.dtype,
            device: self.device.clone(),
        }.into())
    }

    /// An alias for broadcast_as.
    pub fn expand<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        self.broadcast_as(shape)
    }

    /// Casts the input tensor to the target `dtype`.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::new(3.14159265358979f64, &Device::Cpu)?;
    /// assert_eq!(tensor.to_scalar::<f64>()?, 3.14159265358979);
    /// let tensor = tensor.to_dtype(candle_core::DType::F32)?;
    /// assert_eq!(tensor.to_scalar::<f32>()?, 3.1415927);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn to_dtype(&self, dtype: DType) -> Result<Self> {
        if self.dtype() == dtype {
            Ok(self.clone())
        } else {
            let shape = self.shape();
            let storage = self.storage().to_dtype(self.layout(), dtype)?;
            let op = BackpropOp::new1(self, Op::ToDType);
            Ok(from_storage_op(storage, shape.clone(), op))
        }
    }

    /// Returns a tensor that is in row major order. This is the same as the original tensor if it
    /// was already contiguous, otherwise a copy is triggered.
    pub fn contiguous(&self) -> Result<Self> {
        if self.is_contiguous() {
            Ok(self.clone())
        } else {
            let shape = self.shape();
            let mut storage = self.device().zeros(shape, self.dtype())?;
            self.storage()
                .copy_strided_src(&mut storage, 0, self.layout())?;
            let op = BackpropOp::new1(self, Op::Copy);
            Ok(from_storage_op(storage, shape.clone(), op))
        }
    }

    /// Create a variable based on the values currently stored in a tensor. The storage is always
    /// copied.
    pub(crate) fn make_var(&self) -> Result<Self> {
        let shape = self.shape().clone();
        let mut storage = self.device().zeros(&shape, self.dtype())?;
        self.storage()
            .copy_strided_src(&mut storage, 0, self.layout())?;
        Ok(from_storage(storage, shape, None, true))
    }

    // TODO: Do we want to allow target shape using -1 on some dimensions?
    /// Reshape returns a tensor with the target shape provided that the number of elements of the
    /// original tensor is the same.
    /// If the input tensor is contiguous, this is a view on the original data. Otherwise this uses
    /// a new storage and copies the data over, the returned tensor is always contiguous.
    ///
    /// ```rust
    /// # use candle_core::{Tensor, DType, Device, D};
    /// let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    ///
    /// let c = a.reshape((1, 6))?;
    /// assert_eq!(c.shape().dims(), &[1, 6]);
    ///
    /// let c = a.reshape((3, 2))?;
    /// assert_eq!(c.shape().dims(), &[3, 2]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn reshape<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let shape = shape.into();
        if shape.elem_count() != self.elem_count() {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: shape,
                op: "reshape",
            }
            .bt());
        }
        let op = BackpropOp::new1(self, Op::Reshape);
        if self.is_contiguous() {
            Ok(Tensor_ {
                id: TensorId::new(),
                storage: self.storage.clone(),
                layout: Layout::contiguous_with_offset(shape, self.layout.start_offset()),
                op,
                is_variable: false,
                dtype: self.dtype,
                device: self.device.clone(),
            }.into())
        } else {
            let mut storage = self.device().zeros(&shape, self.dtype())?;
            self.storage()
                .copy_strided_src(&mut storage, 0, self.layout())?;
            Ok(from_storage_op(storage, shape, op))
        }
    }

    /// Creates a new tensor with the specified dimension removed if its size was one.
    ///
    /// ```rust
    /// # use candle_core::{Tensor, DType, Device, D};
    /// let a = Tensor::zeros((2, 3, 1), DType::F32, &Device::Cpu)?;
    ///
    /// let c = a.squeeze(2)?;
    /// assert_eq!(c.shape().dims(), &[2, 3]);
    ///
    /// let c = a.squeeze(D::Minus1)?;
    /// assert_eq!(c.shape().dims(), &[2, 3]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn squeeze<D: Dim>(&self, dim: D) -> Result<Self> {
        // The PyTorch semantics are to return the same tensor if the target dimension
        // does not have a size of 1.
        let dims = self.dims();
        let dim = dim.to_index(self.shape(), "squeeze")?;
        if dims[dim] == 1 {
            let mut dims = dims.to_vec();
            dims.remove(dim);
            self.reshape(dims)
        } else {
            Ok(self.clone())
        }
    }

    /// Creates a new tensor with a dimension of size one inserted at the specified position.
    ///
    /// ```rust
    /// # use candle_core::{Tensor, DType, Device, D};
    /// let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    ///
    /// let c = a.unsqueeze(0)?;
    /// assert_eq!(c.shape().dims(), &[1, 2, 3]);
    ///
    /// let c = a.unsqueeze(D::Minus1)?;
    /// assert_eq!(c.shape().dims(), &[2, 3, 1]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn unsqueeze<D: Dim>(&self, dim: D) -> Result<Self> {
        let mut dims = self.dims().to_vec();
        let dim = dim.to_index_plus_one(self.shape(), "unsqueeze")?;
        // Cannot panic because to_index_plus_one already checks dimensions
        dims.insert(dim, 1);
        self.reshape(dims)
    }

    /// Stacks two or more tensors along a particular dimension.
    ///
    /// All tensors must have the same rank, and the output has one additional rank
    ///
    /// ```rust
    /// # use candle_core::{Tensor, DType, Device};
    /// let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    /// let b = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    ///
    /// let c = Tensor::stack(&[&a, &b], 0)?;
    /// assert_eq!(c.shape().dims(), &[2, 2, 3]);
    ///
    /// let c = Tensor::stack(&[&a, &b], 2)?;
    /// assert_eq!(c.shape().dims(), &[2, 3, 2]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn stack<A: AsRef<Self>, D: Dim>(args: &[A], dim: D) -> Result<Self> {
        if args.is_empty() {
            Err(Error::OpRequiresAtLeastOneTensor { op: "stack" }.bt())?
        }
        let dim = dim.to_index_plus_one(args[0].as_ref().shape(), "stack")?;
        let args = args
            .iter()
            .map(|t| t.as_ref().unsqueeze(dim))
            .collect::<Result<Vec<_>>>()?;
        Self::cat(&args, dim)
    }

    /// Concatenates two or more tensors along a particular dimension.
    ///
    /// All tensors must of the same rank, and the output will have
    /// the same rank
    ///
    /// ```rust
    /// # use candle_core::{Tensor, DType, Device};
    /// let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    /// let b = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    ///
    /// let c = Tensor::cat(&[&a, &b], 0)?;
    /// assert_eq!(c.shape().dims(), &[4, 3]);
    ///
    /// let c = Tensor::cat(&[&a, &b], 1)?;
    /// assert_eq!(c.shape().dims(), &[2, 6]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn cat<A: AsRef<Self>, D: Dim>(args: &[A], dim: D) -> Result<Self> {
        if args.is_empty() {
            Err(Error::OpRequiresAtLeastOneTensor { op: "cat" }.bt())?
        }
        let arg0 = args[0].as_ref();
        if args.len() == 1 {
            return Ok(arg0.clone());
        }
        let dim = dim.to_index(arg0.shape(), "cat")?;
        for arg in args {
            arg.as_ref().check_dim(dim, "cat")?;
        }
        if dim == 0 {
            Self::cat0(args)
        } else {
            // TODO: Avoid these transpositions and have an implementation that works
            // for dim != 0...
            let args: Vec<Self> = args
                .iter()
                .map(|a| a.as_ref().transpose(0, dim))
                .collect::<Result<Vec<_>>>()?;
            let cat = Self::cat0(&args)?;
            cat.transpose(0, dim)
        }
    }

    fn cat0<A: AsRef<Self>>(args: &[A]) -> Result<Self> {
        if args.is_empty() {
            Err(Error::OpRequiresAtLeastOneTensor { op: "cat" }.bt())?
        }
        let arg0 = args[0].as_ref();
        if args.len() == 1 {
            return Ok(arg0.clone());
        }
        let rank = arg0.rank();
        let device = arg0.device();
        let dtype = arg0.dtype();
        let first_dims = arg0.shape().dims();
        let mut cat_dims = first_dims.to_vec();
        cat_dims[0] = 0;
        let mut offsets = vec![0usize];
        for (arg_idx, arg) in args.iter().enumerate() {
            let arg = arg.as_ref();
            if arg.dtype() != dtype {
                Err(Error::DTypeMismatchBinaryOp {
                    lhs: dtype,
                    rhs: arg.dtype(),
                    op: "cat",
                }
                .bt())?
            }
            if arg.device().location() != device.location() {
                Err(Error::DeviceMismatchBinaryOp {
                    lhs: device.location(),
                    rhs: arg.device().location(),
                    op: "cat",
                }
                .bt())?
            }
            if rank != arg.rank() {
                Err(Error::UnexpectedNumberOfDims {
                    expected: rank,
                    got: arg.rank(),
                    shape: arg.shape().clone(),
                }
                .bt())?
            }
            for (dim_idx, (v1, v2)) in arg0
                .shape()
                .dims()
                .iter()
                .zip(arg.shape().dims().iter())
                .enumerate()
            {
                if dim_idx == 0 {
                    cat_dims[0] += v2;
                }
                if dim_idx != 0 && v1 != v2 {
                    Err(Error::ShapeMismatchCat {
                        dim: dim_idx,
                        first_shape: arg0.shape().clone(),
                        n: arg_idx + 1,
                        nth_shape: arg.shape().clone(),
                    }
                    .bt())?
                }
            }
            let next_offset = offsets.last().unwrap() + arg.elem_count();
            offsets.push(next_offset);
        }
        let shape = Shape::from(cat_dims);
        let op = BackpropOp::new(args, |args| Op::Cat(args, 0));
        let mut storage = device.zeros(&shape, dtype)?;
        for (arg, &offset) in args.iter().zip(offsets.iter()) {
            let arg = arg.as_ref();
            arg.storage()
                .copy_strided_src(&mut storage, offset, arg.layout())?;
        }
        Ok(from_storage_op(storage, shape, op))
    }

    pub fn pad_with_zeros<D: Dim>(&self, dim: D, left: usize, right: usize) -> Result<Self> {
        if left == 0 && right == 0 {
            Ok(self.clone())
        } else if left == 0 {
            let dim = dim.to_index(self.shape(), "pad_with_zeros")?;
            let mut dims = self.dims().to_vec();
            dims[dim] = right;
            let right = Self::zeros(dims.as_slice(), self.dtype, self.device())?;
            Self::cat(&[self, &right], dim)
        } else if right == 0 {
            let dim = dim.to_index(self.shape(), "pad_with_zeros")?;
            let mut dims = self.dims().to_vec();
            dims[dim] = left;
            let left = Self::zeros(dims.as_slice(), self.dtype, self.device())?;
            Self::cat(&[&left, self], dim)
        } else {
            let dim = dim.to_index(self.shape(), "pad_with_zeros")?;
            let mut dims = self.dims().to_vec();
            dims[dim] = left;
            let left = Self::zeros(dims.as_slice(), self.dtype, self.device())?;
            dims[dim] = right;
            let right = Self::zeros(dims.as_slice(), self.dtype, self.device())?;
            Self::cat(&[&left, self, &right], dim)
        }
    }

    fn storage(&self) -> std::sync::RwLockReadGuard<'_, Storage> {
        self.storage.read().unwrap()
    }

    // If we extend the visibility of this function to be usable outside of this crate, we should
    // make it unsafe.
    pub(crate) fn storage_mut_and_layout(
        &self,
    ) -> (std::sync::RwLockWriteGuard<'_, Storage>, &Layout) {
        let storage = self.storage.write().unwrap();
        (storage, &self.layout)
    }

    /// The storage used by this tensor, together with the layout to use to access it safely.
    pub fn storage_and_layout(&self) -> (std::sync::RwLockReadGuard<'_, Storage>, &Layout) {
        let storage = self.storage.read().unwrap();
        (storage, &self.layout)
    }

    pub(crate) fn same_storage(&self, rhs: &Self) -> bool {
        let lhs: &RwLock<Storage> = self.storage.as_ref();
        let rhs: &RwLock<Storage> = rhs.storage.as_ref();
        std::ptr::eq(lhs, rhs)
    }

    /// Applies a unary custom op.
    pub fn custom_op1_arc(&self, c: Arc<Box<dyn CustomOp1>>) -> Result<Self> {
        let (storage, shape) = self
            .storage()
            .custom_op1(self.layout(), c.as_ref().as_ref())?;
        let op = BackpropOp::new1(self, |s| Op::CustomOp1(s, c.clone()));
        Ok(from_storage_op(storage, shape, op))
    }

    pub fn custom_op1<C: 'static + CustomOp1>(&self, c: C) -> Result<Self> {
        self.custom_op1_arc(Arc::new(Box::new(c)))
    }

    /// Applies a binary custom op.
    pub fn custom_op2_arc(&self, rhs: &Self, c: Arc<Box<dyn CustomOp2>>) -> Result<Self> {
        let (storage, shape) = self.storage().custom_op2(
            self.layout(),
            &rhs.storage(),
            rhs.layout(),
            c.as_ref().as_ref(),
        )?;
        let op = BackpropOp::new2(self, rhs, |t1, t2| Op::CustomOp2(t1, t2, c.clone()));
        Ok(from_storage_op(storage, shape, op))
    }

    pub fn custom_op2<C: 'static + CustomOp2>(&self, r: &Self, c: C) -> Result<Self> {
        self.custom_op2_arc(r, Arc::new(Box::new(c)))
    }

    /// Applies a ternary custom op.
    pub fn custom_op3_arc(&self, t2: &Self, t3: &Self, c: Arc<Box<dyn CustomOp3>>) -> Result<Self> {
        let (storage, shape) = self.storage().custom_op3(
            self.layout(),
            &t2.storage(),
            t2.layout(),
            &t3.storage(),
            t3.layout(),
            c.as_ref().as_ref(),
        )?;
        let op = BackpropOp::new3(self, t2, t3, |t1, t2, t3| {
            Op::CustomOp3(t1, t2, t3, c.clone())
        });
        Ok(from_storage_op(storage, shape, op))
    }

    pub fn custom_op3<C: 'static + CustomOp3>(&self, t2: &Self, t3: &Self, c: C) -> Result<Self> {
        self.custom_op3_arc(t2, t3, Arc::new(Box::new(c)))
    }
}

macro_rules! bin_trait {
    ($trait:ident, $fn1:ident, $mul:expr, $add:expr) => {
        impl<B: std::borrow::Borrow<Self>> std::ops::$trait<B> for Tensor {
            type Output = Result<Self>;

            fn $fn1(self, rhs: B) -> Self::Output {
                Self::$fn1(&self, rhs.borrow())
            }
        }

        impl<B: std::borrow::Borrow<Tensor>> std::ops::$trait<B> for &Tensor {
            type Output = Result<Tensor>;

            fn $fn1(self, rhs: B) -> Self::Output {
                Tensor::$fn1(&self, rhs.borrow())
            }
        }

        impl<B: std::borrow::Borrow<Self>> std::ops::$trait<Result<B>> for Tensor {
            type Output = Result<Self>;

            fn $fn1(self, rhs: Result<B>) -> Self::Output {
                Self::$fn1(&self, rhs?.borrow())
            }
        }

        impl<B: std::borrow::Borrow<Tensor>> std::ops::$trait<Result<B>> for &Tensor {
            type Output = Result<Tensor>;

            fn $fn1(self, rhs: Result<B>) -> Self::Output {
                Tensor::$fn1(&self, rhs?.borrow())
            }
        }

        impl std::ops::$trait<f64> for Tensor {
            type Output = Result<Self>;

            fn $fn1(self, rhs: f64) -> Self::Output {
                self.affine($mul(rhs), $add(rhs))
            }
        }

        impl std::ops::$trait<f64> for &Tensor {
            type Output = Result<Tensor>;

            fn $fn1(self, rhs: f64) -> Self::Output {
                self.affine($mul(rhs), $add(rhs))
            }
        }
    };
}

bin_trait!(Add, add, |_| 1., |v| v);
bin_trait!(Sub, sub, |_| 1., |v: f64| -v);
bin_trait!(Mul, mul, |v| v, |_| 0.);
bin_trait!(Div, div, |v| 1. / v, |_| 0.);
