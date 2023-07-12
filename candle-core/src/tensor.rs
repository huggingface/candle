use crate::backend::{BackendDevice, BackendStorage};
use crate::shape::Dim;
use crate::{op::Op, storage::Storage, DType, Device, Error, Layout, Result, Shape};
use std::sync::Arc;

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
    storage: Arc<Storage>,
    layout: Layout,
    op: Option<Op>,
    is_variable: bool,
}

impl AsRef<Tensor> for Tensor {
    fn as_ref(&self) -> &Tensor {
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
/// use candle::{Tensor, DType, Device};
///
/// let a = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
/// let b = Tensor::arange(0f32, 12f32, &Device::Cpu)?.reshape((3, 4))?;
///
/// let c = a.matmul(&b)?;
/// # Ok::<(), candle::Error>(())
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
                .storage
                .unary_impl::<crate::op::$op_name>(self.layout())?;
            let op = if self.track_op() {
                Some(Op::$op_name(self.clone()))
            } else {
                None
            };
            Ok(from_storage(storage, shape.clone(), op, false))
        }
    };
}

macro_rules! binary_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self, rhs: &Self) -> Result<Self> {
            let shape = self.same_shape_binary_op(rhs, stringify!($fn_name))?;
            let storage = self.storage.binary_impl::<crate::op::$op_name>(
                &rhs.storage,
                self.layout(),
                rhs.layout(),
            )?;
            let op = if self.track_op() || rhs.track_op() {
                Some(Op::$op_name(self.clone(), rhs.clone()))
            } else {
                None
            };
            Ok(from_storage(storage, shape.clone(), op, false))
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
    op: Option<Op>,
    is_variable: bool,
) -> Tensor {
    let tensor_ = Tensor_ {
        id: TensorId::new(),
        storage: Arc::new(storage),
        layout: Layout::contiguous(shape),
        op,
        is_variable,
    };
    Tensor(Arc::new(tensor_))
}

impl Tensor {
    fn ones_impl<S: Into<Shape>>(
        shape: S,
        dtype: DType,
        device: &Device,
        is_variable: bool,
    ) -> Result<Self> {
        let storage = device.ones(&crate::shape::SCALAR, dtype)?;
        from_storage(storage, crate::shape::SCALAR, None, is_variable).broadcast_as(shape)
    }

    /// Creates a new tensor filled with ones.
    ///
    /// ```rust
    /// use candle::{Tensor, DType, Device};
    /// let a = Tensor::ones((2, 3), DType::F32, &Device::Cpu)?;
    /// let b = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0], (2, 3), &Device::Cpu)?;
    /// // a == b
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn ones<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        Self::ones_impl(shape, dtype, device, false)
    }

    pub fn ones_var<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        // Maybe we should allocate some actual storage for vars rather than just using a
        // broadcasted scalar?
        Self::ones_impl(shape, dtype, device, true)
    }

    /// Creates a new tensor filled with ones with same shape, dtype, and device as the other tensor.
    ///
    /// ```rust
    /// use candle::{Tensor, DType, Device};
    /// let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    /// let b = a.ones_like()?;
    /// // b == a + 1
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn ones_like(&self) -> Result<Self> {
        Tensor::ones(self.shape(), self.dtype(), &self.device())
    }

    /// Creates a new tensor filled with zeros.
    ///
    /// ```rust
    /// use candle::{Tensor, DType, Device};
    /// let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    /// let b = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0], (2, 3), &Device::Cpu)?;
    /// // a == b
    /// # Ok::<(), candle::Error>(())
    /// ```
    fn zeros_impl<S: Into<Shape>>(
        shape: S,
        dtype: DType,
        device: &Device,
        is_variable: bool,
    ) -> Result<Self> {
        let storage = device.zeros(&crate::shape::SCALAR, dtype)?;
        from_storage(storage, crate::shape::SCALAR, None, is_variable).broadcast_as(shape)
    }

    /// Creates a new tensor filled with zeros.
    ///
    /// ```rust
    /// use candle::{Tensor, DType, Device};
    /// let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    /// let b = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0], (2, 3), &Device::Cpu)?;
    /// // a == b
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn zeros<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        Self::zeros_impl(shape, dtype, device, false)
    }

    pub fn zeros_var<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        Self::zeros_impl(shape, dtype, device, true)
    }

    /// Creates a new tensor filled with ones with same shape, dtype, and device as the other
    /// tensor.
    ///
    /// ```rust
    /// use candle::{Tensor, DType, Device};
    /// let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    /// let b = a.zeros_like()?;
    /// // b is on CPU f32.
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn zeros_like(&self) -> Result<Self> {
        Tensor::zeros(self.shape(), self.dtype(), &self.device())
    }

    fn rand_impl<S: Into<Shape>>(
        s: S,
        dtype: DType,
        device: &Device,
        lo: f64,
        up: f64,
        is_variable: bool,
    ) -> Result<Self> {
        let s = s.into();
        let storage = device.rand_uniform(&s, dtype, lo, up)?;
        Ok(from_storage(storage, s, None, is_variable))
    }

    /// Creates a new tensor initialized with values sampled uniformly between `lo` and `up`.
    pub fn rand<S: Into<Shape>>(
        s: S,
        dtype: DType,
        device: &Device,
        lo: f64,
        up: f64,
    ) -> Result<Self> {
        Self::rand_impl(s, dtype, device, lo, up, false)
    }

    pub fn rand_var<S: Into<Shape>>(
        s: S,
        dtype: DType,
        device: &Device,
        lo: f64,
        up: f64,
    ) -> Result<Self> {
        Self::rand_impl(s, dtype, device, lo, up, true)
    }

    fn randn_impl<S: Into<Shape>>(
        s: S,
        dtype: DType,
        device: &Device,
        mean: f64,
        std: f64,
        is_variable: bool,
    ) -> Result<Self> {
        let s = s.into();
        let storage = device.rand_normal(&s, dtype, mean, std)?;
        Ok(from_storage(storage, s, None, is_variable))
    }

    /// Creates a new tensor initialized with values sampled from a normal distribution with the
    /// specified `mean` and standard deviation `std`.
    pub fn randn<S: Into<Shape>>(
        s: S,
        dtype: DType,
        device: &Device,
        mean: f64,
        std: f64,
    ) -> Result<Self> {
        Self::randn_impl(s, dtype, device, mean, std, false)
    }

    pub fn randn_var<S: Into<Shape>>(
        s: S,
        dtype: DType,
        device: &Device,
        mean: f64,
        std: f64,
    ) -> Result<Self> {
        Self::randn_impl(s, dtype, device, mean, std, true)
    }

    pub fn new_impl<A: crate::device::NdArray>(
        array: A,
        shape: Shape,
        device: &Device,
        is_variable: bool,
    ) -> Result<Self> {
        let n: usize = shape.elem_count();
        let buffer_size: usize = array.shape()?.elem_count();
        if buffer_size != n {
            return Err(Error::ShapeMismatch { buffer_size, shape });
        }
        let storage = device.storage(array)?;
        Ok(from_storage(storage, shape, None, is_variable))
    }

    /// Creates a new tensor on the specified device using the content and shape of the input.
    pub fn new<A: crate::device::NdArray>(array: A, device: &Device) -> Result<Self> {
        let shape = array.shape()?;
        Self::new_impl(array, shape, device, false)
    }

    /// Creates a new tensor on the specified device using the content and shape of the input.
    /// This is similar to `new` but the resulting tensor is a variable.
    pub fn var<A: crate::device::NdArray>(array: A, device: &Device) -> Result<Self> {
        let shape = array.shape()?;
        Self::new_impl(array, shape, device, true)
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

    fn from_vec_impl<S: Into<Shape>, D: crate::WithDType>(
        data: Vec<D>,
        shape: S,
        device: &Device,
        is_variable: bool,
    ) -> Result<Self> {
        let shape = shape.into();
        let buffer_size = data.len();
        if buffer_size != shape.elem_count() {
            return Err(Error::ShapeMismatch { buffer_size, shape });
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

    pub fn var_from_vec<S: Into<Shape>, D: crate::WithDType>(
        data: Vec<D>,
        shape: S,
        device: &Device,
    ) -> Result<Self> {
        Self::from_vec_impl(data, shape, device, true)
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

    pub fn var_from_slice<S: Into<Shape>, D: crate::WithDType>(
        array: &[D],
        shape: S,
        device: &Device,
    ) -> Result<Self> {
        Self::new_impl(array, shape.into(), device, true)
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
                })?
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
            })
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
            return Err(Error::UnexpectedNumberOfDims {
                expected: 0,
                got: self.rank(),
                shape: self.shape().clone(),
            });
        }
        let from_cpu_storage = |cpu_storage: &crate::CpuStorage| {
            let data = S::cpu_storage_as_slice(cpu_storage)?;
            Ok::<_, Error>(data[self.layout().start_offset()])
        };
        match self.storage.as_ref() {
            Storage::Cpu(cpu_storage) => from_cpu_storage(cpu_storage),
            Storage::Cuda(storage) => from_cpu_storage(&storage.to_cpu_storage()?),
        }
    }

    /// This operation multiplies the input tensor by `mul` then adds `add` and return the result.
    /// The input values `mul` and `add` are casted to the appropriate type so some rounding might
    /// be performed.
    ///
    /// ```rust
    /// use candle::{Tensor, Device};
    /// let a = Tensor::new(&[[0f32, 1.], [2., 3.]], &Device::Cpu)?;
    /// let a = a.affine(4., -2.)?;
    /// assert_eq!(a.to_vec2::<f32>()?, &[[-2.0, 2.0], [6.0, 10.0]]);
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn affine(&self, mul: f64, add: f64) -> Result<Self> {
        let storage = self.storage.affine(self.layout(), mul, add)?;
        let op = if self.track_op() {
            Some(Op::Affine {
                arg: self.clone(),
                mul,
                add,
            })
        } else {
            None
        };
        Ok(from_storage(storage, self.shape(), op, false))
    }

    /// Applies the Exponential Linear Unit (ELU) function on each element of the input tensor.
    pub fn elu(&self, alpha: f64) -> Result<Self> {
        let storage = self.storage.elu(self.layout(), alpha)?;
        let op = if self.track_op() {
            Some(Op::Elu(self.clone(), alpha))
        } else {
            None
        };
        Ok(from_storage(storage, self.shape(), op, false))
    }

    fn check_dim(&self, dim: usize, op: &'static str) -> Result<()> {
        if dim >= self.dims().len() {
            Err(Error::DimOutOfRange {
                shape: self.shape().clone(),
                dim: dim as i32,
                op,
            })?
        } else {
            Ok(())
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
            })?
        }
        if start == 0 && dims[dim] == len {
            Ok(self.clone())
        } else {
            let op = if self.track_op() {
                Some(Op::Narrow(self.clone(), dim, start, len))
            } else {
                None
            };
            let layout = self.layout().narrow(dim, start, len)?;
            let tensor_ = Tensor_ {
                id: TensorId::new(),
                storage: self.storage.clone(),
                layout,
                op,
                is_variable: false,
            };
            Ok(Tensor(Arc::new(tensor_)))
        }
    }

    /// Applies the softmax function to the input tensor, rescaling the element so that elements on
    /// a slice of fixed index on dimension `dim` are between 0 and 1 and sum to 1.
    ///
    /// ```rust
    /// use candle::{Tensor, Device};
    /// let a = Tensor::new(&[[0f32, 1., 0., 1.], [-2., 2., 3., -3.]], &Device::Cpu)?;
    /// let a = a.softmax(1)?;
    /// assert_eq!(
    ///     a.to_vec2::<f32>()?,
    ///     &[
    ///         [0.13447072, 0.3655293, 0.13447072, 0.3655293],
    ///         [0.004892866, 0.26714143, 0.7261657, 0.0017999847],
    ///     ]);
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn softmax<D: Dim>(&self, dim: D) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "softmax")?;
        // TODO: unify the two branches.
        if self.device().is_cuda() {
            // We do not have a cuda kernel for divide_by_sum_over_dim so split
            // the operation.
            let exp = self.exp()?;
            let sum_exp = exp.sum(&[dim])?;
            exp.broadcast_div(&sum_exp)
        } else {
            let shape = self.shape();
            let mut storage = self.storage.unary_impl::<crate::op::Exp>(self.layout())?;
            // The resulting storage is contiguous.
            storage.divide_by_sum_over_dim(shape, dim)?;
            let op = if self.track_op() {
                Some(Op::Softmax(self.clone(), dim))
            } else {
                None
            };
            Ok(from_storage(storage, shape.clone(), op, false))
        }
    }

    /// Returns the sum of all elements in the input tensor. The sum is performed over all the
    /// input dimensions.
    ///
    /// The resulting tensor as a shape that is similar to the shape of the input tensor, except
    /// that the number of elements for each dimension index in `sum_dims` is 1.
    ///
    /// ```rust
    /// use candle::{Tensor, Device};
    /// let a = Tensor::new(&[[0f32, 1.], [2., 3.]], &Device::Cpu)?;
    /// let s = a.sum(&[0])?;
    /// assert_eq!(s.to_vec2::<f32>()?, &[[2., 4.]]);
    /// let s = a.sum(&[1])?;
    /// assert_eq!(s.to_vec2::<f32>()?, &[[1.], [5.]]);
    /// let s = a.sum(&[0, 1])?;
    /// assert_eq!(s.to_vec2::<f32>()?, &[[6.]]);
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn sum(&self, sum_dims: &[usize]) -> Result<Self> {
        for &dim in sum_dims {
            self.check_dim(dim, "sum")?;
        }
        let storage = self.storage.sum(self.layout(), sum_dims)?;
        let op = if self.track_op() {
            Some(Op::Sum(self.clone(), sum_dims.to_vec()))
        } else {
            None
        };
        let mut dims = self.dims().to_vec();
        for &sum_dim in sum_dims.iter() {
            dims[sum_dim] = 1
        }
        Ok(from_storage(storage, dims, op, false))
    }

    /// Applies a 1D convolution over the input tensor.
    pub fn conv1d(&self, kernel: &Self, padding: usize, stride: usize) -> Result<Self> {
        let (c_out, c_in_k, k_size) = kernel.shape().r3()?;
        let (b_size, c_in, l_in) = match *self.dims() {
            [b_size, c_in, l_in] => (Some(b_size), c_in, l_in),
            [c_in, l_in] => (None, c_in, l_in),
            _ => Err(Error::Conv1dInvalidArgs {
                inp_shape: self.shape().clone(),
                k_shape: kernel.shape().clone(),
                padding,
                stride,
                msg: "input rank is not 2 or 3",
            })?,
        };
        if c_in != c_in_k {
            Err(Error::Conv1dInvalidArgs {
                inp_shape: self.shape().clone(),
                k_shape: kernel.shape().clone(),
                padding,
                stride,
                msg: "the number of in-channels on the input doesn't match the kernel size",
            })?
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
            self.storage
                .conv1d(self.layout(), &kernel.storage, kernel.layout(), &params)?;
        let op = if self.track_op() || kernel.track_op() {
            Some(Op::Conv1D {
                arg: self.clone(),
                kernel: kernel.clone(),
                padding,
                stride,
            })
        } else {
            None
        };
        let out_dims = params.out_dims();
        Ok(from_storage(storage, out_dims, op, false))
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
            })?
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
            })?
        }

        let storage = self.storage.matmul(
            &rhs.storage,
            (batching, m, n, k),
            self.layout(),
            rhs.layout(),
        )?;
        let op = if self.track_op() || rhs.track_op() {
            Some(Op::Matmul(self.clone(), rhs.clone()))
        } else {
            None
        };
        Ok(from_storage(storage, c_shape, op, false))
    }

    /// Returns a tensor with the same shape as the input tensor, the values are taken from
    /// `on_true` if the input tensor value is not zero, and `on_false` at the positions where the
    /// input tensor is equal to zero.
    pub fn where_cond(&self, on_true: &Self, on_false: &Self) -> Result<Self> {
        let _shap = self.same_shape_binary_op(on_true, "where_cond")?;
        let shape = self.same_shape_binary_op(on_false, "where_cond")?;
        let storage = self.storage.where_cond(
            self.layout(),
            &on_true.storage,
            on_true.layout(),
            &on_false.storage,
            on_false.layout(),
        )?;
        let op = if self.track_op() || on_true.track_op() || on_false.track_op() {
            Some(Op::WhereCond(
                self.clone(),
                on_true.clone(),
                on_false.clone(),
            ))
        } else {
            None
        };
        Ok(from_storage(storage, shape, op, false))
    }

    pub fn embedding(ids: &Self, rhs: &Self) -> Result<Self> {
        if !rhs.is_contiguous() {
            return Err(Error::RequiresContiguous { op: "embedding" });
        } else if rhs.rank() != 2 || ids.rank() != 1 {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: ids.shape().clone(),
                rhs: rhs.shape().clone(),
                op: "embedding",
            });
        }
        let ids_shape = ids.shape();
        let seq_len = ids_shape.r1()?;
        let (_, hidden_size) = rhs.shape().r2()?;
        let storage = ids
            .storage
            .embedding(ids.layout(), &rhs.storage, rhs.layout())?;
        let shape: Shape = (seq_len, hidden_size).into();
        let op = if ids.track_op() || rhs.track_op() {
            Some(Op::Embedding(ids.clone(), rhs.clone()))
        } else {
            None
        };
        Ok(from_storage(storage, shape, op, false))
    }

    pub(crate) fn strided_index(&self) -> crate::StridedIndex {
        self.layout.strided_index()
    }

    /// Returns data from the underlying storage, this does not take the strides
    /// into account so the size of the resulting buffer might be larger than the
    /// tensor number of elements.
    pub fn storage_data<S: crate::WithDType>(&self) -> Result<std::borrow::Cow<[S]>> {
        match self.storage.as_ref() {
            Storage::Cpu(cpu_storage) => {
                let slice = S::cpu_storage_as_slice(cpu_storage)?;
                Ok(std::borrow::Cow::Borrowed(slice))
            }
            Storage::Cuda(slice) => {
                let cpu_storage = slice.to_cpu_storage()?;
                let storage_data = S::cpu_storage_data(cpu_storage)?;
                Ok(std::borrow::Cow::Owned(storage_data))
            }
        }
    }

    pub fn to_vec1<S: crate::WithDType>(&self) -> Result<Vec<S>> {
        if self.rank() != 1 {
            return Err(Error::UnexpectedNumberOfDims {
                expected: 1,
                got: self.rank(),
                shape: self.shape().clone(),
            });
        }
        match self.storage.as_ref() {
            Storage::Cpu(cpu_storage) => {
                let data = S::cpu_storage_as_slice(cpu_storage)?;
                Ok(self.strided_index().map(|i| data[i]).collect())
            }
            Storage::Cuda(slice) => {
                // TODO: Would it be possible to only fetch the necessary data?
                let cpu_storage = slice.to_cpu_storage()?;
                let data = S::cpu_storage_as_slice(&cpu_storage)?;
                Ok(self.strided_index().map(|i| data[i]).collect())
            }
        }
    }

    pub fn to_vec2<S: crate::WithDType>(&self) -> Result<Vec<Vec<S>>> {
        let (dim1, dim2) = self.shape().r2()?;
        let from_cpu_storage = |cpu_storage: &crate::CpuStorage| {
            let data = S::cpu_storage_as_slice(cpu_storage)?;
            let mut rows = vec![];
            let mut src_index = self.strided_index();
            for _idx_row in 0..dim1 {
                let row = (0..dim2).map(|_| data[src_index.next().unwrap()]).collect();
                rows.push(row)
            }
            assert!(src_index.next().is_none());
            Ok(rows)
        };
        match self.storage.as_ref() {
            Storage::Cpu(storage) => from_cpu_storage(storage),
            Storage::Cuda(storage) => from_cpu_storage(&storage.to_cpu_storage()?),
        }
    }

    pub fn to_vec3<S: crate::WithDType>(&self) -> Result<Vec<Vec<Vec<S>>>> {
        let (dim1, dim2, dim3) = self.shape().r3()?;
        let from_cpu_storage = |cpu_storage: &crate::CpuStorage| {
            let data = S::cpu_storage_as_slice(cpu_storage)?;
            let mut top_rows = vec![];
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
            Ok(top_rows)
        };
        match self.storage.as_ref() {
            Storage::Cpu(storage) => from_cpu_storage(storage),
            Storage::Cuda(storage) => from_cpu_storage(&storage.to_cpu_storage()?),
        }
    }

    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    pub fn device(&self) -> Device {
        self.storage.device()
    }

    pub fn shape(&self) -> &Shape {
        self.layout().shape()
    }

    pub fn dims(&self) -> &[usize] {
        self.shape().dims()
    }

    pub fn dim<D: Dim>(&self, dim: D) -> Result<usize> {
        let dim = dim.to_index(self.shape(), "dim")?;
        Ok(self.dims()[dim])
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn stride(&self) -> &[usize] {
        self.layout.stride()
    }

    pub fn rank(&self) -> usize {
        self.shape().rank()
    }

    pub fn elem_count(&self) -> usize {
        self.shape().elem_count()
    }

    pub fn id(&self) -> TensorId {
        self.id
    }

    pub fn is_variable(&self) -> bool {
        self.is_variable
    }

    pub(crate) fn op(&self) -> &Option<Op> {
        &self.op
    }

    pub fn sum_all(&self) -> Result<Tensor> {
        let dims: Vec<_> = (0..self.rank()).collect();
        self.sum(&dims)
    }

    fn flatten_<D1: Dim, D2: Dim>(
        &self,
        start_dim: Option<D1>,
        end_dim: Option<D2>,
    ) -> Result<Tensor> {
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

    pub fn flatten<D1: Dim, D2: Dim>(&self, start_dim: D1, end_dim: D2) -> Result<Tensor> {
        self.flatten_(Some(start_dim), Some(end_dim))
    }

    pub fn flatten_to<D: Dim>(&self, end_dim: D) -> Result<Tensor> {
        self.flatten_(None::<usize>, Some(end_dim))
    }

    pub fn flatten_from<D: Dim>(&self, start_dim: D) -> Result<Tensor> {
        self.flatten_(Some(start_dim), None::<usize>)
    }

    pub fn flatten_all(&self) -> Result<Tensor> {
        self.flatten_(None::<usize>, None::<usize>)
    }

    pub fn get(&self, i: usize) -> Result<Tensor> {
        let dims = self.dims();
        if dims.is_empty() {
            Ok(self.clone())
        } else {
            self.narrow(0, i, 1)?.reshape(&dims[1..])
        }
    }

    /// Returns a tensor that is a transposed version of the input, the two last dimensions of the
    /// input are swapped.
    pub fn t(&self) -> Result<Tensor> {
        let rank = self.rank();
        if rank < 2 {
            return Err(Error::UnexpectedNumberOfDims {
                expected: 2,
                got: rank,
                shape: self.shape().clone(),
            });
        }
        self.transpose(rank - 2, rank - 1)
    }

    /// Returns a tensor that is a transposed version of the input, the given dimensions are
    /// swapped.
    pub fn transpose<D1: Dim, D2: Dim>(&self, dim1: D1, dim2: D2) -> Result<Tensor> {
        let dim1 = dim1.to_index(self.shape(), "transpose")?;
        let dim2 = dim2.to_index(self.shape(), "transpose")?;
        let op = if self.track_op() {
            Some(Op::Transpose(self.clone(), dim1, dim2))
        } else {
            None
        };
        let tensor_ = Tensor_ {
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: self.layout.transpose(dim1, dim2)?,
            op,
            is_variable: false,
        };
        Ok(Tensor(Arc::new(tensor_)))
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
    pub fn copy(&self) -> Result<Tensor> {
        let tensor_ = Tensor_ {
            id: TensorId::new(),
            storage: Arc::new(self.storage.try_clone(self.layout())?),
            layout: self.layout.clone(),
            op: None, // TODO
            is_variable: false,
        };
        Ok(Tensor(Arc::new(tensor_)))
    }

    /// Returns a new tensor detached from the current graph, gradient are not propagated through
    /// this new node. The storage of this tensor is shared with the initial tensor.
    pub fn detach(&self) -> Result<Tensor> {
        let tensor_ = Tensor_ {
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: self.layout.clone(),
            op: None,
            is_variable: false,
        };
        Ok(Tensor(Arc::new(tensor_)))
    }

    /// If the target device is the same as the tensor device, only a shallow copy is performed.
    pub fn to_device(&self, device: &Device) -> Result<Tensor> {
        if self.device().same_device(device) {
            Ok(self.clone())
        } else {
            let storage = match (self.storage.as_ref(), device) {
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
            let op = if self.track_op() {
                Some(Op::ToDevice(self.clone()))
            } else {
                None
            };
            let tensor_ = Tensor_ {
                id: TensorId::new(),
                storage: Arc::new(storage),
                layout: self.layout.clone(),
                op,
                is_variable: false,
            };
            Ok(Tensor(Arc::new(tensor_)))
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
    pub fn broadcast_as<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let op = if self.track_op() {
            Some(Op::Broadcast(self.clone()))
        } else {
            None
        };
        let tensor_ = Tensor_ {
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: self.layout.broadcast_as(shape)?,
            op,
            is_variable: false,
        };
        Ok(Tensor(Arc::new(tensor_)))
    }

    /// An alias for broadcast_as.
    pub fn expand<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        self.broadcast_as(shape)
    }

    pub fn to_dtype(&self, dtype: DType) -> Result<Self> {
        if self.dtype() == dtype {
            Ok(self.clone())
        } else {
            let shape = self.shape();
            let storage = self.storage.to_dtype(self.layout(), dtype)?;
            let op = if self.track_op() {
                Some(Op::ToDType(self.clone()))
            } else {
                None
            };
            Ok(from_storage(storage, shape.clone(), op, false))
        }
    }

    /// Returns a tensor that is in row major order. This is the same as the original tensor if it
    /// was already contiguous, otherwise a copy is triggered.
    pub fn contiguous(&self) -> Result<Tensor> {
        if self.is_contiguous() {
            Ok(self.clone())
        } else {
            let shape = self.shape();
            let mut storage = self.device().zeros(shape, self.dtype())?;
            self.storage
                .copy_strided_src(&mut storage, 0, self.layout())?;
            Ok(from_storage(
                storage,
                shape.clone(),
                None, // TODO
                false,
            ))
        }
    }

    // TODO: Do we want to allow target shape using -1 on some dimensions?
    /// Reshape returns a tensor with the target shape provided that the number of elements of the
    /// original tensor is the same.
    /// If the input tensor is contiguous, this is a view on the original data. Otherwise this uses
    /// a new storage and copies the data over, the returned tensor is always contiguous.
    ///
    /// ```rust
    /// # use candle::{Tensor, DType, Device, D};
    /// let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    ///
    /// let c = a.reshape((1, 6))?;
    /// assert_eq!(c.shape().dims(), &[1, 6]);
    ///
    /// let c = a.reshape((3, 2))?;
    /// assert_eq!(c.shape().dims(), &[3, 2]);
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn reshape<S: Into<Shape>>(&self, shape: S) -> Result<Tensor> {
        let shape = shape.into();
        if shape.elem_count() != self.elem_count() {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: shape,
                op: "reshape",
            });
        }
        let op = if self.track_op() {
            Some(Op::Reshape(self.clone()))
        } else {
            None
        };
        if self.is_contiguous() {
            let tensor_ = Tensor_ {
                id: TensorId::new(),
                storage: self.storage.clone(),
                layout: Layout::contiguous_with_offset(shape, self.layout.start_offset()),
                op,
                is_variable: false,
            };
            Ok(Tensor(Arc::new(tensor_)))
        } else {
            let mut storage = self.device().zeros(&shape, self.dtype())?;
            self.storage
                .copy_strided_src(&mut storage, 0, self.layout())?;
            Ok(from_storage(storage, shape, op, false))
        }
    }

    /// Creates a new tensor with the specified dimension removed if its size was one.
    ///
    /// ```rust
    /// # use candle::{Tensor, DType, Device, D};
    /// let a = Tensor::zeros((2, 3, 1), DType::F32, &Device::Cpu)?;
    ///
    /// let c = a.squeeze(2)?;
    /// assert_eq!(c.shape().dims(), &[2, 3]);
    ///
    /// let c = a.squeeze(D::Minus1)?;
    /// assert_eq!(c.shape().dims(), &[2, 3]);
    /// # Ok::<(), candle::Error>(())
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
    /// # use candle::{Tensor, DType, Device, D};
    /// let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    ///
    /// let c = a.unsqueeze(0)?;
    /// assert_eq!(c.shape().dims(), &[1, 2, 3]);
    ///
    /// let c = a.unsqueeze(D::Minus1)?;
    /// assert_eq!(c.shape().dims(), &[2, 3, 1]);
    /// # Ok::<(), candle::Error>(())
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
    /// All tensors must have the same rank, and the output has
    /// 1 additional rank
    ///
    /// ```rust
    /// # use candle::{Tensor, DType, Device};
    /// let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    /// let b = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    ///
    /// let c = Tensor::stack(&[&a, &b], 0)?;
    /// assert_eq!(c.shape().dims(), &[2, 2, 3]);
    ///
    /// let c = Tensor::stack(&[&a, &b], 2)?;
    /// assert_eq!(c.shape().dims(), &[2, 3, 2]);
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn stack<A: AsRef<Tensor>, D: Dim>(args: &[A], dim: D) -> Result<Self> {
        if args.is_empty() {
            return Err(Error::OpRequiresAtLeastOneTensor { op: "stack" });
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
    /// # use candle::{Tensor, DType, Device};
    /// let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    /// let b = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    ///
    /// let c = Tensor::cat(&[&a, &b], 0)?;
    /// assert_eq!(c.shape().dims(), &[4, 3]);
    ///
    /// let c = Tensor::cat(&[&a, &b], 1)?;
    /// assert_eq!(c.shape().dims(), &[2, 6]);
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn cat<A: AsRef<Tensor>, D: Dim>(args: &[A], dim: D) -> Result<Self> {
        if args.is_empty() {
            return Err(Error::OpRequiresAtLeastOneTensor { op: "cat" });
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
            let args: Vec<Tensor> = args
                .iter()
                .map(|a| a.as_ref().transpose(0, dim))
                .collect::<Result<Vec<_>>>()?;
            let cat = Self::cat0(&args)?;
            cat.transpose(0, dim)
        }
    }

    fn cat0<A: AsRef<Tensor>>(args: &[A]) -> Result<Self> {
        if args.is_empty() {
            return Err(Error::OpRequiresAtLeastOneTensor { op: "cat" });
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
                // TODO: Improve the error message.
                return Err(Error::DTypeMismatchBinaryOp {
                    lhs: dtype,
                    rhs: arg.dtype(),
                    op: "cat",
                });
            }
            if arg.device().location() != device.location() {
                // TODO: Improve the error message.
                return Err(Error::DeviceMismatchBinaryOp {
                    lhs: device.location(),
                    rhs: arg.device().location(),
                    op: "cat",
                });
            }
            let mut mismatch = arg.rank() != rank;
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
                    // TODO: It would probably be good to have a nicer error message here, i.e.
                    // mention the problematic dimension and the values.
                    mismatch = true;
                }
            }
            if mismatch {
                return Err(Error::ShapeMismatchCat {
                    dim: 0, // TODO: not the appropriate error message
                    first_shape: arg0.shape().clone(),
                    n: arg_idx + 1,
                    nth_shape: arg.shape().clone(),
                });
            }
            let next_offset = offsets.last().unwrap() + arg.elem_count();
            offsets.push(next_offset);
        }
        let shape = Shape::from(cat_dims);
        let op = if args.iter().any(|arg| arg.as_ref().track_op()) {
            let args: Vec<Tensor> = args.iter().map(|arg| arg.as_ref().clone()).collect();
            Some(Op::Cat(args, 0))
        } else {
            None
        };
        let mut storage = device.zeros(&shape, dtype)?;
        for (arg, &offset) in args.iter().zip(offsets.iter()) {
            let arg = arg.as_ref();
            arg.storage
                .copy_strided_src(&mut storage, offset, arg.layout())?;
        }
        Ok(from_storage(storage, shape, op, false))
    }
}

macro_rules! bin_trait {
    ($trait:ident, $fn1:ident, $mul:expr, $add:expr) => {
        impl<B: std::borrow::Borrow<Tensor>> std::ops::$trait<B> for Tensor {
            type Output = Result<Tensor>;

            fn $fn1(self, rhs: B) -> Self::Output {
                Tensor::$fn1(&self, rhs.borrow())
            }
        }

        impl<B: std::borrow::Borrow<Tensor>> std::ops::$trait<B> for &Tensor {
            type Output = Result<Tensor>;

            fn $fn1(self, rhs: B) -> Self::Output {
                Tensor::$fn1(&self, rhs.borrow())
            }
        }

        impl<B: std::borrow::Borrow<Tensor>> std::ops::$trait<Result<B>> for Tensor {
            type Output = Result<Tensor>;

            fn $fn1(self, rhs: Result<B>) -> Self::Output {
                Tensor::$fn1(&self, rhs?.borrow())
            }
        }

        impl<B: std::borrow::Borrow<Tensor>> std::ops::$trait<Result<B>> for &Tensor {
            type Output = Result<Tensor>;

            fn $fn1(self, rhs: Result<B>) -> Self::Output {
                Tensor::$fn1(&self, rhs?.borrow())
            }
        }

        impl std::ops::$trait<f64> for Tensor {
            type Output = Result<Tensor>;

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
