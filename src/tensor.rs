use crate::{op::Op, storage::Storage, DType, Device, Error, Result, Shape};
use std::collections::HashMap;
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
    storage: Storage,
    shape: Shape,
    // The strides are given in number of elements and not in bytes.
    stride: Vec<usize>,
    op: Option<Op>,
    is_variable: bool,
}

#[derive(Clone)]
pub struct Tensor(Arc<Tensor_>);

impl std::ops::Deref for Tensor {
    type Target = Tensor_;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:?}, {:?}]", &self.shape().dims(), self.device())
    }
}

macro_rules! unary_op {
    ($fn_name:ident, $op_name:ident, $impl_name:ident) => {
        pub fn $fn_name(&self) -> Result<Self> {
            let shape = self.shape();
            let storage = self.storage.$impl_name(self.shape(), self.stride())?;
            let tensor_ = Tensor_ {
                id: TensorId::new(),
                storage,
                shape: shape.clone(),
                stride: shape.stride_contiguous(),
                op: Some(Op::$op_name(self.clone())),
                is_variable: false,
            };
            Ok(Self(Arc::new(tensor_)))
        }
    };
}

macro_rules! binary_op {
    ($fn_name:ident, $op_name:ident, $impl_name:ident) => {
        pub fn $fn_name(&self, rhs: &Self) -> Result<Self> {
            let shape = self.same_shape_binary_op(rhs, stringify!($fn_name))?;
            let storage =
                self.storage
                    .$impl_name(&rhs.storage, shape, self.stride(), rhs.stride())?;
            let tensor_ = Tensor_ {
                id: TensorId::new(),
                storage,
                shape: shape.clone(),
                stride: shape.stride_contiguous(),
                op: Some(Op::$op_name(self.clone(), rhs.clone())),
                is_variable: false,
            };
            Ok(Self(Arc::new(tensor_)))
        }
    };
}

impl Tensor {
    fn ones_impl<S: Into<Shape>>(
        shape: S,
        dtype: DType,
        device: &Device,
        is_variable: bool,
    ) -> Result<Self> {
        let shape = shape.into();
        let storage = device.ones(&shape, dtype)?;
        let stride = shape.stride_contiguous();
        let tensor_ = Tensor_ {
            id: TensorId::new(),
            storage,
            shape,
            stride,
            op: None,
            is_variable,
        };
        Ok(Self(Arc::new(tensor_)))
    }

    pub fn ones<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        Self::ones_impl(shape, dtype, device, false)
    }

    pub fn ones_var<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        Self::ones_impl(shape, dtype, device, true)
    }

    pub fn ones_like(&self) -> Result<Self> {
        Tensor::ones(self.shape(), self.dtype(), &self.device())
    }

    fn zeros_impl<S: Into<Shape>>(
        shape: S,
        dtype: DType,
        device: &Device,
        is_variable: bool,
    ) -> Result<Self> {
        let shape = shape.into();
        let storage = device.zeros(&shape, dtype)?;
        let stride = shape.stride_contiguous();
        let tensor_ = Tensor_ {
            id: TensorId::new(),
            storage,
            shape,
            stride,
            op: None,
            is_variable,
        };
        Ok(Self(Arc::new(tensor_)))
    }

    pub fn zeros<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        Self::zeros_impl(shape, dtype, device, false)
    }

    pub fn zeros_var<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        Self::zeros_impl(shape, dtype, device, true)
    }

    pub fn zeros_like(&self) -> Result<Self> {
        Tensor::zeros(self.shape(), self.dtype(), &self.device())
    }

    pub fn new_impl<A: crate::device::NdArray>(
        array: A,
        shape: Shape,
        device: &Device,
        is_variable: bool,
    ) -> Result<Self> {
        // let shape = array.shape()?;
        let storage = device.storage(array)?;
        let stride = shape.stride_contiguous();
        let tensor_ = Tensor_ {
            id: TensorId::new(),
            storage,
            shape,
            stride,
            op: None,
            is_variable,
        };
        Ok(Self(Arc::new(tensor_)))
    }

    pub fn new<A: crate::device::NdArray>(array: A, device: &Device) -> Result<Self> {
        let shape = array.shape()?.clone();
        Self::new_impl(array, shape, device, false)
    }

    pub fn var<A: crate::device::NdArray>(array: A, device: &Device) -> Result<Self> {
        let shape = array.shape()?.clone();
        Self::new_impl(array, shape, device, true)
    }

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

    // TODO: Also make an inplace version or a pre-allocated? This could be tricky
    // if this can create cycles in the compute graph.
    binary_op!(add, Add, add_impl);
    binary_op!(mul, Mul, mul_impl);
    binary_op!(sub, Sub, sub_impl);
    binary_op!(div, Div, div_impl);

    unary_op!(neg, Neg, neg_impl);
    unary_op!(sqr, Sqr, sqr_impl);
    unary_op!(sqrt, Sqrt, sqrt_impl);
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
            Ok::<_, Error>(data[0])
        };
        match &self.storage {
            Storage::Cpu(cpu_storage) => from_cpu_storage(cpu_storage),
            Storage::Cuda(storage) => from_cpu_storage(&storage.to_cpu_storage()?),
        }
    }

    pub fn affine(&self, mul: f64, add: f64) -> Result<Self> {
        let shape = self.shape();
        let storage = self
            .storage
            .affine_impl(self.shape(), self.stride(), mul, add)?;
        let tensor_ = Tensor_ {
            id: TensorId::new(),
            storage,
            shape: shape.clone(),
            stride: shape.stride_contiguous(),
            op: Some(Op::Affine {
                arg: self.clone(),
                mul,
                add,
            }),
            is_variable: false,
        };
        Ok(Self(Arc::new(tensor_)))
    }

    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        let a_dims = self.shape().dims();
        let b_dims = rhs.shape().dims();

        let dim = a_dims.len();

        // TODO
        // if dim < 2 {
        //     return Err(SmeltError::InsufficientRank { minimum_rank: 2 });
        // }
        if b_dims.len() != dim {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: rhs.shape().clone(),
                op: "matmul",
            });
        }

        let m = a_dims[dim - 2];
        let k = a_dims[dim - 1];
        let k2 = b_dims[dim - 2];
        let n = b_dims[dim - 1];
        if k != k2 {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: rhs.shape().clone(),
                op: "matmul",
            });
        }

        let mut c_shape: Vec<_> = a_dims[..dim - 2].into();
        c_shape.extend(&[m, n]);
        let c_shape: Shape = Shape(c_shape);
        let batching: usize = a_dims[..dim - 2].iter().product();

        let storage = self.storage.matmul_impl(
            &rhs.storage,
            (batching, m, n, k),
            self.stride(),
            rhs.stride(),
        )?;
        let tensor_ = Tensor_ {
            id: TensorId::new(),
            storage,
            shape: c_shape.clone(),
            stride: c_shape.stride_contiguous(),
            op: Some(Op::Matmul(self.clone(), rhs.clone())),
            is_variable: false,
        };
        Ok(Self(Arc::new(tensor_)))
    }

    pub(crate) fn strided_index(&self) -> crate::StridedIndex {
        crate::StridedIndex::new(self.dims(), self.stride())
    }

    pub fn as_slice<S: crate::WithDType>(&self) -> Result<&[S]> {
        match &self.storage {
            Storage::Cpu(cpu_storage) => S::cpu_storage_as_slice(cpu_storage),
            Storage::Cuda { .. } => todo!(),
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
        match &self.storage {
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
        match &self.storage {
            Storage::Cpu(storage) => from_cpu_storage(storage),
            Storage::Cuda(storage) => from_cpu_storage(&storage.to_cpu_storage()?),
        }
    }

    pub fn to_vec3<S: crate::WithDType>(&self) -> Result<Vec<Vec<Vec<S>>>> {
        let (dim1, dim2, dim3) = self.shape().r3()?;
        match &self.storage {
            Storage::Cpu(cpu_storage) => {
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
            }
            Storage::Cuda { .. } => todo!(),
        }
    }

    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    pub fn device(&self) -> Device {
        self.storage.device()
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dims(&self) -> &[usize] {
        self.shape().dims()
    }

    pub fn stride(&self) -> &[usize] {
        &self.stride
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

    pub fn t(&self) -> Result<Tensor> {
        let mut stride = self.stride().to_vec();
        let mut shape = self.shape().clone();
        let n = stride.len();
        if n < 2 {
            return Err(Error::UnexpectedNumberOfDims {
                expected: 2,
                got: n,
                shape: self.shape().clone(),
            });
        }
        (shape.0[n - 2], shape.0[n - 1]) = (shape.0[n - 1], shape.0[n - 2]);
        (stride[n - 2], stride[n - 1]) = (stride[n - 1], stride[n - 2]);
        let tensor_ = Tensor_ {
            id: TensorId::new(),
            storage: self.storage.clone(),
            shape,
            stride,
            // TODO The op should have a backward
            op: None,
            is_variable: false,
        };
        Ok(Tensor(Arc::new(tensor_)))
    }

    pub fn is_contiguous(&self) -> bool {
        self.shape.is_contiguous(&self.stride)
    }

    /// Return all the nodes that lead to this value in a topologically sorted vec, the first
    /// elements having dependencies on the latter ones, e.g. the first element if any is the
    /// argument.
    /// This assumes that the op graph is a DAG.
    fn sorted_nodes(&self) -> Vec<&Tensor> {
        // The vec of sorted nodes is passed as an owned value rather than a mutable reference
        // to get around some lifetime limitations.
        fn walk<'a>(
            node: &'a Tensor,
            nodes: Vec<&'a Tensor>,
            already_seen: &mut HashMap<TensorId, bool>,
        ) -> (bool, Vec<&'a Tensor>) {
            if let Some(&tg) = already_seen.get(&node.id) {
                return (tg, nodes);
            }
            let mut track_grad = false;
            let mut nodes = if node.is_variable {
                // Do not call recursively on the "leaf" nodes.
                track_grad = true;
                nodes
            } else if let Some(op) = &node.op {
                match op {
                    Op::Add(lhs, rhs)
                    | Op::Mul(lhs, rhs)
                    | Op::Sub(lhs, rhs)
                    | Op::Div(lhs, rhs)
                    | Op::Matmul(lhs, rhs) => {
                        let (tg, nodes) = walk(lhs, nodes, already_seen);
                        track_grad |= tg;
                        let (tg, nodes) = walk(rhs, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }
                    Op::Affine { arg, mul, .. } => {
                        if *mul == 0. {
                            nodes
                        } else {
                            let (tg, nodes) = walk(arg, nodes, already_seen);
                            track_grad |= tg;
                            nodes
                        }
                    }
                    Op::Sqr(node) | Op::Sqrt(node) | Op::Neg(node) => {
                        let (tg, nodes) = walk(node, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }
                }
            } else {
                nodes
            };
            already_seen.insert(node.id, track_grad);
            if track_grad {
                nodes.push(node);
            }
            (track_grad, nodes)
        }
        let (_tg, mut nodes) = walk(self, vec![], &mut HashMap::new());
        nodes.reverse();
        nodes
    }

    pub fn backward(&self) -> Result<GradStore> {
        let sorted_nodes = self.sorted_nodes();
        println!("{}", sorted_nodes.len());
        let mut grads = GradStore::new();
        grads.insert(self, self.ones_like()?);
        for node in sorted_nodes.iter() {
            if node.is_variable {
                continue;
            }
            let grad = grads.remove(node).unwrap();
            // TODO: We should perform all these operations in place (or at least not track the
            // whole graph).
            // The only drawback would be if we wanted to support grad of grad but this is out of
            // scope.
            if let Some(op) = &node.op {
                match op {
                    Op::Add(lhs, rhs) => {
                        let lhs_sum_grad = grads.or_insert(lhs)?;
                        *lhs_sum_grad = lhs_sum_grad.add(&grad)?;
                        let rhs_sum_grad = grads.or_insert(rhs)?;
                        *rhs_sum_grad = rhs_sum_grad.add(&grad)?;
                    }
                    Op::Sub(lhs, rhs) => {
                        let lhs_sum_grad = grads.or_insert(lhs)?;
                        *lhs_sum_grad = lhs_sum_grad.add(&grad)?;
                        let rhs_sum_grad = grads.or_insert(rhs)?;
                        *rhs_sum_grad = rhs_sum_grad.add(&grad.neg()?)?;
                    }
                    Op::Mul(lhs, rhs) => {
                        let lhs_grad = grad.mul(rhs)?;
                        let lhs_sum_grad = grads.or_insert(lhs)?;
                        *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad)?;
                        let rhs_grad = grad.mul(lhs)?;
                        let rhs_sum_grad = grads.or_insert(rhs)?;
                        *rhs_sum_grad = rhs_sum_grad.add(&rhs_grad)?;
                    }
                    Op::Div(lhs, rhs) => {
                        let lhs_grad = grad.div(rhs)?;
                        let lhs_sum_grad = grads.or_insert(lhs)?;
                        *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad)?;
                        let rhs_grad = grad.mul(lhs)?.div(&rhs.sqr()?)?;
                        let rhs_sum_grad = grads.or_insert(rhs)?;
                        *rhs_sum_grad = rhs_sum_grad.add(&rhs_grad)?;
                    }
                    Op::Matmul(lhs, rhs) => {
                        // Skipping checks, the op went ok, we can skip
                        // the matmul size checks for now.

                        let lhs_grad = grad.matmul(&rhs.t()?)?;
                        let lhs_sum_grad = grads.or_insert(lhs)?;
                        *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad)?;

                        let rhs_grad = lhs.t()?.matmul(&grad)?;
                        let rhs_sum_grad = grads.or_insert(rhs)?;
                        *rhs_sum_grad = rhs_sum_grad.add(&rhs_grad)?;
                    }
                    Op::Affine { arg, mul, .. } => {
                        let arg_grad = grad.affine(*mul, 0.)?;
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.add(&arg_grad)?
                    }
                    Op::Neg(arg) => {
                        let arg_grad = grad.neg()?;
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.add(&arg_grad)?
                    }
                    Op::Sqr(arg) => {
                        let arg_grad = arg.mul(&grad)?.affine(2., 0.)?;
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.add(&arg_grad)?
                    }
                    Op::Sqrt(arg) => {
                        let arg_grad = grad.div(arg)?.affine(0.5, 0.)?;
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.add(&arg_grad)?
                    }
                };
            }
        }
        Ok(grads)
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

pub struct GradStore(HashMap<TensorId, Tensor>);

impl GradStore {
    fn new() -> Self {
        GradStore(HashMap::new())
    }

    pub fn get_id(&self, id: TensorId) -> Option<&Tensor> {
        self.0.get(&id)
    }

    pub fn get(&self, tensor: &Tensor) -> Option<&Tensor> {
        self.0.get(&tensor.id)
    }

    pub fn remove(&mut self, tensor: &Tensor) -> Option<Tensor> {
        self.0.remove(&tensor.id)
    }

    pub fn insert(&mut self, tensor: &Tensor, grad: Tensor) -> Option<Tensor> {
        self.0.insert(tensor.id, grad)
    }

    fn or_insert(&mut self, tensor: &Tensor) -> Result<&mut Tensor> {
        use std::collections::hash_map::Entry;
        let grad = match self.0.entry(tensor.id) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let grad = tensor.zeros_like()?;
                entry.insert(grad)
            }
        };
        Ok(grad)
    }
}
