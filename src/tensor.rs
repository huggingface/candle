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
        device: Device,
        is_variable: bool,
    ) -> Self {
        let shape = shape.into();
        let storage = device.ones(&shape, dtype);
        let stride = shape.stride_contiguous();
        let tensor_ = Tensor_ {
            id: TensorId::new(),
            storage,
            shape,
            stride,
            op: None,
            is_variable,
        };
        Self(Arc::new(tensor_))
    }

    pub fn ones<S: Into<Shape>>(shape: S, dtype: DType, device: Device) -> Self {
        Self::ones_impl(shape, dtype, device, false)
    }

    pub fn ones_var<S: Into<Shape>>(shape: S, dtype: DType, device: Device) -> Self {
        Self::ones_impl(shape, dtype, device, true)
    }

    pub fn ones_like(&self) -> Self {
        Tensor::ones(self.shape(), self.dtype(), self.device())
    }

    fn zeros_impl<S: Into<Shape>>(
        shape: S,
        dtype: DType,
        device: Device,
        is_variable: bool,
    ) -> Self {
        let shape = shape.into();
        let storage = device.zeros(&shape, dtype);
        let stride = shape.stride_contiguous();
        let tensor_ = Tensor_ {
            id: TensorId::new(),
            storage,
            shape,
            stride,
            op: None,
            is_variable,
        };
        Self(Arc::new(tensor_))
    }

    pub fn zeros<S: Into<Shape>>(shape: S, dtype: DType, device: Device) -> Self {
        Self::zeros_impl(shape, dtype, device, false)
    }

    pub fn zeros_var<S: Into<Shape>>(shape: S, dtype: DType, device: Device) -> Self {
        Self::zeros_impl(shape, dtype, device, true)
    }

    pub fn zeros_like(&self) -> Self {
        Tensor::zeros(self.shape(), self.dtype(), self.device())
    }

    pub fn new_impl<A: crate::device::NdArray>(
        array: A,
        device: Device,
        is_variable: bool,
    ) -> Result<Self> {
        let shape = array.shape()?;
        let storage = device.tensor(array);
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

    pub fn new<A: crate::device::NdArray>(array: A, device: Device) -> Result<Self> {
        Self::new_impl(array, device, false)
    }

    pub fn var<A: crate::device::NdArray>(array: A, device: Device) -> Result<Self> {
        Self::new_impl(array, device, true)
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
        match &self.storage {
            Storage::Cpu(cpu_storage) => {
                let data = S::cpu_storage_as_slice(cpu_storage)?;
                Ok(data[0])
            }
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

    pub(crate) fn strided_index(&self) -> crate::StridedIndex {
        crate::StridedIndex::new(self.dims(), self.stride())
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
        }
    }

    pub fn to_vec2<S: crate::WithDType>(&self) -> Result<Vec<Vec<S>>> {
        let (dim1, dim2) = self.shape().r2()?;
        match &self.storage {
            Storage::Cpu(cpu_storage) => {
                let data = S::cpu_storage_as_slice(cpu_storage)?;
                let mut rows = vec![];
                let mut src_index = self.strided_index();
                for _idx_row in 0..dim1 {
                    let row = (0..dim2).map(|_| data[src_index.next().unwrap()]).collect();
                    rows.push(row)
                }
                assert!(src_index.next().is_none());
                Ok(rows)
            }
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
                    | Op::Div(lhs, rhs) => {
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

    pub fn backward(&self) -> Result<HashMap<TensorId, Tensor>> {
        let sorted_nodes = self.sorted_nodes();
        println!("{}", sorted_nodes.len());
        let mut grads = HashMap::new();
        grads.insert(self.id, self.ones_like());
        for node in sorted_nodes.iter() {
            if node.is_variable {
                continue;
            }
            let grad = grads.remove(&node.id).unwrap();
            // TODO: We should perform all these operations in place (or at least not track the
            // whole graph).
            // The only drawback would be if we wanted to support grad of grad but this is out of
            // scope.
            if let Some(op) = &node.op {
                match op {
                    Op::Add(lhs, rhs) => {
                        let lhs_sum_grad = grads.entry(lhs.id).or_insert_with(|| lhs.zeros_like());
                        *lhs_sum_grad = lhs_sum_grad.add(&grad)?;
                        let rhs_sum_grad = grads.entry(rhs.id).or_insert_with(|| rhs.zeros_like());
                        *rhs_sum_grad = rhs_sum_grad.add(&grad)?;
                    }
                    Op::Sub(lhs, rhs) => {
                        let lhs_sum_grad = grads.entry(lhs.id).or_insert_with(|| lhs.zeros_like());
                        *lhs_sum_grad = lhs_sum_grad.add(&grad)?;
                        let rhs_sum_grad = grads.entry(rhs.id).or_insert_with(|| rhs.zeros_like());
                        *rhs_sum_grad = rhs_sum_grad.add(&grad.neg()?)?;
                    }
                    Op::Mul(lhs, rhs) => {
                        let lhs_grad = grad.mul(rhs)?;
                        let lhs_sum_grad = grads.entry(lhs.id).or_insert_with(|| lhs.zeros_like());
                        *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad)?;
                        let rhs_grad = grad.mul(lhs)?;
                        let rhs_sum_grad = grads.entry(rhs.id).or_insert_with(|| rhs.zeros_like());
                        *rhs_sum_grad = rhs_sum_grad.add(&rhs_grad)?;
                    }
                    Op::Div(lhs, rhs) => {
                        let lhs_grad = grad.div(rhs)?;
                        let lhs_sum_grad = grads.entry(lhs.id).or_insert_with(|| lhs.zeros_like());
                        *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad)?;
                        let rhs_grad = grad.mul(lhs)?.div(&rhs.sqr()?)?;
                        let rhs_sum_grad = grads.entry(rhs.id).or_insert_with(|| rhs.zeros_like());
                        *rhs_sum_grad = rhs_sum_grad.add(&rhs_grad)?;
                    }
                    Op::Affine { arg, mul, .. } => {
                        let arg_grad = grad.affine(*mul, 0.)?;
                        let sum_grad = grads.entry(arg.id).or_insert_with(|| arg.zeros_like());
                        *sum_grad = sum_grad.add(&arg_grad)?
                    }
                    Op::Neg(arg) => {
                        let arg_grad = grad.neg()?;
                        let sum_grad = grads.entry(arg.id).or_insert_with(|| arg.zeros_like());
                        *sum_grad = sum_grad.add(&arg_grad)?
                    }
                    Op::Sqr(arg) => {
                        let arg_grad = arg.mul(&grad)?.affine(2., 0.)?;
                        let sum_grad = grads.entry(arg.id).or_insert_with(|| arg.zeros_like());
                        *sum_grad = sum_grad.add(&arg_grad)?
                    }
                    Op::Sqrt(arg) => {
                        let arg_grad = grad.div(arg)?.affine(0.5, 0.)?;
                        let sum_grad = grads.entry(arg.id).or_insert_with(|| arg.zeros_like());
                        *sum_grad = sum_grad.add(&arg_grad)?
                    }
                };
            }
        }
        Ok(grads)
    }
}
