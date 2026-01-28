use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::Dfs;
use petgraph::Direction::{self, Incoming};
use std::fmt::{Debug, Display};
use std::hash::Hash;

#[derive(thiserror::Error, Debug)]
pub enum LazyError {
    #[error("{0}")]
    Message(String),
}

impl From<String> for LazyError {
    fn from(e: String) -> Self {
        LazyError::Message(e)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EdgeId(usize);

impl NodeId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

impl EdgeId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

pub type OpGraph = DiGraph<Op, OpEdge>;

#[derive(Debug, Clone)]
pub struct LazyEdge<S: Debug + Clone> {
    edge_id: EdgeId,
    layout: Layout,
    dtype: DType,
    state: Option<S>,
}
pub type LazyGraph<S: Debug + Clone> = DiGraph<Op, LazyEdge<S>>;

impl<S: Debug + Clone> From<OpEdge> for LazyEdge<S> {
    fn from(edge: OpEdge) -> Self {
        LazyEdge {
            edge_id: edge.edge_id,
            layout: edge.layout,
            dtype: edge.dtype,
            state: None,
        }
    }
}

impl<S: Debug + Clone> LazyEdge<S> {
    pub fn layout(&self) -> &Layout {
        &self.layout
    }
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }
}

#[derive(Debug, Clone)]
pub struct LazyStorage {
    operations: OpGraph,
    shape: Shape,
    dtype: DType,
    current_node: Option<NodeIndex<u32>>,
    // potentially Arc<RwLock<...>>
    //inner: Option<CpuStorage>,
}

impl LazyStorage {
    fn get_current_node(&self) -> Result<NodeIndex<u32>> {
        self.current_node
            .ok_or(LazyError::Message("No current node in lazy storage".to_string()).into())
    }
}

#[derive(Debug, Clone)]
pub struct LazyResult<S: Clone> {
    lazy: LazyStorage,
    inner: Option<S>,
}

impl<S: Clone> LazyResult<S> {
    pub fn new(shape: Shape, dtype: DType) -> LazyResult<S> {
        LazyResult {
            lazy: LazyStorage::new(shape, dtype),
            inner: None,
        }
    }
}

pub trait Executor {
    type ResultType: Debug + Clone;

    fn eval(
        &self,
        operations: &LazyGraph<Self::ResultType>,
        node: petgraph::graph::NodeIndex<u32>,
        state: Self::ResultType,
    ) -> Result<Self::ResultType>;

    fn run(&self, operations: OpGraph) -> Result<Self::ResultType>;

    /// Converts OpGraph to specialized LazyGraph@
    /// TODO: Rename. preprocess?
    fn prepare(&self, graph: OpGraph) -> LazyGraph<Self::ResultType> {
        graph.map(
            |_, op| op.clone(),
            |_, edge| LazyEdge::<Self::ResultType>::from(edge.clone()),
        )
    }
}

impl LazyStorage {
    fn new(shape: Shape, dtype: DType) -> LazyStorage {
        LazyStorage {
            operations: Default::default(),
            shape,
            dtype,
            current_node: None,
        }
    }

    pub fn execute<E: Executor>(&self, executor: E) -> Result<E::ResultType> {
        // Apply optimizations that apply regardless of backend
        // let mut optimized = self.operations.build::<E::ResultType>();
        //executor.run(&mut optimized)
        /*
        let mut edges = self
            .operations
            .edges_directed(NodeIndex::new(0), petgraph::Incoming);
        let edge = edges.next().unwrap();
        edge.weight()
        */

        executor.run(self.operations.clone())

        /*
        for op in self.graph.operations {
            match op {
                Const(s) => {
                    B::storage_from_cpu_storage_owned(s)
                }
                ToCpu,
                Affine(Layout, f64, f64),
                Powf(Layout, f64),
                Elu(Layout, f64),
                Reduce(ReduceOp, Layout, Vec<usize>),
                Cmp(CmpOp, LazyStorage, Layout, Layout),
                ToDType(Layout, DType),
                Unary(Layout, &'static str),
                Binary(Layout, LazyStorage, Layout, &'static str),
                WhereCond(Layout, LazyStorage, Layout, LazyStorage, Layout),
                Conv1D(Layout, LazyStorage, Layout, crate::conv::ParamsConv1D),
                ConvTranspose1D(Layout, LazyStorage, Layout, crate::conv::ParamsConvTranspose1D),
                Conv2D(Layout, LazyStorage, Layout, crate::conv::ParamsConv2D),
                ConvTranspose2D(Layout, LazyStorage, Layout, crate::conv::ParamsConvTranspose2D),
                AvgPool2D(Layout, (usize, usize), (usize, usize)),
                MaxPool2D(Layout, (usize, usize), (usize, usize)),
                UpsampleNearest1D(Layout, usize),
                UpsampleNearest2D(Layout, usize, usize),
                Gather(Layout, LazyStorage, Layout, usize),
                ScatterSet(Layout, LazyStorage, Layout, LazyStorage, Layout, usize),
                ScatterAddSet(Layout, LazyStorage, Layout, LazyStorage, Layout, usize),
                IndexSelect(Layout, LazyStorage, Layout, usize),
                IndexAdd(Layout, LazyStorage, Layout, LazyStorage, Layout, usize),
                Matmul(LazyStorage, (usize, usize, usize, usize), Layout, Layout),
                CopyStridedSrc(LazyStorage, usize, Layout),
                Copy2D(LazyStorage, usize, usize, usize, usize, usize, usize),
                ConstSet(crate::scalar::Scalar, Layout)
            }
        }
         */
    }
}

#[derive(Debug, Clone)]
pub struct OpEdge {
    edge_id: EdgeId,
    layout: Layout,
    dtype: DType,
}

impl OpEdge {
    pub fn new(layout: Layout, dtype: DType) -> Self {
        OpEdge {
            edge_id: EdgeId::new(),
            layout,
            dtype,
        }
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn dtype(&self) -> &DType {
        &self.dtype
    }
}

#[derive(Debug, Clone)]
pub(crate) enum Op {
    Const(CpuStorage),
    ToCpu,
    Affine(f64, f64),
    Powf(f64),
    Elu(f64),
    Reduce(ReduceOp, Vec<usize>),
    Cmp(CmpOp),
    ToDType(DType),
    Unary(&'static str),
    Binary(&'static str),
    WhereCond,
    Conv1D(crate::conv::ParamsConv1D),
    ConvTranspose1D(crate::conv::ParamsConvTranspose1D),
    Conv2D(crate::conv::ParamsConv2D),
    ConvTranspose2D(crate::conv::ParamsConvTranspose2D),
    AvgPool2D((usize, usize), (usize, usize)),
    MaxPool2D((usize, usize), (usize, usize)),
    UpsampleNearest1D(usize),
    UpsampleNearest2D(usize, usize),
    Gather(usize),
    ScatterSet(usize),
    ScatterAddSet(usize),
    IndexSelect(usize),
    IndexAdd(usize),
    Matmul((usize, usize, usize, usize)),
    CopyStridedSrc(usize),
    Copy2D(usize, usize, usize, usize, usize, usize),
    ConstSet(crate::scalar::Scalar),
}

impl Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Display for OpEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({:?}, {:?}, {:?})",
            self.layout.shape().dims(),
            self.layout.stride(),
            self.dtype
        )
    }
}

impl<S: Debug + Clone> Display for LazyEdge<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({:?}, {:?}, {:?}, {:?})",
            self.layout.shape().dims(),
            self.layout.stride(),
            self.dtype,
            self.state
        )
    }
}

// Very much WIP.
// We want to be able to run "deferred eager" before we start working on actual lazy optimizations.
#[derive(Debug, Clone)]
pub(crate) enum LazyOp {
    Const(CpuStorage),
    ToCpu,
    Affine(Layout, f64, f64),
    Powf(Layout, f64),
    Elu(Layout, f64),
    Reduce(ReduceOp, Layout, Vec<usize>),
    Cmp(CmpOp, LazyStorage, Layout, Layout),
    ToDType(Layout, DType),
    Unary(Layout, &'static str),
    Binary(Layout, LazyStorage, Layout, &'static str),
    WhereCond(Layout, LazyStorage, Layout, LazyStorage, Layout),
    Conv1D(Layout, LazyStorage, Layout, crate::conv::ParamsConv1D),
    ConvTranspose1D(
        Layout,
        LazyStorage,
        Layout,
        crate::conv::ParamsConvTranspose1D,
    ),
    Conv2D(Layout, LazyStorage, Layout, crate::conv::ParamsConv2D),
    ConvTranspose2D(
        Layout,
        LazyStorage,
        Layout,
        crate::conv::ParamsConvTranspose2D,
    ),
    AvgPool2D(Layout, (usize, usize), (usize, usize)),
    MaxPool2D(Layout, (usize, usize), (usize, usize)),
    UpsampleNearest1D(Layout, usize),
    UpsampleNearest2D(Layout, usize, usize),
    Gather(Layout, LazyStorage, Layout, usize),
    ScatterSet(Layout, LazyStorage, Layout, LazyStorage, Layout, usize),
    ScatterAddSet(Layout, LazyStorage, Layout, LazyStorage, Layout, usize),
    IndexSelect(Layout, LazyStorage, Layout, usize),
    IndexAdd(Layout, LazyStorage, Layout, LazyStorage, Layout, usize),
    Matmul(LazyStorage, (usize, usize, usize, usize), Layout, Layout),
    CopyStridedSrc(LazyStorage, usize, Layout),
    Copy2D(LazyStorage, usize, usize, usize, usize, usize, usize),
    ConstSet(crate::scalar::Scalar, Layout),
}

/*
impl<S: Clone> From<Vec<LazyOp>> for OpGraph<S> {
    fn from(ops: Vec<LazyOp>) -> OpGraph<S> {
        //let mut graph = OpGraph<LazyOp>
        for op in ops {}
    }
}
 */

impl BackendStorage for LazyStorage {
    type Device = LazyDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        Ok(self.clone())
    }

    fn dtype(&self) -> DType {
        // Get current node if available
        if let Ok(node) = self.get_current_node() {
            // Get first incoming edge if available
            let mut incoming = self.operations.edges_directed(node, Direction::Incoming);
            if let Some(first) = incoming.next() {
                return first.weight().dtype;
            }
        }
        // Fallback to default dtype
        DType::U32
    }

    fn device(&self) -> &Self::Device {
        &LazyDevice
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        let mut next = self.clone();
        next.operations.add_node(Op::ToCpu);
        //next.operations.push(LazyOp::ToCpu);
        todo!()
    }

    fn affine(&self, l: &Layout, mul: f64, add: f64) -> Result<Self> {
        let mut next = self.clone();

        let op = Op::Affine(mul, add);
        let edge = OpEdge::new(l.clone(), self.dtype);
        let idx = next.operations.add_node(op);
        next.operations
            .add_edge(next.get_current_node()?, idx, edge);
        Ok(next)
    }

    fn powf(&self, l: &Layout, pow: f64) -> Result<Self> {
        let mut next = self.clone();

        let op = Op::Powf(pow);
        let edge = OpEdge::new(l.clone(), self.dtype);
        let idx = next.operations.add_node(op);
        next.operations
            .add_edge(next.get_current_node()?, idx, edge);
        Ok(next)
    }

    fn elu(&self, l: &Layout, alpha: f64) -> Result<Self> {
        let mut next = self.clone();

        let op = Op::Elu(alpha);
        let edge = OpEdge::new(l.clone(), self.dtype);
        let idx = next.operations.add_node(op);
        next.operations
            .add_edge(next.get_current_node()?, idx, edge);
        Ok(next)
    }

    fn reduce_op(&self, reduce: ReduceOp, l: &Layout, dims: &[usize]) -> Result<Self> {
        let mut next = self.clone();

        let op = Op::Reduce(reduce, dims.to_vec());
        let edge = OpEdge::new(l.clone(), self.dtype);
        let idx = next.operations.add_node(op);
        next.operations
            .add_edge(next.get_current_node()?, idx, edge);
        Ok(next)
    }

    fn cmp(&self, cmp_op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        let mut next = self.clone();

        let op = Op::Cmp(cmp_op);
        let idx = next.operations.add_node(op);

        let current_op = next.get_current_node()?;
        let lhs_edge = OpEdge::new(lhs_l.clone(), self.dtype);
        next.operations.add_edge(current_op, idx, lhs_edge);

        let rhs_op = rhs.get_current_node()?;
        let rhs_edge = OpEdge::new(rhs_l.clone(), self.dtype);
        next.operations.add_edge(rhs_op, idx, rhs_edge);

        Ok(next)
    }

    fn to_dtype(&self, l: &Layout, dtype: DType) -> Result<Self> {
        let mut next = self.clone();

        let op = Op::ToDType(dtype);
        let edge = OpEdge::new(l.clone(), self.dtype);
        let idx = next.operations.add_node(op);
        next.operations
            .add_edge(next.get_current_node()?, idx, edge);
        Ok(next)
    }

    fn unary_impl<B: UnaryOpT>(&self, l: &Layout) -> Result<Self> {
        let mut next = self.clone();

        let op = Op::Unary(B::KERNEL);
        let edge = OpEdge::new(l.clone(), self.dtype);
        let idx = next.operations.add_node(op);
        next.operations
            .add_edge(next.get_current_node()?, idx, edge);
        Ok(next)
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let mut next = self.clone();

        let op = Op::Binary(B::KERNEL);
        let idx = next.operations.add_node(op);

        let huh = next.operations.node_weight(idx).unwrap();

        let current_op = next.get_current_node()?;
        let lhs_edge = OpEdge::new(lhs_l.clone(), self.dtype);
        next.operations.add_edge(current_op, idx, lhs_edge);

        let rhs_op = rhs.get_current_node()?;
        let rhs_edge = OpEdge::new(rhs_l.clone(), self.dtype);
        next.operations.add_edge(rhs_op, idx, rhs_edge);

        Ok(next)
    }

    fn where_cond(
        &self,
        l: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        let mut next = self.clone();

        let op = Op::WhereCond;
        let idx = next.operations.add_node(op);

        let current_op = next.get_current_node()?;
        let lhs_edge = OpEdge::new(l.clone(), self.dtype);
        next.operations.add_edge(current_op, idx, lhs_edge);

        let t_op = t.get_current_node()?;
        let t_edge = OpEdge::new(t_l.clone(), self.dtype);
        next.operations.add_edge(t_op, idx, t_edge);

        let f_op = f.get_current_node()?;
        let f_edge = OpEdge::new(f_l.clone(), self.dtype);
        next.operations.add_edge(f_op, idx, f_edge);
        Ok(next)
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        let mut next = self.clone();

        let op = Op::Conv1D(params.clone());
        let idx = next.operations.add_node(op);

        let current_op = next.get_current_node()?;
        let l_edge = OpEdge::new(l.clone(), self.dtype);
        next.operations.add_edge(current_op, idx, l_edge);

        let kernel_op = kernel.get_current_node()?;
        let kernel_edge = OpEdge::new(kernel_l.clone(), self.dtype);
        next.operations.add_edge(kernel_op, idx, kernel_edge);

        Ok(next)
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        let mut next = self.clone();

        let op = Op::ConvTranspose1D(params.clone());
        let idx = next.operations.add_node(op);

        let current_op = next.get_current_node()?;
        let l_edge = OpEdge::new(l.clone(), self.dtype);
        next.operations.add_edge(current_op, idx, l_edge);

        let kernel_op = kernel.get_current_node()?;
        let kernel_edge = OpEdge::new(kernel_l.clone(), self.dtype);
        next.operations.add_edge(kernel_op, idx, kernel_edge);

        Ok(next)
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        let mut next = self.clone();

        let op = Op::Conv2D(params.clone());
        let idx = next.operations.add_node(op);

        let current_op = next.get_current_node()?;
        let l_edge = OpEdge::new(l.clone(), self.dtype);
        next.operations.add_edge(current_op, idx, l_edge);

        let kernel_op = kernel.get_current_node()?;
        let kernel_edge = OpEdge::new(kernel_l.clone(), self.dtype);
        next.operations.add_edge(kernel_op, idx, kernel_edge);

        Ok(next)
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        let mut next = self.clone();

        let op = Op::ConvTranspose2D(params.clone());
        let idx = next.operations.add_node(op);

        let current_op = next.get_current_node()?;
        let l_edge = OpEdge::new(l.clone(), self.dtype);
        next.operations.add_edge(current_op, idx, l_edge);

        let kernel_op = kernel.get_current_node()?;
        let kernel_edge = OpEdge::new(kernel_l.clone(), self.dtype);
        next.operations.add_edge(kernel_op, idx, kernel_edge);

        Ok(next)
    }

    fn avg_pool2d(
        &self,
        l: &Layout,
        (w_k, h_k): (usize, usize),
        (w_stride, h_stride): (usize, usize),
    ) -> Result<Self> {
        let mut next = self.clone();

        let op = Op::AvgPool2D((w_k, h_k), (w_stride, h_stride));
        let idx = next.operations.add_node(op);

        let current_op = next.get_current_node()?;
        let edge = OpEdge::new(l.clone(), self.dtype);
        next.operations.add_edge(current_op, idx, edge);

        Ok(next)
    }

    fn max_pool2d(
        &self,
        l: &Layout,
        (w_k, h_k): (usize, usize),
        (w_stride, h_stride): (usize, usize),
    ) -> Result<Self> {
        let mut next = self.clone();

        let op = Op::MaxPool2D((w_k, h_k), (w_stride, h_stride));
        let idx = next.operations.add_node(op);

        let current_op = next.get_current_node()?;
        let edge = OpEdge::new(l.clone(), self.dtype);
        next.operations.add_edge(current_op, idx, edge);

        Ok(next)
    }

    fn upsample_nearest1d(&self, l: &Layout, sz: usize) -> Result<Self> {
        let mut next = self.clone();

        let op = Op::UpsampleNearest1D(sz);
        let idx = next.operations.add_node(op);

        let current_op = next.get_current_node()?;
        let edge = OpEdge::new(l.clone(), self.dtype);
        next.operations.add_edge(current_op, idx, edge);

        Ok(next)
    }

    fn upsample_nearest2d(&self, l: &Layout, out_w: usize, out_h: usize) -> Result<Self> {
        let mut next = self.clone();

        let op = Op::UpsampleNearest2D(out_w, out_h);
        let idx = next.operations.add_node(op);

        let current_op = next.get_current_node()?;
        let edge = OpEdge::new(l.clone(), self.dtype);
        next.operations.add_edge(current_op, idx, edge);

        Ok(next)
    }

    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        let mut next = self.clone();

        let op = Op::Gather(dim);
        let idx = next.operations.add_node(op);

        let current_op = next.get_current_node()?;
        let edge = OpEdge::new(l.clone(), self.dtype);
        next.operations.add_edge(current_op, idx, edge);

        let ids_op = ids.get_current_node()?;
        let ids_edge = OpEdge::new(ids_l.clone(), self.dtype);
        next.operations.add_edge(ids_op, idx, ids_edge);

        Ok(next)
    }

    fn scatter_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let op = Op::ScatterSet(dim);
        let idx = self.operations.add_node(op);

        let current_op = self.get_current_node()?;
        let edge = OpEdge::new(l.clone(), self.dtype);
        self.operations.add_edge(current_op, idx, edge);

        let ids_op = ids.get_current_node()?;
        let ids_edge = OpEdge::new(ids_l.clone(), self.dtype);
        self.operations.add_edge(ids_op, idx, ids_edge);

        let src_op = src.get_current_node()?;
        let src_edge = OpEdge::new(src_l.clone(), self.dtype);
        self.operations.add_edge(src_op, idx, src_edge);

        Ok(())
    }

    fn scatter_add_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let op = Op::ScatterAddSet(dim);
        let idx = self.operations.add_node(op);

        let current_op = self.get_current_node()?;
        let edge = OpEdge::new(l.clone(), self.dtype);
        self.operations.add_edge(current_op, idx, edge);

        let ids_op = ids.get_current_node()?;
        let ids_edge = OpEdge::new(ids_l.clone(), self.dtype);
        self.operations.add_edge(ids_op, idx, ids_edge);

        let src_op = src.get_current_node()?;
        let src_edge = OpEdge::new(src_l.clone(), self.dtype);
        self.operations.add_edge(src_op, idx, src_edge);

        Ok(())
    }

    fn index_select(&self, ids: &Self, src_l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        let mut next = self.clone();

        let op = Op::IndexSelect(dim);
        let idx = next.operations.add_node(op);

        let current_op = next.get_current_node()?;
        let edge = OpEdge::new(src_l.clone(), self.dtype);
        next.operations.add_edge(current_op, idx, edge);

        let ids_op = ids.get_current_node()?;
        let ids_edge = OpEdge::new(ids_l.clone(), self.dtype);
        next.operations.add_edge(ids_op, idx, ids_edge);

        Ok(next)
    }

    fn index_add(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        let mut next = self.clone();

        let op = Op::IndexAdd(dim);
        let idx = next.operations.add_node(op);

        let current_op = next.get_current_node()?;
        let edge = OpEdge::new(l.clone(), self.dtype);
        next.operations.add_edge(current_op, idx, edge);

        let ids_op = ids.get_current_node()?;
        let ids_edge = OpEdge::new(ids_l.clone(), self.dtype);
        next.operations.add_edge(ids_op, idx, ids_edge);

        let src_op = src.get_current_node()?;
        let src_edge = OpEdge::new(src_l.clone(), self.dtype);
        next.operations.add_edge(src_op, idx, src_edge);

        Ok(next)
    }

    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let mut next = self.clone();

        let op = Op::Matmul((b, m, n, k));
        let idx = next.operations.add_node(op);

        let current_op = next.get_current_node()?;
        let edge = OpEdge::new(lhs_l.clone(), self.dtype);
        next.operations.add_edge(current_op, idx, edge);

        let rhs_op = rhs.get_current_node()?;
        let rhs_edge = OpEdge::new(rhs_l.clone(), self.dtype);
        next.operations.add_edge(rhs_op, idx, rhs_edge);

        Ok(next)
    }

    fn copy_strided_src(&self, rhs: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        // TODO: Validate that self -> mut rhs graph logic is correct.
        let op = Op::CopyStridedSrc(dst_offset);
        let idx = rhs.operations.add_node(op);

        let current_op = self.get_current_node()?;
        let edge = OpEdge::new(src_l.clone(), self.dtype);
        rhs.operations.add_edge(current_op, idx, edge);

        Ok(())
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()> {
        // TODO: Validate that self -> mut dst graph logic is correct.
        let op = Op::Copy2D(d1, d2, src_s, dst_s, src_o, dst_o);
        let idx = dst.operations.add_node(op);

        let current_op = self.get_current_node()?;
        // TODO: May have to use self.layout here
        let edge = OpEdge::new(Layout::contiguous(self.shape.clone()), self.dtype);
        dst.operations.add_edge(current_op, idx, edge);

        Ok(())
    }

    fn const_set(&mut self, s: crate::scalar::Scalar, l: &Layout) -> Result<()> {
        let op = Op::ConstSet(s);
        let idx = self.operations.add_node(op);

        let current_op = self.get_current_node()?;
        let edge = OpEdge::new(l.clone(), self.dtype);
        self.operations.add_edge(current_op, idx, edge);
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct LazyDevice;

impl BackendDevice for LazyDevice {
    type Storage = LazyStorage;

    fn new(_ordinal: usize) -> Result<Self> {
        Ok(LazyDevice)
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Lazy
    }

    fn same_device(&self, _rhs: &Self) -> bool {
        true
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        Ok(LazyStorage::new(shape.clone(), dtype))
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let mut storage = unsafe { self.alloc_uninit(shape, dtype)? };
        storage.const_set(
            crate::scalar::Scalar::zero(dtype),
            &Layout::contiguous(shape.clone()),
        )?;
        Ok(storage)
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        let s = T::to_cpu_storage(s);
        self.storage_from_cpu_storage_owned(s)
    }

    fn storage_from_cpu_storage(&self, s: &CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage_owned(s.clone())
    }

    fn storage_from_cpu_storage_owned(&self, s: CpuStorage) -> Result<Self::Storage> {
        let shape = Shape::from(s.len());
        let dtype = s.dtype();
        let mut storage = unsafe { self.alloc_uninit(&shape, dtype)? };

        let op = Op::Const(s);
        let idx = storage.operations.add_node(op);
        storage.current_node = Some(idx);

        Ok(storage)
    }

    fn rand_uniform(
        &self,
        _shape: &Shape,
        _dtype: DType,
        _min: f64,
        _max: f64,
    ) -> Result<Self::Storage> {
        todo!()
    }

    fn rand_normal(
        &self,
        _shape: &Shape,
        _dtype: DType,
        _mean: f64,
        _stddev: f64,
    ) -> Result<Self::Storage> {
        todo!()
    }

    fn set_seed(&self, _seed: u64) -> Result<()> {
        todo!()
    }

    fn get_current_seed(&self) -> Result<u64> {
        todo!()
    }

    fn synchronize(&self) -> Result<()> {
        todo!()
    }
}
