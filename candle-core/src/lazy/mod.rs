pub mod custom;
pub mod ops;

use crate::backend::{BackendDevice, BackendStorage};
use crate::lazy::custom::{CustomOp, LazyCustomFn, LazyCustomOp};
use crate::lazy::ops::{
    Affine, Binary, Cmp, Copy2D, CopyStridedSrc, CustomOpContainer, Elu, IndexSelect, LazyOp,
    Matmul, Output, Powf, Reduce, SingleCopy2D, Sink, ToDType, Unary, WhereCond,
};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::sync::Arc;
use std::sync::{atomic, LazyLock, Mutex};

#[derive(thiserror::Error, Debug, Clone)]
pub enum LazyError {
    #[error("{0}")]
    Message(String),

    #[error("Incorrect incoming edges to node {0:?}")]
    InvalidIncoming(NodeId),

    #[error("Incorrect outgoing edges to node {0:?}")]
    InvalidOutgoing(NodeId),

    #[error("No buffer found for {0:?}")]
    BufferNotFound(BufferId),
}

impl From<String> for LazyError {
    fn from(e: String) -> Self {
        LazyError::Message(e)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(usize);

impl NodeId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }

    pub fn index(&self) -> usize {
        self.0
    }
}

#[derive(Debug)]
pub struct BufferId(atomic::AtomicUsize);

impl BufferId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed).into())
    }

    pub fn inner(&self) -> usize {
        self.0.load(atomic::Ordering::Relaxed)
    }
}

impl Clone for BufferId {
    fn clone(&self) -> Self {
        Self(self.inner().into())
    }
}

impl Hash for BufferId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inner().hash(state);
    }
}

impl PartialEq for BufferId {
    fn eq(&self, other: &Self) -> bool {
        self.inner() == other.inner()
    }
}
impl Eq for BufferId {}

impl PartialOrd for BufferId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.inner().partial_cmp(&other.inner())
    }
}

impl Ord for BufferId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.inner().cmp(&other.inner())
    }
}

static CUSTOM_OP_REGISTRY: LazyLock<Mutex<HashMap<String, CustomOp>>> =
    LazyLock::new(|| Mutex::new(HashMap::<String, CustomOp>::new()));

#[derive(Debug, Clone)]
pub struct LazyStorage {
    id: NodeId,
    // TODO: Arc<Mutex<Op>>. Set as Op::Resolved(BufferId) when resolved
    op: Arc<Op>,
    custom_op_fallbacks: HashMap<String, LazyStorage>,
    layout: Layout,
    dtype: DType,
    buffer_id: Arc<BufferId>,
    // Set once by the backend executor after this node has been evaluated
    resolved_id: Arc<std::sync::OnceLock<BufferId>>,
}

impl LazyStorage {
    pub fn new(op: Op, layout: &Layout, dtype: DType) -> Self {
        Self {
            id: NodeId::new(),
            op: Arc::new(op),
            custom_op_fallbacks: HashMap::new(),
            layout: layout.clone(),
            dtype,
            buffer_id: Arc::new(BufferId::new()),
            resolved_id: Arc::new(std::sync::OnceLock::new()),
        }
    }

    pub fn copy(source: &LazyStorage, layout: &Layout) -> Self {
        // Creates a copy, or view, of the lazy storage with updated layout.
        Self {
            id: source.id,
            op: Arc::new(source.op().clone()),
            custom_op_fallbacks: source.custom_op_fallbacks.clone(),
            layout: layout.clone(),
            dtype: source.dtype,
            buffer_id: source.buffer_id.clone(),
            resolved_id: source.resolved_id.clone(),
        }
    }

    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn op(&self) -> &Op {
        self.op.as_ref()
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn add_custom_fallback<S: Into<String>>(&mut self, name: S, fallback: LazyStorage) {
        self.custom_op_fallbacks.insert(name.into(), fallback);
    }

    pub fn output(&self) -> Result<Self> {
        // Only add output node if not already present
        if !matches!(self.op.as_ref(), Op::Output(_)) {
            return Ok(LazyStorage::new(
                Op::Output(Output::new(self.clone())),
                self.layout(),
                self.dtype(),
            ));
        };

        Ok(self.clone())
    }

    pub(crate) fn custom_ops(&self) -> HashMap<String, CustomOp> {
        CUSTOM_OP_REGISTRY.lock().unwrap().clone()
    }

    pub fn register_custom_op(&mut self, op: CustomOp) {
        match op {
            CustomOp::One(ref custom_op1) => {
                let mut binding = CUSTOM_OP_REGISTRY.lock().unwrap();
                binding.insert(custom_op1.name().to_string(), op.clone());
            }
            CustomOp::Two(ref custom_op2) => {
                let mut binding = CUSTOM_OP_REGISTRY.lock().unwrap();
                binding.insert(custom_op2.name().to_string(), op.clone());
            }
            CustomOp::Three(ref custom_op3) => {
                let mut binding = CUSTOM_OP_REGISTRY.lock().unwrap();
                binding.insert(custom_op3.name().to_string(), op.clone());
            }
        }
    }

    pub fn custom_op(&self, op: Box<dyn LazyCustomOp>, args: &[(&Self, &Layout)]) -> Result<Self> {
        // Include self / src with layout in the args list
        let mut src_and_args = vec![(self.clone(), self.layout().clone())];

        src_and_args.extend(args.iter().map(|(b, l)| {
            let b = LazyStorage::copy(b, l);
            (b, (*l).clone())
        }));

        let expected_edges = op.expected_edges();
        if expected_edges != src_and_args.len() {
            crate::bail!(
                "Incorrect args len. Expected {}, got {}",
                expected_edges,
                src_and_args.len()
            );
        }

        let container = CustomOpContainer::new(src_and_args, op);
        Ok(LazyStorage::new(
            Op::CustomOp(container),
            self.layout(),
            self.dtype(),
        ))
    }

    pub fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    pub(crate) fn buffer_id(&self) -> &BufferId {
        &self.buffer_id
    }

    pub fn resolved_buffer_id(&self) -> Option<&BufferId> {
        self.resolved_id.get()
    }

    pub fn mark_resolved(&self, buf_id: BufferId) {
        let _ = self.resolved_id.set(buf_id);
    }

    pub fn set_buffer_id(&self, buffer_id: BufferId) {
        self.buffer_id
            .0
            .store(buffer_id.inner(), atomic::Ordering::Relaxed);
    }

    pub(crate) fn execution_order(&self) -> Vec<&LazyStorage> {
        let mut done = HashSet::new();
        let mut pending = HashSet::new();
        let mut order = Vec::new();

        let mut stack: Vec<(&LazyStorage, usize)> = vec![(self, 0)];
        while let Some((cur_t, cur_src)) = stack.pop() {
            // Only evaluate nodes once. If already resolved then we do not process.
            let effective_len = if cur_t.resolved_buffer_id().is_some() {
                0
            } else {
                cur_t.op().srcs().len()
            };
            let all_deps_done = cur_src == effective_len;

            if all_deps_done {
                done.insert(cur_t.id());
                pending.remove(&cur_t.id());
                order.push(cur_t);
                continue;
            }

            let (srcs_with_deps, srcs_without_deps): (Vec<_>, Vec<_>) = cur_t
                .op()
                .srcs()
                .iter()
                .partition(|s| s.op().srcs().is_empty());

            let all_srcs = srcs_with_deps
                .into_iter()
                .chain(srcs_without_deps)
                .collect::<Vec<_>>();

            let precursor: &LazyStorage = all_srcs[cur_src];

            if done.contains(&precursor.id()) {
                stack.push((cur_t, cur_src + 1));
            } else if pending.contains(&precursor.id()) {
                panic!(
                    "Cycle detected whilst computing topological order: {:?}. Try plotting with feature `plotting`.",
                    precursor.id()
                );
            } else {
                pending.insert(precursor.id());
                stack.push((cur_t, cur_src));
                stack.push((precursor, 0));
            }
        }
        order
    }
}

impl PartialEq for LazyStorage {
    fn eq(&self, other: &Self) -> bool {
        // Two lazy storages are equal regardless of wether it has been resolved,
        // so we exclude resolved_id from this impl.
        self.id == other.id
            && self.op == other.op
            && self.custom_op_fallbacks == other.custom_op_fallbacks
            && self.layout == other.layout
            && self.dtype == other.dtype
            && self.buffer_id == other.buffer_id
    }
}

struct Dot(LazyStorage);
static INDENT: &str = "    ";
impl Dot {
    pub fn new(leaf: LazyStorage) -> Self {
        Dot(leaf)
    }

    pub fn graph_fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "digraph {{ ")?;
        writeln!(f, "{}ordering=\"in\"", INDENT)?;
        let graph = self.0.execution_order();
        // output all labels
        let mut max_id = 0;
        for node in graph.iter() {
            let node_id = node.id().index();
            if node_id > max_id {
                max_id = node_id;
            }
            write!(f, "{}{} [ ", INDENT, node.id().index())?;
            write!(f, "label = \"")?;
            write!(f, "{}", node.op())?;
            write!(f, "\" ")?;
            writeln!(f, "]")?;
        }
        max_id += 1;

        write!(f, "{}{} [ ", INDENT, max_id)?;
        writeln!(f, "label = \"\"]")?;

        // output all edges
        for node in graph.iter() {
            for source in node.srcs() {
                write!(
                    f,
                    "{}{} -> {} [ ",
                    INDENT,
                    source.id().index(),
                    node.id().index(),
                )?;
                write!(f, "label = \"")?;
                write!(
                    f,
                    "{} ({:?}, {:?}, {})",
                    source.buffer_id().inner(),
                    source.shape(),
                    source.layout().stride(),
                    source.dtype().as_str(),
                )?;
                write!(f, "\" ")?;
                writeln!(f, "]")?;
            }
        }

        let last_node = graph.last().unwrap();
        write!(f, "{}{} -> {} [ ", INDENT, last_node.id().index(), max_id)?;
        write!(f, "label = \"")?;
        write!(
            f,
            "{} ({:?}, {:?}, {})",
            last_node.buffer_id().inner(),
            last_node.shape(),
            last_node.layout().stride(),
            last_node.dtype().as_str(),
        )?;
        write!(f, "\" ")?;
        writeln!(f, "]")?;

        writeln!(f, "}}")?;

        Ok(())
    }
}

impl std::fmt::Display for Dot {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.graph_fmt(f)
    }
}

pub fn graph_to_dot(node: &LazyStorage) -> String {
    format!("{}", Dot::new(node.clone()))
}

pub trait LazyBuffer: Debug + Clone + PartialEq {}

pub trait LazyAllocator<B: LazyBuffer> {
    fn initialize(&mut self, graph: &[&LazyStorage]) -> Result<()>;
    fn insert(&mut self, id: BufferId, buffer: B) -> Result<()>;
    fn allocate(&mut self, id: BufferId, shape: &Shape, dtype: DType) -> Result<&B>;
    fn get(&self, id: &BufferId) -> Result<&B>;
    fn get_or_allocate(&mut self, id: BufferId, shape: &Shape, dtype: DType) -> Result<&B>;
}

#[derive(Debug, Clone)]
pub struct MemoryPlan {
    pub reusage: BTreeMap<BufferId, BufferId>,
    pub allocations: BTreeMap<BufferId, (Layout, DType)>,
}

pub fn determine_tensor_source(source: &LazyStorage) -> &LazyStorage {
    /* TODO: Support inplace properly
    let mut source = source;
    loop {
        if source.op().srcs().is_empty() {
            break;
        }
        let to_modify = true_source.op().srcs()[0];
        source = edge;
    }
    */
    source
}

pub fn calculate_usage_records(
    graph: &[&LazyStorage],
) -> BTreeMap<BufferId, (Option<BufferId>, Option<usize>, usize, Layout, DType)> {
    let mut records = BTreeMap::new();
    let topo_len = graph.len() - 1;
    for (i, node) in graph.iter().rev().enumerate() {
        let buffer_id = node.buffer_id();

        if node.op().resolved() || node.resolved_buffer_id().is_some() {
            continue;
        }
        for source in node.srcs() {
            if source.op().resolved() || source.resolved_buffer_id().is_some() {
                continue;
            }
            let true_source = determine_tensor_source(source);
            records
                .entry(true_source.buffer_id().clone())
                .or_insert_with(|| {
                    (
                        None,
                        None,
                        topo_len - i,
                        true_source.layout.clone(),
                        true_source.dtype(),
                    )
                });
        }

        if let Some(record) = records.get_mut(&buffer_id) {
            record.0 = Some(buffer_id.clone());
            record.1 = Some(topo_len - i);
            // Update layout to the producer's layout to ensure allocation is large enough.
            // or_insert_with stores a consumer's (possibly narrowed) view of the layout.
            // Producer determines the true output size.
            record.3 = node.layout.clone();
            record.4 = node.dtype();
        }
    }
    //filter records with no producer
    records.retain(|_, v| v.1.is_some());
    records
}

// https://arxiv.org/pdf/2001.03288.pdf
pub fn greedy_by_size(graph: &[&LazyStorage]) -> Result<MemoryPlan> {
    let mut reusage = BTreeMap::new();
    let mut allocations = BTreeMap::new();

    // Plan Const (input) allocations
    for node in graph {
        let buffer_id = node.buffer_id();

        if let Op::Const(s) = node.op() {
            let shape = Shape::from(s.len());
            allocations.insert(buffer_id.clone(), (Layout::contiguous(shape), s.dtype()));
        }
    }

    // Find buffer usage spans in the graph
    let record_map = calculate_usage_records(graph);

    let mut shared_objects: Vec<BufferId> = Vec::with_capacity(record_map.len());

    let current_size =
        |layout: &Layout, dtype: &DType| layout.shape().elem_count() * dtype.size_in_bytes();

    for (buffer_id, (_record_buffer_id, producer, last_consumer, layout, dtype)) in
        record_map.iter()
    {
        let record_producer = producer.unwrap();
        let needed_size = current_size(layout, dtype);
        let mut best_buffer: Option<(BufferId, usize)> = None;

        for obj in shared_objects.iter() {
            // Size check: the canonical buffer must be large enough to hold this tensor.
            let Some((obj_layout, obj_dtype)) = allocations.get(obj) else {
                continue;
            };
            let obj_size = current_size(obj_layout, obj_dtype);
            if obj_size < needed_size {
                continue;
            }

            let mut suitable = true;
            for (
                inner_buffer_id,
                (_, inner_producer, inner_last_consumer, _inner_layout, _inner_dtype),
            ) in record_map.iter()
            {
                let max_first = std::cmp::max(record_producer, inner_producer.unwrap());
                let min_last = *std::cmp::min(last_consumer, inner_last_consumer);
                if max_first <= min_last
                    && (inner_buffer_id == obj || reusage.get(inner_buffer_id) == Some(obj))
                {
                    suitable = false;
                    break;
                }
            }
            if suitable {
                // Prefer smallest suitable buffer (best fit) to reduce fragmentation.
                if best_buffer
                    .as_ref()
                    .map_or(true, |(_, best_size)| obj_size < *best_size)
                {
                    best_buffer = Some((obj.clone(), obj_size));
                }
            }
        }
        if let Some((best, _)) = best_buffer {
            reusage.insert(buffer_id.clone(), best);
        } else {
            allocations.insert(buffer_id.clone(), (layout.clone(), dtype.clone()));
            shared_objects.push(buffer_id.clone());
        }
    }

    // Handle final output edge
    let output = graph.last().unwrap();
    let source = determine_tensor_source(output);
    allocations.insert(
        output.buffer_id().clone(),
        (source.layout.clone(), source.dtype()),
    );

    Ok(MemoryPlan {
        reusage,
        allocations,
    })
}

pub trait Executor {
    type BufferType: LazyBuffer;
    type AllocatorType: LazyAllocator<Self::BufferType>;

    fn optimize(&self, _lazy_storage: &LazyStorage) {
        // TODO: Generic optimizations
    }

    /// Backend specific optimizations. Defaults to noop.
    fn specialize(&self, _lazy_storage: &LazyStorage) {}

    fn run(&self, lazy_storage: LazyStorage) -> Result<Self::BufferType>;

    fn eval(&self, lazy_storage: &LazyStorage, allocator: &mut Self::AllocatorType) -> Result<()>;

    fn allocator(&self) -> Self::AllocatorType;

    fn get_specialized_op(
        &self,
        _op: &Box<dyn LazyCustomOp>,
    ) -> Option<Box<dyn LazyCustomFn<Self::BufferType>>> {
        None
    }

    fn install_custom_op(
        &mut self,
        op: &Box<dyn LazyCustomOp>,
        func: Box<dyn LazyCustomFn<Self::BufferType>>,
    ) -> Option<&Box<dyn LazyCustomFn<Self::BufferType>>>;
}

impl LazyStorage {
    pub fn execute<E: Executor>(&self, executor: E) -> Result<E::BufferType> {
        // TODO: Apply backend agnostic optimizations
        executor.run(self.clone())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Uninit,
    Const(Arc<CpuStorage>),
    ToCpu,
    Affine(Affine),
    Powf(Powf),
    Elu(Elu),
    Reduce(Reduce),
    Cmp(Cmp),
    ToDType(ToDType),
    Unary(Unary),
    Binary(Binary),
    WhereCond(WhereCond),
    /*
    Conv1D(Conv1D),
    ConvTranspose1D(ConvTranspose1D),
    Conv2D(Conv2D),
    ConvTranspose2D(ConvTranspose2D),
    AvgPool2D(AvgPool2D),
    MaxPool2D(MaxPool2D),
    UpsampleNearest1D(UpsampleNearest1D),
    UpsampleNearest2D(UpsampleNearest2D),
    Gather(Gather),
    ScatterSet(usize),
    ScatterAddSet(usize),
    */
    IndexSelect(IndexSelect),
    //IndexAdd(usize),
    Matmul(Matmul),
    CopyStridedSrc(CopyStridedSrc),
    Copy2D(Copy2D),
    ConstSet(crate::scalar::Scalar),
    Sink(Sink),
    Aggregate,
    CustomOp(CustomOpContainer),
    Output(Output),
}

impl Op {
    pub fn srcs(&self) -> Vec<&LazyStorage> {
        match self {
            Op::Uninit => vec![],
            Op::Const(_) => vec![],
            Op::ToCpu => vec![],
            Op::Affine(op) => op.srcs(),
            Op::Powf(op) => op.srcs(),
            Op::Elu(op) => op.srcs(),
            Op::Reduce(op) => op.srcs(),
            Op::Cmp(op) => op.srcs(),
            Op::ToDType(op) => op.srcs(),
            Op::Unary(op) => op.srcs(),
            Op::Binary(op) => op.srcs(),
            Op::WhereCond(op) => op.srcs(),
            Op::IndexSelect(op) => op.srcs(),
            Op::Matmul(op) => op.srcs(),
            Op::CopyStridedSrc(op) => op.srcs(),
            Op::Copy2D(op) => op.srcs(),
            Op::ConstSet(_scalar) => vec![],
            Op::Sink(op) => op.srcs(),
            Op::Aggregate => vec![],
            Op::CustomOp(op) => op.srcs(),
            Op::Output(op) => op.srcs(),
        }
    }

    fn resolved(&self) -> bool {
        matches!(self, Op::Const(_))
    }
}

impl Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Op::Uninit => write!(f, "Uninit"),
            Op::Const(_) => write!(f, "Const"),
            Op::ToCpu => write!(f, "ToCpu"),
            Op::Affine(affine) => write!(f, "Affine({}, {})", affine.add, affine.mul),
            Op::Powf(_powf) => write!(f, "Powf"),
            Op::Elu(_elu) => write!(f, "Elu"),
            Op::Reduce(reduce) => write!(
                f,
                "Reduce({}, ({:?}, {:?}, {}), {})",
                reduce.op().name(),
                reduce.reduction_layout().shape().dims(),
                reduce.reduction_layout().stride(),
                reduce.reduction_layout().start_offset(),
                reduce.count()
            ),
            Op::Cmp(_cmp) => write!(f, "Cmp"),
            Op::ToDType(to_dtype) => write!(f, "ToDType({})", to_dtype.dtype.as_str()),
            Op::Unary(unary) => write!(f, "Unary({})", unary.op()),
            Op::Binary(binary) => write!(f, "Binary({})", binary.op()),
            Op::WhereCond(_where_cond) => write!(f, "WhereCond"),
            Op::IndexSelect(_index_select) => write!(f, "IndexSelect"),
            Op::Matmul(_matmul) => write!(f, "Matmul"),
            Op::CopyStridedSrc(_copy_strided_src) => write!(f, "CopyStridedSrc"),
            Op::Copy2D(_copy2d) => write!(f, "Copy2D"),
            Op::ConstSet(_scalar) => write!(f, "ConstSet"),
            Op::Sink(_sink) => write!(f, "Sink"),
            Op::Aggregate => write!(f, "Aggregate"),
            Op::Output(_output) => write!(f, "Output"),
            Op::CustomOp(custom_op_container) => {
                write!(f, "CustomOp({})", custom_op_container.name())
            }
        }
    }
}

impl LazyStorage {
    fn srcs(&self) -> Vec<&LazyStorage> {
        self.op.srcs()
    }
}

impl BackendStorage for LazyStorage {
    type Device = LazyDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        Ok(self.clone())
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &LazyDevice
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        todo!()
    }

    fn affine(&self, l: &Layout, mul: f64, add: f64) -> Result<Self> {
        let op = Op::Affine(Affine::new(self.clone(), l.clone(), mul, add));
        Ok(LazyStorage::new(op, l, self.dtype()))
    }

    fn powf(&self, l: &Layout, exp: f64) -> Result<Self> {
        let op = Op::Powf(Powf::new(self.clone(), l.clone(), exp));
        Ok(LazyStorage::new(op, l, self.dtype()))
    }

    fn elu(&self, l: &Layout, alpha: f64) -> Result<Self> {
        let op = Op::Elu(Elu::new(self.clone(), l.clone(), alpha));
        Ok(LazyStorage::new(op, l, self.dtype()))
    }

    fn reduce_op(&self, reduce: ReduceOp, l: &Layout, reduce_dims: &[usize]) -> Result<Self> {
        // Calculate reduction layout and number of destination elements.
        let src_stride = l.stride();
        let src_dims = l.dims();
        // Source dims and strides with the sum dims at the end.
        let mut dims = vec![];
        let mut stride = vec![];
        let mut dst_el: usize = 1;
        for (dim_idx, &d) in src_dims.iter().enumerate() {
            if !reduce_dims.contains(&dim_idx) {
                dst_el *= d;
                dims.push(d);
                stride.push(src_stride[dim_idx]);
            }
        }

        let mut dst_dims = src_dims.to_vec();
        for &dim_idx in reduce_dims.iter() {
            dims.push(src_dims[dim_idx]);
            stride.push(src_stride[dim_idx]);
            dst_dims[dim_idx] = 1;
        }

        let reduction_shape = Shape::from(dims.clone());
        let reduction_layout = Layout::new(reduction_shape, stride, 0);

        let dst_shape = Shape::from(dst_dims);
        let dst_layout = Layout::contiguous(dst_shape);

        let op = Op::Reduce(Reduce::new(
            self.clone(),
            reduce,
            reduce_dims.to_vec(),
            reduction_layout.clone(),
            dst_el,
        ));
        let dtype = match reduce {
            ReduceOp::ArgMin | ReduceOp::ArgMax => DType::U32,
            _ => self.dtype(),
        };
        Ok(LazyStorage::new(op, &dst_layout, dtype))
    }

    fn cmp(&self, cmp_op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        let op = Op::Cmp(Cmp::new(
            self.clone(),
            lhs_l.clone(),
            rhs.clone(),
            rhs_l.clone(),
            cmp_op,
        ));
        Ok(LazyStorage::new(op, self.layout(), DType::U8))
    }

    fn to_dtype(&self, l: &Layout, dtype: DType) -> Result<Self> {
        let src = LazyStorage::copy(self, l);
        if self.dtype() != dtype {
            let op = Op::ToDType(ToDType::new(src, l.clone(), dtype));
            return Ok(LazyStorage::new(op, l, dtype));
        }
        Ok(src)
    }

    fn unary_impl<B: UnaryOpT>(&self, l: &Layout) -> Result<Self> {
        let src = LazyStorage::copy(self, l);
        let op = Op::Unary(Unary::new(src, l.clone(), B::KERNEL));
        Ok(LazyStorage::new(op, l, self.dtype()))
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let lhs = LazyStorage::copy(self, lhs_l);
        let rhs = LazyStorage::copy(rhs, rhs_l);

        let op = Op::Binary(Binary::new(
            lhs,
            lhs_l.clone(),
            rhs,
            rhs_l.clone(),
            B::KERNEL,
        ));
        Ok(LazyStorage::new(op, lhs_l, self.dtype()))
    }

    fn where_cond(
        &self,
        l: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        let src = LazyStorage::copy(self, l);
        let t = LazyStorage::copy(t, t_l);
        let f = LazyStorage::copy(f, f_l);
        let dtype = f.dtype();

        let op = Op::WhereCond(WhereCond::new(
            src,
            l.clone(),
            t,
            t_l.clone(),
            f,
            f_l.clone(),
        ));
        Ok(LazyStorage::new(op, f_l, dtype))
    }

    fn conv1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        todo!()
    }

    fn conv_transpose1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        todo!()
    }

    fn conv2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        todo!()
    }

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        todo!()
    }

    fn avg_pool2d(
        &self,
        _l: &Layout,
        (_w_k, _h_k): (usize, usize),
        (_w_stride, _h_stride): (usize, usize),
    ) -> Result<Self> {
        todo!()
    }

    fn max_pool2d(
        &self,
        _l: &Layout,
        (_w_k, _h_k): (usize, usize),
        (_w_stride, _h_stride): (usize, usize),
    ) -> Result<Self> {
        todo!()
    }

    fn upsample_nearest1d(&self, _l: &Layout, _sz: usize) -> Result<Self> {
        todo!()
    }

    fn upsample_nearest2d(&self, _l: &Layout, _out_w: usize, _out_h: usize) -> Result<Self> {
        todo!()
    }

    fn gather(&self, _l: &Layout, _ids: &Self, _ids_l: &Layout, _dim: usize) -> Result<Self> {
        todo!()
    }

    fn scatter_set(
        &mut self,
        _l: &Layout,
        _ids: &Self,
        _ids_l: &Layout,
        _src: &Self,
        _src_l: &Layout,
        _dim: usize,
    ) -> Result<()> {
        todo!()
    }

    fn scatter_add_set(
        &mut self,
        _l: &Layout,
        _ids: &Self,
        _ids_l: &Layout,
        _src: &Self,
        _src_l: &Layout,
        _dim: usize,
    ) -> Result<()> {
        todo!()
    }

    fn index_select(&self, ids: &Self, src_l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        let src = LazyStorage::copy(self, src_l);
        let ids = LazyStorage::copy(ids, ids_l);

        // Calculate amount of elements in result, and use to calculate size of output edge buffer.
        let left_size: usize = src_l.dims()[..dim].iter().product();
        let right_size: usize = src_l.dims()[dim + 1..].iter().product();
        let ids_el = ids_l.shape().elem_count();
        // TODO: Instead of using dst_el directly as the layout,
        // perhaps we should calculate actual layout/shape of dst tensor.
        let dst_el = ids_el * left_size * right_size;
        let dst_layout = Layout::contiguous(dst_el);

        let op = Op::IndexSelect(IndexSelect::new(
            src,
            src_l.clone(),
            ids,
            ids_l.clone(),
            dim,
        ));
        Ok(LazyStorage::new(op, &dst_layout, self.dtype()))
    }

    fn index_add(
        &self,
        _l: &Layout,
        _ids: &Self,
        _ids_l: &Layout,
        _src: &Self,
        _src_l: &Layout,
        _dim: usize,
    ) -> Result<Self> {
        todo!()
    }

    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let lhs = LazyStorage::copy(self, lhs_l);
        let rhs = LazyStorage::copy(rhs, rhs_l);

        let op = Op::Matmul(Matmul::new(
            lhs,
            lhs_l.clone(),
            rhs,
            rhs_l.clone(),
            (b, m, n, k),
        ));
        let lhs_dims = lhs_l.shape().dims();
        let dim = lhs_dims.len();
        let dst_shape = Shape::from(&lhs_dims[..dim - 2]).extend(&[m, n]);
        let dst_layout = Layout::contiguous(dst_shape);

        Ok(LazyStorage::new(op, &dst_layout, self.dtype()))
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        let src = LazyStorage::copy(self, src_l);

        if let Op::CopyStridedSrc(copies) = dst.op() {
            let mut copies = copies.clone();
            copies.add(src, src_l.clone(), dst_offset);
            dst.op = Op::CopyStridedSrc(copies).into();
            return Ok(());
        }

        let op = Op::CopyStridedSrc(CopyStridedSrc::new(src, src_l.clone(), dst_offset));
        dst.op = op.into();
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
        if let Op::Copy2D(copies) = dst.op() {
            let mut copies = copies.clone();
            let copy = SingleCopy2D::new(self.clone(), d1, d2, src_s, dst_s, src_o, dst_o);
            copies.add(copy);
            let op = Op::Copy2D(copies);
            dst.op = op.into();
            return Ok(());
        }
        let copy = SingleCopy2D::new(self.clone(), d1, d2, src_s, dst_s, src_o, dst_o);
        let op = Op::Copy2D(Copy2D::new(vec![copy]));

        dst.op = op.into();

        Ok(())
    }

    fn const_set(&mut self, _s: crate::scalar::Scalar, _l: &Layout) -> Result<()> {
        todo!();
    }

    fn upsample_bilinear2d(
        &self,
        _: &Layout,
        _: usize,
        _: usize,
        _: bool,
        _: Option<f64>,
        _: Option<f64>,
    ) -> Result<Self> {
        todo!()
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
        Ok(LazyStorage::new(
            Op::Uninit,
            &Layout::contiguous(shape),
            dtype,
        ))
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
        let op = Op::Const(s.into());
        let storage = LazyStorage::new(op, &Layout::contiguous(shape), dtype);

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

#[cfg(test)]
mod tests {
    use crate::{lazy::LazyDevice, Device, Result, Shape, Tensor, D};

    #[test]
    fn lazy_unary() -> Result<()> {
        let t1 = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0],
            Shape::from(4),
            &Device::Lazy(LazyDevice),
        )?;
        let t1 = t1.sqrt()?.exp()?;
        assert_eq!(
            t1.to_vec1::<f32>()?,
            &[2.718282, 4.1132507, 5.6522336, 7.389056]
        );

        let t2 = Tensor::from_slice(
            &[5.0f32, 6.0, 7.0, 8.0],
            Shape::from(4),
            &Device::Lazy(LazyDevice),
        )?;

        let t2 = t2.sqr()?.affine(0.5, 1.2)?;
        assert_eq!(t2.to_vec1::<f32>()?, &[13.7, 19.2, 25.7, 33.2]);

        //let t3 = t2.powf(1.2)?;
        //assert_eq!(t3.to_vec1::<f32>()?, &[13.7, 19.2, 25.7, 33.2]);

        Ok(())
    }

    #[test]
    fn lazy_concat() -> Result<()> {
        let t1 = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0],
            Shape::from(4),
            &Device::Lazy(LazyDevice),
        )?;
        let t1 = t1.sqrt()?;
        assert_eq!(t1.to_vec1::<f32>()?, &[1.0, 1.4142135, 1.7320508, 2.0]);

        let t2 = Tensor::from_slice(
            &[5.0f32, 6.0, 7.0, 8.0],
            Shape::from(4),
            &Device::Lazy(LazyDevice),
        )?;

        // TODO: bug. copy2d uses first copy data twice _and_ applies affine to it.
        // So even if t2 is correct below, the final result is wrong in two different ways.
        let t2 = t2.affine(0.5, 1.2)?;
        assert_eq!(t2.to_vec1::<f32>()?, &[3.7, 4.2, 4.7, 5.2]);

        let result = Tensor::cat(&[t1, t2], 0)?;

        let result = result.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(
            result,
            &[1.0, 1.4142135, 1.7320508, 2.0, 3.7, 4.2, 4.7, 5.2]
        );
        Ok(())
    }

    #[test]
    fn lazy_binary() -> Result<()> {
        let t1 = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0],
            Shape::from(4),
            &Device::Lazy(LazyDevice),
        )?;

        let t2 = Tensor::from_slice(
            &[5.0f32, 6.0, 7.0, 8.0],
            Shape::from(4),
            &Device::Lazy(LazyDevice),
        )?;

        let result = (t1 + t2)?;

        let result = result.to_vec1::<f32>()?;
        assert_eq!(result, &[6.0, 8.0, 10.0, 12.0]);
        Ok(())
    }

    #[test]
    fn lazy_matmul() -> Result<()> {
        let t1 = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Shape::from((4, 2)),
            &Device::Lazy(LazyDevice),
        )?;

        let t2 = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0],
            Shape::from((2, 2)),
            &Device::Lazy(LazyDevice),
        )?;

        let result = t1.matmul(&t2)?;

        let result = result.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(result, &[7.0, 10.0, 15.0, 22.0, 23.0, 34.0, 31.0, 46.0]);
        Ok(())
    }

    #[test]
    fn lazy_where_cond() -> Result<()> {
        let src = Tensor::from_slice(
            &[0u8, 1, 0, 1, 0, 0, 1, 1],
            Shape::from(8),
            &Device::Lazy(LazyDevice),
        )?;

        let t = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Shape::from(8),
            &Device::Lazy(LazyDevice),
        )?;

        let f = Tensor::from_slice(
            &[9.0f32, 10., 11., 12., 13., 14., 15., 16.],
            Shape::from(8),
            &Device::Lazy(LazyDevice),
        )?;

        let result = src.where_cond(&t, &f)?;
        assert_eq!(
            result.to_vec1::<f32>()?,
            &[9., 2., 11., 4., 13., 14., 7., 8.]
        );
        Ok(())
    }

    #[test]
    fn lazy_reduce() -> Result<()> {
        let t = Tensor::from_slice(
            &[0f32, 1., 2., 3., 4., 3., 2., 1.],
            Shape::from((4, 2)),
            &Device::Lazy(LazyDevice),
        )?;

        let result = t.max(1)?;
        assert_eq!(result.to_vec1::<f32>()?, &[1., 3., 4., 2.]);

        let t = Tensor::from_slice(
            &[0f32, 1., 2., 3.],
            Shape::from((4, 1)),
            &Device::Lazy(LazyDevice),
        )?;

        let result = t.max(0)?;
        assert_eq!(result.to_vec1::<f32>()?, &[3.0]);
        Ok(())
    }

    #[test]
    fn lazy_softmax() -> Result<()> {
        let src = Tensor::from_slice(
            &[0f32, 1., 2., 3., 4., 5., 6., 7.],
            Shape::from(8),
            &Device::Lazy(LazyDevice),
            //&Device::new_metal(0)?,
        )?;

        let t1 = src.affine(1.25, 0.0)?;
        assert_eq!(
            t1.to_vec1::<f32>()?,
            &[0.0, 1.25, 2.5, 3.75, 5.0, 6.25, 7.5, 8.75]
        );
        let t2 = t1.max_keepdim(D::Minus1)?;
        assert_eq!(t2.to_vec1::<f32>()?, &[8.75]);
        let t3 = src.broadcast_sub(&t2)?;
        assert_eq!(
            t3.to_vec1::<f32>()?,
            &[-8.75, -7.75, -6.75, -5.75, -4.75, -3.75, -2.75, -1.75]
        );
        let t4 = t3.exp()?;

        assert_eq!(
            t4.to_vec1::<f32>()?,
            &[
                0.0001584613,
                0.00043074263,
                0.0011708796,
                0.003182782,
                0.008651696,
                0.02351775,
                0.06392787,
                0.17377394
            ]
        );
        let t5 = t4.sum_keepdim(D::Minus1)?;
        assert_eq!(t5.to_vec1::<f32>()?, &[0.27481413]);
        let result = t4.broadcast_div(&t5)?;

        assert_eq!(
            result.to_vec1::<f32>()?,
            &[
                0.00057661266,
                0.0015673962,
                0.0042606234,
                0.011581581,
                0.031481992,
                0.08557693,
                0.23262219,
                0.6323326
            ]
        );
        Ok(())
    }

    #[test]
    fn lazy_rmsnorm() -> Result<()> {
        let device = Device::Lazy(LazyDevice);
        //let device = Device::new_metal(0)?;

        let x = Tensor::from_slice(&[0f32, 1., 2., 3., 4., 5., 6., 7.], Shape::from(8), &device)?;

        let alpha = Tensor::from_slice(
            &[1f32, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
            Shape::from(8),
            &device,
        )?;

        let eps = 1.2;
        let hidden_size = x.dim(D::Minus1)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + eps as f64)?.sqrt()?)?;
        let result = x_normed.broadcast_mul(&alpha)?;

        assert_eq!(
            result.to_vec1::<f32>()?,
            &[0.0, 0.2543735, 0.5549967, 0.90186965, 1.2949923, 1.7343647, 2.219987, 2.7518587]
        );
        Ok(())
    }

    #[test]
    fn lazy_binary_matmul_layout_mismatch_case() -> Result<()> {
        let device = Device::Lazy(LazyDevice);
        //let device = Device::new_metal(0)?;

        // buffer 1: ([128256, 2048], [2048, 1], BF16, BufferId(9))
        // buffer 2: ([1, 2048], [4096, 1], BF16, BufferId(2368))
        // matmul((1, 1, 128256, 2048))
        // dst buffer: ([1, 128256], [128256, 1], BF16, BufferId(2220))

        let x = Tensor::from_slice(
            &[0.0f32, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            (4, 2),
            &device,
        )?;
        let y = Tensor::from_slice(&[1.0f32, 2.0], (2, 1), &device)?;

        let result = x.matmul(&y)?.flatten_all()?;
        assert_eq!(result.to_vec1::<f32>()?, &[2.0, 10.0, 17.0, 23.0]);
        Ok(())
    }
}
