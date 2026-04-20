#![allow(dead_code)]

pub mod custom;
pub mod ops;

use crate::backend::{BackendDevice, BackendStorage};
use crate::lazy::custom::{CustomOp, LazyCustomFn, LazyCustomOp};
use crate::lazy::ops::{
    Affine, Binary, Cmp, Copy2D, CopyStridedSrc, CustomOpContainer, Elu, IndexSelect, LazyOp,
    Matmul, Output, Powf, Reduce, SingleCopy2D, Sink, ToDType, Unary, WhereCond,
};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::tensor_source::TensorSource;
use crate::{CpuStorage, DType, Layout, Result, Shape};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ptr::NonNull;
use std::sync::atomic::AtomicPtr;
use std::sync::{atomic, Arc, LazyLock, Mutex};

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
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }

    pub fn index(&self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BufferId(usize);

impl BufferId {
    fn new() -> Self {
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }

    pub fn inner(&self) -> usize {
        self.0
    }
}

static CUSTOM_OP_REGISTRY: LazyLock<Mutex<HashMap<String, CustomOp>>> =
    LazyLock::new(|| Mutex::new(HashMap::<String, CustomOp>::new()));

#[derive(Debug)]
pub struct AtomicOp {
    ptr: AtomicPtr<Op>,
}

impl AtomicOp {
    fn get_ptr(op: Op) -> *mut Op {
        // Safe way to get raw ptr
        let arc_op = Arc::new(op);
        Arc::into_raw(arc_op) as *mut _
    }

    pub fn new(op: Op) -> AtomicOp {
        AtomicOp {
            ptr: AtomicPtr::new(AtomicOp::get_ptr(op)),
        }
    }

    pub fn swap(&self, new: Op) -> Op {
        let new = AtomicOp::get_ptr(new);
        let old = self.ptr.swap(new, std::sync::atomic::Ordering::SeqCst);
        let value = NonNull::new(old).unwrap();
        unsafe { value.read() }
    }

    pub fn store(&self, new: Op) {
        drop(self.swap(new));
    }

    pub fn load_ref(&self) -> &Op {
        let raw = self.ptr.load(std::sync::atomic::Ordering::Relaxed);
        let value = NonNull::new(raw).unwrap();
        unsafe { value.as_ref() }
    }
}

#[derive(Debug)]
pub struct LazyStorage {
    id: NodeId,
    op: Arc<AtomicOp>,
    pub(crate) custom_op_fallbacks: HashMap<String, LazyStorage>,
    // Layout after applying op. As it is used by subsequent nodes it can be
    // considered the consumer layout.
    layout: Layout,
    // Layout used when applying op, e.g. producer layout.
    // Not always needed, hence optional. When this node was created via `copy()` with
    // a different (e.g. narrowed) layout, this holds the source node's original layout.
    original_layout: Option<Layout>,
    dtype: DType,
    buffer_id: Arc<BufferId>,
    // Weak pointer back to the `Arc<LazyStorage>` in `Storage::Lazy`.
    // Set by `from_storage`. Shared across copies.
    self_arc: Arc<std::sync::OnceLock<std::sync::Weak<LazyStorage>>>,
    // Set to true after the first `optimize()` pass so subsequent calls are no-ops.
    is_optimized: Arc<atomic::AtomicBool>,
    // The lazy device that owns this storage and will evaluate it.
    lazy_device: LazyDevice,
}

impl Clone for LazyStorage {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            op: self.op.clone(),
            custom_op_fallbacks: self.custom_op_fallbacks.clone(),
            layout: self.layout.clone(),
            original_layout: self.original_layout.clone(),
            dtype: self.dtype,
            buffer_id: self.buffer_id.clone(),
            self_arc: self.self_arc.clone(),
            is_optimized: self.is_optimized.clone(),
            lazy_device: self.lazy_device.clone(),
        }
    }
}

impl LazyStorage {
    pub fn new(op: Op, layout: &Layout, dtype: DType, lazy_device: LazyDevice) -> Self {
        Self {
            id: NodeId::new(),
            op: Arc::new(AtomicOp::new(op)),
            custom_op_fallbacks: HashMap::new(),
            layout: layout.clone(),
            original_layout: None,
            dtype,
            buffer_id: Arc::new(BufferId::new()),
            self_arc: Arc::new(std::sync::OnceLock::new()),
            is_optimized: Arc::new(atomic::AtomicBool::new(false)),
            lazy_device,
        }
    }

    pub fn copy(source: &LazyStorage, layout: &Layout) -> Self {
        // Ensures we at least have room for one value in the result.
        let safe_layout_clone = |l: &Layout| {
            if l.dims().is_empty() {
                Layout::contiguous(&[1])
            } else {
                l.clone()
            }
        };
        // Creates a copy, or view, of the lazy storage with updated layout.
        // `producer_layout` preserves the original layout for ops that require it.
        let new_layout = safe_layout_clone(layout);
        let original = source.producer_layout().clone();
        Self {
            id: source.id,
            op: source.op.clone(),
            custom_op_fallbacks: source.custom_op_fallbacks.clone(),
            original_layout: Some(original),
            layout: new_layout,
            dtype: source.dtype,
            buffer_id: source.buffer_id.clone(),
            self_arc: source.self_arc.clone(),
            is_optimized: source.is_optimized.clone(),
            lazy_device: source.lazy_device.clone(),
        }
    }

    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn op(&self) -> &Op {
        self.op.load_ref()
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// The layout to use when producing this node's output.
    ///
    /// For nodes where [`copy`] has updated the layout of a node, this
    /// returns the source [`original_layout`].
    /// Otherwise it is identical to [`layout`].
    pub fn producer_layout(&self) -> &Layout {
        self.original_layout.as_ref().unwrap_or(&self.layout)
    }

    pub fn add_custom_fallback<S: Into<String>>(&mut self, name: S, fallback: LazyStorage) {
        self.custom_op_fallbacks.insert(name.into(), fallback);
    }

    pub fn output(&self) -> Result<Self> {
        // Only add output node if not already present
        if !matches!(self.op(), Op::Output(_)) {
            return Ok(LazyStorage::new(
                Op::Output(Output::new(self.clone())),
                self.layout(),
                self.dtype(),
                self.lazy_device.clone(),
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
            self.lazy_device.clone(),
        ))
    }

    pub fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    pub(crate) fn buffer_id(&self) -> &BufferId {
        &self.buffer_id
    }

    pub fn set_op(&self, op: Op) {
        self.op.store(op);
    }

    /// Store a weak back-reference to the `Arc<LazyStorage>` in `Storage::Lazy`.
    /// Shared across all copies; used by `is_externally_held` and `self_weak`.
    pub(crate) fn set_self_arc(&self, arc: &Arc<LazyStorage>) {
        let _ = self.self_arc.set(Arc::downgrade(arc));
    }

    /// Returns the weak back-reference set by `set_self_arc`, or `None` if not yet set.
    pub(crate) fn self_weak(&self) -> Option<std::sync::Weak<LazyStorage>> {
        self.self_arc.get().cloned()
    }

    /// Returns `true` if the owning tensor is still alive outside the graph.
    pub(crate) fn is_externally_held(&self) -> bool {
        self.self_arc.get().and_then(|w| w.upgrade()).is_some()
    }

    /// `Arc` friendly `const_set`
    pub(crate) fn const_set_lazy(&self, s: crate::scalar::Scalar) {
        self.op.store(Op::ConstSet(s));
    }

    /// `Arc` friendly `copy_strided_src`
    pub(crate) fn copy_strided_src_into(
        &self,
        src: &LazyStorage,
        src_l: &Layout,
        dst_offset: usize,
    ) {
        let src_copy = LazyStorage::copy(src, src_l);
        let new_op = if let Op::CopyStridedSrc(copies) = self.op() {
            let mut copies = copies.clone();
            copies.add(src_copy, dst_offset);
            Op::CopyStridedSrc(copies)
        } else {
            Op::CopyStridedSrc(CopyStridedSrc::new(src_copy, dst_offset))
        };
        self.set_op(new_op);
    }

    /// `Arc` friendly `copy2d`
    pub(crate) fn copy2d_into(
        &self,
        src: &LazyStorage,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o: usize,
    ) {
        let copy = SingleCopy2D::new(src.clone(), d1, d2, src_s, dst_s, src_o, dst_o);
        let new_op = if let Op::Copy2D(copies) = self.op() {
            let mut copies = copies.clone();
            copies.add(copy);
            Op::Copy2D(copies)
        } else {
            Op::Copy2D(Copy2D::new(vec![copy]))
        };
        self.set_op(new_op);
    }

    /// Topological order of operations in the graph from this node
    pub(crate) fn execution_order(&self) -> Vec<&LazyStorage> {
        let mut done = HashSet::new();
        let mut pending = HashSet::new();
        let mut order = Vec::new();

        let mut stack: Vec<(&LazyStorage, usize)> = vec![(self, 0)];
        while let Some((cur_t, cur_src)) = stack.pop() {
            let cur_op = cur_t.op();
            let cur_srcs = cur_op.srcs();
            // Only evaluate nodes once. If already resolved then we do not process.
            let effective_len = cur_srcs.len();
            let all_deps_done = cur_src == effective_len;

            if all_deps_done {
                done.insert(cur_t.id());
                pending.remove(&cur_t.id());
                order.push(cur_t);
                continue;
            }
            let precursor: &LazyStorage = cur_srcs[cur_src];

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

    // Only used by `Dot::graph_fmt` to display the entire graph.
    pub(crate) fn full_execution_order(&self) -> Vec<&LazyStorage> {
        let mut done = HashSet::new();
        let mut pending = HashSet::new();
        let mut order = Vec::new();

        let mut stack: Vec<(&LazyStorage, usize)> = vec![(self, 0)];
        while let Some((cur_t, cur_src)) = stack.pop() {
            let cur_op = cur_t.op();
            let cur_srcs = cur_op.srcs();
            // Only evaluate nodes once. If already resolved then we do not process.
            let effective_len = cur_srcs.len();
            let all_deps_done = cur_src == effective_len;

            if all_deps_done {
                done.insert(cur_t.id());
                pending.remove(&cur_t.id());
                order.push(cur_t);
                continue;
            }
            let precursor: &LazyStorage = cur_srcs[cur_src];

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

    pub(crate) fn resolve(&self) -> Option<Op> {
        if matches!(self.op(), Op::Resolved(_) | Op::Output(_) | Op::Sink(_)) {
            None
        } else {
            Option::Some(self.op.swap(Op::Resolved(self.buffer_id.clone())))
        }
    }
}

impl PartialEq for LazyStorage {
    fn eq(&self, other: &Self) -> bool {
        // Two lazy storages are equal regardless of wether it has been resolved,
        // so we exclude resolved_id from this impl.
        self.id == other.id
            && self.op() == other.op()
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
        let graph = self.0.full_execution_order();
        // output all labels
        let mut max_id = 0;
        let mut visited = HashSet::new();

        for node in graph.iter() {
            let node_id = node.id().index();
            visited.insert(node_id);
            if node_id > max_id {
                max_id = node_id;
            }
            write!(f, "{}{} [ ", INDENT, node.id().index())?;
            writeln!(f, "label = \"{}\"]", node.op())?;
        }
        max_id += 1;

        for node in graph.iter() {
            for source in node.srcs() {
                if !visited.contains(&source.id().index()) {
                    write!(f, "{}{} [ ", INDENT, source.id().index())?;
                    writeln!(f, "label = \"{}\"]", source.op())?;
                }
            }
        }

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
    fn initialize(&mut self, graph: &[&LazyStorage], pinned: &HashSet<BufferId>) -> Result<()>;
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

struct UsageRecord {
    id: Option<BufferId>,
    producer: Option<usize>,
    last_consumer: usize,
    layout: Layout,
    dtype: DType,
}

fn calculate_usage_records(graph: &[&LazyStorage]) -> BTreeMap<BufferId, UsageRecord> {
    let mut records = BTreeMap::new();
    let topo_len = graph.len() - 1;
    for (i, node) in graph.iter().rev().enumerate() {
        let buffer_id = node.buffer_id();

        if node.op().resolved() {
            continue;
        }
        for source in node.srcs() {
            if source.op().resolved() {
                continue;
            }
            let true_source = determine_tensor_source(source);
            records
                .entry(*true_source.buffer_id())
                .or_insert_with(|| UsageRecord {
                    id: None,
                    producer: None,
                    last_consumer: topo_len - i,
                    layout: true_source.producer_layout().clone(),
                    dtype: true_source.dtype(),
                });
        }

        if let Some(record) = records.get_mut(buffer_id) {
            record.id = Some(*buffer_id);
            record.producer = Some(topo_len - i);
            // Use `producer_layout` so that copied nodes (which can be a narrowed view stored as
            // a source inside another op) allocate the full buffer rather than the narrow slice.
            // For regular nodes producer_layout == layout.
            record.layout = node.producer_layout().clone();
            record.dtype = node.dtype();
        }
    }
    // filter records with no producer
    records.retain(|_, v| v.producer.is_some());
    records
}

// https://arxiv.org/pdf/2001.03288.pdf
pub fn greedy_by_size(graph: &[&LazyStorage], pinned: &HashSet<BufferId>) -> Result<MemoryPlan> {
    let mut reusage = BTreeMap::new();
    let mut allocations = BTreeMap::new();

    // Op::Const nodes are excluded. Allocated at runtime.

    // Find buffer usage spans in the graph
    let record_map = calculate_usage_records(graph);

    let mut shared_objects: Vec<BufferId> = Vec::with_capacity(record_map.len());

    let current_size =
        |layout: &Layout, dtype: &DType| layout.shape().elem_count() * dtype.size_in_bytes();

    for (buffer_id, record) in record_map.iter() {
        // Pinned buffers (external references) must never reuse another buffer.
        // They need a dedicated allocation that survives until the post-execution pinning step.
        if pinned.contains(buffer_id) {
            allocations.insert(*buffer_id, (record.layout.clone(), record.dtype));
            // Skipping since pinned should not be in shared_objects
            continue;
        }

        // Debugging: disables reuse to investigate correctness
        if std::env::var("CANDLE_LAZY_NO_REUSE").is_ok() {
            allocations.insert(*buffer_id, (record.layout.clone(), record.dtype));
            shared_objects.push(*buffer_id);
            continue;
        }

        let record_producer = record.producer.unwrap();
        let needed_size = current_size(&record.layout, &record.dtype);
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
            for (inner_buffer_id, inner) in record_map.iter() {
                let max_first = std::cmp::max(record_producer, inner.producer.unwrap());
                let min_last = std::cmp::min(record.last_consumer, inner.last_consumer);
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
                    .is_none_or(|(_, best_size)| obj_size < *best_size)
                {
                    best_buffer = Some((*obj, obj_size));
                }
            }
        }
        if let Some((best, _)) = best_buffer {
            reusage.insert(*buffer_id, best);
        } else {
            allocations.insert(*buffer_id, (record.layout.clone(), record.dtype));
            shared_objects.push(*buffer_id);
        }
    }

    let output = graph.last().unwrap();
    // Resolved nodes already have a buffer saved in the resolved_map; initialize() will
    // restore it.  Do not add them to allocations here or initialize() would overwrite the
    // resolved buffer with a freshly-allocated (uninitialized) one.
    if !output.op().resolved() {
        let source = determine_tensor_source(output);
        allocations.insert(
            *output.buffer_id(),
            (source.producer_layout().clone(), source.dtype()),
        );
    }

    Ok(MemoryPlan {
        reusage,
        allocations,
    })
}

/// Allocation plan for a lazy forward pass.
///
/// Stored after the first run and reused on subsequent runs with identical graph structure,
/// skipping `greedy_by_size` entirely.
///
/// Rather than caching the physical buffer objects, we cache the aliasing structure, denoting
/// which topological order position is the owner of each buffer.
/// This requires that the topological ordering is stable between runs.
///
/// The fast path allocates new pool buffers every token (may already be allocated), applying the same aliasing,
/// so every kernel always writes before it reads.
#[derive(Debug, Clone)]
pub struct CachedPlan {
    /// Structural hash: op type discriminant at each topological position.
    /// Ensures stability even as tensor shapes grow.
    pub key: u64,
    /// Canonical positions (position owns a buffer)
    /// Given `let Some(value) = canonical_position[pos]`, if value is
    ///   `value == pos`: canonical node. Allocate a fresh buffer
    ///   `value < pos` : alias node. Share buffer allocated for this position.
    ///   `None`        : Const, Resolved, or output node. Handled per run.
    pub canonical_posisitons: Vec<Option<usize>>,
}
fn apply_backend_agnostic_passes(
    _node: &LazyStorage,
    _consumer_map: &HashMap<NodeId, Vec<&LazyStorage>>,
) -> bool {
    // let mut changed = false;
    // for example:
    // changed |= fuse_elementwise_pass(node);
    // changed
    false
}

pub trait Executor {
    type BufferType: LazyBuffer;
    type AllocatorType: LazyAllocator<Self::BufferType>;

    /// Optimization loop
    /// Applies both backend agnostic and backend specific passes
    fn optimize(&self, leaf: &LazyStorage) {
        // Already optimized on a previous run - skip
        if leaf.is_optimized.load(atomic::Ordering::Relaxed) {
            return;
        }
        let mut worklist: VecDeque<&LazyStorage> = VecDeque::new();
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut consumer_map: HashMap<NodeId, Vec<&LazyStorage>> = HashMap::new();

        visited.insert(leaf.id());
        worklist.push_back(leaf);

        while let Some(node) = worklist.pop_front() {
            let mut changed = apply_backend_agnostic_passes(node, &consumer_map);
            changed |= self.apply_passes(node, &consumer_map);
            for source in node.srcs() {
                consumer_map
                    .entry(source.id())
                    .and_modify(|c| c.push(node))
                    .or_insert_with(|| vec![node]);
                if visited.insert(source.id()) {
                    worklist.push_back(source);
                }
            }
            if changed {
                let consumers = consumer_map
                    .get(&node.id())
                    .map(|c| c.as_slice())
                    .unwrap_or_else(|| &[]);
                for consumer in consumers {
                    worklist.push_back(consumer);
                }
            }
        }
        leaf.is_optimized.store(true, atomic::Ordering::Relaxed);
    }

    /// Backend specific optimizations. Defaults to noop.
    fn apply_passes(
        &self,
        _node: &LazyStorage,
        _consumer_map: &HashMap<NodeId, Vec<&LazyStorage>>,
    ) -> bool {
        false
    }

    fn run(&self, lazy_storage: LazyStorage) -> Result<Self::BufferType>;

    fn eval(&self, lazy_storage: &LazyStorage, allocator: &mut Self::AllocatorType) -> Result<()>;

    fn allocator(&self) -> Self::AllocatorType;

    fn get_specialized_op(
        &self,
        _op: &dyn LazyCustomOp,
    ) -> Option<Box<dyn LazyCustomFn<Self::BufferType>>> {
        None
    }

    fn install_custom_op(
        &mut self,
        op: &dyn LazyCustomOp,
        func: Box<dyn LazyCustomFn<Self::BufferType>>,
    ) -> Option<&dyn LazyCustomFn<Self::BufferType>>;

    fn inner_precision(op: &Op) -> Option<DType>;

    /// Hash key of graph.
    /// Hashing Op variant discriminants so the key is stable even
    /// as tensor shapes grow between tokens.
    /// This lets us cache plans and optimizations.
    #[inline(always)]
    fn graph_key(graph: &[&LazyStorage]) -> u64 {
        let mut h = (graph.len() as u64).saturating_mul(0x9e3779b97f4a7c15_u64);
        for (i, node) in graph.iter().enumerate() {
            let disc: u64 = match node.op() {
                Op::Uninit => 0,
                Op::Const(_) => 1,
                Op::Resolved(_) => 2,
                Op::ToCpu => 3,
                Op::Affine(_) => 4,
                Op::Powf(_) => 5,
                Op::Elu(_) => 6,
                Op::Reduce(_) => 7,
                Op::Cmp(_) => 8,
                Op::ToDType(_) => 9,
                Op::Unary(_) => 10,
                Op::Binary(_) => 11,
                Op::WhereCond(_) => 12,
                Op::IndexSelect(_) => 13,
                Op::Matmul(_) => 14,
                Op::CopyStridedSrc(_) => 15,
                Op::Copy2D(_) => 16,
                Op::ConstSet(_) => 17,
                Op::Sink(_) => 18,
                Op::Aggregate => 19,
                Op::CustomOp(_) => 20,
                Op::Output(_) => 21,
            };
            h ^= disc
                .wrapping_mul(0x9e3779b97f4a7c15_u64)
                .wrapping_add(i as u64);
        }
        h
    }

    /// Collect buffer ids of nodes that are externally held or already resolved.
    /// Pinned buffers indicate they should be excluded from buffer reusage.
    fn get_pinned_buffers(graph: &[&LazyStorage]) -> HashSet<BufferId> {
        graph
            .iter()
            .filter(|n| n.is_externally_held() && !matches!(n.op(), Op::Resolved(_)))
            .map(|n| *n.buffer_id())
            .collect()
    }
}

impl LazyStorage {
    pub fn execute<E: Executor>(&self, executor: E) -> Result<E::BufferType> {
        // TODO: Apply backend agnostic optimizations
        executor.run(self.clone())
    }
}

#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum Op {
    Uninit,
    Const(Arc<dyn TensorSource>),
    Resolved(Arc<BufferId>),
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

impl PartialEq for Op {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Const(a), Self::Const(b)) => Arc::ptr_eq(a, b),
            (Self::Resolved(a), Self::Resolved(b)) => a == b,
            (Self::Affine(a), Self::Affine(b)) => a == b,
            (Self::Powf(a), Self::Powf(b)) => a == b,
            (Self::Elu(a), Self::Elu(b)) => a == b,
            (Self::Reduce(a), Self::Reduce(b)) => a == b,
            (Self::Cmp(a), Self::Cmp(b)) => a == b,
            (Self::ToDType(a), Self::ToDType(b)) => a == b,
            (Self::Unary(a), Self::Unary(b)) => a == b,
            (Self::Binary(a), Self::Binary(b)) => a == b,
            (Self::WhereCond(a), Self::WhereCond(b)) => a == b,
            (Self::IndexSelect(a), Self::IndexSelect(b)) => a == b,
            (Self::Matmul(a), Self::Matmul(b)) => a == b,
            (Self::CopyStridedSrc(a), Self::CopyStridedSrc(b)) => a == b,
            (Self::Copy2D(a), Self::Copy2D(b)) => a == b,
            (Self::ConstSet(a), Self::ConstSet(b)) => a == b,
            (Self::Sink(a), Self::Sink(b)) => a == b,
            (Self::CustomOp(a), Self::CustomOp(b)) => a == b,
            (Self::Output(a), Self::Output(b)) => a == b,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

impl Op {
    pub fn srcs(&self) -> Vec<&LazyStorage> {
        match self {
            Op::Uninit => vec![],
            Op::Const(_) => vec![],
            Op::Resolved(_) => vec![],
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

    pub fn resolved(&self) -> bool {
        matches!(self, Op::Resolved(_))
    }
}

impl Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Op::Uninit => write!(f, "Uninit"),
            Op::Const(_) => write!(f, "Const"),
            Op::Resolved(buffer_id) => write!(f, "Resolved({})", buffer_id.inner()),
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
    pub fn srcs(&self) -> Vec<&LazyStorage> {
        self.op().srcs()
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
        &self.lazy_device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        todo!()
    }

    fn affine(&self, l: &Layout, mul: f64, add: f64) -> Result<Self> {
        let op = Op::Affine(Affine::new(self.clone(), mul, add));
        Ok(LazyStorage::new(
            op,
            l,
            self.dtype(),
            self.lazy_device.clone(),
        ))
    }

    fn powf(&self, l: &Layout, exp: f64) -> Result<Self> {
        let op = Op::Powf(Powf::new(self.clone(), exp));
        Ok(LazyStorage::new(
            op,
            l,
            self.dtype(),
            self.lazy_device.clone(),
        ))
    }

    fn elu(&self, l: &Layout, alpha: f64) -> Result<Self> {
        let op = Op::Elu(Elu::new(self.clone(), alpha));
        Ok(LazyStorage::new(
            op,
            l,
            self.dtype(),
            self.lazy_device.clone(),
        ))
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
        Ok(LazyStorage::new(
            op,
            &dst_layout,
            dtype,
            self.lazy_device.clone(),
        ))
    }

    fn cmp(&self, cmp_op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        let lhs = LazyStorage::copy(self, lhs_l);
        let rhs = LazyStorage::copy(rhs, rhs_l);
        let dst_l = Layout::contiguous(lhs.shape());
        let op = Op::Cmp(Cmp::new(lhs, rhs, cmp_op));
        Ok(LazyStorage::new(
            op,
            &dst_l,
            DType::U8,
            self.lazy_device.clone(),
        ))
    }

    fn to_dtype(&self, l: &Layout, dtype: DType) -> Result<Self> {
        let src = LazyStorage::copy(self, l);
        if self.dtype() != dtype {
            let l = src.layout().clone();
            let op = Op::ToDType(ToDType::new(src.clone(), dtype));
            return Ok(LazyStorage::new(op, &l, dtype, self.lazy_device.clone()));
        }
        Ok(src)
    }

    fn unary_impl<B: UnaryOpT>(&self, l: &Layout) -> Result<Self> {
        let src = LazyStorage::copy(self, l);
        let op = Op::Unary(Unary::new(src, B::KERNEL));
        Ok(LazyStorage::new(
            op,
            l,
            self.dtype(),
            self.lazy_device.clone(),
        ))
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let lhs = LazyStorage::copy(self, lhs_l);
        let rhs = LazyStorage::copy(rhs, rhs_l);
        let dst_l = Layout::contiguous(lhs_l.shape());
        let op = Op::Binary(Binary::new(lhs, rhs, B::KERNEL));
        Ok(LazyStorage::new(
            op,
            &dst_l,
            self.dtype(),
            self.lazy_device.clone(),
        ))
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
        let f_l = f.layout().clone();
        let dtype = f.dtype();

        let op = Op::WhereCond(WhereCond::new(src, t, f));
        Ok(LazyStorage::new(op, &f_l, dtype, self.lazy_device.clone()))
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

        let op = Op::IndexSelect(IndexSelect::new(src, ids, dim));
        Ok(LazyStorage::new(
            op,
            &dst_layout,
            self.dtype(),
            self.lazy_device.clone(),
        ))
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

        let op = Op::Matmul(Matmul::new(lhs, rhs, (b, m, n, k)));
        let lhs_dims = lhs_l.shape().dims();
        let dim = lhs_dims.len();
        let dst_shape = Shape::from(&lhs_dims[..dim - 2]).extend(&[m, n]);
        let dst_layout = Layout::contiguous(dst_shape);

        Ok(LazyStorage::new(
            op,
            &dst_layout,
            self.dtype(),
            self.lazy_device.clone(),
        ))
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        dst.copy_strided_src_into(self, src_l, dst_offset);
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
        dst.copy2d_into(self, d1, d2, src_s, dst_s, src_o, dst_o);
        Ok(())
    }

    fn const_set(&mut self, s: crate::scalar::Scalar, _l: &Layout) -> Result<()> {
        self.const_set_lazy(s);
        Ok(())
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LazyDevice {
    ordinal: usize,
}

impl LazyDevice {
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }
}

impl BackendDevice for LazyDevice {
    type Storage = LazyStorage;

    fn new(ordinal: usize) -> Result<Self> {
        Ok(Self { ordinal })
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Lazy
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.ordinal == rhs.ordinal
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        Ok(LazyStorage::new(
            Op::Uninit,
            &Layout::contiguous(shape),
            dtype,
            self.clone(),
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
        let op = Op::Const(Arc::new(s) as Arc<dyn TensorSource>);
        let storage = LazyStorage::new(op, &Layout::contiguous(shape), dtype, self.clone());

        Ok(storage)
    }

    fn storage_from_tensor_source<S: TensorSource + 'static>(
        &self,
        source: S,
    ) -> Result<Self::Storage> {
        let element_count = source.element_count();
        let dtype = source.data_type();
        let op = Op::Const(Arc::new(source) as Arc<dyn TensorSource>);
        let storage = LazyStorage::new(
            op,
            &Layout::contiguous(Shape::from(element_count)),
            dtype,
            self.clone(),
        );
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
    use crate::backend::BackendDevice;
    use crate::{lazy::LazyDevice, DType, Device, Result, Shape, Tensor, D};

    fn run_cmp<F>(mut f: F, devices: &[&Device]) -> Result<()>
    where
        F: FnMut(&Device) -> Result<Tensor>,
    {
        assert!(
            devices.len() >= 2,
            "Requires at least 2 devices to compare results"
        );
        let mut results: Vec<Tensor> = vec![];
        for d in devices {
            results.push(f(d)?);
        }
        for w in results.windows(2) {
            cmp_tensors(&w[0], &w[1])?
        }
        Ok(())
    }

    fn cmp_tensors(a: &Tensor, b: &Tensor) -> Result<()> {
        let a_device = a.device();
        let b_device = b.device();

        assert!(
            !a_device.same_device(b_device),
            "Test not configured correctly, both devices are {a_device:?}"
        );

        let a_vals = a.flatten_all()?.to_vec1::<f32>()?;
        let b_vals = b.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(
            a_vals.len(),
            b_vals.len(),
            "shape mismatch: {a_device:?}={:?} {b_device:?}={:?}",
            a.shape(),
            b.shape()
        );
        for (i, (l, e)) in a_vals.iter().zip(b_vals.iter()).enumerate() {
            let diff = (l - e).abs();
            assert!(
                diff < 1e-3,
                "element {i}: {a_device:?}={l} {b_device:?}={e} diff={diff} ({a_device:?} shape={:?})",
                a.shape()
            );
        }
        Ok(())
    }

    #[test]
    fn lazy_unary() -> Result<()> {
        let unary = |device: &Device| {
            let t1 = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], Shape::from(4), device)?;
            let t1 = t1.sqrt()?.exp()?;
            assert_eq!(
                t1.to_vec1::<f32>()?,
                &[2.718282, 4.1132507, 5.6522336, 7.389056]
            );

            let t2 = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], Shape::from(4), device)?;

            let t2 = t2.sqr()?.affine(0.5, 1.2)?;
            assert_eq!(t2.to_vec1::<f32>()?, &[13.7, 19.2, 25.7, 33.2]);

            //let t3 = t2.powf(1.2)?;
            //assert_eq!(t3.to_vec1::<f32>()?, &[13.7, 19.2, 25.7, 33.2]);

            Ok(t2)
        };

        run_cmp(
            unary,
            &[&Device::Lazy(LazyDevice::new(0)?), &Device::new_metal(0)?],
        )?;

        Ok(())
    }

    #[test]
    fn lazy_concat() -> Result<()> {
        let concat = |device: &Device| {
            let t1 = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], Shape::from(4), device)?;
            let t1 = t1.sqrt()?;
            assert_eq!(t1.to_vec1::<f32>()?, &[1.0, 1.4142135, 1.7320508, 2.0]);

            let t2 = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], Shape::from(4), device)?;

            let t2 = t2.affine(0.5, 1.2)?;
            assert_eq!(t2.to_vec1::<f32>()?, &[3.7, 4.2, 4.7, 5.2]);

            let t3 = Tensor::cat(&[t1, t2], 0)?;

            let result = t3.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(
                result,
                &[1.0, 1.4142135, 1.7320508, 2.0, 3.7, 4.2, 4.7, 5.2]
            );

            Ok(t3)
        };

        run_cmp(
            concat,
            &[&Device::Lazy(LazyDevice::new(0)?), &Device::new_metal(0)?],
        )?;

        Ok(())
    }

    #[test]
    fn lazy_binary() -> Result<()> {
        let binary = |device: &Device| {
            let t1 = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], Shape::from(4), device)?;
            let t2 = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], Shape::from(4), device)?;

            t1 + t2
        };

        run_cmp(
            binary,
            &[&Device::Lazy(LazyDevice::new(0)?), &Device::new_metal(0)?],
        )?;

        Ok(())
    }

    #[test]
    fn lazy_matmul() -> Result<()> {
        let matmul = |device: &Device| {
            let t1 = Tensor::from_slice(
                &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                Shape::from((4, 2)),
                device,
            )?;

            let t2 = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], Shape::from((2, 2)), device)?;

            t1.matmul(&t2)
        };

        run_cmp(
            matmul,
            &[&Device::Lazy(LazyDevice::new(0)?), &Device::new_metal(0)?],
        )?;

        Ok(())
    }

    #[test]
    fn lazy_where_cond() -> Result<()> {
        let where_cond = |device: &Device| {
            let src = Tensor::from_slice(&[0u8, 1, 0, 1, 0, 0, 1, 1], Shape::from(8), device)?;

            let t = Tensor::from_slice(
                &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                Shape::from(8),
                device,
            )?;

            let f = Tensor::from_slice(
                &[9.0f32, 10., 11., 12., 13., 14., 15., 16.],
                Shape::from(8),
                device,
            )?;

            let result = src.where_cond(&t, &f)?;
            assert_eq!(
                result.to_vec1::<f32>()?,
                &[9., 2., 11., 4., 13., 14., 7., 8.]
            );

            Ok(result)
        };

        run_cmp(
            where_cond,
            &[&Device::Lazy(LazyDevice::new(0)?), &Device::new_metal(0)?],
        )?;

        Ok(())
    }

    #[test]
    fn lazy_reduce() -> Result<()> {
        let reduce = |device: &Device| {
            let t = Tensor::from_slice(
                &[0f32, 1., 2., 3., 4., 3., 2., 1.],
                Shape::from((4, 2)),
                device,
            )?;

            let result = t.max(1)?;
            assert_eq!(result.to_vec1::<f32>()?, &[1., 3., 4., 2.]);

            let t = Tensor::from_slice(&[0f32, 1., 2., 3.], Shape::from((4, 1)), device)?;

            t.max(0)
        };

        run_cmp(
            reduce,
            &[&Device::Lazy(LazyDevice::new(0)?), &Device::new_metal(0)?],
        )?;

        Ok(())
    }

    #[test]
    fn lazy_softmax() -> Result<()> {
        let softmax = |device: &Device| {
            let src =
                Tensor::from_slice(&[0f32, 1., 2., 3., 4., 5., 6., 7.], Shape::from(8), device)?;

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
            Ok(result)
        };

        run_cmp(
            softmax,
            &[&Device::Lazy(LazyDevice::new(0)?), &Device::new_metal(0)?],
        )?;

        Ok(())
    }

    #[test]
    fn lazy_rmsnorm() -> Result<()> {
        let rmsnorm = |device: &Device| {
            let x =
                Tensor::from_slice(&[0f32, 1., 2., 3., 4., 5., 6., 7.], Shape::from(8), device)?;

            let alpha = Tensor::from_slice(
                &[1f32, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
                Shape::from(8),
                device,
            )?;

            let eps = 1.2;
            let hidden_size = x.dim(D::Minus1)?;
            let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
            let x_normed = x.broadcast_div(&(norm_x + eps)?.sqrt()?)?;
            let result = x_normed.broadcast_mul(&alpha)?;

            assert_eq!(
                result.to_vec1::<f32>()?,
                &[
                    0.0, 0.2543735, 0.5549967, 0.90186965, 1.2949923, 1.7343647, 2.219987,
                    2.7518587
                ]
            );

            Ok(result)
        };

        run_cmp(
            rmsnorm,
            &[&Device::Lazy(LazyDevice::new(0)?), &Device::new_metal(0)?],
        )?;

        Ok(())
    }

    #[test]
    fn lazy_to_dtype() -> Result<()> {
        let dtype = |device: &Device| {
            let data: Vec<f32> = (0..8).map(|x| x as f32 * 0.1).collect();
            Tensor::from_slice(&data, 8, device)?
                .to_dtype(DType::BF16)?
                .to_dtype(DType::F32)
        };

        run_cmp(
            dtype,
            &[&Device::Lazy(LazyDevice::new(0)?), &Device::new_metal(0)?],
        )?;

        Ok(())
    }

    #[test]
    fn lazy_index_select() -> Result<()> {
        let index_select = |device: &Device| {
            let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
            let ids = vec![2u32, 0, 3, 1];
            Tensor::from_slice(&data, (4, 4), device)?
                .index_select(&Tensor::from_slice(&ids, 4, device)?, 0)
        };

        run_cmp(
            index_select,
            &[&Device::Lazy(LazyDevice::new(0)?), &Device::new_metal(0)?],
        )?;

        Ok(())
    }

    #[test]
    fn lazy_reshape_matmul() -> Result<()> {
        let reshape_matmul = |device: &Device| {
            // Basic Q/K/V projection + reshape
            // x: [1, seq, hidden] @ w: [hidden, 3 * head_dim] -> [1, seq, 3 * head_dim]
            let (seq, hidden, heads, head_dim) = (2usize, 4usize, 2usize, 2usize);

            let x_data: Vec<f32> = (0..seq * hidden).map(|i| i as f32 * 0.1).collect();
            let w_data: Vec<f32> = (0..hidden * heads * head_dim)
                .map(|i| (i as f32 + 1.0) * 0.1)
                .collect();

            let x = Tensor::from_slice(&x_data, (1, seq, hidden), device)?;
            let w = Tensor::from_slice(&w_data, (hidden, heads * head_dim), device)?;

            // Flatten batch + seq for matmul
            x.flatten(0, 1)?
                .matmul(&w)?
                .reshape((1, seq, heads, head_dim))
        };

        run_cmp(
            reshape_matmul,
            &[&Device::Lazy(LazyDevice::new(0)?), &Device::new_metal(0)?],
        )?;
        Ok(())
    }

    #[test]
    fn lazy_to_dtype_matmul() -> Result<()> {
        let bf16_matmul = |device: &Device| {
            // to_dtype(bf16) -> matmul
            let x_data: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
            let w_data: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) * 0.1).collect();

            // x: [1, 4], w: [4, 2] in f32, cast to bf16, matmul
            let x = Tensor::from_slice(&x_data, (1, 4), device)?.to_dtype(DType::BF16)?;
            let w = Tensor::from_slice(&w_data, (4, 2), device)?.to_dtype(DType::BF16)?;

            x.matmul(&w)?.to_dtype(DType::F32)
        };

        run_cmp(
            bf16_matmul,
            &[&Device::Lazy(LazyDevice::new(0)?), &Device::new_metal(0)?],
        )?;
        Ok(())
    }

    #[test]
    fn lazy_narrow_unary() -> Result<()> {
        let narrow_unary = |device: &Device| {
            let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
            let t = Tensor::from_slice(&data, (4, 4), device)?;
            // narrow + affine
            t.narrow(0, 1, 2)?.affine(2.0, 0.5)
        };

        run_cmp(
            narrow_unary,
            &[&Device::Lazy(LazyDevice::new(0)?), &Device::new_metal(0)?],
        )?;
        Ok(())
    }

    #[test]
    fn lazy_narrow_mul() -> Result<()> {
        let narrow_broadcast_mul = |device: &Device| {
            let (b, h, s, d) = (1usize, 2usize, 2usize, 4usize);
            let x_data: Vec<f32> = (0..b * h * s * d).map(|i| i as f32 * 0.1).collect();

            let x = Tensor::from_slice(&x_data, (b, h, s, d), device)?;

            // Narrow #1
            let x1 = x.narrow(3, 0, d / 2)?;
            // Narrow #2
            let x2 = x.narrow(3, d / 2, d / 2)?;

            // Test op after narrow
            x1 * x2
        };

        run_cmp(
            narrow_broadcast_mul,
            &[&Device::Lazy(LazyDevice::new(0)?), &Device::new_metal(0)?],
        )?;
        Ok(())
    }

    #[test]
    fn lazy_rope_like() -> Result<()> {
        // Mimic the rotary embedding operation core:
        //   x = [b, h, s, d/2, 2]
        //   x_r, x_i = x.chunk(2, last_dim)
        //   cos, sin = narrow from table
        //   result = [x_r * cos - x_i * sin, x_r * sin + x_i * cos]

        let (b, h, s, d) = (1usize, 2usize, 2usize, 4usize);
        let x_data: Vec<f32> = (0..b * h * s * d).map(|i| i as f32 * 0.1).collect();
        let cos_data: Vec<f32> = (0..s * (d / 2)).map(|i| (i as f32 * 0.3).cos()).collect();
        let sin_data: Vec<f32> = (0..s * (d / 2)).map(|i| (i as f32 * 0.3).sin()).collect();

        fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
            let (b, h, s, d) = x.dims4()?;
            let x1 = x.narrow(3, 0, d / 2)?;
            let x2 = x.narrow(3, d / 2, d / 2)?;
            let cos = cos.broadcast_as((b, h, s, d / 2))?;
            let sin = sin.broadcast_as((b, h, s, d / 2))?;
            let r1 = (x1
                .broadcast_mul(&cos)?
                .broadcast_sub(&x2.broadcast_mul(&sin)?))?;
            let r2 = (x1
                .broadcast_mul(&sin)?
                .broadcast_add(&x2.broadcast_mul(&cos)?))?;
            Tensor::cat(&[r1, r2], 3)
        }

        let rope_like = |device: &Device| {
            let x = Tensor::from_slice(&x_data, (b, h, s, d), device)?.to_dtype(DType::BF16)?;
            let cos = Tensor::from_slice(&cos_data, (s, d / 2), device)?
                .to_dtype(DType::BF16)?
                .unsqueeze(0)?;
            let sin = Tensor::from_slice(&sin_data, (s, d / 2), device)?
                .to_dtype(DType::BF16)?
                .unsqueeze(0)?;

            rope(&x, &cos, &sin)?.to_dtype(DType::F32)
        };

        run_cmp(
            rope_like,
            &[&Device::Lazy(LazyDevice::new(0)?), &Device::new_metal(0)?],
        )?;
        Ok(())
    }
}
