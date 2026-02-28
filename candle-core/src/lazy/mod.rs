use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape, Tensor};
use petgraph::graph::{DiGraph, Edge, EdgeIndex, EdgeReference, IndexType, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction::{Incoming, Outgoing};
use petgraph::{EdgeType, Graph};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::sync::atomic;
use std::sync::Arc;

static OP_COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);

fn count() -> usize {
    OP_COUNTER.fetch_add(1, atomic::Ordering::Relaxed)
}
fn get_count() -> usize {
    OP_COUNTER.load(atomic::Ordering::Relaxed)
}

#[derive(thiserror::Error, Debug, Clone)]
pub enum LazyError {
    #[error("{0}")]
    Message(String),

    #[error("Incorrect incoming edges to node {0:?}")]
    InvalidIncoming(NodeId),

    #[error("Incorrect outgoing edges to node {0:?}")]
    InvalidOutgoing(NodeId),

    #[error("Edge {0:?} expected to have state")]
    InvalidEdgeState(EdgeId),

    #[error("No buffer found for {0:?}")]
    BufferNotFound(BufferId),
}

impl From<String> for LazyError {
    fn from(e: String) -> Self {
        LazyError::Message(e)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LazyStorageId(usize);

impl LazyStorageId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EdgeId(usize);

impl EdgeId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferId(usize);

impl BufferId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

pub type OpGraph = DiGraph<OpNode, OpEdge>;

pub fn update_all_outgoing(
    g: &mut OpGraph,
    node: NodeIndex<u32>,
    buffer_id: BufferId,
) -> Result<()> {
    let edges: Vec<(NodeIndex, NodeIndex, OpEdge)> = g
        .edges_directed(node, Outgoing)
        .map(|e| (e.source(), e.target(), e.weight().clone()))
        .collect();

    if edges.is_empty() {
        todo!("Add error type for this state")
        //return Err(LazyError::InvalidOutgoing(node));
    }
    for (source, target, mut weight) in edges {
        weight.buffer_id = buffer_id;
        g.update_edge(source, target, weight);
    }
    Ok(())
}

pub struct Ancestors<'a, N, E, D, Idx>
where
    E: 'a,
    Idx: 'a + IndexType,
    D: EdgeType,
{
    graph: &'a Graph<N, E, D, Idx>,
    edges: &'a [Edge<E, Idx>],
    visited: HashSet<EdgeIndex<Idx>>,
    to_visit: VecDeque<EdgeIndex<Idx>>,
}

impl<'a, N, E, D, Idx> Ancestors<'a, N, E, D, Idx>
where
    E: 'a,
    Idx: 'a + IndexType,
    D: EdgeType,
{
    pub fn of(g: &'a Graph<N, E, D, Idx>, start: NodeIndex<Idx>) -> Self
    where
        D: EdgeType,
    {
        Ancestors {
            graph: &g,
            edges: g.raw_edges(),
            visited: HashSet::new(),
            to_visit: g.edges_directed(start, Incoming).map(|e| e.id()).collect(),
        }
    }
}

impl<'a, N, E, D, Idx> Iterator for Ancestors<'a, N, E, D, Idx>
where
    E: 'a,
    Idx: 'a + IndexType,
    D: EdgeType,
{
    type Item = EdgeIndex<Idx>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(e) = self.to_visit.pop_front() {
            if self.visited.contains(&e) {
                continue;
            }
            self.visited.insert(e);
            let edge = &self.edges[e.index()];

            let source = edge.source();
            self.to_visit
                .extend(self.graph.edges_directed(source, Incoming).map(|e| e.id()));
            return Some(e);
        }
        None
    }
}

// Return a subgraph which includes all nodes and edges that are ascendants of the provided node index
// TODO: pub fn ancestors<N, E>(g: &DiGraph<N, E>, node: NodeIndex<u32>) -> DiGraph<N, E> {
pub fn ancestors(g: &OpGraph, node: NodeIndex<u32>) -> OpGraph {
    let ancestors_iter = Ancestors::of(g, node);
    let mut ancestors = OpGraph::new();
    let mut idx_map = HashMap::new();

    ancestors_iter.for_each(|e| {
        let e = &g.raw_edges()[e.index()];
        let source = e.source();
        let target = e.target();
        let edge_weight = e.weight.clone();

        let source_id = idx_map
            .entry(source)
            .or_insert_with(|| {
                let w = g.node_weight(source).unwrap().clone();
                ancestors.add_node(w)
            })
            .clone();

        let target_id = idx_map
            .entry(target)
            .or_insert_with(|| {
                let w = g.node_weight(target).unwrap().clone();
                ancestors.add_node(w)
            })
            .clone();

        ancestors.add_edge(source_id, target_id, edge_weight);
    });
    ancestors
}

pub fn get_incoming_edges<N, E>(
    g: &DiGraph<N, E>,
    idx: NodeIndex<u32>,
    expected: Option<usize>,
) -> Vec<EdgeReference<'_, E>> {
    let edges = g.edges_directed(idx, Incoming);
    let result: Vec<EdgeReference<E>> = edges.collect();

    if let Some(n) = expected {
        // TODO: Return Error instead
        assert_eq!(result.len(), n);
    }

    result
}

#[derive(Debug, Clone)]
pub struct LazyStorage {
    id: LazyStorageId,
    operations: OpGraph,
    layout: Layout,
    initial_dtype: DType,
    current_node: Option<NodeIndex<u32>>,
    // potentially Arc<RwLock<...>>
    //inner: Option<CpuStorage>,
}

pub trait LazyCustomOp1 {
    fn name(&self) -> &'static str;

    fn lazy_fwd(&self, _: &LazyStorage, _: &Layout) -> Result<(LazyStorage, Shape)>;

    fn fallback(&self) -> Result<crate::Tensor> {
        Err(crate::Error::Msg(
            format!("no lazy fallback for {}", self.name()).into(),
        ))
    }
}

pub trait LazyCustomOp2 {
    fn name(&self) -> &'static str;

    fn lazy_fwd(
        &self,
        _: &LazyStorage,
        _: &Layout,
        _: &LazyStorage,
        _: &Layout,
    ) -> Result<(LazyStorage, Shape)>;

    fn fallback(&self, _: &Tensor, _: &Tensor) -> Result<Tensor> {
        Err(crate::Error::Msg(
            format!("no lazy fallback for {}", self.name()).into(),
        ))
    }
}

impl LazyStorage {
    pub fn get_current_node(&self) -> Result<NodeIndex<u32>> {
        self.current_node
            .ok_or(LazyError::Message("No current node in lazy storage".to_string()).into())
    }

    pub fn output(&self) -> Result<Self> {
        count();

        let mut next = self.clone();
        let current_node = next.get_current_node()?;

        let current_op = self.operations.node_weight(current_node).unwrap();
        // Only add output node if not already present
        if !matches!(current_op.op(), Op::Output) {
            let idx = next.add_operation(Op::Output);
            let edge = OpEdge::new(self.layout.clone(), self.dtype());
            next.operations.add_edge(current_node, idx, edge);
        };

        Ok(next)
    }

    pub fn custom_op(&self, _: Box<dyn LazyCustomOp1>) -> Result<Self> {
        count();

        let mut next = self.clone();
        let idx = next.add_operation(Op::Sink);
        let current_op = next.get_current_node()?;

        let previous_edge = next.operations.first_edge(current_op, Incoming).unwrap();
        let previous_edge = next.operations.edge_weight(previous_edge).unwrap();

        let edge = OpEdge::new(previous_edge.layout().clone(), self.dtype());
        next.operations.add_edge(current_op, idx, edge);

        Ok(next)
    }

    pub fn operations(&self) -> &OpGraph {
        &self.operations
    }

    pub fn shape(&self) -> &Shape {
        self.layout.shape()
    }
}

pub fn graph_to_dot<G>(g: &G) -> String
where
    G: petgraph::visit::NodeIndexable
        + petgraph::visit::GraphProp
        + petgraph::visit::IntoNodeReferences
        + petgraph::visit::IntoEdgeReferences,
    <G as petgraph::visit::Data>::EdgeWeight: Display,
    <G as petgraph::visit::Data>::NodeWeight: Display,
{
    format!("{}", petgraph::dot::Dot::new(g))
}

pub trait LazyBuffer: Debug + Clone + PartialEq {}

pub trait LazyAllocator<B: LazyBuffer> {
    fn initialize(
        &mut self,
        graph: &mut OpGraph,
        edges: &[EdgeIndex],
        last_node: NodeIndex,
    ) -> Result<()>;
    fn insert(&mut self, id: BufferId, buffer: B) -> Result<()>;
    fn allocate(&mut self, id: BufferId, shape: &Shape, dtype: DType) -> Result<&B>;
    fn get(&self, id: BufferId) -> Result<&B>;
    fn get_or_allocate(&mut self, id: BufferId, shape: &Shape, dtype: DType) -> Result<&B>;
}

pub fn determine_tensor_source<'a>(graph: &'a OpGraph, edge: &'a Edge<OpEdge>) -> &'a Edge<OpEdge> {
    // TODO: Something is not quite right here.
    /*
    let mut source = edge;
    loop {
        let next_edge = source.next_edge(petgraph::Incoming);
        if next_edge == EdgeIndex::end() {
            break;
        }
        let edge = &graph.raw_edges()[next_edge.index()];

        let source_node_idx = edge.source();
        let source_node = &graph.raw_nodes()[source_node_idx.index()];
        if !source_node.weight.supports_inplace() {
            break;
        }
        source = edge;
    }
    source
    */
    edge
}

pub fn calculate_usage_records(
    graph: &OpGraph,
    edges: &[EdgeIndex],
) -> HashMap<BufferId, (Option<BufferId>, Option<usize>, usize, Layout, DType)> {
    let mut records = HashMap::with_capacity(edges.len());
    let topo_len = edges.len() - 1;
    for (i, edge_idx) in edges.iter().rev().enumerate() {
        let edge = &graph.raw_edges()[edge_idx.index()];
        let buffer_id = edge.weight.buffer_id();
        let node_idx = edge.source();

        let t = &graph[node_idx];
        if t.resolved() {
            continue;
        }
        let incoming = graph.edges_directed(node_idx, petgraph::Incoming);
        for in_idx in incoming {
            let in_edge: &Edge<OpEdge> = &graph.raw_edges()[in_idx.id().index()];
            let source_idx = in_edge.source();

            let source = &graph[source_idx];
            if source.resolved() {
                continue;
            }
            let true_source = determine_tensor_source(graph, in_edge);
            records
                .entry(true_source.weight.buffer_id())
                .or_insert_with(|| {
                    (
                        None,
                        None,
                        topo_len - i,
                        true_source.weight.layout.clone(),
                        true_source.weight.dtype(),
                    )
                });
        }

        if let Some(record) = records.get_mut(&buffer_id) {
            record.0 = Some(buffer_id);
            record.1 = Some(topo_len - i);
        }
    }
    //filter records with no producer
    records.retain(|_, v| v.1.is_some());
    records
}

pub struct MemoryPlan {
    pub plan: HashMap<BufferId, BufferId>,
    pub allocations: HashMap<BufferId, (Layout, DType)>,
}

// https://arxiv.org/pdf/2001.03288.pdf
pub fn greedy_by_size(graph: &OpGraph, edges: &[EdgeIndex]) -> Result<MemoryPlan> {
    let record_map = calculate_usage_records(graph, edges);
    let mut shared_objects: Vec<BufferId> = Vec::with_capacity(record_map.len());

    let mut memory_plan = HashMap::with_capacity(record_map.len());
    let mut allocations = HashMap::with_capacity(record_map.len());

    for (buffer_id, (record_buffer_id, producer, last_consumer, layout, dtype)) in record_map.iter()
    {
        let record_producer = producer.unwrap();
        let mut best_buffer: Option<BufferId> = None;

        for obj in shared_objects.iter().cloned() {
            let mut suitable = true;
            for (
                inner_buffer_id,
                (_, inner_producer, inner_last_consumer, inner_layout, inner_dtype),
            ) in record_map.iter()
            {
                let max_first = std::cmp::max(record_producer, inner_producer.unwrap());
                let min_last = *std::cmp::min(last_consumer, inner_last_consumer);
                if max_first <= min_last
                    && allocations.contains_key(inner_buffer_id)
                    && inner_buffer_id == &obj
                {
                    suitable = false;
                    break;
                }
            }
            if suitable {
                best_buffer = Some(obj);
            }
        }
        if let Some(best) = best_buffer {
            memory_plan.insert(buffer_id.clone(), best);
            //allocator.insert(*buffer_id, obj.clone())?;
        } else {
            //let rounded_size = (record.size - 1).next_power_of_two();
            allocations.insert(buffer_id.clone(), (layout.clone(), dtype.clone()));
            //let buffer = allocator.allocate(*buffer_id, layout.shape(), *dtype)?;
            shared_objects.push(buffer_id.clone());
        }
    }

    // Loop through and add inplace assignments
    /*
    for edge_idx in edges.iter() {
        let edge = &graph.raw_edges()[edge_idx.index()];
        let node_idx = edge.source();
        let t = &graph[node_idx];
        if t.resolved() {
            continue;
        }
        let incoming = graph.edges_directed(node_idx, petgraph::Incoming);
        for in_idx in incoming {
            let in_edge: &Edge<OpEdge> = &graph.raw_edges()[in_idx.id().index()];

            let true_source = determine_tensor_source(graph, in_edge);
            if true_source.weight.buffer_id() != in_edge.weight.buffer_id() {
                if let Ok(buf) = allocator.get(true_source.weight.buffer_id()) {
                    allocator.insert(in_edge.weight.buffer_id(), buf.clone())?;
                }
            }
        }
    }
    */
    Ok(MemoryPlan {
        plan: memory_plan,
        allocations,
    })
}

pub trait Executor {
    type BufferType: LazyBuffer;
    type AllocatorType: LazyAllocator<Self::BufferType>;

    fn run(&self, operations: OpGraph) -> Result<Self::BufferType>;

    fn optimize(&self, _graph: &mut OpGraph) {
        // TODO: Generic optimizations
    }

    /// Backend specific optimizations. Defaults to noop.
    fn specialize(&self, _graph: &mut OpGraph) {}

    fn eval(
        &self,
        operations: &mut OpGraph,
        allocator: &mut Self::AllocatorType,
        node: NodeIndex,
    ) -> Result<()>;

    fn allocator(&self) -> Self::AllocatorType;
}

impl LazyStorage {
    fn new(shape: Shape, dtype: DType) -> LazyStorage {
        LazyStorage {
            id: LazyStorageId::new(),
            operations: Default::default(),
            layout: Layout::contiguous(shape),
            initial_dtype: dtype,
            current_node: None,
        }
    }

    pub fn execute<E: Executor>(&self, executor: E) -> Result<E::BufferType> {
        // TODO: Apply backend agnostic optimizations
        // TODO: Apply backend specific optimizations
        executor.run(self.operations.clone())
    }
}

#[derive(Debug, Clone)]
pub struct OpEdge {
    id: EdgeId,
    layout: Layout,
    dtype: DType,
    buffer_id: BufferId,
}

impl OpEdge {
    pub fn new(layout: Layout, dtype: DType) -> Self {
        OpEdge {
            id: EdgeId::new(),
            layout,
            dtype,
            buffer_id: BufferId::new(),
        }
    }

    pub fn id(&self) -> EdgeId {
        self.id
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn shape(&self) -> &Shape {
        &self.layout.shape()
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn buffer_id(&self) -> BufferId {
        self.buffer_id
    }

    pub fn update_buffer_id(&mut self, buffer_id: BufferId) {
        self.buffer_id = buffer_id
    }

    pub fn bytes(&self) -> usize {
        self.layout.shape().elem_count() * self.dtype.size_in_bytes()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OpNode {
    id: NodeId,
    op: Op,
}

impl OpNode {
    fn new(op: Op) -> Self {
        Self {
            id: NodeId::new(),
            op: op,
        }
    }

    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn op(&self) -> &Op {
        &self.op
    }

    pub fn resolved(&self) -> bool {
        matches!(self.op, Op::Const(_))
    }

    pub fn supports_inplace(&self) -> bool {
        false //matches!(self.op, Op::Const(_)) // | Op::Copy2D(_, _, _, _, _, _))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Const(Arc<CpuStorage>),
    ToCpu,
    Affine(f64, f64),
    Powf(f64),
    Elu(f64),
    Reduce(ReduceOp, Vec<usize>, Layout, usize),
    Cmp(CmpOp),
    ToDType(DType),
    Unary(&'static str),
    Binary(&'static str, Layout, Layout),
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
    Sink,
    Aggregate,
    Output,
}

impl Display for OpNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.id.0, self.op)
    }
}

impl Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Op::Const(_) => write!(f, "Const"),
            _ => write!(f, "{:?}", self),
        }
    }
}

impl Display for OpEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} ({:?}, {:?}, {:?}, {:?})",
            self.id.0,
            self.layout.shape().dims(),
            self.layout.stride(),
            self.dtype,
            self.buffer_id
        )
    }
}

impl LazyStorage {
    fn node(&self, idx: NodeIndex<u32>) -> Option<&OpNode> {
        self.operations.node_weight(idx)
    }

    fn node_op(&self, idx: NodeIndex<u32>) -> Option<Op> {
        self.node(idx).map(|n| n.op.clone())
    }

    fn add_operation(&mut self, op: Op) -> NodeIndex<u32> {
        self.operations.add_node(OpNode::new(op))
    }

    fn merge(
        &mut self,
        other: &Self,
        source_node: NodeIndex<u32>,
        target_node: NodeIndex<u32>,
        edge: OpEdge,
    ) -> Result<()> {
        // TODO: Optimize. Perhaps we should use a GraphMap instead of Graph.

        let source_id = other.operations.node_weight(source_node).unwrap().id;
        let _target_id = self.operations.node_weight(target_node).unwrap().id;
        debug_assert!(target_node.index() <= self.operations().node_count());

        // Add nodes from other graph.
        // Store old->new index mapping for updating edges.
        // TODO: Use self.operations.visit_map() instead of hashmap
        let mut nodes: HashMap<NodeId, NodeIndex> = self
            .operations
            .raw_nodes()
            .iter()
            .enumerate()
            .map(|(i, n)| (n.weight.id, NodeIndex::new(i)))
            .collect();

        let edges: HashSet<EdgeId> = self
            .operations
            .raw_edges()
            .iter()
            .map(|e| e.weight.id)
            .collect();

        // Add all relevant nodes if not already present
        other.operations.raw_nodes().iter().for_each(|o| {
            let oid = o.weight.id;
            if !nodes.contains_key(&oid) {
                let i = self.operations.add_node(o.weight.clone());
                nodes.insert(oid, i);
            }
        });

        // Add edges from other graph, using the index mapping.
        other.operations.raw_edges().iter().for_each(|other_edge| {
            let oid = other_edge.weight.id;
            if !edges.contains(&oid) {
                let source = other_edge.source();
                let target = other_edge.target();
                let e = other_edge.weight.clone();
                let s_id = other.operations.node_weight(source).unwrap().id;
                let t_id = other.operations.node_weight(target).unwrap().id;
                let source_idx = nodes.get(&s_id).unwrap();
                let target_idx = nodes.get(&t_id).unwrap();
                self.operations.add_edge(*source_idx, *target_idx, e);
            }
        });

        // Find source_node in self graph by unique id
        let source_node = self
            .operations()
            .raw_nodes()
            .iter()
            .enumerate()
            .find_map(|(i, n)| {
                if n.weight.id == source_id {
                    return Some(NodeIndex::new(i));
                }
                None
            })
            .unwrap();

        // Connect the two graphs using the provided node indices and edge weight.
        self.operations.add_edge(source_node, target_node, edge);

        Ok(())
    }
}

impl BackendStorage for LazyStorage {
    type Device = LazyDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        Ok(self.clone())
    }

    fn dtype(&self) -> DType {
        // Get current node if available
        if let Ok(node) = self.get_current_node() {
            // If Const get dtype from inner storage
            if let Some(Op::Const(s)) = self.node_op(node) {
                return s.dtype();
            }
            // If ToDType use that dtype
            if let Some(Op::ToDType(dtype)) = self.node_op(node) {
                return dtype;
            }
            // Otherwise get dtype from first incoming edge if available
            let mut incoming = self.operations.edges_directed(node, Incoming);
            if let Some(first) = incoming.next() {
                return first.weight().dtype;
            }
        }
        // TODO: unreachable? A node must either have incoming edges or be Const.
        // Fallback to default dtype
        self.initial_dtype
    }

    fn device(&self) -> &Self::Device {
        &LazyDevice
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        let mut next = self.clone();
        next.add_operation(Op::ToCpu);
        todo!()
    }

    fn affine(&self, l: &Layout, mul: f64, add: f64) -> Result<Self> {
        count();

        let mut next = self.clone();

        let op = Op::Affine(mul, add);
        let edge = OpEdge::new(l.clone(), self.dtype());
        let idx = next.add_operation(op);
        next.operations
            .add_edge(next.get_current_node()?, idx, edge);

        next.current_node = Some(idx);
        Ok(next)
    }

    fn powf(&self, l: &Layout, pow: f64) -> Result<Self> {
        count();

        let mut next = self.clone();

        let op = Op::Powf(pow);
        let edge = OpEdge::new(l.clone(), self.dtype());
        let idx = next.add_operation(op);
        next.operations
            .add_edge(next.get_current_node()?, idx, edge);

        next.current_node = Some(idx);
        Ok(next)
    }

    fn elu(&self, l: &Layout, alpha: f64) -> Result<Self> {
        count();

        let mut next = self.clone();

        let op = Op::Elu(alpha);
        let edge = OpEdge::new(l.clone(), self.dtype());
        let idx = next.add_operation(op);
        next.operations
            .add_edge(next.get_current_node()?, idx, edge);

        next.current_node = Some(idx);
        Ok(next)
    }

    fn reduce_op(&self, reduce: ReduceOp, l: &Layout, reduce_dims: &[usize]) -> Result<Self> {
        count();

        let mut next = self.clone();

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
        println!("reduction_shape: {reduction_shape:?}");
        let reduction_layout = Layout::new(reduction_shape, stride, 0);
        println!("reduction_layout: {reduction_layout:?}");

        let dst_shape = Shape::from(dst_dims);
        println!("dst_shape: {dst_shape:?}");
        let dst_layout = Layout::contiguous(dst_shape);
        println!("dst_layout: {dst_layout:?}");

        let op = Op::Reduce(
            reduce,
            reduce_dims.to_vec(),
            reduction_layout.clone(),
            dst_el,
        );
        let edge = OpEdge::new(reduction_layout.clone(), self.dtype());
        let idx = next.add_operation(op);
        next.operations
            .add_edge(next.get_current_node()?, idx, edge);

        next.layout = dst_layout;
        next.current_node = Some(idx);
        Ok(next)
    }

    fn cmp(&self, cmp_op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        count();

        let mut next = self.clone();

        let op = Op::Cmp(cmp_op);
        let idx = next.add_operation(op);

        let current_op = next.get_current_node()?;
        let lhs_edge = OpEdge::new(lhs_l.clone(), self.dtype());
        next.operations.add_edge(current_op, idx, lhs_edge);

        let rhs_op = rhs.get_current_node()?;
        let rhs_edge = OpEdge::new(rhs_l.clone(), self.dtype());

        next.merge(rhs, rhs_op, idx, rhs_edge)?;

        next.current_node = Some(idx);
        Ok(next)
    }

    fn to_dtype(&self, l: &Layout, dtype: DType) -> Result<Self> {
        if self.dtype() != dtype {
            count();

            let mut next = self.clone();

            let op = Op::ToDType(dtype);
            let edge = OpEdge::new(l.clone(), self.dtype());
            let idx = next.add_operation(op);
            next.operations
                .add_edge(next.get_current_node()?, idx, edge);

            next.current_node = Some(idx);
            return Ok(next);
        }
        Ok(self.clone())
    }

    fn unary_impl<B: UnaryOpT>(&self, l: &Layout) -> Result<Self> {
        count();

        let mut next = self.clone();

        let op = Op::Unary(B::KERNEL);
        let edge = OpEdge::new(l.clone(), self.dtype());
        let idx = next.add_operation(op);
        next.operations
            .add_edge(next.get_current_node()?, idx, edge);

        next.layout = l.clone();
        next.current_node = Some(idx);
        Ok(next)
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        count();

        println!("lazy binary lhs_l: {lhs_l:?}");
        println!("lazy binary rhs_l: {rhs_l:?}");
        println!("lazy binary self.layout: {:?}", self.layout);
        println!("lazy binary rhs.layout: {:?}", rhs.layout);
        let mut next = self.clone();

        let op = Op::Binary(B::KERNEL, lhs_l.clone(), rhs_l.clone());
        let idx = next.add_operation(op);

        let current_op = next.get_current_node()?;
        let lhs_edge = OpEdge::new(self.layout.clone(), self.dtype());
        next.operations.add_edge(current_op, idx, lhs_edge);

        let rhs_op = rhs.get_current_node()?;
        let rhs_edge = OpEdge::new(rhs.layout.clone(), self.dtype());

        next.merge(rhs, rhs_op, idx, rhs_edge)?;

        next.current_node = Some(idx);
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
        count();

        let mut next = self.clone();

        let idx = next.add_operation(Op::WhereCond);

        let current_op = next.get_current_node()?;
        let src = OpEdge::new(l.clone(), self.dtype());
        next.operations.add_edge(current_op, idx, src);

        let t_op = t.get_current_node()?;
        let t_edge = OpEdge::new(t_l.clone(), t.dtype());
        next.merge(t, t_op, idx, t_edge)?;

        let f_op = f.get_current_node()?;
        let f_edge = OpEdge::new(f_l.clone(), f.dtype());
        next.merge(f, f_op, idx, f_edge)?;

        next.current_node = Some(idx);

        Ok(next)
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        count();

        let mut next = self.clone();

        let op = Op::Conv1D(params.clone());
        let idx = next.add_operation(op);

        let current_op = next.get_current_node()?;
        let l_edge = OpEdge::new(l.clone(), self.dtype());
        next.operations.add_edge(current_op, idx, l_edge);

        let kernel_op = kernel.get_current_node()?;
        let kernel_edge = OpEdge::new(kernel_l.clone(), self.dtype());
        next.merge(kernel, kernel_op, idx, kernel_edge)?;

        next.current_node = Some(idx);
        Ok(next)
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        count();

        let mut next = self.clone();

        let op = Op::ConvTranspose1D(params.clone());
        let idx = next.add_operation(op);

        let current_op = next.get_current_node()?;
        let l_edge = OpEdge::new(l.clone(), self.dtype());
        next.operations.add_edge(current_op, idx, l_edge);

        let kernel_op = kernel.get_current_node()?;
        let kernel_edge = OpEdge::new(kernel_l.clone(), self.dtype());
        next.merge(kernel, kernel_op, idx, kernel_edge)?;

        next.current_node = Some(idx);
        Ok(next)
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        count();

        let mut next = self.clone();

        let op = Op::Conv2D(params.clone());
        let idx = next.add_operation(op);

        let current_op = next.get_current_node()?;
        let l_edge = OpEdge::new(l.clone(), self.dtype());
        next.operations.add_edge(current_op, idx, l_edge);

        let kernel_op = kernel.get_current_node()?;
        let kernel_edge = OpEdge::new(kernel_l.clone(), self.dtype());
        next.merge(kernel, kernel_op, idx, kernel_edge)?;

        next.current_node = Some(idx);
        Ok(next)
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        count();

        let mut next = self.clone();

        let op = Op::ConvTranspose2D(params.clone());
        let idx = next.add_operation(op);

        let current_op = next.get_current_node()?;
        let l_edge = OpEdge::new(l.clone(), self.dtype());
        next.operations.add_edge(current_op, idx, l_edge);

        let kernel_op = kernel.get_current_node()?;
        let kernel_edge = OpEdge::new(kernel_l.clone(), self.dtype());
        next.merge(kernel, kernel_op, idx, kernel_edge)?;

        next.current_node = Some(idx);
        Ok(next)
    }

    fn avg_pool2d(
        &self,
        l: &Layout,
        (w_k, h_k): (usize, usize),
        (w_stride, h_stride): (usize, usize),
    ) -> Result<Self> {
        count();

        let mut next = self.clone();

        let op = Op::AvgPool2D((w_k, h_k), (w_stride, h_stride));
        let idx = next.add_operation(op);

        let current_op = next.get_current_node()?;
        let edge = OpEdge::new(l.clone(), self.dtype());
        next.operations.add_edge(current_op, idx, edge);

        next.current_node = Some(idx);
        Ok(next)
    }

    fn max_pool2d(
        &self,
        l: &Layout,
        (w_k, h_k): (usize, usize),
        (w_stride, h_stride): (usize, usize),
    ) -> Result<Self> {
        count();

        let mut next = self.clone();

        let op = Op::MaxPool2D((w_k, h_k), (w_stride, h_stride));
        let idx = next.add_operation(op);

        let current_op = next.get_current_node()?;
        let edge = OpEdge::new(l.clone(), self.dtype());
        next.operations.add_edge(current_op, idx, edge);

        next.current_node = Some(idx);
        Ok(next)
    }

    fn upsample_nearest1d(&self, l: &Layout, sz: usize) -> Result<Self> {
        count();

        let mut next = self.clone();

        let op = Op::UpsampleNearest1D(sz);
        let idx = next.add_operation(op);

        let current_op = next.get_current_node()?;
        let edge = OpEdge::new(l.clone(), self.dtype());
        next.operations.add_edge(current_op, idx, edge);

        next.current_node = Some(idx);
        Ok(next)
    }

    fn upsample_nearest2d(&self, l: &Layout, out_w: usize, out_h: usize) -> Result<Self> {
        count();

        let mut next = self.clone();

        let op = Op::UpsampleNearest2D(out_w, out_h);
        let idx = next.add_operation(op);

        let current_op = next.get_current_node()?;
        let edge = OpEdge::new(l.clone(), self.dtype());
        next.operations.add_edge(current_op, idx, edge);

        next.current_node = Some(idx);
        Ok(next)
    }

    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        count();

        let mut next = self.clone();

        let op = Op::Gather(dim);
        let idx = next.add_operation(op);

        let current_op = next.get_current_node()?;
        let edge = OpEdge::new(l.clone(), self.dtype());
        next.operations.add_edge(current_op, idx, edge);

        let ids_op = ids.get_current_node()?;
        let ids_edge = OpEdge::new(ids_l.clone(), self.dtype());
        next.merge(ids, ids_op, idx, ids_edge)?;

        next.current_node = Some(idx);
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
        count();

        let op = Op::ScatterSet(dim);
        let idx = self.add_operation(op);

        let current_op = self.get_current_node()?;
        let edge = OpEdge::new(l.clone(), self.dtype());
        self.operations.add_edge(current_op, idx, edge);

        let ids_op = ids.get_current_node()?;
        let ids_edge = OpEdge::new(ids_l.clone(), self.dtype());
        self.merge(ids, ids_op, idx, ids_edge)?;

        let src_op = src.get_current_node()?;
        let src_edge = OpEdge::new(src_l.clone(), self.dtype());
        self.merge(src, src_op, idx, src_edge)?;

        self.current_node = Some(idx);
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
        count();

        let op = Op::ScatterAddSet(dim);
        let idx = self.add_operation(op);

        let current_op = self.get_current_node()?;
        let edge = OpEdge::new(l.clone(), self.dtype());
        self.operations.add_edge(current_op, idx, edge);

        let ids_op = ids.get_current_node()?;
        let ids_edge = OpEdge::new(ids_l.clone(), self.dtype());
        self.merge(ids, ids_op, idx, ids_edge)?;

        let src_op = src.get_current_node()?;
        let src_edge = OpEdge::new(src_l.clone(), self.dtype());
        self.merge(src, src_op, idx, src_edge)?;

        self.current_node = Some(idx);
        Ok(())
    }

    fn index_select(&self, ids: &Self, src_l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        count();

        let mut next = self.clone();

        let op = Op::IndexSelect(dim);
        let idx = next.add_operation(op);

        let current_op = next.get_current_node()?;
        let edge = OpEdge::new(src_l.clone(), self.dtype());
        next.operations.add_edge(current_op, idx, edge);

        let ids_op = ids.get_current_node()?;
        let ids_edge = OpEdge::new(ids_l.clone(), ids.dtype());
        next.merge(ids, ids_op, idx, ids_edge)?;

        // Calculate amount of elements in result, and use to calculate size of output edge buffer.
        let left_size: usize = src_l.dims()[..dim].iter().product();
        let right_size: usize = src_l.dims()[dim + 1..].iter().product();
        let ids_el = ids_l.shape().elem_count();
        // TODO: Instead of using dst_el directly as the layout of the OpEdge from index select node -> sink node,
        // perhaps we should calculate actual layout/shape of dst tensor.
        let dst_el = ids_el * left_size * right_size;
        let dst_layout = Layout::contiguous(dst_el);

        let sink_idx = next.add_operation(Op::Sink);
        let sink_edge = OpEdge::new(dst_layout.clone(), self.dtype());
        next.operations.add_edge(idx, sink_idx, sink_edge);

        next.layout = dst_layout;
        next.current_node = Some(sink_idx);
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
        count();

        let mut next = self.clone();

        let op = Op::IndexAdd(dim);
        let idx = next.add_operation(op);

        let current_op = next.get_current_node()?;
        let edge = OpEdge::new(l.clone(), self.dtype());
        next.operations.add_edge(current_op, idx, edge);

        let ids_op = ids.get_current_node()?;
        let ids_edge = OpEdge::new(ids_l.clone(), self.dtype());
        next.merge(ids, ids_op, idx, ids_edge)?;

        let src_op = src.get_current_node()?;
        let src_edge = OpEdge::new(src_l.clone(), self.dtype());
        next.merge(src, src_op, idx, src_edge)?;

        next.current_node = Some(idx);
        Ok(next)
    }

    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        count();

        let mut next = self.clone();

        let op = Op::Matmul((b, m, n, k));
        let idx = next.add_operation(op);

        let current_op = next.get_current_node()?;
        let edge = OpEdge::new(lhs_l.clone(), self.dtype());
        next.operations.add_edge(current_op, idx, edge);

        let rhs_op = rhs.get_current_node()?;
        let rhs_edge = OpEdge::new(rhs_l.clone(), self.dtype());
        next.merge(rhs, rhs_op, idx, rhs_edge)?;

        let lhs_dims = lhs_l.shape().dims();
        let dim = lhs_dims.len();
        let out_shape = Shape::from(&lhs_dims[..dim - 2]).extend(&[m, n]);

        next.layout = Layout::contiguous(out_shape);
        next.current_node = Some(idx);
        Ok(next)
    }

    fn copy_strided_src(&self, rhs: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        count();

        // TODO: Validate that self -> mut rhs graph logic is correct.
        let op = Op::CopyStridedSrc(dst_offset);
        let idx = rhs.add_operation(op);

        if let Ok(rhs_current_node) = rhs.get_current_node() {
            let edge = OpEdge::new(src_l.clone(), rhs.dtype());
            rhs.operations.add_edge(rhs_current_node, idx, edge);
        }

        let current_op = self.get_current_node()?;
        let edge = OpEdge::new(src_l.clone(), self.dtype());
        rhs.merge(self, current_op, idx, edge)?;

        rhs.current_node = Some(idx);

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
        count();

        // Add Copy2D node to dst graph. This is the node we will be merging on later.
        let op = Op::Copy2D(d1, d2, src_s, dst_s, src_o, dst_o);
        let copy_idx = dst.add_operation(op);

        let self_current_node = self.get_current_node()?;

        let in_edge = OpEdge::new(self.layout.clone(), self.dtype());

        let sink_idx = if let Ok(dst_current_node) = dst.get_current_node() {
            let dst_node_weight = dst.node(dst_current_node).unwrap();

            if matches!(dst_node_weight.op, Op::Sink) {
                // Already aggregating copies. Continue using this node.
                dst_current_node
            } else {
                dst.operations
                    .add_edge(dst_current_node, copy_idx, in_edge.clone());

                // Current node is not sink. Add sink node.
                let sink_idx = dst.add_operation(Op::Sink);
                dst.current_node = Some(sink_idx);
                sink_idx
            }
        } else {
            // No current node exsits for dst. Add sink node.
            let sink_idx = dst.add_operation(Op::Sink);
            dst.current_node = Some(sink_idx);
            sink_idx
        };
        let out_edge = OpEdge::new(self.layout.clone(), self.dtype());
        dst.operations
            .add_edge(copy_idx, sink_idx, out_edge.clone());
        dst.merge(self, self_current_node, copy_idx, out_edge)?;
        Ok(())
    }

    fn const_set(&mut self, s: crate::scalar::Scalar, l: &Layout) -> Result<()> {
        count();

        let op = Op::ConstSet(s);
        let idx = self.add_operation(op);

        let current_op = self.get_current_node()?;
        let edge = OpEdge::new(l.clone(), self.dtype());
        self.operations.add_edge(current_op, idx, edge);

        self.current_node = Some(idx);
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
        count();

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
        count();

        let shape = Shape::from(s.len());
        let dtype = s.dtype();
        let mut storage = LazyStorage::new(shape.clone(), dtype);

        let op = Op::Const(s.into());
        let idx = storage.add_operation(op);
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
            Shape::from((4, 1)),
            &Device::Lazy(LazyDevice),
        )?;

        let result = t.max(1)?;
        assert_eq!(result.to_vec1::<f32>()?, &[0., 1., 2., 3.]);
        Ok(())
    }

    #[test]
    fn lazy_check() -> Result<()> {
        let t1 = Tensor::from_slice(
            &[0f32, 1., 2., 3., 4., 5., 6., 7.],
            Shape::from(8),
            &Device::Lazy(LazyDevice),
            //&Device::new_metal(0)?,
        )?;

        let t2 = t1.affine(1.25, 0.0)?;
        assert_eq!(
            t2.to_vec1::<f32>()?,
            &[0.0, 1.25, 2.5, 3.75, 5.0, 6.25, 7.5, 8.75]
        );
        let t3 = t2.max_keepdim(D::Minus1)?;
        assert_eq!(t3.to_vec1::<f32>()?, &[8.75]);
        let t4 = t1.broadcast_sub(&t3)?;
        assert_eq!(
            t4.to_vec1::<f32>()?,
            &[-8.75, -7.75, -6.75, -5.75, -4.75, -3.75, -2.75, -1.75]
        );
        let t5 = t4.exp()?;
        assert_eq!(
            t5.to_vec1::<f32>()?,
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
        let t6 = t5.sum_keepdim(D::Minus1)?;
        assert_eq!(t6.to_vec1::<f32>()?, &[0.27481413]);
        let t7 = t5.broadcast_div(&t6)?;

        let result = t7;
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
}
