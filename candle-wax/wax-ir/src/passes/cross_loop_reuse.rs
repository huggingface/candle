//! Cross-loop reuse refers to optimizing the pattern of when a program accesses data items or
//! cache lines in one loop nest and then again in a later part of the program.
//!
//! Our approach is to identify loops that read the same view more than once, and mark the first
//! for caching.
//!
//! Two `wax.for` loops in one block, over the same bounds, both loading the same partition view.
//! The first loop can cache what it reads in thread-private registers so the second reads them
//! back instead of re-loading.
//!
//! [`CrossLoopReuseAnalysis`] analyses the IR and computes the reuse plans. [`CrossLoopReusePass`] asks the
//! [`AnalysisManager`] for those plans, writes the appropriate attributes onto the IR, and reports through
//! [`PassResult`] whether the IR changed and which analyses remain valid.
use crate::attr::Attribute;
use crate::opcode::Opcode;
use crate::source::WaxSource;
use crate::types::{ScalarType, TileElementType, Type};
use pliron::context::{Context, Ptr};
use pliron::irbuild::IRStatus;
use pliron::operation::Operation;
use pliron::pass::{Analysis, AnalysisManager, Pass, PassResult};
use pliron::result::Result;

/// Set on the `wax.for` that produces the cache, i.e. caches a view in registers for reuse.
/// Value is the cache length.
pub const ATTR_CACHE_PRODUCER: &str = "wax_cache_producer";
/// Set on the `wax.for` that consumes what the producer loop cached.
pub const ATTR_CACHE_CONSUMER: &str = "wax_cache_consumer";

/// Bytes per thread of the producer cache, set alongside [`ATTR_CACHE_PRODUCER`].
pub const ATTR_CACHE_BYTES: &str = "wax_cache_bytes";

/// Unique id of a caching producer/consumer pair. Written to both loops of a plan.
///
/// Removes the need for re-deriving pairs later.
pub const ATTR_CACHE_ID: &str = "wax_cache_id";

/// A producer/consumer pair.
// TODO: Perhaps this should be a group with a single producer and multiple consumers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Plan {
    pub producer: Ptr<Operation>,
    pub consumer: Ptr<Operation>,
    /// Bytes per thread of the producer cache. See [`ATTR_CACHE_BYTES`].
    pub bytes_per_thread: i32,
    pub num_chunks: i32,
}

/// Every loop pair in the function worth caching.
pub struct CrossLoopReuseAnalysis {
    pub plans: Vec<Plan>,
}

impl Analysis for CrossLoopReuseAnalysis {
    fn name(&self) -> &str {
        "wax-cross-loop-reuse"
    }

    fn compute(op: Ptr<Operation>, ctx: &Context, _analyses: &mut AnalysisManager) -> Result<Self> {
        Ok(CrossLoopReuseAnalysis {
            plans: find_plans(ctx, op),
        })
    }
}

/// Find loop pairs where the view read by first loop can safely be reused by the second.
///
/// Returns one [`Plan`] per pair. A plan states that the producer loop should write what it
/// loads into a thread-private cache, and the consumer loop should read that cache instead
/// of loading from memory aagain. The plan carries the cache length (`num_chunks`) and what
/// holding it costs (`bytes_per_thread`). Whether that cost is worth paying is up to the
/// backend, not this pass.
///
///
/// # When is a pair eligible?
///
/// 1. Both `wax.for` ops are in the function's entry block.
/// 2. Iteration bounds match, and each bound is a compile time constant (`matching_bounds`).
///    One iteration caches one value, so the iteration count is the cache length.
///    Must be static because the cache lives in registers and register allocation is static (runtime
///    bounds means the cache would have to live in memory, which makes the entire pass moot).
/// 3. Both loops load the same partition view directly in their bodies (`views_read`). That
///    shared view is the one that gets reused.
/// 4. The element width of the view is known, so that we can compute the cost (`view_elem_bytes`).
/// 5. There are no writes to the shared view inside or between the two loops (`may_be_written`).
///    Such a write would make the cache stale.
///
/// # On supporting blocks in general
///
/// The first condition is stricter than technically required.
/// We can change it to "Both `wax.for` ops are in the same block". All sibling
/// (two ops in the same block) loops are eligible for this pass because they are not
/// conditional wrt each other. If the loops are inside a `wax.if` body either both run or neither.
/// Same goes for the body of another `wax.for` op.
///
/// The only reason we do not do this right away is that we first need to guarantee that the lowering
/// correctly handles cache allocation. A naive lowering of the IR we are emitting with this pass will
/// likely emit allocation instructions at the producer loop. This is completely fine with our current
/// first condition, but opening up to support any block means nested blocks -> nested allocations ->
/// potential stack overflow.
/// Simply put, we need to guarantee that these nested block allocations are hoisted out of the loop.
///
/// # On supporting differing depths
///
/// A `wax.for` pair that have different nesting depths is significantly harder. The cache would have
/// to stay live across the inner loop's iterations and remain valid through all of them.
/// This would require live-variable analysis.
///
/// # On overlapping plans
///
/// Currently plans do not overlap. A loop carries a single [`ATTR_CACHE_ID`], so it belongs to exactly
/// one producer/consumer pair.
///
/// This rules out one loop consuming from two different producers (two caches), as well as a loop that
/// produces for one group while consuming from another. Both would need a second id.
///
/// It does not rule out one loop producing for several consumers. However it is currently not supported
/// simply because `find_plans` saves both loops in `paired`, so the producer would be skipped by the
/// "already paired" check. We could change the [`Plan`] to model groups rather than pairs.
pub fn find_plans(ctx: &Context, func: Ptr<Operation>) -> Vec<Plan> {
    let src = DialectView { ctx };
    let Some(block) = src
        .regions(func)
        .first()
        .and_then(|r| src.region_blocks(*r).first().copied())
    else {
        return Vec::new();
    };

    let fors: Vec<Ptr<Operation>> = src
        .block_ops(block)
        .into_iter()
        .filter(|&o| src.opcode(o) == Opcode::For)
        .collect();

    let mut plans: Vec<Plan> = Vec::new();
    let mut paired: Vec<Ptr<Operation>> = Vec::new();
    for a in 0..fors.len() {
        for b in (a + 1)..fors.len() {
            let (first, second) = (fors[a], fors[b]);
            if paired.contains(&first) || paired.contains(&second) {
                continue;
            }
            let Some(num_chunks) = matching_bounds(&src, first, second) else {
                continue;
            };
            let read_by_first = views_read(&src, first);
            if read_by_first.is_empty() {
                continue;
            }
            let read_by_second = views_read(&src, second);
            let Some(shared) = read_by_first.iter().find(|v| read_by_second.contains(v)) else {
                continue;
            };
            let Some(elem_bytes) = view_elem_bytes(&src, *shared) else {
                continue;
            };
            // Ensure no values are written to the source (producer) of the cache.
            if may_be_written(&src, block, first, second, *shared) {
                continue;
            }
            plans.push(Plan {
                producer: first,
                consumer: second,
                num_chunks,
                bytes_per_thread: num_chunks.saturating_mul(elem_bytes),
            });
            paired.push(first);
            paired.push(second);
        }
    }
    plans
}

/// Writes [`CrossLoopReuseAnalysis`]'s conclusion onto the IR.
/// Each plan is annotated with an [`ATTR_CACHE_ID`] on the producing and consumer loop.
#[derive(Default)]
pub struct CrossLoopReusePass {
    /// Pairs marked by the last run.
    pub marked: usize,
}

impl Pass for CrossLoopReusePass {
    fn name(&self) -> &str {
        "wax-cross-loop-reuse-pass"
    }

    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let plans = analyses
            .get_analysis::<CrossLoopReuseAnalysis>(op, ctx)?
            .plans
            .clone();

        let mut result = PassResult::default();
        self.marked = plans.len();
        if plans.is_empty() {
            // `IRStatus::Unchanged` implies all analyses are preserved.
            return Ok(result);
        }
        apply(ctx, &plans);
        result.ir_changed = IRStatus::Changed;

        // Since we only added attributes we mark the analysis as preserved - no recomputation required.
        result.set_preserved::<CrossLoopReuseAnalysis>();
        Ok(result)
    }
}

/// Write one batch of reuse plans onto the IR.
fn apply(ctx: &Context, plans: &[Plan]) {
    for (id, p) in plans.iter().enumerate() {
        let id = id as i64;
        set_int_attr(ctx, p.producer, ATTR_CACHE_PRODUCER, p.num_chunks as i64);
        set_int_attr(ctx, p.producer, ATTR_CACHE_BYTES, p.bytes_per_thread as i64);
        set_int_attr(ctx, p.producer, ATTR_CACHE_ID, id);
        set_int_attr(ctx, p.consumer, ATTR_CACHE_CONSUMER, 1);
        set_int_attr(ctx, p.consumer, ATTR_CACHE_ID, id);
    }
}

/// Performs croos-loop reuse analysis and applies it to one function.
///
/// TODO: Take `&mut Context`, as that is what `Pass::run` requires.
pub fn annotate_with(
    analyses: &mut AnalysisManager,
    ctx: &Context,
    func: Ptr<Operation>,
) -> Result<usize> {
    let plans = analyses
        .get_analysis::<CrossLoopReuseAnalysis>(func, ctx)?
        .plans
        .clone();
    apply(ctx, &plans);
    Ok(plans.len())
}

/// Returns trip count of two `wax.for` loops if their bounds match.
///
/// `wax.for`'s `lo`, `hi`, and `step` decide the trip count.
///
/// All bounds must compile-time constants.
// TODO: Investigate if we can capture more patterns.
fn matching_bounds(
    src: &DialectView<'_>,
    first: Ptr<Operation>,
    second: Ptr<Operation>,
) -> Option<i32> {
    let (o1, o2) = (src.operands(first), src.operands(second));
    if o1.len() < 3 || o2.len() < 3 {
        return None;
    }
    let (lo1, lo2) = (const_i32(src, o1[0])?, const_i32(src, o2[0])?);
    let (hi1, hi2) = (const_i32(src, o1[1])?, const_i32(src, o2[1])?);
    let (st1, st2) = (const_i32(src, o1[2])?, const_i32(src, o2[2])?);
    if lo1 != 0 || lo2 != 0 || hi1 != hi2 || st1 != st2 || st1 <= 0 {
        return None;
    }
    let span = hi1.checked_sub(lo1)?;
    let trips = span.checked_add(st1 - 1)? / st1;
    (trips > 0).then_some(trips)
}

/// How deep [`view_base`] traverses a chain of view ops.
///
/// Hitting depth 8 is unlikely. This is here to avoid spinning on malformed/cyclic chains.
const MAX_VIEW_CHAIN: usize = 8;

/// The base a view reads from, or `None` when it cannot be identified.
///
/// Views can be chained (`make_partition_view(make_tensor_view(t, ..), ..)`). The defining entity
/// of a view is an op. The defining entity of a tensor is a block. We walk through the view chain
/// until we reach a block argument value.
///
/// Any other defining op results in `None`.
///
/// This includes `int_to_ptr` bases, as different values does not necessarily mean different
/// tensors.
fn view_base(src: &DialectView<'_>, mut v: pliron::value::Value) -> Option<pliron::value::Value> {
    use pliron::value::DefiningEntity;
    for _ in 0..MAX_VIEW_CHAIN {
        // Defining entity of value is an op. Continue traversal.
        let DefiningEntity::Op(op) = v.defining_entity() else {
            // Defining entity is a block.
            // This means the value is a block argument -> our root.
            return Some(v);
        };
        match src.opcode(op) {
            Opcode::MakePartitionView
            | Opcode::MakeTensorView
            | Opcode::MakeStridedView
            | Opcode::MakeGatherScatterView => {
                v = *src.operands(op).first()?;
            }
            _ => return None, // some other producer. not identifiable.
        }
    }
    None // too deep.
}

/// Walk through the tree of ops starting at `op`. Stops early if `f` is true.
fn walk_ops(
    src: &DialectView<'_>,
    op: Ptr<Operation>,
    f: &mut impl FnMut(Ptr<Operation>) -> bool,
) -> bool {
    if f(op) {
        return true;
    }
    for r in src.regions(op) {
        for b in src.region_blocks(r) {
            for inner in src.block_ops(b) {
                if walk_ops(src, inner, f) {
                    return true;
                }
            }
        }
    }
    false
}

/// Checks if there are any writes to the producer loop inside or between the reuse loop pair.
///
/// The cache a `consumer` reads consists of values read in the `producer` loop. Any write to the
/// `producer` values inside or between the loops makes the cache stale.
/// More specifically a write disqualifes the plan when its destination view shares a base with the
/// base of the `producer` view.
///
/// # How we determine a shared base
///
/// [`view_base`] traverses the view chain to its root, which for a tensor view is a pointer
/// operand. Roots are compared by IR identity, as opposed to value.
/// Since `view_base` only accepts kernel parameters we can safely determine the shared base:
///
/// - Multiple views of the same parameter resolve to the same root and count as a conflict. This
///   is the common case of two partition views based on one tensor.
/// - Two different parameters are assumed not to alias. This assumption is part of the DSL, as
///   every kernel argument gets its own ordering token on the same basis. A host binding the same
///   buffer to two parameters breaks it, and breaks the memory model too.
/// - Anything else, including a pointer manually built by `int_to_ptr`, is not identifiable and
///   counts as a conflict. This means that a kernel that gathers through a reconstructed address
///   would not benefit from this pass.
fn may_be_written(
    src: &DialectView<'_>,
    block: Ptr<pliron::basic_block::BasicBlock>,
    producer: Ptr<Operation>,
    consumer: Ptr<Operation>,
    cached: pliron::value::Value,
) -> bool {
    let Some(cached_base) = view_base(src, cached) else {
        return true;
    };

    let writes_cached = |op: Ptr<Operation>| -> bool {
        // Operand differes per opcode.
        let dst_index = match src.opcode(op) {
            Opcode::StoreViewTko => 1,
            Opcode::AtomicRMW | Opcode::AtomicCAS => 0,
            _ => return false,
        };
        // A write whose destination we cannot read has to count as a conflict.
        let Some(&dst) = src.operands(op).get(dst_index) else {
            return true;
        };
        match view_base(src, dst) {
            Some(b) => b == cached_base,
            None => true,
        }
    };

    let mut writes_cached = writes_cached;
    let ops = src.block_ops(block);
    let (Some(i), Some(j)) = (
        ops.iter().position(|&o| o == producer),
        ops.iter().position(|&o| o == consumer),
    ) else {
        return true;
    };
    // `producer` preceding `consumer` is upheld by construction, but with this we avoid an
    // unecessary panic.
    if i >= j {
        return true;
    }
    // Check if any of the ops between `producer` and `consumer` makes the cache stale.
    if ops[i + 1..j]
        .iter()
        .any(|&o| walk_ops(src, o, &mut writes_cached))
    {
        return true;
    }

    // Check if any of the ops inside `producer` or `consumer` makes the cache stale.
    for for_op in [producer, consumer] {
        if walk_ops(src, for_op, &mut writes_cached) {
            return true;
        }
    }
    false
}

/// The partition views `for_op` loads on all of its iterations.
///
/// Conditional loads are disqualified. A load inside `wax.if` could leave the cache partly filled,
/// and a consumer cannot tell those apart.
///
/// This is the opposite of [`may_be_written`] which recurses through everything because it has to
/// detect writes that would make the cache stale / kernel wrong.
/// Having `views_read` not detect a load just means a caching opportunity is lost. Not dangerous.
///
/// Improving this would require proving that nested loads execute every iteration, or that two
/// views are conditionally equal.
// TODO: Investigate supporting nested loads.
fn views_read(src: &DialectView<'_>, for_op: Ptr<Operation>) -> Vec<pliron::value::Value> {
    let mut views = Vec::new();
    for r in src.regions(for_op) {
        for b in src.region_blocks(r) {
            for op in src.block_ops(b) {
                if src.opcode(op) != Opcode::LoadViewTko {
                    continue;
                }
                let opds = src.operands(op);
                let Some(&pv) = opds.first() else { continue };
                if !matches!(src.value_type(pv), Type::PartitionView(_)) {
                    continue;
                }
                views.push(pv);
            }
        }
    }
    views
}

/// Bytes held by the view of one thread.
fn view_elem_bytes(src: &DialectView<'_>, v: pliron::value::Value) -> Option<i32> {
    match src.value_type(v) {
        Type::PartitionView(pv) => Some(pv.tensor_view.element_type.byte_width() as i32),
        _ => None,
    }
}

/// Extract a constant i32 from the value, if it has one.
fn const_i32(src: &DialectView<'_>, v: pliron::value::Value) -> Option<i32> {
    use pliron::value::DefiningEntity;
    let DefiningEntity::Op(op) = v.defining_entity() else {
        return None;
    };
    if src.opcode(op) != Opcode::Constant {
        return None;
    }
    src.attributes(op).into_iter().find_map(|(k, a)| match a {
        Attribute::DenseElements(de)
            if k == "value"
                && de.shape.is_empty()                  // a scalar, not a tile of values
                && is_scalarlike_i32(&de.element_type)  // and actually an i32
                && de.data.len() >= 4 =>
        {
            Some(i32::from_le_bytes(de.data[..4].try_into().unwrap()))
        }
        _ => None,
    })
}

/// Checks if type is a scalar-like i32
///
/// The frontend `build_constant_op` emits a constant by its element type. For a loop bound this is
/// a rank 0 tile of i32. This is why we check for scalar-like i32, not just the bare scalar i32.
fn is_scalarlike_i32(t: &Type) -> bool {
    match t {
        Type::Scalar(ScalarType::I32) => true,
        Type::Tile(tt) => {
            tt.shape.is_empty()
                && matches!(tt.element_type, TileElementType::Scalar(ScalarType::I32))
        }
        _ => false,
    }
}

/// Set an integer attribute on the operation.
fn set_int_attr(ctx: &Context, op: Ptr<Operation>, key: &str, v: i64) {
    use crate::dialect::attr_mirror::WaxAttrs;
    use crate::dialect::ops::ATTR_KEY_WAX_ATTRS;
    let mut attrs = op
        .deref(ctx)
        .attributes
        .get::<WaxAttrs>(&ATTR_KEY_WAX_ATTRS.try_into().unwrap())
        .map(|a| a.0.clone())
        .unwrap_or_default();
    attrs.retain(|(k, _)| k != key);
    attrs.push((
        key.to_string(),
        Attribute::Integer(v, Type::Scalar(ScalarType::I32)),
    ));
    op.deref_mut(ctx)
        .attributes
        .set(ATTR_KEY_WAX_ATTRS.try_into().unwrap(), WaxAttrs(attrs));
}

/// Dialect read through [`WaxSource`] so the pass sees ops the same way lowering does.
struct DialectView<'a> {
    ctx: &'a Context,
}

impl WaxSource for DialectView<'_> {
    type Op = Ptr<Operation>;
    type Val = pliron::value::Value;
    type Block = Ptr<pliron::basic_block::BasicBlock>;
    type Region = Ptr<pliron::region::Region>;

    fn opcode(&self, op: Self::Op) -> Opcode {
        crate::dialect::source::opcode_of(self.ctx, op)
    }
    fn operands(&self, op: Self::Op) -> Vec<Self::Val> {
        op.deref(self.ctx).operands().collect()
    }
    fn result_types(&self, op: Self::Op) -> Vec<Type> {
        use pliron::r#type::Typed;
        let o = op.deref(self.ctx);
        (0..o.get_num_results())
            .filter_map(|i| {
                crate::dialect::types::from_pliron(self.ctx, o.get_result(i).get_type(self.ctx))
            })
            .collect()
    }
    fn attributes(&self, op: Self::Op) -> Vec<(String, Attribute)> {
        use crate::dialect::attr_mirror::WaxAttrs;
        use crate::dialect::ops::ATTR_KEY_WAX_ATTRS;
        op.deref(self.ctx)
            .attributes
            .get::<WaxAttrs>(&ATTR_KEY_WAX_ATTRS.try_into().unwrap())
            .map(|a| a.0.clone())
            .unwrap_or_default()
    }
    fn regions(&self, op: Self::Op) -> Vec<Self::Region> {
        op.deref(self.ctx).regions().collect()
    }
    fn op_result(&self, op: Self::Op, i: u32) -> Option<Self::Val> {
        let o = op.deref(self.ctx);
        ((i as usize) < o.get_num_results()).then(|| o.get_result(i as usize))
    }
    fn value_type(&self, v: Self::Val) -> Type {
        use pliron::r#type::Typed;
        crate::dialect::types::from_pliron(self.ctx, v.get_type(self.ctx)).unwrap_or(Type::Token)
    }
    fn region_blocks(&self, r: Self::Region) -> Vec<Self::Block> {
        use pliron::linked_list::ContainsLinkedList;
        r.deref(self.ctx).iter(self.ctx).collect()
    }
    fn block_ops(&self, b: Self::Block) -> Vec<Self::Op> {
        use pliron::linked_list::ContainsLinkedList;
        b.deref(self.ctx).iter(self.ctx).collect()
    }
    fn block_args(&self, b: Self::Block) -> Vec<(Self::Val, Type)> {
        let blk = b.deref(self.ctx);
        blk.arguments().map(|v| (v, self.value_type(v))).collect()
    }
}
