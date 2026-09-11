//! Builds a `wax` module that the frontend emits into.
//!
//! Operations, blocks, regions, and use-def chains are all represented by pliron concepts.
//! The module layer is only what a frontend needs on top to interface with pliron efficiently:
//!
//! - `value_types` stores the [`Type`] a value was created with, so reading one back is a
//!   lookup rather than a downcast. The frontend uses the lookup frequently.
//! - `pending`. A pliron region belongs to an operation, but the frontend can (for example)
//!   emit a loop body before the `For` exists. We store these dangling blocks as pending and
//!   assign them to the owning op once it has been created.
//!
//! Note that to wax the pliron `FuncOp` is not treated as an ordinary op for two reasons.
//! For one it owns its entry block and arguments, so it cannot adopt a region from the frontend
//! cleanly. However it's usage is to represent/encompasses the entire module, so we only need
//! one function per program*. [`Module::begin_function`] creates the function up-front and
//! hands back the block to emit into.
//!
//! *This is also a limitation if we ever want to support multiple functions.

use crate::attr::Attribute;
use crate::dialect::attr_mirror::WaxAttrs;
use crate::dialect::ops::ATTR_KEY_WAX_ATTRS;
use crate::dialect::types::{from_pliron, to_pliron};
use crate::opcode::Opcode;
use crate::source::WaxSource;
use crate::types::Type;
use pliron::basic_block::BasicBlock;
use pliron::builtin::ops::FuncOp;
use pliron::builtin::types::FunctionType;
use pliron::context::{Context, Ptr};
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::{TypeHandle, Typed};
use std::cell::RefCell;
use std::collections::HashMap;

pub type OpId = Ptr<Operation>;
pub type Value = pliron::value::Value;
pub type BlockId = Ptr<BasicBlock>;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct RegionId(pub u32);

#[derive(Clone, Debug, Default)]
pub struct Region {
    pub blocks: Vec<BlockId>,
}

pub struct Module {
    pub ctx: Context,
    name: String,
    /// Entry functions, in declaration order.
    pub functions: Vec<OpId>,
    /// Module-scope globals. Carried, not consumed - a backend synthesises whatever
    /// shared-memory globals its lowering needs.
    pub globals: Vec<crate::global::Global>,
    /// Regions awaiting the op that will own them.
    pending: Vec<Vec<BlockId>>,
    /// The plain type each value was created with.
    value_types: HashMap<Value, Type>,
    /// Types read back out of pliron, keyed by handle. Sound only because it cannot outlive
    /// `ctx` - handles are arena indices and a different `Context` reuses them.
    read_cache: RefCell<HashMap<TypeHandle, Type>>,
}

/// Frontend emits into the module layer.
impl Module {
    pub fn new(name: impl Into<String>) -> Self {
        Module {
            ctx: Context::new(),
            name: name.into(),
            functions: Vec::new(),
            globals: Vec::new(),
            pending: Vec::new(),
            value_types: HashMap::new(),
            read_cache: RefCell::new(HashMap::new()),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// A readable dump, for eyeballing a failing kernel.
    pub fn ir_text(&self) -> String {
        use pliron::printable::Printable;
        let mut s = format!("module @{} {{\n", self.name);
        for f in &self.functions {
            s.push_str(&format!("{}\n", f.disp(&self.ctx)));
        }
        s.push_str("}\n");
        s
    }

    /// Create a function and hand back the block to emit its body into.
    ///
    /// Created first, because pliron's `FuncOp` owns its block and arguments rather than adopting
    /// a region built beforehand.
    pub fn begin_function(
        &mut self,
        name: &str,
        param_types: &[Type],
    ) -> (OpId, BlockId, Vec<Value>) {
        let tys: Vec<TypeHandle> = param_types
            .iter()
            .map(|t| to_pliron(&mut self.ctx, t).expect("a parameter cannot be a function type"))
            .collect();
        let fn_ty = FunctionType::get(&self.ctx, tys, vec![]);
        let sym = name.try_into().expect("kernel symbol");
        let func = FuncOp::new(&mut self.ctx, sym, fn_ty);
        let block = func.get_entry_block(&self.ctx);
        let args: Vec<Value> = (0..param_types.len())
            .map(|i| block.deref(&self.ctx).get_argument(i))
            .collect();
        for (v, t) in args.iter().zip(param_types.iter()) {
            self.value_types.insert(*v, t.clone());
        }
        // Not registered as an entry. Caller decides what is an entry point, and
        // registering here could cause the function to be emitted twice.
        (func.get_operation(), block, args)
    }

    pub fn value_type(&self, v: Value) -> &Type {
        self.value_types
            .get(&v)
            .expect("value was not created by this module")
    }

    /// A type read back out of pliron - from cache where possible.
    pub(crate) fn ty_of(&self, h: TypeHandle) -> Type {
        if let Some(t) = self.read_cache.borrow().get(&h) {
            return t.clone();
        }
        let t = from_pliron(&self.ctx, h).expect("type has no tile equivalent");
        self.read_cache.borrow_mut().insert(h, t.clone());
        t
    }

    pub(crate) fn record_type(&mut self, v: Value, t: Type) {
        self.value_types.insert(v, t);
    }

    /// Retrieve the attributes of an op.
    pub fn op_attrs(&self, op: OpId) -> Vec<(String, Attribute)> {
        op.deref(&self.ctx)
            .attributes
            .get::<WaxAttrs>(&ATTR_KEY_WAX_ATTRS.try_into().unwrap())
            .map(|a| a.0.clone())
            .unwrap_or_default()
    }

    /// Add an attribute to an op.
    pub fn push_op_attr(&mut self, op: OpId, key: &str, value: Attribute) {
        let mut attrs = self.op_attrs(op);
        attrs.push((key.to_string(), value));
        self.set_op_attrs(op, attrs);
    }

    pub(crate) fn set_op_attrs(&self, op: OpId, attrs: Vec<(String, Attribute)>) {
        op.deref_mut(&self.ctx)
            .attributes
            .set(ATTR_KEY_WAX_ATTRS.try_into().unwrap(), WaxAttrs(attrs));
    }

    pub fn alloc_region(&mut self, r: Region) -> RegionId {
        let id = RegionId(self.pending.len() as u32);
        self.pending.push(r.blocks);
        id
    }

    pub(crate) fn take_region(&mut self, r: RegionId) -> Vec<BlockId> {
        std::mem::take(&mut self.pending[r.0 as usize])
    }

    /// Create a block with the given argument types. An op claims the block afterwards.
    pub fn build_block(&mut self, arg_types: &[Type]) -> (BlockId, Vec<Value>) {
        let handles: Vec<TypeHandle> = arg_types
            .iter()
            .map(|t| {
                to_pliron(&mut self.ctx, t).expect("a block argument cannot be a function type")
            })
            .collect();
        let blk = BasicBlock::new(&mut self.ctx, None, handles);
        let vals: Vec<Value> = (0..arg_types.len())
            .map(|i| blk.deref(&self.ctx).get_argument(i))
            .collect();
        for (v, t) in vals.iter().zip(arg_types.iter()) {
            self.value_types.insert(*v, t.clone());
        }
        (blk, vals)
    }
}

impl WaxSource for Module {
    type Op = OpId;
    type Val = Value;
    type Block = BlockId;
    type Region = Ptr<pliron::region::Region>;

    fn opcode(&self, op: Self::Op) -> Opcode {
        crate::dialect::source::opcode_of(&self.ctx, op)
    }
    fn operands(&self, op: Self::Op) -> Vec<Self::Val> {
        op.deref(&self.ctx).operands().collect()
    }
    fn result_types(&self, op: Self::Op) -> Vec<Type> {
        let o = op.deref(&self.ctx);
        (0..o.get_num_results())
            .map(|i| self.ty_of(o.get_result(i).get_type(&self.ctx)))
            .collect()
    }
    fn attributes(&self, op: Self::Op) -> Vec<(String, Attribute)> {
        self.op_attrs(op)
    }
    fn regions(&self, op: Self::Op) -> Vec<Self::Region> {
        op.deref(&self.ctx).regions().collect()
    }
    fn op_result(&self, op: Self::Op, result_index: u32) -> Option<Self::Val> {
        let o = op.deref(&self.ctx);
        if (result_index as usize) >= o.get_num_results() {
            return None;
        }
        let v = o.get_result(result_index as usize);
        (v.num_uses(&self.ctx) > 0).then_some(v)
    }
    fn value_type(&self, v: Self::Val) -> Type {
        match self.value_types.get(&v) {
            Some(t) => t.clone(),
            None => self.ty_of(v.get_type(&self.ctx)),
        }
    }
    fn region_blocks(&self, r: Self::Region) -> Vec<Self::Block> {
        r.deref(&self.ctx).iter(&self.ctx).collect()
    }
    fn block_ops(&self, b: Self::Block) -> Vec<Self::Op> {
        b.deref(&self.ctx).iter(&self.ctx).collect()
    }
    fn block_args(&self, b: Self::Block) -> Vec<(Self::Val, Type)> {
        let blk = b.deref(&self.ctx);
        blk.arguments()
            .map(|v| (v, WaxSource::value_type(self, v)))
            .collect()
    }
}
