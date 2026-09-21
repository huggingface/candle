//! Builds a `wax` module that the frontend emits into.

use crate::attr::Attribute;
use crate::dialect::attr_mirror::WaxAttr;
use crate::dialect::types::{from_pliron, to_pliron};
use crate::opcode::Opcode;
use crate::source::WaxSource;
use crate::types::Type;
use pliron::basic_block::BasicBlock;
use pliron::builtin::op_interfaces::SingleBlockRegionInterface;
use pliron::builtin::ops::{FuncOp, ModuleOp};
use pliron::builtin::types::FunctionType;
use pliron::context::{Context, Ptr};
use pliron::identifier::Identifier;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::{TypeHandle, Typed};
use std::cell::RefCell;
use std::collections::HashMap;

pliron::dict_key!(
    /// Marks a function as an entry point. Set by [`Module::mark_entry`].
    ATTR_ENTRY,
    "wax_entry"
);
pliron::dict_key!(
    /// Symbol name of entry function as an attribute. Not the same as pliron's
    /// `ATTR_KEY_SYM_NAME`.
    ATTR_SYM_NAME,
    "sym_name"
);
pliron::dict_key!(
    /// A constant op's value.
    ATTR_VALUE,
    "value"
);
pliron::dict_key!(
    /// Function type attribute.
    ATTR_FUNCTION_TYPE,
    "function_type"
);
pliron::dict_key!(
    /// Predicate of an `Assume` attribute.
    ATTR_PREDICATE,
    "predicate"
);

pub type OpId = Ptr<Operation>;
pub type Value = pliron::value::Value;
pub type BlockId = Ptr<BasicBlock>;

// TODO: Consider borrowed `Context` originating from shared `ModuleGroup` representing all
// variants of this module.
pub struct Module {
    pub ctx: Context,
    name: String,
    /// The pliron module owning every function this module built
    module_op: ModuleOp,
    /// Types read back out of pliron, keyed by handle.
    read_cache: RefCell<HashMap<TypeHandle, Type>>,
}

/// Frontend emits into the module layer.
impl Module {
    pub fn new(name: impl Into<String>) -> Self {
        let mut ctx = Context::new();
        let name = name.into();
        // TODO: Perhaps this should error instead of defaulting to "wax"?
        let sym = name
            .as_str()
            .try_into()
            .unwrap_or_else(|_| pliron::ident!("wax"));
        let module_op = ModuleOp::new(&mut ctx, sym);
        Module {
            ctx,
            module_op,
            name,
            read_cache: RefCell::new(HashMap::new()),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// IR in readable text for for debugging purposes.
    // TODO: Replace with proper Printable/Parsable
    pub fn ir_text(&self) -> String {
        use pliron::printable::Printable;
        let mut s = format!("module @{} {{\n", self.name);
        for f in self.functions() {
            s.push_str(&format!("{}\n", f.disp(&self.ctx)));
        }
        s.push_str("}\n");
        s
    }

    /// Verifies the module and all its nested regions and blocks.
    // TODO: Impl proper verification on all ops, types, and attributes.
    pub fn verify(&self) -> Result<(), String> {
        for (i, f) in self.functions().into_iter().enumerate() {
            pliron::operation::verify_operation(f, &self.ctx)
                .map_err(|e| format!("{}: function {i}: {e}", self.name))?;
        }
        Ok(())
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
        begin_function_in(&mut self.ctx, self.module_op, name, param_types)
    }

    /// The pliron module owning this module's functions.
    pub fn module_op(&self) -> ModuleOp {
        self.module_op
    }

    /// Every function in the module.
    pub fn functions(&self) -> Vec<OpId> {
        functions_in(&self.ctx, self.module_op)
    }

    /// Functions in the module registered as entry points.
    pub fn entries(&self) -> Vec<OpId> {
        self.functions()
            .into_iter()
            .filter(|f| self.op_attr(*f, &ATTR_ENTRY).is_some())
            .collect()
    }

    /// Register `func` as an entry point. Idempotent.
    pub fn mark_entry(&mut self, func: OpId) {
        mark_entry_in(&self.ctx, func);
    }

    /// A type read back out of pliron. From cache where possible.
    pub(crate) fn ty_of(&self, h: TypeHandle) -> Type {
        if let Some(t) = self.read_cache.borrow().get(&h) {
            return t.clone();
        }
        let t = from_pliron(&self.ctx, h).expect("type has no tile equivalent");
        self.read_cache.borrow_mut().insert(h, t.clone());
        t
    }

    /// All wax attributes.
    pub fn op_attrs(&self, op: OpId) -> Vec<(String, Attribute)> {
        op.deref(&self.ctx)
            .attributes
            .0
            .iter()
            .filter_map(|(k, v)| {
                v.downcast_ref::<WaxAttr>()
                    .map(|a| (k.to_string(), a.0.clone()))
            })
            .collect()
    }

    /// One wax attribute by key.
    pub fn op_attr(&self, op: OpId, key: &Identifier) -> Option<Attribute> {
        op.deref(&self.ctx)
            .attributes
            .get::<WaxAttr>(key)
            .map(|a| a.0.clone())
    }

    /// Set an attribute on an op. Overrides previous value if present.
    pub fn set_op_attr(&self, op: OpId, key: &Identifier, value: Attribute) {
        crate::builder::set_op_attr_in(&self.ctx, op, key, value);
    }

    /// Create a block with the provided argument types. An op claims the block afterwards.
    pub fn build_block(&mut self, arg_types: &[Type]) -> (BlockId, Vec<Value>) {
        crate::builder::build_block_in(&mut self.ctx, arg_types)
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
    fn native_attributes(
        &self,
        op: Self::Op,
    ) -> Vec<(pliron::identifier::Identifier, pliron::attribute::AttrObj)> {
        use crate::dialect::attr_mirror::WaxAttr;
        op.deref(&self.ctx)
            .attributes
            .0
            .iter()
            .filter(|(_, v)| v.downcast_ref::<WaxAttr>().is_none())
            .map(|(k, v)| (k.clone(), pliron::dyn_clone::clone_box(&**v)))
            .collect()
    }
    fn location(&self, op: Self::Op) -> crate::location::Location {
        crate::location::from_pliron(&self.ctx, op.deref(&self.ctx).loc())
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
        self.ty_of(v.get_type(&self.ctx))
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

/// [`Module::begin_function`] against a bare [`Context`] and the module to append into.
///
/// This form allows a pliron pass to add a function to the module, which is needed by for
/// example GEMM split-K.
pub fn begin_function_in(
    ctx: &mut Context,
    module_op: ModuleOp,
    name: &str,
    param_types: &[Type],
) -> (OpId, BlockId, Vec<Value>) {
    let tys: Vec<TypeHandle> = param_types
        .iter()
        .map(|t| to_pliron(ctx, t).expect("a parameter cannot be a function type"))
        .collect();
    let fn_ty = FunctionType::get(ctx, tys, vec![]);
    let sym = name.try_into().expect("kernel symbol");
    let func = FuncOp::new(ctx, sym, fn_ty);
    module_op.append_operation(ctx, func.get_operation(), 0);
    let block = func.get_entry_block(ctx);
    let args: Vec<Value> = (0..param_types.len())
        .map(|i| block.deref(ctx).get_argument(i))
        .collect();
    // Not registered as an entry. Caller decides what is an entry point, and
    // registering here could cause the function to be emitted twice.
    (func.get_operation(), block, args)
}

/// All functions of a `ModuleOp` in the provided `Context`.
pub fn functions_in(ctx: &Context, module_op: ModuleOp) -> Vec<OpId> {
    let body = module_op.get_body(ctx, 0);
    body.deref(ctx).iter(ctx).collect()
}

/// A function's pliron symbol name.
pub fn symbol_name_in(ctx: &Context, func: OpId) -> Option<String> {
    use pliron::builtin::op_interfaces::SymbolOpInterface;
    use pliron::op::Op;
    Some(
        FuncOp::from_operation(func)
            .get_symbol_name(ctx)
            .to_string(),
    )
}

/// Mark `func` as an entry point in the provided `Context`.
pub fn mark_entry_in(ctx: &Context, func: OpId) {
    crate::builder::set_op_attr_in(ctx, func, &ATTR_ENTRY, Attribute::Bool(true));
}
