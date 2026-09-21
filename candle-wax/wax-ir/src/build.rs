//! Builds a `wax` module that the frontend emits into.

use crate::attr::Attribute;
use crate::dialect::attr_mirror::WaxAttr;
use crate::dialect::types::{from_pliron, to_pliron};
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

// TODO: Move or delete.
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
    ) -> pliron::result::Result<(OpId, BlockId, Vec<Value>)> {
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

    /// Clone `func`  from `src` into this module, with provided `name`.
    pub fn clone_function_from(
        &mut self,
        src: &Module,
        func: OpId,
        name: &str,
    ) -> Result<OpId, String> {
        clone_function_in(&src.ctx, func, &mut self.ctx, self.module_op, name)
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
    pub fn build_block(
        &mut self,
        arg_types: &[Type],
    ) -> pliron::result::Result<(BlockId, Vec<Value>)> {
        crate::builder::build_block_in(&mut self.ctx, arg_types)
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
) -> pliron::result::Result<(OpId, BlockId, Vec<Value>)> {
    use crate::builder::BuilderErr;
    let mut tys: Vec<TypeHandle> = Vec::with_capacity(param_types.len());
    for t in param_types {
        let Some(h) = to_pliron(ctx, t) else {
            return pliron::input_err_noloc!(BuilderErr::Make {
                opcode: "begin_function".to_string(),
                reason: format!("parameter type {t:?} has no pliron variant"),
            });
        };
        tys.push(h);
    }
    let fn_ty = FunctionType::get(ctx, tys, vec![]);
    let Ok(sym) = TryInto::<Identifier>::try_into(name) else {
        return pliron::input_err_noloc!(BuilderErr::Make {
            opcode: "begin_function".to_string(),
            reason: format!("`{name}` is not a valid symbol"),
        });
    };
    let func = FuncOp::new(ctx, sym, fn_ty);
    module_op.append_operation(ctx, func.get_operation(), 0);
    let block = func.get_entry_block(ctx);
    let args: Vec<Value> = (0..param_types.len())
        .map(|i| block.deref(ctx).get_argument(i))
        .collect();
    // Not registered as an entry. Caller decides what is an entry point, and
    // registering here could cause the function to be emitted twice.
    Ok((func.get_operation(), block, args))
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

/// The wax types of a block's arguments.
pub fn block_arg_types_in(ctx: &Context, block: BlockId) -> Option<Vec<Type>> {
    block
        .deref(ctx)
        .arguments()
        .map(|v| from_pliron(ctx, v.get_type(ctx)))
        .collect()
}

/// Clone entirety of `func` from source context to provided `ModuleOp` in destination context.
/// Returns the cloned function id. Cloned function is not explicitly marked as an entry function.
pub fn clone_function_in(
    src_ctx: &Context,
    func: OpId,
    dst_ctx: &mut Context,
    dst_module_op: ModuleOp,
    name: &str,
) -> Result<OpId, String> {
    use pliron::builtin::op_interfaces::SymbolOpInterface;
    use pliron::irbuild::cloning::{IrMapping, clone_operation_into_context};
    use pliron::irbuild::listener::DummyListener;
    use pliron::irbuild::rewriter::IRRewriter;

    let mut mapper = IrMapping::new();
    let mut rewriter: IRRewriter<DummyListener> = IRRewriter::default();
    let cloned = clone_operation_into_context(func, src_ctx, dst_ctx, &mut rewriter, &mut mapper)
        .map_err(|e| e.to_string())?;
    let sym = TryInto::<pliron::identifier::Identifier>::try_into(name)
        .map_err(|_| format!("`{name}` is not a valid symbol"))?;
    FuncOp::from_operation(cloned).set_symbol_name(dst_ctx, sym);
    dst_module_op.append_operation(dst_ctx, cloned, 0);
    Ok(cloned)
}
