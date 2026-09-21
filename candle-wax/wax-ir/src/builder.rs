//! Create op with builder pattern.

use crate::attr::Attribute;
use crate::build::{BlockId, Module, OpId, Value};
use crate::dialect::attr_mirror::WaxAttr;
use crate::dialect::make::make_op;
use crate::dialect::types::to_pliron;
use crate::location::Location;
use crate::opcode::Opcode;
use crate::types::Type;
use pliron::basic_block::BasicBlock;
use pliron::context::Context;
use pliron::identifier::Identifier;
use pliron::input_err_noloc;
use pliron::result::Result;
use pliron::r#type::TypeHandle;

pub struct OpBuilder {
    opcode: Opcode,
    location: Location,
    operands: Vec<Value>,
    result_types: Vec<Type>,
    attributes: Vec<(String, Attribute)>,
    regions: Vec<Vec<BlockId>>,
}

impl OpBuilder {
    pub fn new(opcode: Opcode, location: Location) -> Self {
        OpBuilder {
            opcode,
            location,
            operands: Vec::new(),
            result_types: Vec::new(),
            attributes: Vec::new(),
            regions: Vec::new(),
        }
    }

    pub fn operand(mut self, v: Value) -> Self {
        self.operands.push(v);
        self
    }
    pub fn operands(mut self, vs: impl IntoIterator<Item = Value>) -> Self {
        self.operands.extend(vs);
        self
    }
    pub fn result(mut self, t: Type) -> Self {
        self.result_types.push(t);
        self
    }
    pub fn results(mut self, ts: impl IntoIterator<Item = Type>) -> Self {
        self.result_types.extend(ts);
        self
    }
    pub fn attr(mut self, k: impl Into<String>, v: Attribute) -> Self {
        self.attributes.push((k.into(), v));
        self
    }
    pub fn attrs(mut self, kvs: impl IntoIterator<Item = (String, Attribute)>) -> Self {
        self.attributes.extend(kvs);
        self
    }
    /// The blocks of one region this op will own.
    pub fn region(mut self, blocks: Vec<BlockId>) -> Self {
        self.regions.push(blocks);
        self
    }
    pub fn regions(mut self, rs: impl IntoIterator<Item = Vec<BlockId>>) -> Self {
        self.regions.extend(rs);
        self
    }
    /// Build in provided [`Context`], and verify the result.
    pub fn build_in(self, ctx: &mut Context) -> Result<(OpId, Vec<Value>)> {
        let mut tys: Vec<TypeHandle> = Vec::with_capacity(self.result_types.len());
        for t in &self.result_types {
            match to_pliron(ctx, t) {
                Some(h) => tys.push(h),
                None => {
                    return input_err_noloc!(BuilderErr::ResultType {
                        opcode: format!("{:?}", self.opcode),
                        ty: format!("{t:?}"),
                    });
                }
            }
        }

        let op = match make_op(
            ctx,
            self.opcode,
            &tys,
            self.operands,
            &self.attributes,
            self.regions.len(),
        ) {
            Ok(op) => op,
            Err(e) => {
                return input_err_noloc!(BuilderErr::Make {
                    opcode: format!("{:?}", self.opcode),
                    reason: e.to_string(),
                });
            }
        };

        // Now that the op owns regions we can add the blocks back in.
        for (i, blocks) in self.regions.into_iter().enumerate() {
            let region = op.deref(ctx).get_region(i);
            for b in blocks {
                b.insert_at_back(region, ctx);
            }
        }

        for (k, v) in self.attributes {
            set_op_attr_str_in(ctx, op, &k, v);
        }

        // Set location
        op.deref_mut(ctx)
            .set_loc(crate::location::to_pliron(ctx, &self.location));

        // Verify the operation we just built before returning it.
        pliron::operation::verify_operation(op, ctx)?;

        let n = op.deref(ctx).get_num_results();
        let results: Vec<Value> = (0..n).map(|i| op.deref(ctx).get_result(i)).collect();
        Ok((op, results))
    }

    /// [`OpBuilder::build_in`] for a [`Module`].
    pub fn build(self, m: &mut Module) -> Result<(OpId, Vec<Value>)> {
        self.build_in(&mut m.ctx)
    }

    /// Helper for the common case of build and append in one step.
    pub fn build_and_append(self, m: &mut Module, block: BlockId) -> Result<(OpId, Vec<Value>)> {
        let (id, vals) = self.build(m)?;
        append_op(m, block, id);
        Ok((id, vals))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BuilderErr {
    #[error("{opcode}: result type `{ty}` has no pliron spelling")]
    ResultType { opcode: String, ty: String },
    #[error("{opcode}: {reason}")]
    Make { opcode: String, reason: String },
}

/// Append an operation to a block.
pub fn append_op(m: &mut Module, block: BlockId, op: OpId) {
    append_op_in(&m.ctx, block, op);
}

/// [`append_op`] against a bare [`Context`].
pub fn append_op_in(ctx: &Context, block: BlockId, op: OpId) {
    op.insert_at_back(block, ctx);
}

/// Set the operand segment sizes of an op.
///
/// Note that cuda-tile is MLIR, and spells the key `operandSegmentSizes`.
/// We conform to pliron's spelling, translated on conversion.
pub fn set_operand_segment_sizes(ctx: &Context, op: OpId, sizes: Vec<u32>) {
    use pliron::builtin::attributes::OperandSegmentSizesAttr;
    use pliron::builtin::op_interfaces::ATTR_KEY_OPERAND_SEGMENT_SIZES;
    op.deref_mut(ctx).attributes.set(
        ATTR_KEY_OPERAND_SEGMENT_SIZES.clone(),
        OperandSegmentSizesAttr(sizes),
    );
}

/// The sizes carried by a source dialect's MLIR-spelled `operandSegmentSizes` attribute.
pub(crate) fn segment_sizes_of(a: &Attribute) -> Option<Vec<u32>> {
    match a {
        Attribute::DenseI32Array(v) => Some(v.iter().map(|&x| x as u32).collect()),
        Attribute::Array(v) => v
            .iter()
            .map(|e| match e {
                Attribute::Integer(n, _) => Some(*n as u32),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

/// Set an attribute with a string key.
///
/// Used mainly when importing from `cutile_ir`.
pub fn set_op_attr_str_in(ctx: &Context, op: OpId, key: &str, value: Attribute) {
    let ident: Identifier = key
        .try_into()
        .unwrap_or_else(|_| panic!("attribute key `{key}` is not a valid pliron Identifier"));
    set_op_attr_in(ctx, op, &ident, value);
}

pub fn set_op_attr_in(ctx: &Context, op: OpId, key: &Identifier, value: Attribute) {
    op.deref_mut(ctx)
        .attributes
        .set(key.clone(), WaxAttr(value));
}

/// Create a block with the provided argument types. An op claims the block afterwards.
pub fn build_block_in(ctx: &mut Context, arg_types: &[Type]) -> (BlockId, Vec<Value>) {
    let handles: Vec<TypeHandle> = arg_types
        .iter()
        .map(|t| to_pliron(ctx, t).expect("a block argument cannot be a function type"))
        .collect();
    let blk = BasicBlock::new(ctx, None, handles);
    let vals: Vec<Value> = (0..arg_types.len())
        .map(|i| blk.deref(ctx).get_argument(i))
        .collect();
    (blk, vals)
}

/// [`build_block_in`] with [`Module`]'s context.
// TODO: Fix the `build_block -> m.build_block -> build_block_in` dance.
pub fn build_block(m: &mut Module, arg_types: &[Type]) -> (BlockId, Vec<Value>) {
    m.build_block(arg_types)
}

/// A region containing a single block.
pub fn build_single_block_region(
    m: &mut Module,
    arg_types: &[Type],
) -> (Vec<BlockId>, BlockId, Vec<Value>) {
    let (block, args) = m.build_block(arg_types);
    (vec![block], block, args)
}
