//! Op construction into pliron
use crate::attr::Attribute;
use crate::build::{BlockId, Module, OpId, Region, RegionId, Value};
use crate::dialect::make::make_op;
use crate::dialect::types::to_pliron;
use crate::location::Location;
use crate::opcode::Opcode;
use crate::types::Type;
use pliron::r#type::TypeHandle;

pub struct OpBuilder {
    opcode: Opcode,
    operands: Vec<Value>,
    result_types: Vec<Type>,
    attributes: Vec<(String, Attribute)>,
    regions: Vec<RegionId>,
}

impl OpBuilder {
    /// TODO: Use `location` when emitting errors.
    pub fn new(opcode: Opcode, _location: Location) -> Self {
        OpBuilder {
            opcode,
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
    pub fn region(mut self, r: RegionId) -> Self {
        self.regions.push(r);
        self
    }
    pub fn regions(mut self, rs: impl IntoIterator<Item = RegionId>) -> Self {
        self.regions.extend(rs);
        self
    }

    pub fn build(self, m: &mut Module) -> (OpId, Vec<Value>) {
        let tys: Vec<TypeHandle> = self
            .result_types
            .iter()
            .map(|t| to_pliron(&mut m.ctx, t).expect("a result cannot be a function type"))
            .collect();

        // The blocks each region was built from.
        let adopted: Vec<Vec<BlockId>> = self.regions.iter().map(|r| m.take_region(*r)).collect();

        let op = make_op(
            &mut m.ctx,
            self.opcode,
            &tys,
            self.operands,
            &self.attributes,
            adopted.len(),
        )
        .unwrap_or_else(|e| panic!("{:?}: {e}", self.opcode));

        // Now that the op owns regions we can add the blocks back in.
        for (i, blocks) in adopted.into_iter().enumerate() {
            let region = op.deref(&m.ctx).get_region(i);
            for b in blocks {
                b.insert_at_back(region, &m.ctx);
            }
        }

        m.set_op_attrs(op, self.attributes);

        let n = op.deref(&m.ctx).get_num_results();
        let results: Vec<Value> = (0..n).map(|i| op.deref(&m.ctx).get_result(i)).collect();
        for (v, t) in results.iter().zip(self.result_types.iter()) {
            m.record_type(*v, t.clone());
        }
        (op, results)
    }

    /// Helper for the common case of build and append in one step.
    pub fn build_and_append(self, m: &mut Module, block: BlockId) -> (OpId, Vec<Value>) {
        let (id, vals) = self.build(m);
        append_op(m, block, id);
        (id, vals)
    }
}

/// Append an operation to a block.
pub fn append_op(m: &mut Module, block: BlockId, op: OpId) {
    op.insert_at_back(block, &m.ctx);
}

/// A block with provided argument types.
pub fn build_block(m: &mut Module, arg_types: &[Type]) -> (BlockId, Vec<Value>) {
    m.build_block(arg_types)
}

/// A region containing a single block.
pub fn build_single_block_region(
    m: &mut Module,
    arg_types: &[Type],
) -> (RegionId, BlockId, Vec<Value>) {
    let (block, args) = m.build_block(arg_types);
    let region = m.alloc_region(Region {
        blocks: vec![block],
    });
    (region, block, args)
}
