//! A [`WaxSource`] over the pliron dialect itself.
//!
//! Sits between the dialect and every analysis that is generic over its source. A pass or a
//! conversion holding only a `Context` and a function can hand this to `fragment_set` or
//! `uniformity` and get the same answer the backend gets from its materialised `WaxIr`, which
//! is the whole point of those analyses being generic.

use crate::attr::Attribute;
use crate::opcode::Opcode;
use crate::source::WaxSource;
use crate::types::Type;
use pliron::context::{Context, Ptr};
use pliron::operation::Operation;

/// Dialect read through [`WaxSource`] so the pass sees ops the same way lowering does.
pub struct DialectView<'a> {
    pub ctx: &'a Context,
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
        use crate::dialect::attr_mirror::WaxAttr;
        op.deref(self.ctx)
            .attributes
            .0
            .iter()
            .filter_map(|(k, v)| {
                v.downcast_ref::<WaxAttr>()
                    .map(|a| (k.to_string(), a.0.clone()))
            })
            .collect()
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
