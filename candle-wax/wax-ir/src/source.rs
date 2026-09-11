//! Generically reading a source of wax IR.

use crate::attr::Attribute;
use crate::opcode::Opcode;
use crate::types::Type;

/// A source of wax IR.
///
/// Anything that implements this trait can be used as the source of wax IR.
/// A wax source can be read for optimization, passes, lowering to a backend, etc.
///
/// The associated handles lets the consumer work without being bound by the perticularities of
/// a producer's type system. We want these to be `Copy` so that passing them around is cheap.
pub trait WaxSource {
    /// An operation.
    type Op: Copy + Eq + std::hash::Hash + std::fmt::Debug;
    /// An SSA value.
    type Val: Copy + Eq + std::hash::Hash + std::fmt::Debug;
    /// A block.
    type Block: Copy + Eq + std::hash::Hash + std::fmt::Debug;
    /// A region.
    type Region: Copy + Eq + std::hash::Hash + std::fmt::Debug;

    /// What kind of operation this is.
    fn opcode(&self, op: Self::Op) -> Opcode;

    fn operands(&self, op: Self::Op) -> Vec<Self::Val>;
    fn result_types(&self, op: Self::Op) -> Vec<Type>;
    fn attributes(&self, op: Self::Op) -> Vec<(String, Attribute)>;
    fn regions(&self, op: Self::Op) -> Vec<Self::Region>;

    /// Extract result from op by index. Returns `None` if unused.
    fn op_result(&self, op: Self::Op, result_index: u32) -> Option<Self::Val>;

    fn value_type(&self, v: Self::Val) -> Type;

    fn region_blocks(&self, r: Self::Region) -> Vec<Self::Block>;
    fn block_ops(&self, b: Self::Block) -> Vec<Self::Op>;
    fn block_args(&self, b: Self::Block) -> Vec<(Self::Val, Type)>;
}
