//! The materialised tile-based wax IR a backend lowers from.
//!
//! Built from any [`WaxSource`], so a backend can be written against one concrete type
//! rather than being generic over the trait. Keeps control flow and emission decisions plain.

use crate::attr::Attribute;
use crate::opcode::Opcode;
use crate::source::WaxSource;
use crate::types::Type;
use std::collections::HashMap;

/// Handle to [`WaxIr`]'s `ops` arena.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct OpRef(pub u32);

/// Handle to [`WaxIr`]'s `value_types` and `producers` arenas.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ValRef(pub u32);

/// Handle to [`WaxIr`]'s `blocks` arena.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct BlockRef(pub u32);

/// Handle to [`WaxIr`]'s `regions` arena.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct RegionRef(pub u32);

#[derive(Clone, PartialEq, Debug)]
pub struct OpData {
    pub opcode: Opcode,
    pub operands: Vec<ValRef>,
    pub result_types: Vec<Type>,
    /// References to the value of a result. `None` indicates a dead/unused result.
    pub results: Vec<Option<ValRef>>,
    pub attributes: Vec<(String, Attribute)>,
    pub regions: Vec<RegionRef>,
}

/// The origin of a value. By materialising the producer we only have to walk the source
/// once to trace the origin. Typical cases include checking whether an index is a constant,
/// or which block a dynamic stride arrives on, etc.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Producer {
    Op { op: OpRef, result_index: u32 },
    BlockArg { block: BlockRef, arg_index: u32 },
}

#[derive(Clone, PartialEq, Debug)]
pub struct BlockData {
    pub args: Vec<(ValRef, Type)>,
    pub ops: Vec<OpRef>,
}

#[derive(Clone, PartialEq, Debug)]
pub struct RegionData {
    pub blocks: Vec<BlockRef>,
}

/// Materialized wax IR representing one function.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct WaxIr {
    pub ops: Vec<OpData>,
    pub blocks: Vec<BlockData>,
    pub regions: Vec<RegionData>,
    pub value_types: Vec<Type>,
    /// Where every value comes from. Indexed by [`ValRef`].
    /// `None` until the defining op or block (producer) is materialised.
    pub producers: Vec<Option<Producer>>,
    /// The root op of this function.
    pub root: Option<OpRef>,
}

impl WaxIr {
    pub fn op(&self, r: OpRef) -> &OpData {
        &self.ops[r.0 as usize]
    }
    pub fn block(&self, r: BlockRef) -> &BlockData {
        &self.blocks[r.0 as usize]
    }
    pub fn region(&self, r: RegionRef) -> &RegionData {
        &self.regions[r.0 as usize]
    }
    pub fn value_type(&self, v: ValRef) -> &Type {
        &self.value_types[v.0 as usize]
    }
    pub fn producer(&self, v: ValRef) -> Option<Producer> {
        self.producers[v.0 as usize]
    }
    /// The op that defines `v`, if an op does.
    pub fn defining_op(&self, v: ValRef) -> Option<OpRef> {
        match self.producer(v) {
            Some(Producer::Op { op, .. }) => Some(op),
            _ => None,
        }
    }
    /// The entry block of this function.
    pub fn entry_block(&self) -> BlockRef {
        let root = self.root.expect("no root op");
        let region = *self
            .op(root)
            .regions
            .first()
            .expect("function has a region");
        *self
            .region(region)
            .blocks
            .first()
            .expect("function region has an entry block")
    }
}

/// Create `WaxIr` from any source.
pub fn from_source<S: WaxSource>(src: &S, root: S::Op) -> WaxIr {
    let mut b = Materialiser {
        src,
        ir: WaxIr::default(),
        vals: HashMap::new(),
    };
    let r = b.op(root);
    b.ir.root = Some(r);
    b.ir
}

struct Materialiser<'a, S: WaxSource> {
    src: &'a S,
    ir: WaxIr,
    vals: HashMap<S::Val, ValRef>,
}

impl<S: WaxSource> Materialiser<'_, S> {
    /// Extract value and type from source value.
    fn val(&mut self, v: S::Val) -> ValRef {
        if let Some(r) = self.vals.get(&v) {
            return *r;
        }
        let r = ValRef(self.ir.value_types.len() as u32);
        self.ir.value_types.push(self.src.value_type(v));
        self.ir.producers.push(None);
        self.vals.insert(v, r);
        r
    }

    /// Extract op data from the source.
    fn op(&mut self, op: S::Op) -> OpRef {
        // This op ref will be returned at the end of the function.
        // A region inside this op can create ops, and so creating it later would give us the wrong op ref.
        let slot = OpRef(self.ir.ops.len() as u32);

        self.ir.ops.push(OpData {
            opcode: self.src.opcode(op),
            operands: Vec::new(),
            result_types: Vec::new(),
            results: Vec::new(),
            attributes: Vec::new(),
            regions: Vec::new(),
        });

        // Loop through operands, result types, results, attributes, and regions to extract the op data.
        let operands: Vec<ValRef> = self
            .src
            .operands(op)
            .into_iter()
            .map(|v| self.val(v))
            .collect();
        let result_types = self.src.result_types(op);
        let results: Vec<Option<ValRef>> = (0..result_types.len())
            .map(|i| self.src.op_result(op, i as u32).map(|v| self.val(v)))
            .collect();
        for (i, r) in results.iter().enumerate() {
            if let Some(v) = r {
                self.ir.producers[v.0 as usize] = Some(Producer::Op {
                    op: slot,
                    result_index: i as u32,
                });
            }
        }
        let attributes = self.src.attributes(op);

        let mut regions = Vec::new();
        for r in self.src.regions(op) {
            regions.push(self.region(r));
        }

        let d = &mut self.ir.ops[slot.0 as usize];
        d.operands = operands;
        d.result_types = result_types;
        d.results = results;
        d.attributes = attributes;
        d.regions = regions;
        slot
    }

    /// Extract a region from the source.
    fn region(&mut self, r: S::Region) -> RegionRef {
        let slot = RegionRef(self.ir.regions.len() as u32);
        self.ir.regions.push(RegionData { blocks: Vec::new() });
        let mut blocks = Vec::new();
        for b in self.src.region_blocks(r) {
            blocks.push(self.block(b));
        }
        self.ir.regions[slot.0 as usize].blocks = blocks;
        slot
    }

    /// Extract a block from the source.
    fn block(&mut self, b: S::Block) -> BlockRef {
        let slot = BlockRef(self.ir.blocks.len() as u32);
        self.ir.blocks.push(BlockData {
            args: Vec::new(),
            ops: Vec::new(),
        });
        let args: Vec<(ValRef, Type)> = self
            .src
            .block_args(b)
            .into_iter()
            .map(|(v, t)| (self.val(v), t))
            .collect();
        for (i, (v, _)) in args.iter().enumerate() {
            self.ir.producers[v.0 as usize] = Some(Producer::BlockArg {
                block: slot,
                arg_index: i as u32,
            });
        }
        let mut ops = Vec::new();
        for op in self.src.block_ops(b) {
            ops.push(self.op(op));
        }
        let d = &mut self.ir.blocks[slot.0 as usize];
        d.args = args;
        d.ops = ops;
        slot
    }
}
