//! Build a pliron [`Module`] from any [`WaxSource`].
//!
//! The counterpart to [`crate::ir::from_source`], which materialises the same source into the
//! flat [`WaxIr`](crate::ir::WaxIr) a backend lowers from. Both read the same trait; they differ
//! in what they produce and therefore in what they are for.
//!
//! This one exists because the dialect is where our IR-level work lives - `cross_loop_reuse`,
//! `verify_operation`, and everything plan section 8 has queued. A frontend that emits wax IR
//! directly builds the module as it goes and needs none of this. One that hands us its own IR
//! does: `wax-frontend` compiles through `cutile-frontend`, whose output is a
//! `cutile_ir::Module`, and without this the whole dialect layer is bypassed.
//!
//! Ordering follows from wax IR being SSA and structured: a definition always precedes its uses
//! within a block, and a value from an enclosing scope is a block argument. So blocks are
//! created with their arguments first, then ops in order, and every operand is resolvable by the
//! time it is read.

use crate::attr::Attribute;
use crate::build::{BlockId, Module, Region, Value};
use crate::builder::OpBuilder;
use crate::location::Location;
use crate::source::WaxSource;
use crate::types::Type;
use std::collections::HashMap;

/// What went wrong building the module.
#[derive(Debug)]
pub enum BuildErr {
    /// An operand was read before anything defined it. Means the source is not in SSA order,
    /// which this builder relies on rather than checks for up front.
    UndefinedOperand,
    /// The source's root is not a function.
    RootNotAFunction,
    /// The dialect has no op for this opcode.
    Op(String),
}

impl std::fmt::Display for BuildErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UndefinedOperand => write!(f, "operand used before definition"),
            Self::RootNotAFunction => write!(f, "source root is not a function"),
            Self::Op(e) => write!(f, "building op: {e}"),
        }
    }
}

/// Build a module named `name` from `src`'s function at `root`.
pub fn module_from_source<S: WaxSource>(
    name: &str,
    src: &S,
    root: S::Op,
    param_types: &[Type],
) -> Result<Module, BuildErr> {
    let mut m = Module::new(name);
    let (func, entry, args) = m.begin_function(name, param_types);

    // Function-level attributes, carried across. `begin_function` builds a fresh `FuncOp`, so
    // without this everything the frontend recorded ON THE FUNCTION is silently dropped -
    // `const_param_slots`, which decides that a buffer binds in constant address space, and
    // `tg_scratch_rows`, which sizes threadgroup scratch. Both are read from the function much
    // later, and their absence reads as "not requested" rather than as an error.
    for (k, v) in src.attributes(root) {
        m.set_op_attr(func, &k, v);
    }

    let mut b = Builder {
        src,
        m: &mut m,
        vals: HashMap::new(),
    };

    // The entry block's arguments are the function's parameters. Binding them before anything
    // else is what lets the first op resolve its operands.
    let regions = b.src.regions(root);
    let Some(&first) = regions.first() else {
        return Err(BuildErr::RootNotAFunction);
    };
    let blocks = b.src.region_blocks(first);
    let Some(&src_entry) = blocks.first() else {
        return Err(BuildErr::RootNotAFunction);
    };
    for ((v, _), a) in b.src.block_args(src_entry).into_iter().zip(args) {
        b.vals.insert(v, a);
    }

    // Blocks first, then ops: a terminator can target a block that appears later.
    let mut mapped = vec![entry];
    for &blk in blocks.iter().skip(1) {
        let arg_tys: Vec<Type> = b.src.block_args(blk).into_iter().map(|(_, t)| t).collect();
        let (id, vals) = b.m.build_block(&arg_tys);
        for ((v, _), a) in b.src.block_args(blk).into_iter().zip(vals) {
            b.vals.insert(v, a);
        }
        mapped.push(id);
    }
    for (&src_blk, &dst_blk) in blocks.iter().zip(mapped.iter()) {
        b.ops_into(src_blk, dst_blk)?;
    }

    m.functions.push(func);
    Ok(m)
}

struct Builder<'a, S: WaxSource> {
    src: &'a S,
    m: &'a mut Module,
    vals: HashMap<S::Val, Value>,
}

impl<S: WaxSource> Builder<'_, S> {
    fn resolve(&self, v: S::Val) -> Result<Value, BuildErr> {
        self.vals.get(&v).copied().ok_or(BuildErr::UndefinedOperand)
    }

    /// Build every op of `src_blk` into `dst_blk`, recursing into regions.
    fn ops_into(&mut self, src_blk: S::Block, dst_blk: BlockId) -> Result<(), BuildErr> {
        for op in self.src.block_ops(src_blk) {
            let opcode = self.src.opcode(op);
            let operands: Vec<Value> = self
                .src
                .operands(op)
                .into_iter()
                .map(|v| self.resolve(v))
                .collect::<Result<_, _>>()?;
            let result_types = self.src.result_types(op);
            let attrs: Vec<(String, Attribute)> = self.src.attributes(op);

            // Regions are built first and adopted by the op, because pliron's `FuncOp` owns its
            // blocks: an op cannot take a region that was never allocated to it.
            let mut region_ids = Vec::new();
            for r in self.src.regions(op) {
                let src_blocks = self.src.region_blocks(r);
                let mut ids = Vec::new();
                for &sb in &src_blocks {
                    let arg_tys: Vec<Type> = self
                        .src
                        .block_args(sb)
                        .into_iter()
                        .map(|(_, t)| t)
                        .collect();
                    let (id, vals) = self.m.build_block(&arg_tys);
                    for ((v, _), a) in self.src.block_args(sb).into_iter().zip(vals) {
                        self.vals.insert(v, a);
                    }
                    ids.push(id);
                }
                for (&sb, &db) in src_blocks.iter().zip(ids.iter()) {
                    self.ops_into(sb, db)?;
                }
                region_ids.push(self.m.alloc_region(Region { blocks: ids }));
            }

            // `OpBuilder` owns the awkward part: a region has to be allocated before the op
            // exists and adopted after, because pliron ops own their regions.
            let (_, results) = OpBuilder::new(opcode, Location::Unknown)
                .operands(operands)
                .results(result_types.clone())
                .attrs(attrs)
                .regions(region_ids)
                .build_and_append(self.m, dst_blk);

            for (i, _) in result_types.iter().enumerate() {
                if let (Some(sv), Some(v)) = (self.src.op_result(op, i as u32), results.get(i)) {
                    self.vals.insert(sv, *v);
                }
            }
        }
        Ok(())
    }
}
