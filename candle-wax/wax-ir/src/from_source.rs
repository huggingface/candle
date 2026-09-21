//! Build a pliron [`Module`] in the `wax` dialect from any [`WaxSource`].

use crate::attr::Attribute;
use crate::build::{BlockId, Module, Value};
use crate::builder::OpBuilder;
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
    module_from_source_multi(name, src, &[(root, name.to_string(), param_types.to_vec())])
}

/// Build a module named `name` from several of `src`'s functions, each kept under its own
/// name. The single-root form is [`module_from_source`]. Each function is cloned
/// independently, and no value crosses between them, so the clone is the same one-function
/// walk, run once per root.
pub fn module_from_source_multi<S: WaxSource>(
    name: &str,
    src: &S,
    roots: &[(S::Op, String, Vec<Type>)],
) -> Result<Module, BuildErr> {
    let mut m = Module::new(name);
    for (root, fname, param_types) in roots {
        clone_function(&mut m, src, *root, fname, param_types)?;
    }
    Ok(m)
}

fn clone_function<S: WaxSource>(
    m: &mut Module,
    src: &S,
    root: S::Op,
    name: &str,
    param_types: &[Type],
) -> Result<(), BuildErr> {
    let (func, entry, args) = m.begin_function(name, param_types);

    // Clone function attributes.
    for (k, v) in src.attributes(root) {
        crate::builder::set_op_attr_str_in(&m.ctx, func, &k, v);
    }

    let mut b = Builder {
        src,
        m,
        vals: HashMap::new(),
    };

    // Entry function's block arguments are cloned first.
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

    // Blocks first, then ops. Terminators can target a block that appears later.
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

    m.mark_entry(func);
    Ok(())
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

            // Get operand segment sizes.
            // Converted from MLIR's `operandSegmentSizes` to pliron's `OperandSegmentSizesAttr`.
            let mut segment_sizes = None;
            let attrs: Vec<(String, Attribute)> = self
                .src
                .attributes(op)
                .into_iter()
                .filter(|(k, v)| {
                    if k == "operandSegmentSizes" {
                        segment_sizes = crate::builder::segment_sizes_of(v);
                        false
                    } else {
                        true
                    }
                })
                .collect();

            // Regions are built first and taken by the op, because `FuncOp` owns its blocks.
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
                region_ids.push(ids);
            }

            // `OpBuilder` adopts the regions built above.
            let (id, results) = OpBuilder::new(opcode, self.src.location(op))
                .operands(operands)
                .results(result_types.clone())
                .attrs(attrs)
                .regions(region_ids)
                .build_and_append(self.m, dst_blk)
                .map_err(|e| BuildErr::Op(e.to_string()))?;
            if let Some(sizes) = segment_sizes {
                crate::builder::set_operand_segment_sizes(&self.m.ctx, id, sizes);
            }
            // Clone pliron native attributes (like builtins). Only applicable for a wax to wax clone.
            for (k, v) in self.src.native_attributes(op) {
                id.deref_mut(&self.m.ctx).attributes.0.insert(k, v);
            }

            for (i, _) in result_types.iter().enumerate() {
                if let (Some(sv), Some(v)) = (self.src.op_result(op, i as u32), results.get(i)) {
                    self.vals.insert(sv, *v);
                }
            }
        }
        Ok(())
    }
}
