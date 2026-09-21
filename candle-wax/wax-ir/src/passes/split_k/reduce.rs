//! Build a split-k reduce kernel that sums partials.
//!
//! All wax split-K kernels produce partials as `splits` contiguous `rows * cols` f32s.
//! [`build`] creates the kernel that sums them. A single tile block per `lanes`-column, so that
//! each lane has exclusive access to its column. This means the sum is a simple loop with no
//! inter-lane communication.
//! `lanes` is the SIMD width of the target. Not known in wax.
//!
//! The kernel has all extents baked into it. No scalar promotion etc.
//!
//! [`build`] can be used directly, and is needed when you want to apply split-K reduction to a
//! kernel that has no wax loop to split, but still needs to sum the partials.

use super::{ATTR_REDUCE, Partials};
use crate::attr::Attribute;
use crate::build::OpId;
use crate::builder::OpBuilder;
use crate::location::Location;
use crate::opcode::Opcode;
use crate::passes::launch::{Launch, Tile, TileBlock};
use crate::passes::utils::{
    assume_bounded, at_end_of, bin, binf, const_i32, const_scalar, convert, i32, load_view,
    make_token, partition_view, ptr_tile, stamp_in, store_view, tensor_view, terminator, tile,
    tile_block_id,
};
use crate::types::{PaddingValue, ScalarType, TensorViewType};
use pliron::context::Context;
use pliron::irbuild::IRStatus;
use pliron::irbuild::inserter::Inserter;
use pliron::pass::{AnalysisManager, Pass, PassResult};

/// Get the name for the reduce kernel: the entry's pliron symbol.
fn reduce_name_for(ctx: &Context, entry: OpId) -> String {
    crate::build::symbol_name_in(ctx, entry).unwrap_or_else(|| "k".into())
}

/// Builds the reduce that finishes a split, as a [`Pass`] over the module.
///
/// At the module level because it adds a function, and a pliron pass can not touch IR outside
/// the op it is applied to.
///
/// Reads [`Partials`] stamped by [`SplitKPass`].
///
/// No-op if nothing is split.
pub struct SplitKReducePass {
    /// SIMD width of the target.
    lanes: i64,
}

impl SplitKReducePass {
    pub fn new(lanes: i64) -> Self {
        SplitKReducePass { lanes }
    }
}

impl Pass for SplitKReducePass {
    fn name(&self) -> &str {
        "wax-split-k-reduce"
    }

    fn run(
        &mut self,
        op: OpId,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
        use pliron::op::Op;
        let module_op = pliron::builtin::ops::ModuleOp::from_operation(op);
        // Extract partials from IR.
        let Some((entry, partials)) = crate::build::functions_in(ctx, module_op)
            .into_iter()
            .find_map(|f| Partials::of(ctx, f).map(|p| (f, p)))
        else {
            return Ok(PassResult::default());
        };
        // Named after the function it reduces, which the pass has already found.
        let name = reduce_name_for(ctx, entry);
        build(
            ctx,
            module_op,
            &format!("{name}_reduce"),
            &format!("{name}_reduce"),
            &partials.out_elem,
            partials.rows,
            partials.cols,
            partials.splits,
            partials.ldc,
            self.lanes,
        )?;
        let mut res = PassResult::default();
        res.ir_changed = IRStatus::Changed;
        Ok(res)
    }
}

/// Split-K reduce launch definition.
// TODO: embed `Launch` directly.
#[derive(Debug, Clone)]
pub struct SplitKReduce {
    pub entry: String,
    pub block: TileBlock,
    pub tile: Tile,
    pub scratch_elems: usize,
}

impl SplitKReduce {
    /// Extract split-K reduce launch definition from IR
    pub fn of(ctx: &Context, module_op: pliron::builtin::ops::ModuleOp) -> Option<SplitKReduce> {
        let funcs = crate::build::functions_in(ctx, module_op);
        let partials = funcs.iter().find_map(|f| Partials::of(ctx, *f))?;
        let reduce = funcs.into_iter().find(|f| {
            f.deref(ctx)
                .attributes
                .get::<super::partials::ReduceMark>(&ATTR_REDUCE)
                .is_some()
        })?;
        let launch = Launch::of(ctx, reduce)?;
        Some(SplitKReduce {
            entry: crate::build::symbol_name_in(ctx, reduce)?,
            block: launch.block,
            tile: launch.tile,
            scratch_elems: partials.scratch_elems(),
        })
    }
}

/// Builds a split-K reduce kernel into `module_op` and returns the launch definition.
#[allow(clippy::too_many_arguments)]
pub fn build(
    ctx: &mut Context,
    module_op: pliron::builtin::ops::ModuleOp,
    func_name: &str,
    entry_name: &str,
    out_elem: &ScalarType,
    rows: i64,
    cols: i64,
    splits: i32,
    ldc: i64,
    lanes: i64,
) -> pliron::result::Result<SplitKReduce> {
    let flat = rows * cols;
    // Column chunks per row.
    let ncol = cols / lanes + i64::from(cols % lanes != 0);
    let f32 = ScalarType::F32;
    let params = vec![ptr_tile(out_elem), ptr_tile(&f32)];
    let (f, entry, args) = crate::build::begin_function_in(ctx, module_op, func_name, &params);
    let (out_ptr, part_ptr) = (args[0], args[1]);

    // Stamp entry name as attribute so it is easily accessible when lowering.
    crate::builder::set_op_attr_in(
        ctx,
        f,
        &crate::build::ATTR_SYM_NAME,
        Attribute::String(entry_name.to_string()),
    );
    stamp_in(ctx, f, &params);
    f.deref_mut(ctx)
        .attributes
        .set(ATTR_REDUCE.clone(), super::partials::ReduceMark);

    // Stamp updated launch geometry.
    Launch {
        block: TileBlock {
            x: ncol as usize,
            y: rows as usize,
            z: 1,
        },
        tile: Tile(vec![1, lanes as usize]),
    }
    .stamp(ctx, f);

    // Define output view.
    let out_tv = TensorViewType {
        element_type: out_elem.clone(),
        shape: vec![rows, cols],
        strides: vec![ldc, 1],
    };
    let ins = &mut at_end_of(entry);
    let out_view = tensor_view(ctx, ins, out_ptr, &out_tv)?;

    // Define partials view
    let part_tv = TensorViewType {
        element_type: f32.clone(),
        shape: vec![splits as i64 * rows, cols],
        strides: vec![cols, 1],
    };
    let part_view = tensor_view(ctx, ins, part_ptr, &part_tv)?;

    let tok_out = make_token(ctx, ins)?;
    let tok_part = make_token(ctx, ins)?;
    let bid = tile_block_id(ctx, ins)?;
    let blk = assume_bounded(ctx, ins, bid[0], 0, None)?;
    let row = assume_bounded(ctx, ins, bid[1], 0, Some(rows - 1))?;

    // Output partition
    let pv_out = partition_view(
        ctx,
        ins,
        out_view,
        &[1, lanes as i32],
        &out_tv,
        Some(PaddingValue::Zero),
    )?;
    // Partials partition
    let pv_part = partition_view(
        ctx,
        ins,
        part_view,
        &[1, lanes as i32],
        &part_tv,
        Some(PaddingValue::Zero),
    )?;

    // Reduction loop definitions
    let row_tile = tile(&[1, lanes], f32.clone());
    let lo = const_i32(ctx, ins, 0)?;
    let hi = const_i32(ctx, ins, splits as i64)?;
    let step = const_i32(ctx, ins, 1)?;
    let acc = const_scalar(ctx, ins, &row_tile, ScalarType::F32, [0, 0, 0, 0])?;

    let (body, body_args) = crate::builder::build_block_in(ctx, &[i32(), row_tile.clone()]);
    let (iv, carry) = (body_args[0], body_args[1]);
    let bins = &mut at_end_of(body);
    let s_a = assume_bounded(ctx, bins, iv, 0, Some(splits as i64 - 1))?;

    let rows_c = const_i32(ctx, bins, rows)?;
    let s_off = bin(ctx, bins, Opcode::MulI, s_a, rows_c)?;
    let prow = bin(ctx, bins, Opcode::AddI, s_off, row)?;
    let load = load_view(ctx, bins, pv_part, prow, blk, tok_part, &row_tile)?;
    let sum = binf(ctx, bins, Opcode::AddF, carry, load)?;
    terminator(ctx, bins, Opcode::Continue, &[sum])?;

    let res = {
        // Build reduction for loop
        let (id, res) = OpBuilder::new(Opcode::For, Location::Unknown)
            .operands([lo, hi, step, acc])
            .results(std::iter::once(row_tile.clone()))
            .region(vec![body])
            .build_in(ctx)?;
        ins.append_operation(ctx, id);
        res[0]
    };

    // Add output conversion if needed
    let value = if *out_elem == ScalarType::F32 {
        res
    } else {
        convert(ctx, ins, res, &tile(&[1, lanes], out_elem.clone()))?
    };
    // Store output and return.
    store_view(ctx, ins, value, pv_out, row, blk, tok_out)?;
    terminator(ctx, ins, Opcode::Return, &[])?;
    crate::build::mark_entry_in(ctx, f);

    Ok(SplitKReduce {
        entry: entry_name.to_string(),
        block: TileBlock {
            x: ncol as usize,
            y: rows as usize,
            z: 1,
        },
        tile: Tile(vec![1, lanes as usize]),
        scratch_elems: (splits as usize) * (flat as usize),
    })
}
