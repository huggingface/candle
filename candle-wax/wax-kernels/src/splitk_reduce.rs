//! The split-K plane reduce, as a DSL kernel.
//!
//! The same computation `wax_ir::passes::splitk_reduce::build` stamps by hand,
//! written in the DSL. Prototype for checking if kernel substitution passes via pre-written
//! kernels is an improvement over IR builders.
//!
//! The split-k gemm leaves its partials as `SPLITS` contiguous planes of `ROWS x COLS` elements.
//! Since it is contiguous each lane owns its column across all planes, and the sum is a straight
//! loop without any cross-lane work.

use wax_frontend::ast;

#[allow(unused_unsafe)]
#[wax_macro::module]
pub mod kernels {
    use cutile::core::*;

    #[wax_macro::entry()]
    pub fn splitk_reduce<
        E: ElementType,
        const ROWS: i32,
        const COLS: i32,
        const SPLITS: i32,
        const PROWS: i32,
    >(
        out: &mut Tensor<E, { [ROWS, COLS] }>,
        part: &Tensor<f32, { [PROWS, COLS] }>,
    ) {
        let ts = const_shape![1, 32];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let ptok: Token = get_tensor_token(part);
        // Padded load keeps reads within plane.
        let pp: Partition<f32, { [1, 32] }> =
            unsafe { make_partition_view(part, ts, padding::Zero, dim_map::Identity, ptok) };
        // Partials are f32, so the accumulator is as well. Narrow on store.
        let mut acc: Tile<f32, { [1, 32] }> = constant(0.0f32, ts);
        // Accumulate
        for s in 0i32..SPLITS {
            acc = acc + pp.load([s * ROWS + pid.1, pid.0]);
        }
        let otok: Token = get_tensor_token(out);
        let mut op: PartitionMut<E, { [1, 32] }> =
            unsafe { make_partition_view_mut(out, ts, padding::Zero, otok) };
        let narrowed: Tile<E, { [1, 32] }> = convert_tile(acc);
        unsafe { op.store(narrowed, [pid.1, pid.0]) };
    }

    /// The same reduce with a plain store, for a `COLS` that 32 divides.
    #[wax_macro::entry()]
    pub fn splitk_reduce_packed<
        E: ElementType,
        const ROWS: i32,
        const COLS: i32,
        const SPLITS: i32,
        const PROWS: i32,
    >(
        out: &mut Tensor<E, { [ROWS, COLS] }>,
        part: &Tensor<f32, { [PROWS, COLS] }>,
    ) {
        let ts = const_shape![1, 32];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let pp = part.partition(ts);
        let mut acc: Tile<f32, { [1, 32] }> = constant(0.0f32, ts);
        for s in 0i32..SPLITS {
            acc = acc + pp.load([s * ROWS + pid.1, pid.0]);
        }
        let mut op: PartitionMut<E, { [1, 32] }> = unsafe { out.partition_full_mut(ts) };
        let narrowed: Tile<E, { [1, 32] }> = convert_tile(acc);
        unsafe { op.store(narrowed, [pid.1, pid.0]) };
    }
}

pub const MODULE_NAME: &str = "kernels";
pub const ENTRY: &str = "splitk_reduce";
pub const ENTRY_PACKED: &str = "splitk_reduce_packed";
pub const SOURCE_HASH: &str = kernels::SOURCE_HASH;

pub fn module_ast() -> ast::Module {
    kernels::__module_ast_self()
}
