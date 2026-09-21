//! `C[M, 1] = A[M, K] @ B[K]`. The N == 1 shape.

use wax_frontend::ast;

#[allow(unused_unsafe)]
#[wax_macro::module]
pub mod kernels {
    use cutile::core::*;

    #[wax_macro::entry()]
    pub fn gemv<E: ElementType, const BK: i32, const K: i32>(
        out: &mut Tensor<E, { [-1, 1] }>,
        a: &Tensor<E, { [-1, K] }>,
        v: &Tensor<E, { [1, K] }>,
    ) {
        let ts = const_shape![1, BK];
        let row: i32 = get_tile_block_id().0;
        let ap = a.partition(ts);
        let vp = v.partition(ts);
        // Explicit f32 to avoid accumulating 4096 products in f16.
        let mut acc: Tile<f32, { [1, BK] }> = constant(0.0f32, ts);
        for i in 0i32..(K / BK) {
            let xa: Tile<f32, { [1, BK] }> = convert_tile(ap.load([row, i]));
            let xv: Tile<f32, { [1, BK] }> = convert_tile(vp.load([0i32, i]));
            acc = acc + xa * xv;
        }
        let dot: Tile<f32, { [1] }> = reduce_sum(acc, 1i32);
        let narrowed: Tile<E, { [1, 1] }> = convert_tile(dot.reshape(const_shape![1, 1]));
        let mut op: PartitionMut<E, { [1, 1] }> =
            unsafe { out.partition_full_mut(const_shape![1, 1]) };
        unsafe { op.store(narrowed, [row, 0i32]) };
    }
}

pub const MODULE_NAME: &str = "kernels";
pub const ENTRY: &str = "gemv";
pub const SOURCE_HASH: &str = kernels::SOURCE_HASH;

pub fn module_ast() -> ast::Module {
    kernels::__module_ast_self()
}
