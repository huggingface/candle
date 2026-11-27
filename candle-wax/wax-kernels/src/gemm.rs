//! The canonical blocked GEMM: the tile loop over `mmaf`, shape-generic in M and N.
//!
//! Single kernel per element type, four entries, same computation:
//!
//! - `gemm`: the plain store. Correct when the tiling divides the shape.
//! - `gemm_padded`: the padded store. Correct for any shape. The padding masks the partial
//!   edge tile. It promotes the output's extents to kernel parameters, which is what makes
//!   the module shape-generic in M and N. The backend fills them from the problem it was
//!   handed.
//! - `gemm_batched`: `gemm` with batching on the grid's x axis.
//! - `gemm_batched_padded`: `gemm_batched` with padded store.

use wax_frontend::ast;

#[allow(unused_unsafe)]
#[wax_macro::module]
pub mod kernels {
    use cutile::core::*;

    #[wax_macro::entry()]
    pub fn gemm<E: ElementType, const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<E, { [BM, BN] }>,
        x: &Tensor<E, { [-1, K] }>,
        y: &Tensor<E, { [K, -1] }>,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for i in 0i32..((K + BK - 1) / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i, pid.1]);
            acc = mmaf(tile_x, tile_y, acc);
        }
        let out: Tile<E, { [BM, BN] }> = convert_tile(acc);
        z.store(out);
    }

    /// Same loop with a padded store. The store is `padding::Zero` on a mutable output,
    /// promoting `M` and `N` to kernel parameters, which the backend must bind correctly.
    #[wax_macro::entry()]
    pub fn gemm_padded<
        E: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const K: i32,
    >(
        z: &mut Tensor<E, { [-1, -1] }>,
        x: &Tensor<E, { [-1, K] }>,
        y: &Tensor<E, { [K, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let xp = x.partition(const_shape![BM, BK]);
        let yp = y.partition(const_shape![BK, BN]);
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for i in 0i32..((K + BK - 1) / BK) {
            acc = mmaf(xp.load([pid.0, i]), yp.load([i, pid.1]), acc);
        }
        let ztok: Token = get_tensor_token(z);
        let mut zp: PartitionMut<E, { [BM, BN] }> =
            unsafe { make_partition_view_mut(z, const_shape![BM, BN], padding::Zero, ztok) };
        let narrowed: Tile<E, { [BM, BN] }> = convert_tile(acc);
        unsafe { zp.store(narrowed, [pid.0, pid.1]) };
    }

    /// Batched gemm on the grid's x axis.
    #[wax_macro::entry()]
    pub fn gemm_batched<
        E: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const K: i32,
    >(
        z: &mut Tensor<E, { [1, BM, BN] }>,
        x: &Tensor<E, { [-1, -1, K] }>,
        y: &Tensor<E, { [-1, K, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let batch_idx = pid.0;
        let m_idx = pid.1;
        let n_idx = pid.2;
        let part_x = x.partition(const_shape![1, BM, BK]);
        let part_y = y.partition(const_shape![1, BK, BN]);
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for i in 0i32..((K + BK - 1) / BK) {
            let tile_x: Tile<E, { [BM, BK] }> = part_x
                .load([batch_idx, m_idx, i])
                .reshape(const_shape![BM, BK]);
            let tile_y: Tile<E, { [BK, BN] }> = part_y
                .load([batch_idx, i, n_idx])
                .reshape(const_shape![BK, BN]);
            acc = mmaf(tile_x, tile_y, acc);
        }
        let out: Tile<E, { [BM, BN] }> = convert_tile(acc);
        z.store(out.reshape(const_shape![1, BM, BN]));
    }

    /// `gemm_batched` with padded store.
    #[wax_macro::entry()]
    pub fn gemm_batched_padded<
        E: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const K: i32,
    >(
        z: &mut Tensor<E, { [-1, -1, -1] }>,
        x: &Tensor<E, { [-1, -1, K] }>,
        y: &Tensor<E, { [-1, K, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let part_x = x.partition(const_shape![1, BM, BK]);
        let part_y = y.partition(const_shape![1, BK, BN]);
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for i in 0i32..((K + BK - 1) / BK) {
            let tile_x: Tile<E, { [BM, BK] }> =
                part_x.load([pid.0, pid.1, i]).reshape(const_shape![BM, BK]);
            let tile_y: Tile<E, { [BK, BN] }> =
                part_y.load([pid.0, i, pid.2]).reshape(const_shape![BK, BN]);
            acc = mmaf(tile_x, tile_y, acc);
        }
        let ztok: Token = get_tensor_token(z);
        let mut zp: PartitionMut<E, { [1, BM, BN] }> =
            unsafe { make_partition_view_mut(z, const_shape![1, BM, BN], padding::Zero, ztok) };
        let flat: Tile<E, { [BM, BN] }> = convert_tile(acc);
        let narrowed: Tile<E, { [1, BM, BN] }> = flat.reshape(const_shape![1, BM, BN]);
        unsafe { zp.store(narrowed, [pid.0, pid.1, pid.2]) };
    }
}

pub const MODULE_NAME: &str = "kernels";
pub const ENTRY: &str = "gemm";
pub const ENTRY_PADDED: &str = "gemm_padded";
pub const ENTRY_BATCHED: &str = "gemm_batched";
pub const ENTRY_BATCHED_PADDED: &str = "gemm_batched_padded";
pub const SOURCE_HASH: &str = kernels::SOURCE_HASH;

pub fn module_ast() -> ast::Module {
    kernels::__module_ast_self()
}
