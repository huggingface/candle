// Ported verbatim from the metal-tile-rs POC (metal-tile-bench tests/src/benches):
// its 119 `#[metal_tile::entry()]` kernels, import-swapped for candle3
// (`#[metal_tile::entry()]` -> `#[wax::entry()]`), bodies otherwise unedited.
// Identical duplicate definitions across POC files are emitted once; the same
// name with a DIFFERENT body is suffixed `__<file>`. Two kernels (`bitops`,
// `shr_u32`) were not ported: they use integer tile shifts, which the POC's
// vendored DSL exposes but candle3's cutile frontend does not.

mod bench {
    use cutile::core::*;

        /// Port of the POC's `StoreTileAtCurrentBlock::store` trait body (`store_tile_2d`):
        /// candle3's pinned cutile frontend does not resolve the trait method, but every
        /// primitive it inlines to is supported, so the body is inlined here.
        fn store_current_block_2d<E: ElementType, const S: [i32; 2]>(
            y: &mut Tensor<E, S>,
            result: Tile<E, S>,
        ) {
            let pid: (i32, i32, i32) = get_tile_block_id();
            let tile_shape: Shape<S> = Shape::<S> { dims: &[] };
            let tensor_token: Token = get_tensor_token(y);
            let mut y_partition: PartitionMut<E, S> =
                unsafe { make_partition_view_mut(y, tile_shape, padding::None, tensor_token) };
            unsafe {
                store_view_tko_mut(
                    &mut y_partition,
                    result,
                    [pid.0, pid.1],
                    ordering::Weak,
                    scope::TileBlock,
                    None,
                    tma::Enabled,
                )
            };
            let new_token: Token = get_partition_token_mut(&y_partition);
            set_tensor_token(y, new_token);
        }

        /// Same, with the POC `gemm_dtype_out` narrowing: the f32 SGMMA accumulator is
        /// down-converted into an `E`-typed tensor on the way out.
        fn store_narrow_2d<E: ElementType, const S: [i32; 2]>(
            y: &mut Tensor<E, S>,
            result: Tile<f32, S>,
        ) {
            let narrowed: Tile<E, S> = convert_tile(result);
            store_current_block_2d(y, narrowed);
        }

        #[wax::entry()]
        pub fn gemm<E: ElementType, const BM: i32, const BN: i32, const BK: i32, const K: i32>(
            z: &mut Tensor<f32, { [BM, BN] }>,
            x: &Tensor<E, { [-1, K] }>,
            y: &Tensor<E, { [K, -1] }>,
        ) {
            let part_x = x.partition(const_shape![BM, BK]);
            let part_y = y.partition(const_shape![BK, BN]);
            let pid: (i32, i32, i32) = get_tile_block_id();
            // Explicit f32: the frontend infers `mmaf`'s accumulator element from the
            // inputs (f16/bf16) when the type is left open, but the SGMMA register
            // - and the POC's reference lowering - accumulate in f32.
            let mut tile_z: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            for i in 0i32..(K / BK) {
                let tile_x = part_x.load([pid.0, i]);
                let tile_y = part_y.load([i, pid.1]);
                tile_z = mmaf(tile_x, tile_y, tile_z);
            }
            store_current_block_2d(z, tile_z);
        }

        #[wax::entry()]
        pub fn gemm__gemm_kernels<E: ElementType, const BM: i32, const BN: i32, const BK: i32, const K: i32>(
            z: &mut Tensor<E, { [BM, BN] }>,
            x: &Tensor<E, { [-1, K] }>,
            y: &Tensor<E, { [K, -1] }>,
        ) {
            let part_x = x.partition(const_shape![BM, BK]);
            let part_y = y.partition(const_shape![BK, BN]);
            let pid: (i32, i32, i32) = get_tile_block_id();
            // Explicit f32: the frontend infers `mmaf`'s accumulator element from the
            // inputs (f16/bf16) when the type is left open, but the SGMMA register
            // - and the POC's reference lowering - accumulate in f32.
            let mut tile_z: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            for i in 0i32..(K / BK) {
                let tile_x = part_x.load([pid.0, i]);
                let tile_y = part_y.load([i, pid.1]);
                tile_z = mmaf(tile_x, tile_y, tile_z);
            }
            store_narrow_2d(z, tile_z);
        }

        #[wax::entry()]
        pub fn gemm__bench_shapes<E: ElementType, const BM: i32, const BN: i32, const BK: i32, const K: i32>(
            z: &mut Tensor<E, { [BM, BN] }>,
            x: &Tensor<E, { [-1, K] }>,
            y: &Tensor<E, { [K, -1] }>,
        ) {
            let part_x = x.partition(const_shape![BM, BK]);
            let part_y = y.partition(const_shape![BK, BN]);
            let pid: (i32, i32, i32) = get_tile_block_id();
            let mut tile_z = constant(E::ZERO, const_shape![BM, BN]);
            for i in 0i32..(K / BK) {
                let tile_x = part_x.load([pid.0, i]);
                let tile_y = part_y.load([i, pid.1]);
                tile_z = mma(tile_x, tile_y, tile_z);
            }
            store_current_block_2d(z, tile_z);
        }

        #[wax::entry()]
        pub fn gemv<E: ElementType, const BK: i32, const K: i32>(
            a: &Tensor<E, { [-1, K] }>,
            vec: &Tensor<E, { [-1, K] }>,
            output: &mut Tensor<E, { [-1, BK] }>,
        ) {
            let tile_shape = const_shape![1, BK];
            let pid = get_tile_block_id();
            let row = pid.0;
            let a_part = a.partition(tile_shape);
            let v_part = vec.partition(tile_shape);
            let num_chunks = K / BK;
            let mut acc: Tile<E, { [1, BK] }> = a_part.load([row, 0i32]) * v_part.load([0i32, 0i32]);
            for i in 1i32..num_chunks {
                let ac = a_part.load([row, i]);
                let vc = v_part.load([0i32, i]);
                acc = acc + ac * vc;
            }
            let dot: Tile<E, { [1] }> = reduce_sum(acc, 1i32);
            let dot_bc: Tile<E, { [1, BK] }> = dot.reshape(const_shape![1, 1]).broadcast(tile_shape);
            let mut out_part: PartitionMut<E, { [1, BK] }> =
                unsafe { output.partition_full_mut(tile_shape) };
            unsafe { out_part.store(dot_bc, [row, 0i32]) };
        }

        #[wax::entry()]
        pub fn gemv__gemm_kernels<E: ElementType, const BK: i32, const K: i32>(
            output: &mut Tensor<E, { [-1, BK] }>,
            a: &Tensor<E, { [-1, K] }>,
            vec: &Tensor<E, { [-1, K] }>,
        ) {
            let tile_shape = const_shape![1, BK];
            let pid = get_tile_block_id();
            let row = pid.0;
            let a_part = a.partition(tile_shape);
            let v_part = vec.partition(tile_shape);
            let num_chunks = K / BK;
            let mut acc: Tile<E, { [1, BK] }> = a_part.load([row, 0i32]) * v_part.load([0i32, 0i32]);
            for i in 1i32..num_chunks {
                let ac = a_part.load([row, i]);
                let vc = v_part.load([0i32, i]);
                acc = acc + ac * vc;
            }
            let dot: Tile<E, { [1] }> = reduce_sum(acc, 1i32);
            let dot_bc: Tile<E, { [1, BK] }> = dot.reshape(const_shape![1, 1]).broadcast(tile_shape);
            let mut out_part: PartitionMut<E, { [1, BK] }> =
                unsafe { output.partition_full_mut(tile_shape) };
            unsafe { out_part.store(dot_bc, [row, 0i32]) };
        }

        #[wax::entry()]
        pub fn gemv__epilogue_fusion<E: ElementType, const BK: i32, const K: i32>(
            a: &Tensor<E, { [-1, K] }>,
            vec: &Tensor<E, { [-1, K] }>,
            out: &mut Tensor<E, { [-1, 1] }>,
        ) {
            let ts = const_shape![1, BK];
            let row = get_tile_block_id().0;
            let ap = a.partition(ts);
            let vp = vec.partition(ts);
            let nc = K / BK;
            let a0: Tile<f32, { [1, BK] }> = convert_tile(ap.load([row, 0i32]));
            let v0: Tile<f32, { [1, BK] }> = convert_tile(vp.load([0i32, 0i32]));
            let mut acc = a0 * v0;
            for i in 1i32..nc {
                let ai: Tile<f32, { [1, BK] }> = convert_tile(ap.load([row, i]));
                let vi: Tile<f32, { [1, BK] }> = convert_tile(vp.load([0i32, i]));
                acc = acc + ai * vi;
            }
            let dot: Tile<f32, { [1] }> = reduce_sum(acc, 1i32);
            let out_ts = const_shape![1, 1];
            let dot1: Tile<E, { [1, 1] }> = convert_tile(dot.reshape(out_ts));
            let mut op: PartitionMut<E, { [1, 1] }> = unsafe { out.partition_full_mut(out_ts) };
            unsafe { op.store(dot1, [row, 0i32]) };
        }

        #[wax::entry()]
        pub fn gemv__gemv_verify<E: ElementType, const BN: i32, const BK: i32, const K: i32>(
            z: &mut Tensor<E, { [1, BN] }>,
            x: &Tensor<E, { [-1, K] }>,
            y: &Tensor<E, { [K, -1] }>,
        ) {
            let part_x = x.partition(const_shape![1, BK]);
            let part_y = y.partition(const_shape![BK, BN]);
            let pid: (i32, i32, i32) = get_tile_block_id();
            let mut tile_z = constant(E::ZERO, const_shape![1, BN]);
            for i in 0i32..(K / BK) {
                let tile_x = part_x.load([pid.0, i]);
                let tile_y = part_y.load([i, pid.1]);
                tile_z = mma(tile_x, tile_y, tile_z);
            }
            store_current_block_2d(z, tile_z);
        }

        #[wax::entry()]
        pub fn rms_norm<E: ElementType, const N: i32, const BLOCK_SIZE: i32>(
            x: &Tensor<E, { [-1, N] }>,
            w: &Tensor<E, { [-1] }>,
            out: &mut Tensor<E, { [1, N] }>,
        ) {
            let tile_shape = const_shape![1, BLOCK_SIZE];
            let pid = get_tile_block_id();
            let row = pid.0;
            let num_chunks = N / BLOCK_SIZE;

            let x_part = x.partition(tile_shape);
            let mut rms = constant(0.0f32, tile_shape);
            for j in 0i32..num_chunks {
                let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(x_part.load([row, j]));
                rms = rms + tx * tx;
            }

            let rms: Tile<f32, { [1] }> = reduce_sum(rms, 1i32);
            let rms: Tile<f32, { [] }> = rms.reshape(const_shape![]);
            let rms: f32 = tile_to_scalar(rms);
            let n: f32 = convert_scalar(N);
            let one: f32 = convert_scalar(1.0f32);
            let eps: f32 = convert_scalar(1e-5f32);
            let rms: f32 = one / (rms / n + eps);
            let rms: Tile<f32, { [] }> =
                sqrt(scalar_to_tile(rms), rounding::NegativeInf, ftz::Disabled);
            let rms: f32 = tile_to_scalar(rms);
            let rms: Tile<f32, { [1, BLOCK_SIZE] }> = broadcast_scalar(rms, tile_shape);

            let w_part = w.partition(const_shape![BLOCK_SIZE]);
            let mut out_part: PartitionMut<E, { [1, BLOCK_SIZE] }> =
                unsafe { out.partition_full_mut(tile_shape) };
            for j in 0i32..num_chunks {
                let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(x_part.load([row, j]));
                let tw: Tile<f32, { [1, BLOCK_SIZE] }> =
                    convert_tile(w_part.load([j]).reshape(tile_shape));
                let y: Tile<E, { [1, BLOCK_SIZE] }> = convert_tile(tx * rms * tw);
                unsafe { out_part.store(y, [row, j]) };
            }
        }

        #[wax::entry()]
        pub fn rms_norm__bitcode_vs_metal_as<const N: i32, const BLOCK_SIZE: i32>(
            x: &Tensor<f32, { [-1, N] }>,
            w: &Tensor<f32, { [-1] }>,
            out: &mut Tensor<f32, { [-1, N] }>,
        ) {
            let tile_shape = const_shape![1, BLOCK_SIZE];
            let pid = get_tile_block_id();
            let row = pid.0;
            let num_chunks = N / BLOCK_SIZE;

            let x_part = x.partition(tile_shape);
            let mut rms = constant(0.0f32, tile_shape);
            for j in 0i32..num_chunks {
                let tx = x_part.load([row, j]);
                rms = rms + tx * tx;
            }

            let rms: Tile<f32, { [1] }> = reduce_sum(rms, 1i32);
            let rms: Tile<f32, { [] }> = rms.reshape(const_shape![]);
            let rms: f32 = tile_to_scalar(rms);
            let n: f32 = convert_scalar(N);
            let one: f32 = convert_scalar(1.0f32);
            let eps: f32 = convert_scalar(1e-5f32);
            let rms: f32 = one / (rms / n + eps);
            let rms: Tile<f32, { [] }> =
                sqrt(scalar_to_tile(rms), rounding::NegativeInf, ftz::Disabled);
            let rms: f32 = tile_to_scalar(rms);
            let rms: Tile<f32, { [1, BLOCK_SIZE] }> = broadcast_scalar(rms, tile_shape);

            let w_part = w.partition(const_shape![BLOCK_SIZE]);
            let mut out_part: PartitionMut<f32, { [1, BLOCK_SIZE] }> =
                unsafe { out.partition_full_mut(tile_shape) };
            for j in 0i32..num_chunks {
                let tx = x_part.load([row, j]);
                let tw = w_part.load([j]).reshape(tile_shape);
                unsafe { out_part.store(tx * rms * tw, [row, j]) };
            }
        }

        #[wax::entry()]
        pub fn rms_norm__prop<const N: i32, const BLOCK_SIZE: i32>(
            x: &Tensor<f32, { [-1, N] }>,
            w: &Tensor<f32, { [-1] }>,
            out: &mut Tensor<f32, { [1, N] }>,
        ) {
            let tile_shape = const_shape![1, BLOCK_SIZE];
            let pid = get_tile_block_id();
            let row = pid.0;
            let num_chunks = N / BLOCK_SIZE;

            let x_part = x.partition(tile_shape);
            let mut rms = constant(0.0f32, tile_shape);
            for j in 0i32..num_chunks {
                let tx = x_part.load([row, j]);
                rms = rms + tx * tx;
            }

            let rms: Tile<f32, { [1] }> = reduce_sum(rms, 1i32);
            let rms: Tile<f32, { [] }> = rms.reshape(const_shape![]);
            let rms: f32 = tile_to_scalar(rms);
            let n: f32 = convert_scalar(N);
            let one: f32 = convert_scalar(1.0f32);
            let eps: f32 = convert_scalar(1e-5f32);
            let rms: f32 = one / (rms / n + eps);
            let rms: Tile<f32, { [] }> =
                sqrt(scalar_to_tile(rms), rounding::NegativeInf, ftz::Disabled);
            let rms: f32 = tile_to_scalar(rms);
            let rms: Tile<f32, { [1, BLOCK_SIZE] }> = broadcast_scalar(rms, tile_shape);

            let w_part = w.partition(const_shape![BLOCK_SIZE]);
            let mut out_part: PartitionMut<f32, { [1, BLOCK_SIZE] }> =
                unsafe { out.partition_full_mut(tile_shape) };
            for j in 0i32..num_chunks {
                let tx = x_part.load([row, j]);
                let tw = w_part.load([j]).reshape(tile_shape);
                unsafe { out_part.store(tx * rms * tw, [row, j]) };
            }
        }

        #[wax::entry()]
        pub fn softmax<E: ElementType, const N: i32, const BLOCK_SIZE: i32>(
            x: &Tensor<E, { [-1, N] }>,
            out: &mut Tensor<E, { [1, N] }>,
        ) {
            let tile_shape = const_shape![1, BLOCK_SIZE];
            let pid = get_tile_block_id();
            let row = pid.0;
            let num_chunks = N / BLOCK_SIZE;

            let x_part = x.partition(tile_shape);

            // Pass 1: row-wise maximum.
            let mut gmax: Tile<E, { [1] }> = reduce_max(x_part.load([row, 0i32]), 1i32);
            for j in 1i32..num_chunks {
                let chunk_max = reduce_max(x_part.load([row, j]), 1i32);
                gmax = maxf(gmax, chunk_max, nan::Disabled, ftz::Disabled);
            }
            let gmax_bc: Tile<E, { [1, BLOCK_SIZE] }> =
                gmax.reshape(const_shape![1, 1]).broadcast(tile_shape);

            // Pass 2: sum of exp(x - max).
            let first_exp = exp(x_part.load([row, 0i32]) - gmax_bc);
            let mut gsum: Tile<E, { [1] }> = reduce_sum(first_exp, 1i32);
            for j in 1i32..num_chunks {
                let num = exp(x_part.load([row, j]) - gmax_bc);
                gsum = gsum + reduce_sum(num, 1i32);
            }
            let gsum_bc: Tile<E, { [1, BLOCK_SIZE] }> =
                gsum.reshape(const_shape![1, 1]).broadcast(tile_shape);

            // Pass 3: normalise and store.
            let mut out_part: PartitionMut<E, { [1, BLOCK_SIZE] }> =
                unsafe { out.partition_full_mut(tile_shape) };
            for j in 0i32..num_chunks {
                let num = exp(x_part.load([row, j]) - gmax_bc);
                unsafe { out_part.store(num / gsum_bc, [row, j]) };
            }
        }

        #[wax::entry()]
        pub fn softmax__bitcode_vs_metal_as<const N: i32, const BLOCK_SIZE: i32>(
            x: &Tensor<f32, { [-1, N] }>,
            out: &mut Tensor<f32, { [-1, N] }>,
        ) {
            let tile_shape = const_shape![1, BLOCK_SIZE];
            let pid = get_tile_block_id();
            let row = pid.0;
            let num_chunks = N / BLOCK_SIZE;

            let x_part = x.partition(tile_shape);

            let mut gmax: Tile<f32, { [1] }> = reduce_max(x_part.load([row, 0i32]), 1i32);
            for j in 1i32..num_chunks {
                let chunk_max = reduce_max(x_part.load([row, j]), 1i32);
                gmax = maxf(gmax, chunk_max, nan::Disabled, ftz::Disabled);
            }
            let gmax_bc: Tile<f32, { [1, BLOCK_SIZE] }> =
                gmax.reshape(const_shape![1, 1]).broadcast(tile_shape);

            let first_exp = exp(x_part.load([row, 0i32]) - gmax_bc);
            let mut gsum: Tile<f32, { [1] }> = reduce_sum(first_exp, 1i32);
            for j in 1i32..num_chunks {
                let num = exp(x_part.load([row, j]) - gmax_bc);
                gsum = gsum + reduce_sum(num, 1i32);
            }
            let gsum_bc: Tile<f32, { [1, BLOCK_SIZE] }> =
                gsum.reshape(const_shape![1, 1]).broadcast(tile_shape);

            let mut out_part: PartitionMut<f32, { [1, BLOCK_SIZE] }> =
                unsafe { out.partition_full_mut(tile_shape) };
            for j in 0i32..num_chunks {
                let num = exp(x_part.load([row, j]) - gmax_bc);
                unsafe { out_part.store(num / gsum_bc, [row, j]) };
            }
        }

        #[wax::entry()]
        pub fn softmax__prop<const N: i32, const BLOCK_SIZE: i32>(
            x: &Tensor<f32, { [-1, N] }>,
            out: &mut Tensor<f32, { [1, N] }>,
        ) {
            let tile_shape = const_shape![1, BLOCK_SIZE];
            let pid = get_tile_block_id();
            let row = pid.0;
            let num_chunks = N / BLOCK_SIZE;

            let x_part = x.partition(tile_shape);

            let mut gmax: Tile<f32, { [1] }> = reduce_max(x_part.load([row, 0i32]), 1i32);
            for j in 1i32..num_chunks {
                let chunk_max = reduce_max(x_part.load([row, j]), 1i32);
                gmax = maxf(gmax, chunk_max, nan::Disabled, ftz::Disabled);
            }
            let gmax_bc: Tile<f32, { [1, BLOCK_SIZE] }> =
                gmax.reshape(const_shape![1, 1]).broadcast(tile_shape);

            let first_exp = exp(x_part.load([row, 0i32]) - gmax_bc);
            let mut gsum: Tile<f32, { [1] }> = reduce_sum(first_exp, 1i32);
            for j in 1i32..num_chunks {
                let num = exp(x_part.load([row, j]) - gmax_bc);
                gsum = gsum + reduce_sum(num, 1i32);
            }
            let gsum_bc: Tile<f32, { [1, BLOCK_SIZE] }> =
                gsum.reshape(const_shape![1, 1]).broadcast(tile_shape);

            let mut out_part: PartitionMut<f32, { [1, BLOCK_SIZE] }> =
                unsafe { out.partition_full_mut(tile_shape) };
            for j in 0i32..num_chunks {
                let num = exp(x_part.load([row, j]) - gmax_bc);
                unsafe { out_part.store(num / gsum_bc, [row, j]) };
            }
        }

        #[wax::entry()]
        pub fn sdpa_sd_d64<
            E: ElementType,
            const NH: i32,
            const NKV: i32,
            const BN: i32,
            const BN32: i32,
        >(
            q: &Tensor<E, { [-1, 64] }>,
            k: &Tensor<E, { [-1, 64] }>,
            v: &Tensor<E, { [-1, 64] }>,
            meta: &Tensor<f32, { [-1] }>,
            out: &mut Tensor<E, { [-1, 64] }>,
        ) {
            let chunk = const_shape![1, 32];
            let wide = const_shape![BN, 32];
            let one = const_shape![1];
            let head = get_tile_block_id().0;

            let mp = meta.partition(one);
            // Pinned in two steps: `convert_scalar` is generic in both directions, so wrapping the
            // read in it leaves nothing to fix the tile's element type or rank. Line below reads the
            // same way and never needed it, because `f32` there pins both.
            let n_f: f32 = tile_to_scalar(mp.load([0i32]).reshape(const_shape![]));
            let n: i32 = convert_scalar(n_f);
            let scale: f32 = tile_to_scalar(mp.load([1i32]).reshape(const_shape![]));

            let bi = head / NH;
            let h = head % NH;
            let kvh = h / (NH / NKV);
            // The row offset, in COLUMN units: 64 elements per row / 32 per tile column = 2.
            let col0 = (bi * NKV + kvh) * n * 2i32;

            let scale_bc: Tile<f32, { [1, 32] }> = broadcast_scalar(scale, chunk);
            let qp = q.partition(chunk);
            let tq0: Tile<f32, { [1, 32] }> = convert_tile(qp.load([head, 0i32])) * scale_bc;
            let tq1: Tile<f32, { [1, 32] }> = convert_tile(qp.load([head, 1i32])) * scale_bc;
            // An identity in registers: every row already holds this head's q on its lanes.
            let tq0w: Tile<f32, { [BN, 32] }> = tq0.broadcast(wide);
            let tq1w: Tile<f32, { [BN, 32] }> = tq1.broadcast(wide);

            let kp = k.partition(wide);
            let vp = v.partition(wide);

            // FULL tiles first, then ONE masked tail. Masking every iteration costs the inner
            // loop a compare and a select it does not need - measured at 0.84x against the
            // `get_simd_group_id` spelling, which avoided the mask entirely by giving each
            // group its own `cnt`. The tail runs unconditionally: when `n % BN == 0` its mask
            // is all-false, every row takes -inf, and `exp(-inf - m)` contributes nothing while
            // `max(m, -inf)` leaves the running max alone.
            let full = n / BN;
            let mut m_i: Tile<f32, { [BN, 32] }> = broadcast_scalar(-3.4028235e38f32, wide);
            let mut l_i: Tile<f32, { [BN, 32] }> = constant(0.0f32, wide);
            let mut acc0: Tile<f32, { [BN, 32] }> = constant(0.0f32, wide);
            let mut acc1: Tile<f32, { [BN, 32] }> = constant(0.0f32, wide);
            for jj in 0i32..full {
                let tk0: Tile<f32, { [BN, 32] }> = convert_tile(kp.load([jj, col0]));
                let tk1: Tile<f32, { [BN, 32] }> = convert_tile(kp.load([jj, col0 + 1i32]));
                let dot_r: Tile<f32, { [BN] }> = reduce_sum(tq0w * tk0 + tq1w * tk1, 1i32);
                let dot: Tile<f32, { [BN, 32] }> = dot_r.reshape(const_shape![BN, 1]).broadcast(wide);
                let m_ij: Tile<f32, { [BN, 32] }> = maxf(m_i, dot, nan::Disabled, ftz::Disabled);
                let alpha: Tile<f32, { [BN, 32] }> = exp(m_i - m_ij);
                let p_val: Tile<f32, { [BN, 32] }> = exp(dot - m_ij);
                l_i = l_i * alpha + p_val;
                let tv0: Tile<f32, { [BN, 32] }> = convert_tile(vp.load([jj, col0]));
                let tv1: Tile<f32, { [BN, 32] }> = convert_tile(vp.load([jj, col0 + 1i32]));
                acc0 = acc0 * alpha + tv0 * p_val;
                acc1 = acc1 * alpha + tv1 * p_val;
                m_i = m_ij;
            }
            // Only when there IS a tail. At an aligned `n` this whole block is pure waste -
            // one extra iteration out of `full`, which is 6% of the work at n=512 and 0.8% at
            // n=4096, and that is exactly the residual it was costing.
            if n - full * BN > 0i32 {
                let tk0: Tile<f32, { [BN, 32] }> = convert_tile(kp.load([full, col0]));
                let tk1: Tile<f32, { [BN, 32] }> = convert_tile(kp.load([full, col0 + 1i32]));
                let dot_r: Tile<f32, { [BN] }> = reduce_sum(tq0w * tk0 + tq1w * tk1, 1i32);
                let dot0: Tile<f32, { [BN, 32] }> = dot_r.reshape(const_shape![BN, 1]).broadcast(wide);
                // `s*32` for this thread's row (see 2 above), and the pad value - both live
                // ONLY here. Hoisting them above the loop kept three extra registers alive
                // through it, and this kernel trades registers for occupancy.
                let iw1: Tile<i32, { [BN32] }> = iota(const_shape![BN32]);
                let iw: Tile<i32, { [BN, 32] }> = iw1.reshape(wide);
                let il1: Tile<i32, { [32] }> = iota(const_shape![32]);
                let il2: Tile<i32, { [1, 32] }> = il1.reshape(chunk);
                let il: Tile<i32, { [BN, 32] }> = il2.broadcast(wide);
                let s32: Tile<i32, { [BN, 32] }> = iw - il;
                let ninf: Tile<f32, { [BN, 32] }> = broadcast_scalar(-3.4028235e38f32, wide);
                let lim: Tile<i32, { [BN, 32] }> = broadcast_scalar(32i32 * (n - full * BN), wide);
                let inb: Tile<bool, { [BN, 32] }> = cmpi(s32, lim, predicate::LessThan);
                let dot: Tile<f32, { [BN, 32] }> = select(inb, dot0, ninf);
                let m_ij: Tile<f32, { [BN, 32] }> = maxf(m_i, dot, nan::Disabled, ftz::Disabled);
                let alpha: Tile<f32, { [BN, 32] }> = exp(m_i - m_ij);
                let p_val: Tile<f32, { [BN, 32] }> = exp(dot - m_ij);
                l_i = l_i * alpha + p_val;
                // The V rows past key `n` must be NEUTRALISED, not merely down-weighted. The
                // mask above only reaches the SCORES: it drives `p_val` to zero for an
                // out-of-range row, and zero times a finite garbage value is zero. But the
                // bytes past the end of a RECYCLED kv-cache buffer are not finite - read as
                // f16 the all-ones exponent turns up constantly - and `inf * 0` is NaN, which
                // then rides `acc` through every remaining iteration and out into the logits.
                // A fresh allocation is zero-filled, so this is invisible until the allocator
                // starts handing back used memory.
                let vzero: Tile<f32, { [BN, 32] }> = constant(0.0f32, wide);
                let tv0r: Tile<f32, { [BN, 32] }> = convert_tile(vp.load([full, col0]));
                let tv1r: Tile<f32, { [BN, 32] }> = convert_tile(vp.load([full, col0 + 1i32]));
                let tv0: Tile<f32, { [BN, 32] }> = select(inb, tv0r, vzero);
                let tv1: Tile<f32, { [BN, 32] }> = select(inb, tv1r, vzero);
                acc0 = acc0 * alpha + tv0 * p_val;
                acc1 = acc1 * alpha + tv1 * p_val;
                m_i = m_ij;
            }
            // The merge the old kernel already used, now over state that is a tile to begin
            // with: max over rows, rescaled sum, each accumulator weighted by exp(m_s - gm).
            // `m_i` and `l_i` are lane-INVARIANT: `dot` is broadcast across each row, so every
            // lane of row s carries row s's value. The uniformity analysis cannot see that -
            // it traces them back to a divergent load - and a column reduce over a source it
            // believes divergent pays a full threadgroup redistribution. Passing them through
            // a WITHIN-ROW reduce first is a value no-op (all lanes already agree, hence `max`
            // and not `sum`) that makes the invariance visible, and a within-SIMD reduce is the
            // analysis's convergence point. This is worth 8 of the kernel's 16 barriers - the
            // `get_simd_group_id` spelling got the same elision for free, because it reduced to
            // a scalar and re-broadcast it.
            let m_row: Tile<f32, { [BN] }> = reduce_max(m_i, 1i32);
            let mt: Tile<f32, { [BN, 32] }> = m_row.reshape(const_shape![BN, 1]).broadcast(wide);
            let l_row: Tile<f32, { [BN] }> = reduce_max(l_i, 1i32);
            let lt: Tile<f32, { [BN, 32] }> = l_row.reshape(const_shape![BN, 1]).broadcast(wide);
            let gm_w: Tile<f32, { [32] }> = reduce_max(mt, 0i32);
            let gm: Tile<f32, { [BN, 32] }> = gm_w.reshape(chunk).broadcast(wide);
            let f: Tile<f32, { [BN, 32] }> = exp(mt - gm);
            let gl_w: Tile<f32, { [32] }> = reduce_sum(lt * f, 0i32);
            let gl: Tile<f32, { [1, 32] }> = gl_w.reshape(chunk);
            // Hoist BOTH products before BOTH reduces. `batch_column_reduces` shares one set of
            // barriers across ADJACENT independent column reduces, and writing the multiply
            // inline (`reduce_sum(acc0 * f, 0)`) interleaves mul/reduce/mul/reduce so the two
            // reduces are no longer adjacent and each pays its own barriers.
            let a0: Tile<f32, { [BN, 32] }> = acc0 * f;
            let a1: Tile<f32, { [BN, 32] }> = acc1 * f;
            let n0: Tile<f32, { [32] }> = reduce_sum(a0, 0i32);
            let n1: Tile<f32, { [32] }> = reduce_sum(a1, 0i32);
            let o0: Tile<E, { [1, 32] }> = convert_tile(n0.reshape(chunk) / gl);
            let o1: Tile<E, { [1, 32] }> = convert_tile(n1.reshape(chunk) / gl);
            let mut op: PartitionMut<E, { [1, 32] }> = unsafe { out.partition_full_mut(chunk) };
            unsafe { op.store(o0, [head, 0i32]) };
            unsafe { op.store(o1, [head, 1i32]) };
        }

        #[wax::entry()]
        pub fn sdpa_sd_d64__candle_wide_kernels<
            E: ElementType,
            const NH: i32,
            const NKV: i32,
            const BN: i32,
            const BN32: i32,
        >(
            q: &Tensor<E, { [-1, 64] }>,
            k: &Tensor<E, { [-1, 64] }>,
            v: &Tensor<E, { [-1, 64] }>,
            meta: &Tensor<f32, { [-1] }>,
            out: &mut Tensor<E, { [-1, 64] }>,
        ) {
            let chunk = const_shape![1, 32];
            let wide = const_shape![BN, 32];
            let one = const_shape![1];
            let head = get_tile_block_id().0;

            let mp = meta.partition(one);
            let n_f: f32 = tile_to_scalar(mp.load([0i32]).reshape(const_shape![]));
            let n: i32 = convert_scalar(n_f);
            let scale: f32 = tile_to_scalar(mp.load([1i32]).reshape(const_shape![]));
            let khs_f: f32 = tile_to_scalar(mp.load([2i32]).reshape(const_shape![]));
            let khs: i32 = convert_scalar(khs_f);

            let bi = head / NH;
            let h = head % NH;
            let kvh = h / (NH / NKV);
            // The row offset, in COLUMN units: 64 elements per row / 32 per tile column = 2.
            let col0 = (bi * NKV + kvh) * khs * 2i32;

            let scale_bc: Tile<f32, { [1, 32] }> = broadcast_scalar(scale, chunk);
            let qp = q.partition(chunk);
            let tq0: Tile<f32, { [1, 32] }> = convert_tile(qp.load([head, 0i32])) * scale_bc;
            let tq1: Tile<f32, { [1, 32] }> = convert_tile(qp.load([head, 1i32])) * scale_bc;
            // An identity in registers: every row already holds this head's q on its lanes.
            let tq0w: Tile<f32, { [BN, 32] }> = tq0.broadcast(wide);
            let tq1w: Tile<f32, { [BN, 32] }> = tq1.broadcast(wide);

            let kp = k.partition(wide);
            let vp = v.partition(wide);

            // FULL tiles first, then ONE masked tail. Masking every iteration costs the inner
            // loop a compare and a select it does not need - measured at 0.84x against the
            // `get_simd_group_id` spelling, which avoided the mask entirely by giving each
            // group its own `cnt`. The tail runs unconditionally: when `n % BN == 0` its mask
            // is all-false, every row takes -inf, and `exp(-inf - m)` contributes nothing while
            // `max(m, -inf)` leaves the running max alone.
            let full = n / BN;
            let mut m_i: Tile<f32, { [BN, 32] }> = broadcast_scalar(-3.4028235e38f32, wide);
            let mut l_i: Tile<f32, { [BN, 32] }> = constant(0.0f32, wide);
            let mut acc0: Tile<f32, { [BN, 32] }> = constant(0.0f32, wide);
            let mut acc1: Tile<f32, { [BN, 32] }> = constant(0.0f32, wide);
            for jj in 0i32..full {
                let tk0: Tile<f32, { [BN, 32] }> = convert_tile(kp.load([jj, col0]));
                let tk1: Tile<f32, { [BN, 32] }> = convert_tile(kp.load([jj, col0 + 1i32]));
                let dot_r: Tile<f32, { [BN] }> = reduce_sum(tq0w * tk0 + tq1w * tk1, 1i32);
                let dot: Tile<f32, { [BN, 32] }> = dot_r.reshape(const_shape![BN, 1]).broadcast(wide);
                let m_ij: Tile<f32, { [BN, 32] }> = maxf(m_i, dot, nan::Disabled, ftz::Disabled);
                let alpha: Tile<f32, { [BN, 32] }> = exp(m_i - m_ij);
                let p_val: Tile<f32, { [BN, 32] }> = exp(dot - m_ij);
                l_i = l_i * alpha + p_val;
                let tv0: Tile<f32, { [BN, 32] }> = convert_tile(vp.load([jj, col0]));
                let tv1: Tile<f32, { [BN, 32] }> = convert_tile(vp.load([jj, col0 + 1i32]));
                acc0 = acc0 * alpha + tv0 * p_val;
                acc1 = acc1 * alpha + tv1 * p_val;
                m_i = m_ij;
            }
            // Only when there IS a tail. At an aligned `n` this whole block is pure waste -
            // one extra iteration out of `full`, which is 6% of the work at n=512 and 0.8% at
            // n=4096, and that is exactly the residual it was costing.
            if n - full * BN > 0i32 {
                let tk0: Tile<f32, { [BN, 32] }> = convert_tile(kp.load([full, col0]));
                let tk1: Tile<f32, { [BN, 32] }> = convert_tile(kp.load([full, col0 + 1i32]));
                let dot_r: Tile<f32, { [BN] }> = reduce_sum(tq0w * tk0 + tq1w * tk1, 1i32);
                let dot0: Tile<f32, { [BN, 32] }> = dot_r.reshape(const_shape![BN, 1]).broadcast(wide);
                // `s*32` for this thread's row (see 2 above), and the pad value - both live
                // ONLY here. Hoisting them above the loop kept three extra registers alive
                // through it, and this kernel trades registers for occupancy.
                let iw1: Tile<i32, { [BN32] }> = iota(const_shape![BN32]);
                let iw: Tile<i32, { [BN, 32] }> = iw1.reshape(wide);
                let il1: Tile<i32, { [32] }> = iota(const_shape![32]);
                let il2: Tile<i32, { [1, 32] }> = il1.reshape(chunk);
                let il: Tile<i32, { [BN, 32] }> = il2.broadcast(wide);
                let s32: Tile<i32, { [BN, 32] }> = iw - il;
                let ninf: Tile<f32, { [BN, 32] }> = broadcast_scalar(-3.4028235e38f32, wide);
                let lim: Tile<i32, { [BN, 32] }> = broadcast_scalar(32i32 * (n - full * BN), wide);
                let inb: Tile<bool, { [BN, 32] }> = cmpi(s32, lim, predicate::LessThan);
                let dot: Tile<f32, { [BN, 32] }> = select(inb, dot0, ninf);
                let m_ij: Tile<f32, { [BN, 32] }> = maxf(m_i, dot, nan::Disabled, ftz::Disabled);
                let alpha: Tile<f32, { [BN, 32] }> = exp(m_i - m_ij);
                let p_val: Tile<f32, { [BN, 32] }> = exp(dot - m_ij);
                l_i = l_i * alpha + p_val;
                let tv0: Tile<f32, { [BN, 32] }> = convert_tile(vp.load([full, col0]));
                let tv1: Tile<f32, { [BN, 32] }> = convert_tile(vp.load([full, col0 + 1i32]));
                acc0 = acc0 * alpha + tv0 * p_val;
                acc1 = acc1 * alpha + tv1 * p_val;
                m_i = m_ij;
            }
            // The merge the old kernel already used, now over state that is a tile to begin
            // with: max over rows, rescaled sum, each accumulator weighted by exp(m_s - gm).
            // `m_i` and `l_i` are lane-INVARIANT: `dot` is broadcast across each row, so every
            // lane of row s carries row s's value. The uniformity analysis cannot see that -
            // it traces them back to a divergent load - and a column reduce over a source it
            // believes divergent pays a full threadgroup redistribution. Passing them through
            // a WITHIN-ROW reduce first is a value no-op (all lanes already agree, hence `max`
            // and not `sum`) that makes the invariance visible, and a within-SIMD reduce is the
            // analysis's convergence point. This is worth 8 of the kernel's 16 barriers - the
            // `get_simd_group_id` spelling got the same elision for free, because it reduced to
            // a scalar and re-broadcast it.
            let m_row: Tile<f32, { [BN] }> = reduce_max(m_i, 1i32);
            let mt: Tile<f32, { [BN, 32] }> = m_row.reshape(const_shape![BN, 1]).broadcast(wide);
            let l_row: Tile<f32, { [BN] }> = reduce_max(l_i, 1i32);
            let lt: Tile<f32, { [BN, 32] }> = l_row.reshape(const_shape![BN, 1]).broadcast(wide);
            let gm_w: Tile<f32, { [32] }> = reduce_max(mt, 0i32);
            let gm: Tile<f32, { [BN, 32] }> = gm_w.reshape(chunk).broadcast(wide);
            let f: Tile<f32, { [BN, 32] }> = exp(mt - gm);
            let gl_w: Tile<f32, { [32] }> = reduce_sum(lt * f, 0i32);
            let gl: Tile<f32, { [1, 32] }> = gl_w.reshape(chunk);
            // Hoist BOTH products before BOTH reduces. `batch_column_reduces` shares one set of
            // barriers across ADJACENT independent column reduces, and writing the multiply
            // inline (`reduce_sum(acc0 * f, 0)`) interleaves mul/reduce/mul/reduce so the two
            // reduces are no longer adjacent and each pays its own barriers.
            let a0: Tile<f32, { [BN, 32] }> = acc0 * f;
            let a1: Tile<f32, { [BN, 32] }> = acc1 * f;
            let n0: Tile<f32, { [32] }> = reduce_sum(a0, 0i32);
            let n1: Tile<f32, { [32] }> = reduce_sum(a1, 0i32);
            let o0: Tile<E, { [1, 32] }> = convert_tile(n0.reshape(chunk) / gl);
            let o1: Tile<E, { [1, 32] }> = convert_tile(n1.reshape(chunk) / gl);
            let mut op: PartitionMut<E, { [1, 32] }> = unsafe { out.partition_full_mut(chunk) };
            unsafe { op.store(o0, [head, 0i32]) };
            unsafe { op.store(o1, [head, 1i32]) };
        }

        #[wax::entry()]
        pub fn flash_attn_decode_d64<const N: i32>(
            q: &Tensor<f32, { [-1, 64] }>,
            k: &Tensor<f32, { [-1, 64] }>,
            v: &Tensor<f32, { [-1, 64] }>,
            out: &mut Tensor<f32, { [-1, 64] }>,
        ) {
            let chunk = const_shape![1, 32];
            let qb = get_tile_block_id().0;
            let q_part = q.partition(chunk);
            let k_part = k.partition(chunk);
            let v_part = v.partition(chunk);
            let tq0 = q_part.load([qb, 0i32]);
            let tq1 = q_part.load([qb, 1i32]);
            let mut m_i: Tile<f32, { [1] }> = constant(-3.4028235e38f32, const_shape![1]);
            let mut l_i: Tile<f32, { [1] }> = constant(0.0f32, const_shape![1]);
            let mut acc0: Tile<f32, { [1, 32] }> = constant(0.0f32, chunk);
            let mut acc1: Tile<f32, { [1, 32] }> = constant(0.0f32, chunk);
            for j in 0i32..N {
                let tk0 = k_part.load([j, 0i32]);
                let tk1 = k_part.load([j, 1i32]);
                let dot_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk0, 1i32);
                let dot_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk1, 1i32);
                let dot: Tile<f32, { [1] }> = dot_0 + dot_1;
                let m_ij: Tile<f32, { [1] }> = maxf(m_i, dot, nan::Disabled, ftz::Disabled);
                let alpha: Tile<f32, { [1] }> = exp(m_i - m_ij);
                let p_val: Tile<f32, { [1] }> = exp(dot - m_ij);
                l_i = l_i * alpha + p_val;
                let alpha_bc = alpha.reshape(const_shape![1, 1]).broadcast(chunk);
                let p_bc = p_val.reshape(const_shape![1, 1]).broadcast(chunk);
                let tv0 = v_part.load([j, 0i32]);
                let tv1 = v_part.load([j, 1i32]);
                acc0 = acc0 * alpha_bc + tv0 * p_bc;
                acc1 = acc1 * alpha_bc + tv1 * p_bc;
                m_i = m_ij;
            }
            let l_bc = l_i.reshape(const_shape![1, 1]).broadcast(chunk);
            acc0 = acc0 / l_bc;
            acc1 = acc1 / l_bc;
            let mut out_part = unsafe { out.partition_full_mut(chunk) };
            unsafe { out_part.store(acc0, [qb, 0i32]) };
            unsafe { out_part.store(acc1, [qb, 1i32]) };
        }

        #[wax::entry()]
        pub fn flash_attn_decode_d128<const N: i32>(
            q: &Tensor<f32, { [-1, 128] }>,
            k: &Tensor<f32, { [-1, 128] }>,
            v: &Tensor<f32, { [-1, 128] }>,
            out: &mut Tensor<f32, { [-1, 128] }>,
        ) {
            let chunk = const_shape![1, 32];
            let qb = get_tile_block_id().0;
            let q_part = q.partition(chunk);
            let k_part = k.partition(chunk);
            let v_part = v.partition(chunk);
            let tq0 = q_part.load([qb, 0i32]);
            let tq1 = q_part.load([qb, 1i32]);
            let tq2 = q_part.load([qb, 2i32]);
            let tq3 = q_part.load([qb, 3i32]);
            let mut m_i: Tile<f32, { [1] }> = constant(-3.4028235e38f32, const_shape![1]);
            let mut l_i: Tile<f32, { [1] }> = constant(0.0f32, const_shape![1]);
            let mut acc0: Tile<f32, { [1, 32] }> = constant(0.0f32, chunk);
            let mut acc1: Tile<f32, { [1, 32] }> = constant(0.0f32, chunk);
            let mut acc2: Tile<f32, { [1, 32] }> = constant(0.0f32, chunk);
            let mut acc3: Tile<f32, { [1, 32] }> = constant(0.0f32, chunk);
            for j in 0i32..N {
                let tk0 = k_part.load([j, 0i32]);
                let tk1 = k_part.load([j, 1i32]);
                let tk2 = k_part.load([j, 2i32]);
                let tk3 = k_part.load([j, 3i32]);
                // Hoist V loads before dot computation - V latency overlaps with
                // K-wait + exp, so V is ready by the time we accumulate.
                let tv0 = v_part.load([j, 0i32]);
                let tv1 = v_part.load([j, 1i32]);
                let tv2 = v_part.load([j, 2i32]);
                let tv3 = v_part.load([j, 3i32]);
                let dot_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk0, 1i32);
                let dot_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk1, 1i32);
                let dot_2: Tile<f32, { [1] }> = reduce_sum(tq2 * tk2, 1i32);
                let dot_3: Tile<f32, { [1] }> = reduce_sum(tq3 * tk3, 1i32);
                let dot: Tile<f32, { [1] }> = dot_0 + dot_1 + dot_2 + dot_3;
                let m_ij: Tile<f32, { [1] }> = maxf(m_i, dot, nan::Disabled, ftz::Disabled);
                let alpha: Tile<f32, { [1] }> = exp(m_i - m_ij);
                let p_val: Tile<f32, { [1] }> = exp(dot - m_ij);
                l_i = l_i * alpha + p_val;
                let alpha_bc = alpha.reshape(const_shape![1, 1]).broadcast(chunk);
                let p_bc = p_val.reshape(const_shape![1, 1]).broadcast(chunk);
                acc0 = acc0 * alpha_bc + tv0 * p_bc;
                acc1 = acc1 * alpha_bc + tv1 * p_bc;
                acc2 = acc2 * alpha_bc + tv2 * p_bc;
                acc3 = acc3 * alpha_bc + tv3 * p_bc;
                m_i = m_ij;
            }
            let l_bc = l_i.reshape(const_shape![1, 1]).broadcast(chunk);
            acc0 = acc0 / l_bc;
            acc1 = acc1 / l_bc;
            acc2 = acc2 / l_bc;
            acc3 = acc3 / l_bc;
            let mut out_part = unsafe { out.partition_full_mut(chunk) };
            unsafe { out_part.store(acc0, [qb, 0i32]) };
            unsafe { out_part.store(acc1, [qb, 1i32]) };
            unsafe { out_part.store(acc2, [qb, 2i32]) };
            unsafe { out_part.store(acc3, [qb, 3i32]) };
        }

        #[wax::entry()]
        pub fn flash_attn_partial_batched_d64<const CHUNK_SIZE: i32, const SPLITS: i32>(
            q: &Tensor<f32, { [-1, 64] }>,
            k: &Tensor<f32, { [-1, 64] }>,
            v: &Tensor<f32, { [-1, 64] }>,
            partial_m: &mut Tensor<f32, { [-1, 32] }>,
            partial_l: &mut Tensor<f32, { [-1, 32] }>,
            partial_acc: &mut Tensor<f32, { [-1, 64] }>,
        ) {
            let chunk = const_shape![1, 32];
            let block_id = get_tile_block_id().0;
            let q_id = block_id / SPLITS;
            let split_id = block_id % SPLITS;
            let kv_start = split_id * CHUNK_SIZE;
            let q_part = q.partition(chunk);
            let k_part = k.partition(chunk);
            let v_part = v.partition(chunk);
            let tq0 = q_part.load([q_id, 0i32]);
            let tq1 = q_part.load([q_id, 1i32]);
            let mut m_i: Tile<f32, { [1] }> = constant(-3.4028235e38f32, const_shape![1]);
            let mut l_i: Tile<f32, { [1] }> = constant(0.0f32, const_shape![1]);
            let mut acc0: Tile<f32, { [1, 32] }> = constant(0.0f32, chunk);
            let mut acc1: Tile<f32, { [1, 32] }> = constant(0.0f32, chunk);
            for j in 0i32..CHUNK_SIZE {
                let kv = kv_start + j;
                let tk0 = k_part.load([kv, 0i32]);
                let tk1 = k_part.load([kv, 1i32]);
                let dot_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk0, 1i32);
                let dot_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk1, 1i32);
                let dot: Tile<f32, { [1] }> = dot_0 + dot_1;
                let m_ij: Tile<f32, { [1] }> = maxf(m_i, dot, nan::Disabled, ftz::Disabled);
                let alpha: Tile<f32, { [1] }> = exp(m_i - m_ij);
                let p_val: Tile<f32, { [1] }> = exp(dot - m_ij);
                l_i = l_i * alpha + p_val;
                let alpha_bc = alpha.reshape(const_shape![1, 1]).broadcast(chunk);
                let p_bc = p_val.reshape(const_shape![1, 1]).broadcast(chunk);
                let tv0 = v_part.load([kv, 0i32]);
                let tv1 = v_part.load([kv, 1i32]);
                acc0 = acc0 * alpha_bc + tv0 * p_bc;
                acc1 = acc1 * alpha_bc + tv1 * p_bc;
                m_i = m_ij;
            }
            let out_id = q_id * SPLITS + split_id;
            let m_bc = broadcast_scalar(tile_to_scalar(m_i.reshape(const_shape![])), chunk);
            let l_bc = broadcast_scalar(tile_to_scalar(l_i.reshape(const_shape![])), chunk);
            let mut pm = unsafe { partial_m.partition_full_mut(chunk) };
            unsafe { pm.store(m_bc, [out_id, 0i32]) };
            let mut pl = unsafe { partial_l.partition_full_mut(chunk) };
            unsafe { pl.store(l_bc, [out_id, 0i32]) };
            let mut pa = unsafe { partial_acc.partition_full_mut(chunk) };
            unsafe { pa.store(acc0, [out_id, 0i32]) };
            unsafe { pa.store(acc1, [out_id, 1i32]) };
        }

        #[wax::entry()]
        pub fn flash_attn_combine_batched_d64<const SPLITS: i32>(
            partial_m: &Tensor<f32, { [-1, 32] }>,
            partial_l: &Tensor<f32, { [-1, 32] }>,
            partial_acc: &Tensor<f32, { [-1, 64] }>,
            out: &mut Tensor<f32, { [-1, 64] }>,
        ) {
            let chunk = const_shape![1, 32];
            let q_id = get_tile_block_id().0;
            let base = q_id * SPLITS;
            let m_part = partial_m.partition(chunk);
            let l_part = partial_l.partition(chunk);
            let acc_part = partial_acc.partition(chunk);
            let mut m_g = m_part.load([base, 0i32]);
            let mut l_g = l_part.load([base, 0i32]);
            let mut acc0_g = acc_part.load([base, 0i32]);
            let mut acc1_g = acc_part.load([base, 1i32]);
            for s in 1i32..SPLITS {
                let idx = base + s;
                let m_s = m_part.load([idx, 0i32]);
                let l_s = l_part.load([idx, 0i32]);
                let acc0_s = acc_part.load([idx, 0i32]);
                let acc1_s = acc_part.load([idx, 1i32]);
                let m_new: Tile<f32, { [1, 32] }> = maxf(m_g, m_s, nan::Disabled, ftz::Disabled);
                let alpha: Tile<f32, { [1, 32] }> = exp(m_g - m_new);
                let beta: Tile<f32, { [1, 32] }> = exp(m_s - m_new);
                l_g = l_g * alpha + l_s * beta;
                acc0_g = acc0_g * alpha + acc0_s * beta;
                acc1_g = acc1_g * alpha + acc1_s * beta;
                m_g = m_new;
            }
            acc0_g = acc0_g / l_g;
            acc1_g = acc1_g / l_g;
            let mut out_part = unsafe { out.partition_full_mut(chunk) };
            unsafe { out_part.store(acc0_g, [q_id, 0i32]) };
            unsafe { out_part.store(acc1_g, [q_id, 1i32]) };
        }

        #[wax::entry()]
        pub fn flash_attn_partial_batched_d128<const CHUNK_SIZE: i32, const SPLITS: i32>(
            q: &Tensor<f32, { [-1, 128] }>,
            k: &Tensor<f32, { [-1, 128] }>,
            v: &Tensor<f32, { [-1, 128] }>,
            partial_m: &mut Tensor<f32, { [-1, 32] }>,
            partial_l: &mut Tensor<f32, { [-1, 32] }>,
            partial_acc: &mut Tensor<f32, { [-1, 128] }>,
        ) {
            let chunk = const_shape![1, 32];
            let block_id = get_tile_block_id().0;
            let q_id = block_id / SPLITS;
            let split_id = block_id % SPLITS;
            let kv_start = split_id * CHUNK_SIZE;
            let q_part = q.partition(chunk);
            let k_part = k.partition(chunk);
            let v_part = v.partition(chunk);
            let tq0 = q_part.load([q_id, 0i32]);
            let tq1 = q_part.load([q_id, 1i32]);
            let tq2 = q_part.load([q_id, 2i32]);
            let tq3 = q_part.load([q_id, 3i32]);
            let mut m_i: Tile<f32, { [1] }> = constant(-3.4028235e38f32, const_shape![1]);
            let mut l_i: Tile<f32, { [1] }> = constant(0.0f32, const_shape![1]);
            let mut acc0: Tile<f32, { [1, 32] }> = constant(0.0f32, chunk);
            let mut acc1: Tile<f32, { [1, 32] }> = constant(0.0f32, chunk);
            let mut acc2: Tile<f32, { [1, 32] }> = constant(0.0f32, chunk);
            let mut acc3: Tile<f32, { [1, 32] }> = constant(0.0f32, chunk);
            for j in 0i32..CHUNK_SIZE {
                let kv = kv_start + j;
                let tk0 = k_part.load([kv, 0i32]);
                let tk1 = k_part.load([kv, 1i32]);
                let tk2 = k_part.load([kv, 2i32]);
                let tk3 = k_part.load([kv, 3i32]);
                // Hoist V loads: all 8 tiles (4K+4V) issue together, V latency
                // overlaps with K-wait + exp so V is ready by accumulate time.
                let tv0 = v_part.load([kv, 0i32]);
                let tv1 = v_part.load([kv, 1i32]);
                let tv2 = v_part.load([kv, 2i32]);
                let tv3 = v_part.load([kv, 3i32]);
                let dot_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk0, 1i32);
                let dot_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk1, 1i32);
                let dot_2: Tile<f32, { [1] }> = reduce_sum(tq2 * tk2, 1i32);
                let dot_3: Tile<f32, { [1] }> = reduce_sum(tq3 * tk3, 1i32);
                let dot: Tile<f32, { [1] }> = dot_0 + dot_1 + dot_2 + dot_3;
                let m_ij: Tile<f32, { [1] }> = maxf(m_i, dot, nan::Disabled, ftz::Disabled);
                let alpha: Tile<f32, { [1] }> = exp(m_i - m_ij);
                let p_val: Tile<f32, { [1] }> = exp(dot - m_ij);
                l_i = l_i * alpha + p_val;
                let alpha_bc = alpha.reshape(const_shape![1, 1]).broadcast(chunk);
                let p_bc = p_val.reshape(const_shape![1, 1]).broadcast(chunk);
                acc0 = acc0 * alpha_bc + tv0 * p_bc;
                acc1 = acc1 * alpha_bc + tv1 * p_bc;
                acc2 = acc2 * alpha_bc + tv2 * p_bc;
                acc3 = acc3 * alpha_bc + tv3 * p_bc;
                m_i = m_ij;
            }
            let out_id = q_id * SPLITS + split_id;
            let m_bc = broadcast_scalar(tile_to_scalar(m_i.reshape(const_shape![])), chunk);
            let l_bc = broadcast_scalar(tile_to_scalar(l_i.reshape(const_shape![])), chunk);
            let mut pm = unsafe { partial_m.partition_full_mut(chunk) };
            unsafe { pm.store(m_bc, [out_id, 0i32]) };
            let mut pl = unsafe { partial_l.partition_full_mut(chunk) };
            unsafe { pl.store(l_bc, [out_id, 0i32]) };
            let mut pa = unsafe { partial_acc.partition_full_mut(chunk) };
            unsafe { pa.store(acc0, [out_id, 0i32]) };
            unsafe { pa.store(acc1, [out_id, 1i32]) };
            unsafe { pa.store(acc2, [out_id, 2i32]) };
            unsafe { pa.store(acc3, [out_id, 3i32]) };
        }

        #[wax::entry()]
        pub fn flash_attn_decode_d64_b8<const N: i32>(
            q: &Tensor<f32, { [-1, 64] }>,
            k: &Tensor<f32, { [-1, 64] }>,
            v: &Tensor<f32, { [-1, 64] }>,
            out: &mut Tensor<f32, { [-1, 64] }>,
        ) {
            let chunk = const_shape![1, 32];
            let qb = get_tile_block_id().0;
            let q_part = q.partition(chunk);
            let k_part = k.partition(chunk);
            let v_part = v.partition(chunk);
            let tq0 = q_part.load([qb, 0i32]);
            let tq1 = q_part.load([qb, 1i32]);
            let mut m_i: Tile<f32, { [1] }> = constant(-3.4028235e38f32, const_shape![1]);
            let mut l_i: Tile<f32, { [1] }> = constant(0.0f32, const_shape![1]);
            let mut acc0: Tile<f32, { [1, 32] }> = constant(0.0f32, chunk);
            let mut acc1: Tile<f32, { [1, 32] }> = constant(0.0f32, chunk);
            for jb in 0i32..(N / 8) {
                let kv = jb * 8;
                let tk0_0 = k_part.load([kv, 0i32]);
                let tk0_1 = k_part.load([kv, 1i32]);
                let tk1_0 = k_part.load([kv + 1, 0i32]);
                let tk1_1 = k_part.load([kv + 1, 1i32]);
                let tk2_0 = k_part.load([kv + 2, 0i32]);
                let tk2_1 = k_part.load([kv + 2, 1i32]);
                let tk3_0 = k_part.load([kv + 3, 0i32]);
                let tk3_1 = k_part.load([kv + 3, 1i32]);
                let tk4_0 = k_part.load([kv + 4, 0i32]);
                let tk4_1 = k_part.load([kv + 4, 1i32]);
                let tk5_0 = k_part.load([kv + 5, 0i32]);
                let tk5_1 = k_part.load([kv + 5, 1i32]);
                let tk6_0 = k_part.load([kv + 6, 0i32]);
                let tk6_1 = k_part.load([kv + 6, 1i32]);
                let tk7_0 = k_part.load([kv + 7, 0i32]);
                let tk7_1 = k_part.load([kv + 7, 1i32]);
                let tv0_0 = v_part.load([kv, 0i32]);
                let tv0_1 = v_part.load([kv, 1i32]);
                let tv1_0 = v_part.load([kv + 1, 0i32]);
                let tv1_1 = v_part.load([kv + 1, 1i32]);
                let tv2_0 = v_part.load([kv + 2, 0i32]);
                let tv2_1 = v_part.load([kv + 2, 1i32]);
                let tv3_0 = v_part.load([kv + 3, 0i32]);
                let tv3_1 = v_part.load([kv + 3, 1i32]);
                let tv4_0 = v_part.load([kv + 4, 0i32]);
                let tv4_1 = v_part.load([kv + 4, 1i32]);
                let tv5_0 = v_part.load([kv + 5, 0i32]);
                let tv5_1 = v_part.load([kv + 5, 1i32]);
                let tv6_0 = v_part.load([kv + 6, 0i32]);
                let tv6_1 = v_part.load([kv + 6, 1i32]);
                let tv7_0 = v_part.load([kv + 7, 0i32]);
                let tv7_1 = v_part.load([kv + 7, 1i32]);
                let dot0_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk0_0, 1i32);
                let dot0_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk0_1, 1i32);
                let dot0: Tile<f32, { [1] }> = dot0_0 + dot0_1;
                let dot1_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk1_0, 1i32);
                let dot1_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk1_1, 1i32);
                let dot1: Tile<f32, { [1] }> = dot1_0 + dot1_1;
                let dot2_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk2_0, 1i32);
                let dot2_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk2_1, 1i32);
                let dot2: Tile<f32, { [1] }> = dot2_0 + dot2_1;
                let dot3_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk3_0, 1i32);
                let dot3_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk3_1, 1i32);
                let dot3: Tile<f32, { [1] }> = dot3_0 + dot3_1;
                let dot4_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk4_0, 1i32);
                let dot4_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk4_1, 1i32);
                let dot4: Tile<f32, { [1] }> = dot4_0 + dot4_1;
                let dot5_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk5_0, 1i32);
                let dot5_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk5_1, 1i32);
                let dot5: Tile<f32, { [1] }> = dot5_0 + dot5_1;
                let dot6_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk6_0, 1i32);
                let dot6_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk6_1, 1i32);
                let dot6: Tile<f32, { [1] }> = dot6_0 + dot6_1;
                let dot7_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk7_0, 1i32);
                let dot7_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk7_1, 1i32);
                let dot7: Tile<f32, { [1] }> = dot7_0 + dot7_1;
                let m01 = maxf(dot0, dot1, nan::Disabled, ftz::Disabled);
                let m23 = maxf(dot2, dot3, nan::Disabled, ftz::Disabled);
                let m45 = maxf(dot4, dot5, nan::Disabled, ftz::Disabled);
                let m67 = maxf(dot6, dot7, nan::Disabled, ftz::Disabled);
                let m0123 = maxf(m01, m23, nan::Disabled, ftz::Disabled);
                let m4567 = maxf(m45, m67, nan::Disabled, ftz::Disabled);
                let m_blk = maxf(m0123, m4567, nan::Disabled, ftz::Disabled);
                let m_ij = maxf(m_i, m_blk, nan::Disabled, ftz::Disabled);
                let alpha: Tile<f32, { [1] }> = exp(m_i - m_ij);
                let p0: Tile<f32, { [1] }> = exp(dot0 - m_ij);
                let p1: Tile<f32, { [1] }> = exp(dot1 - m_ij);
                let p2: Tile<f32, { [1] }> = exp(dot2 - m_ij);
                let p3: Tile<f32, { [1] }> = exp(dot3 - m_ij);
                let p4: Tile<f32, { [1] }> = exp(dot4 - m_ij);
                let p5: Tile<f32, { [1] }> = exp(dot5 - m_ij);
                let p6: Tile<f32, { [1] }> = exp(dot6 - m_ij);
                let p7: Tile<f32, { [1] }> = exp(dot7 - m_ij);
                l_i = l_i * alpha + p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7;
                let alpha_bc = alpha.reshape(const_shape![1, 1]).broadcast(chunk);
                let p0_bc = p0.reshape(const_shape![1, 1]).broadcast(chunk);
                let p1_bc = p1.reshape(const_shape![1, 1]).broadcast(chunk);
                let p2_bc = p2.reshape(const_shape![1, 1]).broadcast(chunk);
                let p3_bc = p3.reshape(const_shape![1, 1]).broadcast(chunk);
                let p4_bc = p4.reshape(const_shape![1, 1]).broadcast(chunk);
                let p5_bc = p5.reshape(const_shape![1, 1]).broadcast(chunk);
                let p6_bc = p6.reshape(const_shape![1, 1]).broadcast(chunk);
                let p7_bc = p7.reshape(const_shape![1, 1]).broadcast(chunk);
                acc0 = acc0 * alpha_bc
                    + tv0_0 * p0_bc
                    + tv1_0 * p1_bc
                    + tv2_0 * p2_bc
                    + tv3_0 * p3_bc
                    + tv4_0 * p4_bc
                    + tv5_0 * p5_bc
                    + tv6_0 * p6_bc
                    + tv7_0 * p7_bc;
                acc1 = acc1 * alpha_bc
                    + tv0_1 * p0_bc
                    + tv1_1 * p1_bc
                    + tv2_1 * p2_bc
                    + tv3_1 * p3_bc
                    + tv4_1 * p4_bc
                    + tv5_1 * p5_bc
                    + tv6_1 * p6_bc
                    + tv7_1 * p7_bc;
                m_i = m_ij;
            }
            let l_bc = l_i.reshape(const_shape![1, 1]).broadcast(chunk);
            acc0 = acc0 / l_bc;
            acc1 = acc1 / l_bc;
            let mut out_part = unsafe { out.partition_full_mut(chunk) };
            unsafe { out_part.store(acc0, [qb, 0i32]) };
            unsafe { out_part.store(acc1, [qb, 1i32]) };
        }

        #[wax::entry()]
        pub fn flash_attn_decode_d128_b8<const N: i32>(
            q: &Tensor<f32, { [-1, 128] }>,
            k: &Tensor<f32, { [-1, 128] }>,
            v: &Tensor<f32, { [-1, 128] }>,
            out: &mut Tensor<f32, { [-1, 128] }>,
        ) {
            let chunk = const_shape![1, 32];
            let qb = get_tile_block_id().0;
            let q_part = q.partition(chunk);
            let k_part = k.partition(chunk);
            let v_part = v.partition(chunk);
            let tq0 = q_part.load([qb, 0i32]);
            let tq1 = q_part.load([qb, 1i32]);
            let tq2 = q_part.load([qb, 2i32]);
            let tq3 = q_part.load([qb, 3i32]);
            let mut m_i: Tile<f32, { [1] }> = constant(-3.4028235e38f32, const_shape![1]);
            let mut l_i: Tile<f32, { [1] }> = constant(0.0f32, const_shape![1]);
            let mut acc0: Tile<f32, { [1, 32] }> = constant(0.0f32, chunk);
            let mut acc1: Tile<f32, { [1, 32] }> = constant(0.0f32, chunk);
            let mut acc2: Tile<f32, { [1, 32] }> = constant(0.0f32, chunk);
            let mut acc3: Tile<f32, { [1, 32] }> = constant(0.0f32, chunk);
            for jb in 0i32..(N / 8) {
                let kv = jb * 8;
                let tk0_0 = k_part.load([kv, 0i32]);
                let tk0_1 = k_part.load([kv, 1i32]);
                let tk0_2 = k_part.load([kv, 2i32]);
                let tk0_3 = k_part.load([kv, 3i32]);
                let tk1_0 = k_part.load([kv + 1, 0i32]);
                let tk1_1 = k_part.load([kv + 1, 1i32]);
                let tk1_2 = k_part.load([kv + 1, 2i32]);
                let tk1_3 = k_part.load([kv + 1, 3i32]);
                let tk2_0 = k_part.load([kv + 2, 0i32]);
                let tk2_1 = k_part.load([kv + 2, 1i32]);
                let tk2_2 = k_part.load([kv + 2, 2i32]);
                let tk2_3 = k_part.load([kv + 2, 3i32]);
                let tk3_0 = k_part.load([kv + 3, 0i32]);
                let tk3_1 = k_part.load([kv + 3, 1i32]);
                let tk3_2 = k_part.load([kv + 3, 2i32]);
                let tk3_3 = k_part.load([kv + 3, 3i32]);
                let tk4_0 = k_part.load([kv + 4, 0i32]);
                let tk4_1 = k_part.load([kv + 4, 1i32]);
                let tk4_2 = k_part.load([kv + 4, 2i32]);
                let tk4_3 = k_part.load([kv + 4, 3i32]);
                let tk5_0 = k_part.load([kv + 5, 0i32]);
                let tk5_1 = k_part.load([kv + 5, 1i32]);
                let tk5_2 = k_part.load([kv + 5, 2i32]);
                let tk5_3 = k_part.load([kv + 5, 3i32]);
                let tk6_0 = k_part.load([kv + 6, 0i32]);
                let tk6_1 = k_part.load([kv + 6, 1i32]);
                let tk6_2 = k_part.load([kv + 6, 2i32]);
                let tk6_3 = k_part.load([kv + 6, 3i32]);
                let tk7_0 = k_part.load([kv + 7, 0i32]);
                let tk7_1 = k_part.load([kv + 7, 1i32]);
                let tk7_2 = k_part.load([kv + 7, 2i32]);
                let tk7_3 = k_part.load([kv + 7, 3i32]);
                let tv0_0 = v_part.load([kv, 0i32]);
                let tv0_1 = v_part.load([kv, 1i32]);
                let tv0_2 = v_part.load([kv, 2i32]);
                let tv0_3 = v_part.load([kv, 3i32]);
                let tv1_0 = v_part.load([kv + 1, 0i32]);
                let tv1_1 = v_part.load([kv + 1, 1i32]);
                let tv1_2 = v_part.load([kv + 1, 2i32]);
                let tv1_3 = v_part.load([kv + 1, 3i32]);
                let tv2_0 = v_part.load([kv + 2, 0i32]);
                let tv2_1 = v_part.load([kv + 2, 1i32]);
                let tv2_2 = v_part.load([kv + 2, 2i32]);
                let tv2_3 = v_part.load([kv + 2, 3i32]);
                let tv3_0 = v_part.load([kv + 3, 0i32]);
                let tv3_1 = v_part.load([kv + 3, 1i32]);
                let tv3_2 = v_part.load([kv + 3, 2i32]);
                let tv3_3 = v_part.load([kv + 3, 3i32]);
                let tv4_0 = v_part.load([kv + 4, 0i32]);
                let tv4_1 = v_part.load([kv + 4, 1i32]);
                let tv4_2 = v_part.load([kv + 4, 2i32]);
                let tv4_3 = v_part.load([kv + 4, 3i32]);
                let tv5_0 = v_part.load([kv + 5, 0i32]);
                let tv5_1 = v_part.load([kv + 5, 1i32]);
                let tv5_2 = v_part.load([kv + 5, 2i32]);
                let tv5_3 = v_part.load([kv + 5, 3i32]);
                let tv6_0 = v_part.load([kv + 6, 0i32]);
                let tv6_1 = v_part.load([kv + 6, 1i32]);
                let tv6_2 = v_part.load([kv + 6, 2i32]);
                let tv6_3 = v_part.load([kv + 6, 3i32]);
                let tv7_0 = v_part.load([kv + 7, 0i32]);
                let tv7_1 = v_part.load([kv + 7, 1i32]);
                let tv7_2 = v_part.load([kv + 7, 2i32]);
                let tv7_3 = v_part.load([kv + 7, 3i32]);
                let dot0_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk0_0, 1i32);
                let dot0_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk0_1, 1i32);
                let dot0_2: Tile<f32, { [1] }> = reduce_sum(tq2 * tk0_2, 1i32);
                let dot0_3: Tile<f32, { [1] }> = reduce_sum(tq3 * tk0_3, 1i32);
                let dot0: Tile<f32, { [1] }> = dot0_0 + dot0_1 + dot0_2 + dot0_3;
                let dot1_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk1_0, 1i32);
                let dot1_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk1_1, 1i32);
                let dot1_2: Tile<f32, { [1] }> = reduce_sum(tq2 * tk1_2, 1i32);
                let dot1_3: Tile<f32, { [1] }> = reduce_sum(tq3 * tk1_3, 1i32);
                let dot1: Tile<f32, { [1] }> = dot1_0 + dot1_1 + dot1_2 + dot1_3;
                let dot2_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk2_0, 1i32);
                let dot2_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk2_1, 1i32);
                let dot2_2: Tile<f32, { [1] }> = reduce_sum(tq2 * tk2_2, 1i32);
                let dot2_3: Tile<f32, { [1] }> = reduce_sum(tq3 * tk2_3, 1i32);
                let dot2: Tile<f32, { [1] }> = dot2_0 + dot2_1 + dot2_2 + dot2_3;
                let dot3_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk3_0, 1i32);
                let dot3_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk3_1, 1i32);
                let dot3_2: Tile<f32, { [1] }> = reduce_sum(tq2 * tk3_2, 1i32);
                let dot3_3: Tile<f32, { [1] }> = reduce_sum(tq3 * tk3_3, 1i32);
                let dot3: Tile<f32, { [1] }> = dot3_0 + dot3_1 + dot3_2 + dot3_3;
                let dot4_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk4_0, 1i32);
                let dot4_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk4_1, 1i32);
                let dot4_2: Tile<f32, { [1] }> = reduce_sum(tq2 * tk4_2, 1i32);
                let dot4_3: Tile<f32, { [1] }> = reduce_sum(tq3 * tk4_3, 1i32);
                let dot4: Tile<f32, { [1] }> = dot4_0 + dot4_1 + dot4_2 + dot4_3;
                let dot5_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk5_0, 1i32);
                let dot5_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk5_1, 1i32);
                let dot5_2: Tile<f32, { [1] }> = reduce_sum(tq2 * tk5_2, 1i32);
                let dot5_3: Tile<f32, { [1] }> = reduce_sum(tq3 * tk5_3, 1i32);
                let dot5: Tile<f32, { [1] }> = dot5_0 + dot5_1 + dot5_2 + dot5_3;
                let dot6_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk6_0, 1i32);
                let dot6_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk6_1, 1i32);
                let dot6_2: Tile<f32, { [1] }> = reduce_sum(tq2 * tk6_2, 1i32);
                let dot6_3: Tile<f32, { [1] }> = reduce_sum(tq3 * tk6_3, 1i32);
                let dot6: Tile<f32, { [1] }> = dot6_0 + dot6_1 + dot6_2 + dot6_3;
                let dot7_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk7_0, 1i32);
                let dot7_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk7_1, 1i32);
                let dot7_2: Tile<f32, { [1] }> = reduce_sum(tq2 * tk7_2, 1i32);
                let dot7_3: Tile<f32, { [1] }> = reduce_sum(tq3 * tk7_3, 1i32);
                let dot7: Tile<f32, { [1] }> = dot7_0 + dot7_1 + dot7_2 + dot7_3;
                let m01 = maxf(dot0, dot1, nan::Disabled, ftz::Disabled);
                let m23 = maxf(dot2, dot3, nan::Disabled, ftz::Disabled);
                let m45 = maxf(dot4, dot5, nan::Disabled, ftz::Disabled);
                let m67 = maxf(dot6, dot7, nan::Disabled, ftz::Disabled);
                let m0123 = maxf(m01, m23, nan::Disabled, ftz::Disabled);
                let m4567 = maxf(m45, m67, nan::Disabled, ftz::Disabled);
                let m_blk = maxf(m0123, m4567, nan::Disabled, ftz::Disabled);
                let m_ij = maxf(m_i, m_blk, nan::Disabled, ftz::Disabled);
                let alpha: Tile<f32, { [1] }> = exp(m_i - m_ij);
                let p0: Tile<f32, { [1] }> = exp(dot0 - m_ij);
                let p1: Tile<f32, { [1] }> = exp(dot1 - m_ij);
                let p2: Tile<f32, { [1] }> = exp(dot2 - m_ij);
                let p3: Tile<f32, { [1] }> = exp(dot3 - m_ij);
                let p4: Tile<f32, { [1] }> = exp(dot4 - m_ij);
                let p5: Tile<f32, { [1] }> = exp(dot5 - m_ij);
                let p6: Tile<f32, { [1] }> = exp(dot6 - m_ij);
                let p7: Tile<f32, { [1] }> = exp(dot7 - m_ij);
                l_i = l_i * alpha + p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7;
                let alpha_bc = alpha.reshape(const_shape![1, 1]).broadcast(chunk);
                let p0_bc = p0.reshape(const_shape![1, 1]).broadcast(chunk);
                let p1_bc = p1.reshape(const_shape![1, 1]).broadcast(chunk);
                let p2_bc = p2.reshape(const_shape![1, 1]).broadcast(chunk);
                let p3_bc = p3.reshape(const_shape![1, 1]).broadcast(chunk);
                let p4_bc = p4.reshape(const_shape![1, 1]).broadcast(chunk);
                let p5_bc = p5.reshape(const_shape![1, 1]).broadcast(chunk);
                let p6_bc = p6.reshape(const_shape![1, 1]).broadcast(chunk);
                let p7_bc = p7.reshape(const_shape![1, 1]).broadcast(chunk);
                acc0 = acc0 * alpha_bc
                    + tv0_0 * p0_bc
                    + tv1_0 * p1_bc
                    + tv2_0 * p2_bc
                    + tv3_0 * p3_bc
                    + tv4_0 * p4_bc
                    + tv5_0 * p5_bc
                    + tv6_0 * p6_bc
                    + tv7_0 * p7_bc;
                acc1 = acc1 * alpha_bc
                    + tv0_1 * p0_bc
                    + tv1_1 * p1_bc
                    + tv2_1 * p2_bc
                    + tv3_1 * p3_bc
                    + tv4_1 * p4_bc
                    + tv5_1 * p5_bc
                    + tv6_1 * p6_bc
                    + tv7_1 * p7_bc;
                acc2 = acc2 * alpha_bc
                    + tv0_2 * p0_bc
                    + tv1_2 * p1_bc
                    + tv2_2 * p2_bc
                    + tv3_2 * p3_bc
                    + tv4_2 * p4_bc
                    + tv5_2 * p5_bc
                    + tv6_2 * p6_bc
                    + tv7_2 * p7_bc;
                acc3 = acc3 * alpha_bc
                    + tv0_3 * p0_bc
                    + tv1_3 * p1_bc
                    + tv2_3 * p2_bc
                    + tv3_3 * p3_bc
                    + tv4_3 * p4_bc
                    + tv5_3 * p5_bc
                    + tv6_3 * p6_bc
                    + tv7_3 * p7_bc;
                m_i = m_ij;
            }
            let l_bc = l_i.reshape(const_shape![1, 1]).broadcast(chunk);
            acc0 = acc0 / l_bc;
            acc1 = acc1 / l_bc;
            acc2 = acc2 / l_bc;
            acc3 = acc3 / l_bc;
            let mut out_part = unsafe { out.partition_full_mut(chunk) };
            unsafe { out_part.store(acc0, [qb, 0i32]) };
            unsafe { out_part.store(acc1, [qb, 1i32]) };
            unsafe { out_part.store(acc2, [qb, 2i32]) };
            unsafe { out_part.store(acc3, [qb, 3i32]) };
        }

        #[wax::entry()]
        pub fn flash_attn_partial_batched_d64_b8<const CHUNK_SIZE: i32, const SPLITS: i32>(
            q: &Tensor<f32, { [-1, 64] }>,
            k: &Tensor<f32, { [-1, 64] }>,
            v: &Tensor<f32, { [-1, 64] }>,
            partial_m: &mut Tensor<f32, { [-1, 32] }>,
            partial_l: &mut Tensor<f32, { [-1, 32] }>,
            partial_acc: &mut Tensor<f32, { [-1, 64] }>,
        ) {
            let chunk = const_shape![1, 32];
            let block_id = get_tile_block_id().0;
            let q_id = block_id / SPLITS;
            let split_id = block_id % SPLITS;
            let kv_start = split_id * CHUNK_SIZE;
            let q_part = q.partition(chunk);
            let k_part = k.partition(chunk);
            let v_part = v.partition(chunk);
            let tq0 = q_part.load([q_id, 0i32]);
            let tq1 = q_part.load([q_id, 1i32]);
            let mut m_i: Tile<f32, { [1] }> = constant(-3.4028235e38f32, const_shape![1]);
            let mut l_i: Tile<f32, { [1] }> = constant(0.0f32, const_shape![1]);
            let mut acc0: Tile<f32, { [1, 32] }> = constant(0.0f32, chunk);
            let mut acc1: Tile<f32, { [1, 32] }> = constant(0.0f32, chunk);
            for jb in 0i32..(CHUNK_SIZE / 8) {
                let kv = kv_start + jb * 8;
                let tk0_0 = k_part.load([kv, 0i32]);
                let tk0_1 = k_part.load([kv, 1i32]);
                let tk1_0 = k_part.load([kv + 1, 0i32]);
                let tk1_1 = k_part.load([kv + 1, 1i32]);
                let tk2_0 = k_part.load([kv + 2, 0i32]);
                let tk2_1 = k_part.load([kv + 2, 1i32]);
                let tk3_0 = k_part.load([kv + 3, 0i32]);
                let tk3_1 = k_part.load([kv + 3, 1i32]);
                let tk4_0 = k_part.load([kv + 4, 0i32]);
                let tk4_1 = k_part.load([kv + 4, 1i32]);
                let tk5_0 = k_part.load([kv + 5, 0i32]);
                let tk5_1 = k_part.load([kv + 5, 1i32]);
                let tk6_0 = k_part.load([kv + 6, 0i32]);
                let tk6_1 = k_part.load([kv + 6, 1i32]);
                let tk7_0 = k_part.load([kv + 7, 0i32]);
                let tk7_1 = k_part.load([kv + 7, 1i32]);
                let tv0_0 = v_part.load([kv, 0i32]);
                let tv0_1 = v_part.load([kv, 1i32]);
                let tv1_0 = v_part.load([kv + 1, 0i32]);
                let tv1_1 = v_part.load([kv + 1, 1i32]);
                let tv2_0 = v_part.load([kv + 2, 0i32]);
                let tv2_1 = v_part.load([kv + 2, 1i32]);
                let tv3_0 = v_part.load([kv + 3, 0i32]);
                let tv3_1 = v_part.load([kv + 3, 1i32]);
                let tv4_0 = v_part.load([kv + 4, 0i32]);
                let tv4_1 = v_part.load([kv + 4, 1i32]);
                let tv5_0 = v_part.load([kv + 5, 0i32]);
                let tv5_1 = v_part.load([kv + 5, 1i32]);
                let tv6_0 = v_part.load([kv + 6, 0i32]);
                let tv6_1 = v_part.load([kv + 6, 1i32]);
                let tv7_0 = v_part.load([kv + 7, 0i32]);
                let tv7_1 = v_part.load([kv + 7, 1i32]);
                let dot0_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk0_0, 1i32);
                let dot0_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk0_1, 1i32);
                let dot0: Tile<f32, { [1] }> = dot0_0 + dot0_1;
                let dot1_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk1_0, 1i32);
                let dot1_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk1_1, 1i32);
                let dot1: Tile<f32, { [1] }> = dot1_0 + dot1_1;
                let dot2_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk2_0, 1i32);
                let dot2_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk2_1, 1i32);
                let dot2: Tile<f32, { [1] }> = dot2_0 + dot2_1;
                let dot3_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk3_0, 1i32);
                let dot3_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk3_1, 1i32);
                let dot3: Tile<f32, { [1] }> = dot3_0 + dot3_1;
                let dot4_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk4_0, 1i32);
                let dot4_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk4_1, 1i32);
                let dot4: Tile<f32, { [1] }> = dot4_0 + dot4_1;
                let dot5_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk5_0, 1i32);
                let dot5_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk5_1, 1i32);
                let dot5: Tile<f32, { [1] }> = dot5_0 + dot5_1;
                let dot6_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk6_0, 1i32);
                let dot6_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk6_1, 1i32);
                let dot6: Tile<f32, { [1] }> = dot6_0 + dot6_1;
                let dot7_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk7_0, 1i32);
                let dot7_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk7_1, 1i32);
                let dot7: Tile<f32, { [1] }> = dot7_0 + dot7_1;
                let m01 = maxf(dot0, dot1, nan::Disabled, ftz::Disabled);
                let m23 = maxf(dot2, dot3, nan::Disabled, ftz::Disabled);
                let m45 = maxf(dot4, dot5, nan::Disabled, ftz::Disabled);
                let m67 = maxf(dot6, dot7, nan::Disabled, ftz::Disabled);
                let m0123 = maxf(m01, m23, nan::Disabled, ftz::Disabled);
                let m4567 = maxf(m45, m67, nan::Disabled, ftz::Disabled);
                let m_blk = maxf(m0123, m4567, nan::Disabled, ftz::Disabled);
                let m_ij = maxf(m_i, m_blk, nan::Disabled, ftz::Disabled);
                let alpha: Tile<f32, { [1] }> = exp(m_i - m_ij);
                let p0: Tile<f32, { [1] }> = exp(dot0 - m_ij);
                let p1: Tile<f32, { [1] }> = exp(dot1 - m_ij);
                let p2: Tile<f32, { [1] }> = exp(dot2 - m_ij);
                let p3: Tile<f32, { [1] }> = exp(dot3 - m_ij);
                let p4: Tile<f32, { [1] }> = exp(dot4 - m_ij);
                let p5: Tile<f32, { [1] }> = exp(dot5 - m_ij);
                let p6: Tile<f32, { [1] }> = exp(dot6 - m_ij);
                let p7: Tile<f32, { [1] }> = exp(dot7 - m_ij);
                l_i = l_i * alpha + p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7;
                let alpha_bc = alpha.reshape(const_shape![1, 1]).broadcast(chunk);
                let p0_bc = p0.reshape(const_shape![1, 1]).broadcast(chunk);
                let p1_bc = p1.reshape(const_shape![1, 1]).broadcast(chunk);
                let p2_bc = p2.reshape(const_shape![1, 1]).broadcast(chunk);
                let p3_bc = p3.reshape(const_shape![1, 1]).broadcast(chunk);
                let p4_bc = p4.reshape(const_shape![1, 1]).broadcast(chunk);
                let p5_bc = p5.reshape(const_shape![1, 1]).broadcast(chunk);
                let p6_bc = p6.reshape(const_shape![1, 1]).broadcast(chunk);
                let p7_bc = p7.reshape(const_shape![1, 1]).broadcast(chunk);
                acc0 = acc0 * alpha_bc
                    + tv0_0 * p0_bc
                    + tv1_0 * p1_bc
                    + tv2_0 * p2_bc
                    + tv3_0 * p3_bc
                    + tv4_0 * p4_bc
                    + tv5_0 * p5_bc
                    + tv6_0 * p6_bc
                    + tv7_0 * p7_bc;
                acc1 = acc1 * alpha_bc
                    + tv0_1 * p0_bc
                    + tv1_1 * p1_bc
                    + tv2_1 * p2_bc
                    + tv3_1 * p3_bc
                    + tv4_1 * p4_bc
                    + tv5_1 * p5_bc
                    + tv6_1 * p6_bc
                    + tv7_1 * p7_bc;
                m_i = m_ij;
            }
            let out_id = q_id * SPLITS + split_id;
            let m_bc = broadcast_scalar(tile_to_scalar(m_i.reshape(const_shape![])), chunk);
            let l_bc = broadcast_scalar(tile_to_scalar(l_i.reshape(const_shape![])), chunk);
            let mut pm = unsafe { partial_m.partition_full_mut(chunk) };
            unsafe { pm.store(m_bc, [out_id, 0i32]) };
            let mut pl = unsafe { partial_l.partition_full_mut(chunk) };
            unsafe { pl.store(l_bc, [out_id, 0i32]) };
            let mut pa = unsafe { partial_acc.partition_full_mut(chunk) };
            unsafe { pa.store(acc0, [out_id, 0i32]) };
            unsafe { pa.store(acc1, [out_id, 1i32]) };
        }

        #[wax::entry()]
        pub fn flash_attn_partial_batched_d128_b8<const CHUNK_SIZE: i32, const SPLITS: i32>(
            q: &Tensor<f32, { [-1, 128] }>,
            k: &Tensor<f32, { [-1, 128] }>,
            v: &Tensor<f32, { [-1, 128] }>,
            partial_m: &mut Tensor<f32, { [-1, 32] }>,
            partial_l: &mut Tensor<f32, { [-1, 32] }>,
            partial_acc: &mut Tensor<f32, { [-1, 128] }>,
        ) {
            let chunk = const_shape![1, 32];
            let block_id = get_tile_block_id().0;
            let q_id = block_id / SPLITS;
            let split_id = block_id % SPLITS;
            let kv_start = split_id * CHUNK_SIZE;
            let q_part = q.partition(chunk);
            let k_part = k.partition(chunk);
            let v_part = v.partition(chunk);
            let tq0 = q_part.load([q_id, 0i32]);
            let tq1 = q_part.load([q_id, 1i32]);
            let tq2 = q_part.load([q_id, 2i32]);
            let tq3 = q_part.load([q_id, 3i32]);
            let mut m_i: Tile<f32, { [1] }> = constant(-3.4028235e38f32, const_shape![1]);
            let mut l_i: Tile<f32, { [1] }> = constant(0.0f32, const_shape![1]);
            let mut acc0: Tile<f32, { [1, 32] }> = constant(0.0f32, chunk);
            let mut acc1: Tile<f32, { [1, 32] }> = constant(0.0f32, chunk);
            let mut acc2: Tile<f32, { [1, 32] }> = constant(0.0f32, chunk);
            let mut acc3: Tile<f32, { [1, 32] }> = constant(0.0f32, chunk);
            for jb in 0i32..(CHUNK_SIZE / 8) {
                let kv = kv_start + jb * 8;
                let tk0_0 = k_part.load([kv, 0i32]);
                let tk0_1 = k_part.load([kv, 1i32]);
                let tk0_2 = k_part.load([kv, 2i32]);
                let tk0_3 = k_part.load([kv, 3i32]);
                let tk1_0 = k_part.load([kv + 1, 0i32]);
                let tk1_1 = k_part.load([kv + 1, 1i32]);
                let tk1_2 = k_part.load([kv + 1, 2i32]);
                let tk1_3 = k_part.load([kv + 1, 3i32]);
                let tk2_0 = k_part.load([kv + 2, 0i32]);
                let tk2_1 = k_part.load([kv + 2, 1i32]);
                let tk2_2 = k_part.load([kv + 2, 2i32]);
                let tk2_3 = k_part.load([kv + 2, 3i32]);
                let tk3_0 = k_part.load([kv + 3, 0i32]);
                let tk3_1 = k_part.load([kv + 3, 1i32]);
                let tk3_2 = k_part.load([kv + 3, 2i32]);
                let tk3_3 = k_part.load([kv + 3, 3i32]);
                let tk4_0 = k_part.load([kv + 4, 0i32]);
                let tk4_1 = k_part.load([kv + 4, 1i32]);
                let tk4_2 = k_part.load([kv + 4, 2i32]);
                let tk4_3 = k_part.load([kv + 4, 3i32]);
                let tk5_0 = k_part.load([kv + 5, 0i32]);
                let tk5_1 = k_part.load([kv + 5, 1i32]);
                let tk5_2 = k_part.load([kv + 5, 2i32]);
                let tk5_3 = k_part.load([kv + 5, 3i32]);
                let tk6_0 = k_part.load([kv + 6, 0i32]);
                let tk6_1 = k_part.load([kv + 6, 1i32]);
                let tk6_2 = k_part.load([kv + 6, 2i32]);
                let tk6_3 = k_part.load([kv + 6, 3i32]);
                let tk7_0 = k_part.load([kv + 7, 0i32]);
                let tk7_1 = k_part.load([kv + 7, 1i32]);
                let tk7_2 = k_part.load([kv + 7, 2i32]);
                let tk7_3 = k_part.load([kv + 7, 3i32]);
                let tv0_0 = v_part.load([kv, 0i32]);
                let tv0_1 = v_part.load([kv, 1i32]);
                let tv0_2 = v_part.load([kv, 2i32]);
                let tv0_3 = v_part.load([kv, 3i32]);
                let tv1_0 = v_part.load([kv + 1, 0i32]);
                let tv1_1 = v_part.load([kv + 1, 1i32]);
                let tv1_2 = v_part.load([kv + 1, 2i32]);
                let tv1_3 = v_part.load([kv + 1, 3i32]);
                let tv2_0 = v_part.load([kv + 2, 0i32]);
                let tv2_1 = v_part.load([kv + 2, 1i32]);
                let tv2_2 = v_part.load([kv + 2, 2i32]);
                let tv2_3 = v_part.load([kv + 2, 3i32]);
                let tv3_0 = v_part.load([kv + 3, 0i32]);
                let tv3_1 = v_part.load([kv + 3, 1i32]);
                let tv3_2 = v_part.load([kv + 3, 2i32]);
                let tv3_3 = v_part.load([kv + 3, 3i32]);
                let tv4_0 = v_part.load([kv + 4, 0i32]);
                let tv4_1 = v_part.load([kv + 4, 1i32]);
                let tv4_2 = v_part.load([kv + 4, 2i32]);
                let tv4_3 = v_part.load([kv + 4, 3i32]);
                let tv5_0 = v_part.load([kv + 5, 0i32]);
                let tv5_1 = v_part.load([kv + 5, 1i32]);
                let tv5_2 = v_part.load([kv + 5, 2i32]);
                let tv5_3 = v_part.load([kv + 5, 3i32]);
                let tv6_0 = v_part.load([kv + 6, 0i32]);
                let tv6_1 = v_part.load([kv + 6, 1i32]);
                let tv6_2 = v_part.load([kv + 6, 2i32]);
                let tv6_3 = v_part.load([kv + 6, 3i32]);
                let tv7_0 = v_part.load([kv + 7, 0i32]);
                let tv7_1 = v_part.load([kv + 7, 1i32]);
                let tv7_2 = v_part.load([kv + 7, 2i32]);
                let tv7_3 = v_part.load([kv + 7, 3i32]);
                let dot0_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk0_0, 1i32);
                let dot0_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk0_1, 1i32);
                let dot0_2: Tile<f32, { [1] }> = reduce_sum(tq2 * tk0_2, 1i32);
                let dot0_3: Tile<f32, { [1] }> = reduce_sum(tq3 * tk0_3, 1i32);
                let dot0: Tile<f32, { [1] }> = dot0_0 + dot0_1 + dot0_2 + dot0_3;
                let dot1_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk1_0, 1i32);
                let dot1_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk1_1, 1i32);
                let dot1_2: Tile<f32, { [1] }> = reduce_sum(tq2 * tk1_2, 1i32);
                let dot1_3: Tile<f32, { [1] }> = reduce_sum(tq3 * tk1_3, 1i32);
                let dot1: Tile<f32, { [1] }> = dot1_0 + dot1_1 + dot1_2 + dot1_3;
                let dot2_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk2_0, 1i32);
                let dot2_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk2_1, 1i32);
                let dot2_2: Tile<f32, { [1] }> = reduce_sum(tq2 * tk2_2, 1i32);
                let dot2_3: Tile<f32, { [1] }> = reduce_sum(tq3 * tk2_3, 1i32);
                let dot2: Tile<f32, { [1] }> = dot2_0 + dot2_1 + dot2_2 + dot2_3;
                let dot3_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk3_0, 1i32);
                let dot3_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk3_1, 1i32);
                let dot3_2: Tile<f32, { [1] }> = reduce_sum(tq2 * tk3_2, 1i32);
                let dot3_3: Tile<f32, { [1] }> = reduce_sum(tq3 * tk3_3, 1i32);
                let dot3: Tile<f32, { [1] }> = dot3_0 + dot3_1 + dot3_2 + dot3_3;
                let dot4_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk4_0, 1i32);
                let dot4_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk4_1, 1i32);
                let dot4_2: Tile<f32, { [1] }> = reduce_sum(tq2 * tk4_2, 1i32);
                let dot4_3: Tile<f32, { [1] }> = reduce_sum(tq3 * tk4_3, 1i32);
                let dot4: Tile<f32, { [1] }> = dot4_0 + dot4_1 + dot4_2 + dot4_3;
                let dot5_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk5_0, 1i32);
                let dot5_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk5_1, 1i32);
                let dot5_2: Tile<f32, { [1] }> = reduce_sum(tq2 * tk5_2, 1i32);
                let dot5_3: Tile<f32, { [1] }> = reduce_sum(tq3 * tk5_3, 1i32);
                let dot5: Tile<f32, { [1] }> = dot5_0 + dot5_1 + dot5_2 + dot5_3;
                let dot6_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk6_0, 1i32);
                let dot6_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk6_1, 1i32);
                let dot6_2: Tile<f32, { [1] }> = reduce_sum(tq2 * tk6_2, 1i32);
                let dot6_3: Tile<f32, { [1] }> = reduce_sum(tq3 * tk6_3, 1i32);
                let dot6: Tile<f32, { [1] }> = dot6_0 + dot6_1 + dot6_2 + dot6_3;
                let dot7_0: Tile<f32, { [1] }> = reduce_sum(tq0 * tk7_0, 1i32);
                let dot7_1: Tile<f32, { [1] }> = reduce_sum(tq1 * tk7_1, 1i32);
                let dot7_2: Tile<f32, { [1] }> = reduce_sum(tq2 * tk7_2, 1i32);
                let dot7_3: Tile<f32, { [1] }> = reduce_sum(tq3 * tk7_3, 1i32);
                let dot7: Tile<f32, { [1] }> = dot7_0 + dot7_1 + dot7_2 + dot7_3;
                let m01 = maxf(dot0, dot1, nan::Disabled, ftz::Disabled);
                let m23 = maxf(dot2, dot3, nan::Disabled, ftz::Disabled);
                let m45 = maxf(dot4, dot5, nan::Disabled, ftz::Disabled);
                let m67 = maxf(dot6, dot7, nan::Disabled, ftz::Disabled);
                let m0123 = maxf(m01, m23, nan::Disabled, ftz::Disabled);
                let m4567 = maxf(m45, m67, nan::Disabled, ftz::Disabled);
                let m_blk = maxf(m0123, m4567, nan::Disabled, ftz::Disabled);
                let m_ij = maxf(m_i, m_blk, nan::Disabled, ftz::Disabled);
                let alpha: Tile<f32, { [1] }> = exp(m_i - m_ij);
                let p0: Tile<f32, { [1] }> = exp(dot0 - m_ij);
                let p1: Tile<f32, { [1] }> = exp(dot1 - m_ij);
                let p2: Tile<f32, { [1] }> = exp(dot2 - m_ij);
                let p3: Tile<f32, { [1] }> = exp(dot3 - m_ij);
                let p4: Tile<f32, { [1] }> = exp(dot4 - m_ij);
                let p5: Tile<f32, { [1] }> = exp(dot5 - m_ij);
                let p6: Tile<f32, { [1] }> = exp(dot6 - m_ij);
                let p7: Tile<f32, { [1] }> = exp(dot7 - m_ij);
                l_i = l_i * alpha + p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7;
                let alpha_bc = alpha.reshape(const_shape![1, 1]).broadcast(chunk);
                let p0_bc = p0.reshape(const_shape![1, 1]).broadcast(chunk);
                let p1_bc = p1.reshape(const_shape![1, 1]).broadcast(chunk);
                let p2_bc = p2.reshape(const_shape![1, 1]).broadcast(chunk);
                let p3_bc = p3.reshape(const_shape![1, 1]).broadcast(chunk);
                let p4_bc = p4.reshape(const_shape![1, 1]).broadcast(chunk);
                let p5_bc = p5.reshape(const_shape![1, 1]).broadcast(chunk);
                let p6_bc = p6.reshape(const_shape![1, 1]).broadcast(chunk);
                let p7_bc = p7.reshape(const_shape![1, 1]).broadcast(chunk);
                acc0 = acc0 * alpha_bc
                    + tv0_0 * p0_bc
                    + tv1_0 * p1_bc
                    + tv2_0 * p2_bc
                    + tv3_0 * p3_bc
                    + tv4_0 * p4_bc
                    + tv5_0 * p5_bc
                    + tv6_0 * p6_bc
                    + tv7_0 * p7_bc;
                acc1 = acc1 * alpha_bc
                    + tv0_1 * p0_bc
                    + tv1_1 * p1_bc
                    + tv2_1 * p2_bc
                    + tv3_1 * p3_bc
                    + tv4_1 * p4_bc
                    + tv5_1 * p5_bc
                    + tv6_1 * p6_bc
                    + tv7_1 * p7_bc;
                acc2 = acc2 * alpha_bc
                    + tv0_2 * p0_bc
                    + tv1_2 * p1_bc
                    + tv2_2 * p2_bc
                    + tv3_2 * p3_bc
                    + tv4_2 * p4_bc
                    + tv5_2 * p5_bc
                    + tv6_2 * p6_bc
                    + tv7_2 * p7_bc;
                acc3 = acc3 * alpha_bc
                    + tv0_3 * p0_bc
                    + tv1_3 * p1_bc
                    + tv2_3 * p2_bc
                    + tv3_3 * p3_bc
                    + tv4_3 * p4_bc
                    + tv5_3 * p5_bc
                    + tv6_3 * p6_bc
                    + tv7_3 * p7_bc;
                m_i = m_ij;
            }
            let out_id = q_id * SPLITS + split_id;
            let m_bc = broadcast_scalar(tile_to_scalar(m_i.reshape(const_shape![])), chunk);
            let l_bc = broadcast_scalar(tile_to_scalar(l_i.reshape(const_shape![])), chunk);
            let mut pm = unsafe { partial_m.partition_full_mut(chunk) };
            unsafe { pm.store(m_bc, [out_id, 0i32]) };
            let mut pl = unsafe { partial_l.partition_full_mut(chunk) };
            unsafe { pl.store(l_bc, [out_id, 0i32]) };
            let mut pa = unsafe { partial_acc.partition_full_mut(chunk) };
            unsafe { pa.store(acc0, [out_id, 0i32]) };
            unsafe { pa.store(acc1, [out_id, 1i32]) };
            unsafe { pa.store(acc2, [out_id, 2i32]) };
            unsafe { pa.store(acc3, [out_id, 3i32]) };
        }

        #[wax::entry()]
        pub fn flash_attn_combine_batched_d128<const SPLITS: i32>(
            partial_m: &Tensor<f32, { [-1, 32] }>,
            partial_l: &Tensor<f32, { [-1, 32] }>,
            partial_acc: &Tensor<f32, { [-1, 128] }>,
            out: &mut Tensor<f32, { [-1, 128] }>,
        ) {
            let chunk = const_shape![1, 32];
            let q_id = get_tile_block_id().0;
            let base = q_id * SPLITS;
            let m_part = partial_m.partition(chunk);
            let l_part = partial_l.partition(chunk);
            let acc_part = partial_acc.partition(chunk);
            let mut m_g = m_part.load([base, 0i32]);
            let mut l_g = l_part.load([base, 0i32]);
            let mut acc0_g = acc_part.load([base, 0i32]);
            let mut acc1_g = acc_part.load([base, 1i32]);
            let mut acc2_g = acc_part.load([base, 2i32]);
            let mut acc3_g = acc_part.load([base, 3i32]);
            for s in 1i32..SPLITS {
                let idx = base + s;
                let m_s = m_part.load([idx, 0i32]);
                let l_s = l_part.load([idx, 0i32]);
                let acc0_s = acc_part.load([idx, 0i32]);
                let acc1_s = acc_part.load([idx, 1i32]);
                let acc2_s = acc_part.load([idx, 2i32]);
                let acc3_s = acc_part.load([idx, 3i32]);
                let m_new: Tile<f32, { [1, 32] }> = maxf(m_g, m_s, nan::Disabled, ftz::Disabled);
                let alpha: Tile<f32, { [1, 32] }> = exp(m_g - m_new);
                let beta: Tile<f32, { [1, 32] }> = exp(m_s - m_new);
                l_g = l_g * alpha + l_s * beta;
                acc0_g = acc0_g * alpha + acc0_s * beta;
                acc1_g = acc1_g * alpha + acc1_s * beta;
                acc2_g = acc2_g * alpha + acc2_s * beta;
                acc3_g = acc3_g * alpha + acc3_s * beta;
                m_g = m_new;
            }
            acc0_g = acc0_g / l_g;
            acc1_g = acc1_g / l_g;
            acc2_g = acc2_g / l_g;
            acc3_g = acc3_g / l_g;
            let mut out_part = unsafe { out.partition_full_mut(chunk) };
            unsafe { out_part.store(acc0_g, [q_id, 0i32]) };
            unsafe { out_part.store(acc1_g, [q_id, 1i32]) };
            unsafe { out_part.store(acc2_g, [q_id, 2i32]) };
            unsafe { out_part.store(acc3_g, [q_id, 3i32]) };
        }

        #[wax::entry()]
        pub fn gemm_padded<
            E: ElementType,
            const BM: i32,
            const BN: i32,
            const BK: i32,
            const M: i32,
            const N: i32,
            const K: i32,
            const KB: i32,
        >(
            z: &mut Tensor<E, { [M, N] }>,
            x: &Tensor<E, { [M, K] }>,
            y: &Tensor<E, { [K, N] }>,
        ) {
            let pid: (i32, i32, i32) = get_tile_block_id();
            // `Tensor::partition` already declares `padding::Zero` - that is upstream's own
            // default - so the loads need no special spelling, only a backend that honours it.
            let xp = x.partition(const_shape![BM, BK]);
            let yp = y.partition(const_shape![BK, BN]);
            // Explicit f32: the frontend infers `mmaf`'s accumulator element from the
            // inputs when the type is left open, but the SGMMA register - and the POC's
            // reference lowering - accumulate in f32.
            let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            for i in 0i32..KB {
                acc = mmaf(xp.load([pid.0, i]), yp.load([i, pid.1]), acc);
            }
            let ztok: Token = get_tensor_token(z);
            let mut zp: PartitionMut<E, { [BM, BN] }> =
                unsafe { make_partition_view_mut(z, const_shape![BM, BN], padding::Zero, ztok) };
            let narrowed: Tile<E, { [BM, BN] }> = convert_tile(acc);
            unsafe { zp.store(narrowed, [pid.0, pid.1]) };
        }

        #[wax::entry()]
        pub fn vecmat_tg<E: ElementType, const N: i32, const K: i32, const SG: i32, const KS: i32>(
            out: &mut Tensor<E, { [1, N] }>,
            // `[K, 32]`, not `[1, K]`: with a column stride of 0 this IS the K-vector, viewed
            // as K rows each replicated across 32 lanes. The declared shape has to admit the
            // `[SG, 32]` tile - the frontend bounds-checks it - and this is the shape that
            // honestly describes the broadcast view the strides already build.
            x: &Tensor<E, { [K, 32] }>,
            y: &Tensor<E, { [K, N] }>,
        ) {
            let wide = const_shape![SG, 32];
            let blk = get_tile_block_id().0;
            let xp = x.partition(wide);
            let yp = y.partition(wide);
            let mut acc: Tile<f32, { [SG, 32] }> = constant(0.0f32, wide);
            for kb in 0i32..KS {
                let a: Tile<f32, { [SG, 32] }> = convert_tile(xp.load([kb, 0i32]));
                let b: Tile<f32, { [SG, 32] }> = convert_tile(yp.load([kb, blk]));
                acc = acc + a * b;
            }
            let tot: Tile<f32, { [32] }> = reduce_sum(acc, 0i32);
            let total: Tile<f32, { [1, 32] }> = tot.reshape(const_shape![1, 32]);
            let o: Tile<E, { [1, 32] }> = convert_tile(total);
            let otok: Token = get_tensor_token(out);
            let mut op: PartitionMut<E, { [1, 32] }> =
                unsafe { make_partition_view_mut(out, const_shape![1, 32], padding::Zero, otok) };
            unsafe { op.store(o, [0i32, blk]) };
        }

        #[wax::entry()]
        pub fn vm_probe_c<E: ElementType, const N: i32, const K: i32, const SG: i32, const KS: i32>(
            out: &mut Tensor<E, { [1, N] }>,
            x: &Tensor<E, { [K, 32] }>,
            y: &Tensor<E, { [K, N] }>,
        ) {
            let wide = const_shape![SG, 32];
            let blk = get_tile_block_id().0;
            let yp = y.partition(wide);
            let acc: Tile<f32, { [SG, 32] }> = convert_tile(yp.load([0i32, blk]));
            let tot: Tile<f32, { [32] }> = reduce_sum(acc, 0i32);
            let total: Tile<f32, { [1, 32] }> = tot.reshape(const_shape![1, 32]);
            let o: Tile<E, { [1, 32] }> = convert_tile(total);
            let otok: Token = get_tensor_token(out);
            let mut op: PartitionMut<E, { [1, 32] }> =
                unsafe { make_partition_view_mut(out, const_shape![1, 32], padding::Zero, otok) };
            unsafe { op.store(o, [0i32, blk]) };
        }

        #[wax::entry()]
        pub fn vm_probe_d<E: ElementType, const N: i32, const K: i32, const SG: i32, const KS: i32>(
            out: &mut Tensor<E, { [SG, N] }>,
            x: &Tensor<E, { [K, 32] }>,
            y: &Tensor<E, { [K, N] }>,
        ) {
            let wide = const_shape![SG, 32];
            let blk = get_tile_block_id().0;
            let yp = y.partition(wide);
            let mut acc: Tile<f32, { [SG, 32] }> = constant(0.0f32, wide);
            for kb in 0i32..KS {
                let b: Tile<f32, { [SG, 32] }> = convert_tile(yp.load([kb, blk]));
                acc = acc + b;
            }
            let o: Tile<E, { [SG, 32] }> = convert_tile(acc);
            let mut op: PartitionMut<E, { [SG, 32] }> = unsafe { out.partition_full_mut(wide) };
            unsafe { op.store(o, [0i32, blk]) };
        }

        #[wax::entry()]
        pub fn vm_probe_a<E: ElementType, const N: i32, const K: i32, const SG: i32, const KS: i32>(
            out: &mut Tensor<E, { [1, N] }>,
            x: &Tensor<E, { [K, 32] }>,
            y: &Tensor<E, { [K, N] }>,
        ) {
            let wide = const_shape![SG, 32];
            let blk = get_tile_block_id().0;
            let yp = y.partition(wide);
            let mut acc: Tile<f32, { [SG, 32] }> = constant(0.0f32, wide);
            for kb in 0i32..KS {
                let b: Tile<f32, { [SG, 32] }> = convert_tile(yp.load([kb, blk]));
                acc = acc + b;
            }
            let tot: Tile<f32, { [32] }> = reduce_sum(acc, 0i32);
            let total: Tile<f32, { [1, 32] }> = tot.reshape(const_shape![1, 32]);
            let o: Tile<E, { [1, 32] }> = convert_tile(total);
            let otok: Token = get_tensor_token(out);
            let mut op: PartitionMut<E, { [1, 32] }> =
                unsafe { make_partition_view_mut(out, const_shape![1, 32], padding::Zero, otok) };
            unsafe { op.store(o, [0i32, blk]) };
        }

        #[wax::entry()]
        pub fn vm_probe_b<E: ElementType, const N: i32, const K: i32, const SG: i32, const KS: i32>(
            out: &mut Tensor<E, { [1, N] }>,
            x: &Tensor<E, { [K, 32] }>,
            y: &Tensor<E, { [K, N] }>,
        ) {
            let wide = const_shape![SG, 32];
            let blk = get_tile_block_id().0;
            let xp = x.partition(wide);
            let mut acc: Tile<f32, { [SG, 32] }> = constant(0.0f32, wide);
            for kb in 0i32..KS {
                let a: Tile<f32, { [SG, 32] }> = convert_tile(xp.load([kb, 0i32]));
                acc = acc + a;
            }
            let tot: Tile<f32, { [32] }> = reduce_sum(acc, 0i32);
            let total: Tile<f32, { [1, 32] }> = tot.reshape(const_shape![1, 32]);
            let o: Tile<E, { [1, 32] }> = convert_tile(total);
            let otok: Token = get_tensor_token(out);
            let mut op: PartitionMut<E, { [1, 32] }> =
                unsafe { make_partition_view_mut(out, const_shape![1, 32], padding::Zero, otok) };
            unsafe { op.store(o, [0i32, blk]) };
        }

        #[wax::entry()]
        pub fn vecmat_reduce<E: ElementType, const N: i32, const SPLITS: i32>(
            out: &mut Tensor<E, { [1, N] }>,
            // `-1` on the plane axis: the addressing comes from the row pitch, and dim 0 needs
            // no bound because `s < S` by construction. (A const generic in the shape of a
            // CONCRETELY typed tensor is not something the frontend instantiates.)
            part: &Tensor<f32, { [-1, N] }>,
        ) {
            let ts = const_shape![1, 32];
            // The S partial planes are consecutive rows of pitch N, which is exactly what a
            // `[S, N]` declaration says - so plane `s`, chunk `blk` is the partition index
            // `[s, blk]` and needs no element-granular pointer.
            let blk = get_tile_block_id().0;
            let pp = part.partition(ts);
            let mut total: Tile<f32, { [1, 32] }> = pp.load([0i32, blk]);
            for s in 1i32..SPLITS {
                total = total + pp.load([s, blk]);
            }
            let acc: Tile<E, { [1, 32] }> = convert_tile(total);
            let otok: Token = get_tensor_token(out);
            let mut op: PartitionMut<E, { [1, 32] }> =
                unsafe { make_partition_view_mut(out, ts, padding::Zero, otok) };
            unsafe { op.store(acc, [0i32, blk]) };
        }

        #[wax::entry()]
        pub fn vector_add(
            a: &Tensor<f32, { [256] }>,
            b: &Tensor<f32, { [256] }>,
            c: &mut Tensor<f32, { [256] }>,
        ) {
            let tile_a = a.partition(const_shape![256]).load([0i32]);
            let tile_b = b.partition(const_shape![256]).load([0i32]);
            store_current_block_2d(c, tile_a + tile_b);
        }

        #[wax::entry()]
        pub fn relu(x: &Tensor<f32, { [256] }>, z: &mut Tensor<f32, { [256] }>) {
            let tile_x = x.partition(const_shape![256]).load([0i32]);
            let zero = constant(0.0f32, const_shape![256]);
            store_current_block_2d(z, maxf(tile_x, zero, nan::Disabled, ftz::Disabled));
        }

        #[wax::entry()]
        pub fn relu__if_else(x: &Tensor<f32, { [4] }>, z: &mut Tensor<f32, { [4] }>) {
            let tile_x = x.partition(const_shape![4]).load([0i32]);
            let zero = constant(0.0f32, const_shape![4]);
            store_current_block_2d(z, maxf(tile_x, zero, nan::Disabled, ftz::Disabled));
        }

        #[wax::entry()]
        pub fn scan_sum_test<const D: i32>(
            input: &Tensor<f32, { [-1, D] }>,
            output: &mut Tensor<f32, { [-1, D] }>,
        ) {
            let tile_shape = const_shape![1, D];
            let row = get_tile_block_id().0;
            let in_part = input.partition(tile_shape);
            let tile = in_part.load([row, 0i32]);
            let scanned: Tile<f32, { [1, D] }> = scan_sum(tile, 1i32, reverse::Forward, 0.0f32);
            let mut out_part: PartitionMut<f32, { [1, D] }> =
                unsafe { output.partition_full_mut(tile_shape) };
            unsafe { out_part.store(scanned, [row, 0i32]) };
        }

        #[wax::entry()]
        pub fn scan_sum_test__flash_attn<const D: i32>(
            input: &Tensor<f32, { [-1, D] }>,
            output: &mut Tensor<f32, { [-1, D] }>,
        ) {
            let tile_shape = const_shape![1, D];
            let pid = get_tile_block_id();
            let row = pid.0;
            let in_part = input.partition(tile_shape);
            let tile = in_part.load([row, 0i32]);
            let scanned: Tile<f32, { [1, D] }> = scan_sum(tile, 1i32, reverse::Forward, 0.0f32);
            let mut out_part: PartitionMut<f32, { [1, D] }> =
                unsafe { output.partition_full_mut(tile_shape) };
            unsafe { out_part.store(scanned, [row, 0i32]) };
        }

        #[wax::entry()]
        pub fn int_to_float(x: &Tensor<i32, { [1, 32] }>, out: &mut Tensor<f32, { [-1, 32] }>) {
            let ts = const_shape![1, 32];
            let t: Tile<i32, { [1, 32] }> = x.partition(ts).load([0i32, 0i32]);
            let f: Tile<f32, { [1, 32] }> = convert_tile(t);
            let mut op: PartitionMut<f32, { [1, 32] }> = unsafe { out.partition_full_mut(ts) };
            unsafe { op.store(f, [0i32, 0i32]) };
        }

        #[wax::entry()]
        pub fn int_to_float__int_load(x: &Tensor<i32, { [1, 32] }>, out: &mut Tensor<f32, { [1, 32] }>) {
            let ts = const_shape![1, 32];
            let t: Tile<i32, { [1, 32] }> = x.partition(ts).load([0i32, 0i32]);
            let f: Tile<f32, { [1, 32] }> = convert_tile(t);
            let mut op: PartitionMut<f32, { [1, 32] }> = unsafe { out.partition_full_mut(ts) };
            unsafe { op.store(f, [0i32, 0i32]) };
        }

        #[wax::entry()]
        pub fn softmax_bn<E: ElementType, const N: i32, const BNW: i32>(
            x: &Tensor<E, { [-1, N] }>,
            out: &mut Tensor<E, { [1, N] }>,
        ) {
            let ts = const_shape![1, BNW];
            let row = get_tile_block_id().0;
            let nblk = (N + BNW - 1i32) / BNW;

            // `iota` is the element index within the tile, which here spans the whole
            // threadgroup - so it is the thread's column offset inside a block.
            let lanes1: Tile<i32, { [BNW] }> = iota(const_shape![BNW]);
            let lanes: Tile<i32, { [1, BNW] }> = lanes1.reshape(ts);
            let n_bc: Tile<i32, { [1, BNW] }> = broadcast_scalar(N, ts);
            let ninf: Tile<f32, { [1, BNW] }> = constant(-3.4028235e38f32, ts);
            let zero: Tile<f32, { [1, BNW] }> = constant(0.0f32, ts);

            // Pass 1: global max. Tail lanes read past N (masked to -inf), as before.
            let mut mx: Tile<f32, { [1, BNW] }> = constant(-3.4028235e38f32, ts);
            for jj in 0i32..nblk {
                let col: Tile<i32, { [1, BNW] }> = lanes + broadcast_scalar(jj * BNW, ts);
                let mask: Tile<bool, { [1, BNW] }> = cmpi(col, n_bc, predicate::LessThan);
                let tx: Tile<f32, { [1, BNW] }> = convert_tile(load_tile(x, ts, [row, jj]));
                mx = maxf(mx, select(mask, tx, ninf), nan::Disabled, ftz::Disabled);
            }
            let m1: Tile<f32, { [1] }> = reduce_max(mx, 1i32);
            let gmax_bc: Tile<f32, { [1, BNW] }> =
                broadcast_scalar(tile_to_scalar(m1.reshape(const_shape![])), ts);

            // Pass 2: global sum of exp(x - gmax).
            let mut sm: Tile<f32, { [1, BNW] }> = constant(0.0f32, ts);
            for jj in 0i32..nblk {
                let col: Tile<i32, { [1, BNW] }> = lanes + broadcast_scalar(jj * BNW, ts);
                let mask: Tile<bool, { [1, BNW] }> = cmpi(col, n_bc, predicate::LessThan);
                let tx: Tile<f32, { [1, BNW] }> = convert_tile(load_tile(x, ts, [row, jj]));
                let e: Tile<f32, { [1, BNW] }> = exp(tx - gmax_bc);
                sm = sm + select(mask, e, zero);
            }
            let s1: Tile<f32, { [1] }> = reduce_sum(sm, 1i32);
            let one_bc: Tile<f32, { [1, BNW] }> = constant(1.0f32, ts);
            let gsum_bc: Tile<f32, { [1, BNW] }> =
                broadcast_scalar(tile_to_scalar(s1.reshape(const_shape![])), ts);
            let inv_bc: Tile<f32, { [1, BNW] }> = one_bc / gsum_bc;

            // Pass 3: normalize and store. The view declares `padding`, so the backend drops
            // the tail lanes against the tensor's runtime extent - the thread's own column
            // index is no longer needed, and with it the last reason this kernel asked which
            // SIMD group and lane it was.
            let otok: Token = get_tensor_token(out);
            let mut op: PartitionMut<E, { [1, BNW] }> =
                unsafe { make_partition_view_mut(out, ts, padding::Zero, otok) };
            for jj in 0i32..nblk {
                let tx: Tile<f32, { [1, BNW] }> = convert_tile(load_tile(x, ts, [row, jj]));
                let y: Tile<E, { [1, BNW] }> = convert_tile(exp(tx - gmax_bc) * inv_bc);
                unsafe { op.store(y, [row, jj]) };
            }
        }

        #[wax::entry()]
        pub fn argmax_bn<E: ElementType, const N: i32, const BNW: i32>(
            x: &Tensor<E, { [-1, N] }>,
            out: &mut Tensor<i32, { [-1, 1] }>,
        ) {
            let ts = const_shape![1, BNW];
            let row = get_tile_block_id().0;
            let nblk = (N + BNW - 1i32) / BNW;

            let lanes1: Tile<i32, { [BNW] }> = iota(const_shape![BNW]);
            let lanes: Tile<i32, { [1, BNW] }> = lanes1.reshape(ts);
            let n_bc: Tile<i32, { [1, BNW] }> = broadcast_scalar(N, ts);
            let ninf: Tile<f32, { [1, BNW] }> = constant(-3.4028235e38f32, ts);
            // Above any valid index, so a lane that matches nothing never wins the min.
            let big: Tile<f32, { [1, BNW] }> = constant(2147483600.0f32, ts);

            // Phase 1: global max.
            let mut mx: Tile<f32, { [1, BNW] }> = constant(-3.4028235e38f32, ts);
            for jj in 0i32..nblk {
                let col: Tile<i32, { [1, BNW] }> = lanes + broadcast_scalar(jj * BNW, ts);
                let inb: Tile<bool, { [1, BNW] }> = cmpi(col, n_bc, predicate::LessThan);
                let tx: Tile<f32, { [1, BNW] }> = convert_tile(load_tile(x, ts, [row, jj]));
                mx = maxf(mx, select(inb, tx, ninf), nan::Disabled, ftz::Disabled);
            }
            let m1: Tile<f32, { [1] }> = reduce_max(mx, 1i32);
            let gmax: Tile<f32, { [1, BNW] }> =
                broadcast_scalar(tile_to_scalar(m1.reshape(const_shape![])), ts);

            // Phase 2: smallest index attaining the max. An out-of-row lane holds -inf, so
            // it can only tie when the whole row is -inf - and then its index (>= N) still
            // loses the min to a real lane.
            let mut best: Tile<f32, { [1, BNW] }> = big;
            for jj in 0i32..nblk {
                let col: Tile<i32, { [1, BNW] }> = lanes + broadcast_scalar(jj * BNW, ts);
                let inb: Tile<bool, { [1, BNW] }> = cmpi(col, n_bc, predicate::LessThan);
                let tx: Tile<f32, { [1, BNW] }> = convert_tile(load_tile(x, ts, [row, jj]));
                let txm: Tile<f32, { [1, BNW] }> = select(inb, tx, ninf);
                let hit: Tile<bool, { [1, BNW] }> =
                    cmpf(txm, gmax, predicate::Equal, cmp_ordering::Ordered);
                let colf: Tile<f32, { [1, BNW] }> = convert_tile(col);
                best = minf(best, select(hit, colf, big), nan::Disabled, ftz::Disabled);
            }
            let i1: Tile<f32, { [1] }> = reduce_min(best, 1i32);

            // Only threadgroup position 0 stores. The reduce leaves the same value on every
            // thread, so this is not a correctness condition any more (it was, before the
            // lane term came from the tile extent: a [1,1] tile has none) - dropping the
            // guard was TESTED and still correct. It stays because 1024 threads writing the
            // same address is 1024 stores instead of one.
            let out_ts = const_shape![1, 1];
            let r1: Tile<i32, { [1, 1] }> = convert_tile(i1.reshape(out_ts));
            let mut op: PartitionMut<i32, { [1, 1] }> = unsafe { out.partition_full_mut(out_ts) };
            // No guard: `reduce_min` leaves every thread with the same value and a `[1,1]`
            // tile's lane term is `tp.x & 0` = 0, so they all form the same address. The guard
            // bounded redundant identical writes, not correctness. Matches candle.
            unsafe { op.store(r1, [row, 0i32]) };
        }

        #[wax::entry()]
        pub fn sdpa_sd_d128<
            E: ElementType,
            const NH: i32,
            const NKV: i32,
            const BN: i32,
            const BN32: i32,
        >(
            q: &Tensor<E, { [-1, 32, 4] }>,
            k: &Tensor<E, { [-1, 32, 4] }>,
            v: &Tensor<E, { [-1, 128] }>,
            meta: &Tensor<f32, { [-1] }>,
            out: &mut Tensor<E, { [-1, 128] }>,
        ) {
            let chunk = const_shape![1, 32];
            let wide = const_shape![BN, 32];
            let one = const_shape![1];
            let head = get_tile_block_id().0;

            let mp = meta.partition(one);
            let n_f: f32 = tile_to_scalar(mp.load([0i32]).reshape(const_shape![]));
            let n: i32 = convert_scalar(n_f);
            let scale: f32 = tile_to_scalar(mp.load([1i32]).reshape(const_shape![]));
            let khs_f: f32 = tile_to_scalar(mp.load([2i32]).reshape(const_shape![]));
            let khs: i32 = convert_scalar(khs_f);
            let scale_w: Tile<f32, { [BN, 32] }> = broadcast_scalar(scale, wide);

            let bi = head / NH;
            let h = head % NH;
            let kvh = h / (NH / NKV);
            let kv_base = (bi * NKV + kvh) * khs;

            let lanevec = const_shape![1, 32, 1];
            let qp = q.partition(lanevec);
            let q0: Tile<f32, { [1, 32] }> = convert_tile(qp.load([head, 0i32, 0i32]).reshape(chunk));
            let q1: Tile<f32, { [1, 32] }> = convert_tile(qp.load([head, 0i32, 1i32]).reshape(chunk));
            let q2: Tile<f32, { [1, 32] }> = convert_tile(qp.load([head, 0i32, 2i32]).reshape(chunk));
            let q3: Tile<f32, { [1, 32] }> = convert_tile(qp.load([head, 0i32, 3i32]).reshape(chunk));
            // Identity in registers: every row already holds this head's q on its own lanes.
            let q0w: Tile<f32, { [BN, 32] }> = q0.broadcast(wide);
            let q1w: Tile<f32, { [BN, 32] }> = q1.broadcast(wide);
            let q2w: Tile<f32, { [BN, 32] }> = q2.broadcast(wide);
            let q3w: Tile<f32, { [BN, 32] }> = q3.broadcast(wide);

            let kw = const_shape![BN, 32, 1];
            let kp = k.partition(kw);
            let vp = v.partition(wide);
            let v0 = kv_base * 4i32;

            let full = n / BN;
            let mut m_i: Tile<f32, { [BN, 32] }> = broadcast_scalar(-3.4028235e38f32, wide);
            let mut l_i: Tile<f32, { [BN, 32] }> = constant(0.0f32, wide);
            let mut acc0: Tile<f32, { [BN, 32] }> = constant(0.0f32, wide);
            let mut acc1: Tile<f32, { [BN, 32] }> = constant(0.0f32, wide);
            let mut acc2: Tile<f32, { [BN, 32] }> = constant(0.0f32, wide);
            let mut acc3: Tile<f32, { [BN, 32] }> = constant(0.0f32, wide);
            for jj in 0i32..full {
                let k0: Tile<f32, { [BN, 32] }> =
                    convert_tile(kp.load([jj, kv_base, 0i32]).reshape(wide));
                let k1: Tile<f32, { [BN, 32] }> =
                    convert_tile(kp.load([jj, kv_base, 1i32]).reshape(wide));
                let k2: Tile<f32, { [BN, 32] }> =
                    convert_tile(kp.load([jj, kv_base, 2i32]).reshape(wide));
                let k3: Tile<f32, { [BN, 32] }> =
                    convert_tile(kp.load([jj, kv_base, 3i32]).reshape(wide));
                let hs: Tile<f32, { [BN, 32] }> = q0w * k0 + q1w * k1 + q2w * k2 + q3w * k3;
                let raw: Tile<f32, { [BN] }> = reduce_sum(hs, 1i32);
                let dot: Tile<f32, { [BN, 32] }> =
                    raw.reshape(const_shape![BN, 1]).broadcast(wide) * scale_w;
                let m_ij: Tile<f32, { [BN, 32] }> = maxf(m_i, dot, nan::Disabled, ftz::Disabled);
                let alpha: Tile<f32, { [BN, 32] }> = exp(m_i - m_ij);
                let p_val: Tile<f32, { [BN, 32] }> = exp(dot - m_ij);
                l_i = l_i * alpha + p_val;
                let tv0: Tile<f32, { [BN, 32] }> = convert_tile(vp.load([jj, v0]));
                let tv1: Tile<f32, { [BN, 32] }> = convert_tile(vp.load([jj, v0 + 1i32]));
                let tv2: Tile<f32, { [BN, 32] }> = convert_tile(vp.load([jj, v0 + 2i32]));
                let tv3: Tile<f32, { [BN, 32] }> = convert_tile(vp.load([jj, v0 + 3i32]));
                acc0 = acc0 * alpha + tv0 * p_val;
                acc1 = acc1 * alpha + tv1 * p_val;
                acc2 = acc2 * alpha + tv2 * p_val;
                acc3 = acc3 * alpha + tv3 * p_val;
                m_i = m_ij;
            }
            if n - full * BN > 0i32 {
                let iw1: Tile<i32, { [BN32] }> = iota(const_shape![BN32]);
                let iw: Tile<i32, { [BN, 32] }> = iw1.reshape(wide);
                let il1: Tile<i32, { [32] }> = iota(const_shape![32]);
                let il2: Tile<i32, { [1, 32] }> = il1.reshape(chunk);
                let il: Tile<i32, { [BN, 32] }> = il2.broadcast(wide);
                let s32: Tile<i32, { [BN, 32] }> = iw - il;
                let ninf: Tile<f32, { [BN, 32] }> = broadcast_scalar(-3.4028235e38f32, wide);
                let lim: Tile<i32, { [BN, 32] }> = broadcast_scalar(32i32 * (n - full * BN), wide);
                let inb: Tile<bool, { [BN, 32] }> = cmpi(s32, lim, predicate::LessThan);
                let k0: Tile<f32, { [BN, 32] }> =
                    convert_tile(kp.load([full, kv_base, 0i32]).reshape(wide));
                let k1: Tile<f32, { [BN, 32] }> =
                    convert_tile(kp.load([full, kv_base, 1i32]).reshape(wide));
                let k2: Tile<f32, { [BN, 32] }> =
                    convert_tile(kp.load([full, kv_base, 2i32]).reshape(wide));
                let k3: Tile<f32, { [BN, 32] }> =
                    convert_tile(kp.load([full, kv_base, 3i32]).reshape(wide));
                let hs: Tile<f32, { [BN, 32] }> = q0w * k0 + q1w * k1 + q2w * k2 + q3w * k3;
                let raw: Tile<f32, { [BN] }> = reduce_sum(hs, 1i32);
                let dot0: Tile<f32, { [BN, 32] }> =
                    raw.reshape(const_shape![BN, 1]).broadcast(wide) * scale_w;
                let dot: Tile<f32, { [BN, 32] }> = select(inb, dot0, ninf);
                let m_ij: Tile<f32, { [BN, 32] }> = maxf(m_i, dot, nan::Disabled, ftz::Disabled);
                let alpha: Tile<f32, { [BN, 32] }> = exp(m_i - m_ij);
                let p_val: Tile<f32, { [BN, 32] }> = exp(dot - m_ij);
                l_i = l_i * alpha + p_val;
                let tv0: Tile<f32, { [BN, 32] }> = convert_tile(vp.load([full, v0]));
                let tv1: Tile<f32, { [BN, 32] }> = convert_tile(vp.load([full, v0 + 1i32]));
                let tv2: Tile<f32, { [BN, 32] }> = convert_tile(vp.load([full, v0 + 2i32]));
                let tv3: Tile<f32, { [BN, 32] }> = convert_tile(vp.load([full, v0 + 3i32]));
                acc0 = acc0 * alpha + tv0 * p_val;
                acc1 = acc1 * alpha + tv1 * p_val;
                acc2 = acc2 * alpha + tv2 * p_val;
                acc3 = acc3 * alpha + tv3 * p_val;
                m_i = m_ij;
            }
            // Within-row reduce first: a value no-op (every lane of a row already agrees, so
            // `max` and not `sum`) that lets the uniformity analysis see the invariance and
            // skip two redistributions.
            let m_row: Tile<f32, { [BN] }> = reduce_max(m_i, 1i32);
            let mt: Tile<f32, { [BN, 32] }> = m_row.reshape(const_shape![BN, 1]).broadcast(wide);
            let l_row: Tile<f32, { [BN] }> = reduce_max(l_i, 1i32);
            let lt: Tile<f32, { [BN, 32] }> = l_row.reshape(const_shape![BN, 1]).broadcast(wide);
            let gm_w: Tile<f32, { [32] }> = reduce_max(mt, 0i32);
            let gm: Tile<f32, { [BN, 32] }> = gm_w.reshape(chunk).broadcast(wide);
            let f: Tile<f32, { [BN, 32] }> = exp(mt - gm);
            let gl_w: Tile<f32, { [32] }> = reduce_sum(lt * f, 0i32);
            let gl: Tile<f32, { [1, 32] }> = gl_w.reshape(chunk);
            // Hoist ALL FOUR products before ANY reduce: `batch_column_reduces` shares one set
            // of barriers across ADJACENT column reduces, and an interleaved mul breaks that.
            let a0: Tile<f32, { [BN, 32] }> = acc0 * f;
            let a1: Tile<f32, { [BN, 32] }> = acc1 * f;
            let a2: Tile<f32, { [BN, 32] }> = acc2 * f;
            let a3: Tile<f32, { [BN, 32] }> = acc3 * f;
            let n0: Tile<f32, { [32] }> = reduce_sum(a0, 0i32);
            let n1: Tile<f32, { [32] }> = reduce_sum(a1, 0i32);
            let n2: Tile<f32, { [32] }> = reduce_sum(a2, 0i32);
            let n3: Tile<f32, { [32] }> = reduce_sum(a3, 0i32);
            let mut op: PartitionMut<E, { [1, 32] }> = unsafe { out.partition_full_mut(chunk) };
            let o0: Tile<E, { [1, 32] }> = convert_tile(n0.reshape(chunk) / gl);
            let o1: Tile<E, { [1, 32] }> = convert_tile(n1.reshape(chunk) / gl);
            let o2: Tile<E, { [1, 32] }> = convert_tile(n2.reshape(chunk) / gl);
            let o3: Tile<E, { [1, 32] }> = convert_tile(n3.reshape(chunk) / gl);
            unsafe { op.store(o0, [head, 0i32]) };
            unsafe { op.store(o1, [head, 1i32]) };
            unsafe { op.store(o2, [head, 2i32]) };
            unsafe { op.store(o3, [head, 3i32]) };
        }

        #[wax::entry()]
        pub fn sdpa_dbg<const NH: i32, const NKV: i32, const BN: i32, const BN32: i32>(
            out: &mut Tensor<f32, { [-1, 32] }>,
            q: &Tensor<f32, { [-1, 64] }>,
            k: &Tensor<f32, { [-1, 64] }>,
            meta: &Tensor<f32, { [-1] }>,
        ) {
            let chunk = const_shape![1, 32];
            let wide = const_shape![BN, 32];
            let one = const_shape![1];
            let head = get_tile_block_id().0;
            let mp = meta.partition(one);
            let n_f: f32 = tile_to_scalar(mp.load([0i32]).reshape(const_shape![]));
            let n: i32 = convert_scalar(n_f);
            let scale: f32 = tile_to_scalar(mp.load([1i32]).reshape(const_shape![]));
            let khs_f: f32 = tile_to_scalar(mp.load([2i32]).reshape(const_shape![]));
            let khs: i32 = convert_scalar(khs_f);
            let bi = head / NH;
            let h = head % NH;
            let kvh = h / (NH / NKV);
            let col0 = (bi * NKV + kvh) * khs * 2i32;
            let scale_bc: Tile<f32, { [1, 32] }> = broadcast_scalar(scale, chunk);
            let qp = q.partition(chunk);
            let tq0: Tile<f32, { [1, 32] }> = qp.load([head, 0i32]) * scale_bc;
            let tq1: Tile<f32, { [1, 32] }> = qp.load([head, 1i32]) * scale_bc;
            let tq0w: Tile<f32, { [BN, 32] }> = tq0.broadcast(wide);
            let tq1w: Tile<f32, { [BN, 32] }> = tq1.broadcast(wide);
            let kp = k.partition(wide);
            let full = n / BN;
            let mut m_i: Tile<f32, { [BN, 32] }> = broadcast_scalar(-3.4028235e38f32, wide);
            let mut acc: Tile<f32, { [BN, 32] }> = constant(0.0f32, wide);
            let mut acc2: Tile<f32, { [BN, 32] }> = constant(0.0f32, wide);
            let mut l_i: Tile<f32, { [BN, 32] }> = constant(0.0f32, wide);
            for jj in 0i32..full {
                let tk0: Tile<f32, { [BN, 32] }> = kp.load([jj, col0]);
                let tk1: Tile<f32, { [BN, 32] }> = kp.load([jj, col0 + 1i32]);
                let dot_r: Tile<f32, { [BN] }> = reduce_sum(tq0w * tk0 + tq1w * tk1, 1i32);
                let dot: Tile<f32, { [BN, 32] }> = dot_r.reshape(const_shape![BN, 1]).broadcast(wide);
                m_i = maxf(m_i, dot, nan::Disabled, ftz::Disabled);
                acc = acc + tk0;
                acc2 = acc2 + tk1;
                l_i = l_i + dot;
            }
            // The conditional tail: an `if` carrying all four [BN,32] tiles. At n == full*BN
            // it does not run, but it is still in the CFG.
            if n - full * BN > 0i32 {
                let tk0: Tile<f32, { [BN, 32] }> = kp.load([full, col0]);
                let tk1: Tile<f32, { [BN, 32] }> = kp.load([full, col0 + 1i32]);
                let dot_r: Tile<f32, { [BN] }> = reduce_sum(tq0w * tk0 + tq1w * tk1, 1i32);
                let dot: Tile<f32, { [BN, 32] }> = dot_r.reshape(const_shape![BN, 1]).broadcast(wide);
                m_i = maxf(m_i, dot, nan::Disabled, ftz::Disabled);
                acc = acc + tk0;
                acc2 = acc2 + tk1;
                l_i = l_i + dot;
            }
            // The merge: column max over the loop-carried max, then a column SUM of a second
            // carry scaled by something derived from it - the shape the real merge has.
            let gm_w: Tile<f32, { [32] }> = reduce_max(m_i, 0i32);
            let gm: Tile<f32, { [BN, 32] }> = gm_w.reshape(chunk).broadcast(wide);
            let f: Tile<f32, { [BN, 32] }> = exp(m_i - gm);
            let gl_w: Tile<f32, { [32] }> = reduce_sum(l_i * f, 0i32);
            let gl: Tile<f32, { [1, 32] }> = gl_w.reshape(chunk);
            let a0: Tile<f32, { [BN, 32] }> = acc * f;
            let a1: Tile<f32, { [BN, 32] }> = acc2 * f;
            let n0: Tile<f32, { [32] }> = reduce_sum(a0, 0i32);
            let n1: Tile<f32, { [32] }> = reduce_sum(a1, 0i32);
            let _unused: Tile<f32, { [1, 32] }> = n1.reshape(chunk) + gl;
            let res: Tile<f32, { [BN, 32] }> = n0.reshape(chunk).broadcast(wide);
            let mut op: PartitionMut<f32, { [BN, 32] }> = unsafe { out.partition_full_mut(wide) };
            unsafe { op.store(res, [0i32, 0i32]) };
        }

        #[wax::entry()]
        pub fn sdpa_sd_d128_p1<
            E: ElementType,
            const NH: i32,
            const NKV: i32,
            const SPLITS: i32,
            const GQA: i32,
        >(
            q: &Tensor<E, { [-1, 32, 4] }>,
            k: &Tensor<E, { [-1, 32, 4] }>,
            v: &Tensor<E, { [-1, 128] }>,
            meta: &Tensor<f32, { [-1] }>,
            partial_m: &mut Tensor<f32, { [-1, -1, 32] }>,
            partial_l: &mut Tensor<f32, { [-1, -1, 32] }>,
            partial_o: &mut Tensor<f32, { [-1, -1, 128] }>,
        ) {
            let chunk = const_shape![1, 32];
            let wide = const_shape![GQA, 32];
            let one = const_shape![1];
            let blk = get_tile_block_id().0;
            let kv_lin = blk / SPLITS;
            let split = blk % SPLITS;

            let mp = meta.partition(one);
            let n_f: f32 = tile_to_scalar(mp.load([0i32]).reshape(const_shape![]));
            let n: i32 = convert_scalar(n_f);
            let scale: f32 = tile_to_scalar(mp.load([1i32]).reshape(const_shape![]));
            let khs_f: f32 = tile_to_scalar(mp.load([2i32]).reshape(const_shape![]));
            let khs: i32 = convert_scalar(khs_f);
            let scale_w: Tile<f32, { [GQA, 32] }> = broadcast_scalar(scale, wide);

            let bi = kv_lin / NKV;
            let kvh = kv_lin % NKV;
            let head_base = bi * NH + kvh * GQA;
            let kv_base = kv_lin * khs;

            // q: dim 1 has coefficient 32*4 = 128, one row, so the head base goes in there.
            let qw = const_shape![GQA, 32, 1];
            let qp = q.partition(qw);
            let q0: Tile<f32, { [GQA, 32] }> =
                convert_tile(qp.load([0i32, head_base, 0i32]).reshape(wide));
            let q1: Tile<f32, { [GQA, 32] }> =
                convert_tile(qp.load([0i32, head_base, 1i32]).reshape(wide));
            let q2: Tile<f32, { [GQA, 32] }> =
                convert_tile(qp.load([0i32, head_base, 2i32]).reshape(wide));
            let q3: Tile<f32, { [GQA, 32] }> =
                convert_tile(qp.load([0i32, head_base, 3i32]).reshape(wide));

            // k and v are the SAME for every head in the group, so they load once as a
            // `[1,32]` tile and broadcast - an identity in registers.
            let lanevec = const_shape![1, 32, 1];
            let kp = k.partition(lanevec);
            let vp = v.partition(chunk);

            let cnt = (n - split + (SPLITS - 1i32)) / SPLITS;
            let mut m_i: Tile<f32, { [GQA, 32] }> = broadcast_scalar(-3.4028235e38f32, wide);
            let mut l_i: Tile<f32, { [GQA, 32] }> = constant(0.0f32, wide);
            let mut acc0: Tile<f32, { [GQA, 32] }> = constant(0.0f32, wide);
            let mut acc1: Tile<f32, { [GQA, 32] }> = constant(0.0f32, wide);
            let mut acc2: Tile<f32, { [GQA, 32] }> = constant(0.0f32, wide);
            let mut acc3: Tile<f32, { [GQA, 32] }> = constant(0.0f32, wide);
            for jj in 0i32..cnt {
                let key = kv_base + split + jj * SPLITS;
                let k0c: Tile<f32, { [1, 32] }> =
                    convert_tile(kp.load([key, 0i32, 0i32]).reshape(chunk));
                let k1c: Tile<f32, { [1, 32] }> =
                    convert_tile(kp.load([key, 0i32, 1i32]).reshape(chunk));
                let k2c: Tile<f32, { [1, 32] }> =
                    convert_tile(kp.load([key, 0i32, 2i32]).reshape(chunk));
                let k3c: Tile<f32, { [1, 32] }> =
                    convert_tile(kp.load([key, 0i32, 3i32]).reshape(chunk));
                let hs: Tile<f32, { [GQA, 32] }> = q0 * k0c.broadcast(wide)
                    + q1 * k1c.broadcast(wide)
                    + q2 * k2c.broadcast(wide)
                    + q3 * k3c.broadcast(wide);
                let raw: Tile<f32, { [GQA] }> = reduce_sum(hs, 1i32);
                let dot: Tile<f32, { [GQA, 32] }> =
                    raw.reshape(const_shape![GQA, 1]).broadcast(wide) * scale_w;
                let m_ij: Tile<f32, { [GQA, 32] }> = maxf(m_i, dot, nan::Disabled, ftz::Disabled);
                let alpha: Tile<f32, { [GQA, 32] }> = exp(m_i - m_ij);
                let p_val: Tile<f32, { [GQA, 32] }> = exp(dot - m_ij);
                l_i = l_i * alpha + p_val;
                let tv0: Tile<f32, { [1, 32] }> = convert_tile(vp.load([key, 0i32]));
                let tv1: Tile<f32, { [1, 32] }> = convert_tile(vp.load([key, 1i32]));
                let tv2: Tile<f32, { [1, 32] }> = convert_tile(vp.load([key, 2i32]));
                let tv3: Tile<f32, { [1, 32] }> = convert_tile(vp.load([key, 3i32]));
                acc0 = acc0 * alpha + tv0.broadcast(wide) * p_val;
                acc1 = acc1 * alpha + tv1.broadcast(wide) * p_val;
                acc2 = acc2 * alpha + tv2.broadcast(wide) * p_val;
                acc3 = acc3 * alpha + tv3.broadcast(wide) * p_val;
                m_i = m_ij;
            }
            // `partial_m`/`_l` hold ONE float per (head, split), but a `[GQA,1]` tile cannot
            // deliver it: with only GQA elements its lane term is `tp.x & (GQA-1)`, which maps
            // threads to elements by the LOW BITS of the thread index rather than by row, so
            // row 0's lanes would write row 1's slot. Declaring a third dim with STRIDE 0
            // instead makes all 32 lanes of a row form the SAME address holding the SAME value
            // - one word written redundantly, which is the write-side mirror of the stride-0
            // uniform READ this codebase already uses for `meta`. It also removes the need to
            // reduce first: every lane of a row already carries the row's value.
            let narrow = const_shape![GQA, 1, 32];
            let mut pmw: PartitionMut<f32, { [GQA, 1, 32] }> =
                unsafe { partial_m.partition_full_mut(narrow) };
            let mut plw: PartitionMut<f32, { [GQA, 1, 32] }> =
                unsafe { partial_l.partition_full_mut(narrow) };
            unsafe { pmw.store(m_i.reshape(narrow), [kv_lin, split, 0i32]) };
            unsafe { plw.store(l_i.reshape(narrow), [kv_lin, split, 0i32]) };
            let ow = const_shape![GQA, 1, 32];
            let mut op: PartitionMut<f32, { [GQA, 1, 32] }> =
                unsafe { partial_o.partition_full_mut(ow) };
            unsafe { op.store(acc0.reshape(ow), [kv_lin, split, 0i32]) };
            unsafe { op.store(acc1.reshape(ow), [kv_lin, split, 1i32]) };
            unsafe { op.store(acc2.reshape(ow), [kv_lin, split, 2i32]) };
            unsafe { op.store(acc3.reshape(ow), [kv_lin, split, 3i32]) };
        }

        #[wax::entry()]
        pub fn sdpa_sd_d128_p2<E: ElementType, const SPLITS: i32, const R: i32>(
            partial_m: &Tensor<f32, { [-1, 32] }>,
            partial_l: &Tensor<f32, { [-1, 32] }>,
            partial_o: &Tensor<f32, { [-1, 128] }>,
            partial_m_t: &Tensor<f32, { [-1, 32, 32] }>,
            out: &mut Tensor<E, { [-1, 128] }>,
        ) {
            let chunk = const_shape![1, 32];
            let wide = const_shape![32, 32];
            let head = get_tile_block_id().0;
            let row0 = head * R;

            let pm = partial_m.partition(chunk);
            let pl = partial_l.partition(chunk);
            let mut mmax: Tile<f32, { [1, 32] }> = constant(-3.4028235e38f32, chunk);
            for b in 0i32..R {
                let mb: Tile<f32, { [1, 32] }> = pm.load([row0 + b, 0i32]);
                mmax = maxf(mmax, mb, nan::Disabled, ftz::Disabled);
            }
            let gm1: Tile<f32, { [1] }> = reduce_max(mmax, 1i32);
            let gm: Tile<f32, { [1, 32] }> = gm1.reshape(const_shape![1, 1]).broadcast(chunk);

            let mut lsum: Tile<f32, { [1, 32] }> = constant(0.0f32, chunk);
            for b in 0i32..R {
                let mb: Tile<f32, { [1, 32] }> = pm.load([row0 + b, 0i32]);
                let lb: Tile<f32, { [1, 32] }> = pl.load([row0 + b, 0i32]);
                lsum = lsum + lb * exp(mb - gm);
            }
            let gl1: Tile<f32, { [1] }> = reduce_sum(lsum, 1i32);
            let gl: Tile<f32, { [1, 32] }> = gl1.reshape(const_shape![1, 1]).broadcast(chunk);

            // Row s folds the splits {b*32 + s}: the same work group `sg` used to do.
            let mt = const_shape![32, 1, 32];
            let pmt = partial_m_t.partition(mt);
            let pop = partial_o.partition(wide);
            let gmw: Tile<f32, { [32, 32] }> = gm.broadcast(wide);
            let mut acc0: Tile<f32, { [32, 32] }> = constant(0.0f32, wide);
            let mut acc1: Tile<f32, { [32, 32] }> = constant(0.0f32, wide);
            let mut acc2: Tile<f32, { [32, 32] }> = constant(0.0f32, wide);
            let mut acc3: Tile<f32, { [32, 32] }> = constant(0.0f32, wide);
            for b in 0i32..R {
                let ms: Tile<f32, { [32, 32] }> = pmt.load([row0 + b, 0i32, 0i32]).reshape(wide);
                let f: Tile<f32, { [32, 32] }> = exp(ms - gmw);
                // `pop.load([row0 + b, c])`: a `[32,32]` tile steps `i0` by 32*128, and
                // `head*SPLITS + b*32` is `(head*R + b)*32` because SPLITS = R*32 - so the
                // split index folds into the row index with nothing left over.
                acc0 = acc0 + pop.load([row0 + b, 0i32]) * f;
                acc1 = acc1 + pop.load([row0 + b, 1i32]) * f;
                acc2 = acc2 + pop.load([row0 + b, 2i32]) * f;
                acc3 = acc3 + pop.load([row0 + b, 3i32]) * f;
            }
            let n0: Tile<f32, { [32] }> = reduce_sum(acc0, 0i32);
            let n1: Tile<f32, { [32] }> = reduce_sum(acc1, 0i32);
            let n2: Tile<f32, { [32] }> = reduce_sum(acc2, 0i32);
            let n3: Tile<f32, { [32] }> = reduce_sum(acc3, 0i32);
            let mut op: PartitionMut<E, { [1, 32] }> = unsafe { out.partition_full_mut(chunk) };
            let o0: Tile<E, { [1, 32] }> = convert_tile(n0.reshape(chunk) / gl);
            let o1: Tile<E, { [1, 32] }> = convert_tile(n1.reshape(chunk) / gl);
            let o2: Tile<E, { [1, 32] }> = convert_tile(n2.reshape(chunk) / gl);
            let o3: Tile<E, { [1, 32] }> = convert_tile(n3.reshape(chunk) / gl);
            unsafe { op.store(o0, [head, 0i32]) };
            unsafe { op.store(o1, [head, 1i32]) };
            unsafe { op.store(o2, [head, 2i32]) };
            unsafe { op.store(o3, [head, 3i32]) };
        }

        #[wax::entry()]
        pub fn copy2d<E: ElementType, const SG: i32>(
            src: &Tensor<E, { [-1, -1, 32] }>,
            meta: &Tensor<i32, { [-1, 32] }>,
            out: &mut Tensor<E, { [-1, -1, 32] }>,
        ) {
            let wide = const_shape![1, SG, 32];
            let one = const_shape![1, 1];
            let cb = get_tile_block_id().0;
            let row = get_tile_block_id().1;
            let sbr: i32 = tile_to_scalar(load_tile(meta, one, [0i32, 0i32]).reshape(const_shape![]));
            let dbr: i32 = tile_to_scalar(load_tile(meta, one, [1i32, 0i32]).reshape(const_shape![]));
            let cpr: i32 = tile_to_scalar(load_tile(meta, one, [2i32, 0i32]).reshape(const_shape![]));
            let c0 = min(cb * SG, cpr - SG);
            let t: Tile<E, { [1, SG, 32] }> = src.partition(wide).load([row * sbr + c0, 0i32, 0i32]);
            let mut op: PartitionMut<E, { [1, SG, 32] }> = unsafe { out.partition_full_mut(wide) };
            unsafe { op.store(t, [row * dbr + c0, 0i32, 0i32]) };
        }

        #[wax::entry()]
        pub fn wide_carry<const N: i32, const KB: i32>(
            out: &mut Tensor<f32, { [8, N] }>,
            x: &Tensor<f32, { [-1, N] }>,
        ) {
            let ts = const_shape![8, 32];
            let blk = get_tile_block_id().0;
            let xp = x.partition(ts);
            let mut acc: Tile<f32, { [8, 32] }> = constant(0.0f32, ts);
            for i in 0i32..KB {
                acc = acc + xp.load([i, blk]);
            }
            let mut op: PartitionMut<f32, { [8, 32] }> = unsafe { out.partition_full_mut(ts) };
            unsafe { op.store(acc, [0i32, blk]) };
        }

        #[wax::entry()]
        pub fn uscale<const N: i32>(
            x: &Tensor<f32, { [-1, N] }>,
            meta: &Tensor<f32, { [-1] }>,
            out: &mut Tensor<f32, { [1, N] }>,
        ) {
            let ts = const_shape![1, 32];
            let row = get_tile_block_id().0;
            let mp = meta.partition(const_shape![1]);
            let s: f32 = tile_to_scalar(mp.load([0i32]).reshape(const_shape![]));
            let sb: Tile<f32, { [1, 32] }> = broadcast_scalar(s, ts);
            let nchunks = N / 32i32;
            let xp = x.partition(ts);
            let mut op: PartitionMut<f32, { [1, 32] }> = unsafe { out.partition_full_mut(ts) };
            for c in 0i32..nchunks {
                let t: Tile<f32, { [1, 32] }> = xp.load([row, c]);
                unsafe { op.store(t * sb, [row, c]) };
            }
        }

        #[wax::entry()]
        pub fn atan2_test<const D: i32>(
            y_in: &Tensor<f32, { [-1, D] }>,
            x_in: &Tensor<f32, { [-1, D] }>,
            output: &mut Tensor<f32, { [-1, D] }>,
        ) {
            let tile_shape = const_shape![1, D];
            let pid = get_tile_block_id();
            let row = pid.0;
            let y_part = y_in.partition(tile_shape);
            let x_part = x_in.partition(tile_shape);
            let ty = y_part.load([row, 0i32]);
            let tx = x_part.load([row, 0i32]);
            let out_tile = atan2(ty, tx);
            let mut out_part: PartitionMut<f32, { [1, D] }> =
                unsafe { output.partition_full_mut(tile_shape) };
            unsafe { out_part.store(out_tile, [row, 0i32]) };
        }

        #[wax::entry()]
        pub fn reduce_sum_test<const D: i32>(
            input: &Tensor<f32, { [-1, D] }>,
            output: &mut Tensor<f32, { [-1, D] }>,
        ) {
            let tile_shape = const_shape![1, D];
            let pid = get_tile_block_id();
            let row = pid.0;
            let in_part = input.partition(tile_shape);
            let tile = in_part.load([row, 0i32]);
            let sum: Tile<f32, { [1] }> = reduce_sum(tile, 1i32);
            let sum_bc: Tile<f32, { [1, D] }> = sum.reshape(const_shape![1, 1]).broadcast(tile_shape);
            let mut out_part: PartitionMut<f32, { [1, D] }> =
                unsafe { output.partition_full_mut(tile_shape) };
            unsafe { out_part.store(sum_bc, [row, 0i32]) };
        }

        #[wax::entry()]
        pub fn gemv_silu<E: ElementType, const BK: i32, const K: i32>(
            a: &Tensor<E, { [-1, K] }>,
            vec: &Tensor<E, { [-1, K] }>,
            out: &mut Tensor<E, { [-1, 1] }>,
        ) {
            let ts = const_shape![1, BK];
            let row = get_tile_block_id().0;
            let ap = a.partition(ts);
            let vp = vec.partition(ts);
            let nc = K / BK;
            let a0: Tile<f32, { [1, BK] }> = convert_tile(ap.load([row, 0i32]));
            let v0: Tile<f32, { [1, BK] }> = convert_tile(vp.load([0i32, 0i32]));
            let mut acc = a0 * v0;
            for i in 1i32..nc {
                let ai: Tile<f32, { [1, BK] }> = convert_tile(ap.load([row, i]));
                let vi: Tile<f32, { [1, BK] }> = convert_tile(vp.load([0i32, i]));
                acc = acc + ai * vi;
            }
            let dot: Tile<f32, { [1] }> = reduce_sum(acc, 1i32);
            let out_ts = const_shape![1, 1];
            let d: Tile<f32, { [1, 1] }> = dot.reshape(out_ts);
            let one: Tile<f32, { [1, 1] }> = constant(1.0f32, out_ts);
            let zero: Tile<f32, { [1, 1] }> = constant(0.0f32, out_ts);
            let y: Tile<E, { [1, 1] }> = convert_tile(d * (one / (one + exp(zero - d))));
            let mut op: PartitionMut<E, { [1, 1] }> = unsafe { out.partition_full_mut(out_ts) };
            unsafe { op.store(y, [row, 0i32]) };
        }

        #[wax::entry()]
        pub fn silu_vec<E: ElementType, const BK: i32>(
            inp: &Tensor<E, { [-1] }>,
            out: &mut Tensor<E, { [-1] }>,
        ) {
            let ts = const_shape![BK];
            let b = get_tile_block_id().0;
            let x: Tile<f32, { [BK] }> = convert_tile(inp.partition(ts).load([b]));
            let one: Tile<f32, { [BK] }> = constant(1.0f32, ts);
            let zero: Tile<f32, { [BK] }> = constant(0.0f32, ts);
            let y: Tile<E, { [BK] }> = convert_tile(x * (one / (one + exp(zero - x))));
            let mut op: PartitionMut<E, { [BK] }> = unsafe { out.partition_full_mut(ts) };
            unsafe { op.store(y, [b]) };
        }

            #[wax::entry()]
            pub fn row_sum<E: ElementType, const N: i32, const BS: i32>(
                x: &Tensor<E, { [-1, N] }>,
                out: &mut Tensor<E, { [-1, N] }>,
            ) {
                let ts = const_shape![1, BS];
                let row = get_tile_block_id().0;
                let nc = N / BS;
                let xp = x.partition(ts);
                let mut acc: Tile<E, { [1, BS] }> = xp.load([row, 0i32]);
                for j in 1i32..nc {
                    acc = acc + xp.load([row, j]);
                }
                let s: Tile<E, { [1] }> = reduce_sum(acc, 1i32);
                let bc: Tile<E, { [1, BS] }> = s.reshape(const_shape![1, 1]).broadcast(ts);
                let mut op: PartitionMut<E, { [1, BS] }> = unsafe { out.partition_full_mut(ts) };
                for j in 0i32..nc {
                    unsafe { op.store(bc, [row, j]) };
                }
            }

        #[wax::entry()]
        pub fn reduce_max_test<const D: i32>(
            input: &Tensor<f32, { [-1, D] }>,
            output: &mut Tensor<f32, { [-1, D] }>,
        ) {
            let tile_shape = const_shape![1, D];
            let pid = get_tile_block_id();
            let row = pid.0;
            let in_part = input.partition(tile_shape);
            let tile = in_part.load([row, 0i32]);
            let mx: Tile<f32, { [1] }> = reduce_max(tile, 1i32);
            let mx_bc: Tile<f32, { [1, D] }> = mx.reshape(const_shape![1, 1]).broadcast(tile_shape);
            let mut out_part: PartitionMut<f32, { [1, D] }> =
                unsafe { output.partition_full_mut(tile_shape) };
            unsafe { out_part.store(mx_bc, [row, 0i32]) };
        }

        #[wax::entry()]
        pub fn reduce_min_test<const D: i32>(
            input: &Tensor<f32, { [-1, D] }>,
            output: &mut Tensor<f32, { [-1, D] }>,
        ) {
            let tile_shape = const_shape![1, D];
            let pid = get_tile_block_id();
            let row = pid.0;
            let in_part = input.partition(tile_shape);
            let tile = in_part.load([row, 0i32]);
            let mn: Tile<f32, { [1] }> = reduce_min(tile, 1i32);
            let mn_bc: Tile<f32, { [1, D] }> = mn.reshape(const_shape![1, 1]).broadcast(tile_shape);
            let mut out_part: PartitionMut<f32, { [1, D] }> =
                unsafe { output.partition_full_mut(tile_shape) };
            unsafe { out_part.store(mn_bc, [row, 0i32]) };
        }

        #[wax::entry()]
        pub fn convert_i32_i64_test<const D: i32>(
            input: &Tensor<f32, { [-1, D] }>,
            output: &mut Tensor<f32, { [-1, D] }>,
        ) {
            let tile_shape = const_shape![1, D];
            let pid = get_tile_block_id();
            let row: i32 = pid.0;
            let row_i64: i64 = convert_scalar(row);
            let row_i32: i32 = convert_scalar(row_i64);
            let in_part = input.partition(tile_shape);
            let tile = in_part.load([row_i32, 0i32]);
            let mut out_part: PartitionMut<f32, { [1, D] }> =
                unsafe { output.partition_full_mut(tile_shape) };
            unsafe { out_part.store(tile, [row_i32, 0i32]) };
        }

        #[wax::entry()]
        pub fn loop_sum_test<const D: i32, const N: i32>(
            input: &Tensor<f32, { [-1, D] }>,
            output: &mut Tensor<f32, { [-1, D] }>,
        ) {
            let tile_shape = const_shape![1, D];
            let pid = get_tile_block_id();
            let row = pid.0;
            let in_part = input.partition(tile_shape);
            let mut acc: Tile<f32, { [1, D] }> = constant(0.0f32, tile_shape);
            for j in 0i32..N {
                let t = in_part.load([row * N + j, 0i32]);
                acc = acc + t;
            }
            let mut out_part: PartitionMut<f32, { [1, D] }> =
                unsafe { output.partition_full_mut(tile_shape) };
            unsafe { out_part.store(acc, [row, 0i32]) };
        }

        #[wax::entry()]
        pub fn gemv_test<const BK: i32, const K: i32>(
            a: &Tensor<f32, { [-1, K] }>,
            vec: &Tensor<f32, { [-1, K] }>,
            output: &mut Tensor<f32, { [-1, BK] }>,
        ) {
            let tile_shape = const_shape![1, BK];
            let pid = get_tile_block_id();
            let row = pid.0;
            let a_part = a.partition(tile_shape);
            let v_part = vec.partition(tile_shape);
            let num_chunks = K / BK;
            let mut acc: Tile<f32, { [1, BK] }> = constant(0.0f32, tile_shape);
            for i in 0i32..num_chunks {
                let ac = a_part.load([row, i]);
                let vc = v_part.load([0i32, i]);
                acc = acc + ac * vc;
            }
            let dot: Tile<f32, { [1] }> = reduce_sum(acc, 1i32);
            let dot_bc: Tile<f32, { [1, BK] }> = dot.reshape(const_shape![1, 1]).broadcast(tile_shape);
            let mut out_part: PartitionMut<f32, { [1, BK] }> =
                unsafe { output.partition_full_mut(tile_shape) };
            unsafe { out_part.store(dot_bc, [row, 0i32]) };
        }

        #[wax::entry()]
        pub fn flash_attn_decode<const D: i32, const N: i32>(
            q: &Tensor<f32, { [-1, D] }>,
            k: &Tensor<f32, { [-1, D] }>,
            v: &Tensor<f32, { [-1, D] }>,
            out: &mut Tensor<f32, { [-1, D] }>,
        ) {
            let tile_shape = const_shape![1, D];
            let pid = get_tile_block_id();
            let q_block = pid.0;

            let q_part = q.partition(tile_shape);
            let k_part = k.partition(tile_shape);
            let v_part = v.partition(tile_shape);

            // Q stays in registers for the whole KV loop.
            let tq = q_part.load([q_block, 0i32]);

            // Online softmax state (per-SIMD-group scalars via [1] tiles).
            let mut m_i: Tile<f32, { [1] }> = constant(-3.4028235e38f32, const_shape![1]);
            let mut l_i: Tile<f32, { [1] }> = constant(0.0f32, const_shape![1]);
            // Running weighted-value accumulator.
            let mut acc: Tile<f32, { [1, D] }> = constant(0.0f32, tile_shape);

            for j in 0i32..N {
                let tk = k_part.load([j, 0i32]);

                // Q - K[j]^T  --  each thread multiplies its own element, then
                // air.simd_sum (+ hierarchical reduce for D > 32) gives the scalar.
                let dot: Tile<f32, { [1] }> = reduce_sum(tq * tk, 1i32);

                // Online softmax max update.
                let m_ij: Tile<f32, { [1] }> = maxf(m_i, dot, nan::Disabled, ftz::Disabled);

                // Correction factor  alpha = exp(m_i - m_ij).
                let alpha: Tile<f32, { [1] }> = exp(m_i - m_ij);

                // Attention weight for this key  p = exp(dot - m_ij).
                let p_val: Tile<f32, { [1] }> = exp(dot - m_ij);

                // l_i <- l_i - alpha + p
                l_i = l_i * alpha + p_val;

                // Broadcast scalars -> [1, D] for the accumulator update.
                let alpha_bc: Tile<f32, { [1, D] }> =
                    alpha.reshape(const_shape![1, 1]).broadcast(tile_shape);
                let p_bc: Tile<f32, { [1, D] }> =
                    p_val.reshape(const_shape![1, 1]).broadcast(tile_shape);

                let tv = v_part.load([j, 0i32]);
                acc = acc * alpha_bc + tv * p_bc;

                m_i = m_ij;
            }

            // Normalize: acc / l_i
            let l_bc: Tile<f32, { [1, D] }> = l_i.reshape(const_shape![1, 1]).broadcast(tile_shape);
            acc = acc / l_bc;

            let mut out_part: PartitionMut<f32, { [1, D] }> =
                unsafe { out.partition_full_mut(tile_shape) };
            unsafe { out_part.store(acc, [q_block, 0i32]) };
        }

        #[wax::entry()]
        pub fn ragged_gemm<const M: i32, const N: i32, const K: i32, const KB: i32>(
            z: &mut Tensor<f32, { [M, N] }>,
            x: &Tensor<f32, { [M, K] }>,
            y: &Tensor<f32, { [K, N] }>,
        ) {
            let ts = const_shape![8, 8];
            let rb = get_tile_block_id().0;
            let cb = get_tile_block_id().1;
            let xtok: Token = get_tensor_token(x);
            let xp: Partition<f32, { [8, 8] }> =
                make_partition_view(x, ts, padding::Zero, dim_map::Identity, xtok);
            let ytok: Token = get_tensor_token(y);
            let yp: Partition<f32, { [8, 8] }> =
                make_partition_view(y, ts, padding::Zero, dim_map::Identity, ytok);
            let mut acc: Tile<f32, { [8, 8] }> = constant(0.0f32, ts);
            for i in 0i32..KB {
                acc = mmaf(xp.load([rb, i]), yp.load([i, cb]), acc);
            }
            let ztok: Token = get_tensor_token(z);
            let mut zp: PartitionMut<f32, { [8, 8] }> =
                unsafe { make_partition_view_mut(z, ts, padding::Zero, ztok) };
            unsafe { zp.store(acc, [rb, cb]) };
        }

        #[wax::entry()]
        pub fn frag_store(
            a: &Tensor<f32, { [-1, 8] }>,
            id: &Tensor<f32, { [-1, 8] }>,
            out: &mut Tensor<f32, { [-1, 8] }>,
        ) {
            let ts = const_shape![8, 8];
            let rb = get_tile_block_id().0;
            let cb = get_tile_block_id().1;
            // `padding::None` on both inputs: `a` is deliberately allocated out to the full
            // fragment grid and `id` is a single tile, so every read IS in bounds. It has to be
            // said explicitly, because `partition()` declares `padding::Zero` by default and a
            // padded fragment load inside a kernel that masks its store now gets a real bound -
            // promoted to a kernel parameter, which would change this kernel's ABI.
            let atok: Token = get_tensor_token(a);
            let ap: Partition<f32, { [8, 8] }> =
                make_partition_view(a, ts, padding::None, dim_map::Identity, atok);
            let av: Tile<f32, { [8, 8] }> = ap.load([rb, cb]);
            let itok: Token = get_tensor_token(id);
            let ip: Partition<f32, { [8, 8] }> =
                make_partition_view(id, ts, padding::None, dim_map::Identity, itok);
            let idv: Tile<f32, { [8, 8] }> = ip.load([0i32, 0i32]);
            let zero: Tile<f32, { [8, 8] }> = constant(0.0f32, ts);
            let acc: Tile<f32, { [8, 8] }> = mmaf(av, idv, zero);
            let otok: Token = get_tensor_token(out);
            let mut op: PartitionMut<f32, { [8, 8] }> =
                unsafe { make_partition_view_mut(out, ts, padding::Zero, otok) };
            unsafe { op.store(acc, [rb, cb]) };
        }

        #[wax::entry()]
        pub fn three_addrs<const BK: i32>(
            src: &Tensor<f32, { [-1] }>,
            addr: &Tensor<i64, { [-1] }>,
            idx: &Tensor<i32, { [-1] }>,
            out: &mut Tensor<f32, { [-1] }>,
        ) {
            let ts = const_shape![BK];
            let one = const_shape![1];
            let blk = get_tile_block_id().0;
            let ap = addr.partition(one);
            let o: Tile<i32, { [BK] }> = idx.partition(ts).load([blk]);
            let o1: Tile<i32, { [BK] }> = idx.partition(ts).load([blk + 1i32]);
            let o2: Tile<i32, { [BK] }> = idx.partition(ts).load([blk + 2i32]);

            let a0: Tile<i64, { [1] }> = ap.load([0i32]);
            let b0: PointerTile<*mut f32, { [1] }> = int_to_ptr(a0);
            let w0: PointerTile<*mut f32, { [BK] }> = broadcast_ptr(b0, ts);
            let p0: PointerTile<*mut f32, { [BK] }> = addptr_tile(w0, o);
            let (v0, _t0): (Tile<f32, { [BK] }>, Token) = load_ptr_tko(
                p0,
                ordering::Weak,
                None::<scope::Device>,
                None,
                None,
                None,
                Latency::<0>,
            );

            let a1: Tile<i64, { [1] }> = ap.load([1i32]);
            let b1: PointerTile<*mut f32, { [1] }> = int_to_ptr(a1);
            let w1: PointerTile<*mut f32, { [BK] }> = broadcast_ptr(b1, ts);
            let p1: PointerTile<*mut f32, { [BK] }> = addptr_tile(w1, o1);
            let (v1, _t1): (Tile<f32, { [BK] }>, Token) = load_ptr_tko(
                p1,
                ordering::Weak,
                None::<scope::Device>,
                None,
                None,
                None,
                Latency::<0>,
            );

            let a2: Tile<i64, { [1] }> = ap.load([2i32]);
            let b2: PointerTile<*mut f32, { [1] }> = int_to_ptr(a2);
            let w2: PointerTile<*mut f32, { [BK] }> = broadcast_ptr(b2, ts);
            let p2: PointerTile<*mut f32, { [BK] }> = addptr_tile(w2, o2);
            let (v2, _t2): (Tile<f32, { [BK] }>, Token) = load_ptr_tko(
                p2,
                ordering::Weak,
                None::<scope::Device>,
                None,
                None,
                None,
                Latency::<0>,
            );

            let s: Tile<f32, { [BK] }> = v0 + v1 + v2;
            let mut op: PartitionMut<f32, { [BK] }> = unsafe { out.partition_full_mut(ts) };
            unsafe { op.store(s, [blk]) };
        }

        #[wax::entry()]
        pub fn one_addr<const BK: i32>(
            src: &Tensor<f32, { [-1] }>,
            addr: &Tensor<i64, { [-1] }>,
            idx: &Tensor<i32, { [-1] }>,
            out: &mut Tensor<f32, { [-1] }>,
        ) {
            let ts = const_shape![BK];
            let one = const_shape![1];
            let blk = get_tile_block_id().0;
            let o: Tile<i32, { [BK] }> = idx.partition(ts).load([blk]);
            let o1: Tile<i32, { [BK] }> = idx.partition(ts).load([blk + 1i32]);
            let o2: Tile<i32, { [BK] }> = idx.partition(ts).load([blk + 2i32]);
            let a0: Tile<i64, { [1] }> = addr.partition(one).load([0i32]);
            let b0: PointerTile<*mut f32, { [1] }> = int_to_ptr(a0);
            let base: PointerTile<*mut f32, { [BK] }> = broadcast_ptr(b0, ts);
            let q0: PointerTile<*mut f32, { [BK] }> = addptr_tile(base, o);
            let q1: PointerTile<*mut f32, { [BK] }> = addptr_tile(base, o1);
            let q2: PointerTile<*mut f32, { [BK] }> = addptr_tile(base, o2);

            let (v0, _t0): (Tile<f32, { [BK] }>, Token) = load_ptr_tko(
                q0,
                ordering::Weak,
                None::<scope::Device>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let (v1, _t1): (Tile<f32, { [BK] }>, Token) = load_ptr_tko(
                q1,
                ordering::Weak,
                None::<scope::Device>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let (v2, _t2): (Tile<f32, { [BK] }>, Token) = load_ptr_tko(
                q2,
                ordering::Weak,
                None::<scope::Device>,
                None,
                None,
                None,
                Latency::<0>,
            );

            let s: Tile<f32, { [BK] }> = v0 + v1 + v2;
            let mut op: PartitionMut<f32, { [BK] }> = unsafe { out.partition_full_mut(ts) };
            unsafe { op.store(s, [blk]) };
        }

        #[wax::entry()]
        pub fn gemm_bf16_8x8(
            z: &mut Tensor<f32, { [8, 8] }>,
            x: &Tensor<bf16, { [8, 8] }>,
            y: &Tensor<bf16, { [8, 8] }>,
        ) {
            let tile_x = x.partition(const_shape![8, 8]).load([0i32, 0i32]);
            let tile_y = y.partition(const_shape![8, 8]).load([0i32, 0i32]);
            let acc = constant(0.0f32, const_shape![8, 8]);
            let tile_z = mmaf(tile_x, tile_y, acc);
            store_current_block_2d(z, tile_z);
        }

        #[wax::entry()]
        pub fn gemm_bf16_16x16(
            z: &mut Tensor<f32, { [16, 16] }>,
            x: &Tensor<bf16, { [16, 16] }>,
            y: &Tensor<bf16, { [16, 16] }>,
        ) {
            let tile_x = x.partition(const_shape![16, 16]).load([0i32, 0i32]);
            let tile_y = y.partition(const_shape![16, 16]).load([0i32, 0i32]);
            let acc = constant(0.0f32, const_shape![16, 16]);
            let tile_z = mmaf(tile_x, tile_y, acc);
            store_current_block_2d(z, tile_z);
        }

        #[wax::entry()]
        pub fn gemm_eout<E: ElementType, const BM: i32, const BN: i32, const BK: i32, const K: i32>(
            z: &mut Tensor<E, { [BM, BN] }>,
            x: &Tensor<E, { [-1, K] }>,
            y: &Tensor<E, { [K, -1] }>,
        ) {
            let part_x = x.partition(const_shape![BM, BK]);
            let part_y = y.partition(const_shape![BK, BN]);
            let pid: (i32, i32, i32) = get_tile_block_id();
            // Explicit f32: the frontend infers `mmaf`'s accumulator element from the
            // inputs (f16/bf16) when the type is left open, but the SGMMA register
            // - and the POC's reference lowering - accumulate in f32.
            let mut tile_z: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            for i in 0i32..(K / BK) {
                let tile_x = part_x.load([pid.0, i]);
                let tile_y = part_y.load([i, pid.1]);
                tile_z = mmaf(tile_x, tile_y, tile_z);
            }
            store_narrow_2d(z, tile_z);
        }

        #[wax::entry()]
        pub fn gemm_f16_8x8(
            z: &mut Tensor<f32, { [8, 8] }>,
            x: &Tensor<f16, { [8, 8] }>,
            y: &Tensor<f16, { [8, 8] }>,
        ) {
            let tile_x = x.partition(const_shape![8, 8]).load([0i32, 0i32]);
            let tile_y = y.partition(const_shape![8, 8]).load([0i32, 0i32]);
            let acc = constant(0.0f32, const_shape![8, 8]);
            let tile_z = mmaf(tile_x, tile_y, acc);
            store_current_block_2d(z, tile_z);
        }

        #[wax::entry()]
        pub fn gemm_f16_16x16(
            z: &mut Tensor<f32, { [16, 16] }>,
            x: &Tensor<f16, { [16, 16] }>,
            y: &Tensor<f16, { [16, 16] }>,
        ) {
            let tile_x = x.partition(const_shape![16, 16]).load([0i32, 0i32]);
            let tile_y = y.partition(const_shape![16, 16]).load([0i32, 0i32]);
            let acc = constant(0.0f32, const_shape![16, 16]);
            let tile_z = mmaf(tile_x, tile_y, acc);
            store_current_block_2d(z, tile_z);
        }

        #[wax::entry()]
        pub fn dual_path(x: &Tensor<f32, { [2, 4] }>, z: &mut Tensor<f32, { [2, 4] }>) {
            let x_part = x.partition(const_shape![1, 4]);
            let mut z_part = unsafe { z.partition_full_mut(const_shape![1, 4]) };
            for i in 0i32..2i32 {
                let tile = x_part.load([i, 0i32]);
                let result = if i == 0i32 {
                    tile + constant(10.0f32, const_shape![1, 4])
                } else {
                    tile + constant(20.0f32, const_shape![1, 4])
                };
                unsafe { z_part.store(result, [i, 0i32]) };
            }
        }

        #[wax::entry()]
        pub fn int_copy(x: &Tensor<i32, { [1, 32] }>, out: &mut Tensor<i32, { [1, 32] }>) {
            let ts = const_shape![1, 32];
            let t: Tile<i32, { [1, 32] }> = x.partition(ts).load([0i32, 0i32]);
            let mut op: PartitionMut<i32, { [1, 32] }> = unsafe { out.partition_full_mut(ts) };
            unsafe { op.store(t, [0i32, 0i32]) };
        }

        #[wax::entry()]
        pub fn index_select<const D: i32>(
            x: &Tensor<f32, { [-1, D] }>,
            ids: &Tensor<i32, { [-1, 32] }>,
            out: &mut Tensor<f32, { [1, D] }>,
        ) {
            let ts = const_shape![1, 32];
            let row = get_tile_block_id().0;
            let idt: Tile<i32, { [1, 32] }> = load_tile(ids, ts, [row, 0i32]);
            let idf: Tile<f32, { [1, 32] }> = convert_tile(idt);
            let idr: Tile<f32, { [1] }> = reduce_max(idf, 1i32);
            let id_f: f32 = tile_to_scalar(idr.reshape(const_shape![]));
            let id: i32 = convert_scalar(id_f);
            let nchunks = D / 32i32;
            let mut op: PartitionMut<f32, { [1, 32] }> = unsafe { out.partition_full_mut(ts) };
            for c in 0i32..nchunks {
                let t: Tile<f32, { [1, 32] }> = load_tile(x, ts, [id, c]);
                unsafe { op.store(t, [row, c]) };
            }
        }

        #[wax::entry()]
        pub fn where_bcast(
            cond: &Tensor<i32, { [-1] }>,
            t: &Tensor<f32, { [-1] }>,
            f: &Tensor<f32, { [-1] }>,
            addr: &Tensor<i64, { [-1] }>,
            meta: &Tensor<i32, { [-1] }>,
            out: &mut Tensor<f32, { [-1] }>,
        ) {
            let ts = const_shape![32];
            let one = const_shape![1];
            let blk = get_tile_block_id().0;
            let mp = meta.partition(ts);
            let cp: Tile<i32, { [32] }> = mp.load([0i32]);
            let tp: Tile<i32, { [32] }> = mp.load([1i32]);
            let fp: Tile<i32, { [32] }> = mp.load([2i32]);
            let ap = addr.partition(one);
            let g: Tile<i32, { [32] }> = iota(ts) + broadcast_scalar(blk * 32i32, ts);

            let ca: Tile<i64, { [1] }> = ap.load([0i32]);
            let cbase: PointerTile<*mut i32, { [1] }> = int_to_ptr(ca);
            let cb32: PointerTile<*mut i32, { [32] }> = broadcast_ptr(cbase, ts);
            let cptr: PointerTile<*mut i32, { [32] }> = addptr_tile(cb32, g % cp);
            let (c, _ct): (Tile<i32, { [32] }>, Token) = load_ptr_tko(
                cptr,
                ordering::Weak,
                None::<scope::Device>,
                None,
                None,
                None,
                Latency::<0>,
            );

            let ta: Tile<i64, { [1] }> = ap.load([1i32]);
            let tbase: PointerTile<*mut f32, { [1] }> = int_to_ptr(ta);
            let tb32: PointerTile<*mut f32, { [32] }> = broadcast_ptr(tbase, ts);
            let tptr: PointerTile<*mut f32, { [32] }> = addptr_tile(tb32, g % tp);
            let (tv, _tt): (Tile<f32, { [32] }>, Token) = load_ptr_tko(
                tptr,
                ordering::Weak,
                None::<scope::Device>,
                None,
                None,
                None,
                Latency::<0>,
            );

            let fa: Tile<i64, { [1] }> = ap.load([2i32]);
            let fbase: PointerTile<*mut f32, { [1] }> = int_to_ptr(fa);
            let fb32: PointerTile<*mut f32, { [32] }> = broadcast_ptr(fbase, ts);
            let fptr: PointerTile<*mut f32, { [32] }> = addptr_tile(fb32, g % fp);
            let (fv, _ft): (Tile<f32, { [32] }>, Token) = load_ptr_tko(
                fptr,
                ordering::Weak,
                None::<scope::Device>,
                None,
                None,
                None,
                Latency::<0>,
            );

            let zero: Tile<i32, { [32] }> = broadcast_scalar(0i32, ts);
            let mask: Tile<bool, { [32] }> = cmpi(c, zero, predicate::NotEqual);
            let z: Tile<f32, { [32] }> = select(mask, tv, fv);
            let mut op: PartitionMut<f32, { [32] }> = unsafe { out.partition_full_mut(ts) };
            unsafe { op.store(z, [blk]) };
        }

        #[wax::entry()]
        pub fn gather1d(
            x: &Tensor<f32, { [-1] }>,
            addr: &Tensor<i64, { [-1] }>,
            off: &Tensor<i32, { [-1] }>,
            out: &mut Tensor<f32, { [-1] }>,
        ) {
            let ts = const_shape![32];
            let one = const_shape![1];
            let blk = get_tile_block_id().0;
            let o: Tile<i32, { [32] }> = off.partition(ts).load([blk]);
            let a: Tile<i64, { [1] }> = addr.partition(one).load([0i32]);
            let base: PointerTile<*mut f32, { [1] }> = int_to_ptr(a);
            let b32: PointerTile<*mut f32, { [32] }> = broadcast_ptr(base, ts);
            let ptrs: PointerTile<*mut f32, { [32] }> = addptr_tile(b32, o);
            let (g, _tok): (Tile<f32, { [32] }>, Token) = load_ptr_tko(
                ptrs,
                ordering::Weak,
                None::<scope::Device>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let mut op: PartitionMut<f32, { [32] }> = unsafe { out.partition_full_mut(ts) };
            unsafe { op.store(g, [blk]) };
        }

        #[wax::entry()]
        pub fn rows_per_simd_group<const SG: i32>(
            inp: &Tensor<f32, { [-1, 32] }>,
            out: &mut Tensor<f32, { [-1, 32] }>,
        ) {
            // Deliberately a `[1,32]` tile: this kernel exists to pin the `tp.x & 31` lane
            // term that shape produces, so it must NOT become a threadgroup-spanning tile.
            // The row index comes from the BLOCK instead of the SIMD group.
            let ts = const_shape![1, 32];
            let blk = get_tile_block_id().0;
            let ip = inp.partition(ts);
            let row: Tile<f32, { [1, 32] }> = ip.load([blk, 0i32]);
            let mut op: PartitionMut<f32, { [1, 32] }> = unsafe { out.partition_full_mut(ts) };
            unsafe { op.store(row * constant(2.0f32, ts), [blk, 0i32]) };
        }

        #[wax::entry()]
        pub fn rows_via_tile_axis<const SG: i32>(
            inp: &Tensor<f32, { [-1, 32] }>,
            out: &mut Tensor<f32, { [-1, 32] }>,
        ) {
            let wide = const_shape![SG, 32];
            let rows: Tile<f32, { [SG, 32] }> = inp.partition(wide).load([0i32, 0i32]);
            let mut op: PartitionMut<f32, { [SG, 32] }> = unsafe { out.partition_full_mut(wide) };
            unsafe { op.store(rows * constant(2.0f32, wide), [0i32, 0i32]) };
        }

        #[wax::entry()]
        pub fn iota_within_tile<const SG: i32>(out: &mut Tensor<i32, { [-1, 32] }>) {
            let ts = const_shape![1, 32];
            let wide = const_shape![SG, 32];
            let col1: Tile<i32, { [32] }> = iota(const_shape![32]);
            let col: Tile<i32, { [1, 32] }> = col1.reshape(ts);
            // Every row gets the same 0..31, which is the claim: `iota` is the index WITHIN
            // the tile, so a `[32]` tile restarts per SIMD group.
            let cw: Tile<i32, { [SG, 32] }> = col.broadcast(wide);
            let mut op: PartitionMut<i32, { [SG, 32] }> = unsafe { out.partition_full_mut(wide) };
            unsafe { op.store(cw, [0i32, 0i32]) };
        }

        #[wax::entry()]
        pub fn wide_tile<const SG: i32>(
            winp: &Tensor<f32, { [-1, 64] }>,
            out: &mut Tensor<f32, { [-1, 64] }>,
        ) {
            let ts = const_shape![1, 64];
            let ip = winp.partition(ts);
            let row: Tile<f32, { [1, 64] }> = ip.load([0i32, 0i32]);
            let mut op: PartitionMut<f32, { [1, 64] }> = unsafe { out.partition_full_mut(ts) };
            unsafe { op.store(row, [0i32, 0i32]) };
        }

        #[wax::entry()]
        pub fn uniform_scalar<const SG: i32>(
            scalars: &Tensor<f32, { [-1] }>,
            out: &mut Tensor<f32, { [-1, 32] }>,
        ) {
            let ts = const_shape![1, 32];
            let wide = const_shape![SG, 32];
            let one = const_shape![1];
            let sp = scalars.partition(one);
            // Index 1 of a 2-element buffer - unreachable if the address carried a
            // lane term, since lane 31 would run off the end.
            let v: f32 = tile_to_scalar(sp.load([1i32]).reshape(const_shape![]));
            let vw: Tile<f32, { [SG, 32] }> = broadcast_scalar(v, wide);
            let mut op: PartitionMut<f32, { [SG, 32] }> = unsafe { out.partition_full_mut(wide) };
            unsafe { op.store(vw, [0i32, 0i32]) };
        }

        #[wax::entry()]
        pub fn vcopy_f32(x: &Tensor<f32, { [-1, 32, 4] }>, out: &mut Tensor<f32, { [-1, 32, 4] }>) {
            let ts = const_shape![1, 32, 1];
            let blk = get_tile_block_id().0;
            let xp = x.partition(ts);
            let t0: Tile<f32, { [1, 32, 1] }> = xp.load([blk, 0i32, 0i32]);
            let t1: Tile<f32, { [1, 32, 1] }> = xp.load([blk, 0i32, 1i32]);
            let t2: Tile<f32, { [1, 32, 1] }> = xp.load([blk, 0i32, 2i32]);
            let t3: Tile<f32, { [1, 32, 1] }> = xp.load([blk, 0i32, 3i32]);
            let mut op: PartitionMut<f32, { [1, 32, 1] }> = unsafe { out.partition_full_mut(ts) };
            unsafe { op.store(t0, [blk, 0i32, 0i32]) };
            unsafe { op.store(t1, [blk, 0i32, 1i32]) };
            unsafe { op.store(t2, [blk, 0i32, 2i32]) };
            unsafe { op.store(t3, [blk, 0i32, 3i32]) };
        }

        #[wax::entry()]
        pub fn vcopy_f16(x: &Tensor<f16, { [-1, 32, 8] }>, out: &mut Tensor<f16, { [-1, 32, 8] }>) {
            let ts = const_shape![1, 32, 1];
            let blk = get_tile_block_id().0;
            let xp = x.partition(ts);
            let t0: Tile<f16, { [1, 32, 1] }> = xp.load([blk, 0i32, 0i32]);
            let t1: Tile<f16, { [1, 32, 1] }> = xp.load([blk, 0i32, 1i32]);
            let t2: Tile<f16, { [1, 32, 1] }> = xp.load([blk, 0i32, 2i32]);
            let t3: Tile<f16, { [1, 32, 1] }> = xp.load([blk, 0i32, 3i32]);
            let t4: Tile<f16, { [1, 32, 1] }> = xp.load([blk, 0i32, 4i32]);
            let t5: Tile<f16, { [1, 32, 1] }> = xp.load([blk, 0i32, 5i32]);
            let t6: Tile<f16, { [1, 32, 1] }> = xp.load([blk, 0i32, 6i32]);
            let t7: Tile<f16, { [1, 32, 1] }> = xp.load([blk, 0i32, 7i32]);
            let mut op: PartitionMut<f16, { [1, 32, 1] }> = unsafe { out.partition_full_mut(ts) };
            unsafe { op.store(t0, [blk, 0i32, 0i32]) };
            unsafe { op.store(t1, [blk, 0i32, 1i32]) };
            unsafe { op.store(t2, [blk, 0i32, 2i32]) };
            unsafe { op.store(t3, [blk, 0i32, 3i32]) };
            unsafe { op.store(t4, [blk, 0i32, 4i32]) };
            unsafe { op.store(t5, [blk, 0i32, 5i32]) };
            unsafe { op.store(t6, [blk, 0i32, 6i32]) };
            unsafe { op.store(t7, [blk, 0i32, 7i32]) };
        }

        #[wax::entry()]
        pub fn vcopy_bf16(x: &Tensor<bf16, { [-1, 32, 8] }>, out: &mut Tensor<bf16, { [-1, 32, 8] }>) {
            let ts = const_shape![1, 32, 1];
            let blk = get_tile_block_id().0;
            let xp = x.partition(ts);
            let t0: Tile<bf16, { [1, 32, 1] }> = xp.load([blk, 0i32, 0i32]);
            let t1: Tile<bf16, { [1, 32, 1] }> = xp.load([blk, 0i32, 1i32]);
            let t2: Tile<bf16, { [1, 32, 1] }> = xp.load([blk, 0i32, 2i32]);
            let t3: Tile<bf16, { [1, 32, 1] }> = xp.load([blk, 0i32, 3i32]);
            let t4: Tile<bf16, { [1, 32, 1] }> = xp.load([blk, 0i32, 4i32]);
            let t5: Tile<bf16, { [1, 32, 1] }> = xp.load([blk, 0i32, 5i32]);
            let t6: Tile<bf16, { [1, 32, 1] }> = xp.load([blk, 0i32, 6i32]);
            let t7: Tile<bf16, { [1, 32, 1] }> = xp.load([blk, 0i32, 7i32]);
            let mut op: PartitionMut<bf16, { [1, 32, 1] }> = unsafe { out.partition_full_mut(ts) };
            unsafe { op.store(t0, [blk, 0i32, 0i32]) };
            unsafe { op.store(t1, [blk, 0i32, 1i32]) };
            unsafe { op.store(t2, [blk, 0i32, 2i32]) };
            unsafe { op.store(t3, [blk, 0i32, 3i32]) };
            unsafe { op.store(t4, [blk, 0i32, 4i32]) };
            unsafe { op.store(t5, [blk, 0i32, 5i32]) };
            unsafe { op.store(t6, [blk, 0i32, 6i32]) };
            unsafe { op.store(t7, [blk, 0i32, 7i32]) };
        }

        #[wax::entry()]
        pub fn narrow_copy<const W: i32, const N: i32>(
            out: &mut Tensor<f32, { [-1, W] }>,
            x: &Tensor<f32, { [1, N] }>,
        ) {
            let ts = const_shape![1, W];
            let chunk = get_tile_block_id().0;
            let t: Tile<f32, { [1, W] }> = x.partition(ts).load([0i32, chunk]);
            let mut op: PartitionMut<f32, { [1, W] }> = unsafe { out.partition_full_mut(ts) };
            unsafe { op.store(t, [0i32, chunk]) };
        }

        #[wax::entry()]
        pub fn padded_copy<const M: i32>(x: &Tensor<f32, { [M] }>, out: &mut Tensor<f32, { [-1] }>) {
            let ts = const_shape![32];
            let blk = get_tile_block_id().0;
            let tok: Token = get_tensor_token(x);
            let xp: Partition<f32, { [32] }> =
                make_partition_view(x, ts, padding::Zero, dim_map::Identity, tok);
            let t: Tile<f32, { [32] }> = xp.load([blk]);
            let mut op: PartitionMut<f32, { [32] }> = unsafe { out.partition_full_mut(ts) };
            unsafe { op.store(t, [blk]) };
        }

        #[wax::entry()]
        pub fn padded_store<const M: i32>(x: &Tensor<f32, { [-1] }>, out: &mut Tensor<f32, { [M] }>) {
            let ts = const_shape![32];
            let blk = get_tile_block_id().0;
            let t: Tile<f32, { [32] }> = load_tile(x, ts, [blk]);
            let tok: Token = get_tensor_token(out);
            let mut op: PartitionMut<f32, { [32] }> =
                unsafe { make_partition_view_mut(out, ts, padding::Zero, tok) };
            unsafe { op.store(t, [blk]) };
        }

        #[wax::entry()]
        pub fn unpadded_copy<const M: i32>(x: &Tensor<f32, { [M] }>, out: &mut Tensor<f32, { [-1] }>) {
            let ts = const_shape![32];
            let blk = get_tile_block_id().0;
            let tok: Token = get_tensor_token(x);
            let xp: Partition<f32, { [32] }> =
                make_partition_view(x, ts, padding::None, dim_map::Identity, tok);
            let t: Tile<f32, { [32] }> = xp.load([blk]);
            let mut op: PartitionMut<f32, { [32] }> = unsafe { out.partition_full_mut(ts) };
            unsafe { op.store(t, [blk]) };
        }

        #[wax::entry()]
        pub fn f_sqrt(x: &Tensor<f32, { [256] }>, z: &mut Tensor<f32, { [256] }>) {
            let t = x.partition(const_shape![256]).load([0i32]);
            store_current_block_2d(z, sqrt(t, rounding::NegativeInf, ftz::Disabled));
        }

        #[wax::entry()]
        pub fn f_log(x: &Tensor<f32, { [256] }>, z: &mut Tensor<f32, { [256] }>) {
            let t = x.partition(const_shape![256]).load([0i32]);
            store_current_block_2d(z, log(t));
        }

        #[wax::entry()]
        pub fn f_log2(x: &Tensor<f32, { [256] }>, z: &mut Tensor<f32, { [256] }>) {
            let t = x.partition(const_shape![256]).load([0i32]);
            store_current_block_2d(z, log2(t));
        }

        #[wax::entry()]
        pub fn f_exp2(x: &Tensor<f32, { [256] }>, z: &mut Tensor<f32, { [256] }>) {
            let t = x.partition(const_shape![256]).load([0i32]);
            store_current_block_2d(z, exp2(t, ftz::Disabled));
        }

        #[wax::entry()]
        pub fn f_sin(x: &Tensor<f32, { [256] }>, z: &mut Tensor<f32, { [256] }>) {
            let t = x.partition(const_shape![256]).load([0i32]);
            store_current_block_2d(z, sin(t));
        }

        #[wax::entry()]
        pub fn f_cos(x: &Tensor<f32, { [256] }>, z: &mut Tensor<f32, { [256] }>) {
            let t = x.partition(const_shape![256]).load([0i32]);
            store_current_block_2d(z, cos(t));
        }

        #[wax::entry()]
        pub fn f_rsqrt(x: &Tensor<f32, { [256] }>, z: &mut Tensor<f32, { [256] }>) {
            let t = x.partition(const_shape![256]).load([0i32]);
            store_current_block_2d(z, rsqrt(t, ftz::Disabled));
        }

        #[wax::entry()]
        pub fn f_sqrt_h(x: &Tensor<f16, { [256] }>, z: &mut Tensor<f16, { [256] }>) {
            let t = x.partition(const_shape![256]).load([0i32]);
            store_current_block_2d(z, sqrt(t, rounding::NegativeInf, ftz::Disabled));
        }

        #[wax::entry()]
        pub fn f_log_h(x: &Tensor<f16, { [256] }>, z: &mut Tensor<f16, { [256] }>) {
            let t = x.partition(const_shape![256]).load([0i32]);
            store_current_block_2d(z, log(t));
        }

        #[wax::entry()]
        pub fn f_sqrt_b(x: &Tensor<bf16, { [256] }>, z: &mut Tensor<bf16, { [256] }>) {
            let t = x.partition(const_shape![256]).load([0i32]);
            store_current_block_2d(z, sqrt(t, rounding::NegativeInf, ftz::Disabled));
        }

        #[wax::entry()]
        pub fn ptr_gather<const BK: i32>(
            src: &Tensor<f32, { [-1] }>,
            addr: &Tensor<i64, { [-1] }>,
            idx: &Tensor<i32, { [-1] }>,
            out: &mut Tensor<f32, { [-1] }>,
        ) {
            let ts = const_shape![BK];
            let one = const_shape![1];
            let blk = get_tile_block_id().0;
            let off: Tile<i32, { [BK] }> = idx.partition(ts).load([blk]);
            let a1: Tile<i64, { [1] }> = addr.partition(one).load([0i32]);
            let a: Tile<i64, { [1] }> = a1;
            let base0: PointerTile<*mut f32, { [1] }> = int_to_ptr(a);
            let base: PointerTile<*mut f32, { [BK] }> = broadcast_ptr(base0, ts);
            let ptrs: PointerTile<*mut f32, { [BK] }> = addptr_tile(base, off);
            let (v, _tok): (Tile<f32, { [BK] }>, Token) = load_ptr_tko(
                ptrs,
                ordering::Weak,
                None::<scope::Device>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let mut op: PartitionMut<f32, { [BK] }> = unsafe { out.partition_full_mut(ts) };
            unsafe { op.store(v, [blk]) };
        }

        #[wax::entry()]
        pub fn store_raised<const GQA: i32, const SPLITS: i32>(
            pm: &mut Tensor<f32, { [-1, -1, 32] }>,
            po: &mut Tensor<f32, { [-1, -1, 128] }>,
            src: &Tensor<f32, { [-1, 32] }>,
        ) {
            let wide = const_shape![GQA, 32];
            let narrow = const_shape![GQA, 1, 32];
            let ow = const_shape![GQA, 1, 32];
            let blk = get_tile_block_id().0;
            let kv_lin = blk / SPLITS;
            let split = blk % SPLITS;
            // Element (s,l) = src[s*32+l], so each row is distinguishable.
            let t: Tile<f32, { [GQA, 32] }> = src.partition(wide).load([0i32, 0i32]);
            // A stride-0 lane dim: all 32 lanes of a row form the SAME address and hold the
            // same value, so the write is one word per row, redundantly. That sidesteps a
            // `[GQA,1]` tile, whose lane term is `tp.x & (GQA-1)` - which maps threads to
            // elements by the low bits of the thread index, not by ROW.
            let mut pmw: PartitionMut<f32, { [GQA, 1, 32] }> = unsafe { pm.partition_full_mut(narrow) };
            unsafe { pmw.store(t.reshape(narrow), [kv_lin, split, 0i32]) };
            let mut pow: PartitionMut<f32, { [GQA, 1, 32] }> = unsafe { po.partition_full_mut(ow) };
            unsafe { pow.store(t.reshape(ow), [kv_lin, split, 0i32]) };
        }

        #[wax::entry()]
        pub fn sgmma_8x8(
            a: &Tensor<f32, { [8, 8] }>,
            b: &Tensor<f32, { [8, 8] }>,
            c: &mut Tensor<f32, { [8, 8] }>,
        ) {
            let tile_a = a.partition(const_shape![8, 8]).load([0i32, 0i32]);
            let tile_b = b.partition(const_shape![8, 8]).load([0i32, 0i32]);
            let acc = constant(0.0f32, const_shape![8, 8]);
            store_current_block_2d(c, mmaf(tile_a, tile_b, acc));
        }

        #[wax::entry()]
        pub fn r3_load<const BN: i32>(
            out: &mut Tensor<f32, { [-1, 32] }>,
            k: &Tensor<f32, { [-1, -1, 64] }>,
        ) {
            let ts = const_shape![1, BN, 32];
            let h = get_tile_block_id().0;
            let kp = k.partition(ts);
            let t: Tile<f32, { [1, BN, 32] }> = kp.load([h, 0i32, 0i32]);
            let flat: Tile<f32, { [BN, 32] }> = t.reshape(const_shape![BN, 32]);
            // Each block writes its OWN slab, so the blocks cannot race for the answer.
            let mut op: PartitionMut<f32, { [BN, 32] }> =
                unsafe { out.partition_full_mut(const_shape![BN, 32]) };
            unsafe { op.store(flat, [h, 0i32]) };
        }

        #[wax::entry()]
        pub fn col_reduce<const BN: i32>(
            out: &mut Tensor<f32, { [-1, 32] }>,
            x: &Tensor<f32, { [-1, 32] }>,
        ) {
            let wide = const_shape![BN, 32];
            let chunk = const_shape![1, 32];
            let xp = x.partition(wide);
            let t: Tile<f32, { [BN, 32] }> = xp.load([0i32, 0i32]);
            // FOUR chained column reduces sharing the scratch, as the SDPA merge does: a
            // max, then three sums that depend on it. One reduce alone cannot expose a
            // write-after-read hazard in the shared slab.
            let mx: Tile<f32, { [32] }> = reduce_max(t, 0i32);
            let mxw: Tile<f32, { [BN, 32] }> = mx.reshape(chunk).broadcast(wide);
            let d: Tile<f32, { [BN, 32] }> = t - mxw;
            let s1: Tile<f32, { [32] }> = reduce_sum(d, 0i32);
            let a: Tile<f32, { [BN, 32] }> = d * mxw;
            let b: Tile<f32, { [BN, 32] }> = d + mxw;
            let s2: Tile<f32, { [32] }> = reduce_sum(a, 0i32);
            let s3: Tile<f32, { [32] }> = reduce_sum(b, 0i32);
            let r: Tile<f32, { [32] }> = s1 + s2 + s3;
            let o: Tile<f32, { [1, 32] }> = r.reshape(chunk);
            let mut op: PartitionMut<f32, { [1, 32] }> = unsafe { out.partition_full_mut(chunk) };
            unsafe { op.store(o, [0i32, 0i32]) };
        }

        #[wax::entry()]
        pub fn rope<const D: i32, const HALF: i32, const T: i32>(
            x: &Tensor<f32, { [-1, D] }>,
            cos: &Tensor<f32, { [-1, HALF] }>,
            sin: &Tensor<f32, { [-1, HALF] }>,
            out: &mut Tensor<f32, { [1, D] }>,
        ) {
            let ts = const_shape![1, 32];
            let row = get_tile_block_id().0;
            let i_t = row % T;
            let h = HALF / 32i32;
            let mut op: PartitionMut<f32, { [1, 32] }> = unsafe { out.partition_full_mut(ts) };
            for c in 0i32..h {
                let a: Tile<f32, { [1, 32] }> = load_tile(x, ts, [row, c]);
                let b: Tile<f32, { [1, 32] }> = load_tile(x, ts, [row, c + h]);
                let cs: Tile<f32, { [1, 32] }> = load_tile(cos, ts, [i_t, c]);
                let sn: Tile<f32, { [1, 32] }> = load_tile(sin, ts, [i_t, c]);
                let oa: Tile<f32, { [1, 32] }> = a * cs - b * sn;
                let ob: Tile<f32, { [1, 32] }> = a * sn + b * cs;
                unsafe { op.store(oa, [row, c]) };
                unsafe { op.store(ob, [row, c + h]) };
            }
        }

        #[wax::entry()]
        pub fn row_reduce<const BN: i32>(
            out: &mut Tensor<f32, { [-1, 32] }>,
            x: &Tensor<f32, { [-1, 32] }>,
        ) {
            let wide = const_shape![BN, 32];
            let xp = x.partition(wide);
            let t: Tile<f32, { [BN, 32] }> = xp.load([0i32, 0i32]);
            let r: Tile<f32, { [BN] }> = reduce_sum(t, 1i32);
            // Broadcast the row result back across the row and store the whole grid, so the
            // test can see exactly which value landed on which (row, lane).
            let rb: Tile<f32, { [BN, 32] }> = r.reshape(const_shape![BN, 1]).broadcast(wide);
            let mut op: PartitionMut<f32, { [BN, 32] }> = unsafe { out.partition_full_mut(wide) };
            unsafe { op.store(rb, [0i32, 0i32]) };
        }

        #[wax::entry()]
        pub fn trivial(x: &Tensor<f32, { [256] }>, z: &mut Tensor<f32, { [256] }>) {
            let tile_x = x.partition(const_shape![256]).load([0i32]);
            store_current_block_2d(z, tile_x);
        }

        #[wax::entry()]
        pub fn softmax_masked<const N: i32>(
            x: &Tensor<f32, { [-1, N] }>,
            out: &mut Tensor<f32, { [-1, N] }>,
        ) {
            let ts = const_shape![1, 32];
            let row = get_tile_block_id().0;
            let num_chunks = (N + 31i32) / 32i32;

            // Per-lane column index within a chunk: element c lives on lane c.
            let lanes1: Tile<i32, { [32] }> = iota(const_shape![32]);
            let lanes: Tile<i32, { [1, 32] }> = lanes1.reshape(ts);
            let n_bc: Tile<i32, { [1, 32] }> = broadcast_scalar(N, ts);

            // Pass 1: row max (padding lanes forced to -inf so simd_max ignores them).
            let mut mx: Tile<f32, { [1, 32] }> = constant(-3.4028235e38f32, ts);
            for j in 0i32..num_chunks {
                let col: Tile<i32, { [1, 32] }> = lanes + broadcast_scalar(j * 32i32, ts);
                let mask: Tile<bool, { [1, 32] }> = cmpi(col, n_bc, predicate::LessThan);
                let tx: Tile<f32, { [1, 32] }> = load_tile(x, ts, [row, j]);
                let ninf: Tile<f32, { [1, 32] }> = constant(-3.4028235e38f32, ts);
                let txm: Tile<f32, { [1, 32] }> = select(mask, tx, ninf);
                mx = maxf(mx, txm, nan::Disabled, ftz::Disabled);
            }
            let m1: Tile<f32, { [1] }> = reduce_max(mx, 1i32);
            let m: f32 = tile_to_scalar(m1.reshape(const_shape![]));
            let m_bc: Tile<f32, { [1, 32] }> = broadcast_scalar(m, ts);

            // Pass 2: sum of exp(x - max) (padding lanes forced to 0).
            let mut sm: Tile<f32, { [1, 32] }> = constant(0.0f32, ts);
            for j in 0i32..num_chunks {
                let col: Tile<i32, { [1, 32] }> = lanes + broadcast_scalar(j * 32i32, ts);
                let mask: Tile<bool, { [1, 32] }> = cmpi(col, n_bc, predicate::LessThan);
                let tx: Tile<f32, { [1, 32] }> = load_tile(x, ts, [row, j]);
                let e: Tile<f32, { [1, 32] }> = exp(tx - m_bc);
                let zero: Tile<f32, { [1, 32] }> = constant(0.0f32, ts);
                let em: Tile<f32, { [1, 32] }> = select(mask, e, zero);
                sm = sm + em;
            }
            let l1: Tile<f32, { [1] }> = reduce_sum(sm, 1i32);
            let l: f32 = tile_to_scalar(l1.reshape(const_shape![]));
            let inv: f32 = 1.0f32 / l;
            let inv_bc: Tile<f32, { [1, 32] }> = broadcast_scalar(inv, ts);

            // Pass 3: normalize + store. The view declares `padding`, so the backend drops
            // the tail lanes (col >= N) itself against the tensor's runtime extent - which is
            // the bound that matters here, since the declared `[1, N]` is not the real shape
            // (the dispatch runs one block per row).
            let otok: Token = get_tensor_token(out);
            let mut out_part: PartitionMut<f32, { [1, 32] }> =
                unsafe { make_partition_view_mut(out, ts, padding::Zero, otok) };
            for j in 0i32..num_chunks {
                let tx: Tile<f32, { [1, 32] }> = load_tile(x, ts, [row, j]);
                let y: Tile<f32, { [1, 32] }> = exp(tx - m_bc) * inv_bc;
                unsafe { out_part.store(y, [row, j]) };
            }
        }

        #[wax::entry()]
        pub fn sgmma_dyn_row_stride(
            z: &mut Tensor<f32, { [8, 8] }>,
            x: &Tensor<f32, { [8, 8] }>,
            y: &Tensor<f32, { [8, 8] }>,
        ) {
            let tile_x = x.partition(const_shape![8, 8]).load([0i32, 0i32]);
            let tile_y = y.partition(const_shape![8, 8]).load([0i32, 0i32]);
            let acc = constant(0.0f32, const_shape![8, 8]);
            let tile_z = mmaf(tile_x, tile_y, acc);
            store_current_block_2d(z, tile_z);
        }

        #[wax::entry()]
        pub fn colsum_direct<const R: i32>(
            inp: &Tensor<f32, { [-1, 32] }>,
            out: &mut Tensor<f32, { [-1] }>,
        ) {
            let ts = const_shape![R, 32];
            let m: Tile<f32, { [R, 32] }> = load_tile(inp, ts, [0i32, 0i32]);
            let cs: Tile<f32, { [32] }> = reduce_sum(m, 0i32);
            let out_ts = const_shape![32];
            let mut op: PartitionMut<f32, { [32] }> = unsafe { out.partition_full_mut(out_ts) };
            // No `sg == 0` guard: a column reduce leaves every group holding the same 32
            // values and a `[32]` tile forms the same address in each, so the guard only
            // bounded redundant identical writes.
            unsafe { op.store(cs, [0i32]) };
        }

        #[wax::entry()]
        pub fn tcolsum_reduce<const BN: i32, const BNW: i32>(
            inp: &Tensor<f32, { [-1, 32] }>,
            out: &mut Tensor<f32, { [-1] }>,
        ) {
            // Load through a `[1, BN*32]` shape and reshape to `[BN, 32]` rather than
            // loading `[BN,32]` directly: a 2-D tile whose extents are both multiples of 8
            // is claimed by the SGMMA fragment classifier, which would try to load it as
            // simdgroup fragments. `[1, N]` is never SGMMA-shaped, and `reshape` is an
            // identity in the backend - both shapes put element `tp.x` on thread `tp.x`,
            // so `[BN,32]` element (r,c) is thread r*32+c either way.
            let flat_ts = const_shape![1, BNW];
            let ts = const_shape![BN, 32];
            let flat: Tile<f32, { [1, BNW] }> = load_tile(inp, flat_ts, [0i32, 0i32]);
            let m: Tile<f32, { [BN, 32] }> = flat.reshape(ts);
            let cs: Tile<f32, { [32] }> = reduce_sum(m, 0i32);
            let out_ts = const_shape![32];
            let mut op: PartitionMut<f32, { [32] }> = unsafe { out.partition_full_mut(out_ts) };
            // No `sg == 0` guard: a column reduce leaves every group holding the same 32
            // values and a `[32]` tile forms the same address in each, so the guard only
            // bounded redundant identical writes.
            unsafe { op.store(cs, [0i32]) };
        }

        #[wax::entry()]
        pub fn tile_load<const BN: i32>(
            out: &mut Tensor<f32, { [-1, 32] }>,
            x: &Tensor<f32, { [-1, 64] }>,
        ) {
            let wide = const_shape![BN, 32];
            let c = get_tile_block_id().0;
            let xp = x.partition(wide);
            let t: Tile<f32, { [BN, 32] }> = xp.load([0i32, c]);
            let mut op: PartitionMut<f32, { [BN, 32] }> = unsafe { out.partition_full_mut(wide) };
            unsafe { op.store(t, [c, 0i32]) };
        }

        #[wax::entry()]
        pub fn invariant_colsum<const BN: i32>(
            x: &Tensor<f32, { [-1, 32] }>,
            out: &mut Tensor<f32, { [-1, 32] }>,
        ) {
            let ts = const_shape![1, 32];
            let wide = const_shape![BN, 32];
            // A `[BN,32]` tile spans the threadgroup, so row s IS group s's row - the same
            // data `load([sg, 0])` fetched, without asking which group this is.
            let xw: Tile<f32, { [BN, 32] }> = x.partition(wide).load([0i32, 0i32]);
            let r1: Tile<f32, { [BN] }> = reduce_sum(xw, 1i32);
            let w: Tile<f32, { [BN, 32] }> = r1.reshape(const_shape![BN, 1]).broadcast(wide);
            let c: Tile<f32, { [32] }> = reduce_sum(w, 0i32);
            let cw: Tile<f32, { [BN, 32] }> = c.reshape(ts).broadcast(wide);
            let mut op: PartitionMut<f32, { [BN, 32] }> = unsafe { out.partition_full_mut(wide) };
            unsafe { op.store(cw, [0i32, 0i32]) };
        }

        #[wax::entry()]
        pub fn divergent_colsum<const BN: i32>(
            x: &Tensor<f32, { [-1, 32] }>,
            out: &mut Tensor<f32, { [-1, 32] }>,
        ) {
            let ts = const_shape![1, 32];
            let wide = const_shape![BN, 32];
            let w: Tile<f32, { [BN, 32] }> = x.partition(wide).load([0i32, 0i32]);
            let c: Tile<f32, { [32] }> = reduce_sum(w, 0i32);
            let cw: Tile<f32, { [BN, 32] }> = c.reshape(ts).broadcast(wide);
            let mut op: PartitionMut<f32, { [BN, 32] }> = unsafe { out.partition_full_mut(wide) };
            unsafe { op.store(cw, [0i32, 0i32]) };
        }

}