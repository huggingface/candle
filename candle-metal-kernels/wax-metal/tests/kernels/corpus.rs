// The corpus kernels, as DSL source. Kept as text (rather than a `#[wax::module]`) because
// the corpus driver compiles one entry per `KernelCompiler::new`, the way the shipping
// frontends do.

mod kernels {
    use cutile::core::*;

    /// Straight-line elementwise: loads, arithmetic, a store.
    #[wax::entry()]
    pub fn scale<E: ElementType, const N: i32>(
        out: &mut Tensor<E, { [1, N] }>,
        x: &Tensor<E, { [1, N] }>,
    ) {
        let ts = const_shape![1, N];
        let px = x.partition(ts);
        let mut po: PartitionMut<E, { [1, N] }> = unsafe { out.partition_full_mut(ts) };
        let v = px.load([0i32, 0i32]);
        unsafe { po.store(v + v, [0i32, 0i32]) };
    }

    /// A COUNTED LOOP with a carry - the phi/block-argument case, in a real kernel.
    #[wax::entry()]
    pub fn accumulate<E: ElementType, const N: i32, const K: i32>(
        out: &mut Tensor<E, { [1, N] }>,
        x: &Tensor<E, { [K, N] }>,
    ) {
        let ts = const_shape![1, N];
        let px = x.partition(ts);
        let mut acc = px.load([0i32, 0i32]);
        for i in 1i32..K {
            acc = acc + px.load([i, 0i32]);
        }
        let mut po: PartitionMut<E, { [1, N] }> = unsafe { out.partition_full_mut(ts) };
        unsafe { po.store(acc, [0i32, 0i32]) };
    }

    /// A CONDITIONAL inside a loop - the if block-argument shape.
    ///
    /// A `break` inside a `for` is rejected by the DSL (and `loop` lowers to an opcode the
    /// dialect has no builder for), so the exit is spelled as a guarded body: same kernel,
    /// `wax.break` coverage lives in the hand-built `wax_to_air_break` test.
    #[wax::entry()]
    pub fn guard_early<const N: i32, const K: i32>(output: &mut Tensor<f32, { [N] }>) {
        let ts = const_shape![1];
        let mut out_part = unsafe { output.partition_full_mut(ts) };
        for i in 0i32..N {
            if i < K {
                let one: Tile<f32, { [1] }> = constant(1.0f32, ts);
                unsafe { out_part.store(one, [i]) };
            }
        }
    }

    /// A WITHIN-SIMD REDUCTION: exercises the `air.simd_*` intrinsic call.
    #[wax::entry()]
    pub fn row_sum<const N: i32>(out: &mut Tensor<f32, { [1, 1] }>, x: &Tensor<f32, { [1, N] }>) {
        let v = x.partition(const_shape![1, N]).load([0i32, 0i32]);
        let s: Tile<f32, { [1] }> = reduce_sum(v, 1i32);
        let mut po: PartitionMut<f32, { [1, 1] }> =
            unsafe { out.partition_full_mut(const_shape![1, 1]) };
        unsafe { po.store(s.reshape(const_shape![1, 1]), [0i32, 0i32]) };
    }

    /// SGMMA without a loop: fragment load, mma, fragment store.
    #[wax::entry()]
    pub fn gemm_8x8(
        z: &mut Tensor<f32, { [8, 8] }>,
        x: &Tensor<f32, { [8, 8] }>,
        y: &Tensor<f32, { [8, 8] }>,
    ) {
        let tile_x = x.partition(const_shape![8, 8]).load([0i32, 0i32]);
        let tile_y = y.partition(const_shape![8, 8]).load([0i32, 0i32]);
        let acc = constant(0.0f32, const_shape![8, 8]);
        z.store(mma(tile_x, tile_y, acc));
    }

    /// SGMMA in f16: A and B are f16 tiles, the accumulator f32 - the mma call name
    /// carries the precision, everything else is f32-register shaped.
    #[wax::entry()]
    pub fn gemm_8x8_f16(
        z: &mut Tensor<f32, { [8, 8] }>,
        x: &Tensor<f16, { [8, 8] }>,
        y: &Tensor<f16, { [8, 8] }>,
    ) {
        let tile_x = x.partition(const_shape![8, 8]).load([0i32, 0i32]);
        let tile_y = y.partition(const_shape![8, 8]).load([0i32, 0i32]);
        let acc = constant(0.0f32, const_shape![8, 8]);
        z.store(mma(tile_x, tile_y, acc));
    }

    /// The bf16 twin of `gemm_8x8_f16`.
    #[wax::entry()]
    pub fn gemm_8x8_bf16(
        z: &mut Tensor<f32, { [8, 8] }>,
        x: &Tensor<bf16, { [8, 8] }>,
        y: &Tensor<bf16, { [8, 8] }>,
    ) {
        let tile_x = x.partition(const_shape![8, 8]).load([0i32, 0i32]);
        let tile_y = y.partition(const_shape![8, 8]).load([0i32, 0i32]);
        let acc = constant(0.0f32, const_shape![8, 8]);
        z.store(mma(tile_x, tile_y, acc));
    }

    /// SGMMA WITH a loop: the fragment CARRY - the mma's result crosses the back edge and
    /// the fragment store consumes the loop result.
    #[wax::entry()]
    pub fn gemm_loop<E: ElementType, const BM: i32, const BN: i32, const BK: i32, const K: i32>(
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
        z.store(tile_z);
    }

    /// A COLUMN reduce: spreads across SIMD groups, needs threadgroup scratch and barriers
    /// the conversion route must reproduce exactly.
    #[wax::entry()]
    pub fn tg_colsum(inp: &Tensor<f32, { [-1, 32] }>, out: &mut Tensor<f32, { [-1, 32] }>) {
        let wide = const_shape![32, 32];
        let w: Tile<f32, { [32, 32] }> = inp.partition(wide).load([0i32, 0i32]);
        let cs: Tile<f32, { [32] }> = reduce_sum(w, 0i32);
        let csw: Tile<f32, { [32, 32] }> = cs.reshape(const_shape![1, 32]).broadcast(wide);
        let mut ow: PartitionMut<f32, { [32, 32] }> = unsafe { out.partition_full_mut(wide) };
        unsafe { ow.store(csw, [0i32, 0i32]) };
    }

    /// A tensor shape query: `get_tensor_shape` on static extents lowers one i32 constant
    /// per dim; used here as load offsets (each dim minus its own stride, i.e. zero) so
    /// both values survive to code as real indices.
    #[wax::entry()]
    pub fn shape_q(out: &mut Tensor<f32, { [1, 1] }>, t: &Tensor<f32, { [16, 8] }>) {
        let s: [i32; 2] = get_tensor_shape(t);
        let v: Tile<f32, { [1, 1] }> = t
            .partition(const_shape![1, 1])
            .load([s[1] - 8, s[0] - 16]);
        out.store(v);
    }

    /// A COLUMN reduce over 16 rows: `rows != 32` takes the serial column walk
    /// instead of the transposed SIMD-intrinsic round trip.
    #[wax::entry()]
    pub fn tg_colsum16(inp: &Tensor<f32, { [-1, 32] }>, out: &mut Tensor<f32, { [-1, 1] }>) {
        let w: Tile<f32, { [16, 32] }> = inp.partition(const_shape![16, 32]).load([0i32, 0i32]);
        let cs: Tile<f32, { [16] }> = reduce_sum(w, 0i32);
        let o: Tile<f32, { [16, 1] }> = cs.reshape(const_shape![16, 1]);
        let mut po: PartitionMut<f32, { [16, 1] }> = unsafe { out.partition_full_mut(const_shape![16, 1]) };
        unsafe { po.store(o, [0i32, 0i32]) };
    }

    /// A COLUMN reduce of f16: computes in f32 in the float scratch (extended before the
    /// store, truncated back at the end), the way the bf16 twin does.
    #[wax::entry()]
    pub fn tg_colsum_f16(inp: &Tensor<f16, { [-1, 32] }>, out: &mut Tensor<f16, { [-1, 1] }>) {
        let w: Tile<f16, { [32, 32] }> = inp.partition(const_shape![32, 32]).load([0i32, 0i32]);
        let cs: Tile<f16, { [32] }> = reduce_sum(w, 0i32);
        let o: Tile<f16, { [32, 1] }> = cs.reshape(const_shape![32, 1]);
        let mut po: PartitionMut<f16, { [32, 1] }> = unsafe { out.partition_full_mut(const_shape![32, 1]) };
        unsafe { po.store(o, [0i32, 0i32]) };
    }

    /// A COLUMN reduce of bf16: computes in f32, truncates the result back.
    #[wax::entry()]
    pub fn tg_colsum_bf16(inp: &Tensor<bf16, { [-1, 32] }>, out: &mut Tensor<bf16, { [-1, 1] }>) {
        let w: Tile<bf16, { [32, 32] }> = inp.partition(const_shape![32, 32]).load([0i32, 0i32]);
        let cs: Tile<bf16, { [32] }> = reduce_sum(w, 0i32);
        let o: Tile<bf16, { [32, 1] }> = cs.reshape(const_shape![32, 1]);
        let mut po: PartitionMut<bf16, { [32, 1] }> = unsafe { out.partition_full_mut(const_shape![32, 1]) };
        unsafe { po.store(o, [0i32, 0i32]) };
    }

    /// The f16 twin of `tg_colsum16`: the serial column walk over the float scratch,
    /// f32 compute, f16 in and out.
    #[wax::entry()]
    pub fn tg_colsum16_f16(inp: &Tensor<f16, { [-1, 32] }>, out: &mut Tensor<f16, { [-1, 1] }>) {
        let w: Tile<f16, { [16, 32] }> = inp.partition(const_shape![16, 32]).load([0i32, 0i32]);
        let cs: Tile<f16, { [16] }> = reduce_sum(w, 0i32);
        let o: Tile<f16, { [16, 1] }> = cs.reshape(const_shape![16, 1]);
        let mut po: PartitionMut<f16, { [16, 1] }> = unsafe { out.partition_full_mut(const_shape![16, 1]) };
        unsafe { po.store(o, [0i32, 0i32]) };
    }

    /// A per-ROW reduce of an SGMMA fragment: the result of an mma, reduced over dim 1.
    /// The 8x8 fragment's rows sit two-per-lane-pair across the four groups, so the
    /// shuffle walk's per-lane row values are parked in the scratch's row region and
    /// read back per store lane.
    #[wax::entry()]
    pub fn frag_rowmax(
        out: &mut Tensor<f32, { [8, 1] }>,
        x: &Tensor<f32, { [8, 8] }>,
        y: &Tensor<f32, { [8, 8] }>,
    ) {
        let tile_x = x.partition(const_shape![8, 8]).load([0i32, 0i32]);
        let tile_y = y.partition(const_shape![8, 8]).load([0i32, 0i32]);
        let acc = constant(0.0f32, const_shape![8, 8]);
        let z: Tile<f32, { [8, 8] }> = mma(tile_x, tile_y, acc);
        let r: Tile<f32, { [8] }> = reduce_max(z, 1i32);
        out.store(r.reshape(const_shape![8, 1]));
    }

    /// A dim-1 (per-row) reduce of a `[16, 16]` tile: `cols < 32`, so the SIMD lanes span
    /// rows, not columns - the scratch row round trip (within-row shuffle butterfly, park
    /// the rows, read back per lane) rather than the whole-tile diamond.
    #[wax::entry()]
    pub fn tg_widesum(out: &mut Tensor<f32, { [16, 1] }>, inp: &Tensor<f32, { [16, 16] }>) {
        let w: Tile<f32, { [16, 16] }> = inp.partition(const_shape![16, 16]).load([0i32, 0i32]);
        let s: Tile<f32, { [16] }> = reduce_sum(w, 1i32);
        out.store(s.reshape(const_shape![16, 1]));
    }

    /// The d128 SDPA key-load pattern: a `[1,32,1]` tile loaded as four `j = 0..3`
    /// accesses at a runtime row (a scalar tensor, the way d128 gets its head). Per lane
    /// the four elements are contiguous (lane stride 4), so the CFG fuses the run into
    /// one `<4 x float>` 128-bit transaction. The conversion must emit the identical wide
    /// load, not four scalars.
    #[wax::entry()]
    pub fn lane_vec(out: &mut Tensor<f32, { [1, 32] }>, k: &Tensor<f32, { [-1, 32, 4] }>, row: &Tensor<i32, { [1] }>) {
        let mp = row.partition(const_shape![1]);
        let row: i32 = tile_to_scalar(mp.load([0i32]).reshape(const_shape![]));
        let kp = k.partition(const_shape![1, 32, 1]);
        let a: Tile<f32, { [1, 32] }> = kp.load([row, 0i32, 0i32]).reshape(const_shape![1, 32]);
        let b: Tile<f32, { [1, 32] }> = kp.load([row, 0i32, 1i32]).reshape(const_shape![1, 32]);
        let c: Tile<f32, { [1, 32] }> = kp.load([row, 0i32, 2i32]).reshape(const_shape![1, 32]);
        let e: Tile<f32, { [1, 32] }> = kp.load([row, 0i32, 3i32]).reshape(const_shape![1, 32]);
        out.store(a + b + c + e);
    }

    /// A real kernel from the metal-tile-bench SDPA suite (ported verbatim apart from the
    /// entry attribute): a cut-down tiled d64 that ends in a COLUMN REDUCE over state an
    /// `if` carries. The regression it guards: an `If` yields its results from the arms'
    /// terminators, and the uniformity analysis's "divergent operand makes divergent
    /// result" rule must see them or the column reduce elides its redistribution and
    /// hands every lane the same number.
    ///
    /// Exercises, on the conversion route: `get_tile_block_id`, scalar `meta` loads and
    /// `tile_to_scalar`/`convert_scalar`, a `for` with four `[BN,32]` carries, a tail `if`
    /// carrying them, `maxf`/`exp`, row reduces, and the column reduces (rows==32 fast
    /// path at BN=32, serial walk at BN=16).
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

    /// The flagship real kernel, ported from metal-tile-bench's `sdpa128_kernels`
    /// verbatim apart from the entry attribute: tiled d128 single-pass SDPA with the
    /// BN-way key split as a tile dimension. The four per-lane q/k loads are contiguous,
    /// so both routes must fuse each run into one 128-bit transaction; the `[BN,32]`
    /// column reduces take the rows==32 scratch path at BN=32 and the serial walk at
    /// smaller BN; `convert_tile` covers the f16/bf16 element types.
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
        let m_row: Tile<f32, { [BN] }> = reduce_max(m_i, 1i32);
        let mt: Tile<f32, { [BN, 32] }> = m_row.reshape(const_shape![BN, 1]).broadcast(wide);
        let l_row: Tile<f32, { [BN] }> = reduce_max(l_i, 1i32);
        let lt: Tile<f32, { [BN, 32] }> = l_row.reshape(const_shape![BN, 1]).broadcast(wide);
        let gm_w: Tile<f32, { [32] }> = reduce_max(mt, 0i32);
        let gm: Tile<f32, { [BN, 32] }> = gm_w.reshape(chunk).broadcast(wide);
        let f: Tile<f32, { [BN, 32] }> = exp(mt - gm);
        let gl_w: Tile<f32, { [32] }> = reduce_sum(lt * f, 0i32);
        let gl: Tile<f32, { [1, 32] }> = gl_w.reshape(chunk);
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

    /// The two-pass GQA chain, ported from metal-tile-bench's `sdpa2pass_kernels`.
    /// `p1` splits the K/V range across threadgroups (one partial softmax per split,
    /// written to rank-raised stride-0 views); `p2` folds the partials back - its
    /// `[32, 1, 32]` tile reads `partial_m` transposed with the lane axis stride-0.
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
    }}
