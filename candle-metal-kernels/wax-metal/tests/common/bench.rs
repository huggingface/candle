/// One row: the entry, its generics, and the per-buffer strides, as the POC's
use super::Case;

/// tests instantiate it. `n as i32`-style symbolic strides are filled in with
/// the concrete value the POC test ran.
pub const CORPUS: &[Case] = &[
    // prop.rs / air_ir.rs / if_else.rs
    Case::same("sgmma_8x8", &[], &[("a", &[8, 1]), ("b", &[8, 1]), ("c", &[8, 1])]),
    Case::same("softmax", &["f32", "512", "32"], &[("x", &[512, 1]), ("out", &[512, 1])]),
    Case::same("softmax", &["f16", "2048", "32"], &[("x", &[2048, 1]), ("out", &[2048, 1])]),
    Case::same("softmax__prop", &["32", "32"], &[("x", &[32, 1]), ("out", &[32, 1])]),
    Case::same("softmax__prop", &["128", "32"], &[("x", &[128, 1]), ("out", &[128, 1])]),
    Case::same(
        "softmax__bitcode_vs_metal_as",
        &["256", "32"],
        &[("x", &[256, 1]), ("out", &[256, 1])],
    ),
    Case::same(
        "softmax__bitcode_vs_metal_as",
        &["512", "32"],
        &[("x", &[512, 1]), ("out", &[512, 1])],
    ),
    Case::same(
        "rms_norm",
        &["f32", "512", "512"],
        &[("x", &[512, 1]), ("w", &[1]), ("out", &[512, 1])],
    ),
    Case::same(
        "rms_norm",
        &["f16", "2048", "1024"],
        &[("x", &[2048, 1]), ("w", &[1]), ("out", &[2048, 1])],
    ),
    Case::same(
        "rms_norm__prop",
        &["32", "32"],
        &[("x", &[32, 1]), ("w", &[1]), ("out", &[32, 1])],
    ),
    Case::same(
        "rms_norm__prop",
        &["128", "32"],
        &[("x", &[128, 1]), ("w", &[1]), ("out", &[128, 1])],
    ),
    Case::same(
        "rms_norm__bitcode_vs_metal_as",
        &["512", "32"],
        &[("x", &[512, 1]), ("w", &[1]), ("out", &[512, 1])],
    ),
    Case::same(
        "rms_norm__bitcode_vs_metal_as",
        &["256", "256"],
        &[("x", &[256, 1]), ("w", &[1]), ("out", &[256, 1])],
    ),
    Case::same(
        "vector_add",
        &[],
        &[("a", &[1]), ("b", &[1]), ("c", &[1])],
    ),
    Case::same("relu", &[], &[("x", &[1]), ("z", &[1])]),
    Case::same("relu__if_else", &[], &[("x", &[1]), ("z", &[1])]),
    Case::same("relu__if_else", &[], &[("x", &[1]), ("z", &[1])]),
    Case::same("dual_path", &[], &[("x", &[4, 1]), ("z", &[4, 1])]),
    // int_load.rs / bitcode_vs_metal_as.rs
    Case::same("int_to_float", &[], &[("x", &[32, 1]), ("out", &[32, 1])]),
    Case::same("int_to_float__int_load", &[], &[("x", &[32, 1]), ("out", &[32, 1])]),
    Case::same("int_copy", &[], &[("x", &[32, 1]), ("out", &[32, 1])]),
    Case::same(
        "index_select",
        &["64"],
        &[("x", &[64, 1]), ("ids", &[1, 0]), ("out", &[64, 1])],
    ),
    Case::same(
        "where_bcast",
        &[],
        &[("cond", &[1]), ("t", &[1]), ("f", &[1]), ("addr", &[1]), ("meta", &[1]), ("out", &[1])],
    ),
    Case::same(
        "gather1d",
        &[],
        &[("x", &[1]), ("addr", &[1]), ("off", &[1]), ("out", &[1])],
    ),
    // flash_attn.rs test kernels (reduce/scan/loop/gemv/atan2/convert)
    Case::same("reduce_sum_test", &["32"], &[("input", &[32, 1]), ("output", &[32, 1])]),
    Case::same("reduce_sum_test", &["64"], &[("input", &[64, 1]), ("output", &[64, 1])]),
    Case::same("reduce_max_test", &["32"], &[("input", &[32, 1]), ("output", &[32, 1])]),
    Case::same("reduce_max_test", &["64"], &[("input", &[64, 1]), ("output", &[64, 1])]),
    Case::same("reduce_min_test", &["32"], &[("input", &[32, 1]), ("output", &[32, 1])]),
    Case::same("reduce_min_test", &["64"], &[("input", &[64, 1]), ("output", &[64, 1])]),
    Case::same("scan_sum_test", &["32"], &[("input", &[32, 1]), ("output", &[32, 1])]),
    Case::same("scan_sum_test", &["64"], &[("input", &[64, 1]), ("output", &[64, 1])]),
    Case::same(
        "scan_sum_test__flash_attn",
        &["32"],
        &[("input", &[32, 1]), ("output", &[32, 1])],
    ),
    Case::same(
        "scan_sum_test__flash_attn",
        &["64"],
        &[("input", &[64, 1]), ("output", &[64, 1])],
    ),
    Case::same("loop_sum_test", &["32", "4"], &[("input", &[32, 1]), ("output", &[32, 1])]),
    Case::same(
        "gemv_test",
        &["32", "128"],
        &[("a", &[128, 1]), ("vec", &[128, 1]), ("output", &[32, 1])],
    ),
    Case::same(
        "atan2_test",
        &["32"],
        &[("y_in", &[32, 1]), ("x_in", &[32, 1]), ("output", &[32, 1])],
    ),
    Case::same(
        "atan2_test",
        &["64"],
        &[("y_in", &[64, 1]), ("x_in", &[64, 1]), ("output", &[64, 1])],
    ),
    // flash attention proper (tests/flash_attn.rs + src/attn.rs builders)
    Case::same(
        "flash_attn_decode",
        &["64", "32"],
        &[("q", &[64, 1]), ("k", &[64, 1]), ("v", &[64, 1]), ("out", &[64, 1])],
    ),
    Case::same(
        "flash_attn_decode",
        &["128", "64"],
        &[("q", &[128, 1]), ("k", &[128, 1]), ("v", &[128, 1]), ("out", &[128, 1])],
    ),
    Case::same(
        "flash_attn_decode_d64",
        &["32"],
        &[("q", &[64, 1]), ("k", &[64, 1]), ("v", &[64, 1]), ("out", &[64, 1])],
    ),
    Case::same(
        "flash_attn_decode_d128",
        &["32"],
        &[("q", &[128, 1]), ("k", &[128, 1]), ("v", &[128, 1]), ("out", &[128, 1])],
    ),
    Case::same(
        "flash_attn_decode_d64_b8",
        &["64"],
        &[("q", &[64, 1]), ("k", &[64, 1]), ("v", &[64, 1]), ("out", &[64, 1])],
    ),
    Case::same(
        "flash_attn_decode_d128_b8",
        &["128"],
        &[("q", &[128, 1]), ("k", &[128, 1]), ("v", &[128, 1]), ("out", &[128, 1])],
    ),
    Case::same(
        "flash_attn_partial_batched_d64",
        &["64", "2"],
        &[
            ("q", &[64, 1]),
            ("k", &[64, 1]),
            ("v", &[64, 1]),
            ("partial_m", &[32, 1]),
            ("partial_l", &[32, 1]),
            ("partial_acc", &[64, 1]),
        ],
    ),
    Case::same(
        "flash_attn_partial_batched_d128",
        &["128", "2"],
        &[
            ("q", &[128, 1]),
            ("k", &[128, 1]),
            ("v", &[128, 1]),
            ("partial_m", &[32, 1]),
            ("partial_l", &[32, 1]),
            ("partial_acc", &[128, 1]),
        ],
    ),
    Case::same(
        "flash_attn_partial_batched_d64_b8",
        &["64", "2"],
        &[
            ("q", &[64, 1]),
            ("k", &[64, 1]),
            ("v", &[64, 1]),
            ("partial_m", &[32, 1]),
            ("partial_l", &[32, 1]),
            ("partial_acc", &[64, 1]),
        ],
    ),
    Case::same(
        "flash_attn_partial_batched_d128_b8",
        &["128", "2"],
        &[
            ("q", &[128, 1]),
            ("k", &[128, 1]),
            ("v", &[128, 1]),
            ("partial_m", &[32, 1]),
            ("partial_l", &[32, 1]),
            ("partial_acc", &[128, 1]),
        ],
    ),
    Case::same(
        "flash_attn_combine_batched_d64",
        &["2"],
        &[
            ("partial_m", &[32, 1]),
            ("partial_l", &[32, 1]),
            ("partial_acc", &[64, 1]),
            ("out", &[64, 1]),
        ],
    ),
    Case::same(
        "flash_attn_combine_batched_d128",
        &["2"],
        &[
            ("partial_m", &[32, 1]),
            ("partial_l", &[32, 1]),
            ("partial_acc", &[128, 1]),
            ("out", &[128, 1]),
        ],
    ),
    // candle_wide_kernels.rs (sdpa d64, the wide-N softmax/argmax, copy2d, d128)
    Case::same(
        "sdpa_sd_d64",
        &["f32", "4", "4", "8", "256"],
        &[
            ("q", &[64, 1]),
            ("k", &[64, 1]),
            ("v", &[64, 1]),
            ("meta", &[1]),
            ("out", &[64, 1]),
        ],
    ),
    Case::same(
        "sdpa_sd_d64",
        &["f32", "4", "4", "32", "1024"],
        &[
            ("q", &[64, 1]),
            ("k", &[64, 1]),
            ("v", &[64, 1]),
            ("meta", &[1]),
            ("out", &[64, 1]),
        ],
    ),
    Case::same(
        "sdpa_sd_d64__candle_wide_kernels",
        &["f32", "1", "1", "32", "1024"],
        &[
            ("q", &[64, 1]),
            ("k", &[64, 1]),
            ("v", &[64, 1]),
            ("meta", &[1]),
            ("out", &[64, 1]),
        ],
    ),
    Case::same(
        "softmax_bn",
        &["f32", "128", "128"],
        &[("x", &[128, 1]), ("out", &[128, 1])],
    ),
    Case::same(
        "softmax_bn",
        &["f32", "1000", "1024"],
        &[("x", &[1000, 1]), ("out", &[1000, 1])],
    ),
    Case::same(
        "argmax_bn",
        &["f32", "1000", "1024"],
        &[("x", &[1000, 1]), ("out", &[1, 1])],
    ),
    Case::same(
        "copy2d",
        &["f32", "4"],
        &[("src", &[32, 32, 1]), ("meta", &[1, 0]), ("out", &[32, 32, 1])],
    ),
    Case::same(
        "sdpa_sd_d128",
        &["f32", "8", "2", "8", "256"],
        &[
            ("q", &[128, 4, 1]),
            ("k", &[128, 4, 1]),
            ("v", &[128, 1]),
            ("meta", &[1]),
            ("out", &[128, 1]),
        ],
    ),
    Case::same(
        "sdpa_sd_d128",
        &["f16", "4", "4", "32", "1024"],
        &[
            ("q", &[128, 4, 1]),
            ("k", &[128, 4, 1]),
            ("v", &[128, 1]),
            ("meta", &[1]),
            ("out", &[128, 1]),
        ],
    ),
    Case::same(
        "sdpa_dbg",
        &["1", "1", "32", "1024"],
        &[("out", &[32, 1]), ("q", &[64, 1]), ("k", &[64, 1]), ("meta", &[1])],
    ),
    // Two-pass chain (the POC's fully-tiled variant: SPLITS=32 on a tile axis)
    Case::same(
        "sdpa_sd_d128_p1",
        &["f32", "2", "2", "32", "1"],
        &[
            ("q", &[128, 4, 1]),
            ("k", &[128, 4, 1]),
            ("v", &[128, 1]),
            ("meta", &[1]),
            ("partial_m", &[32, 1, 0]),
            ("partial_l", &[32, 1, 0]),
            ("partial_o", &[32 * 128, 128, 1]),
        ],
    ),
    Case::same(
        "sdpa_sd_d128_p2",
        &["f32", "32", "1"],
        &[
            ("partial_m", &[32, 1]),
            ("partial_l", &[32, 1]),
            ("partial_o", &[128, 1]),
            ("partial_m_t", &[1, 32, 0]),
            ("out", &[128, 1]),
        ],
    ),
    Case::same(
        "sdpa_sd_d128_p2",
        &["f16", "32", "1"],
        &[
            ("partial_m", &[32, 1]),
            ("partial_l", &[32, 1]),
            ("partial_o", &[128, 1]),
            ("partial_m_t", &[1, 32, 0]),
            ("out", &[128, 1]),
        ],
    ),
    // GEMM family
    Case::same(
        "gemm",
        &["f32", "32", "32", "8", "64"],
        &[("z", &[32, 1]), ("x", &[64, 1]), ("y", &[32, 1])],
    ),
    Case::same(
        "gemm",
        &["f16", "32", "32", "8", "256"],
        &[("z", &[32, 1]), ("x", &[256, 1]), ("y", &[32, 1])],
    ),
    Case::same(
        "gemm__gemm_kernels",
        &["f32", "32", "32", "8", "64"],
        &[("z", &[32, 1]), ("x", &[64, 1]), ("y", &[32, 1])],
    ),
    Case::same(
        "gemm__gemm_kernels",
        &["bf16", "32", "32", "8", "128"],
        &[("z", &[32, 1]), ("x", &[128, 1]), ("y", &[32, 1])],
    ),
    Case::same(
        "gemm__bench_shapes",
        &["f32", "16", "16", "8", "32"],
        &[("z", &[16, 1]), ("x", &[32, 1]), ("y", &[16, 1])],
    ),
    Case::same(
        "gemm__gemm_kernels",
        &["f32", "8", "8", "8", "64"],
        &[("z", &[32, 1]), ("x", &[64, 1]), ("y", &[32, 1])],
    ),
    Case::same(
        "gemm",
        &["f32", "8", "8", "8", "64"],
        &[("z", &[64, 1]), ("x", &[64, 1]), ("y", &[64, 1])],
    ),
    Case::same(
        "gemm_padded",
        &["f32", "16", "16", "8", "64", "64", "64", "8"],
        &[("z", &[64, 1]), ("x", &[64, 1]), ("y", &[1, 64])],
    ),
    // NT is `gemm_padded` over the transposed VIEW of the same bytes: y strides [1, K].
    Case::same(
        "gemm_padded",
        &["f16", "8", "32", "8", "72", "130", "64", "8"],
        &[("z", &[130, 1]), ("x", &[64, 1]), ("y", &[1, 64])],
    ),
    Case::same(
        "gemm_eout",
        &["f16", "32", "32", "8", "64"],
        &[("z", &[64, 1]), ("x", &[64, 1]), ("y", &[64, 1])],
    ),
    Case::same(
        "gemm_eout",
        &["bf16", "8", "32", "32", "64"],
        &[("z", &[64, 1]), ("x", &[64, 1]), ("y", &[64, 1])],
    ),
    Case::same("gemm_f16_8x8", &[], &[("z", &[8, 1]), ("x", &[8, 1]), ("y", &[8, 1])]),
    Case::same("gemm_f16_16x16", &[], &[("z", &[16, 1]), ("x", &[16, 1]), ("y", &[16, 1])]),
    Case::same("gemm_bf16_8x8", &[], &[("z", &[8, 1]), ("x", &[8, 1]), ("y", &[8, 1])]),
    Case::same("gemm_bf16_16x16", &[], &[("z", &[16, 1]), ("x", &[16, 1]), ("y", &[16, 1])]),
    Case::same(
        "ragged_gemm",
        &["20", "12", "12", "2"],
        &[("z", &[12, 1]), ("x", &[12, 1]), ("y", &[12, 1])],
    ),
    // GEMV family (reduce-based and the epilogue-fused variants; the M=1 SGMMA
    // variants are deliberately absent - see the module docs)
    Case::same(
        "gemv",
        &["f32", "32", "128"],
        &[("a", &[128, 1]), ("vec", &[128, 1]), ("output", &[32, 1])],
    ),
    Case::same(
        "gemv",
        &["f16", "32", "256"],
        &[("a", &[256, 1]), ("vec", &[256, 1]), ("output", &[32, 1])],
    ),
    Case::same(
        "gemv__epilogue_fusion",
        &["f32", "32", "2048"],
        &[("a", &[2048, 1]), ("vec", &[2048, 1]), ("out", &[1, 1])],
    ),
    Case::same(
        "gemv_silu",
        &["f32", "32", "2048"],
        &[("a", &[2048, 1]), ("vec", &[2048, 1]), ("out", &[1, 1])],
    ),
    Case::same("silu_vec", &["f32", "32"], &[("inp", &[1]), ("out", &[1])]),
    // vecmat (M=1 tile GEMM) and its split-K reduce
    Case::same(
        "vecmat_tg",
        &["f32", "1024", "1024", "32", "32"],
        &[("out", &[1024, 1]), ("x", &[1, 0]), ("y", &[1024, 1])],
    ),
    Case::same(
        "vecmat_tg",
        &["f32", "1000", "1024", "32", "32"],
        &[("out", &[1000, 1]), ("x", &[1, 0]), ("y", &[1000, 1])],
    ),
    Case::same(
        "vecmat_reduce",
        &["f16", "4096", "2"],
        &[("out", &[4096, 1]), ("part", &[4096, 1])],
    ),
    Case::same(
        "vm_probe_a",
        &["f32", "64", "128", "32", "4"],
        &[("out", &[64, 1]), ("x", &[128, 1]), ("y", &[128, 1])],
    ),
    Case::same(
        "vm_probe_b",
        &["f32", "64", "128", "32", "4"],
        &[("out", &[64, 1]), ("x", &[128, 1]), ("y", &[128, 1])],
    ),
    Case::same(
        "vm_probe_c",
        &["f32", "64", "128", "32", "4"],
        &[("out", &[64, 1]), ("x", &[128, 1]), ("y", &[128, 1])],
    ),
    Case::same(
        "vm_probe_d",
        &["f32", "64", "128", "32", "4"],
        &[("out", &[64, 1]), ("x", &[128, 1]), ("y", &[128, 1])],
    ),
    // Wide tile carries and lane geometry
    Case::same(
        "wide_carry",
        &["256", "4"],
        &[("out", &[256, 1]), ("x", &[256, 1])],
    ),
    Case::same(
        "rows_per_simd_group",
        &["4"],
        &[("inp", &[32, 1]), ("out", &[32, 1]), ("scalars", &[1]), ("winp", &[64, 1])],
    ),
    Case::same(
        "rows_via_tile_axis",
        &["4"],
        &[("inp", &[32, 1]), ("out", &[32, 1]), ("scalars", &[1]), ("winp", &[64, 1])],
    ),
    Case::same(
        "iota_within_tile",
        &["4"],
        &[("inp", &[32, 1]), ("out", &[32, 1]), ("scalars", &[1]), ("winp", &[64, 1])],
    ),
    Case::same(
        "wide_tile",
        &["4"],
        &[("inp", &[32, 1]), ("out", &[32, 1]), ("scalars", &[1]), ("winp", &[64, 1])],
    ),
    Case::same(
        "uniform_scalar",
        &["4"],
        &[("inp", &[32, 1]), ("out", &[32, 1]), ("scalars", &[1]), ("winp", &[64, 1])],
    ),
    // Lane-vector fusion at every float dtype
    Case::same("vcopy_f32", &[], &[("x", &[128, 4, 1]), ("out", &[128, 4, 1])]),
    Case::same("vcopy_f16", &[], &[("x", &[256, 8, 1]), ("out", &[256, 8, 1])]),
    Case::same("vcopy_bf16", &[], &[("x", &[256, 8, 1]), ("out", &[256, 8, 1])]),
    // Narrow / padded tiles
    Case::same("narrow_copy", &["7", "21"], &[("out", &[21, 1]), ("x", &[21, 1])]),
    Case::same("padded_copy", &["100"], &[("x", &[1]), ("out", &[1])]),
    Case::same("padded_store", &["100"], &[("x", &[1]), ("out", &[1])]),
    Case::same("unpadded_copy", &["100"], &[("x", &[1]), ("out", &[1])]),
    Case::same("tile_load", &["8"], &[("out", &[32, 1]), ("x", &[64, 1])]),
    Case::same("tile_load", &["16"], &[("out", &[32, 1]), ("x", &[64, 1])]),
    Case::same("tile_load", &["32"], &[("out", &[32, 1]), ("x", &[64, 1])]),
    Case::same("trivial", &[], &[("x", &[1]), ("z", &[1])]),
    // Precise math intrinsics
    Case::same("f_sqrt", &[], &[("x", &[1]), ("z", &[1])]),
    Case::same("f_log", &[], &[("x", &[1]), ("z", &[1])]),
    Case::same("f_log2", &[], &[("x", &[1]), ("z", &[1])]),
    Case::same("f_exp2", &[], &[("x", &[1]), ("z", &[1])]),
    Case::same("f_sin", &[], &[("x", &[1]), ("z", &[1])]),
    Case::same("f_cos", &[], &[("x", &[1]), ("z", &[1])]),
    Case::same("f_rsqrt", &[], &[("x", &[1]), ("z", &[1])]),
    Case::same("f_sqrt_h", &[], &[("x", &[1]), ("z", &[1])]),
    Case::same("f_log_h", &[], &[("x", &[1]), ("z", &[1])]),
    Case::same("f_sqrt_b", &[], &[("x", &[1]), ("z", &[1])]),
    // Pointer-tile gathers (pure upstream DSL: int_to_ptr + addptr + load_ptr_tko)
    Case::same(
        "ptr_gather",
        &["32"],
        &[("src", &[1]), ("addr", &[1]), ("idx", &[1]), ("out", &[1])],
    ),
    Case::same(
        "three_addrs",
        &["32"],
        &[("src", &[1]), ("addr", &[1]), ("idx", &[1]), ("out", &[1])],
    ),
    Case::same(
        "one_addr",
        &["32"],
        &[("src", &[1]), ("addr", &[1]), ("idx", &[1]), ("out", &[1])],
    ),
    // Rank-3 / rank-raised views
    Case::same(
        "store_raised",
        &["2", "4"],
        &[("pm", &[4, 1, 0]), ("po", &[4 * 128, 128, 1]), ("src", &[32, 1])],
    ),
    Case::same(
        "r3_load",
        &["8"],
        &[("out", &[32, 1]), ("k", &[64 * 64, 64, 1])],
    ),
    // Column / row reduces at the widths the POC sweeps
    Case::same("col_reduce", &["8"], &[("out", &[32, 1]), ("x", &[32, 1])]),
    Case::same("col_reduce", &["16"], &[("out", &[32, 1]), ("x", &[32, 1])]),
    Case::same("col_reduce", &["32"], &[("out", &[32, 1]), ("x", &[32, 1])]),
    Case::same("row_reduce", &["32"], &[("out", &[32, 1]), ("x", &[32, 1])]),
    Case::same("row_sum", &["f16", "64", "32"], &[("x", &[64, 1]), ("out", &[64, 1])]),
    Case::same("row_sum", &["bf16", "64", "32"], &[("x", &[64, 1]), ("out", &[64, 1])]),
    Case::same("colsum_direct", &["32"], &[("inp", &[64, 1]), ("out", &[1])]),
    Case::same(
        "tcolsum_reduce",
        &["32", "1024"],
        &[("inp", &[32, 1]), ("out", &[1])],
    ),
    Case::same(
        "invariant_colsum",
        &["32"],
        &[("x", &[32, 1]), ("out", &[32, 1])],
    ),
    Case::same(
        "divergent_colsum",
        &["32"],
        &[("x", &[32, 1]), ("out", &[32, 1])],
    ),
    // Masked-fragment gemm/store, rope, scalar-const scale, masked softmax
    Case::same(
        "frag_store",
        &[],
        &[("a", &[-1, 1]), ("id", &[8, 1]), ("out", &[-1, 1])],
    ),
    Case::same(
        "rope",
        &["64", "32", "8"],
        &[("x", &[64, 1]), ("cos", &[32, 1]), ("sin", &[32, 1]), ("out", &[64, 1])],
    ),
    Case::same(
        "uscale",
        &["64"],
        &[("x", &[64, 1]), ("meta", &[1]), ("out", &[64, 1])],
    ),
    Case::same(
        "softmax_masked",
        &["128"],
        &[("x", &[128, 1]), ("out", &[128, 1])],
    ),
    // Dynamic (runtime scalar) row strides - the `-1` stride entries
    Case::same(
        "sgmma_dyn_row_stride",
        &[],
        &[("z", &[-1, 1]), ("x", &[-1, 1]), ("y", &[-1, 1])],
    ),
    // And the same kernel with fully static strides (no scalar args)
    Case::same(
        "sgmma_dyn_row_stride",
        &[],
        &[("z", &[8, 1]), ("x", &[8, 1]), ("y", &[8, 1])],
    ),
];

