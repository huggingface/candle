/// The corpus, one row per kernel. `miss` is the allowlist: the cases the conversion is
use super::Case;

/// expected to refuse, with the opcode that stops them. Shrinking it to empty (and growing
/// `same`) is the progress metric.
pub const CORPUS: &[Case] = &[
    Case::same("scale", &["f32", "32"], &[("out", &[32, 1]), ("x", &[32, 1])]),
    Case::same("scale", &["f16", "32"], &[("out", &[32, 1]), ("x", &[32, 1])]),
    Case::same("scale", &["bf16", "32"], &[("out", &[32, 1]), ("x", &[32, 1])]),
    Case::same(
        "accumulate",
        &["f32", "32", "8"],
        &[("out", &[32, 1]), ("x", &[32, 1])],
    ),
    Case::same("guard_early", &["16", "8"], &[("output", &[1])]),
    Case::same(
        "row_sum",
        &["32"],
        &[("out", &[1, 1]), ("x", &[32, 1])],
    ),
    Case::same(
        "gemm_8x8",
        &[],
        &[("z", &[8, 1]), ("x", &[8, 1]), ("y", &[8, 1])],
    ),
    Case::same(
        "gemm_8x8_f16",
        &[],
        &[("z", &[8, 1]), ("x", &[8, 1]), ("y", &[8, 1])],
    ),
    Case::same(
        "gemm_8x8_bf16",
        &[],
        &[("z", &[8, 1]), ("x", &[8, 1]), ("y", &[8, 1])],
    ),
    Case::same(
        "gemm_loop",
        &["f32", "8", "8", "8", "64"],
        &[("z", &[8, 1]), ("x", &[64, 1]), ("y", &[8, 1])],
    ),
    Case::same(
        "tg_colsum",
        &[],
        &[("inp", &[32, 1]), ("out", &[32, 1])],
    ),
    Case::same(
        "tg_colsum16",
        &[],
        &[("inp", &[16, 1]), ("out", &[16, 1])],
    ),
    Case::same(
        "tg_colsum16_f16",
        &[],
        &[("inp", &[16, 1]), ("out", &[16, 1])],
    ),
    Case::same(
        "tg_colsum_f16",
        &[],
        &[("inp", &[32, 1]), ("out", &[32, 1])],
    ),
    Case::same(
        "tg_colsum_bf16",
        &[],
        &[("inp", &[32, 1]), ("out", &[32, 1])],
    ),
    Case::same(
        "frag_rowmax",
        &[],
        &[("out", &[8, 1]), ("x", &[8, 1]), ("y", &[8, 1])],
    ),
    Case::same(
        "lane_vec",
        &[],
        &[("out", &[32, 1]), ("k", &[128, 4, 1]), ("row", &[1])],
    ),
    // Real kernels ported from metal-tile-bench. The POC's own CPU test runs these
    // shapes: (bs, nh, nkv, n, bn) in { (1,4,4,100,8), (1,4,4,100,32), ... }.
    Case::same(
        "sdpa_sd_d128",
        &["f32", "4", "4", "8", "256"],
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
        &["f32", "4", "4", "32", "1024"],
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
        &["4", "2", "32", "1024"],
        &[
            ("out", &[32, 1]),
            ("q", &[64, 1]),
            ("k", &[64, 1]),
            ("meta", &[1]),
        ],
    ),
    Case::same(
        "sdpa_dbg",
        &["4", "2", "16", "512"],
        &[
            ("out", &[32, 1]),
            ("q", &[64, 1]),
            ("k", &[64, 1]),
            ("meta", &[1]),
        ],
    ),
    // The two-pass GQA chain: rank-raised stride-0 views (p1's partials), the transposed
    // stride-0 `[32,1,32]` read and the split fold (p2, with its cross-loop-reuse
    // staging: the `mmax` loop writes its `pm` loads to a thread-private cache the
    // `lsum` loop reads back).
    Case::same(
        "sdpa_sd_d128_p1",
        &["f32", "4", "2", "32", "2"],
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
    // R = 2: the staging cache has two slots, so the slot index (the loop IV) is not a
    // constant - the `gep_array` over the private `[2 x elem]` must track the induction.
    Case::same(
        "sdpa_sd_d128_p2",
        &["f32", "64", "2"],
        &[
            ("partial_m", &[32, 1]),
            ("partial_l", &[32, 1]),
            ("partial_o", &[128, 1]),
            ("partial_m_t", &[1, 32, 0]),
            ("out", &[128, 1]),
        ],
    ),
    // E = f16: the fold's `convert_tile` runs f32 -> f16 around the staged loops.
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
    Case::same(
        "tg_widesum",
        &[],
        &[("out", &[16, 1]), ("inp", &[16, 1])],
    ),
    Case::same(
        "shape_q",
        &[],
        &[("out", &[1, 1]), ("t", &[8, 1])],
    ),
];

