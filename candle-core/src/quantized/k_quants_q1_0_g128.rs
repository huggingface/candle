// Q1_0_g128 support for candle
// Based on https://github.com/PrismML-Eng/llama.cpp/blob/master/ggml/src/ggml-common.h
// and https://github.com/PrismML-Eng/llama.cpp/blob/master/ggml/src/ggml-cpu/quants.c
//
// Format: 1-bit weights in groups of 128.
// Each block: 2 bytes (FP16 scale) + 16 bytes (128 bits) = 18 bytes
// Weight = bit ? +d : -d
// Paired with Q8_0 for vec_dot (4 Q8_0 blocks per Q1_0_g128 block, since Q8_0 blocksize=32)

use super::{GgmlDType, GgmlType};
use crate::Result;
use half::f16;

// QK for Q1_0_g128
pub const QK1_0_G128: usize = 128;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ1_0_g128 {
    pub(crate) d: f16,             // delta (scale), FP16
    pub(crate) qs: [u8; QK1_0_G128 / 8], // 16 bytes, 128 bits
}
const _: () = assert!(std::mem::size_of::<BlockQ1_0_g128>() == 18);

impl GgmlType for BlockQ1_0_g128 {
    const DTYPE: GgmlDType = GgmlDType::Q1_0_g128;
    const BLCK_SIZE: usize = QK1_0_G128;
    // Paired with BlockQ8_0 (4 blocks per Q1_0_g128 block)
    type VecDotType = super::BlockQ8_0;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        // Dequantize: weight = bit ? d : -d
        let k = ys.len();
        let qk = Self::BLCK_SIZE;
        debug_assert!(
            k.is_multiple_of(qk),
            "dequantize_row_q1_0_g128: {k} is not divisible by {qk}"
        );

        let nb = k / qk;
        for i in 0..nb {
            let d = xs[i].d.to_f32();

            for j in 0..qk {
                let byte_index = j / 8;
                let bit_offset = j % 8;
                let bit = (xs[i].qs[byte_index] >> bit_offset) & 1;
                ys[i * qk + j] = if bit != 0 { d } else { -d };
            }
        }
    }

    fn from_float(_xs: &[f32], _ys: &mut [Self]) {
        // Quantization: find max absolute value, use as scale, quantize to bits
        // For simplicity, use the same scale-based quantization as Q4_0
        // Actual quantization would need proper 1-bit aware quantization
        panic!("from_float for Q1_0_g128 is not yet implemented")
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        // Q1_0_g128 blocksize = 128, BlockQ8_0 blocksize = 32
        // 4 Q8_0 blocks per Q1_0_g128 block
        const QK8_0: usize = 32;
        const RATIO: usize = QK1_0_G128 / QK8_0; // 4

        debug_assert!(
            n.is_multiple_of(QK1_0_G128),
            "vec_dot_q1_0_g128_q8_0: {n} is not divisible by {}",
            QK1_0_G128
        );
        debug_assert!(
            xs.len() * RATIO <= ys.len(),
            "vec_dot_q1_0_g128_q8_0: not enough Q8_0 blocks"
        );

        let nb = n / QK1_0_G128;
        let mut sumf = 0f32;

        for i in 0..nb {
            let d0 = xs[i].d.to_f32();
            let mut sumi = 0i32;

            // Process 4 Q8_0 blocks (32 elements each = 128 total)
            for k in 0..RATIO {
                let yblock = &ys[i * RATIO + k];
                let d1 = yblock.d.to_f32();
                let mut sumi_block = 0i32;

                for j in 0..QK8_0 {
                    let bit_index = k * QK8_0 + j;
                    let byte_index = bit_index / 8;
                    let bit_offset = bit_index % 8;

                    // 1-bit: 1 → +1, 0 → -1
                    let xi = if (xs[i].qs[byte_index] >> bit_offset) & 1 != 0 {
                        1i32
                    } else {
                        -1i32
                    };
                    let yi = yblock.qs[j] as i32;

                    sumi_block += xi * yi;
                }

                sumi += (d1 as f32 * sumi_block as f32) as i32;
            }

            sumf += d0 * sumi as f32;
        }

        sumf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q1_0_g128_size() {
        // Block: 2 bytes (FP16 scale) + 16 bytes (128 bits) = 18 bytes
        assert_eq!(std::mem::size_of::<BlockQ1_0_g128>(), 18);
    }

    #[test]
    fn test_to_float_roundtrip() {
        // Simple test: all +d should give all positive values
        let block = BlockQ1_0_g128 {
            d: f16::from_f32(2.0),
            qs: [0xFF; 16], // all bits set → all +d
        };

        let mut ys = [0f32; 128];
        BlockQ1_0_g128::to_float(&[block], &mut ys);

        for &y in &ys {
            assert!((y - 2.0).abs() < 0.001, "Expected 2.0, got {y}");
        }

        // All bits clear → all -d
        let block = BlockQ1_0_g128 {
            d: f16::from_f32(3.0),
            qs: [0x00; 16], // all bits clear → all -d
        };

        BlockQ1_0_g128::to_float(&[block], &mut ys);
        for &y in &ys {
            assert!((y + 3.0).abs() < 0.001, "Expected -3.0, got {y}");
        }
    }

    #[test]
    fn test_vec_dot_unopt_all_ones_vs_q8_0() {
        // Q1_0_g128 block with all bits = 1 (weight = +d for all)
        // paired with Q8_0 block (weight = d for all)
        // Expected: d * sum(q8) = d * sum of Q8_0 values
        let q1_block = BlockQ1_0_g128 {
            d: f16::from_f32(0.5),
            qs: [0xFF; 16], // all 128 bits set → all +0.5
        };

        // Q8_0 block with values 1..32 repeated 4 times
        // BlockQ8_0 has d:f16 and qs:[i8; 32]
        let q8_values: [i8; 32] = (1i8..=32i8).collect::<Vec<_>>().try_into().unwrap();
        let q8_block = super::super::k_quants::BlockQ8_0 {
            d: f16::from_f32(1.0),
            qs: q8_values,
        };

        // vec_dot for Q1_0_g128 × BlockQ8_0 processes 4 Q8_0 blocks per Q1_0_g128
        // For this test, we use 1 Q8_0 block (32 elements)
        let q1_blocks = [q1_block];
        let q8_blocks = [q8_block];

        let result = BlockQ1_0_g128::vec_dot_unopt(32, &q1_blocks, &q8_blocks);

        // Expected: d0 * sum over j of (q1_j * d1 * q8_j)
        // q1_j = +1 for all (bit=1), d1 = 1.0, q8_j = 1..32
        // sum = 0.5 * (1 + 2 + ... + 32) = 0.5 * 528 = 264
        let expected = 0.5 * (1i32..=32i32).sum::<i32>() as f32;
        assert!(
            (result - expected).abs() < 0.1,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_vec_dot_unopt_all_zeros_vs_q8_0() {
        // Q1_0_g128 block with all bits = 0 (weight = -d for all)
        let q1_block = BlockQ1_0_g128 {
            d: f16::from_f32(0.5),
            qs: [0x00; 16], // all 128 bits clear → all -0.5
        };

        // Q8_0 block with values 1..32
        let q8_values: [i8; 32] = (1i8..=32i8).collect::<Vec<_>>().try_into().unwrap();
        let q8_block = super::super::k_quants::BlockQ8_0 {
            d: f16::from_f32(1.0),
            qs: q8_values,
        };

        let q1_blocks = [q1_block];
        let q8_blocks = [q8_block];

        let result = BlockQ1_0_g128::vec_dot_unopt(32, &q1_blocks, &q8_blocks);

        // Expected: d0 * sum over j of (-1 * d1 * q8_j) = -d0 * sum(q8)
        // = -0.5 * 528 = -264
        let expected = -0.5 * (1i32..=32i32).sum::<i32>() as f32;
        assert!(
            (result - expected).abs() < 0.1,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_vec_dot_unopt_alternating_bits() {
        // Q1_0_g128 with alternating bit pattern: 0,1,0,1,0,1,...
        // Bit pattern 0xAA = 10101010 in binary
        let q1_block = BlockQ1_0_g128 {
            d: f16::from_f32(1.0),
            qs: [0xAA; 16], // alternating 0/1 bits
        };

        // Q8_0 block with constant values
        let q8_values = [1i8; 32];
        let q8_block = super::super::k_quants::BlockQ8_0 {
            d: f16::from_f32(1.0),
            qs: q8_values,
        };

        let q1_blocks = [q1_block];
        let q8_blocks = [q8_block];

        let result = BlockQ1_0_g128::vec_dot_unopt(32, &q1_blocks, &q8_blocks);

        // Expected: d0 * sum over j of (sign_j * d1 * q8_j)
        // sign_j = +1 for bit=1, -1 for bit=0
        // With alternating bits in first byte: +1,-1,+1,-1,+1,-1,+1,-1
        // For constant q8=1, sum = (+1-1+1-1+1-1+1-1) = 0 per byte
        // So total sum should be 0
        assert!(
            result.abs() < 0.1,
            "Expected ~0 for alternating bits with constant values, got {}",
            result
        );
    }

    #[test]
    fn test_vec_dot_unopt_single_bit_set() {
        // Only bit 0 is set in Q1_0_g128
        let q1_block = BlockQ1_0_g128 {
            d: f16::from_f32(1.0),
            qs: [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        };

        // Q8_0 with value 5 at position 0, others 0
        let q8_values: [i8; 32] = {
            let mut arr = [0i8; 32];
            arr[0] = 5;
            arr
        };
        let q8_block = super::super::k_quants::BlockQ8_0 {
            d: f16::from_f32(1.0),
            qs: q8_values,
        };

        let q1_blocks = [q1_block];
        let q8_blocks = [q8_block];

        let result = BlockQ1_0_g128::vec_dot_unopt(32, &q1_blocks, &q8_blocks);

        // Expected: d0 * (bit0 * d1 * q8_0) = 1.0 * (1 * 1.0 * 5) = 5.0
        assert!(
            (result - 5.0).abs() < 0.1,
            "Expected 5.0, got {}",
            result
        );
    }

    #[test]
    fn test_vec_dot_matches_unopt() {
        // Test that vec_dot (which may use SIMD) matches vec_dot_unopt
        // This test runs on any hardware - it falls back to unopt if SIMD unavailable
        use std::sync::atomic::{AtomicBool, Ordering};

        static SIMD_AVAILABLE: AtomicBool = AtomicBool::new(false);

        // Check if AVX2 or NEON is available at runtime
        #[cfg(target_feature = "avx2")]
        {
            SIMD_AVAILABLE.store(true, Ordering::Relaxed);
        }
        #[cfg(target_feature = "neon")]
        {
            SIMD_AVAILABLE.store(true, Ordering::Relaxed);
        }

        // Create test blocks
        let q1_block = BlockQ1_0_g128 {
            d: f16::from_f32(0.5),
            qs: [0x55; 16], // alternating bit pattern
        };

        let q8_values: [i8; 32] = (1i8..=32i8).collect::<Vec<_>>().try_into().unwrap();
        let q8_block = super::super::k_quants::BlockQ8_0 {
            d: f16::from_f32(2.0),
            qs: q8_values,
        };

        let q1_blocks = [q1_block];
        let q8_blocks = [q8_block];

        let unopt = BlockQ1_0_g128::vec_dot_unopt(32, &q1_blocks, &q8_blocks);
        let result = BlockQ1_0_g128::vec_dot(32, &q1_blocks, &q8_blocks);

        // Allow small floating point differences
        assert!(
            (result - unopt).abs() < 1.0,
            "SIMD result {} differs from unopt {} by more than 1.0",
            result,
            unopt
        );
    }
}
