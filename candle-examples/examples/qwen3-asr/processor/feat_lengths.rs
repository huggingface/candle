//! Feature-length math used for placeholder expansion and audio encoder shapes.
//!
//! Mirrors `_get_feat_extract_output_lengths` from the official Python processor:
//! `../../../Qwen3-ASR/qwen_asr/core/transformers_backend/processing_qwen3_asr.py`.

/// Python-style floor division for signed integers.
///
/// Rust truncates division toward zero; Python floors toward -inf.
fn div_floor(a: i64, b: i64) -> i64 {
    let q = a / b;
    let r = a % b;
    if r != 0 && ((r > 0) != (b > 0)) {
        q - 1
    } else {
        q
    }
}

/// Compute the audio encoder output length for a single input length.
///
/// This is used to expand the `<|audio|>` placeholder token to the exact number
/// of audio embeddings produced by the audio tower.
pub fn feat_extract_output_length(input_len: usize) -> usize {
    let input_len = i64::try_from(input_len).unwrap_or(i64::MAX);
    let leave = input_len % 100;
    let feat_len = div_floor(leave - 1, 2) + 1;
    let base = div_floor(div_floor(feat_len - 1, 2), 2) + 1;
    let extra = div_floor(input_len, 100) * 13;
    let out = base + extra;
    usize::try_from(out).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::feat_extract_output_length;

    #[test]
    fn test_feat_extract_output_length_known_values() -> anyhow::Result<()> {
        // Values sanity-checked against the Python formula by manual evaluation.
        let cases = [
            (0usize, 0usize),
            (1usize, 1usize),
            (2usize, 1usize),
            (3usize, 1usize),
            (4usize, 1usize),
            (5usize, 1usize),
            (99usize, 13usize),
            (100usize, 13usize),
            (101usize, 14usize),
            (199usize, 26usize),
            (200usize, 26usize),
        ];

        for (inp, exp) in cases {
            let got = feat_extract_output_length(inp);
            if got != exp {
                anyhow::bail!("input_len={inp}: expected {exp}, got {got}");
            }
        }
        Ok(())
    }
}
