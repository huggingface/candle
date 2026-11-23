use candle::{DType, Device, Result, Tensor};

// With zero Q/K, attention must average V over exactly the permitted keys.
// This checks masking against an analytic reference, independent of FA2.
#[test]
fn local_windows_match_uniform_attention_reference() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let lengths = [1usize, 7, 65, 129];
    let heads = 4;
    let kv_heads = 2;
    let total: usize = lengths.iter().sum();
    let mut offsets = vec![0u32];
    for length in lengths {
        offsets.push(offsets.last().unwrap() + length as u32);
    }
    let seqlens = Tensor::from_vec(offsets, lengths.len() + 1, &device)?;
    let windows = [
        (None, None),
        (None, Some(0)),
        (Some(2), Some(3)),
        (Some(4), Some(0)),
        (None, Some(3)),
        (Some(2), None),
    ];
    for dtype in [DType::F16, DType::BF16] {
        let tolerance = if dtype == DType::F16 { 0.03125 } else { 0.125 };
        for width in [64, 128] {
            let q = Tensor::zeros((total, heads, width), dtype, &device)?;
            let k = Tensor::zeros((total, kv_heads, width), dtype, &device)?;
            let mut values = Vec::new();
            for (sequence, length) in lengths.into_iter().enumerate() {
                for position in 0..length {
                    for head in 0..kv_heads {
                        values.extend(std::iter::repeat_n(
                            (position % 17) as f32 + head as f32 * 0.25 + sequence as f32 * 0.5,
                            width,
                        ));
                    }
                }
            }
            let v = Tensor::from_vec(values, (total, kv_heads, width), &device)?.to_dtype(dtype)?;
            for (left, right) in windows {
                let y = candle_flash_attn_v3::flash_attn_varlen_windowed(
                    &q, &k, &v, &seqlens, &seqlens, 129, 129, 1., left, right, false,
                )?
                .to_dtype(DType::F32)?
                .to_vec3::<f32>()?;
                let mut offset = 0;
                for (sequence, length) in lengths.into_iter().enumerate() {
                    for position in 0..length {
                        let begin = left.map_or(0, |l| position.saturating_sub(l));
                        let end = right.map_or(length - 1, |r| (position + r).min(length - 1));
                        for head in 0..heads {
                            let expected = (begin..=end)
                                .map(|p| {
                                    (p % 17) as f32
                                        + (head / 2) as f32 * 0.25
                                        + sequence as f32 * 0.5
                                })
                                .sum::<f32>()
                                / (end - begin + 1) as f32;
                            for &actual in &y[offset + position][head] {
                                assert!(
                                    actual.is_finite() && (actual - expected).abs() <= tolerance,
                                    "{dtype:?} width={width} length={length} position={position} window={left:?}/{right:?}: {actual} != {expected}"
                                );
                            }
                        }
                    }
                    offset += length;
                }
            }
        }
    }
    Ok(())
}
