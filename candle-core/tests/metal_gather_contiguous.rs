#![cfg(feature = "metal")]
//! Metal gather indexes the source assuming a contiguous layout; a strided
//! source silently returned wrong values. It now errors like the CPU backend.
use candle_core::{Device, Result, Tensor};

#[test]
fn gather_strided_source_errors_like_cpu() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let (r, c) = (4usize, 6usize);
    let data: Vec<f32> = (0..r * c).map(|i| i as f32).collect();
    let strided = Tensor::from_vec(data, (c, r), &dev)?.transpose(0, 1)?; // (r,c), non-contiguous
    assert!(!strided.is_contiguous());
    let ids: Vec<u32> = (0..r * c).map(|i| ((i * 5 + 2) % c) as u32).collect();
    let idx = Tensor::from_vec(ids, (r, c), &dev)?;
    // Was: silently wrong. Now: a clear error (as on CPU).
    assert!(strided.gather(&idx, 1).is_err());
    // A contiguous source still gathers correctly.
    let out: Vec<f32> = strided
        .contiguous()?
        .gather(&idx, 1)?
        .flatten_all()?
        .to_vec1()?;
    let cpu = Device::Cpu;
    let sc = Tensor::from_vec(
        (0..r * c).map(|i| i as f32).collect::<Vec<_>>(),
        (c, r),
        &cpu,
    )?
    .t()?
    .contiguous()?;
    let ic = Tensor::from_vec(
        (0..r * c)
            .map(|i| ((i * 5 + 2) % c) as u32)
            .collect::<Vec<_>>(),
        (r, c),
        &cpu,
    )?;
    let want: Vec<f32> = sc.gather(&ic, 1)?.flatten_all()?.to_vec1()?;
    assert_eq!(out, want);
    Ok(())
}
