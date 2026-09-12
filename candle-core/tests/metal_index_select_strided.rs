#![cfg(feature = "metal")]
//! Regression: index_select on a non-contiguous (strided) source on Metal.
//! The strided path passed the selected-dim size to get_strided_index where the
//! source rank was expected, so it read the wrong offsets whenever the rank
//! differed from that dim size -- silently returning wrong values vs the CPU
//! backend. (The pre-existing 2x2 test happened to have rank == dim size.)

use candle_core::{Device, Result, Tensor};

fn assert_matches_cpu(rows: usize, cols: usize, dim: usize, ids: &[u32]) -> Result<()> {
    let dev = Device::new_metal(0)?;
    let cpu = Device::Cpu;
    let data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
    // Metal source is (cols, rows) transposed -> (rows, cols) non-contiguous.
    // The CPU reference is the same logical tensor laid out contiguously.
    let m = Tensor::from_vec(data.clone(), (cols, rows), &dev)?.transpose(0, 1)?;
    let c = Tensor::from_vec(data, (cols, rows), &cpu)?
        .t()?
        .contiguous()?;
    assert!(!m.is_contiguous());
    let idx_m = Tensor::from_vec(ids.to_vec(), ids.len(), &dev)?;
    let idx_c = Tensor::from_vec(ids.to_vec(), ids.len(), &cpu)?;
    let got: Vec<f32> = m.index_select(&idx_m, dim)?.flatten_all()?.to_vec1()?;
    let want: Vec<f32> = c.index_select(&idx_c, dim)?.flatten_all()?.to_vec1()?;
    assert_eq!(got, want, "rows={rows} cols={cols} dim={dim} ids={ids:?}");
    Ok(())
}

#[test]
fn index_select_strided_matches_cpu() -> Result<()> {
    // rank (2) != selected-dim size -> exercises the previously-broken path.
    assert_matches_cpu(17, 40, 1, &[0, 3, 1, 39, 7, 39])?;
    assert_matches_cpu(17, 40, 0, &[0, 5, 16, 2, 16])?;
    assert_matches_cpu(3, 5, 1, &[4, 0, 2])?;
    assert_matches_cpu(8, 3, 0, &[7, 0, 3])?;
    Ok(())
}
