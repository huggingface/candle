//! The shared-memory transpose fast path in `copy_strided_src`.
//!
//! Two things to pin: [`super::ops_transpose::as_transpose2d`] must recognise
//! exactly the layouts the kernel can serve, and the kernel must agree with the
//! generic path it pre-empts — including at sizes leaving a partial tile, where
//! its bounds checks are all that stand between it and a wrong answer.

use super::ops_transpose::as_transpose2d;
use super::tests::rocm_device;
use crate::{DType, Device, IndexOp, Layout, Result, Shape, Tensor};

fn layout(dims: &[usize], strides: &[usize]) -> Layout {
    Layout::new(Shape::from(dims.to_vec()), strides.to_vec(), 0)
}

#[test]
fn as_transpose2d_accepts_a_swapped_tail_and_rejects_everything_else() {
    // The plain 2D case: the view is (3, 5), memory holds (5, 3).
    assert_eq!(as_transpose2d(&layout(&[3, 5], &[1, 3])), Some((1, 3, 5)));
    // Batched, with the matrices packed back to back.
    assert_eq!(
        as_transpose2d(&layout(&[4, 3, 5], &[15, 1, 3])),
        Some((4, 3, 5))
    );
    assert_eq!(
        as_transpose2d(&layout(&[2, 4, 3, 5], &[60, 15, 1, 3])),
        Some((8, 3, 5))
    );
    // A batch dimension of extent 1 carries an arbitrary stride.
    assert_eq!(
        as_transpose2d(&layout(&[1, 3, 5], &[999, 1, 3])),
        Some((1, 3, 5))
    );

    // Contiguous — handled before this is ever reached, and not a transpose.
    assert_eq!(as_transpose2d(&layout(&[3, 5], &[5, 1])), None);
    // A transposed tail sitting on a *broadcast* batch is not a contiguous
    // stack of matrices, so one launch cannot cover it.
    assert_eq!(as_transpose2d(&layout(&[4, 3, 5], &[0, 1, 3])), None);
    // Gaps between the matrices, e.g. a narrowed batch.
    assert_eq!(as_transpose2d(&layout(&[4, 3, 5], &[30, 1, 3])), None);
    // The last two dims are permuted, but so is the row stride.
    assert_eq!(as_transpose2d(&layout(&[3, 5], &[2, 6])), None);
    // Fewer than two dimensions, and an empty tensor.
    assert_eq!(as_transpose2d(&layout(&[5], &[1])), None);
    assert_eq!(as_transpose2d(&layout(&[0, 5], &[1, 0])), None);
}

/// The kernel against the CPU backend, over every dtype `copy_strided_src`
/// dispatches on and over sizes that do and do not fill a 32x32 tile.
#[test]
fn transposed_contiguous_matches_the_cpu() -> Result<()> {
    let dev = rocm_device!();
    // 1 and 33 straddle the tile edge in both directions; 64 is exact.
    for (rows, cols) in [(1usize, 7usize), (7, 1), (33, 31), (32, 32), (64, 96)] {
        // Small non-negative integers, so u8 through f64 all hold the value
        // exactly and a mismatch can only be the transpose.
        let vals: Vec<f32> = (0..rows * cols).map(|i| (i % 251) as f32).collect();
        let base = Tensor::from_slice(&vals, (rows, cols), &Device::Cpu)?;
        let want = base.t()?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        for dtype in [
            DType::U8,
            DType::U32,
            DType::I64,
            DType::BF16,
            DType::F16,
            DType::F32,
            DType::F64,
        ] {
            let got = base
                .to_device(&dev)?
                .to_dtype(dtype)?
                .t()?
                .contiguous()?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(got, want, "{dtype:?} at {rows}x{cols}");
        }
    }
    Ok(())
}

/// Batched, and reached through the op every model actually uses.
#[test]
fn transposed_batch_matches_the_cpu() -> Result<()> {
    let dev = rocm_device!();
    let dims = (2usize, 3usize, 40usize, 17usize);
    let cpu = Tensor::rand(-1f32, 1f32, dims, &Device::Cpu)?;
    let want = cpu
        .transpose(2, 3)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let got = cpu
        .to_device(&dev)?
        .transpose(2, 3)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(got, want);
    Ok(())
}

/// A start offset, which the fast path has to carry into the source pointer —
/// and a narrowed batch, which it has to *decline* because the matrices are no
/// longer packed.
#[test]
fn transposed_views_of_a_slice_match_the_cpu() -> Result<()> {
    let dev = rocm_device!();
    let cpu = Tensor::rand(-1f32, 1f32, (5usize, 24usize, 9usize), &Device::Cpu)?;
    for (start, len) in [(0usize, 5usize), (2, 3), (1, 1)] {
        let want = cpu
            .narrow(0, start, len)?
            .transpose(1, 2)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let got = cpu
            .to_device(&dev)?
            .narrow(0, start, len)?
            .transpose(1, 2)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(got, want, "narrow({start}, {len})");
    }
    // Narrowing the *matrix* leaves gaps the single launch cannot express, so
    // this has to come back through the generic path with the same answer.
    let want = cpu
        .i((.., 2..20, ..))?
        .transpose(1, 2)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let got = cpu
        .to_device(&dev)?
        .i((.., 2..20, ..))?
        .transpose(1, 2)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(got, want);
    Ok(())
}

/// GB/s moved (read plus write). Ignored by default, like the quantized
/// harness:
///
/// ```text
/// cargo test -p candle-core --features rocm --lib --release -- \
///     rocm_backend::tests_transpose::bench --ignored --nocapture
/// ```
#[test]
#[ignore = "benchmark: needs a GPU and takes seconds"]
fn bench_transpose_against_the_strided_copy() -> Result<()> {
    let dev = rocm_device!();
    println!(
        "{:>6} {:>7} {:>6} {:>10} {:>10}",
        "dtype", "rows", "cols", "ms", "GB/s"
    );
    for (rows, cols) in [(2048usize, 2048usize), (8192, 2048), (2048, 8192)] {
        for dtype in [DType::F16, DType::F32] {
            let t = Tensor::rand(-1f32, 1f32, (rows, cols), &dev)?
                .to_dtype(dtype)?
                .t()?;
            for _ in 0..3 {
                let _ = t.contiguous()?;
            }
            dev.synchronize()?;
            let start = std::time::Instant::now();
            const ITERS: usize = 50;
            for _ in 0..ITERS {
                let _ = t.contiguous()?;
            }
            dev.synchronize()?;
            let ms = start.elapsed().as_secs_f64() * 1e3 / ITERS as f64;
            let bytes = 2. * (rows * cols * dtype.size_in_bytes()) as f64;
            println!(
                "{:>6} {rows:>7} {cols:>6} {ms:>10.4} {:>10.1}",
                format!("{dtype:?}"),
                bytes / (ms * 1e-3) / 1e9
            );
        }
    }
    Ok(())
}
