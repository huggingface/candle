#![cfg(feature = "metal")]
//! Regression tests for Metal `arg_sort` on rows longer than a single
//! threadgroup (issue #2570). The bitonic kernel sorts a whole row inside one
//! threadgroup and silently returned wrong indices past ~1024 elements; large
//! ascending F32/U32 sorts now go through the multi-block MLX kernel, and the
//! cases that remain unsupported error instead of corrupting the output.

use candle_core::{DType, Device, Result, Tensor};

// Assert that `idx` is a valid ascending argsort of `values`: the indices form
// a permutation of 0..n and gather the values in non-decreasing order.
fn assert_valid_ascending_argsort(values: &[f32], idx: &[u32]) {
    assert_eq!(values.len(), idx.len(), "index length mismatch");
    let n = values.len();
    let mut seen = vec![false; n];
    let mut prev = f32::NEG_INFINITY;
    for &i in idx {
        let i = i as usize;
        assert!(i < n, "index {i} out of range for n={n}");
        assert!(!seen[i], "index {i} returned twice (not a permutation)");
        seen[i] = true;
        let v = values[i];
        assert!(v >= prev, "not sorted ascending: {v} < {prev}");
        prev = v;
    }
}

#[test]
fn argsort_large_ascending_f32() -> Result<()> {
    let device = Device::new_metal(0)?;
    // Straddle the single-threadgroup limit and the block/multi-block MLX paths.
    for n in [1024usize, 1025, 2048, 4096, 4097, 130_000] {
        let d = Tensor::rand(-256f32, 255., (1, n), &device)?;
        let idx = d.arg_sort_last_dim(true)?;
        let values: Vec<f32> = d.get(0)?.to_vec1()?;
        let order: Vec<u32> = idx.get(0)?.to_vec1()?;
        assert_valid_ascending_argsort(&values, &order);
    }
    Ok(())
}

#[test]
fn argsort_large_ascending_multi_row_f32() -> Result<()> {
    // nrows > 1: each row must be sorted independently.
    let device = Device::new_metal(0)?;
    let (rows, n) = (3usize, 5000usize);
    let d = Tensor::rand(-10f32, 10., (rows, n), &device)?;
    let idx = d.arg_sort_last_dim(true)?;
    for r in 0..rows {
        let values: Vec<f32> = d.get(r)?.to_vec1()?;
        let order: Vec<u32> = idx.get(r)?.to_vec1()?;
        assert_valid_ascending_argsort(&values, &order);
    }
    Ok(())
}

#[test]
fn argsort_large_ascending_u32() -> Result<()> {
    let device = Device::new_metal(0)?;
    let n = 3000usize;
    let vals: Vec<u32> = (0..n as u32).rev().collect(); // strictly descending input
    let d = Tensor::from_vec(vals, (1, n), &device)?;
    let idx = d.arg_sort_last_dim(true)?;
    let order: Vec<u32> = idx.get(0)?.to_vec1()?;
    // Ascending argsort of a reversed range is the reversed index range.
    let expected: Vec<u32> = (0..n as u32).rev().collect();
    assert_eq!(order, expected);
    Ok(())
}

#[test]
fn argsort_large_descending_errors_not_corrupts() -> Result<()> {
    // Descending large sorts are not supported by the multi-block path; they must
    // return a clear error rather than silently producing wrong indices.
    let device = Device::new_metal(0)?;
    let d = Tensor::rand(-1f32, 1., (1, 2048), &device)?;
    let res = d.arg_sort_last_dim(false);
    assert!(
        res.is_err(),
        "expected an error for a large descending Metal argsort, got Ok"
    );
    Ok(())
}

#[test]
fn argsort_large_unsupported_dtype_errors() -> Result<()> {
    // A dtype the multi-block path does not cover (F16) must error above the
    // threadgroup limit instead of returning garbage.
    let device = Device::new_metal(0)?;
    let d = Tensor::rand(-1f32, 1., (1, 2048), &device)?.to_dtype(DType::F16)?;
    let res = d.arg_sort_last_dim(true);
    assert!(
        res.is_err(),
        "expected an error for a large F16 Metal argsort, got Ok"
    );
    Ok(())
}

#[test]
fn argsort_small_still_works_all_paths() -> Result<()> {
    // The <=1024 bitonic path must be unaffected, ascending and descending.
    let device = Device::new_metal(0)?;
    let d = Tensor::rand(-1f32, 1., (1, 512), &device)?;
    let values: Vec<f32> = d.get(0)?.to_vec1()?;

    let asc: Vec<u32> = d.arg_sort_last_dim(true)?.get(0)?.to_vec1()?;
    assert_valid_ascending_argsort(&values, &asc);

    // Descending: values gathered in non-increasing order, still a permutation.
    let desc: Vec<u32> = d.arg_sort_last_dim(false)?.get(0)?.to_vec1()?;
    let neg: Vec<f32> = values.iter().map(|v| -v).collect();
    assert_valid_ascending_argsort(&neg, &desc);
    Ok(())
}
