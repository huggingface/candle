use anyhow::Result;
use candle_core::{Device, IndexOp, StepRange, Tensor, Var};
use std::ops::Bound::{Excluded, Included, Unbounded};

#[test]
fn step_index_forward() -> Result<()> {
    let tensor = Tensor::arange(0u32, 8, &Device::Cpu)?;
    assert_eq!(
        tensor.i(StepRange::new(.., 2))?.to_vec1::<u32>()?,
        [0, 2, 4, 6]
    );
    assert_eq!(tensor.i(StepRange::new(1..7, 3))?.to_vec1::<u32>()?, [1, 4]);
    assert_eq!(
        tensor.i(StepRange::new(..=6, 3))?.to_vec1::<u32>()?,
        [0, 3, 6]
    );
    assert_eq!(
        tensor.i(StepRange::new(2..=6, 2))?.to_vec1::<u32>()?,
        [2, 4, 6]
    );
    assert_eq!(
        tensor.i(StepRange::new(2..=5, 2))?.to_vec1::<u32>()?,
        [2, 4]
    );
    assert_eq!(
        tensor
            .i(StepRange::new((Excluded(1), Excluded(7)), 2))?
            .to_vec1::<u32>()?,
        [2, 4, 6]
    );
    Ok(())
}

#[test]
#[allow(clippy::reversed_empty_ranges)] // Bounds are traversed in the signed step direction.
fn step_index_backward() -> Result<()> {
    let tensor = Tensor::arange(0i64, 8, &Device::Cpu)?;
    assert_eq!(
        tensor.i(StepRange::new(.., -2))?.to_vec1::<i64>()?,
        [7, 5, 3, 1]
    );
    assert_eq!(
        tensor.i(StepRange::new(6..1, -2))?.to_vec1::<i64>()?,
        [6, 4, 2]
    );
    assert_eq!(
        tensor.i(StepRange::new(6..=0, -3))?.to_vec1::<i64>()?,
        [6, 3, 0]
    );
    assert_eq!(
        tensor.i(StepRange::new(6..=1, -2))?.to_vec1::<i64>()?,
        [6, 4, 2]
    );
    assert_eq!(
        tensor.i(StepRange::new(..2, -2))?.to_vec1::<i64>()?,
        [7, 5, 3]
    );
    assert_eq!(
        tensor.i(StepRange::new(..=3, -2))?.to_vec1::<i64>()?,
        [7, 5, 3]
    );
    assert_eq!(
        tensor
            .i(StepRange::new((Excluded(7), Included(0)), -2))?
            .to_vec1::<i64>()?,
        [6, 4, 2, 0]
    );
    assert_eq!(tensor.i(StepRange::new(0.., -1))?.to_vec1::<i64>()?, [0]);
    Ok(())
}

#[test]
#[allow(clippy::reversed_empty_ranges)] // Reversed forward bounds intentionally produce no indices.
fn step_index_empty_and_extreme_steps() -> Result<()> {
    let tensor = Tensor::arange(0u32, 6, &Device::Cpu)?;
    for range in [
        StepRange::new(3..3, 2),
        StepRange::new(3..1, 2),
        StepRange::new(1..3, -2),
        StepRange::new(0..0, -1),
        StepRange::new((Excluded(0), Unbounded), -1),
        StepRange::new(6.., 1),
    ] {
        let out = tensor.i(range)?;
        assert_eq!(out.dims(), [0]);
        assert!(out.to_vec1::<u32>()?.is_empty());
    }
    assert_eq!(
        tensor
            .i(StepRange::new(1.., isize::MAX))?
            .to_vec1::<u32>()?,
        [1]
    );
    assert_eq!(
        tensor.i(StepRange::new(.., isize::MIN))?.to_vec1::<u32>()?,
        [5]
    );
    let empty = Tensor::zeros((0, 3), candle_core::DType::F32, &Device::Cpu)?;
    for step in [-2, -1, 1, 2] {
        let out = empty.i(StepRange::new(.., step))?;
        assert_eq!(out.dims(), [0, 3]);
        assert!(out.to_vec2::<f32>()?.is_empty());
    }
    let matrix = tensor.reshape((2, 3))?;
    assert_eq!(matrix.i((.., StepRange::new(1..1, -1)))?.dims(), [2, 0]);
    let empty = Tensor::zeros((3, 0, 2), candle_core::DType::F32, &Device::Cpu)?;
    for step in [-2, 2] {
        let out = empty.i(StepRange::new(.., step))?;
        assert_eq!(out.dims(), [2, 0, 2]);
        assert!(out.flatten_all()?.to_vec1::<f32>()?.is_empty());
    }
    let empty = Tensor::zeros((usize::MAX, 0), candle_core::DType::F32, &Device::Cpu)?;
    assert_eq!(empty.i(StepRange::new(.., 1))?.dims(), [usize::MAX, 0]);
    for step in [-2, 2] {
        let out = empty.i(StepRange::new(.., step))?;
        assert_eq!(out.dims(), [usize::MAX / 2 + 1, 0]);
    }
    Ok(())
}

#[test]
fn step_index_invalid_args() -> Result<()> {
    let tensor = Tensor::arange(0u32, 6, &Device::Cpu)?;
    for range in [
        StepRange::new(.., 0),
        StepRange::new(7.., 1),
        StepRange::new(..7, 1),
        StepRange::new(..=6, 1),
        StepRange::new(6.., -1),
        StepRange::new(..6, -1),
        StepRange::new(..=6, -1),
        StepRange::new((Excluded(usize::MAX), Unbounded), 1),
        StepRange::new(..=usize::MAX, 1),
        StepRange::new(usize::MAX.., -1),
    ] {
        let description = format!("{range:?}");
        assert!(tensor.i(range).is_err(), "accepted invalid {description}");
    }
    assert!(tensor.i((.., StepRange::new(.., 2))).is_err());
    let scalar = Tensor::new(1u32, &Device::Cpu)?;
    assert!(scalar.i(StepRange::new(.., -1)).is_err());
    Ok(())
}

#[test]
fn step_index_mixed_dims_and_strides() -> Result<()> {
    let tensor = Tensor::arange(0u32, 24, &Device::Cpu)?.reshape((2, 3, 4))?;
    let out = tensor.i((1, StepRange::new(.., -1), StepRange::new(.., 2)))?;
    assert_eq!(out.dims(), [3, 2]);
    assert_eq!(out.to_vec2::<u32>()?, [[20, 22], [16, 18], [12, 14]]);
    let out = tensor.i((StepRange::new(.., -1), 1, 1..3))?;
    assert_eq!(out.to_vec2::<u32>()?, [[17, 18], [5, 6]]);
    let indexes = Tensor::new(&[2u32, 0], &Device::Cpu)?;
    let out = tensor.i((0, &indexes, StepRange::new(.., -2)))?;
    assert_eq!(out.to_vec2::<u32>()?, [[11, 9], [3, 1]]);
    let matrix = Tensor::arange(0u32, 20, &Device::Cpu)?.reshape((4, 5))?;
    let transposed = matrix.t()?.narrow(0, 1, 3)?;
    let out = transposed.i((StepRange::new(.., -2), StepRange::new(1.., 2)))?;
    assert_eq!(out.to_vec2::<u32>()?, [[8, 18], [6, 16]]);
    assert_eq!(
        matrix.i(StepRange::new(.., 1))?.to_vec2::<u32>()?,
        matrix.to_vec2::<u32>()?
    );
    Ok(())
}

#[test]
fn step_index_matches_reference() -> Result<()> {
    for len in 1..8u32 {
        let tensor = Tensor::arange(0u32, len, &Device::Cpu)?;
        for start in 0..len {
            for end in 0..len {
                for step in [-3isize, -2, -1, 1, 2, 3] {
                    let expected: Vec<u32> = if step > 0 {
                        (start..end).step_by(step as usize).collect()
                    } else if start > end {
                        ((end + 1)..=start)
                            .rev()
                            .step_by(step.unsigned_abs())
                            .collect()
                    } else {
                        Vec::new()
                    };
                    let out = tensor.i(StepRange::new(start as usize..end as usize, step))?;
                    assert_eq!(
                        out.to_vec1::<u32>()?,
                        expected,
                        "{len}: {start}..{end}, step {step}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn step_index_grad() -> Result<()> {
    let var = Var::new(&[[1f32, 2., 3., 4.], [5., 6., 7., 8.]], &Device::Cpu)?;
    let out = var.i((StepRange::new(.., -1), StepRange::new(.., -2)))?;
    let grads = out.sqr()?.sum_all()?.backward()?;
    let grad = grads.get(&var).expect("gradient for stepped input");
    assert_eq!(
        grad.to_vec2::<f32>()?,
        [[0., 4., 0., 8.], [0., 12., 0., 16.]]
    );
    let out = var.t()?.i((StepRange::new(.., -2), ..))?;
    let grads = out.sqr()?.sum_all()?.backward()?;
    let grad = grads.get(&var).expect("gradient for strided stepped input");
    assert_eq!(
        grad.to_vec2::<f32>()?,
        [[0., 4., 0., 8.], [0., 12., 0., 16.]]
    );
    let empty = var.i((.., StepRange::new(1..1, 2)))?;
    let grads = empty.sum_all()?.backward()?;
    let grad = grads.get(&var).expect("gradient for empty stepped input");
    assert_eq!(grad.to_vec2::<f32>()?, [[0.; 4]; 2]);
    Ok(())
}

#[test]
fn integer_index() -> Result<()> {
    let dev = Device::Cpu;

    let tensor = Tensor::arange(0u32, 2 * 3, &dev)?.reshape((2, 3))?;
    let result = tensor.i(1)?;
    assert_eq!(result.dims(), &[3]);
    assert_eq!(result.to_vec1::<u32>()?, &[3, 4, 5]);

    let result = tensor.i((.., 2))?;
    assert_eq!(result.dims(), &[2]);
    assert_eq!(result.to_vec1::<u32>()?, &[2, 5]);

    Ok(())
}

#[test]
fn range_index() -> Result<()> {
    let dev = Device::Cpu;
    // RangeFull
    let tensor = Tensor::arange(0u32, 2 * 3, &dev)?.reshape((2, 3))?;
    let result = tensor.i(..)?;
    assert_eq!(result.dims(), &[2, 3]);
    assert_eq!(result.to_vec2::<u32>()?, &[[0, 1, 2], [3, 4, 5]]);

    // Range
    let tensor = Tensor::arange(0u32, 4 * 3, &dev)?.reshape((4, 3))?;
    let result = tensor.i(1..3)?;
    assert_eq!(result.dims(), &[2, 3]);
    assert_eq!(result.to_vec2::<u32>()?, &[[3, 4, 5], [6, 7, 8]]);

    // RangeFrom
    let result = tensor.i(2..)?;
    assert_eq!(result.dims(), &[2, 3]);
    assert_eq!(result.to_vec2::<u32>()?, &[[6, 7, 8], [9, 10, 11]]);

    // RangeTo
    let result = tensor.i(..2)?;
    assert_eq!(result.dims(), &[2, 3]);
    assert_eq!(result.to_vec2::<u32>()?, &[[0, 1, 2], [3, 4, 5]]);

    // RangeInclusive
    let result = tensor.i(1..=2)?;
    assert_eq!(result.dims(), &[2, 3]);
    assert_eq!(result.to_vec2::<u32>()?, &[[3, 4, 5], [6, 7, 8]]);

    // RangeTo
    let result = tensor.i(..1)?;
    assert_eq!(result.dims(), &[1, 3]);
    assert_eq!(result.to_vec2::<u32>()?, &[[0, 1, 2]]);

    // RangeToInclusive
    let result = tensor.i(..=1)?;
    assert_eq!(result.dims(), &[2, 3]);
    assert_eq!(result.to_vec2::<u32>()?, &[[0, 1, 2], [3, 4, 5]]);

    // Empty range
    let result = tensor.i(1..1)?;
    assert_eq!(result.dims(), &[0, 3]);
    let empty: [[u32; 3]; 0] = [];
    assert_eq!(result.to_vec2::<u32>()?, &empty);

    // Similar to PyTorch, allow empty ranges when the computed length is negative.
    #[allow(clippy::reversed_empty_ranges)]
    let result = tensor.i(1..0)?;
    assert_eq!(result.dims(), &[0, 3]);
    let empty: [[u32; 3]; 0] = [];
    assert_eq!(result.to_vec2::<u32>()?, &empty);
    Ok(())
}

#[test]
fn index_3d() -> Result<()> {
    let tensor = Tensor::from_iter(0..24u32, &Device::Cpu)?.reshape((2, 3, 4))?;
    assert_eq!(tensor.i((0, 0, 0))?.to_scalar::<u32>()?, 0);
    assert_eq!(tensor.i((1, 0, 0))?.to_scalar::<u32>()?, 12);
    assert_eq!(tensor.i((0, 1, 0))?.to_scalar::<u32>()?, 4);
    assert_eq!(tensor.i((0, 1, 3))?.to_scalar::<u32>()?, 7);
    assert_eq!(tensor.i((0..2, 0, 0))?.to_vec1::<u32>()?, &[0, 12]);
    assert_eq!(
        tensor.i((0..2, .., 0))?.to_vec2::<u32>()?,
        &[[0, 4, 8], [12, 16, 20]]
    );
    assert_eq!(
        tensor.i((..2, .., 3))?.to_vec2::<u32>()?,
        &[[3, 7, 11], [15, 19, 23]]
    );
    assert_eq!(tensor.i((1, .., 3))?.to_vec1::<u32>()?, &[15, 19, 23]);
    Ok(())
}

#[test]
fn slice_assign() -> Result<()> {
    let dev = Device::Cpu;

    let tensor = Tensor::arange(0u32, 4 * 5, &dev)?.reshape((4, 5))?;
    let src = Tensor::arange(0u32, 2 * 3, &dev)?.reshape((3, 2))?;
    let out = tensor.slice_assign(&[1..4, 3..5], &src)?;
    assert_eq!(
        out.to_vec2::<u32>()?,
        &[
            [0, 1, 2, 3, 4],
            [5, 6, 7, 0, 1],
            [10, 11, 12, 2, 3],
            [15, 16, 17, 4, 5]
        ]
    );
    let out = tensor.slice_assign(&[0..3, 0..2], &src)?;
    assert_eq!(
        out.to_vec2::<u32>()?,
        &[
            [0, 1, 2, 3, 4],
            [2, 3, 7, 8, 9],
            [4, 5, 12, 13, 14],
            [15, 16, 17, 18, 19]
        ]
    );
    Ok(())
}
