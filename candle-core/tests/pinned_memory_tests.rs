#![cfg(feature = "pinned-memory")]

use candle_core::{utils, Device, Result, Tensor};

#[test]
fn pinned_memory_roundtrip() -> Result<()> {
    if !utils::cuda_is_available() {
        return Ok(());
    }

    let device = Device::new_cuda(0)?;
    let len = 32_usize;

    // Allocate pinned memory and fill it with data
    let mut pinned_in = unsafe { device.cuda_alloc_pinned::<f32>(len)? };
    let expected: Vec<f32> = (0..len).map(|v| v as f32 * 2.5).collect();
    pinned_in
        .as_mut_slice()
        .expect("pinned slice")
        .copy_from_slice(&expected);

    // Create a CPU tensor from pinned memory using the new API
    let tensor_cpu = Tensor::new_pinned(&pinned_in)?;
    drop(pinned_in);

    // Verify the CPU tensor contains the correct data
    let tensor_data = tensor_cpu.to_vec1::<f32>()?;
    assert_eq!(tensor_data.len(), expected.len());
    for (i, (actual, expected_val)) in tensor_data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected_val).abs() < 1e-5,
            "Mismatch at index {}: got {}, expected {}",
            i,
            actual,
            expected_val
        );
    }

    // Run some tensor operations to ensure it works normally
    let sum = tensor_cpu.sum_all()?.to_scalar::<f32>()?;
    let expected_sum: f32 = expected.iter().copied().sum();
    assert!((sum - expected_sum).abs() < 1e-4);

    // Test roundtrip: create a CUDA tensor from pinned memory, then copy back
    let mut pinned_in2 = unsafe { device.cuda_alloc_pinned::<f32>(len)? };
    pinned_in2
        .as_mut_slice()
        .expect("pinned slice")
        .copy_from_slice(&expected);
    let tensor_cuda = Tensor::new_pinned(&pinned_in2)?;
    let tensor_cuda = tensor_cuda.to_device(&device)?;
    drop(pinned_in2);
    
    let mut pinned_out = unsafe { device.cuda_alloc_pinned::<f32>(len)? };
    tensor_cuda.copy_to_pinned_host(&mut pinned_out)?;
    let host = pinned_out.as_slice().expect("pinned slice");
    assert_eq!(host.len(), expected.len());
    for (i, (actual, expected_val)) in host.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected_val).abs() < 1e-5,
            "Roundtrip mismatch at index {}: got {}, expected {}",
            i,
            actual,
            expected_val
        );
    }

    Ok(())
}

#[test]
#[cfg(all(feature = "pinned-memory", feature = "cuda"))]
fn cpu_storage_operations_work_as_usual() -> Result<()> {
    // This test verifies that CpuStorage operations work correctly
    // with pinned memory backed tensors. It creates tensors from pinned memory
    // and ensures all CPU operations work as expected.

    if !utils::cuda_is_available() {
        return Ok(());
    }

    let cuda_device = Device::new_cuda(0)?;
    
    // Test zeros - create from pinned memory
    let zeros_len = 3 * 4;
    let mut pinned_zeros = unsafe { cuda_device.cuda_alloc_pinned::<f32>(zeros_len)? };
    pinned_zeros
        .as_mut_slice()
        .expect("pinned slice")
        .fill(0.0);
    let zeros = Tensor::from_pinned_host(&pinned_zeros, (3, 4))?;
    drop(pinned_zeros);
    let zeros_data = zeros.to_vec2::<f32>()?;
    assert_eq!(zeros_data.len(), 3);
    assert_eq!(zeros_data[0].len(), 4);
    for row in &zeros_data {
        for &val in row {
            assert_eq!(val, 0.0);
        }
    }

    // Test ones - create from pinned memory
    let ones_len = 2 * 3;
    let mut pinned_ones = unsafe { cuda_device.cuda_alloc_pinned::<f32>(ones_len)? };
    pinned_ones
        .as_mut_slice()
        .expect("pinned slice")
        .fill(1.0);
    let ones = Tensor::from_pinned_host(&pinned_ones, (2, 3))?;
    drop(pinned_ones);
    let ones_data = ones.to_vec2::<f32>()?;
    assert_eq!(ones_data.len(), 2);
    assert_eq!(ones_data[0].len(), 3);
    for row in &ones_data {
        for &val in row {
            assert_eq!(val, 1.0);
        }
    }

    // Test from_pinned_host with data
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut pinned_data = unsafe { cuda_device.cuda_alloc_pinned::<f32>(data.len())? };
    pinned_data
        .as_mut_slice()
        .expect("pinned slice")
        .copy_from_slice(&data);
    let tensor = Tensor::from_pinned_host(&pinned_data, (2, 3))?;
    drop(pinned_data);
    let tensor_data = tensor.to_vec2::<f32>()?;
    assert_eq!(tensor_data, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    // Test arithmetic operations - create from pinned memory
    let a_data = [1.0f32, 2.0, 3.0];
    let mut pinned_a = unsafe { cuda_device.cuda_alloc_pinned::<f32>(a_data.len())? };
    pinned_a
        .as_mut_slice()
        .expect("pinned slice")
        .copy_from_slice(&a_data);
    let a = Tensor::new_pinned(&pinned_a)?;
    drop(pinned_a);

    let b_data = [4.0f32, 5.0, 6.0];
    let mut pinned_b = unsafe { cuda_device.cuda_alloc_pinned::<f32>(b_data.len())? };
    pinned_b
        .as_mut_slice()
        .expect("pinned slice")
        .copy_from_slice(&b_data);
    let b = Tensor::new_pinned(&pinned_b)?;
    drop(pinned_b);
    let sum = (&a + &b)?;
    let sum_data = sum.to_vec1::<f32>()?;
    assert_eq!(sum_data, [5.0, 7.0, 9.0]);

    // Test multiplication
    let prod = (&a * &b)?;
    let prod_data = prod.to_vec1::<f32>()?;
    assert_eq!(prod_data, [4.0, 10.0, 18.0]);

    // Test unary operations
    let neg = (&a).neg()?;
    let neg_data = neg.to_vec1::<f32>()?;
    assert_eq!(neg_data, [-1.0, -2.0, -3.0]);

    // Test reductions
    let sum_all = (&a).sum_all()?.to_scalar::<f32>()?;
    assert_eq!(sum_all, 6.0);

    let mean_all = (&a).mean_all()?.to_scalar::<f32>()?;
    assert!((mean_all - 2.0).abs() < 1e-5);

    // Test reshape - create from pinned memory
    let reshape_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut pinned_reshape = unsafe { cuda_device.cuda_alloc_pinned::<f32>(reshape_data.len())? };
    pinned_reshape
        .as_mut_slice()
        .expect("pinned slice")
        .copy_from_slice(&reshape_data);
    let flat = Tensor::new_pinned(&pinned_reshape)?;
    drop(pinned_reshape);
    let reshaped = flat.reshape((2, 3))?;
    let reshaped_data = reshaped.to_vec2::<f32>()?;
    assert_eq!(reshaped_data, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    // Test different dtypes - create from pinned memory
    let u8_data = [1u8, 2, 3, 4];
    let mut pinned_u8 = unsafe { cuda_device.cuda_alloc_pinned::<u8>(u8_data.len())? };
    pinned_u8
        .as_mut_slice()
        .expect("pinned slice")
        .copy_from_slice(&u8_data);
    let u8_tensor = Tensor::from_pinned_host(&pinned_u8, (2, 2))?;
    drop(pinned_u8);
    let u8_tensor_data = u8_tensor.to_vec2::<u8>()?;
    assert_eq!(u8_tensor_data, [[1, 2], [3, 4]]);

    let i64_data = [10i64, 20, 30];
    let mut pinned_i64 = unsafe { cuda_device.cuda_alloc_pinned::<i64>(i64_data.len())? };
    pinned_i64
        .as_mut_slice()
        .expect("pinned slice")
        .copy_from_slice(&i64_data);
    let i64_tensor = Tensor::new_pinned(&pinned_i64)?;
    drop(pinned_i64);
    let i64_tensor_data = i64_tensor.to_vec1::<i64>()?;
    assert_eq!(i64_tensor_data, [10, 20, 30]);

    Ok(())
}

#[test]
fn pinned_memory_with_different_dtypes() -> Result<()> {
    if !utils::cuda_is_available() {
        return Ok(());
    }

    let device = Device::new_cuda(0)?;
    let len = 16_usize;

    // Test with f32
    let mut pinned_f32 = unsafe { device.cuda_alloc_pinned::<f32>(len)? };
    let f32_data: Vec<f32> = (0..len).map(|i| i as f32 * 1.5).collect();
    pinned_f32
        .as_mut_slice()
        .expect("pinned slice")
        .copy_from_slice(&f32_data);

    let tensor_f32 = Tensor::new_pinned(&pinned_f32)?;
    let tensor_f32_data = tensor_f32.to_vec1::<f32>()?;
    assert_eq!(tensor_f32_data.len(), f32_data.len());
    for (actual, expected) in tensor_f32_data.iter().zip(f32_data.iter()) {
        assert!((actual - expected).abs() < 1e-5);
    }

    // Test with u32
    let mut pinned_u32 = unsafe { device.cuda_alloc_pinned::<u32>(len)? };
    let u32_data: Vec<u32> = (0..len as u32).collect();
    pinned_u32
        .as_mut_slice()
        .expect("pinned slice")
        .copy_from_slice(&u32_data);

    let tensor_u32 = Tensor::new_pinned(&pinned_u32)?;
    let tensor_u32_data = tensor_u32.to_vec1::<u32>()?;
    assert_eq!(tensor_u32_data, u32_data);

    Ok(())
}

#[test]
fn pinned_memory_with_shape() -> Result<()> {
    if !utils::cuda_is_available() {
        return Ok(());
    }

    let device = Device::new_cuda(0)?;
    let len = 12_usize;

    let mut pinned = unsafe { device.cuda_alloc_pinned::<f32>(len)? };
    let data: Vec<f32> = (0..len).map(|i| i as f32).collect();
    pinned
        .as_mut_slice()
        .expect("pinned slice")
        .copy_from_slice(&data);

    // Test with different shapes
    let tensor_1d = Tensor::from_pinned_host(&pinned, len)?;
    assert_eq!(tensor_1d.shape().dims(), &[len]);

    let tensor_2d = Tensor::from_pinned_host(&pinned, (3, 4))?;
    assert_eq!(tensor_2d.shape().dims(), &[3, 4]);
    let tensor_2d_data = tensor_2d.to_vec2::<f32>()?;
    assert_eq!(tensor_2d_data, [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]]);

    let tensor_3d = Tensor::from_pinned_host(&pinned, (2, 2, 3))?;
    assert_eq!(tensor_3d.shape().dims(), &[2, 2, 3]);

    Ok(())
}

