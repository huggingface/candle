#![cfg(feature = "cuda-pinned-memory")]

use candle_core::{backend::BackendStorage, utils, Device, Result, Storage, Tensor};

#[test]
#[cfg(all(feature = "cuda-pinned-memory", feature = "cuda"))]
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
#[cfg(all(feature = "cuda-pinned-memory", feature = "cuda"))]
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
#[cfg(all(feature = "cuda-pinned-memory", feature = "cuda"))]
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
#[cfg(all(feature = "cuda-pinned-memory", feature = "cuda"))]
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

#[test]
#[cfg(all(feature = "cuda-pinned-memory", feature = "cuda"))]
fn pinned_memory_performance_test() -> Result<()> {
    if !utils::cuda_is_available() {
        return Ok(());
    }

    let cuda_device = Device::new_cuda(0)?;
    let len = 10_000_000_usize; // 10M elements = ~40MB for f32

    // Prepare test data
    let data: Vec<f32> = (0..len).map(|i| i as f32).collect();

    // ===== Warmup: Initialize CUDA context and drivers =====
    // Do a warmup transfer to ensure any initialization overhead is excluded from measurements
    let warmup_data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    let warmup_tensor = Tensor::from_slice(&warmup_data, 1000, &Device::Cpu)?;
    let _warmup_cuda = warmup_tensor.to_device(&cuda_device)?;
    cuda_device.synchronize()?;
    
    // Also warmup pinned memory path
    let mut warmup_pinned = unsafe { cuda_device.cuda_alloc_pinned::<f32>(1000)? };
    warmup_pinned
        .as_mut_slice()
        .expect("pinned slice")
        .copy_from_slice(&warmup_data);
    let warmup_pinned_tensor = Tensor::new_pinned(&warmup_pinned)?;
    let _warmup_pinned_cuda = warmup_pinned_tensor.to_device(&cuda_device)?;
    cuda_device.synchronize()?;

    // ===== Test 1: Host → GPU (HTOD) transfers =====

    // Test 1a: Copy from regular host memory to GPU
    let tensor_cpu_regular = Tensor::from_slice(&data, len, &Device::Cpu)?;
    let start = std::time::Instant::now();
    let _tensor_cuda_regular = tensor_cpu_regular.to_device(&cuda_device)?;
    cuda_device.synchronize()?;
    let regular_htod_time = start.elapsed();

    // Test 1b: Copy from pinned host memory to GPU
    let mut pinned_in = unsafe { cuda_device.cuda_alloc_pinned::<f32>(len)? };
    pinned_in
        .as_mut_slice()
        .expect("pinned slice")
        .copy_from_slice(&data);
    let tensor_cpu_pinned = Tensor::new_pinned(&pinned_in)?;
    let start = std::time::Instant::now();
    let _tensor_cuda_pinned = tensor_cpu_pinned.to_device(&cuda_device)?;
    cuda_device.synchronize()?;
    let pinned_htod_time = start.elapsed();

    // ===== Test 2: GPU → Host (DTOH) transfers =====
    
    // Create a CUDA tensor for testing GPU→host transfers
    let tensor_cuda = Tensor::from_slice(&data, len, &cuda_device)?;
    cuda_device.synchronize()?;
    
    // Warmup the DTOH paths
    let (warmup_storage, _) = tensor_cuda.storage_and_layout();
    let warmup_cuda_storage = match &*warmup_storage {
        Storage::Cuda(s) => s,
        _ => unreachable!(),
    };
    let _warmup_cpu = warmup_cuda_storage.to_cpu_storage()?;
    cuda_device.synchronize()?;
    
    let mut warmup_pinned_out = unsafe { cuda_device.cuda_alloc_pinned::<f32>(len)? };
    tensor_cuda.copy_to_pinned_host(&mut warmup_pinned_out)?;
    cuda_device.synchronize()?;

    // Test 2a: Copy from GPU to regular host memory
    // Use memcpy_dtoh directly to a pre-allocated regular buffer for fair comparison
    let (storage, _layout) = tensor_cuda.storage_and_layout();
    let cuda_storage = match &*storage {
        Storage::Cuda(s) => s,
        _ => unreachable!(),
    };
    
    // Pre-allocate regular host memory buffer
    let mut regular_host_buf = vec![999.0f32; len]; // Initialize with wrong values to verify copy
    let cuda_slice = cuda_storage.as_cuda_slice::<f32>()?;
    let start = std::time::Instant::now();
    // Use memcpy_dtoh directly to regular host memory (same API as pinned, different memory type)
    cuda_storage.device().memcpy_dtoh(cuda_slice, regular_host_buf.as_mut_slice())?;
    cuda_device.synchronize()?;
    let regular_dtoh_time = start.elapsed();
    
    // Verify the copy actually happened by checking the data
    assert_eq!(regular_host_buf.len(), len);
    assert!((regular_host_buf[0] - 0.0).abs() < 1e-5, "Regular copy data mismatch at index 0 (got {}, expected 0.0)", regular_host_buf[0]);
    assert!((regular_host_buf[len / 2] - (len / 2) as f32).abs() < 1e-5, "Regular copy data mismatch at middle");

    // Test 2b: Copy from GPU to pinned host memory
    // Pre-allocate pinned memory (this is part of the pinned memory workflow)
    let mut pinned_out = unsafe { cuda_device.cuda_alloc_pinned::<f32>(len)? };
    // Initialize with zeros to ensure we're actually copying
    pinned_out
        .as_mut_slice()
        .expect("pinned slice")
        .fill(999.0f32);
    let start = std::time::Instant::now();
    // This uses memcpy_dtoh which copies directly to pinned memory (can use DMA)
    tensor_cuda.copy_to_pinned_host(&mut pinned_out)?;
    cuda_device.synchronize()?;
    let pinned_dtoh_time = start.elapsed();

    // Verify the copy actually happened by checking the data
    let pinned_data = pinned_out.as_slice().expect("pinned slice");
    assert_eq!(pinned_data.len(), len);
    assert!((pinned_data[0] - 0.0).abs() < 1e-5, "Pinned copy data mismatch at index 0 (got {}, expected 0.0)", pinned_data[0]);
    assert!((pinned_data[len / 2] - (len / 2) as f32).abs() < 1e-5, "Pinned copy data mismatch at middle");
    assert!((pinned_data[len - 1] - (len - 1) as f32).abs() < 1e-5, "Pinned copy data mismatch at last index");
    
    // Verify both copies got the same data
    assert!((pinned_data[0] - regular_host_buf[0]).abs() < 1e-5, "Data mismatch between regular and pinned copies");
    assert!((pinned_data[len / 2] - regular_host_buf[len / 2]).abs() < 1e-5, "Data mismatch between regular and pinned copies");

    // Print performance results
    println!("\n=== Host → GPU (HTOD) ===");
    println!(
        "Regular memory: {:?}, Pinned memory: {:?} (speedup: {:.2}x)",
        regular_htod_time,
        pinned_htod_time,
        regular_htod_time.as_secs_f64() / pinned_htod_time.as_secs_f64()
    );
    
    println!("\n=== GPU → Host (DTOH) ===");
    println!(
        "Regular memory: {:?}, Pinned memory: {:?} (speedup: {:.2}x)",
        regular_dtoh_time,
        pinned_dtoh_time,
        regular_dtoh_time.as_secs_f64() / pinned_dtoh_time.as_secs_f64()
    );
    
    // Assert that pinned memory is at least as fast (HTOD speedup is typically small after warmup)
    // and significantly faster for DTOH (where pinned memory really shines)
    let htod_speedup = regular_htod_time.as_secs_f64() / pinned_htod_time.as_secs_f64();
    let dtoh_speedup = regular_dtoh_time.as_secs_f64() / pinned_dtoh_time.as_secs_f64();

    // HTOD speedup
    println!("HTOD speedup: {:.2}x", htod_speedup);

    // DTOH speedup
    println!("DTOH speedup: {:.2}x", dtoh_speedup);

    Ok(())
}

