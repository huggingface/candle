//! Integration tests for the Vulkan support.
//!
//! Device selection:
//! `CANDLE_VULKAN_TEST_GPU=<n> cargo test --features vulkan --test vulkan_integration_tests`
//! (defaults to physical device 0).
#![cfg(feature = "vulkan")]
use candle_core::{backend::BackendDevice, backend::BackendStorage, VulkanDevice};
use std::sync::LazyLock;
static INIT: LazyLock<()> = LazyLock::new(|| {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_test_writer()
        .init();
});
static GPU_ID: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("CANDLE_VULKAN_TEST_GPU")
        .map(|value| value.parse().unwrap())
        .unwrap_or_default()
});
#[test]
fn test_vulkan_device_supports_bf16() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let device = VulkanDevice::new(gpu_id).unwrap();
    tracing::debug!(
        "Vulkan device {gpu_id} has bfloat16 support?: {}",
        device.supports_bf16()
    );
}
#[test]
fn test_vulkan_device_storage() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let device = VulkanDevice::new(gpu_id).unwrap();
    // A 1-D shape of 256 elements (NOT 256 zero-sized dims).
    let shape = candle_core::Shape::from(256);
    let dtype = candle_core::DType::F16;
    let storage = device.zeros_impl(&shape, dtype).unwrap();
    let cpu = storage.to_cpu_storage().unwrap();
    let data = cpu.as_slice::<half::f16>().unwrap();
    assert_eq!(data.len(), 256);
    assert!(
        data.iter().all(|value| value.to_bits() == 0),
        "expected every f16 element to be zero, got {data:?}"
    );
    tracing::debug!("Vulkan device {gpu_id} zeroed f16 storage round-tripped OK");
}
#[test]
fn test_vulkan_device_fill_f16() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let device = VulkanDevice::new(gpu_id).unwrap();
    // 65,536 elements: one slot for every possible f16 bit pattern.
    let shape = candle_core::Shape::from(65536);
    let storage = device.zeros_impl(&shape, candle_core::DType::F16).unwrap();
    storage.fill_f16().unwrap();
    let cpu = storage.to_cpu_storage().unwrap();
    let data = cpu.as_slice::<half::f16>().unwrap();
    assert_eq!(data.len(), 65536);
    let mismatch = data
        .iter()
        .enumerate()
        .find(|(index, value)| value.to_bits() != *index as u16);
    if let Some((index, value)) = mismatch {
        panic!(
            "fill_f16 mismatch at index {index}: got bits {:04x}, expected {:04x}",
            value.to_bits(),
            index as u16
        );
    }
    // All 65,536 bit patterns must be distinct.
    let unique = data
        .iter()
        .map(|v| v.to_bits())
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(unique.len(), 65536, "fill_f16 produced duplicate values");
    tracing::debug!("Vulkan device {gpu_id} fill_f16 round-tripped OK");
}

#[test]
fn test_vulkan_device_fill_u8() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let device = VulkanDevice::new(gpu_id).unwrap();
    // 256 elements: one slot for every possible byte value.
    let shape = candle_core::Shape::from(256);
    let storage = device.zeros_impl(&shape, candle_core::DType::U8).unwrap();
    storage.fill_u8().unwrap();
    let cpu = storage.to_cpu_storage().unwrap();
    let data = cpu.as_slice::<u8>().unwrap().to_vec();
    assert_eq!(data.len(), 256);
    // Every element must be a valid byte matching its index.
    for (index, value) in data.iter().enumerate() {
        assert_eq!(*value, index as u8, "fill_u8 mismatch at index {index}");
    }
    // All 256 values must be distinct.
    let unique = data.iter().collect::<std::collections::HashSet<_>>();
    assert_eq!(unique.len(), 256, "fill_u8 produced duplicate values");
    tracing::debug!("Vulkan device {gpu_id} fill_u8 round-tripped OK");
}
#[test]
fn test_vulkan_device_fill_i16() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let device = VulkanDevice::new(gpu_id).unwrap();
    // 65,536 elements: one slot for every possible 16-bit bit pattern.
    let shape = candle_core::Shape::from(65536);
    let storage = device.zeros_impl(&shape, candle_core::DType::I16).unwrap();
    storage.fill_i16().unwrap();
    let cpu = storage.to_cpu_storage().unwrap();
    let data = cpu.as_slice::<i16>().unwrap().to_vec();
    assert_eq!(data.len(), 65536);
    // Every element must hold the bit pattern of its index (a valid i16).
    for (index, value) in data.iter().enumerate() {
        assert_eq!(
            *value as u16, index as u16,
            "fill_i16 mismatch at index {index}"
        );
    }
    // All 65,536 bit patterns must be distinct.
    let unique = data
        .iter()
        .map(|v| *v as u16)
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(unique.len(), 65536, "fill_i16 produced duplicate values");
    tracing::debug!("Vulkan device {gpu_id} fill_i16 round-tripped OK");
}
#[test]
fn test_vulkan_device_fill_bf16() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let device = VulkanDevice::new(gpu_id).unwrap();
    // 65,536 elements: one slot for every possible 16-bit bit pattern.
    let shape = candle_core::Shape::from(65536);
    let storage = device.zeros_impl(&shape, candle_core::DType::BF16).unwrap();
    storage.fill_bf16().unwrap();
    let cpu = storage.to_cpu_storage().unwrap();
    let data = cpu.as_slice::<half::bf16>().unwrap().to_vec();
    assert_eq!(data.len(), 65536);
    // Every element must hold the bit pattern of its index (a valid bf16).
    for (index, value) in data.iter().enumerate() {
        assert_eq!(
            value.to_bits(),
            index as u16,
            "fill_bf16 mismatch at index {index}"
        );
    }
    // All 65,536 bit patterns must be distinct.
    let unique = data
        .iter()
        .map(|v| v.to_bits())
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(unique.len(), 65536, "fill_bf16 produced duplicate values");
    tracing::debug!(
        "Vulkan device {gpu_id} fill_bf16 round-tripped OK (emulated path; hw_bf16={})",
        device.supports_bf16()
    );
}
#[test]
fn test_vulkan_device_fill_bf16_emulated() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let device = VulkanDevice::new(gpu_id).unwrap();
    // 65,536 elements: one slot for every possible 16-bit bit pattern.
    let shape = candle_core::Shape::from(65536);
    let storage = device.zeros_impl(&shape, candle_core::DType::BF16).unwrap();
    // Force the emulated (raw 16-bit) path regardless of native support.
    storage.fill_bf16_emulated().unwrap();
    let cpu = storage.to_cpu_storage().unwrap();
    let data = cpu.as_slice::<half::bf16>().unwrap().to_vec();
    assert_eq!(data.len(), 65536);
    for (index, value) in data.iter().enumerate() {
        assert_eq!(
            value.to_bits(),
            index as u16,
            "fill_bf16_emulated mismatch at index {index}"
        );
    }
    let unique = data
        .iter()
        .map(|v| v.to_bits())
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(
        unique.len(),
        65536,
        "fill_bf16_emulated produced duplicate values"
    );
    tracing::debug!("Vulkan device {gpu_id} fill_bf16_emulated round-tripped OK");
}
#[test]
fn test_vulkan_device_fill_u32() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let device = VulkanDevice::new(gpu_id).unwrap();
    let shape = candle_core::Shape::from(256);
    let storage = device.zeros_impl(&shape, candle_core::DType::U32).unwrap();
    storage.fill_u32().unwrap();
    let cpu = storage.to_cpu_storage().unwrap();
    let data = cpu.as_slice::<u32>().unwrap().to_vec();
    assert_eq!(data.len(), 256);
    // Each element must be (i + 1) * 100000 + 65536, all beyond the 16-bit range.
    for (index, value) in data.iter().enumerate() {
        let expected = (index as u32 + 1) * 100000 + 65536;
        assert_eq!(*value, expected, "fill_u32 mismatch at index {index}");
    }
    // A truncated 16-bit buffer could not produce any of these values.
    assert!(
        data.iter().all(|v| *v > u16::MAX as u32),
        "fill_u32 value not beyond the 16-bit range"
    );
    // All 256 values must be distinct.
    let unique = data.iter().collect::<std::collections::HashSet<_>>();
    assert_eq!(unique.len(), 256, "fill_u32 produced duplicate values");
    tracing::debug!("Vulkan device {gpu_id} fill_u32 round-tripped OK");
}
#[test]
fn test_vulkan_device_fill_i32() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let device = VulkanDevice::new(gpu_id).unwrap();
    let shape = candle_core::Shape::from(256);
    let storage = device.zeros_impl(&shape, candle_core::DType::I32).unwrap();
    storage.fill_i32().unwrap();
    let cpu = storage.to_cpu_storage().unwrap();
    let data = cpu.as_slice::<i32>().unwrap().to_vec();
    assert_eq!(data.len(), 256);
    // Each element must be (i - 128) * 100000, straddling the 16-bit range.
    for (index, value) in data.iter().enumerate() {
        let expected = (index as i32 - 128) * 100000;
        assert_eq!(*value, expected, "fill_i32 mismatch at index {index}");
    }
    // A truncated i16 buffer could not produce these out-of-range values.
    assert!(
        data.iter().any(|v| *v > i16::MAX as i32) && data.iter().any(|v| *v < i16::MIN as i32),
        "fill_i32 values do not straddle the 16-bit range"
    );
    // All 256 values must be distinct.
    let unique = data.iter().collect::<std::collections::HashSet<_>>();
    assert_eq!(unique.len(), 256, "fill_i32 produced duplicate values");
    tracing::debug!("Vulkan device {gpu_id} fill_i32 round-tripped OK");
}
#[test]
fn test_vulkan_device_fill_f32() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let device = VulkanDevice::new(gpu_id).unwrap();
    let shape = candle_core::Shape::from(256);
    let storage = device.zeros_impl(&shape, candle_core::DType::F32).unwrap();
    storage.fill_f32().unwrap();
    let cpu = storage.to_cpu_storage().unwrap();
    let data = cpu.as_slice::<f32>().unwrap().to_vec();
    assert_eq!(data.len(), 256);
    // Each element must be (i + 1) * 1000.0, some exceeding the f16 max (65504).
    for (index, value) in data.iter().enumerate() {
        let expected = (index as f32 + 1.0) * 1000.0;
        assert_eq!(*value, expected, "fill_f32 mismatch at index {index}");
    }
    // An f16 buffer could not hold values above its max finite value.
    assert!(
        data.iter().any(|v| *v > 65504.0),
        "fill_f32 no value above the f16 max"
    );
    // All 256 values must be distinct (compared by bit pattern).
    let unique = data
        .iter()
        .map(|v| v.to_bits())
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(unique.len(), 256, "fill_f32 produced duplicate values");
    tracing::debug!("Vulkan device {gpu_id} fill_f32 round-tripped OK");
}
#[test]
fn test_vulkan_device_fill_i64() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let device = VulkanDevice::new(gpu_id).unwrap();
    let shape = candle_core::Shape::from(256);
    let storage = device.zeros_impl(&shape, candle_core::DType::I64).unwrap();
    storage.fill_i64().unwrap();
    let cpu = storage.to_cpu_storage().unwrap();
    let data = cpu.as_slice::<i64>().unwrap().to_vec();
    assert_eq!(data.len(), 256);
    // Each element must be (i - 128) * 1,000,000,000,000, beyond the 32-bit range.
    for (index, value) in data.iter().enumerate() {
        let expected = (index as i64 - 128) * 1_000_000_000_000;
        assert_eq!(*value, expected, "fill_i64 mismatch at index {index}");
    }
    // A truncated i32 buffer could not produce these out-of-range values.
    assert!(
        data.iter().any(|v| *v > i32::MAX as i64) && data.iter().any(|v| *v < i32::MIN as i64),
        "fill_i64 values do not straddle the 32-bit range"
    );
    // All 256 values must be distinct.
    let unique = data.iter().collect::<std::collections::HashSet<_>>();
    assert_eq!(unique.len(), 256, "fill_i64 produced duplicate values");
    tracing::debug!("Vulkan device {gpu_id} fill_i64 round-tripped OK");
}
#[test]
fn test_vulkan_device_fill_f64() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let device = VulkanDevice::new(gpu_id).unwrap();
    let shape = candle_core::Shape::from(256);
    let storage = device.zeros_impl(&shape, candle_core::DType::F64).unwrap();
    storage.fill_f64().unwrap();
    let cpu = storage.to_cpu_storage().unwrap();
    let data = cpu.as_slice::<f64>().unwrap().to_vec();
    assert_eq!(data.len(), 256);
    // Each element must be (i + 1) * 16,777,217.0 (2^24 + 1), beyond f32 precision.
    for (index, value) in data.iter().enumerate() {
        let expected = (index as f64 + 1.0) * 16_777_217.0;
        assert_eq!(*value, expected, "fill_f64 mismatch at index {index}");
    }
    // An f32 buffer could not hold values that are not exactly representable in f32.
    assert!(
        data.iter().any(|v| (*v as f32) as f64 != *v),
        "fill_f64 all values exactly representable in f32"
    );
    // All 256 values must be distinct (compared by bit pattern).
    let unique = data
        .iter()
        .map(|v| v.to_bits())
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(unique.len(), 256, "fill_f64 produced duplicate values");
    tracing::debug!("Vulkan device {gpu_id} fill_f64 round-tripped OK");
}

#[test]
fn test_vulkan_device_fill_f4() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let device = VulkanDevice::new(gpu_id).unwrap();
    // 16 elements: one slot for every possible raw value.
    let shape = candle_core::Shape::from(16);
    let storage = device.zeros_impl(&shape, candle_core::DType::F4).unwrap();
    storage.fill_f4().unwrap();
    let cpu = storage.to_cpu_storage().unwrap();
    let data = match cpu {
        candle_core::CpuStorage::F4(data) => data,
        _ => panic!("expected F4 storage"),
    };
    assert_eq!(data.len(), 16);
    // Every element must be a raw byte matching its index.
    for (index, value) in data.iter().enumerate() {
        assert_eq!(
            value.to_bits(),
            index as u8,
            "fill_f4 mismatch at index {index}"
        );
    }
    // All values must be distinct.
    let unique = data
        .iter()
        .map(|v| v.to_bits())
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(unique.len(), 16, "fill_f4 produced duplicate values");
    tracing::debug!("Vulkan device {gpu_id} fill_f4 round-tripped OK");
}

#[test]
fn test_vulkan_device_fill_f6e2m3() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let device = VulkanDevice::new(gpu_id).unwrap();
    // 64 elements: one slot for every possible raw value.
    let shape = candle_core::Shape::from(64);
    let storage = device
        .zeros_impl(&shape, candle_core::DType::F6E2M3)
        .unwrap();
    storage.fill_f6e2m3().unwrap();
    let cpu = storage.to_cpu_storage().unwrap();
    let data = match cpu {
        candle_core::CpuStorage::F6E2M3(data) => data,
        _ => panic!("expected F6E2M3 storage"),
    };
    assert_eq!(data.len(), 64);
    // Every element must be a raw byte matching its index.
    for (index, value) in data.iter().enumerate() {
        assert_eq!(
            value.to_bits(),
            index as u8,
            "fill_f6e2m3 mismatch at index {index}"
        );
    }
    // All values must be distinct.
    let unique = data
        .iter()
        .map(|v| v.to_bits())
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(unique.len(), 64, "fill_f6e2m3 produced duplicate values");
    tracing::debug!("Vulkan device {gpu_id} fill_f6e2m3 round-tripped OK");
}

#[test]
fn test_vulkan_device_fill_f6e3m2() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let device = VulkanDevice::new(gpu_id).unwrap();
    // 64 elements: one slot for every possible raw value.
    let shape = candle_core::Shape::from(64);
    let storage = device
        .zeros_impl(&shape, candle_core::DType::F6E3M2)
        .unwrap();
    storage.fill_f6e3m2().unwrap();
    let cpu = storage.to_cpu_storage().unwrap();
    let data = match cpu {
        candle_core::CpuStorage::F6E3M2(data) => data,
        _ => panic!("expected F6E3M2 storage"),
    };
    assert_eq!(data.len(), 64);
    // Every element must be a raw byte matching its index.
    for (index, value) in data.iter().enumerate() {
        assert_eq!(
            value.to_bits(),
            index as u8,
            "fill_f6e3m2 mismatch at index {index}"
        );
    }
    // All values must be distinct.
    let unique = data
        .iter()
        .map(|v| v.to_bits())
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(unique.len(), 64, "fill_f6e3m2 produced duplicate values");
    tracing::debug!("Vulkan device {gpu_id} fill_f6e3m2 round-tripped OK");
}

#[test]
fn test_vulkan_device_fill_f8e4m3() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let device = VulkanDevice::new(gpu_id).unwrap();
    // 256 elements: one slot for every possible raw value.
    let shape = candle_core::Shape::from(256);
    let storage = device
        .zeros_impl(&shape, candle_core::DType::F8E4M3)
        .unwrap();
    storage.fill_f8e4m3().unwrap();
    let cpu = storage.to_cpu_storage().unwrap();
    let data = match cpu {
        candle_core::CpuStorage::F8E4M3(data) => data,
        _ => panic!("expected F8E4M3 storage"),
    };
    assert_eq!(data.len(), 256);
    // Every element must be a raw byte matching its index.
    for (index, value) in data.iter().enumerate() {
        assert_eq!(
            value.to_bits(),
            index as u8,
            "fill_f8e4m3 mismatch at index {index}"
        );
    }
    // All values must be distinct.
    let unique = data
        .iter()
        .map(|v| v.to_bits())
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(unique.len(), 256, "fill_f8e4m3 produced duplicate values");
    tracing::debug!("Vulkan device {gpu_id} fill_f8e4m3 round-tripped OK");
}

#[test]
fn test_vulkan_device_fill_f8e8m0() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let device = VulkanDevice::new(gpu_id).unwrap();
    // 256 elements: one slot for every possible raw value.
    let shape = candle_core::Shape::from(256);
    let storage = device
        .zeros_impl(&shape, candle_core::DType::F8E8M0)
        .unwrap();
    storage.fill_f8e8m0().unwrap();
    let cpu = storage.to_cpu_storage().unwrap();
    let data = match cpu {
        candle_core::CpuStorage::F8E8M0(data) => data,
        _ => panic!("expected F8E8M0 storage"),
    };
    assert_eq!(data.len(), 256);
    // Every element must be a raw byte matching its index.
    for (index, value) in data.iter().enumerate() {
        assert_eq!(
            value.to_bits(),
            index as u8,
            "fill_f8e8m0 mismatch at index {index}"
        );
    }
    // All values must be distinct.
    let unique = data
        .iter()
        .map(|v| v.to_bits())
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(unique.len(), 256, "fill_f8e8m0 produced duplicate values");
    tracing::debug!("Vulkan device {gpu_id} fill_f8e8m0 round-tripped OK");
}

/// `synchronize` must drain all deferred GPU work on the device.
#[test]
fn test_vulkan_device_synchronize() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let device = VulkanDevice::new(gpu_id).unwrap();
    let shape = candle_core::Shape::from((4, 4));
    let storage = device.zeros_impl(&shape, candle_core::DType::F32).unwrap();
    device.synchronize().unwrap();
    let cpu = storage.to_cpu_storage().unwrap();
    let data = cpu.as_slice::<f32>().unwrap();
    assert!(data.iter().all(|value| *value == 0.0));
    tracing::debug!("Vulkan device {gpu_id} synchronize drained OK");
}

/// `set_seed` must store the seed on the device.
#[test]
fn test_vulkan_device_set_seed() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let device = VulkanDevice::new(gpu_id).unwrap();
    device.set_seed(0xDEADBEEF).unwrap();
    assert_eq!(device.seed_atomic().load(std::sync::atomic::Ordering::Relaxed), 0xDEADBEEF);
}

/// `get_current_seed` must return the seed stored on the device.
#[test]
fn test_vulkan_device_get_current_seed() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let device = VulkanDevice::new(gpu_id).unwrap();
    device.set_seed(42).unwrap();
    assert_eq!(device.get_current_seed().unwrap(), 42);
}

/// `storage_from_slice` must upload host data to the device.
#[test]
fn test_vulkan_device_storage_from_slice() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let device = VulkanDevice::new(gpu_id).unwrap();
    let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.5 - 8.0).collect();
    let storage = device.storage_from_slice(&data).unwrap();
    assert_eq!(storage.dtype(), candle_core::DType::F32);
    let cpu = storage.to_cpu_storage().unwrap();
    let back = cpu.as_slice::<f32>().unwrap();
    assert_eq!(back.len(), data.len());
    assert!(back.iter().zip(&data).all(|(a, b)| a == b));
    tracing::debug!("Vulkan device {gpu_id} storage_from_slice round-tripped OK");
}

/// `storage_from_cpu_storage` must upload a CPU tensor to the device.
#[test]
fn test_vulkan_device_storage_from_cpu_storage() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let device = VulkanDevice::new(gpu_id).unwrap();
    let cpu = candle_core::Device::Cpu;
    let t = candle_core::Tensor::new(vec![1.5f32, -2.25, 0.0, 42.0], &cpu).unwrap();
    let cpu_storage = candle_core::CpuStorage::F32(t.to_vec1::<f32>().unwrap());
    let v = device.storage_from_cpu_storage(&cpu_storage).unwrap();
    assert_eq!(v.dtype(), candle_core::DType::F32);
    let back = v.to_cpu_storage().unwrap();
    let data = back.as_slice::<f32>().unwrap();
    assert_eq!(data, &[1.5f32, -2.25, 0.0, 42.0]);
    tracing::debug!("Vulkan device {gpu_id} storage_from_cpu_storage round-tripped OK");
}

/// `storage_from_cpu_storage_owned` must consume the CPU storage.
#[test]
fn test_vulkan_device_storage_from_cpu_storage_owned() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let device = VulkanDevice::new(gpu_id).unwrap();
    let cpu = candle_core::Device::Cpu;
    let t = candle_core::Tensor::new(vec![7.5f32, 8.25, 9.0], &cpu).unwrap();
    let cpu_storage = candle_core::CpuStorage::F32(t.to_vec1::<f32>().unwrap());
    let v = device.storage_from_cpu_storage_owned(cpu_storage).unwrap();
    assert_eq!(v.dtype(), candle_core::DType::F32);
    let back = v.to_cpu_storage().unwrap();
    let data = back.as_slice::<f32>().unwrap();
    assert_eq!(data, &[7.5f32, 8.25, 9.0]);
    tracing::debug!("Vulkan device {gpu_id} owned upload round-tripped OK");
}

/// `rand_uniform` must draw in [min, max] and be reproducible per seed.
#[test]
fn test_vulkan_device_rand_uniform() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let device = VulkanDevice::new(gpu_id).unwrap();
    let shape = candle_core::Shape::from(1024);
    device.set_seed(1234).unwrap();
    let a = device.rand_uniform(&shape, candle_core::DType::F32, -1.0, 2.0).unwrap();
    device.set_seed(1234).unwrap();
    let b = device.rand_uniform(&shape, candle_core::DType::F32, -1.0, 2.0).unwrap();
    let va = a.to_cpu_storage().unwrap().as_slice::<f32>().unwrap().to_vec();
    let vb = b.to_cpu_storage().unwrap().as_slice::<f32>().unwrap().to_vec();
    assert_eq!(va, vb, "same seed must give the same draw");
    assert!(va.iter().all(|x| *x >= -1.0 && *x <= 2.0));
    assert!(va.iter().any(|x| *x < -0.99) && va.iter().any(|x| *x > 1.99),
            "draw should span the range: min {} max {}", va.iter().cloned().fold(f32::INFINITY, f32::min), va.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    tracing::debug!("Vulkan device {gpu_id} rand_uniform OK");
}

/// `rand_normal` must be reproducible per seed with the right moment.
#[test]
fn test_vulkan_device_rand_normal() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let device = VulkanDevice::new(gpu_id).unwrap();
    let shape = candle_core::Shape::from(4096);
    device.set_seed(99).unwrap();
    let a = device.rand_normal(&shape, candle_core::DType::F32, 1.0, 2.0).unwrap();
    device.set_seed(99).unwrap();
    let b = device.rand_normal(&shape, candle_core::DType::F32, 1.0, 2.0).unwrap();
    let va = a.to_cpu_storage().unwrap().as_slice::<f32>().unwrap().to_vec();
    let vb = b.to_cpu_storage().unwrap().as_slice::<f32>().unwrap().to_vec();
    assert_eq!(va, vb, "same seed must give the same draw");
    let n = va.len() as f64;
    let mean = va.iter().map(|x| *x as f64).sum::<f64>() / n;
    let var = va.iter().map(|x| (*x as f64 - 1.0) * (*x as f64 - 1.0)).sum::<f64>() / n;
    assert!((mean - 1.0).abs() < 0.1, "mean {mean}");
    assert!((var - 4.0).abs() < 0.5, "var {var}");
    tracing::debug!("Vulkan device {gpu_id} rand_normal OK (mean {mean}, var {var})");
}

/// `const_set` fills a contiguous region with a constant value.
#[test]
fn test_vulkan_const_set() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let device = VulkanDevice::new(gpu_id).unwrap();
    let shape = candle_core::Shape::from((4, 8));
    let mut storage = device.zeros_impl(&shape, candle_core::DType::F32).unwrap();
    let l = candle_core::Layout::contiguous(shape);
    storage
        .const_set(candle_core::scalar::Scalar::F32(3.5), &l)
        .unwrap();
    device.synchronize().unwrap();
    let binding = storage.to_cpu_storage().unwrap();
    let data = binding.as_slice::<f32>().unwrap();
    assert!(data.iter().all(|x| *x == 3.5), "expected all 3.5, got {:?}", &data[..4]);
    tracing::debug!("Vulkan device {gpu_id} const_set OK");
}

/// All 19 `UnaryOp`s, compared element-wise against the CPU backend.
#[test]
fn test_vulkan_unary_ops() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let dev = candle_core::Device::new_vulkan(gpu_id).unwrap();
    let cpu = candle_core::Device::Cpu;
    // Mix of positive/negative, in/out of the "interesting" ranges.
    let xs: Vec<f32> = vec![
        0.0, 1.0, -1.0, 0.5, -0.5, 2.5, -2.5, 10.0, -10.0, 0.25, -0.75, 3.7, -3.7,
    ];
    let t_cpu = candle_core::Tensor::new(xs.as_slice(), &cpu).unwrap();
    let t = candle_core::Tensor::new(xs.as_slice(), &dev).unwrap();

    let pairs: Vec<(&str, Vec<f32>)> = vec![
        ("exp", t.exp().unwrap().to_vec1::<f32>().unwrap()),
        ("log", t.log().unwrap().to_vec1::<f32>().unwrap()),
        ("sin", t.sin().unwrap().to_vec1::<f32>().unwrap()),
        ("cos", t.cos().unwrap().to_vec1::<f32>().unwrap()),
        ("abs", t.abs().unwrap().to_vec1::<f32>().unwrap()),
        ("neg", t.neg().unwrap().to_vec1::<f32>().unwrap()),
        ("recip", t.recip().unwrap().to_vec1::<f32>().unwrap()),
        ("sqr", t.sqr().unwrap().to_vec1::<f32>().unwrap()),
        ("sqrt", t.sqrt().unwrap().to_vec1::<f32>().unwrap()),
        ("gelu", t.gelu().unwrap().to_vec1::<f32>().unwrap()),
        ("gelu_erf", t.gelu_erf().unwrap().to_vec1::<f32>().unwrap()),
        ("erf", t.erf().unwrap().to_vec1::<f32>().unwrap()),
        ("relu", t.relu().unwrap().to_vec1::<f32>().unwrap()),
        ("silu", t.silu().unwrap().to_vec1::<f32>().unwrap()),
        ("tanh", t.tanh().unwrap().to_vec1::<f32>().unwrap()),
        ("floor", t.floor().unwrap().to_vec1::<f32>().unwrap()),
        ("ceil", t.ceil().unwrap().to_vec1::<f32>().unwrap()),
        ("round", t.round().unwrap().to_vec1::<f32>().unwrap()),
        ("sign", t.sign().unwrap().to_vec1::<f32>().unwrap()),
    ];
    let mut failures = 0usize;
    for (name, gpu) in &pairs {
        let cpu_t = match *name {
            "exp" => t_cpu.exp().unwrap(),
            "log" => t_cpu.log().unwrap(),
            "sin" => t_cpu.sin().unwrap(),
            "cos" => t_cpu.cos().unwrap(),
            "abs" => t_cpu.abs().unwrap(),
            "neg" => t_cpu.neg().unwrap(),
            "recip" => t_cpu.recip().unwrap(),
            "sqr" => t_cpu.sqr().unwrap(),
            "sqrt" => t_cpu.sqrt().unwrap(),
            "gelu" => t_cpu.gelu().unwrap(),
            "gelu_erf" => t_cpu.gelu_erf().unwrap(),
            "erf" => t_cpu.erf().unwrap(),
            "relu" => t_cpu.relu().unwrap(),
            "silu" => t_cpu.silu().unwrap(),
            "tanh" => t_cpu.tanh().unwrap(),
            "floor" => t_cpu.floor().unwrap(),
            "ceil" => t_cpu.ceil().unwrap(),
            "round" => t_cpu.round().unwrap(),
            "sign" => t_cpu.sign().unwrap(),
            _ => unreachable!(),
        }
        .to_vec1::<f32>()
        .unwrap();
        for (i, (g, c)) in gpu.iter().zip(cpu_t.iter()).enumerate() {
            let tol = if g.is_infinite() || c.is_infinite() {
                0.0
            } else {
                1e-5 * (1.0 + g.abs().max(c.abs()))
            };
            if (g - c).abs() > tol.max(1e-5) {
                failures += 1;
                tracing::debug!(
                    "unary {name}: mismatch at {i}: gpu {g} cpu {c}"
                );
            }
        }
    }
    assert_eq!(failures, 0, "{failures} unary mismatches");
    tracing::debug!("Vulkan device {gpu_id} all 19 unary ops OK");
}

/// Binary ops (all 6) including scalar and rank-broadcast against CPU.
#[test]
fn test_vulkan_binary_ops() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let dev = candle_core::Device::new_vulkan(gpu_id).unwrap();
    let cpu = candle_core::Device::Cpu;
    let shape = candle_core::Shape::from((4, 8));
    let lhs_v: Vec<f32> = (0..32).map(|i| (i as f32) * 0.25 - 4.0).collect();
    let rhs_v: Vec<f32> = (0..32).map(|i| 1.0 / (1.0 + (i as f32))).collect();
    let l = candle_core::Tensor::new(lhs_v.as_slice(), &dev).unwrap().reshape(shape.clone()).unwrap();
    let r = candle_core::Tensor::new(rhs_v.as_slice(), &dev).unwrap().reshape(shape.clone()).unwrap();
    let lc = candle_core::Tensor::new(lhs_v.as_slice(), &cpu).unwrap().reshape(shape.clone()).unwrap();
    let rc = candle_core::Tensor::new(rhs_v.as_slice(), &cpu).unwrap().reshape(shape.clone()).unwrap();
    // 8 elements: broadcasts over the last dim of (4, 8).
    let row = candle_core::Tensor::new(&[0.5f32, -1.5, 2.0, 3.0, -2.5, 1.0, 0.25, -0.75], &dev).unwrap();
    let rowc = candle_core::Tensor::new(&[0.5f32, -1.5, 2.0, 3.0, -2.5, 1.0, 0.25, -0.75], &cpu).unwrap();
    let rowr = row.broadcast_as(shape.clone()).unwrap();
    let rowrc = rowc.broadcast_as(shape.clone()).unwrap();
    let s_dev = candle_core::Tensor::new(1.5f32, &dev).unwrap();
    let s_cpu = candle_core::Tensor::new(1.5f32, &cpu).unwrap();
    let n_dev = candle_core::Tensor::new(-2.0f32, &dev).unwrap();
    let n_cpu = candle_core::Tensor::new(-2.0f32, &cpu).unwrap();

    let cases: Vec<(&str, candle_core::Tensor, candle_core::Tensor)> = vec![
        ("add", l.add(&r).unwrap(), lc.add(&rc).unwrap()),
        ("sub", l.sub(&r).unwrap(), lc.sub(&rc).unwrap()),
        ("mul", l.mul(&r).unwrap(), lc.mul(&rc).unwrap()),
        ("div", l.div(&r).unwrap(), lc.div(&rc).unwrap()),
        ("maximum", l.maximum(&r).unwrap(), lc.maximum(&rc).unwrap()),
        ("minimum", l.minimum(&r).unwrap(), lc.minimum(&rc).unwrap()),
        // Scalar (0-dim) broadcast.
        ("add_scalar", l.add(&sbcast_dev(s_dev, &shape)).unwrap(), lc.add(&sbcast_cpu(s_cpu, &shape)).unwrap()),
        ("mul_scalar", l.mul(&sbcast_dev(n_dev, &shape)).unwrap(), lc.mul(&sbcast_cpu(n_cpu, &shape)).unwrap()),
        ("maximum_scalar", l.maximum(0.0f32).unwrap(), lc.maximum(0.0f32).unwrap()),
        // Rank-1 row broadcast over the last dim.
        ("add_row", l.add(&rowr).unwrap(), lc.add(&rowrc).unwrap()),
        ("mul_row", l.mul(&rowr).unwrap(), lc.mul(&rowrc).unwrap()),
        ("minimum_row", l.minimum(&rowr).unwrap(), lc.minimum(&rowrc).unwrap()),
    ];
    let mut failures = 0usize;
    for (name, g, c) in &cases {
        let gv: Vec<f32> = g.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
        let cv: Vec<f32> = c.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
        assert_eq!(gv.len(), cv.len());
        for (i, (a, b)) in gv.iter().zip(cv.iter()).enumerate() {
            let tol = 1e-5 * (1.0 + a.abs().max(b.abs()));
            if (a - b).abs() > tol {
                failures += 1;
                tracing::debug!("binary {name}: mismatch at {i}: gpu {a} cpu {b}");
            }
        }
    }
    assert_eq!(failures, 0, "{failures} binary mismatches");
    tracing::debug!("Vulkan device {gpu_id} binary ops OK (incl. broadcast)");
}

fn sbcast_dev(t: candle_core::Tensor, shape: &candle_core::Shape) -> candle_core::Tensor {
    t.broadcast_as(shape.clone()).unwrap()
}
fn sbcast_cpu(t: candle_core::Tensor, shape: &candle_core::Shape) -> candle_core::Tensor {
    t.broadcast_as(shape.clone()).unwrap()
}

/// All 6 CmpOps, same-shape and scalar rhs, compared against the CPU backend.
#[test]
fn test_vulkan_cmp_ops() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let dev = candle_core::Device::new_vulkan(gpu_id).unwrap();
    let cpu = candle_core::Device::Cpu;
    let shape = candle_core::Shape::from((4, 8));
    let lhs_v: Vec<f32> = (0..32).map(|i| (i as f32) * 0.5 - 8.0).collect();
    let rhs_v: Vec<f32> = (0..32).map(|i| ((i as f32) % 3.0) * 2.0 - 3.0).collect();
    let l = candle_core::Tensor::new(lhs_v.as_slice(), &dev).unwrap().reshape(shape.clone()).unwrap();
    let r = candle_core::Tensor::new(rhs_v.as_slice(), &dev).unwrap().reshape(shape.clone()).unwrap();
    let lc = candle_core::Tensor::new(lhs_v.as_slice(), &cpu).unwrap().reshape(shape.clone()).unwrap();
    let rc = candle_core::Tensor::new(rhs_v.as_slice(), &cpu).unwrap().reshape(shape.clone()).unwrap();

    let cases: Vec<(&str, candle_core::Tensor, candle_core::Tensor)> = vec![
        ("eq", l.eq(&r).unwrap(), lc.eq(&rc).unwrap()),
        ("ne", l.ne(&r).unwrap(), lc.ne(&rc).unwrap()),
        ("lt", l.lt(&r).unwrap(), lc.lt(&rc).unwrap()),
        ("le", l.le(&r).unwrap(), lc.le(&rc).unwrap()),
        ("gt", l.gt(&r).unwrap(), lc.gt(&rc).unwrap()),
        ("ge", l.ge(&r).unwrap(), lc.ge(&rc).unwrap()),
        // Scalar rhs.
        ("eq_s", l.eq(0.0f32).unwrap(), lc.eq(0.0f32).unwrap()),
        ("gt_s", l.gt(-4.0f32).unwrap(), lc.gt(-4.0f32).unwrap()),
        ("le_s", l.le(-4.0f32).unwrap(), lc.le(-4.0f32).unwrap()),
    ];
    let mut failures = 0usize;
    for (name, g, c) in &cases {
        let gv: Vec<u8> = g.to_vec2::<u8>().unwrap().into_iter().flatten().collect();
        let cv: Vec<u8> = c.to_vec2::<u8>().unwrap().into_iter().flatten().collect();
        assert_eq!(gv.len(), cv.len());
        for (i, (a, b)) in gv.iter().zip(cv.iter()).enumerate() {
            if a != b {
                failures += 1;
                tracing::debug!("cmp {name}: mismatch at {i}: gpu {a} cpu {b}");
            }
        }
    }
    assert_eq!(failures, 0, "{failures} cmp mismatches");
    tracing::debug!("Vulkan device {gpu_id} cmp ops OK");
}

/// `where_cond` with a cmp predicate, incl. a broadcast on_false.
#[test]
fn test_vulkan_where_cond() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let dev = candle_core::Device::new_vulkan(gpu_id).unwrap();
    let cpu = candle_core::Device::Cpu;
    let shape = candle_core::Shape::from((4, 8));
    let cond_v: Vec<f32> = (0..32).map(|i| (i as f32) - 15.0).collect();
    let t_v: Vec<f32> = (0..32).map(|i| 100.0 + i as f32).collect();
    let l = candle_core::Tensor::new(cond_v.as_slice(), &dev).unwrap().reshape(shape.clone()).unwrap();
    let lc = candle_core::Tensor::new(cond_v.as_slice(), &cpu).unwrap().reshape(shape.clone()).unwrap();
    let t = candle_core::Tensor::new(t_v.as_slice(), &dev).unwrap().reshape(shape.clone()).unwrap();
    let tc = candle_core::Tensor::new(t_v.as_slice(), &cpu).unwrap().reshape(shape.clone()).unwrap();
    // on_false: a single scalar (broadcast view) and a full tensor.
    let f_scalar_dev = candle_core::Tensor::new(-1.0f32, &dev).unwrap();
    let f_scalar_cpu = candle_core::Tensor::new(-1.0f32, &cpu).unwrap();
    let f_v: Vec<f32> = (0..32).map(|i| -50.0 - i as f32).collect();
    let f = candle_core::Tensor::new(f_v.as_slice(), &dev).unwrap().reshape(shape.clone()).unwrap();
    let fc = candle_core::Tensor::new(f_v.as_slice(), &cpu).unwrap().reshape(shape.clone()).unwrap();

    let g1 = l.gt(0.0f32).unwrap().where_cond(&t, &f_scalar_dev.broadcast_as(shape.clone()).unwrap()).unwrap();
    let c1 = lc.gt(0.0f32).unwrap().where_cond(&tc, &f_scalar_cpu.broadcast_as(shape.clone()).unwrap()).unwrap();
    let g2 = l.lt(0.0f32).unwrap().where_cond(&t, &f).unwrap();
    let c2 = lc.lt(0.0f32).unwrap().where_cond(&tc, &fc).unwrap();
    for (name, g, c) in [("where_scalar", g1, c1), ("where_tensor", g2, c2)] {
        let gv: Vec<f32> = g.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
        let cv: Vec<f32> = c.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
        assert_eq!(gv.len(), cv.len());
        for (i, (a, b)) in gv.iter().zip(cv.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "where {name}: mismatch at {i}: gpu {a} cpu {b}");
        }
    }
    tracing::debug!("Vulkan device {gpu_id} where_cond OK");
}

/// `affine` (y = x * mul + add) vs the CPU backend.
#[test]
fn test_vulkan_affine() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let dev = candle_core::Device::new_vulkan(gpu_id).unwrap();
    let cpu = candle_core::Device::Cpu;
    let v: Vec<f32> = (0..64).map(|i| (i as f32) * 0.25 - 8.0).collect();
    let g = candle_core::Tensor::new(v.as_slice(), &dev).unwrap().reshape(candle_core::Shape::from((4, 16))).unwrap();
    let c = candle_core::Tensor::new(v.as_slice(), &cpu).unwrap().reshape(candle_core::Shape::from((4, 16))).unwrap();
    let (mul, add) = (2.5f64, -1.0f64);
    let gv: Vec<f32> = g.affine(mul, add).unwrap().to_vec2::<f32>().unwrap().into_iter().flatten().collect();
    let cv: Vec<f32> = c.affine(mul, add).unwrap().to_vec2::<f32>().unwrap().into_iter().flatten().collect();
    assert_eq!(gv.len(), cv.len());
    for (i, (a, b)) in gv.iter().zip(cv.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5 * (1.0 + a.abs().max(b.abs())),
            "affine: mismatch at {i}: gpu {a} cpu {b}"
        );
    }
    // mul = 0 -> constant fill via affine.
    let g2 = candle_core::Tensor::new(v.as_slice(), &dev).unwrap().affine(0.0f64, 3.5f64).unwrap();
    let c2 = candle_core::Tensor::new(v.as_slice(), &cpu).unwrap().affine(0.0f64, 3.5f64).unwrap();
    for (a, b) in g2.to_vec1::<f32>().unwrap().iter().zip(c2.to_vec1::<f32>().unwrap().iter()) {
        assert!((a - b).abs() < 1e-6);
    }
    tracing::debug!("Vulkan device {gpu_id} affine OK");
}

/// `powf` (x^e) vs the CPU backend.
#[test]
fn test_vulkan_powf() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let dev = candle_core::Device::new_vulkan(gpu_id).unwrap();
    let cpu = candle_core::Device::Cpu;
    let v: Vec<f32> = (0..64).map(|i| 0.5 + i as f32 * 0.0625).collect();
    for e in [2.0f64, 0.5f64, 3.0f64] {
        let g = candle_core::Tensor::new(v.as_slice(), &dev).unwrap().powf(e).unwrap();
        let c = candle_core::Tensor::new(v.as_slice(), &cpu).unwrap().powf(e).unwrap();
        let gv = g.to_vec1::<f32>().unwrap();
        let cv = c.to_vec1::<f32>().unwrap();
        assert_eq!(gv.len(), cv.len());
        for (i, (a, b)) in gv.iter().zip(cv.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4 * (1.0 + b.abs()),
                "powf e={e}: mismatch at {i}: gpu {a} cpu {b}"
            );
        }
    }
    tracing::debug!("Vulkan device {gpu_id} powf OK");
}

/// `elu` (x >= 0 ? x : alpha * (exp(x) - 1)) vs the CPU backend.
#[test]
fn test_vulkan_elu() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let dev = candle_core::Device::new_vulkan(gpu_id).unwrap();
    let cpu = candle_core::Device::Cpu;
    let v: Vec<f32> = (0..64).map(|i| (i as f32) * 0.2 - 6.0).collect();
    for alpha in [1.0f64, 0.5f64] {
        let g = candle_core::Tensor::new(v.as_slice(), &dev).unwrap().elu(alpha).unwrap();
        let c = candle_core::Tensor::new(v.as_slice(), &cpu).unwrap().elu(alpha).unwrap();
        let gv = g.to_vec1::<f32>().unwrap();
        let cv = c.to_vec1::<f32>().unwrap();
        assert_eq!(gv.len(), cv.len());
        for (i, (a, b)) in gv.iter().zip(cv.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5 * (1.0 + b.abs()),
                "elu alpha={alpha}: mismatch at {i}: gpu {a} cpu {b}"
            );
        }
    }
    tracing::debug!("Vulkan device {gpu_id} elu OK");
}

/// `to_dtype` conversions (F32/F16/BF16) vs the CPU backend.
#[test]
fn test_vulkan_to_dtype() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let dev = candle_core::Device::new_vulkan(gpu_id).unwrap();
    let cpu = candle_core::Device::Cpu;
    let v: Vec<f32> = (0..64).map(|i| (i as f32) * 0.3 - 8.0).collect();
    let base_g = candle_core::Tensor::new(v.as_slice(), &dev).unwrap();
    let base_c = candle_core::Tensor::new(v.as_slice(), &cpu).unwrap();

    // F32 -> F16 -> F32 round trip.
    let r1g = base_g.clone().to_dtype(candle_core::DType::F16).unwrap().to_dtype(candle_core::DType::F32).unwrap();
    let r1c = base_c.clone().to_dtype(candle_core::DType::F16).unwrap().to_dtype(candle_core::DType::F32).unwrap();
    for (a, b) in r1g.to_vec1::<f32>().unwrap().iter().zip(r1c.to_vec1::<f32>().unwrap().iter()) {
        assert!((a - b).abs() < 1e-3 * (1.0 + b.abs()), "f32->f16->f32: gpu {a} cpu {b}");
    }
    // F32 -> BF16 -> F32 round trip (coarser: 8 mantissa bits).
    let r2g = base_g.clone().to_dtype(candle_core::DType::BF16).unwrap().to_dtype(candle_core::DType::F32).unwrap();
    let r2c = base_c.clone().to_dtype(candle_core::DType::BF16).unwrap().to_dtype(candle_core::DType::F32).unwrap();
    for (a, b) in r2g.to_vec1::<f32>().unwrap().iter().zip(r2c.to_vec1::<f32>().unwrap().iter()) {
        assert!((a - b).abs() < 1e-2 * (1.0 + b.abs()), "f32->bf16->f32: gpu {a} cpu {b}");
    }
    // Direct dtype comparison on the GPU side via back-conversion equality.
    let f16g = base_g.clone().to_dtype(candle_core::DType::F16).unwrap();
    let bf16g = base_g.clone().to_dtype(candle_core::DType::BF16).unwrap();
    assert_eq!(f16g.dtype(), candle_core::DType::F16);
    assert_eq!(bf16g.dtype(), candle_core::DType::BF16);
    let f16c = base_c.clone().to_dtype(candle_core::DType::F16).unwrap();
    let bf16c = base_c.clone().to_dtype(candle_core::DType::BF16).unwrap();
    let f16g_b: Vec<f32> = f16g.to_dtype(candle_core::DType::F32).unwrap().to_vec1::<f32>().unwrap();
    let f16c_b: Vec<f32> = f16c.to_dtype(candle_core::DType::F32).unwrap().to_vec1::<f32>().unwrap();
    for (a, b) in f16g_b.iter().zip(f16c_b.iter()) {
        assert!((a - b).abs() < 1e-6, "f16 value: gpu {a} cpu {b}");
    }
    let bf16g_b: Vec<f32> = bf16g.to_dtype(candle_core::DType::F32).unwrap().to_vec1::<f32>().unwrap();
    let bf16c_b: Vec<f32> = bf16c.to_dtype(candle_core::DType::F32).unwrap().to_vec1::<f32>().unwrap();
    for (a, b) in bf16g_b.iter().zip(bf16c_b.iter()) {
        assert!((a - b).abs() < 1e-6, "bf16 value: gpu {a} cpu {b}");
    }
    tracing::debug!("Vulkan device {gpu_id} to_dtype OK");
}

/// `copy_strided_src` via `slice_scatter0` (contiguous src at a nonzero dst
/// offset) and via `contiguous()` on a transposed view.
#[test]
fn test_vulkan_copy_strided_src() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let dev = candle_core::Device::new_vulkan(gpu_id).unwrap();
    let cpu = candle_core::Device::Cpu;
    let base_v: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let src_v: Vec<f32> = (0..16).map(|i| 100.0 + i as f32).collect();
    let base = candle_core::Tensor::new(base_v.as_slice(), &dev).unwrap().reshape(candle_core::Shape::from((4, 8))).unwrap();
    let base_c = candle_core::Tensor::new(base_v.as_slice(), &cpu).unwrap().reshape(candle_core::Shape::from((4, 8))).unwrap();
    let src = candle_core::Tensor::new(src_v.as_slice(), &dev).unwrap().reshape(candle_core::Shape::from((2, 8))).unwrap();
    let src_c = candle_core::Tensor::new(src_v.as_slice(), &cpu).unwrap().reshape(candle_core::Shape::from((2, 8))).unwrap();
    let g = base.clone().slice_scatter0(&src, 1).unwrap();
    let c = base_c.clone().slice_scatter0(&src_c, 1).unwrap();
    let gv: Vec<f32> = g.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
    let cv: Vec<f32> = c.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
    assert_eq!(gv, cv);

    // Transposed view -> contiguous (strided source).
    let t_g = base.clone().t().unwrap();
    let t_c = base_c.clone().t().unwrap();
    let cg = t_g.contiguous().unwrap();
    let cc = t_c.contiguous().unwrap();
    let gv: Vec<f32> = cg.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
    let cv: Vec<f32> = cc.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
    assert_eq!(gv, cv);
    tracing::debug!("Vulkan device {gpu_id} copy_strided_src OK");
}

/// `copy2d` via `Tensor::cat` on a non-contiguous (transposed) source.
#[test]
fn test_vulkan_copy2d() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let dev = candle_core::Device::new_vulkan(gpu_id).unwrap();
    let cpu = candle_core::Device::Cpu;
    let a_v: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let b_v: Vec<f32> = (0..24).map(|i| 100.0 + i as f32).collect();
    let a = candle_core::Tensor::new(a_v.as_slice(), &dev).unwrap().reshape(candle_core::Shape::from((4, 6))).unwrap();
    let b = candle_core::Tensor::new(b_v.as_slice(), &dev).unwrap().reshape(candle_core::Shape::from((4, 6))).unwrap();
    let ac = candle_core::Tensor::new(a_v.as_slice(), &cpu).unwrap().reshape(candle_core::Shape::from((4, 6))).unwrap();
    let bc = candle_core::Tensor::new(b_v.as_slice(), &cpu).unwrap().reshape(candle_core::Shape::from((4, 6))).unwrap();
    // cat along dim 0 (contiguous rows) and dim 1.
    let g0 = candle_core::Tensor::cat(&[a.clone(), b.clone()], 0).unwrap();
    let c0 = candle_core::Tensor::cat(&[ac.clone(), bc.clone()], 0).unwrap();
    assert_eq!(g0.to_vec2::<f32>().unwrap(), c0.to_vec2::<f32>().unwrap());
    let g1 = candle_core::Tensor::cat(&[a.clone(), b.clone()], 1).unwrap();
    let c1 = candle_core::Tensor::cat(&[ac.clone(), bc.clone()], 1).unwrap();
    assert_eq!(g1.to_vec2::<f32>().unwrap(), c1.to_vec2::<f32>().unwrap());
    // Non-contiguous (transposed) source in a cat.
    let at = a.t().unwrap();
    let bt = b.t().unwrap();
    let g2 = candle_core::Tensor::cat(&[at, bt], 0).unwrap();
    let c2 = candle_core::Tensor::cat(&[ac.t().unwrap(), bc.t().unwrap()], 0).unwrap();
    assert_eq!(g2.to_vec2::<f32>().unwrap(), c2.to_vec2::<f32>().unwrap());
    tracing::debug!("Vulkan device {gpu_id} copy2d OK");
}

/// `reduce_op` over the last axis: sum, max, min, argmax, argmin.
#[test]
fn test_vulkan_reduce_op() {
    (*INIT);
    let gpu_id = *GPU_ID;
    let dev = candle_core::Device::new_vulkan(gpu_id).unwrap();
    let cpu = candle_core::Device::Cpu;
    let shape = candle_core::Shape::from((4, 8));
    let v: Vec<f32> = (0..32).map(|i| (i as f32) * 0.7 - 10.0).collect();
    let g = candle_core::Tensor::new(v.as_slice(), &dev).unwrap().reshape(shape.clone()).unwrap();
    let c = candle_core::Tensor::new(v.as_slice(), &cpu).unwrap().reshape(shape.clone()).unwrap();

    let (gs, cs) = (g.clone().sum(1).unwrap(), c.clone().sum(1).unwrap());
    for (a, b) in gs.to_vec1::<f32>().unwrap().iter().zip(cs.to_vec1::<f32>().unwrap().iter()) {
        assert!((a - b).abs() < 1e-4 * (1.0 + b.abs()), "sum: gpu {a} cpu {b}");
    }
    let (gm, cm) = (g.clone().max(1).unwrap(), c.clone().max(1).unwrap());
    assert_eq!(gm.to_vec1::<f32>().unwrap(), cm.to_vec1::<f32>().unwrap());
    let (gn, cn) = (g.clone().min(1).unwrap(), c.clone().min(1).unwrap());
    assert_eq!(gn.to_vec1::<f32>().unwrap(), cn.to_vec1::<f32>().unwrap());
    let (ga, ca) = (g.clone().argmax(1).unwrap(), c.clone().argmax(1).unwrap());
    assert_eq!(ga.to_vec1::<u32>().unwrap(), ca.to_vec1::<u32>().unwrap());
    let (gb, cb) = (g.clone().argmin(1).unwrap(), c.clone().argmin(1).unwrap());
    assert_eq!(gb.to_vec1::<u32>().unwrap(), cb.to_vec1::<u32>().unwrap());
    tracing::debug!("Vulkan device {gpu_id} reduce_op OK");
}
