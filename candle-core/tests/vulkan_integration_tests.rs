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
