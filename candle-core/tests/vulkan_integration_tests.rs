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
    tracing::debug!("Vulkan device {gpu_id} fill_f16 round-tripped OK");
}
