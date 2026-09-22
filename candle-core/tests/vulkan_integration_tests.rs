//! Integration tests for the Vulkan support.
//!
//! Device selection:
//! `CANDLE_VULKAN_TEST_GPU=<n> cargo test --features vulkan --test vulkan_integration_tests`
//! (defaults to physical device 0).

#![cfg(feature = "vulkan")]

use std::sync::LazyLock;

use candle_core::{backend::BackendDevice, VulkanDevice};

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
    let data = vec![0usize; 256];
    let shape = candle_core::Shape::from(data);
    let dtype = candle_core::DType::F16;
    let storage = device.zeros_impl(&shape, dtype).unwrap();
    tracing::debug!("Vulkan device {gpu_id} storage: {:?}", storage.buffer());
}
