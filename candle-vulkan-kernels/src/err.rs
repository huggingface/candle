//! Errors from the candle-vulkan-kernels crate.

use thiserror::Error;

/// Errors that can occur when building or dispatching Vulkan compute kernels.
#[derive(Debug, Error)]
pub enum VulkanKernelError {
    #[error("failed to parse GLSL source: {0}")]
    ParseError(String),

    #[error("naga validation failed: {0}")]
    ValidationError(String),

    #[error("failed to write SPIR-V: {0}")]
    SpvError(String),

    #[error("shader entry point not found")]
    EntryPoint,

    #[error("failed to create descriptor set: {0}")]
    DescriptorSet(String),

    #[error("failed to build compute pipeline: {0}")]
    Pipeline(String),

    #[error("command buffer error: {0}")]
    CommandBuffer(String),

    #[error("failed to submit work: {0}")]
    Execute(String),

    #[error("failed to wait on GPU: {0}")]
    Fence(String),

    #[error("buffer error: {0}")]
    Buffer(String),

    #[error("{0}")]
    Message(String),
}

impl From<String> for VulkanKernelError {
    fn from(e: String) -> Self {
        VulkanKernelError::Message(e)
    }
}
