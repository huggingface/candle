// Created by: TEAM-502
// Dummy ROCm quantization module for when ROCm feature is disabled
// CUDA parity: cuda_backend/quantized/cuda.rs (structure)
//
// This module provides stub implementations that should never be called.
// The QStorage::Rocm variant is gated behind #[cfg(feature = "rocm")],
// so these stubs only exist to satisfy the type system.

use super::GgmlDType;

/// Dummy QRocmStorage that should never be instantiated
/// 
/// This type exists only to satisfy the module system when the rocm feature is disabled.
/// The actual QStorage::Rocm variant is gated behind #[cfg(feature = "rocm")],
/// so this type should never actually be used.
#[derive(Clone, Debug)]
pub struct QRocmStorage {
    _private: (),
}
