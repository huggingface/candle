//! Turning float weights into the packed layout `QRocmStorage` holds.
//!
//! There is no device-side quantizer here, and the CUDA backend has none
//! either: `quantized/cuda.rs` copies the weights to the host, runs the same
//! `k_quants` code the CPU backend runs, and uploads the result. Quantization
//! happens once at load time, so a kernel would buy nothing; matching the CPU
//! bit for bit, on the other hand, is what lets a ROCm-quantized tensor be
//! compared against a CPU-quantized one in the tests.

use super::{upload, GgmlDType, QRocmStorage, QStorage, RocmStorage, RocmStorageSlice};
use crate::{CpuStorage, Result};

/// Whether `k_quants` implements `from_float_imatrix` for `dtype`.
///
/// The trait's default body is a `panic!`, and the dtype reaching it comes from
/// a config file or a CLI flag, so the check has to happen before the call.
fn supports_imatrix(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q2K | GgmlDType::Q3K | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K
    )
}

impl QRocmStorage {
    pub fn quantize(&mut self, src: &RocmStorage) -> Result<()> {
        let src = self.src_to_host(src)?;
        self.quantize_from_f32(&src)
    }

    pub fn quantize_onto(&mut self, src: &CpuStorage) -> Result<()> {
        self.quantize_from_f32(src.as_slice::<f32>()?)
    }

    pub fn quantize_imatrix(
        &mut self,
        src: &RocmStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        let src = self.src_to_host(src)?;
        self.quantize_imatrix_from_f32(&src, imatrix_weights, n_per_row)
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        src: &CpuStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        self.quantize_imatrix_from_f32(src.as_slice::<f32>()?, imatrix_weights, n_per_row)
    }

    fn src_to_host(&self, src: &RocmStorage) -> Result<Vec<f32>> {
        match &src.slice {
            RocmStorageSlice::F32(data) => self.device.clone_dtoh(data),
            slice => crate::bail!("only f32 can be quantized, got {:?}", slice.dtype()),
        }
    }

    /// Quantization itself runs on the CPU; only the upload is device side.
    fn quantize_from_f32(&mut self, src: &[f32]) -> Result<()> {
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src.len(), self.dtype)?;
        match &mut qcpu_storage {
            QStorage::Cpu(storage) => storage.from_float(src),
            _ => crate::bail!("Device::Cpu.qzeros did not return a cpu storage"),
        }
        self.store(qcpu_storage)
    }

    fn quantize_imatrix_from_f32(
        &mut self,
        src: &[f32],
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        if !supports_imatrix(self.dtype) {
            crate::bail!(
                "imatrix quantization is not implemented for {:?}",
                self.dtype
            )
        }
        if imatrix_weights.len() != n_per_row {
            crate::bail!(
                "imatrix has {} weights, expected one per row element ({n_per_row})",
                imatrix_weights.len()
            )
        }
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src.len(), self.dtype)?;
        match &mut qcpu_storage {
            QStorage::Cpu(storage) => storage.from_float_imatrix(src, imatrix_weights, n_per_row),
            _ => crate::bail!("Device::Cpu.qzeros did not return a cpu storage"),
        }
        self.store(qcpu_storage)
    }

    /// Replace the device buffer with the freshly quantized payload.
    fn store(&mut self, qcpu_storage: QStorage) -> Result<()> {
        let data = qcpu_storage.data()?;
        self.data = upload(&self.device, &data, self.dtype)?;
        self.len = data.len();
        Ok(())
    }
}
