//! Module to load `safetensor` files into CPU/GPU memory.
//!
//! There are multiple ways to load tensors from safetensor files:
//! - `load` function for loading directly into memory and returning a HashMap of tensors
//! - `MmapedSafetensors` for memory mapping files and avoiding full allocation
//! - `SliceSafetensors` for working with in-memory buffers
//! - `BufferedSafetensors` for owning a buffer of data
//!
//! Tensors can also be serialized to safetensor format using the `save` function or
//! `Tensor::save_safetensors` method.
//!
//! # Endianness Handling
//!
//! SafeTensors stores all multi-byte data in little-endian format. On big-endian systems,
//! this module automatically converts between little-endian (file format) and big-endian
//! (system format) by reversing byte order for multi-byte types. Single-byte types (U8, F8E4M3)
//! require no conversion. This conversion is applied during both loading and saving operations.
//!
//! **Note on Dummy Types**: The dummy types (F6E2M3, F6E3M2, F4, F8E8M0) store raw packed bytes
//! whose internal structure is opaque. Loading or saving these types on big-endian systems will
//! return an error, as we cannot safely perform byte swapping without knowing their internal format.
//! These types are only supported on little-endian architectures.
//!
use crate::op::BackpropOp;
use crate::storage::Storage;
use crate::tensor::from_storage;
use crate::{DType, Device, Error, Result, Tensor, WithDType};
use safetensors::tensor as st;
use safetensors::tensor::SafeTensors;
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

impl From<DType> for st::Dtype {
    fn from(value: DType) -> Self {
        match value {
            DType::U8 => st::Dtype::U8,
            DType::U32 => st::Dtype::U32,
            DType::I16 => st::Dtype::I16,
            DType::I32 => st::Dtype::I32,
            DType::I64 => st::Dtype::I64,
            DType::BF16 => st::Dtype::BF16,
            DType::F16 => st::Dtype::F16,
            DType::F32 => st::Dtype::F32,
            DType::F64 => st::Dtype::F64,
            DType::F8E4M3 => st::Dtype::F8_E4M3,
            DType::F6E2M3 => st::Dtype::F6_E2M3,
            DType::F6E3M2 => st::Dtype::F6_E3M2,
            DType::F4 => st::Dtype::F4,
            DType::F8E8M0 => st::Dtype::F8_E8M0,
        }
    }
}

impl TryFrom<st::Dtype> for DType {
    type Error = Error;
    fn try_from(value: st::Dtype) -> Result<Self> {
        match value {
            st::Dtype::U8 => Ok(DType::U8),
            st::Dtype::U32 => Ok(DType::U32),
            st::Dtype::I16 => Ok(DType::I16),
            st::Dtype::I32 => Ok(DType::I32),
            st::Dtype::I64 => Ok(DType::I64),
            st::Dtype::BF16 => Ok(DType::BF16),
            st::Dtype::F16 => Ok(DType::F16),
            st::Dtype::F32 => Ok(DType::F32),
            st::Dtype::F64 => Ok(DType::F64),
            st::Dtype::F8_E4M3 => Ok(DType::F8E4M3),
            st::Dtype::F6_E2M3 => Ok(DType::F6E2M3),
            st::Dtype::F6_E3M2 => Ok(DType::F6E3M2),
            st::Dtype::F4 => Ok(DType::F4),
            st::Dtype::F8_E8M0 => Ok(DType::F8E8M0),
            dtype => Err(Error::UnsupportedSafeTensorDtype(dtype)),
        }
    }
}

impl st::View for Tensor {
    fn dtype(&self) -> st::Dtype {
        self.dtype().into()
    }
    fn shape(&self) -> &[usize] {
        self.shape().dims()
    }

    fn data(&self) -> Cow<'_, [u8]> {
        // This copies data from GPU to CPU.
        // TODO: Avoid the unwrap here.
        Cow::Owned(convert_back(self).unwrap())
    }

    fn data_len(&self) -> usize {
        let n: usize = self.shape().elem_count();
        let bytes_per_element = self.dtype().size_in_bytes();
        n * bytes_per_element
    }
}

impl st::View for &Tensor {
    fn dtype(&self) -> st::Dtype {
        (*self).dtype().into()
    }
    fn shape(&self) -> &[usize] {
        self.dims()
    }

    fn data(&self) -> Cow<'_, [u8]> {
        // This copies data from GPU to CPU.
        // TODO: Avoid the unwrap here.
        Cow::Owned(convert_back(self).unwrap())
    }

    fn data_len(&self) -> usize {
        let n: usize = self.dims().iter().product();
        let bytes_per_element = (*self).dtype().size_in_bytes();
        n * bytes_per_element
    }
}

impl Tensor {
    pub fn save_safetensors<P: AsRef<Path>>(&self, name: &str, filename: P) -> Result<()> {
        let data = [(name, self.clone())];
        Ok(st::serialize_to_file(data, None, filename.as_ref())?)
    }
}

/// Convert byte slice to appropriate endianness for the current system.
/// SafeTensors uses little-endian format; this swaps bytes on big-endian systems.
#[inline]
fn maybe_swap_endianness(data: &[u8], _size_in_bytes: usize) -> Cow<'_, [u8]> {
    #[cfg(target_endian = "big")]
    {
        if _size_in_bytes == 1 {
            Cow::Borrowed(data)
        } else {
            let mut swapped = data.to_vec();
            for chunk in swapped.chunks_exact_mut(_size_in_bytes) {
                chunk.reverse();
            }
            Cow::Owned(swapped)
        }
    }
    #[cfg(target_endian = "little")]
    {
        Cow::Borrowed(data)
    }
}

/// Handle endianness conversion for dummy types.
/// Returns an error on big-endian systems since the internal structure is opaque
/// and we cannot safely swap bytes without knowing the format.
fn maybe_swap_dummy_endianness(data: &[u8]) -> Result<Cow<'_, [u8]>> {
    #[cfg(target_endian = "big")]
    {
        // Dummy types store raw bytes with opaque internal structure.
        // Without knowing the format, we cannot safely swap bytes.
        Err(Error::Msg(
            "Dummy types (F6E2M3, F6E3M2, F4, F8E8M0) are not supported on big-endian systems due to unknown internal byte structure".to_string()
        ))
    }
    #[cfg(target_endian = "little")]
    {
        Ok(Cow::Borrowed(data))
    }
}

/// Checks if data is properly aligned for the given size in bytes.
/// On big-endian systems with multi-byte types, swapped data may not be aligned.
fn check_alignment(data: &Cow<'_, [u8]>, size_in_bytes: usize) -> bool {
    #[cfg(target_endian = "big")]
    {
        let is_aligned = (data.as_ptr() as usize).is_multiple_of(size_in_bytes);
        if size_in_bytes > 1 && matches!(data, Cow::Owned(_)) {
            // Swapped data from Vec may not maintain alignment, use unaligned path
            false
        } else {
            is_aligned
        }
    }
    
    #[cfg(target_endian = "little")]
    {
        (data.as_ptr() as usize).is_multiple_of(size_in_bytes)
    }
}

fn convert_slice<T: WithDType>(data: &[u8], shape: &[usize], device: &Device) -> Result<Tensor> {
    let size_in_bytes = T::DTYPE.size_in_bytes();
    let elem_count = data.len() / size_in_bytes;
    let data = maybe_swap_endianness(data, size_in_bytes);
    let is_aligned = check_alignment(&data, size_in_bytes);
    
    if is_aligned {
        // SAFETY This is safe because we just checked that this
        // was correctly aligned.
        let data: &[T] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const T, elem_count) };
        Tensor::from_slice(data, shape, device)
    } else {
        // XXX: We need to specify `T` here, otherwise the compiler will infer u8 because of the following cast
        // Making this vector too small to fit a full f16/f32/f64 weights, resulting in out-of-bounds access
        let mut c: Vec<T> = Vec::with_capacity(elem_count);
        // SAFETY: We just created c, so the allocated memory is necessarily
        // contiguous and non overlapping with the view's data.
        // We're downgrading the `c` pointer from T to u8, which removes alignment
        // constraints.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), c.as_mut_ptr() as *mut u8, data.len());
            c.set_len(elem_count)
        }
        Tensor::from_slice(&c, shape, device)
    }
}

fn convert_slice_with_cast<T: Sized + Copy, U: WithDType, F: Fn(T) -> Result<U>>(
    data: &[u8],
    shape: &[usize],
    device: &Device,
    conv: F,
) -> Result<Tensor> {
    let size_in_bytes = std::mem::size_of::<T>();
    let elem_count = data.len() / size_in_bytes;
    let data = maybe_swap_endianness(data, size_in_bytes);
    let is_aligned = check_alignment(&data, size_in_bytes);
    
    if is_aligned {
        // SAFETY This is safe because we just checked that this
        // was correctly aligned.
        let data: &[T] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const T, elem_count) };
        let data = data.iter().map(|t| conv(*t)).collect::<Result<Vec<_>>>()?;
        Tensor::from_vec(data, shape, device)
    } else {
        // XXX: We need to specify `T` here, otherwise the compiler will infer u8 because of the following cast
        // Making this vector too small to fit a full f16/f32/f64 weights, resulting in out-of-bounds access
        let mut c: Vec<T> = Vec::with_capacity(elem_count);
        // SAFETY: We just created c, so the allocated memory is necessarily
        // contiguous and non overlapping with the view's data.
        // We're downgrading the `c` pointer from T to u8, which removes alignment
        // constraints.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), c.as_mut_ptr() as *mut u8, data.len());
            c.set_len(elem_count)
        }
        let c = c.into_iter().map(conv).collect::<Result<Vec<_>>>()?;
        Tensor::from_vec(c, shape, device)
    }
}

fn convert_with_cast_<T: Sized + Copy, U: WithDType, F: Fn(T) -> Result<U>>(
    view: &st::TensorView<'_>,
    device: &Device,
    conv: F,
) -> Result<Tensor> {
    convert_slice_with_cast::<T, U, F>(view.data(), view.shape(), device, conv)
}

fn convert_<T: WithDType>(view: &st::TensorView<'_>, device: &Device) -> Result<Tensor> {
    convert_slice::<T>(view.data(), view.shape(), device)
}

fn convert_back_<T: WithDType>(mut vs: Vec<T>) -> Vec<u8> {
    let size_in_bytes = T::DTYPE.size_in_bytes();
    let length = vs.len() * size_in_bytes;
    let capacity = vs.capacity() * size_in_bytes;
    let ptr = vs.as_mut_ptr() as *mut u8;
    std::mem::forget(vs);
    
    // SAFETY: Every T is larger than u8, so no alignment issues.
    // Re-interpret Vec<T> as Vec<u8>, then swap bytes for big-endian.
    let bytes = unsafe { Vec::from_raw_parts(ptr, length, capacity) };
    
    #[cfg(target_endian = "big")]
    {
        if size_in_bytes > 1 {
            let mut bytes = bytes;
            for chunk in bytes.chunks_exact_mut(size_in_bytes) {
                chunk.reverse();
            }
            return bytes;
        }
    }
    
    bytes
}

pub trait Load {
    fn load(&self, device: &Device) -> Result<Tensor>;
}

impl Load for st::TensorView<'_> {
    fn load(&self, device: &Device) -> Result<Tensor> {
        convert(self, device)
    }
}

impl Tensor {
    pub fn from_raw_buffer(
        data: &[u8],
        dtype: DType,
        shape: &[usize],
        device: &Device,
    ) -> Result<Self> {
        match dtype {
            DType::U8 => convert_slice::<u8>(data, shape, device),
            DType::U32 => convert_slice::<u32>(data, shape, device),
            DType::I16 => convert_slice::<i16>(data, shape, device),
            DType::I32 => convert_slice::<i32>(data, shape, device),
            DType::I64 => convert_slice::<i64>(data, shape, device),
            DType::BF16 => convert_slice::<half::bf16>(data, shape, device),
            DType::F16 => convert_slice::<half::f16>(data, shape, device),
            DType::F32 => convert_slice::<f32>(data, shape, device),
            DType::F64 => convert_slice::<f64>(data, shape, device),
            DType::F8E4M3 => convert_slice::<float8::F8E4M3>(data, shape, device),
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                // For dummy types, create storage with raw bytes (see module docs for limitations)
                let data = maybe_swap_dummy_endianness(data)?;
                let storage = match device {
                    Device::Cpu => {
                        let cpu_storage = match dtype {
                            DType::F6E2M3 => crate::cpu_backend::CpuStorage::F6E2M3(data.to_vec()),
                            DType::F6E3M2 => crate::cpu_backend::CpuStorage::F6E3M2(data.to_vec()),
                            DType::F4 => crate::cpu_backend::CpuStorage::F4(data.to_vec()),
                            DType::F8E8M0 => crate::cpu_backend::CpuStorage::F8E8M0(data.to_vec()),
                            _ => unreachable!(),
                        };
                        Storage::Cpu(cpu_storage)
                    }
                    #[cfg(feature = "cuda")]
                    Device::Cuda(device) => {
                        let mut slice = unsafe { device.alloc::<u8>(data.len())? };
                        device.memcpy_htod(data, &mut slice)?;

                        let slice = match dtype {
                            DType::F6E2M3 => crate::cuda_backend::CudaStorageSlice::F6E2M3(slice),
                            DType::F6E3M2 => crate::cuda_backend::CudaStorageSlice::F6E3M2(slice),
                            DType::F4 => crate::cuda_backend::CudaStorageSlice::F4(slice),
                            DType::F8E8M0 => crate::cuda_backend::CudaStorageSlice::F8E8M0(slice),
                            _ => unreachable!(),
                        };
                        let storage = crate::cuda_backend::CudaStorage {
                            slice,
                            device: device.clone(),
                        };
                        Storage::Cuda(storage)
                    }
                    #[cfg(not(feature = "cuda"))]
                    Device::Cuda(_) => {
                        return Err(Error::Msg("CUDA support not compiled".to_string()));
                    }
                    #[cfg(feature = "metal")]
                    Device::Metal(device) => {
                        let buffer = device.new_buffer_with_data(data)?;

                        let storage = crate::metal_backend::MetalStorage::new(
                            buffer,
                            device.clone(),
                            data.len(),
                            dtype,
                        );
                        Storage::Metal(storage)
                    }
                    #[cfg(not(feature = "metal"))]
                    Device::Metal(_) => {
                        return Err(Error::Msg("Metal support not compiled".to_string()));
                    }
                };

                let op = BackpropOp::none();
                Ok(from_storage(storage, shape, op, false))
            }
        }
    }
}

fn convert(view: &st::TensorView<'_>, device: &Device) -> Result<Tensor> {
    match view.dtype() {
        st::Dtype::U8 => convert_::<u8>(view, device),
        st::Dtype::U16 => {
            let conv = |x| Ok(u32::from(x));
            convert_with_cast_::<u16, u32, _>(view, device, conv)
        }
        st::Dtype::U32 => convert_::<u32>(view, device),
        st::Dtype::I16 => convert_::<i16>(view, device),
        st::Dtype::I32 => convert_::<i32>(view, device),
        st::Dtype::I64 => convert_::<i64>(view, device),
        st::Dtype::BF16 => convert_::<half::bf16>(view, device),
        st::Dtype::F16 => convert_::<half::f16>(view, device),
        st::Dtype::F32 => convert_::<f32>(view, device),
        st::Dtype::F64 => convert_::<f64>(view, device),
        st::Dtype::F8_E4M3 => convert_::<float8::F8E4M3>(view, device),
        st::Dtype::F6_E2M3 | st::Dtype::F6_E3M2 | st::Dtype::F4 | st::Dtype::F8_E8M0 => {
            // For dummy types, we need to handle loading by creating a dummy tensor
            // Since these types don't have actual data representation, we'll create
            // a tensor that indicates it's a dummy type
            convert_dummy(view, device)
        }
        dtype => Err(Error::UnsupportedSafeTensorDtype(dtype)),
    }
}

fn convert_dummy(view: &st::TensorView<'_>, device: &Device) -> Result<Tensor> {
    // For dummy types, we'll create the appropriate storage variant that preserves
    // both the raw data and the correct dtype
    let (dtype, _dtype_name) = match view.dtype() {
        st::Dtype::F6_E2M3 => (DType::F6E2M3, "F6_E2M3 (MX6)"),
        st::Dtype::F6_E3M2 => (DType::F6E3M2, "F6_E3M2 (MX6)"),
        st::Dtype::F4 => (DType::F4, "F4 (MX4)"),
        st::Dtype::F8_E8M0 => (DType::F8E8M0, "F8_E8M0"),
        _ => unreachable!("convert_dummy called with non-dummy dtype"),
    };

    let data = view.data();
    let shape = view.shape();
    let data = maybe_swap_dummy_endianness(data)?;

    // Create storage with the appropriate dummy type variant
    let storage = match device {
        Device::Cpu => {
            let cpu_storage = match dtype {
                DType::F6E2M3 => crate::cpu_backend::CpuStorage::F6E2M3(data.to_vec()),
                DType::F6E3M2 => crate::cpu_backend::CpuStorage::F6E3M2(data.to_vec()),
                DType::F4 => crate::cpu_backend::CpuStorage::F4(data.to_vec()),
                DType::F8E8M0 => crate::cpu_backend::CpuStorage::F8E8M0(data.to_vec()),
                _ => unreachable!(),
            };
            Storage::Cpu(cpu_storage)
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(device) => {
            let mut slice = unsafe { device.alloc::<u8>(data.len())? };
            device.memcpy_htod(data, &mut slice)?;

            let slice = match dtype {
                DType::F6E2M3 => crate::cuda_backend::CudaStorageSlice::F6E2M3(slice),
                DType::F6E3M2 => crate::cuda_backend::CudaStorageSlice::F6E3M2(slice),
                DType::F4 => crate::cuda_backend::CudaStorageSlice::F4(slice),
                DType::F8E8M0 => crate::cuda_backend::CudaStorageSlice::F8E8M0(slice),
                _ => unreachable!(),
            };
            let storage = crate::cuda_backend::CudaStorage {
                slice,
                device: device.clone(),
            };
            Storage::Cuda(storage)
        }
        #[cfg(not(feature = "cuda"))]
        Device::Cuda(_) => {
            return Err(Error::Msg("CUDA support not compiled".to_string()));
        }
        #[cfg(feature = "metal")]
        Device::Metal(device) => {
            let buffer = device.new_buffer_with_data(data)?;

            let storage =
                crate::metal_backend::MetalStorage::new(buffer, device.clone(), data.len(), dtype);
            Storage::Metal(storage)
        }
        #[cfg(not(feature = "metal"))]
        Device::Metal(_) => {
            return Err(Error::Msg("Metal support not compiled".to_string()));
        }
    };

    // Create tensor with correct dtype
    let op = BackpropOp::none();
    Ok(from_storage(storage, shape, op, false))
}

fn convert_back(tensor: &Tensor) -> Result<Vec<u8>> {
    // TODO: This makes an unnecessary copy when the tensor is on the cpu.
    let tensor = tensor.flatten_all()?;
    match tensor.dtype() {
        DType::U8 => Ok(convert_back_::<u8>(tensor.to_vec1()?)),
        DType::U32 => Ok(convert_back_::<u32>(tensor.to_vec1()?)),
        DType::I16 => Ok(convert_back_::<i16>(tensor.to_vec1()?)),
        DType::I32 => Ok(convert_back_::<i32>(tensor.to_vec1()?)),
        DType::I64 => Ok(convert_back_::<i64>(tensor.to_vec1()?)),
        DType::F16 => Ok(convert_back_::<half::f16>(tensor.to_vec1()?)),
        DType::BF16 => Ok(convert_back_::<half::bf16>(tensor.to_vec1()?)),
        DType::F32 => Ok(convert_back_::<f32>(tensor.to_vec1()?)),
        DType::F64 => Ok(convert_back_::<f64>(tensor.to_vec1()?)),
        DType::F8E4M3 => Ok(convert_back_::<float8::F8E4M3>(tensor.to_vec1()?)),
        DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
            Err(Error::Msg("Internal error: dtype mismatch in storage".to_string()).bt())
        }
    }
}

pub fn load<P: AsRef<Path>>(filename: P, device: &Device) -> Result<HashMap<String, Tensor>> {
    let data = std::fs::read(filename.as_ref())?;
    load_buffer(&data[..], device)
}

pub fn load_buffer(data: &[u8], device: &Device) -> Result<HashMap<String, Tensor>> {
    let st = safetensors::SafeTensors::deserialize(data)?;
    st.tensors()
        .into_iter()
        .map(|(name, view)| Ok((name, view.load(device)?)))
        .collect()
}

pub fn save<K: AsRef<str> + Ord + std::fmt::Display, P: AsRef<Path>>(
    tensors: &HashMap<K, Tensor>,
    filename: P,
) -> Result<()> {
    Ok(st::serialize_to_file(tensors, None, filename.as_ref())?)
}

#[derive(yoke::Yokeable)]
struct SafeTensors_<'a>(SafeTensors<'a>);

pub struct MmapedSafetensors {
    safetensors: Vec<yoke::Yoke<SafeTensors_<'static>, memmap2::Mmap>>,
    routing: Option<HashMap<String, usize>>,
}

impl MmapedSafetensors {
    /// Creates a wrapper around a memory mapped file and deserialize the safetensors header.
    ///
    /// # Safety
    ///
    /// The unsafe is inherited from [`memmap2::MmapOptions`].
    pub unsafe fn new<P: AsRef<Path>>(p: P) -> Result<Self> {
        let p = p.as_ref();
        let file = std::fs::File::open(p).map_err(|e| Error::from(e).with_path(p))?;
        let file = memmap2::MmapOptions::new()
            .map(&file)
            .map_err(|e| Error::from(e).with_path(p))?;
        let safetensors = yoke::Yoke::<SafeTensors_<'static>, memmap2::Mmap>::try_attach_to_cart(
            file,
            |data: &[u8]| {
                let st = safetensors::SafeTensors::deserialize(data)
                    .map_err(|e| Error::from(e).with_path(p))?;
                Ok::<_, Error>(SafeTensors_(st))
            },
        )?;
        Ok(Self {
            safetensors: vec![safetensors],
            routing: None,
        })
    }

    /// Creates a wrapper around multiple memory mapped file and deserialize the safetensors headers.
    ///
    /// If a tensor name appears in multiple files, the last entry is returned.
    ///
    /// # Safety
    ///
    /// The unsafe is inherited from [`memmap2::MmapOptions`].
    pub unsafe fn multi<P: AsRef<Path>>(paths: &[P]) -> Result<Self> {
        let mut routing = HashMap::new();
        let mut safetensors = vec![];
        for (index, p) in paths.iter().enumerate() {
            let p = p.as_ref();
            let file = std::fs::File::open(p).map_err(|e| Error::from(e).with_path(p))?;
            let file = memmap2::MmapOptions::new()
                .map(&file)
                .map_err(|e| Error::from(e).with_path(p))?;
            let data = yoke::Yoke::<SafeTensors_<'static>, memmap2::Mmap>::try_attach_to_cart(
                file,
                |data: &[u8]| {
                    let st = safetensors::SafeTensors::deserialize(data)
                        .map_err(|e| Error::from(e).with_path(p))?;
                    Ok::<_, Error>(SafeTensors_(st))
                },
            )?;
            for k in data.get().0.names() {
                routing.insert(k.to_string(), index);
            }
            safetensors.push(data)
        }
        Ok(Self {
            safetensors,
            routing: Some(routing),
        })
    }

    pub fn load(&self, name: &str, dev: &Device) -> Result<Tensor> {
        self.get(name)?.load(dev)
    }

    pub fn tensors(&self) -> Vec<(String, st::TensorView<'_>)> {
        let mut tensors = vec![];
        for safetensors in self.safetensors.iter() {
            tensors.push(safetensors.get().0.tensors())
        }
        tensors.into_iter().flatten().collect()
    }

    pub fn get(&self, name: &str) -> Result<st::TensorView<'_>> {
        let index = match &self.routing {
            None => 0,
            Some(routing) => {
                let index = routing.get(name).ok_or_else(|| {
                    Error::CannotFindTensor {
                        path: name.to_string(),
                    }
                    .bt()
                })?;
                *index
            }
        };
        Ok(self.safetensors[index].get().0.tensor(name)?)
    }
}

pub struct SliceSafetensors<'a> {
    safetensors: SafeTensors<'a>,
}

impl<'a> SliceSafetensors<'a> {
    /// Creates a wrapper around a binary buffer and deserialize the safetensors header.
    pub fn new(buffer: &'a [u8]) -> Result<Self> {
        let safetensors = safetensors::SafeTensors::deserialize(buffer)?;
        Ok(Self { safetensors })
    }

    pub fn load(&self, name: &str, dev: &Device) -> Result<Tensor> {
        self.safetensors.tensor(name)?.load(dev)
    }

    pub fn tensors(&self) -> Vec<(String, st::TensorView<'_>)> {
        self.safetensors.tensors()
    }

    pub fn get(&self, name: &str) -> Result<st::TensorView<'_>> {
        Ok(self.safetensors.tensor(name)?)
    }
}

pub struct BufferedSafetensors {
    safetensors: yoke::Yoke<SafeTensors_<'static>, Vec<u8>>,
}

impl BufferedSafetensors {
    /// Creates a wrapper around a binary buffer and deserialize the safetensors header.
    pub fn new(buffer: Vec<u8>) -> Result<Self> {
        let safetensors = yoke::Yoke::<SafeTensors_<'static>, Vec<u8>>::try_attach_to_cart(
            buffer,
            |data: &[u8]| {
                let st = safetensors::SafeTensors::deserialize(data)?;
                Ok::<_, Error>(SafeTensors_(st))
            },
        )?;
        Ok(Self { safetensors })
    }

    pub fn load(&self, name: &str, dev: &Device) -> Result<Tensor> {
        self.get(name)?.load(dev)
    }

    pub fn tensors(&self) -> Vec<(String, st::TensorView<'_>)> {
        self.safetensors.get().0.tensors()
    }

    pub fn get(&self, name: &str) -> Result<st::TensorView<'_>> {
        Ok(self.safetensors.get().0.tensor(name)?)
    }
}

pub struct MmapedFile {
    path: std::path::PathBuf,
    inner: memmap2::Mmap,
}

impl MmapedFile {
    /// Creates a wrapper around a memory mapped file from which you can retrieve
    /// tensors using [`MmapedFile::deserialize`]
    ///
    /// # Safety
    ///
    /// The unsafe is inherited from [`memmap2::MmapOptions`].
    pub unsafe fn new<P: AsRef<Path>>(p: P) -> Result<Self> {
        let p = p.as_ref();
        let file = std::fs::File::open(p).map_err(|e| Error::from(e).with_path(p))?;
        let inner = memmap2::MmapOptions::new()
            .map(&file)
            .map_err(|e| Error::from(e).with_path(p))?;
        Ok(Self {
            inner,
            path: p.to_path_buf(),
        })
    }

    pub fn deserialize(&self) -> Result<SafeTensors<'_>> {
        let st = safetensors::SafeTensors::deserialize(&self.inner)
            .map_err(|e| Error::from(e).with_path(&self.path))?;
        Ok(st)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn save_single_tensor() {
        let t = Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap();
        t.save_safetensors("t", "t.safetensors").unwrap();
        let bytes = std::fs::read("t.safetensors").unwrap();
        assert_eq!(bytes, b"@\0\0\0\0\0\0\0{\"t\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}       \0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0");
        std::fs::remove_file("t.safetensors").unwrap();
    }

    #[test]
    fn save_load_multiple_tensors() {
        let t = Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap();
        let u = Tensor::zeros((1, 2), DType::F32, &Device::Cpu).unwrap();
        let map: HashMap<_, _> = [("t", t), ("u", u)].into_iter().collect();
        save(&map, "multi.safetensors").unwrap();

        let weights = load("multi.safetensors", &Device::Cpu).unwrap();
        assert_eq!(weights.get("t").unwrap().dims(), &[2, 2]);
        assert_eq!(weights.get("u").unwrap().dims(), &[1, 2]);
        let bytes = std::fs::read("multi.safetensors").unwrap();
        assert_eq!(bytes, b"x\0\0\0\0\0\0\0{\"t\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]},\"u\":{\"dtype\":\"F32\",\"shape\":[1,2],\"data_offsets\":[16,24]}}      \0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0");
        std::fs::remove_file("multi.safetensors").unwrap();
    }

    #[test]
    fn load_u8() {
        let bytes = b"8\0\0\0\0\0\0\0{\"x\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[0,2]}}   \x01\x03";
        std::fs::write("test_u8.safetensors", bytes).unwrap();
        let weights = load("test_u8.safetensors", &Device::Cpu).unwrap();
        let tensor = weights.get("x").unwrap();
        assert_eq!(tensor.dims(), &[2]);
        assert_eq!(tensor.dtype(), DType::U8);
        let data: Vec<u8> = tensor.to_vec1().unwrap();
        assert_eq!(data, vec![1, 3]);
        std::fs::remove_file("test_u8.safetensors").unwrap();
    }

    #[test]
    fn test_endianness_multi_byte_types() {
        // Test that multi-byte types are correctly handled
        // SafeTensors stores data in little-endian format
        
        // Test F32: value 1.0 in little-endian is [0x00, 0x00, 0x80, 0x3f]
        // Header: 8 bytes length (0x36 = 54) + 54 bytes JSON, then data
        let f32_bytes = b"6\0\0\0\0\0\0\0{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}\x00\x00\x80\x3f";
        std::fs::write("test_f32_endian.safetensors", f32_bytes).unwrap();
        let weights = load("test_f32_endian.safetensors", &Device::Cpu).unwrap();
        let tensor = weights.get("x").unwrap();
        let data: Vec<f32> = tensor.to_vec1().unwrap();
        assert_eq!(data, vec![1.0f32]);
        std::fs::remove_file("test_f32_endian.safetensors").unwrap();

        // Test I32: value 256 in little-endian is [0x00, 0x01, 0x00, 0x00]
        let i32_bytes = b"6\0\0\0\0\0\0\0{\"x\":{\"dtype\":\"I32\",\"shape\":[1],\"data_offsets\":[0,4]}}\x00\x01\x00\x00";
        std::fs::write("test_i32_endian.safetensors", i32_bytes).unwrap();
        let weights = load("test_i32_endian.safetensors", &Device::Cpu).unwrap();
        let tensor = weights.get("x").unwrap();
        let data: Vec<i32> = tensor.to_vec1().unwrap();
        assert_eq!(data, vec![256i32]);
        std::fs::remove_file("test_i32_endian.safetensors").unwrap();

        // Test U32: value 65536 in little-endian is [0x00, 0x00, 0x01, 0x00]
        let u32_bytes = b"6\0\0\0\0\0\0\0{\"x\":{\"dtype\":\"U32\",\"shape\":[1],\"data_offsets\":[0,4]}}\x00\x00\x01\x00";
        std::fs::write("test_u32_endian.safetensors", u32_bytes).unwrap();
        let weights = load("test_u32_endian.safetensors", &Device::Cpu).unwrap();
        let tensor = weights.get("x").unwrap();
        let data: Vec<u32> = tensor.to_vec1().unwrap();
        assert_eq!(data, vec![65536u32]);
        std::fs::remove_file("test_u32_endian.safetensors").unwrap();
    }

    #[test]
    fn test_save_load_roundtrip_endianness() {
        // Test that save and load are symmetric for multi-byte types
        // Covers edge cases: zero, negative, max/min, special floats, non-power-of-2
        
        // F32: zero, negative, special values, non-power-of-2
        let original_f32 = Tensor::from_slice(
            &[0.0f32, -1.5f32, 3.14159f32, f32::INFINITY, f32::NEG_INFINITY, f32::NAN],
            (6,),
            &Device::Cpu
        ).unwrap();
        
        // I32: zero, negative, min, max, non-power-of-2
        let original_i32 = Tensor::from_slice(
            &[0i32, -256i32, i32::MIN, i32::MAX, 12345i32],
            (5,),
            &Device::Cpu
        ).unwrap();
        
        // U32: zero, max, non-power-of-2
        let original_u32 = Tensor::from_slice(
            &[0u32, u32::MAX, 98765u32],
            (3,),
            &Device::Cpu
        ).unwrap();

        let map: HashMap<_, _> = [
            ("f32", original_f32.clone()),
            ("i32", original_i32.clone()),
            ("u32", original_u32.clone()),
        ].into_iter().collect();
        
        save(&map, "test_roundtrip.safetensors").unwrap();
        let loaded = load("test_roundtrip.safetensors", &Device::Cpu).unwrap();

        let loaded_f32: Vec<f32> = loaded.get("f32").unwrap().to_vec1().unwrap();
        let loaded_i32: Vec<i32> = loaded.get("i32").unwrap().to_vec1().unwrap();
        let loaded_u32: Vec<u32> = loaded.get("u32").unwrap().to_vec1().unwrap();

        // Verify exact values (NaN requires special handling)
        assert_eq!(loaded_f32[0], 0.0f32);
        assert_eq!(loaded_f32[1], -1.5f32);
        assert_eq!(loaded_f32[2], 3.14159f32);
        assert_eq!(loaded_f32[3], f32::INFINITY);
        assert_eq!(loaded_f32[4], f32::NEG_INFINITY);
        assert!(loaded_f32[5].is_nan()); // NaN != NaN, so use is_nan()
        
        assert_eq!(loaded_i32, vec![0i32, -256i32, i32::MIN, i32::MAX, 12345i32]);
        assert_eq!(loaded_u32, vec![0u32, u32::MAX, 98765u32]);

        std::fs::remove_file("test_roundtrip.safetensors").unwrap();
    }

    #[test]
    fn test_u16_to_u32_cast_endianness() {
        // Test U16 to U32 casting with endianness handling
        // U16 value 256 in little-endian is [0x00, 0x01]
        let u16_bytes = b"6\0\0\0\0\0\0\0{\"x\":{\"dtype\":\"U16\",\"shape\":[1],\"data_offsets\":[0,2]}}\x00\x01";
        std::fs::write("test_u16_cast.safetensors", u16_bytes).unwrap();
        let weights = load("test_u16_cast.safetensors", &Device::Cpu).unwrap();
        let tensor = weights.get("x").unwrap();
        // U16 gets cast to U32
        assert_eq!(tensor.dtype(), DType::U32);
        let data: Vec<u32> = tensor.to_vec1().unwrap();
        assert_eq!(data, vec![256u32]);
        std::fs::remove_file("test_u16_cast.safetensors").unwrap();
    }
}
