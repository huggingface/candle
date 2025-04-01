use crate::backend::BackendDevice;
use crate::{CpuStorage, CpuStorageRef, DType, Layout, Result, Shape};
pub use candle_kernels as kernels;
pub use cudarc;
use cudarc::driver::{CudaFunction, CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg};
use half::{bf16, f16};
use std::sync::{Arc, Mutex};

use super::{CudaError, CudaStorage, CudaStorageSlice, WrapErr};

/// Unique identifier for cuda devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

struct CudaRng(cudarc::curand::CudaRng);
unsafe impl Send for CudaRng {}

#[derive(Clone)]
pub struct CudaDevice {
    id: DeviceId,
    context: Arc<cudarc::driver::CudaContext>,
    stream: Arc<cudarc::driver::CudaStream>,
    pub(crate) blas: Arc<cudarc::cublas::CudaBlas>,
    curand: Arc<Mutex<CudaRng>>,
}

impl std::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaDevice({:?})", self.id)
    }
}

impl std::ops::Deref for CudaDevice {
    type Target = Arc<cudarc::driver::CudaStream>;

    fn deref(&self) -> &Self::Target {
        &self.stream
    }
}

impl CudaDevice {
    pub fn cuda_stream(&self) -> Arc<cudarc::driver::CudaStream> {
        self.stream.clone()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn compile(
        &self,
        func_name: &'static str,
        kernel: ug::lang::ssa::Kernel,
    ) -> Result<CudaFunction> {
        let mut buf = vec![];
        ug_cuda::code_gen::gen(&mut buf, func_name, &kernel)?;
        let cuda_code = String::from_utf8(buf)?;
        let opts = cudarc::nvrtc::CompileOptions {
            use_fast_math: Some(true),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::safe::compile_ptx_with_opts(cuda_code, opts).w()?;
        let module = self.context.load_module(ptx).w()?;
        let func = module.load_function(func_name).w()?;
        Ok(func)
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    fn const_impl(&self, v: f64, shape: &Shape, dtype: DType) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(elem_count as u32);
        let slice = match dtype {
            DType::U8 => {
                // SAFETY: Set later by running the fill kernel.
                let data = unsafe { self.alloc::<u8>(elem_count) }.w()?;
                let func = self.get_or_load_func("fill_u8", kernels::FILL)?;
                let mut builder = self.stream.launch_builder(&func);
                let v = v as u8;
                builder.arg(&data);
                builder.arg(&v);
                builder.arg(&elem_count);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::U8(data)
            }
            DType::U32 => {
                // SAFETY: Set later by running the fill kernel.
                let data = unsafe { self.alloc::<u32>(elem_count) }.w()?;
                let func = self.get_or_load_func("fill_u32", kernels::FILL)?;
                let mut builder = self.stream.launch_builder(&func);
                let v = v as u32;
                builder.arg(&data);
                builder.arg(&v);
                builder.arg(&elem_count);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::U32(data)
            }
            DType::I64 => {
                // SAFETY: Set later by running the fill kernel.
                let data = unsafe { self.alloc::<i64>(elem_count) }.w()?;
                let func = self.get_or_load_func("fill_i64", kernels::FILL)?;
                let mut builder = self.stream.launch_builder(&func);
                let v = v as i64;
                builder.arg(&data);
                builder.arg(&v);
                builder.arg(&elem_count);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::I64(data)
            }
            DType::BF16 => {
                // SAFETY: Set later by running the fill kernel.
                let data = unsafe { self.alloc::<bf16>(elem_count) }.w()?;
                let func = self.get_or_load_func("fill_bf16", kernels::FILL)?;
                let mut builder = self.stream.launch_builder(&func);
                let v = bf16::from_f64(v);
                builder.arg(&data);
                builder.arg(&v);
                builder.arg(&elem_count);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::BF16(data)
            }
            DType::F16 => {
                // SAFETY: Set later by running the fill kernel.
                let data = unsafe { self.alloc::<f16>(elem_count) }.w()?;
                let func = self.get_or_load_func("fill_f16", kernels::FILL)?;
                let mut builder = self.stream.launch_builder(&func);
                let v = f16::from_f64(v);
                builder.arg(&data);
                builder.arg(&v);
                builder.arg(&elem_count);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F16(data)
            }
            DType::F32 => {
                // SAFETY: Set later by running the fill kernel.
                let data = unsafe { self.alloc::<f32>(elem_count) }.w()?;
                let func = self.get_or_load_func("fill_f32", kernels::FILL)?;
                let mut builder = self.stream.launch_builder(&func);
                let v = v as f32;
                builder.arg(&data);
                builder.arg(&v);
                builder.arg(&elem_count);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                // SAFETY: Set later by running the fill kernel.
                let data = unsafe { self.alloc::<f64>(elem_count) }.w()?;
                let func = self.get_or_load_func("fill_f64", kernels::FILL)?;
                let mut builder = self.stream.launch_builder(&func);
                builder.arg(&data);
                builder.arg(&v);
                builder.arg(&elem_count);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F64(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    pub fn get_or_load_func(&self, module_name: &str, ptx: &'static str) -> Result<CudaFunction> {
        if !self.has_func(module_name, module_name) {
            // Leaking the string here is a bit sad but we need a &'static str and this is only
            // done once per kernel name.
            let static_module_name = Box::leak(module_name.to_string().into_boxed_str());
            self.context
                .load_module(ptx.into(), module_name, &[static_module_name])
                .map_err(|cuda| CudaError::Load {
                    cuda,
                    module_name: module_name.to_string(),
                })
                .w()?;
        }
        self.get_func(module_name, module_name)
            // Clippy recommends this `ok_or` rather than `ok_or_else` so hopefully the compiler is
            // able to only build the error value if needed.
            .ok_or(CudaError::MissingKernel {
                module_name: module_name.to_string(),
            })
            .w()
    }
}

impl CudaDevice {
    pub fn new_with_stream(ordinal: usize) -> Result<Self> {
        let context = cudarc::driver::CudaContext::new(ordinal).w()?;
        let stream = context.new_stream().w()?;
        let blas = cudarc::cublas::CudaBlas::new(stream.clone()).w()?;
        let curand = cudarc::curand::CudaRng::new(299792458, stream.clone()).w()?;
        Ok(Self {
            id: DeviceId::new(),
            context,
            stream,
            blas: Arc::new(blas),
            curand: Arc::new(Mutex::new(CudaRng(curand))),
        })
    }
}

impl BackendDevice for CudaDevice {
    type Storage = CudaStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let context = cudarc::driver::CudaContext::new(ordinal).w()?;
        let stream = context.default_stream();
        let blas = cudarc::cublas::CudaBlas::new(stream.clone()).w()?;
        let curand = cudarc::curand::CudaRng::new(299792458, stream.clone()).w()?;
        Ok(Self {
            id: DeviceId::new(),
            context,
            stream,
            blas: Arc::new(blas),
            curand: Arc::new(Mutex::new(CudaRng(curand))),
        })
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        // We do not call set_seed but instead create a new curand object. This ensures that the
        // state will be identical and the same random numbers will be generated.
        let mut curand = self.curand.lock().unwrap();
        curand.0 = cudarc::curand::CudaRng::new(seed, self.stream.clone()).w()?;
        Ok(())
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Cuda {
            gpu_id: self.context.ordinal(),
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.id == rhs.id
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => {
                let data = self.alloc_zeros::<u8>(elem_count).w()?;
                CudaStorageSlice::U8(data)
            }
            DType::U32 => {
                let data = self.alloc_zeros::<u32>(elem_count).w()?;
                CudaStorageSlice::U32(data)
            }
            DType::I64 => {
                let data = self.alloc_zeros::<i64>(elem_count).w()?;
                CudaStorageSlice::I64(data)
            }
            DType::BF16 => {
                let data = self.alloc_zeros::<bf16>(elem_count).w()?;
                CudaStorageSlice::BF16(data)
            }
            DType::F16 => {
                let data = self.alloc_zeros::<f16>(elem_count).w()?;
                CudaStorageSlice::F16(data)
            }
            DType::F32 => {
                let data = self.alloc_zeros::<f32>(elem_count).w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let data = self.alloc_zeros::<f64>(elem_count).w()?;
                CudaStorageSlice::F64(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, up: f64) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let curand = self.curand.lock().unwrap();
        let slice = match dtype {
            // TODO: Add support for F16 and BF16 though this is likely to require some upstream
            // cudarc changes.
            DType::U8 | DType::U32 | DType::I64 | DType::F16 | DType::BF16 => {
                Err(CudaError::UnsupportedDtype {
                    dtype,
                    op: "rand_uniform",
                })
                .w()?
            }
            DType::F32 => {
                let mut data = unsafe { self.alloc::<f32>(elem_count) }.w()?;
                curand.0.fill_with_uniform(&mut data).w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let mut data = unsafe { self.alloc::<f64>(elem_count) }.w()?;
                curand.0.fill_with_uniform(&mut data).w()?;
                CudaStorageSlice::F64(data)
            }
        };
        let slice = if lo == 0. && up == 1.0 {
            slice
        } else {
            use super::utils::Map1;
            let layout = Layout::contiguous(shape);
            super::Affine(up - lo, lo).map(&slice, self, &layout)?
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<CudaStorage> {
        // TODO: Add support for F16 and BF16 though this is likely to require some upstream
        // cudarc changes.
        let elem_count = shape.elem_count();
        let curand = self.curand.lock().unwrap();
        // curand can only generate an odd number of values.
        // https://github.com/huggingface/candle/issues/734
        let elem_count_round = if elem_count % 2 == 1 {
            elem_count + 1
        } else {
            elem_count
        };
        let slice = match dtype {
            DType::U8 | DType::U32 | DType::I64 | DType::F16 | DType::BF16 => {
                Err(CudaError::UnsupportedDtype {
                    dtype,
                    op: "rand_normal",
                })
                .w()?
            }
            DType::F32 => {
                let mut data = unsafe { self.alloc::<f32>(elem_count_round) }.w()?;
                curand
                    .0
                    .fill_with_normal(&mut data, mean as f32, std as f32)
                    .w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let mut data = unsafe { self.alloc::<f64>(elem_count_round) }.w()?;
                curand.0.fill_with_normal(&mut data, mean, std).w()?;
                CudaStorageSlice::F64(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn ones_impl(&self, shape: &Shape, dtype: DType) -> Result<CudaStorage> {
        self.const_impl(1., shape, dtype)
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => {
                let data = self.alloc::<u8>(elem_count).w()?;
                CudaStorageSlice::U8(data)
            }
            DType::U32 => {
                let data = self.alloc::<u32>(elem_count).w()?;
                CudaStorageSlice::U32(data)
            }
            DType::I64 => {
                let data = self.alloc::<i64>(elem_count).w()?;
                CudaStorageSlice::I64(data)
            }
            DType::BF16 => {
                let data = self.alloc::<bf16>(elem_count).w()?;
                CudaStorageSlice::BF16(data)
            }
            DType::F16 => {
                let data = self.alloc::<f16>(elem_count).w()?;
                CudaStorageSlice::F16(data)
            }
            DType::F32 => {
                let data = self.alloc::<f32>(elem_count).w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let data = self.alloc::<f64>(elem_count).w()?;
                CudaStorageSlice::F64(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        let slice = match T::cpu_storage_ref(s) {
            CpuStorageRef::U8(storage) => {
                let data = self.memcpy_stod(storage).w()?;
                CudaStorageSlice::U8(data)
            }
            CpuStorageRef::U32(storage) => {
                let data = self.memcpy_stod(storage).w()?;
                CudaStorageSlice::U32(data)
            }
            CpuStorageRef::I64(storage) => {
                let data = self.memcpy_stod(storage).w()?;
                CudaStorageSlice::I64(data)
            }
            CpuStorageRef::BF16(storage) => {
                let data = self.memcpy_stod(storage).w()?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorageRef::F16(storage) => {
                let data = self.memcpy_stod(storage).w()?;
                CudaStorageSlice::F16(data)
            }
            CpuStorageRef::F32(storage) => {
                let data = self.memcpy_stod(storage).w()?;
                CudaStorageSlice::F32(data)
            }
            CpuStorageRef::F64(storage) => {
                let data = self.memcpy_stod(storage).w()?;
                CudaStorageSlice::F64(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<CudaStorage> {
        let slice = match storage {
            CpuStorage::U8(storage) => {
                let data = self.memcpy_stod(storage).w()?;
                CudaStorageSlice::U8(data)
            }
            CpuStorage::U32(storage) => {
                let data = self.memcpy_stod(storage).w()?;
                CudaStorageSlice::U32(data)
            }
            CpuStorage::I64(storage) => {
                let data = self.memcpy_stod(storage).w()?;
                CudaStorageSlice::I64(data)
            }
            CpuStorage::BF16(storage) => {
                let data = self.memcpy_stod(storage).w()?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorage::F16(storage) => {
                let data = self.memcpy_stod(storage).w()?;
                CudaStorageSlice::F16(data)
            }
            CpuStorage::F32(storage) => {
                let data = self.memcpy_stod(storage).w()?;
                CudaStorageSlice::F32(data)
            }
            CpuStorage::F64(storage) => {
                let data = self.memcpy_stod(storage).w()?;
                CudaStorageSlice::F64(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<CudaStorage> {
        let slice = match storage {
            CpuStorage::U8(storage) => {
                let data = self.memcpy_stod(&storage).w()?;
                CudaStorageSlice::U8(data)
            }
            CpuStorage::U32(storage) => {
                let data = self.memcpy_stod(&storage).w()?;
                CudaStorageSlice::U32(data)
            }
            CpuStorage::I64(storage) => {
                let data = self.memcpy_stod(&storage).w()?;
                CudaStorageSlice::I64(data)
            }
            CpuStorage::BF16(storage) => {
                let data = self.memcpy_stod(&storage).w()?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorage::F16(storage) => {
                let data = self.memcpy_stod(&storage).w()?;
                CudaStorageSlice::F16(data)
            }
            CpuStorage::F32(storage) => {
                let data = self.memcpy_stod(&storage).w()?;
                CudaStorageSlice::F32(data)
            }
            CpuStorage::F64(storage) => {
                let data = self.memcpy_stod(&storage).w()?;
                CudaStorageSlice::F64(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().map_err(crate::Error::wrap)?;
        Ok(())
    }
}
