// candle-core/src/rocm_backend/mod.rs
// Created by: TEAM-488 (Phase 1)
// Updated by: TEAM-489 (Phase 2 Step 3) - Added kernel imports
// Updated by: TEAM-492 (Phase 2 Step 3) - Direct kernel loading
// Updated by: TEAM-493 (Phase 3) - Added utils module for Map traits
// ROCm backend using rocm-rs - thin wrappers, don't reimplement

pub mod device;
pub mod error;
pub mod kernels;
pub mod storage_slice;
pub mod utils;

pub use device::{device_count, is_available, runtime_version, RocmDevice};
pub use error::RocmError;
pub use storage_slice::RocmStorageSlice;

// Re-export rocm-rs types we use directly
pub use rocm_rs::hip::{Dim3, DeviceMemory, Function, Module, Stream};

// Type alias for convenience (matches CUDA backend pattern)
pub type S = RocmStorageSlice;
pub type Result<T> = std::result::Result<T, RocmError>;

// ============================================================================
// RocmStorage - matches CUDA backend pattern
// ============================================================================

/// ROCm storage - wraps storage slice + device
/// MATCHES: cuda_backend/mod.rs CudaStorage (line 1132)
#[derive(Debug)]
pub struct RocmStorage {
    pub slice: RocmStorageSlice,
    pub device: RocmDevice,
}

impl RocmStorage {
    pub fn new(slice: RocmStorageSlice, device: RocmDevice) -> Self {
        Self { slice, device }
    }

    pub fn device(&self) -> &RocmDevice {
        &self.device
    }

    pub fn dtype(&self) -> crate::DType {
        match &self.slice {
            S::U8(_) => crate::DType::U8,
            S::U32(_) => crate::DType::U32,
            S::I64(_) => crate::DType::I64,
            S::BF16(_) => crate::DType::BF16,
            S::F16(_) => crate::DType::F16,
            S::F32(_) => crate::DType::F32,
            S::F64(_) => crate::DType::F64,
            S::F8E4M3(_) => crate::DType::F8E4M3,
        }
    }
}

// ============================================================================
// Helper Structs for Map1 Pattern (matches CUDA)
// ============================================================================

struct Clone;
struct Affine(f64, f64);
struct Powf(f64);
struct Elu(f64);

// ============================================================================
// Map1 Implementations
// ============================================================================

impl utils::Map1 for Clone {
    fn f<T: crate::WithDType>(
        &self,
        src: &DeviceMemory<T>,
        dev: &RocmDevice,
        layout: &crate::Layout,
    ) -> Result<DeviceMemory<T>> {
        let el = layout.shape().elem_count();
        let mut dst = dev.hip_device().alloc::<T>(el)?;
        dst.copy_from_device(src)?;
        Ok(dst)
    }
}

impl utils::Map1 for Affine {
    fn f<T: crate::WithDType>(
        &self,
        src: &DeviceMemory<T>,
        dev: &RocmDevice,
        layout: &crate::Layout,
    ) -> Result<DeviceMemory<T>> {
        let kernel_name = format!("affine_{}", T::DTYPE.as_str());
        kernels::launch_affine(
            &kernel_name,
            dev.hip_device(),
            src,
            layout,
            T::from_f64(self.0),
            T::from_f64(self.1),
        )
    }
}

impl utils::Map1 for Powf {
    fn f<T: crate::WithDType>(
        &self,
        src: &DeviceMemory<T>,
        dev: &RocmDevice,
        layout: &crate::Layout,
    ) -> Result<DeviceMemory<T>> {
        let kernel_name = format!("upowf_{}", T::DTYPE.as_str());
        kernels::launch_unary(&kernel_name, dev.hip_device(), src, layout)
    }
}

impl utils::Map1 for Elu {
    fn f<T: crate::WithDType>(
        &self,
        src: &DeviceMemory<T>,
        dev: &RocmDevice,
        layout: &crate::Layout,
    ) -> Result<DeviceMemory<T>> {
        let kernel_name = format!("uelu_{}", T::DTYPE.as_str());
        kernels::launch_unary(&kernel_name, dev.hip_device(), src, layout)
    }
}

// ============================================================================
// BackendStorage Implementation
// ============================================================================

impl crate::backend::BackendStorage for RocmStorage {
    type Device = RocmDevice;

    fn try_clone(&self, layout: &crate::Layout) -> Result<Self> {
        let slice = Clone.map(&self.slice, self.device(), layout)?;
        let device = self.device.clone();
        Ok(Self { slice, device })
    }

    fn dtype(&self) -> crate::DType {
        self.dtype()
    }

    fn device(&self) -> &RocmDevice {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<crate::CpuStorage> {
        use crate::DType;
        let cpu_storage = match &self.slice {
            S::U8(slice) => {
                let data = slice.copy_to_host()?;
                crate::CpuStorage::U8(data)
            }
            S::U32(slice) => {
                let data = slice.copy_to_host()?;
                crate::CpuStorage::U32(data)
            }
            S::I64(slice) => {
                let data = slice.copy_to_host()?;
                crate::CpuStorage::I64(data)
            }
            S::BF16(slice) => {
                let data = slice.copy_to_host()?;
                crate::CpuStorage::BF16(data)
            }
            S::F16(slice) => {
                let data = slice.copy_to_host()?;
                crate::CpuStorage::F16(data)
            }
            S::F32(slice) => {
                let data = slice.copy_to_host()?;
                crate::CpuStorage::F32(data)
            }
            S::F64(slice) => {
                let data = slice.copy_to_host()?;
                crate::CpuStorage::F64(data)
            }
            S::F8E4M3(slice) => {
                let data = slice.copy_to_host()?;
                crate::CpuStorage::F8E4M3(data)
            }
        };
        Ok(cpu_storage)
    }

    fn to_dtype(&self, layout: &crate::Layout, dtype: crate::DType) -> Result<Self> {
        use crate::DType;
        let shape = layout.shape();
        let el = shape.elem_count();
        let dev = self.device.hip_device();
        
        let kernel_name = format!("cast_{}_{}", self.dtype().as_str(), dtype.as_str());
        
        let slice = match (&self.slice, dtype) {
            (S::U8(src), DType::U8) => S::U8(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::U8(src), DType::U32) => S::U32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::U8(src), DType::I64) => S::I64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::U8(src), DType::BF16) => S::BF16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::U8(src), DType::F16) => S::F16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::U8(src), DType::F32) => S::F32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::U8(src), DType::F64) => S::F64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::U8(src), DType::F8E4M3) => S::F8E4M3(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            
            (S::U32(src), DType::U8) => S::U8(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::U32(src), DType::U32) => S::U32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::U32(src), DType::I64) => S::I64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::U32(src), DType::BF16) => S::BF16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::U32(src), DType::F16) => S::F16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::U32(src), DType::F32) => S::F32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::U32(src), DType::F64) => S::F64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::U32(src), DType::F8E4M3) => S::F8E4M3(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            
            (S::I64(src), DType::U8) => S::U8(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::I64(src), DType::U32) => S::U32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::I64(src), DType::I64) => S::I64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::I64(src), DType::BF16) => S::BF16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::I64(src), DType::F16) => S::F16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::I64(src), DType::F32) => S::F32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::I64(src), DType::F64) => S::F64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::I64(src), DType::F8E4M3) => S::F8E4M3(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            
            (S::BF16(src), DType::U8) => S::U8(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::BF16(src), DType::U32) => S::U32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::BF16(src), DType::I64) => S::I64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::BF16(src), DType::BF16) => S::BF16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::BF16(src), DType::F16) => S::F16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::BF16(src), DType::F32) => S::F32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::BF16(src), DType::F64) => S::F64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::BF16(src), DType::F8E4M3) => S::F8E4M3(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            
            (S::F16(src), DType::U8) => S::U8(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F16(src), DType::U32) => S::U32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F16(src), DType::I64) => S::I64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F16(src), DType::BF16) => S::BF16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F16(src), DType::F16) => S::F16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F16(src), DType::F32) => S::F32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F16(src), DType::F64) => S::F64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F16(src), DType::F8E4M3) => S::F8E4M3(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            
            (S::F32(src), DType::U8) => S::U8(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F32(src), DType::U32) => S::U32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F32(src), DType::I64) => S::I64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F32(src), DType::BF16) => S::BF16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F32(src), DType::F16) => S::F16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F32(src), DType::F32) => S::F32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F32(src), DType::F64) => S::F64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F32(src), DType::F8E4M3) => S::F8E4M3(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            
            (S::F64(src), DType::U8) => S::U8(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F64(src), DType::U32) => S::U32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F64(src), DType::I64) => S::I64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F64(src), DType::BF16) => S::BF16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F64(src), DType::F16) => S::F16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F64(src), DType::F32) => S::F32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F64(src), DType::F64) => S::F64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F64(src), DType::F8E4M3) => S::F8E4M3(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            
            (S::F8E4M3(src), DType::U8) => S::U8(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F8E4M3(src), DType::U32) => S::U32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F8E4M3(src), DType::I64) => S::I64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F8E4M3(src), DType::BF16) => S::BF16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F8E4M3(src), DType::F16) => S::F16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F8E4M3(src), DType::F32) => S::F32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F8E4M3(src), DType::F64) => S::F64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F8E4M3(src), DType::F8E4M3) => S::F8E4M3(kernels::launch_cast(&kernel_name, dev, src, layout)?),
        };
        
        Ok(Self {
            slice,
            device: self.device.clone(),
        })
    }

    fn affine(&self, layout: &crate::Layout, mul: f64, add: f64) -> Result<Self> {
        let device = self.device().clone();
        let slice = Affine(mul, add).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn powf(&self, layout: &crate::Layout, e: f64) -> Result<Self> {
        let device = self.device().clone();
        let slice = Powf(e).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn elu(&self, layout: &crate::Layout, alpha: f64) -> Result<Self> {
        let device = self.device().clone();
        let slice = Elu(alpha).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    // ==========================================================================
    // TODO: TEAM-494 - Wire Existing rocm-rs Operations
    // ==========================================================================
    // These operations EXIST in rocm-rs but aren't wired to Candle yet!
    //
    // 1. reduce_op - rocm-rs has reduce_sum/min/max in rocarray/kernels.rs
    //    - Create ReduceSum/ReduceMin/ReduceMax structs
    //    - Implement Map1Any trait for them
    //    - Call rocm_rs::rocarray::kernels::reduce_*()
    //
    // 2. binary_impl - rocm-rs has elementwise_add/sub/mul/div
    //    - Create Add/Sub/Mul/Div structs
    //    - Implement Map2 trait for them
    //    - Call rocm_rs::rocarray::kernels::elementwise_*()
    //
    // 3. unary_impl - rocm-rs has many unary kernels (exp, log, sin, etc.)
    //    - Create generic dispatch based on UnaryOpT trait
    //    - Call appropriate kernels::launch_unary() with kernel name
    //
    // 4. cmp - MISSING from rocm-rs (see TODO in rocm-rs/src/rocarray/kernels.hip)
    //    - TEAM-495 needs to add comparison kernels first
    //    - Then create Cmp struct and wire it here
    // ==========================================================================

    fn reduce_op(&self, _op: crate::op::ReduceOp, _layout: &crate::Layout, _sum_dims: &[usize]) -> Result<Self> {
        // TODO: TEAM-494 - Wire rocm-rs reduce_sum/min/max
        unimplemented!("reduce_op - rocm-rs HAS reduce_sum/min/max, need to wire them")
    }

    fn cmp(&self, _op: crate::op::CmpOp, _rhs: &Self, _lhs_l: &crate::Layout, _rhs_l: &crate::Layout) -> Result<Self> {
        // TODO: TEAM-495 - Add comparison kernels to rocm-rs first, then wire here
        unimplemented!("cmp - need to add comparison kernels to rocm-rs/src/rocarray/kernels.hip")
    }

    fn unary_impl<B: crate::op::UnaryOpT>(&self, _layout: &crate::Layout) -> Result<Self> {
        // TODO: TEAM-494 - Create generic dispatch to existing unary kernels
        unimplemented!("unary_impl - rocm-rs HAS unary kernels (exp, log, sin, etc.), need generic dispatch")
    }

    fn binary_impl<B: crate::op::BinaryOpT>(&self, _rhs: &Self, _lhs_l: &crate::Layout, _rhs_l: &crate::Layout) -> Result<Self> {
        // TODO: TEAM-494 - Wire rocm-rs elementwise_add/sub/mul/div
        unimplemented!("binary_impl - rocm-rs HAS elementwise_add/sub/mul/div, need to wire them")
    }

    fn where_cond(&self, layout: &crate::Layout, t: &Self, layout_t: &crate::Layout, f: &Self, layout_f: &crate::Layout) -> Result<Self> {
        use crate::DType;
        let dev = self.device.hip_device();
        
        // Determine kernel name from condition type and value type
        let cond_type = match &self.slice {
            S::U8(_) => "u8",
            S::U32(_) => "u32",
            S::I64(_) => "i64",
            _ => return Err(RocmError::InternalError("where_cond: condition must be u8/u32/i64").into()),
        };
        
        let val_type = t.dtype().as_str();
        let kernel_name = format!("where_{}_{}", cond_type, val_type);
        
        let slice = match (&self.slice, &t.slice, &f.slice) {
            (S::U8(cond), S::F32(t_val), S::F32(f_val)) => {
                S::F32(kernels::launch_ternary(&kernel_name, dev, cond, layout, t_val, layout_t, f_val, layout_f)?)
            }
            (S::U8(cond), S::F64(t_val), S::F64(f_val)) => {
                S::F64(kernels::launch_ternary(&kernel_name, dev, cond, layout, t_val, layout_t, f_val, layout_f)?)
            }
            (S::U8(cond), S::F16(t_val), S::F16(f_val)) => {
                S::F16(kernels::launch_ternary(&kernel_name, dev, cond, layout, t_val, layout_t, f_val, layout_f)?)
            }
            (S::U8(cond), S::BF16(t_val), S::BF16(f_val)) => {
                S::BF16(kernels::launch_ternary(&kernel_name, dev, cond, layout, t_val, layout_t, f_val, layout_f)?)
            }
            (S::U8(cond), S::U8(t_val), S::U8(f_val)) => {
                S::U8(kernels::launch_ternary(&kernel_name, dev, cond, layout, t_val, layout_t, f_val, layout_f)?)
            }
            (S::U8(cond), S::U32(t_val), S::U32(f_val)) => {
                S::U32(kernels::launch_ternary(&kernel_name, dev, cond, layout, t_val, layout_t, f_val, layout_f)?)
            }
            (S::U8(cond), S::I64(t_val), S::I64(f_val)) => {
                S::I64(kernels::launch_ternary(&kernel_name, dev, cond, layout, t_val, layout_t, f_val, layout_f)?)
            }
            (S::U8(cond), S::F8E4M3(t_val), S::F8E4M3(f_val)) => {
                S::F8E4M3(kernels::launch_ternary(&kernel_name, dev, cond, layout, t_val, layout_t, f_val, layout_f)?)
            }
            _ => return Err(RocmError::InternalError("where_cond: unsupported type combination").into()),
        };
        
        Ok(Self {
            slice,
            device: self.device.clone(),
        })
    }

    // Advanced operations - unimplemented for now
    fn conv1d(&self, _l: &crate::Layout, _kernel: &Self, _kernel_l: &crate::Layout, _params: &crate::conv::ParamsConv1D) -> Result<Self> {
        unimplemented!("conv1d - need MIOpen integration")
    }

    fn conv_transpose1d(&self, _l: &crate::Layout, _kernel: &Self, _kernel_l: &crate::Layout, _params: &crate::conv::ParamsConvTranspose1D) -> Result<Self> {
        unimplemented!("conv_transpose1d - need MIOpen integration")
    }

    fn conv2d(&self, _l: &crate::Layout, _kernel: &Self, _kernel_l: &crate::Layout, _params: &crate::conv::ParamsConv2D) -> Result<Self> {
        unimplemented!("conv2d - need MIOpen integration")
    }

    fn conv_transpose2d(&self, _l: &crate::Layout, _kernel: &Self, _kernel_l: &crate::Layout, _params: &crate::conv::ParamsConvTranspose2D) -> Result<Self> {
        unimplemented!("conv_transpose2d - need MIOpen integration")
    }

    fn avg_pool2d(&self, _: &crate::Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        unimplemented!("avg_pool2d - need MIOpen integration")
    }

    fn max_pool2d(&self, _: &crate::Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        unimplemented!("max_pool2d - need MIOpen integration")
    }

    fn upsample_nearest1d(&self, _: &crate::Layout, _: usize) -> Result<Self> {
        unimplemented!("upsample_nearest1d")
    }

    fn upsample_nearest2d(&self, _: &crate::Layout, _: usize, _: usize) -> Result<Self> {
        unimplemented!("upsample_nearest2d")
    }

    fn gather(&self, _: &crate::Layout, _: &Self, _: &crate::Layout, _: usize) -> Result<Self> {
        unimplemented!("gather - need custom kernels")
    }

    fn scatter_set(&mut self, _: &crate::Layout, _: &Self, _: &crate::Layout, _: &Self, _: &crate::Layout, _: usize) -> Result<()> {
        unimplemented!("scatter_set - need custom kernels")
    }

    fn scatter_add_set(&mut self, _: &crate::Layout, _: &Self, _: &crate::Layout, _: &Self, _: &crate::Layout, _: usize) -> Result<()> {
        unimplemented!("scatter_add_set - need custom kernels")
    }

    fn index_select(&self, _: &Self, _: &crate::Layout, _: &crate::Layout, _: usize) -> Result<Self> {
        unimplemented!("index_select - need custom kernels")
    }

    fn index_add(&self, _: &crate::Layout, _: &Self, _: &crate::Layout, _: &Self, _: &crate::Layout, _: usize) -> Result<Self> {
        unimplemented!("index_add - need custom kernels")
    }

    fn matmul(&self, _: &Self, _: (usize, usize, usize, usize), _: &crate::Layout, _: &crate::Layout) -> Result<Self> {
        unimplemented!("matmul - need rocBLAS integration")
    }

    fn copy2d(&self, _: &mut Self, _: usize, _: usize, _: usize, _: usize, _: usize, _: usize) -> Result<()> {
        unimplemented!("copy2d")
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &crate::Layout) -> Result<()> {
        unimplemented!("copy_strided_src")
    }
}
