//! Operation structs for ROCm backend
//!
//! These structs implement Map1/Map2/Map1Any traits to dispatch to HIP kernels.
//! Matches cuda_backend/mod.rs pattern (lines 54-150).

use crate::rocm_backend::{kernels, utils, RocmDevice};
use crate::{Result, WithDType};
use rocm_rs::hip::DeviceMemory;

// ============================================================================
// Operation Structs
// ============================================================================

// Map1 operations (single input)
pub(crate) struct Clone;
pub(crate) struct Affine(pub f64, pub f64);
pub(crate) struct Powf(pub f64);
pub(crate) struct Elu(pub f64);

// Map2 operations (binary - two inputs)
pub(crate) struct BinaryAdd;
pub(crate) struct BinarySub;
pub(crate) struct BinaryMul;
pub(crate) struct BinaryDiv;

// Map2 operations (comparison - two inputs, u8 output)
pub(crate) struct CmpEq;
pub(crate) struct CmpNe;
pub(crate) struct CmpLt;
pub(crate) struct CmpLe;
pub(crate) struct CmpGt;
pub(crate) struct CmpGe;

// Map1Any operations (reduce - returns different type)
pub(crate) struct ReduceSum {
    pub sum_dims: Vec<usize>,
}
pub(crate) struct ReduceMin {
    pub sum_dims: Vec<usize>,
}
pub(crate) struct ReduceMax {
    pub sum_dims: Vec<usize>,
}

// Generic unary operation dispatcher
pub(crate) struct UnaryOp<T: crate::op::UnaryOpT> {
    pub(crate) _phantom: std::marker::PhantomData<T>,
}

impl<T: crate::op::UnaryOpT> UnaryOp<T> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

// ============================================================================
// Map1 Implementations
// ============================================================================

impl utils::Map1 for Clone {
    fn f<T: WithDType>(
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
    fn f<T: WithDType>(
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
    fn f<T: WithDType>(
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
    fn f<T: WithDType>(
        &self,
        src: &DeviceMemory<T>,
        dev: &RocmDevice,
        layout: &crate::Layout,
    ) -> Result<DeviceMemory<T>> {
        let kernel_name = format!("uelu_{}", T::DTYPE.as_str());
        kernels::launch_unary(&kernel_name, dev.hip_device(), src, layout)
    }
}

impl<T: crate::op::UnaryOpT> utils::Map1 for UnaryOp<T> {
    fn f<U: WithDType>(
        &self,
        src: &DeviceMemory<U>,
        dev: &RocmDevice,
        layout: &crate::Layout,
    ) -> Result<DeviceMemory<U>> {
        let kernel_name = format!("u{}_{}", T::KERNEL_NAME, U::DTYPE.as_str());
        kernels::launch_unary(&kernel_name, dev.hip_device(), src, layout)
    }
}

// ============================================================================
// Map2 Implementations for Binary Operations
// ============================================================================

impl utils::Map2 for BinaryAdd {
    fn f<T: WithDType>(
        &self,
        src1: &DeviceMemory<T>,
        layout1: &crate::Layout,
        src2: &DeviceMemory<T>,
        layout2: &crate::Layout,
        dev: &RocmDevice,
    ) -> Result<DeviceMemory<T>> {
        let kernel_name = format!("badd_{}", T::DTYPE.as_str());
        kernels::launch_binary(&kernel_name, dev.hip_device(), src1, layout1, src2, layout2)
    }
}

impl utils::Map2 for BinarySub {
    fn f<T: WithDType>(
        &self,
        src1: &DeviceMemory<T>,
        layout1: &crate::Layout,
        src2: &DeviceMemory<T>,
        layout2: &crate::Layout,
        dev: &RocmDevice,
    ) -> Result<DeviceMemory<T>> {
        let kernel_name = format!("bsub_{}", T::DTYPE.as_str());
        kernels::launch_binary(&kernel_name, dev.hip_device(), src1, layout1, src2, layout2)
    }
}

impl utils::Map2 for BinaryMul {
    fn f<T: WithDType>(
        &self,
        src1: &DeviceMemory<T>,
        layout1: &crate::Layout,
        src2: &DeviceMemory<T>,
        layout2: &crate::Layout,
        dev: &RocmDevice,
    ) -> Result<DeviceMemory<T>> {
        let kernel_name = format!("bmul_{}", T::DTYPE.as_str());
        kernels::launch_binary(&kernel_name, dev.hip_device(), src1, layout1, src2, layout2)
    }
}

impl utils::Map2 for BinaryDiv {
    fn f<T: WithDType>(
        &self,
        src1: &DeviceMemory<T>,
        layout1: &crate::Layout,
        src2: &DeviceMemory<T>,
        layout2: &crate::Layout,
        dev: &RocmDevice,
    ) -> Result<DeviceMemory<T>> {
        let kernel_name = format!("bdiv_{}", T::DTYPE.as_str());
        kernels::launch_binary(&kernel_name, dev.hip_device(), src1, layout1, src2, layout2)
    }
}

// ============================================================================
// Map2 Implementations for Comparison Operations
// ============================================================================

impl utils::Map2 for CmpEq {
    fn f<T: WithDType>(
        &self,
        src1: &DeviceMemory<T>,
        layout1: &crate::Layout,
        src2: &DeviceMemory<T>,
        layout2: &crate::Layout,
        dev: &RocmDevice,
    ) -> Result<DeviceMemory<T>> {
        let kernel_name = format!("ceq_{}", T::DTYPE.as_str());
        kernels::launch_binary(&kernel_name, dev.hip_device(), src1, layout1, src2, layout2)
    }
}

impl utils::Map2 for CmpNe {
    fn f<T: WithDType>(
        &self,
        src1: &DeviceMemory<T>,
        layout1: &crate::Layout,
        src2: &DeviceMemory<T>,
        layout2: &crate::Layout,
        dev: &RocmDevice,
    ) -> Result<DeviceMemory<T>> {
        let kernel_name = format!("cne_{}", T::DTYPE.as_str());
        kernels::launch_binary(&kernel_name, dev.hip_device(), src1, layout1, src2, layout2)
    }
}

impl utils::Map2 for CmpLt {
    fn f<T: WithDType>(
        &self,
        src1: &DeviceMemory<T>,
        layout1: &crate::Layout,
        src2: &DeviceMemory<T>,
        layout2: &crate::Layout,
        dev: &RocmDevice,
    ) -> Result<DeviceMemory<T>> {
        let kernel_name = format!("clt_{}", T::DTYPE.as_str());
        kernels::launch_binary(&kernel_name, dev.hip_device(), src1, layout1, src2, layout2)
    }
}

impl utils::Map2 for CmpLe {
    fn f<T: WithDType>(
        &self,
        src1: &DeviceMemory<T>,
        layout1: &crate::Layout,
        src2: &DeviceMemory<T>,
        layout2: &crate::Layout,
        dev: &RocmDevice,
    ) -> Result<DeviceMemory<T>> {
        let kernel_name = format!("cle_{}", T::DTYPE.as_str());
        kernels::launch_binary(&kernel_name, dev.hip_device(), src1, layout1, src2, layout2)
    }
}

impl utils::Map2 for CmpGt {
    fn f<T: WithDType>(
        &self,
        src1: &DeviceMemory<T>,
        layout1: &crate::Layout,
        src2: &DeviceMemory<T>,
        layout2: &crate::Layout,
        dev: &RocmDevice,
    ) -> Result<DeviceMemory<T>> {
        let kernel_name = format!("cgt_{}", T::DTYPE.as_str());
        kernels::launch_binary(&kernel_name, dev.hip_device(), src1, layout1, src2, layout2)
    }
}

impl utils::Map2 for CmpGe {
    fn f<T: WithDType>(
        &self,
        src1: &DeviceMemory<T>,
        layout1: &crate::Layout,
        src2: &DeviceMemory<T>,
        layout2: &crate::Layout,
        dev: &RocmDevice,
    ) -> Result<DeviceMemory<T>> {
        let kernel_name = format!("cge_{}", T::DTYPE.as_str());
        kernels::launch_binary(&kernel_name, dev.hip_device(), src1, layout1, src2, layout2)
    }
}

// ============================================================================
// Map1Any Implementations for Reduce Operations
// ============================================================================

impl utils::Map1Any for ReduceSum {
    fn f<T: WithDType, W: Fn(Vec<usize>) -> Result<DeviceMemory<T>>>(
        &self,
        src: &DeviceMemory<T>,
        dev: &RocmDevice,
        layout: &crate::Layout,
        wrap: W,
    ) -> Result<DeviceMemory<T>> {
        let kernel_name = format!("fast_sum_{}", T::DTYPE.as_str());
        kernels::launch_reduce(
            &kernel_name,
            dev.hip_device(),
            src,
            layout,
            &self.sum_dims,
            wrap,
        )
    }
}

impl utils::Map1Any for ReduceMin {
    fn f<T: WithDType, W: Fn(Vec<usize>) -> Result<DeviceMemory<T>>>(
        &self,
        src: &DeviceMemory<T>,
        dev: &RocmDevice,
        layout: &crate::Layout,
        wrap: W,
    ) -> Result<DeviceMemory<T>> {
        let kernel_name = format!("fast_min_{}", T::DTYPE.as_str());
        kernels::launch_reduce(
            &kernel_name,
            dev.hip_device(),
            src,
            layout,
            &self.sum_dims,
            wrap,
        )
    }
}

impl utils::Map1Any for ReduceMax {
    fn f<T: WithDType, W: Fn(Vec<usize>) -> Result<DeviceMemory<T>>>(
        &self,
        src: &DeviceMemory<T>,
        dev: &RocmDevice,
        layout: &crate::Layout,
        wrap: W,
    ) -> Result<DeviceMemory<T>> {
        let kernel_name = format!("fast_max_{}", T::DTYPE.as_str());
        kernels::launch_reduce(
            &kernel_name,
            dev.hip_device(),
            src,
            layout,
            &self.sum_dims,
            wrap,
        )
    }
}
