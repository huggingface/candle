#![cfg(feature = "provisional")]
//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_NV_cuda_kernel_launch.html>

use crate::read_into_uninitialized_vector;
use crate::vk;
use crate::RawPtr;
use crate::VkResult;
use alloc::vec::Vec;
use core::mem;

impl crate::nv::cuda_kernel_launch::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateCudaModuleNV.html>
    #[inline]
    pub unsafe fn create_cuda_module(
        &self,
        create_info: &vk::CudaModuleCreateInfoNV<'_>,
        allocator: Option<&vk::AllocationCallbacks>,
    ) -> VkResult<vk::CudaModuleNV> {
        let mut module = mem::MaybeUninit::uninit();
        (self.fp.create_cuda_module_nv)(
            self.handle,
            create_info,
            allocator.to_raw_ptr(),
            module.as_mut_ptr(),
        )
        .assume_init_on_success(module)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetCudaModuleCacheNV.html>
    #[inline]
    pub unsafe fn get_cuda_module_cache(&self, module: vk::CudaModuleNV) -> VkResult<Vec<u8>> {
        read_into_uninitialized_vector(|cache_size, cache_data: *mut u8| {
            (self.fp.get_cuda_module_cache_nv)(self.handle, module, cache_size, cache_data.cast())
        })
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateCudaFunctionNV.html>
    #[inline]
    pub unsafe fn create_cuda_function(
        &self,
        create_info: &vk::CudaFunctionCreateInfoNV<'_>,
        allocator: Option<&vk::AllocationCallbacks>,
    ) -> VkResult<vk::CudaFunctionNV> {
        let mut function = mem::MaybeUninit::uninit();
        (self.fp.create_cuda_function_nv)(
            self.handle,
            create_info,
            allocator.to_raw_ptr(),
            function.as_mut_ptr(),
        )
        .assume_init_on_success(function)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkDestroyCudaModuleNV.html>
    #[inline]
    pub unsafe fn destroy_cuda_module(
        &self,
        module: vk::CudaModuleNV,
        allocator: Option<&vk::AllocationCallbacks>,
    ) {
        (self.fp.destroy_cuda_module_nv)(self.handle, module, allocator.to_raw_ptr())
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkDestroyCudaFunctionNV.html>
    #[inline]
    pub unsafe fn destroy_cuda_function(
        &self,
        function: vk::CudaFunctionNV,
        allocator: Option<&vk::AllocationCallbacks>,
    ) {
        (self.fp.destroy_cuda_function_nv)(self.handle, function, allocator.to_raw_ptr())
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdCudaLaunchKernelNV.html>
    #[inline]
    pub unsafe fn cmd_cuda_launch_kernel(
        &self,
        command_buffer: vk::CommandBuffer,
        launch_info: &vk::CudaLaunchInfoNV<'_>,
    ) {
        (self.fp.cmd_cuda_launch_kernel_nv)(command_buffer, launch_info)
    }
}
