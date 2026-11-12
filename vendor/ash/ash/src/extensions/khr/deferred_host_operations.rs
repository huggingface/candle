//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_deferred_host_operations.html>

use crate::vk;
use crate::RawPtr;
use crate::VkResult;
use core::mem;

impl crate::khr::deferred_host_operations::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateDeferredOperationKHR.html>
    #[inline]
    pub unsafe fn create_deferred_operation(
        &self,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) -> VkResult<vk::DeferredOperationKHR> {
        let mut operation = mem::MaybeUninit::uninit();
        (self.fp.create_deferred_operation_khr)(
            self.handle,
            allocation_callbacks.to_raw_ptr(),
            operation.as_mut_ptr(),
        )
        .assume_init_on_success(operation)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkDeferredOperationJoinKHR.html>
    #[inline]
    pub unsafe fn deferred_operation_join(
        &self,
        operation: vk::DeferredOperationKHR,
    ) -> VkResult<()> {
        (self.fp.deferred_operation_join_khr)(self.handle, operation).result()
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkDestroyDeferredOperationKHR.html>
    #[inline]
    pub unsafe fn destroy_deferred_operation(
        &self,
        operation: vk::DeferredOperationKHR,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) {
        (self.fp.destroy_deferred_operation_khr)(
            self.handle,
            operation,
            allocation_callbacks.to_raw_ptr(),
        )
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetDeferredOperationMaxConcurrencyKHR.html>
    #[inline]
    pub unsafe fn get_deferred_operation_max_concurrency(
        &self,
        operation: vk::DeferredOperationKHR,
    ) -> u32 {
        (self.fp.get_deferred_operation_max_concurrency_khr)(self.handle, operation)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetDeferredOperationResultKHR.html>
    #[inline]
    pub unsafe fn get_deferred_operation_result(
        &self,
        operation: vk::DeferredOperationKHR,
    ) -> VkResult<()> {
        (self.fp.get_deferred_operation_result_khr)(self.handle, operation).result()
    }
}
