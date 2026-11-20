//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_private_data.html>

use crate::vk;
use crate::RawPtr;
use crate::VkResult;
use core::mem;

impl crate::ext::private_data::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCreatePrivateDataSlotEXT.html>
    #[inline]
    pub unsafe fn create_private_data_slot(
        &self,
        create_info: &vk::PrivateDataSlotCreateInfoEXT<'_>,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) -> VkResult<vk::PrivateDataSlotEXT> {
        let mut private_data_slot = mem::MaybeUninit::uninit();
        (self.fp.create_private_data_slot_ext)(
            self.handle,
            create_info,
            allocation_callbacks.to_raw_ptr(),
            private_data_slot.as_mut_ptr(),
        )
        .assume_init_on_success(private_data_slot)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkDestroyPrivateDataSlotEXT.html>
    #[inline]
    pub unsafe fn destroy_private_data_slot(
        &self,
        private_data_slot: vk::PrivateDataSlotEXT,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) {
        (self.fp.destroy_private_data_slot_ext)(
            self.handle,
            private_data_slot,
            allocation_callbacks.to_raw_ptr(),
        )
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkSetPrivateDataEXT.html>
    #[inline]
    pub unsafe fn set_private_data<T: vk::Handle>(
        &self,
        object: T,
        private_data_slot: vk::PrivateDataSlotEXT,
        data: u64,
    ) -> VkResult<()> {
        (self.fp.set_private_data_ext)(
            self.handle,
            T::TYPE,
            object.as_raw(),
            private_data_slot,
            data,
        )
        .result()
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPrivateDataEXT.html>
    #[inline]
    pub unsafe fn get_private_data<T: vk::Handle>(
        &self,
        object: T,
        private_data_slot: vk::PrivateDataSlotEXT,
    ) -> u64 {
        let mut data = mem::MaybeUninit::uninit();
        (self.fp.get_private_data_ext)(
            self.handle,
            T::TYPE,
            object.as_raw(),
            private_data_slot,
            data.as_mut_ptr(),
        );
        data.assume_init()
    }
}
