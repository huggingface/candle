//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_NV_cooperative_vector.html>

use crate::vk;
use crate::VkResult;
use core::mem;
use core::ptr;

impl crate::nv::cooperative_vector::Instance {
    /// Retrieve the number of elements to pass to [`get_physical_device_cooperative_vector_properties()`][Self::get_physical_device_cooperative_vector_properties()]
    #[inline]
    #[doc(alias = "vkGetPhysicalDeviceCooperativeVectorPropertiesNV")]
    pub unsafe fn get_physical_device_cooperative_vector_properties_len(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> VkResult<usize> {
        let mut count = mem::MaybeUninit::uninit();
        (self.fp.get_physical_device_cooperative_vector_properties_nv)(
            physical_device,
            count.as_mut_ptr(),
            ptr::null_mut(),
        )
        .assume_init_on_success(count)
        .map(|c| c as usize)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDeviceCooperativeVectorPropertiesNV.html>
    ///
    /// Call [`get_physical_device_cooperative_vector_properties_len()`][Self::get_physical_device_cooperative_vector_properties_len()] to query the number of elements to pass to `out`.
    /// Be sure to [`Default::default()`]-initialize these elements and optionally set their `p_next` pointer.
    #[inline]
    #[doc(alias = "vkGetPhysicalDeviceCooperativeVectorPropertiesNV")]
    pub unsafe fn get_physical_device_cooperative_vector_properties(
        &self,
        physical_device: vk::PhysicalDevice,
        out: &mut [vk::CooperativeVectorPropertiesNV<'_>],
    ) -> VkResult<()> {
        let mut count = out.len() as u32;
        (self.fp.get_physical_device_cooperative_vector_properties_nv)(
            physical_device,
            &mut count,
            out.as_mut_ptr(),
        )
        .result()?;
        assert_eq!(count as usize, out.len());
        Ok(())
    }
}

impl crate::nv::cooperative_vector::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdConvertCooperativeVectorMatrixNV.html>
    #[inline]
    #[doc(alias = "vkCmdConvertCooperativeVectorMatrixNV")]
    pub unsafe fn cmd_convert_cooperative_vector_matrix(
        &self,
        command_buffer: vk::CommandBuffer,
        infos: &[vk::ConvertCooperativeVectorMatrixInfoNV<'_>],
    ) {
        (self.fp.cmd_convert_cooperative_vector_matrix_nv)(
            command_buffer,
            infos.len() as u32,
            infos.as_ptr(),
        )
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkConvertCooperativeVectorMatrixNV.html>
    #[inline]
    #[doc(alias = "vkConvertCooperativeVectorMatrixNV")]
    pub unsafe fn convert_cooperative_vector_matrix(
        &self,
        info: &vk::ConvertCooperativeVectorMatrixInfoNV<'_>,
    ) -> VkResult<()> {
        (self.fp.convert_cooperative_vector_matrix_nv)(self.handle, info).result()
    }
}
