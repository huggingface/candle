//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_cooperative_matrix.html>

use crate::vk;
use crate::VkResult;
use core::mem;
use core::ptr;

impl crate::khr::cooperative_matrix::Instance {
    /// Retrieve the number of elements to pass to [`get_physical_device_cooperative_matrix_properties()`][Self::get_physical_device_cooperative_matrix_properties()]
    #[inline]
    pub unsafe fn get_physical_device_cooperative_matrix_properties_len(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> VkResult<usize> {
        let mut count = mem::MaybeUninit::uninit();
        (self
            .fp
            .get_physical_device_cooperative_matrix_properties_khr)(
            physical_device,
            count.as_mut_ptr(),
            ptr::null_mut(),
        )
        .assume_init_on_success(count)
        .map(|c| c as usize)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR.html>
    ///
    /// Call [`get_physical_device_cooperative_matrix_properties_len()`][Self::get_physical_device_cooperative_matrix_properties_len()] to query the number of elements to pass to `out`.
    /// Be sure to [`Default::default()`]-initialize these elements and optionally set their `p_next` pointer.
    #[inline]
    pub unsafe fn get_physical_device_cooperative_matrix_properties(
        &self,
        physical_device: vk::PhysicalDevice,
        out: &mut [vk::CooperativeMatrixPropertiesKHR<'_>],
    ) -> VkResult<()> {
        let mut count = out.len() as u32;
        (self
            .fp
            .get_physical_device_cooperative_matrix_properties_khr)(
            physical_device,
            &mut count,
            out.as_mut_ptr(),
        )
        .result()?;
        assert_eq!(count as usize, out.len());
        Ok(())
    }
}
