//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_OHOS_surface.html>

use crate::vk;
use crate::RawPtr;
use crate::VkResult;
use core::mem;

impl crate::ohos::surface::Instance {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateSurfaceOHOS.html>
    #[inline]
    pub unsafe fn create_surface(
        &self,
        create_info: &vk::SurfaceCreateInfoOHOS<'_>,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) -> VkResult<vk::SurfaceKHR> {
        let mut surface = mem::MaybeUninit::uninit();
        (self.fp.create_surface_ohos)(
            self.handle,
            create_info,
            allocation_callbacks.to_raw_ptr(),
            surface.as_mut_ptr(),
        )
        .assume_init_on_success(surface)
    }
}
