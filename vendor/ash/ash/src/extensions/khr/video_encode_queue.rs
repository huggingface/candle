//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_video_encode_queue.html>

use crate::vk;
use crate::RawMutPtr;
use crate::VkResult;
use core::mem;
use core::ptr;

impl crate::khr::video_encode_queue::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdEncodeVideoKHR.html>
    #[inline]
    #[doc(alias = "vkCmdEncodeVideoKHR")]
    pub unsafe fn cmd_encode_video(
        &self,
        command_buffer: vk::CommandBuffer,
        encode_info: &vk::VideoEncodeInfoKHR<'_>,
    ) {
        (self.fp.cmd_encode_video_khr)(command_buffer, encode_info)
    }

    /// Retrieve the length of the byte slice to pass to [`get_encoded_video_session_parameters()`][Self::get_encoded_video_session_parameters()].
    #[inline]
    pub unsafe fn get_encoded_video_session_parameters_len(
        &self,
        session_parameters_info: &vk::VideoEncodeSessionParametersGetInfoKHR<'_>,
        feedback_info: Option<&mut vk::VideoEncodeSessionParametersFeedbackInfoKHR<'_>>,
    ) -> VkResult<usize> {
        let mut size = mem::MaybeUninit::uninit();
        (self.fp.get_encoded_video_session_parameters_khr)(
            self.handle,
            session_parameters_info,
            feedback_info.to_raw_mut_ptr(),
            size.as_mut_ptr(),
            ptr::null_mut(),
        )
        .assume_init_on_success(size)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetEncodedVideoSessionParametersKHR.html>
    ///
    /// Call [`get_encoded_video_session_parameters_len()`][Self::get_encoded_video_session_parameters_len()] to query the length of the byte slice to pass to `out`.
    #[inline]
    #[doc(alias = "vkGetEncodedVideoSessionParametersKHR")]
    pub unsafe fn get_encoded_video_session_parameters(
        &self,
        session_parameters_info: &vk::VideoEncodeSessionParametersGetInfoKHR<'_>,
        feedback_info: Option<&mut vk::VideoEncodeSessionParametersFeedbackInfoKHR<'_>>,
        out: &mut [mem::MaybeUninit<u8>],
    ) -> VkResult<()> {
        let mut size = out.len();
        (self.fp.get_encoded_video_session_parameters_khr)(
            self.handle,
            session_parameters_info,
            feedback_info.to_raw_mut_ptr(),
            &mut size,
            out.as_mut_ptr() as _,
        )
        .result()?;
        assert_eq!(size, out.len());
        Ok(())
    }
}

impl crate::khr::video_encode_queue::Instance {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR.html>
    #[inline]
    #[doc(alias = "vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR")]
    pub unsafe fn get_physical_device_video_encode_quality_level_properties(
        &self,
        physical_device: vk::PhysicalDevice,
        quality_level_info: &vk::PhysicalDeviceVideoEncodeQualityLevelInfoKHR<'_>,
        quality_level_properties: &mut vk::VideoEncodeQualityLevelPropertiesKHR<'_>,
    ) -> VkResult<()> {
        (self
            .fp
            .get_physical_device_video_encode_quality_level_properties_khr)(
            physical_device,
            quality_level_info,
            quality_level_properties,
        )
        .result()
    }
}
