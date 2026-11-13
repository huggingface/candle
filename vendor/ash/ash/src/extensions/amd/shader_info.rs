//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_AMD_shader_info.html>

use crate::read_into_uninitialized_vector;
use crate::vk;
use crate::VkResult;
use alloc::vec::Vec;
use core::mem;
use core::mem::size_of_val; // TODO: Remove when bumping MSRV to 1.80

impl crate::amd::shader_info::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetShaderInfoAMD.html> with [`vk::ShaderInfoTypeAMD::STATISTICS`]
    #[inline]
    pub unsafe fn get_shader_info_statistics(
        &self,
        pipeline: vk::Pipeline,
        shader_stage: vk::ShaderStageFlags,
    ) -> VkResult<vk::ShaderStatisticsInfoAMD> {
        let mut info = mem::MaybeUninit::<vk::ShaderStatisticsInfoAMD>::uninit();
        let mut size = size_of_val(&info);
        (self.fp.get_shader_info_amd)(
            self.handle,
            pipeline,
            shader_stage,
            vk::ShaderInfoTypeAMD::STATISTICS,
            &mut size,
            info.as_mut_ptr().cast(),
        )
        .result()?;
        assert_eq!(size, size_of_val(&info));
        Ok(info.assume_init())
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetShaderInfoAMD.html> with [`vk::ShaderInfoTypeAMD::BINARY`]
    #[inline]
    pub unsafe fn get_shader_info_binary(
        &self,
        pipeline: vk::Pipeline,
        shader_stage: vk::ShaderStageFlags,
    ) -> VkResult<Vec<u8>> {
        read_into_uninitialized_vector(|count, data: *mut u8| {
            (self.fp.get_shader_info_amd)(
                self.handle,
                pipeline,
                shader_stage,
                vk::ShaderInfoTypeAMD::BINARY,
                count,
                data.cast(),
            )
        })
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetShaderInfoAMD.html> with [`vk::ShaderInfoTypeAMD::DISASSEMBLY`]
    #[inline]
    pub unsafe fn get_shader_info_disassembly(
        &self,
        pipeline: vk::Pipeline,
        shader_stage: vk::ShaderStageFlags,
    ) -> VkResult<Vec<u8>> {
        read_into_uninitialized_vector(|count, data: *mut u8| {
            (self.fp.get_shader_info_amd)(
                self.handle,
                pipeline,
                shader_stage,
                vk::ShaderInfoTypeAMD::DISASSEMBLY,
                count,
                data.cast(),
            )
        })
    }
}
