//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_NV_memory_decompression.html>

use crate::vk;

impl crate::nv::memory_decompression::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdDecompressMemoryNV.html>
    pub unsafe fn cmd_decompress_memory(
        &self,
        command_buffer: vk::CommandBuffer,
        decompress_memory_regions: &[vk::DecompressMemoryRegionNV],
    ) {
        (self.fp.cmd_decompress_memory_nv)(
            command_buffer,
            decompress_memory_regions.len() as u32,
            decompress_memory_regions.as_ptr(),
        )
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdDecompressMemoryIndirectCountNV.html>
    pub unsafe fn cmd_decompress_memory_indirect_count(
        &self,
        command_buffer: vk::CommandBuffer,
        indirect_commands_address: vk::DeviceAddress,
        indirect_commands_count_address: vk::DeviceAddress,
        stride: u32,
    ) {
        (self.fp.cmd_decompress_memory_indirect_count_nv)(
            command_buffer,
            indirect_commands_address,
            indirect_commands_count_address,
            stride,
        )
    }
}
