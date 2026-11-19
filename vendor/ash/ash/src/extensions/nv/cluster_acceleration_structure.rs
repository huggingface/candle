//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_NV_cluster_acceleration_structure.html>

use crate::vk;

impl crate::nv::cluster_acceleration_structure::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetClusterAccelerationStructureBuildSizesNV.html>
    #[inline]
    pub unsafe fn get_cluster_acceleration_structure_build_sizes(
        &self,
        info: &vk::ClusterAccelerationStructureInputInfoNV<'_>,
        size_info: &mut vk::AccelerationStructureBuildSizesInfoKHR<'_>,
    ) {
        (self.fp.get_cluster_acceleration_structure_build_sizes_nv)(self.handle, info, size_info)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdBuildClusterAccelerationStructureIndirectNV.html>
    #[inline]
    pub unsafe fn cmd_build_cluster_acceleration_structure_indirect(
        &self,
        command_buffer: vk::CommandBuffer,
        command_infos: &vk::ClusterAccelerationStructureCommandsInfoNV<'_>,
    ) {
        (self.fp.cmd_build_cluster_acceleration_structure_indirect_nv)(
            command_buffer,
            command_infos,
        )
    }
}
