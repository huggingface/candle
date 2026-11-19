use crate::{
    command_buffer::{
        sys::{CommandBuffer, RecordingCommandBuffer},
        CommandBufferLevel,
    },
    device::{DeviceOwned, QueueFlags},
    ValidationError, VulkanObject,
};
use smallvec::SmallVec;
use std::cmp::min;

impl RecordingCommandBuffer {
    #[inline]
    #[track_caller]
    pub unsafe fn execute_commands(&mut self, command_buffers: &[&CommandBuffer]) -> &mut Self {
        unsafe { self.try_execute_commands(command_buffers) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_execute_commands(
        &mut self,
        command_buffers: &[&CommandBuffer],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_execute_commands(command_buffers.iter().copied())?;

        Ok(unsafe { self.execute_commands_unchecked(command_buffers) })
    }

    pub(crate) fn validate_execute_commands<'a>(
        &self,
        command_buffers: impl Iterator<Item = &'a CommandBuffer>,
    ) -> Result<(), Box<ValidationError>> {
        if self.level() != CommandBufferLevel::Primary {
            return Err(Box::new(ValidationError {
                problem: "this command buffer is not a primary command buffer".into(),
                vuids: &["VUID-vkCmdExecuteCommands-bufferlevel"],
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::TRANSFER | QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    transfer, graphics or compute operations"
                    .into(),
                vuids: &["VUID-vkCmdExecuteCommands-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        for (command_buffer_index, command_buffer) in command_buffers.enumerate() {
            // VUID-vkCmdExecuteCommands-commonparent
            assert_eq!(self.device(), command_buffer.device());

            if command_buffer.level() != CommandBufferLevel::Secondary {
                return Err(Box::new(ValidationError {
                    context: format!("command_buffers[{}]", command_buffer_index).into(),
                    problem: "is not a secondary command buffer".into(),
                    vuids: &["VUID-vkCmdExecuteCommands-pCommandBuffers-00088"],
                    ..Default::default()
                }));
            }

            // TODO:
            // VUID-vkCmdExecuteCommands-pCommandBuffers-00094
        }

        // TODO:
        // VUID-vkCmdExecuteCommands-pCommandBuffers-00091
        // VUID-vkCmdExecuteCommands-pCommandBuffers-00092
        // VUID-vkCmdExecuteCommands-pCommandBuffers-00093
        // VUID-vkCmdExecuteCommands-pCommandBuffers-00105

        // VUID-vkCmdExecuteCommands-pCommandBuffers-00089
        // Partially ensured by the `RawCommandBuffer` type.

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn execute_commands_unchecked(
        &mut self,
        command_buffers: &[&CommandBuffer],
    ) -> &mut Self {
        if command_buffers.is_empty() {
            return self;
        }

        let command_buffers_vk: SmallVec<[_; 4]> =
            command_buffers.iter().map(|cb| cb.handle()).collect();

        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_execute_commands)(
                self.handle(),
                command_buffers_vk.len() as u32,
                command_buffers_vk.as_ptr(),
            )
        };

        // If the secondary is non-concurrent or one-time use, that restricts the primary as
        // well.
        self.usage = command_buffers
            .iter()
            .map(|cb| cb.usage())
            .fold(self.usage, min);

        self
    }
}
