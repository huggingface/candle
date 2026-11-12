use crate::{
    command_buffer::sys::RecordingCommandBuffer,
    device::{DeviceOwned, QueueFlags},
    instance::debug::DebugUtilsLabel,
    Requires, RequiresAllOf, RequiresOneOf, ValidationError, VulkanObject,
};

impl RecordingCommandBuffer {
    #[inline]
    #[track_caller]
    pub unsafe fn begin_debug_utils_label(&mut self, label_info: &DebugUtilsLabel) -> &mut Self {
        unsafe { self.try_begin_debug_utils_label(label_info) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_begin_debug_utils_label(
        &mut self,
        label_info: &DebugUtilsLabel,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_begin_debug_utils_label(label_info)?;

        Ok(unsafe { self.begin_debug_utils_label_unchecked(label_info) })
    }

    pub(crate) fn validate_begin_debug_utils_label(
        &self,
        _label_info: &DebugUtilsLabel,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .device()
            .instance()
            .enabled_extensions()
            .ext_debug_utils
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "ext_debug_utils",
                )])]),
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics or compute operations"
                    .into(),
                vuids: &["VUID-vkCmdBeginDebugUtilsLabelEXT-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn begin_debug_utils_label_unchecked(
        &mut self,
        label_info: &DebugUtilsLabel,
    ) -> &mut Self {
        let label_info_fields1_vk = label_info.to_vk_fields1();
        let label_info_vk = label_info.to_vk(&label_info_fields1_vk);

        let fns = self.device().fns();
        unsafe {
            (fns.ext_debug_utils.cmd_begin_debug_utils_label_ext)(self.handle(), &label_info_vk)
        };

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn end_debug_utils_label(&mut self) -> &mut Self {
        unsafe { self.try_end_debug_utils_label() }.unwrap()
    }

    #[inline]
    pub unsafe fn try_end_debug_utils_label(&mut self) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_end_debug_utils_label()?;

        Ok(unsafe { self.end_debug_utils_label_unchecked() })
    }

    pub(crate) fn validate_end_debug_utils_label(&self) -> Result<(), Box<ValidationError>> {
        if !self
            .device()
            .instance()
            .enabled_extensions()
            .ext_debug_utils
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "ext_debug_utils",
                )])]),
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics or compute operations"
                    .into(),
                vuids: &["VUID-vkCmdEndDebugUtilsLabelEXT-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn end_debug_utils_label_unchecked(&mut self) -> &mut Self {
        let fns = self.device().fns();
        unsafe { (fns.ext_debug_utils.cmd_end_debug_utils_label_ext)(self.handle()) };

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn insert_debug_utils_label(&mut self, label_info: &DebugUtilsLabel) -> &mut Self {
        unsafe { self.try_insert_debug_utils_label(label_info) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_insert_debug_utils_label(
        &mut self,
        label_info: &DebugUtilsLabel,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_insert_debug_utils_label(label_info)?;

        Ok(unsafe { self.insert_debug_utils_label_unchecked(label_info) })
    }

    pub(crate) fn validate_insert_debug_utils_label(
        &self,
        _label_info: &DebugUtilsLabel,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .device()
            .instance()
            .enabled_extensions()
            .ext_debug_utils
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "ext_debug_utils",
                )])]),
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics or compute operations"
                    .into(),
                vuids: &["VUID-vkCmdInsertDebugUtilsLabelEXT-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn insert_debug_utils_label_unchecked(
        &mut self,
        label_info: &DebugUtilsLabel,
    ) -> &mut Self {
        let label_info_fields1_vk = label_info.to_vk_fields1();
        let label_info_vk = label_info.to_vk(&label_info_fields1_vk);

        let fns = self.device().fns();
        unsafe {
            (fns.ext_debug_utils.cmd_insert_debug_utils_label_ext)(self.handle(), &label_info_vk)
        };

        self
    }
}
