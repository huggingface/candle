use crate::{
    buffer::{Buffer, BufferUsage},
    command_buffer::{
        sys::RecordingCommandBuffer, DispatchIndirectCommand, DrawIndexedIndirectCommand,
        DrawIndirectCommand, DrawMeshTasksIndirectCommand,
    },
    device::{DeviceOwned, QueueFlags},
    pipeline::ray_tracing::ShaderBindingTableAddresses,
    DeviceSize, Requires, RequiresAllOf, RequiresOneOf, ValidationError, Version, VulkanObject,
};
use ash::vk::DeviceAddress;

impl RecordingCommandBuffer {
    #[inline]
    #[track_caller]
    pub unsafe fn dispatch(&mut self, group_counts: [u32; 3]) -> &mut Self {
        unsafe { self.try_dispatch(group_counts) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_dispatch(
        &mut self,
        group_counts: [u32; 3],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_dispatch(group_counts)?;

        Ok(unsafe { self.dispatch_unchecked(group_counts) })
    }

    pub(crate) fn validate_dispatch(
        &self,
        group_counts: [u32; 3],
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    compute operations"
                    .into(),
                vuids: &["VUID-vkCmdDispatch-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        let properties = self.device().physical_device().properties();

        if group_counts[0] > properties.max_compute_work_group_count[0] {
            return Err(Box::new(ValidationError {
                context: "group_counts[0]".into(),
                problem: "is greater than the `max_compute_work_group_count[0]` limit".into(),
                vuids: &["VUID-vkCmdDispatch-groupCountX-00386"],
                ..Default::default()
            }));
        }

        if group_counts[1] > properties.max_compute_work_group_count[1] {
            return Err(Box::new(ValidationError {
                context: "group_counts[1]".into(),
                problem: "is greater than the `max_compute_work_group_count[1]` limit".into(),
                vuids: &["VUID-vkCmdDispatch-groupCountY-00387"],
                ..Default::default()
            }));
        }

        if group_counts[2] > properties.max_compute_work_group_count[2] {
            return Err(Box::new(ValidationError {
                context: "group_counts[2]".into(),
                problem: "is greater than the `max_compute_work_group_count[2]` limit".into(),
                vuids: &["VUID-vkCmdDispatch-groupCountZ-00388"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn dispatch_unchecked(&mut self, group_counts: [u32; 3]) -> &mut Self {
        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_dispatch)(
                self.handle(),
                group_counts[0],
                group_counts[1],
                group_counts[2],
            )
        };

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn dispatch_indirect(&mut self, buffer: &Buffer, offset: DeviceSize) -> &mut Self {
        unsafe { self.try_dispatch_indirect(buffer, offset) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_dispatch_indirect(
        &mut self,
        buffer: &Buffer,
        offset: DeviceSize,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_dispatch_indirect(buffer, offset)?;

        Ok(unsafe { self.dispatch_indirect_unchecked(buffer, offset) })
    }

    pub(crate) fn validate_dispatch_indirect(
        &self,
        buffer: &Buffer,
        offset: DeviceSize,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    compute operations"
                    .into(),
                vuids: &["VUID-vkCmdDispatchIndirect-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        // VUID-vkCmdDispatchIndirect-commonparent
        assert_eq!(self.device(), buffer.device());

        if !buffer.usage().intersects(BufferUsage::INDIRECT_BUFFER) {
            return Err(Box::new(ValidationError {
                context: "buffer.usage()".into(),
                problem: "does not contain `BufferUsage::INDIRECT_BUFFER`".into(),
                vuids: &["VUID-vkCmdDispatchIndirect-buffer-02709"],
                ..Default::default()
            }));
        }

        if !offset.is_multiple_of(4) {
            return Err(Box::new(ValidationError {
                context: "offset".into(),
                problem: "is not a multiple of 4".into(),
                vuids: &["VUID-vkCmdDispatchIndirect-offset-02710"],
                ..Default::default()
            }));
        }

        if offset
            .checked_add(size_of::<DispatchIndirectCommand>() as DeviceSize)
            .is_none_or(|end| end > buffer.size())
        {
            return Err(Box::new(ValidationError {
                problem: "`offset + size_of::<DispatchIndirectCommand>()` is greater than \
                    `buffer.size()`"
                    .into(),
                vuids: &["VUID-vkCmdDispatchIndirect-offset-00407"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn dispatch_indirect_unchecked(
        &mut self,
        buffer: &Buffer,
        offset: DeviceSize,
    ) -> &mut Self {
        let fns = self.device().fns();
        unsafe { (fns.v1_0.cmd_dispatch_indirect)(self.handle(), buffer.handle(), offset) };

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> &mut Self {
        unsafe { self.try_draw(vertex_count, instance_count, first_vertex, first_instance) }
            .unwrap()
    }

    #[inline]
    pub unsafe fn try_draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_draw(vertex_count, instance_count, first_vertex, first_instance)?;

        Ok(unsafe {
            self.draw_unchecked(vertex_count, instance_count, first_vertex, first_instance)
        })
    }

    pub(crate) fn validate_draw(
        &self,
        _vertex_count: u32,
        _instance_count: u32,
        _first_vertex: u32,
        _first_instance: u32,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdDraw-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn draw_unchecked(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> &mut Self {
        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_draw)(
                self.handle(),
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            )
        };

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn draw_indirect(
        &mut self,
        buffer: &Buffer,
        offset: DeviceSize,
        draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        unsafe { self.try_draw_indirect(buffer, offset, draw_count, stride) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_draw_indirect(
        &mut self,
        buffer: &Buffer,
        offset: DeviceSize,
        draw_count: u32,
        stride: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_draw_indirect(buffer, offset, draw_count, stride)?;

        Ok(unsafe { self.draw_indirect_unchecked(buffer, offset, draw_count, stride) })
    }

    pub(crate) fn validate_draw_indirect(
        &self,
        buffer: &Buffer,
        offset: DeviceSize,
        draw_count: u32,
        stride: u32,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdDrawIndirect-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        if !buffer.usage().intersects(BufferUsage::INDIRECT_BUFFER) {
            return Err(Box::new(ValidationError {
                context: "buffer.usage()".into(),
                problem: "does not contain `BufferUsage::INDIRECT_BUFFER`".into(),
                vuids: &["VUID-vkCmdDrawIndirect-buffer-02709"],
                ..Default::default()
            }));
        }

        if !offset.is_multiple_of(4) {
            return Err(Box::new(ValidationError {
                context: "offset".into(),
                problem: "is not a multiple of 4".into(),
                vuids: &["VUID-vkCmdDrawIndirect-offset-02710"],
                ..Default::default()
            }));
        }

        if draw_count > 1 {
            if !self.device().enabled_features().multi_draw_indirect {
                return Err(Box::new(ValidationError {
                    context: "draw_count".into(),
                    problem: "is greater than 1".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "multi_draw_indirect",
                    )])]),
                    vuids: &["VUID-vkCmdDrawIndirect-drawCount-02718"],
                }));
            }

            if !stride.is_multiple_of(4) {
                return Err(Box::new(ValidationError {
                    problem: "`draw_count` is greater than 1, but `stride` is not a multiple of 4"
                        .into(),
                    vuids: &["VUID-vkCmdDrawIndirect-drawCount-00476"],
                    ..Default::default()
                }));
            }

            if (stride as DeviceSize) < size_of::<DrawIndirectCommand>() as DeviceSize {
                return Err(Box::new(ValidationError {
                    problem: "`draw_count` is greater than 1, but `stride` is less than \
                        `size_of::<DrawIndirectCommand>()`"
                        .into(),
                    vuids: &["VUID-vkCmdDrawIndirect-drawCount-00476"],
                    ..Default::default()
                }));
            }

            if (stride as DeviceSize)
                .checked_mul(draw_count as DeviceSize - 1)
                .and_then(|x| x.checked_add(offset))
                .and_then(|x| x.checked_add(size_of::<DrawIndirectCommand>() as DeviceSize))
                .is_none_or(|end| end > buffer.size())
            {
                return Err(Box::new(ValidationError {
                    problem: "`draw_count` is greater than 1, but \
                        `stride * (draw_count - 1) + offset + size_of::<DrawIndirectCommand>()` is \
                        greater than `buffer.size()`"
                        .into(),
                    vuids: &["VUID-vkCmdDrawIndirect-drawCount-00488"],
                    ..Default::default()
                }));
            }
        } else {
            if offset
                .checked_add(size_of::<DrawIndirectCommand>() as DeviceSize)
                .is_none_or(|end| end > buffer.size())
            {
                return Err(Box::new(ValidationError {
                    problem: "`draw_count` is 1, but `offset + size_of::<DrawIndirectCommand>()` \
                        is greater than `buffer.size()`"
                        .into(),
                    vuids: &["VUID-vkCmdDrawIndirect-drawCount-00487"],
                    ..Default::default()
                }));
            }
        }

        let properties = self.device().physical_device().properties();

        if draw_count > properties.max_draw_indirect_count {
            return Err(Box::new(ValidationError {
                context: "draw_count".into(),
                problem: "is greater than the `max_draw_indirect_count` limit".into(),
                vuids: &["VUID-vkCmdDrawIndirect-drawCount-02719"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn draw_indirect_unchecked(
        &mut self,
        buffer: &Buffer,
        offset: DeviceSize,
        draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_draw_indirect)(self.handle(), buffer.handle(), offset, draw_count, stride)
        };

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn draw_indirect_count(
        &mut self,
        buffer: &Buffer,
        offset: DeviceSize,
        count_buffer: &Buffer,
        count_buffer_offset: DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        unsafe {
            self.try_draw_indirect_count(
                buffer,
                offset,
                count_buffer,
                count_buffer_offset,
                max_draw_count,
                stride,
            )
        }
        .unwrap()
    }

    #[inline]
    pub unsafe fn try_draw_indirect_count(
        &mut self,
        buffer: &Buffer,
        offset: DeviceSize,
        count_buffer: &Buffer,
        count_buffer_offset: DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_draw_indirect_count(
            buffer,
            offset,
            count_buffer,
            count_buffer_offset,
            max_draw_count,
            stride,
        )?;

        Ok(unsafe {
            self.draw_indirect_count_unchecked(
                buffer,
                offset,
                count_buffer,
                count_buffer_offset,
                max_draw_count,
                stride,
            )
        })
    }

    pub(crate) fn validate_draw_indirect_count(
        &self,
        buffer: &Buffer,
        offset: DeviceSize,
        count_buffer: &Buffer,
        count_buffer_offset: DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) -> Result<(), Box<ValidationError>> {
        if !self.device().enabled_features().draw_indirect_count {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "draw_indirect_count",
                )])]),
                vuids: &["VUID-vkCmdDrawIndirectCount-None-04445"],
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdDrawIndirectCount-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        if !buffer.usage().intersects(BufferUsage::INDIRECT_BUFFER) {
            return Err(Box::new(ValidationError {
                context: "buffer.usage()".into(),
                problem: "does not contain `BufferUsage::INDIRECT_BUFFER`".into(),
                vuids: &["VUID-vkCmdDrawIndirectCount-buffer-02709"],
                ..Default::default()
            }));
        }

        if !offset.is_multiple_of(4) {
            return Err(Box::new(ValidationError {
                context: "offset".into(),
                problem: "is not a multiple of 4".into(),
                vuids: &["VUID-vkCmdDrawIndirectCount-offset-02710"],
                ..Default::default()
            }));
        }

        if !count_buffer
            .usage()
            .intersects(BufferUsage::INDIRECT_BUFFER)
        {
            return Err(Box::new(ValidationError {
                context: "count_buffer.usage()".into(),
                problem: "does not contain `BufferUsage::INDIRECT_BUFFER`".into(),
                vuids: &["VUID-vkCmdDrawIndirectCount-countBuffer-02715"],
                ..Default::default()
            }));
        }

        if !count_buffer_offset.is_multiple_of(4) {
            return Err(Box::new(ValidationError {
                context: "count_buffer_offset".into(),
                problem: "is not a multiple of 4".into(),
                vuids: &["VUID-vkCmdDrawIndirectCount-countBufferOffset-02716"],
                ..Default::default()
            }));
        }

        if !stride.is_multiple_of(4) {
            return Err(Box::new(ValidationError {
                context: "stride".into(),
                problem: "is not a multiple of 4".into(),
                vuids: &["VUID-vkCmdDrawIndirectCount-stride-03110"],
                ..Default::default()
            }));
        }

        if (stride as DeviceSize) < size_of::<DrawIndirectCommand>() as DeviceSize {
            return Err(Box::new(ValidationError {
                context: "stride".into(),
                problem: "is less than `size_of::<DrawIndirectCommand>()`".into(),
                vuids: &["VUID-vkCmdDrawIndirectCount-stride-03110"],
                ..Default::default()
            }));
        }

        if max_draw_count >= 1
            && (stride as DeviceSize)
                .checked_mul(max_draw_count as DeviceSize - 1)
                .and_then(|x| x.checked_add(offset))
                .and_then(|x| x.checked_add(size_of::<DrawIndirectCommand>() as DeviceSize))
                .is_none_or(|end| end > buffer.size())
        {
            return Err(Box::new(ValidationError {
                problem: "`max_draw_count` is equal to or greater than 1, but \
                    `stride * (max_draw_count - 1) + offset + size_of::<DrawIndirectCommand>()` is \
                    greater than `buffer.size()`"
                    .into(),
                vuids: &["VUID-vkCmdDrawIndirectCount-maxDrawCount-03111"],
                ..Default::default()
            }));
        }

        if count_buffer_offset
            .checked_add(size_of::<u32>() as DeviceSize)
            .is_none_or(|end| end > count_buffer.size())
        {
            return Err(Box::new(ValidationError {
                problem: "`count_buffer_offset + size_of::<u32>()` is greater than \
                    `count_buffer.size()`"
                    .into(),
                vuids: &["VUID-vkCmdDrawIndirectCount-countBufferOffset-04129"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn draw_indirect_count_unchecked(
        &mut self,
        buffer: &Buffer,
        offset: DeviceSize,
        count_buffer: &Buffer,
        count_buffer_offset: DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        let device = self.device();
        let fns = device.fns();
        let cmd_draw_indirect_count = if device.api_version() >= Version::V1_2 {
            fns.v1_2.cmd_draw_indirect_count
        } else if device.enabled_extensions().khr_draw_indirect_count {
            fns.khr_draw_indirect_count.cmd_draw_indirect_count_khr
        } else if device.enabled_extensions().amd_draw_indirect_count {
            fns.amd_draw_indirect_count.cmd_draw_indirect_count_amd
        } else {
            std::process::abort();
        };

        unsafe {
            cmd_draw_indirect_count(
                self.handle(),
                buffer.handle(),
                offset,
                count_buffer.handle(),
                count_buffer_offset,
                max_draw_count,
                stride,
            )
        };

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> &mut Self {
        unsafe {
            self.try_draw_indexed(
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            )
        }
        .unwrap()
    }

    #[inline]
    pub unsafe fn try_draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_draw_indexed(
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
        )?;

        Ok(unsafe {
            self.draw_indexed_unchecked(
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            )
        })
    }

    pub(crate) fn validate_draw_indexed(
        &self,
        _index_count: u32,
        _instance_count: u32,
        _first_index: u32,
        _vertex_offset: i32,
        _first_instance: u32,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdDrawIndexed-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn draw_indexed_unchecked(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> &mut Self {
        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_draw_indexed)(
                self.handle(),
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            )
        };

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn draw_indexed_indirect(
        &mut self,
        buffer: &Buffer,
        offset: DeviceSize,
        draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        unsafe { self.try_draw_indexed_indirect(buffer, offset, draw_count, stride) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_draw_indexed_indirect(
        &mut self,
        buffer: &Buffer,
        offset: DeviceSize,
        draw_count: u32,
        stride: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_draw_indexed_indirect(buffer, offset, draw_count, stride)?;

        Ok(unsafe { self.draw_indexed_indirect_unchecked(buffer, offset, draw_count, stride) })
    }

    pub(crate) fn validate_draw_indexed_indirect(
        &self,
        buffer: &Buffer,
        offset: DeviceSize,
        draw_count: u32,
        stride: u32,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdDrawIndexedIndirect-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        if !buffer.usage().intersects(BufferUsage::INDIRECT_BUFFER) {
            return Err(Box::new(ValidationError {
                context: "buffer.usage()".into(),
                problem: "does not contain `BufferUsage::INDIRECT_BUFFER`".into(),
                vuids: &["VUID-vkCmdDrawIndexedIndirect-buffer-02709"],
                ..Default::default()
            }));
        }

        if !offset.is_multiple_of(4) {
            return Err(Box::new(ValidationError {
                context: "offset".into(),
                problem: "is not a multiple of 4".into(),
                vuids: &["VUID-vkCmdDrawIndexedIndirect-offset-02710"],
                ..Default::default()
            }));
        }

        if draw_count > 1 {
            if !self.device().enabled_features().multi_draw_indirect {
                return Err(Box::new(ValidationError {
                    context: "draw_count".into(),
                    problem: "is greater than 1".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "multi_draw_indirect",
                    )])]),
                    vuids: &["VUID-vkCmdDrawIndexedIndirect-drawCount-02718"],
                }));
            }

            if !stride.is_multiple_of(4) {
                return Err(Box::new(ValidationError {
                    problem: "`draw_count` is greater than 1, but `stride` is not a multiple of 4"
                        .into(),
                    vuids: &["VUID-vkCmdDrawIndexedIndirect-drawCount-00528"],
                    ..Default::default()
                }));
            }

            if (stride as DeviceSize) < size_of::<DrawIndexedIndirectCommand>() as DeviceSize {
                return Err(Box::new(ValidationError {
                    problem: "`draw_count` is greater than 1, but `stride` is less than \
                        `size_of::<DrawIndexedIndirectCommand>()`"
                        .into(),
                    vuids: &["VUID-vkCmdDrawIndexedIndirect-drawCount-00528"],
                    ..Default::default()
                }));
            }

            if (stride as DeviceSize)
                .checked_mul(draw_count as DeviceSize - 1)
                .and_then(|x| x.checked_add(offset))
                .and_then(|x| x.checked_add(size_of::<DrawIndexedIndirectCommand>() as DeviceSize))
                .is_none_or(|end| end > buffer.size())
            {
                return Err(Box::new(ValidationError {
                    problem: "`draw_count` is greater than 1, but `stride * (draw_count - 1) + \
                        offset + size_of::<DrawIndexedIndirectCommand>()` is greater than \
                        `buffer.size()`"
                        .into(),
                    vuids: &["VUID-vkCmdDrawIndexedIndirect-drawCount-00540"],
                    ..Default::default()
                }));
            }
        } else {
            if offset
                .checked_add(size_of::<DrawIndexedIndirectCommand>() as DeviceSize)
                .is_none_or(|end| end > buffer.size())
            {
                return Err(Box::new(ValidationError {
                    problem: "`draw_count` is 1, but `offset + \
                        size_of::<DrawIndexedIndirectCommand>()` is greater than `buffer.size()`"
                        .into(),
                    vuids: &["VUID-vkCmdDrawIndexedIndirect-drawCount-00539"],
                    ..Default::default()
                }));
            }
        }

        let properties = self.device().physical_device().properties();

        if draw_count > properties.max_draw_indirect_count {
            return Err(Box::new(ValidationError {
                context: "draw_count".into(),
                problem: "is greater than the `max_draw_indirect_count` limit".into(),
                vuids: &["VUID-vkCmdDrawIndexedIndirect-drawCount-02719"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn draw_indexed_indirect_unchecked(
        &mut self,
        buffer: &Buffer,
        offset: DeviceSize,
        draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_draw_indexed_indirect)(
                self.handle(),
                buffer.handle(),
                offset,
                draw_count,
                stride,
            )
        };

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn draw_indexed_indirect_count(
        &mut self,
        buffer: &Buffer,
        offset: DeviceSize,
        count_buffer: &Buffer,
        count_buffer_offset: DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        unsafe {
            self.try_draw_indexed_indirect_count(
                buffer,
                offset,
                count_buffer,
                count_buffer_offset,
                max_draw_count,
                stride,
            )
        }
        .unwrap()
    }

    #[inline]
    pub unsafe fn try_draw_indexed_indirect_count(
        &mut self,
        buffer: &Buffer,
        offset: DeviceSize,
        count_buffer: &Buffer,
        count_buffer_offset: DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_draw_indexed_indirect_count(
            buffer,
            offset,
            count_buffer,
            count_buffer_offset,
            max_draw_count,
            stride,
        )?;

        Ok(unsafe {
            self.draw_indexed_indirect_count_unchecked(
                buffer,
                offset,
                count_buffer,
                count_buffer_offset,
                max_draw_count,
                stride,
            )
        })
    }

    pub(crate) fn validate_draw_indexed_indirect_count(
        &self,
        buffer: &Buffer,
        offset: DeviceSize,
        count_buffer: &Buffer,
        count_buffer_offset: DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) -> Result<(), Box<ValidationError>> {
        if !self.device().enabled_features().draw_indirect_count {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "draw_indirect_count",
                )])]),
                vuids: &["VUID-vkCmdDrawIndexedIndirectCount-None-04445"],
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdDrawIndexedIndirectCount-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        if !buffer.usage().intersects(BufferUsage::INDIRECT_BUFFER) {
            return Err(Box::new(ValidationError {
                context: "buffer.usage()".into(),
                problem: "does not contain `BufferUsage::INDIRECT_BUFFER`".into(),
                vuids: &["VUID-vkCmdDrawIndexedIndirectCount-buffer-02709"],
                ..Default::default()
            }));
        }

        if !offset.is_multiple_of(4) {
            return Err(Box::new(ValidationError {
                context: "offset".into(),
                problem: "is not a multiple of 4".into(),
                vuids: &["VUID-vkCmdDrawIndexedIndirectCount-offset-02710"],
                ..Default::default()
            }));
        }

        if !count_buffer
            .usage()
            .intersects(BufferUsage::INDIRECT_BUFFER)
        {
            return Err(Box::new(ValidationError {
                context: "count_buffer.usage()".into(),
                problem: "does not contain `BufferUsage::INDIRECT_BUFFER`".into(),
                vuids: &["VUID-vkCmdDrawIndexedIndirectCount-countBuffer-02715"],
                ..Default::default()
            }));
        }

        if !count_buffer_offset.is_multiple_of(4) {
            return Err(Box::new(ValidationError {
                context: "count_buffer_offset".into(),
                problem: "is not a multiple of 4".into(),
                vuids: &["VUID-vkCmdDrawIndexedIndirectCount-countBufferOffset-02716"],
                ..Default::default()
            }));
        }

        if !stride.is_multiple_of(4) {
            return Err(Box::new(ValidationError {
                context: "stride".into(),
                problem: "is not a multiple of 4".into(),
                vuids: &["VUID-vkCmdDrawIndexedIndirectCount-stride-03142"],
                ..Default::default()
            }));
        }

        if (stride as DeviceSize) < size_of::<DrawIndirectCommand>() as DeviceSize {
            return Err(Box::new(ValidationError {
                context: "stride".into(),
                problem: "is less than `size_of::<DrawIndirectCommand>()`".into(),
                vuids: &["VUID-vkCmdDrawIndexedIndirectCount-stride-03142"],
                ..Default::default()
            }));
        }

        if max_draw_count >= 1
            && (stride as DeviceSize)
                .checked_mul(max_draw_count as DeviceSize - 1)
                .and_then(|x| x.checked_add(offset))
                .and_then(|x| x.checked_add(size_of::<DrawIndirectCommand>() as DeviceSize))
                .is_none_or(|end| end > buffer.size())
        {
            return Err(Box::new(ValidationError {
                problem: "`max_draw_count` is equal to or greater than 1, but `stride * \
                    (max_draw_count - 1) + offset + size_of::<DrawIndirectCommand>()` is greater \
                    than `buffer.size()`"
                    .into(),
                vuids: &["VUID-vkCmdDrawIndexedIndirectCount-maxDrawCount-03143"],
                ..Default::default()
            }));
        }

        if count_buffer_offset
            .checked_add(size_of::<u32>() as DeviceSize)
            .is_none_or(|end| end > count_buffer.size())
        {
            return Err(Box::new(ValidationError {
                problem: "`count_buffer_offset + size_of::<u32>()` is greater than \
                    `count_buffer.size()`"
                    .into(),
                vuids: &["VUID-vkCmdDrawIndexedIndirectCount-countBufferOffset-04129"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn draw_indexed_indirect_count_unchecked(
        &mut self,
        buffer: &Buffer,
        offset: DeviceSize,
        count_buffer: &Buffer,
        count_buffer_offset: DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        let device = self.device();
        let fns = device.fns();
        let cmd_draw_indexed_indirect_count = if device.api_version() >= Version::V1_2 {
            fns.v1_2.cmd_draw_indexed_indirect_count
        } else if device.enabled_extensions().khr_draw_indirect_count {
            fns.khr_draw_indirect_count
                .cmd_draw_indexed_indirect_count_khr
        } else if device.enabled_extensions().amd_draw_indirect_count {
            fns.amd_draw_indirect_count
                .cmd_draw_indexed_indirect_count_amd
        } else {
            std::process::abort();
        };

        unsafe {
            cmd_draw_indexed_indirect_count(
                self.handle(),
                buffer.handle(),
                offset,
                count_buffer.handle(),
                count_buffer_offset,
                max_draw_count,
                stride,
            )
        };

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn draw_mesh_tasks(&mut self, group_counts: [u32; 3]) -> &mut Self {
        unsafe { self.try_draw_mesh_tasks(group_counts) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_draw_mesh_tasks(
        &mut self,
        group_counts: [u32; 3],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_draw_mesh_tasks(group_counts)?;

        Ok(unsafe { self.draw_mesh_tasks_unchecked(group_counts) })
    }

    pub(crate) fn validate_draw_mesh_tasks(
        &self,
        _group_counts: [u32; 3],
    ) -> Result<(), Box<ValidationError>> {
        if !self.device().enabled_extensions().ext_mesh_shader {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "ext_mesh_shader",
                )])]),
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdDrawMeshTasksEXT-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn draw_mesh_tasks_unchecked(&mut self, group_counts: [u32; 3]) -> &mut Self {
        let fns = self.device().fns();
        unsafe {
            (fns.ext_mesh_shader.cmd_draw_mesh_tasks_ext)(
                self.handle(),
                group_counts[0],
                group_counts[1],
                group_counts[2],
            )
        };

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn draw_mesh_tasks_indirect(
        &mut self,
        buffer: &Buffer,
        offset: DeviceSize,
        draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        unsafe { self.try_draw_mesh_tasks_indirect(buffer, offset, draw_count, stride) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_draw_mesh_tasks_indirect(
        &mut self,
        buffer: &Buffer,
        offset: DeviceSize,
        draw_count: u32,
        stride: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_draw_mesh_tasks_indirect(buffer, offset, draw_count, stride)?;

        Ok(unsafe { self.draw_mesh_tasks_indirect_unchecked(buffer, offset, draw_count, stride) })
    }

    pub(crate) fn validate_draw_mesh_tasks_indirect(
        &self,
        buffer: &Buffer,
        offset: DeviceSize,
        draw_count: u32,
        stride: u32,
    ) -> Result<(), Box<ValidationError>> {
        if !self.device().enabled_extensions().ext_mesh_shader {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "ext_mesh_shader",
                )])]),
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdDrawMeshTasksIndirectEXT-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        // VUID-vkCmdDrawMeshTasksIndirectEXT-commonparent
        assert_eq!(self.device(), buffer.device());

        if !buffer.usage().intersects(BufferUsage::INDIRECT_BUFFER) {
            return Err(Box::new(ValidationError {
                context: "buffer.usage()".into(),
                problem: "does not contain `BufferUsage::INDIRECT_BUFFER`".into(),
                vuids: &["VUID-vkCmdDrawMeshTasksIndirectEXT-buffer-02709"],
                ..Default::default()
            }));
        }

        if !offset.is_multiple_of(4) {
            return Err(Box::new(ValidationError {
                context: "offset".into(),
                problem: "is not a multiple of 4".into(),
                vuids: &["VUID-vkCmdDrawMeshTasksIndirectEXT-offset-02710"],
                ..Default::default()
            }));
        }

        if draw_count > 1 {
            if !self.device().enabled_features().multi_draw_indirect {
                return Err(Box::new(ValidationError {
                    context: "draw_count".into(),
                    problem: "is greater than 1".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "multi_draw_indirect",
                    )])]),
                    vuids: &["VUID-vkCmdDrawMeshTasksIndirectEXT-drawCount-02718"],
                }));
            }

            if !stride.is_multiple_of(4) {
                return Err(Box::new(ValidationError {
                    problem: "`draw_count` is greater than 1, but `stride` is not a multiple of 4"
                        .into(),
                    vuids: &["VUID-vkCmdDrawMeshTasksIndirectEXT-drawCount-07088"],
                    ..Default::default()
                }));
            }

            if (stride as DeviceSize) < size_of::<DrawMeshTasksIndirectCommand>() as DeviceSize {
                return Err(Box::new(ValidationError {
                    problem: "`draw_count` is greater than 1, but `stride` is less than \
                        `size_of::<DrawMeshTasksIndirectCommand>()`"
                        .into(),
                    vuids: &["VUID-vkCmdDrawMeshTasksIndirectEXT-drawCount-07088"],
                    ..Default::default()
                }));
            }

            if (stride as DeviceSize)
                .checked_mul(draw_count as DeviceSize - 1)
                .and_then(|x| x.checked_add(offset))
                .and_then(|x| {
                    x.checked_add(size_of::<DrawMeshTasksIndirectCommand>() as DeviceSize)
                })
                .is_none_or(|end| end > buffer.size())
            {
                return Err(Box::new(ValidationError {
                    problem: "`draw_count` is greater than 1, but \
                        `stride * (draw_count - 1) + size_of::<DrawMeshTasksIndirectCommand>()` \
                        is greater than `buffer.size()`"
                        .into(),
                    vuids: &["VUID-vkCmdDrawMeshTasksIndirectEXT-drawCount-07090"],
                    ..Default::default()
                }));
            }
        } else {
            if offset
                .checked_add(size_of::<DrawMeshTasksIndirectCommand>() as DeviceSize)
                .is_none_or(|end| end > buffer.size())
            {
                return Err(Box::new(ValidationError {
                    problem: "`draw_count` is 1, but `offset + \
                        size_of::<DrawMeshTasksIndirectCommand>()` is greater than `buffer.size()`"
                        .into(),
                    vuids: &["VUID-vkCmdDrawMeshTasksIndirectEXT-drawCount-07089"],
                    ..Default::default()
                }));
            }
        }

        let properties = self.device().physical_device().properties();

        if draw_count > properties.max_draw_indirect_count {
            return Err(Box::new(ValidationError {
                context: "draw_count".into(),
                problem: "is greater than the `max_draw_indirect_count` limit".into(),
                vuids: &["VUID-vkCmdDrawMeshTasksIndirectEXT-drawCount-02719"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn draw_mesh_tasks_indirect_unchecked(
        &mut self,
        buffer: &Buffer,
        offset: DeviceSize,
        draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        let fns = self.device().fns();
        unsafe {
            (fns.ext_mesh_shader.cmd_draw_mesh_tasks_indirect_ext)(
                self.handle(),
                buffer.handle(),
                offset,
                draw_count,
                stride,
            )
        };

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn draw_mesh_tasks_indirect_count(
        &mut self,
        buffer: &Buffer,
        offset: DeviceSize,
        count_buffer: &Buffer,
        count_buffer_offset: DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        unsafe {
            self.try_draw_mesh_tasks_indirect_count(
                buffer,
                offset,
                count_buffer,
                count_buffer_offset,
                max_draw_count,
                stride,
            )
        }
        .unwrap()
    }

    #[inline]
    pub unsafe fn try_draw_mesh_tasks_indirect_count(
        &mut self,
        buffer: &Buffer,
        offset: DeviceSize,
        count_buffer: &Buffer,
        count_buffer_offset: DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_draw_mesh_tasks_indirect_count(
            buffer,
            offset,
            count_buffer,
            count_buffer_offset,
            max_draw_count,
            stride,
        )?;

        Ok(unsafe {
            self.draw_mesh_tasks_indirect_count_unchecked(
                buffer,
                offset,
                count_buffer,
                count_buffer_offset,
                max_draw_count,
                stride,
            )
        })
    }

    pub(crate) fn validate_draw_mesh_tasks_indirect_count(
        &self,
        buffer: &Buffer,
        offset: DeviceSize,
        count_buffer: &Buffer,
        count_buffer_offset: DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) -> Result<(), Box<ValidationError>> {
        if !self.device().enabled_extensions().ext_mesh_shader {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "ext_mesh_shader",
                )])]),
                ..Default::default()
            }));
        }

        if !self.device().enabled_features().draw_indirect_count {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "draw_indirect_count",
                )])]),
                vuids: &["VUID-vkCmdDrawMeshTasksIndirectCountEXT-None-04445"],
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdDrawMeshTasksIndirectCountEXT-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        if !buffer.usage().intersects(BufferUsage::INDIRECT_BUFFER) {
            return Err(Box::new(ValidationError {
                context: "buffer.usage()".into(),
                problem: "does not contain `BufferUsage::INDIRECT_BUFFER`".into(),
                vuids: &["VUID-vkCmdDrawMeshTasksIndirectCountEXT-buffer-02709"],
                ..Default::default()
            }));
        }

        if !offset.is_multiple_of(4) {
            return Err(Box::new(ValidationError {
                context: "offset".into(),
                problem: "is not a multiple of 4".into(),
                vuids: &["VUID-vkCmdDrawMeshTasksIndirectCountEXT-offset-02710"],
                ..Default::default()
            }));
        }

        if !count_buffer
            .usage()
            .intersects(BufferUsage::INDIRECT_BUFFER)
        {
            return Err(Box::new(ValidationError {
                context: "count_buffer.usage()".into(),
                problem: "does not contain `BufferUsage::INDIRECT_BUFFER`".into(),
                vuids: &["VUID-vkCmdDrawMeshTasksIndirectCountEXT-countBuffer-02715"],
                ..Default::default()
            }));
        }

        if !count_buffer_offset.is_multiple_of(4) {
            return Err(Box::new(ValidationError {
                context: "count_buffer_offset".into(),
                problem: "is not a multiple of 4".into(),
                vuids: &["VUID-vkCmdDrawMeshTasksIndirectCountEXT-countBufferOffset-02716"],
                ..Default::default()
            }));
        }

        if !stride.is_multiple_of(4) {
            return Err(Box::new(ValidationError {
                context: "stride".into(),
                problem: "is not a multiple of 4".into(),
                vuids: &["VUID-vkCmdDrawMeshTasksIndirectCountEXT-stride-07096"],
                ..Default::default()
            }));
        }

        if (stride as DeviceSize) < size_of::<DrawMeshTasksIndirectCommand>() as DeviceSize {
            return Err(Box::new(ValidationError {
                context: "stride".into(),
                problem: "is less than `size_of::<DrawMeshTasksIndirectCommand>()`".into(),
                vuids: &["VUID-vkCmdDrawMeshTasksIndirectCountEXT-stride-07096"],
                ..Default::default()
            }));
        }

        if max_draw_count >= 1
            && (stride as DeviceSize)
                .checked_mul(max_draw_count as DeviceSize - 1)
                .and_then(|x| x.checked_add(offset))
                .and_then(|x| {
                    x.checked_add(size_of::<DrawMeshTasksIndirectCommand>() as DeviceSize)
                })
                .is_none_or(|end| end > buffer.size())
        {
            return Err(Box::new(ValidationError {
                problem: "`max_draw_count` is equal to or greater than 1, but `stride * \
                    (max_draw_count - 1) + offset + size_of::<DrawMeshTasksIndirectCommand>()` is \
                    greater than `buffer.size()`"
                    .into(),
                vuids: &["VUID-vkCmdDrawMeshTasksIndirectCountEXT-maxDrawCount-07097"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn draw_mesh_tasks_indirect_count_unchecked(
        &mut self,
        buffer: &Buffer,
        offset: DeviceSize,
        count_buffer: &Buffer,
        count_buffer_offset: DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        let fns = self.device().fns();

        unsafe {
            (fns.ext_mesh_shader.cmd_draw_mesh_tasks_indirect_count_ext)(
                self.handle(),
                buffer.handle(),
                offset,
                count_buffer.handle(),
                count_buffer_offset,
                max_draw_count,
                stride,
            )
        };

        self
    }

    #[track_caller]
    pub unsafe fn trace_rays(
        &mut self,
        shader_binding_table_addresses: &ShaderBindingTableAddresses,
        dimensions: [u32; 3],
    ) -> &mut Self {
        unsafe { self.try_trace_rays(shader_binding_table_addresses, dimensions) }.unwrap()
    }

    pub unsafe fn try_trace_rays(
        &mut self,
        shader_binding_table_addresses: &ShaderBindingTableAddresses,
        dimensions: [u32; 3],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_trace_rays(shader_binding_table_addresses, dimensions)?;

        Ok(unsafe { self.trace_rays_unchecked(shader_binding_table_addresses, dimensions) })
    }

    pub(crate) fn validate_trace_rays(
        &self,
        _shader_binding_table_addresses: &ShaderBindingTableAddresses,
        dimensions: [u32; 3],
    ) -> Result<(), Box<ValidationError>> {
        if !self.device().enabled_features().ray_tracing_pipeline {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "ray_tracing_pipeline",
                )])]),
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    compute operations"
                    .into(),
                vuids: &["VUID-vkCmdTraceRaysKHR-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        let device_properties = self.device().physical_device().properties();

        let width = dimensions[0] as u64;
        let height = dimensions[1] as u64;
        let depth = dimensions[2] as u64;

        let max_width = device_properties.max_compute_work_group_count[0] as u64
            * device_properties.max_compute_work_group_size[0] as u64;

        if width > max_width {
            return Err(Box::new(ValidationError {
                context: "width".into(),
                problem: "exceeds `max_compute_work_group_count[0] * \
                    max_compute_work_group_size[0]`"
                    .into(),
                vuids: &["VUID-vkCmdTraceRaysKHR-width-03638"],
                ..Default::default()
            }));
        }

        let max_height = device_properties.max_compute_work_group_count[1] as u64
            * device_properties.max_compute_work_group_size[1] as u64;

        if height > max_height {
            return Err(Box::new(ValidationError {
                context: "height".into(),
                problem: "exceeds `max_compute_work_group_count[1] * \
                    max_compute_work_group_size[1]`"
                    .into(),
                vuids: &["VUID-vkCmdTraceRaysKHR-height-03639"],
                ..Default::default()
            }));
        }

        let max_depth = device_properties.max_compute_work_group_count[2] as u64
            * device_properties.max_compute_work_group_size[2] as u64;

        if depth > max_depth {
            return Err(Box::new(ValidationError {
                context: "depth".into(),
                problem: "exceeds `max_compute_work_group_count[2] * \
                    max_compute_work_group_size[2]`"
                    .into(),
                vuids: &["VUID-vkCmdTraceRaysKHR-depth-03640"],
                ..Default::default()
            }));
        }

        let total_invocations = width * height * depth;
        let max_invocations = device_properties.max_ray_dispatch_invocation_count.unwrap() as u64;

        if total_invocations > max_invocations {
            return Err(Box::new(ValidationError {
                context: "width * height * depth".into(),
                problem: "exceeds `max_ray_dispatch_invocation_count`".into(),
                vuids: &["VUID-vkCmdTraceRaysKHR-width-03641"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn trace_rays_unchecked(
        &mut self,
        shader_binding_table_addresses: &ShaderBindingTableAddresses,
        dimensions: [u32; 3],
    ) -> &mut Self {
        let raygen = shader_binding_table_addresses.raygen.to_vk();
        let miss = shader_binding_table_addresses.miss.to_vk();
        let hit = shader_binding_table_addresses.hit.to_vk();
        let callable = shader_binding_table_addresses.callable.to_vk();

        let fns = self.device().fns();

        unsafe {
            (fns.khr_ray_tracing_pipeline.cmd_trace_rays_khr)(
                self.handle(),
                &raygen,
                &miss,
                &hit,
                &callable,
                dimensions[0],
                dimensions[1],
                dimensions[2],
            )
        };

        self
    }

    #[track_caller]
    pub unsafe fn trace_rays_indirect(
        &mut self,
        shader_binding_table_addresses: &ShaderBindingTableAddresses,
        indirect_device_address: DeviceAddress,
    ) -> &mut Self {
        unsafe {
            self.try_trace_rays_indirect(shader_binding_table_addresses, indirect_device_address)
        }
        .unwrap()
    }

    pub unsafe fn try_trace_rays_indirect(
        &mut self,
        shader_binding_table_addresses: &ShaderBindingTableAddresses,
        indirect_device_address: DeviceAddress,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_trace_rays_indirect(shader_binding_table_addresses, indirect_device_address)?;

        Ok(unsafe {
            self.trace_rays_indirect_unchecked(
                shader_binding_table_addresses,
                indirect_device_address,
            )
        })
    }

    pub(crate) fn validate_trace_rays_indirect(
        &self,
        _shader_binding_table_addresses: &ShaderBindingTableAddresses,
        indirect_device_address: DeviceAddress,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .device()
            .enabled_features()
            .ray_tracing_pipeline_trace_rays_indirect
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "ray_tracing_pipeline_trace_rays_indirect",
                )])]),
                vuids: &[
                    "VUID-vkCmdTraceRaysIndirectKHR-rayTracingPipelineTraceRaysIndirect-03637",
                ],
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    compute operations"
                    .into(),
                vuids: &["VUID-vkCmdTraceRaysIndirectKHR-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        if !indirect_device_address.is_multiple_of(4) {
            return Err(Box::new(ValidationError {
                context: "indirect_device_address".into(),
                problem: "is not a multiple of 4".into(),
                vuids: &["VUID-vkCmdTraceRaysIndirectKHR-indirectDeviceAddress-03634"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn trace_rays_indirect_unchecked(
        &mut self,
        shader_binding_table_addresses: &ShaderBindingTableAddresses,
        indirect_device_address: DeviceAddress,
    ) -> &mut Self {
        let raygen = shader_binding_table_addresses.raygen.to_vk();
        let miss = shader_binding_table_addresses.miss.to_vk();
        let hit = shader_binding_table_addresses.hit.to_vk();
        let callable = shader_binding_table_addresses.callable.to_vk();

        let fns = self.device().fns();

        unsafe {
            (fns.khr_ray_tracing_pipeline.cmd_trace_rays_indirect_khr)(
                self.handle(),
                &raygen,
                &miss,
                &hit,
                &callable,
                indirect_device_address,
            )
        };

        self
    }
}
