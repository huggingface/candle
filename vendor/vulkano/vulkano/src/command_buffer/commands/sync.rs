use crate::{
    buffer::Buffer,
    command_buffer::sys::RecordingCommandBuffer,
    device::{Device, DeviceOwned, QueueFlags},
    image::{
        Image, ImageAspects, ImageCreateFlags, ImageLayout, ImageSubresourceRange, ImageUsage,
    },
    sync::{
        event::Event, AccessFlags, DependencyFlags, PipelineStages, QueueFamilyOwnershipTransfer,
    },
    DeviceSize, Requires, RequiresAllOf, RequiresOneOf, ValidationError, Version, VulkanObject,
};
use ash::vk;
use smallvec::SmallVec;

impl RecordingCommandBuffer {
    #[inline]
    #[track_caller]
    pub unsafe fn pipeline_barrier(&mut self, dependency_info: &DependencyInfo<'_>) -> &mut Self {
        unsafe { self.try_pipeline_barrier(dependency_info) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_pipeline_barrier(
        &mut self,
        dependency_info: &DependencyInfo<'_>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_pipeline_barrier(dependency_info)?;

        Ok(unsafe { self.pipeline_barrier_unchecked(dependency_info) })
    }

    fn validate_pipeline_barrier(
        &self,
        dependency_info: &DependencyInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        let queue_family_properties = self.queue_family_properties();

        if !queue_family_properties.queue_flags.intersects(
            QueueFlags::TRANSFER
                | QueueFlags::GRAPHICS
                | QueueFlags::COMPUTE
                | QueueFlags::VIDEO_DECODE
                | QueueFlags::VIDEO_ENCODE,
        ) {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    transfer, graphics, compute, video decode or video encode operations"
                    .into(),
                vuids: &["VUID-vkCmdPipelineBarrier2-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        dependency_info
            .validate(self.device())
            .map_err(|err| err.add_context("dependency_info"))?;

        let &DependencyInfo {
            dependency_flags: _,
            memory_barriers,
            buffer_memory_barriers,
            image_memory_barriers,
            _ne,
        } = dependency_info;

        let supported_pipeline_stages = PipelineStages::from(queue_family_properties.queue_flags);

        for (barrier_index, memory_barrier) in memory_barriers.iter().enumerate() {
            let &MemoryBarrier {
                src_stages,
                src_access: _,
                dst_stages,
                dst_access: _,
                _ne: _,
            } = memory_barrier;

            if !supported_pipeline_stages.contains(src_stages) {
                return Err(Box::new(ValidationError {
                    context: format!(
                        "dependency_info.memory_barriers[{}].src_stages",
                        barrier_index
                    )
                    .into(),
                    problem: "contains stages that are not supported by the queue family of the \
                        command buffer"
                        .into(),
                    vuids: &["VUID-vkCmdPipelineBarrier2-srcStageMask-03849"],
                    ..Default::default()
                }));
            }

            if !supported_pipeline_stages.contains(dst_stages) {
                return Err(Box::new(ValidationError {
                    context: format!(
                        "dependency_info.memory_barriers[{}].dst_stages",
                        barrier_index
                    )
                    .into(),
                    problem: "contains stages that are not supported by the queue family of the \
                        command buffer"
                        .into(),
                    vuids: &["VUID-vkCmdPipelineBarrier2-dstStageMask-03850"],
                    ..Default::default()
                }));
            }
        }

        for (barrier_index, buffer_memory_barrier) in buffer_memory_barriers.iter().enumerate() {
            let &BufferMemoryBarrier {
                src_stages,
                src_access: _,
                dst_stages,
                dst_access: _,
                queue_family_ownership_transfer: _,
                buffer: _,
                offset: _,
                size: _,
                _ne: _,
            } = buffer_memory_barrier;

            if !supported_pipeline_stages.contains(src_stages) {
                return Err(Box::new(ValidationError {
                    context: format!(
                        "dependency_info.buffer_memory_barriers[{}].src_stages",
                        barrier_index
                    )
                    .into(),
                    problem: "contains stages that are not supported by the queue family of the \
                        command buffer"
                        .into(),
                    vuids: &["VUID-vkCmdPipelineBarrier2-srcStageMask-03849"],
                    ..Default::default()
                }));
            }

            if !supported_pipeline_stages.contains(dst_stages) {
                return Err(Box::new(ValidationError {
                    context: format!(
                        "dependency_info.buffer_memory_barriers[{}].dst_stages",
                        barrier_index
                    )
                    .into(),
                    problem: "contains stages that are not supported by the queue family of the \
                        command buffer"
                        .into(),
                    vuids: &["VUID-vkCmdPipelineBarrier2-dstStageMask-03850"],
                    ..Default::default()
                }));
            }
        }

        for (barrier_index, image_memory_barrier) in image_memory_barriers.iter().enumerate() {
            let &ImageMemoryBarrier {
                src_stages,
                src_access: _,
                dst_stages,
                dst_access: _,
                old_layout: _,
                new_layout: _,
                queue_family_ownership_transfer: _,
                image: _,
                subresource_range: _,
                _ne: _,
            } = image_memory_barrier;

            if !supported_pipeline_stages.contains(src_stages) {
                return Err(Box::new(ValidationError {
                    context: format!(
                        "dependency_info.image_memory_barriers[{}].src_stages",
                        barrier_index
                    )
                    .into(),
                    problem: "contains stages that are not supported by the queue family of the \
                        command buffer"
                        .into(),
                    vuids: &["VUID-vkCmdPipelineBarrier2-srcStageMask-03849"],
                    ..Default::default()
                }));
            }

            if !supported_pipeline_stages.contains(dst_stages) {
                return Err(Box::new(ValidationError {
                    context: format!(
                        "dependency_info.image_memory_barriers[{}].dst_stages",
                        barrier_index
                    )
                    .into(),
                    problem: "contains stages that are not supported by the queue family of the \
                        command buffer"
                        .into(),
                    vuids: &["VUID-vkCmdPipelineBarrier2-dstStageMask-03850"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn pipeline_barrier_unchecked(
        &mut self,
        dependency_info: &DependencyInfo<'_>,
    ) -> &mut Self {
        if dependency_info.is_empty() {
            return self;
        }

        if self.device().enabled_features().synchronization2 {
            let DependencyInfo2Fields1Vk {
                memory_barriers_vk,
                buffer_memory_barriers_vk,
                image_memory_barriers_vk,
            } = dependency_info.to_vk2_fields1();
            let dependency_info_vk = dependency_info.to_vk2(
                &memory_barriers_vk,
                &buffer_memory_barriers_vk,
                &image_memory_barriers_vk,
            );

            let fns = self.device().fns();

            if self.device().api_version() >= Version::V1_3 {
                unsafe { (fns.v1_3.cmd_pipeline_barrier2)(self.handle(), &dependency_info_vk) };
            } else {
                unsafe {
                    (fns.khr_synchronization2.cmd_pipeline_barrier2_khr)(
                        self.handle(),
                        &dependency_info_vk,
                    )
                };
            }
        } else {
            let DependencyInfoFields1Vk {
                memory_barriers_vk,
                buffer_memory_barriers_vk,
                image_memory_barriers_vk,
                src_stage_mask_vk,
                dst_stage_mask_vk,
            } = dependency_info.to_vk_fields1();
            let dependency_flags_vk = dependency_info.to_vk_dependency_flags();

            let fns = self.device().fns();
            unsafe {
                (fns.v1_0.cmd_pipeline_barrier)(
                    self.handle(),
                    src_stage_mask_vk,
                    dst_stage_mask_vk,
                    dependency_flags_vk,
                    memory_barriers_vk.len() as u32,
                    memory_barriers_vk.as_ptr(),
                    buffer_memory_barriers_vk.len() as u32,
                    buffer_memory_barriers_vk.as_ptr(),
                    image_memory_barriers_vk.len() as u32,
                    image_memory_barriers_vk.as_ptr(),
                )
            };
        }

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn set_event(
        &mut self,
        event: &Event,
        dependency_info: &DependencyInfo<'_>,
    ) -> &mut Self {
        unsafe { self.try_set_event(event, dependency_info) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_set_event(
        &mut self,
        event: &Event,
        dependency_info: &DependencyInfo<'_>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_event(event, dependency_info)?;

        Ok(unsafe { self.set_event_unchecked(event, dependency_info) })
    }

    fn validate_set_event(
        &self,
        event: &Event,
        dependency_info: &DependencyInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        let queue_family_properties = self.queue_family_properties();

        if !queue_family_properties.queue_flags.intersects(
            QueueFlags::GRAPHICS
                | QueueFlags::COMPUTE
                | QueueFlags::VIDEO_DECODE
                | QueueFlags::VIDEO_ENCODE,
        ) {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics, compute, video decode or video encode operations"
                    .into(),
                vuids: &["VUID-vkCmdSetEvent2-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        // VUID-vkCmdSetEvent2-commonparent
        assert_eq!(self.device(), event.device());

        dependency_info
            .validate(self.device())
            .map_err(|err| err.add_context("dependency_info"))?;

        let &DependencyInfo {
            dependency_flags,
            memory_barriers,
            buffer_memory_barriers,
            image_memory_barriers,
            _ne,
        } = dependency_info;

        if !dependency_flags.is_empty() {
            return Err(Box::new(ValidationError {
                context: "dependency_info.dependency_flags".into(),
                problem: "is not empty".into(),
                vuids: &["VUID-vkCmdSetEvent2-dependencyFlags-03825"],
                ..Default::default()
            }));
        }

        let supported_pipeline_stages = PipelineStages::from(queue_family_properties.queue_flags);

        for (barrier_index, memory_barrier) in memory_barriers.iter().enumerate() {
            let &MemoryBarrier {
                src_stages,
                src_access: _,
                dst_stages,
                dst_access: _,
                _ne: _,
            } = memory_barrier;

            if !supported_pipeline_stages.contains(src_stages) {
                return Err(Box::new(ValidationError {
                    context: format!(
                        "dependency_info.memory_barriers[{}].src_stages",
                        barrier_index
                    )
                    .into(),
                    problem: "contains stages that are not supported by the queue family of the \
                        command buffer"
                        .into(),
                    vuids: &["VUID-vkCmdSetEvent2-srcStageMask-03827"],
                    ..Default::default()
                }));
            }

            if !supported_pipeline_stages.contains(dst_stages) {
                return Err(Box::new(ValidationError {
                    context: format!(
                        "dependency_info.memory_barriers[{}].dst_stages",
                        barrier_index
                    )
                    .into(),
                    problem: "contains stages that are not supported by the queue family of the \
                        command buffer"
                        .into(),
                    vuids: &["VUID-vkCmdSetEvent2-dstStageMask-03828"],
                    ..Default::default()
                }));
            }
        }

        for (barrier_index, buffer_memory_barrier) in buffer_memory_barriers.iter().enumerate() {
            let &BufferMemoryBarrier {
                src_stages,
                src_access: _,
                dst_stages,
                dst_access: _,
                queue_family_ownership_transfer: _,
                buffer: _,
                offset: _,
                size: _,
                _ne: _,
            } = buffer_memory_barrier;

            if !supported_pipeline_stages.contains(src_stages) {
                return Err(Box::new(ValidationError {
                    context: format!(
                        "dependency_info.buffer_memory_barriers[{}].src_stages",
                        barrier_index
                    )
                    .into(),
                    problem: "contains stages that are not supported by the queue family of the \
                        command buffer"
                        .into(),
                    vuids: &["VUID-vkCmdSetEvent2-srcStageMask-03827"],
                    ..Default::default()
                }));
            }

            if !supported_pipeline_stages.contains(dst_stages) {
                return Err(Box::new(ValidationError {
                    context: format!(
                        "dependency_info.buffer_memory_barriers[{}].dst_stages",
                        barrier_index
                    )
                    .into(),
                    problem: "contains stages that are not supported by the queue family of the \
                        command buffer"
                        .into(),
                    vuids: &["VUID-vkCmdSetEvent2-dstStageMask-03828"],
                    ..Default::default()
                }));
            }
        }

        for (barrier_index, image_memory_barrier) in image_memory_barriers.iter().enumerate() {
            let &ImageMemoryBarrier {
                src_stages,
                src_access: _,
                dst_stages,
                dst_access: _,
                old_layout: _,
                new_layout: _,
                queue_family_ownership_transfer: _,
                image: _,
                subresource_range: _,
                _ne: _,
            } = image_memory_barrier;

            if !supported_pipeline_stages.contains(src_stages) {
                return Err(Box::new(ValidationError {
                    context: format!(
                        "dependency_info.image_memory_barriers[{}].src_stages",
                        barrier_index
                    )
                    .into(),
                    problem: "contains stages that are not supported by the queue family of the \
                        command buffer"
                        .into(),
                    vuids: &["VUID-vkCmdSetEvent2-srcStageMask-03827"],
                    ..Default::default()
                }));
            }

            if !supported_pipeline_stages.contains(dst_stages) {
                return Err(Box::new(ValidationError {
                    context: format!(
                        "dependency_info.image_memory_barriers[{}].dst_stages",
                        barrier_index
                    )
                    .into(),
                    problem: "contains stages that are not supported by the queue family of the \
                        command buffer"
                        .into(),
                    vuids: &["VUID-vkCmdSetEvent2-dstStageMask-03828"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_event_unchecked(
        &mut self,
        event: &Event,
        dependency_info: &DependencyInfo<'_>,
    ) -> &mut Self {
        let fns = self.device().fns();

        if self.device().enabled_features().synchronization2 {
            let DependencyInfo2Fields1Vk {
                memory_barriers_vk,
                buffer_memory_barriers_vk,
                image_memory_barriers_vk,
            } = dependency_info.to_vk2_fields1();
            let dependency_info_vk = dependency_info.to_vk2(
                &memory_barriers_vk,
                &buffer_memory_barriers_vk,
                &image_memory_barriers_vk,
            );

            if self.device().api_version() >= Version::V1_3 {
                unsafe {
                    (fns.v1_3.cmd_set_event2)(self.handle(), event.handle(), &dependency_info_vk)
                };
            } else {
                unsafe {
                    (fns.khr_synchronization2.cmd_set_event2_khr)(
                        self.handle(),
                        event.handle(),
                        &dependency_info_vk,
                    )
                };
            }
        } else {
            // The original function only takes a source stage mask; the rest of the info is
            // provided with `wait_events` instead. Therefore, we condense the source stages
            // here and ignore the rest.
            let stage_mask_vk = dependency_info.to_vk_src_stage_mask();

            unsafe { (fns.v1_0.cmd_set_event)(self.handle(), event.handle(), stage_mask_vk) };
        }

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn wait_events(&mut self, events: &[(&Event, &DependencyInfo<'_>)]) -> &mut Self {
        unsafe { self.try_wait_events(events) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_wait_events(
        &mut self,
        events: &[(&Event, &DependencyInfo<'_>)],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_wait_events(events)?;

        Ok(unsafe { self.wait_events_unchecked(events) })
    }

    fn validate_wait_events(
        &self,
        events: &[(&Event, &DependencyInfo<'_>)],
    ) -> Result<(), Box<ValidationError>> {
        let queue_family_properties = self.queue_family_properties();

        if !queue_family_properties.queue_flags.intersects(
            QueueFlags::GRAPHICS
                | QueueFlags::COMPUTE
                | QueueFlags::VIDEO_DECODE
                | QueueFlags::VIDEO_ENCODE,
        ) {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics, compute, video decode or video encode operations"
                    .into(),
                vuids: &["VUID-vkCmdWaitEvents2-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        for (event_index, (event, dependency_info)) in events.iter().enumerate() {
            // VUID-vkCmdWaitEvents2-commandBuffer-cmdpool
            assert_eq!(self.device(), event.device());

            dependency_info
                .validate(self.device())
                .map_err(|err| err.add_context(format!("events[{}].1", event_index)))?;

            let &DependencyInfo {
                dependency_flags: _,
                memory_barriers,
                buffer_memory_barriers,
                image_memory_barriers,
                _ne,
            } = dependency_info;

            let supported_pipeline_stages =
                PipelineStages::from(queue_family_properties.queue_flags);

            for (barrier_index, memory_barrier) in memory_barriers.iter().enumerate() {
                let &MemoryBarrier {
                    src_stages,
                    src_access: _,
                    dst_stages,
                    dst_access: _,
                    _ne: _,
                } = memory_barrier;

                if !supported_pipeline_stages.contains(src_stages) {
                    return Err(Box::new(ValidationError {
                        context: format!(
                            "events[{}].1.memory_barriers[{}].src_stages",
                            event_index, barrier_index
                        )
                        .into(),
                        problem: "contains stages that are not supported by the queue family of \
                            the command buffer"
                            .into(),
                        vuids: &["VUID-vkCmdSetEvent2-srcStageMask-03827"],
                        ..Default::default()
                    }));
                }

                if !supported_pipeline_stages.contains(dst_stages) {
                    return Err(Box::new(ValidationError {
                        context: format!(
                            "events[{}].1.memory_barriers[{}].dst_stages",
                            event_index, barrier_index
                        )
                        .into(),
                        problem: "contains stages that are not supported by the queue family of \
                            the command buffer"
                            .into(),
                        vuids: &["VUID-vkCmdSetEvent2-dstStageMask-03828"],
                        ..Default::default()
                    }));
                }
            }

            for (barrier_index, buffer_memory_barrier) in buffer_memory_barriers.iter().enumerate()
            {
                let &BufferMemoryBarrier {
                    src_stages,
                    src_access: _,
                    dst_stages,
                    dst_access: _,
                    queue_family_ownership_transfer: _,
                    buffer: _,
                    offset: _,
                    size: _,
                    _ne: _,
                } = buffer_memory_barrier;

                if !supported_pipeline_stages.contains(src_stages) {
                    return Err(Box::new(ValidationError {
                        context: format!(
                            "events[{}].1.buffer_memory_barriers[{}].src_stages",
                            event_index, barrier_index
                        )
                        .into(),
                        problem: "contains stages that are not supported by the queue family of \
                            the command buffer"
                            .into(),
                        vuids: &["VUID-vkCmdSetEvent2-srcStageMask-03827"],
                        ..Default::default()
                    }));
                }

                if !supported_pipeline_stages.contains(dst_stages) {
                    return Err(Box::new(ValidationError {
                        context: format!(
                            "events[{}].1.buffer_memory_barriers[{}].dst_stages",
                            event_index, barrier_index
                        )
                        .into(),
                        problem: "contains stages that are not supported by the queue family of \
                            the command buffer"
                            .into(),
                        vuids: &["VUID-vkCmdSetEvent2-dstStageMask-03828"],
                        ..Default::default()
                    }));
                }
            }

            for (barrier_index, image_memory_barrier) in image_memory_barriers.iter().enumerate() {
                let &ImageMemoryBarrier {
                    src_stages,
                    src_access: _,
                    dst_stages,
                    dst_access: _,
                    old_layout: _,
                    new_layout: _,
                    queue_family_ownership_transfer: _,
                    image: _,
                    subresource_range: _,
                    _ne: _,
                } = image_memory_barrier;

                if !supported_pipeline_stages.contains(src_stages) {
                    return Err(Box::new(ValidationError {
                        context: format!(
                            "events[{}].1.image_memory_barriers[{}].src_stages",
                            event_index, barrier_index
                        )
                        .into(),
                        problem: "contains stages that are not supported by the queue family of \
                            the command buffer"
                            .into(),
                        vuids: &["VUID-vkCmdSetEvent2-srcStageMask-03827"],
                        ..Default::default()
                    }));
                }

                if !supported_pipeline_stages.contains(dst_stages) {
                    return Err(Box::new(ValidationError {
                        context: format!(
                            "events[{}].1.image_memory_barriers[{}].dst_stages",
                            event_index, barrier_index
                        )
                        .into(),
                        problem: "contains stages that are not supported by the queue family of \
                            the command buffer"
                            .into(),
                        vuids: &["VUID-vkCmdSetEvent2-dstStageMask-03828"],
                        ..Default::default()
                    }));
                }
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn wait_events_unchecked(
        &mut self,
        events: &[(&Event, &DependencyInfo<'_>)],
    ) -> &mut Self {
        let fns = self.device().fns();

        if self.device().enabled_features().synchronization2 {
            let events_vk: SmallVec<[_; 4]> =
                events.iter().map(|(event, _)| event.handle()).collect();
            let dependency_infos_fields1_vk: SmallVec<[_; 4]> = events
                .iter()
                .map(|(_, dependency_info)| dependency_info.to_vk2_fields1())
                .collect();
            let dependency_infos_vk: SmallVec<[_; 4]> = events
                .iter()
                .zip(&dependency_infos_fields1_vk)
                .map(|((_, dependency_info), dependency_info_fields1_vk)| {
                    let DependencyInfo2Fields1Vk {
                        memory_barriers_vk,
                        buffer_memory_barriers_vk,
                        image_memory_barriers_vk,
                    } = dependency_info_fields1_vk;

                    dependency_info.to_vk2(
                        memory_barriers_vk,
                        buffer_memory_barriers_vk,
                        image_memory_barriers_vk,
                    )
                })
                .collect();

            if self.device().api_version() >= Version::V1_3 {
                unsafe {
                    (fns.v1_3.cmd_wait_events2)(
                        self.handle(),
                        events_vk.len() as u32,
                        events_vk.as_ptr(),
                        dependency_infos_vk.as_ptr(),
                    )
                };
            } else {
                unsafe {
                    (fns.khr_synchronization2.cmd_wait_events2_khr)(
                        self.handle(),
                        events_vk.len() as u32,
                        events_vk.as_ptr(),
                        dependency_infos_vk.as_ptr(),
                    )
                };
            }
        } else {
            // With the original function, you can only specify a single dependency info for all
            // events at once, rather than separately for each event. Therefore, to achieve the
            // same behaviour as the "2" function, we split it up into multiple Vulkan API calls,
            // one per event.

            for (event, dependency_info) in events {
                let events_vk = [event.handle()];
                let DependencyInfoFields1Vk {
                    memory_barriers_vk,
                    buffer_memory_barriers_vk,
                    image_memory_barriers_vk,
                    src_stage_mask_vk,
                    dst_stage_mask_vk,
                } = dependency_info.to_vk_fields1();

                unsafe {
                    (fns.v1_0.cmd_wait_events)(
                        self.handle(),
                        1,
                        events_vk.as_ptr(),
                        src_stage_mask_vk,
                        dst_stage_mask_vk,
                        memory_barriers_vk.len() as u32,
                        memory_barriers_vk.as_ptr(),
                        buffer_memory_barriers_vk.len() as u32,
                        buffer_memory_barriers_vk.as_ptr(),
                        image_memory_barriers_vk.len() as u32,
                        image_memory_barriers_vk.as_ptr(),
                    )
                };
            }
        }

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn reset_event(&mut self, event: &Event, stages: PipelineStages) -> &mut Self {
        unsafe { self.try_reset_event(event, stages) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_reset_event(
        &mut self,
        event: &Event,
        stages: PipelineStages,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_reset_event(event, stages)?;

        Ok(unsafe { self.reset_event_unchecked(event, stages) })
    }

    fn validate_reset_event(
        &self,
        event: &Event,
        stages: PipelineStages,
    ) -> Result<(), Box<ValidationError>> {
        if !self.queue_family_properties().queue_flags.intersects(
            QueueFlags::GRAPHICS
                | QueueFlags::COMPUTE
                | QueueFlags::VIDEO_DECODE
                | QueueFlags::VIDEO_ENCODE,
        ) {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics, compute, video decode or video encode operations"
                    .into(),
                vuids: &["VUID-vkCmdResetEvent2-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        let device = self.device();

        // VUID-vkCmdResetEvent2-commonparent
        assert_eq!(device, event.device());

        stages.validate_device(device).map_err(|err| {
            err.add_context("stages")
                .set_vuids(&["VUID-vkCmdResetEvent2-stageMask-parameter"])
        })?;

        if !device.enabled_features().synchronization2 {
            if stages.contains_flags2() {
                return Err(Box::new(ValidationError {
                    context: "stages".into(),
                    problem: "contains flags from `VkPipelineStageFlagBits2`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "synchronization2",
                    )])]),
                    ..Default::default()
                }));
            }
        }

        if !device.enabled_features().geometry_shader {
            if stages.intersects(PipelineStages::GEOMETRY_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "stages".into(),
                    problem: "contains `PipelineStages::GEOMETRY_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "geometry_shader",
                    )])]),
                    vuids: &["VUID-vkCmdResetEvent2-stageMask-03929"],
                }));
            }
        }

        if !device.enabled_features().tessellation_shader {
            if stages.intersects(
                PipelineStages::TESSELLATION_CONTROL_SHADER
                    | PipelineStages::TESSELLATION_EVALUATION_SHADER,
            ) {
                return Err(Box::new(ValidationError {
                    context: "stages".into(),
                    problem: "contains `PipelineStages::TESSELLATION_CONTROL_SHADER` or \
                        `PipelineStages::TESSELLATION_EVALUATION_SHADER`"
                        .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "tessellation_shader",
                    )])]),
                    vuids: &["VUID-vkCmdResetEvent2-stageMask-03930"],
                }));
            }
        }

        if !device.enabled_features().conditional_rendering {
            if stages.intersects(PipelineStages::CONDITIONAL_RENDERING) {
                return Err(Box::new(ValidationError {
                    context: "stages".into(),
                    problem: "contains `PipelineStages::CONDITIONAL_RENDERING`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "conditional_rendering",
                    )])]),
                    vuids: &["VUID-vkCmdResetEvent2-stageMask-03931"],
                }));
            }
        }

        if !device.enabled_features().fragment_density_map {
            if stages.intersects(PipelineStages::FRAGMENT_DENSITY_PROCESS) {
                return Err(Box::new(ValidationError {
                    context: "stages".into(),
                    problem: "contains `PipelineStages::FRAGMENT_DENSITY_PROCESS`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "fragment_density_map",
                    )])]),
                    vuids: &["VUID-vkCmdResetEvent2-stageMask-03932"],
                }));
            }
        }

        if !device.enabled_features().transform_feedback {
            if stages.intersects(PipelineStages::TRANSFORM_FEEDBACK) {
                return Err(Box::new(ValidationError {
                    context: "stages".into(),
                    problem: "contains `PipelineStages::TRANSFORM_FEEDBACK`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "transform_feedback",
                    )])]),
                    vuids: &["VUID-vkCmdResetEvent2-stageMask-03933"],
                }));
            }
        }

        if !device.enabled_features().mesh_shader {
            if stages.intersects(PipelineStages::MESH_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "stages".into(),
                    problem: "contains `PipelineStages::MESH_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "mesh_shader",
                    )])]),
                    vuids: &["VUID-vkCmdResetEvent2-stageMask-03934"],
                }));
            }
        }

        if !device.enabled_features().task_shader {
            if stages.intersects(PipelineStages::TASK_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "stages".into(),
                    problem: "contains `PipelineStages::TASK_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "task_shader",
                    )])]),
                    vuids: &["VUID-vkCmdResetEvent2-stageMask-03935"],
                }));
            }
        }

        if !(device.enabled_features().attachment_fragment_shading_rate
            || device.enabled_features().shading_rate_image)
        {
            if stages.intersects(PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT) {
                return Err(Box::new(ValidationError {
                    context: "stages".into(),
                    problem: "contains `PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT`".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::DeviceFeature(
                            "attachment_fragment_shading_rate",
                        )]),
                        RequiresAllOf(&[Requires::DeviceFeature("shading_rate_image")]),
                    ]),
                    vuids: &["VUID-VkImageMemoryBarrier2-shadingRateImage-07316"],
                }));
            }
        }

        if !device.enabled_features().subpass_shading {
            if stages.intersects(PipelineStages::SUBPASS_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "stages".into(),
                    problem: "contains `PipelineStages::SUBPASS_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "subpass_shading",
                    )])]),
                    vuids: &["VUID-vkCmdResetEvent2-stageMask-04957"],
                }));
            }
        }

        if !device.enabled_features().invocation_mask {
            if stages.intersects(PipelineStages::INVOCATION_MASK) {
                return Err(Box::new(ValidationError {
                    context: "stages".into(),
                    problem: "contains `PipelineStages::INVOCATION_MASK`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "invocation_mask",
                    )])]),
                    vuids: &["VUID-vkCmdResetEvent2-stageMask-04995"],
                }));
            }
        }

        if !(device.enabled_extensions().nv_ray_tracing
            || device.enabled_features().ray_tracing_pipeline)
        {
            if stages.intersects(PipelineStages::RAY_TRACING_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "stages".into(),
                    problem: "contains `PipelineStages::RAY_TRACING_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "ray_tracing_pipeline",
                    )])]),
                    vuids: &["VUID-vkCmdResetEvent2-stageMask-07946"],
                }));
            }
        }

        if stages.intersects(PipelineStages::HOST) {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "contains `PipelineStages::HOST`".into(),
                vuids: &["VUID-vkCmdResetEvent2-stageMask-03830"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn reset_event_unchecked(
        &mut self,
        event: &Event,
        stages: PipelineStages,
    ) -> &mut Self {
        let fns = self.device().fns();

        if self.device().enabled_features().synchronization2 {
            if self.device().api_version() >= Version::V1_3 {
                unsafe {
                    (fns.v1_3.cmd_reset_event2)(self.handle(), event.handle(), stages.into())
                };
            } else {
                unsafe {
                    (fns.khr_synchronization2.cmd_reset_event2_khr)(
                        self.handle(),
                        event.handle(),
                        stages.into(),
                    )
                };
            }
        } else {
            unsafe { (fns.v1_0.cmd_reset_event)(self.handle(), event.handle(), stages.into()) };
        }

        self
    }
}

/// Dependency info for barriers in a pipeline barrier or event command.
///
/// A pipeline barrier creates a dependency between commands submitted before the barrier (the
/// source scope) and commands submitted after it (the destination scope). An event command acts
/// like a split pipeline barrier: the source scope and destination scope are defined
/// relative to different commands. Each `DependencyInfo` consists of multiple individual barriers
/// that concern a either single resource or operate globally.
///
/// Each barrier has a set of source/destination pipeline stages and source/destination memory
/// access types. The pipeline stages create an *execution dependency*: the `src_stages` of
/// commands submitted before the barrier must be completely finished before before any of the
/// `dst_stages` of commands after the barrier are allowed to start. The memory access types
/// create a *memory dependency*: in addition to the execution dependency, any `src_access`
/// performed before the barrier must be made available and visible before any `dst_access`
/// are made after the barrier.
#[derive(Clone, Debug)]
pub struct DependencyInfo<'a> {
    /// Flags to modify how the execution and memory dependencies are formed.
    ///
    /// The default value is empty.
    pub dependency_flags: DependencyFlags,

    /// Memory barriers for global operations and accesses, not limited to a single resource.
    ///
    /// The default value is empty.
    pub memory_barriers: &'a [MemoryBarrier<'a>],

    /// Memory barriers for individual buffers.
    ///
    /// The default value is empty.
    pub buffer_memory_barriers: &'a [BufferMemoryBarrier<'a>],

    /// Memory barriers for individual images.
    ///
    /// The default value is empty.
    pub image_memory_barriers: &'a [ImageMemoryBarrier<'a>],

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for DependencyInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl DependencyInfo<'_> {
    /// Returns a default `DependencyInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            dependency_flags: DependencyFlags::empty(),
            memory_barriers: &[],
            buffer_memory_barriers: &[],
            image_memory_barriers: &[],
            _ne: crate::NE,
        }
    }

    /// Returns whether `self` contains any barriers.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.memory_barriers.is_empty()
            && self.buffer_memory_barriers.is_empty()
            && self.image_memory_barriers.is_empty()
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            dependency_flags,
            memory_barriers,
            buffer_memory_barriers,
            image_memory_barriers,
            _ne: _,
        } = self;

        dependency_flags.validate_device(device).map_err(|err| {
            err.add_context("dependency_flags")
                .set_vuids(&["VUID-VkDependencyInfo-dependencyFlags-parameter"])
        })?;

        for (barrier_index, memory_barrier) in memory_barriers.iter().enumerate() {
            memory_barrier
                .validate(device)
                .map_err(|err| err.add_context(format!("memory_barriers[{}]", barrier_index)))?;
        }

        for (barrier_index, buffer_memory_barrier) in buffer_memory_barriers.iter().enumerate() {
            buffer_memory_barrier.validate(device).map_err(|err| {
                err.add_context(format!("buffer_memory_barriers[{}]", barrier_index))
            })?;
        }

        for (barrier_index, image_memory_barrier) in image_memory_barriers.iter().enumerate() {
            image_memory_barrier.validate(device).map_err(|err| {
                err.add_context(format!("image_memory_barriers[{}]", barrier_index))
            })?;
        }

        Ok(())
    }

    pub(crate) fn to_vk2<'a>(
        &self,
        memory_barriers_vk: &'a [vk::MemoryBarrier2<'_>],
        buffer_memory_barriers_vk: &'a [vk::BufferMemoryBarrier2<'_>],
        image_memory_barriers_vk: &'a [vk::ImageMemoryBarrier2<'_>],
    ) -> vk::DependencyInfo<'a> {
        let &Self {
            dependency_flags,
            memory_barriers: _,
            buffer_memory_barriers: _,
            image_memory_barriers: _,
            _ne: _,
        } = self;

        vk::DependencyInfo::default()
            .dependency_flags(dependency_flags.into())
            .memory_barriers(memory_barriers_vk)
            .buffer_memory_barriers(buffer_memory_barriers_vk)
            .image_memory_barriers(image_memory_barriers_vk)
    }

    pub(crate) fn to_vk2_fields1(&self) -> DependencyInfo2Fields1Vk {
        let &Self {
            dependency_flags: _,
            memory_barriers,
            buffer_memory_barriers,
            image_memory_barriers,
            _ne: _,
        } = self;

        let memory_barriers_vk = memory_barriers.iter().map(MemoryBarrier::to_vk2).collect();
        let buffer_memory_barriers_vk = buffer_memory_barriers
            .iter()
            .map(BufferMemoryBarrier::to_vk2)
            .collect();
        let image_memory_barriers_vk = image_memory_barriers
            .iter()
            .map(ImageMemoryBarrier::to_vk2)
            .collect();

        DependencyInfo2Fields1Vk {
            memory_barriers_vk,
            buffer_memory_barriers_vk,
            image_memory_barriers_vk,
        }
    }

    pub(crate) fn to_vk_dependency_flags(&self) -> vk::DependencyFlags {
        self.dependency_flags.into()
    }

    pub(crate) fn to_vk_fields1(&self) -> DependencyInfoFields1Vk {
        let &Self {
            dependency_flags: _,
            memory_barriers,
            buffer_memory_barriers,
            image_memory_barriers,
            _ne: _,
        } = self;

        let mut src_stage_mask_vk = vk::PipelineStageFlags::empty();
        let mut dst_stage_mask_vk = vk::PipelineStageFlags::empty();

        let memory_barriers_vk = memory_barriers
            .iter()
            .inspect(|barrier| {
                src_stage_mask_vk |= barrier.src_stages.into();
                dst_stage_mask_vk |= barrier.dst_stages.into();
            })
            .map(MemoryBarrier::to_vk)
            .collect();
        let buffer_memory_barriers_vk = buffer_memory_barriers
            .iter()
            .inspect(|barrier| {
                src_stage_mask_vk |= barrier.src_stages.into();
                dst_stage_mask_vk |= barrier.dst_stages.into();
            })
            .map(BufferMemoryBarrier::to_vk)
            .collect();
        let image_memory_barriers_vk = image_memory_barriers
            .iter()
            .inspect(|barrier| {
                src_stage_mask_vk |= barrier.src_stages.into();
                dst_stage_mask_vk |= barrier.dst_stages.into();
            })
            .map(ImageMemoryBarrier::to_vk)
            .collect();

        if src_stage_mask_vk.is_empty() {
            // "VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT is [...] equivalent to
            // VK_PIPELINE_STAGE_2_NONE in the first scope."
            src_stage_mask_vk |= vk::PipelineStageFlags::TOP_OF_PIPE;
        }

        if dst_stage_mask_vk.is_empty() {
            // "VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT is [...] equivalent to
            // VK_PIPELINE_STAGE_2_NONE in the second scope."
            dst_stage_mask_vk |= vk::PipelineStageFlags::BOTTOM_OF_PIPE;
        }

        DependencyInfoFields1Vk {
            memory_barriers_vk,
            buffer_memory_barriers_vk,
            image_memory_barriers_vk,
            src_stage_mask_vk,
            dst_stage_mask_vk,
        }
    }

    pub(crate) fn to_vk_src_stage_mask(&self) -> vk::PipelineStageFlags {
        let &Self {
            dependency_flags: _,
            memory_barriers,
            buffer_memory_barriers,
            image_memory_barriers,
            _ne: _,
        } = self;

        let mut src_stage_mask_vk = vk::PipelineStageFlags::empty();

        for barrier in memory_barriers {
            src_stage_mask_vk |= barrier.src_stages.into();
        }

        for barrier in buffer_memory_barriers {
            src_stage_mask_vk |= barrier.src_stages.into();
        }

        for barrier in image_memory_barriers {
            src_stage_mask_vk |= barrier.src_stages.into();
        }

        if src_stage_mask_vk.is_empty() {
            // "VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT is [...] equivalent to
            // VK_PIPELINE_STAGE_2_NONE in the first scope."
            src_stage_mask_vk |= vk::PipelineStageFlags::TOP_OF_PIPE;
        }

        src_stage_mask_vk
    }
}

pub(crate) struct DependencyInfo2Fields1Vk {
    pub(crate) memory_barriers_vk: SmallVec<[vk::MemoryBarrier2<'static>; 2]>,
    pub(crate) buffer_memory_barriers_vk: SmallVec<[vk::BufferMemoryBarrier2<'static>; 8]>,
    pub(crate) image_memory_barriers_vk: SmallVec<[vk::ImageMemoryBarrier2<'static>; 8]>,
}

pub(crate) struct DependencyInfoFields1Vk {
    pub(crate) memory_barriers_vk: SmallVec<[vk::MemoryBarrier<'static>; 2]>,
    pub(crate) buffer_memory_barriers_vk: SmallVec<[vk::BufferMemoryBarrier<'static>; 8]>,
    pub(crate) image_memory_barriers_vk: SmallVec<[vk::ImageMemoryBarrier<'static>; 8]>,
    pub(crate) src_stage_mask_vk: vk::PipelineStageFlags,
    pub(crate) dst_stage_mask_vk: vk::PipelineStageFlags,
}

/// A memory barrier that is applied globally.
#[derive(Clone, Debug)]
pub struct MemoryBarrier<'a> {
    /// The pipeline stages in the source scope to wait for.
    ///
    /// The default value is [`PipelineStages::empty()`].
    pub src_stages: PipelineStages,

    /// The memory accesses in the source scope to make available and visible.
    ///
    /// The default value is [`AccessFlags::empty()`].
    pub src_access: AccessFlags,

    /// The pipeline stages in the destination scope that must wait for `src_stages`.
    ///
    /// The default value is [`PipelineStages::empty()`].
    pub dst_stages: PipelineStages,

    /// The memory accesses in the destination scope that must wait for `src_access` to be made
    /// available and visible.
    pub dst_access: AccessFlags,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for MemoryBarrier<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryBarrier<'_> {
    /// Returns a default `MemoryBarrier`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            src_stages: PipelineStages::empty(),
            src_access: AccessFlags::empty(),
            dst_stages: PipelineStages::empty(),
            dst_access: AccessFlags::empty(),
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            src_stages,
            src_access,
            dst_stages,
            dst_access,
            _ne: _,
        } = self;

        src_stages.validate_device(device).map_err(|err| {
            err.add_context("src_stages")
                .set_vuids(&["VUID-VkMemoryBarrier2-srcStageMask-parameter"])
        })?;

        dst_stages.validate_device(device).map_err(|err| {
            err.add_context("dst_stages")
                .set_vuids(&["VUID-VkMemoryBarrier2-dstStageMask-parameter"])
        })?;

        src_access.validate_device(device).map_err(|err| {
            err.add_context("src_access")
                .set_vuids(&["VUID-VkMemoryBarrier2-srcAccessMask-parameter"])
        })?;

        dst_access.validate_device(device).map_err(|err| {
            err.add_context("dst_access")
                .set_vuids(&["VUID-VkMemoryBarrier2-dstAccessMask-parameter"])
        })?;

        if !device.enabled_features().synchronization2 {
            if src_stages.contains_flags2() {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains flags from `VkPipelineStageFlagBits2`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "synchronization2",
                    )])]),
                    ..Default::default()
                }));
            }

            if dst_stages.contains_flags2() {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains flags from `VkPipelineStageFlagBits2`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "synchronization2",
                    )])]),
                    ..Default::default()
                }));
            }

            if src_access.contains_flags2() {
                return Err(Box::new(ValidationError {
                    context: "src_access".into(),
                    problem: "contains flags from `VkAccessFlagBits2`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "synchronization2",
                    )])]),
                    ..Default::default()
                }));
            }

            if dst_access.contains_flags2() {
                return Err(Box::new(ValidationError {
                    context: "dst_access".into(),
                    problem: "contains flags from `VkAccessFlagBits2`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "synchronization2",
                    )])]),
                    ..Default::default()
                }));
            }
        }

        if !device.enabled_features().geometry_shader {
            if src_stages.intersects(PipelineStages::GEOMETRY_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::GEOMETRY_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "geometry_shader",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-srcStageMask-03929"],
                }));
            }

            if dst_stages.intersects(PipelineStages::GEOMETRY_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::GEOMETRY_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "geometry_shader",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-dstStageMask-03929"],
                }));
            }
        }

        if !device.enabled_features().tessellation_shader {
            if src_stages.intersects(
                PipelineStages::TESSELLATION_CONTROL_SHADER
                    | PipelineStages::TESSELLATION_EVALUATION_SHADER,
            ) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::TESSELLATION_CONTROL_SHADER` or \
                        `PipelineStages::TESSELLATION_EVALUATION_SHADER`"
                        .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "tessellation_shader",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-srcStageMask-03930"],
                }));
            }

            if dst_stages.intersects(
                PipelineStages::TESSELLATION_CONTROL_SHADER
                    | PipelineStages::TESSELLATION_EVALUATION_SHADER,
            ) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::TESSELLATION_CONTROL_SHADER` or \
                        `PipelineStages::TESSELLATION_EVALUATION_SHADER`"
                        .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "tessellation_shader",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-dstStageMask-03930"],
                }));
            }
        }

        if !device.enabled_features().conditional_rendering {
            if src_stages.intersects(PipelineStages::CONDITIONAL_RENDERING) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::CONDITIONAL_RENDERING`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "conditional_rendering",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-srcStageMask-03931"],
                }));
            }

            if dst_stages.intersects(PipelineStages::CONDITIONAL_RENDERING) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::CONDITIONAL_RENDERING`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "conditional_rendering",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-dstStageMask-03931"],
                }));
            }
        }

        if !device.enabled_features().fragment_density_map {
            if src_stages.intersects(PipelineStages::FRAGMENT_DENSITY_PROCESS) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::FRAGMENT_DENSITY_PROCESS`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "fragment_density_map",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-srcStageMask-03932"],
                }));
            }

            if dst_stages.intersects(PipelineStages::FRAGMENT_DENSITY_PROCESS) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::FRAGMENT_DENSITY_PROCESS`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "fragment_density_map",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-dstStageMask-03932"],
                }));
            }
        }

        if !device.enabled_features().transform_feedback {
            if src_stages.intersects(PipelineStages::TRANSFORM_FEEDBACK) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::TRANSFORM_FEEDBACK`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "transform_feedback",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-srcStageMask-03933"],
                }));
            }

            if dst_stages.intersects(PipelineStages::TRANSFORM_FEEDBACK) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::TRANSFORM_FEEDBACK`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "transform_feedback",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-dstStageMask-03933"],
                }));
            }
        }

        if !device.enabled_features().mesh_shader {
            if src_stages.intersects(PipelineStages::MESH_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::MESH_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "mesh_shader",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-srcStageMask-03934"],
                }));
            }

            if dst_stages.intersects(PipelineStages::MESH_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::MESH_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "mesh_shader",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-dstStageMask-03934"],
                }));
            }
        }

        if !device.enabled_features().task_shader {
            if src_stages.intersects(PipelineStages::TASK_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::TASK_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "task_shader",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-srcStageMask-03935"],
                }));
            }

            if dst_stages.intersects(PipelineStages::TASK_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::TASK_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "task_shader",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-dstStageMask-03935"],
                }));
            }
        }

        if !(device.enabled_features().attachment_fragment_shading_rate
            || device.enabled_features().shading_rate_image)
        {
            if src_stages.intersects(PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT`".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::DeviceFeature(
                            "attachment_fragment_shading_rate",
                        )]),
                        RequiresAllOf(&[Requires::DeviceFeature("shading_rate_image")]),
                    ]),
                    vuids: &["VUID-VkMemoryBarrier2-shadingRateImage-07316"],
                }));
            }

            if dst_stages.intersects(PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT`".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::DeviceFeature(
                            "attachment_fragment_shading_rate",
                        )]),
                        RequiresAllOf(&[Requires::DeviceFeature("shading_rate_image")]),
                    ]),
                    vuids: &["VUID-VkMemoryBarrier2-shadingRateImage-07316"],
                }));
            }
        }

        if !device.enabled_features().subpass_shading {
            if src_stages.intersects(PipelineStages::SUBPASS_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::SUBPASS_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "subpass_shading",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-srcStageMask-04957"],
                }));
            }

            if dst_stages.intersects(PipelineStages::SUBPASS_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::SUBPASS_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "subpass_shading",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-dstStageMask-04957"],
                }));
            }
        }

        if !device.enabled_features().invocation_mask {
            if src_stages.intersects(PipelineStages::INVOCATION_MASK) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::INVOCATION_MASK`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "invocation_mask",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-srcStageMask-04995"],
                }));
            }

            if dst_stages.intersects(PipelineStages::INVOCATION_MASK) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::INVOCATION_MASK`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "invocation_mask",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-dstStageMask-04995"],
                }));
            }
        }

        if !(device.enabled_extensions().nv_ray_tracing
            || device.enabled_features().ray_tracing_pipeline)
        {
            if src_stages.intersects(PipelineStages::RAY_TRACING_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::RAY_TRACING_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "ray_tracing_pipeline",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-srcStageMask-07946"],
                }));
            }

            if dst_stages.intersects(PipelineStages::RAY_TRACING_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::RAY_TRACING_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "ray_tracing_pipeline",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-dstStageMask-07946"],
                }));
            }
        }

        if !AccessFlags::from(src_stages).contains(src_access) {
            return Err(Box::new(ValidationError {
                problem: "`src_access` contains one or more access types that are not performed \
                    by any stage in `src_stages`"
                    .into(),
                vuids: &[
                    "VUID-VkMemoryBarrier2-srcAccessMask-03900",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03901",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03902",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03903",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03904",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03905",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03906",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03907",
                    "VUID-VkMemoryBarrier2-srcAccessMask-07454",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03909",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03910",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03911",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03912",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03913",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03914",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03915",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03916",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03917",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03918",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03919",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03920",
                    "VUID-VkMemoryBarrier2-srcAccessMask-04747",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03922",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03923",
                    "VUID-VkMemoryBarrier2-srcAccessMask-04994",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03924",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03925",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03926",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03927",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03928",
                    "VUID-VkMemoryBarrier2-srcAccessMask-06256",
                    "VUID-VkMemoryBarrier2-srcAccessMask-07272",
                    "VUID-VkMemoryBarrier2-srcAccessMask-04858",
                    "VUID-VkMemoryBarrier2-srcAccessMask-04859",
                    "VUID-VkMemoryBarrier2-srcAccessMask-04860",
                    "VUID-VkMemoryBarrier2-srcAccessMask-04861",
                    "VUID-VkMemoryBarrier2-srcAccessMask-07455",
                    "VUID-VkMemoryBarrier2-srcAccessMask-07456",
                    "VUID-VkMemoryBarrier2-srcAccessMask-07457",
                    "VUID-VkMemoryBarrier2-srcAccessMask-07458",
                    "VUID-VkMemoryBarrier2-srcAccessMask-08118",
                ],
                ..Default::default()
            }));
        }

        if !AccessFlags::from(dst_stages).contains(dst_access) {
            return Err(Box::new(ValidationError {
                problem: "`dst_access` contains one or more access types that are not performed \
                    by any stage in `dst_stages`"
                    .into(),
                vuids: &[
                    "VUID-VkMemoryBarrier2-dstAccessMask-03900",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03901",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03902",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03903",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03904",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03905",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03906",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03907",
                    "VUID-VkMemoryBarrier2-dstAccessMask-07454",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03909",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03910",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03911",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03912",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03913",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03914",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03915",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03916",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03917",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03918",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03919",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03920",
                    "VUID-VkMemoryBarrier2-dstAccessMask-04747",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03922",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03923",
                    "VUID-VkMemoryBarrier2-dstAccessMask-04994",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03924",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03925",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03926",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03927",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03928",
                    "VUID-VkMemoryBarrier2-dstAccessMask-06256",
                    "VUID-VkMemoryBarrier2-dstAccessMask-07272",
                    "VUID-VkMemoryBarrier2-dstAccessMask-04858",
                    "VUID-VkMemoryBarrier2-dstAccessMask-04859",
                    "VUID-VkMemoryBarrier2-dstAccessMask-04860",
                    "VUID-VkMemoryBarrier2-dstAccessMask-04861",
                    "VUID-VkMemoryBarrier2-dstAccessMask-07455",
                    "VUID-VkMemoryBarrier2-dstAccessMask-07456",
                    "VUID-VkMemoryBarrier2-dstAccessMask-07457",
                    "VUID-VkMemoryBarrier2-dstAccessMask-07458",
                    "VUID-VkMemoryBarrier2-dstAccessMask-08118",
                ],
                ..Default::default()
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk2(&self) -> vk::MemoryBarrier2<'static> {
        let &Self {
            src_stages,
            src_access,
            dst_stages,
            dst_access,
            _ne: _,
        } = self;

        vk::MemoryBarrier2::default()
            .src_stage_mask(src_stages.into())
            .src_access_mask(src_access.into())
            .dst_stage_mask(dst_stages.into())
            .dst_access_mask(dst_access.into())
    }

    pub(crate) fn to_vk(&self) -> vk::MemoryBarrier<'static> {
        let &Self {
            src_stages: _,
            src_access,
            dst_stages: _,
            dst_access,
            _ne: _,
        } = self;

        vk::MemoryBarrier::default()
            .src_access_mask(src_access.into())
            .dst_access_mask(dst_access.into())
    }
}

/// A memory barrier that is applied to a single buffer.
#[derive(Clone, Debug)]
pub struct BufferMemoryBarrier<'a> {
    /// The pipeline stages in the source scope to wait for.
    ///
    /// The default value is [`PipelineStages::empty()`].
    pub src_stages: PipelineStages,

    /// The memory accesses in the source scope to make available and visible.
    ///
    /// The default value is [`AccessFlags::empty()`].
    pub src_access: AccessFlags,

    /// The pipeline stages in the destination scope that must wait for `src_stages`.
    ///
    /// The default value is [`PipelineStages::empty()`].
    pub dst_stages: PipelineStages,

    /// The memory accesses in the destination scope that must wait for `src_access` to be made
    /// available and visible.
    pub dst_access: AccessFlags,

    /// For resources created with [`Sharing::Exclusive`](crate::sync::Sharing), transfers
    /// ownership of a resource from one queue family to another.
    pub queue_family_ownership_transfer: Option<QueueFamilyOwnershipTransfer>,

    /// The buffer to apply the barrier to.
    pub buffer: &'a Buffer,

    /// The byte offset from `buffer` to apply the barrier to.
    pub offset: DeviceSize,

    /// The byte size to apply the barrier to.
    ///
    /// If set to `None`, the barrier applies until the end of the buffer.
    pub size: Option<DeviceSize>,

    pub _ne: crate::NonExhaustive<'a>,
}

impl<'a> BufferMemoryBarrier<'a> {
    /// Returns a default `BufferMemoryBarrier` with the provided `buffer`.
    #[inline]
    pub const fn new(buffer: &'a Buffer) -> Self {
        Self {
            src_stages: PipelineStages::empty(),
            src_access: AccessFlags::empty(),
            dst_stages: PipelineStages::empty(),
            dst_access: AccessFlags::empty(),
            queue_family_ownership_transfer: None,
            buffer,
            offset: 0,
            size: None,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            src_stages,
            src_access,
            dst_stages,
            dst_access,
            queue_family_ownership_transfer,
            buffer,
            offset,
            size,
            _ne,
        } = self;

        src_stages.validate_device(device).map_err(|err| {
            err.add_context("src_stages")
                .set_vuids(&["VUID-VkBufferMemoryBarrier2-srcStageMask-parameter"])
        })?;

        dst_stages.validate_device(device).map_err(|err| {
            err.add_context("dst_stages")
                .set_vuids(&["VUID-VkBufferMemoryBarrier2-dstStageMask-parameter"])
        })?;

        src_access.validate_device(device).map_err(|err| {
            err.add_context("src_access")
                .set_vuids(&["VUID-VkBufferMemoryBarrier2-srcAccessMask-parameter"])
        })?;

        dst_access.validate_device(device).map_err(|err| {
            err.add_context("dst_access")
                .set_vuids(&["VUID-VkBufferMemoryBarrier2-dstAccessMask-parameter"])
        })?;

        if !device.enabled_features().synchronization2 {
            if src_stages.contains_flags2() {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains flags from `VkPipelineStageFlagBits2`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "synchronization2",
                    )])]),
                    ..Default::default()
                }));
            }

            if dst_stages.contains_flags2() {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains flags from `VkPipelineStageFlagBits2`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "synchronization2",
                    )])]),
                    ..Default::default()
                }));
            }

            if src_access.contains_flags2() {
                return Err(Box::new(ValidationError {
                    context: "src_access".into(),
                    problem: "contains flags from `VkAccessFlagBits2`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "synchronization2",
                    )])]),
                    ..Default::default()
                }));
            }

            if dst_access.contains_flags2() {
                return Err(Box::new(ValidationError {
                    context: "dst_access".into(),
                    problem: "contains flags from `VkAccessFlagBits2`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "synchronization2",
                    )])]),
                    ..Default::default()
                }));
            }
        }

        if !device.enabled_features().geometry_shader {
            if src_stages.intersects(PipelineStages::GEOMETRY_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::GEOMETRY_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "geometry_shader",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-srcStageMask-03929"],
                }));
            }

            if dst_stages.intersects(PipelineStages::GEOMETRY_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::GEOMETRY_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "geometry_shader",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-dstStageMask-03929"],
                }));
            }
        }

        if !device.enabled_features().tessellation_shader {
            if src_stages.intersects(
                PipelineStages::TESSELLATION_CONTROL_SHADER
                    | PipelineStages::TESSELLATION_EVALUATION_SHADER,
            ) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::TESSELLATION_CONTROL_SHADER` or \
                        `PipelineStages::TESSELLATION_EVALUATION_SHADER`"
                        .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "tessellation_shader",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-srcStageMask-03930"],
                }));
            }

            if dst_stages.intersects(
                PipelineStages::TESSELLATION_CONTROL_SHADER
                    | PipelineStages::TESSELLATION_EVALUATION_SHADER,
            ) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::TESSELLATION_CONTROL_SHADER` or \
                        `PipelineStages::TESSELLATION_EVALUATION_SHADER`"
                        .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "tessellation_shader",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-dstStageMask-03930"],
                }));
            }
        }

        if !device.enabled_features().conditional_rendering {
            if src_stages.intersects(PipelineStages::CONDITIONAL_RENDERING) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::CONDITIONAL_RENDERING`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "conditional_rendering",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-srcStageMask-03931"],
                }));
            }

            if dst_stages.intersects(PipelineStages::CONDITIONAL_RENDERING) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::CONDITIONAL_RENDERING`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "conditional_rendering",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-dstStageMask-03931"],
                }));
            }
        }

        if !device.enabled_features().fragment_density_map {
            if src_stages.intersects(PipelineStages::FRAGMENT_DENSITY_PROCESS) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::FRAGMENT_DENSITY_PROCESS`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "fragment_density_map",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-srcStageMask-03932"],
                }));
            }

            if dst_stages.intersects(PipelineStages::FRAGMENT_DENSITY_PROCESS) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::FRAGMENT_DENSITY_PROCESS`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "fragment_density_map",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-dstStageMask-03932"],
                }));
            }
        }

        if !device.enabled_features().transform_feedback {
            if src_stages.intersects(PipelineStages::TRANSFORM_FEEDBACK) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::TRANSFORM_FEEDBACK`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "transform_feedback",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-srcStageMask-03933"],
                }));
            }

            if dst_stages.intersects(PipelineStages::TRANSFORM_FEEDBACK) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::TRANSFORM_FEEDBACK`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "transform_feedback",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-dstStageMask-03933"],
                }));
            }
        }

        if !device.enabled_features().mesh_shader {
            if src_stages.intersects(PipelineStages::MESH_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::MESH_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "mesh_shader",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-srcStageMask-03934"],
                }));
            }

            if dst_stages.intersects(PipelineStages::MESH_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::MESH_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "mesh_shader",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-dstStageMask-03934"],
                }));
            }
        }

        if !device.enabled_features().task_shader {
            if src_stages.intersects(PipelineStages::TASK_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::TASK_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "task_shader",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-srcStageMask-03935"],
                }));
            }

            if dst_stages.intersects(PipelineStages::TASK_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::TASK_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "task_shader",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-dstStageMask-03935"],
                }));
            }
        }

        if !(device.enabled_features().attachment_fragment_shading_rate
            || device.enabled_features().shading_rate_image)
        {
            if src_stages.intersects(PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT`".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::DeviceFeature(
                            "attachment_fragment_shading_rate",
                        )]),
                        RequiresAllOf(&[Requires::DeviceFeature("shading_rate_image")]),
                    ]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-shadingRateImage-07316"],
                }));
            }

            if dst_stages.intersects(PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT`".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::DeviceFeature(
                            "attachment_fragment_shading_rate",
                        )]),
                        RequiresAllOf(&[Requires::DeviceFeature("shading_rate_image")]),
                    ]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-shadingRateImage-07316"],
                }));
            }
        }

        if !device.enabled_features().subpass_shading {
            if src_stages.intersects(PipelineStages::SUBPASS_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::SUBPASS_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "subpass_shading",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-srcStageMask-04957"],
                }));
            }

            if dst_stages.intersects(PipelineStages::SUBPASS_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::SUBPASS_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "subpass_shading",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-dstStageMask-04957"],
                }));
            }
        }

        if !device.enabled_features().invocation_mask {
            if src_stages.intersects(PipelineStages::INVOCATION_MASK) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::INVOCATION_MASK`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "invocation_mask",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-srcStageMask-04995"],
                }));
            }

            if dst_stages.intersects(PipelineStages::INVOCATION_MASK) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::INVOCATION_MASK`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "invocation_mask",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-dstStageMask-04995"],
                }));
            }
        }

        if !(device.enabled_extensions().nv_ray_tracing
            || device.enabled_features().ray_tracing_pipeline)
        {
            if src_stages.intersects(PipelineStages::RAY_TRACING_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::RAY_TRACING_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "ray_tracing_pipeline",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-srcStageMask-07946"],
                }));
            }

            if dst_stages.intersects(PipelineStages::RAY_TRACING_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::RAY_TRACING_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "ray_tracing_pipeline",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-dstStageMask-07946"],
                }));
            }
        }

        if !AccessFlags::from(src_stages).contains(src_access) {
            return Err(Box::new(ValidationError {
                problem: "`src_access` contains one or more access types that are not performed \
                    by any stage in `src_stages`"
                    .into(),
                vuids: &[
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03900",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03901",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03902",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03903",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03904",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03905",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03906",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03907",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-07454",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03909",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03910",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03911",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03912",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03913",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03914",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03915",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03916",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03917",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03918",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03919",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03920",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-04747",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03922",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03923",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-04994",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03924",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03925",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03926",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03927",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03928",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-06256",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-07272",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-04858",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-04859",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-04860",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-04861",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-07455",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-07456",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-07457",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-07458",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-08118",
                ],
                ..Default::default()
            }));
        }

        if !AccessFlags::from(dst_stages).contains(dst_access) {
            return Err(Box::new(ValidationError {
                problem: "`dst_access` contains one or more access types that are not performed \
                    by any stage in `dst_stages`"
                    .into(),
                vuids: &[
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03900",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03901",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03902",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03903",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03904",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03905",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03906",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03907",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-07454",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03909",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03910",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03911",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03912",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03913",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03914",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03915",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03916",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03917",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03918",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03919",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03920",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-04747",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03922",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03923",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-04994",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03924",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03925",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03926",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03927",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03928",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-06256",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-07272",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-04858",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-04859",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-04860",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-04861",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-07455",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-07456",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-07457",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-07458",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-08118",
                ],
                ..Default::default()
            }));
        }

        if size == Some(0) {
            return Err(Box::new(ValidationError {
                context: "size".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkBufferMemoryBarrier2-size-01188"],
                ..Default::default()
            }));
        }

        if offset >= buffer.size() {
            return Err(Box::new(ValidationError {
                problem: "`offset` is not less than `buffer.size()`".into(),
                vuids: &["VUID-VkBufferMemoryBarrier2-offset-01187"],
                ..Default::default()
            }));
        }

        if let Some(size) = size {
            if size > buffer.size() - offset {
                return Err(Box::new(ValidationError {
                    problem: "`offset + size` is greater than `buffer.size()`".into(),
                    vuids: &["VUID-VkBufferMemoryBarrier2-size-01189"],
                    ..Default::default()
                }));
            }
        }

        if let Some(queue_family_ownership_transfer) = queue_family_ownership_transfer {
            if src_stages.intersects(PipelineStages::HOST) {
                return Err(Box::new(ValidationError {
                    problem: "`src_stages` contains `PipelineStages::HOST`, but \
                        `queue_family_ownership_transfer` is `Some`"
                        .into(),
                    vuids: &["VUID-VkBufferMemoryBarrier2-srcStageMask-03851"],
                    ..Default::default()
                }));
            }

            if dst_stages.intersects(PipelineStages::HOST) {
                return Err(Box::new(ValidationError {
                    problem: "`dst_stages` contains `PipelineStages::HOST`, but \
                        `queue_family_ownership_transfer` is `Some`"
                        .into(),
                    vuids: &["VUID-VkBufferMemoryBarrier2-srcStageMask-03851"],
                    ..Default::default()
                }));
            }

            // VUID-VkBufferMemoryBarrier2-srcQueueFamilyIndex-04087
            // VUID-VkBufferMemoryBarrier2-buffer-04088
            // Ensured by the use of an enum.

            let queue_family_count =
                device.physical_device().queue_family_properties().len() as u32;

            match queue_family_ownership_transfer {
                QueueFamilyOwnershipTransfer::ExclusiveBetweenLocal {
                    src_index,
                    dst_index,
                } => {
                    if src_index >= queue_family_count {
                        return Err(Box::new(ValidationError {
                            context: "queue_family_ownership_transfer.src_index".into(),
                            problem: "is not less than the number of queue families in the \
                                physical device"
                                .into(),
                            vuids: &["VUID-VkBufferMemoryBarrier2-buffer-04089"],
                            ..Default::default()
                        }));
                    }

                    if dst_index >= queue_family_count {
                        return Err(Box::new(ValidationError {
                            context: "queue_family_ownership_transfer.dst_index".into(),
                            problem: "is not less than the number of queue families in the \
                                physical device"
                                .into(),
                            vuids: &["VUID-VkBufferMemoryBarrier2-buffer-04089"],
                            ..Default::default()
                        }));
                    }

                    if src_index == dst_index {
                        return Err(Box::new(ValidationError {
                            problem: "`queue_family_ownership_transfer.src_index` is equal to \
                                `queue_family_ownership_transfer.dst_index`"
                                .into(),
                            // just for sanity, so that the value is `Some`
                            // if and only if there is a transfer
                            ..Default::default()
                        }));
                    }
                }
                QueueFamilyOwnershipTransfer::ExclusiveToExternal { src_index } => {
                    if src_index >= queue_family_count {
                        return Err(Box::new(ValidationError {
                            context: "queue_family_ownership_transfer.src_index".into(),
                            problem: "is not less than the number of queue families in the \
                                physical device"
                                .into(),
                            vuids: &["VUID-VkBufferMemoryBarrier2-buffer-04089"],
                            ..Default::default()
                        }));
                    }
                }
                QueueFamilyOwnershipTransfer::ExclusiveFromExternal { dst_index } => {
                    if dst_index >= queue_family_count {
                        return Err(Box::new(ValidationError {
                            context: "queue_family_ownership_transfer.dst_index".into(),
                            problem: "is not less than the number of queue families in the \
                                physical device"
                                .into(),
                            vuids: &["VUID-VkBufferMemoryBarrier2-buffer-04089"],
                            ..Default::default()
                        }));
                    }
                }
                QueueFamilyOwnershipTransfer::ExclusiveToForeign { src_index } => {
                    if src_index >= queue_family_count {
                        return Err(Box::new(ValidationError {
                            context: "queue_family_ownership_transfer.src_index".into(),
                            problem: "is not less than the number of queue families in the \
                                physical device"
                                .into(),
                            vuids: &["VUID-VkBufferMemoryBarrier2-buffer-04089"],
                            ..Default::default()
                        }));
                    }
                }
                QueueFamilyOwnershipTransfer::ExclusiveFromForeign { dst_index } => {
                    if dst_index >= queue_family_count {
                        return Err(Box::new(ValidationError {
                            context: "queue_family_ownership_transfer.dst_index".into(),
                            problem: "is not less than the number of queue families in the \
                                physical device"
                                .into(),
                            vuids: &["VUID-VkBufferMemoryBarrier2-buffer-04089"],
                            ..Default::default()
                        }));
                    }
                }
                QueueFamilyOwnershipTransfer::ConcurrentToExternal
                | QueueFamilyOwnershipTransfer::ConcurrentFromExternal
                | QueueFamilyOwnershipTransfer::ConcurrentToForeign
                | QueueFamilyOwnershipTransfer::ConcurrentFromForeign => (),
            }
        }

        Ok(())
    }

    pub(crate) fn to_vk2(&self) -> vk::BufferMemoryBarrier2<'static> {
        let &Self {
            src_stages,
            src_access,
            dst_stages,
            dst_access,
            queue_family_ownership_transfer,
            buffer,
            offset,
            size,
            _ne: _,
        } = self;

        let (src_queue_family_index, dst_queue_family_index) =
            queue_family_ownership_transfer.as_ref().map_or(
                (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED),
                QueueFamilyOwnershipTransfer::to_vk,
            );

        vk::BufferMemoryBarrier2::default()
            .src_stage_mask(src_stages.into())
            .src_access_mask(src_access.into())
            .dst_stage_mask(dst_stages.into())
            .dst_access_mask(dst_access.into())
            .src_queue_family_index(src_queue_family_index)
            .dst_queue_family_index(dst_queue_family_index)
            .buffer(buffer.handle())
            .offset(offset)
            .size(size.unwrap_or(vk::WHOLE_SIZE))
    }

    pub(crate) fn to_vk(&self) -> vk::BufferMemoryBarrier<'static> {
        let &Self {
            src_stages: _,
            src_access,
            dst_stages: _,
            dst_access,
            queue_family_ownership_transfer,
            buffer,
            offset,
            size,
            _ne: _,
        } = self;

        let (src_queue_family_index, dst_queue_family_index) =
            queue_family_ownership_transfer.as_ref().map_or(
                (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED),
                QueueFamilyOwnershipTransfer::to_vk,
            );

        vk::BufferMemoryBarrier::default()
            .src_access_mask(src_access.into())
            .dst_access_mask(dst_access.into())
            .src_queue_family_index(src_queue_family_index)
            .dst_queue_family_index(dst_queue_family_index)
            .buffer(buffer.handle())
            .offset(offset)
            .size(size.unwrap_or(vk::WHOLE_SIZE))
    }
}

/// A memory barrier that is applied to a single image.
#[derive(Clone, Debug)]
pub struct ImageMemoryBarrier<'a> {
    /// The pipeline stages in the source scope to wait for.
    ///
    /// The default value is [`PipelineStages::empty()`].
    pub src_stages: PipelineStages,

    /// The memory accesses in the source scope to make available and visible.
    ///
    /// The default value is [`AccessFlags::empty()`].
    pub src_access: AccessFlags,

    /// The pipeline stages in the destination scope that must wait for `src_stages`.
    ///
    /// The default value is [`PipelineStages::empty()`].
    pub dst_stages: PipelineStages,

    /// The memory accesses in the destination scope that must wait for `src_access` to be made
    /// available and visible.
    ///
    /// The default value is [`AccessFlags::empty()`].
    pub dst_access: AccessFlags,

    /// The layout that the specified `subresource_range` of `image` is expected to be in when the
    /// source scope completes.
    ///
    /// The default value is [`ImageLayout::Undefined`].
    pub old_layout: ImageLayout,

    /// The layout that the specified `subresource_range` of `image` will be transitioned to before
    /// the destination scope begins.
    ///
    /// The default value is [`ImageLayout::Undefined`], which must be overridden.
    pub new_layout: ImageLayout,

    /// For resources created with [`Sharing::Exclusive`](crate::sync::Sharing), transfers
    /// ownership of a resource from one queue family to another.
    ///
    /// The default value is `None`.
    pub queue_family_ownership_transfer: Option<QueueFamilyOwnershipTransfer>,

    /// The image to apply the barrier to.
    ///
    /// There is no default value.
    pub image: &'a Image,

    /// The subresource range of `image` to apply the barrier to.
    ///
    /// The default value is [`ImageSubresourceRange::default()`].
    pub subresource_range: ImageSubresourceRange,

    pub _ne: crate::NonExhaustive<'a>,
}

impl<'a> ImageMemoryBarrier<'a> {
    /// Returns a default `ImageMemoryBarrier` with the provided `image`.
    #[inline]
    pub const fn new(image: &'a Image) -> Self {
        Self {
            src_stages: PipelineStages::empty(),
            src_access: AccessFlags::empty(),
            dst_stages: PipelineStages::empty(),
            dst_access: AccessFlags::empty(),
            old_layout: ImageLayout::Undefined,
            new_layout: ImageLayout::Undefined,
            queue_family_ownership_transfer: None,
            image,
            subresource_range: ImageSubresourceRange::new(),
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            src_stages,
            src_access,
            dst_stages,
            dst_access,
            old_layout,
            new_layout,
            queue_family_ownership_transfer,
            image,
            subresource_range,
            _ne,
        } = self;

        src_stages.validate_device(device).map_err(|err| {
            err.add_context("src_stages")
                .set_vuids(&["VUID-VkImageMemoryBarrier2-srcStageMask-parameter"])
        })?;

        dst_stages.validate_device(device).map_err(|err| {
            err.add_context("dst_stages")
                .set_vuids(&["VUID-VkImageMemoryBarrier2-dstStageMask-parameter"])
        })?;

        src_access.validate_device(device).map_err(|err| {
            err.add_context("src_access")
                .set_vuids(&["VUID-VkImageMemoryBarrier2-srcAccessMask-parameter"])
        })?;

        dst_access.validate_device(device).map_err(|err| {
            err.add_context("dst_access")
                .set_vuids(&["VUID-VkImageMemoryBarrier2-dstAccessMask-parameter"])
        })?;

        if !device.enabled_features().synchronization2 {
            if src_stages.contains_flags2() {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains flags from `VkPipelineStageFlagBits2`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "synchronization2",
                    )])]),
                    ..Default::default()
                }));
            }

            if dst_stages.contains_flags2() {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains flags from `VkPipelineStageFlagBits2`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "synchronization2",
                    )])]),
                    ..Default::default()
                }));
            }

            if src_access.contains_flags2() {
                return Err(Box::new(ValidationError {
                    context: "src_access".into(),
                    problem: "contains flags from `VkAccessFlagBits2`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "synchronization2",
                    )])]),
                    ..Default::default()
                }));
            }

            if dst_access.contains_flags2() {
                return Err(Box::new(ValidationError {
                    context: "dst_access".into(),
                    problem: "contains flags from `VkAccessFlagBits2`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "synchronization2",
                    )])]),
                    ..Default::default()
                }));
            }
        }

        if !device.enabled_features().geometry_shader {
            if src_stages.intersects(PipelineStages::GEOMETRY_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::GEOMETRY_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "geometry_shader",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-srcStageMask-03929"],
                }));
            }

            if dst_stages.intersects(PipelineStages::GEOMETRY_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::GEOMETRY_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "geometry_shader",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-dstStageMask-03929"],
                }));
            }
        }

        if !device.enabled_features().tessellation_shader {
            if src_stages.intersects(
                PipelineStages::TESSELLATION_CONTROL_SHADER
                    | PipelineStages::TESSELLATION_EVALUATION_SHADER,
            ) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::TESSELLATION_CONTROL_SHADER` or \
                        `PipelineStages::TESSELLATION_EVALUATION_SHADER`"
                        .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "tessellation_shader",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-srcStageMask-03930"],
                }));
            }

            if dst_stages.intersects(
                PipelineStages::TESSELLATION_CONTROL_SHADER
                    | PipelineStages::TESSELLATION_EVALUATION_SHADER,
            ) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::TESSELLATION_CONTROL_SHADER` or \
                        `PipelineStages::TESSELLATION_EVALUATION_SHADER`"
                        .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "tessellation_shader",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-dstStageMask-03930"],
                }));
            }
        }

        if !device.enabled_features().conditional_rendering {
            if src_stages.intersects(PipelineStages::CONDITIONAL_RENDERING) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::CONDITIONAL_RENDERING`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "conditional_rendering",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-srcStageMask-03931"],
                }));
            }

            if dst_stages.intersects(PipelineStages::CONDITIONAL_RENDERING) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::CONDITIONAL_RENDERING`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "conditional_rendering",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-dstStageMask-03931"],
                }));
            }
        }

        if !device.enabled_features().fragment_density_map {
            if src_stages.intersects(PipelineStages::FRAGMENT_DENSITY_PROCESS) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::FRAGMENT_DENSITY_PROCESS`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "fragment_density_map",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-srcStageMask-03932"],
                }));
            }

            if dst_stages.intersects(PipelineStages::FRAGMENT_DENSITY_PROCESS) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::FRAGMENT_DENSITY_PROCESS`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "fragment_density_map",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-dstStageMask-03932"],
                }));
            }
        }

        if !device.enabled_features().transform_feedback {
            if src_stages.intersects(PipelineStages::TRANSFORM_FEEDBACK) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::TRANSFORM_FEEDBACK`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "transform_feedback",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-srcStageMask-03933"],
                }));
            }

            if dst_stages.intersects(PipelineStages::TRANSFORM_FEEDBACK) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::TRANSFORM_FEEDBACK`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "transform_feedback",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-dstStageMask-03933"],
                }));
            }
        }

        if !device.enabled_features().mesh_shader {
            if src_stages.intersects(PipelineStages::MESH_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::MESH_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "mesh_shader",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-srcStageMask-03934"],
                }));
            }

            if dst_stages.intersects(PipelineStages::MESH_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::MESH_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "mesh_shader",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-dstStageMask-03934"],
                }));
            }
        }

        if !device.enabled_features().task_shader {
            if src_stages.intersects(PipelineStages::TASK_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::TASK_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "task_shader",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-srcStageMask-03935"],
                }));
            }

            if dst_stages.intersects(PipelineStages::TASK_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::TASK_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "task_shader",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-dstStageMask-03935"],
                }));
            }
        }

        if !(device.enabled_features().attachment_fragment_shading_rate
            || device.enabled_features().shading_rate_image)
        {
            if src_stages.intersects(PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT`".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::DeviceFeature(
                            "attachment_fragment_shading_rate",
                        )]),
                        RequiresAllOf(&[Requires::DeviceFeature("shading_rate_image")]),
                    ]),
                    vuids: &["VUID-VkImageMemoryBarrier2-shadingRateImage-07316"],
                }));
            }

            if dst_stages.intersects(PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT`".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::DeviceFeature(
                            "attachment_fragment_shading_rate",
                        )]),
                        RequiresAllOf(&[Requires::DeviceFeature("shading_rate_image")]),
                    ]),
                    vuids: &["VUID-VkImageMemoryBarrier2-shadingRateImage-07316"],
                }));
            }
        }

        if !device.enabled_features().subpass_shading {
            if src_stages.intersects(PipelineStages::SUBPASS_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::SUBPASS_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "subpass_shading",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-srcStageMask-04957"],
                }));
            }

            if dst_stages.intersects(PipelineStages::SUBPASS_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::SUBPASS_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "subpass_shading",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-dstStageMask-04957"],
                }));
            }
        }

        if !device.enabled_features().invocation_mask {
            if src_stages.intersects(PipelineStages::INVOCATION_MASK) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::INVOCATION_MASK`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "invocation_mask",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-srcStageMask-04995"],
                }));
            }

            if dst_stages.intersects(PipelineStages::INVOCATION_MASK) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::INVOCATION_MASK`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "invocation_mask",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-dstStageMask-04995"],
                }));
            }
        }

        if !(device.enabled_extensions().nv_ray_tracing
            || device.enabled_features().ray_tracing_pipeline)
        {
            if src_stages.intersects(PipelineStages::RAY_TRACING_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::RAY_TRACING_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "ray_tracing_pipeline",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-srcStageMask-07946"],
                }));
            }

            if dst_stages.intersects(PipelineStages::RAY_TRACING_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::RAY_TRACING_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "ray_tracing_pipeline",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-dstStageMask-07946"],
                }));
            }
        }

        if !AccessFlags::from(src_stages).contains(src_access) {
            return Err(Box::new(ValidationError {
                problem: "`src_access` contains one or more access types that are not performed \
                    by any stage in `src_stages`"
                    .into(),
                vuids: &[
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03900",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03901",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03902",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03903",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03904",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03905",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03906",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03907",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-07454",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03909",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03910",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03911",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03912",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03913",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03914",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03915",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03916",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03917",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03918",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03919",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03920",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-04747",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03922",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03923",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-04994",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03924",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03925",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03926",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03927",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03928",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-06256",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-07272",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-04858",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-04859",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-04860",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-04861",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-07455",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-07456",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-07457",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-07458",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-08118",
                ],
                ..Default::default()
            }));
        }

        if !AccessFlags::from(dst_stages).contains(dst_access) {
            return Err(Box::new(ValidationError {
                problem: "`dst_access` contains one or more access types that are not performed \
                    by any stage in `dst_stages`"
                    .into(),
                vuids: &[
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03900",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03901",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03902",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03903",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03904",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03905",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03906",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03907",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-07454",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03909",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03910",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03911",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03912",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03913",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03914",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03915",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03916",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03917",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03918",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03919",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03920",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-04747",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03922",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03923",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-04994",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03924",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03925",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03926",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03927",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03928",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-06256",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-07272",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-04858",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-04859",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-04860",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-04861",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-07455",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-07456",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-07457",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-07458",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-08118",
                ],
                ..Default::default()
            }));
        }

        // VUID-VkImageMemoryBarrier2-synchronization2-07793
        // If the synchronization2 feature is not enabled, oldLayout must not be
        // VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR or VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL_KHR

        // VUID-VkImageMemoryBarrier2-synchronization2-07794
        // If the synchronization2 feature is not enabled, newLayout must not be
        // VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR or VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL_KHR

        // VUID-VkImageMemoryBarrier2-attachmentFeedbackLoopLayout-07313
        // If the attachmentFeedbackLoopLayout feature is not enabled, newLayout must not be
        // VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT

        subresource_range
            .validate(device)
            .map_err(|err| err.add_context("subresource_range"))?;

        if subresource_range.base_mip_level >= image.mip_levels() {
            return Err(Box::new(ValidationError {
                problem: "`subresource_range.base_mip_level` is not less than `image.mip_levels()`"
                    .into(),
                vuids: &["VUID-VkImageMemoryBarrier2-subresourceRange-01486"],
                ..Default::default()
            }));
        }

        if let Some(subresource_range_level_count) = subresource_range.level_count {
            if subresource_range_level_count > image.mip_levels() - subresource_range.base_mip_level
            {
                return Err(Box::new(ValidationError {
                    problem: "`subresource_range.base_mip_level + subresource_range.level_count` \
                        is greater than `image.mip_levels()`"
                        .into(),
                    vuids: &["VUID-VkImageMemoryBarrier2-subresourceRange-01724"],
                    ..Default::default()
                }));
            }
        }

        if subresource_range.base_array_layer >= image.array_layers() {
            return Err(Box::new(ValidationError {
                problem: "`subresource_range.base_array_layer` is not less than \
                    `image.array_layers()`"
                    .into(),
                vuids: &["VUID-VkImageMemoryBarrier2-subresourceRange-01488"],
                ..Default::default()
            }));
        }

        if let Some(subresource_range_layer_count) = subresource_range.layer_count {
            if subresource_range_layer_count
                > image.array_layers() - subresource_range.base_array_layer
            {
                return Err(Box::new(ValidationError {
                    problem: "`subresource_range.base_array_layer + \
                        subresource_range.layer_count` is greater than `image.array_layers()`"
                        .into(),
                    vuids: &["VUID-VkImageMemoryBarrier2-subresourceRange-01725"],
                    ..Default::default()
                }));
            }
        }

        let image_format_aspects = image.format().aspects();

        if !image_format_aspects.contains(subresource_range.aspects) {
            return Err(Box::new(ValidationError {
                problem: "`subresource_range.aspects` is not a subset of \
                    `image.format().aspects()`"
                    .into(),
                vuids: &[
                    "VUID-VkImageMemoryBarrier2-image-01672",
                    "VUID-VkImageMemoryBarrier2-image-03319",
                ],
                ..Default::default()
            }));
        }

        if image_format_aspects.intersects(ImageAspects::COLOR)
            && !image.flags().intersects(ImageCreateFlags::DISJOINT)
            && subresource_range.aspects != ImageAspects::COLOR
        {
            return Err(Box::new(ValidationError {
                problem: "`image.format()` is a color format, and \
                    `image.flags()` does not contain `ImageCreateFlags::DISJOINT`, but \
                    `subresource_range.aspects` is not `ImageAspects::COLOR`"
                    .into(),
                vuids: &["VUID-VkImageMemoryBarrier2-image-01671"],
                ..Default::default()
            }));
        }

        if image_format_aspects.contains(ImageAspects::DEPTH | ImageAspects::STENCIL)
            && !subresource_range
                .aspects
                .contains(ImageAspects::DEPTH | ImageAspects::STENCIL)
            && !device.enabled_features().separate_depth_stencil_layouts
        {
            return Err(Box::new(ValidationError {
                problem: "`image.format()` has both a depth and a stencil component, and \
                    `subresource_range.aspects` does not contain both \
                    `ImageAspects::DEPTH` and `ImageAspects::STENCIL`"
                    .into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "separate_depth_stencil_layouts",
                )])]),
                vuids: &["VUID-VkImageMemoryBarrier2-image-03320"],
                ..Default::default()
            }));
        }

        if subresource_range.aspects.intersects(ImageAspects::DEPTH) {
            if matches!(
                old_layout,
                ImageLayout::StencilAttachmentOptimal | ImageLayout::StencilReadOnlyOptimal
            ) {
                return Err(Box::new(ValidationError {
                    problem: "`subresource_range.aspects` contains `ImageAspects::DEPTH`, but \
                        `old_layout` is `ImageLayout::StencilAttachmentOptimal` or \
                        `ImageLayout::StencilReadOnlyOptimal`"
                        .into(),
                    vuids: &["VUID-VkImageMemoryBarrier2-aspectMask-08702"],
                    ..Default::default()
                }));
            }

            if matches!(
                new_layout,
                ImageLayout::StencilAttachmentOptimal | ImageLayout::StencilReadOnlyOptimal
            ) {
                return Err(Box::new(ValidationError {
                    problem: "`subresource_range.aspects` contains `ImageAspects::DEPTH`, but \
                        `new_layout` is `ImageLayout::StencilAttachmentOptimal` or \
                        `ImageLayout::StencilReadOnlyOptimal`"
                        .into(),
                    vuids: &["VUID-VkImageMemoryBarrier2-aspectMask-08702"],
                    ..Default::default()
                }));
            }
        }

        if subresource_range.aspects.intersects(ImageAspects::STENCIL) {
            if matches!(
                old_layout,
                ImageLayout::StencilAttachmentOptimal | ImageLayout::StencilReadOnlyOptimal
            ) {
                return Err(Box::new(ValidationError {
                    problem: "`subresource_range.aspects` contains `ImageAspects::STENCIL`, but \
                        `old_layout` is `ImageLayout::DepthAttachmentOptimal` or \
                        `ImageLayout::DepthReadOnlyOptimal`"
                        .into(),
                    vuids: &["VUID-VkImageMemoryBarrier2-aspectMask-08703"],
                    ..Default::default()
                }));
            }

            if matches!(
                new_layout,
                ImageLayout::DepthAttachmentOptimal | ImageLayout::DepthReadOnlyOptimal
            ) {
                return Err(Box::new(ValidationError {
                    problem: "`subresource_range.aspects` contains `ImageAspects::STENCIL`, but \
                        `new_layout` is `ImageLayout::DepthAttachmentOptimal` or \
                        `ImageLayout::DepthReadOnlyOptimal`"
                        .into(),
                    vuids: &["VUID-VkImageMemoryBarrier2-aspectMask-08703"],
                    ..Default::default()
                }));
            }
        }

        if let Some(queue_family_ownership_transfer) = queue_family_ownership_transfer {
            if src_stages.intersects(PipelineStages::HOST) {
                return Err(Box::new(ValidationError {
                    problem: "`src_stages` contains `PipelineStages::HOST`, but \
                        `queue_family_ownership_transfer` is `Some`"
                        .into(),
                    vuids: &["VUID-VkImageMemoryBarrier2-srcStageMask-03854"],
                    ..Default::default()
                }));
            }

            if dst_stages.intersects(PipelineStages::HOST) {
                return Err(Box::new(ValidationError {
                    problem: "`dst_stages` contains `PipelineStages::HOST`, but \
                        `queue_family_ownership_transfer` is `Some`"
                        .into(),
                    vuids: &["VUID-VkImageMemoryBarrier2-srcStageMask-03854"],
                    ..Default::default()
                }));
            }

            // VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04070
            // VUID-VkImageMemoryBarrier2-image-04071
            // Ensured by the use of an enum.

            let queue_family_count =
                device.physical_device().queue_family_properties().len() as u32;

            match queue_family_ownership_transfer {
                QueueFamilyOwnershipTransfer::ExclusiveBetweenLocal {
                    src_index,
                    dst_index,
                } => {
                    if src_index >= queue_family_count {
                        return Err(Box::new(ValidationError {
                            context: "queue_family_ownership_transfer.src_index".into(),
                            problem: "is not less than the number of queue families in the \
                                physical device"
                                .into(),
                            vuids: &["VUID-VkImageMemoryBarrier2-image-04072"],
                            ..Default::default()
                        }));
                    }

                    if dst_index >= queue_family_count {
                        return Err(Box::new(ValidationError {
                            context: "queue_family_ownership_transfer.dst_index".into(),
                            problem: "is not less than the number of queue families in the \
                                physical device"
                                .into(),
                            vuids: &["VUID-VkImageMemoryBarrier2-image-04072"],
                            ..Default::default()
                        }));
                    }

                    if src_index == dst_index {
                        return Err(Box::new(ValidationError {
                            problem: "`queue_family_ownership_transfer.src_index` is equal to \
                                `queue_family_ownership_transfer.dst_index`"
                                .into(),
                            // just for sanity, so that the value is `Some`
                            // if and only if there is a transfer
                            ..Default::default()
                        }));
                    }
                }
                QueueFamilyOwnershipTransfer::ExclusiveToExternal { src_index } => {
                    if src_index >= queue_family_count {
                        return Err(Box::new(ValidationError {
                            context: "queue_family_ownership_transfer.src_index".into(),
                            problem: "is not less than the number of queue families in the \
                                physical device"
                                .into(),
                            vuids: &["VUID-VkImageMemoryBarrier2-image-04072"],
                            ..Default::default()
                        }));
                    }
                }
                QueueFamilyOwnershipTransfer::ExclusiveFromExternal { dst_index } => {
                    if dst_index >= queue_family_count {
                        return Err(Box::new(ValidationError {
                            context: "queue_family_ownership_transfer.dst_index".into(),
                            problem: "is not less than the number of queue families in the \
                                physical device"
                                .into(),
                            vuids: &["VUID-VkImageMemoryBarrier2-image-04072"],
                            ..Default::default()
                        }));
                    }
                }
                QueueFamilyOwnershipTransfer::ExclusiveToForeign { src_index } => {
                    if src_index >= queue_family_count {
                        return Err(Box::new(ValidationError {
                            context: "queue_family_ownership_transfer.src_index".into(),
                            problem: "is not less than the number of queue families in the \
                                physical device"
                                .into(),
                            vuids: &["VUID-VkImageMemoryBarrier2-image-04072"],
                            ..Default::default()
                        }));
                    }
                }
                QueueFamilyOwnershipTransfer::ExclusiveFromForeign { dst_index } => {
                    if dst_index >= queue_family_count {
                        return Err(Box::new(ValidationError {
                            context: "queue_family_ownership_transfer.dst_index".into(),
                            problem: "is not less than the number of queue families in the \
                                physical device"
                                .into(),
                            vuids: &["VUID-VkImageMemoryBarrier2-image-04072"],
                            ..Default::default()
                        }));
                    }
                }
                QueueFamilyOwnershipTransfer::ConcurrentToExternal
                | QueueFamilyOwnershipTransfer::ConcurrentFromExternal
                | QueueFamilyOwnershipTransfer::ConcurrentToForeign
                | QueueFamilyOwnershipTransfer::ConcurrentFromForeign => (),
            }
        }

        let is_image_layout_transition =
            !device.enabled_features().synchronization2 || old_layout != new_layout;
        let is_queue_family_ownership_transfer = queue_family_ownership_transfer.is_some();

        if is_image_layout_transition || is_queue_family_ownership_transfer {
            match old_layout {
                ImageLayout::ColorAttachmentOptimal => {
                    if !image.usage().intersects(ImageUsage::COLOR_ATTACHMENT) {
                        return Err(Box::new(ValidationError {
                            problem: "`old_layout` is `ImageLayout::ColorAttachmentOptimal`, but \
                                `image.usage()` does not contain `ImageUsage::COLOR_ATTACHMENT`"
                                .into(),
                            vuids: &["VUID-VkImageMemoryBarrier2-oldLayout-01208"],
                            ..Default::default()
                        }));
                    }
                }
                ImageLayout::DepthStencilAttachmentOptimal
                | ImageLayout::DepthStencilReadOnlyOptimal
                | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                | ImageLayout::DepthAttachmentStencilReadOnlyOptimal
                | ImageLayout::DepthAttachmentOptimal
                | ImageLayout::StencilAttachmentOptimal => {
                    if !image
                        .usage()
                        .intersects(ImageUsage::DEPTH_STENCIL_ATTACHMENT)
                    {
                        return Err(Box::new(ValidationError {
                            problem: "`old_layout` is \
                                `ImageLayout::DepthStencilAttachmentOptimal`, \
                                `ImageLayout::DepthStencilReadOnlyOptimal`, \
                                `ImageLayout::DepthReadOnlyStencilAttachmentOptimal`, \
                                `ImageLayout::DepthAttachmentStencilReadOnlyOptimal`, \
                                `ImageLayout::DepthAttachmentOptimal` or \
                                `ImageLayout::StencilAttachmentOptimal`, but \
                                `image.usage()` does not contain \
                                `ImageUsage::DEPTH_STENCIL_ATTACHMENT`"
                                .into(),
                            vuids: &[
                                "VUID-VkImageMemoryBarrier2-oldLayout-01209",
                                "VUID-VkImageMemoryBarrier2-oldLayout-01210",
                                "VUID-VkImageMemoryBarrier2-oldLayout-01658",
                                "VUID-VkImageMemoryBarrier2-oldLayout-01659",
                                "VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04066",
                                "VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04068",
                            ],
                            ..Default::default()
                        }));
                    }
                }
                ImageLayout::ShaderReadOnlyOptimal => {
                    if !image
                        .usage()
                        .intersects(ImageUsage::SAMPLED | ImageUsage::INPUT_ATTACHMENT)
                    {
                        return Err(Box::new(ValidationError {
                            problem: "`old_layout` is `ImageLayout::ShaderReadOnlyOptimal`, but \
                                `image.usage()` does not contain `ImageUsage::SAMPLED` or \
                                `ImageUsage::INPUT_ATTACHMENT`"
                                .into(),
                            vuids: &["VUID-VkImageMemoryBarrier2-oldLayout-01211"],
                            ..Default::default()
                        }));
                    }
                }
                ImageLayout::TransferSrcOptimal => {
                    if !image.usage().intersects(ImageUsage::TRANSFER_SRC) {
                        return Err(Box::new(ValidationError {
                            problem: "`old_layout` is `ImageLayout::TransferSrcOptimal`, but \
                                `image.usage()` does not contain `ImageUsage::TRANSFER_SRC`"
                                .into(),
                            vuids: &["VUID-VkImageMemoryBarrier2-oldLayout-01212"],
                            ..Default::default()
                        }));
                    }
                }
                ImageLayout::TransferDstOptimal => {
                    if !image.usage().intersects(ImageUsage::TRANSFER_DST) {
                        return Err(Box::new(ValidationError {
                            problem: "`old_layout` is `ImageLayout::TransferDstOptimal`, but \
                                `image.usage()` does not contain `ImageUsage::TRANSFER_DST`"
                                .into(),
                            vuids: &["VUID-VkImageMemoryBarrier2-oldLayout-01213"],
                            ..Default::default()
                        }));
                    }
                }
                ImageLayout::Preinitialized => todo!(),
                ImageLayout::DepthReadOnlyOptimal | ImageLayout::StencilReadOnlyOptimal => {
                    if !image.usage().intersects(
                        ImageUsage::DEPTH_STENCIL_ATTACHMENT
                            | ImageUsage::SAMPLED
                            | ImageUsage::INPUT_ATTACHMENT,
                    ) {
                        return Err(Box::new(ValidationError {
                            problem: "`old_layout` is `ImageLayout::DepthReadOnlyOptimal` or \
                                `ImageLayout::StencilReadOnlyOptimal`, but \
                                `image.usage()` does not contain \
                                `ImageUsage::DEPTH_STENCIL_ATTACHMENT`, `ImageUsage::SAMPLED` or \
                                `ImageUsage::INPUT_ATTACHMENT`"
                                .into(),
                            vuids: &[
                                "VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04065",
                                "VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04067",
                            ],
                            ..Default::default()
                        }));
                    }
                }
                ImageLayout::Undefined | ImageLayout::General | ImageLayout::PresentSrc => (),
            }

            match new_layout {
                ImageLayout::Undefined | ImageLayout::Preinitialized => {
                    return Err(Box::new(ValidationError {
                        context: "new_layout".into(),
                        problem: "is `ImageLayout::Undefined` or `ImageLayout::Preinitialized`"
                            .into(),
                        vuids: &["VUID-VkImageMemoryBarrier2-newLayout-01198"],
                        ..Default::default()
                    }));
                }
                ImageLayout::ColorAttachmentOptimal => {
                    if !image.usage().intersects(ImageUsage::COLOR_ATTACHMENT) {
                        return Err(Box::new(ValidationError {
                            problem: "`new_layout` is `ImageLayout::ColorAttachmentOptimal`, but \
                                `image.usage()` does not contain `ImageUsage::COLOR_ATTACHMENT`"
                                .into(),
                            vuids: &["VUID-VkImageMemoryBarrier2-oldLayout-01208"],
                            ..Default::default()
                        }));
                    }
                }
                ImageLayout::DepthStencilAttachmentOptimal
                | ImageLayout::DepthStencilReadOnlyOptimal
                | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                | ImageLayout::DepthAttachmentStencilReadOnlyOptimal
                | ImageLayout::DepthAttachmentOptimal
                | ImageLayout::StencilAttachmentOptimal => {
                    if !image
                        .usage()
                        .intersects(ImageUsage::DEPTH_STENCIL_ATTACHMENT)
                    {
                        return Err(Box::new(ValidationError {
                            problem: "`new_layout` is \
                                `ImageLayout::DepthStencilAttachmentOptimal`, \
                                `ImageLayout::DepthStencilReadOnlyOptimal`, \
                                `ImageLayout::DepthReadOnlyStencilAttachmentOptimal`, \
                                `ImageLayout::DepthAttachmentStencilReadOnlyOptimal`, \
                                `ImageLayout::DepthAttachmentOptimal` or \
                                `ImageLayout::StencilAttachmentOptimal`, but \
                                `image.usage()` does not contain \
                                `ImageUsage::DEPTH_STENCIL_ATTACHMENT`"
                                .into(),
                            vuids: &[
                                "VUID-VkImageMemoryBarrier2-oldLayout-01209",
                                "VUID-VkImageMemoryBarrier2-oldLayout-01210",
                                "VUID-VkImageMemoryBarrier2-oldLayout-01658",
                                "VUID-VkImageMemoryBarrier2-oldLayout-01659",
                                "VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04066",
                                "VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04068",
                            ],
                            ..Default::default()
                        }));
                    }
                }
                ImageLayout::ShaderReadOnlyOptimal => {
                    if !image
                        .usage()
                        .intersects(ImageUsage::SAMPLED | ImageUsage::INPUT_ATTACHMENT)
                    {
                        return Err(Box::new(ValidationError {
                            problem: "`new_layout` is `ImageLayout::ShaderReadOnlyOptimal`, but \
                                `image.usage()` does not contain `ImageUsage::SAMPLED` or \
                                `ImageUsage::INPUT_ATTACHMENT`"
                                .into(),
                            vuids: &["VUID-VkImageMemoryBarrier2-oldLayout-01211"],
                            ..Default::default()
                        }));
                    }
                }
                ImageLayout::TransferSrcOptimal => {
                    if !image.usage().intersects(ImageUsage::TRANSFER_SRC) {
                        return Err(Box::new(ValidationError {
                            problem: "`new_layout` is `ImageLayout::TransferSrcOptimal`, but \
                                `image.usage()` does not contain `ImageUsage::TRANSFER_SRC`"
                                .into(),
                            vuids: &["VUID-VkImageMemoryBarrier2-oldLayout-01212"],
                            ..Default::default()
                        }));
                    }
                }
                ImageLayout::TransferDstOptimal => {
                    if !image.usage().intersects(ImageUsage::TRANSFER_DST) {
                        return Err(Box::new(ValidationError {
                            problem: "`new_layout` is `ImageLayout::TransferDstOptimal`, but \
                                `image.usage()` does not contain `ImageUsage::TRANSFER_DST`"
                                .into(),
                            vuids: &["VUID-VkImageMemoryBarrier2-oldLayout-01213"],
                            ..Default::default()
                        }));
                    }
                }
                ImageLayout::DepthReadOnlyOptimal | ImageLayout::StencilReadOnlyOptimal => {
                    if !image.usage().intersects(
                        ImageUsage::DEPTH_STENCIL_ATTACHMENT
                            | ImageUsage::SAMPLED
                            | ImageUsage::INPUT_ATTACHMENT,
                    ) {
                        return Err(Box::new(ValidationError {
                            problem: "`new_layout` is `ImageLayout::DepthReadOnlyOptimal` or \
                                `ImageLayout::StencilReadOnlyOptimal`, but \
                                `image.usage()` does not contain \
                                `ImageUsage::DEPTH_STENCIL_ATTACHMENT`, `ImageUsage::SAMPLED` or \
                                `ImageUsage::INPUT_ATTACHMENT`"
                                .into(),
                            vuids: &[
                                "VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04065",
                                "VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04067",
                            ],
                            ..Default::default()
                        }));
                    }
                }
                ImageLayout::General | ImageLayout::PresentSrc => (),
            }

            if src_stages.intersects(PipelineStages::HOST)
                && !matches!(
                    old_layout,
                    ImageLayout::Preinitialized | ImageLayout::Undefined | ImageLayout::General
                )
            {
                return Err(Box::new(ValidationError {
                    problem: "`src_stages` contains `PipelineStages::HOST`, but \
                        `old_layout` is not `ImageLayout::Preinitialized`, \
                        `ImageLayout::Undefined` or `ImageLayout::General`"
                        .into(),
                    vuids: &["VUID-VkImageMemoryBarrier2-srcStageMask-03855"],
                    ..Default::default()
                }));
            }

            // VUID-VkImageMemoryBarrier2-oldLayout-01197
            // Unsafe
        }

        Ok(())
    }

    pub(crate) fn to_vk2(&self) -> vk::ImageMemoryBarrier2<'static> {
        let &Self {
            src_stages,
            src_access,
            dst_stages,
            dst_access,
            old_layout,
            new_layout,
            queue_family_ownership_transfer,
            image,
            subresource_range,
            _ne: _,
        } = self;

        let (src_queue_family_index, dst_queue_family_index) =
            queue_family_ownership_transfer.as_ref().map_or(
                (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED),
                QueueFamilyOwnershipTransfer::to_vk,
            );

        vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src_stages.into())
            .src_access_mask(src_access.into())
            .dst_stage_mask(dst_stages.into())
            .dst_access_mask(dst_access.into())
            .old_layout(old_layout.into())
            .new_layout(new_layout.into())
            .src_queue_family_index(src_queue_family_index)
            .dst_queue_family_index(dst_queue_family_index)
            .image(image.handle())
            .subresource_range(subresource_range.to_vk())
    }

    pub(crate) fn to_vk(&self) -> vk::ImageMemoryBarrier<'static> {
        let &Self {
            src_stages: _,
            src_access,
            dst_stages: _,
            dst_access,
            old_layout,
            new_layout,
            queue_family_ownership_transfer,
            image,
            subresource_range,
            _ne: _,
        } = self;

        let (src_queue_family_index, dst_queue_family_index) =
            queue_family_ownership_transfer.as_ref().map_or(
                (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED),
                QueueFamilyOwnershipTransfer::to_vk,
            );

        vk::ImageMemoryBarrier::default()
            .src_access_mask(src_access.into())
            .dst_access_mask(dst_access.into())
            .old_layout(old_layout.into())
            .new_layout(new_layout.into())
            .src_queue_family_index(src_queue_family_index)
            .dst_queue_family_index(dst_queue_family_index)
            .image(image.handle())
            .subresource_range(subresource_range.to_vk())
    }
}
