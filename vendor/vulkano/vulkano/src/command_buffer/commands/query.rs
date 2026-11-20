use crate::{
    buffer::{Buffer, BufferUsage},
    command_buffer::sys::RecordingCommandBuffer,
    device::{DeviceOwned, QueueFlags},
    query::{QueryControlFlags, QueryPool, QueryResultElement, QueryResultFlags, QueryType},
    sync::{PipelineStage, PipelineStages},
    DeviceSize, Requires, RequiresAllOf, RequiresOneOf, ValidationError, Version, VulkanObject,
};
use ash::vk;
use std::sync::Arc;

impl RecordingCommandBuffer {
    #[inline]
    #[track_caller]
    pub unsafe fn begin_query(
        &mut self,
        query_pool: &QueryPool,
        query: u32,
        flags: QueryControlFlags,
    ) -> &mut Self {
        unsafe { self.try_begin_query(query_pool, query, flags) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_begin_query(
        &mut self,
        query_pool: &QueryPool,
        query: u32,
        flags: QueryControlFlags,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_begin_query(query_pool, query, flags)?;

        Ok(unsafe { self.begin_query_unchecked(query_pool, query, flags) })
    }

    pub(crate) fn validate_begin_query(
        &self,
        query_pool: &QueryPool,
        query: u32,
        flags: QueryControlFlags,
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
                vuids: &["VUID-vkCmdBeginQuery-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        let device = self.device();

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-vkCmdBeginQuery-flags-parameter"])
        })?;

        // VUID-vkCmdBeginQuery-commonparent
        assert_eq!(device, query_pool.device());

        if query > query_pool.query_count() {
            return Err(Box::new(ValidationError {
                problem: "`query` is greater than `query_pool.query_count()`".into(),
                vuids: &["VUID-vkCmdBeginQuery-query-00802"],
                ..Default::default()
            }));
        }

        match query_pool.query_type() {
            QueryType::Occlusion => {
                if !queue_family_properties
                    .queue_flags
                    .intersects(QueueFlags::GRAPHICS)
                {
                    return Err(Box::new(ValidationError {
                        problem: "`query_pool.query_type()` is `QueryType::Occlusion`, but \
                            the queue family of the command buffer does not support \
                            graphics operations"
                            .into(),
                        vuids: &["VUID-vkCmdBeginQuery-queryType-00803"],
                        ..Default::default()
                    }));
                }
            }
            QueryType::PipelineStatistics => {
                if query_pool.pipeline_statistics().is_graphics()
                    && !queue_family_properties
                        .queue_flags
                        .intersects(QueueFlags::GRAPHICS)
                {
                    return Err(Box::new(ValidationError {
                        problem: "`query_pool.query_type()` is `QueryType::PipelineStatistics`, \
                            and `query_pool.pipeline_statistics()` includes a graphics flag, but \
                            the queue family of the command buffer does not support \
                            graphics operations"
                            .into(),
                        vuids: &["VUID-vkCmdBeginQuery-queryType-00804"],
                        ..Default::default()
                    }));
                }

                if query_pool.pipeline_statistics().is_compute()
                    && !queue_family_properties
                        .queue_flags
                        .intersects(QueueFlags::COMPUTE)
                {
                    return Err(Box::new(ValidationError {
                        problem: "`query_pool.query_type()` is `QueryType::PipelineStatistics`, \
                            and `query_pool.pipeline_statistics()` includes a compute flag, but \
                            the queue family of the command buffer does not support \
                            compute operations"
                            .into(),
                        vuids: &["VUID-vkCmdBeginQuery-queryType-00805"],
                        ..Default::default()
                    }));
                }
            }
            QueryType::MeshPrimitivesGenerated => {
                if !queue_family_properties
                    .queue_flags
                    .intersects(QueueFlags::GRAPHICS)
                {
                    return Err(Box::new(ValidationError {
                        problem: "`query_pool.query_type()` is \
                            `QueryType::MeshPrimitivesGenerated`, but \
                            the queue family of the command buffer does not support \
                            graphics operations"
                            .into(),
                        vuids: &["VUID-vkCmdBeginQuery-queryType-07070"],
                        ..Default::default()
                    }));
                }
            }
            QueryType::Timestamp
            | QueryType::AccelerationStructureCompactedSize
            | QueryType::AccelerationStructureSerializationSize
            | QueryType::AccelerationStructureSerializationBottomLevelPointers
            | QueryType::AccelerationStructureSize => {
                return Err(Box::new(ValidationError {
                    context: "query_pool.query_type()".into(),
                    problem: "is not allowed for this command".into(),
                    vuids: &[
                        "VUID-vkCmdBeginQuery-queryType-02804",
                        "VUID-vkCmdBeginQuery-queryType-04728",
                        "VUID-vkCmdBeginQuery-queryType-06741",
                    ],
                    ..Default::default()
                }));
            }
        }

        if flags.intersects(QueryControlFlags::PRECISE) {
            if !device.enabled_features().occlusion_query_precise {
                return Err(Box::new(ValidationError {
                    context: "flags".into(),
                    problem: "contains `QueryControlFlags::PRECISE`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "occlusion_query_precise",
                    )])]),
                    vuids: &["VUID-vkCmdBeginQuery-queryType-00800"],
                }));
            }

            if !matches!(query_pool.query_type(), QueryType::Occlusion) {
                return Err(Box::new(ValidationError {
                    problem: "`flags` contains `QueryControlFlags::PRECISE`, but \
                        `query_pool.query_type()` is not `QueryType::Occlusion`"
                        .into(),
                    vuids: &["VUID-vkCmdBeginQuery-queryType-00800"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn begin_query_unchecked(
        &mut self,
        query_pool: &QueryPool,
        query: u32,
        flags: QueryControlFlags,
    ) -> &mut Self {
        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_begin_query)(self.handle(), query_pool.handle(), query, flags.into())
        };

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn end_query(&mut self, query_pool: &QueryPool, query: u32) -> &mut Self {
        unsafe { self.try_end_query(query_pool, query) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_end_query(
        &mut self,
        query_pool: &QueryPool,
        query: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_end_query(query_pool, query)?;

        Ok(unsafe { self.end_query_unchecked(query_pool, query) })
    }

    pub(crate) fn validate_end_query(
        &self,
        query_pool: &QueryPool,
        query: u32,
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
                vuids: &["VUID-vkCmdEndQuery-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        let device = self.device();

        // VUID-vkCmdEndQuery-commonparent
        assert_eq!(device, query_pool.device());

        if query > query_pool.query_count() {
            return Err(Box::new(ValidationError {
                problem: "`query` is greater than `query_pool.query_count()`".into(),
                vuids: &["VUID-vkCmdEndQuery-query-00810"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn end_query_unchecked(&mut self, query_pool: &QueryPool, query: u32) -> &mut Self {
        let fns = self.device().fns();
        unsafe { (fns.v1_0.cmd_end_query)(self.handle(), query_pool.handle(), query) };

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn write_timestamp(
        &mut self,
        query_pool: &QueryPool,
        query: u32,
        stage: PipelineStage,
    ) -> &mut Self {
        unsafe { self.try_write_timestamp(query_pool, query, stage) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_write_timestamp(
        &mut self,
        query_pool: &QueryPool,
        query: u32,
        stage: PipelineStage,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_write_timestamp(query_pool, query, stage)?;

        Ok(unsafe { self.write_timestamp_unchecked(query_pool, query, stage) })
    }

    pub(crate) fn validate_write_timestamp(
        &self,
        query_pool: &QueryPool,
        query: u32,
        stage: PipelineStage,
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
                vuids: &["VUID-vkCmdWriteTimestamp2-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        let device = self.device();

        stage.validate_device(device).map_err(|err| {
            err.add_context("stage")
                .set_vuids(&["VUID-vkCmdWriteTimestamp2-stage-parameter"])
        })?;

        if !device.enabled_features().synchronization2
            && PipelineStages::from(stage).contains_flags2()
        {
            return Err(Box::new(ValidationError {
                context: "stage".into(),
                problem: "is a stage flag from `VkPipelineStageFlagBits2`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "synchronization2",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-vkCmdWriteTimestamp2-commonparent
        assert_eq!(device, query_pool.device());

        if !PipelineStages::from(queue_family_properties.queue_flags).contains_enum(stage) {
            return Err(Box::new(ValidationError {
                context: "stage".into(),
                problem: "is not supported by the queue family of the command buffer".into(),
                vuids: &["VUID-vkCmdWriteTimestamp2-stage-03860"],
                ..Default::default()
            }));
        }

        match stage {
            PipelineStage::GeometryShader => {
                if !device.enabled_features().geometry_shader {
                    return Err(Box::new(ValidationError {
                        context: "stage".into(),
                        problem: "is `PipelineStage::GeometryShader`".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                            Requires::DeviceFeature("geometry_shadere"),
                        ])]),
                        vuids: &["VUID-vkCmdWriteTimestamp2-stage-03929"],
                    }));
                }
            }
            PipelineStage::TessellationControlShader
            | PipelineStage::TessellationEvaluationShader => {
                if !device.enabled_features().tessellation_shader {
                    return Err(Box::new(ValidationError {
                        context: "stage".into(),
                        problem: "is `PipelineStage::TessellationControlShader` or \
                            `PipelineStage::TessellationEvaluationShader`"
                            .into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                            Requires::DeviceFeature("tessellation_shader"),
                        ])]),
                        vuids: &["VUID-vkCmdWriteTimestamp2-stage-03930"],
                    }));
                }
            }
            PipelineStage::ConditionalRendering => {
                if !device.enabled_features().conditional_rendering {
                    return Err(Box::new(ValidationError {
                        context: "stage".into(),
                        problem: "is `PipelineStage::ConditionalRendering`".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                            Requires::DeviceFeature("conditional_rendering"),
                        ])]),
                        vuids: &["VUID-vkCmdWriteTimestamp2-stage-03931"],
                    }));
                }
            }
            PipelineStage::FragmentDensityProcess => {
                if !device.enabled_features().fragment_density_map {
                    return Err(Box::new(ValidationError {
                        context: "stage".into(),
                        problem: "is `PipelineStage::FragmentDensityProcess`".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                            Requires::DeviceFeature("fragment_density_map"),
                        ])]),
                        vuids: &["VUID-vkCmdWriteTimestamp2-stage-03932"],
                    }));
                }
            }
            PipelineStage::TransformFeedback => {
                if !device.enabled_features().transform_feedback {
                    return Err(Box::new(ValidationError {
                        context: "stage".into(),
                        problem: "is `PipelineStage::TransformFeedback`".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                            Requires::DeviceFeature("transform_feedback"),
                        ])]),
                        vuids: &["VUID-vkCmdWriteTimestamp2-stage-03933"],
                    }));
                }
            }
            PipelineStage::MeshShader => {
                if !device.enabled_features().mesh_shader {
                    return Err(Box::new(ValidationError {
                        context: "stage".into(),
                        problem: "is `PipelineStage::MeshShader`".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                            Requires::DeviceFeature("mesh_shader"),
                        ])]),
                        vuids: &["VUID-vkCmdWriteTimestamp2-stage-03934"],
                    }));
                }
            }
            PipelineStage::TaskShader => {
                if !device.enabled_features().task_shader {
                    return Err(Box::new(ValidationError {
                        context: "stage".into(),
                        problem: "is `PipelineStage::TaskShader`".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                            Requires::DeviceFeature("task_shader"),
                        ])]),
                        vuids: &["VUID-vkCmdWriteTimestamp2-stage-03935"],
                    }));
                }
            }
            PipelineStage::FragmentShadingRateAttachment => {
                if !(device.enabled_features().attachment_fragment_shading_rate
                    || device.enabled_features().shading_rate_image)
                {
                    return Err(Box::new(ValidationError {
                        context: "stage".into(),
                        problem: "is `PipelineStage::FragmentShadingRateAttachment`".into(),
                        requires_one_of: RequiresOneOf(&[
                            RequiresAllOf(&[Requires::DeviceFeature(
                                "attachment_fragment_shading_rate",
                            )]),
                            RequiresAllOf(&[Requires::DeviceFeature("shading_rate_image")]),
                        ]),
                        vuids: &["VUID-vkCmdWriteTimestamp2-shadingRateImage-07316"],
                    }));
                }
            }
            PipelineStage::SubpassShader => {
                if !device.enabled_features().subpass_shading {
                    return Err(Box::new(ValidationError {
                        context: "stage".into(),
                        problem: "is `PipelineStage::SubpassShader`".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                            Requires::DeviceFeature("subpass_shading"),
                        ])]),
                        vuids: &["VUID-vkCmdWriteTimestamp2-stage-04957"],
                    }));
                }
            }
            PipelineStage::InvocationMask => {
                if !device.enabled_features().invocation_mask {
                    return Err(Box::new(ValidationError {
                        context: "stage".into(),
                        problem: "is `PipelineStage::InvocationMask`".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                            Requires::DeviceFeature("invocation_mask"),
                        ])]),
                        vuids: &["VUID-vkCmdWriteTimestamp2-stage-04995"],
                    }));
                }
            }
            _ => (),
        }

        if !matches!(query_pool.query_type(), QueryType::Timestamp) {
            return Err(Box::new(ValidationError {
                context: "query_pool.query_type()".into(),
                problem: "is not `QueryType::Timestamp`".into(),
                vuids: &["VUID-vkCmdWriteTimestamp2-queryPool-03861"],
                ..Default::default()
            }));
        }

        if queue_family_properties.timestamp_valid_bits.is_none() {
            return Err(Box::new(ValidationError {
                problem: "the `timestamp_valid_bits` value of the queue family properties of \
                    the command buffer is `None`"
                    .into(),
                vuids: &["VUID-vkCmdWriteTimestamp2-timestampValidBits-03863"],
                ..Default::default()
            }));
        }

        if query > query_pool.query_count() {
            return Err(Box::new(ValidationError {
                problem: "`query` is greater than `query_pool.query_count()`".into(),
                vuids: &["VUID-vkCmdWriteTimestamp2-query-04903"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn write_timestamp_unchecked(
        &mut self,
        query_pool: &QueryPool,
        query: u32,
        stage: PipelineStage,
    ) -> &mut Self {
        let fns = self.device().fns();

        if self.device().enabled_features().synchronization2 {
            if self.device().api_version() >= Version::V1_3 {
                unsafe {
                    (fns.v1_3.cmd_write_timestamp2)(
                        self.handle(),
                        stage.into(),
                        query_pool.handle(),
                        query,
                    )
                };
            } else {
                unsafe {
                    (fns.khr_synchronization2.cmd_write_timestamp2_khr)(
                        self.handle(),
                        stage.into(),
                        query_pool.handle(),
                        query,
                    )
                };
            }
        } else {
            unsafe {
                (fns.v1_0.cmd_write_timestamp)(
                    self.handle(),
                    stage.into(),
                    query_pool.handle(),
                    query,
                )
            };
        }

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn copy_query_pool_results<T>(
        &mut self,
        query_pool: &QueryPool,
        first_query: u32,
        query_count: u32,
        dst_buffer: &Arc<Buffer>,
        dst_offset: DeviceSize,
        stride: DeviceSize,
        flags: QueryResultFlags,
    ) -> &mut Self
    where
        T: QueryResultElement,
    {
        unsafe {
            self.try_copy_query_pool_results::<T>(
                query_pool,
                first_query,
                query_count,
                dst_buffer,
                dst_offset,
                stride,
                flags,
            )
        }
        .unwrap()
    }

    #[inline]
    pub unsafe fn try_copy_query_pool_results<T>(
        &mut self,
        query_pool: &QueryPool,
        first_query: u32,
        query_count: u32,
        dst_buffer: &Arc<Buffer>,
        dst_offset: DeviceSize,
        stride: DeviceSize,
        flags: QueryResultFlags,
    ) -> Result<&mut Self, Box<ValidationError>>
    where
        T: QueryResultElement,
    {
        self.validate_copy_query_pool_results::<T>(
            query_pool,
            first_query,
            query_count,
            dst_buffer,
            dst_offset,
            stride,
            flags,
        )?;

        Ok(unsafe {
            self.copy_query_pool_results_unchecked::<T>(
                query_pool,
                first_query,
                query_count,
                dst_buffer,
                dst_offset,
                stride,
                flags,
            )
        })
    }

    pub(crate) fn validate_copy_query_pool_results<T>(
        &self,
        query_pool: &QueryPool,
        first_query: u32,
        query_count: u32,
        dst_buffer: &Arc<Buffer>,
        dst_offset: DeviceSize,
        stride: DeviceSize,
        flags: QueryResultFlags,
    ) -> Result<(), Box<ValidationError>>
    where
        T: QueryResultElement,
    {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics or compute operations"
                    .into(),
                vuids: &["VUID-vkCmdCopyQueryPoolResults-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        let device = self.device();

        // VUID-vkCmdCopyQueryPoolResults-commonparent
        assert_eq!(device, dst_buffer.device());
        assert_eq!(device, query_pool.device());

        if dst_offset >= dst_buffer.size() {
            return Err(Box::new(ValidationError {
                context: "dst_offset".into(),
                problem: "is greater than or equal to `dst_buffer.size()`".into(),
                vuids: &["VUID-vkCmdCopyQueryPoolResults-dstOffset-00819"],
                ..Default::default()
            }));
        }

        if !dst_offset.is_multiple_of(size_of::<T>() as DeviceSize) {
            return Err(Box::new(ValidationError {
                context: "dst_offset".into(),
                problem: "is not a multiple of the result type's size".into(),
                vuids: &[
                    "VUID-vkCmdCopyQueryPoolResults-flags-00822",
                    "VUID-vkCmdCopyQueryPoolResults-flags-00823",
                ],
                ..Default::default()
            }));
        }

        if !first_query
            .checked_add(query_count)
            .is_some_and(|end| end <= query_pool.query_count())
        {
            return Err(Box::new(ValidationError {
                problem: "`first_query + query_count` is greater than `query_pool.query_count()`"
                    .into(),
                vuids: &[
                    "VUID-vkCmdCopyQueryPoolResults-firstQuery-00820",
                    "VUID-vkCmdCopyQueryPoolResults-firstQuery-00821",
                ],
                ..Default::default()
            }));
        }

        let per_query_len = query_pool.result_len(flags);
        let required_len = per_query_len * query_count as DeviceSize;

        if (dst_buffer.size() - dst_offset) / (size_of::<T>() as DeviceSize) < required_len {
            return Err(Box::new(ValidationError {
                problem: "`dst_buffer` is smaller than the size required to write the results"
                    .into(),
                vuids: &["VUID-vkCmdCopyQueryPoolResults-dstBuffer-00824"],
                ..Default::default()
            }));
        }

        if !dst_buffer.usage().intersects(BufferUsage::TRANSFER_DST) {
            return Err(Box::new(ValidationError {
                context: "dst_buffer.usage()".into(),
                problem: "does not contain `BufferUsage::TRANSFER_DST`".into(),
                vuids: &["VUID-vkCmdCopyQueryPoolResults-dstBuffer-00825"],
                ..Default::default()
            }));
        }

        if matches!(query_pool.query_type(), QueryType::Timestamp)
            && flags.intersects(QueryResultFlags::PARTIAL)
        {
            return Err(Box::new(ValidationError {
                problem: "`query_pool.query_type()` is `QueryType::Timestamp`, but \
                    `flags` contains `QueryResultFlags::PARTIAL`"
                    .into(),
                vuids: &["VUID-vkCmdCopyQueryPoolResults-queryType-00827"],
                ..Default::default()
            }));
        }

        if query_count > 1 {
            if stride == 0 {
                return Err(Box::new(ValidationError {
                    problem: "`query_count` is greater than 1, but `stride` is zero".into(),
                    vuids: &["VUID-vkCmdCopyQueryPoolResults-queryCount-09438"],
                    ..Default::default()
                }));
            }

            if !stride.is_multiple_of(size_of::<T>() as DeviceSize) {
                return Err(Box::new(ValidationError {
                    problem: "`query_count` is greater than 1, but `stride` is not a multiple of \
                        the result type's size"
                        .into(),
                    vuids: &[
                        "VUID-vkCmdCopyQueryPoolResults-queryCount-12254",
                        "VUID-vkCmdCopyQueryPoolResults-queryCount-12255",
                    ],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn copy_query_pool_results_unchecked<T>(
        &mut self,
        query_pool: &QueryPool,
        first_query: u32,
        query_count: u32,
        dst_buffer: &Arc<Buffer>,
        dst_offset: DeviceSize,
        stride: DeviceSize,
        flags: QueryResultFlags,
    ) -> &mut Self
    where
        T: QueryResultElement,
    {
        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_copy_query_pool_results)(
                self.handle(),
                query_pool.handle(),
                first_query,
                query_count,
                dst_buffer.handle(),
                dst_offset,
                stride,
                vk::QueryResultFlags::from(flags) | T::FLAG,
            )
        };

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn reset_query_pool(
        &mut self,
        query_pool: &QueryPool,
        first_query: u32,
        query_count: u32,
    ) -> &mut Self {
        unsafe { self.try_reset_query_pool(query_pool, first_query, query_count) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_reset_query_pool(
        &mut self,
        query_pool: &QueryPool,
        first_query: u32,
        query_count: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_reset_query_pool(query_pool, first_query, query_count)?;

        Ok(unsafe { self.reset_query_pool_unchecked(query_pool, first_query, query_count) })
    }

    pub(crate) fn validate_reset_query_pool(
        &self,
        query_pool: &QueryPool,
        first_query: u32,
        query_count: u32,
    ) -> Result<(), Box<ValidationError>> {
        let queue_family_properties = self.queue_family_properties();

        if !queue_family_properties.queue_flags.intersects(
            QueueFlags::GRAPHICS
                | QueueFlags::COMPUTE
                | QueueFlags::VIDEO_DECODE
                | QueueFlags::VIDEO_ENCODE
                | QueueFlags::OPTICAL_FLOW,
        ) {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics, compute, video decode, video encode or optical flow operations"
                    .into(),
                vuids: &["VUID-vkCmdResetQueryPool-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        let device = self.device();

        // VUID-vkCmdResetQueryPool-commonparent
        assert_eq!(device, query_pool.device());

        if !first_query
            .checked_add(query_count)
            .is_some_and(|end| end <= query_pool.query_count())
        {
            return Err(Box::new(ValidationError {
                problem: "`first_query + query_count` is greater than `query_pool.query_count()`"
                    .into(),
                vuids: &[
                    "VUID-vkCmdResetQueryPool-firstQuery-00796",
                    "VUID-vkCmdResetQueryPool-firstQuery-00797",
                ],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn reset_query_pool_unchecked(
        &mut self,
        query_pool: &QueryPool,
        first_query: u32,
        query_count: u32,
    ) -> &mut Self {
        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_reset_query_pool)(
                self.handle(),
                query_pool.handle(),
                first_query,
                query_count,
            )
        };

        self
    }
}
