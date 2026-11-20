use crate::{
    buffer::{Buffer, BufferUsage},
    command_buffer::sys::RecordingCommandBuffer,
    device::{Device, DeviceOwned, QueueFlags},
    format::{Format, FormatFeatures},
    image::{
        mip_level_extent, sampler::Filter, Image, ImageAspects, ImageLayout,
        ImageSubresourceLayers, ImageTiling, ImageType, ImageUsage, SampleCount,
    },
    DeviceSize, Requires, RequiresAllOf, RequiresOneOf, ValidationError, Version, VulkanObject,
};
use ash::vk;
use smallvec::{smallvec, SmallVec};
use std::cmp::{max, min};

impl RecordingCommandBuffer {
    #[inline]
    #[track_caller]
    pub unsafe fn copy_buffer(&mut self, copy_buffer_info: &CopyBufferInfo<'_>) -> &mut Self {
        unsafe { self.try_copy_buffer(copy_buffer_info) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_copy_buffer(
        &mut self,
        copy_buffer_info: &CopyBufferInfo<'_>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_copy_buffer(copy_buffer_info)?;

        Ok(unsafe { self.copy_buffer_unchecked(copy_buffer_info) })
    }

    pub(crate) fn validate_copy_buffer(
        &self,
        copy_buffer_info: &CopyBufferInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::TRANSFER | QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    transfer, graphics or compute operations"
                    .into(),
                vuids: &["VUID-vkCmdCopyBuffer2-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        copy_buffer_info
            .validate(self.device())
            .map_err(|err| err.add_context("copy_buffer_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn copy_buffer_unchecked(
        &mut self,
        copy_buffer_info: &CopyBufferInfo<'_>,
    ) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3
            || self.device().enabled_extensions().khr_copy_commands2
        {
            let regions_vk = copy_buffer_info.to_vk2_regions();
            let copy_buffer_info_vk = copy_buffer_info.to_vk2(&regions_vk);

            if self.device().api_version() >= Version::V1_3 {
                unsafe { (fns.v1_3.cmd_copy_buffer2)(self.handle(), &copy_buffer_info_vk) };
            } else {
                unsafe {
                    (fns.khr_copy_commands2.cmd_copy_buffer2_khr)(
                        self.handle(),
                        &copy_buffer_info_vk,
                    )
                };
            }
        } else {
            let regions_vk = copy_buffer_info.to_vk_regions();
            let CopyBufferInfoVk {
                src_buffer_vk,
                dst_buffer_vk,
            } = copy_buffer_info.to_vk();

            unsafe {
                (fns.v1_0.cmd_copy_buffer)(
                    self.handle(),
                    src_buffer_vk,
                    dst_buffer_vk,
                    regions_vk.len() as u32,
                    regions_vk.as_ptr(),
                )
            };
        }

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn copy_image(&mut self, copy_image_info: &CopyImageInfo<'_>) -> &mut Self {
        unsafe { self.try_copy_image(copy_image_info) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_copy_image(
        &mut self,
        copy_image_info: &CopyImageInfo<'_>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_copy_image(copy_image_info)?;

        Ok(unsafe { self.copy_image_unchecked(copy_image_info) })
    }

    pub(crate) fn validate_copy_image(
        &self,
        copy_image_info: &CopyImageInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        let queue_family_properties = self.queue_family_properties();

        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::TRANSFER | QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    transfer, graphics or compute operations"
                    .into(),
                vuids: &["VUID-vkCmdCopyImage2-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        copy_image_info
            .validate(self.device())
            .map_err(|err| err.add_context("copy_image_info"))?;

        let &CopyImageInfo {
            src_image,
            src_image_layout: _,
            dst_image,
            dst_image_layout: _,
            regions,
            _ne: _,
        } = copy_image_info;

        let src_image_format = src_image.format();
        let src_image_format_subsampled_extent = src_image_format
            .ycbcr_chroma_sampling()
            .map_or(src_image.extent(), |s| {
                s.subsampled_extent(src_image.extent())
            });

        let dst_image_format = dst_image.format();
        let dst_image_format_subsampled_extent = dst_image_format
            .ycbcr_chroma_sampling()
            .map_or(dst_image.extent(), |s| {
                s.subsampled_extent(dst_image.extent())
            });

        let min_image_transfer_granularity =
            // `[1; 3]` means the granularity is 1x1x1 texel, so we can ignore it.
            // Only check this if there are values greater than 1.
            (queue_family_properties.min_image_transfer_granularity != [1; 3]).then(|| {
                // `[0; 3]` means only the whole subresource can be copied.
                (queue_family_properties.min_image_transfer_granularity != [0; 3]).then(|| {
                    // Spec:
                    // "The value returned in minImageTransferGranularity has a unit of
                    // compressed texel blocks for images having a block-compressed format,
                    // and a unit of texels otherwise.""

                    let src_granularity = if src_image_format.compression().is_some() {
                        let granularity = queue_family_properties.min_image_transfer_granularity;
                        let block_extent = src_image_format.block_extent();

                        [
                            granularity[0] * block_extent[0],
                            granularity[1] * block_extent[1],
                            granularity[2] * block_extent[2],
                        ]
                    } else {
                        queue_family_properties.min_image_transfer_granularity
                    };

                    let dst_granularity = if dst_image_format.compression().is_some() {
                        let granularity = queue_family_properties.min_image_transfer_granularity;
                        let block_extent = dst_image_format.block_extent();

                        [
                            granularity[0] * block_extent[0],
                            granularity[1] * block_extent[1],
                            granularity[2] * block_extent[2],
                        ]
                    } else {
                        queue_family_properties.min_image_transfer_granularity
                    };

                    (src_granularity, dst_granularity)
                })
            });

        if min_image_transfer_granularity.is_some() {
            for (region_index, region) in regions.iter().enumerate() {
                let &ImageCopy {
                    src_subresource,
                    src_offset,
                    dst_subresource,
                    dst_offset,
                    extent,
                    _ne: _,
                } = region;

                if let Some(min_image_transfer_granularity) = &min_image_transfer_granularity {
                    let mut src_subresource_extent =
                        mip_level_extent(src_image.extent(), src_subresource.mip_level).unwrap();

                    if matches!(
                        src_subresource.aspects,
                        ImageAspects::PLANE_1 | ImageAspects::PLANE_2
                    ) {
                        src_subresource_extent = src_image_format_subsampled_extent;
                    }

                    let mut dst_subresource_extent =
                        mip_level_extent(dst_image.extent(), dst_subresource.mip_level).unwrap();

                    if matches!(
                        dst_subresource.aspects,
                        ImageAspects::PLANE_1 | ImageAspects::PLANE_2
                    ) {
                        dst_subresource_extent = dst_image_format_subsampled_extent;
                    }

                    if let Some((src_granularity, dst_granularity)) =
                        &min_image_transfer_granularity
                    {
                        /*
                           Check src
                        */

                        for i in 0..3 {
                            if src_offset[i] % src_granularity[i] != 0 {
                                return Err(Box::new(ValidationError {
                                    context: "copy_image_info".into(),
                                    problem: format!(
                                        "the `min_image_transfer_granularity` property of the \
                                    queue family of the command buffer is not `[0; 3]`, but \
                                    `regions[{}].src_offset[{1}]` is not a multiple of \
                                    `min_image_transfer_granularity[{1}]` texel blocks",
                                        region_index, i,
                                    )
                                    .into(),
                                    vuids: &["VUID-VkCopyImageInfo2-srcOffset-01783"],
                                    ..Default::default()
                                }));
                            }

                            if src_offset[i] + extent[i] != src_subresource_extent[i]
                                && extent[i] % src_granularity[i] != 0
                            {
                                return Err(Box::new(ValidationError {
                                    context: "copy_image_info".into(),
                                    problem: format!(
                                        "the `min_image_transfer_granularity` property of the \
                                    queue family of the command buffer is not `[0; 3]`, and \
                                    `regions[{0}].src_offset[{1}] + regions[{0}].extent[{1}]` \
                                    is not equal to coordinate {1} of the extent of the \
                                    subresource of `src_image` selected by \
                                    `regions[{0}].src_subresource`, but \
                                    `regions[{}].extent[{1}]` is not a multiple of \
                                    `min_image_transfer_granularity[{1}]` texel blocks",
                                        region_index, i,
                                    )
                                    .into(),
                                    vuids: &["VUID-VkCopyImageInfo2-srcOffset-01783"],
                                    ..Default::default()
                                }));
                            }
                        }

                        /*
                           Check dst
                        */

                        for i in 0..3 {
                            if dst_offset[i] % dst_granularity[i] != 0 {
                                return Err(Box::new(ValidationError {
                                    context: "copy_image_info".into(),
                                    problem: format!(
                                        "the `min_image_transfer_granularity` property of the \
                                    queue family of the command buffer is not `[0; 3]`, but \
                                    `regions[{}].dst_offset[{1}]` is not a multiple of \
                                    `min_image_transfer_granularity[{1}]` texel blocks",
                                        region_index, i,
                                    )
                                    .into(),
                                    vuids: &["VUID-VkCopyImageInfo2-dstOffset-01784"],
                                    ..Default::default()
                                }));
                            }

                            if dst_offset[i] + extent[i] != dst_subresource_extent[i]
                                && extent[i] % dst_granularity[i] != 0
                            {
                                return Err(Box::new(ValidationError {
                                    context: "copy_image_info".into(),
                                    problem: format!(
                                        "the `min_image_transfer_granularity` property of the \
                                    queue family of the command buffer is not `[0; 3]`, and \
                                    `regions[{0}].dst_offset[{1}] + regions[{0}].extent[{1}]` \
                                    is not equal to coordinate {1} of the extent of the \
                                    subresource of `dst_image` selected by \
                                    `regions[{0}].dst_subresource`, but \
                                    `regions[{}].extent[{1}]` is not a multiple of \
                                    `min_image_transfer_granularity[{1}]` texel blocks",
                                        region_index, i,
                                    )
                                    .into(),
                                    vuids: &["VUID-VkCopyImageInfo2-dstOffset-01784"],
                                    ..Default::default()
                                }));
                            }
                        }
                    } else {
                        /*
                           Check src
                        */

                        for i in 0..3 {
                            if src_offset[i] != 0 {
                                return Err(Box::new(ValidationError {
                                    context: "copy_image_info".into(),
                                    problem: format!(
                                        "the `min_image_transfer_granularity` property of the \
                                    queue family of the command buffer is `[0; 3]`, but \
                                    `regions[{}].src_offset[{}]` is not 0",
                                        region_index, i,
                                    )
                                    .into(),
                                    vuids: &["VUID-VkCopyImageInfo2-srcOffset-01783"],
                                    ..Default::default()
                                }));
                            }

                            if src_offset[i] + extent[i] != src_subresource_extent[i] {
                                return Err(Box::new(ValidationError {
                                    context: "copy_image_info".into(),
                                    problem: format!(
                                        "the `min_image_transfer_granularity` property of the \
                                    queue family of the command buffer is `[0; 3]`, but \
                                    `regions[{0}].src_offset[{1}] + regions[{0}].extent[{1}]` \
                                    is not equal to coordinate {1} of the extent of the \
                                    subresource of `src_image` selected by \
                                    `regions[{0}].src_subresource`",
                                        region_index, i,
                                    )
                                    .into(),
                                    vuids: &["VUID-VkCopyImageInfo2-srcOffset-01783"],
                                    ..Default::default()
                                }));
                            }
                        }

                        /*
                           Check dst
                        */

                        for i in 0..3 {
                            if dst_offset[i] != 0 {
                                return Err(Box::new(ValidationError {
                                    context: "copy_image_info".into(),
                                    problem: format!(
                                        "the `min_image_transfer_granularity` property of the \
                                    queue family of the command buffer is `[0; 3]`, but \
                                    `regions[{}].dst_offset[{}]` is not 0",
                                        region_index, i,
                                    )
                                    .into(),
                                    vuids: &["VUID-VkCopyImageInfo2-dstOffset-01784"],
                                    ..Default::default()
                                }));
                            }

                            if dst_offset[i] + extent[i] != dst_subresource_extent[i] {
                                return Err(Box::new(ValidationError {
                                    context: "copy_image_info".into(),
                                    problem: format!(
                                        "the `min_image_transfer_granularity` property of the \
                                    queue family of the command buffer is `[0; 3]`, but \
                                    `regions[{0}].dst_offset[{1}] + regions[{0}].extent[{1}]` \
                                    is not equal to coordinate {1} of the extent of the \
                                    subresource of `dst_image` selected by \
                                    `regions[{0}].dst_subresource`",
                                        region_index, i,
                                    )
                                    .into(),
                                    vuids: &["VUID-VkCopyImageInfo2-dstOffset-01784"],
                                    ..Default::default()
                                }));
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn copy_image_unchecked(
        &mut self,
        copy_image_info: &CopyImageInfo<'_>,
    ) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3
            || self.device().enabled_extensions().khr_copy_commands2
        {
            let regions_vk = copy_image_info.to_vk2_regions();
            let copy_image_info_vk = copy_image_info.to_vk2(&regions_vk);

            if self.device().api_version() >= Version::V1_3 {
                unsafe { (fns.v1_3.cmd_copy_image2)(self.handle(), &copy_image_info_vk) };
            } else {
                unsafe {
                    (fns.khr_copy_commands2.cmd_copy_image2_khr)(self.handle(), &copy_image_info_vk)
                };
            }
        } else {
            let regions_vk = copy_image_info.to_vk_regions();
            let CopyImageInfoVk {
                src_image_vk,
                src_image_layout_vk,
                dst_image_vk,
                dst_image_layout_vk,
            } = copy_image_info.to_vk();

            unsafe {
                (fns.v1_0.cmd_copy_image)(
                    self.handle(),
                    src_image_vk,
                    src_image_layout_vk,
                    dst_image_vk,
                    dst_image_layout_vk,
                    regions_vk.len() as u32,
                    regions_vk.as_ptr(),
                )
            };
        }

        self
    }

    #[inline]
    pub unsafe fn copy_buffer_to_image(
        &mut self,
        copy_buffer_to_image_info: &CopyBufferToImageInfo<'_>,
    ) -> &mut Self {
        unsafe { self.try_copy_buffer_to_image(copy_buffer_to_image_info) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_copy_buffer_to_image(
        &mut self,
        copy_buffer_to_image_info: &CopyBufferToImageInfo<'_>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_copy_buffer_to_image(copy_buffer_to_image_info)?;

        Ok(unsafe { self.copy_buffer_to_image_unchecked(copy_buffer_to_image_info) })
    }

    pub(crate) fn validate_copy_buffer_to_image(
        &self,
        copy_buffer_to_image_info: &CopyBufferToImageInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        let queue_family_properties = self.queue_family_properties();

        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::TRANSFER | QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    transfer, graphics or compute operations"
                    .into(),
                vuids: &["VUID-vkCmdCopyBufferToImage2-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        copy_buffer_to_image_info
            .validate(self.device())
            .map_err(|err| err.add_context("copy_buffer_to_image_info"))?;

        let &CopyBufferToImageInfo {
            src_buffer: _,
            dst_image,
            dst_image_layout: _,
            regions,
            _ne,
        } = copy_buffer_to_image_info;

        let dst_image_format = dst_image.format();
        let dst_image_format_subsampled_extent = dst_image_format
            .ycbcr_chroma_sampling()
            .map_or(dst_image.extent(), |s| {
                s.subsampled_extent(dst_image.extent())
            });

        let min_image_transfer_granularity =
            // `[1; 3]` means the granularity is 1x1x1 texel, so we can ignore it.
            // Only check this if there are values greater than 1.
            (queue_family_properties.min_image_transfer_granularity != [1; 3]).then(|| {
                // `[0; 3]` means only the whole subresource can be copied.
                (queue_family_properties.min_image_transfer_granularity != [0; 3]).then(|| {
                    // Spec:
                    // "The value returned in minImageTransferGranularity has a unit of
                    // compressed texel blocks for images having a block-compressed format,
                    // and a unit of texels otherwise.""

                    if dst_image_format.compression().is_some() {
                        let granularity = queue_family_properties.min_image_transfer_granularity;
                        let block_extent = dst_image_format.block_extent();

                        [
                            granularity[0] * block_extent[0],
                            granularity[1] * block_extent[1],
                            granularity[2] * block_extent[2],
                        ]
                    } else {
                        queue_family_properties.min_image_transfer_granularity
                    }
                })
            });

        let queue_family_no_graphics = !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS);
        let queue_family_no_compute = !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::COMPUTE);

        if min_image_transfer_granularity.is_some() || queue_family_no_graphics {
            for (region_index, region) in regions.iter().enumerate() {
                let &BufferImageCopy {
                    buffer_offset,
                    buffer_row_length: _,
                    buffer_image_height: _,
                    image_subresource,
                    image_offset,
                    image_extent,
                    _ne,
                } = region;

                if queue_family_no_graphics {
                    if queue_family_no_compute && buffer_offset % 4 != 0 {
                        return Err(Box::new(ValidationError {
                            context: "create_info".into(),
                            problem: format!(
                                "the queue family of the command buffer does not support \
                                graphics or compute operations, but \
                                `regions[{}].buffer_offset` is not a multiple of 4",
                                region_index
                            )
                            .into(),
                            vuids: &["VUID-vkCmdCopyBufferToImage2-commandBuffer-07737"],
                            ..Default::default()
                        }));
                    }

                    if image_subresource
                        .aspects
                        .intersects(ImageAspects::DEPTH | ImageAspects::STENCIL)
                    {
                        return Err(Box::new(ValidationError {
                            context: "create_info".into(),
                            problem: format!(
                                "the queue family of the command buffer does not support \
                                graphics operations, but \
                                `regions[{}].image_subresource.aspects` contains \
                                `ImageAspects::DEPTH` or `ImageAspects::STENCIL`",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-vkCmdCopyBufferToImage2-commandBuffer-07739"],
                            ..Default::default()
                        }));
                    }
                }

                if let Some(min_image_transfer_granularity) = &min_image_transfer_granularity {
                    let mut image_subresource_extent =
                        mip_level_extent(dst_image.extent(), image_subresource.mip_level).unwrap();

                    if matches!(
                        image_subresource.aspects,
                        ImageAspects::PLANE_1 | ImageAspects::PLANE_2
                    ) {
                        image_subresource_extent = dst_image_format_subsampled_extent;
                    }

                    if let Some(dst_granularity) = &min_image_transfer_granularity {
                        for i in 0..3 {
                            if image_offset[i] % dst_granularity[i] != 0 {
                                return Err(Box::new(ValidationError {
                                    context: "copy_image_info".into(),
                                    problem: format!(
                                        "the `min_image_transfer_granularity` property of the \
                                        queue family of the command buffer is not `[0; 3]`, but \
                                        `regions[{}].image_offset[{1}]` is not a multiple of \
                                        `min_image_transfer_granularity[{1}]` texel blocks",
                                        region_index, i,
                                    )
                                    .into(),
                                    vuids: &["VUID-vkCmdCopyBufferToImage2-imageOffset-07738"],
                                    ..Default::default()
                                }));
                            }

                            if image_offset[i] + image_extent[i] != image_subresource_extent[i]
                                && image_extent[i] % dst_granularity[i] != 0
                            {
                                return Err(Box::new(ValidationError {
                                    context: "copy_image_info".into(),
                                    problem: format!(
                                        "the `min_image_transfer_granularity` property of the \
                                        queue family of the command buffer is not `[0; 3]`, and \
                                        `regions[{0}].image_offset[{1}] + \
                                        regions[{0}].image_extent[{1}]` \
                                        is not equal to coordinate {1} of the extent of the \
                                        subresource of `dst_image` selected by \
                                        `regions[{0}].image_subresource`, but \
                                        `regions[{}].image_extent[{1}]` is not a multiple of \
                                        `min_image_transfer_granularity[{1}]` texel blocks",
                                        region_index, i,
                                    )
                                    .into(),
                                    vuids: &["VUID-vkCmdCopyBufferToImage2-imageOffset-07738"],
                                    ..Default::default()
                                }));
                            }
                        }
                    } else {
                        for i in 0..3 {
                            if image_offset[i] != 0 {
                                return Err(Box::new(ValidationError {
                                    context: "copy_image_info".into(),
                                    problem: format!(
                                        "the `min_image_transfer_granularity` property of the \
                                        queue family of the command buffer is `[0; 3]`, but \
                                        `regions[{}].image_offset[{}]` is not 0",
                                        region_index, i,
                                    )
                                    .into(),
                                    vuids: &["VUID-vkCmdCopyBufferToImage2-imageOffset-07738"],
                                    ..Default::default()
                                }));
                            }

                            if image_offset[i] + image_extent[i] != image_subresource_extent[i] {
                                return Err(Box::new(ValidationError {
                                    context: "copy_image_info".into(),
                                    problem: format!(
                                        "the `min_image_transfer_granularity` property of the \
                                        queue family of the command buffer is `[0; 3]`, but \
                                        `regions[{0}].image_offset[{1}] + \
                                        regions[{0}].image_extent[{1}]` \
                                        is not equal to coordinate {1} of the extent of the \
                                        subresource of `dst_image` selected by \
                                        `regions[{0}].image_subresource`",
                                        region_index, i,
                                    )
                                    .into(),
                                    vuids: &["VUID-vkCmdCopyBufferToImage2-imageOffset-07738"],
                                    ..Default::default()
                                }));
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn copy_buffer_to_image_unchecked(
        &mut self,
        copy_buffer_to_image_info: &CopyBufferToImageInfo<'_>,
    ) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3
            || self.device().enabled_extensions().khr_copy_commands2
        {
            let regions_vk = copy_buffer_to_image_info.to_vk2_regions();
            let copy_buffer_to_image_info_vk = copy_buffer_to_image_info.to_vk2(&regions_vk);

            if self.device().api_version() >= Version::V1_3 {
                unsafe {
                    (fns.v1_3.cmd_copy_buffer_to_image2)(
                        self.handle(),
                        &copy_buffer_to_image_info_vk,
                    )
                };
            } else {
                unsafe {
                    (fns.khr_copy_commands2.cmd_copy_buffer_to_image2_khr)(
                        self.handle(),
                        &copy_buffer_to_image_info_vk,
                    )
                };
            }
        } else {
            let regions_vk = copy_buffer_to_image_info.to_vk_regions();
            let CopyBufferToImageInfoVk {
                src_buffer_vk,
                dst_image_vk,
                dst_image_layout_vk,
            } = copy_buffer_to_image_info.to_vk();

            unsafe {
                (fns.v1_0.cmd_copy_buffer_to_image)(
                    self.handle(),
                    src_buffer_vk,
                    dst_image_vk,
                    dst_image_layout_vk,
                    regions_vk.len() as u32,
                    regions_vk.as_ptr(),
                )
            };
        }

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn copy_image_to_buffer(
        &mut self,
        copy_image_to_buffer_info: &CopyImageToBufferInfo<'_>,
    ) -> &mut Self {
        unsafe { self.try_copy_image_to_buffer(copy_image_to_buffer_info) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_copy_image_to_buffer(
        &mut self,
        copy_image_to_buffer_info: &CopyImageToBufferInfo<'_>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_copy_image_to_buffer(copy_image_to_buffer_info)?;

        Ok(unsafe { self.copy_image_to_buffer_unchecked(copy_image_to_buffer_info) })
    }

    pub(crate) fn validate_copy_image_to_buffer(
        &self,
        copy_image_to_buffer_info: &CopyImageToBufferInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        let queue_family_properties = self.queue_family_properties();

        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::TRANSFER | QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    transfer, graphics or compute operations"
                    .into(),
                vuids: &["VUID-vkCmdCopyImageToBuffer2-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        copy_image_to_buffer_info
            .validate(self.device())
            .map_err(|err| err.add_context("copy_image_to_buffer_info"))?;

        let &CopyImageToBufferInfo {
            src_image,
            src_image_layout: _,
            dst_buffer: _,
            regions,
            _ne,
        } = copy_image_to_buffer_info;

        let src_image_format = src_image.format();
        let src_image_format_subsampled_extent = src_image_format
            .ycbcr_chroma_sampling()
            .map_or(src_image.extent(), |s| {
                s.subsampled_extent(src_image.extent())
            });

        let min_image_transfer_granularity =
            // `[1; 3]` means the granularity is 1x1x1 texel, so we can ignore it.
            // Only check this if there are values greater than 1.
            (queue_family_properties.min_image_transfer_granularity != [1; 3]).then(|| {
                // `[0; 3]` means only the whole subresource can be copied.
                (queue_family_properties.min_image_transfer_granularity != [0; 3]).then(|| {
                    // Spec:
                    // "The value returned in minImageTransferGranularity has a unit of
                    // compressed texel blocks for images having a block-compressed format,
                    // and a unit of texels otherwise.""

                    if src_image_format.compression().is_some() {
                        let granularity = queue_family_properties.min_image_transfer_granularity;
                        let block_extent = src_image_format.block_extent();

                        [
                            granularity[0] * block_extent[0],
                            granularity[1] * block_extent[1],
                            granularity[2] * block_extent[2],
                        ]
                    } else {
                        queue_family_properties.min_image_transfer_granularity
                    }
                })
            });

        let queue_family_no_graphics = !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS);
        let queue_family_no_compute = !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::COMPUTE);

        if min_image_transfer_granularity.is_some() || queue_family_no_graphics {
            for (region_index, region) in regions.iter().enumerate() {
                let &BufferImageCopy {
                    buffer_offset,
                    buffer_row_length: _,
                    buffer_image_height: _,
                    image_subresource,
                    image_offset,
                    image_extent,
                    _ne,
                } = region;

                if queue_family_no_graphics && queue_family_no_compute && buffer_offset % 4 != 0 {
                    return Err(Box::new(ValidationError {
                        context: "create_info".into(),
                        problem: format!(
                            "the queue family of the command buffer does not support graphics or \
                            compute operations, but `regions[{}].buffer_offset` is not a multiple \
                            of 4",
                            region_index,
                        )
                        .into(),
                        vuids: &["VUID-vkCmdCopyImageToBuffer2-commandBuffer-07746"],
                        ..Default::default()
                    }));
                }

                if let Some(min_image_transfer_granularity) = &min_image_transfer_granularity {
                    let mut image_subresource_extent =
                        mip_level_extent(src_image.extent(), image_subresource.mip_level).unwrap();

                    if matches!(
                        image_subresource.aspects,
                        ImageAspects::PLANE_1 | ImageAspects::PLANE_2
                    ) {
                        image_subresource_extent = src_image_format_subsampled_extent;
                    }

                    if let Some(src_granularity) = &min_image_transfer_granularity {
                        for i in 0..3 {
                            if image_offset[i] % src_granularity[i] != 0 {
                                return Err(Box::new(ValidationError {
                                    context: "copy_image_info".into(),
                                    problem: format!(
                                        "the `min_image_transfer_granularity` property of the \
                                        queue family of the command buffer is not `[0; 3]`, but \
                                        `regions[{}].image_offset[{1}]` is not a multiple of \
                                        `min_image_transfer_granularity[{1}]` texel blocks",
                                        region_index, i,
                                    )
                                    .into(),
                                    vuids: &["VUID-vkCmdCopyImageToBuffer2-imageOffset-07747"],
                                    ..Default::default()
                                }));
                            }

                            if image_offset[i] + image_extent[i] != image_subresource_extent[i]
                                && image_extent[i] % src_granularity[i] != 0
                            {
                                return Err(Box::new(ValidationError {
                                    context: "copy_image_info".into(),
                                    problem: format!(
                                        "the `min_image_transfer_granularity` property of the \
                                        queue family of the command buffer is not `[0; 3]`, and \
                                        `regions[{0}].image_offset[{1}] + \
                                        regions[{0}].image_extent[{1}]` \
                                        is not equal to coordinate {1} of the extent of the \
                                        subresource of `src_image` selected by \
                                        `regions[{0}].image_subresource`, but \
                                        `regions[{}].image_extent[{1}]` is not a multiple of \
                                        `min_image_transfer_granularity[{1}]` texel blocks",
                                        region_index, i,
                                    )
                                    .into(),
                                    vuids: &["VUID-vkCmdCopyImageToBuffer2-imageOffset-07747"],
                                    ..Default::default()
                                }));
                            }
                        }
                    } else {
                        for i in 0..3 {
                            if image_offset[i] != 0 {
                                return Err(Box::new(ValidationError {
                                    context: "copy_image_info".into(),
                                    problem: format!(
                                        "the `min_image_transfer_granularity` property of the \
                                        queue family of the command buffer is `[0; 3]`, but \
                                        `regions[{}].image_offset[{}]` is not 0",
                                        region_index, i,
                                    )
                                    .into(),
                                    vuids: &["VUID-vkCmdCopyImageToBuffer2-imageOffset-07747"],
                                    ..Default::default()
                                }));
                            }

                            if image_offset[i] + image_extent[i] != image_subresource_extent[i] {
                                return Err(Box::new(ValidationError {
                                    context: "copy_image_info".into(),
                                    problem: format!(
                                        "the `min_image_transfer_granularity` property of the \
                                        queue family of the command buffer is `[0; 3]`, but \
                                        `regions[{0}].image_offset[{1}] + \
                                        regions[{0}].image_extent[{1}]` \
                                        is not equal to coordinate {1} of the extent of the \
                                        subresource of `src_image` selected by \
                                        `regions[{0}].image_subresource`",
                                        region_index, i,
                                    )
                                    .into(),
                                    vuids: &["VUID-vkCmdCopyImageToBuffer2-imageOffset-07747"],
                                    ..Default::default()
                                }));
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn copy_image_to_buffer_unchecked(
        &mut self,
        copy_image_to_buffer_info: &CopyImageToBufferInfo<'_>,
    ) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3
            || self.device().enabled_extensions().khr_copy_commands2
        {
            let regions_vk = copy_image_to_buffer_info.to_vk2_regions();
            let copy_image_to_buffer_info_vk = copy_image_to_buffer_info.to_vk2(&regions_vk);

            if self.device().api_version() >= Version::V1_3 {
                unsafe {
                    (fns.v1_3.cmd_copy_image_to_buffer2)(
                        self.handle(),
                        &copy_image_to_buffer_info_vk,
                    )
                };
            } else {
                unsafe {
                    (fns.khr_copy_commands2.cmd_copy_image_to_buffer2_khr)(
                        self.handle(),
                        &copy_image_to_buffer_info_vk,
                    )
                };
            }
        } else {
            let regions_vk = copy_image_to_buffer_info.to_vk_regions();
            let CopyImageToBufferInfoVk {
                src_image_vk,
                src_image_layout_vk,
                dst_buffer_vk,
            } = copy_image_to_buffer_info.to_vk();

            unsafe {
                (fns.v1_0.cmd_copy_image_to_buffer)(
                    self.handle(),
                    src_image_vk,
                    src_image_layout_vk,
                    dst_buffer_vk,
                    regions_vk.len() as u32,
                    regions_vk.as_ptr(),
                )
            };
        }

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn blit_image(&mut self, blit_image_info: &BlitImageInfo<'_>) -> &mut Self {
        unsafe { self.try_blit_image(blit_image_info) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_blit_image(
        &mut self,
        blit_image_info: &BlitImageInfo<'_>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_blit_image(blit_image_info)?;

        Ok(unsafe { self.blit_image_unchecked(blit_image_info) })
    }

    pub(crate) fn validate_blit_image(
        &self,
        blit_image_info: &BlitImageInfo<'_>,
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
                vuids: &["VUID-vkCmdBlitImage2-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        blit_image_info
            .validate(self.device())
            .map_err(|err| err.add_context("blit_image_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn blit_image_unchecked(
        &mut self,
        blit_image_info: &BlitImageInfo<'_>,
    ) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3
            || self.device().enabled_extensions().khr_copy_commands2
        {
            let regions_vk = blit_image_info.to_vk2_regions();
            let blit_image_info_vk = blit_image_info.to_vk2(&regions_vk);

            if self.device().api_version() >= Version::V1_3 {
                unsafe { (fns.v1_3.cmd_blit_image2)(self.handle(), &blit_image_info_vk) };
            } else {
                unsafe {
                    (fns.khr_copy_commands2.cmd_blit_image2_khr)(self.handle(), &blit_image_info_vk)
                };
            }
        } else {
            let regions_vk = blit_image_info.to_vk_regions();
            let BlitImageInfoVk {
                src_image_vk,
                src_image_layout_vk,
                dst_image_vk,
                dst_image_layout_vk,
                filter_vk,
            } = blit_image_info.to_vk();

            unsafe {
                (fns.v1_0.cmd_blit_image)(
                    self.handle(),
                    src_image_vk,
                    src_image_layout_vk,
                    dst_image_vk,
                    dst_image_layout_vk,
                    regions_vk.len() as u32,
                    regions_vk.as_ptr(),
                    filter_vk,
                )
            };
        }

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn resolve_image(&mut self, resolve_image_info: &ResolveImageInfo<'_>) -> &mut Self {
        unsafe { self.try_resolve_image(resolve_image_info) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_resolve_image(
        &mut self,
        resolve_image_info: &ResolveImageInfo<'_>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_resolve_image(resolve_image_info)?;

        Ok(unsafe { self.resolve_image_unchecked(resolve_image_info) })
    }

    pub(crate) fn validate_resolve_image(
        &self,
        resolve_image_info: &ResolveImageInfo<'_>,
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
                vuids: &["VUID-vkCmdResolveImage2-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        resolve_image_info
            .validate(self.device())
            .map_err(|err| err.add_context("resolve_image_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn resolve_image_unchecked(
        &mut self,
        resolve_image_info: &ResolveImageInfo<'_>,
    ) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3
            || self.device().enabled_extensions().khr_copy_commands2
        {
            let regions_vk = resolve_image_info.to_vk2_regions();
            let resolve_image_info_vk = resolve_image_info.to_vk2(&regions_vk);

            if self.device().api_version() >= Version::V1_3 {
                unsafe { (fns.v1_3.cmd_resolve_image2)(self.handle(), &resolve_image_info_vk) };
            } else {
                unsafe {
                    (fns.khr_copy_commands2.cmd_resolve_image2_khr)(
                        self.handle(),
                        &resolve_image_info_vk,
                    )
                };
            }
        } else {
            let regions_vk = resolve_image_info.to_vk_regions();
            let ResolveImageInfoVk {
                src_image_vk,
                src_image_layout_vk,
                dst_image_vk,
                dst_image_layout_vk,
            } = resolve_image_info.to_vk();

            unsafe {
                (fns.v1_0.cmd_resolve_image)(
                    self.handle(),
                    src_image_vk,
                    src_image_layout_vk,
                    dst_image_vk,
                    dst_image_layout_vk,
                    regions_vk.len() as u32,
                    regions_vk.as_ptr(),
                )
            };
        }

        self
    }
}

/// Parameters to copy data from a buffer to another buffer.
///
/// The fields of `regions` represent bytes.
#[derive(Clone, Debug)]
pub struct CopyBufferInfo<'a> {
    /// The buffer to copy from.
    ///
    /// There is no default value.
    pub src_buffer: &'a Buffer,

    /// The buffer to copy to.
    ///
    /// There is no default value.
    pub dst_buffer: &'a Buffer,

    /// The regions of both buffers to copy between, specified in bytes.
    ///
    /// The default value is a single region, with zero offsets and a `size` equal to the smallest
    /// of the two buffers.
    pub regions: &'a [BufferCopy<'a>],

    pub _ne: crate::NonExhaustive<'a>,
}

impl<'a> CopyBufferInfo<'a> {
    /// Returns a default `CopyBufferInfo` with the provided `src_buffer` and `dst_buffer`.
    #[inline]
    pub const fn new(src_buffer: &'a Buffer, dst_buffer: &'a Buffer) -> Self {
        Self {
            src_buffer,
            dst_buffer,
            regions: &[],
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            src_buffer,
            dst_buffer,
            regions,
            _ne: _,
        } = self;

        // VUID-VkCopyBufferInfo2-commonparent
        assert_eq!(device, src_buffer.device().as_ref());
        assert_eq!(device, dst_buffer.device().as_ref());

        if !src_buffer.usage().intersects(BufferUsage::TRANSFER_SRC) {
            return Err(Box::new(ValidationError {
                context: "src_buffer.usage()".into(),
                problem: "does not contain `BufferUsage::TRANSFER_SRC`".into(),
                vuids: &["VUID-VkCopyBufferInfo2-srcBuffer-00118"],
                ..Default::default()
            }));
        }

        if !dst_buffer.usage().intersects(BufferUsage::TRANSFER_DST) {
            return Err(Box::new(ValidationError {
                context: "dst_buffer.usage()".into(),
                problem: "does not contain `BufferUsage::TRANSFER_DST`".into(),
                vuids: &["VUID-VkCopyBufferInfo2-dstBuffer-00120"],
                ..Default::default()
            }));
        }

        let same_buffer = src_buffer == dst_buffer;
        let mut overlap_indices = None;

        for (region_index, region) in regions.iter().enumerate() {
            region
                .validate(device)
                .map_err(|err| err.add_context(format!("regions[{}]", region_index)))?;

            let &BufferCopy {
                src_offset,
                dst_offset,
                size,
                _ne: _,
            } = region;

            if src_offset
                .checked_add(size)
                .is_none_or(|end| end > src_buffer.size())
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].src_offset + regions[{0}].size` is greater than \
                        `src_buffer.size()`",
                        region_index,
                    )
                    .into(),
                    vuids: &[
                        "VUID-VkCopyBufferInfo2-srcOffset-00113",
                        "VUID-VkCopyBufferInfo2-size-00115",
                    ],
                    ..Default::default()
                }));
            }

            if dst_offset
                .checked_add(size)
                .is_none_or(|end| end > dst_buffer.size())
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].dst_offset + regions[{0}].size` is greater than \
                        `dst_buffer.size()`",
                        region_index,
                    )
                    .into(),
                    vuids: &[
                        "VUID-VkCopyBufferInfo2-dstOffset-00114",
                        "VUID-VkCopyBufferInfo2-size-00116",
                    ],
                    ..Default::default()
                }));
            }

            // VUID-VkCopyBufferInfo2-pRegions-00117
            if same_buffer {
                let src_region_index = region_index;
                let src_range = src_offset..src_offset + size;

                for (dst_region_index, dst_region) in regions.iter().enumerate() {
                    let &BufferCopy { dst_offset, .. } = dst_region;

                    let dst_range = dst_offset..dst_offset + size;

                    if src_range.start >= dst_range.end || dst_range.start >= src_range.end {
                        // The regions do not overlap
                        continue;
                    }

                    overlap_indices = Some((src_region_index, dst_region_index));
                }
            }
        }

        if let Some((src_region_index, dst_region_index)) = overlap_indices {
            return Err(Box::new(ValidationError {
                problem: format!(
                    "`src_buffer` is equal to `dst_buffer`, and the source of `regions[{}]` \
                    overlaps with the destination of `regions[{}]`",
                    src_region_index, dst_region_index,
                )
                .into(),
                vuids: &["VUID-VkCopyBufferInfo2-pRegions-00117"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk2(
        &self,
        regions_vk: &'a [vk::BufferCopy2<'static>],
    ) -> vk::CopyBufferInfo2<'a> {
        let &Self {
            src_buffer,
            dst_buffer,
            regions: _,
            _ne: _,
        } = self;

        vk::CopyBufferInfo2::default()
            .src_buffer(src_buffer.handle())
            .dst_buffer(dst_buffer.handle())
            .regions(regions_vk)
    }

    pub(crate) fn to_vk2_regions(&self) -> SmallVec<[vk::BufferCopy2<'static>; 8]> {
        let &Self {
            src_buffer,
            dst_buffer,
            regions,
            _ne: _,
        } = self;

        if regions.is_empty() {
            let region_vk = vk::BufferCopy2::default()
                .src_offset(0)
                .dst_offset(0)
                .size(min(src_buffer.size(), dst_buffer.size()));

            smallvec![region_vk]
        } else {
            regions.iter().map(BufferCopy::to_vk2).collect()
        }
    }

    pub(crate) fn to_vk(&self) -> CopyBufferInfoVk {
        let &Self {
            src_buffer,
            dst_buffer,
            regions: _,
            _ne: _,
        } = self;

        CopyBufferInfoVk {
            src_buffer_vk: src_buffer.handle(),
            dst_buffer_vk: dst_buffer.handle(),
        }
    }

    pub(crate) fn to_vk_regions(&self) -> SmallVec<[vk::BufferCopy; 8]> {
        let &Self {
            src_buffer,
            dst_buffer,
            regions,
            _ne: _,
        } = self;

        if regions.is_empty() {
            let region_vk = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: min(src_buffer.size(), dst_buffer.size()),
            };

            smallvec![region_vk]
        } else {
            regions.iter().map(BufferCopy::to_vk).collect()
        }
    }
}

pub(crate) struct CopyBufferInfoVk {
    pub(crate) src_buffer_vk: vk::Buffer,
    pub(crate) dst_buffer_vk: vk::Buffer,
}

/// A region of data to copy between buffers.
#[derive(Clone, Debug)]
pub struct BufferCopy<'a> {
    /// The offset in bytes or elements from the start of `src_buffer` that copying will
    /// start from.
    ///
    /// The default value is `0`.
    pub src_offset: DeviceSize,

    /// The offset in bytes or elements from the start of `dst_buffer` that copying will
    /// start from.
    ///
    /// The default value is `0`.
    pub dst_offset: DeviceSize,

    /// The number of bytes or elements to copy.
    ///
    /// The default value is `0`, which must be overridden.
    pub size: DeviceSize,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for BufferCopy<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl BufferCopy<'_> {
    /// Returns a default `BufferCopy`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            src_offset: 0,
            dst_offset: 0,
            size: 0,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, _device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            src_offset: _,
            dst_offset: _,
            size,
            _ne: _,
        } = self;

        if size == 0 {
            return Err(Box::new(ValidationError {
                context: "size".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkBufferCopy2-size-01988"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk2(&self) -> vk::BufferCopy2<'static> {
        let &Self {
            src_offset,
            dst_offset,
            size,
            _ne,
        } = self;

        vk::BufferCopy2::default()
            .src_offset(src_offset)
            .dst_offset(dst_offset)
            .size(size)
    }

    pub(crate) fn to_vk(&self) -> vk::BufferCopy {
        let &Self {
            src_offset,
            dst_offset,
            size,
            _ne,
        } = self;

        vk::BufferCopy {
            src_offset,
            dst_offset,
            size,
        }
    }
}

/// Parameters to copy data from an image to another image.
#[derive(Clone, Debug)]
pub struct CopyImageInfo<'a> {
    /// The image to copy from.
    ///
    /// There is no default value.
    pub src_image: &'a Image,

    /// The layout used for `src_image` during the copy operation.
    ///
    /// The following layouts are allowed:
    /// - [`ImageLayout::TransferSrcOptimal`]
    /// - [`ImageLayout::General`]
    ///
    /// The default value is [`ImageLayout::TransferSrcOptimal`].
    pub src_image_layout: ImageLayout,

    /// The image to copy to.
    ///
    /// There is no default value.
    pub dst_image: &'a Image,

    /// The layout used for `dst_image` during the copy operation.
    ///
    /// The following layouts are allowed:
    /// - [`ImageLayout::TransferDstOptimal`]
    /// - [`ImageLayout::General`]
    ///
    /// The default value is [`ImageLayout::TransferDstOptimal`].
    pub dst_image_layout: ImageLayout,

    /// The regions of both images to copy between.
    ///
    /// The default value is a single region, covering the first mip level, and the smallest of the
    /// array layers and extent of the two images. All aspects of each image are selected, or
    /// `plane0` if the image is multi-planar.
    pub regions: &'a [ImageCopy<'a>],

    pub _ne: crate::NonExhaustive<'a>,
}

impl<'a> CopyImageInfo<'a> {
    /// Returns a default `CopyImageInfo` with the provided `src_image` and `dst_image`.
    #[inline]
    pub fn new(src_image: &'a Image, dst_image: &'a Image) -> Self {
        Self {
            src_image,
            src_image_layout: ImageLayout::TransferSrcOptimal,
            dst_image,
            dst_image_layout: ImageLayout::TransferDstOptimal,
            regions: &[],
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            src_image,
            src_image_layout,
            dst_image,
            dst_image_layout,
            regions,
            _ne: _,
        } = self;

        src_image_layout.validate_device(device).map_err(|err| {
            err.add_context("src_image_layout")
                .set_vuids(&["VUID-VkCopyImageInfo2-srcImageLayout-parameter"])
        })?;

        dst_image_layout.validate_device(device).map_err(|err| {
            err.add_context("dst_image_layout")
                .set_vuids(&["VUID-VkCopyImageInfo2-dstImageLayout-parameter"])
        })?;

        // VUID-VkCopyImageInfo2-commonparent
        assert_eq!(device, src_image.device().as_ref());
        assert_eq!(device, dst_image.device().as_ref());

        let src_image_format = src_image.format();
        let src_image_format_aspects = src_image_format.aspects();
        let src_image_format_planes = src_image_format.planes();
        let src_image_format_subsampled_extent = src_image_format
            .ycbcr_chroma_sampling()
            .map_or(src_image.extent(), |s| {
                s.subsampled_extent(src_image.extent())
            });

        let dst_image_format = dst_image.format();
        let dst_image_format_aspects = dst_image_format.aspects();
        let dst_image_format_planes = dst_image_format.planes();
        let dst_image_format_subsampled_extent = dst_image_format
            .ycbcr_chroma_sampling()
            .map_or(dst_image.extent(), |s| {
                s.subsampled_extent(dst_image.extent())
            });

        if device.api_version() >= Version::V1_1 || device.enabled_extensions().khr_maintenance1 {
            if !src_image
                .format_features()
                .intersects(FormatFeatures::TRANSFER_SRC)
            {
                return Err(Box::new(ValidationError {
                    context: "src_image.format_features()".into(),
                    problem: "does not contain `FormatFeatures::TRANSFER_SRC`".into(),
                    vuids: &["VUID-VkCopyImageInfo2-srcImage-01995"],
                    ..Default::default()
                }));
            }

            if !dst_image
                .format_features()
                .intersects(FormatFeatures::TRANSFER_DST)
            {
                return Err(Box::new(ValidationError {
                    context: "dst_image.format_features()".into(),
                    problem: "does not contain `FormatFeatures::TRANSFER_DST`".into(),
                    vuids: &["VUID-VkCopyImageInfo2-dstImage-01996"],
                    ..Default::default()
                }));
            }
        }

        if src_image.samples() != dst_image.samples() {
            return Err(Box::new(ValidationError {
                problem: "`src_image.samples()` does not equal `dst_image.samples()`".into(),
                vuids: &["VUID-VkCopyImageInfo2-srcImage-00136"],
                ..Default::default()
            }));
        }

        if !matches!(
            src_image_layout,
            ImageLayout::TransferSrcOptimal | ImageLayout::General
        ) {
            return Err(Box::new(ValidationError {
                context: "src_image_layout".into(),
                problem: "is not `ImageLayout::TransferSrcOptimal` or `ImageLayout::General`"
                    .into(),
                vuids: &["VUID-VkCopyImageInfo2-srcImageLayout-01917"],
                ..Default::default()
            }));
        }

        if !matches!(
            dst_image_layout,
            ImageLayout::TransferDstOptimal | ImageLayout::General
        ) {
            return Err(Box::new(ValidationError {
                context: "dst_image_layout".into(),
                problem: "is not `ImageLayout::TransferDstOptimal` or `ImageLayout::General`"
                    .into(),
                vuids: &["VUID-VkCopyImageInfo2-dstImageLayout-01395"],
                ..Default::default()
            }));
        }

        if src_image.image_type() != dst_image.image_type() {
            if !(matches!(src_image.image_type(), ImageType::Dim2d | ImageType::Dim3d)
                && matches!(dst_image.image_type(), ImageType::Dim2d | ImageType::Dim3d))
            {
                return Err(Box::new(ValidationError {
                    problem: "`src_image.image_type()` does not equal `dst_image.image_type()`, \
                        but they are not both `ImageType::Dim2d` or `ImageType::Dim3d`"
                        .into(),
                    vuids: &["VUID-VkCopyImageInfo2-srcImage-07743"],
                    ..Default::default()
                }));
            }

            if !(device.api_version() >= Version::V1_1
                || device.enabled_extensions().khr_maintenance1)
            {
                return Err(Box::new(ValidationError {
                    problem: "`src_image.image_type()` does not equal `dst_image.image_type()`, \
                        and are both `ImageType::Dim2d` or `ImageType::Dim3d`"
                        .into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                        RequiresAllOf(&[Requires::DeviceExtension("khr_maintenance1")]),
                    ]),
                    vuids: &["VUID-VkCopyImageInfo2-apiVersion-07933"],
                    ..Default::default()
                }));
            }
        }

        let is_same_image = src_image == dst_image;
        let mut overlap_subresource_indices = None;
        let mut overlap_extent_indices = None;

        for (region_index, region) in regions.iter().enumerate() {
            region
                .validate(device)
                .map_err(|err| err.add_context(format!("regions[{}]", region_index)))?;

            let &ImageCopy {
                src_subresource,
                src_offset,
                dst_subresource,
                dst_offset,
                extent,
                _ne,
            } = region;

            /*
               Check src
            */

            if src_subresource.mip_level >= src_image.mip_levels() {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].src_subresource.mip_level` is not less than \
                        `src_image.mip_levels()`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageInfo2-srcSubresource-07967"],
                    ..Default::default()
                }));
            }

            let mut src_subresource_format = src_image_format;
            let mut src_subresource_extent =
                mip_level_extent(src_image.extent(), src_subresource.mip_level).unwrap();

            if src_image_format_planes.is_empty() {
                if !src_image_format_aspects.contains(src_subresource.aspects) {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`regions[{}].src_subresource.aspects` is not a subset of \
                            `src_image.format().aspects()`",
                            region_index,
                        )
                        .into(),
                        vuids: &["VUID-VkCopyImageInfo2-aspectMask-00142"],
                        ..Default::default()
                    }));
                }
            } else if src_image_format_planes.len() == 2 {
                match src_subresource.aspects {
                    ImageAspects::PLANE_0 => {
                        src_subresource_format = src_image_format_planes[0];
                    }
                    ImageAspects::PLANE_1 => {
                        src_subresource_format = src_image_format_planes[1];
                        src_subresource_extent = src_image_format_subsampled_extent;
                    }
                    _ => {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.format()` is a multi-planar format with two planes, \
                                but `regions[{}].src_subresource.aspect` is not \
                                `ImageAspects::PLANE_0` or `ImageAspects::PLANE_1`",
                                region_index,
                            )
                            .into(),
                            vuids: &[
                                "VUID-VkCopyImageInfo2-srcImage-08713",
                                "VUID-VkCopyImageInfo2-aspectMask-00142",
                            ],
                            ..Default::default()
                        }));
                    }
                }
            } else if src_image_format_planes.len() == 3 {
                match src_subresource.aspects {
                    ImageAspects::PLANE_0 => {
                        src_subresource_format = src_image_format_planes[0];
                    }
                    ImageAspects::PLANE_1 => {
                        src_subresource_format = src_image_format_planes[1];
                        src_subresource_extent = src_image_format_subsampled_extent;
                    }
                    ImageAspects::PLANE_2 => {
                        src_subresource_format = src_image_format_planes[2];
                        src_subresource_extent = src_image_format_subsampled_extent;
                    }
                    _ => {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.format()` is a multi-planar format with three planes, \
                                but `regions[{}].src_subresource.aspect` is not \
                                `ImageAspects::PLANE_0`, `ImageAspects::PLANE_1` or \
                                `ImageAspects::PLANE_2`",
                                region_index,
                            )
                            .into(),
                            vuids: &[
                                "VUID-VkCopyImageInfo2-srcImage-08713",
                                "VUID-VkCopyImageInfo2-aspectMask-00142",
                            ],
                            ..Default::default()
                        }));
                    }
                }
            }

            match src_image.image_type() {
                ImageType::Dim1d => {
                    if src_offset[1] != 0 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].src_offset[1]` is not 0",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyImageInfo2-srcImage-00146"],
                            ..Default::default()
                        }));
                    }

                    if extent[1] != 1 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].extent[1]` is not 1",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyImageInfo2-srcImage-00146"],
                            ..Default::default()
                        }));
                    }

                    if src_offset[2] != 0 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].src_offset[2]` is not 0",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyImageInfo2-srcImage-01785"],
                            ..Default::default()
                        }));
                    }

                    if extent[2] != 1 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].extent[2]` is not 1",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyImageInfo2-srcImage-01785"],
                            ..Default::default()
                        }));
                    }
                }
                ImageType::Dim2d => {
                    if src_offset[2] != 0 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.image_type()` is `ImageType::Dim2d`, but \
                                `regions[{}].src_offset[2]` is not 0",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyImageInfo2-srcImage-01787"],
                            ..Default::default()
                        }));
                    }
                }
                ImageType::Dim3d => {
                    if !(src_subresource.base_array_layer == 0
                        && src_subresource.layer_count == Some(1))
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.image_type()` is `ImageType::Dim3d`, but \
                                `regions[{0}].src_subresource.base_array_layer` and \
                                `regions[{0}].src_subresource.layer_count` are not `0` and \
                                `Some(1)`, respectively",
                                region_index,
                            )
                            .into(),
                            vuids: &[
                                "VUID-VkCopyImageInfo2-srcImage-04443",
                                "VUID-VkCopyImageInfo2-apiVersion-07932",
                            ],
                            ..Default::default()
                        }));
                    }
                }
            }

            if src_subresource.base_array_layer >= src_image.array_layers() {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].src_subresource.base_array_layer` is not less than \
                        `src_image.array_layers()`",
                        region_index,
                    )
                    .into(),
                    // vuids?
                    ..Default::default()
                }));
            }

            if let Some(src_subresource_layer_count) = src_subresource.layer_count {
                if src_subresource_layer_count
                    > src_image.array_layers() - src_subresource.base_array_layer
                {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`regions[{0}].src_subresource.base_array_layer + \
                            regions[{0}].src_subresource.layer_count` is greater than \
                            `src_image.array_layers()`",
                            region_index,
                        )
                        .into(),
                        vuids: &["VUID-VkCopyImageInfo2-srcSubresource-07968"],
                        ..Default::default()
                    }));
                }
            }

            if src_offset[0]
                .checked_add(extent[0])
                .is_none_or(|end| end > src_subresource_extent[0])
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].src_offset[0] + regions[{0}].extent[0]` is greater \
                        than coordinate 0 of the extent of the subresource of `src_image` \
                        selected by `regions[{0}].src_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageInfo2-srcOffset-00144"],
                    ..Default::default()
                }));
            }

            if src_offset[1]
                .checked_add(extent[1])
                .is_none_or(|end| end > src_subresource_extent[1])
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].src_offset[1] + regions[{0}].extent[1]` is greater \
                        than coordinate 1 of the extent of the subresource of `src_image` \
                        selected by `regions[{0}].src_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageInfo2-srcOffset-00145"],
                    ..Default::default()
                }));
            }

            if src_offset[2]
                .checked_add(extent[2])
                .is_none_or(|end| end > src_subresource_extent[2])
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].src_offset[2] + regions[{0}].extent[2]` is greater \
                        than coordinate 2 of the extent of the subresource of `src_image` \
                        selected by `regions[{0}].src_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageInfo2-srcOffset-00147"],
                    ..Default::default()
                }));
            }

            let src_subresource_format_block_extent = src_subresource_format.block_extent();

            if src_offset[0] % src_subresource_format_block_extent[0] != 0 {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].src_offset[0]` is not a multiple of coordinate 0 of the \
                        block extent of the format of the subresource of `src_image` \
                        selected by `regions[{0}].src_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageInfo2-pRegions-07278"],
                    ..Default::default()
                }));
            }

            if src_offset[1] % src_subresource_format_block_extent[1] != 0 {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].src_offset[1]` is not a multiple of coordinate 1 of the \
                        block extent of the format of the subresource of `src_image` \
                        selected by `regions[{0}].src_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageInfo2-pRegions-07279"],
                    ..Default::default()
                }));
            }

            if src_offset[2] % src_subresource_format_block_extent[2] != 0 {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].src_offset[2]` is not a multiple of coordinate 2 of the \
                        block extent of the format of the subresource of `src_image` \
                        selected by `regions[{0}].src_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageInfo2-pRegions-07280"],
                    ..Default::default()
                }));
            }

            if src_offset[0] + extent[0] != src_subresource_extent[0]
                && (src_offset[0] + extent[0]) % src_subresource_format_block_extent[0] != 0
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].src_offset[0] + regions[{0}].extent[0]` is not \
                        equal to the extent of the subresource of `src_image` \
                        selected by `regions[{0}].src_subresource`, but \
                        it is also not a multiple of coordinate 0 of the block extent of the \
                        format of that subresource",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageInfo2-srcImage-01728"],
                    ..Default::default()
                }));
            }

            if src_offset[1] + extent[1] != src_subresource_extent[1]
                && (src_offset[1] + extent[1]) % src_subresource_format_block_extent[1] != 0
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].src_offset[1] + regions[{0}].extent[1]` is not \
                        equal to the extent of the subresource of `src_image` \
                        selected by `regions[{0}].src_subresource`, but \
                        it is also not a multiple of coordinate 1 of the block extent of the \
                        format of that subresource",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageInfo2-srcImage-01729"],
                    ..Default::default()
                }));
            }

            if src_offset[2] + extent[2] != src_subresource_extent[2]
                && (src_offset[2] + extent[2]) % src_subresource_format_block_extent[2] != 0
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].src_offset[2] + regions[{0}].extent[2]` is not \
                        equal to the extent of the subresource of `src_image` \
                        selected by `regions[{0}].src_subresource`, but \
                        it is also not a multiple of coordinate 2 of the block extent of the \
                        format of that subresource",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageInfo2-srcImage-01730"],
                    ..Default::default()
                }));
            }

            if !(src_subresource.aspects - ImageAspects::STENCIL).is_empty()
                && !src_image.usage().intersects(ImageUsage::TRANSFER_SRC)
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].src_subresource.aspects` contains aspects other than \
                        `ImageAspects::STENCIL`, but \
                        `src_image.usage()` does not contain `ImageUsage::TRANSFER_SRC`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageInfo2-aspect-06662"],
                    ..Default::default()
                }));
            }

            if src_subresource.aspects.intersects(ImageAspects::STENCIL)
                && !src_image
                    .stencil_usage()
                    .unwrap_or(src_image.usage())
                    .intersects(ImageUsage::TRANSFER_SRC)
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].src_subresource.aspects` contains \
                        `ImageAspects::STENCIL`, but \
                        `src_image.stencil_usage()` does not contain `ImageUsage::TRANSFER_SRC`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageInfo2-aspect-06664"],
                    ..Default::default()
                }));
            }

            /*
               Check dst
            */

            if dst_subresource.mip_level >= dst_image.mip_levels() {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].dst_subresource.mip_level` is not less than \
                        `dst_image.mip_levels()`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageInfo2-dstSubresource-07967"],
                    ..Default::default()
                }));
            }

            let mut dst_subresource_format = dst_image_format;
            let mut dst_subresource_extent =
                mip_level_extent(dst_image.extent(), dst_subresource.mip_level).unwrap();

            if dst_image_format_planes.is_empty() {
                if !dst_image_format_aspects.contains(dst_subresource.aspects) {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`regions[{}].dst_subresource.aspects` is not a subset of \
                            `dst_image.format().aspects()`",
                            region_index,
                        )
                        .into(),
                        vuids: &["VUID-VkCopyImageInfo2-aspectMask-00143"],
                        ..Default::default()
                    }));
                }
            } else if dst_image_format_planes.len() == 2 {
                match dst_subresource.aspects {
                    ImageAspects::PLANE_0 => {
                        dst_subresource_format = dst_image_format_planes[0];
                    }
                    ImageAspects::PLANE_1 => {
                        dst_subresource_format = dst_image_format_planes[1];
                        dst_subresource_extent = dst_image_format_subsampled_extent;
                    }
                    _ => {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`dst_image.format()` is a multi-planar format with two planes, \
                                but `regions[{}].dst_subresource.aspect` is not \
                                `ImageAspects::PLANE_0` or `ImageAspects::PLANE_1`",
                                region_index,
                            )
                            .into(),
                            vuids: &[
                                "VUID-VkCopyImageInfo2-dstImage-08714",
                                "VUID-VkCopyImageInfo2-aspectMask-00143",
                            ],
                            ..Default::default()
                        }));
                    }
                }
            } else if dst_image_format_planes.len() == 3 {
                match dst_subresource.aspects {
                    ImageAspects::PLANE_0 => {
                        dst_subresource_format = dst_image_format_planes[0];
                    }
                    ImageAspects::PLANE_1 => {
                        dst_subresource_format = dst_image_format_planes[1];
                        dst_subresource_extent = dst_image_format_subsampled_extent;
                    }
                    ImageAspects::PLANE_2 => {
                        dst_subresource_format = dst_image_format_planes[2];
                        dst_subresource_extent = dst_image_format_subsampled_extent;
                    }
                    _ => {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`dst_image.format()` is a multi-planar format with three planes, \
                                but `regions[{}].dst_subresource.aspect` is not \
                                `ImageAspects::PLANE_0`, `ImageAspects::PLANE_1` or \
                                `ImageAspects::PLANE_2`",
                                region_index,
                            )
                            .into(),
                            vuids: &[
                                "VUID-VkCopyImageInfo2-dstImage-08714",
                                "VUID-VkCopyImageInfo2-aspectMask-00143",
                            ],
                            ..Default::default()
                        }));
                    }
                }
            }

            match dst_image.image_type() {
                ImageType::Dim1d => {
                    if dst_offset[1] != 0 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`dst_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].dst_offset[1]` is not 0",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyImageInfo2-dstImage-00152"],
                            ..Default::default()
                        }));
                    }

                    if extent[1] != 1 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`dst_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].extent[1]` is not 1",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyImageInfo2-dstImage-00152"],
                            ..Default::default()
                        }));
                    }

                    if dst_offset[2] != 0 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`dst_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].dst_offset[2]` is not 0",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyImageInfo2-dstImage-01786"],
                            ..Default::default()
                        }));
                    }

                    if extent[2] != 1 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`dst_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].extent[2]` is not 1",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyImageInfo2-dstImage-01786"],
                            ..Default::default()
                        }));
                    }
                }
                ImageType::Dim2d => {
                    if dst_offset[2] != 0 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`dst_image.image_type()` is `ImageType::Dim2d`, but \
                                `regions[{}].dst_offset[2]` is not 0",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyImageInfo2-dstImage-01788"],
                            ..Default::default()
                        }));
                    }
                }
                ImageType::Dim3d => {
                    if !(dst_subresource.base_array_layer == 0
                        && dst_subresource.layer_count == Some(1))
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`dst_image.image_type()` is `ImageType::Dim3d`, but \
                                `regions[{0}].dst_subresource.base_array_layer` and \
                                `regions[{0}].dst_subresource.layer_count` are not `0` and \
                                `Some(1)`, respectively",
                                region_index,
                            )
                            .into(),
                            vuids: &[
                                "VUID-VkCopyImageInfo2-dstImage-04444",
                                "VUID-VkCopyImageInfo2-apiVersion-07932",
                            ],
                            ..Default::default()
                        }));
                    }
                }
            }

            if dst_subresource.base_array_layer >= dst_image.array_layers() {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].dst_subresource.base_array_layer` is not less than \
                        `dst_image.array_layers()`",
                        region_index,
                    )
                    .into(),
                    // vuids?
                    ..Default::default()
                }));
            }

            if let Some(dst_subresource_layer_count) = dst_subresource.layer_count {
                if dst_subresource_layer_count
                    > dst_image.array_layers() - dst_subresource.base_array_layer
                {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`regions[{0}].dst_subresource.base_array_layer + \
                            regions[{0}].dst_subresource.layer_count` is greater than \
                            `dst_image.array_layers()`",
                            region_index,
                        )
                        .into(),
                        vuids: &["VUID-VkCopyImageInfo2-dstSubresource-07968"],
                        ..Default::default()
                    }));
                }
            }

            if dst_offset[0]
                .checked_add(extent[0])
                .is_none_or(|end| end > dst_subresource_extent[0])
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].dst_offset[0] + regions[{0}].extent[0]` is greater \
                        than coordinate 0 of the extent of the subresource of `dst_image` \
                        selected by `regions[{0}].dst_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageInfo2-dstOffset-00150"],
                    ..Default::default()
                }));
            }

            if dst_offset[1]
                .checked_add(extent[1])
                .is_none_or(|end| end > dst_subresource_extent[1])
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].dst_offset[1] + regions[{0}].extent[1]` is greater \
                        than coordinate 1 of the extent of the subresource of `dst_image` \
                        selected by `regions[{0}].dst_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageInfo2-dstOffset-00151"],
                    ..Default::default()
                }));
            }

            if dst_offset[2]
                .checked_add(extent[2])
                .is_none_or(|end| end > dst_subresource_extent[2])
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].dst_offset[2] + regions[{0}].extent[2]` is greater \
                        than coordinate 2 of the extent of the subresource of `dst_image` \
                        selected by `regions[{0}].dst_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageInfo2-dstOffset-00153"],
                    ..Default::default()
                }));
            }

            let dst_subresource_format_block_extent = dst_subresource_format.block_extent();

            if dst_offset[0] % dst_subresource_format_block_extent[0] != 0 {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].dst_offset[0]` is not a multiple of coordinate 0 of the \
                        block extent of the format of the subresource of `dst_image` \
                        selected by `regions[{0}].dst_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageInfo2-pRegions-07281"],
                    ..Default::default()
                }));
            }

            if dst_offset[1] % dst_subresource_format_block_extent[1] != 0 {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].dst_offset[1]` is not a multiple of coordinate 1 of the \
                        block extent of the format of the subresource of `dst_image` \
                        selected by `regions[{0}].dst_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageInfo2-pRegions-07282"],
                    ..Default::default()
                }));
            }

            if dst_offset[2] % dst_subresource_format_block_extent[2] != 0 {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].dst_offset[2]` is not a multiple of coordinate 2 of the \
                        block extent of the format of the subresource of `dst_image` \
                        selected by `regions[{0}].dst_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageInfo2-pRegions-07283"],
                    ..Default::default()
                }));
            }

            if dst_offset[0] + extent[0] != dst_subresource_extent[0]
                && (dst_offset[0] + extent[0]) % dst_subresource_format_block_extent[0] != 0
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].dst_offset[0] + regions[{0}].extent[0]` is not \
                        equal to the extent of the subresource of `dst_image` \
                        selected by `regions[{0}].dst_subresource`, but \
                        it is also not a multiple of coordinate 0 of the block extent of the \
                        format of that subresource",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageInfo2-dstImage-01732"],
                    ..Default::default()
                }));
            }

            if dst_offset[1] + extent[1] != dst_subresource_extent[1]
                && (dst_offset[1] + extent[1]) % dst_subresource_format_block_extent[1] != 0
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].dst_offset[1] + regions[{0}].extent[1]` is not \
                        equal to the extent of the subresource of `dst_image` \
                        selected by `regions[{0}].dst_subresource`, but \
                        it is also not a multiple of coordinate 1 of the block extent of the \
                        format of that subresource",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageInfo2-dstImage-01733"],
                    ..Default::default()
                }));
            }

            if dst_offset[2] + extent[2] != dst_subresource_extent[2]
                && (dst_offset[2] + extent[2]) % dst_subresource_format_block_extent[2] != 0
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].dst_offset[2] + regions[{0}].extent[2]` is not \
                        equal to the extent of the subresource of `dst_image` \
                        selected by `regions[{0}].dst_subresource`, but \
                        it is also not a multiple of coordinate 2 of the block extent of the \
                        format of that subresource",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageInfo2-dstImage-01734"],
                    ..Default::default()
                }));
            }

            if !(dst_subresource.aspects - ImageAspects::STENCIL).is_empty()
                && !dst_image.usage().intersects(ImageUsage::TRANSFER_DST)
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].dst_subresource.aspects` contains aspects other than \
                        `ImageAspects::STENCIL`, but \
                        `dst_image.usage()` does not contain `ImageUsage::TRANSFER_DST`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageInfo2-aspect-06663"],
                    ..Default::default()
                }));
            }

            if dst_subresource.aspects.intersects(ImageAspects::STENCIL)
                && !dst_image
                    .stencil_usage()
                    .unwrap_or(dst_image.usage())
                    .intersects(ImageUsage::TRANSFER_DST)
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].dst_subresource.aspects` contains \
                        `ImageAspects::STENCIL`, but \
                        `dst_image.stencil_usage()` does not contain `ImageUsage::TRANSFER_DST`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageInfo2-aspect-06665"],
                    ..Default::default()
                }));
            }

            /*
                Check src and dst together
            */

            match (
                src_image_format_planes.is_empty(),
                dst_image_format_planes.is_empty(),
            ) {
                (true, true) => {
                    if src_subresource.aspects != dst_subresource.aspects {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.format()` and `dst_image.format()` are both not \
                                multi-planar formats, but \
                                `regions[{0}].src_subresource.aspects` does not equal \
                                `regions[{0}].dst_subresource.aspects`",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyImageInfo2-srcImage-01551"],
                            ..Default::default()
                        }));
                    }

                    if src_image_format.block_size() != dst_image_format.block_size() {
                        return Err(Box::new(ValidationError {
                            problem: "`src_image.format()` and `dst_image.format()` are both not \
                                multi-planar formats, but \
                                `src_image.format().block_size()` does not equal \
                                `dst_image.format().block_size()`"
                                .into(),
                            vuids: &["VUID-VkCopyImageInfo2-srcImage-01548"],
                            ..Default::default()
                        }));
                    }
                }
                (false, true) => {
                    if dst_subresource.aspects != ImageAspects::COLOR {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.format()` is a multi-planar format, and \
                                `dst_image.format()` is not a multi-planar format, but \
                                `regions[{}].dst_subresource.aspects` is not \
                                `ImageAspects::COLOR`",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyImageInfo2-srcImage-01556"],
                            ..Default::default()
                        }));
                    }

                    if src_subresource_format.block_size() != dst_image_format.block_size() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.format()` is a multi-planar format, and \
                                `dst_image.format()` is not a multi-planar format, but \
                                the block size of the plane of `src_image.format()` selected by \
                                `regions[{}].src_subresource.aspects` does not equal \
                                `dst_image.format().block_size()`",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyImageInfo2-None-01549"],
                            ..Default::default()
                        }));
                    }
                }
                (true, false) => {
                    if src_subresource.aspects != ImageAspects::COLOR {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.format()` is not a multi-planar format, and \
                                `dst_image.format()` is a multi-planar format, but \
                                `regions[{}].src_subresource.aspects` is not \
                                `ImageAspects::COLOR`",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyImageInfo2-dstImage-01557"],
                            ..Default::default()
                        }));
                    }

                    if src_image_format.block_size() != dst_subresource_format.block_size() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.format()` is not a multi-planar format, and \
                                `dst_image.format()` is a multi-planar format, but \
                                `src_image.format().block_size()` does not equal \
                                the block size of the plane of `dst_image.format()` selected by \
                                `regions[{}].dst_subresource.aspects`",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyImageInfo2-None-01549"],
                            ..Default::default()
                        }));
                    }
                }
                (false, false) => {
                    if src_subresource_format.block_size() != dst_subresource_format.block_size() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.format()` and `dst_image.format()` are both \
                                multi-planar formats, but \
                                the block size of the plane of `src_image.format()` selected by \
                                `regions[{0}].src_subresource.aspects` does not equal \
                                the block size of the plane of `dst_image.format()` selected by \
                                `regions[{0}].dst_subresource.aspects`",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyImageInfo2-None-01549"],
                            ..Default::default()
                        }));
                    }
                }
            }

            if src_image.image_type() == dst_image.image_type() {
                match (src_subresource.layer_count, dst_subresource.layer_count) {
                    (Some(src_subresource_layer_count), Some(dst_subresource_layer_count)) => {
                        if src_subresource_layer_count != dst_subresource_layer_count {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "`src_image.image_type()` equals `dst_image.image_type()`, \
                                    and `regions[{0}].src_subresource.layer_count` and \
                                    `regions[{0}].dst_subresource.layer_count` are both `Some`, \
                                    but `regions[{0}].src_subresource.layer_count` does not equal \
                                    `regions[{0}].dst_subresource.layer_count`",
                                    region_index,
                                )
                                .into(),
                                vuids: &["VUID-VkCopyImageInfo2-srcImage-08793"],
                                ..Default::default()
                            }));
                        }
                    }
                    (Some(src_subresource_layer_count), None) => {
                        if src_subresource_layer_count
                            != src_image.array_layers() - src_subresource_layer_count
                        {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "`src_image.image_type()` equals `dst_image.image_type()`, \
                                    and `regions[{0}].src_subresource.layer_count` and \
                                    `regions[{0}].dst_subresource.layer_count` are `Some` and \
                                    `None`, respectively, \
                                    but `regions[{0}].src_subresource.layer_count` does not equal \
                                    `src_image.array_layers() - \
                                    regions[{0}].src_subresource.base_array_layer`",
                                    region_index,
                                )
                                .into(),
                                vuids: &["VUID-VkCopyImageInfo2-srcImage-08794"],
                                ..Default::default()
                            }));
                        }
                    }
                    (None, Some(dst_subresource_layer_count)) => {
                        if dst_subresource_layer_count
                            != dst_image.array_layers() - dst_subresource_layer_count
                        {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "`src_image.image_type()` equals `dst_image.image_type()`, \
                                    and `regions[{0}].src_subresource.layer_count` and \
                                    `regions[{0}].dst_subresource.layer_count` are `None` and \
                                    `Some`, respectively, \
                                    but `regions[{0}].dst_subresource.layer_count` does not equal \
                                    `dst_image.array_layers() - \
                                    regions[{0}].dst_subresource.base_array_layer`",
                                    region_index,
                                )
                                .into(),
                                vuids: &["VUID-VkCopyImageInfo2-srcImage-08794"],
                                ..Default::default()
                            }));
                        }
                    }
                    (None, None) => {}
                }
            }

            match (src_image.image_type(), dst_image.image_type()) {
                (ImageType::Dim2d, ImageType::Dim2d) => {
                    if extent[2] != 1 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.image_type()` and `dst_image.image_type()` are \
                                both `ImageType::Dim2d`, but `regions[{}].extent[2]` is not 1",
                                region_index,
                            )
                            .into(),
                            vuids: &[
                                "VUID-VkCopyImageInfo2-srcImage-01790",
                                "VUID-VkCopyImageInfo2-apiVersion-08969",
                            ],
                            ..Default::default()
                        }));
                    }
                }
                (ImageType::Dim2d, ImageType::Dim3d) => {
                    if Some(extent[2]) != src_subresource.layer_count {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.image_type()` is `ImageType::Dim2d` and \
                                `dst_image.image_type()` is `ImageType::Dim3d`, but \
                                `regions[{0}].extent[2]` does not equal \
                                `regions[{0}].src_subresource.layer_count`",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyImageInfo2-srcImage-01791"],
                            ..Default::default()
                        }));
                    }
                }
                (ImageType::Dim3d, ImageType::Dim2d) => {
                    if Some(extent[2]) != dst_subresource.layer_count {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.image_type()` is `ImageType::Dim3d` and \
                                `dst_image.image_type()` is `ImageType::Dim2d`, but \
                                `regions[{0}].extent[2]` does not equal \
                                `regions[{0}].dst_subresource.layer_count`",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyImageInfo2-dstImage-01792"],
                            ..Default::default()
                        }));
                    }
                }
                _ => (),
            }

            // VUID-VkCopyImageInfo2-pRegions-00124
            if is_same_image {
                let src_region_index = region_index;
                let src_subresource_axes = [
                    src_subresource.mip_level..src_subresource.mip_level + 1,
                    src_subresource.base_array_layer
                        ..src_subresource.layer_count.map_or(
                            src_image.array_layers(),
                            |src_subresource_layer_count| {
                                src_subresource.base_array_layer + src_subresource_layer_count
                            },
                        ),
                ];
                let src_extent_axes = [
                    src_offset[0]..src_offset[0] + extent[0],
                    src_offset[1]..src_offset[1] + extent[1],
                    src_offset[2]..src_offset[2] + extent[2],
                ];

                for (dst_region_index, dst_region) in regions.iter().enumerate() {
                    let &ImageCopy {
                        dst_subresource,
                        dst_offset,
                        ..
                    } = dst_region;

                    // For a single-plane image, the aspects must always be identical anyway
                    if src_image_format_aspects.intersects(ImageAspects::PLANE_0)
                        && src_subresource.aspects != dst_subresource.aspects
                    {
                        continue;
                    }

                    let dst_subresource_axes = [
                        dst_subresource.mip_level..dst_subresource.mip_level + 1,
                        dst_subresource.base_array_layer
                            ..dst_subresource.layer_count.map_or(
                                dst_image.array_layers(),
                                |dst_subresource_layer_count| {
                                    dst_subresource.base_array_layer + dst_subresource_layer_count
                                },
                            ),
                    ];

                    if src_subresource_axes.iter().zip(dst_subresource_axes).any(
                        |(src_range, dst_range)| {
                            src_range.start >= dst_range.end || dst_range.start >= src_range.end
                        },
                    ) {
                        continue;
                    }

                    // If the subresource axes all overlap, then the source and destination must
                    // have the same layout.
                    overlap_subresource_indices = Some((src_region_index, dst_region_index));

                    let dst_extent_axes = [
                        dst_offset[0]..dst_offset[0] + extent[0],
                        dst_offset[1]..dst_offset[1] + extent[1],
                        dst_offset[2]..dst_offset[2] + extent[2],
                    ];

                    // There is only overlap if all of the axes overlap.
                    if src_extent_axes
                        .iter()
                        .zip(dst_extent_axes)
                        .any(|(src_range, dst_range)| {
                            src_range.start >= dst_range.end || dst_range.start >= src_range.end
                        })
                    {
                        continue;
                    }

                    overlap_extent_indices = Some((src_region_index, dst_region_index));
                }
            }
        }

        if let Some((src_region_index, dst_region_index)) = overlap_extent_indices {
            return Err(Box::new(ValidationError {
                problem: format!(
                    "`src_image` is equal to `dst_image`, and `regions[{0}].src_subresource` \
                    overlaps with `regions[{1}].dst_subresource`, but \
                    the `src_offset` and `extent` of `regions[{0}]` overlaps with \
                    the `dst_offset` and `extent` of `regions[{1}]`",
                    src_region_index, dst_region_index,
                )
                .into(),
                vuids: &["VUID-VkCopyImageInfo2-pRegions-00124"],
                ..Default::default()
            }));
        }

        if let Some((src_region_index, dst_region_index)) = overlap_subresource_indices {
            if src_image_layout != dst_image_layout {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`src_image` is equal to `dst_image`, and `regions[{0}].src_subresource` \
                        overlaps with `regions[{1}].dst_subresource`, but \
                        `src_image_layout` does not equal `dst_image_layout`",
                        src_region_index, dst_region_index,
                    )
                    .into(),
                    vuids: &[
                        "VUID-VkCopyImageInfo2-srcImageLayout-00128",
                        "VUID-VkCopyImageInfo2-dstImageLayout-00133",
                    ],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    pub(crate) fn to_vk2(
        &self,
        regions_vk: &'a [vk::ImageCopy2<'static>],
    ) -> vk::CopyImageInfo2<'a> {
        let &Self {
            src_image,
            src_image_layout,
            dst_image,
            dst_image_layout,
            regions: _,
            _ne: _,
        } = self;

        vk::CopyImageInfo2::default()
            .src_image(src_image.handle())
            .src_image_layout(src_image_layout.into())
            .dst_image(dst_image.handle())
            .dst_image_layout(dst_image_layout.into())
            .regions(regions_vk)
    }

    pub(crate) fn to_vk2_regions(&self) -> SmallVec<[vk::ImageCopy2<'static>; 8]> {
        let &Self {
            src_image,
            dst_image,
            regions,
            ..
        } = self;

        if regions.is_empty() {
            let remaining_array_layers = min(src_image.array_layers(), dst_image.array_layers());
            let src_extent = src_image.extent();
            let dst_extent = dst_image.extent();
            let region_vk = vk::ImageCopy2::default()
                .src_subresource(src_image.subresource_layers().to_vk(remaining_array_layers))
                .src_offset(convert_offset([0; 3]))
                .dst_subresource(dst_image.subresource_layers().to_vk(remaining_array_layers))
                .dst_offset(convert_offset([0; 3]))
                .extent(convert_extent([
                    min(src_extent[0], dst_extent[0]),
                    min(src_extent[1], dst_extent[1]),
                    min(src_extent[2], dst_extent[2]),
                ]));

            smallvec![region_vk]
        } else {
            regions
                .iter()
                .map(|region| region.to_vk2(src_image, dst_image))
                .collect()
        }
    }

    pub(crate) fn to_vk(&self) -> CopyImageInfoVk {
        let &Self {
            src_image,
            src_image_layout,
            dst_image,
            dst_image_layout,
            regions: _,
            _ne: _,
        } = self;

        CopyImageInfoVk {
            src_image_vk: src_image.handle(),
            src_image_layout_vk: src_image_layout.into(),
            dst_image_vk: dst_image.handle(),
            dst_image_layout_vk: dst_image_layout.into(),
        }
    }

    pub(crate) fn to_vk_regions(&self) -> SmallVec<[vk::ImageCopy; 8]> {
        let &Self {
            src_image,
            dst_image,
            regions,
            ..
        } = self;

        if regions.is_empty() {
            let remaining_array_layers = min(src_image.array_layers(), dst_image.array_layers());
            let src_extent = src_image.extent();
            let dst_extent = dst_image.extent();
            let region_vk = vk::ImageCopy {
                src_subresource: src_image.subresource_layers().to_vk(remaining_array_layers),
                src_offset: convert_offset([0; 3]),
                dst_subresource: dst_image.subresource_layers().to_vk(remaining_array_layers),
                dst_offset: convert_offset([0; 3]),
                extent: convert_extent([
                    min(src_extent[0], dst_extent[0]),
                    min(src_extent[1], dst_extent[1]),
                    min(src_extent[2], dst_extent[2]),
                ]),
            };

            smallvec![region_vk]
        } else {
            regions
                .iter()
                .map(|region| region.to_vk(src_image, dst_image))
                .collect()
        }
    }
}

pub(crate) struct CopyImageInfoVk {
    pub(crate) src_image_vk: vk::Image,
    pub(crate) src_image_layout_vk: vk::ImageLayout,
    pub(crate) dst_image_vk: vk::Image,
    pub(crate) dst_image_layout_vk: vk::ImageLayout,
}

/// A region of data to copy between images.
#[derive(Clone, Debug)]
pub struct ImageCopy<'a> {
    /// The subresource of `src_image` to copy from.
    ///
    /// The default value is [`ImageSubresourceLayers::default()`].
    pub src_subresource: ImageSubresourceLayers,

    /// The offset from the zero coordinate of `src_image` that copying will start from.
    ///
    /// The default value is `[0; 3]`.
    pub src_offset: [u32; 3],

    /// The subresource of `dst_image` to copy to.
    ///
    /// The default value is [`ImageSubresourceLayers::default()`].
    pub dst_subresource: ImageSubresourceLayers,

    /// The offset from the zero coordinate of `dst_image` that copying will start from.
    ///
    /// The default value is `[0; 3]`.
    pub dst_offset: [u32; 3],

    /// The extent of texels to copy.
    ///
    /// The default value is `[0; 3]`, which must be overridden.
    pub extent: [u32; 3],

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for ImageCopy<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl ImageCopy<'_> {
    /// Returns a default `ImageCopy`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            src_subresource: ImageSubresourceLayers::new(),
            src_offset: [0; 3],
            dst_subresource: ImageSubresourceLayers::new(),
            dst_offset: [0; 3],
            extent: [0; 3],
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            src_subresource,
            src_offset: _,
            dst_subresource,
            dst_offset: _,
            extent,
            _ne,
        } = self;

        src_subresource
            .validate(device)
            .map_err(|err| err.add_context("src_subresource"))?;

        dst_subresource
            .validate(device)
            .map_err(|err| err.add_context("dst_subresource"))?;

        if device.api_version() < Version::V1_1 {
            if src_subresource.aspects != dst_subresource.aspects
                && !device.enabled_extensions().khr_sampler_ycbcr_conversion
            {
                return Err(Box::new(ValidationError {
                    problem: "`src_subresource.aspects` does not equal `dst_subresource.aspects`"
                        .into(),
                    vuids: &["VUID-VkImageCopy2-apiVersion-07940"],
                    ..Default::default()
                }));
            }

            if src_subresource.layer_count != dst_subresource.layer_count
                && !device.enabled_extensions().khr_maintenance1
            {
                return Err(Box::new(ValidationError {
                    problem: "`src_subresource.layer_count` does not equal \
                        `dst_subresource.layer_count`"
                        .into(),
                    vuids: &["VUID-VkImageCopy2-extent-00140"],
                    ..Default::default()
                }));
            }
        }

        if extent[0] == 0 {
            return Err(Box::new(ValidationError {
                context: "extent[0]".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkImageCopy2-extent-06668"],
                ..Default::default()
            }));
        }

        if extent[1] == 0 {
            return Err(Box::new(ValidationError {
                context: "extent[1]".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkImageCopy2-extent-06669"],
                ..Default::default()
            }));
        }

        if extent[2] == 0 {
            return Err(Box::new(ValidationError {
                context: "extent[2]".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkImageCopy2-extent-06670"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk2(&self, src_image: &Image, dst_image: &Image) -> vk::ImageCopy2<'static> {
        let &Self {
            src_subresource,
            src_offset,
            dst_subresource,
            dst_offset,
            extent,
            _ne: _,
        } = self;

        let remaining_array_layers = min(
            src_subresource
                .layer_count
                .unwrap_or(src_image.array_layers() - src_subresource.base_array_layer),
            dst_subresource
                .layer_count
                .unwrap_or(dst_image.array_layers() - dst_subresource.base_array_layer),
        );

        vk::ImageCopy2::default()
            .src_subresource(src_subresource.to_vk(remaining_array_layers))
            .src_offset(vk::Offset3D {
                x: src_offset[0] as i32,
                y: src_offset[1] as i32,
                z: src_offset[2] as i32,
            })
            .dst_subresource(dst_subresource.to_vk(remaining_array_layers))
            .dst_offset(vk::Offset3D {
                x: dst_offset[0] as i32,
                y: dst_offset[1] as i32,
                z: dst_offset[2] as i32,
            })
            .extent(vk::Extent3D {
                width: extent[0],
                height: extent[1],
                depth: extent[2],
            })
    }

    pub(crate) fn to_vk(&self, src_image: &Image, dst_image: &Image) -> vk::ImageCopy {
        let &Self {
            src_subresource,
            src_offset,
            dst_subresource,
            dst_offset,
            extent,
            _ne: _,
        } = self;

        let remaining_array_layers = min(
            src_subresource
                .layer_count
                .unwrap_or(src_image.array_layers() - src_subresource.base_array_layer),
            dst_subresource
                .layer_count
                .unwrap_or(dst_image.array_layers() - dst_subresource.base_array_layer),
        );

        vk::ImageCopy {
            src_subresource: src_subresource.to_vk(remaining_array_layers),
            src_offset: vk::Offset3D {
                x: src_offset[0] as i32,
                y: src_offset[1] as i32,
                z: src_offset[2] as i32,
            },
            dst_subresource: dst_subresource.to_vk(remaining_array_layers),
            dst_offset: vk::Offset3D {
                x: dst_offset[0] as i32,
                y: dst_offset[1] as i32,
                z: dst_offset[2] as i32,
            },
            extent: vk::Extent3D {
                width: extent[0],
                height: extent[1],
                depth: extent[2],
            },
        }
    }
}

/// Parameters to copy data from a buffer to an image.
#[derive(Clone, Debug)]
pub struct CopyBufferToImageInfo<'a> {
    /// The buffer to copy from.
    ///
    /// There is no default value.
    pub src_buffer: &'a Buffer,

    /// The image to copy to.
    ///
    /// There is no default value.
    pub dst_image: &'a Image,

    /// The layout used for `dst_image` during the copy operation.
    ///
    /// The following layouts are allowed:
    /// - [`ImageLayout::TransferDstOptimal`]
    /// - [`ImageLayout::General`]
    ///
    /// The default value is [`ImageLayout::TransferDstOptimal`].
    pub dst_image_layout: ImageLayout,

    /// The regions of the buffer and image to copy between.
    ///
    /// The default value is a single region, covering all of the buffer and the first mip level of
    /// the image. All aspects of the image are selected, or `plane0` if the image is multi-planar.
    pub regions: &'a [BufferImageCopy<'a>],

    pub _ne: crate::NonExhaustive<'a>,
}

impl<'a> CopyBufferToImageInfo<'a> {
    /// Returns a default `CopyBufferToImageInfo` with the provided `src_buffer` and
    /// `dst_image`.
    #[inline]
    pub const fn new(src_buffer: &'a Buffer, dst_image: &'a Image) -> Self {
        Self {
            src_buffer,
            dst_image,
            dst_image_layout: ImageLayout::TransferDstOptimal,
            regions: &[],
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            src_buffer,
            dst_image,
            dst_image_layout,
            regions,
            _ne: _,
        } = self;

        dst_image_layout.validate_device(device).map_err(|err| {
            err.add_context("dst_image_layout")
                .set_vuids(&["VUID-VkCopyBufferToImageInfo2-dstImageLayout-parameter"])
        })?;

        // VUID-VkCopyBufferToImageInfo2-commonparent
        assert_eq!(device, src_buffer.device().as_ref());
        assert_eq!(device, dst_image.device().as_ref());

        let dst_image_format = dst_image.format();
        let dst_image_format_aspects = dst_image_format.aspects();
        let dst_image_format_planes = dst_image_format.planes();
        let dst_image_format_subsampled_extent = dst_image_format
            .ycbcr_chroma_sampling()
            .map_or(dst_image.extent(), |s| {
                s.subsampled_extent(dst_image.extent())
            });

        if !src_buffer.usage().intersects(BufferUsage::TRANSFER_SRC) {
            return Err(Box::new(ValidationError {
                context: "src_buffer.usage()".into(),
                problem: "does not contain `BufferUsage::TRANSFER_SRC`".into(),
                vuids: &["VUID-VkCopyBufferToImageInfo2-srcBuffer-00174"],
                ..Default::default()
            }));
        }

        if !dst_image.usage().intersects(ImageUsage::TRANSFER_DST) {
            return Err(Box::new(ValidationError {
                context: "dst_image.usage()".into(),
                problem: "does not contain `ImageUsage::TRANSFER_DST`".into(),
                vuids: &["VUID-VkCopyBufferToImageInfo2-dstImage-00177"],
                ..Default::default()
            }));
        }

        if device.api_version() >= Version::V1_1 || device.enabled_extensions().khr_maintenance1 {
            if !dst_image
                .format_features()
                .intersects(FormatFeatures::TRANSFER_DST)
            {
                return Err(Box::new(ValidationError {
                    context: "dst_image.format_features()".into(),
                    problem: "does not contain `FormatFeatures::TRANSFER_DST`".into(),
                    vuids: &["VUID-VkCopyBufferToImageInfo2-dstImage-01997"],
                    ..Default::default()
                }));
            }
        }

        if dst_image.samples() != SampleCount::Sample1 {
            return Err(Box::new(ValidationError {
                context: "dst_image.samples()".into(),
                problem: "is not `SampleCount::Sample1`".into(),
                vuids: &["VUID-VkCopyBufferToImageInfo2-dstImage-07973"],
                ..Default::default()
            }));
        }

        if !matches!(
            dst_image_layout,
            ImageLayout::TransferDstOptimal | ImageLayout::General
        ) {
            return Err(Box::new(ValidationError {
                context: "dst_image_layout".into(),
                problem: "is not `ImageLayout::TransferDstOptimal` or `ImageLayout::General`"
                    .into(),
                vuids: &["VUID-VkCopyBufferToImageInfo2-dstImageLayout-01396"],
                ..Default::default()
            }));
        }

        for (region_index, region) in regions.iter().enumerate() {
            region
                .validate(device)
                .map_err(|err| err.add_context(format!("regions[{}]", region_index)))?;

            let &BufferImageCopy {
                buffer_offset,
                buffer_row_length,
                buffer_image_height,
                image_subresource,
                image_offset,
                image_extent,
                _ne: _,
            } = region;

            /*
               Check image
            */

            if image_subresource.mip_level >= dst_image.mip_levels() {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].image_subresource.mip_level` is not less than \
                        `dst_image.mip_levels()`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyBufferToImageInfo2-imageSubresource-01701"],
                    ..Default::default()
                }));
            }

            let mut image_subresource_format = dst_image_format;
            let mut image_subresource_extent =
                mip_level_extent(dst_image.extent(), image_subresource.mip_level).unwrap();

            if dst_image_format_planes.is_empty() {
                if !dst_image_format_aspects.contains(image_subresource.aspects) {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`regions[{}].image_subresource.aspects` is not a subset of \
                            `dst_image.format().aspects()`",
                            region_index,
                        )
                        .into(),
                        vuids: &["VUID-VkCopyBufferToImageInfo2-aspectMask-00211"],
                        ..Default::default()
                    }));
                }
            } else if dst_image_format_planes.len() == 2 {
                match image_subresource.aspects {
                    ImageAspects::PLANE_0 => {
                        image_subresource_format = dst_image_format_planes[0];
                    }
                    ImageAspects::PLANE_1 => {
                        image_subresource_format = dst_image_format_planes[1];
                        image_subresource_extent = dst_image_format_subsampled_extent;
                    }
                    _ => {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`dst_image.format()` is a multi-planar format with two planes, \
                                but `regions[{}].image_subresource.aspect` is not \
                                `ImageAspects::PLANE_0` or `ImageAspects::PLANE_1`",
                                region_index,
                            )
                            .into(),
                            vuids: &[
                                "VUID-VkCopyBufferToImageInfo2-dstImage-07981",
                                "VUID-VkCopyBufferToImageInfo2-aspectMask-00211",
                            ],
                            ..Default::default()
                        }));
                    }
                }
            } else if dst_image_format_planes.len() == 3 {
                match image_subresource.aspects {
                    ImageAspects::PLANE_0 => {
                        image_subresource_format = dst_image_format_planes[0];
                    }
                    ImageAspects::PLANE_1 => {
                        image_subresource_format = dst_image_format_planes[1];
                        image_subresource_extent = dst_image_format_subsampled_extent;
                    }
                    ImageAspects::PLANE_2 => {
                        image_subresource_format = dst_image_format_planes[2];
                        image_subresource_extent = dst_image_format_subsampled_extent;
                    }
                    _ => {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`dst_image.format()` is a multi-planar format with three planes, \
                                but `regions[{}].image_subresource.aspect` is not \
                                `ImageAspects::PLANE_0`, `ImageAspects::PLANE_1` or \
                                `ImageAspects::PLANE_2`",
                                region_index,
                            )
                            .into(),
                            vuids: &[
                                "VUID-VkCopyBufferToImageInfo2-dstImage-07982",
                                "VUID-VkCopyBufferToImageInfo2-aspectMask-00211",
                            ],
                            ..Default::default()
                        }));
                    }
                }
            }

            match dst_image.image_type() {
                ImageType::Dim1d => {
                    if image_offset[1] != 0 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`dst_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].image_offset[1]` is not 0",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyBufferToImageInfo2-dstImage-07979"],
                            ..Default::default()
                        }));
                    }

                    if image_extent[1] != 1 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`dst_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].image_extent[1]` is not 1",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyBufferToImageInfo2-dstImage-07979"],
                            ..Default::default()
                        }));
                    }

                    if image_offset[2] != 0 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`dst_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].image_offset[2]` is not 0",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyBufferToImageInfo2-dstImage-07980"],
                            ..Default::default()
                        }));
                    }

                    if image_extent[2] != 1 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`dst_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].image_extent[2]` is not 1",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyBufferToImageInfo2-dstImage-07980"],
                            ..Default::default()
                        }));
                    }
                }
                ImageType::Dim2d => {
                    if image_offset[2] != 0 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`dst_image.image_type()` is `ImageType::Dim2d`, but \
                                `regions[{}].image_offset[2]` is not 0",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyBufferToImageInfo2-dstImage-07980"],
                            ..Default::default()
                        }));
                    }

                    if image_extent[2] != 1 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`dst_image.image_type()` is `ImageType::Dim2d`, but \
                                `regions[{}].image_extent[2]` is not 1",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyBufferToImageInfo2-dstImage-07980"],
                            ..Default::default()
                        }));
                    }
                }
                ImageType::Dim3d => {
                    if !(image_subresource.base_array_layer == 0
                        && image_subresource.layer_count == Some(1))
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`dst_image.image_type()` is `ImageType::Dim3d`, but \
                                `regions[{0}].image_subresource.base_array_layer` and \
                                `regions[{0}].image_subresource.layer_count` are not `0` and \
                                `Some(1)`, respectively",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyBufferToImageInfo2-dstImage-07983"],
                            ..Default::default()
                        }));
                    }
                }
            }

            if image_subresource.base_array_layer >= dst_image.array_layers() {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].image_subresource.base_array_layer` is not less than \
                        `dst_image.array_layers()`",
                        region_index,
                    )
                    .into(),
                    // vuids?
                    ..Default::default()
                }));
            }

            if let Some(image_subresource_layer_count) = image_subresource.layer_count {
                if image_subresource_layer_count
                    > dst_image.array_layers() - image_subresource.base_array_layer
                {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`regions[{0}].image_subresource.base_array_layer + \
                            regions[{0}].image_subresource.layer_count` is greater than \
                            `dst_image.array_layers()`",
                            region_index,
                        )
                        .into(),
                        vuids: &["VUID-VkCopyBufferToImageInfo2-imageSubresource-07968"],
                        ..Default::default()
                    }));
                }
            }

            if image_offset[0]
                .checked_add(image_extent[0])
                .is_none_or(|end| end > image_subresource_extent[0])
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].image_offset[0] + regions[{0}].image_extent[0]` is greater \
                        than coordinate 0 of the extent of the subresource of `dst_image` \
                        selected by `regions[{0}].image_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyBufferToImageInfo2-pRegions-06223"],
                    ..Default::default()
                }));
            }

            if image_offset[1]
                .checked_add(image_extent[1])
                .is_none_or(|end| end > image_subresource_extent[1])
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].image_offset[1] + regions[{0}].image_extent[1]` is greater \
                        than coordinate 1 of the extent of the subresource of `dst_image` \
                        selected by `regions[{0}].image_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyBufferToImageInfo2-pRegions-06224"],
                    ..Default::default()
                }));
            }

            if image_offset[2]
                .checked_add(image_extent[2])
                .is_none_or(|end| end > image_subresource_extent[2])
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].image_offset[2] + regions[{0}].image_extent[2]` is greater \
                        than coordinate 2 of the extent of the subresource of `dst_image` \
                        selected by `regions[{0}].image_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyBufferToImageInfo2-imageOffset-00200"],
                    ..Default::default()
                }));
            }

            let image_subresource_format_block_extent = image_subresource_format.block_extent();

            if image_offset[0] % image_subresource_format_block_extent[0] != 0 {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].image_offset[0]` is not a multiple of coordinate 0 of the \
                        block extent of the format of the subresource of `dst_image` \
                        selected by `regions[{0}].image_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyBufferToImageInfo2-pRegions-07274"],
                    ..Default::default()
                }));
            }

            if image_offset[1] % image_subresource_format_block_extent[1] != 0 {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].image_offset[1]` is not a multiple of coordinate 1 of the \
                        block extent of the format of the subresource of `dst_image` \
                        selected by `regions[{0}].image_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyBufferToImageInfo2-pRegions-07275"],
                    ..Default::default()
                }));
            }

            if image_offset[2] % image_subresource_format_block_extent[2] != 0 {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].image_offset[2]` is not a multiple of coordinate 2 of the \
                        block extent of the format of the subresource of `dst_image` \
                        selected by `regions[{0}].image_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyBufferToImageInfo2-pRegions-07276"],
                    ..Default::default()
                }));
            }

            if image_offset[0] + image_extent[0] != image_subresource_extent[0]
                && (image_offset[0] + image_extent[0]) % image_subresource_format_block_extent[0]
                    != 0
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].image_offset[0] + regions[{0}].image_extent[0]` is not \
                        equal to the extent of the subresource of `dst_image` \
                        selected by `regions[{0}].image_subresource`, but \
                        it is also not a multiple of coordinate 0 of the block extent of the \
                        format of that subresource",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyBufferToImageInfo2-imageExtent-00207"],
                    ..Default::default()
                }));
            }

            if image_offset[1] + image_extent[1] != image_subresource_extent[1]
                && (image_offset[1] + image_extent[1]) % image_subresource_format_block_extent[1]
                    != 0
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].image_offset[1] + regions[{0}].image_extent[1]` is not \
                        equal to the extent of the subresource of `dst_image` \
                        selected by `regions[{0}].image_subresource`, but \
                        it is also not a multiple of coordinate 1 of the block extent of the \
                        format of that subresource",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyBufferToImageInfo2-imageExtent-00208"],
                    ..Default::default()
                }));
            }

            if image_offset[2] + image_extent[2] != image_subresource_extent[2]
                && (image_offset[2] + image_extent[2]) % image_subresource_format_block_extent[2]
                    != 0
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].image_offset[2] + regions[{0}].image_extent[2]` is not \
                        equal to the extent of the subresource of `dst_image` \
                        selected by `regions[{0}].image_subresource`, but \
                        it is also not a multiple of coordinate 2 of the block extent of the \
                        format of that subresource",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyBufferToImageInfo2-imageExtent-00209"],
                    ..Default::default()
                }));
            }

            /*
               Check buffer and image together
            */

            let image_subresource_format_block_size = image_subresource_format.block_size();

            if dst_image_format_aspects.intersects(ImageAspects::DEPTH | ImageAspects::STENCIL) {
                if buffer_offset % 4 != 0 {
                    return Err(Box::new(ValidationError {
                        problem: "`dst_image.format()` is a depth/stencil format, but \
                            `buffer_offset` is not a multiple of 4"
                            .into(),
                        vuids: &["VUID-VkCopyBufferToImageInfo2-dstImage-07978"],
                        ..Default::default()
                    }));
                }
            } else {
                if buffer_offset % image_subresource_format_block_size != 0 {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`dst_image.format()` is not a depth/stencil format, but \
                            `buffer_offset` is not a multiple of the block size of the format of \
                            the subresource of `dst_image` selected by \
                            `regions[{0}].image_subresource`",
                            region_index,
                        )
                        .into(),
                        vuids: &[
                            "VUID-VkCopyBufferToImageInfo2-dstImage-07975",
                            "VUID-VkCopyBufferToImageInfo2-dstImage-07976",
                        ],
                        ..Default::default()
                    }));
                }
            }

            if buffer_row_length % image_subresource_format_block_extent[0] != 0 {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].buffer_row_length` is not a multiple of coordinate 0 of \
                        the block extent of the format of the subresource of `dst_image` \
                        selected by `regions[{0}].image_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyBufferToImageInfo2-bufferRowLength-00203"],
                    ..Default::default()
                }));
            }

            if buffer_image_height % image_subresource_format_block_extent[1] != 0 {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].buffer_image_height` is not a multiple of coordinate 1 of \
                        the block extent of the format of the subresource of `dst_image` \
                        selected by `regions[{0}].image_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyBufferToImageInfo2-bufferImageHeight-00204"],
                    ..Default::default()
                }));
            }

            if (buffer_row_length / image_subresource_format_block_extent[0]) as DeviceSize
                * image_subresource_format_block_size
                > 0x7FFFFFFF
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].buffer_row_length`, divided by the block size of the \
                        format of the subresource of `dst_image` selected by \
                        `regions[{0}].image_subresource`, and then multiplied by the block size \
                        of that subresource, is greater than 0x7FFFFFFF",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyBufferToImageInfo2-bufferRowLength-00203"],
                    ..Default::default()
                }));
            }

            if region
                .buffer_copy_size(image_subresource_format, dst_image.array_layers())
                .and_then(|copy_size| buffer_offset.checked_add(copy_size))
                .is_none_or(|end| end > src_buffer.size())
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].buffer_offset` plus the number of bytes being copied \
                        is greater than `src_buffer.size()`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyBufferToImageInfo2-pRegions-00171"],
                    ..Default::default()
                }));
            }
        }

        // VUID-VkCopyBufferToImageInfo2-pRegions-00173
        // Can't occur as long as memory aliasing isn't allowed.

        // VUID-VkCopyBufferToImageInfo2-pRegions-07931
        // Unsafe, can't validate

        Ok(())
    }

    pub(crate) fn to_vk2(
        &self,
        regions_vk: &'a [vk::BufferImageCopy2<'static>],
    ) -> vk::CopyBufferToImageInfo2<'a> {
        let &Self {
            src_buffer,
            dst_image,
            dst_image_layout,
            regions: _,
            _ne: _,
        } = self;

        vk::CopyBufferToImageInfo2::default()
            .src_buffer(src_buffer.handle())
            .dst_image(dst_image.handle())
            .dst_image_layout(dst_image_layout.into())
            .regions(regions_vk)
    }

    pub(crate) fn to_vk2_regions(&self) -> SmallVec<[vk::BufferImageCopy2<'static>; 8]> {
        let &Self {
            dst_image, regions, ..
        } = self;

        if regions.is_empty() {
            let remaining_array_layers = dst_image.array_layers();
            let region_vk = vk::BufferImageCopy2::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(dst_image.subresource_layers().to_vk(remaining_array_layers))
                .image_offset(convert_offset([0; 3]))
                .image_extent(convert_extent(dst_image.extent()));

            smallvec![region_vk]
        } else {
            regions
                .iter()
                .map(|region| region.to_vk2(dst_image))
                .collect()
        }
    }

    pub(crate) fn to_vk(&self) -> CopyBufferToImageInfoVk {
        let &Self {
            src_buffer,
            dst_image,
            dst_image_layout,
            regions: _,
            _ne: _,
        } = self;

        CopyBufferToImageInfoVk {
            src_buffer_vk: src_buffer.handle(),
            dst_image_vk: dst_image.handle(),
            dst_image_layout_vk: dst_image_layout.into(),
        }
    }

    pub(crate) fn to_vk_regions(&self) -> SmallVec<[vk::BufferImageCopy; 8]> {
        let &Self {
            dst_image, regions, ..
        } = self;

        if regions.is_empty() {
            let remaining_array_layers = dst_image.array_layers();
            let region_vk = vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: dst_image.subresource_layers().to_vk(remaining_array_layers),
                image_offset: convert_offset([0; 3]),
                image_extent: convert_extent(dst_image.extent()),
            };

            smallvec![region_vk]
        } else {
            regions
                .iter()
                .map(|region| region.to_vk(dst_image))
                .collect()
        }
    }
}

pub(crate) struct CopyBufferToImageInfoVk {
    pub(crate) src_buffer_vk: vk::Buffer,
    pub(crate) dst_image_vk: vk::Image,
    pub(crate) dst_image_layout_vk: vk::ImageLayout,
}

/// Parameters to copy data from an image to a buffer.
#[derive(Clone, Debug)]
pub struct CopyImageToBufferInfo<'a> {
    /// The image to copy from.
    ///
    /// There is no default value.
    pub src_image: &'a Image,

    /// The layout used for `src_image` during the copy operation.
    ///
    /// The following layouts are allowed:
    /// - [`ImageLayout::TransferSrcOptimal`]
    /// - [`ImageLayout::General`]
    ///
    /// The default value is [`ImageLayout::TransferSrcOptimal`].
    pub src_image_layout: ImageLayout,

    /// The buffer to copy to.
    ///
    /// There is no default value.
    pub dst_buffer: &'a Buffer,

    /// The regions of the image and buffer to copy between.
    ///
    /// The default value is a single region, covering all of the buffer and the first mip level of
    /// the image. All aspects of the image are selected, or `plane0` if the image is multi-planar.
    pub regions: &'a [BufferImageCopy<'a>],

    pub _ne: crate::NonExhaustive<'a>,
}

impl<'a> CopyImageToBufferInfo<'a> {
    /// Returns a default `CopyImageToBufferInfo` with the provided `src_image` and
    /// `dst_buffer`.
    #[inline]
    pub const fn new(src_image: &'a Image, dst_buffer: &'a Buffer) -> Self {
        Self {
            src_image,
            src_image_layout: ImageLayout::TransferSrcOptimal,
            dst_buffer,
            regions: &[],
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            src_image,
            src_image_layout,
            dst_buffer,
            regions,
            _ne: _,
        } = self;

        src_image_layout.validate_device(device).map_err(|err| {
            err.add_context("src_image_layout")
                .set_vuids(&["VUID-VkCopyImageToBufferInfo2-srcImageLayout-parameter"])
        })?;

        // VUID-VkCopyImageToBufferInfo2-commonparent
        assert_eq!(device, src_image.device().as_ref());
        assert_eq!(device, dst_buffer.device().as_ref());

        let src_image_format = src_image.format();
        let src_image_format_aspects = src_image_format.aspects();
        let src_image_format_planes = src_image_format.planes();
        let src_image_format_subsampled_extent = src_image_format
            .ycbcr_chroma_sampling()
            .map_or(src_image.extent(), |s| {
                s.subsampled_extent(src_image.extent())
            });

        if !dst_buffer.usage().intersects(BufferUsage::TRANSFER_DST) {
            return Err(Box::new(ValidationError {
                context: "dst_buffer.usage()".into(),
                problem: "does not contain `BufferUsage::TRANSFER_DST`".into(),
                vuids: &["VUID-VkCopyImageToBufferInfo2-dstBuffer-00191"],
                ..Default::default()
            }));
        }

        if !src_image.usage().intersects(ImageUsage::TRANSFER_SRC) {
            return Err(Box::new(ValidationError {
                context: "src_image.usage()".into(),
                problem: "does not contain `ImageUsage::TRANSFER_SRC`".into(),
                vuids: &["VUID-VkCopyImageToBufferInfo2-srcImage-00186"],
                ..Default::default()
            }));
        }

        if device.api_version() >= Version::V1_1 || device.enabled_extensions().khr_maintenance1 {
            if !src_image
                .format_features()
                .intersects(FormatFeatures::TRANSFER_SRC)
            {
                return Err(Box::new(ValidationError {
                    context: "src_image.format_features()".into(),
                    problem: "does not contain `FormatFeatures::TRANSFER_SRC`".into(),
                    vuids: &["VUID-VkCopyImageToBufferInfo2-srcImage-01998"],
                    ..Default::default()
                }));
            }
        }

        if src_image.samples() != SampleCount::Sample1 {
            return Err(Box::new(ValidationError {
                context: "src_image.samples()".into(),
                problem: "is not `SampleCount::Sample1`".into(),
                vuids: &["VUID-VkCopyImageToBufferInfo2-srcImage-07973"],
                ..Default::default()
            }));
        }

        if !matches!(
            src_image_layout,
            ImageLayout::TransferSrcOptimal | ImageLayout::General
        ) {
            return Err(Box::new(ValidationError {
                context: "src_image_layout".into(),
                problem: "is not `ImageLayout::TransferSrcOptimal` or `ImageLayout::General`"
                    .into(),
                vuids: &["VUID-VkCopyImageToBufferInfo2-srcImageLayout-01397"],
                ..Default::default()
            }));
        }

        for (region_index, region) in regions.iter().enumerate() {
            region
                .validate(device)
                .map_err(|err| err.add_context(format!("regions[{}]", region_index)))?;

            let &BufferImageCopy {
                buffer_offset,
                buffer_row_length,
                buffer_image_height,
                image_subresource,
                image_offset,
                image_extent,
                _ne: _,
            } = region;

            /*
               Check image
            */

            if image_subresource.mip_level >= src_image.mip_levels() {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].image_subresource.mip_level` is not less than \
                        `src_image.mip_levels()`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageToBufferInfo2-imageSubresource-07967"],
                    ..Default::default()
                }));
            }

            let mut image_subresource_format = src_image_format;
            let mut image_subresource_extent =
                mip_level_extent(src_image.extent(), image_subresource.mip_level).unwrap();

            if src_image_format_planes.is_empty() {
                if !src_image_format_aspects.contains(image_subresource.aspects) {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`regions[{}].image_subresource.aspects` is not a subset of \
                            `src_image.format().aspects()`",
                            region_index,
                        )
                        .into(),
                        vuids: &["VUID-VkCopyImageToBufferInfo2-aspectMask-00211"],
                        ..Default::default()
                    }));
                }
            } else if src_image_format_planes.len() == 2 {
                match image_subresource.aspects {
                    ImageAspects::PLANE_0 => {
                        image_subresource_format = src_image_format_planes[0];
                    }
                    ImageAspects::PLANE_1 => {
                        image_subresource_format = src_image_format_planes[1];
                        image_subresource_extent = src_image_format_subsampled_extent;
                    }
                    _ => {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.format()` is a multi-planar format with two planes, \
                                but `regions[{}].image_subresource.aspect` is not \
                                `ImageAspects::PLANE_0` or `ImageAspects::PLANE_1`",
                                region_index,
                            )
                            .into(),
                            vuids: &[
                                "VUID-VkCopyImageToBufferInfo2-srcImage-07981",
                                "VUID-VkCopyImageToBufferInfo2-aspectMask-00211",
                            ],
                            ..Default::default()
                        }));
                    }
                }
            } else if src_image_format_planes.len() == 3 {
                match image_subresource.aspects {
                    ImageAspects::PLANE_0 => {
                        image_subresource_format = src_image_format_planes[0];
                    }
                    ImageAspects::PLANE_1 => {
                        image_subresource_format = src_image_format_planes[1];
                        image_subresource_extent = src_image_format_subsampled_extent;
                    }
                    ImageAspects::PLANE_2 => {
                        image_subresource_format = src_image_format_planes[2];
                        image_subresource_extent = src_image_format_subsampled_extent;
                    }
                    _ => {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.format()` is a multi-planar format with three planes, \
                                but `regions[{}].image_subresource.aspect` is not \
                                `ImageAspects::PLANE_0`, `ImageAspects::PLANE_1` or \
                                `ImageAspects::PLANE_2`",
                                region_index,
                            )
                            .into(),
                            vuids: &[
                                "VUID-VkCopyImageToBufferInfo2-srcImage-07982",
                                "VUID-VkCopyImageToBufferInfo2-aspectMask-00211",
                            ],
                            ..Default::default()
                        }));
                    }
                }
            }

            match src_image.image_type() {
                ImageType::Dim1d => {
                    if image_offset[1] != 0 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].image_offset[1]` is not 0",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyImageToBufferInfo2-srcImage-07979"],
                            ..Default::default()
                        }));
                    }

                    if image_extent[1] != 1 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].image_extent[1]` is not 1",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyImageToBufferInfo2-srcImage-07979"],
                            ..Default::default()
                        }));
                    }

                    if image_offset[2] != 0 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].image_offset[2]` is not 0",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyImageToBufferInfo2-srcImage-07980"],
                            ..Default::default()
                        }));
                    }

                    if image_extent[2] != 1 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].image_extent[2]` is not 1",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyImageToBufferInfo2-srcImage-07980"],
                            ..Default::default()
                        }));
                    }
                }
                ImageType::Dim2d => {
                    if image_offset[2] != 0 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.image_type()` is `ImageType::Dim2d`, but \
                                `regions[{}].image_offset[2]` is not 0",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyImageToBufferInfo2-srcImage-07980"],
                            ..Default::default()
                        }));
                    }

                    if image_extent[2] != 1 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.image_type()` is `ImageType::Dim2d`, but \
                                `regions[{}].image_extent[2]` is not 1",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyImageToBufferInfo2-srcImage-07980"],
                            ..Default::default()
                        }));
                    }
                }
                ImageType::Dim3d => {
                    if !(image_subresource.base_array_layer == 0
                        && image_subresource.layer_count == Some(1))
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.image_type()` is `ImageType::Dim3d`, but \
                                `regions[{0}].image_subresource.base_array_layer` and \
                                `regions[{0}].image_subresource.layer_count` are not `0` and \
                                `Some(1)`, respectively",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkCopyImageToBufferInfo2-srcImage-07983"],
                            ..Default::default()
                        }));
                    }
                }
            }

            if image_subresource.base_array_layer >= src_image.array_layers() {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].image_subresource.base_array_layer` is not less than \
                        `src_image.array_layers()`",
                        region_index,
                    )
                    .into(),
                    // vuids?
                    ..Default::default()
                }));
            }

            if let Some(image_subresource_layer_count) = image_subresource.layer_count {
                if image_subresource_layer_count
                    > src_image.array_layers() - image_subresource.base_array_layer
                {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`regions[{0}].image_subresource.base_array_layer + \
                            regions[{0}].image_subresource.layer_count` is greater than \
                            `src_image.array_layers()`",
                            region_index,
                        )
                        .into(),
                        vuids: &["VUID-VkCopyImageToBufferInfo2-imageSubresource-07968"],
                        ..Default::default()
                    }));
                }
            }

            if image_offset[0]
                .checked_add(image_extent[0])
                .is_none_or(|end| end > image_subresource_extent[0])
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].image_offset[0] + regions[{0}].image_extent[0]` is greater \
                        than coordinate 0 of the extent of the subresource of `src_image` \
                        selected by `regions[{0}].image_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageToBufferInfo2-imageOffset-00197"],
                    ..Default::default()
                }));
            }

            if image_offset[1]
                .checked_add(image_extent[1])
                .is_none_or(|end| end > image_subresource_extent[1])
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].image_offset[1] + regions[{0}].image_extent[1]` is greater \
                        than coordinate 1 of the extent of the subresource of `src_image` \
                        selected by `regions[{0}].image_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageToBufferInfo2-imageOffset-00198"],
                    ..Default::default()
                }));
            }

            if image_offset[2]
                .checked_add(image_extent[2])
                .is_none_or(|end| end > image_subresource_extent[2])
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].image_offset[2] + regions[{0}].image_extent[2]` is greater \
                        than coordinate 2 of the extent of the subresource of `src_image` \
                        selected by `regions[{0}].image_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageToBufferInfo2-imageOffset-00200"],
                    ..Default::default()
                }));
            }

            let image_subresource_format_block_extent = image_subresource_format.block_extent();

            if image_offset[0] % image_subresource_format_block_extent[0] != 0 {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].image_offset[0]` is not a multiple of coordinate 0 of the \
                        block extent of the format of the subresource of `src_image` \
                        selected by `regions[{0}].image_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageToBufferInfo2-pRegions-07274"],
                    ..Default::default()
                }));
            }

            if image_offset[1] % image_subresource_format_block_extent[1] != 0 {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].image_offset[1]` is not a multiple of coordinate 1 of the \
                        block extent of the format of the subresource of `src_image` \
                        selected by `regions[{0}].image_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageToBufferInfo2-pRegions-07275"],
                    ..Default::default()
                }));
            }

            if image_offset[2] % image_subresource_format_block_extent[2] != 0 {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].image_offset[2]` is not a multiple of coordinate 2 of the \
                        block extent of the format of the subresource of `src_image` \
                        selected by `regions[{0}].image_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageToBufferInfo2-pRegions-07276"],
                    ..Default::default()
                }));
            }

            if image_offset[0] + image_extent[0] != image_subresource_extent[0]
                && (image_offset[0] + image_extent[0]) % image_subresource_format_block_extent[0]
                    != 0
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].image_offset[0] + regions[{0}].image_extent[0]` is not \
                        equal to the extent of the subresource of `src_image` \
                        selected by `regions[{0}].image_subresource`, but \
                        it is also not a multiple of coordinate 0 of the block extent of the \
                        format of that subresource",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageToBufferInfo2-imageExtent-00207"],
                    ..Default::default()
                }));
            }

            if image_offset[1] + image_extent[1] != image_subresource_extent[1]
                && (image_offset[1] + image_extent[1]) % image_subresource_format_block_extent[1]
                    != 0
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].image_offset[1] + regions[{0}].image_extent[1]` is not \
                        equal to the extent of the subresource of `src_image` \
                        selected by `regions[{0}].image_subresource`, but \
                        it is also not a multiple of coordinate 1 of the block extent of the \
                        format of that subresource",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageToBufferInfo2-imageExtent-00208"],
                    ..Default::default()
                }));
            }

            if image_offset[2] + image_extent[2] != image_subresource_extent[2]
                && (image_offset[2] + image_extent[2]) % image_subresource_format_block_extent[2]
                    != 0
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].image_offset[2] + regions[{0}].image_extent[2]` is not \
                        equal to the extent of the subresource of `src_image` \
                        selected by `regions[{0}].image_subresource`, but \
                        it is also not a multiple of coordinate 2 of the block extent of the \
                        format of that subresource",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageToBufferInfo2-imageExtent-00209"],
                    ..Default::default()
                }));
            }

            /*
               Check buffer and image together
            */

            let image_subresource_format_block_size = image_subresource_format.block_size();

            if src_image_format_aspects.intersects(ImageAspects::DEPTH | ImageAspects::STENCIL) {
                if buffer_offset % 4 != 0 {
                    return Err(Box::new(ValidationError {
                        problem: "`src_image.format()` is a depth/stencil format, but \
                            `buffer_offset` is not a multiple of 4"
                            .into(),
                        vuids: &["VUID-VkCopyImageToBufferInfo2-srcImage-07978"],
                        ..Default::default()
                    }));
                }
            } else {
                if buffer_offset % image_subresource_format_block_size != 0 {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`src_image.format()` is not a depth/stencil format, but \
                            `buffer_offset` is not a multiple of the block size of the format of \
                            the subresource of `src_image` selected by \
                            `regions[{0}].image_subresource`",
                            region_index,
                        )
                        .into(),
                        vuids: &[
                            "VUID-VkCopyImageToBufferInfo2-srcImage-07975",
                            "VUID-VkCopyImageToBufferInfo2-srcImage-07976",
                        ],
                        ..Default::default()
                    }));
                }
            }

            if buffer_row_length % image_subresource_format_block_extent[0] != 0 {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].buffer_row_length` is not a multiple of coordinate 0 of \
                        the block extent of the format of the subresource of `src_image` \
                        selected by `regions[{0}].image_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageToBufferInfo2-bufferRowLength-00203"],
                    ..Default::default()
                }));
            }

            if buffer_image_height % image_subresource_format_block_extent[1] != 0 {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].buffer_image_height` is not a multiple of coordinate 1 of \
                        the block extent of the format of the subresource of `src_image` \
                        selected by `regions[{0}].image_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageToBufferInfo2-bufferImageHeight-00204"],
                    ..Default::default()
                }));
            }

            if (buffer_row_length / image_subresource_format_block_extent[0]) as DeviceSize
                * image_subresource_format_block_size
                > 0x7FFFFFFF
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].buffer_row_length`, divided by the block size of the \
                        format of the subresource of `src_image` selected by \
                        `regions[{0}].image_subresource`, and then multiplied by the block size \
                        of that subresource, is greater than 0x7FFFFFFF",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageToBufferInfo2-pRegions-07277"],
                    ..Default::default()
                }));
            }

            if region
                .buffer_copy_size(image_subresource_format, src_image.array_layers())
                .and_then(|copy_size| buffer_offset.checked_add(copy_size))
                .is_none_or(|end| end > dst_buffer.size())
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].buffer_offset` plus the number of bytes being copied \
                        is greater than `dst_buffer.size()`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkCopyImageToBufferInfo2-pRegions-00183"],
                    ..Default::default()
                }));
            }
        }

        // VUID-VkCopyImageToBufferInfo2-pRegions-00184
        // Can't occur as long as memory aliasing isn't allowed.

        Ok(())
    }

    pub(crate) fn to_vk2(
        &self,
        regions_vk: &'a [vk::BufferImageCopy2<'static>],
    ) -> vk::CopyImageToBufferInfo2<'a> {
        let &Self {
            src_image,
            src_image_layout,
            dst_buffer,
            regions: _,
            _ne: _,
        } = self;

        vk::CopyImageToBufferInfo2::default()
            .src_image(src_image.handle())
            .src_image_layout(src_image_layout.into())
            .dst_buffer(dst_buffer.handle())
            .regions(regions_vk)
    }

    pub(crate) fn to_vk2_regions(&self) -> SmallVec<[vk::BufferImageCopy2<'static>; 8]> {
        let &Self {
            src_image, regions, ..
        } = self;

        if regions.is_empty() {
            let remaining_array_layers = src_image.array_layers();
            let region_vk = vk::BufferImageCopy2::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(src_image.subresource_layers().to_vk(remaining_array_layers))
                .image_offset(convert_offset([0; 3]))
                .image_extent(convert_extent(src_image.extent()));

            smallvec![region_vk]
        } else {
            regions
                .iter()
                .map(|region| region.to_vk2(src_image))
                .collect()
        }
    }

    pub(crate) fn to_vk(&self) -> CopyImageToBufferInfoVk {
        let &Self {
            src_image,
            src_image_layout,
            dst_buffer,
            regions: _,
            _ne: _,
        } = self;

        CopyImageToBufferInfoVk {
            src_image_vk: src_image.handle(),
            src_image_layout_vk: src_image_layout.into(),
            dst_buffer_vk: dst_buffer.handle(),
        }
    }

    pub(crate) fn to_vk_regions(&self) -> SmallVec<[vk::BufferImageCopy; 8]> {
        let &Self {
            src_image, regions, ..
        } = self;

        if regions.is_empty() {
            let remaining_array_layers = src_image.array_layers();
            let region_vk = vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: src_image.subresource_layers().to_vk(remaining_array_layers),
                image_offset: convert_offset([0; 3]),
                image_extent: convert_extent(src_image.extent()),
            };

            smallvec![region_vk]
        } else {
            regions
                .iter()
                .map(|region| region.to_vk(src_image))
                .collect()
        }
    }
}

pub(crate) struct CopyImageToBufferInfoVk {
    pub(crate) src_image_vk: vk::Image,
    pub(crate) src_image_layout_vk: vk::ImageLayout,
    pub(crate) dst_buffer_vk: vk::Buffer,
}

/// A region of data to copy between a buffer and an image.
#[derive(Clone, Debug)]
pub struct BufferImageCopy<'a> {
    /// The offset in bytes from the start of the buffer that copying will start from.
    ///
    /// The default value is `0`.
    pub buffer_offset: DeviceSize,

    /// The number of texels between successive rows of image data in the buffer.
    ///
    /// If set to `0`, the width of the image is used.
    ///
    /// The default value is `0`.
    pub buffer_row_length: u32,

    /// The number of rows between successive depth slices of image data in the buffer.
    ///
    /// If set to `0`, the height of the image is used.
    ///
    /// The default value is `0`.
    pub buffer_image_height: u32,

    /// The subresource of the image to copy from/to.
    ///
    /// The default value is [`ImageSubresourceLayers::default()`].
    pub image_subresource: ImageSubresourceLayers,

    /// The offset from the zero coordinate of the image that copying will start from.
    ///
    /// The default value is `[0; 3]`.
    pub image_offset: [u32; 3],

    /// The extent of texels in the image to copy.
    ///
    /// The default value is `[0; 3]`, which must be overridden.
    pub image_extent: [u32; 3],

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for BufferImageCopy<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl BufferImageCopy<'_> {
    /// Returns a default `BufferImageCopy`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: ImageSubresourceLayers::new(),
            image_offset: [0; 3],
            image_extent: [0; 3],
            _ne: crate::NE,
        }
    }

    // Following
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap20.html#copies-buffers-images-addressing
    pub(crate) fn buffer_copy_size(&self, format: Format, array_layers: u32) -> Option<DeviceSize> {
        let &Self {
            buffer_offset: _,
            mut buffer_row_length,
            mut buffer_image_height,
            image_subresource,
            image_offset: _,
            mut image_extent,
            _ne: _,
        } = self;

        if buffer_row_length == 0 {
            buffer_row_length = image_extent[0];
        }

        if buffer_image_height == 0 {
            buffer_image_height = image_extent[1];
        }

        // Scale down from texels to texel blocks, rounding up if needed.
        let block_extent = format.block_extent();
        buffer_row_length = buffer_row_length.div_ceil(block_extent[0]);
        buffer_image_height = buffer_image_height.div_ceil(block_extent[1]);

        for i in 0..3 {
            image_extent[i] = image_extent[i].div_ceil(block_extent[i]);
        }

        // Only one of these is greater than 1, take the greater number.
        image_extent[2] = max(
            image_extent[2],
            image_subresource
                .layer_count
                .unwrap_or(array_layers - image_subresource.base_array_layer),
        );

        let blocks_to_last_slice = (image_extent[2] as DeviceSize - 1)
            .checked_mul(buffer_image_height as DeviceSize)?
            .checked_mul(buffer_row_length as DeviceSize)?;
        let blocks_to_last_row =
            (image_extent[1] as DeviceSize - 1).checked_mul(buffer_row_length as DeviceSize)?;
        let num_blocks = blocks_to_last_slice
            .checked_add(blocks_to_last_row)?
            .checked_add(image_extent[0] as DeviceSize)?;

        num_blocks.checked_mul(format.block_size())
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            buffer_offset: _,
            buffer_row_length,
            buffer_image_height,
            image_subresource,
            image_offset: _,
            image_extent,
            _ne: _,
        } = self;

        image_subresource
            .validate(device)
            .map_err(|err| err.add_context("image_subresource"))?;

        if !(buffer_row_length == 0 || buffer_row_length >= image_extent[0]) {
            return Err(Box::new(ValidationError {
                problem: "`buffer_row_length` is not either zero, or greater than or equal to \
                    `image_extent[0]`"
                    .into(),
                vuids: &["VUID-VkBufferImageCopy2-bufferRowLength-00195"],
                ..Default::default()
            }));
        }

        if !(buffer_image_height == 0 || buffer_image_height >= image_extent[1]) {
            return Err(Box::new(ValidationError {
                problem: "`buffer_image_height` is not either zero, or greater than or equal to \
                    `image_extent[1]`"
                    .into(),
                vuids: &["VUID-VkBufferImageCopy2-bufferImageHeight-00196"],
                ..Default::default()
            }));
        }

        if image_subresource.aspects.count() != 1 {
            return Err(Box::new(ValidationError {
                context: "image_subresource.aspects".into(),
                problem: "contains more than one aspect".into(),
                vuids: &["VUID-VkBufferImageCopy2-aspectMask-00212"],
                ..Default::default()
            }));
        }

        if image_extent[0] == 0 {
            return Err(Box::new(ValidationError {
                context: "image_extent[0]".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkBufferImageCopy2-imageExtent-06659"],
                ..Default::default()
            }));
        }

        if image_extent[1] == 0 {
            return Err(Box::new(ValidationError {
                context: "image_extent[1]".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkBufferImageCopy2-imageExtent-06660"],
                ..Default::default()
            }));
        }

        if image_extent[2] == 0 {
            return Err(Box::new(ValidationError {
                context: "image_extent[2]".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkBufferImageCopy2-imageExtent-06661"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk2(&self, image: &Image) -> vk::BufferImageCopy2<'static> {
        let &Self {
            buffer_offset,
            buffer_row_length,
            buffer_image_height,
            image_subresource,
            image_offset,
            image_extent,
            _ne: _,
        } = self;

        let remaining_array_layers = image.array_layers() - image_subresource.base_array_layer;

        vk::BufferImageCopy2::default()
            .buffer_offset(buffer_offset)
            .buffer_row_length(buffer_row_length)
            .buffer_image_height(buffer_image_height)
            .image_subresource(image_subresource.to_vk(remaining_array_layers))
            .image_offset(vk::Offset3D {
                x: image_offset[0] as i32,
                y: image_offset[1] as i32,
                z: image_offset[2] as i32,
            })
            .image_extent(vk::Extent3D {
                width: image_extent[0],
                height: image_extent[1],
                depth: image_extent[2],
            })
    }

    pub(crate) fn to_vk(&self, image: &Image) -> vk::BufferImageCopy {
        let &Self {
            buffer_offset,
            buffer_row_length,
            buffer_image_height,
            image_subresource,
            image_offset,
            image_extent,
            _ne: _,
        } = self;

        let remaining_array_layers = image.array_layers() - image_subresource.base_array_layer;

        vk::BufferImageCopy {
            buffer_offset,
            buffer_row_length,
            buffer_image_height,
            image_subresource: image_subresource.to_vk(remaining_array_layers),
            image_offset: vk::Offset3D {
                x: image_offset[0] as i32,
                y: image_offset[1] as i32,
                z: image_offset[2] as i32,
            },
            image_extent: vk::Extent3D {
                width: image_extent[0],
                height: image_extent[1],
                depth: image_extent[2],
            },
        }
    }
}

/// Parameters to blit image data.
#[derive(Clone, Debug)]
pub struct BlitImageInfo<'a> {
    /// The image to blit from.
    ///
    /// There is no default value.
    pub src_image: &'a Image,

    /// The layout used for `src_image` during the blit operation.
    ///
    /// The following layouts are allowed:
    /// - [`ImageLayout::TransferSrcOptimal`]
    /// - [`ImageLayout::General`]
    ///
    /// The default value is [`ImageLayout::TransferSrcOptimal`].
    pub src_image_layout: ImageLayout,

    /// The image to blit to.
    ///
    /// There is no default value.
    pub dst_image: &'a Image,

    /// The layout used for `dst_image` during the blit operation.
    ///
    /// The following layouts are allowed:
    /// - [`ImageLayout::TransferDstOptimal`]
    /// - [`ImageLayout::General`]
    ///
    /// The default value is [`ImageLayout::TransferDstOptimal`].
    pub dst_image_layout: ImageLayout,

    /// The regions of both images to blit between.
    ///
    /// The default value is a single region, covering the first mip level, and the smallest of the
    /// array layers of the two images. The whole extent of each image is covered, scaling if
    /// necessary. All aspects of each image are selected, or `plane0` if the image is
    /// multi-planar.
    pub regions: &'a [ImageBlit<'a>],

    /// The filter to use for sampling `src_image` when the `src_extent` and
    /// `dst_extent` of a region are not the same size.
    ///
    /// The default value is [`Filter::Nearest`].
    pub filter: Filter,

    pub _ne: crate::NonExhaustive<'a>,
}

impl<'a> BlitImageInfo<'a> {
    /// Returns a default `BlitImageInfo` with the provided `src_image` and `dst_image`.
    #[inline]
    pub const fn new(src_image: &'a Image, dst_image: &'a Image) -> Self {
        Self {
            src_image,
            src_image_layout: ImageLayout::TransferSrcOptimal,
            dst_image,
            dst_image_layout: ImageLayout::TransferDstOptimal,
            regions: &[],
            filter: Filter::Nearest,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            src_image,
            src_image_layout,
            dst_image,
            dst_image_layout,
            regions,
            filter,
            _ne: _,
        } = self;

        src_image_layout.validate_device(device).map_err(|err| {
            err.add_context("src_image_layout")
                .set_vuids(&["VUID-VkBlitImageInfo2-srcImageLayout-parameter"])
        })?;

        dst_image_layout.validate_device(device).map_err(|err| {
            err.add_context("dst_image_layout")
                .set_vuids(&["VUID-VkBlitImageInfo2-dstImageLayout-parameter"])
        })?;

        filter.validate_device(device).map_err(|err| {
            err.add_context("filter")
                .set_vuids(&["VUID-VkBlitImageInfo2-filter-parameter"])
        })?;

        // VUID-VkBlitImageInfo2-commonparent
        assert_eq!(device, src_image.device().as_ref());
        assert_eq!(device, dst_image.device().as_ref());

        let src_image_format = src_image.format();
        let src_image_format_aspects = src_image_format.aspects();

        let dst_image_format = dst_image.format();
        let dst_image_format_aspects = dst_image_format.aspects();

        if !src_image.usage().intersects(ImageUsage::TRANSFER_SRC) {
            return Err(Box::new(ValidationError {
                context: "src_image.usage()".into(),
                problem: "does not contain `ImageUsage::TRANSFER_SRC`".into(),
                vuids: &["VUID-VkBlitImageInfo2-srcImage-00219"],
                ..Default::default()
            }));
        }

        if !src_image
            .format_features()
            .intersects(FormatFeatures::BLIT_SRC)
        {
            return Err(Box::new(ValidationError {
                context: "src_image.format_features()".into(),
                problem: "does not contain `FormatFeatures::BLIT_SRC`".into(),
                vuids: &["VUID-VkBlitImageInfo2-srcImage-01999"],
                ..Default::default()
            }));
        }

        if src_image_format.ycbcr_chroma_sampling().is_some() {
            return Err(Box::new(ValidationError {
                context: "src_image.format()".into(),
                problem: "is a YCbCr format".into(),
                vuids: &["VUID-VkBlitImageInfo2-srcImage-06421"],
                ..Default::default()
            }));
        }

        if src_image.samples() != SampleCount::Sample1 {
            return Err(Box::new(ValidationError {
                context: "src_image.samples()".into(),
                problem: "is not `SampleCount::Sample1`".into(),
                vuids: &["VUID-VkBlitImageInfo2-srcImage-00233"],
                ..Default::default()
            }));
        }

        if !matches!(
            src_image_layout,
            ImageLayout::TransferSrcOptimal | ImageLayout::General
        ) {
            return Err(Box::new(ValidationError {
                context: "src_image_layout".into(),
                problem: "is not `ImageLayout::TransferSrcOptimal` or `ImageLayout::General`"
                    .into(),
                vuids: &["VUID-VkBlitImageInfo2-srcImageLayout-01398"],
                ..Default::default()
            }));
        }

        if !dst_image.usage().intersects(ImageUsage::TRANSFER_DST) {
            return Err(Box::new(ValidationError {
                context: "dst_image.usage()".into(),
                problem: "does not contain `ImageUsage::TRANSFER_DST`".into(),
                vuids: &["VUID-VkBlitImageInfo2-dstImage-00224"],
                ..Default::default()
            }));
        }

        if !dst_image
            .format_features()
            .intersects(FormatFeatures::BLIT_DST)
        {
            return Err(Box::new(ValidationError {
                context: "dst_image.format_features()".into(),
                problem: "does not contain `FormatFeatures::BLIT_DST`".into(),
                vuids: &["VUID-VkBlitImageInfo2-dstImage-02000"],
                ..Default::default()
            }));
        }

        if dst_image_format.ycbcr_chroma_sampling().is_some() {
            return Err(Box::new(ValidationError {
                context: "dst_image.format()".into(),
                problem: "is a YCbCr format".into(),
                vuids: &["VUID-VkBlitImageInfo2-dstImage-06422"],
                ..Default::default()
            }));
        }

        if dst_image.samples() != SampleCount::Sample1 {
            return Err(Box::new(ValidationError {
                context: "dst_image.samples()".into(),
                problem: "is not `SampleCount::Sample1`".into(),
                vuids: &["VUID-VkBlitImageInfo2-dstImage-00234"],
                ..Default::default()
            }));
        }

        if !matches!(
            dst_image_layout,
            ImageLayout::TransferDstOptimal | ImageLayout::General
        ) {
            return Err(Box::new(ValidationError {
                context: "dst_image_layout".into(),
                problem: "is not `ImageLayout::TransferDstOptimal` or `ImageLayout::General`"
                    .into(),
                vuids: &["VUID-VkBlitImageInfo2-dstImageLayout-01399"],
                ..Default::default()
            }));
        }

        if src_image_format_aspects.intersects(ImageAspects::DEPTH | ImageAspects::STENCIL)
            || dst_image_format_aspects.intersects(ImageAspects::DEPTH | ImageAspects::STENCIL)
        {
            if src_image_format != dst_image_format {
                return Err(Box::new(ValidationError {
                    problem: "one of `src_image.format()` or `dst_image.format()` is a \
                        depth/stencil format, but they are not equal"
                        .into(),
                    vuids: &["VUID-VkBlitImageInfo2-srcImage-00231"],
                    ..Default::default()
                }));
            }
        } else {
            if src_image_format
                .numeric_format_color()
                .unwrap()
                .numeric_type()
                != dst_image_format
                    .numeric_format_color()
                    .unwrap()
                    .numeric_type()
            {
                return Err(Box::new(ValidationError {
                    problem: "neither `src_image.format()` nor `dst_image.format()` is a \
                        depth/stencil format, but their numeric types are not equal"
                        .into(),
                    vuids: &[
                        "VUID-VkBlitImageInfo2-srcImage-00229",
                        "VUID-VkBlitImageInfo2-srcImage-00230",
                    ],
                    ..Default::default()
                }));
            }
        }

        if src_image_format_aspects.intersects(ImageAspects::DEPTH | ImageAspects::STENCIL)
            && filter != Filter::Nearest
        {
            return Err(Box::new(ValidationError {
                problem: "`src_image.format()` is a depth/stencil format, but \
                    `filter` is not `Filter::Nearest`"
                    .into(),
                vuids: &["VUID-VkBlitImageInfo2-srcImage-00232"],
                ..Default::default()
            }));
        }

        match filter {
            Filter::Nearest => (),
            Filter::Linear => {
                if !src_image
                    .format_features()
                    .intersects(FormatFeatures::SAMPLED_IMAGE_FILTER_LINEAR)
                {
                    return Err(Box::new(ValidationError {
                        problem: "`filter` is `Filter::Linear`, but \
                            `src_image.format_features()` do not contain \
                            `FormatFeatures::SAMPLED_IMAGE_FILTER_LINEAR`"
                            .into(),
                        vuids: &["VUID-VkBlitImageInfo2-filter-02001"],
                        ..Default::default()
                    }));
                }
            }
            Filter::Cubic => {
                if !src_image
                    .format_features()
                    .intersects(FormatFeatures::SAMPLED_IMAGE_FILTER_CUBIC)
                {
                    return Err(Box::new(ValidationError {
                        problem: "`filter` is `Filter::Cubic`, but \
                            `src_image.format_features()` do not contain \
                            `FormatFeatures::SAMPLED_IMAGE_FILTER_CUBIC`"
                            .into(),
                        vuids: &["VUID-VkBlitImageInfo2-filter-02002"],
                        ..Default::default()
                    }));
                }

                if src_image.image_type() != ImageType::Dim2d {
                    return Err(Box::new(ValidationError {
                        problem: "`filter` is `Filter::Cubic`, but \
                            `src_image.image_type()` is not `ImageType::Dim2d`"
                            .into(),
                        vuids: &["VUID-VkBlitImageInfo2-filter-00237"],
                        ..Default::default()
                    }));
                }
            }
        }

        let is_same_image = src_image == dst_image;
        let mut overlap_subresource_indices = None;
        let mut overlap_extent_indices = None;

        for (region_index, region) in regions.iter().enumerate() {
            region
                .validate(device)
                .map_err(|err| err.add_context(format!("regions[{}]", region_index)))?;

            let &ImageBlit {
                src_subresource,
                src_offsets,
                dst_subresource,
                dst_offsets,
                _ne: _,
            } = region;

            /*
               Check src
            */

            if src_subresource.mip_level >= src_image.mip_levels() {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].src_subresource.mip_level` is not less than \
                        `src_image.mip_levels()`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkBlitImageInfo2-srcSubresource-01705"],
                    ..Default::default()
                }));
            }

            let src_subresource_extent =
                mip_level_extent(src_image.extent(), src_subresource.mip_level).unwrap();

            if !src_image_format_aspects.contains(src_subresource.aspects) {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].src_subresource.aspects` is not a subset of \
                        `src_image.format().aspects()`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkBlitImageInfo2-aspectMask-00241"],
                    ..Default::default()
                }));
            }

            match src_image.image_type() {
                ImageType::Dim1d => {
                    if src_offsets[0][1] != 0 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].src_offsets[0][1]` is not 0",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkBlitImageInfo2-srcImage-00245"],
                            ..Default::default()
                        }));
                    }

                    if src_offsets[1][1] != 1 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].src_offsets[1][1]` is not 1",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkBlitImageInfo2-srcImage-00245"],
                            ..Default::default()
                        }));
                    }

                    if src_offsets[0][2] != 0 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].src_offsets[0][2]` is not 0",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkBlitImageInfo2-srcImage-00247"],
                            ..Default::default()
                        }));
                    }

                    if src_offsets[1][2] != 1 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].src_offsets[1][2]` is not 1",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkBlitImageInfo2-srcImage-00247"],
                            ..Default::default()
                        }));
                    }
                }
                ImageType::Dim2d => {
                    if src_offsets[0][2] != 0 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.image_type()` is `ImageType::Dim2d`, but \
                                `regions[{}].src_offsets[0][2]` is not 0",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkBlitImageInfo2-srcImage-00247"],
                            ..Default::default()
                        }));
                    }

                    if src_offsets[1][2] != 1 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.image_type()` is `ImageType::Dim2d`, but \
                                `regions[{}].src_offsets[1][2]` is not 1",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkBlitImageInfo2-srcImage-00247"],
                            ..Default::default()
                        }));
                    }
                }
                ImageType::Dim3d => {
                    if !(src_subresource.base_array_layer == 0
                        && src_subresource.layer_count == Some(1))
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.image_type()` is `ImageType::Dim3d`, but \
                                `regions[{0}].src_subresource.base_array_layer` and \
                                `regions[{0}].src_subresource.layer_count` are not `0` and \
                                `Some(1)`, respectively",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkBlitImageInfo2-srcImage-00240"],
                            ..Default::default()
                        }));
                    }
                }
            }

            if src_subresource.base_array_layer >= src_image.array_layers() {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].src_subresource.base_array_layer` is not less than \
                        `src_image.array_layers()`",
                        region_index,
                    )
                    .into(),
                    // vuids?
                    ..Default::default()
                }));
            }

            if let Some(src_subresource_layer_count) = src_subresource.layer_count {
                if src_subresource_layer_count
                    > src_image.array_layers() - src_subresource.base_array_layer
                {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`regions[{0}].src_subresource.base_array_layer + \
                            regions[{0}].src_subresource.layer_count` is greater than \
                            `src_image.array_layers()`",
                            region_index,
                        )
                        .into(),
                        vuids: &["VUID-VkBlitImageInfo2-srcSubresource-01707"],
                        ..Default::default()
                    }));
                }
            }

            let src_offsets_max = [
                max(src_offsets[0][0], src_offsets[1][0]),
                max(src_offsets[0][1], src_offsets[1][1]),
                max(src_offsets[0][2], src_offsets[1][2]),
            ];

            if src_offsets_max[0] > src_subresource_extent[0] {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`max(regions[{0}].src_offsets[0][0], regions[{0}].src_offsets[1][0])` is \
                        greater than coordinate 0 of the extent of the subresource of \
                        `src_image` selected by `regions[{0}].src_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkBlitImageInfo2-srcOffset-00243"],
                    ..Default::default()
                }));
            }

            if src_offsets_max[1] > src_subresource_extent[1] {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`max(regions[{0}].src_offsets[0][1], regions[{0}].src_offsets[1][1])` is \
                        greater than coordinate 1 of the extent of the subresource of \
                        `src_image` selected by `regions[{0}].src_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkBlitImageInfo2-srcOffset-00244"],
                    ..Default::default()
                }));
            }

            if src_offsets_max[2] > src_subresource_extent[2] {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`max(regions[{0}].src_offsets[0][2], regions[{0}].src_offsets[1][2])` is \
                        greater than coordinate 2 of the extent of the subresource of \
                        `src_image` selected by `regions[{0}].src_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkBlitImageInfo2-srcOffset-00246"],
                    ..Default::default()
                }));
            }

            /*
               Check dst
            */

            if dst_subresource.mip_level >= dst_image.mip_levels() {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].dst_subresource.mip_level` is not less than \
                        `dst_image.mip_levels()`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkBlitImageInfo2-srcSubresource-01705"],
                    ..Default::default()
                }));
            }

            let dst_subresource_extent =
                mip_level_extent(dst_image.extent(), dst_subresource.mip_level).unwrap();

            if !dst_image_format_aspects.contains(dst_subresource.aspects) {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].dst_subresource.aspects` is not a subset of \
                        `dst_image.format().aspects()`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkBlitImageInfo2-aspectMask-00242"],
                    ..Default::default()
                }));
            }

            match dst_image.image_type() {
                ImageType::Dim1d => {
                    if dst_offsets[0][1] != 0 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`dst_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].dst_offsets[0][1]` is not 0",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkBlitImageInfo2-dstImage-00250"],
                            ..Default::default()
                        }));
                    }

                    if dst_offsets[1][1] != 1 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`dst_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].dst_offsets[1][1]` is not 1",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkBlitImageInfo2-dstImage-00250"],
                            ..Default::default()
                        }));
                    }

                    if dst_offsets[0][2] != 0 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`dst_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].dst_offsets[0][2]` is not 0",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkBlitImageInfo2-dstImage-00252"],
                            ..Default::default()
                        }));
                    }

                    if dst_offsets[1][2] != 1 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`dst_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].dst_offsets[1][2]` is not 1",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkBlitImageInfo2-dstImage-00252"],
                            ..Default::default()
                        }));
                    }
                }
                ImageType::Dim2d => {
                    if dst_offsets[0][2] != 0 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`dst_image.image_type()` is `ImageType::Dim2d`, but \
                                `regions[{}].dst_offsets[0][2]` is not 0",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkBlitImageInfo2-dstImage-00252"],
                            ..Default::default()
                        }));
                    }

                    if dst_offsets[1][2] != 1 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`dst_image.image_type()` is `ImageType::Dim2d`, but \
                                `regions[{}].dst_offsets[1][2]` is not 1",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkBlitImageInfo2-dstImage-00252"],
                            ..Default::default()
                        }));
                    }
                }
                ImageType::Dim3d => {
                    if !(dst_subresource.base_array_layer == 0
                        && dst_subresource.layer_count == Some(1))
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`dst_image.image_type()` is `ImageType::Dim3d`, but \
                                `regions[{0}].dst_subresource.base_array_layer` and \
                                `regions[{0}].dst_subresource.layer_count` are not `0` and \
                                `Some(1)`, respectively",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkBlitImageInfo2-srcImage-00240"],
                            ..Default::default()
                        }));
                    }
                }
            }

            if dst_subresource.base_array_layer >= dst_image.array_layers() {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].dst_subresource.base_array_layer` is not less than \
                        `dst_image.array_layers()`",
                        region_index,
                    )
                    .into(),
                    // vuids?
                    ..Default::default()
                }));
            }

            if let Some(dst_subresource_layer_count) = dst_subresource.layer_count {
                if dst_subresource_layer_count
                    > dst_image.array_layers() - dst_subresource.base_array_layer
                {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`regions[{0}].dst_subresource.base_array_layer + \
                            regions[{0}].dst_subresource.layer_count` is greater than \
                            `dst_image.array_layers()`",
                            region_index,
                        )
                        .into(),
                        vuids: &["VUID-VkBlitImageInfo2-srcSubresource-01707"],
                        ..Default::default()
                    }));
                }
            }

            let dst_offsets_max = [
                max(dst_offsets[0][0], dst_offsets[1][0]),
                max(dst_offsets[0][1], dst_offsets[1][1]),
                max(dst_offsets[0][2], dst_offsets[1][2]),
            ];

            if dst_offsets_max[0] > dst_subresource_extent[0] {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`max(regions[{0}].dst_offsets[0][0], regions[{0}].dst_offsets[1][0])` is \
                        greater than coordinate 0 of the extent of the subresource of \
                        `dst_image` selected by `regions[{0}].dst_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkBlitImageInfo2-dstOffset-00248"],
                    ..Default::default()
                }));
            }

            if dst_offsets_max[1] > dst_subresource_extent[1] {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`max(regions[{0}].dst_offsets[0][1], regions[{0}].dst_offsets[1][1])` is \
                        greater than coordinate 1 of the extent of the subresource of \
                        `dst_image` selected by `regions[{0}].dst_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkBlitImageInfo2-dstOffset-00249"],
                    ..Default::default()
                }));
            }

            if dst_offsets_max[2] > dst_subresource_extent[2] {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`max(regions[{0}].dst_offsets[0][2], regions[{0}].dst_offsets[1][2])` is \
                        greater than coordinate 2 of the extent of the subresource of \
                        `dst_image` selected by `regions[{0}].dst_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkBlitImageInfo2-dstOffset-00251"],
                    ..Default::default()
                }));
            }

            // VUID-VkBlitImageInfo2-pRegions-00217
            if is_same_image {
                let src_region_index = region_index;
                let src_subresource_axes = [
                    src_subresource.mip_level..src_subresource.mip_level + 1,
                    src_subresource.base_array_layer
                        ..src_subresource.layer_count.map_or(
                            src_image.array_layers(),
                            |src_subresource_layer_count| {
                                src_subresource.base_array_layer + src_subresource_layer_count
                            },
                        ),
                ];
                let src_extent_axes = [
                    min(src_offsets[0][0], src_offsets[1][0])
                        ..max(src_offsets[0][0], src_offsets[1][0]),
                    min(src_offsets[0][1], src_offsets[1][1])
                        ..max(src_offsets[0][1], src_offsets[1][1]),
                    min(src_offsets[0][2], src_offsets[1][2])
                        ..max(src_offsets[0][2], src_offsets[1][2]),
                ];

                for (dst_region_index, dst_region) in regions.iter().enumerate() {
                    let &ImageBlit {
                        dst_subresource,
                        dst_offsets,
                        ..
                    } = dst_region;

                    let dst_subresource_axes = [
                        dst_subresource.mip_level..dst_subresource.mip_level + 1,
                        dst_subresource.base_array_layer
                            ..dst_subresource.layer_count.map_or(
                                dst_image.array_layers(),
                                |dst_subresource_layer_count| {
                                    dst_subresource.base_array_layer + dst_subresource_layer_count
                                },
                            ),
                    ];

                    if src_subresource_axes.iter().zip(dst_subresource_axes).any(
                        |(src_range, dst_range)| {
                            src_range.start >= dst_range.end || dst_range.start >= src_range.end
                        },
                    ) {
                        continue;
                    }

                    // If the subresource axes all overlap, then the source and destination must
                    // have the same layout.
                    overlap_subresource_indices = Some((src_region_index, dst_region_index));

                    let dst_extent_axes = [
                        min(dst_offsets[0][0], dst_offsets[1][0])
                            ..max(dst_offsets[0][0], dst_offsets[1][0]),
                        min(dst_offsets[0][1], dst_offsets[1][1])
                            ..max(dst_offsets[0][1], dst_offsets[1][1]),
                        min(dst_offsets[0][2], dst_offsets[1][2])
                            ..max(dst_offsets[0][2], dst_offsets[1][2]),
                    ];

                    if src_extent_axes
                        .iter()
                        .zip(dst_extent_axes)
                        .any(|(src_range, dst_range)| {
                            src_range.start >= dst_range.end || dst_range.start >= src_range.end
                        })
                    {
                        continue;
                    }

                    // If the extent axes *also* overlap, then that's an error.
                    overlap_extent_indices = Some((src_region_index, dst_region_index));
                }
            }
        }

        if let Some((src_region_index, dst_region_index)) = overlap_extent_indices {
            return Err(Box::new(ValidationError {
                problem: format!(
                    "`src_image` is equal to `dst_image`, and `regions[{0}].src_subresource` \
                    overlaps with `regions[{1}].dst_subresource`, but \
                    the `src_offsets` of `regions[{0}]` overlaps with \
                    the `dst_offsets` of `regions[{1}]`",
                    src_region_index, dst_region_index,
                )
                .into(),
                vuids: &["VUID-VkBlitImageInfo2-pRegions-00217"],
                ..Default::default()
            }));
        }

        if let Some((src_region_index, dst_region_index)) = overlap_subresource_indices {
            if src_image_layout != dst_image_layout {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`src_image` is equal to `dst_image`, and `regions[{0}].src_subresource` \
                        overlaps with `regions[{1}].dst_subresource`, but \
                        `src_image_layout` does not equal `dst_image_layout`",
                        src_region_index, dst_region_index,
                    )
                    .into(),
                    vuids: &[
                        "VUID-VkBlitImageInfo2-srcImageLayout-00221",
                        "VUID-VkBlitImageInfo2-dstImageLayout-00226",
                    ],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    pub(crate) fn to_vk2(
        &self,
        regions_vk: &'a [vk::ImageBlit2<'static>],
    ) -> vk::BlitImageInfo2<'a> {
        let &Self {
            src_image,
            src_image_layout,
            dst_image,
            dst_image_layout,
            regions: _,
            filter,
            _ne: _,
        } = self;

        vk::BlitImageInfo2::default()
            .src_image(src_image.handle())
            .src_image_layout(src_image_layout.into())
            .dst_image(dst_image.handle())
            .dst_image_layout(dst_image_layout.into())
            .regions(regions_vk)
            .filter(filter.into())
    }

    pub(crate) fn to_vk2_regions(&self) -> SmallVec<[vk::ImageBlit2<'static>; 8]> {
        let &Self {
            src_image,
            dst_image,
            regions,
            ..
        } = self;

        if regions.is_empty() {
            let remaining_array_layers = min(src_image.array_layers(), dst_image.array_layers());
            let region_vk = vk::ImageBlit2::default()
                .src_subresource(src_image.subresource_layers().to_vk(remaining_array_layers))
                .src_offsets([[0; 3], src_image.extent()].map(convert_offset))
                .dst_subresource(dst_image.subresource_layers().to_vk(remaining_array_layers))
                .dst_offsets([[0; 3], dst_image.extent()].map(convert_offset));

            smallvec![region_vk]
        } else {
            regions
                .iter()
                .map(|region| region.to_vk2(src_image, dst_image))
                .collect()
        }
    }

    pub(crate) fn to_vk(&self) -> BlitImageInfoVk {
        let &Self {
            src_image,
            src_image_layout,
            dst_image,
            dst_image_layout,
            regions: _,
            filter,
            _ne: _,
        } = self;

        BlitImageInfoVk {
            src_image_vk: src_image.handle(),
            src_image_layout_vk: src_image_layout.into(),
            dst_image_vk: dst_image.handle(),
            dst_image_layout_vk: dst_image_layout.into(),
            filter_vk: filter.into(),
        }
    }

    pub(crate) fn to_vk_regions(&self) -> SmallVec<[vk::ImageBlit; 8]> {
        let &Self {
            src_image,
            dst_image,
            regions,
            ..
        } = self;

        if regions.is_empty() {
            let remaining_array_layers = min(src_image.array_layers(), dst_image.array_layers());
            let region_vk = vk::ImageBlit {
                src_subresource: src_image.subresource_layers().to_vk(remaining_array_layers),
                src_offsets: [[0; 3], src_image.extent()].map(convert_offset),
                dst_subresource: dst_image.subresource_layers().to_vk(remaining_array_layers),
                dst_offsets: [[0; 3], dst_image.extent()].map(convert_offset),
            };

            smallvec![region_vk]
        } else {
            regions
                .iter()
                .map(|region| region.to_vk(src_image, dst_image))
                .collect()
        }
    }
}

pub(crate) struct BlitImageInfoVk {
    pub(crate) src_image_vk: vk::Image,
    pub(crate) src_image_layout_vk: vk::ImageLayout,
    pub(crate) dst_image_vk: vk::Image,
    pub(crate) dst_image_layout_vk: vk::ImageLayout,
    pub(crate) filter_vk: vk::Filter,
}

/// A region of data to blit between images.
#[derive(Clone, Debug)]
pub struct ImageBlit<'a> {
    /// The subresource of `src_image` to blit from.
    ///
    /// The default value is [`ImageSubresourceLayers::default()`].
    pub src_subresource: ImageSubresourceLayers,

    /// The offsets from the zero coordinate of `src_image`, defining two corners of the region
    /// to blit from.
    /// If the ordering of the two offsets differs between source and destination, the image will
    /// be flipped.
    ///
    /// The default value is `[[0; 3]; 2]`, which must be overridden.
    pub src_offsets: [[u32; 3]; 2],

    /// The subresource of `dst_image` to blit to.
    ///
    /// The default value is [`ImageSubresourceLayers::default()`].
    pub dst_subresource: ImageSubresourceLayers,

    /// The offset from the zero coordinate of `dst_image` defining two corners of the
    /// region to blit to.
    /// If the ordering of the two offsets differs between source and destination, the image will
    /// be flipped.
    ///
    /// The default value is `[[0; 3]; 2]`, which must be overridden.
    pub dst_offsets: [[u32; 3]; 2],

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for ImageBlit<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl ImageBlit<'_> {
    /// Returns a default `ImageBlit`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            src_subresource: ImageSubresourceLayers::new(),
            src_offsets: [[0; 3]; 2],
            dst_subresource: ImageSubresourceLayers::new(),
            dst_offsets: [[0; 3]; 2],
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            src_subresource,
            src_offsets: _,
            dst_subresource,
            dst_offsets: _,
            _ne: _,
        } = self;

        src_subresource
            .validate(device)
            .map_err(|err| err.add_context("src_subresource"))?;

        dst_subresource
            .validate(device)
            .map_err(|err| err.add_context("dst_subresource"))?;

        if src_subresource.aspects != dst_subresource.aspects {
            return Err(Box::new(ValidationError {
                problem: "`src_subresource.aspects` does not equal `dst_subresource.aspects`"
                    .into(),
                vuids: &["VUID-VkImageBlit2-aspectMask-00238"],
                ..Default::default()
            }));
        }

        if src_subresource.layer_count != dst_subresource.layer_count {
            return Err(Box::new(ValidationError {
                problem: "`src_subresource.layer_count` does not equal \
                    `dst_subresource.layer_count`"
                    .into(),
                vuids: &["VUID-VkImageBlit2-layerCount-00239"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk2(&self, src_image: &Image, dst_image: &Image) -> vk::ImageBlit2<'static> {
        let &Self {
            src_subresource,
            src_offsets,
            dst_subresource,
            dst_offsets,
            _ne: _,
        } = self;

        let remaining_array_layers = min(
            src_subresource
                .layer_count
                .unwrap_or(src_image.array_layers() - src_subresource.base_array_layer),
            dst_subresource
                .layer_count
                .unwrap_or(dst_image.array_layers() - dst_subresource.base_array_layer),
        );

        vk::ImageBlit2::default()
            .src_subresource(src_subresource.to_vk(remaining_array_layers))
            .src_offsets([
                vk::Offset3D {
                    x: src_offsets[0][0] as i32,
                    y: src_offsets[0][1] as i32,
                    z: src_offsets[0][2] as i32,
                },
                vk::Offset3D {
                    x: src_offsets[1][0] as i32,
                    y: src_offsets[1][1] as i32,
                    z: src_offsets[1][2] as i32,
                },
            ])
            .dst_subresource(dst_subresource.to_vk(remaining_array_layers))
            .dst_offsets([
                vk::Offset3D {
                    x: dst_offsets[0][0] as i32,
                    y: dst_offsets[0][1] as i32,
                    z: dst_offsets[0][2] as i32,
                },
                vk::Offset3D {
                    x: dst_offsets[1][0] as i32,
                    y: dst_offsets[1][1] as i32,
                    z: dst_offsets[1][2] as i32,
                },
            ])
    }

    pub(crate) fn to_vk(&self, src_image: &Image, dst_image: &Image) -> vk::ImageBlit {
        let &Self {
            src_subresource,
            src_offsets,
            dst_subresource,
            dst_offsets,
            _ne: _,
        } = self;

        let remaining_array_layers = min(
            src_subresource
                .layer_count
                .unwrap_or(src_image.array_layers() - src_subresource.base_array_layer),
            dst_subresource
                .layer_count
                .unwrap_or(dst_image.array_layers() - dst_subresource.base_array_layer),
        );

        vk::ImageBlit {
            src_subresource: src_subresource.to_vk(remaining_array_layers),
            src_offsets: [
                vk::Offset3D {
                    x: src_offsets[0][0] as i32,
                    y: src_offsets[0][1] as i32,
                    z: src_offsets[0][2] as i32,
                },
                vk::Offset3D {
                    x: src_offsets[1][0] as i32,
                    y: src_offsets[1][1] as i32,
                    z: src_offsets[1][2] as i32,
                },
            ],
            dst_subresource: dst_subresource.to_vk(remaining_array_layers),
            dst_offsets: [
                vk::Offset3D {
                    x: dst_offsets[0][0] as i32,
                    y: dst_offsets[0][1] as i32,
                    z: dst_offsets[0][2] as i32,
                },
                vk::Offset3D {
                    x: dst_offsets[1][0] as i32,
                    y: dst_offsets[1][1] as i32,
                    z: dst_offsets[1][2] as i32,
                },
            ],
        }
    }
}

/// Parameters to resolve image data.
#[derive(Clone, Debug)]
pub struct ResolveImageInfo<'a> {
    /// The multisampled image to resolve from.
    ///
    /// There is no default value.
    pub src_image: &'a Image,

    /// The layout used for `src_image` during the resolve operation.
    ///
    /// The following layouts are allowed:
    /// - [`ImageLayout::TransferSrcOptimal`]
    /// - [`ImageLayout::General`]
    ///
    /// The default value is [`ImageLayout::TransferSrcOptimal`].
    pub src_image_layout: ImageLayout,

    /// The non-multisampled image to resolve into.
    ///
    /// There is no default value.
    pub dst_image: &'a Image,

    /// The layout used for `dst_image` during the resolve operation.
    ///
    /// The following layouts are allowed:
    /// - [`ImageLayout::TransferDstOptimal`]
    /// - [`ImageLayout::General`]
    ///
    /// The default value is [`ImageLayout::TransferDstOptimal`].
    pub dst_image_layout: ImageLayout,

    /// The regions of both images to resolve between.
    ///
    /// The default value is a single region, covering the first mip level, and the smallest of the
    /// array layers and extent of the two images. All aspects of each image are selected, or
    /// `plane0` if the image is multi-planar.
    pub regions: &'a [ImageResolve<'a>],

    pub _ne: crate::NonExhaustive<'a>,
}

impl<'a> ResolveImageInfo<'a> {
    /// Returns a default `ResolveImageInfo` with the provided `src_image` and `dst_image`.
    #[inline]
    pub const fn new(src_image: &'a Image, dst_image: &'a Image) -> Self {
        Self {
            src_image,
            src_image_layout: ImageLayout::TransferSrcOptimal,
            dst_image,
            dst_image_layout: ImageLayout::TransferDstOptimal,
            regions: &[],
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            src_image,
            src_image_layout,
            dst_image,
            dst_image_layout,
            regions,
            _ne: _,
        } = self;

        src_image_layout.validate_device(device).map_err(|err| {
            err.add_context("src_image_layout")
                .set_vuids(&["VUID-VkResolveImageInfo2-srcImageLayout-parameter"])
        })?;

        dst_image_layout.validate_device(device).map_err(|err| {
            err.add_context("dst_image_layout")
                .set_vuids(&["VUID-VkResolveImageInfo2-dstImageLayout-parameter"])
        })?;

        // VUID-VkResolveImageInfo2-commonparent
        assert_eq!(device, src_image.device().as_ref());
        assert_eq!(device, dst_image.device().as_ref());

        let src_image_format = src_image.format();
        let dst_image_format = dst_image.format();

        if src_image.samples() == SampleCount::Sample1 {
            return Err(Box::new(ValidationError {
                context: "src_image.samples()".into(),
                problem: "is `SampleCount::Sample1`".into(),
                vuids: &["VUID-VkResolveImageInfo2-srcImage-00257"],
                ..Default::default()
            }));
        }

        if !src_image.usage().intersects(ImageUsage::TRANSFER_SRC) {
            return Err(Box::new(ValidationError {
                context: "src_image.usage()".into(),
                problem: "does not contain `ImageUsage::TRANSFER_SRC`".into(),
                vuids: &["VUID-VkResolveImageInfo2-srcImage-06762"],
                ..Default::default()
            }));
        }

        if !src_image
            .format_features()
            .intersects(FormatFeatures::TRANSFER_SRC)
        {
            return Err(Box::new(ValidationError {
                context: "src_image.format_features()".into(),
                problem: "does not contain `FormatFeatures::TRANSFER_SRC`".into(),
                vuids: &["VUID-VkResolveImageInfo2-srcImage-06763"],
                ..Default::default()
            }));
        }

        if !matches!(
            src_image_layout,
            ImageLayout::TransferSrcOptimal | ImageLayout::General
        ) {
            return Err(Box::new(ValidationError {
                context: "src_image_layout".into(),
                problem: "is not `ImageLayout::TransferSrcOptimal` or `ImageLayout::General`"
                    .into(),
                vuids: &["VUID-VkResolveImageInfo2-srcImageLayout-01400"],
                ..Default::default()
            }));
        }

        if dst_image.samples() != SampleCount::Sample1 {
            return Err(Box::new(ValidationError {
                context: "dst_image.samples()".into(),
                problem: "is not `SampleCount::Sample1`".into(),
                vuids: &["VUID-VkResolveImageInfo2-dstImage-00259"],
                ..Default::default()
            }));
        }

        if !dst_image.usage().intersects(ImageUsage::TRANSFER_DST) {
            return Err(Box::new(ValidationError {
                context: "dst_image.usage()".into(),
                problem: "does not contain `ImageUsage::TRANSFER_DST`".into(),
                vuids: &["VUID-VkResolveImageInfo2-dstImage-06764"],
                ..Default::default()
            }));
        }

        if !dst_image
            .format_features()
            .contains(FormatFeatures::TRANSFER_DST | FormatFeatures::COLOR_ATTACHMENT)
        {
            return Err(Box::new(ValidationError {
                context: "dst_image.format_features()".into(),
                problem: "does not contain both `FormatFeatures::TRANSFER_DST` and \
                    `FormatFeatures::COLOR_ATTACHMENT`"
                    .into(),
                vuids: &[
                    "VUID-VkResolveImageInfo2-dstImage-06765",
                    "VUID-VkResolveImageInfo2-dstImage-02003",
                ],
                ..Default::default()
            }));
        }

        if device.enabled_features().linear_color_attachment
            && dst_image.tiling() == ImageTiling::Linear
            && !dst_image
                .format_features()
                .contains(FormatFeatures::LINEAR_COLOR_ATTACHMENT)
        {
            return Err(Box::new(ValidationError {
                problem: "the `linear_color_attachment` feature is enabled on the device, and \
                    `dst_image.tiling()` is `ImageTiling::Linear`, but \
                    `dst_image.format_features()` does not contain \
                    `FormatFeatures::LINEAR_COLOR_ATTACHMENT`"
                    .into(),
                vuids: &["VUID-VkResolveImageInfo2-linearColorAttachment-06519"],
                ..Default::default()
            }));
        }

        if !matches!(
            dst_image_layout,
            ImageLayout::TransferDstOptimal | ImageLayout::General
        ) {
            return Err(Box::new(ValidationError {
                context: "dst_image_layout".into(),
                problem: "is not `ImageLayout::TransferDstOptimal` or `ImageLayout::General`"
                    .into(),
                vuids: &["VUID-VkResolveImageInfo2-dstImageLayout-01401"],
                ..Default::default()
            }));
        }

        if src_image_format != dst_image_format {
            return Err(Box::new(ValidationError {
                problem: "`src_image.format()` does not equal `dst_image.format()`".into(),
                vuids: &["VUID-VkResolveImageInfo2-srcImage-01386"],
                ..Default::default()
            }));
        }

        for (region_index, region) in regions.iter().enumerate() {
            region
                .validate(device)
                .map_err(|err| err.add_context(format!("regions[{}]", region_index)))?;

            let &ImageResolve {
                src_subresource,
                src_offset,
                dst_subresource,
                dst_offset,
                extent,
                _ne: _,
            } = region;

            /*
               Check src
            */

            if src_subresource.mip_level >= src_image.mip_levels() {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].src_subresource.mip_level` is not less than \
                        `src_image.mip_levels()`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkResolveImageInfo2-srcSubresource-01709"],
                    ..Default::default()
                }));
            }

            let src_subresource_extent =
                mip_level_extent(src_image.extent(), src_subresource.mip_level).unwrap();

            match src_image.image_type() {
                ImageType::Dim1d => {
                    if src_offset[1] != 0 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].src_offset[1]` is not 0",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkResolveImageInfo2-srcImage-00271"],
                            ..Default::default()
                        }));
                    }

                    if extent[1] != 1 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].extent[1]` is not 1",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkResolveImageInfo2-srcImage-00271"],
                            ..Default::default()
                        }));
                    }

                    if src_offset[2] != 0 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].src_offset[2]` is not 0",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkResolveImageInfo2-srcImage-00273"],
                            ..Default::default()
                        }));
                    }

                    if extent[2] != 1 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].extent[2]` is not 1",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkResolveImageInfo2-srcImage-00273"],
                            ..Default::default()
                        }));
                    }
                }
                ImageType::Dim2d => {
                    if src_offset[2] != 0 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.image_type()` is `ImageType::Dim2d`, but \
                                `regions[{}].src_offset[2]` is not 0",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkResolveImageInfo2-srcImage-00273"],
                            ..Default::default()
                        }));
                    }

                    if extent[2] != 1 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.image_type()` is `ImageType::Dim2d`, but \
                                `regions[{}].extent[2]` is not 1",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkResolveImageInfo2-srcImage-00273"],
                            ..Default::default()
                        }));
                    }
                }
                ImageType::Dim3d => {
                    if !(src_subresource.base_array_layer == 0
                        && src_subresource.layer_count == Some(1))
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`src_image.image_type()` is `ImageType::Dim3d`, but \
                                `regions[{0}].src_subresource.base_array_layer` and \
                                `regions[{0}].src_subresource.layer_count` are not `0` and \
                                `Some(1)`, respectively",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkResolveImageInfo2-srcImage-04446"],
                            ..Default::default()
                        }));
                    }
                }
            }

            if src_subresource.base_array_layer >= src_image.array_layers() {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].src_subresource.base_array_layer` is not less than \
                        `src_image.array_layers()`",
                        region_index,
                    )
                    .into(),
                    // vuids?
                    ..Default::default()
                }));
            }

            if let Some(src_subresource_layer_count) = src_subresource.layer_count {
                if src_subresource_layer_count
                    > src_image.array_layers() - src_subresource.base_array_layer
                {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`regions[{0}].src_subresource.base_array_layer + \
                            regions[{0}].src_subresource.layer_count` is greater than \
                            `src_image.array_layers()`",
                            region_index,
                        )
                        .into(),
                        vuids: &["VUID-VkResolveImageInfo2-srcSubresource-01711"],
                        ..Default::default()
                    }));
                }
            }

            if src_offset[0]
                .checked_add(extent[0])
                .is_none_or(|end| end > src_subresource_extent[0])
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].src_offset[0] + regions[{0}].extent[0]` is greater \
                        than coordinate 0 of the extent of the subresource of `src_image` \
                        selected by `regions[{0}].src_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkResolveImageInfo2-srcOffset-00269"],
                    ..Default::default()
                }));
            }

            if src_offset[1]
                .checked_add(extent[1])
                .is_none_or(|end| end > src_subresource_extent[1])
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].src_offset[1] + regions[{0}].extent[1]` is greater \
                        than coordinate 1 of the extent of the subresource of `src_image` \
                        selected by `regions[{0}].src_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkResolveImageInfo2-srcOffset-00270"],
                    ..Default::default()
                }));
            }

            if src_offset[2]
                .checked_add(extent[2])
                .is_none_or(|end| end > src_subresource_extent[2])
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].src_offset[2] + regions[{0}].extent[2]` is greater \
                        than coordinate 2 of the extent of the subresource of `src_image` \
                        selected by `regions[{0}].src_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkResolveImageInfo2-srcOffset-00272"],
                    ..Default::default()
                }));
            }

            /*
               Check dst
            */

            if dst_subresource.mip_level >= dst_image.mip_levels() {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].dst_subresource.mip_level` is not less than \
                        `dst_image.mip_levels()`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkResolveImageInfo2-dstSubresource-01710"],
                    ..Default::default()
                }));
            }

            let dst_subresource_extent =
                mip_level_extent(dst_image.extent(), dst_subresource.mip_level).unwrap();

            match dst_image.image_type() {
                ImageType::Dim1d => {
                    if dst_offset[1] != 0 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`dst_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].dst_offset[1]` is not 0",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkResolveImageInfo2-dstImage-00276"],
                            ..Default::default()
                        }));
                    }

                    if extent[1] != 1 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`dst_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].extent[1]` is not 1",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkResolveImageInfo2-dstImage-00276"],
                            ..Default::default()
                        }));
                    }

                    if dst_offset[2] != 0 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`dst_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].dst_offset[2]` is not 0",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkResolveImageInfo2-dstImage-00278"],
                            ..Default::default()
                        }));
                    }

                    if extent[2] != 1 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`dst_image.image_type()` is `ImageType::Dim1d`, but \
                                `regions[{}].extent[2]` is not 1",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkResolveImageInfo2-dstImage-00278"],
                            ..Default::default()
                        }));
                    }
                }
                ImageType::Dim2d => {
                    if dst_offset[2] != 0 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`dst_image.image_type()` is `ImageType::Dim2d`, but \
                                `regions[{}].dst_offset[2]` is not 0",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkResolveImageInfo2-dstImage-00278"],
                            ..Default::default()
                        }));
                    }

                    if extent[2] != 1 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`dst_image.image_type()` is `ImageType::Dim2d`, but \
                                `regions[{}].extent[2]` is not 1",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkResolveImageInfo2-dstImage-00278"],
                            ..Default::default()
                        }));
                    }
                }
                ImageType::Dim3d => {
                    if !(dst_subresource.base_array_layer == 0
                        && dst_subresource.layer_count == Some(1))
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`dst_image.image_type()` is `ImageType::Dim3d`, but \
                                `regions[{0}].dst_subresource.base_array_layer` and \
                                `regions[{0}].dst_subresource.layer_count` are not `0` and \
                                `Some(1)`, respectively",
                                region_index,
                            )
                            .into(),
                            vuids: &["VUID-VkResolveImageInfo2-srcImage-04447"],
                            ..Default::default()
                        }));
                    }
                }
            }

            if dst_subresource.base_array_layer >= dst_image.array_layers() {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].dst_subresource.base_array_layer` is not less than \
                        `dst_image.array_layers()`",
                        region_index,
                    )
                    .into(),
                    // vuids?
                    ..Default::default()
                }));
            }

            if let Some(dst_subresource_layer_count) = dst_subresource.layer_count {
                if dst_subresource_layer_count
                    > dst_image.array_layers() - dst_subresource.base_array_layer
                {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`regions[{0}].dst_subresource.base_array_layer + \
                            regions[{0}].dst_subresource.layer_count` is greater than \
                            `dst_image.array_layers()`",
                            region_index,
                        )
                        .into(),
                        vuids: &["VUID-VkResolveImageInfo2-dstSubresource-01712"],
                        ..Default::default()
                    }));
                }
            }

            if dst_offset[0]
                .checked_add(extent[0])
                .is_none_or(|end| end > dst_subresource_extent[0])
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].dst_offset[0] + regions[{0}].extent[0]` is greater \
                        than coordinate 0 of the extent of the subresource of `dst_image` \
                        selected by `regions[{0}].dst_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkResolveImageInfo2-dstOffset-00274"],
                    ..Default::default()
                }));
            }

            if dst_offset[1]
                .checked_add(extent[1])
                .is_none_or(|end| end > dst_subresource_extent[1])
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].dst_offset[1] + regions[{0}].extent[1]` is greater \
                        than coordinate 1 of the extent of the subresource of `dst_image` \
                        selected by `regions[{0}].dst_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkResolveImageInfo2-dstOffset-00275"],
                    ..Default::default()
                }));
            }

            if dst_offset[2]
                .checked_add(extent[2])
                .is_none_or(|end| end > dst_subresource_extent[2])
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{0}].dst_offset[2] + regions[{0}].extent[2]` is greater \
                        than coordinate 2 of the extent of the subresource of `dst_image` \
                        selected by `regions[{0}].dst_subresource`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-VkResolveImageInfo2-dstOffset-00277"],
                    ..Default::default()
                }));
            }
        }

        // VUID-VkResolveImageInfo2-pRegions-00255
        // Can't occur as long as memory aliasing isn't allowed, because `src_image` and
        // `dst_image` must have different sample counts and therefore can never be the same image.

        Ok(())
    }

    pub(crate) fn to_vk2(
        &self,
        regions_vk: &'a [vk::ImageResolve2<'static>],
    ) -> vk::ResolveImageInfo2<'a> {
        let &Self {
            src_image,
            src_image_layout,
            dst_image,
            dst_image_layout,
            regions: _,
            _ne: _,
        } = self;

        vk::ResolveImageInfo2::default()
            .src_image(src_image.handle())
            .src_image_layout(src_image_layout.into())
            .dst_image(dst_image.handle())
            .dst_image_layout(dst_image_layout.into())
            .regions(regions_vk)
    }

    pub(crate) fn to_vk2_regions(&self) -> SmallVec<[vk::ImageResolve2<'static>; 8]> {
        let &Self {
            src_image,
            dst_image,
            regions,
            ..
        } = self;

        if regions.is_empty() {
            let remaining_array_layers = min(src_image.array_layers(), dst_image.array_layers());
            let src_extent = src_image.extent();
            let dst_extent = dst_image.extent();
            let region_vk = vk::ImageResolve2::default()
                .src_subresource(src_image.subresource_layers().to_vk(remaining_array_layers))
                .src_offset(convert_offset([0; 3]))
                .dst_subresource(dst_image.subresource_layers().to_vk(remaining_array_layers))
                .dst_offset(convert_offset([0; 3]))
                .extent(convert_extent([
                    min(src_extent[0], dst_extent[0]),
                    min(src_extent[1], dst_extent[1]),
                    min(src_extent[2], dst_extent[2]),
                ]));

            smallvec![region_vk]
        } else {
            regions
                .iter()
                .map(|region| region.to_vk2(src_image, dst_image))
                .collect()
        }
    }

    pub(crate) fn to_vk(&self) -> ResolveImageInfoVk {
        let &Self {
            src_image,
            src_image_layout,
            dst_image,
            dst_image_layout,
            regions: _,
            _ne: _,
        } = self;

        ResolveImageInfoVk {
            src_image_vk: src_image.handle(),
            src_image_layout_vk: src_image_layout.into(),
            dst_image_vk: dst_image.handle(),
            dst_image_layout_vk: dst_image_layout.into(),
        }
    }

    pub(crate) fn to_vk_regions(&self) -> SmallVec<[vk::ImageResolve; 8]> {
        let &Self {
            src_image,
            dst_image,
            regions,
            ..
        } = self;

        if regions.is_empty() {
            let remaining_array_layers = min(src_image.array_layers(), dst_image.array_layers());
            let src_extent = src_image.extent();
            let dst_extent = dst_image.extent();
            let region_vk = vk::ImageResolve {
                src_subresource: src_image.subresource_layers().to_vk(remaining_array_layers),
                src_offset: convert_offset([0; 3]),
                dst_subresource: dst_image.subresource_layers().to_vk(remaining_array_layers),
                dst_offset: convert_offset([0; 3]),
                extent: convert_extent([
                    min(src_extent[0], dst_extent[0]),
                    min(src_extent[1], dst_extent[1]),
                    min(src_extent[2], dst_extent[2]),
                ]),
            };

            smallvec![region_vk]
        } else {
            regions
                .iter()
                .map(|region| region.to_vk(src_image, dst_image))
                .collect()
        }
    }
}

pub(crate) struct ResolveImageInfoVk {
    pub(crate) src_image_vk: vk::Image,
    pub(crate) src_image_layout_vk: vk::ImageLayout,
    pub(crate) dst_image_vk: vk::Image,
    pub(crate) dst_image_layout_vk: vk::ImageLayout,
}

/// A region of data to resolve between images.
#[derive(Clone, Debug)]
pub struct ImageResolve<'a> {
    /// The subresource of `src_image` to resolve from.
    ///
    /// The default value is [`ImageSubresourceLayers::default()`].
    pub src_subresource: ImageSubresourceLayers,

    /// The offset from the zero coordinate of `src_image` that resolving will start from.
    ///
    /// The default value is `[0; 3]`.
    pub src_offset: [u32; 3],

    /// The subresource of `dst_image` to resolve into.
    ///
    /// The default value is [`ImageSubresourceLayers::default()`].
    pub dst_subresource: ImageSubresourceLayers,

    /// The offset from the zero coordinate of `dst_image` that resolving will start from.
    ///
    /// The default value is `[0; 3]`.
    pub dst_offset: [u32; 3],

    /// The extent of texels to resolve.
    ///
    /// The default value is `[0; 3]`, which must be overridden.
    pub extent: [u32; 3],

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for ImageResolve<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl ImageResolve<'_> {
    /// Returns a default `ImageResolve`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            src_subresource: ImageSubresourceLayers::new(),
            src_offset: [0; 3],
            dst_subresource: ImageSubresourceLayers::new(),
            dst_offset: [0; 3],
            extent: [0; 3],
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            src_subresource,
            src_offset: _,
            dst_subresource,
            dst_offset: _,
            extent: _,
            _ne: _,
        } = self;

        src_subresource
            .validate(device)
            .map_err(|err| err.add_context("src_subresource"))?;

        dst_subresource
            .validate(device)
            .map_err(|err| err.add_context("dst_subresource"))?;

        if src_subresource.aspects != ImageAspects::COLOR {
            return Err(Box::new(ValidationError {
                problem: "`src_subresource.aspects` is not `ImageAspects::COLOR`".into(),
                vuids: &["VUID-VkImageResolve2-aspectMask-00266"],
                ..Default::default()
            }));
        }

        if dst_subresource.aspects != ImageAspects::COLOR {
            return Err(Box::new(ValidationError {
                problem: "`dst_subresource.aspects` is not `ImageAspects::COLOR`".into(),
                vuids: &["VUID-VkImageResolve2-aspectMask-00266"],
                ..Default::default()
            }));
        }

        if src_subresource.layer_count != dst_subresource.layer_count {
            return Err(Box::new(ValidationError {
                problem: "`src_subresource.layer_count` does not equal \
                    `dst_subresource.layer_count`"
                    .into(),
                vuids: &["VUID-VkImageResolve2-layerCount-00267"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk2(
        &self,
        src_image: &Image,
        dst_image: &Image,
    ) -> vk::ImageResolve2<'static> {
        let &Self {
            src_subresource,
            src_offset,
            dst_subresource,
            dst_offset,
            extent,
            _ne: _,
        } = self;

        let remaining_array_layers = min(
            src_subresource
                .layer_count
                .unwrap_or(src_image.array_layers() - src_subresource.base_array_layer),
            dst_subresource
                .layer_count
                .unwrap_or(dst_image.array_layers() - dst_subresource.base_array_layer),
        );

        vk::ImageResolve2::default()
            .src_subresource(src_subresource.to_vk(remaining_array_layers))
            .src_offset(vk::Offset3D {
                x: src_offset[0] as i32,
                y: src_offset[1] as i32,
                z: src_offset[2] as i32,
            })
            .dst_subresource(dst_subresource.to_vk(remaining_array_layers))
            .dst_offset(vk::Offset3D {
                x: dst_offset[0] as i32,
                y: dst_offset[1] as i32,
                z: dst_offset[2] as i32,
            })
            .extent(vk::Extent3D {
                width: extent[0],
                height: extent[1],
                depth: extent[2],
            })
    }

    pub(crate) fn to_vk(&self, src_image: &Image, dst_image: &Image) -> vk::ImageResolve {
        let &Self {
            src_subresource,
            src_offset,
            dst_subresource,
            dst_offset,
            extent,
            _ne: _,
        } = self;

        let remaining_array_layers = min(
            src_subresource
                .layer_count
                .unwrap_or(src_image.array_layers() - src_subresource.base_array_layer),
            dst_subresource
                .layer_count
                .unwrap_or(dst_image.array_layers() - dst_subresource.base_array_layer),
        );

        vk::ImageResolve {
            src_subresource: src_subresource.to_vk(remaining_array_layers),
            src_offset: vk::Offset3D {
                x: src_offset[0] as i32,
                y: src_offset[1] as i32,
                z: src_offset[2] as i32,
            },
            dst_subresource: dst_subresource.to_vk(remaining_array_layers),
            dst_offset: vk::Offset3D {
                x: dst_offset[0] as i32,
                y: dst_offset[1] as i32,
                z: dst_offset[2] as i32,
            },
            extent: vk::Extent3D {
                width: extent[0],
                height: extent[1],
                depth: extent[2],
            },
        }
    }
}

fn convert_offset(offset: [u32; 3]) -> vk::Offset3D {
    vk::Offset3D {
        x: offset[0] as i32,
        y: offset[1] as i32,
        z: offset[2] as i32,
    }
}

fn convert_extent(extent: [u32; 3]) -> vk::Extent3D {
    vk::Extent3D {
        width: extent[0],
        height: extent[1],
        depth: extent[2],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::Format;

    /// Computes the minimum required len in elements for buffer with image data in specified
    /// format of specified size.
    fn required_size_for_format(format: Format, extent: [u32; 3], layer_count: u32) -> DeviceSize {
        let num_blocks = extent
            .into_iter()
            .zip(format.block_extent())
            .map(|(extent, block_extent)| {
                let extent = extent as DeviceSize;
                let block_extent = block_extent as DeviceSize;
                extent.div_ceil(block_extent)
            })
            .product::<DeviceSize>()
            * layer_count as DeviceSize;

        num_blocks * format.block_size()
    }

    #[test]
    fn test_required_len_for_format() {
        // issue #1292
        assert_eq!(
            required_size_for_format(Format::BC1_RGB_UNORM_BLOCK, [2048, 2048, 1], 1),
            2097152
        );
        // other test cases
        assert_eq!(
            required_size_for_format(Format::R8G8B8A8_UNORM, [2048, 2048, 1], 1),
            16777216
        );
        assert_eq!(
            required_size_for_format(Format::R4G4_UNORM_PACK8, [512, 512, 1], 1),
            262144
        );
        assert_eq!(
            required_size_for_format(Format::R8G8B8_USCALED, [512, 512, 1], 1),
            786432
        );
        assert_eq!(
            required_size_for_format(Format::R32G32_UINT, [512, 512, 1], 1),
            2097152
        );
        assert_eq!(
            required_size_for_format(Format::R32G32_UINT, [512, 512, 1], 1),
            2097152
        );
        assert_eq!(
            required_size_for_format(Format::ASTC_8x8_UNORM_BLOCK, [512, 512, 1], 1),
            65536
        );
        assert_eq!(
            required_size_for_format(Format::ASTC_12x12_SRGB_BLOCK, [512, 512, 1], 1),
            29584
        );
    }
}
