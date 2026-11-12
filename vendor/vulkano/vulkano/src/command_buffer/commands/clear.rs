use crate::{
    buffer::{Buffer, BufferContents, BufferUsage},
    command_buffer::sys::RecordingCommandBuffer,
    device::{Device, DeviceOwned, QueueFlags},
    format::{ClearColorValue, ClearDepthStencilValue, FormatFeatures},
    image::{Image, ImageAspects, ImageLayout, ImageSubresourceRange, ImageUsage},
    DeviceSize, Requires, RequiresAllOf, RequiresOneOf, ValidationError, Version, VulkanObject,
};
use ash::vk;
use smallvec::{smallvec, SmallVec};
use std::ffi::c_void;

impl RecordingCommandBuffer {
    #[inline]
    #[track_caller]
    pub unsafe fn clear_color_image(&mut self, clear_info: &ClearColorImageInfo<'_>) -> &mut Self {
        unsafe { self.try_clear_color_image(clear_info) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_clear_color_image(
        &mut self,
        clear_info: &ClearColorImageInfo<'_>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_clear_color_image(clear_info)?;

        Ok(unsafe { self.clear_color_image_unchecked(clear_info) })
    }

    pub(crate) fn validate_clear_color_image(
        &self,
        clear_info: &ClearColorImageInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics or compute operations"
                    .into(),
                vuids: &["VUID-vkCmdClearColorImage-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        clear_info
            .validate(self.device())
            .map_err(|err| err.add_context("clear_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn clear_color_image_unchecked(
        &mut self,
        clear_info: &ClearColorImageInfo<'_>,
    ) -> &mut Self {
        let clear_info_vk = clear_info.to_vk();
        let ranges_vk = clear_info.to_vk_ranges();

        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_clear_color_image)(
                self.handle(),
                clear_info_vk.image,
                clear_info_vk.image_layout,
                &clear_info_vk.color,
                ranges_vk.len() as u32,
                ranges_vk.as_ptr(),
            )
        };

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn clear_depth_stencil_image(
        &mut self,
        clear_info: &ClearDepthStencilImageInfo<'_>,
    ) -> &mut Self {
        unsafe { self.try_clear_depth_stencil_image(clear_info) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_clear_depth_stencil_image(
        &mut self,
        clear_info: &ClearDepthStencilImageInfo<'_>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_clear_depth_stencil_image(clear_info)?;

        Ok(unsafe { self.clear_depth_stencil_image_unchecked(clear_info) })
    }

    pub(crate) fn validate_clear_depth_stencil_image(
        &self,
        clear_info: &ClearDepthStencilImageInfo<'_>,
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
                vuids: &["VUID-vkCmdClearDepthStencilImage-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        clear_info
            .validate(self.device())
            .map_err(|err| err.add_context("clear_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn clear_depth_stencil_image_unchecked(
        &mut self,
        clear_info: &ClearDepthStencilImageInfo<'_>,
    ) -> &mut Self {
        let clear_info_vk = clear_info.to_vk();
        let ranges_vk = clear_info.to_vk_ranges();

        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_clear_depth_stencil_image)(
                self.handle(),
                clear_info_vk.image,
                clear_info_vk.image_layout,
                &clear_info_vk.depth_stencil,
                ranges_vk.len() as u32,
                ranges_vk.as_ptr(),
            )
        };

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn fill_buffer(&mut self, fill_info: &FillBufferInfo<'_>) -> &mut Self {
        unsafe { self.try_fill_buffer(fill_info) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_fill_buffer(
        &mut self,
        fill_info: &FillBufferInfo<'_>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_fill_buffer(fill_info)?;

        Ok(unsafe { self.fill_buffer_unchecked(fill_info) })
    }

    pub(crate) fn validate_fill_buffer(
        &self,
        fill_info: &FillBufferInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        let device = self.device();
        let queue_family_properties = self.queue_family_properties();

        if device.api_version() >= Version::V1_1 || device.enabled_extensions().khr_maintenance1 {
            if !queue_family_properties
                .queue_flags
                .intersects(QueueFlags::TRANSFER | QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
            {
                return Err(Box::new(ValidationError {
                    problem: "the queue family of the command buffer does not support \
                        transfer, graphics or compute operations"
                        .into(),
                    vuids: &["VUID-vkCmdFillBuffer-commandBuffer-cmdpool"],
                    ..Default::default()
                }));
            }
        } else {
            if !queue_family_properties
                .queue_flags
                .intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
            {
                return Err(Box::new(ValidationError {
                    problem: "the queue family of the command buffer does not support \
                        graphics or compute operations"
                        .into(),
                    vuids: &["VUID-vkCmdFillBuffer-commandBuffer-00030"],
                    ..Default::default()
                }));
            }
        }

        fill_info
            .validate(self.device())
            .map_err(|err| err.add_context("fill_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn fill_buffer_unchecked(&mut self, fill_info: &FillBufferInfo<'_>) -> &mut Self {
        let fill_info_vk = fill_info.to_vk();

        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_fill_buffer)(
                self.handle(),
                fill_info_vk.dst_buffer,
                fill_info_vk.dst_offset,
                fill_info_vk.size,
                fill_info_vk.data,
            )
        };

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn update_buffer(
        &mut self,
        dst_buffer: &Buffer,
        dst_offset: DeviceSize,
        data: &(impl BufferContents + ?Sized),
    ) -> &mut Self {
        unsafe { self.try_update_buffer(dst_buffer, dst_offset, data) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_update_buffer(
        &mut self,
        dst_buffer: &Buffer,
        dst_offset: DeviceSize,
        data: &(impl BufferContents + ?Sized),
    ) -> Result<&mut Self, Box<ValidationError>> {
        if size_of_val(data) == 0 {
            return Ok(self);
        }

        self.validate_update_buffer(dst_buffer, dst_offset, size_of_val(data) as DeviceSize)?;

        Ok(unsafe { self.update_buffer_unchecked(dst_buffer, dst_offset, data) })
    }

    pub(crate) fn validate_update_buffer(
        &self,
        dst_buffer: &Buffer,
        dst_offset: DeviceSize,
        data_size: DeviceSize,
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
                vuids: &["VUID-vkCmdUpdateBuffer-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        let device = self.device();

        // VUID-vkCmdUpdateBuffer-commonparent
        assert_eq!(device, dst_buffer.device());

        // VUID-vkCmdUpdateBuffer-dataSize-arraylength
        // Ensured because we return when the size is 0.

        if dst_offset >= dst_buffer.size() {
            return Err(Box::new(ValidationError {
                context: "dst_offset".into(),
                problem: "is not less than `dst_buffer.size()`".into(),
                vuids: &["VUID-vkCmdUpdateBuffer-dataSize-00032"],
                ..Default::default()
            }));
        }

        if data_size > dst_buffer.size() - dst_offset {
            return Err(Box::new(ValidationError {
                problem: "the size of `data` is greater than `dst_buffer.size() - dst_offset`"
                    .into(),
                vuids: &["VUID-vkCmdUpdateBuffer-dataSize-00033"],
                ..Default::default()
            }));
        }

        if !dst_buffer.usage().intersects(BufferUsage::TRANSFER_DST) {
            return Err(Box::new(ValidationError {
                context: "dst_buffer.usage()".into(),
                problem: "does not contain `BufferUsage::TRANSFER_DST`".into(),
                vuids: &["VUID-vkCmdUpdateBuffer-dstBuffer-00034"],
                ..Default::default()
            }));
        }

        if !dst_offset.is_multiple_of(4) {
            return Err(Box::new(ValidationError {
                context: "dst_buffer.offset()".into(),
                problem: "is not a multiple of 4".into(),
                vuids: &["VUID-vkCmdUpdateBuffer-dstOffset-00036"],
                ..Default::default()
            }));
        }

        if data_size > 65536 {
            return Err(Box::new(ValidationError {
                context: "data".into(),
                problem: "the size is greater than 65536 bytes".into(),
                vuids: &["VUID-vkCmdUpdateBuffer-dataSize-00037"],
                ..Default::default()
            }));
        }

        if !data_size.is_multiple_of(4) {
            return Err(Box::new(ValidationError {
                context: "data".into(),
                problem: "the size is not a multiple of 4".into(),
                vuids: &["VUID-vkCmdUpdateBuffer-dataSize-00038"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn update_buffer_unchecked(
        &mut self,
        dst_buffer: &Buffer,
        dst_offset: DeviceSize,
        data: &(impl BufferContents + ?Sized),
    ) -> &mut Self {
        unsafe {
            self.update_buffer_unchecked_inner(
                dst_buffer,
                dst_offset,
                <*const _>::cast(data),
                size_of_val(data) as DeviceSize,
            )
        }
    }

    unsafe fn update_buffer_unchecked_inner(
        &mut self,
        dst_buffer: &Buffer,
        dst_offset: DeviceSize,
        data: *const c_void,
        data_size: DeviceSize,
    ) -> &mut Self {
        if data_size == 0 {
            return self;
        }

        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_update_buffer)(
                self.handle(),
                dst_buffer.handle(),
                dst_offset,
                data_size,
                data,
            )
        };

        self
    }
}

/// Parameters to clear a color image.
#[derive(Clone, Debug)]
pub struct ClearColorImageInfo<'a> {
    /// The image to clear.
    ///
    /// There is no default value.
    pub image: &'a Image,

    /// The layout used for `image` during the clear operation.
    ///
    /// The following layouts are allowed:
    /// - [`ImageLayout::TransferDstOptimal`]
    /// - [`ImageLayout::General`]
    ///
    /// The default value is [`ImageLayout::TransferDstOptimal`].
    pub image_layout: ImageLayout,

    /// The color value to clear the image to.
    ///
    /// The default value is `ClearColorValue::Float([0.0; 4])`.
    pub clear_value: ClearColorValue,

    /// The subresource ranges of `image` to clear.
    ///
    /// The default value is a single region, covering the whole image.
    pub regions: &'a [ImageSubresourceRange],

    pub _ne: crate::NonExhaustive<'a>,
}

impl<'a> ClearColorImageInfo<'a> {
    /// Returns a default `ClearColorImageInfo` with the provided `image`.
    #[inline]
    pub const fn new(image: &'a Image) -> Self {
        Self {
            image,
            image_layout: ImageLayout::TransferDstOptimal,
            clear_value: ClearColorValue::Float([0.0; 4]),
            regions: &[],
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            image,
            image_layout,
            clear_value: _,
            regions,
            _ne: _,
        } = self;

        image_layout.validate_device(device).map_err(|err| {
            err.add_context("image_layout")
                .set_vuids(&["VUID-vkCmdClearColorImage-imageLayout-parameter"])
        })?;

        // VUID-vkCmdClearColorImage-commonparent
        assert_eq!(device, image.device().as_ref());

        if !image.usage().intersects(ImageUsage::TRANSFER_DST) {
            return Err(Box::new(ValidationError {
                context: "image.usage()".into(),
                problem: "does not contain `ImageUsage::TRANSFER_DST`".into(),
                vuids: &["VUID-vkCmdClearColorImage-image-00002"],
                ..Default::default()
            }));
        }

        if device.api_version() >= Version::V1_1 || device.enabled_extensions().khr_maintenance1 {
            if !image
                .format_features()
                .intersects(FormatFeatures::TRANSFER_DST)
            {
                return Err(Box::new(ValidationError {
                    context: "image.format_features()".into(),
                    problem: "does not contain `FormatFeatures::TRANSFER_DST`".into(),
                    vuids: &["VUID-vkCmdClearColorImage-image-01993"],
                    ..Default::default()
                }));
            }
        }

        let image_aspects = image.format().aspects();

        if image_aspects.intersects(ImageAspects::DEPTH | ImageAspects::STENCIL) {
            return Err(Box::new(ValidationError {
                context: "image.format()".into(),
                problem: "is a depth/stencil format".into(),
                vuids: &["VUID-vkCmdClearColorImage-image-00007"],
                ..Default::default()
            }));
        }

        if image.format().compression().is_some() {
            return Err(Box::new(ValidationError {
                context: "image.format()".into(),
                problem: "is a compressed format".into(),
                vuids: &["VUID-vkCmdClearColorImage-image-00007"],
                ..Default::default()
            }));
        }

        if image.format().ycbcr_chroma_sampling().is_some() {
            return Err(Box::new(ValidationError {
                context: "image.format()".into(),
                problem: "is a YCbCr format".into(),
                vuids: &["VUID-vkCmdClearColorImage-image-01545"],
                ..Default::default()
            }));
        }

        if !matches!(
            image_layout,
            ImageLayout::TransferDstOptimal | ImageLayout::General
        ) {
            return Err(Box::new(ValidationError {
                context: "image_layout".into(),
                problem: "is not `ImageLayout::TransferDstOptimal` or `ImageLayout::General`"
                    .into(),
                vuids: &["VUID-vkCmdClearColorImage-imageLayout-01394"],
                ..Default::default()
            }));
        }

        for (region_index, subresource_range) in regions.iter().enumerate() {
            subresource_range
                .validate(device)
                .map_err(|err| err.add_context(format!("regions[{}]", region_index)))?;

            if !image_aspects.contains(subresource_range.aspects) {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].aspects` is not a subset of `image.format().aspects()`",
                        region_index
                    )
                    .into(),
                    vuids: &["VUID-vkCmdClearColorImage-aspectMask-02498"],
                    ..Default::default()
                }));
            }

            if subresource_range.base_mip_level >= image.mip_levels() {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].base_mip_level` is not less than `image.mip_levels()`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-vkCmdClearColorImage-baseMipLevel-01470"],
                    ..Default::default()
                }));
            }

            if let Some(subresource_range_level_count) = subresource_range.level_count {
                if subresource_range_level_count
                    > image.mip_levels() - subresource_range.base_mip_level
                {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`regions[{0}].base_mip_level + regions[{0}].level_count` is greater \
                            than `image.mip_levels()`",
                            region_index,
                        )
                        .into(),
                        vuids: &["VUID-vkCmdClearColorImage-pRanges-01692"],
                        ..Default::default()
                    }));
                }
            }

            if subresource_range.base_array_layer >= image.mip_levels() {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].base_array_level` is not less than `image.array_layers()`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-vkCmdClearColorImage-baseArrayLayer-01472"],
                    ..Default::default()
                }));
            }

            if let Some(subresource_range_layer_count) = subresource_range.layer_count {
                if subresource_range_layer_count
                    > image.array_layers() - subresource_range.base_array_layer
                {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`regions[{0}].base_array_level + regions[{0}].layer_count` is \
                            greater than `image.array_layers()`",
                            region_index,
                        )
                        .into(),
                        vuids: &["VUID-vkCmdClearColorImage-pRanges-01693"],
                        ..Default::default()
                    }));
                }
            }
        }

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> ClearColorImageInfoVk {
        let &Self {
            image,
            image_layout,
            clear_value,
            regions: _,
            _ne: _,
        } = self;

        ClearColorImageInfoVk {
            image: image.handle(),
            image_layout: image_layout.into(),
            color: clear_value.to_vk(),
        }
    }

    pub(crate) fn to_vk_ranges(&self) -> SmallVec<[vk::ImageSubresourceRange; 8]> {
        let &Self { image, regions, .. } = self;

        if regions.is_empty() {
            let region_vk = image.subresource_range().to_vk();

            smallvec![region_vk]
        } else {
            regions.iter().map(ImageSubresourceRange::to_vk).collect()
        }
    }
}

pub(crate) struct ClearColorImageInfoVk {
    pub(crate) image: vk::Image,
    pub(crate) image_layout: vk::ImageLayout,
    pub(crate) color: vk::ClearColorValue,
}

/// Parameters to clear a depth/stencil image.
#[derive(Clone, Debug)]
pub struct ClearDepthStencilImageInfo<'a> {
    /// The image to clear.
    ///
    /// There is no default value.
    pub image: &'a Image,

    /// The layout used for `image` during the clear operation.
    ///
    /// The following layouts are allowed:
    /// - [`ImageLayout::TransferDstOptimal`]
    /// - [`ImageLayout::General`]
    ///
    /// The default value is [`ImageLayout::TransferDstOptimal`].
    pub image_layout: ImageLayout,

    /// The depth/stencil values to clear the image to.
    ///
    /// The default value is zero for both.
    pub clear_value: ClearDepthStencilValue,

    /// The subresource ranges of `image` to clear.
    ///
    /// The default value is a single region, covering the whole image.
    pub regions: &'a [ImageSubresourceRange],

    pub _ne: crate::NonExhaustive<'a>,
}

impl<'a> ClearDepthStencilImageInfo<'a> {
    /// Returns a default `ClearDepthStencilImageInfo` with the provided `image`.
    #[inline]
    pub const fn new(image: &'a Image) -> Self {
        Self {
            image,
            image_layout: ImageLayout::TransferDstOptimal,
            clear_value: ClearDepthStencilValue {
                depth: 0.0,
                stencil: 0,
            },
            regions: &[],
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            image,
            image_layout,
            clear_value,
            regions,
            _ne: _,
        } = self;

        image_layout.validate_device(device).map_err(|err| {
            err.add_context("image_layout")
                .set_vuids(&["VUID-vkCmdClearDepthStencilImage-imageLayout-parameter"])
        })?;

        // VUID-vkCmdClearDepthStencilImage-commonparent
        assert_eq!(device, image.device().as_ref());

        if device.api_version() >= Version::V1_1 || device.enabled_extensions().khr_maintenance1 {
            if !image
                .format_features()
                .intersects(FormatFeatures::TRANSFER_DST)
            {
                return Err(Box::new(ValidationError {
                    context: "image.format_features()".into(),
                    problem: "does not contain `FormatFeatures::TRANSFER_DST`".into(),
                    vuids: &["VUID-vkCmdClearDepthStencilImage-image-01994"],
                    ..Default::default()
                }));
            }
        }

        let image_aspects = image.format().aspects();

        if !image_aspects.intersects(ImageAspects::DEPTH | ImageAspects::STENCIL) {
            return Err(Box::new(ValidationError {
                context: "image.format()".into(),
                problem: "is not a depth/stencil format".into(),
                vuids: &["VUID-vkCmdClearDepthStencilImage-image-00014"],
                ..Default::default()
            }));
        }

        if !matches!(
            image_layout,
            ImageLayout::TransferDstOptimal | ImageLayout::General
        ) {
            return Err(Box::new(ValidationError {
                context: "image_layout".into(),
                problem: "is not `ImageLayout::TransferDstOptimal` or `ImageLayout::General`"
                    .into(),
                vuids: &["VUID-vkCmdClearDepthStencilImage-imageLayout-00012"],
                ..Default::default()
            }));
        }

        if !(0.0..=1.0).contains(&clear_value.depth)
            && !device.enabled_extensions().ext_depth_range_unrestricted
        {
            return Err(Box::new(ValidationError {
                context: "clear_value.depth".into(),
                problem: "is not between `0.0` and `1.0` inclusive".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "ext_depth_range_unrestricted",
                )])]),
                vuids: &["VUID-VkClearDepthStencilValue-depth-00022"],
            }));
        }

        for (region_index, subresource_range) in regions.iter().enumerate() {
            subresource_range
                .validate(device)
                .map_err(|err| err.add_context(format!("regions[{}]", region_index)))?;

            if !image_aspects.contains(subresource_range.aspects) {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].aspects` is not a subset of `image.format().aspects()`",
                        region_index
                    )
                    .into(),
                    vuids: &[
                        "VUID-vkCmdClearDepthStencilImage-aspectMask-02824",
                        "VUID-vkCmdClearDepthStencilImage-image-02825",
                        "VUID-vkCmdClearDepthStencilImage-image-02826",
                    ],
                    ..Default::default()
                }));
            }

            if subresource_range.aspects.intersects(ImageAspects::STENCIL)
                && !image
                    .stencil_usage()
                    .unwrap_or(image.usage())
                    .intersects(ImageUsage::TRANSFER_DST)
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].aspects` contains `ImageAspects::STENCIL`, but \
                        `image.stencil_usage()` does not contain `ImageUsage::TRANSFER_DST`",
                        region_index
                    )
                    .into(),
                    vuids: &[
                        "VUID-vkCmdClearDepthStencilImage-pRanges-02658",
                        "VUID-vkCmdClearDepthStencilImage-pRanges-02659",
                    ],
                    ..Default::default()
                }));
            }

            if !(subresource_range.aspects - ImageAspects::STENCIL).is_empty()
                && !image.usage().intersects(ImageUsage::TRANSFER_DST)
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].aspects` contains aspects other than \
                        `ImageAspects::STENCIL`, but \
                        `image.usage()` does not contain `ImageUsage::TRANSFER_DST`",
                        region_index
                    )
                    .into(),
                    vuids: &["VUID-vkCmdClearDepthStencilImage-pRanges-02660"],
                    ..Default::default()
                }));
            }

            if subresource_range.base_mip_level >= image.mip_levels() {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].base_mip_level` is not less than `image.mip_levels()`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-vkCmdClearDepthStencilImage-baseMipLevel-01474"],
                    ..Default::default()
                }));
            }

            if let Some(subresource_range_level_count) = subresource_range.level_count {
                if subresource_range_level_count
                    > image.mip_levels() - subresource_range.base_mip_level
                {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`regions[{0}].base_mip_level + regions[{0}].level_count` is greater \
                            than `image.mip_levels()`",
                            region_index,
                        )
                        .into(),
                        vuids: &["VUID-vkCmdClearDepthStencilImage-pRanges-01694"],
                        ..Default::default()
                    }));
                }
            }

            if subresource_range.base_array_layer >= image.array_layers() {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].base_array_layer` is not less than `image.array_layers()`",
                        region_index,
                    )
                    .into(),
                    vuids: &["VUID-vkCmdClearDepthStencilImage-baseArrayLayer-01476"],
                    ..Default::default()
                }));
            }

            if let Some(subresource_range_layer_count) = subresource_range.layer_count {
                if subresource_range_layer_count
                    > image.array_layers() - subresource_range.base_array_layer
                {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`regions[{0}].base_array_layer + regions[{0}].layer_count` is \
                            greater than `image.array_layers()`",
                            region_index,
                        )
                        .into(),
                        vuids: &["VUID-vkCmdClearDepthStencilImage-pRanges-01695"],
                        ..Default::default()
                    }));
                }
            }
        }

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> ClearDepthStencilImageInfoVk {
        let &Self {
            image,
            image_layout,
            clear_value,
            regions: _,
            _ne: _,
        } = self;

        ClearDepthStencilImageInfoVk {
            image: image.handle(),
            image_layout: image_layout.into(),
            depth_stencil: clear_value.to_vk(),
        }
    }

    pub(crate) fn to_vk_ranges(&self) -> SmallVec<[vk::ImageSubresourceRange; 8]> {
        let &Self { image, regions, .. } = self;

        if regions.is_empty() {
            let region_vk = image.subresource_range().to_vk();

            smallvec![region_vk]
        } else {
            regions.iter().map(ImageSubresourceRange::to_vk).collect()
        }
    }
}

pub(crate) struct ClearDepthStencilImageInfoVk {
    pub(crate) image: vk::Image,
    pub(crate) image_layout: vk::ImageLayout,
    pub(crate) depth_stencil: vk::ClearDepthStencilValue,
}

/// Parameters to fill a region of a buffer with repeated copies of a value.
#[derive(Clone, Debug)]
pub struct FillBufferInfo<'a> {
    /// The buffer to fill.
    ///
    /// There is no default value.
    pub dst_buffer: &'a Buffer,

    /// The offset in bytes from the start of `dst_buffer` that filling will start from.
    ///
    /// This must be a multiple of 4.
    ///
    /// The default value is `0`.
    pub dst_offset: DeviceSize,

    /// The number of bytes to fill.
    ///
    /// If set to `Some`, this must be a multiple of 4.
    ///
    /// If set to `None`, fills until the end of the buffer.
    ///
    /// The default value is `None`.
    pub size: Option<DeviceSize>,

    /// The data to fill with.
    ///
    /// The default value is `0`.
    pub data: u32,

    pub _ne: crate::NonExhaustive<'a>,
}

impl<'a> FillBufferInfo<'a> {
    /// Returns a default `FillBufferInfo` with the provided `dst_buffer`.
    #[inline]
    pub const fn new(dst_buffer: &'a Buffer) -> Self {
        Self {
            dst_buffer,
            dst_offset: 0,
            size: None,
            data: 0,
            _ne: crate::NE,
        }
    }

    fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            dst_buffer,
            dst_offset,
            size,
            data: _,
            _ne: _,
        } = self;

        // VUID-vkCmdFillBuffer-commonparent
        assert_eq!(device, dst_buffer.device().as_ref());

        if dst_offset >= dst_buffer.size() {
            return Err(Box::new(ValidationError {
                context: "dst_offset".into(),
                problem: "is not less than `dst_buffer.size()`".into(),
                vuids: &["VUID-vkCmdFillBuffer-dstOffset-00024"],
                ..Default::default()
            }));
        }

        if dst_offset % 4 != 0 {
            return Err(Box::new(ValidationError {
                context: "dst_offset".into(),
                problem: "is not a multiple of 4".into(),
                vuids: &["VUID-vkCmdFillBuffer-dstOffset-00025"],
                ..Default::default()
            }));
        }

        if size == Some(0) {
            return Err(Box::new(ValidationError {
                context: "size".into(),
                problem: "is zero".into(),
                vuids: &["VUID-vkCmdFillBuffer-dstOffset-00026"],
                ..Default::default()
            }));
        }

        if let Some(size) = size {
            if size > dst_buffer.size() - dst_offset {
                return Err(Box::new(ValidationError {
                    problem: "`dst_offset + size` is greater than `dst_buffer.size()`".into(),
                    vuids: &["VUID-vkCmdFillBuffer-dstOffset-00027"],
                    ..Default::default()
                }));
            }

            if size % 4 != 0 {
                return Err(Box::new(ValidationError {
                    context: "size".into(),
                    problem: "is not a multiple of 4".into(),
                    vuids: &["VUID-vkCmdFillBuffer-dstOffset-00028"],
                    ..Default::default()
                }));
            }
        }

        if !dst_buffer.usage().intersects(BufferUsage::TRANSFER_DST) {
            return Err(Box::new(ValidationError {
                context: "dst_buffer.usage()".into(),
                problem: "does not contain `BufferUsage::TRANSFER_DST`".into(),
                vuids: &["VUID-vkCmdFillBuffer-dstBuffer-00029"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    fn to_vk(&self) -> FillBufferInfoVk {
        let &Self {
            dst_buffer,
            dst_offset,
            size,
            data,
            _ne: _,
        } = self;

        FillBufferInfoVk {
            dst_buffer: dst_buffer.handle(),
            dst_offset,
            size: size.unwrap_or(vk::WHOLE_SIZE),
            data,
        }
    }
}

pub(crate) struct FillBufferInfoVk {
    pub(crate) dst_buffer: vk::Buffer,
    pub(crate) dst_offset: DeviceSize,
    pub(crate) size: DeviceSize,
    pub(crate) data: u32,
}
