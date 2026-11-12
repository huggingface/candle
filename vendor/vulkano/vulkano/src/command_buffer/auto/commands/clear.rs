use crate::{
    buffer::{BufferContents, Subbuffer},
    command_buffer::{
        auto::Resource, raw, sys::RecordingCommandBuffer, AutoCommandBufferBuilder,
        ResourceInCommand,
    },
    format::{ClearColorValue, ClearDepthStencilValue},
    image::{Image, ImageLayout, ImageSubresourceRange},
    sync::PipelineStageAccessFlags,
    DeviceSize, SafeDeref, ValidationError,
};
use smallvec::{smallvec, SmallVec};
use std::sync::Arc;

/// # Commands to fill resources with new data.
impl<L> AutoCommandBufferBuilder<L> {
    /// Clears a color image with a specific value.
    pub fn clear_color_image(
        &mut self,
        clear_info: ClearColorImageInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_clear_color_image(&clear_info)?;

        Ok(unsafe { self.clear_color_image_unchecked(clear_info) })
    }

    fn validate_clear_color_image(
        &self,
        clear_info: &ClearColorImageInfo,
    ) -> Result<(), Box<ValidationError>> {
        let clear_info_raw = raw::ClearColorImageInfo {
            image: &clear_info.image,
            image_layout: clear_info.image_layout,
            clear_value: clear_info.clear_value,
            regions: &clear_info.regions,
            _ne: crate::NE,
        };
        self.inner.validate_clear_color_image(&clear_info_raw)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdClearColorImage-renderpass"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn clear_color_image_unchecked(
        &mut self,
        clear_info: ClearColorImageInfo,
    ) -> &mut Self {
        let &ClearColorImageInfo {
            ref image,
            image_layout,
            clear_value: _,
            ref regions,
            _ne: _,
        } = &clear_info;

        self.add_command(
            "clear_color_image",
            regions
                .iter()
                .cloned()
                .flat_map(|subresource_range| {
                    [(
                        ResourceInCommand::Destination.into(),
                        Resource::Image {
                            image: image.clone(),
                            subresource_range,
                            memory_access: PipelineStageAccessFlags::Clear_TransferWrite,
                            start_layout: image_layout,
                            end_layout: image_layout,
                        },
                    )]
                })
                .collect(),
            move |out: &mut RecordingCommandBuffer| {
                let clear_info_raw = raw::ClearColorImageInfo {
                    image: &clear_info.image,
                    image_layout: clear_info.image_layout,
                    clear_value: clear_info.clear_value,
                    regions: &clear_info.regions,
                    _ne: crate::NE,
                };
                unsafe { out.clear_color_image_unchecked(&clear_info_raw) };
            },
        );

        self
    }

    /// Clears a depth/stencil image with a specific value.
    pub fn clear_depth_stencil_image(
        &mut self,
        clear_info: ClearDepthStencilImageInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_clear_depth_stencil_image(&clear_info)?;

        Ok(unsafe { self.clear_depth_stencil_image_unchecked(clear_info) })
    }

    fn validate_clear_depth_stencil_image(
        &self,
        clear_info: &ClearDepthStencilImageInfo,
    ) -> Result<(), Box<ValidationError>> {
        let clear_info_raw = raw::ClearDepthStencilImageInfo {
            image: &clear_info.image,
            image_layout: clear_info.image_layout,
            clear_value: clear_info.clear_value,
            regions: &clear_info.regions,
            _ne: crate::NE,
        };
        self.inner
            .validate_clear_depth_stencil_image(&clear_info_raw)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdClearDepthStencilImage-renderpass"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn clear_depth_stencil_image_unchecked(
        &mut self,
        clear_info: ClearDepthStencilImageInfo,
    ) -> &mut Self {
        let &ClearDepthStencilImageInfo {
            ref image,
            image_layout,
            clear_value: _,
            ref regions,
            _ne: _,
        } = &clear_info;

        self.add_command(
            "clear_depth_stencil_image",
            regions
                .iter()
                .cloned()
                .flat_map(|subresource_range| {
                    [(
                        ResourceInCommand::Destination.into(),
                        Resource::Image {
                            image: image.clone(),
                            subresource_range,
                            memory_access: PipelineStageAccessFlags::Clear_TransferWrite,
                            start_layout: image_layout,
                            end_layout: image_layout,
                        },
                    )]
                })
                .collect(),
            move |out: &mut RecordingCommandBuffer| {
                let clear_info_raw = raw::ClearDepthStencilImageInfo {
                    image: &clear_info.image,
                    image_layout: clear_info.image_layout,
                    clear_value: clear_info.clear_value,
                    regions: &clear_info.regions,
                    _ne: crate::NE,
                };
                unsafe { out.clear_depth_stencil_image_unchecked(&clear_info_raw) };
            },
        );

        self
    }

    /// Fills a region of a buffer with repeated copies of a value.
    ///
    /// This function is similar to the `memset` function in C. The `data` parameter is a number
    /// that will be repeatedly written through the entire buffer.
    pub fn fill_buffer(
        &mut self,
        dst_buffer: Subbuffer<[u32]>,
        data: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_fill_buffer(&dst_buffer, data)?;

        Ok(unsafe { self.fill_buffer_unchecked(dst_buffer, data) })
    }

    fn validate_fill_buffer(
        &self,
        dst_buffer: &Subbuffer<[u32]>,
        data: u32,
    ) -> Result<(), Box<ValidationError>> {
        let fill_info_raw = raw::FillBufferInfo {
            dst_buffer: dst_buffer.buffer(),
            dst_offset: dst_buffer.offset(),
            size: Some(dst_buffer.size()),
            data,
            _ne: crate::NE,
        };
        self.inner.validate_fill_buffer(&fill_info_raw)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdFillBuffer-renderpass"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn fill_buffer_unchecked(
        &mut self,
        dst_buffer: Subbuffer<[u32]>,
        data: u32,
    ) -> &mut Self {
        self.add_command(
            "fill_buffer",
            [(
                ResourceInCommand::Destination.into(),
                Resource::Buffer {
                    buffer: dst_buffer.as_bytes().clone(),
                    range: 0..dst_buffer.size(),
                    memory_access: PipelineStageAccessFlags::Clear_TransferWrite,
                },
            )]
            .into_iter()
            .collect(),
            move |out: &mut RecordingCommandBuffer| {
                let fill_info_raw = raw::FillBufferInfo {
                    dst_buffer: dst_buffer.buffer(),
                    dst_offset: dst_buffer.offset(),
                    size: Some(dst_buffer.size()),
                    data,
                    _ne: crate::NE,
                };
                unsafe { out.fill_buffer_unchecked(&fill_info_raw) };
            },
        );

        self
    }

    /// Writes data to a region of a buffer.
    pub fn update_buffer<D, Dd>(
        &mut self,
        dst_buffer: Subbuffer<D>,
        data: Dd,
    ) -> Result<&mut Self, Box<ValidationError>>
    where
        D: BufferContents + ?Sized,
        Dd: SafeDeref<Target = D> + Send + Sync + 'static,
    {
        self.validate_update_buffer(
            dst_buffer.as_bytes(),
            size_of_val(data.deref()) as DeviceSize,
        )?;

        Ok(unsafe { self.update_buffer_unchecked(dst_buffer, data) })
    }

    fn validate_update_buffer(
        &self,
        dst_buffer: &Subbuffer<[u8]>,
        data_size: DeviceSize,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_update_buffer(dst_buffer.buffer(), dst_buffer.offset(), data_size)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdUpdateBuffer-renderpass"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn update_buffer_unchecked<D, Dd>(
        &mut self,
        dst_buffer: Subbuffer<D>,
        data: Dd,
    ) -> &mut Self
    where
        D: BufferContents + ?Sized,
        Dd: SafeDeref<Target = D> + Send + Sync + 'static,
    {
        self.add_command(
            "update_buffer",
            [(
                ResourceInCommand::Destination.into(),
                Resource::Buffer {
                    buffer: dst_buffer.as_bytes().clone(),
                    range: 0..size_of_val(data.deref()) as DeviceSize,
                    memory_access: PipelineStageAccessFlags::Clear_TransferWrite,
                },
            )]
            .into_iter()
            .collect(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe {
                    out.update_buffer_unchecked(dst_buffer.buffer(), dst_buffer.offset(), &*data)
                };
            },
        );

        self
    }
}

/// Parameters to clear a color image.
#[derive(Clone, Debug)]
pub struct ClearColorImageInfo {
    /// The image to clear.
    ///
    /// There is no default value.
    pub image: Arc<Image>,

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
    pub regions: SmallVec<[ImageSubresourceRange; 1]>,

    pub _ne: crate::NonExhaustive<'static>,
}

impl ClearColorImageInfo {
    /// Returns a default `ClearColorImageInfo` with the provided `image`.
    #[inline]
    pub fn new(image: Arc<Image>) -> Self {
        let range = image.subresource_range();

        Self {
            image,
            image_layout: ImageLayout::TransferDstOptimal,
            clear_value: ClearColorValue::Float([0.0; 4]),
            regions: smallvec![range],
            _ne: crate::NE,
        }
    }

    #[deprecated(since = "0.36.0", note = "use `new` instead")]
    #[inline]
    pub fn image(image: Arc<Image>) -> Self {
        Self::new(image)
    }
}

/// Parameters to clear a depth/stencil image.
#[derive(Clone, Debug)]
pub struct ClearDepthStencilImageInfo {
    /// The image to clear.
    ///
    /// There is no default value.
    pub image: Arc<Image>,

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
    pub regions: SmallVec<[ImageSubresourceRange; 1]>,

    pub _ne: crate::NonExhaustive<'static>,
}

impl ClearDepthStencilImageInfo {
    /// Returns a default `ClearDepthStencilImageInfo` with the provided `image`.
    #[inline]
    pub fn new(image: Arc<Image>) -> Self {
        let range = image.subresource_range();

        Self {
            image,
            image_layout: ImageLayout::TransferDstOptimal,
            clear_value: ClearDepthStencilValue::default(),
            regions: smallvec![range],
            _ne: crate::NE,
        }
    }

    #[deprecated(since = "0.36.0", note = "use `new` instead")]
    #[inline]
    pub fn image(image: Arc<Image>) -> Self {
        Self::new(image)
    }
}
