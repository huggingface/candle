use crate::{
    buffer::Subbuffer,
    command_buffer::{
        auto::Resource, raw, sys::RecordingCommandBuffer, AutoCommandBufferBuilder,
        ResourceInCommand,
    },
    image::{sampler::Filter, Image, ImageLayout, ImageSubresourceLayers},
    sync::PipelineStageAccessFlags,
    DeviceSize, ValidationError,
};
use smallvec::{smallvec, SmallVec};
use std::{
    cmp::{max, min},
    sync::Arc,
};

/// # Commands to transfer data between resources.
impl<L> AutoCommandBufferBuilder<L> {
    /// Copies data from a buffer to another buffer.
    ///
    /// # Panics
    ///
    /// - Panics if `src_buffer` or `dst_buffer` were not created from the same device as `self`.
    pub fn copy_buffer(
        &mut self,
        copy_buffer_info: impl Into<CopyBufferInfo>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        let copy_buffer_info = copy_buffer_info.into();
        self.validate_copy_buffer(&copy_buffer_info)?;

        Ok(unsafe { self.copy_buffer_unchecked(copy_buffer_info) })
    }

    fn validate_copy_buffer(
        &self,
        copy_buffer_info: &CopyBufferInfo,
    ) -> Result<(), Box<ValidationError>> {
        let regions_raw = copy_buffer_info
            .regions
            .iter()
            .map(|region| raw::BufferCopy {
                src_offset: copy_buffer_info
                    .src_buffer
                    .offset()
                    .checked_add(region.src_offset)
                    .unwrap(),
                dst_offset: copy_buffer_info
                    .dst_buffer
                    .offset()
                    .checked_add(region.dst_offset)
                    .unwrap(),
                size: region.size,
                _ne: crate::NE,
            })
            .collect::<SmallVec<[_; 1]>>();
        let copy_buffer_info_raw = raw::CopyBufferInfo {
            src_buffer: copy_buffer_info.src_buffer.buffer(),
            dst_buffer: copy_buffer_info.dst_buffer.buffer(),
            regions: &regions_raw,
            _ne: crate::NE,
        };
        self.inner.validate_copy_buffer(&copy_buffer_info_raw)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdCopyBuffer2-renderpass"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn copy_buffer_unchecked(
        &mut self,
        copy_buffer_info: impl Into<CopyBufferInfo>,
    ) -> &mut Self {
        let copy_buffer_info = copy_buffer_info.into();
        let CopyBufferInfo {
            src_buffer,
            dst_buffer,
            regions,
            _ne: _,
        } = &copy_buffer_info;

        self.add_command(
            "copy_buffer",
            regions
                .iter()
                .flat_map(|region| {
                    let &BufferCopy {
                        src_offset,
                        dst_offset,
                        size,
                        _ne: _,
                    } = region;

                    [
                        (
                            ResourceInCommand::Source.into(),
                            Resource::Buffer {
                                buffer: src_buffer.clone(),
                                range: src_offset..src_offset + size,
                                memory_access: PipelineStageAccessFlags::Copy_TransferRead,
                            },
                        ),
                        (
                            ResourceInCommand::Destination.into(),
                            Resource::Buffer {
                                buffer: dst_buffer.clone(),
                                range: dst_offset..dst_offset + size,
                                memory_access: PipelineStageAccessFlags::Copy_TransferWrite,
                            },
                        ),
                    ]
                })
                .collect(),
            move |out: &mut RecordingCommandBuffer| {
                let regions_raw = copy_buffer_info
                    .regions
                    .iter()
                    .map(|region| raw::BufferCopy {
                        src_offset: copy_buffer_info
                            .src_buffer
                            .offset()
                            .checked_add(region.src_offset)
                            .unwrap(),
                        dst_offset: copy_buffer_info
                            .dst_buffer
                            .offset()
                            .checked_add(region.dst_offset)
                            .unwrap(),
                        size: region.size,
                        _ne: crate::NE,
                    })
                    .collect::<SmallVec<[_; 1]>>();
                let copy_buffer_info_raw = raw::CopyBufferInfo {
                    src_buffer: copy_buffer_info.src_buffer.buffer(),
                    dst_buffer: copy_buffer_info.dst_buffer.buffer(),
                    regions: &regions_raw,
                    _ne: crate::NE,
                };
                unsafe { out.copy_buffer_unchecked(&copy_buffer_info_raw) };
            },
        );

        self
    }

    /// Copies data from an image to another image.
    ///
    /// There are several restrictions:
    ///
    /// - The number of samples in the source and destination images must be equal.
    /// - The size of the uncompressed element format of the source image must be equal to the
    ///   compressed element format of the destination.
    /// - If you copy between depth, stencil or depth-stencil images, the format of both images
    ///   must match exactly.
    /// - For two-dimensional images, the Z coordinate must be 0 for the image offsets and 1 for
    ///   the extent. Same for the Y coordinate for one-dimensional images.
    /// - For non-array images, the base array layer must be 0 and the number of layers must be 1.
    ///
    /// If `layer_count` is greater than 1, the copy will happen between each individual layer as
    /// if they were separate images.
    ///
    /// # Panics
    ///
    /// - Panics if `src_image` or `dst_image` were not created from the same device as `self`.
    pub fn copy_image(
        &mut self,
        copy_image_info: CopyImageInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_copy_image(&copy_image_info)?;

        Ok(unsafe { self.copy_image_unchecked(copy_image_info) })
    }

    fn validate_copy_image(
        &self,
        copy_image_info: &CopyImageInfo,
    ) -> Result<(), Box<ValidationError>> {
        let regions_raw = copy_image_info
            .regions
            .iter()
            .map(|region| raw::ImageCopy {
                src_subresource: region.src_subresource,
                src_offset: region.src_offset,
                dst_subresource: region.dst_subresource,
                dst_offset: region.dst_offset,
                extent: region.extent,
                _ne: crate::NE,
            })
            .collect::<SmallVec<[_; 1]>>();
        let copy_image_info_raw = raw::CopyImageInfo {
            src_image: &copy_image_info.src_image,
            src_image_layout: copy_image_info.src_image_layout,
            dst_image: &copy_image_info.dst_image,
            dst_image_layout: copy_image_info.dst_image_layout,
            regions: &regions_raw,
            _ne: crate::NE,
        };
        self.inner.validate_copy_image(&copy_image_info_raw)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdCopyImage2-renderpass"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn copy_image_unchecked(&mut self, copy_image_info: CopyImageInfo) -> &mut Self {
        let &CopyImageInfo {
            ref src_image,
            src_image_layout,
            ref dst_image,
            dst_image_layout,
            ref regions,
            _ne: _,
        } = &copy_image_info;

        self.add_command(
            "copy_image",
            regions
                .iter()
                .flat_map(|region| {
                    let &ImageCopy {
                        src_subresource,
                        src_offset: _,
                        dst_subresource,
                        dst_offset: _,
                        extent: _,
                        _ne: _,
                    } = region;

                    [
                        (
                            ResourceInCommand::Source.into(),
                            Resource::Image {
                                image: src_image.clone(),
                                subresource_range: src_subresource.into(),
                                memory_access: PipelineStageAccessFlags::Copy_TransferRead,
                                start_layout: src_image_layout,
                                end_layout: src_image_layout,
                            },
                        ),
                        (
                            ResourceInCommand::Destination.into(),
                            Resource::Image {
                                image: dst_image.clone(),
                                subresource_range: dst_subresource.into(),
                                memory_access: PipelineStageAccessFlags::Copy_TransferWrite,
                                start_layout: dst_image_layout,
                                end_layout: dst_image_layout,
                            },
                        ),
                    ]
                })
                .collect(),
            move |out: &mut RecordingCommandBuffer| {
                let regions_raw = copy_image_info
                    .regions
                    .iter()
                    .map(|region| raw::ImageCopy {
                        src_subresource: region.src_subresource,
                        src_offset: region.src_offset,
                        dst_subresource: region.dst_subresource,
                        dst_offset: region.dst_offset,
                        extent: region.extent,
                        _ne: crate::NE,
                    })
                    .collect::<SmallVec<[_; 1]>>();
                let copy_image_info_raw = raw::CopyImageInfo {
                    src_image: &copy_image_info.src_image,
                    src_image_layout: copy_image_info.src_image_layout,
                    dst_image: &copy_image_info.dst_image,
                    dst_image_layout: copy_image_info.dst_image_layout,
                    regions: &regions_raw,
                    _ne: crate::NE,
                };
                unsafe { out.copy_image_unchecked(&copy_image_info_raw) };
            },
        );

        self
    }

    /// Copies from a buffer to an image.
    pub fn copy_buffer_to_image(
        &mut self,
        copy_buffer_to_image_info: CopyBufferToImageInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_copy_buffer_to_image(&copy_buffer_to_image_info)?;

        Ok(unsafe { self.copy_buffer_to_image_unchecked(copy_buffer_to_image_info) })
    }

    fn validate_copy_buffer_to_image(
        &self,
        copy_buffer_to_image_info: &CopyBufferToImageInfo,
    ) -> Result<(), Box<ValidationError>> {
        let regions_raw = copy_buffer_to_image_info
            .regions
            .iter()
            .map(|region| raw::BufferImageCopy {
                buffer_offset: copy_buffer_to_image_info
                    .src_buffer
                    .offset()
                    .checked_add(region.buffer_offset)
                    .unwrap(),
                buffer_row_length: region.buffer_row_length,
                buffer_image_height: region.buffer_image_height,
                image_subresource: region.image_subresource,
                image_offset: region.image_offset,
                image_extent: region.image_extent,
                _ne: crate::NE,
            })
            .collect::<SmallVec<[_; 1]>>();
        let copy_buffer_to_image_info_raw = raw::CopyBufferToImageInfo {
            src_buffer: copy_buffer_to_image_info.src_buffer.buffer(),
            dst_image: &copy_buffer_to_image_info.dst_image,
            dst_image_layout: copy_buffer_to_image_info.dst_image_layout,
            regions: &regions_raw,
            _ne: crate::NE,
        };
        self.inner
            .validate_copy_buffer_to_image(&copy_buffer_to_image_info_raw)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdCopyBufferToImage2-renderpass"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn copy_buffer_to_image_unchecked(
        &mut self,
        copy_buffer_to_image_info: CopyBufferToImageInfo,
    ) -> &mut Self {
        let &CopyBufferToImageInfo {
            ref src_buffer,
            ref dst_image,
            dst_image_layout,
            ref regions,
            _ne: _,
        } = &copy_buffer_to_image_info;

        self.add_command(
            "copy_buffer_to_image",
            regions
                .iter()
                .flat_map(|region| {
                    let &BufferImageCopy {
                        buffer_offset,
                        buffer_row_length: _,
                        buffer_image_height: _,
                        image_subresource,
                        image_offset: _,
                        image_extent: _,
                        _ne: _,
                    } = region;

                    [
                        (
                            ResourceInCommand::Source.into(),
                            Resource::Buffer {
                                buffer: src_buffer.clone(),
                                range: buffer_offset
                                    ..buffer_offset + region.buffer_copy_size(dst_image),
                                memory_access: PipelineStageAccessFlags::Copy_TransferRead,
                            },
                        ),
                        (
                            ResourceInCommand::Destination.into(),
                            Resource::Image {
                                image: dst_image.clone(),
                                subresource_range: image_subresource.into(),
                                memory_access: PipelineStageAccessFlags::Copy_TransferWrite,
                                start_layout: dst_image_layout,
                                end_layout: dst_image_layout,
                            },
                        ),
                    ]
                })
                .collect(),
            move |out: &mut RecordingCommandBuffer| {
                let regions_raw = copy_buffer_to_image_info
                    .regions
                    .iter()
                    .map(|region| raw::BufferImageCopy {
                        buffer_offset: copy_buffer_to_image_info
                            .src_buffer
                            .offset()
                            .checked_add(region.buffer_offset)
                            .unwrap(),
                        buffer_row_length: region.buffer_row_length,
                        buffer_image_height: region.buffer_image_height,
                        image_subresource: region.image_subresource,
                        image_offset: region.image_offset,
                        image_extent: region.image_extent,
                        _ne: crate::NE,
                    })
                    .collect::<SmallVec<[_; 1]>>();
                let copy_buffer_to_image_info_raw = raw::CopyBufferToImageInfo {
                    src_buffer: copy_buffer_to_image_info.src_buffer.buffer(),
                    dst_image: &copy_buffer_to_image_info.dst_image,
                    dst_image_layout: copy_buffer_to_image_info.dst_image_layout,
                    regions: &regions_raw,
                    _ne: crate::NE,
                };
                unsafe { out.copy_buffer_to_image_unchecked(&copy_buffer_to_image_info_raw) };
            },
        );

        self
    }

    /// Copies from an image to a buffer.
    pub fn copy_image_to_buffer(
        &mut self,
        copy_image_to_buffer_info: CopyImageToBufferInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_copy_image_to_buffer(&copy_image_to_buffer_info)?;

        Ok(unsafe { self.copy_image_to_buffer_unchecked(copy_image_to_buffer_info) })
    }

    fn validate_copy_image_to_buffer(
        &self,
        copy_image_to_buffer_info: &CopyImageToBufferInfo,
    ) -> Result<(), Box<ValidationError>> {
        let regions_raw = copy_image_to_buffer_info
            .regions
            .iter()
            .map(|region| raw::BufferImageCopy {
                buffer_offset: copy_image_to_buffer_info
                    .dst_buffer
                    .offset()
                    .checked_add(region.buffer_offset)
                    .unwrap(),
                buffer_row_length: region.buffer_row_length,
                buffer_image_height: region.buffer_image_height,
                image_subresource: region.image_subresource,
                image_offset: region.image_offset,
                image_extent: region.image_extent,
                _ne: crate::NE,
            })
            .collect::<SmallVec<[_; 1]>>();
        let copy_image_to_buffer_info_raw = raw::CopyImageToBufferInfo {
            src_image: &copy_image_to_buffer_info.src_image,
            src_image_layout: copy_image_to_buffer_info.src_image_layout,
            dst_buffer: copy_image_to_buffer_info.dst_buffer.buffer(),
            regions: &regions_raw,
            _ne: crate::NE,
        };
        self.inner
            .validate_copy_image_to_buffer(&copy_image_to_buffer_info_raw)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdCopyImageToBuffer2-renderpass"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn copy_image_to_buffer_unchecked(
        &mut self,
        copy_image_to_buffer_info: CopyImageToBufferInfo,
    ) -> &mut Self {
        let &CopyImageToBufferInfo {
            ref src_image,
            src_image_layout,
            ref dst_buffer,
            ref regions,
            _ne: _,
        } = &copy_image_to_buffer_info;

        self.add_command(
            "copy_image_to_buffer",
            regions
                .iter()
                .flat_map(|region| {
                    let &BufferImageCopy {
                        buffer_offset,
                        buffer_row_length: _,
                        buffer_image_height: _,
                        image_subresource,
                        image_offset: _,
                        image_extent: _,
                        _ne: _,
                    } = region;

                    [
                        (
                            ResourceInCommand::Source.into(),
                            Resource::Image {
                                image: src_image.clone(),
                                subresource_range: image_subresource.into(),
                                memory_access: PipelineStageAccessFlags::Copy_TransferRead,
                                start_layout: src_image_layout,
                                end_layout: src_image_layout,
                            },
                        ),
                        (
                            ResourceInCommand::Destination.into(),
                            Resource::Buffer {
                                buffer: dst_buffer.clone(),
                                range: buffer_offset
                                    ..buffer_offset + region.buffer_copy_size(src_image),
                                memory_access: PipelineStageAccessFlags::Copy_TransferWrite,
                            },
                        ),
                    ]
                })
                .collect(),
            move |out: &mut RecordingCommandBuffer| {
                let regions_raw = copy_image_to_buffer_info
                    .regions
                    .iter()
                    .map(|region| raw::BufferImageCopy {
                        buffer_offset: copy_image_to_buffer_info
                            .dst_buffer
                            .offset()
                            .checked_add(region.buffer_offset)
                            .unwrap(),
                        buffer_row_length: region.buffer_row_length,
                        buffer_image_height: region.buffer_image_height,
                        image_subresource: region.image_subresource,
                        image_offset: region.image_offset,
                        image_extent: region.image_extent,
                        _ne: crate::NE,
                    })
                    .collect::<SmallVec<[_; 1]>>();
                let copy_image_to_buffer_info_raw = raw::CopyImageToBufferInfo {
                    src_image: &copy_image_to_buffer_info.src_image,
                    src_image_layout: copy_image_to_buffer_info.src_image_layout,
                    dst_buffer: copy_image_to_buffer_info.dst_buffer.buffer(),
                    regions: &regions_raw,
                    _ne: crate::NE,
                };
                unsafe { out.copy_image_to_buffer_unchecked(&copy_image_to_buffer_info_raw) };
            },
        );

        self
    }

    /// Blits an image to another.
    ///
    /// A *blit* is similar to an image copy operation, except that the portion of the image that
    /// is transferred can be resized. You choose an area of the source and an area of the
    /// destination, and the implementation will resize the area of the source so that it matches
    /// the size of the area of the destination before writing it.
    ///
    /// Blit operations have several restrictions:
    ///
    /// - Blit operations are only allowed on queue families that support graphics operations.
    /// - The format of the source and destination images must support blit operations, which
    ///   depends on the Vulkan implementation. Vulkan guarantees that some specific formats must
    ///   always be supported. See tables 52 to 61 of the specifications.
    /// - Only single-sampled images are allowed.
    /// - You can only blit between two images whose formats belong to the same type. The types
    ///   are: floating-point, signed integers, unsigned integers, depth-stencil.
    /// - If you blit between depth, stencil or depth-stencil images, the format of both images
    ///   must match exactly.
    /// - If you blit between depth, stencil or depth-stencil images, only the `Nearest` filter is
    ///   allowed.
    /// - For two-dimensional images, the Z coordinate must be 0 for the top-left offset and 1 for
    ///   the bottom-right offset. Same for the Y coordinate for one-dimensional images.
    /// - For non-array images, the base array layer must be 0 and the number of layers must be 1.
    ///
    /// If `layer_count` is greater than 1, the blit will happen between each individual layer as
    /// if they were separate images.
    ///
    /// # Panics
    ///
    /// - Panics if the source or the destination was not created with `device`.
    pub fn blit_image(
        &mut self,
        blit_image_info: BlitImageInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_blit_image(&blit_image_info)?;

        Ok(unsafe { self.blit_image_unchecked(blit_image_info) })
    }

    fn validate_blit_image(
        &self,
        blit_image_info: &BlitImageInfo,
    ) -> Result<(), Box<ValidationError>> {
        let regions_raw = blit_image_info
            .regions
            .iter()
            .map(|region| raw::ImageBlit {
                src_subresource: region.src_subresource,
                src_offsets: region.src_offsets,
                dst_subresource: region.dst_subresource,
                dst_offsets: region.dst_offsets,
                _ne: crate::NE,
            })
            .collect::<SmallVec<[_; 1]>>();
        let blit_image_info_raw = raw::BlitImageInfo {
            src_image: &blit_image_info.src_image,
            src_image_layout: blit_image_info.src_image_layout,
            dst_image: &blit_image_info.dst_image,
            dst_image_layout: blit_image_info.dst_image_layout,
            regions: &regions_raw,
            filter: blit_image_info.filter,
            _ne: crate::NE,
        };
        self.inner.validate_blit_image(&blit_image_info_raw)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdBlitImage2-renderpass"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn blit_image_unchecked(&mut self, blit_image_info: BlitImageInfo) -> &mut Self {
        let &BlitImageInfo {
            ref src_image,
            src_image_layout,
            ref dst_image,
            dst_image_layout,
            ref regions,
            filter: _,
            _ne: _,
        } = &blit_image_info;

        self.add_command(
            "blit_image",
            regions
                .iter()
                .flat_map(|region| {
                    let &ImageBlit {
                        src_subresource,
                        src_offsets: _,
                        dst_subresource,
                        dst_offsets: _,
                        _ne: _,
                    } = region;

                    [
                        (
                            ResourceInCommand::Source.into(),
                            Resource::Image {
                                image: src_image.clone(),
                                subresource_range: src_subresource.into(),
                                memory_access: PipelineStageAccessFlags::Blit_TransferRead,
                                start_layout: src_image_layout,
                                end_layout: src_image_layout,
                            },
                        ),
                        (
                            ResourceInCommand::Destination.into(),
                            Resource::Image {
                                image: dst_image.clone(),
                                subresource_range: dst_subresource.into(),
                                memory_access: PipelineStageAccessFlags::Blit_TransferWrite,
                                start_layout: dst_image_layout,
                                end_layout: dst_image_layout,
                            },
                        ),
                    ]
                })
                .collect(),
            move |out: &mut RecordingCommandBuffer| {
                let regions_raw = blit_image_info
                    .regions
                    .iter()
                    .map(|region| raw::ImageBlit {
                        src_subresource: region.src_subresource,
                        src_offsets: region.src_offsets,
                        dst_subresource: region.dst_subresource,
                        dst_offsets: region.dst_offsets,
                        _ne: crate::NE,
                    })
                    .collect::<SmallVec<[_; 1]>>();
                let blit_image_info_raw = raw::BlitImageInfo {
                    src_image: &blit_image_info.src_image,
                    src_image_layout: blit_image_info.src_image_layout,
                    dst_image: &blit_image_info.dst_image,
                    dst_image_layout: blit_image_info.dst_image_layout,
                    regions: &regions_raw,
                    filter: blit_image_info.filter,
                    _ne: crate::NE,
                };
                unsafe { out.blit_image_unchecked(&blit_image_info_raw) };
            },
        );

        self
    }

    /// Resolves a multisampled image into a single-sampled image.
    ///
    /// # Panics
    ///
    /// - Panics if `src_image` or `dst_image` were not created from the same device as `self`.
    pub fn resolve_image(
        &mut self,
        resolve_image_info: ResolveImageInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_resolve_image(&resolve_image_info)?;

        Ok(unsafe { self.resolve_image_unchecked(resolve_image_info) })
    }

    fn validate_resolve_image(
        &self,
        resolve_image_info: &ResolveImageInfo,
    ) -> Result<(), Box<ValidationError>> {
        let regions_raw = resolve_image_info
            .regions
            .iter()
            .map(|region| raw::ImageResolve {
                src_subresource: region.src_subresource,
                src_offset: region.src_offset,
                dst_subresource: region.dst_subresource,
                dst_offset: region.dst_offset,
                extent: region.extent,
                _ne: crate::NE,
            })
            .collect::<SmallVec<[_; 1]>>();
        let resolve_image_info_raw = raw::ResolveImageInfo {
            src_image: &resolve_image_info.src_image,
            src_image_layout: resolve_image_info.src_image_layout,
            dst_image: &resolve_image_info.dst_image,
            dst_image_layout: resolve_image_info.dst_image_layout,
            regions: &regions_raw,
            _ne: crate::NE,
        };
        self.inner.validate_resolve_image(&resolve_image_info_raw)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdResolveImage2-renderpass"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn resolve_image_unchecked(
        &mut self,
        resolve_image_info: ResolveImageInfo,
    ) -> &mut Self {
        let &ResolveImageInfo {
            ref src_image,
            src_image_layout,
            ref dst_image,
            dst_image_layout,
            ref regions,
            _ne: _,
        } = &resolve_image_info;

        self.add_command(
            "resolve_image",
            regions
                .iter()
                .flat_map(|region| {
                    let &ImageResolve {
                        src_subresource,
                        src_offset: _,
                        dst_subresource,
                        dst_offset: _,
                        extent: _,
                        _ne: _,
                    } = region;

                    [
                        (
                            ResourceInCommand::Source.into(),
                            Resource::Image {
                                image: src_image.clone(),
                                subresource_range: src_subresource.into(),
                                memory_access: PipelineStageAccessFlags::Resolve_TransferRead,
                                start_layout: src_image_layout,
                                end_layout: src_image_layout,
                            },
                        ),
                        (
                            ResourceInCommand::Destination.into(),
                            Resource::Image {
                                image: dst_image.clone(),
                                subresource_range: dst_subresource.into(),
                                memory_access: PipelineStageAccessFlags::Resolve_TransferWrite,
                                start_layout: dst_image_layout,
                                end_layout: dst_image_layout,
                            },
                        ),
                    ]
                })
                .collect(),
            move |out: &mut RecordingCommandBuffer| {
                let regions_raw = resolve_image_info
                    .regions
                    .iter()
                    .map(|region| raw::ImageResolve {
                        src_subresource: region.src_subresource,
                        src_offset: region.src_offset,
                        dst_subresource: region.dst_subresource,
                        dst_offset: region.dst_offset,
                        extent: region.extent,
                        _ne: crate::NE,
                    })
                    .collect::<SmallVec<[_; 1]>>();
                let resolve_image_info_raw = raw::ResolveImageInfo {
                    src_image: &resolve_image_info.src_image,
                    src_image_layout: resolve_image_info.src_image_layout,
                    dst_image: &resolve_image_info.dst_image,
                    dst_image_layout: resolve_image_info.dst_image_layout,
                    regions: &regions_raw,
                    _ne: crate::NE,
                };
                unsafe { out.resolve_image_unchecked(&resolve_image_info_raw) };
            },
        );

        self
    }
}

/// Parameters to copy data from a buffer to another buffer.
///
/// The fields of `regions` represent bytes.
#[derive(Clone, Debug)]
pub struct CopyBufferInfo {
    /// The buffer to copy from.
    ///
    /// There is no default value.
    pub src_buffer: Subbuffer<[u8]>,

    /// The buffer to copy to.
    ///
    /// There is no default value.
    pub dst_buffer: Subbuffer<[u8]>,

    /// The regions of both buffers to copy between, specified in bytes.
    ///
    /// The default value is a single region, with zero offsets and a `size` equal to the smallest
    /// of the two buffers.
    pub regions: SmallVec<[BufferCopy; 1]>,

    pub _ne: crate::NonExhaustive<'static>,
}

impl CopyBufferInfo {
    /// Returns a default `CopyBufferInfo` with the provided `src_buffer` and `dst_buffer`.
    #[inline]
    pub fn new(src_buffer: Subbuffer<impl ?Sized>, dst_buffer: Subbuffer<impl ?Sized>) -> Self {
        let region = BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: min(src_buffer.size(), dst_buffer.size()),
            ..Default::default()
        };

        Self {
            src_buffer: src_buffer.into_bytes(),
            dst_buffer: dst_buffer.into_bytes(),
            regions: smallvec![region],
            _ne: crate::NE,
        }
    }

    #[deprecated(since = "0.36.0", note = "use `new` instead")]
    #[inline]
    pub fn buffers(src_buffer: Subbuffer<impl ?Sized>, dst_buffer: Subbuffer<impl ?Sized>) -> Self {
        Self::new(src_buffer, dst_buffer)
    }
}

/// Parameters to copy data from a buffer to another buffer, with type information.
///
/// The fields of `regions` represent elements of `T`.
#[derive(Clone, Debug)]
pub struct CopyBufferInfoTyped<T> {
    /// The buffer to copy from.
    ///
    /// There is no default value.
    pub src_buffer: Subbuffer<[T]>,

    /// The buffer to copy to.
    ///
    /// There is no default value.
    pub dst_buffer: Subbuffer<[T]>,

    /// The regions of both buffers to copy between, specified in elements of `T`.
    ///
    /// The default value is a single region, with zero offsets and a `size` equal to the smallest
    /// of the two buffers.
    pub regions: SmallVec<[BufferCopy; 1]>,

    pub _ne: crate::NonExhaustive<'static>,
}

impl<T> CopyBufferInfoTyped<T> {
    /// Returns a default `CopyBufferInfoTyped` with the provided `src_buffer` and `dst_buffer`.
    #[inline]
    pub fn new(src_buffer: Subbuffer<[T]>, dst_buffer: Subbuffer<[T]>) -> Self {
        let region = BufferCopy {
            size: min(src_buffer.len(), dst_buffer.len()),
            ..Default::default()
        };

        Self {
            src_buffer,
            dst_buffer,
            regions: smallvec![region],
            _ne: crate::NE,
        }
    }

    #[deprecated(since = "0.36.0", note = "use `new` instead")]
    #[inline]
    pub fn buffers(src_buffer: Subbuffer<[T]>, dst_buffer: Subbuffer<[T]>) -> Self {
        Self::new(src_buffer, dst_buffer)
    }
}

impl<T> From<CopyBufferInfoTyped<T>> for CopyBufferInfo {
    fn from(typed: CopyBufferInfoTyped<T>) -> Self {
        let CopyBufferInfoTyped {
            src_buffer,
            dst_buffer,
            mut regions,
            _ne: _,
        } = typed;

        for region in &mut regions {
            region.src_offset *= size_of::<T>() as DeviceSize;
            region.dst_offset *= size_of::<T>() as DeviceSize;
            region.size *= size_of::<T>() as DeviceSize;
        }

        Self {
            src_buffer: src_buffer.as_bytes().clone(),
            dst_buffer: dst_buffer.as_bytes().clone(),
            regions,
            _ne: crate::NE,
        }
    }
}

/// A region of data to copy between buffers.
#[derive(Clone, Debug)]
pub struct BufferCopy {
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

    pub _ne: crate::NonExhaustive<'static>,
}

impl Default for BufferCopy {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl BufferCopy {
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
}

/// Parameters to copy data from an image to another image.
#[derive(Clone, Debug)]
pub struct CopyImageInfo {
    /// The image to copy from.
    ///
    /// There is no default value.
    pub src_image: Arc<Image>,

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
    pub dst_image: Arc<Image>,

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
    pub regions: SmallVec<[ImageCopy; 1]>,

    pub _ne: crate::NonExhaustive<'static>,
}

impl CopyImageInfo {
    /// Returns a default `CopyImageInfo` with the provided `src_image` and `dst_image`.
    #[inline]
    pub fn new(src_image: Arc<Image>, dst_image: Arc<Image>) -> Self {
        let min_array_layers = src_image.array_layers().min(dst_image.array_layers());
        let region = ImageCopy {
            src_subresource: ImageSubresourceLayers {
                layer_count: Some(min_array_layers),
                ..src_image.subresource_layers()
            },
            dst_subresource: ImageSubresourceLayers {
                layer_count: Some(min_array_layers),
                ..dst_image.subresource_layers()
            },
            extent: {
                let src_extent = src_image.extent();
                let dst_extent = dst_image.extent();

                [
                    src_extent[0].min(dst_extent[0]),
                    src_extent[1].min(dst_extent[1]),
                    src_extent[2].min(dst_extent[2]),
                ]
            },
            ..Default::default()
        };

        Self {
            src_image,
            src_image_layout: ImageLayout::TransferSrcOptimal,
            dst_image,
            dst_image_layout: ImageLayout::TransferDstOptimal,
            regions: smallvec![region],
            _ne: crate::NE,
        }
    }

    #[deprecated(since = "0.36.0", note = "use `new` instead")]
    #[inline]
    pub fn images(src_image: Arc<Image>, dst_image: Arc<Image>) -> Self {
        Self::new(src_image, dst_image)
    }
}

/// A region of data to copy between images.
#[derive(Clone, Debug)]
pub struct ImageCopy {
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

    pub _ne: crate::NonExhaustive<'static>,
}

impl Default for ImageCopy {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl ImageCopy {
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
}

/// Parameters to copy data from a buffer to an image.
#[derive(Clone, Debug)]
pub struct CopyBufferToImageInfo {
    /// The buffer to copy from.
    ///
    /// There is no default value.
    pub src_buffer: Subbuffer<[u8]>,

    /// The image to copy to.
    ///
    /// There is no default value.
    pub dst_image: Arc<Image>,

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
    pub regions: SmallVec<[BufferImageCopy; 1]>,

    pub _ne: crate::NonExhaustive<'static>,
}

impl CopyBufferToImageInfo {
    /// Returns a default `CopyBufferToImageInfo` with the provided `src_buffer` and
    /// `dst_image`.
    #[inline]
    pub fn new(src_buffer: Subbuffer<impl ?Sized>, dst_image: Arc<Image>) -> Self {
        let region = BufferImageCopy {
            image_subresource: dst_image.subresource_layers(),
            image_extent: dst_image.extent(),
            ..Default::default()
        };

        Self {
            src_buffer: src_buffer.into_bytes(),
            dst_image,
            dst_image_layout: ImageLayout::TransferDstOptimal,
            regions: smallvec![region],
            _ne: crate::NE,
        }
    }

    #[deprecated(since = "0.36.0", note = "use `new` instead")]
    #[inline]
    pub fn buffer_image(src_buffer: Subbuffer<impl ?Sized>, dst_image: Arc<Image>) -> Self {
        Self::new(src_buffer, dst_image)
    }
}

/// Parameters to copy data from an image to a buffer.
#[derive(Clone, Debug)]
pub struct CopyImageToBufferInfo {
    /// The image to copy from.
    ///
    /// There is no default value.
    pub src_image: Arc<Image>,

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
    pub dst_buffer: Subbuffer<[u8]>,

    /// The regions of the image and buffer to copy between.
    ///
    /// The default value is a single region, covering all of the buffer and the first mip level of
    /// the image. All aspects of the image are selected, or `plane0` if the image is multi-planar.
    pub regions: SmallVec<[BufferImageCopy; 1]>,

    pub _ne: crate::NonExhaustive<'static>,
}

impl CopyImageToBufferInfo {
    /// Returns a default `CopyImageToBufferInfo` with the provided `src_image` and
    /// `dst_buffer`.
    #[inline]
    pub fn new(src_image: Arc<Image>, dst_buffer: Subbuffer<impl ?Sized>) -> Self {
        let region = BufferImageCopy {
            image_subresource: src_image.subresource_layers(),
            image_extent: src_image.extent(),
            ..Default::default()
        };

        Self {
            src_image,
            src_image_layout: ImageLayout::TransferSrcOptimal,
            dst_buffer: dst_buffer.into_bytes(),
            regions: smallvec![region],
            _ne: crate::NE,
        }
    }

    #[deprecated(since = "0.36.0", note = "use `new` instead")]
    #[inline]
    pub fn image_buffer(src_image: Arc<Image>, dst_buffer: Subbuffer<impl ?Sized>) -> Self {
        Self::new(src_image, dst_buffer)
    }
}

/// A region of data to copy between a buffer and an image.
#[derive(Clone, Debug)]
pub struct BufferImageCopy {
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

    pub _ne: crate::NonExhaustive<'static>,
}

impl Default for BufferImageCopy {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl BufferImageCopy {
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
    pub(crate) fn buffer_copy_size(&self, image: &Image) -> DeviceSize {
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
        let block_extent = image.format().block_extent();
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
                .unwrap_or(image.array_layers() - image_subresource.base_array_layer),
        );

        let blocks_to_last_slice = (image_extent[2] as DeviceSize - 1)
            * buffer_image_height as DeviceSize
            * buffer_row_length as DeviceSize;
        let blocks_to_last_row =
            (image_extent[1] as DeviceSize - 1) * buffer_row_length as DeviceSize;
        let num_blocks = blocks_to_last_slice + blocks_to_last_row + image_extent[0] as DeviceSize;

        num_blocks * image.format().block_size()
    }
}

/// Parameters to blit image data.
#[derive(Clone, Debug)]
pub struct BlitImageInfo {
    /// The image to blit from.
    ///
    /// There is no default value.
    pub src_image: Arc<Image>,

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
    pub dst_image: Arc<Image>,

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
    pub regions: SmallVec<[ImageBlit; 1]>,

    /// The filter to use for sampling `src_image` when the `src_extent` and
    /// `dst_extent` of a region are not the same size.
    ///
    /// The default value is [`Filter::Nearest`].
    pub filter: Filter,

    pub _ne: crate::NonExhaustive<'static>,
}

impl BlitImageInfo {
    /// Returns a default `BlitImageInfo` with the provided `src_image` and `dst_image`.
    #[inline]
    pub fn new(src_image: Arc<Image>, dst_image: Arc<Image>) -> Self {
        let min_array_layers = src_image.array_layers().min(dst_image.array_layers());
        let region = ImageBlit {
            src_subresource: ImageSubresourceLayers {
                layer_count: Some(min_array_layers),
                ..src_image.subresource_layers()
            },
            src_offsets: [[0; 3], src_image.extent()],
            dst_subresource: ImageSubresourceLayers {
                layer_count: Some(min_array_layers),
                ..dst_image.subresource_layers()
            },
            dst_offsets: [[0; 3], dst_image.extent()],
            ..Default::default()
        };

        Self {
            src_image,
            src_image_layout: ImageLayout::TransferSrcOptimal,
            dst_image,
            dst_image_layout: ImageLayout::TransferDstOptimal,
            regions: smallvec![region],
            filter: Filter::Nearest,
            _ne: crate::NE,
        }
    }

    #[deprecated(since = "0.36.0", note = "use `new` instead")]
    #[inline]
    pub fn images(src_image: Arc<Image>, dst_image: Arc<Image>) -> Self {
        Self::new(src_image, dst_image)
    }
}

/// A region of data to blit between images.
#[derive(Clone, Debug)]
pub struct ImageBlit {
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

    pub _ne: crate::NonExhaustive<'static>,
}

impl Default for ImageBlit {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl ImageBlit {
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
}

/// Parameters to resolve image data.
#[derive(Clone, Debug)]
pub struct ResolveImageInfo {
    /// The multisampled image to resolve from.
    ///
    /// There is no default value.
    pub src_image: Arc<Image>,

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
    pub dst_image: Arc<Image>,

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
    pub regions: SmallVec<[ImageResolve; 1]>,

    pub _ne: crate::NonExhaustive<'static>,
}

impl ResolveImageInfo {
    /// Returns a default `ResolveImageInfo` with the provided `src_image` and `dst_image`.
    #[inline]
    pub fn new(src_image: Arc<Image>, dst_image: Arc<Image>) -> Self {
        let min_array_layers = src_image.array_layers().min(dst_image.array_layers());
        let region = ImageResolve {
            src_subresource: ImageSubresourceLayers {
                layer_count: Some(min_array_layers),
                ..src_image.subresource_layers()
            },
            dst_subresource: ImageSubresourceLayers {
                layer_count: Some(min_array_layers),
                ..dst_image.subresource_layers()
            },
            extent: {
                let src_extent = src_image.extent();
                let dst_extent = dst_image.extent();

                [
                    src_extent[0].min(dst_extent[0]),
                    src_extent[1].min(dst_extent[1]),
                    src_extent[2].min(dst_extent[2]),
                ]
            },
            ..Default::default()
        };

        Self {
            src_image,
            src_image_layout: ImageLayout::TransferSrcOptimal,
            dst_image,
            dst_image_layout: ImageLayout::TransferDstOptimal,
            regions: smallvec![region],
            _ne: crate::NE,
        }
    }

    #[deprecated(since = "0.36.0", note = "use `new` instead")]
    #[inline]
    pub fn images(src_image: Arc<Image>, dst_image: Arc<Image>) -> Self {
        Self::new(src_image, dst_image)
    }
}

/// A region of data to resolve between images.
#[derive(Clone, Debug)]
pub struct ImageResolve {
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

    pub _ne: crate::NonExhaustive<'static>,
}

impl Default for ImageResolve {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl ImageResolve {
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
}
