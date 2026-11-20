use crate::{
    command_buffer::{RecordingCommandBuffer, Result},
    resource::{AccessTypes, ImageLayoutType},
    Id,
};
use ash::vk;
use smallvec::SmallVec;
use std::cmp;
use vulkano::{
    buffer::Buffer,
    device::DeviceOwned,
    image::{sampler::Filter, Image, ImageSubresourceLayers},
    DeviceSize, Version, VulkanObject,
};

/// # Commands to transfer data between resources
impl RecordingCommandBuffer<'_> {
    /// Copies data from a buffer to another buffer, panicking on a validation error.
    ///
    /// This is a shortcut for `try_copy_buffer().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_copy_buffer`] returns a [`ValidationError`].
    ///
    /// [`try_copy_buffer`]: Self::try_copy_buffer
    #[track_caller]
    pub unsafe fn copy_buffer(&mut self, copy_buffer_info: &CopyBufferInfo<'_>) -> &mut Self {
        unsafe { self.try_copy_buffer(copy_buffer_info) }.unwrap()
    }

    /// Copies data from a buffer to another buffer.
    pub unsafe fn try_copy_buffer(
        &mut self,
        copy_buffer_info: &CopyBufferInfo<'_>,
    ) -> Result<&mut Self> {
        Ok(unsafe { self.copy_buffer_unchecked(copy_buffer_info) })
    }

    pub unsafe fn copy_buffer_unchecked(
        &mut self,
        copy_buffer_info: &CopyBufferInfo<'_>,
    ) -> &mut Self {
        let &CopyBufferInfo {
            src_buffer,
            dst_buffer,
            regions,
            _ne: _,
        } = copy_buffer_info;

        let src_buffer = unsafe { self.accesses.buffer_unchecked(src_buffer) };
        let dst_buffer = unsafe { self.accesses.buffer_unchecked(dst_buffer) };

        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3
            || self.device().enabled_extensions().khr_copy_commands2
        {
            let cmd_copy_buffer2 = if self.device().api_version() >= Version::V1_3 {
                fns.v1_3.cmd_copy_buffer2
            } else {
                fns.khr_copy_commands2.cmd_copy_buffer2_khr
            };

            if regions.is_empty() {
                let regions_vk = [vk::BufferCopy2::default()
                    .src_offset(0)
                    .dst_offset(0)
                    .size(cmp::min(src_buffer.size(), dst_buffer.size()))];

                let copy_buffer_info_vk = vk::CopyBufferInfo2::default()
                    .src_buffer(src_buffer.handle())
                    .dst_buffer(dst_buffer.handle())
                    .regions(&regions_vk);

                unsafe { cmd_copy_buffer2(self.handle(), &copy_buffer_info_vk) };
            } else {
                let regions_vk = regions
                    .iter()
                    .map(|region| {
                        let &BufferCopy {
                            src_offset,
                            dst_offset,
                            size,
                            _ne,
                        } = region;

                        vk::BufferCopy2::default()
                            .src_offset(src_offset)
                            .dst_offset(dst_offset)
                            .size(size)
                    })
                    .collect::<SmallVec<[_; 8]>>();

                let copy_buffer_info_vk = vk::CopyBufferInfo2::default()
                    .src_buffer(src_buffer.handle())
                    .dst_buffer(dst_buffer.handle())
                    .regions(&regions_vk);

                unsafe { cmd_copy_buffer2(self.handle(), &copy_buffer_info_vk) };
            }
        } else {
            let cmd_copy_buffer = fns.v1_0.cmd_copy_buffer;

            if regions.is_empty() {
                let region_vk = vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: cmp::min(src_buffer.size(), dst_buffer.size()),
                };

                unsafe {
                    cmd_copy_buffer(
                        self.handle(),
                        src_buffer.handle(),
                        dst_buffer.handle(),
                        1,
                        &region_vk,
                    )
                };
            } else {
                let regions_vk = regions
                    .iter()
                    .map(|copy| {
                        let &BufferCopy {
                            src_offset,
                            dst_offset,
                            size,
                            _ne: _,
                        } = copy;

                        vk::BufferCopy {
                            src_offset,
                            dst_offset,
                            size,
                        }
                    })
                    .collect::<SmallVec<[_; 8]>>();

                unsafe {
                    cmd_copy_buffer(
                        self.handle(),
                        src_buffer.handle(),
                        dst_buffer.handle(),
                        regions_vk.len() as u32,
                        regions_vk.as_ptr(),
                    )
                };
            }
        }

        self
    }

    /// Copies data from an image to another image, panicking on a validation error.
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
    /// This is a shortcut for `try_copy_image().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_copy_image`] returns a [`ValidationError`].
    ///
    /// [`try_copy_image`]: Self::try_copy_image
    #[track_caller]
    pub unsafe fn copy_image(&mut self, copy_image_info: &CopyImageInfo<'_>) -> &mut Self {
        unsafe { self.try_copy_image(copy_image_info) }.unwrap()
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
    pub unsafe fn try_copy_image(
        &mut self,
        copy_image_info: &CopyImageInfo<'_>,
    ) -> Result<&mut Self> {
        Ok(unsafe { self.copy_image_unchecked(copy_image_info) })
    }

    pub unsafe fn copy_image_unchecked(
        &mut self,
        copy_image_info: &CopyImageInfo<'_>,
    ) -> &mut Self {
        let &CopyImageInfo {
            src_image,
            src_image_layout,
            dst_image,
            dst_image_layout,
            regions,
            _ne: _,
        } = copy_image_info;

        let src_image = unsafe { self.accesses.image_unchecked(src_image) };
        let src_image_layout = AccessTypes::COPY_TRANSFER_READ.image_layout(src_image_layout);
        let dst_image = unsafe { self.accesses.image_unchecked(dst_image) };
        let dst_image_layout = AccessTypes::COPY_TRANSFER_WRITE.image_layout(dst_image_layout);

        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3
            || self.device().enabled_extensions().khr_copy_commands2
        {
            let cmd_copy_image2 = if self.device().api_version() >= Version::V1_3 {
                fns.v1_3.cmd_copy_image2
            } else {
                fns.khr_copy_commands2.cmd_copy_image2_khr
            };

            if regions.is_empty() {
                let remaining_array_layers =
                    cmp::min(src_image.array_layers(), dst_image.array_layers());
                let src_extent = src_image.extent();
                let dst_extent = dst_image.extent();
                let regions_vk = [vk::ImageCopy2::default()
                    .src_subresource(src_image.subresource_layers().to_vk(remaining_array_layers))
                    .src_offset(convert_offset([0; 3]))
                    .dst_subresource(dst_image.subresource_layers().to_vk(remaining_array_layers))
                    .dst_offset(convert_offset([0; 3]))
                    .extent(convert_extent([
                        cmp::min(src_extent[0], dst_extent[0]),
                        cmp::min(src_extent[1], dst_extent[1]),
                        cmp::min(src_extent[2], dst_extent[2]),
                    ]))];

                let copy_image_info_vk = vk::CopyImageInfo2::default()
                    .src_image(src_image.handle())
                    .src_image_layout(src_image_layout.into())
                    .dst_image(dst_image.handle())
                    .dst_image_layout(dst_image_layout.into())
                    .regions(&regions_vk);

                unsafe { cmd_copy_image2(self.handle(), &copy_image_info_vk) };
            } else {
                let regions_vk = regions
                    .iter()
                    .map(|region| {
                        let &ImageCopy {
                            src_subresource,
                            src_offset,
                            dst_subresource,
                            dst_offset,
                            extent,
                            _ne: _,
                        } = region;

                        let remaining_array_layers = cmp::min(
                            src_subresource.layer_count.unwrap_or(
                                src_image.array_layers() - src_subresource.base_array_layer,
                            ),
                            dst_subresource.layer_count.unwrap_or(
                                dst_image.array_layers() - dst_subresource.base_array_layer,
                            ),
                        );

                        vk::ImageCopy2::default()
                            .src_subresource(src_subresource.to_vk(remaining_array_layers))
                            .src_offset(convert_offset(src_offset))
                            .dst_subresource(dst_subresource.to_vk(remaining_array_layers))
                            .dst_offset(convert_offset(dst_offset))
                            .extent(convert_extent(extent))
                    })
                    .collect::<SmallVec<[_; 8]>>();

                let copy_image_info_vk = vk::CopyImageInfo2::default()
                    .src_image(src_image.handle())
                    .src_image_layout(src_image_layout.into())
                    .dst_image(dst_image.handle())
                    .dst_image_layout(dst_image_layout.into())
                    .regions(&regions_vk);

                unsafe { cmd_copy_image2(self.handle(), &copy_image_info_vk) };
            }
        } else {
            let cmd_copy_image = fns.v1_0.cmd_copy_image;

            if regions.is_empty() {
                let remaining_array_layers =
                    cmp::min(src_image.array_layers(), dst_image.array_layers());
                let src_extent = src_image.extent();
                let dst_extent = dst_image.extent();
                let region_vk = vk::ImageCopy {
                    src_subresource: src_image.subresource_layers().to_vk(remaining_array_layers),
                    src_offset: convert_offset([0; 3]),
                    dst_subresource: dst_image.subresource_layers().to_vk(remaining_array_layers),
                    dst_offset: convert_offset([0; 3]),
                    extent: convert_extent([
                        cmp::min(src_extent[0], dst_extent[0]),
                        cmp::min(src_extent[1], dst_extent[1]),
                        cmp::min(src_extent[2], dst_extent[2]),
                    ]),
                };

                unsafe {
                    cmd_copy_image(
                        self.handle(),
                        src_image.handle(),
                        src_image_layout.into(),
                        dst_image.handle(),
                        dst_image_layout.into(),
                        1,
                        &region_vk,
                    )
                };
            } else {
                let regions_vk: SmallVec<[_; 8]> = regions
                    .iter()
                    .map(|region| {
                        let &ImageCopy {
                            src_subresource,
                            src_offset,
                            dst_subresource,
                            dst_offset,
                            extent,
                            _ne: _,
                        } = region;

                        let remaining_array_layers = cmp::min(
                            src_subresource.layer_count.unwrap_or(
                                src_image.array_layers() - src_subresource.base_array_layer,
                            ),
                            dst_subresource.layer_count.unwrap_or(
                                dst_image.array_layers() - dst_subresource.base_array_layer,
                            ),
                        );

                        vk::ImageCopy {
                            src_subresource: src_subresource.to_vk(remaining_array_layers),
                            src_offset: convert_offset(src_offset),
                            dst_subresource: dst_subresource.to_vk(remaining_array_layers),
                            dst_offset: convert_offset(dst_offset),
                            extent: convert_extent(extent),
                        }
                    })
                    .collect();

                unsafe {
                    cmd_copy_image(
                        self.handle(),
                        src_image.handle(),
                        src_image_layout.into(),
                        dst_image.handle(),
                        dst_image_layout.into(),
                        regions_vk.len() as u32,
                        regions_vk.as_ptr(),
                    )
                };
            }
        }

        self
    }

    /// Copies from a buffer to an image, panicking on a validation error.
    ///
    /// This is a shortcut for `try_copy_buffer_to_image().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_copy_buffer_to_image`] returns a [`ValidationError`].
    ///
    /// [`try_copy_buffer_to_image`]: Self::try_copy_buffer_to_image
    #[track_caller]
    pub unsafe fn copy_buffer_to_image(
        &mut self,
        copy_buffer_to_image_info: &CopyBufferToImageInfo<'_>,
    ) -> &mut Self {
        unsafe { self.try_copy_buffer_to_image(copy_buffer_to_image_info) }.unwrap()
    }

    /// Copies from a buffer to an image.
    pub unsafe fn try_copy_buffer_to_image(
        &mut self,
        copy_buffer_to_image_info: &CopyBufferToImageInfo<'_>,
    ) -> Result<&mut Self> {
        Ok(unsafe { self.copy_buffer_to_image_unchecked(copy_buffer_to_image_info) })
    }

    pub unsafe fn copy_buffer_to_image_unchecked(
        &mut self,
        copy_buffer_to_image_info: &CopyBufferToImageInfo<'_>,
    ) -> &mut Self {
        let &CopyBufferToImageInfo {
            src_buffer,
            dst_image,
            dst_image_layout,
            regions,
            _ne: _,
        } = copy_buffer_to_image_info;

        let src_buffer = unsafe { self.accesses.buffer_unchecked(src_buffer) };
        let dst_image = unsafe { self.accesses.image_unchecked(dst_image) };
        let dst_image_layout = AccessTypes::COPY_TRANSFER_WRITE.image_layout(dst_image_layout);

        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3
            || self.device().enabled_extensions().khr_copy_commands2
        {
            let cmd_copy_buffer_to_image2 = if self.device().api_version() >= Version::V1_3 {
                fns.v1_3.cmd_copy_buffer_to_image2
            } else {
                fns.khr_copy_commands2.cmd_copy_buffer_to_image2_khr
            };

            if regions.is_empty() {
                let remaining_array_layers = dst_image.array_layers();
                let regions_vk = [vk::BufferImageCopy2::default()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(dst_image.subresource_layers().to_vk(remaining_array_layers))
                    .image_offset(convert_offset([0; 3]))
                    .image_extent(convert_extent(dst_image.extent()))];

                let copy_buffer_to_image_info_vk = vk::CopyBufferToImageInfo2::default()
                    .src_buffer(src_buffer.handle())
                    .dst_image(dst_image.handle())
                    .dst_image_layout(dst_image_layout.into())
                    .regions(&regions_vk);

                unsafe { cmd_copy_buffer_to_image2(self.handle(), &copy_buffer_to_image_info_vk) };
            } else {
                let regions_vk = regions
                    .iter()
                    .map(|region| {
                        let &BufferImageCopy {
                            buffer_offset,
                            buffer_row_length,
                            buffer_image_height,
                            image_subresource,
                            image_offset,
                            image_extent,
                            _ne: _,
                        } = region;

                        let remaining_array_layers =
                            dst_image.array_layers() - image_subresource.base_array_layer;

                        vk::BufferImageCopy2::default()
                            .buffer_offset(buffer_offset)
                            .buffer_row_length(buffer_row_length)
                            .buffer_image_height(buffer_image_height)
                            .image_subresource(image_subresource.to_vk(remaining_array_layers))
                            .image_offset(convert_offset(image_offset))
                            .image_extent(convert_extent(image_extent))
                    })
                    .collect::<SmallVec<[_; 8]>>();

                let copy_buffer_to_image_info_vk = vk::CopyBufferToImageInfo2::default()
                    .src_buffer(src_buffer.handle())
                    .dst_image(dst_image.handle())
                    .dst_image_layout(dst_image_layout.into())
                    .regions(&regions_vk);

                unsafe { cmd_copy_buffer_to_image2(self.handle(), &copy_buffer_to_image_info_vk) };
            }
        } else {
            let cmd_copy_buffer_to_image = fns.v1_0.cmd_copy_buffer_to_image;

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

                unsafe {
                    cmd_copy_buffer_to_image(
                        self.handle(),
                        src_buffer.handle(),
                        dst_image.handle(),
                        dst_image_layout.into(),
                        1,
                        &region_vk,
                    )
                };
            } else {
                let regions_vk = regions
                    .iter()
                    .map(|region| {
                        let &BufferImageCopy {
                            buffer_offset,
                            buffer_row_length,
                            buffer_image_height,
                            image_subresource,
                            image_offset,
                            image_extent,
                            _ne: _,
                        } = region;

                        let remaining_array_layers =
                            dst_image.array_layers() - image_subresource.base_array_layer;

                        vk::BufferImageCopy {
                            buffer_offset,
                            buffer_row_length,
                            buffer_image_height,
                            image_subresource: image_subresource.to_vk(remaining_array_layers),
                            image_offset: convert_offset(image_offset),
                            image_extent: convert_extent(image_extent),
                        }
                    })
                    .collect::<SmallVec<[_; 8]>>();

                unsafe {
                    cmd_copy_buffer_to_image(
                        self.handle(),
                        src_buffer.handle(),
                        dst_image.handle(),
                        dst_image_layout.into(),
                        regions_vk.len() as u32,
                        regions_vk.as_ptr(),
                    )
                };
            }
        }

        self
    }

    /// Copies from an image to a buffer, panicking on a validation error.
    ///
    /// This is a shortcut for `try_copy_image_to_buffer().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_copy_image_to_buffer`] returns a [`ValidationError`].
    ///
    /// [`try_copy_image_to_buffer`]: Self::try_copy_image_to_buffer
    #[track_caller]
    pub unsafe fn copy_image_to_buffer(
        &mut self,
        copy_image_to_buffer_info: &CopyImageToBufferInfo<'_>,
    ) -> &mut Self {
        unsafe { self.try_copy_image_to_buffer(copy_image_to_buffer_info) }.unwrap()
    }

    /// Copies from an image to a buffer.
    pub unsafe fn try_copy_image_to_buffer(
        &mut self,
        copy_image_to_buffer_info: &CopyImageToBufferInfo<'_>,
    ) -> Result<&mut Self> {
        Ok(unsafe { self.copy_image_to_buffer_unchecked(copy_image_to_buffer_info) })
    }

    pub unsafe fn copy_image_to_buffer_unchecked(
        &mut self,
        copy_image_to_buffer_info: &CopyImageToBufferInfo<'_>,
    ) -> &mut Self {
        let &CopyImageToBufferInfo {
            src_image,
            src_image_layout,
            dst_buffer,
            regions,
            _ne: _,
        } = copy_image_to_buffer_info;

        let src_image = unsafe { self.accesses.image_unchecked(src_image) };
        let src_image_layout = AccessTypes::COPY_TRANSFER_READ.image_layout(src_image_layout);
        let dst_buffer = unsafe { self.accesses.buffer_unchecked(dst_buffer) };

        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3
            || self.device().enabled_extensions().khr_copy_commands2
        {
            let cmd_copy_image_to_buffer2 = if self.device().api_version() >= Version::V1_3 {
                fns.v1_3.cmd_copy_image_to_buffer2
            } else {
                fns.khr_copy_commands2.cmd_copy_image_to_buffer2_khr
            };

            if regions.is_empty() {
                let remaining_array_layers = src_image.array_layers();
                let regions_vk = [vk::BufferImageCopy2::default()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(src_image.subresource_layers().to_vk(remaining_array_layers))
                    .image_offset(convert_offset([0; 3]))
                    .image_extent(convert_extent(src_image.extent()))];

                let copy_image_to_buffer_info_vk = vk::CopyImageToBufferInfo2::default()
                    .src_image(src_image.handle())
                    .src_image_layout(src_image_layout.into())
                    .dst_buffer(dst_buffer.handle())
                    .regions(&regions_vk);

                unsafe { cmd_copy_image_to_buffer2(self.handle(), &copy_image_to_buffer_info_vk) };
            } else {
                let regions_vk = regions
                    .iter()
                    .map(|region| {
                        let &BufferImageCopy {
                            buffer_offset,
                            buffer_row_length,
                            buffer_image_height,
                            image_subresource,
                            image_offset,
                            image_extent,
                            _ne: _,
                        } = region;

                        let remaining_array_layers =
                            src_image.array_layers() - image_subresource.base_array_layer;

                        vk::BufferImageCopy2::default()
                            .buffer_offset(buffer_offset)
                            .buffer_row_length(buffer_row_length)
                            .buffer_image_height(buffer_image_height)
                            .image_subresource(image_subresource.to_vk(remaining_array_layers))
                            .image_offset(convert_offset(image_offset))
                            .image_extent(convert_extent(image_extent))
                    })
                    .collect::<SmallVec<[_; 8]>>();

                let copy_image_to_buffer_info_vk = vk::CopyImageToBufferInfo2::default()
                    .src_image(src_image.handle())
                    .src_image_layout(src_image_layout.into())
                    .dst_buffer(dst_buffer.handle())
                    .regions(&regions_vk);

                unsafe { cmd_copy_image_to_buffer2(self.handle(), &copy_image_to_buffer_info_vk) };
            }
        } else {
            let cmd_copy_image_to_buffer = fns.v1_0.cmd_copy_image_to_buffer;

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

                unsafe {
                    cmd_copy_image_to_buffer(
                        self.handle(),
                        src_image.handle(),
                        src_image_layout.into(),
                        dst_buffer.handle(),
                        1,
                        &region_vk,
                    )
                };
            } else {
                let regions_vk = regions
                    .iter()
                    .map(|region| {
                        let &BufferImageCopy {
                            buffer_offset,
                            buffer_row_length,
                            buffer_image_height,
                            image_subresource,
                            image_offset,
                            image_extent,
                            _ne: _,
                        } = region;

                        let remaining_array_layers =
                            src_image.array_layers() - image_subresource.base_array_layer;

                        vk::BufferImageCopy {
                            buffer_offset,
                            buffer_row_length,
                            buffer_image_height,
                            image_subresource: image_subresource.to_vk(remaining_array_layers),
                            image_offset: convert_offset(image_offset),
                            image_extent: convert_extent(image_extent),
                        }
                    })
                    .collect::<SmallVec<[_; 8]>>();

                unsafe {
                    cmd_copy_image_to_buffer(
                        self.handle(),
                        src_image.handle(),
                        src_image_layout.into(),
                        dst_buffer.handle(),
                        regions_vk.len() as u32,
                        regions_vk.as_ptr(),
                    )
                };
            }
        }

        self
    }

    /// Blits an image to another, panicking on a validation error.
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
    /// This is a shortcut for `try_blit_image().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_blit_image`] returns a [`ValidationError`].
    ///
    /// [`try_blit_image`]: Self::try_blit_image
    #[track_caller]
    pub unsafe fn blit_image(&mut self, blit_image_info: &BlitImageInfo<'_>) -> &mut Self {
        unsafe { self.try_blit_image(blit_image_info) }.unwrap()
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
    pub unsafe fn try_blit_image(
        &mut self,
        blit_image_info: &BlitImageInfo<'_>,
    ) -> Result<&mut Self> {
        Ok(unsafe { self.blit_image_unchecked(blit_image_info) })
    }

    pub unsafe fn blit_image_unchecked(
        &mut self,
        blit_image_info: &BlitImageInfo<'_>,
    ) -> &mut Self {
        let &BlitImageInfo {
            src_image,
            src_image_layout,
            dst_image,
            dst_image_layout,
            regions,
            filter,
            _ne: _,
        } = blit_image_info;

        let src_image = unsafe { self.accesses.image_unchecked(src_image) };
        let src_image_layout = AccessTypes::BLIT_TRANSFER_READ.image_layout(src_image_layout);
        let dst_image = unsafe { self.accesses.image_unchecked(dst_image) };
        let dst_image_layout = AccessTypes::BLIT_TRANSFER_WRITE.image_layout(dst_image_layout);

        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3
            || self.device().enabled_extensions().khr_copy_commands2
        {
            let cmd_blit_image2 = if self.device().api_version() >= Version::V1_3 {
                fns.v1_3.cmd_blit_image2
            } else {
                fns.khr_copy_commands2.cmd_blit_image2_khr
            };

            if regions.is_empty() {
                let remaining_array_layers =
                    cmp::min(src_image.array_layers(), dst_image.array_layers());
                let regions_vk = [vk::ImageBlit2::default()
                    .src_subresource(src_image.subresource_layers().to_vk(remaining_array_layers))
                    .src_offsets([[0; 3], src_image.extent()].map(convert_offset))
                    .dst_subresource(dst_image.subresource_layers().to_vk(remaining_array_layers))
                    .dst_offsets([[0; 3], dst_image.extent()].map(convert_offset))];

                let blit_image_info_vk = vk::BlitImageInfo2::default()
                    .src_image(src_image.handle())
                    .src_image_layout(src_image_layout.into())
                    .dst_image(dst_image.handle())
                    .dst_image_layout(dst_image_layout.into())
                    .regions(&regions_vk)
                    .filter(filter.into());

                unsafe { cmd_blit_image2(self.handle(), &blit_image_info_vk) };
            } else {
                let regions_vk = regions
                    .iter()
                    .map(|region| {
                        let &ImageBlit {
                            src_subresource,
                            src_offsets,
                            dst_subresource,
                            dst_offsets,
                            _ne: _,
                        } = region;

                        let remaining_array_layers = cmp::min(
                            src_subresource.layer_count.unwrap_or(
                                src_image.array_layers() - src_subresource.base_array_layer,
                            ),
                            dst_subresource.layer_count.unwrap_or(
                                dst_image.array_layers() - dst_subresource.base_array_layer,
                            ),
                        );

                        vk::ImageBlit2::default()
                            .src_subresource(src_subresource.to_vk(remaining_array_layers))
                            .src_offsets(src_offsets.map(convert_offset))
                            .dst_subresource(dst_subresource.to_vk(remaining_array_layers))
                            .dst_offsets(dst_offsets.map(convert_offset))
                    })
                    .collect::<SmallVec<[_; 8]>>();

                let blit_image_info_vk = vk::BlitImageInfo2::default()
                    .src_image(src_image.handle())
                    .src_image_layout(src_image_layout.into())
                    .dst_image(dst_image.handle())
                    .dst_image_layout(dst_image_layout.into())
                    .regions(&regions_vk)
                    .filter(filter.into());

                unsafe { cmd_blit_image2(self.handle(), &blit_image_info_vk) };
            }
        } else {
            let cmd_blit_image = fns.v1_0.cmd_blit_image;

            if regions.is_empty() {
                let remaining_array_layers =
                    cmp::min(src_image.array_layers(), dst_image.array_layers());
                let region_vk = vk::ImageBlit {
                    src_subresource: src_image.subresource_layers().to_vk(remaining_array_layers),
                    src_offsets: [[0; 3], src_image.extent()].map(convert_offset),
                    dst_subresource: dst_image.subresource_layers().to_vk(remaining_array_layers),
                    dst_offsets: [[0; 3], dst_image.extent()].map(convert_offset),
                };

                unsafe {
                    cmd_blit_image(
                        self.handle(),
                        src_image.handle(),
                        src_image_layout.into(),
                        dst_image.handle(),
                        dst_image_layout.into(),
                        1,
                        &region_vk,
                        filter.into(),
                    )
                };
            } else {
                let regions_vk = regions
                    .iter()
                    .map(|region| {
                        let &ImageBlit {
                            src_subresource,
                            src_offsets,
                            dst_subresource,
                            dst_offsets,
                            _ne: _,
                        } = region;

                        let remaining_array_layers = cmp::min(
                            src_subresource.layer_count.unwrap_or(
                                src_image.array_layers() - src_subresource.base_array_layer,
                            ),
                            dst_subresource.layer_count.unwrap_or(
                                dst_image.array_layers() - dst_subresource.base_array_layer,
                            ),
                        );

                        vk::ImageBlit {
                            src_subresource: src_subresource.to_vk(remaining_array_layers),
                            src_offsets: src_offsets.map(convert_offset),
                            dst_subresource: dst_subresource.to_vk(remaining_array_layers),
                            dst_offsets: dst_offsets.map(convert_offset),
                        }
                    })
                    .collect::<SmallVec<[_; 8]>>();

                unsafe {
                    cmd_blit_image(
                        self.handle(),
                        src_image.handle(),
                        src_image_layout.into(),
                        dst_image.handle(),
                        dst_image_layout.into(),
                        regions_vk.len() as u32,
                        regions_vk.as_ptr(),
                        filter.into(),
                    )
                };
            }
        }

        self
    }

    /// Resolves a multisampled image into a single-sampled image, panicking on a validation error.
    ///
    /// This is a shortcut for `try_resolve_image().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_resolve_image`] returns a [`ValidationError`].
    ///
    /// [`try_resolve_image`]: Self::try_resolve_image
    #[track_caller]
    pub unsafe fn resolve_image(&mut self, resolve_image_info: &ResolveImageInfo<'_>) -> &mut Self {
        unsafe { self.try_resolve_image(resolve_image_info) }.unwrap()
    }

    /// Resolves a multisampled image into a single-sampled image.
    pub unsafe fn try_resolve_image(
        &mut self,
        resolve_image_info: &ResolveImageInfo<'_>,
    ) -> Result<&mut Self> {
        Ok(unsafe { self.resolve_image_unchecked(resolve_image_info) })
    }

    pub unsafe fn resolve_image_unchecked(
        &mut self,
        resolve_image_info: &ResolveImageInfo<'_>,
    ) -> &mut Self {
        let &ResolveImageInfo {
            src_image,
            src_image_layout,
            dst_image,
            dst_image_layout,
            regions,
            _ne: _,
        } = resolve_image_info;

        let src_image = unsafe { self.accesses.image_unchecked(src_image) };
        let src_image_layout = AccessTypes::RESOLVE_TRANSFER_READ.image_layout(src_image_layout);
        let dst_image = unsafe { self.accesses.image_unchecked(dst_image) };
        let dst_image_layout = AccessTypes::RESOLVE_TRANSFER_WRITE.image_layout(dst_image_layout);

        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3
            || self.device().enabled_extensions().khr_copy_commands2
        {
            let cmd_resolve_image2 = if self.device().api_version() >= Version::V1_3 {
                fns.v1_3.cmd_resolve_image2
            } else {
                fns.khr_copy_commands2.cmd_resolve_image2_khr
            };

            if regions.is_empty() {
                let remaining_array_layers =
                    cmp::min(src_image.array_layers(), dst_image.array_layers());
                let src_extent = src_image.extent();
                let dst_extent = dst_image.extent();
                let regions_vk = [vk::ImageResolve2::default()
                    .src_subresource(src_image.subresource_layers().to_vk(remaining_array_layers))
                    .src_offset(convert_offset([0; 3]))
                    .dst_subresource(dst_image.subresource_layers().to_vk(remaining_array_layers))
                    .dst_offset(convert_offset([0; 3]))
                    .extent(convert_extent([
                        cmp::min(src_extent[0], dst_extent[0]),
                        cmp::min(src_extent[1], dst_extent[1]),
                        cmp::min(src_extent[2], dst_extent[2]),
                    ]))];

                let resolve_image_info_vk = vk::ResolveImageInfo2::default()
                    .src_image(src_image.handle())
                    .src_image_layout(src_image_layout.into())
                    .dst_image(dst_image.handle())
                    .dst_image_layout(dst_image_layout.into())
                    .regions(&regions_vk);

                unsafe { cmd_resolve_image2(self.handle(), &resolve_image_info_vk) };
            } else {
                let regions_vk = regions
                    .iter()
                    .map(|region| {
                        let &ImageResolve {
                            src_subresource,
                            src_offset,
                            dst_subresource,
                            dst_offset,
                            extent,
                            _ne: _,
                        } = region;

                        let remaining_array_layers = cmp::min(
                            src_subresource.layer_count.unwrap_or(
                                src_image.array_layers() - src_subresource.base_array_layer,
                            ),
                            dst_subresource.layer_count.unwrap_or(
                                dst_image.array_layers() - dst_subresource.base_array_layer,
                            ),
                        );

                        vk::ImageResolve2::default()
                            .src_subresource(src_subresource.to_vk(remaining_array_layers))
                            .src_offset(convert_offset(src_offset))
                            .dst_subresource(dst_subresource.to_vk(remaining_array_layers))
                            .dst_offset(convert_offset(dst_offset))
                            .extent(convert_extent(extent))
                    })
                    .collect::<SmallVec<[_; 8]>>();

                let resolve_image_info_vk = vk::ResolveImageInfo2::default()
                    .src_image(src_image.handle())
                    .src_image_layout(src_image_layout.into())
                    .dst_image(dst_image.handle())
                    .dst_image_layout(dst_image_layout.into())
                    .regions(&regions_vk);

                unsafe { cmd_resolve_image2(self.handle(), &resolve_image_info_vk) };
            }
        } else {
            let cmd_resolve_image = fns.v1_0.cmd_resolve_image;

            if regions.is_empty() {
                let remaining_array_layers =
                    cmp::min(src_image.array_layers(), dst_image.array_layers());
                let src_extent = src_image.extent();
                let dst_extent = dst_image.extent();
                let regions_vk = [vk::ImageResolve {
                    src_subresource: src_image.subresource_layers().to_vk(remaining_array_layers),
                    src_offset: convert_offset([0; 3]),
                    dst_subresource: dst_image.subresource_layers().to_vk(remaining_array_layers),
                    dst_offset: convert_offset([0; 3]),
                    extent: convert_extent([
                        cmp::min(src_extent[0], dst_extent[0]),
                        cmp::min(src_extent[1], dst_extent[1]),
                        cmp::min(src_extent[2], dst_extent[2]),
                    ]),
                }];

                unsafe {
                    cmd_resolve_image(
                        self.handle(),
                        src_image.handle(),
                        src_image_layout.into(),
                        dst_image.handle(),
                        dst_image_layout.into(),
                        regions_vk.len() as u32,
                        regions_vk.as_ptr(),
                    )
                };
            } else {
                let regions_vk = regions
                    .iter()
                    .map(|region| {
                        let &ImageResolve {
                            src_subresource,
                            src_offset,
                            dst_subresource,
                            dst_offset,
                            extent,
                            _ne: _,
                        } = region;

                        let remaining_array_layers = cmp::min(
                            src_subresource.layer_count.unwrap_or(
                                src_image.array_layers() - src_subresource.base_array_layer,
                            ),
                            dst_subresource.layer_count.unwrap_or(
                                dst_image.array_layers() - dst_subresource.base_array_layer,
                            ),
                        );

                        vk::ImageResolve {
                            src_subresource: src_subresource.to_vk(remaining_array_layers),
                            src_offset: convert_offset(src_offset),
                            dst_subresource: dst_subresource.to_vk(remaining_array_layers),
                            dst_offset: convert_offset(dst_offset),
                            extent: convert_extent(extent),
                        }
                    })
                    .collect::<SmallVec<[_; 8]>>();

                unsafe {
                    cmd_resolve_image(
                        self.handle(),
                        src_image.handle(),
                        src_image_layout.into(),
                        dst_image.handle(),
                        dst_image_layout.into(),
                        regions_vk.len() as u32,
                        regions_vk.as_ptr(),
                    )
                };
            }
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
    /// The default value is [`Id::INVALID`], which must be overridden.
    pub src_buffer: Id<Buffer>,

    /// The buffer to copy to.
    ///
    /// The default value is [`Id::INVALID`], which must be overridden.
    pub dst_buffer: Id<Buffer>,

    /// The regions of both buffers to copy between, specified in bytes.
    ///
    /// The default value is a single region, with zero offsets and a `size` equal to the smallest
    /// of the two buffers.
    pub regions: &'a [BufferCopy<'a>],

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for CopyBufferInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl CopyBufferInfo<'_> {
    /// Returns a default `CopyBufferInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            src_buffer: Id::INVALID,
            dst_buffer: Id::INVALID,
            regions: &[],
            _ne: crate::NE,
        }
    }
}

/// A region of data to copy between buffers.
#[derive(Clone, Debug)]
pub struct BufferCopy<'a> {
    /// The offset in bytes or elements from the start of `src_buffer` that copying will start
    /// from.
    ///
    /// The default value is `0`.
    pub src_offset: DeviceSize,

    /// The offset in bytes or elements from the start of `dst_buffer` that copying will start
    /// from.
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
}

/// Parameters to copy data from an image to another image.
#[derive(Clone, Debug)]
pub struct CopyImageInfo<'a> {
    /// The image to copy from.
    ///
    /// The default value is [`Id::INVALID`], which must be overridden.
    pub src_image: Id<Image>,

    /// The layout used for `src_image` during the copy operation.
    ///
    /// The default value is [`ImageLayoutType::Optimal`].
    pub src_image_layout: ImageLayoutType,

    /// The image to copy to.
    ///
    /// The default value is [`Id::INVALID`], which must be overridden.
    pub dst_image: Id<Image>,

    /// The layout used for `dst_image` during the copy operation.
    ///
    /// The default value is [`ImageLayoutType::Optimal`].
    pub dst_image_layout: ImageLayoutType,

    /// The regions of both images to copy between.
    ///
    /// The default value is a single region, covering the first mip level, and the smallest of the
    /// array layers and extent of the two images. All aspects of each image are selected, or
    /// `plane0` if the image is multi-planar.
    pub regions: &'a [ImageCopy<'a>],

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for CopyImageInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl CopyImageInfo<'_> {
    /// Returns a default `CopyImageInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            src_image: Id::INVALID,
            src_image_layout: ImageLayoutType::Optimal,
            dst_image: Id::INVALID,
            dst_image_layout: ImageLayoutType::Optimal,
            regions: &[],
            _ne: crate::NE,
        }
    }
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
}

/// Parameters to copy data from a buffer to an image.
#[derive(Clone, Debug)]
pub struct CopyBufferToImageInfo<'a> {
    /// The buffer to copy from.
    ///
    /// The default value is [`Id::INVALID`], which must be overridden.
    pub src_buffer: Id<Buffer>,

    /// The image to copy to.
    ///
    /// The default value is [`Id::INVALID`], which must be overridden.
    pub dst_image: Id<Image>,

    /// The layout used for `dst_image` during the copy operation.
    ///
    /// The default value is [`ImageLayoutType::Optimal`].
    pub dst_image_layout: ImageLayoutType,

    /// The regions of the buffer and image to copy between.
    ///
    /// The default value is a single region, covering all of the buffer and the first mip level of
    /// the image. All aspects of the image are selected, or `plane0` if the image is multi-planar.
    pub regions: &'a [BufferImageCopy<'a>],

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for CopyBufferToImageInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl CopyBufferToImageInfo<'_> {
    /// Returns a default `CopyBufferToImageInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            src_buffer: Id::INVALID,
            dst_image: Id::INVALID,
            dst_image_layout: ImageLayoutType::Optimal,
            regions: &[],
            _ne: crate::NE,
        }
    }
}

/// Parameters to copy data from an image to a buffer.
#[derive(Clone, Debug)]
pub struct CopyImageToBufferInfo<'a> {
    /// The image to copy from.
    ///
    /// The default value is [`Id::INVALID`], which must be overridden.
    pub src_image: Id<Image>,

    /// The layout used for `src_image` during the copy operation.
    ///
    /// The default value is [`ImageLayoutType::Optimal`].
    pub src_image_layout: ImageLayoutType,

    /// The buffer to copy to.
    ///
    /// The default value is [`Id::INVALID`], which must be overridden.
    pub dst_buffer: Id<Buffer>,

    /// The regions of the image and buffer to copy between.
    ///
    /// The default value is a single region, covering all of the buffer and the first mip level of
    /// the image. All aspects of the image are selected, or `plane0` if the image is multi-planar.
    pub regions: &'a [BufferImageCopy<'a>],

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for CopyImageToBufferInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl CopyImageToBufferInfo<'_> {
    /// Returns a default `CopyImageToBufferInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            src_image: Id::INVALID,
            src_image_layout: ImageLayoutType::Optimal,
            dst_buffer: Id::INVALID,
            regions: &[],
            _ne: crate::NE,
        }
    }
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
}

/// Parameters to blit image data.
#[derive(Clone, Debug)]
pub struct BlitImageInfo<'a> {
    /// The image to blit from.
    ///
    /// The default value is [`Id::INVALID`], which must be overridden.
    pub src_image: Id<Image>,

    /// The layout used for `src_image` during the blit operation.
    ///
    /// The default value is [`ImageLayoutType::Optimal`].
    pub src_image_layout: ImageLayoutType,

    /// The image to blit to.
    ///
    /// The default value is [`Id::INVALID`], which must be overridden.
    pub dst_image: Id<Image>,

    /// The layout used for `dst_image` during the blit operation.
    ///
    /// The default value is [`ImageLayoutType::Optimal`].
    pub dst_image_layout: ImageLayoutType,

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

impl Default for BlitImageInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl BlitImageInfo<'_> {
    /// Returns a default `BlitImageInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            src_image: Id::INVALID,
            src_image_layout: ImageLayoutType::Optimal,
            dst_image: Id::INVALID,
            dst_image_layout: ImageLayoutType::Optimal,
            regions: &[],
            filter: Filter::Nearest,
            _ne: crate::NE,
        }
    }
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
}

/// Parameters to resolve image data.
#[derive(Clone, Debug)]
pub struct ResolveImageInfo<'a> {
    /// The multisampled image to resolve from.
    ///
    /// The default value is [`Id::INVALID`], which must be overridden.
    pub src_image: Id<Image>,

    /// The layout used for `src_image` during the resolve operation.
    ///
    /// The default value is [`ImageLayoutType::Optimal`].
    pub src_image_layout: ImageLayoutType,

    /// The non-multisampled image to resolve into.
    ///
    /// The default value is [`Id::INVALID`], which must be overridden.
    pub dst_image: Id<Image>,

    /// The layout used for `dst_image` during the resolve operation.
    ///
    /// The default value is [`ImageLayoutType::Optimal`].
    pub dst_image_layout: ImageLayoutType,

    /// The regions of both images to resolve between.
    ///
    /// The default value is a single region, covering the first mip level, and the smallest of the
    /// array layers and extent of the two images. All aspects of each image are selected, or
    /// `plane0` if the image is multi-planar.
    pub regions: &'a [ImageResolve<'a>],

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for ResolveImageInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl ResolveImageInfo<'_> {
    /// Returns a default `ResolveImageInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            src_image: Id::INVALID,
            src_image_layout: ImageLayoutType::Optimal,
            dst_image: Id::INVALID,
            dst_image_layout: ImageLayoutType::Optimal,
            regions: &[],
            _ne: crate::NE,
        }
    }
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
