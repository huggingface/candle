use crate::{
    command_buffer::{RecordingCommandBuffer, Result},
    resource::{AccessTypes, ImageLayoutType},
    Id,
};
use ash::vk;
use smallvec::SmallVec;
use std::ffi::c_void;
use vulkano::{
    buffer::{Buffer, BufferContents},
    device::DeviceOwned,
    format::{ClearColorValue, ClearDepthStencilValue},
    image::{Image, ImageSubresourceRange},
    DeviceSize, VulkanObject,
};

/// # Commands to fill resources with new data
impl RecordingCommandBuffer<'_> {
    /// Clears a color image with a specific value, panicking on a validation error.
    ///
    /// This is a shortcut for `try_clear_color_image().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_clear_color_image`] returns a [`ValidationError`].
    ///
    /// [`try_clear_color_image`]: Self::try_clear_color_image
    #[track_caller]
    pub unsafe fn clear_color_image(&mut self, clear_info: &ClearColorImageInfo<'_>) -> &mut Self {
        unsafe { self.try_clear_color_image(clear_info) }.unwrap()
    }

    /// Clears a color image with a specific value.
    pub unsafe fn try_clear_color_image(
        &mut self,
        clear_info: &ClearColorImageInfo<'_>,
    ) -> Result<&mut Self> {
        Ok(unsafe { self.clear_color_image_unchecked(clear_info) })
    }

    pub unsafe fn clear_color_image_unchecked(
        &mut self,
        clear_info: &ClearColorImageInfo<'_>,
    ) -> &mut Self {
        let &ClearColorImageInfo {
            image,
            image_layout,
            clear_value,
            regions,
            _ne: _,
        } = clear_info;

        let image = unsafe { self.accesses.image_unchecked(image) };
        let image_layout = AccessTypes::CLEAR_TRANSFER_WRITE.image_layout(image_layout);

        let fns = self.device().fns();
        let cmd_clear_color_image = fns.v1_0.cmd_clear_color_image;

        if regions.is_empty() {
            let region_vk = image.subresource_range().to_vk();

            unsafe {
                cmd_clear_color_image(
                    self.handle(),
                    image.handle(),
                    image_layout.into(),
                    &clear_value.to_vk(),
                    1,
                    &region_vk,
                )
            };
        } else {
            let regions_vk = regions
                .iter()
                .map(ImageSubresourceRange::to_vk)
                .collect::<SmallVec<[_; 8]>>();

            unsafe {
                cmd_clear_color_image(
                    self.handle(),
                    image.handle(),
                    image_layout.into(),
                    &clear_value.to_vk(),
                    regions_vk.len() as u32,
                    regions_vk.as_ptr(),
                )
            };
        }

        self
    }

    /// Clears a depth/stencil image with a specific value, panicking on a validation error.
    ///
    /// This is a shortcut for `try_clear_depth_stencil_image().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_clear_depth_stencil_image`] returns a [`ValidationError`].
    ///
    /// [`try_clear_depth_stencil_image`]: Self::try_clear_depth_stencil_image
    #[track_caller]
    pub unsafe fn clear_depth_stencil_image(
        &mut self,
        clear_info: &ClearDepthStencilImageInfo<'_>,
    ) -> &mut Self {
        unsafe { self.try_clear_depth_stencil_image(clear_info) }.unwrap()
    }

    /// Clears a depth/stencil image with a specific value.
    pub unsafe fn try_clear_depth_stencil_image(
        &mut self,
        clear_info: &ClearDepthStencilImageInfo<'_>,
    ) -> Result<&mut Self> {
        Ok(unsafe { self.clear_depth_stencil_image_unchecked(clear_info) })
    }

    pub unsafe fn clear_depth_stencil_image_unchecked(
        &mut self,
        clear_info: &ClearDepthStencilImageInfo<'_>,
    ) -> &mut Self {
        let &ClearDepthStencilImageInfo {
            image,
            image_layout,
            clear_value,
            regions,
            _ne: _,
        } = clear_info;

        let image = unsafe { self.accesses.image_unchecked(image) };
        let image_layout = AccessTypes::CLEAR_TRANSFER_WRITE.image_layout(image_layout);

        let fns = self.device().fns();
        let cmd_clear_depth_stencil_image = fns.v1_0.cmd_clear_depth_stencil_image;

        if regions.is_empty() {
            let region_vk = image.subresource_range().to_vk();

            unsafe {
                cmd_clear_depth_stencil_image(
                    self.handle(),
                    image.handle(),
                    image_layout.into(),
                    &clear_value.to_vk(),
                    1,
                    &region_vk,
                )
            };
        } else {
            let regions_vk = regions
                .iter()
                .map(ImageSubresourceRange::to_vk)
                .collect::<SmallVec<[_; 8]>>();

            unsafe {
                cmd_clear_depth_stencil_image(
                    self.handle(),
                    image.handle(),
                    image_layout.into(),
                    &clear_value.to_vk(),
                    regions_vk.len() as u32,
                    regions_vk.as_ptr(),
                )
            };
        }

        self
    }

    /// Fills a region of a buffer with repeated copies of a value, panicking on a validation
    /// error.
    ///
    /// This function is similar to the `memset` function in C. The `data` parameter is a number
    /// that will be repeatedly written through the entire buffer.
    ///
    /// This is a shortcut for `try_fill_buffer().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_fill_buffer`] returns a [`ValidationError`].
    ///
    /// [`try_fill_buffer`]: Self::try_fill_buffer
    #[track_caller]
    pub unsafe fn fill_buffer(&mut self, fill_info: &FillBufferInfo<'_>) -> &mut Self {
        unsafe { self.try_fill_buffer(fill_info) }.unwrap()
    }

    /// Fills a region of a buffer with repeated copies of a value.
    ///
    /// This function is similar to the `memset` function in C. The `data` parameter is a number
    /// that will be repeatedly written through the entire buffer.
    pub unsafe fn try_fill_buffer(&mut self, fill_info: &FillBufferInfo<'_>) -> Result<&mut Self> {
        Ok(unsafe { self.fill_buffer_unchecked(fill_info) })
    }

    pub unsafe fn fill_buffer_unchecked(&mut self, fill_info: &FillBufferInfo<'_>) -> &mut Self {
        let &FillBufferInfo {
            dst_buffer,
            dst_offset,
            size,
            data,
            _ne: _,
        } = fill_info;

        let dst_buffer = unsafe { self.accesses.buffer_unchecked(dst_buffer) };

        let fns = self.device().fns();
        let cmd_fill_buffer = fns.v1_0.cmd_fill_buffer;
        unsafe {
            cmd_fill_buffer(
                self.handle(),
                dst_buffer.handle(),
                dst_offset,
                size.unwrap_or(vk::WHOLE_SIZE),
                data,
            )
        };

        self
    }

    /// Writes data to a region of a buffer, panicking on a validation error.
    ///
    /// This is a shortcut for `try_update_buffer().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_update_buffer`] returns a [`ValidationError`].
    ///
    /// [`try_update_buffer`]: Self::try_update_buffer
    #[track_caller]
    pub unsafe fn update_buffer(
        &mut self,
        dst_buffer: Id<Buffer>,
        dst_offset: DeviceSize,
        data: &(impl BufferContents + ?Sized),
    ) -> &mut Self {
        unsafe { self.try_update_buffer(dst_buffer, dst_offset, data) }.unwrap()
    }

    /// Writes data to a region of a buffer.
    pub unsafe fn try_update_buffer(
        &mut self,
        dst_buffer: Id<Buffer>,
        dst_offset: DeviceSize,
        data: &(impl BufferContents + ?Sized),
    ) -> Result<&mut Self> {
        Ok(unsafe { self.update_buffer_unchecked(dst_buffer, dst_offset, data) })
    }

    pub unsafe fn update_buffer_unchecked(
        &mut self,
        dst_buffer: Id<Buffer>,
        dst_offset: DeviceSize,
        data: &(impl BufferContents + ?Sized),
    ) -> &mut Self {
        unsafe {
            self.update_buffer_inner(
                dst_buffer,
                dst_offset,
                <*const _>::cast(data),
                size_of_val(data) as DeviceSize,
            )
        }
    }

    unsafe fn update_buffer_inner(
        &mut self,
        dst_buffer: Id<Buffer>,
        dst_offset: DeviceSize,
        data: *const c_void,
        data_size: DeviceSize,
    ) -> &mut Self {
        if data_size == 0 {
            return self;
        }

        let dst_buffer = unsafe { self.accesses.buffer_unchecked(dst_buffer) };

        let fns = self.device().fns();
        let cmd_update_buffer = fns.v1_0.cmd_update_buffer;
        unsafe {
            cmd_update_buffer(
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
    /// The default value is [`Id::INVALID`], which must be overridden.
    pub image: Id<Image>,

    /// The layout used for `image` during the clear operation.
    ///
    /// The default value is [`ImageLayoutType::Optimal`].
    pub image_layout: ImageLayoutType,

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

impl Default for ClearColorImageInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl ClearColorImageInfo<'_> {
    /// Returns a default `ClearColorImageInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            image: Id::INVALID,
            image_layout: ImageLayoutType::Optimal,
            clear_value: ClearColorValue::Float([0.0; 4]),
            regions: &[],
            _ne: crate::NE,
        }
    }
}

/// Parameters to clear a depth/stencil image.
#[derive(Clone, Debug)]
pub struct ClearDepthStencilImageInfo<'a> {
    /// The image to clear.
    ///
    /// The default value is [`Id::INVALID`], which must be overridden.
    pub image: Id<Image>,

    /// The layout used for `image` during the clear operation.
    ///
    /// The default value is [`ImageLayoutType::Optimal`].
    pub image_layout: ImageLayoutType,

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

impl Default for ClearDepthStencilImageInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl ClearDepthStencilImageInfo<'_> {
    /// Returns a default `ClearDepthStencilImageInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            image: Id::INVALID,
            image_layout: ImageLayoutType::Optimal,
            clear_value: ClearDepthStencilValue {
                depth: 0.0,
                stencil: 0,
            },
            regions: &[],
            _ne: crate::NE,
        }
    }
}

/// Parameters to fill a region of a buffer with repeated copies of a value.
#[derive(Clone, Debug)]
pub struct FillBufferInfo<'a> {
    /// The buffer to fill.
    ///
    /// The default value is [`Id::INVALID`], which must be overridden.
    pub dst_buffer: Id<Buffer>,

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

impl Default for FillBufferInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl FillBufferInfo<'_> {
    /// Returns a default `FillBufferInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            dst_buffer: Id::INVALID,
            dst_offset: 0,
            size: None,
            data: 0,
            _ne: crate::NE,
        }
    }
}
