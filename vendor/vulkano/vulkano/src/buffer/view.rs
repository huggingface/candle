//! View of a buffer, in order to use it as a uniform texel buffer or storage texel buffer.
//!
//! In order to use a buffer as a uniform texel buffer or a storage texel buffer, you have to
//! create a `BufferView`, which indicates which format the data is in.
//!
//! In order to create a view from a buffer, the buffer must have been created with either the
//! `uniform_texel_buffer` or the `storage_texel_buffer` usage.
//!
//! # Examples
//!
//! ```
//! # use std::sync::Arc;
//! use vulkano::{
//!     buffer::{view::{BufferView, BufferViewCreateInfo}, Buffer, BufferCreateInfo, BufferUsage},
//!     format::Format,
//!     memory::allocator::{AllocationCreateInfo, DeviceLayout},
//! };
//!
//! # let queue: Arc<vulkano::device::Queue> = return;
//! # let memory_allocator: Arc<vulkano::memory::allocator::StandardMemoryAllocator> = return;
//! #
//! let buffer = Buffer::new(
//!     &memory_allocator,
//!     &BufferCreateInfo {
//!         usage: BufferUsage::STORAGE_TEXEL_BUFFER,
//!         ..Default::default()
//!     },
//!     &AllocationCreateInfo::default(),
//!     DeviceLayout::new_unsized::<[u32]>(128).unwrap(),
//! )
//! .unwrap();
//!
//! let view = BufferView::new(
//!     &buffer,
//!     &BufferViewCreateInfo {
//!         format: Format::R32_UINT,
//!         ..Default::default()
//!     },
//! )
//! .unwrap();
//! ```

use super::BufferUsage;
use crate::{
    buffer::Buffer,
    device::{Device, DeviceOwned},
    format::{Format, FormatFeatures},
    macros::impl_id_counter,
    memory::{is_aligned, DeviceAlignment},
    DeviceSize, Validated, ValidationError, Version, VulkanError, VulkanObject,
};
use ash::vk;
use std::{mem::MaybeUninit, num::NonZero, ptr, sync::Arc};

/// Represents a way for the GPU to interpret buffer data. See the documentation of the
/// `view` module.
#[derive(Debug)]
pub struct BufferView {
    handle: vk::BufferView,
    buffer: Arc<Buffer>,
    id: NonZero<u64>,

    format: Format,
    format_features: FormatFeatures,
    offset: DeviceSize,
    range: Option<DeviceSize>,
}

impl BufferView {
    /// Creates a new `BufferView`, panicking on a validation error.
    ///
    /// This is a shortcut for `try_new().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_new`] returns an error.
    ///
    /// [`try_new`]: Self::try_new
    #[inline]
    #[track_caller]
    pub fn new(
        buffer: &Arc<Buffer>,
        create_info: &BufferViewCreateInfo<'_>,
    ) -> Result<Arc<BufferView>, VulkanError> {
        match Self::try_new(buffer, create_info) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a new `BufferView`, returning an error on failure.
    #[inline]
    pub fn try_new(
        buffer: &Arc<Buffer>,
        create_info: &BufferViewCreateInfo<'_>,
    ) -> Result<Arc<BufferView>, Validated<VulkanError>> {
        Self::validate_new(buffer, create_info)?;

        Ok(unsafe { Self::new_unchecked(buffer, create_info) }?)
    }

    fn validate_new(
        buffer: &Arc<Buffer>,
        create_info: &BufferViewCreateInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        let device = buffer.device();

        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;

        let &BufferViewCreateInfo {
            format,
            offset,
            range,
            _ne: _,
        } = create_info;

        let properties = device.physical_device().properties();

        let format_properties =
            unsafe { device.physical_device().format_properties_unchecked(format) };
        let format_features = format_properties.buffer_features;

        if offset >= buffer.size() {
            return Err(Box::new(ValidationError {
                context: "offset".into(),
                problem: "must be less than `buffer.size()`".into(),
                vuids: &["VUID-VkBufferViewCreateInfo-offset-00925"],
                ..Default::default()
            }));
        }

        if !buffer
            .usage()
            .intersects(BufferUsage::UNIFORM_TEXEL_BUFFER | BufferUsage::STORAGE_TEXEL_BUFFER)
        {
            return Err(Box::new(ValidationError {
                context: "buffer".into(),
                problem: "was not created with the `BufferUsage::UNIFORM_TEXEL_BUFFER` \
                    or `BufferUsage::STORAGE_TEXEL_BUFFER` usage"
                    .into(),
                vuids: &["VUID-VkBufferViewCreateInfo-buffer-00932"],
                ..Default::default()
            }));
        }

        if buffer.usage().intersects(BufferUsage::UNIFORM_TEXEL_BUFFER)
            && !format_features.intersects(FormatFeatures::UNIFORM_TEXEL_BUFFER)
        {
            return Err(Box::new(ValidationError {
                problem: "`buffer` was created with the `BufferUsage::UNIFORM_TEXEL_BUFFER` \
                    usage, but the format features of `create_info.format` do not include \
                    `FormatFeatures::UNIFORM_TEXEL_BUFFER`"
                    .into(),
                vuids: &["VUID-VkBufferViewCreateInfo-buffer-00933"],
                ..Default::default()
            }));
        }

        if buffer.usage().intersects(BufferUsage::STORAGE_TEXEL_BUFFER)
            && !format_features.intersects(FormatFeatures::STORAGE_TEXEL_BUFFER)
        {
            return Err(Box::new(ValidationError {
                problem: "`buffer` was created with the `BufferUsage::STORAGE_TEXEL_BUFFER` \
                    usage, but the format features of `create_info.format` do not include \
                    `FormatFeatures::STORAGE_TEXEL_BUFFER`"
                    .into(),
                vuids: &["VUID-VkBufferViewCreateInfo-buffer-00934"],
                ..Default::default()
            }));
        }

        let block_size = format.block_size();
        let texels_per_block = format.texels_per_block();

        if let Some(range) = range {
            if range == 0 {
                return Err(Box::new(ValidationError {
                    context: "range".into(),
                    problem: "is zero".into(),
                    vuids: &["VUID-VkBufferViewCreateInfo-range-00928"],
                    ..Default::default()
                }));
            }

            if !range.is_multiple_of(block_size) {
                return Err(Box::new(ValidationError {
                    context: "range".into(),
                    problem: "is not a multiple of `create_info.format.block_size()`".into(),
                    vuids: &["VUID-VkBufferViewCreateInfo-range-00929"],
                    ..Default::default()
                }));
            }

            if ((range / block_size) * texels_per_block as DeviceSize) as u32
                > properties.max_texel_buffer_elements
            {
                return Err(Box::new(ValidationError {
                    problem: "`buffer.size() / create_info.format.block_size() * \
                        create_info.format.texels_per_block()` is greater than the \
                        `max_texel_buffer_elements` limit"
                        .into(),
                    vuids: &["VUID-VkBufferViewCreateInfo-range-00930"],
                    ..Default::default()
                }));
            }

            if range > buffer.size() - offset {
                return Err(Box::new(ValidationError {
                    problem: "`create_info.offset + create_info.range` must be less than or \
                        equal to `buffer.size()`"
                        .into(),
                    vuids: &["VUID-VkBufferViewCreateInfo-offset-00931"],
                    ..Default::default()
                }));
            }
        }

        if device.api_version() >= Version::V1_3 || device.enabled_features().texel_buffer_alignment
        {
            let element_size = DeviceAlignment::new(if block_size % 3 == 0 {
                block_size / 3
            } else {
                block_size
            })
            .unwrap();

            if buffer.usage().intersects(BufferUsage::STORAGE_TEXEL_BUFFER) {
                if properties
                    .storage_texel_buffer_offset_single_texel_alignment
                    .unwrap()
                {
                    if !is_aligned(
                        offset,
                        properties
                            .storage_texel_buffer_offset_alignment_bytes
                            .unwrap()
                            .min(element_size),
                    ) {
                        return Err(Box::new(ValidationError {
                            problem: "`buffer` was created with the \
                                `BufferUsage::STORAGE_TEXEL_BUFFER` usage, and the \
                                `storage_texel_buffer_offset_single_texel_alignment` \
                                property is `true`, but \
                                `create_info.offset` is not a multiple of the \
                                minimum of `create_info.format.block_size()` and the \
                                `storage_texel_buffer_offset_alignment_bytes` limit"
                                .into(),
                            vuids: &["VUID-VkBufferViewCreateInfo-buffer-02750"],
                            ..Default::default()
                        }));
                    }
                } else {
                    if !is_aligned(
                        offset,
                        properties
                            .storage_texel_buffer_offset_alignment_bytes
                            .unwrap(),
                    ) {
                        return Err(Box::new(ValidationError {
                            problem: "`buffer` was created with the \
                                `BufferUsage::STORAGE_TEXEL_BUFFER` usage, and the \
                                `storage_texel_buffer_offset_single_texel_alignment` \
                                property is `false`, but \
                                `create_info.offset` is not a multiple of the \
                                `storage_texel_buffer_offset_alignment_bytes` limit"
                                .into(),
                            vuids: &["VUID-VkBufferViewCreateInfo-buffer-02750"],
                            ..Default::default()
                        }));
                    }
                }
            }

            if buffer.usage().intersects(BufferUsage::UNIFORM_TEXEL_BUFFER) {
                if properties
                    .uniform_texel_buffer_offset_single_texel_alignment
                    .unwrap()
                {
                    if !is_aligned(
                        offset,
                        properties
                            .uniform_texel_buffer_offset_alignment_bytes
                            .unwrap()
                            .min(element_size),
                    ) {
                        return Err(Box::new(ValidationError {
                            problem: "`buffer` was created with the \
                                `BufferUsage::UNIFORM_TEXEL_BUFFER` usage, and the \
                                `uniform_texel_buffer_offset_single_texel_alignment` \
                                property is `false`, but \
                                `create_info.offset` is not a multiple of the \
                                minimum of `create_info.format.block_size()` and the \
                                `uniform_texel_buffer_offset_alignment_bytes` limit"
                                .into(),
                            vuids: &["VUID-VkBufferViewCreateInfo-buffer-02751"],
                            ..Default::default()
                        }));
                    }
                } else {
                    if !is_aligned(
                        offset,
                        properties
                            .uniform_texel_buffer_offset_alignment_bytes
                            .unwrap(),
                    ) {
                        return Err(Box::new(ValidationError {
                            problem: "`buffer` was created with the \
                                `BufferUsage::UNIFORM_TEXEL_BUFFER` usage, and the \
                                `uniform_texel_buffer_offset_single_texel_alignment` \
                                property is `false`, but \
                                `create_info.offset` is not a multiple of the \
                                `uniform_texel_buffer_offset_alignment_bytes` limit"
                                .into(),
                            vuids: &["VUID-VkBufferViewCreateInfo-buffer-02751"],
                            ..Default::default()
                        }));
                    }
                }
            }
        } else {
            if !is_aligned(offset, properties.min_texel_buffer_offset_alignment) {
                return Err(Box::new(ValidationError {
                    problem: "`create_info.offset` is not a multiple of the \
                        `min_texel_buffer_offset_alignment` limit"
                        .into(),
                    vuids: &["VUID-VkBufferViewCreateInfo-offset-02749"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        buffer: &Arc<Buffer>,
        create_info: &BufferViewCreateInfo<'_>,
    ) -> Result<Arc<BufferView>, VulkanError> {
        let device = buffer.device();
        let create_info_vk = create_info.to_vk(buffer);

        let fns = device.fns();
        let handle = {
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.v1_0.create_buffer_view)(
                    device.handle(),
                    &create_info_vk,
                    ptr::null(),
                    output.as_mut_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;
            unsafe { output.assume_init() }
        };

        Ok(unsafe { Self::from_handle(buffer, handle, create_info) })
    }

    /// Creates a new `BufferView` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `buffer` and `create_info` must match the info used to create the object.
    pub unsafe fn from_handle(
        buffer: &Arc<Buffer>,
        handle: vk::BufferView,
        create_info: &BufferViewCreateInfo<'_>,
    ) -> Arc<BufferView> {
        let &BufferViewCreateInfo {
            format,
            offset,
            range,
            _ne: _,
        } = create_info;
        let format_properties = unsafe {
            buffer
                .device()
                .physical_device()
                .format_properties_unchecked(format)
        };
        let format_features = format_properties.buffer_features;

        Arc::new(BufferView {
            handle,
            buffer: buffer.clone(),
            id: Self::next_id(),
            format,
            format_features,
            offset,
            range,
        })
    }

    /// Returns the buffer associated to this view.
    #[inline]
    pub fn buffer(&self) -> &Arc<Buffer> {
        &self.buffer
    }

    /// Returns the format of this view.
    #[inline]
    pub fn format(&self) -> Format {
        self.format
    }

    /// Returns the features supported by this view’s format.
    #[inline]
    pub fn format_features(&self) -> FormatFeatures {
        self.format_features
    }

    /// Returns the offset of the view in bytes.
    #[inline]
    pub fn offset(&self) -> DeviceSize {
        self.offset
    }

    /// Returns the range of the view in bytes.
    #[inline]
    pub fn range(&self) -> Option<DeviceSize> {
        self.range
    }
}

impl Drop for BufferView {
    #[inline]
    fn drop(&mut self) {
        let fns = self.buffer.device().fns();
        unsafe {
            (fns.v1_0.destroy_buffer_view)(self.buffer.device().handle(), self.handle, ptr::null())
        };
    }
}

unsafe impl VulkanObject for BufferView {
    type Handle = vk::BufferView;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for BufferView {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.buffer.device()
    }
}

impl_id_counter!(BufferView);

/// Parameters to create a new `BufferView`.
#[derive(Clone, Debug)]
pub struct BufferViewCreateInfo<'a> {
    /// The format of the buffer view.
    ///
    /// The default value is `Format::UNDEFINED`.
    pub format: Format,

    /// The offset in bytes.
    ///
    /// The default value is `0`.
    pub offset: DeviceSize,

    /// The size in bytes.
    ///
    /// If set to `None`, the size until the end of the buffer will be used.
    ///
    /// The default value is `None`.
    pub range: Option<DeviceSize>,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for BufferViewCreateInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl BufferViewCreateInfo<'_> {
    /// Returns a default `BufferViewCreateInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            format: Format::UNDEFINED,
            offset: 0,
            range: None,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let Self {
            format,
            offset: _,
            range: _,
            _ne: _,
        } = self;

        format.validate_device(device).map_err(|err| {
            err.add_context("format")
                .set_vuids(&["VUID-VkBufferViewCreateInfo-format-parameter"])
        })?;

        Ok(())
    }

    pub(crate) fn to_vk(&self, buffer: &Arc<Buffer>) -> vk::BufferViewCreateInfo<'static> {
        let &Self {
            format,
            offset,
            range,
            _ne: _,
        } = self;

        vk::BufferViewCreateInfo::default()
            .flags(vk::BufferViewCreateFlags::empty())
            .buffer(buffer.handle())
            .format(format.into())
            .offset(offset)
            .range(range.unwrap_or(vk::WHOLE_SIZE))
    }
}

#[cfg(test)]
mod tests {
    use super::{BufferView, BufferViewCreateInfo};
    use crate::{
        buffer::{Buffer, BufferCreateInfo, BufferUsage},
        format::Format,
        memory::allocator::{AllocationCreateInfo, DeviceLayout, StandardMemoryAllocator},
    };
    use std::sync::Arc;

    #[test]
    fn create_uniform() {
        // `VK_FORMAT_R8G8B8A8_UNORM` guaranteed to be a supported format
        let (device, _) = gfx_dev_and_queue!();
        let memory_allocator = Arc::new(StandardMemoryAllocator::new(&device, &Default::default()));

        let buffer = Buffer::new(
            &memory_allocator,
            &BufferCreateInfo {
                usage: BufferUsage::UNIFORM_TEXEL_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo::default(),
            DeviceLayout::new_unsized::<[[u8; 4]]>(128).unwrap(),
        )
        .unwrap();

        BufferView::new(
            &buffer,
            &BufferViewCreateInfo {
                format: Format::R8G8B8A8_UNORM,
                ..Default::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn create_storage() {
        // `VK_FORMAT_R8G8B8A8_UNORM` guaranteed to be a supported format
        let (device, _) = gfx_dev_and_queue!();
        let memory_allocator = Arc::new(StandardMemoryAllocator::new(&device, &Default::default()));

        let buffer = Buffer::new(
            &memory_allocator,
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_TEXEL_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo::default(),
            DeviceLayout::new_unsized::<[[u8; 4]]>(128).unwrap(),
        )
        .unwrap();
        BufferView::new(
            &buffer,
            &BufferViewCreateInfo {
                format: Format::R8G8B8A8_UNORM,
                ..Default::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn create_storage_atomic() {
        // `VK_FORMAT_R32_UINT` guaranteed to be a supported format for atomics
        let (device, _) = gfx_dev_and_queue!();
        let memory_allocator = Arc::new(StandardMemoryAllocator::new(&device, &Default::default()));

        let buffer = Buffer::new(
            &memory_allocator,
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_TEXEL_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo::default(),
            DeviceLayout::new_unsized::<[u32]>(128).unwrap(),
        )
        .unwrap();
        BufferView::new(
            &buffer,
            &BufferViewCreateInfo {
                format: Format::R32_UINT,
                ..Default::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn wrong_usage() {
        // `VK_FORMAT_R8G8B8A8_UNORM` guaranteed to be a supported format
        let (device, _) = gfx_dev_and_queue!();
        let memory_allocator = Arc::new(StandardMemoryAllocator::new(&device, &Default::default()));

        let buffer = Buffer::new(
            &memory_allocator,
            &BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST, // Dummy value
                ..Default::default()
            },
            &AllocationCreateInfo::default(),
            DeviceLayout::new_unsized::<[[u8; 4]]>(128).unwrap(),
        )
        .unwrap();

        match BufferView::try_new(
            &buffer,
            &BufferViewCreateInfo {
                format: Format::R8G8B8A8_UNORM,
                ..Default::default()
            },
        ) {
            Err(_) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn unsupported_format() {
        let (device, _) = gfx_dev_and_queue!();
        let memory_allocator = Arc::new(StandardMemoryAllocator::new(&device, &Default::default()));

        let buffer = Buffer::new(
            &memory_allocator,
            &BufferCreateInfo {
                usage: BufferUsage::UNIFORM_TEXEL_BUFFER | BufferUsage::STORAGE_TEXEL_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo::default(),
            DeviceLayout::new_unsized::<[[f64; 4]]>(128).unwrap(),
        )
        .unwrap();

        // TODO: what if R64G64B64A64_SFLOAT is supported?
        match BufferView::try_new(
            &buffer,
            &BufferViewCreateInfo {
                format: Format::R64G64B64A64_SFLOAT,
                ..Default::default()
            },
        ) {
            Err(_) => (),
            _ => panic!(),
        }
    }
}
