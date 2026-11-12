//! Low-level descriptor set.

use super::{
    allocator::{DescriptorSetAlloc, DescriptorSetAllocator},
    pool::DescriptorPool,
    CopyDescriptorSet,
};
use crate::{
    descriptor_set::{layout::DescriptorSetLayout, update::WriteDescriptorSet},
    device::{Device, DeviceOwned},
    Validated, ValidationError, VulkanError, VulkanObject,
};
use ash::vk;
use smallvec::SmallVec;
use std::{
    fmt::Debug,
    hash::{Hash, Hasher},
    mem::ManuallyDrop,
    sync::Arc,
};

/// A raw descriptor set corresponding directly to a `VkDescriptorSet`.
///
/// This descriptor set does not keep track of synchronization, nor does it store any information
/// on what resources have been written to each descriptor.
#[derive(Debug)]
pub struct RawDescriptorSet {
    allocation: ManuallyDrop<DescriptorSetAlloc>,
    allocator: Arc<dyn DescriptorSetAllocator>,
}

impl RawDescriptorSet {
    /// Allocates a new descriptor set and returns it, panicking on a validation error.
    ///
    /// This is a shortcut for `try_new().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_new`] returns a [`ValidationError`].
    ///
    /// [`try_new`]: Self::try_new
    #[inline]
    #[track_caller]
    pub fn new(
        allocator: &Arc<impl DescriptorSetAllocator + ?Sized>,
        layout: &Arc<DescriptorSetLayout>,
        variable_descriptor_count: u32,
    ) -> Result<RawDescriptorSet, VulkanError> {
        Self::new_inner(
            allocator.clone().as_dyn(),
            layout,
            variable_descriptor_count,
        )
    }

    #[inline]
    #[track_caller]
    fn new_inner(
        allocator: Arc<dyn DescriptorSetAllocator>,
        layout: &Arc<DescriptorSetLayout>,
        variable_descriptor_count: u32,
    ) -> Result<RawDescriptorSet, VulkanError> {
        match Self::try_new_inner(allocator, layout, variable_descriptor_count) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Allocates a new descriptor set and returns it.
    #[inline]
    pub fn try_new(
        allocator: &Arc<impl DescriptorSetAllocator + ?Sized>,
        layout: &Arc<DescriptorSetLayout>,
        variable_descriptor_count: u32,
    ) -> Result<RawDescriptorSet, Validated<VulkanError>> {
        Self::try_new_inner(
            allocator.clone().as_dyn(),
            layout,
            variable_descriptor_count,
        )
    }

    fn try_new_inner(
        allocator: Arc<dyn DescriptorSetAllocator>,
        layout: &Arc<DescriptorSetLayout>,
        variable_descriptor_count: u32,
    ) -> Result<RawDescriptorSet, Validated<VulkanError>> {
        let allocation = allocator.try_allocate(layout, variable_descriptor_count)?;

        Ok(unsafe { Self::new_unchecked_inner_inner(allocator, allocation) }?)
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        allocator: &Arc<impl DescriptorSetAllocator + ?Sized>,
        layout: &Arc<DescriptorSetLayout>,
        variable_descriptor_count: u32,
    ) -> Result<RawDescriptorSet, VulkanError> {
        unsafe {
            Self::new_unchecked_inner(
                allocator.clone().as_dyn(),
                layout,
                variable_descriptor_count,
            )
        }
    }

    unsafe fn new_unchecked_inner(
        allocator: Arc<dyn DescriptorSetAllocator>,
        layout: &Arc<DescriptorSetLayout>,
        variable_descriptor_count: u32,
    ) -> Result<RawDescriptorSet, VulkanError> {
        let allocation =
            unsafe { allocator.allocate_unchecked(layout, variable_descriptor_count) }?;

        unsafe { Self::new_unchecked_inner_inner(allocator, allocation) }
    }

    unsafe fn new_unchecked_inner_inner(
        allocator: Arc<dyn DescriptorSetAllocator>,
        allocation: DescriptorSetAlloc,
    ) -> Result<RawDescriptorSet, VulkanError> {
        Ok(RawDescriptorSet {
            allocation: ManuallyDrop::new(allocation),
            allocator,
        })
    }

    /// Returns the allocation of this descriptor set.
    #[inline]
    pub fn alloc(&self) -> &DescriptorSetAlloc {
        &self.allocation
    }

    /// Returns the descriptor pool that the descriptor set was allocated from.
    #[inline]
    pub fn pool(&self) -> &DescriptorPool {
        &self.allocation.pool
    }

    /// Returns the layout of this descriptor set.
    #[inline]
    pub fn layout(&self) -> &Arc<DescriptorSetLayout> {
        self.allocation.inner.layout()
    }

    /// Returns the variable descriptor count that this descriptor set was allocated with.
    #[inline]
    pub fn variable_descriptor_count(&self) -> u32 {
        self.allocation.inner.variable_descriptor_count()
    }

    /// Updates the descriptor set with new values, panicking on a validation error.
    ///
    /// This is a shortcut for `try_update().unwrap()`.
    ///
    /// # Safety
    ///
    /// - The resources in `descriptor_writes` and `descriptor_copies` must be kept alive for as
    ///   long as `self` is in use.
    /// - The descriptor set must not be in use by the device, or be recorded to a command buffer
    ///   as part of a bind command.
    /// - Host access to the descriptor set must be externally synchronized.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_update`] returns a [`ValidationError`].
    ///
    /// [`try_update`]: Self::try_update
    #[inline]
    #[track_caller]
    pub unsafe fn update(
        &self,
        descriptor_writes: &[WriteDescriptorSet<'_>],
        descriptor_copies: &[CopyDescriptorSet<'_>],
    ) {
        unsafe { self.try_update(descriptor_writes, descriptor_copies) }.unwrap()
    }

    /// Updates the descriptor set with new values.
    ///
    /// # Safety
    ///
    /// - The resources in `descriptor_writes` and `descriptor_copies` must be kept alive for as
    ///   long as `self` is in use.
    /// - The descriptor set must not be in use by the device, or be recorded to a command buffer
    ///   as part of a bind command.
    /// - Host access to the descriptor set must be externally synchronized.
    #[inline]
    pub unsafe fn try_update(
        &self,
        descriptor_writes: &[WriteDescriptorSet<'_>],
        descriptor_copies: &[CopyDescriptorSet<'_>],
    ) -> Result<(), Box<ValidationError>> {
        if descriptor_writes.is_empty() && descriptor_copies.is_empty() {
            return Ok(());
        }

        self.validate_update(descriptor_writes, descriptor_copies)?;

        unsafe { self.update_unchecked(descriptor_writes, descriptor_copies) };
        Ok(())
    }

    pub(super) fn validate_update(
        &self,
        descriptor_writes: &[WriteDescriptorSet<'_>],
        descriptor_copies: &[CopyDescriptorSet<'_>],
    ) -> Result<(), Box<ValidationError>> {
        for (index, write) in descriptor_writes.iter().enumerate() {
            write
                .validate(self.layout(), self.variable_descriptor_count())
                .map_err(|err| err.add_context(format!("descriptor_writes[{}]", index)))?;
        }

        for (index, copy) in descriptor_copies.iter().enumerate() {
            copy.validate(self)
                .map_err(|err| err.add_context(format!("descriptor_copies[{}]", index)))?;
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn update_unchecked(
        &self,
        descriptor_writes: &[WriteDescriptorSet<'_>],
        descriptor_copies: &[CopyDescriptorSet<'_>],
    ) {
        if descriptor_writes.is_empty() && descriptor_copies.is_empty() {
            return;
        }

        let set_layout = self.layout();
        let writes_fields1_vk: SmallVec<[_; 8]> = descriptor_writes
            .iter()
            .map(|write| {
                let default_image_layout = set_layout
                    .binding(write.dst_binding())
                    .unwrap()
                    .descriptor_type
                    .default_image_layout();
                write.to_vk_fields1(default_image_layout)
            })
            .collect();
        let mut write_extensions_vk: SmallVec<[_; 8]> = descriptor_writes
            .iter()
            .zip(&writes_fields1_vk)
            .map(|(write, fields1_vk)| write.to_vk_extensions(fields1_vk))
            .collect();
        let writes_vk: SmallVec<[_; 8]> = descriptor_writes
            .iter()
            .zip(&writes_fields1_vk)
            .zip(&mut write_extensions_vk)
            .map(|((write, write_info_vk), write_extension_vk)| {
                write.to_vk(
                    self.handle(),
                    set_layout
                        .binding(write.dst_binding())
                        .unwrap()
                        .descriptor_type,
                    write_info_vk,
                    write_extension_vk,
                )
            })
            .collect();

        let copies_vk: SmallVec<[_; 8]> = descriptor_copies
            .iter()
            .map(|copy| copy.to_vk(self.handle()))
            .collect();

        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.update_descriptor_sets)(
                self.device().handle(),
                writes_vk.len() as u32,
                writes_vk.as_ptr(),
                copies_vk.len() as u32,
                copies_vk.as_ptr(),
            )
        };
    }
}

impl Drop for RawDescriptorSet {
    #[inline]
    fn drop(&mut self) {
        let allocation = unsafe { ManuallyDrop::take(&mut self.allocation) };
        unsafe { self.allocator.deallocate(allocation) };
    }
}

unsafe impl VulkanObject for RawDescriptorSet {
    type Handle = vk::DescriptorSet;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.allocation.inner.handle()
    }
}

unsafe impl DeviceOwned for RawDescriptorSet {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.allocation.inner.device()
    }
}

impl PartialEq for RawDescriptorSet {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.allocation.inner == other.allocation.inner
    }
}

impl Eq for RawDescriptorSet {}

impl Hash for RawDescriptorSet {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.allocation.inner.hash(state);
    }
}
