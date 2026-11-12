//! Bindings between shaders and the resources they access.
//!
//! # Overview
//!
//! In order to access a buffer or an image from a shader, that buffer or image must be put in a
//! *descriptor*. Each descriptor contains one buffer or one image alongside with the way that it
//! can be accessed. A descriptor can also be an array, in which case it contains multiple buffers
//! or images that all have the same layout.
//!
//! Descriptors are grouped in what is called *descriptor sets*. In Vulkan you don't bind
//! individual descriptors one by one, but you create then bind descriptor sets one by one. As
//! binding a descriptor set has (small but non-null) a cost, you are encouraged to put descriptors
//! that are often used together in the same set so that you can keep the same set binding through
//! multiple draws.
//!
//! # Examples
//!
//! <div class="vulkano-alert-note">
//!
//! > Note
//! >
//! > This section describes the simple way to bind resources. There are more optimized ways.
//!
//! </div>
//!
//! There are two steps to give access to a resource in a shader: creating the descriptor set, and
//! passing the descriptor sets when drawing.
//!
//! ## Creating a descriptor set
//!
//! TODO: write
//!
//! ## Passing the descriptor set when drawing
//!
//! TODO: write
//!
//! # When drawing
//!
//! When you call a function that adds a draw command to a command buffer, vulkano will check that
//! the descriptor sets you bound are compatible with the layout of the pipeline.
//!
//! TODO: talk about perfs of changing sets
//!
//! # Descriptor sets creation and management
//!
//! There are three concepts in Vulkan related to descriptor sets:
//!
//! - A `VkDescriptorSetLayout` is a Vulkan object that describes to the Vulkan implementation the
//!   layout of a future descriptor set. When you allocate a descriptor set, you have to pass an
//!   instance of this object. This is represented with the [`DescriptorSetLayout`] type in
//!   vulkano.
//! - A `VkDescriptorPool` is a Vulkan object that holds the memory of descriptor sets and that can
//!   be used to allocate and free individual descriptor sets. This is represented with the
//!   [`DescriptorPool`] type in vulkano.
//! - A `VkDescriptorSet` contains the bindings to resources and is allocated from a pool. This is
//!   represented with the [`RawDescriptorSet`] type in vulkano.
//!
//! In addition to this, vulkano defines the following:
//!
//! - The [`DescriptorSetAllocator`] trait can be implemented on types from which you can allocate
//!   and free descriptor sets. However it is different from Vulkan descriptor pools in the sense
//!   that an implementation of the `DescriptorSetAllocator` trait can manage multiple Vulkan
//!   descriptor pools.
//! - The [`StandardDescriptorSetAllocator`] type is a default implementation of the
//!   [`DescriptorSetAllocator`] trait.
//! - The [`DescriptorSet`] type wraps around an `RawDescriptorSet` a safe way. A Vulkan descriptor
//!   set is inherently unsafe, so we need safe wrappers around them.
//! - The [`DescriptorSetsCollection`] trait is implemented on collections of descriptor sets. It
//!   is what you pass to the bind function.
//!
//! [`StandardDescriptorSetAllocator`]: allocator::StandardDescriptorSetAllocator

pub(crate) use self::update::{
    OwnedDescriptorBufferInfo, OwnedDescriptorImageInfo, OwnedWriteDescriptorSetElements,
};
use self::{
    allocator::DescriptorSetAllocator,
    layout::DescriptorSetLayout,
    pool::{DescriptorPool, DescriptorPoolAlloc},
    sys::RawDescriptorSet,
};
pub use self::{
    collection::DescriptorSetsCollection,
    update::{
        CopyDescriptorSet, DescriptorBufferInfo, DescriptorImageInfo, InvalidateDescriptorSet,
        WriteDescriptorSet, WriteDescriptorSetElements,
    },
};
use crate::{
    acceleration_structure::AccelerationStructure,
    buffer::view::BufferView,
    descriptor_set::layout::{
        DescriptorBindingFlags, DescriptorSetLayoutCreateFlags, DescriptorType,
    },
    device::{Device, DeviceOwned},
    image::ImageLayout,
    Validated, ValidationError, VulkanError, VulkanObject,
};
use ash::vk;
use foldhash::HashMap;
use parking_lot::{RwLock, RwLockReadGuard};
use smallvec::{smallvec, SmallVec};
use std::{
    hash::{Hash, Hasher},
    sync::Arc,
};

pub mod allocator;
mod collection;
pub mod layout;
pub mod pool;
pub mod sys;
mod update;

/// An object that contains a collection of resources that will be accessible by shaders.
///
/// Descriptor sets can be bound when recording a command buffer.
#[derive(Debug)]
pub struct DescriptorSet {
    inner: RawDescriptorSet,
    resources: RwLock<DescriptorSetResources>,
}

impl DescriptorSet {
    /// Creates and returns a new descriptor set with a variable descriptor count of 0.
    pub fn new(
        allocator: &Arc<impl DescriptorSetAllocator>,
        layout: &Arc<DescriptorSetLayout>,
        descriptor_writes: &[WriteDescriptorSet<'_>],
        descriptor_copies: &[CopyDescriptorSet<'_>],
    ) -> Result<Arc<DescriptorSet>, Validated<VulkanError>> {
        Self::new_variable(allocator, layout, 0, descriptor_writes, descriptor_copies)
    }

    /// Creates and returns a new descriptor set with the requested variable descriptor count.
    pub fn new_variable(
        allocator: &Arc<impl DescriptorSetAllocator>,
        layout: &Arc<DescriptorSetLayout>,
        variable_descriptor_count: u32,
        descriptor_writes: &[WriteDescriptorSet<'_>],
        descriptor_copies: &[CopyDescriptorSet<'_>],
    ) -> Result<Arc<DescriptorSet>, Validated<VulkanError>> {
        let mut set = DescriptorSet {
            inner: RawDescriptorSet::new(allocator, layout, variable_descriptor_count)?,
            resources: RwLock::new(DescriptorSetResources::new(
                layout,
                variable_descriptor_count,
            )),
        };

        set.update(descriptor_writes, descriptor_copies)?;

        Ok(Arc::new(set))
    }

    /// Returns the inner raw descriptor set.
    #[inline]
    pub fn as_raw(&self) -> &RawDescriptorSet {
        &self.inner
    }

    /// Returns the allocation of the descriptor set.
    #[inline]
    pub fn alloc(&self) -> &DescriptorPoolAlloc {
        &self.inner.alloc().inner
    }

    /// Returns the descriptor pool that the descriptor set was allocated from.
    #[inline]
    pub fn pool(&self) -> &DescriptorPool {
        self.inner.pool()
    }

    /// Returns the layout of this descriptor set.
    #[inline]
    pub fn layout(&self) -> &Arc<DescriptorSetLayout> {
        self.alloc().layout()
    }

    /// Returns the variable descriptor count that this descriptor set was allocated with.
    #[inline]
    pub fn variable_descriptor_count(&self) -> u32 {
        self.alloc().variable_descriptor_count()
    }

    /// Creates a [`DescriptorSetWithOffsets`] with the given dynamic offsets.
    pub fn offsets(
        self: Arc<Self>,
        dynamic_offsets: impl IntoIterator<Item = u32>,
    ) -> DescriptorSetWithOffsets {
        DescriptorSetWithOffsets::new(self, dynamic_offsets)
    }

    /// Returns the resources bound to this descriptor set.
    #[inline]
    pub(crate) fn resources(&self) -> RwLockReadGuard<'_, DescriptorSetResources> {
        self.resources.read()
    }

    /// Updates the descriptor set with new values.
    pub fn update(
        &mut self,
        descriptor_writes: &[WriteDescriptorSet<'_>],
        descriptor_copies: &[CopyDescriptorSet<'_>],
    ) -> Result<(), Box<ValidationError>> {
        if descriptor_writes.is_empty() && descriptor_copies.is_empty() {
            return Ok(());
        }

        self.inner
            .validate_update(descriptor_writes, descriptor_copies)?;

        unsafe {
            Self::update_inner(
                &self.inner,
                self.resources.get_mut(),
                descriptor_writes,
                descriptor_copies,
            )
        };

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn update_unchecked(
        &mut self,
        descriptor_writes: &[WriteDescriptorSet<'_>],
        descriptor_copies: &[CopyDescriptorSet<'_>],
    ) {
        if descriptor_writes.is_empty() && descriptor_copies.is_empty() {
            return;
        }

        unsafe {
            Self::update_inner(
                &self.inner,
                self.resources.get_mut(),
                descriptor_writes,
                descriptor_copies,
            )
        };
    }

    /// Updates the descriptor set with new values.
    ///
    /// # Safety
    ///
    /// - Host access to the descriptor set must be externally synchronized.
    pub unsafe fn update_by_ref(
        &self,
        descriptor_writes: &[WriteDescriptorSet<'_>],
        descriptor_copies: &[CopyDescriptorSet<'_>],
    ) -> Result<(), Box<ValidationError>> {
        if descriptor_writes.is_empty() && descriptor_copies.is_empty() {
            return Ok(());
        }

        self.inner
            .validate_update(descriptor_writes, descriptor_copies)?;

        unsafe {
            Self::update_inner(
                &self.inner,
                &mut self.resources.write(),
                descriptor_writes,
                descriptor_copies,
            )
        };

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn update_by_ref_unchecked(
        &self,
        descriptor_writes: &[WriteDescriptorSet<'_>],
        descriptor_copies: &[CopyDescriptorSet<'_>],
    ) {
        if descriptor_writes.is_empty() && descriptor_copies.is_empty() {
            return;
        }

        unsafe {
            Self::update_inner(
                &self.inner,
                &mut self.resources.write(),
                descriptor_writes,
                descriptor_copies,
            )
        };
    }

    unsafe fn update_inner(
        inner: &RawDescriptorSet,
        resources: &mut DescriptorSetResources,
        descriptor_writes: &[WriteDescriptorSet<'_>],
        descriptor_copies: &[CopyDescriptorSet<'_>],
    ) {
        unsafe { inner.update_unchecked(descriptor_writes, descriptor_copies) };

        for write in descriptor_writes {
            resources.write(write, inner.layout());
        }

        for copy in descriptor_copies {
            resources.copy(copy);
        }
    }

    /// Invalidates descriptors within a descriptor set. Doesn't actually call into Vulkan and only
    /// invalidates the descriptors inside vulkano's resource tracking. Invalidated descriptors are
    /// equivalent to uninitialized descriptors, in that binding a descriptor set to a particular
    /// pipeline requires all shader-accessible descriptors to be valid.
    ///
    /// The intended use-case is an update-after-bind or bindless system, where entries in an
    /// arrayed binding have to be invalidated so that the backing resource will be freed, and
    /// not stay forever referenced until overridden by some update.
    pub fn invalidate(
        &self,
        descriptor_invalidates: &[InvalidateDescriptorSet],
    ) -> Result<(), Box<ValidationError>> {
        self.validate_invalidate(descriptor_invalidates)?;
        self.invalidate_unchecked(descriptor_invalidates);
        Ok(())
    }

    pub fn invalidate_unchecked(&self, descriptor_invalidates: &[InvalidateDescriptorSet]) {
        let mut resources = self.resources.write();
        for invalidate in descriptor_invalidates {
            resources.invalidate(invalidate);
        }
    }

    pub(super) fn validate_invalidate(
        &self,
        descriptor_invalidates: &[InvalidateDescriptorSet],
    ) -> Result<(), Box<ValidationError>> {
        for (index, write) in descriptor_invalidates.iter().enumerate() {
            write
                .validate(self.layout(), self.variable_descriptor_count())
                .map_err(|err| err.add_context(format!("descriptor_writes[{}]", index)))?;
        }
        Ok(())
    }
}

unsafe impl VulkanObject for DescriptorSet {
    type Handle = vk::DescriptorSet;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.inner.handle()
    }
}

unsafe impl DeviceOwned for DescriptorSet {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

impl PartialEq for DescriptorSet {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl Eq for DescriptorSet {}

impl Hash for DescriptorSet {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

/// The resources that are bound to a descriptor set.
#[derive(Clone, Debug)]
pub(crate) struct DescriptorSetResources {
    binding_resources: HashMap<u32, DescriptorBindingResources>,
}

impl DescriptorSetResources {
    /// Creates a new `DescriptorSetResources` matching the provided descriptor set layout, and
    /// all descriptors set to `None`.
    #[inline]
    pub(crate) fn new(layout: &DescriptorSetLayout, variable_descriptor_count: u32) -> Self {
        assert!(variable_descriptor_count <= layout.variable_descriptor_count());

        let binding_resources = layout
            .bindings()
            .iter()
            .map(|binding| {
                let count = if binding
                    .binding_flags
                    .intersects(DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT)
                {
                    variable_descriptor_count
                } else {
                    binding.descriptor_count
                } as usize;

                let binding_resources = match binding.descriptor_type {
                    DescriptorType::UniformBuffer
                    | DescriptorType::StorageBuffer
                    | DescriptorType::UniformBufferDynamic
                    | DescriptorType::StorageBufferDynamic => {
                        DescriptorBindingResources::Buffer(smallvec![None; count])
                    }
                    DescriptorType::UniformTexelBuffer | DescriptorType::StorageTexelBuffer => {
                        DescriptorBindingResources::BufferView(smallvec![None; count])
                    }
                    DescriptorType::SampledImage
                    | DescriptorType::StorageImage
                    | DescriptorType::InputAttachment
                    | DescriptorType::CombinedImageSampler => {
                        DescriptorBindingResources::Image(smallvec![None; count])
                    }
                    DescriptorType::Sampler => {
                        if binding.immutable_samplers.is_empty()
                            || layout
                                .flags()
                                .intersects(DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR)
                        {
                            DescriptorBindingResources::Image(smallvec![None; count])
                        } else {
                            // For descriptor sets that use immutable samplers, all descriptors are
                            // considered valid from the start but only if the descriptor set
                            // doesn't use push descriptors.
                            DescriptorBindingResources::Image(
                                smallvec![Some(OwnedDescriptorImageInfo::default()); count],
                            )
                        }
                    }
                    DescriptorType::InlineUniformBlock => {
                        DescriptorBindingResources::InlineUniformBlock
                    }
                    DescriptorType::AccelerationStructure => {
                        DescriptorBindingResources::AccelerationStructure(smallvec![None; count])
                    }
                };

                (binding.binding, binding_resources)
            })
            .collect();

        Self { binding_resources }
    }

    /// Returns a reference to the bound resources for `binding`. Returns `None` if the binding
    /// doesn't exist.
    #[inline]
    pub(crate) fn binding(&self, binding: u32) -> Option<&DescriptorBindingResources> {
        self.binding_resources.get(&binding)
    }

    #[inline]
    pub(crate) fn write(&mut self, write: &WriteDescriptorSet<'_>, layout: &DescriptorSetLayout) {
        let descriptor_type = layout
            .binding(write.dst_binding())
            .expect("descriptor write has invalid binding number")
            .descriptor_type;
        self.binding_resources
            .get_mut(&write.dst_binding())
            .expect("descriptor write has invalid binding number")
            .write(write, descriptor_type)
    }

    #[inline]
    pub(crate) fn copy(&mut self, copy: &CopyDescriptorSet<'_>) {
        let resources = copy.src_set.resources();
        let src = resources
            .binding_resources
            .get(&copy.src_binding)
            .expect("descriptor copy has invalid src_binding number");
        self.binding_resources
            .get_mut(&copy.dst_binding)
            .expect("descriptor copy has invalid dst_binding number")
            .copy(
                src,
                copy.src_array_element,
                copy.dst_array_element,
                copy.descriptor_count,
            );
    }

    #[inline]
    pub(crate) fn invalidate(&mut self, invalidate: &InvalidateDescriptorSet) {
        self.binding_resources
            .get_mut(&invalidate.binding)
            .expect("descriptor write has invalid binding number")
            .invalidate(invalidate)
    }
}

/// The resources that are bound to a single descriptor set binding.
#[derive(Clone, Debug)]
pub(crate) enum DescriptorBindingResources {
    Image(Elements<OwnedDescriptorImageInfo>),
    Buffer(Elements<OwnedDescriptorBufferInfo>),
    BufferView(Elements<Option<Arc<BufferView>>>),
    InlineUniformBlock,
    AccelerationStructure(Elements<Option<Arc<AccelerationStructure>>>),
}

type Elements<T> = SmallVec<[Option<T>; 1]>;

impl DescriptorBindingResources {
    pub(crate) fn write(
        &mut self,
        write: &WriteDescriptorSet<'_>,
        descriptor_type: DescriptorType,
    ) {
        fn write_resources<T, U>(
            first: usize,
            resources: &mut [Option<U>],
            elements: &[T],
            element_func: impl Fn(&T) -> U,
        ) {
            resources
                .get_mut(first..first + elements.len())
                .expect("descriptor write for binding out of bounds")
                .iter_mut()
                .zip(elements)
                .for_each(|(resource, element)| {
                    *resource = Some(element_func(element));
                });
        }

        let default_image_layout = descriptor_type.default_image_layout();
        let first = write.dst_array_element() as usize;

        match write.elements() {
            WriteDescriptorSetElements::Image(elements) => match self {
                DescriptorBindingResources::Image(resources) => {
                    write_resources(first, resources, elements, |element| {
                        OwnedDescriptorImageInfo {
                            image_layout: if element.image_view.is_some()
                                && element.image_layout == ImageLayout::Undefined
                            {
                                default_image_layout
                            } else {
                                element.image_layout
                            },
                            ..element.to_owned()
                        }
                    })
                }
                _ => panic!(
                    "descriptor write for binding {} has wrong resource type",
                    write.dst_binding(),
                ),
            },
            WriteDescriptorSetElements::Buffer(elements) => match self {
                DescriptorBindingResources::Buffer(resources) => {
                    write_resources(first, resources, elements, DescriptorBufferInfo::to_owned)
                }
                _ => panic!(
                    "descriptor write for binding {} has wrong resource type",
                    write.dst_binding(),
                ),
            },
            WriteDescriptorSetElements::BufferView(elements) => match self {
                DescriptorBindingResources::BufferView(resources) => {
                    write_resources(first, resources, elements, |element| element.cloned())
                }
                _ => panic!(
                    "descriptor write for binding {} has wrong resource type",
                    write.dst_binding(),
                ),
            },
            WriteDescriptorSetElements::InlineUniformBlock(_) => match self {
                DescriptorBindingResources::InlineUniformBlock => (),
                _ => panic!(
                    "descriptor write for binding {} has wrong resource type",
                    write.dst_binding(),
                ),
            },
            WriteDescriptorSetElements::AccelerationStructure(elements) => match self {
                DescriptorBindingResources::AccelerationStructure(resources) => {
                    write_resources(first, resources, elements, |element| element.cloned())
                }
                _ => panic!(
                    "descriptor write for binding {} has wrong resource type",
                    write.dst_binding(),
                ),
            },
        }
    }

    pub(crate) fn copy(
        &mut self,
        src: &DescriptorBindingResources,
        src_start: u32,
        dst_start: u32,
        count: u32,
    ) {
        let src_start = src_start as usize;
        let dst_start = dst_start as usize;
        let count = count as usize;

        match src {
            DescriptorBindingResources::Image(src) => match self {
                DescriptorBindingResources::Image(dst) => dst[dst_start..dst_start + count]
                    .clone_from_slice(&src[src_start..src_start + count]),
                _ => panic!("descriptor copy has wrong resource type"),
            },
            DescriptorBindingResources::Buffer(src) => match self {
                DescriptorBindingResources::Buffer(dst) => dst[dst_start..dst_start + count]
                    .clone_from_slice(&src[src_start..src_start + count]),
                _ => panic!("descriptor copy has wrong resource type"),
            },
            DescriptorBindingResources::BufferView(src) => match self {
                DescriptorBindingResources::BufferView(dst) => dst[dst_start..dst_start + count]
                    .clone_from_slice(&src[src_start..src_start + count]),
                _ => panic!("descriptor copy has wrong resource type"),
            },
            DescriptorBindingResources::InlineUniformBlock => match self {
                DescriptorBindingResources::InlineUniformBlock => (),
                _ => panic!("descriptor copy has wrong resource type"),
            },
            DescriptorBindingResources::AccelerationStructure(src) => match self {
                DescriptorBindingResources::AccelerationStructure(dst) => dst
                    [dst_start..dst_start + count]
                    .clone_from_slice(&src[src_start..src_start + count]),
                _ => panic!("descriptor copy has wrong resource type"),
            },
        }
    }

    pub(crate) fn invalidate(&mut self, invalidate: &InvalidateDescriptorSet) {
        fn invalidate_resources<T: Clone>(
            resources: &mut [Option<T>],
            invalidate: &InvalidateDescriptorSet,
        ) {
            let first = invalidate.first_array_element as usize;
            resources
                .get_mut(first..first + invalidate.descriptor_count as usize)
                .expect("descriptor write for binding out of bounds")
                .iter_mut()
                .for_each(|resource| {
                    *resource = None;
                });
        }

        match self {
            DescriptorBindingResources::Image(resources) => {
                invalidate_resources(resources, invalidate)
            }
            DescriptorBindingResources::Buffer(resources) => {
                invalidate_resources(resources, invalidate)
            }
            DescriptorBindingResources::BufferView(resources) => {
                invalidate_resources(resources, invalidate)
            }
            DescriptorBindingResources::InlineUniformBlock => (),
            DescriptorBindingResources::AccelerationStructure(resources) => {
                invalidate_resources(resources, invalidate)
            }
        }
    }
}

#[derive(Clone)]
pub struct DescriptorSetWithOffsets {
    descriptor_set: Arc<DescriptorSet>,
    dynamic_offsets: SmallVec<[u32; 4]>,
}

impl DescriptorSetWithOffsets {
    pub fn new(
        descriptor_set: Arc<DescriptorSet>,
        dynamic_offsets: impl IntoIterator<Item = u32>,
    ) -> Self {
        Self {
            descriptor_set,
            dynamic_offsets: dynamic_offsets.into_iter().collect(),
        }
    }

    #[inline]
    pub fn as_ref(&self) -> (&Arc<DescriptorSet>, &[u32]) {
        (&self.descriptor_set, &self.dynamic_offsets)
    }

    #[inline]
    pub fn into_tuple(
        self,
    ) -> (
        Arc<DescriptorSet>,
        impl ExactSizeIterator<Item = u32> + use<>,
    ) {
        (self.descriptor_set, self.dynamic_offsets.into_iter())
    }
}

impl From<Arc<DescriptorSet>> for DescriptorSetWithOffsets {
    #[inline]
    fn from(descriptor_set: Arc<DescriptorSet>) -> Self {
        DescriptorSetWithOffsets::new(descriptor_set, std::iter::empty())
    }
}
