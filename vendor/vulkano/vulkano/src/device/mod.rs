//! Communication channel with a physical device.
//!
//! The `Device` is one of the most important objects of Vulkan. Creating a `Device` is required
//! before you can create buffers, textures, shaders, etc.
//!
//! Basic example:
//!
//! ```no_run
//! use vulkano::{
//!     device::{
//!         physical::PhysicalDevice, Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures,
//!         QueueCreateInfo,
//!     },
//!     instance::{Instance, InstanceExtensions},
//!     Version, VulkanLibrary,
//! };
//!
//! // Creating the instance. See the documentation of the `instance` module.
//! let library = unsafe { VulkanLibrary::new() }
//!     .unwrap_or_else(|err| panic!("couldn't load Vulkan library: {:?}", err));
//! let instance = Instance::new(&library, &Default::default())
//!     .unwrap_or_else(|err| panic!("couldn't create instance: {:?}", err));
//!
//! // We just choose the first physical device. In a real application you would choose depending
//! // on the capabilities of the physical device and the user's preferences.
//! let physical_device = instance
//!     .enumerate_physical_devices()
//!     .unwrap_or_else(|err| panic!("couldn't enumerate physical devices: {:?}", err))
//!     .into_iter()
//!     .next()
//!     .expect("no physical device");
//!
//! // Here is the device-creating code.
//! let device = {
//!     let features = DeviceFeatures::empty();
//!     let extensions = DeviceExtensions::empty();
//!
//!     match Device::new(
//!         &physical_device,
//!         &DeviceCreateInfo {
//!             queue_create_infos: &[QueueCreateInfo {
//!                 queue_family_index: 0,
//!                 ..Default::default()
//!             }],
//!             enabled_extensions: &extensions,
//!             enabled_features: &features,
//!             ..Default::default()
//!         },
//!     ) {
//!         Ok(d) => d,
//!         Err(err) => panic!("couldn't build device: {:?}", err),
//!     }
//! };
//! ```
//!
//! # Features and extensions
//!
//! Two of the parameters that you pass to `Device::new` are the list of the features and the list
//! of extensions to enable on the newly-created device.
//!
//! <div class="vulkano-alert-note">
//!
//! > Note
//! >
//! > Device extensions are the same as instance extensions, except for the device. Features are
//! > similar to extensions, except that they are part of the core Vulkan specifications instead of
//! > being separate documents.
//!
//! </div>
//!
//! Some Vulkan capabilities, such as swapchains (that allow you to render on the screen) or
//! geometry shaders for example, require that you enable a certain feature or extension when you
//! create the device. Contrary to OpenGL, you can't use the functions provided by a feature or an
//! extension if you didn't explicitly enable it when creating the device.
//!
//! Not all physical devices support all possible features and extensions. For example mobile
//! devices tend to not support geometry shaders, because their hardware is not capable of it. You
//! can query what is supported with respectively `PhysicalDevice::supported_features` and
//! `DeviceExtensions::supported_by_device`.
//!
//! <div class="vulkano-alert-note">
//!
//! > Note
//! >
//! > The fact that you need to manually enable features at initialization also means that you
//! > don't need to worry about a capability not being supported later on in your code.
//!
//! </div>
//!
//! # Queues
//!
//! Each physical device proposes one or more *queues* that are divided in *queue families*. A
//! queue is a thread of execution to which you can submit commands that the GPU will execute.
//!
//! <div class="vulkano-alert-note">
//!
//! > Note
//! >
//! > You can think of a queue like a CPU thread. Each queue executes its commands one after the
//! > other, and queues run concurrently. A GPU behaves similarly to the hyper-threading
//! > technology, in the sense that queues will only run partially in parallel.
//!
//! </div>
//!
//! The Vulkan API requires that you specify the list of queues that you are going to use at the
//! same time as when you create the device. This is done in vulkano by passing an iterator where
//! each element is a tuple containing a queue family and a number between 0.0 and 1.0 indicating
//! the priority of execution of the queue relative to the others.
//!
//! TODO: write better doc here
//!
//! The `Device::new` function returns the newly-created device, but also the list of queues.
//!
//! # Extended example
//!
//! TODO: write

use self::physical::PhysicalDevice;
pub use self::{
    properties::DeviceProperties,
    queue::{
        DefaultQueueMutex, DeviceQueueInfo, Queue, QueueFamilyProperties, QueueFlags, QueueGuard,
        QueueMutex,
    },
};
pub use crate::fns::DeviceFunctions;
use crate::{
    acceleration_structure::{
        AccelerationStructureBuildGeometryInfo, AccelerationStructureBuildSizesInfo,
        AccelerationStructureBuildType, AccelerationStructureGeometry,
        AccelerationStructureGeometryData,
    },
    buffer::BufferCreateInfo,
    descriptor_set::layout::{DescriptorSetLayoutCreateInfo, DescriptorSetLayoutSupport},
    image::{sys::ImageCreateInfoExtensionsVk, ImageCreateFlags, ImageCreateInfo, ImageTiling},
    instance::{Instance, InstanceExtensions, InstanceOwned, InstanceOwnedDebugWrapper},
    macros::{impl_id_counter, vulkan_bitflags},
    memory::{
        ExternalMemoryHandleType, MemoryFdProperties, MemoryRequirements,
        MemoryWin32HandleProperties,
    },
    RawFd, Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, Version,
    VulkanError, VulkanObject,
};
use ash::vk::TaggedStructure;
use ash::vk::{self, Handle};

use parking_lot::Mutex;
use smallvec::{smallvec, SmallVec};
use std::{
    ffi::{c_char, CStr, CString},
    fmt::{Debug, Error as FmtError, Formatter},
    marker::PhantomData,
    mem::{ManuallyDrop, MaybeUninit},
    num::NonZero,
    ops::Deref,
    ptr, slice,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
};

pub mod physical;
pub mod private_data;
pub(crate) mod properties;
mod queue;

include!(crate::autogen_output!("device_extensions.rs"));
include!(crate::autogen_output!("features.rs"));

/// Represents a Vulkan context.
pub struct Device {
    handle: vk::Device,
    // NOTE: `physical_devices` always contains this.
    physical_device: InstanceOwnedDebugWrapper<Arc<PhysicalDevice>>,
    id: NonZero<u64>,

    enabled_extensions: DeviceExtensions,
    enabled_features: DeviceFeatures,
    physical_devices: SmallVec<[InstanceOwnedDebugWrapper<Arc<PhysicalDevice>>; 2]>,

    // The highest version that is supported for this device.
    // This is the minimum of Instance::max_api_version and PhysicalDevice::api_version.
    api_version: Version,
    fns: DeviceFunctions,
    active_queue_family_indices: SmallVec<[u32; 2]>,

    queue_locks: Mutex<Vec<(NonZero<u64>, Arc<dyn QueueMutex>)>>,

    // This is required for validation in `memory::device_memory`, the count must only be modified
    // in that module.
    pub(crate) allocation_count: AtomicU32,
    fence_pool: Mutex<Vec<vk::Fence>>,
    semaphore_pool: Mutex<Vec<vk::Semaphore>>,
    event_pool: Mutex<Vec<vk::Event>>,

    borrowed: bool,
}

impl Device {
    /// Creates a new `Device`, panicking on a validation error.
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
        physical_device: &Arc<PhysicalDevice>,
        create_info: &DeviceCreateInfo<'_>,
    ) -> Result<(Arc<Device>, Vec<Arc<Queue>>), VulkanError> {
        match Self::try_new(physical_device, create_info) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a new `Device`.
    #[inline]
    pub fn try_new(
        physical_device: &Arc<PhysicalDevice>,
        create_info: &DeviceCreateInfo<'_>,
    ) -> Result<(Arc<Device>, Vec<Arc<Queue>>), Validated<VulkanError>> {
        Self::validate_new(physical_device, create_info)?;

        Ok(unsafe { Self::new_unchecked(physical_device, create_info) }?)
    }

    fn validate_new(
        physical_device: &PhysicalDevice,
        create_info: &DeviceCreateInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        create_info
            .validate(physical_device)
            .map_err(|err| err.add_context("create_info"))?;

        let &DeviceCreateInfo {
            queue_create_infos: _,
            enabled_extensions: _,
            enabled_features: _,
            physical_devices,
            private_data_slot_request_count: _,
            _ne: _,
        } = create_info;

        if !physical_devices.is_empty()
            && !physical_devices
                .iter()
                .any(|p| p.as_ref() == physical_device)
        {
            return Err(Box::new(ValidationError {
                problem: "`create_info.physical_devices` is not empty, but does not contain \
                    `physical_device`"
                    .into(),
                vuids: &["VUID-VkDeviceGroupDeviceCreateInfo-physicalDeviceCount-00377"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        physical_device: &Arc<PhysicalDevice>,
        create_info: &DeviceCreateInfo<'_>,
    ) -> Result<(Arc<Device>, Vec<Arc<Queue>>), VulkanError> {
        unsafe fn create_fn_raw(
            data: *const (),
            create_info_vk: &vk::DeviceCreateInfo<'_>,
        ) -> Result<vk::Device, VulkanError> {
            // SAFETY: The caller must ensure that `data` is the same one we pass below.
            let physical_device = unsafe { &*data.cast::<Arc<PhysicalDevice>>() };

            let fns = physical_device.instance().fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.v1_0.create_device)(
                    physical_device.handle(),
                    create_info_vk,
                    ptr::null(),
                    output.as_mut_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;

            Ok(unsafe { output.assume_init() })
        }

        unsafe {
            Self::new_with_unchecked_inner(
                physical_device,
                create_info,
                <*const _>::cast(physical_device),
                create_fn_raw,
            )
        }
    }

    /// Creates a new `Device` using a custom instance create function, panicking on a validation
    /// error.
    ///
    /// This is a shortcut for `try_new_with().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - `create_fn` must return a valid Vulkan object handle created from `physical_device`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_new_with`] returns a [`ValidationError`].
    ///
    /// [`try_new_with`]: Self::try_new_with
    #[inline]
    #[track_caller]
    pub unsafe fn new_with(
        physical_device: &Arc<PhysicalDevice>,
        create_info: &DeviceCreateInfo<'_>,
        create_fn: impl FnOnce(&vk::DeviceCreateInfo<'_>) -> Result<vk::Device, VulkanError>,
    ) -> Result<(Arc<Device>, Vec<Arc<Queue>>), VulkanError> {
        match unsafe { Self::try_new_with(physical_device, create_info, create_fn) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a new `Device` using a custom instance create function.
    ///
    /// # Safety
    ///
    /// - `create_fn` must return a valid Vulkan object handle created from `physical_device`.
    #[inline]
    pub unsafe fn try_new_with(
        physical_device: &Arc<PhysicalDevice>,
        create_info: &DeviceCreateInfo<'_>,
        create_fn: impl FnOnce(&vk::DeviceCreateInfo<'_>) -> Result<vk::Device, VulkanError>,
    ) -> Result<(Arc<Device>, Vec<Arc<Queue>>), Validated<VulkanError>> {
        Self::validate_new(physical_device, create_info)?;

        Ok(unsafe { Self::new_with_unchecked(physical_device, create_info, create_fn) }?)
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_with_unchecked(
        physical_device: &Arc<PhysicalDevice>,
        create_info: &DeviceCreateInfo<'_>,
        create_fn: impl FnOnce(&vk::DeviceCreateInfo<'_>) -> Result<vk::Device, VulkanError>,
    ) -> Result<(Arc<Device>, Vec<Arc<Queue>>), VulkanError> {
        #[inline]
        unsafe fn helper<F>(
            physical_device: &Arc<PhysicalDevice>,
            create_info: &DeviceCreateInfo<'_>,
            create_fn: ManuallyDrop<F>,
        ) -> Result<(Arc<Device>, Vec<Arc<Queue>>), VulkanError>
        where
            F: FnOnce(&vk::DeviceCreateInfo<'_>) -> Result<vk::Device, VulkanError>,
        {
            unsafe fn create_fn_raw<F>(
                data: *const (),
                create_info_vk: &vk::DeviceCreateInfo<'_>,
            ) -> Result<vk::Device, VulkanError>
            where
                F: FnOnce(&vk::DeviceCreateInfo<'_>) -> Result<vk::Device, VulkanError>,
            {
                // SAFETY: The caller must ensure that `data` is the same one we pass below.
                (unsafe { data.cast::<F>().read() })(create_info_vk)
            }

            unsafe {
                Device::new_with_unchecked_inner(
                    physical_device,
                    create_info,
                    (&raw const create_fn).cast(),
                    create_fn_raw::<F>,
                )
            }
        }

        unsafe { helper(physical_device, create_info, ManuallyDrop::new(create_fn)) }
    }

    unsafe fn new_with_unchecked_inner(
        physical_device: &Arc<PhysicalDevice>,
        create_info: &DeviceCreateInfo<'_>,
        data: *const (),
        create_fn_raw: unsafe fn(
            *const (),
            &vk::DeviceCreateInfo<'_>,
        ) -> Result<vk::Device, VulkanError>,
    ) -> Result<(Arc<Device>, Vec<Arc<Queue>>), VulkanError> {
        let (enabled_extensions, enabled_features) =
            create_info.enable_dependencies(physical_device);

        let create_info = DeviceCreateInfo {
            enabled_extensions: &enabled_extensions,
            enabled_features: &enabled_features,
            ..*create_info
        };

        let handle = {
            let mut features_vk = None;
            let mut features2_vk = None;
            let mut features2_vk_extensions;

            // VUID-VkDeviceCreateInfo-pNext-00373
            if physical_device.instance().api_version() >= Version::V1_1
                || physical_device
                    .instance()
                    .enabled_extensions()
                    .khr_get_physical_device_properties2
            {
                features2_vk_extensions = create_info.enabled_features.to_vk2_extensions(
                    physical_device.api_version(),
                    create_info.enabled_extensions,
                    physical_device.instance().enabled_extensions(),
                );
                features2_vk_extensions.disable_conflicting();
                features2_vk = Some(
                    create_info
                        .enabled_features
                        .to_vk2(&mut features2_vk_extensions),
                );
            } else {
                features_vk = Some(create_info.enabled_features.to_vk());
            };

            let create_info_fields2_vk = create_info.to_vk_fields2();
            let create_info_fields1_vk =
                create_info.to_vk_fields1(&create_info_fields2_vk, features_vk.as_ref());
            let mut create_info_extensions =
                create_info.to_vk_extensions(&create_info_fields1_vk, features2_vk.as_mut());
            let create_info_vk =
                create_info.to_vk(&create_info_fields1_vk, &mut create_info_extensions);

            // SAFETY: We only call the function once, and `data` is the same as what was passed to
            // us.
            unsafe { create_fn_raw(data, &create_info_vk) }?
        };

        let device = unsafe { Self::from_handle(physical_device, handle, &create_info) };

        let mut queues_to_get: SmallVec<[_; 2]> =
            SmallVec::with_capacity(create_info.queue_create_infos.len());

        for queue_create_info in create_info.queue_create_infos {
            let &QueueCreateInfo {
                flags,
                queue_family_index,
                queues,
                _ne: _,
            } = queue_create_info;

            queues_to_get.extend((0..queues.len() as u32).map(move |queue_index| {
                DeviceQueueInfo {
                    flags,
                    queue_family_index,
                    queue_index,
                    ..Default::default()
                }
            }));
        }

        let queues = queues_to_get
            .into_iter()
            .map(|queue_info| unsafe { Queue::new(&device, &queue_info) })
            .collect();

        Ok((device, queues))
    }

    /// Creates a new `Device` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `physical_device`.
    /// - `create_info` must match the info used to create the object. The `queue_create_infos`
    ///   don't have to match exactly: only the `queue_family_index` parameters are used by this
    ///   function.
    /// - `handle` must not be freed, as it will be owned by the returned `Device`.
    pub unsafe fn from_handle(
        physical_device: &Arc<PhysicalDevice>,
        handle: vk::Device,
        create_info: &DeviceCreateInfo<'_>,
    ) -> Arc<Device> {
        unsafe { Self::from_handle_inner(physical_device, handle, create_info, false) }
    }

    /// Creates a new `Device` from a raw object handle, without owning the handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `physical_device`.
    /// - `create_info` must match the info used to create the object. The `queue_create_infos`
    ///   don't have to match exactly: only the `queue_family_index` parameters are used by this
    ///   function.
    /// - `handle` must not be freed while the returned `Device` object is still alive. This means
    ///   all copies of the returned `Arc` must be dropped first. Note `DeviceOwned` objects all
    ///   hold a reference to the device internally.
    pub unsafe fn from_handle_borrowed(
        physical_device: &Arc<PhysicalDevice>,
        handle: vk::Device,
        create_info: &DeviceCreateInfo<'_>,
    ) -> Arc<Device> {
        unsafe { Self::from_handle_inner(physical_device, handle, create_info, true) }
    }

    unsafe fn from_handle_inner(
        physical_device: &Arc<PhysicalDevice>,
        handle: vk::Device,
        create_info: &DeviceCreateInfo<'_>,
        borrowed: bool,
    ) -> Arc<Device> {
        let &DeviceCreateInfo {
            queue_create_infos,
            enabled_features,
            enabled_extensions,
            physical_devices,
            private_data_slot_request_count: _,
            _ne: _,
        } = create_info;

        let api_version = physical_device.api_version();
        let fns = DeviceFunctions::load(|name| {
            unsafe {
                (physical_device.instance().fns().v1_0.get_device_proc_addr)(handle, name.as_ptr())
            }
            .map_or(ptr::null(), |func| func as _)
        });

        let mut active_queue_family_indices: SmallVec<[_; 2]> =
            SmallVec::with_capacity(queue_create_infos.len());

        for queue_create_info in queue_create_infos {
            active_queue_family_indices.push(queue_create_info.queue_family_index);
        }

        active_queue_family_indices.sort_unstable();
        active_queue_family_indices.dedup();

        let physical_devices = if physical_devices.is_empty() {
            smallvec![InstanceOwnedDebugWrapper(physical_device.clone())]
        } else {
            physical_devices
                .iter()
                .copied()
                .cloned()
                .map(InstanceOwnedDebugWrapper)
                .collect()
        };

        Arc::new(Device {
            handle,
            physical_device: InstanceOwnedDebugWrapper(physical_device.clone()),
            id: Self::next_id(),

            enabled_extensions: *enabled_extensions,
            enabled_features: *enabled_features,
            physical_devices,

            api_version,
            fns,
            active_queue_family_indices,

            queue_locks: Mutex::new(Vec::new()),

            allocation_count: AtomicU32::new(0),
            fence_pool: Mutex::new(Vec::new()),
            semaphore_pool: Mutex::new(Vec::new()),
            event_pool: Mutex::new(Vec::new()),

            borrowed,
        })
    }

    /// Returns the Vulkan version supported by the device.
    ///
    /// This is the lower of the
    /// [physical device's supported version](PhysicalDevice::api_version)
    /// and the instance's [`max_api_version`](Instance::max_api_version).
    #[inline]
    pub fn api_version(&self) -> Version {
        self.api_version
    }

    /// Returns pointers to the raw Vulkan functions of the device.
    #[inline]
    pub fn fns(&self) -> &DeviceFunctions {
        &self.fns
    }

    /// Returns the physical device that was used to create this device.
    #[inline]
    pub fn physical_device(&self) -> &Arc<PhysicalDevice> {
        &self.physical_device
    }

    /// Returns the list of physical devices that was used to create this device. The index of
    /// each physical device in this list is its *device index*.
    ///
    /// This always contains the physical device returned by [`physical_device`].
    ///
    /// [`physical_device`]: Self::physical_device
    #[inline]
    pub fn physical_devices(&self) -> &[Arc<PhysicalDevice>] {
        InstanceOwnedDebugWrapper::cast_slice_inner(&self.physical_devices)
    }

    /// Returns a device mask containing all physical devices in this device. In other words:
    /// every bit that corresponds to a physical device in this device is set to 1.
    #[inline]
    pub fn device_mask(&self) -> u32 {
        (1 << self.physical_devices.len() as u32) - 1
    }

    /// Returns the instance used to create this device.
    #[inline]
    pub fn instance(&self) -> &Arc<Instance> {
        self.physical_device.instance()
    }

    /// Returns the queue family indices that this device uses.
    #[inline]
    pub fn active_queue_family_indices(&self) -> &[u32] {
        &self.active_queue_family_indices
    }

    /// Returns the extensions that have been enabled on the device.
    ///
    /// This includes both the extensions specified in [`DeviceCreateInfo::enabled_extensions`],
    /// and any extensions that are required by those extensions.
    #[inline]
    pub fn enabled_extensions(&self) -> &DeviceExtensions {
        &self.enabled_extensions
    }

    /// Returns the features that have been enabled on the device.
    ///
    /// This includes both the features specified in [`DeviceCreateInfo::enabled_features`],
    /// and any features that are required by the enabled extensions.
    #[inline]
    pub fn enabled_features(&self) -> &DeviceFeatures {
        &self.enabled_features
    }

    /// Returns the current number of active [`DeviceMemory`] allocations the device has.
    ///
    /// [`DeviceMemory`]: crate::memory::DeviceMemory
    #[inline]
    pub fn allocation_count(&self) -> u32 {
        self.allocation_count.load(Ordering::Acquire)
    }

    pub(crate) fn fence_pool(&self) -> &Mutex<Vec<vk::Fence>> {
        &self.fence_pool
    }

    pub(crate) fn semaphore_pool(&self) -> &Mutex<Vec<vk::Semaphore>> {
        &self.semaphore_pool
    }

    pub(crate) fn event_pool(&self) -> &Mutex<Vec<vk::Event>> {
        &self.event_pool
    }

    /// For the given acceleration structure build info and primitive counts, returns the
    /// minimum size required to build the acceleration structure, and the minimum size of the
    /// scratch buffer used during the build operation, panicking on a validation error.
    ///
    /// This is a shortcut for `try_acceleration_structure_build_sizes().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_acceleration_structure_build_sizes`] returns a [`ValidationError`].
    ///
    /// [`try_acceleration_structure_build_sizes`]: Self::try_acceleration_structure_build_sizes
    #[inline]
    #[track_caller]
    pub fn acceleration_structure_build_sizes(
        &self,
        build_type: AccelerationStructureBuildType,
        build_info: &AccelerationStructureBuildGeometryInfo<'_>,
        max_primitive_counts: &[u32],
    ) -> AccelerationStructureBuildSizesInfo {
        self.try_acceleration_structure_build_sizes(build_type, build_info, max_primitive_counts)
            .unwrap()
    }

    /// For the given acceleration structure build info and primitive counts, returns the
    /// minimum size required to build the acceleration structure, and the minimum size of the
    /// scratch buffer used during the build operation.
    #[inline]
    pub fn try_acceleration_structure_build_sizes(
        &self,
        build_type: AccelerationStructureBuildType,
        build_info: &AccelerationStructureBuildGeometryInfo<'_>,
        max_primitive_counts: &[u32],
    ) -> Result<AccelerationStructureBuildSizesInfo, Box<ValidationError>> {
        self.validate_acceleration_structure_build_sizes(
            build_type,
            build_info,
            max_primitive_counts,
        )?;

        Ok(unsafe {
            self.acceleration_structure_build_sizes_unchecked(
                build_type,
                build_info,
                max_primitive_counts,
            )
        })
    }

    fn validate_acceleration_structure_build_sizes(
        &self,
        build_type: AccelerationStructureBuildType,
        build_info: &AccelerationStructureBuildGeometryInfo<'_>,
        max_primitive_counts: &[u32],
    ) -> Result<(), Box<ValidationError>> {
        if !self.enabled_extensions().khr_acceleration_structure {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "khr_acceleration_structure",
                )])]),
                ..Default::default()
            }));
        }

        if !self.enabled_features().acceleration_structure {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "acceleration_structure",
                )])]),
                vuids: &[
                    "VUID-vkGetAccelerationStructureBuildSizesKHR-accelerationStructure-08933",
                ],
                ..Default::default()
            }));
        }

        build_type.validate_device(self).map_err(|err| {
            err.add_context("build_type")
                .set_vuids(&["VUID-vkGetAccelerationStructureBuildSizesKHR-buildType-parameter"])
        })?;

        // VUID-vkGetAccelerationStructureBuildSizesKHR-pBuildInfo-parameter
        build_info
            .validate(self)
            .map_err(|err| err.add_context("build_info"))?;

        let mut total_primitive_count = 0;
        let max_primitive_count = self
            .physical_device()
            .properties()
            .max_primitive_count
            .unwrap();
        let max_instance_count = self
            .physical_device()
            .properties()
            .max_instance_count
            .unwrap();

        for (geometry, &primitive_count) in build_info.geometries.iter().zip(max_primitive_counts) {
            let AccelerationStructureGeometry {
                geometry,
                flags: _,
                _ne: _,
            } = geometry;

            total_primitive_count += primitive_count as u64;

            match geometry {
                AccelerationStructureGeometryData::Triangles(_triangles_data) => {
                    if total_primitive_count > max_primitive_count {
                        return Err(Box::new(ValidationError {
                            context: "info.geometries".into(),
                            problem: "the total number of triangles exceeds the \
                                `max_primitive_count` limit"
                                .into(),
                            vuids: &["VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03795"],
                            ..Default::default()
                        }));
                    }
                }
                AccelerationStructureGeometryData::Aabbs(_aabbs_data) => {
                    if total_primitive_count > max_primitive_count {
                        return Err(Box::new(ValidationError {
                            problem: "the total number of AABBs exceeds the `max_primitive_count` \
                                limit"
                                .into(),
                            vuids: &["VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03794"],
                            ..Default::default()
                        }));
                    }
                }
                AccelerationStructureGeometryData::Instances(_instances_data) => {
                    if primitive_count as u64 > max_instance_count {
                        return Err(Box::new(ValidationError {
                            context: "build_range_infos[0].primitive_count".into(),
                            problem: "exceeds the the `max_instance_count` limit".into(),
                            vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03801"],
                            ..Default::default()
                        }));
                    }
                }
            }
        }

        if build_info.geometries.len() != max_primitive_counts.len() {
            return Err(Box::new(ValidationError {
                problem: "`build_info.geometries` and `max_primitive_counts` do not have the same \
                    length"
                    .into(),
                vuids: &["VUID-vkGetAccelerationStructureBuildSizesKHR-pMaxPrimitiveCounts-11612"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn acceleration_structure_build_sizes_unchecked(
        &self,
        build_type: AccelerationStructureBuildType,
        build_info: &AccelerationStructureBuildGeometryInfo<'_>,
        max_primitive_counts: &[u32],
    ) -> AccelerationStructureBuildSizesInfo {
        let build_info_fields1_vk = build_info.to_vk_fields1();
        let build_info_vk = build_info.to_vk(&build_info_fields1_vk);

        let mut build_sizes_info_vk = AccelerationStructureBuildSizesInfo::to_mut_vk();

        let fns = self.fns();
        unsafe {
            (fns.khr_acceleration_structure
                .get_acceleration_structure_build_sizes_khr)(
                self.handle,
                build_type.into(),
                &build_info_vk,
                max_primitive_counts.as_ptr(),
                &mut build_sizes_info_vk,
            )
        };

        AccelerationStructureBuildSizesInfo::from_vk(&build_sizes_info_vk)
    }

    /// Returns whether a serialized acceleration structure with the specified version data
    /// is compatible with this device, panicking on a validation error.
    ///
    /// This is a shortcut for
    /// `try_acceleration_structure_is_compatible().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_acceleration_structure_is_compatible`] returns a [`ValidationError`].
    ///
    /// [`try_acceleration_structure_is_compatible`]: Self::try_acceleration_structure_is_compatible
    #[inline]
    #[track_caller]
    pub fn acceleration_structure_is_compatible(
        &self,
        version_data: &[u8; 2 * vk::UUID_SIZE],
    ) -> bool {
        self.try_acceleration_structure_is_compatible(version_data)
            .unwrap()
    }

    /// Returns whether a serialized acceleration structure with the specified version data
    /// is compatible with this device.
    #[inline]
    pub fn try_acceleration_structure_is_compatible(
        &self,
        version_data: &[u8; 2 * vk::UUID_SIZE],
    ) -> Result<bool, Box<ValidationError>> {
        self.validate_acceleration_structure_is_compatible(version_data)?;

        Ok(unsafe { self.acceleration_structure_is_compatible_unchecked(version_data) })
    }

    fn validate_acceleration_structure_is_compatible(
        &self,
        _version_data: &[u8; 2 * vk::UUID_SIZE],
    ) -> Result<(), Box<ValidationError>> {
        if !self.enabled_extensions().khr_acceleration_structure {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "khr_acceleration_structure",
                )])]),
                ..Default::default()
            }));
        }

        if !self.enabled_features().acceleration_structure {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::DeviceFeature("acceleration_structure")]),
                ]),
                vuids: &["VUID-vkGetDeviceAccelerationStructureCompatibilityKHR-accelerationStructure-08928"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn acceleration_structure_is_compatible_unchecked(
        &self,
        version_data: &[u8; 2 * vk::UUID_SIZE],
    ) -> bool {
        let version_info_vk =
            vk::AccelerationStructureVersionInfoKHR::default().version_data(version_data);
        let mut compatibility_vk = vk::AccelerationStructureCompatibilityKHR::default();

        let fns = self.fns();
        unsafe {
            (fns.khr_acceleration_structure
                .get_device_acceleration_structure_compatibility_khr)(
                self.handle,
                &version_info_vk,
                &mut compatibility_vk,
            )
        };

        compatibility_vk == vk::AccelerationStructureCompatibilityKHR::COMPATIBLE
    }

    /// Returns whether a descriptor set layout with the given `create_info` could be created
    /// on the device, and additional supported properties where relevant, panicking on a
    /// validation error. `Some` is returned if the descriptor set layout is supported, `None` if
    /// it is not.
    ///
    /// This is primarily useful for checking whether the device supports a descriptor set layout
    /// that goes beyond the [`max_per_set_descriptors`] limit. A layout that does not exceed
    /// that limit is guaranteed to be supported, otherwise this function can be called.
    ///
    /// The device API version must be at least 1.1, or the [`khr_maintenance3`] extension must
    /// be enabled on the device.
    ///
    /// This is a shortcut for `try_descriptor_set_layout_support().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_descriptor_set_layout_support`] returns a [`ValidationError`].
    ///
    /// [`max_per_set_descriptors`]: crate::device::DeviceProperties::max_per_set_descriptors
    /// [`khr_maintenance3`]: crate::device::DeviceExtensions::khr_maintenance3
    /// [`try_descriptor_set_layout_support`]: Self::try_descriptor_set_layout_support
    #[inline]
    #[track_caller]
    pub fn descriptor_set_layout_support(
        &self,
        create_info: &DescriptorSetLayoutCreateInfo<'_>,
    ) -> Option<DescriptorSetLayoutSupport> {
        self.try_descriptor_set_layout_support(create_info).unwrap()
    }

    /// Returns whether a descriptor set layout with the given `create_info` could be created
    /// on the device, and additional supported properties where relevant. `Some` is returned if
    /// the descriptor set layout is supported, `None` if it is not.
    ///
    /// This is primarily useful for checking whether the device supports a descriptor set layout
    /// that goes beyond the [`max_per_set_descriptors`] limit. A layout that does not exceed
    /// that limit is guaranteed to be supported, otherwise this function can be called.
    ///
    /// The device API version must be at least 1.1, or the [`khr_maintenance3`] extension must
    /// be enabled on the device.
    ///
    /// [`max_per_set_descriptors`]: crate::device::DeviceProperties::max_per_set_descriptors
    /// [`khr_maintenance3`]: crate::device::DeviceExtensions::khr_maintenance3
    #[inline]
    pub fn try_descriptor_set_layout_support(
        &self,
        create_info: &DescriptorSetLayoutCreateInfo<'_>,
    ) -> Result<Option<DescriptorSetLayoutSupport>, Box<ValidationError>> {
        self.validate_descriptor_set_layout_support(create_info)?;

        Ok(unsafe { self.descriptor_set_layout_support_unchecked(create_info) })
    }

    fn validate_descriptor_set_layout_support(
        &self,
        create_info: &DescriptorSetLayoutCreateInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        if !(self.api_version() >= Version::V1_1 || self.enabled_extensions().khr_maintenance3) {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                    RequiresAllOf(&[Requires::DeviceExtension("khr_maintenance3")]),
                ]),
                ..Default::default()
            }));
        }

        // VUID-vkGetDescriptorSetLayoutSupport-pCreateInfo-parameter
        create_info
            .validate(self)
            .map_err(|err| err.add_context("create_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn descriptor_set_layout_support_unchecked(
        &self,
        create_info: &DescriptorSetLayoutCreateInfo<'_>,
    ) -> Option<DescriptorSetLayoutSupport> {
        let create_info_fields2_vk = create_info.to_vk_fields2();
        let create_info_fields1_vk = create_info.to_vk_fields1(&create_info_fields2_vk);
        let mut create_info_extensions_vk = create_info.to_vk_extensions(&create_info_fields2_vk);
        let create_info_vk =
            create_info.to_vk(&create_info_fields1_vk, &mut create_info_extensions_vk);

        let mut support_extensions_vk = DescriptorSetLayoutSupport::to_mut_vk_extensions(self);
        let mut support_vk = DescriptorSetLayoutSupport::to_mut_vk(&mut support_extensions_vk);

        let fns = self.fns();

        if self.api_version() >= Version::V1_1 {
            unsafe {
                (fns.v1_1.get_descriptor_set_layout_support)(
                    self.handle(),
                    &create_info_vk,
                    &mut support_vk,
                )
            }
        } else {
            unsafe {
                (fns.khr_maintenance3.get_descriptor_set_layout_support_khr)(
                    self.handle(),
                    &create_info_vk,
                    &mut support_vk,
                )
            }
        }

        // Unborrow
        let support_vk = vk::DescriptorSetLayoutSupport {
            _marker: PhantomData,
            ..support_vk
        };

        DescriptorSetLayoutSupport::from_vk(&support_vk, &support_extensions_vk)
    }

    /// Returns the memory requirements that would apply for a buffer created with the specified
    /// `create_info`, panicking on a validation error.
    ///
    /// The device API version must be at least 1.3, or the [`khr_maintenance4`] extension must
    /// be enabled on the device.
    ///
    /// This is a shortcut for `try_buffer_memory_requirements().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_buffer_memory_requirements`] returns a [`ValidationError`].
    ///
    /// [`khr_maintenance4`]: DeviceExtensions::khr_maintenance4
    /// [`try_buffer_memory_requirements`]: Self::try_buffer_memory_requirements
    #[inline]
    #[track_caller]
    pub fn buffer_memory_requirements(
        &self,
        create_info: &BufferCreateInfo<'_>,
    ) -> MemoryRequirements {
        self.try_buffer_memory_requirements(create_info).unwrap()
    }

    /// Returns the memory requirements that would apply for a buffer created with the specified
    /// `create_info`.
    ///
    /// The device API version must be at least 1.3, or the [`khr_maintenance4`] extension must
    /// be enabled on the device.
    ///
    /// [`khr_maintenance4`]: DeviceExtensions::khr_maintenance4
    #[inline]
    pub fn try_buffer_memory_requirements(
        &self,
        create_info: &BufferCreateInfo<'_>,
    ) -> Result<MemoryRequirements, Box<ValidationError>> {
        self.validate_buffer_memory_requirements(create_info)?;

        Ok(unsafe { self.buffer_memory_requirements_unchecked(create_info) })
    }

    fn validate_buffer_memory_requirements(
        &self,
        create_info: &BufferCreateInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        if !(self.api_version() >= Version::V1_3 || self.enabled_extensions().khr_maintenance4) {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::DeviceExtension("khr_maintenance")]),
                ]),
                ..Default::default()
            }));
        }

        create_info
            .validate(self)
            .map_err(|err| err.add_context("create_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn buffer_memory_requirements_unchecked(
        &self,
        create_info: &BufferCreateInfo<'_>,
    ) -> MemoryRequirements {
        let mut extensions_vk = create_info.to_vk_extensions();
        let create_info_vk = create_info.to_vk(&mut extensions_vk);

        let info_vk = vk::DeviceBufferMemoryRequirements::default().create_info(&create_info_vk);

        let mut memory_requirements2_extensions_vk =
            MemoryRequirements::to_mut_vk2_extensions(self);
        let mut memory_requirements2_vk =
            MemoryRequirements::to_mut_vk2(&mut memory_requirements2_extensions_vk);

        let fns = self.fns();

        if self.api_version() >= Version::V1_3 {
            unsafe {
                (fns.v1_3.get_device_buffer_memory_requirements)(
                    self.handle(),
                    &info_vk,
                    &mut memory_requirements2_vk,
                )
            };
        } else {
            debug_assert!(self.enabled_extensions().khr_maintenance4);
            unsafe {
                (fns.khr_maintenance4
                    .get_device_buffer_memory_requirements_khr)(
                    self.handle(),
                    &info_vk,
                    &mut memory_requirements2_vk,
                )
            };
        }

        // Unborrow
        let memory_requirements2_vk = vk::MemoryRequirements2 {
            _marker: PhantomData,
            ..memory_requirements2_vk
        };

        MemoryRequirements::from_vk2(
            &memory_requirements2_vk,
            &memory_requirements2_extensions_vk,
        )
    }

    /// Returns the memory requirements that would apply for an image created with the specified
    /// `create_info`, panicking on a validation error.
    ///
    /// If `create_info.flags` contains [`ImageCreateFlags::DISJOINT`], then `plane` must specify
    /// the plane number of the format or memory plane (depending on tiling) that memory
    /// requirements will be returned for. Otherwise, `plane` must be `None`.
    ///
    /// The device API version must be at least 1.3, or the [`khr_maintenance4`] extension must
    /// be enabled on the device.
    ///
    /// This is a shortcut for `try_image_memory_requirements().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_image_memory_requirements`] returns a [`ValidationError`].
    ///
    /// [`khr_maintenance4`]: DeviceExtensions::khr_maintenance4
    /// [`try_image_memory_requirements`]: Self::try_image_memory_requirements
    #[inline]
    #[track_caller]
    pub fn image_memory_requirements(
        &self,
        create_info: &ImageCreateInfo<'_>,
        plane: Option<usize>,
    ) -> MemoryRequirements {
        self.try_image_memory_requirements(create_info, plane)
            .unwrap()
    }

    /// Returns the memory requirements that would apply for an image created with the specified
    /// `create_info`.
    ///
    /// If `create_info.flags` contains [`ImageCreateFlags::DISJOINT`], then `plane` must specify
    /// the plane number of the format or memory plane (depending on tiling) that memory
    /// requirements will be returned for. Otherwise, `plane` must be `None`.
    ///
    /// The device API version must be at least 1.3, or the [`khr_maintenance4`] extension must
    /// be enabled on the device.
    ///
    /// [`khr_maintenance4`]: DeviceExtensions::khr_maintenance4
    #[inline]
    pub fn try_image_memory_requirements(
        &self,
        create_info: &ImageCreateInfo<'_>,
        plane: Option<usize>,
    ) -> Result<MemoryRequirements, Box<ValidationError>> {
        self.validate_image_memory_requirements(create_info, plane)?;

        Ok(unsafe { self.image_memory_requirements_unchecked(create_info, plane) })
    }

    fn validate_image_memory_requirements(
        &self,
        create_info: &ImageCreateInfo<'_>,
        plane: Option<usize>,
    ) -> Result<(), Box<ValidationError>> {
        if !(self.api_version() >= Version::V1_3 || self.enabled_extensions().khr_maintenance4) {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::DeviceExtension("khr_maintenance")]),
                ]),
                ..Default::default()
            }));
        }

        create_info
            .validate(self)
            .map_err(|err| err.add_context("create_info"))?;

        let &ImageCreateInfo {
            flags,
            image_type: _,
            format,
            view_formats: _,
            extent: _,
            array_layers: _,
            mip_levels: _,
            samples: _,
            tiling,
            usage: _,
            stencil_usage: _,
            sharing: _,
            initial_layout: _,
            drm_format_modifiers,
            drm_format_modifier_plane_layouts: _,
            external_memory_handle_types: _,
            _ne: _,
        } = create_info;

        if flags.intersects(ImageCreateFlags::DISJOINT) {
            let Some(plane) = plane else {
                return Err(Box::new(ValidationError {
                    problem: "`create_info.flags` contains `ImageCreateFlags::DISJOINT`, but \
                        `plane` is `None`"
                        .into(),
                    vuids: &[
                        "VUID-VkDeviceImageMemoryRequirements-pCreateInfo-06419",
                        "VUID-VkDeviceImageMemoryRequirements-pCreateInfo-06420",
                    ],
                    ..Default::default()
                }));
            };

            match tiling {
                ImageTiling::Linear | ImageTiling::Optimal => {
                    if plane >= format.planes().len() {
                        return Err(Box::new(ValidationError {
                            problem:
                                "`create_info.tiling` is not `ImageTiling::DrmFormatModifier`, \
                                but `plane` is not less than the number of planes in \
                                `create_info.format`"
                                    .into(),
                            vuids: &["VUID-VkDeviceImageMemoryRequirements-pCreateInfo-06419"],
                            ..Default::default()
                        }));
                    }
                }
                ImageTiling::DrmFormatModifier => {
                    // TODO: handle the case where `drm_format_modifiers` contains multiple
                    // elements. See: https://github.com/KhronosGroup/Vulkan-Docs/issues/2309

                    if let &[drm_format_modifier] = drm_format_modifiers {
                        let format_properties =
                            unsafe { self.physical_device.format_properties_unchecked(format) };
                        let drm_format_modifier_properties = format_properties
                            .drm_format_modifier_properties
                            .iter()
                            .find(|properties| {
                                properties.drm_format_modifier == drm_format_modifier
                            })
                            .unwrap();

                        if plane
                            >= drm_format_modifier_properties.drm_format_modifier_plane_count
                                as usize
                        {
                            return Err(Box::new(ValidationError {
                                problem: "`create_info.drm_format_modifiers` has a length of 1, \
                                    but `plane` is not less than `DrmFormatModifierProperties::\
                                    drm_format_modifier_plane_count` for \
                                    `drm_format_modifiers[0]`, as returned by \
                                    `PhysicalDevice::format_properties` for `format`"
                                    .into(),
                                vuids: &["VUID-VkDeviceImageMemoryRequirements-pCreateInfo-06420"],
                                ..Default::default()
                            }));
                        }
                    }
                }
            }
        } else if plane.is_some() {
            return Err(Box::new(ValidationError {
                problem: "`create_info.flags` does not contain `ImageCreateFlags::DISJOINT`, but \
                    `plane` is `Some`"
                    .into(),
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn image_memory_requirements_unchecked(
        &self,
        create_info: &ImageCreateInfo<'_>,
        plane: Option<usize>,
    ) -> MemoryRequirements {
        let create_info_fields1_vk = create_info.to_vk_fields1();
        let mut create_info_extensions_vk = ImageCreateInfoExtensionsVk {
            drm_format_modifier_explicit_vk: None,
            ..create_info.to_vk_extensions(&create_info_fields1_vk)
        };
        let create_info_vk = create_info.to_vk(&mut create_info_extensions_vk);

        // This is currently necessary because of an issue with the spec. The plane aspect should
        // only be needed if the image is disjoint, but the spec currently demands a valid aspect
        // even for non-disjoint DRM format modifier images.
        // See: https://github.com/KhronosGroup/Vulkan-Docs/issues/2309
        // Replace this variable with vk::ImageAspectFlags::NONE when resolved.
        let default_aspect = if create_info.tiling == ImageTiling::DrmFormatModifier {
            // Hopefully valid for any DrmFormatModifier image?
            vk::ImageAspectFlags::MEMORY_PLANE_0_EXT
        } else {
            vk::ImageAspectFlags::NONE
        };
        let plane_aspect = plane.map_or(default_aspect, |plane| match create_info.tiling {
            ImageTiling::Optimal | ImageTiling::Linear => match plane {
                0 => vk::ImageAspectFlags::PLANE_0,
                1 => vk::ImageAspectFlags::PLANE_1,
                2 => vk::ImageAspectFlags::PLANE_2,
                _ => unreachable!(),
            },
            ImageTiling::DrmFormatModifier => match plane {
                0 => vk::ImageAspectFlags::MEMORY_PLANE_0_EXT,
                1 => vk::ImageAspectFlags::MEMORY_PLANE_1_EXT,
                2 => vk::ImageAspectFlags::MEMORY_PLANE_2_EXT,
                3 => vk::ImageAspectFlags::MEMORY_PLANE_3_EXT,
                _ => unreachable!(),
            },
        });

        let info_vk = vk::DeviceImageMemoryRequirements::default()
            .create_info(&create_info_vk)
            .plane_aspect(plane_aspect);

        let mut memory_requirements2_extensions_vk =
            MemoryRequirements::to_mut_vk2_extensions(self);
        let mut memory_requirements2_vk =
            MemoryRequirements::to_mut_vk2(&mut memory_requirements2_extensions_vk);

        let fns = self.fns();

        if self.api_version() >= Version::V1_3 {
            unsafe {
                (fns.v1_3.get_device_image_memory_requirements)(
                    self.handle(),
                    &info_vk,
                    &mut memory_requirements2_vk,
                )
            };
        } else {
            debug_assert!(self.enabled_extensions().khr_maintenance4);
            unsafe {
                (fns.khr_maintenance4
                    .get_device_image_memory_requirements_khr)(
                    self.handle(),
                    &info_vk,
                    &mut memory_requirements2_vk,
                )
            };
        }

        // Unborrow
        let memory_requirements2_vk = vk::MemoryRequirements2 {
            _marker: PhantomData,
            ..memory_requirements2_vk
        };

        MemoryRequirements::from_vk2(
            &memory_requirements2_vk,
            &memory_requirements2_extensions_vk,
        )
    }

    // TODO: image_sparse_memory_requirements

    /// Retrieves the properties of an external file descriptor when imported as a given external
    /// handle type, panicking on a validation error.
    ///
    /// An error will be returned if the [`khr_external_memory_fd`] extension was not enabled on
    /// the device, or if `handle_type` is [`ExternalMemoryHandleType::OpaqueFd`].
    ///
    /// This is a shortcut for `try_memory_fd_properties().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - `fd` must be a handle to external memory that was created outside the Vulkan API.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_memory_fd_properties`] returns a [`ValidationError`].
    ///
    /// [`khr_external_memory_fd`]: DeviceExtensions::khr_external_memory_fd
    /// [`try_memory_fd_properties`]: Self::try_memory_fd_properties
    #[inline]
    #[track_caller]
    pub unsafe fn memory_fd_properties(
        &self,
        handle_type: ExternalMemoryHandleType,
        fd: RawFd,
    ) -> Result<MemoryFdProperties, VulkanError> {
        match unsafe { self.try_memory_fd_properties(handle_type, fd) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Retrieves the properties of an external file descriptor when imported as a given external
    /// handle type.
    ///
    /// An error will be returned if the [`khr_external_memory_fd`] extension was not enabled on
    /// the device, or if `handle_type` is [`ExternalMemoryHandleType::OpaqueFd`].
    ///
    /// # Safety
    ///
    /// - `fd` must be a handle to external memory that was created outside the Vulkan API.
    ///
    /// [`khr_external_memory_fd`]: DeviceExtensions::khr_external_memory_fd
    #[inline]
    pub unsafe fn try_memory_fd_properties(
        &self,
        handle_type: ExternalMemoryHandleType,
        fd: RawFd,
    ) -> Result<MemoryFdProperties, Validated<VulkanError>> {
        self.validate_memory_fd_properties(handle_type, fd)?;

        Ok(unsafe { self.memory_fd_properties_unchecked(handle_type, fd) }?)
    }

    fn validate_memory_fd_properties(
        &self,
        handle_type: ExternalMemoryHandleType,
        _fd: RawFd,
    ) -> Result<(), Box<ValidationError>> {
        if !self.enabled_extensions().khr_external_memory_fd {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "khr_external_memory_fd",
                )])]),
                ..Default::default()
            }));
        }

        handle_type.validate_device(self).map_err(|err| {
            err.add_context("handle_type")
                .set_vuids(&["VUID-vkGetMemoryFdPropertiesKHR-handleType-parameter"])
        })?;

        if handle_type == ExternalMemoryHandleType::OpaqueFd {
            return Err(Box::new(ValidationError {
                context: "handle_type".into(),
                problem: "is `ExternalMemoryHandleType::OpaqueFd`".into(),
                vuids: &["VUID-vkGetMemoryFdPropertiesKHR-handleType-00674"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn memory_fd_properties_unchecked(
        &self,
        handle_type: ExternalMemoryHandleType,
        fd: RawFd,
    ) -> Result<MemoryFdProperties, VulkanError> {
        let mut memory_fd_properties = MemoryFdProperties::to_mut_vk();

        let fns = self.fns();
        unsafe {
            (fns.khr_external_memory_fd.get_memory_fd_properties_khr)(
                self.handle,
                handle_type.into(),
                fd,
                &mut memory_fd_properties,
            )
        }
        .result()
        .map_err(VulkanError::from)?;

        Ok(MemoryFdProperties::from_vk(&memory_fd_properties))
    }

    /// Retrieves the properties of a Windows external memory handle when imported as a given
    /// external handle type, panicking on a validation error.
    ///
    /// The [`khr_external_memory_fd`] extension must be enabled on the device and `handle_type`
    /// must not be one of the `Opaque` handle types.
    ///
    /// This is a shortcut for `try_memory_win32_handle_properties().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid Windows handle to external memory.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_memory_win32_handle_properties`] returns a [`ValidationError`].
    ///
    /// [`khr_external_memory_fd`]: DeviceExtensions::khr_external_memory_fd
    /// [`try_memory_win32_handle_properties`]: Self::try_memory_win32_handle_properties
    #[inline]
    #[track_caller]
    pub unsafe fn memory_win32_handle_properties(
        &self,
        handle_type: ExternalMemoryHandleType,
        handle: vk::HANDLE,
    ) -> Result<MemoryWin32HandleProperties, VulkanError> {
        match unsafe { self.try_memory_win32_handle_properties(handle_type, handle) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Retrieves the properties of a Windows external memory handle when imported as a given
    /// external handle type.
    ///
    /// The [`khr_external_memory_fd`] extension must be enabled on the device and `handle_type`
    /// must not be one of the `Opaque` handle types.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid Windows handle to external memory.
    ///
    /// [`khr_external_memory_fd`]: DeviceExtensions::khr_external_memory_fd
    #[inline]
    pub unsafe fn try_memory_win32_handle_properties(
        &self,
        handle_type: ExternalMemoryHandleType,
        handle: vk::HANDLE,
    ) -> Result<MemoryWin32HandleProperties, Validated<VulkanError>> {
        self.validate_memory_win32_handle_properties(handle_type)?;

        Ok(unsafe { self.memory_win32_handle_properties_unchecked(handle_type, handle)? })
    }

    fn validate_memory_win32_handle_properties(
        &self,
        handle_type: ExternalMemoryHandleType,
    ) -> Result<(), Box<ValidationError>> {
        let enabled_extensions = self.enabled_extensions();

        if !enabled_extensions.khr_external_memory_win32 {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "khr_external_memory_win32",
                )])]),
                vuids: &["VUID-vkGetMemoryWin32HandlePropertiesKHR-device-parameter"],
                ..Default::default()
            }));
        }

        if handle_type == ExternalMemoryHandleType::OpaqueWin32
            || handle_type == ExternalMemoryHandleType::OpaqueWin32Kmt
            || handle_type == ExternalMemoryHandleType::OpaqueFd
        {
            return Err(Box::new(ValidationError {
                context: "handle_type".into(),
                problem: "is `ExternalMemoryHandleType::OpaqueWin32`, \
                    `ExternalMemoryHandleType::OpaqueWin32Kmt` or \
                    `ExternalMemoryHandleType::OpaqueFd`"
                    .into(),
                vuids: &["VUID-vkGetMemoryWin32HandlePropertiesKHR-handleType-00666"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn memory_win32_handle_properties_unchecked(
        &self,
        handle_type: ExternalMemoryHandleType,
        handle: vk::HANDLE,
    ) -> Result<MemoryWin32HandleProperties, VulkanError> {
        let fns = self.fns();
        let mut memory_win32_handle_properties = MemoryWin32HandleProperties::to_mut_vk();

        unsafe {
            (fns.khr_external_memory_win32
                .get_memory_win32_handle_properties_khr)(
                self.handle(),
                handle_type.into(),
                handle,
                &mut memory_win32_handle_properties,
            )
        }
        .result()
        .map_err(VulkanError::from)?;

        Ok(MemoryWin32HandleProperties::from_vk(
            &memory_win32_handle_properties,
        ))
    }

    /// Assigns a human-readable name to `object` for debugging purposes, panicking on a validation
    /// error.
    ///
    /// The [`ext_debug_utils`] extension must be enabled on the instance.
    ///
    /// If `object_name` is `None`, a previously set object name is removed.
    ///
    /// This is a shortcut for `try_set_debug_utils_object_name().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - `object` must not be used concurrently by any other Vulkan command.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_set_debug_utils_object_name`] returns a [`ValidationError`].
    ///
    /// [`ext_debug_utils`]: crate::instance::InstanceExtensions::ext_debug_utils
    /// [`try_set_debug_utils_object_name`]: Self::try_set_debug_utils_object_name
    #[inline]
    #[track_caller]
    pub unsafe fn set_debug_utils_object_name<T: VulkanObject + DeviceOwned>(
        &self,
        object: &T,
        object_name: Option<&str>,
    ) -> Result<(), VulkanError> {
        match unsafe { self.try_set_debug_utils_object_name(object, object_name) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Assigns a human-readable name to `object` for debugging purposes.
    ///
    /// The [`ext_debug_utils`] extension must be enabled on the instance.
    ///
    /// If `object_name` is `None`, a previously set object name is removed.
    ///
    /// # Safety
    ///
    /// - `object` must not be used concurrently by any other Vulkan command.
    ///
    /// [`ext_debug_utils`]: crate::instance::InstanceExtensions::ext_debug_utils
    #[inline]
    pub unsafe fn try_set_debug_utils_object_name<T: VulkanObject + DeviceOwned>(
        &self,
        object: &T,
        object_name: Option<&str>,
    ) -> Result<(), Validated<VulkanError>> {
        self.validate_set_debug_utils_object_name(object)?;

        Ok(unsafe { self.set_debug_utils_object_name_unchecked(object, object_name) }?)
    }

    fn validate_set_debug_utils_object_name<T: VulkanObject + DeviceOwned>(
        &self,
        object: &T,
    ) -> Result<(), Box<ValidationError>> {
        // VUID-vkSetDebugUtilsObjectNameEXT-pNameInfo-07874
        assert_eq!(object.device().handle(), self.handle());

        if !self.instance().enabled_extensions().ext_debug_utils {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "ext_debug_utils",
                )])]),
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_debug_utils_object_name_unchecked<T: VulkanObject + DeviceOwned>(
        &self,
        object: &T,
        object_name: Option<&str>,
    ) -> Result<(), VulkanError> {
        let object_name_vk = object_name.map(|object_name| CString::new(object_name).unwrap());
        let mut info_vk = vk::DebugUtilsObjectNameInfoEXT::default().object_handle(object.handle());

        if let Some(object_name_vk) = &object_name_vk {
            info_vk = info_vk.object_name(object_name_vk);
        }

        let fns = self.fns();
        unsafe { (fns.ext_debug_utils.set_debug_utils_object_name_ext)(self.handle, &info_vk) }
            .result()
            .map_err(VulkanError::from)?;

        Ok(())
    }

    /// Waits until all work on this device has finished. You should never need to call
    /// this function, but it can be useful for debugging or benchmarking purposes.
    ///
    /// <div class="vulkano-alert-note">
    ///
    /// > Note
    /// >
    /// > This is the Vulkan equivalent of OpenGL's `glFinish`.
    ///
    /// </div>
    pub fn wait_idle(&self) -> Result<(), VulkanError> {
        struct UnlockGuard<'a> {
            queue_locks: &'a [(NonZero<u64>, Arc<dyn QueueMutex>)],
            locked_count: usize,
        }

        impl Drop for UnlockGuard<'_> {
            fn drop(&mut self) {
                for (_, queue_lock) in &self.queue_locks[..self.locked_count] {
                    // SAFETY: The code below ensures that the `locked_count` is incremented after
                    // a lock is locked such that our `..locked_count` range always denotes the
                    // locked locks. Notably, this ensures that locks are soundly unlocked even if
                    // one of the locks in the middle panics.
                    unsafe { queue_lock.unlock() };
                }
            }
        }

        let mut unlock_guard = UnlockGuard {
            queue_locks: &self.queue_locks.lock(),
            locked_count: 0,
        };

        for (_, queue_lock) in unlock_guard.queue_locks {
            queue_lock.lock();
            unlock_guard.locked_count += 1;
        }

        let fns = self.fns();
        unsafe { (fns.v1_0.device_wait_idle)(self.handle) }
            .result()
            .map_err(VulkanError::from)?;

        Ok(())
    }
}

impl Debug for Device {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        let Self {
            handle,
            physical_device,
            id,

            enabled_extensions,
            enabled_features,
            physical_devices,

            api_version,
            fns,
            active_queue_family_indices,

            queue_locks: _,

            allocation_count,
            fence_pool: _,
            semaphore_pool: _,
            event_pool: _,

            borrowed: _,
        } = self;

        f.debug_struct("Device")
            .field("handle", handle)
            .field("physical_device", physical_device)
            .field("id", id)
            .field("enabled_extensions", enabled_extensions)
            .field("enabled_features", enabled_features)
            .field("physical_devices", physical_devices)
            .field("api_version", api_version)
            .field("fns", fns)
            .field("active_queue_family_indices", active_queue_family_indices)
            .field("allocation_count", allocation_count)
            .finish_non_exhaustive()
    }
}

impl Drop for Device {
    #[inline]
    fn drop(&mut self) {
        let fns = self.fns();

        for &raw_fence in self.fence_pool.lock().iter() {
            unsafe { (fns.v1_0.destroy_fence)(self.handle, raw_fence, ptr::null()) };
        }

        for &raw_sem in self.semaphore_pool.lock().iter() {
            unsafe { (fns.v1_0.destroy_semaphore)(self.handle, raw_sem, ptr::null()) };
        }

        for &raw_event in self.event_pool.lock().iter() {
            unsafe { (fns.v1_0.destroy_event)(self.handle, raw_event, ptr::null()) };
        }

        if !self.borrowed {
            unsafe { (fns.v1_0.destroy_device)(self.handle, ptr::null()) };
        }
    }
}

unsafe impl VulkanObject for Device {
    type Handle = vk::Device;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl InstanceOwned for Device {
    #[inline]
    fn instance(&self) -> &Arc<Instance> {
        self.physical_device().instance()
    }
}

impl_id_counter!(Device);

/// Parameters to create a new `Device`.
#[derive(Clone, Debug)]
pub struct DeviceCreateInfo<'a> {
    /// The queues to create for the device.
    ///
    /// The default value is empty, which must be overridden.
    pub queue_create_infos: &'a [QueueCreateInfo<'a>],

    /// The extensions to enable on the device.
    ///
    /// You only need to enable the extensions that you need. If the extensions you specified
    /// require additional extensions to be enabled, they will be automatically enabled as well.
    ///
    /// If the [`khr_portability_subset`](DeviceExtensions::khr_portability_subset) extension is
    /// available, it will be enabled automatically, so you do not have to do this yourself.
    /// You are responsible for ensuring that your program can work correctly on such devices.
    /// See [the documentation of the `instance`
    /// module](crate::instance#portability-subset-devices-and-the-enumerate_portability-flag)
    /// for more information.
    ///
    /// The default value is [`DeviceExtensions::empty()`].
    pub enabled_extensions: &'a DeviceExtensions,

    /// The features to enable on the device.
    ///
    /// You only need to enable the features that you need. If the extensions you specified
    /// require certain features to be enabled, they will be automatically enabled as well.
    ///
    /// The default value is [`DeviceFeatures::empty()`].
    pub enabled_features: &'a DeviceFeatures,

    /// A list of physical devices to create this device from, to act together as a single
    /// logical device. The physical devices must all belong to the same device group, as returned
    /// by [`Instance::enumerate_physical_device_groups`], and a physical device must not appear
    /// in the list more than once.
    ///
    /// The index of each physical device in this list becomes that physical
    /// device's *device index*, which can be used in other Vulkan functions to specify
    /// a particular physical device within the group.
    /// If the list is left empty, then it behaves as if it contained the physical device,
    /// that was passed to the `physical_device` parameter of [`Device::new`], as its only element.
    /// Otherwise, that physical device must always be part of the list, but it does not need to
    /// be the first element.
    ///
    /// If the list contains more than one physical device, the instance API version must be at
    /// least 1.1, or the [`khr_device_group_creation`] extension must be enabled on the instance.
    /// In order to use any device-level functionality for dealing with device groups,
    /// the physical device API version should also be at least 1.1,
    /// or `enabled_extensions` should contain [`khr_device_group`].
    ///
    /// The default value is empty.
    ///
    /// [`Instance::enumerate_physical_device_groups`]: Instance::enumerate_physical_device_groups
    /// [`khr_device_group_creation`]: crate::instance::InstanceExtensions::khr_device_group_creation
    /// [`khr_device_group`]: crate::device::DeviceExtensions::khr_device_group
    pub physical_devices: &'a [&'a Arc<PhysicalDevice>],

    /// The number of [private data slots] to reserve when creating the device.
    ///
    /// This is purely an optimization, and it is not necessary to do this in order to use private
    /// data slots, but it may improve performance.
    ///
    /// If not zero, the physical device API version must be at least 1.3, or `enabled_extensions`
    /// must contain [`ext_private_data`].
    ///
    /// The default value is `0`.
    ///
    /// [private data slots]: private_data
    /// [`ext_private_data`]: DeviceExtensions::ext_private_data
    pub private_data_slot_request_count: u32,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for DeviceCreateInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> DeviceCreateInfo<'a> {
    /// Returns a default `DeviceCreateInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            queue_create_infos: &[],
            enabled_extensions: &const { DeviceExtensions::empty() },
            enabled_features: &const { DeviceFeatures::empty() },
            physical_devices: &[],
            private_data_slot_request_count: 0,
            _ne: crate::NE,
        }
    }

    pub(crate) fn enable_dependencies(
        &self,
        physical_device: &PhysicalDevice,
    ) -> (DeviceExtensions, DeviceFeatures) {
        // VUID-vkCreateDevice-ppEnabledExtensionNames-01387
        let mut enabled_extensions = self.enabled_extensions.enable_dependencies(
            physical_device.api_version(),
            physical_device.supported_extensions(),
        );

        // VUID-VkDeviceCreateInfo-pProperties-04451
        if physical_device
            .supported_extensions()
            .khr_portability_subset
        {
            enabled_extensions.khr_portability_subset = true;
        }

        let enabled_features = self
            .enabled_features
            .enable_dependencies(physical_device.api_version(), &enabled_extensions);

        (enabled_extensions, enabled_features)
    }

    pub(crate) fn validate(
        &self,
        physical_device: &PhysicalDevice,
    ) -> Result<(), Box<ValidationError>> {
        let &Self {
            queue_create_infos,
            enabled_extensions: _,
            enabled_features: _,
            physical_devices,
            private_data_slot_request_count,
            _ne: _,
        } = self;
        let (enabled_extensions, enabled_features) = self.enable_dependencies(physical_device);

        if queue_create_infos.is_empty() {
            return Err(Box::new(ValidationError {
                context: "queue_create_infos".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkDeviceCreateInfo-queueCreateInfoCount-arraylength"],
                ..Default::default()
            }));
        }

        for (index, queue_create_info) in queue_create_infos.iter().enumerate() {
            queue_create_info
                .validate(physical_device, &enabled_extensions, &enabled_features)
                .map_err(|err| err.add_context(format!("queue_create_infos[{}]", index)))?;

            let &QueueCreateInfo {
                flags: _,
                queue_family_index,
                queues: _,
                _ne: _,
            } = queue_create_info;

            if queue_create_infos
                .iter()
                .filter(|qc2| qc2.queue_family_index == queue_family_index)
                .count()
                != 1
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`queue_create_infos[{}].queue_family_index` occurs more than once in \
                        `queue_create_infos`",
                        index
                    )
                    .into(),
                    vuids: &["VUID-VkDeviceCreateInfo-queueFamilyIndex-02802"],
                    ..Default::default()
                }));
            }
        }

        enabled_extensions
            .validate(
                physical_device.supported_extensions(),
                physical_device.api_version(),
                physical_device.instance().enabled_extensions(),
            )
            .map_err(|err| err.add_context("enabled_extensions"))?;

        // VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-01840
        // VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-00374
        // Ensured because `DeviceExtensions` doesn't contain obsoleted extensions.

        enabled_features
            .validate(
                physical_device.supported_features(),
                physical_device.api_version(),
                &enabled_extensions,
            )
            .map_err(|err| err.add_context("enabled_features"))?;

        if enabled_extensions.ext_buffer_device_address {
            if enabled_extensions.khr_buffer_device_address {
                return Err(Box::new(ValidationError {
                    context: "enabled_extensions".into(),
                    problem: "contains both `khr_buffer_device_address` and \
                        `ext_buffer_device_address`"
                        .into(),
                    vuids: &["VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-03328"],
                    ..Default::default()
                }));
            }

            if enabled_features.buffer_device_address {
                return Err(Box::new(ValidationError {
                    problem: "`enabled_features` contains `buffer_device_address`, and \
                        `enabled_extensions` contains `ext_buffer_device_address`"
                        .into(),
                    vuids: &["VUID-VkDeviceCreateInfo-pNext-04748"],
                    ..Default::default()
                }));
            }
        }

        if enabled_features.shading_rate_image {
            if enabled_features.pipeline_fragment_shading_rate {
                return Err(Box::new(ValidationError {
                    context: "enabled_features".into(),
                    problem: "contains both `shading_rate_image` and \
                        `pipeline_fragment_shading_rate`"
                        .into(),
                    vuids: &["VUID-VkDeviceCreateInfo-shadingRateImage-04478"],
                    ..Default::default()
                }));
            }

            if enabled_features.primitive_fragment_shading_rate {
                return Err(Box::new(ValidationError {
                    context: "enabled_features".into(),
                    problem: "contains both `shading_rate_image` and \
                        `primitive_fragment_shading_rate`"
                        .into(),
                    vuids: &["VUID-VkDeviceCreateInfo-shadingRateImage-04479"],
                    ..Default::default()
                }));
            }

            if enabled_features.attachment_fragment_shading_rate {
                return Err(Box::new(ValidationError {
                    context: "enabled_features".into(),
                    problem: "contains both `shading_rate_image` and \
                        `attachment_fragment_shading_rate`"
                        .into(),
                    vuids: &["VUID-VkDeviceCreateInfo-shadingRateImage-04480"],
                    ..Default::default()
                }));
            }
        }

        if enabled_features.fragment_density_map {
            if enabled_features.pipeline_fragment_shading_rate {
                return Err(Box::new(ValidationError {
                    context: "enabled_features".into(),
                    problem: "contains both `fragment_density_map` and \
                        `pipeline_fragment_shading_rate`"
                        .into(),
                    vuids: &["VUID-VkDeviceCreateInfo-shadingRateImage-04481"],
                    ..Default::default()
                }));
            }

            if enabled_features.primitive_fragment_shading_rate {
                return Err(Box::new(ValidationError {
                    context: "enabled_features".into(),
                    problem: "contains both `fragment_density_map` and \
                        `primitive_fragment_shading_rate`"
                        .into(),
                    vuids: &["VUID-VkDeviceCreateInfo-shadingRateImage-04482"],
                    ..Default::default()
                }));
            }

            if enabled_features.attachment_fragment_shading_rate {
                return Err(Box::new(ValidationError {
                    context: "enabled_features".into(),
                    problem: "contains both `fragment_density_map` and \
                        `attachment_fragment_shading_rate`"
                        .into(),
                    vuids: &["VUID-VkDeviceCreateInfo-shadingRateImage-04483"],
                    ..Default::default()
                }));
            }
        }

        if enabled_features.descriptor_buffer && enabled_extensions.amd_shader_fragment_mask {
            return Err(Box::new(ValidationError {
                problem: "`enabled_features` contains `descriptor_buffer`, and \
                    `enabled_extensions` contains `amd_shader_fragment_mask`"
                    .into(),
                vuids: &["VUID-VkDeviceCreateInfo-None-08095"],
                ..Default::default()
            }));
        }

        if enabled_features.robust_buffer_access
            && !physical_device
                .properties()
                .robust_buffer_access_update_after_bind
                .unwrap_or(false)
            && (enabled_features.descriptor_binding_storage_buffer_update_after_bind
                || enabled_features.descriptor_binding_uniform_texel_buffer_update_after_bind
                || enabled_features.descriptor_binding_storage_texel_buffer_update_after_bind)
        {
            return Err(Box::new(ValidationError {
                problem: "`enabled_features` contains `robust_buffer_access` and \
                    the `robust_buffer_access_update_after_bind` device property is `false`, but \
                    `enabled_features` also contains one of \
                    `descriptor_binding_storage_buffer_update_after_bind`, \
                    `descriptor_binding_uniform_texel_buffer_update_after_bind`, or \
                    `descriptor_binding_storage_texel_buffer_update_after_bind`"
                    .into(),
                vuids: &["VUID-VkDeviceCreateInfo-robustBufferAccess-10247"],
                ..Default::default()
            }));
        }

        if physical_devices.len() > 1 {
            for (index, group_physical_device) in physical_devices.iter().enumerate() {
                if physical_devices[..index].contains(group_physical_device) {
                    return Err(Box::new(ValidationError {
                        context: "physical_devices".into(),
                        problem: format!(
                            "the physical device at index {} is contained in the list more than \
                            once",
                            index,
                        )
                        .into(),
                        vuids: &["VUID-VkDeviceGroupDeviceCreateInfo-pPhysicalDevices-00375"],
                        ..Default::default()
                    }));
                }
            }

            if !(physical_device.instance().api_version() >= Version::V1_1
                || physical_device
                    .instance()
                    .enabled_extensions()
                    .khr_device_group_creation)
            {
                return Err(Box::new(ValidationError {
                    context: "physical_devices".into(),
                    problem: "the length is greater than 1".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                        RequiresAllOf(&[Requires::InstanceExtension("khr_device_group_creation")]),
                    ]),
                    ..Default::default()
                }));
            }

            if !physical_device
                .instance()
                .is_same_device_group(physical_devices.iter().map(AsRef::as_ref))
            {
                return Err(Box::new(ValidationError {
                    context: "physical_devices".into(),
                    problem: "the physical devices do not all belong to the same device group"
                        .into(),
                    vuids: &["VUID-VkDeviceGroupDeviceCreateInfo-pPhysicalDevices-00376"],
                    ..Default::default()
                }));
            }
        }

        if private_data_slot_request_count != 0
            && !(physical_device.api_version() >= Version::V1_3
                || enabled_extensions.ext_private_data)
        {
            return Err(Box::new(ValidationError {
                context: "private_data_slot_request_count".into(),
                problem: "is not zero".into(),
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::DeviceExtension("ext_private_data")]),
                ]),
                ..Default::default()
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk(
        &self,
        fields1_vk: &'a DeviceCreateInfoFields1Vk<'_>,
        extensions_vk: &'a mut DeviceCreateInfoExtensionsVk<'_, '_>,
    ) -> vk::DeviceCreateInfo<'a> {
        let DeviceCreateInfoFields1Vk {
            queue_create_infos_vk,
            enabled_extension_names_vk,
            features_vk,
            device_group_physical_devices_vk: _,
        } = fields1_vk;

        let mut val_vk = vk::DeviceCreateInfo::default()
            .flags(vk::DeviceCreateFlags::empty())
            .queue_create_infos(queue_create_infos_vk)
            .enabled_extension_names(enabled_extension_names_vk);

        if let Some(features_vk) = features_vk {
            val_vk = val_vk.enabled_features(features_vk);
        }

        let DeviceCreateInfoExtensionsVk {
            device_group_vk,
            features2_vk,
            private_data_vk,
        } = extensions_vk;

        if let Some(next) = device_group_vk {
            val_vk = val_vk.push(next);
        }

        if let Some(next) = features2_vk {
            // `features2_vk` may itself carry a `pNext` chain of feature structs,
            // so use `extend` (which links the chain) rather than `push`
            // (which requires a null `pNext`).
            val_vk = unsafe { val_vk.extend(*next) };
        }

        if let Some(next) = private_data_vk {
            val_vk = val_vk.push(next);
        }

        val_vk
    }

    pub(crate) fn to_vk_extensions<'b>(
        &self,
        fields1_vk: &'a DeviceCreateInfoFields1Vk<'_>,
        features2_vk: Option<&'a mut vk::PhysicalDeviceFeatures2<'b>>,
    ) -> DeviceCreateInfoExtensionsVk<'a, 'b> {
        let DeviceCreateInfoFields1Vk {
            queue_create_infos_vk: _,
            enabled_extension_names_vk: _,
            features_vk: _,
            device_group_physical_devices_vk,
        } = fields1_vk;

        // Length of zero and length of one are completely equivalent,
        // so only do anything special here if more than one physical device was given.
        // Spec:
        // A logical device created without using VkDeviceGroupDeviceCreateInfo,
        // or with physicalDeviceCount equal to zero, is equivalent to a physicalDeviceCount of one
        // and pPhysicalDevices pointing to the physicalDevice parameter to vkCreateDevice.
        let device_group_vk = (device_group_physical_devices_vk.len() > 1).then(|| {
            vk::DeviceGroupDeviceCreateInfo::default()
                .physical_devices(device_group_physical_devices_vk)
        });

        let private_data_vk = (self.private_data_slot_request_count != 0).then(|| {
            vk::DevicePrivateDataCreateInfo::default()
                .private_data_slot_request_count(self.private_data_slot_request_count)
        });

        DeviceCreateInfoExtensionsVk {
            device_group_vk,
            features2_vk,
            private_data_vk,
        }
    }

    pub(crate) fn to_vk_fields1(
        &'a self,
        fields2_vk: &'a DeviceCreateInfoFields2Vk,
        features_vk: Option<&'a vk::PhysicalDeviceFeatures>,
    ) -> DeviceCreateInfoFields1Vk<'a> {
        let DeviceCreateInfoFields2Vk {
            enabled_extensions_extra_vk: enabled_extensions_raw_vk,
        } = fields2_vk;
        let enabled_extensions_vk = self.enabled_extensions.to_vk();

        let queue_create_infos_vk = self
            .queue_create_infos
            .iter()
            .map(QueueCreateInfo::to_vk)
            .collect();
        let enabled_extension_names_vk = enabled_extensions_vk
            .into_iter()
            .chain(enabled_extensions_raw_vk.iter().map(Deref::deref))
            .map(|extension| extension.as_ptr())
            .collect();
        let device_group_physical_devices_vk = self
            .physical_devices
            .iter()
            .map(VulkanObject::handle)
            .collect();

        DeviceCreateInfoFields1Vk {
            queue_create_infos_vk,
            enabled_extension_names_vk,
            features_vk,
            device_group_physical_devices_vk,
        }
    }

    pub(crate) fn to_vk_fields2(&self) -> DeviceCreateInfoFields2Vk {
        DeviceCreateInfoFields2Vk {
            // TODO: allow user to (unsafely) specify custom extensions
            enabled_extensions_extra_vk: Vec::new(),
        }
    }
}

pub(crate) struct DeviceCreateInfoExtensionsVk<'a, 'b> {
    pub(crate) device_group_vk: Option<vk::DeviceGroupDeviceCreateInfo<'a>>,
    pub(crate) features2_vk: Option<&'a mut vk::PhysicalDeviceFeatures2<'b>>,
    pub(crate) private_data_vk: Option<vk::DevicePrivateDataCreateInfo<'static>>,
}

pub(crate) struct DeviceCreateInfoFields1Vk<'a> {
    pub(crate) queue_create_infos_vk: SmallVec<[vk::DeviceQueueCreateInfo<'a>; 2]>,
    pub(crate) enabled_extension_names_vk: SmallVec<[*const c_char; 16]>,
    pub(crate) features_vk: Option<&'a vk::PhysicalDeviceFeatures>,
    pub(crate) device_group_physical_devices_vk: SmallVec<[vk::PhysicalDevice; 2]>,
}

pub(crate) struct DeviceCreateInfoFields2Vk {
    pub(crate) enabled_extensions_extra_vk: Vec<CString>,
}

/// Parameters to create queues in a new `Device`.
#[derive(Clone, Debug)]
pub struct QueueCreateInfo<'a> {
    /// Additional properties of the queue.
    ///
    /// The default value is empty.
    pub flags: QueueCreateFlags,

    /// The index of the queue family to create queues for.
    ///
    /// The default value is `0`.
    pub queue_family_index: u32,

    /// The queues to create for the given queue family, each with a relative priority.
    ///
    /// The relative priority value is an arbitrary number between 0.0 and 1.0. Giving a queue a
    /// higher priority is a hint to the driver that the queue should be given more processing
    /// time. As this is only a hint, different drivers may handle this value differently and
    /// there are no guarantees about its behavior.
    ///
    /// The default value is a single queue with a priority of 0.5.
    pub queues: &'a [f32],

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for QueueCreateInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl QueueCreateInfo<'_> {
    /// Returns a default `QueueCreateInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            flags: QueueCreateFlags::empty(),
            queue_family_index: 0,
            queues: &[0.5],
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(
        &self,
        physical_device: &PhysicalDevice,
        device_extensions: &DeviceExtensions,
        device_features: &DeviceFeatures,
    ) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            queue_family_index,
            queues,
            _ne: _,
        } = self;

        flags
            .validate_device_raw(
                physical_device.api_version(),
                device_features,
                device_extensions,
                physical_device.instance().enabled_extensions(),
            )
            .map_err(|err| {
                err.add_context("flags")
                    .set_vuids(&["VUID-VkDeviceQueueCreateInfo-flags-parameter"])
            })?;

        let queue_family_properties = physical_device
            .queue_family_properties()
            .get(queue_family_index as usize)
            .ok_or_else(|| {
                Box::new(ValidationError {
                    context: "queue_family_index".into(),
                    problem: "is not less than the number of queue families in the physical device"
                        .into(),
                    vuids: &["VUID-VkDeviceQueueCreateInfo-queueFamilyIndex-00381"],
                    ..Default::default()
                })
            })?;

        if queues.is_empty() {
            return Err(Box::new(ValidationError {
                context: "queues".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkDeviceQueueCreateInfo-queueCount-arraylength"],
                ..Default::default()
            }));
        }

        if queues.len() > queue_family_properties.queue_count as usize {
            return Err(Box::new(ValidationError {
                problem: "the length of `queues` is greater than the number of queues in the
                    queue family indicated by `queue_family_index`"
                    .into(),
                vuids: &["VUID-VkDeviceQueueCreateInfo-queueCount-00382"],
                ..Default::default()
            }));
        }

        for (index, &priority) in queues.iter().enumerate() {
            if !(0.0..=1.0).contains(&priority) {
                return Err(Box::new(ValidationError {
                    context: format!("queues[{}]", index).into(),
                    problem: "is not between 0.0 and 1.0 inclusive".into(),
                    vuids: &["VUID-VkDeviceQueueCreateInfo-pQueuePriorities-00383"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::DeviceQueueCreateInfo<'_> {
        let &Self {
            flags,
            queue_family_index,
            queues,
            _ne: _,
        } = self;

        vk::DeviceQueueCreateInfo::default()
            .flags(flags.into())
            .queue_family_index(queue_family_index)
            .queue_priorities(queues)
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags specifying additional properties of a queue.
    QueueCreateFlags = DeviceQueueCreateFlags(u32);

    PROTECTED = PROTECTED
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
    ]),
}

/// Implemented on objects that belong to a Vulkan device.
///
/// # Safety
///
/// - `device()` must return the correct device.
pub unsafe trait DeviceOwned {
    /// Returns the device that owns `self`.
    fn device(&self) -> &Arc<Device>;
}

unsafe impl<T> DeviceOwned for T
where
    T: Deref,
    T::Target: DeviceOwned,
{
    fn device(&self) -> &Arc<Device> {
        (**self).device()
    }
}

/// Implemented on objects that implement both `DeviceOwned` and `VulkanObject`.
pub unsafe trait DeviceOwnedVulkanObject {
    /// Assigns a human-readable name to the object for debugging purposes.
    ///
    /// The [`ext_debug_utils`](crate::instance::InstanceExtensions::ext_debug_utils) extension
    /// must be enabled on the instance.
    ///
    /// If `object_name` is `None`, a previously set object name is removed.
    ///
    /// # Safety
    ///
    /// - The object must not be used concurrently by any other Vulkan command.
    unsafe fn set_debug_utils_object_name(
        &self,
        object_name: Option<&str>,
    ) -> Result<(), VulkanError>;
}

unsafe impl<T> DeviceOwnedVulkanObject for T
where
    T: DeviceOwned + VulkanObject,
{
    unsafe fn set_debug_utils_object_name(
        &self,
        object_name: Option<&str>,
    ) -> Result<(), VulkanError> {
        unsafe { self.device().set_debug_utils_object_name(self, object_name) }
    }
}

/// Same as [`DebugWrapper`], but also prints the device handle for disambiguation.
///
/// [`DebugWrapper`]: crate:: DebugWrapper
#[derive(Clone, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub(crate) struct DeviceOwnedDebugWrapper<T>(pub(crate) T);

impl<T> DeviceOwnedDebugWrapper<T> {
    pub fn cast_slice_inner(slice: &[Self]) -> &[T] {
        // SAFETY: `DeviceOwnedDebugWrapper<T>` and `T` have the same layout.
        unsafe { slice::from_raw_parts(<*const _>::cast(slice), slice.len()) }
    }
}

impl<T> Debug for DeviceOwnedDebugWrapper<T>
where
    T: VulkanObject + DeviceOwned,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(
            f,
            "0x{:x} (device: 0x{:x})",
            self.0.handle().as_raw(),
            self.0.device().handle().as_raw(),
        )
    }
}

impl<T> Deref for DeviceOwnedDebugWrapper<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl PhysicalDeviceFeatures2ExtensionsVk {
    fn disable_conflicting(&mut self) {
        macro_rules! conflicts_with_struct {
            (
                $struct:ident =>
                $($struct_to_disable:ident),+ $(,)?
            ) => {
                if self.$struct.is_some() {
                    $(
                        self.$struct_to_disable = None;
                    )+
                }
            };
        }

        // VUID-VkDeviceCreateInfo-pNext-02829
        conflicts_with_struct!(features_vulkan11_vk =>
            features_16bit_storage_vk,
            features_multiview_vk,
            features_variable_pointers_vk,
            features_protected_memory_vk,
            features_sampler_ycbcr_conversion_vk,
            features_shader_draw_parameters_vk,
        );

        // VUID-VkDeviceCreateInfo-pNext-02830
        conflicts_with_struct!(features_vulkan12_vk =>
            features_8bit_storage_vk,
            features_shader_atomic_int64_vk,
            features_shader_float16_int8_vk,
            features_descriptor_indexing_vk,
            features_scalar_block_layout_vk,
            features_imageless_framebuffer_vk,
            features_uniform_buffer_standard_layout_vk,
            features_shader_subgroup_extended_types_vk,
            features_separate_depth_stencil_layouts_vk,
            features_host_query_reset_vk,
            features_timeline_semaphore_vk,
            features_buffer_device_address_vk,
            features_vulkan_memory_model_vk,
        );

        // VUID-VkDeviceCreateInfo-pNext-06532
        conflicts_with_struct!(features_vulkan13_vk =>
            features_dynamic_rendering_vk,
            features_image_robustness_vk,
            features_inline_uniform_block_vk,
            features_maintenance4_vk,
            features_pipeline_creation_cache_control_vk,
            features_private_data_vk,
            features_shader_demote_to_helper_invocation_vk,
            features_shader_integer_dot_product_vk,
            features_shader_terminate_invocation_vk,
            features_subgroup_size_control_vk,
            features_synchronization2_vk,
            features_texture_compression_astchdr_vk,
            features_zero_initialize_workgroup_memory_vk,
        );

        // VUID-VkDeviceCreateInfo-pNext-10360
        // TODO: Vulkan 1.4
        /*disable_feature_conflicting_structs!(features_vulkan14_vk =>
            features_global_priority_query_vk,
            features_shader_subgroup_rotate_vk,
            features_shader_float_controls2_vk,
            features_shader_expect_assume_vk,
            features_line_rasterization_vk,
            features_vertex_attribute_divisor_vk,
            features_index_type_uint8_vk,
            features_dynamic_rendering_local_read_vk,
            features_maintenance5_vk,
            features_maintenance6_vk,
            features_pipeline_protected_access_vk,
            features_pipeline_robustness_vk,
            features_host_image_copy_vk,
        );*/
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

    #[test]
    fn empty_extensions() {
        let d = DeviceExtensions::empty().to_vk();
        assert!(d.is_empty());
    }

    #[test]
    fn extensions_into_iter() {
        let extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };
        for (name, enabled) in &extensions {
            if name == "VK_KHR_swapchain" {
                assert!(enabled);
            } else {
                assert!(!enabled);
            }
        }
    }

    #[test]
    fn features_into_iter() {
        let features = DeviceFeatures {
            tessellation_shader: true,
            ..DeviceFeatures::empty()
        };
        for (name, enabled) in &features {
            if name == "tessellationShader" {
                assert!(enabled);
            } else {
                assert!(!enabled);
            }
        }
    }

    #[test]
    fn one_ref() {
        let (mut device, _) = gfx_dev_and_queue!();
        assert!(Arc::get_mut(&mut device).is_some());
    }

    #[test]
    fn too_many_queues() {
        let instance = instance!();
        let physical_device = match instance.enumerate_physical_devices() {
            Ok(x) => x.into_iter().next().unwrap(),
            Err(_) => return,
        };

        let queue_family_index = 0;
        let queue_family_properties =
            &physical_device.queue_family_properties()[queue_family_index as usize];
        let queues = (0..queue_family_properties.queue_count + 1)
            .map(|_| 0.5)
            .collect::<Vec<_>>();

        assert!(Device::try_new(
            &physical_device,
            &DeviceCreateInfo {
                queue_create_infos: &[QueueCreateInfo {
                    queue_family_index,
                    queues: &queues,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .is_err());
    }

    #[test]
    fn unsupported_features() {
        let instance = instance!();
        let physical_device = match instance.enumerate_physical_devices() {
            Ok(x) => x.into_iter().next().unwrap(),
            Err(_) => return,
        };

        let features = DeviceFeatures::all();
        // In the unlikely situation where the device supports everything, we ignore the test.
        if physical_device.supported_features().contains(&features) {
            return;
        }

        assert!(Device::try_new(
            &physical_device,
            &DeviceCreateInfo {
                queue_create_infos: &[QueueCreateInfo {
                    queue_family_index: 0,
                    ..Default::default()
                }],
                enabled_features: &features,
                ..Default::default()
            },
        )
        .is_err());
    }

    #[test]
    fn priority_out_of_range() {
        let instance = instance!();
        let physical_device = match instance.enumerate_physical_devices() {
            Ok(x) => x.into_iter().next().unwrap(),
            Err(_) => return,
        };

        assert!(Device::try_new(
            &physical_device,
            &DeviceCreateInfo {
                queue_create_infos: &[QueueCreateInfo {
                    queue_family_index: 0,
                    queues: &[1.4],
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .is_err());

        assert!(Device::try_new(
            &physical_device,
            &DeviceCreateInfo {
                queue_create_infos: &[QueueCreateInfo {
                    queue_family_index: 0,
                    queues: &[-0.2],
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .is_err());
    }

    #[test]
    fn create_with() {
        let instance = instance!();

        let physical_device = match instance.enumerate_physical_devices() {
            Ok(x) => x.into_iter().next().unwrap(),
            Err(_) => return,
        };

        let arc = Arc::new(42);
        let _arc2 = arc.clone();

        let instance_ref = &instance;
        let physical_device_ref = &physical_device;
        let arc3 = arc.clone();
        let create_fn = move |create_info_vk: &vk::DeviceCreateInfo<'_>| {
            // Move one clone to the closure without consuming it in the closure.
            let _ = &arc3;

            let fns = instance_ref.fns();
            let mut device_vk = vk::Device::null();
            unsafe {
                (fns.v1_0.create_device)(
                    physical_device_ref.handle(),
                    create_info_vk,
                    ptr::null(),
                    &mut device_vk,
                )
            }
            .result()
            .map_err(VulkanError::from)?;

            Ok(device_vk)
        };

        let _ = unsafe {
            Device::new_with(
                &physical_device,
                &DeviceCreateInfo {
                    queue_create_infos: &[QueueCreateInfo {
                        queues: &[1.0],
                        ..Default::default()
                    }],
                    ..Default::default()
                },
                create_fn,
            )
        };

        // The closure should have been dropped exactly once, not zero or two times.
        assert_eq!(Arc::strong_count(&arc), 2);

        struct OurPayload;

        let arc3 = arc.clone();
        let create_fn = move |_create_info_vk: &vk::DeviceCreateInfo<'_>| {
            let _ = &arc3;

            resume_unwind(Box::new(OurPayload))
        };

        catch_unwind(AssertUnwindSafe(|| unsafe {
            Device::new_with(
                &physical_device,
                &DeviceCreateInfo {
                    queue_create_infos: &[QueueCreateInfo {
                        queues: &[1.0],
                        ..Default::default()
                    }],
                    ..Default::default()
                },
                create_fn,
            )
        }))
        .unwrap_err()
        .downcast::<OurPayload>()
        .unwrap();

        // The closure should have been dropped exactly once even in the face of panics.
        assert_eq!(Arc::strong_count(&arc), 2);
    }
}
