use super::QueueFamilyProperties;
use crate::{
    buffer::{ExternalBufferInfo, ExternalBufferProperties},
    cache::{OnceCache, WeakArcOnceCache},
    device::{properties::DeviceProperties, DeviceExtensions, DeviceFeatures},
    display::{Display, DisplayPlaneProperties, DisplayPlanePropertiesRaw, DisplayProperties},
    format::{Format, FormatProperties},
    image::{
        ImageFormatInfo, ImageFormatProperties, OwnedImageFormatInfo, SparseImageFormatInfo,
        SparseImageFormatProperties,
    },
    instance::{Instance, InstanceOwned},
    macros::{impl_id_counter, vulkan_bitflags, vulkan_enum},
    memory::{ExternalMemoryHandleType, MemoryBudget, MemoryProperties},
    swapchain::{
        ColorSpace, FullScreenExclusive, PresentMode, Surface, SurfaceApi, SurfaceCapabilities,
        SurfaceInfo, SurfaceInfo2ExtensionsVk,
    },
    sync::{
        fence::{ExternalFenceInfo, ExternalFenceProperties},
        semaphore::{ExternalSemaphoreInfo, ExternalSemaphoreProperties},
    },
    DebugWrapper, ExtensionProperties, Requires, RequiresAllOf, RequiresOneOf, Validated,
    ValidationError, Version, VulkanError, VulkanObject,
};
use ash::vk;
use ash::vk::TaggedStructure;

use parking_lot::RwLock;
#[cfg(feature = "raw_window_handle")]
use raw_window_handle::{HandleError, HasDisplayHandle, RawDisplayHandle};
use std::{
    fmt::{Debug, Error as FmtError, Formatter},
    marker::PhantomData,
    mem::MaybeUninit,
    num::NonZero,
    ptr,
    sync::Arc,
};

/// Represents one of the available physical devices on this machine.
///
/// # Examples
///
/// ```no_run
/// # use vulkano::{
/// #     instance::{Instance, InstanceExtensions},
/// #     Version, VulkanLibrary,
/// # };
/// use vulkano::device::physical::PhysicalDevice;
///
/// # let library = unsafe { VulkanLibrary::new() }.unwrap();
/// # let instance = Instance::new(&library, &Default::default()).unwrap();
/// #
/// for physical_device in instance.enumerate_physical_devices().unwrap() {
///     print_infos(&physical_device);
/// }
///
/// fn print_infos(dev: &PhysicalDevice) {
///     println!("Name: {}", dev.properties().device_name);
/// }
/// ```
pub struct PhysicalDevice {
    handle: vk::PhysicalDevice,
    instance: DebugWrapper<Arc<Instance>>,
    id: NonZero<u64>,

    // Data queried at `PhysicalDevice` creation.
    api_version: Version,
    supported_extensions: DeviceExtensions,
    supported_features: DeviceFeatures,
    properties: DeviceProperties,
    extension_properties: Vec<ExtensionProperties>,
    memory_properties: MemoryProperties,
    queue_family_properties: Vec<QueueFamilyProperties>,

    // Data queried by the user at runtime, cached for faster lookups.
    display_properties: WeakArcOnceCache<vk::DisplayKHR, Display>,
    display_plane_properties: RwLock<Vec<DisplayPlanePropertiesRaw>>,
    external_buffer_properties: OnceCache<ExternalBufferInfo<'static>, ExternalBufferProperties>,
    external_fence_properties: OnceCache<ExternalFenceInfo<'static>, ExternalFenceProperties>,
    external_semaphore_properties:
        OnceCache<ExternalSemaphoreInfo<'static>, ExternalSemaphoreProperties>,
    format_properties: OnceCache<Format, FormatProperties>,
    image_format_properties: OnceCache<OwnedImageFormatInfo, Option<ImageFormatProperties>>,
    sparse_image_format_properties:
        OnceCache<SparseImageFormatInfo<'static>, Vec<SparseImageFormatProperties>>,
}

impl PhysicalDevice {
    /// Creates a new `PhysicalDevice` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `instance`.
    pub unsafe fn from_handle(
        instance: &Arc<Instance>,
        handle: vk::PhysicalDevice,
    ) -> Result<Arc<Self>, VulkanError> {
        let api_version = unsafe { Self::get_api_version(handle, instance) };
        let extension_properties = unsafe { Self::get_extension_properties(handle, instance) }?;
        let supported_extensions = DeviceExtensions::from_vk(
            extension_properties
                .iter()
                .map(|property| property.extension_name.as_str()),
        );

        // Get the remaining infos.
        // If possible, we use VK_KHR_get_physical_device_properties2.
        let (supported_features, properties, memory_properties, queue_family_properties) =
            if api_version >= Version::V1_1
                || instance
                    .enabled_extensions()
                    .khr_get_physical_device_properties2
            {
                (
                    unsafe {
                        Self::get_features2(handle, instance, api_version, &supported_extensions)
                    },
                    unsafe {
                        Self::get_properties2(handle, instance, api_version, &supported_extensions)
                    },
                    unsafe { Self::get_memory_properties2(handle, instance) },
                    unsafe { Self::get_queue_family_properties2(handle, instance) },
                )
            } else {
                (
                    unsafe { Self::get_features(handle, instance) },
                    unsafe { Self::get_properties(handle, instance) },
                    unsafe { Self::get_memory_properties(handle, instance) },
                    unsafe { Self::get_queue_family_properties(handle, instance) },
                )
            };

        Ok(Arc::new(PhysicalDevice {
            handle,
            instance: DebugWrapper(instance.clone()),
            id: Self::next_id(),

            api_version,
            supported_extensions,
            supported_features,
            properties,
            extension_properties,
            memory_properties,
            queue_family_properties,

            display_properties: WeakArcOnceCache::new(),
            display_plane_properties: RwLock::new(Vec::new()),
            external_buffer_properties: OnceCache::new(),
            external_fence_properties: OnceCache::new(),
            external_semaphore_properties: OnceCache::new(),
            format_properties: OnceCache::new(),
            image_format_properties: OnceCache::new(),
            sparse_image_format_properties: OnceCache::new(),
        }))
    }

    unsafe fn get_api_version(handle: vk::PhysicalDevice, instance: &Instance) -> Version {
        let properties = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            unsafe { (fns.v1_0.get_physical_device_properties)(handle, output.as_mut_ptr()) };
            unsafe { output.assume_init() }
        };

        let api_version = Version::from(properties.api_version);
        std::cmp::min(instance.max_api_version(), api_version)
    }

    unsafe fn get_extension_properties(
        handle: vk::PhysicalDevice,
        instance: &Instance,
    ) -> Result<Vec<ExtensionProperties>, VulkanError> {
        let fns = instance.fns();

        loop {
            let mut count = 0;
            unsafe {
                (fns.v1_0.enumerate_device_extension_properties)(
                    handle,
                    ptr::null(),
                    &mut count,
                    ptr::null_mut(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;

            let mut output = Vec::with_capacity(count as usize);
            let result = unsafe {
                (fns.v1_0.enumerate_device_extension_properties)(
                    handle,
                    ptr::null(),
                    &mut count,
                    output.as_mut_ptr(),
                )
            };

            match result {
                vk::Result::SUCCESS => {
                    unsafe { output.set_len(count as usize) };
                    return Ok(output.into_iter().map(Into::into).collect());
                }
                vk::Result::INCOMPLETE => (),
                err => return Err(VulkanError::from(err)),
            }
        }
    }

    unsafe fn get_features(handle: vk::PhysicalDevice, instance: &Instance) -> DeviceFeatures {
        let mut features_vk = DeviceFeatures::to_mut_vk();

        let fns = instance.fns();
        unsafe { (fns.v1_0.get_physical_device_features)(handle, &mut features_vk) };

        DeviceFeatures::from_vk(&features_vk)
    }

    unsafe fn get_features2(
        handle: vk::PhysicalDevice,
        instance: &Instance,
        api_version: Version,
        supported_extensions: &DeviceExtensions,
    ) -> DeviceFeatures {
        let fns = instance.fns();

        let call = |features_vk: &mut vk::PhysicalDeviceFeatures2<'_>| {
            if instance.api_version() >= Version::V1_1 {
                unsafe { (fns.v1_1.get_physical_device_features2)(handle, features_vk) };
            } else {
                unsafe {
                    (fns.khr_get_physical_device_properties2
                        .get_physical_device_features2_khr)(handle, features_vk)
                };
            }
        };

        let mut features2_extensions_vk = DeviceFeatures::to_mut_vk2_extensions(
            api_version,
            supported_extensions,
            instance.enabled_extensions(),
        );
        let mut features2_vk = DeviceFeatures::to_mut_vk2(&mut features2_extensions_vk);

        call(&mut features2_vk);

        // Unborrow
        let features2_vk = vk::PhysicalDeviceFeatures2 {
            _marker: PhantomData,
            ..features2_vk
        };

        DeviceFeatures::from_vk2(&features2_vk, &features2_extensions_vk)
    }

    unsafe fn get_properties(handle: vk::PhysicalDevice, instance: &Instance) -> DeviceProperties {
        let mut properties_vk = DeviceProperties::to_mut_vk();

        let fns = instance.fns();
        unsafe { (fns.v1_0.get_physical_device_properties)(handle, &mut properties_vk) };

        DeviceProperties::from_vk(&properties_vk)
    }

    unsafe fn get_properties2(
        handle: vk::PhysicalDevice,
        instance: &Instance,
        api_version: Version,
        supported_extensions: &DeviceExtensions,
    ) -> DeviceProperties {
        let fns = instance.fns();

        let call = |properties2_vk: &mut vk::PhysicalDeviceProperties2<'_>| {
            if instance.api_version() >= Version::V1_1 {
                unsafe { (fns.v1_1.get_physical_device_properties2)(handle, properties2_vk) };
            } else {
                unsafe {
                    (fns.khr_get_physical_device_properties2
                        .get_physical_device_properties2_khr)(
                        handle, properties2_vk
                    )
                };
            }
        };

        let mut properties2_fields1_vk = DeviceProperties::to_mut_vk2_fields1({
            let mut properties2_extensions_query_count_vk =
                DeviceProperties::to_mut_vk2_extensions_query_count(
                    api_version,
                    supported_extensions,
                    instance.enabled_extensions(),
                );
            let mut properties2_query_count_vk =
                DeviceProperties::to_mut_vk2(&mut properties2_extensions_query_count_vk);

            call(&mut properties2_query_count_vk);

            properties2_extensions_query_count_vk
        });

        let mut properties2_extensions_vk = DeviceProperties::to_mut_vk2_extensions(
            &mut properties2_fields1_vk,
            api_version,
            supported_extensions,
            instance.enabled_extensions(),
        );
        let mut properties2_vk = DeviceProperties::to_mut_vk2(&mut properties2_extensions_vk);

        call(&mut properties2_vk);

        // Unborrow
        let properties2_vk = vk::PhysicalDeviceProperties2 {
            _marker: PhantomData,
            ..properties2_vk
        };
        let properties2_extensions_vk = properties2_extensions_vk.unborrow();

        DeviceProperties::from_vk2(
            &properties2_vk,
            &properties2_extensions_vk,
            &properties2_fields1_vk,
        )
    }

    unsafe fn get_memory_properties(
        handle: vk::PhysicalDevice,
        instance: &Instance,
    ) -> MemoryProperties {
        let properties = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.v1_0.get_physical_device_memory_properties)(handle, output.as_mut_ptr())
            };
            unsafe { output.assume_init() }
        };

        MemoryProperties::from_vk(&properties)
    }

    unsafe fn get_memory_properties2(
        handle: vk::PhysicalDevice,
        instance: &Instance,
    ) -> MemoryProperties {
        let mut properties_vk = MemoryProperties::to_mut_vk2();

        let fns = instance.fns();

        if instance.api_version() >= Version::V1_1 {
            unsafe {
                (fns.v1_1.get_physical_device_memory_properties2)(handle, &mut properties_vk)
            };
        } else {
            unsafe {
                (fns.khr_get_physical_device_properties2
                    .get_physical_device_memory_properties2_khr)(
                    handle, &mut properties_vk
                )
            };
        }

        MemoryProperties::from_vk2(&properties_vk)
    }

    unsafe fn get_queue_family_properties(
        handle: vk::PhysicalDevice,
        instance: &Instance,
    ) -> Vec<QueueFamilyProperties> {
        let fns = instance.fns();

        let mut num = 0;
        unsafe {
            (fns.v1_0.get_physical_device_queue_family_properties)(
                handle,
                &mut num,
                ptr::null_mut(),
            )
        };

        let mut output = Vec::with_capacity(num as usize);
        unsafe {
            (fns.v1_0.get_physical_device_queue_family_properties)(
                handle,
                &mut num,
                output.as_mut_ptr(),
            )
        };
        unsafe { output.set_len(num as usize) };

        output.iter().map(QueueFamilyProperties::from_vk).collect()
    }

    unsafe fn get_queue_family_properties2(
        handle: vk::PhysicalDevice,
        instance: &Instance,
    ) -> Vec<QueueFamilyProperties> {
        let mut num = 0;
        let fns = instance.fns();

        if instance.api_version() >= Version::V1_1 {
            unsafe {
                (fns.v1_1.get_physical_device_queue_family_properties2)(
                    handle,
                    &mut num,
                    ptr::null_mut(),
                )
            };
        } else {
            unsafe {
                (fns.khr_get_physical_device_properties2
                    .get_physical_device_queue_family_properties2_khr)(
                    handle,
                    &mut num,
                    ptr::null_mut(),
                )
            };
        }

        let mut properties_vk = vec![QueueFamilyProperties::to_mut_vk2(); num as usize];

        if instance.api_version() >= Version::V1_1 {
            unsafe {
                (fns.v1_1.get_physical_device_queue_family_properties2)(
                    handle,
                    &mut num,
                    properties_vk.as_mut_ptr(),
                )
            };
        } else {
            unsafe {
                (fns.khr_get_physical_device_properties2
                    .get_physical_device_queue_family_properties2_khr)(
                    handle,
                    &mut num,
                    properties_vk.as_mut_ptr(),
                )
            };
        }

        properties_vk
            .iter()
            .map(QueueFamilyProperties::from_vk2)
            .collect()
    }

    /// Returns the instance that owns the physical device.
    #[inline]
    pub fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }

    /// Returns the version of Vulkan supported by the physical device.
    ///
    /// Unlike the `api_version` property, which is the version reported by the device directly,
    /// this function returns the version the device can actually support, based on the instance's
    /// `max_api_version`.
    #[inline]
    pub fn api_version(&self) -> Version {
        self.api_version
    }

    /// Returns the properties reported by the physical device.
    #[inline]
    pub fn properties(&self) -> &DeviceProperties {
        &self.properties
    }

    /// Returns the extension properties reported by the physical device.
    #[inline]
    pub fn extension_properties(&self) -> &[ExtensionProperties] {
        &self.extension_properties
    }

    /// Returns the extensions that are supported by the physical device.
    #[inline]
    pub fn supported_extensions(&self) -> &DeviceExtensions {
        &self.supported_extensions
    }

    /// Returns the features that are supported by the physical device.
    #[inline]
    pub fn supported_features(&self) -> &DeviceFeatures {
        &self.supported_features
    }

    /// Returns the memory properties reported by the physical device.
    #[inline]
    pub fn memory_properties(&self) -> &MemoryProperties {
        &self.memory_properties
    }

    /// Returns the queue family properties reported by the physical device.
    #[inline]
    pub fn queue_family_properties(&self) -> &[QueueFamilyProperties] {
        &self.queue_family_properties
    }

    /// Retrieves the current memory budget and usage of each memory heap of the physical device,
    /// panicking on a validation error.
    ///
    /// These properties may change during runtime, so the result only reflects the current
    /// situation and is not cached.
    ///
    /// The [`ext_memory_budget`] extension must be supported by the physical device.
    ///
    /// This is a shortcut for `try_memory_budget().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_memory_budget`] returns a [`ValidationError`].
    ///
    /// [`ext_memory_budget`]: crate::device::DeviceExtensions::ext_memory_budget
    /// [`try_memory_budget`]: Self::try_memory_budget
    #[inline]
    #[track_caller]
    pub fn memory_budget(&self) -> MemoryBudget {
        self.try_memory_budget().unwrap()
    }

    /// Retrieves the current memory budget and usage of each memory heap of the physical device.
    ///
    /// These properties may change during runtime, so the result only reflects the current
    /// situation and is not cached.
    ///
    /// The [`ext_memory_budget`] extension must be supported by the physical device.
    ///
    /// [`ext_memory_budget`]: crate::device::DeviceExtensions::ext_memory_budget
    #[inline]
    pub fn try_memory_budget(&self) -> Result<MemoryBudget, Box<ValidationError>> {
        self.validate_memory_budget()?;

        Ok(unsafe { self.memory_budget_unchecked() })
    }

    fn validate_memory_budget(&self) -> Result<(), Box<ValidationError>> {
        if !self.supported_extensions().ext_memory_budget {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "ext_memory_budget",
                )])]),
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn memory_budget_unchecked(&self) -> MemoryBudget {
        let mut budget_vk = MemoryBudget::to_mut_vk2();
        let mut properties2_vk = MemoryProperties::to_mut_vk2().push(&mut budget_vk);

        let fns = self.instance.fns();
        if self.instance.api_version() >= Version::V1_1 {
            unsafe {
                (fns.v1_1.get_physical_device_memory_properties2)(self.handle, &mut properties2_vk)
            };
        } else {
            unsafe {
                (fns.khr_get_physical_device_properties2
                    .get_physical_device_memory_properties2_khr)(
                    self.handle, &mut properties2_vk
                )
            };
        }

        MemoryBudget::from_vk(&budget_vk, self.memory_properties().memory_heaps.len())
    }

    /// Returns the properties of displays attached to the physical device, panicking on a
    /// validation error.
    ///
    /// This is a shortcut for `try_display_properties().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_display_properties`] returns a [`ValidationError`].
    ///
    /// [`try_display_properties`]: Self::try_display_properties
    #[inline]
    #[track_caller]
    pub fn display_properties(self: &Arc<Self>) -> Result<Vec<Arc<Display>>, VulkanError> {
        match self.try_display_properties() {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Returns the properties of displays attached to the physical device.
    #[inline]
    pub fn try_display_properties(
        self: &Arc<Self>,
    ) -> Result<Vec<Arc<Display>>, Validated<VulkanError>> {
        self.validate_display_properties()?;

        Ok(unsafe { self.display_properties_unchecked() }?)
    }

    fn validate_display_properties(&self) -> Result<(), Box<ValidationError>> {
        if !self.instance.enabled_extensions().khr_display {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "khr_display",
                )])]),
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn display_properties_unchecked(
        self: &Arc<Self>,
    ) -> Result<Vec<Arc<Display>>, VulkanError> {
        let fns = self.instance.fns();

        if self
            .instance
            .enabled_extensions()
            .khr_get_display_properties2
        {
            let properties_vk = loop {
                let mut count = 0;
                unsafe {
                    (fns.khr_get_display_properties2
                        .get_physical_device_display_properties2_khr)(
                        self.handle,
                        &mut count,
                        ptr::null_mut(),
                    )
                }
                .result()
                .map_err(VulkanError::from)?;

                let mut properties_vk = vec![DisplayProperties::to_mut_vk2(); count as usize];
                let result = unsafe {
                    (fns.khr_get_display_properties2
                        .get_physical_device_display_properties2_khr)(
                        self.handle,
                        &mut count,
                        properties_vk.as_mut_ptr(),
                    )
                };

                match result {
                    vk::Result::SUCCESS => {
                        unsafe { properties_vk.set_len(count as usize) };
                        break properties_vk;
                    }
                    vk::Result::INCOMPLETE => (),
                    err => return Err(VulkanError::from(err)),
                }
            };

            Ok(properties_vk
                .iter()
                .map(|properties_vk| {
                    let properties_vk = &properties_vk.display_properties;
                    self.display_properties
                        .get_or_insert(properties_vk.display, |&handle| {
                            Display::from_handle(
                                self,
                                handle,
                                DisplayProperties::from_vk(properties_vk),
                            )
                        })
                })
                .collect())
        } else {
            let properties_vk = loop {
                let mut count = 0;
                unsafe {
                    (fns.khr_display.get_physical_device_display_properties_khr)(
                        self.handle,
                        &mut count,
                        ptr::null_mut(),
                    )
                }
                .result()
                .map_err(VulkanError::from)?;

                let mut properties_vk = Vec::with_capacity(count as usize);
                let result = unsafe {
                    (fns.khr_display.get_physical_device_display_properties_khr)(
                        self.handle,
                        &mut count,
                        properties_vk.as_mut_ptr(),
                    )
                };

                match result {
                    vk::Result::SUCCESS => {
                        unsafe { properties_vk.set_len(count as usize) };
                        break properties_vk;
                    }
                    vk::Result::INCOMPLETE => (),
                    err => return Err(VulkanError::from(err)),
                }
            };

            Ok(properties_vk
                .iter()
                .map(|properties_vk| {
                    self.display_properties
                        .get_or_insert(properties_vk.display, |&handle| {
                            Display::from_handle(
                                self,
                                handle,
                                DisplayProperties::from_vk(properties_vk),
                            )
                        })
                })
                .collect())
        }
    }

    /// Returns the properties of the display planes of the physical device, panicking on a
    /// validation error.
    ///
    /// This is a shortcut for `try_display_plane_properties().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_display_plane_properties`] returns a [`ValidationError`].
    ///
    /// [`try_display_plane_properties`]: Self::try_display_plane_properties
    #[inline]
    #[track_caller]
    pub fn display_plane_properties(
        self: &Arc<Self>,
    ) -> Result<Vec<DisplayPlaneProperties>, VulkanError> {
        match self.try_display_plane_properties() {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Returns the properties of the display planes of the physical device.
    #[inline]
    pub fn try_display_plane_properties(
        self: &Arc<Self>,
    ) -> Result<Vec<DisplayPlaneProperties>, Validated<VulkanError>> {
        self.validate_display_plane_properties()?;

        Ok(unsafe { self.display_plane_properties_unchecked() }?)
    }

    fn validate_display_plane_properties(&self) -> Result<(), Box<ValidationError>> {
        if !self.instance.enabled_extensions().khr_display {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "khr_display",
                )])]),
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn display_plane_properties_unchecked(
        self: &Arc<Self>,
    ) -> Result<Vec<DisplayPlaneProperties>, VulkanError> {
        unsafe { self.get_display_plane_properties_raw() }?
            .iter()
            .map(|properties_raw| -> Result<_, VulkanError> {
                let &DisplayPlanePropertiesRaw {
                    current_display,
                    current_stack_index,
                } = properties_raw;

                let current_display = current_display
                    .map(|display_handle| {
                        self.display_properties
                            .get(&display_handle)
                            .map(Ok)
                            .unwrap_or_else(|| -> Result<_, VulkanError> {
                                unsafe { self.display_properties_unchecked() }?;
                                Ok(self.display_properties.get(&display_handle).unwrap())
                            })
                    })
                    .transpose()?;

                Ok(DisplayPlaneProperties {
                    current_display,
                    current_stack_index,
                })
            })
            .collect()
    }

    pub(crate) unsafe fn display_plane_properties_raw(
        &self,
    ) -> Result<Vec<DisplayPlanePropertiesRaw>, VulkanError> {
        {
            let read = self.display_plane_properties.read();

            if !read.is_empty() {
                return Ok(read.clone());
            }
        }

        unsafe { self.get_display_plane_properties_raw() }
    }

    unsafe fn get_display_plane_properties_raw(
        &self,
    ) -> Result<Vec<DisplayPlanePropertiesRaw>, VulkanError> {
        let fns = self.instance.fns();

        let properties_raw: Vec<_> = if self
            .instance
            .enabled_extensions()
            .khr_get_display_properties2
        {
            let properties_vk = loop {
                let mut count = 0;
                unsafe {
                    (fns.khr_get_display_properties2
                        .get_physical_device_display_plane_properties2_khr)(
                        self.handle,
                        &mut count,
                        ptr::null_mut(),
                    )
                }
                .result()
                .map_err(VulkanError::from)?;

                let mut properties = vec![DisplayPlanePropertiesRaw::to_mut_vk2(); count as usize];
                let result = unsafe {
                    (fns.khr_get_display_properties2
                        .get_physical_device_display_plane_properties2_khr)(
                        self.handle,
                        &mut count,
                        properties.as_mut_ptr(),
                    )
                };

                match result {
                    vk::Result::SUCCESS => {
                        unsafe { properties.set_len(count as usize) };
                        break properties;
                    }
                    vk::Result::INCOMPLETE => (),
                    err => return Err(VulkanError::from(err)),
                }
            };

            properties_vk
                .iter()
                .map(|properties_vk| {
                    let properties_vk = &properties_vk.display_plane_properties;
                    DisplayPlanePropertiesRaw::from_vk(properties_vk)
                })
                .collect()
        } else {
            let properties_vk = loop {
                let mut count = 0;
                unsafe {
                    (fns.khr_display
                        .get_physical_device_display_plane_properties_khr)(
                        self.handle,
                        &mut count,
                        ptr::null_mut(),
                    )
                }
                .result()
                .map_err(VulkanError::from)?;

                let mut properties = Vec::with_capacity(count as usize);
                let result = unsafe {
                    (fns.khr_display
                        .get_physical_device_display_plane_properties_khr)(
                        self.handle,
                        &mut count,
                        properties.as_mut_ptr(),
                    )
                };

                match result {
                    vk::Result::SUCCESS => {
                        unsafe { properties.set_len(count as usize) };
                        break properties;
                    }
                    vk::Result::INCOMPLETE => (),
                    err => return Err(VulkanError::from(err)),
                }
            };

            properties_vk
                .iter()
                .map(DisplayPlanePropertiesRaw::from_vk)
                .collect()
        };

        self.display_plane_properties
            .write()
            .clone_from(&properties_raw);

        Ok(properties_raw)
    }

    /// Returns the displays that are supported for the given plane index, panicking on a
    /// validation error.
    ///
    /// The index must be less than the number of elements returned by
    /// [`display_plane_properties`].
    ///
    /// This is a shortcut for `try_display_plane_supported_displays().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_display_plane_supported_displays`] returns a [`ValidationError`].
    ///
    /// [`display_plane_properties`]: Self::display_plane_properties
    /// [`try_display_plane_supported_displays`]: Self::try_display_plane_supported_displays
    #[inline]
    #[track_caller]
    pub fn display_plane_supported_displays(
        self: &Arc<Self>,
        plane_index: u32,
    ) -> Result<Vec<Arc<Display>>, VulkanError> {
        match self.try_display_plane_supported_displays(plane_index) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Returns the displays that are supported for the given plane index.
    ///
    /// The index must be less than the number of elements returned by
    /// [`display_plane_properties`].
    ///
    /// [`display_plane_properties`]: Self::display_plane_properties
    #[inline]
    pub fn try_display_plane_supported_displays(
        self: &Arc<Self>,
        plane_index: u32,
    ) -> Result<Vec<Arc<Display>>, Validated<VulkanError>> {
        self.validate_display_plane_supported_displays(plane_index)?;

        Ok(unsafe { self.display_plane_supported_displays_unchecked(plane_index) }?)
    }

    fn validate_display_plane_supported_displays(
        &self,
        plane_index: u32,
    ) -> Result<(), Box<ValidationError>> {
        if !self.instance.enabled_extensions().khr_display {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "khr_display",
                )])]),
                ..Default::default()
            }));
        }

        let display_plane_properties_raw =
            unsafe { self.display_plane_properties_raw() }.map_err(|_err| {
                Box::new(ValidationError {
                    problem: "`PhysicalDevice::display_plane_properties` \
                        returned an error"
                        .into(),
                    ..Default::default()
                })
            })?;

        if plane_index as usize >= display_plane_properties_raw.len() {
            return Err(Box::new(ValidationError {
                problem: "`plane_index` is not less than the number of display planes on the \
                    physical device"
                    .into(),
                vuids: &["VUID-vkGetDisplayPlaneSupportedDisplaysKHR-planeIndex-01249"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn display_plane_supported_displays_unchecked(
        self: &Arc<Self>,
        plane_index: u32,
    ) -> Result<Vec<Arc<Display>>, VulkanError> {
        let fns = self.instance.fns();

        let displays_vk = loop {
            let mut count = 0;
            unsafe {
                (fns.khr_display.get_display_plane_supported_displays_khr)(
                    self.handle,
                    plane_index,
                    &mut count,
                    ptr::null_mut(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;

            let mut displays = Vec::with_capacity(count as usize);
            let result = unsafe {
                (fns.khr_display.get_display_plane_supported_displays_khr)(
                    self.handle,
                    plane_index,
                    &mut count,
                    displays.as_mut_ptr(),
                )
            };

            match result {
                vk::Result::SUCCESS => {
                    unsafe { displays.set_len(count as usize) };
                    break displays;
                }
                vk::Result::INCOMPLETE => (),
                err => return Err(VulkanError::from(err)),
            }
        };

        let displays: Vec<_> = displays_vk
            .into_iter()
            .map(|display_vk| -> Result<_, VulkanError> {
                Ok(
                    if let Some(display) = self.display_properties.get(&display_vk) {
                        display
                    } else {
                        unsafe { self.display_properties_unchecked() }?;
                        self.display_properties.get(&display_vk).unwrap()
                    },
                )
            })
            .collect::<Result<_, _>>()?;

        Ok(displays)
    }

    /// Retrieves the external memory properties supported for buffers with a given configuration,
    /// panicking on a validation error.
    ///
    /// Instance API version must be at least 1.1, or the [`khr_external_memory_capabilities`]
    /// extension must be enabled on the instance.
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    ///
    /// This is a shortcut for `try_external_buffer_properties().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_external_buffer_properties`] returns a [`ValidationError`].
    ///
    /// [`khr_external_memory_capabilities`]: crate::instance::InstanceExtensions::khr_external_memory_capabilities
    /// [`try_external_buffer_properties`]: Self::try_external_buffer_properties
    #[inline]
    #[track_caller]
    pub fn external_buffer_properties(
        &self,
        info: &ExternalBufferInfo<'_>,
    ) -> ExternalBufferProperties {
        self.try_external_buffer_properties(info).unwrap()
    }

    /// Retrieves the external memory properties supported for buffers with a given configuration.
    ///
    /// Instance API version must be at least 1.1, or the [`khr_external_memory_capabilities`]
    /// extension must be enabled on the instance.
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    ///
    /// [`khr_external_memory_capabilities`]: crate::instance::InstanceExtensions::khr_external_memory_capabilities
    #[inline]
    pub fn try_external_buffer_properties(
        &self,
        info: &ExternalBufferInfo<'_>,
    ) -> Result<ExternalBufferProperties, Box<ValidationError>> {
        self.validate_external_buffer_properties(info)?;

        Ok(unsafe { self.external_buffer_properties_unchecked(info) })
    }

    fn validate_external_buffer_properties(
        &self,
        info: &ExternalBufferInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        if !(self.instance.api_version() >= Version::V1_1
            || self
                .instance
                .enabled_extensions()
                .khr_external_memory_capabilities)
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                    RequiresAllOf(&[Requires::InstanceExtension(
                        "khr_external_memory_capabilities",
                    )]),
                ]),
                ..Default::default()
            }));
        }

        info.validate(self).map_err(|err| err.add_context("info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn external_buffer_properties_unchecked(
        &self,
        info: &ExternalBufferInfo<'_>,
    ) -> ExternalBufferProperties {
        self.external_buffer_properties
            .get_or_insert(info.wrap(), || {
                /* Input */

                let info_vk = info.to_vk();

                /* Output */

                let mut properties_vk = ExternalBufferProperties::to_mut_vk();

                /* Call */

                let fns = self.instance.fns();

                if self.instance.api_version() >= Version::V1_1 {
                    unsafe {
                        (fns.v1_1.get_physical_device_external_buffer_properties)(
                            self.handle,
                            &info_vk,
                            &mut properties_vk,
                        )
                    }
                } else {
                    unsafe {
                        (fns.khr_external_memory_capabilities
                            .get_physical_device_external_buffer_properties_khr)(
                            self.handle,
                            &info_vk,
                            &mut properties_vk,
                        )
                    };
                }

                (
                    info.to_owned(),
                    ExternalBufferProperties::from_vk(&properties_vk),
                )
            })
    }

    /// Retrieves the external handle properties supported for fences with a given
    /// configuration, panicking on a validation error.
    ///
    /// The instance API version must be at least 1.1, or the [`khr_external_fence_capabilities`]
    /// extension must be enabled on the instance.
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    ///
    /// This is a shortcut for `try_external_fence_properties().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_external_fence_properties`] returns a [`ValidationError`].
    ///
    /// [`khr_external_fence_capabilities`]: crate::instance::InstanceExtensions::khr_external_fence_capabilities
    /// [`try_external_fence_properties`]: Self::try_external_fence_properties
    #[inline]
    #[track_caller]
    pub fn external_fence_properties(
        &self,
        info: &ExternalFenceInfo<'_>,
    ) -> ExternalFenceProperties {
        self.try_external_fence_properties(info).unwrap()
    }

    /// Retrieves the external handle properties supported for fences with a given
    /// configuration.
    ///
    /// The instance API version must be at least 1.1, or the [`khr_external_fence_capabilities`]
    /// extension must be enabled on the instance.
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    ///
    /// [`khr_external_fence_capabilities`]: crate::instance::InstanceExtensions::khr_external_fence_capabilities
    #[inline]
    pub fn try_external_fence_properties(
        &self,
        info: &ExternalFenceInfo<'_>,
    ) -> Result<ExternalFenceProperties, Box<ValidationError>> {
        self.validate_external_fence_properties(info)?;

        Ok(unsafe { self.external_fence_properties_unchecked(info) })
    }

    fn validate_external_fence_properties(
        &self,
        info: &ExternalFenceInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        if !(self.instance.api_version() >= Version::V1_1
            || self
                .instance
                .enabled_extensions()
                .khr_external_fence_capabilities)
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                    RequiresAllOf(&[Requires::InstanceExtension(
                        "khr_external_fence_capabilities",
                    )]),
                ]),
                ..Default::default()
            }));
        }

        info.validate(self).map_err(|err| err.add_context("info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn external_fence_properties_unchecked(
        &self,
        info: &ExternalFenceInfo<'_>,
    ) -> ExternalFenceProperties {
        self.external_fence_properties
            .get_or_insert(info.wrap(), || {
                /* Input */

                let info_vk = info.to_vk();

                /* Output */

                let mut properties_vk = ExternalFenceProperties::to_mut_vk();

                /* Call */

                let fns = self.instance.fns();

                if self.instance.api_version() >= Version::V1_1 {
                    unsafe {
                        (fns.v1_1.get_physical_device_external_fence_properties)(
                            self.handle,
                            &info_vk,
                            &mut properties_vk,
                        )
                    }
                } else {
                    unsafe {
                        (fns.khr_external_fence_capabilities
                            .get_physical_device_external_fence_properties_khr)(
                            self.handle,
                            &info_vk,
                            &mut properties_vk,
                        )
                    };
                }

                (
                    info.to_owned(),
                    ExternalFenceProperties::from_vk(&properties_vk),
                )
            })
    }

    /// Retrieves the external handle properties supported for semaphores with a given
    /// configuration, panicking on a validation error
    ///
    /// The instance API version must be at least 1.1, or the
    /// [`khr_external_semaphore_capabilities`] extension must be enabled on the instance.
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    ///
    /// This is a shortcut for `try_external_semaphore_properties().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_external_semaphore_properties`] returns a [`ValidationError`].
    ///
    /// [`khr_external_semaphore_capabilities`]: crate::instance::InstanceExtensions::khr_external_semaphore_capabilities
    /// [`try_external_semaphore_properties`]: Self::try_external_semaphore_properties
    #[inline]
    #[track_caller]
    pub fn external_semaphore_properties(
        &self,
        info: &ExternalSemaphoreInfo<'_>,
    ) -> ExternalSemaphoreProperties {
        self.try_external_semaphore_properties(info).unwrap()
    }

    /// Retrieves the external handle properties supported for semaphores with a given
    /// configuration.
    ///
    /// The instance API version must be at least 1.1, or the
    /// [`khr_external_semaphore_capabilities`] extension must be enabled on the instance.
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    ///
    /// [`khr_external_semaphore_capabilities`]: crate::instance::InstanceExtensions::khr_external_semaphore_capabilities
    #[inline]
    pub fn try_external_semaphore_properties(
        &self,
        info: &ExternalSemaphoreInfo<'_>,
    ) -> Result<ExternalSemaphoreProperties, Box<ValidationError>> {
        self.validate_external_semaphore_properties(info)?;

        Ok(unsafe { self.external_semaphore_properties_unchecked(info) })
    }

    fn validate_external_semaphore_properties(
        &self,
        info: &ExternalSemaphoreInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        if !(self.instance.api_version() >= Version::V1_1
            || self
                .instance
                .enabled_extensions()
                .khr_external_semaphore_capabilities)
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                    RequiresAllOf(&[Requires::InstanceExtension(
                        "khr_external_semaphore_capabilities",
                    )]),
                ]),
                ..Default::default()
            }));
        }

        info.validate(self).map_err(|err| err.add_context("info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn external_semaphore_properties_unchecked(
        &self,
        info: &ExternalSemaphoreInfo<'_>,
    ) -> ExternalSemaphoreProperties {
        self.external_semaphore_properties
            .get_or_insert(info.wrap(), || {
                /* Input */

                let mut info_extensions_vk = info.to_vk_extensions();
                let info_vk = info.to_vk(&mut info_extensions_vk);

                /* Output */

                let mut properties_vk = ExternalSemaphoreProperties::to_mut_vk();

                /* Call */

                let fns = self.instance.fns();

                if self.instance.api_version() >= Version::V1_1 {
                    unsafe {
                        (fns.v1_1.get_physical_device_external_semaphore_properties)(
                            self.handle,
                            &info_vk,
                            &mut properties_vk,
                        )
                    }
                } else {
                    unsafe {
                        (fns.khr_external_semaphore_capabilities
                            .get_physical_device_external_semaphore_properties_khr)(
                            self.handle,
                            &info_vk,
                            &mut properties_vk,
                        )
                    };
                }

                (
                    info.to_owned(),
                    ExternalSemaphoreProperties::from_vk(&properties_vk),
                )
            })
    }

    /// Retrieves the properties of a format when used by this physical device, panicking on a
    /// validation error.
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    ///
    /// This is a shortcut for `try_format_properties().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_format_properties`] returns a [`ValidationError`].
    ///
    /// [`try_format_properties`]: Self::try_format_properties
    #[inline]
    #[track_caller]
    pub fn format_properties(&self, format: Format) -> FormatProperties {
        self.try_format_properties(format).unwrap()
    }

    /// Retrieves the properties of a format when used by this physical device.
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    #[inline]
    pub fn try_format_properties(
        &self,
        format: Format,
    ) -> Result<FormatProperties, Box<ValidationError>> {
        self.validate_format_properties(format)?;

        Ok(unsafe { self.format_properties_unchecked(format) })
    }

    fn validate_format_properties(&self, format: Format) -> Result<(), Box<ValidationError>> {
        format.validate_physical_device(self).map_err(|err| {
            err.add_context("format")
                .set_vuids(&["VUID-vkGetPhysicalDeviceFormatProperties2-format-parameter"])
        })?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn format_properties_unchecked(&self, format: Format) -> FormatProperties {
        self.format_properties.get_or_insert(&format, || {
            let fns = self.instance.fns();
            let call = |format_properties2_vk: &mut vk::FormatProperties2<'_>| {
                if self.api_version() >= Version::V1_1 {
                    unsafe {
                        (fns.v1_1.get_physical_device_format_properties2)(
                            self.handle,
                            format.into(),
                            format_properties2_vk,
                        )
                    };
                } else if self
                    .instance
                    .enabled_extensions()
                    .khr_get_physical_device_properties2
                {
                    unsafe {
                        (fns.khr_get_physical_device_properties2
                            .get_physical_device_format_properties2_khr)(
                            self.handle,
                            format.into(),
                            format_properties2_vk,
                        )
                    };
                } else {
                    unsafe {
                        (fns.v1_0.get_physical_device_format_properties)(
                            self.handle(),
                            format.into(),
                            &mut format_properties2_vk.format_properties,
                        )
                    };
                }
            };

            let mut properties2_fields1_vk = FormatProperties::to_mut_vk2_fields1(
                FormatProperties::to_mut_vk2_extensions_query_count(self).map(
                    |mut properties2_extensions_query_count_vk| {
                        // If `to_mut_vk2_extensions_query_count` returns `Some`, we must query
                        // the element count and then pass it to `to_mut_vk2_fields1`.
                        let mut properties2_query_count_vk = FormatProperties::to_mut_vk2(
                            &mut properties2_extensions_query_count_vk,
                        );
                        call(&mut properties2_query_count_vk);
                        properties2_extensions_query_count_vk
                    },
                ),
            );
            let mut properties2_extensions_vk =
                FormatProperties::to_mut_vk2_extensions(&mut properties2_fields1_vk, self);
            let mut properties2_vk = FormatProperties::to_mut_vk2(&mut properties2_extensions_vk);

            call(&mut properties2_vk);

            // Unborrow
            let properties2_vk = vk::FormatProperties2 {
                _marker: PhantomData,
                ..properties2_vk
            };
            let properties2_extensions_vk = properties2_extensions_vk.unborrow();

            (
                format,
                FormatProperties::from_vk2(
                    &properties2_vk,
                    &properties2_fields1_vk,
                    &properties2_extensions_vk,
                ),
            )
        })
    }

    /// Returns the properties supported for images with a given image configuration, panicking on
    /// a validation error.
    ///
    /// `Some` is returned if the configuration is supported, `None` if it is not.
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    ///
    /// This is a shortcut for `try_image_format_properties().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_image_format_properties`] returns a [`ValidationError`].
    /// - Panics if `image_format_info.format` is `None`.
    ///
    /// [`try_image_format_properties`]: Self::try_image_format_properties
    #[inline]
    #[track_caller]
    pub fn image_format_properties(
        &self,
        image_format_info: &ImageFormatInfo<'_>,
    ) -> Result<Option<ImageFormatProperties>, VulkanError> {
        match self.try_image_format_properties(image_format_info) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Returns the properties supported for images with a given image configuration.
    ///
    /// `Some` is returned if the configuration is supported, `None` if it is not.
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    ///
    /// # Panics
    ///
    /// - Panics if `image_format_info.format` is `None`.
    #[inline]
    pub fn try_image_format_properties(
        &self,
        image_format_info: &ImageFormatInfo<'_>,
    ) -> Result<Option<ImageFormatProperties>, Validated<VulkanError>> {
        self.validate_image_format_properties(image_format_info)?;

        Ok(unsafe { self.image_format_properties_unchecked(image_format_info) }?)
    }

    fn validate_image_format_properties(
        &self,
        image_format_info: &ImageFormatInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        image_format_info
            .validate(self)
            .map_err(|err| err.add_context("image_format_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn image_format_properties_unchecked(
        &self,
        image_format_info: &ImageFormatInfo<'_>,
    ) -> Result<Option<ImageFormatProperties>, VulkanError> {
        self.image_format_properties
            .get_or_try_insert(image_format_info.wrap(), || {
                /* Input */
                let info2_fields1_vk = image_format_info.to_vk2_fields1();
                let mut info2_extensions_vk =
                    image_format_info.to_vk2_extensions(&info2_fields1_vk);
                let info2_vk = image_format_info.to_vk2(&mut info2_extensions_vk);

                /* Output */

                let mut properties2_extensions_vk =
                    ImageFormatProperties::to_mut_vk2_extensions(image_format_info);
                let mut properties2_vk =
                    ImageFormatProperties::to_mut_vk2(&mut properties2_extensions_vk);

                let result = {
                    let fns = self.instance.fns();

                    if self.api_version() >= Version::V1_1 {
                        unsafe {
                            (fns.v1_1.get_physical_device_image_format_properties2)(
                                self.handle,
                                &info2_vk,
                                &mut properties2_vk,
                            )
                        }
                    } else if self
                        .instance
                        .enabled_extensions()
                        .khr_get_physical_device_properties2
                    {
                        unsafe {
                            (fns.khr_get_physical_device_properties2
                                .get_physical_device_image_format_properties2_khr)(
                                self.handle,
                                &info2_vk,
                                &mut properties2_vk,
                            )
                        }
                    } else {
                        // Can't query this, return unsupported
                        if !info2_vk.p_next.is_null() {
                            return Ok((image_format_info.to_owned(), None));
                        }
                        if let Some(ExternalMemoryHandleType::DmaBuf) =
                            image_format_info.external_memory_handle_type
                        {
                            // VUID-vkGetPhysicalDeviceImageFormatProperties-tiling-02248
                            // VUID-VkPhysicalDeviceImageFormatInfo2-tiling-02249
                            return Ok((image_format_info.to_owned(), None));
                        }

                        unsafe {
                            (fns.v1_0.get_physical_device_image_format_properties)(
                                self.handle,
                                info2_vk.format,
                                info2_vk.ty,
                                info2_vk.tiling,
                                info2_vk.usage,
                                info2_vk.flags,
                                &mut properties2_vk.image_format_properties,
                            )
                        }
                    }
                    .result()
                    .map_err(VulkanError::from)
                };

                // Unborrow
                let properties2_vk = vk::ImageFormatProperties2 {
                    _marker: PhantomData,
                    ..properties2_vk
                };

                match result {
                    Ok(_) => Ok((
                        image_format_info.to_owned(),
                        Some(ImageFormatProperties::from_vk2(
                            &properties2_vk,
                            &properties2_extensions_vk,
                        )),
                    )),
                    Err(VulkanError::FormatNotSupported) => {
                        Ok((image_format_info.to_owned(), None))
                    }
                    Err(err) => Err(err),
                }
            })
    }

    /// Returns the properties of sparse images with a given image configuration, panicking on a
    /// validation error.
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    ///
    /// This is a shortcut for `try_sparse_image_format_properties().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_sparse_image_format_properties`] returns a [`ValidationError`].
    /// - Panics if `format_info.format` is `None`.
    ///
    /// [`try_sparse_image_format_properties`]: Self::try_sparse_image_format_properties
    #[inline]
    #[track_caller]
    pub fn sparse_image_format_properties(
        &self,
        format_info: &SparseImageFormatInfo<'_>,
    ) -> Vec<SparseImageFormatProperties> {
        self.try_sparse_image_format_properties(format_info)
            .unwrap()
    }

    /// Returns the properties of sparse images with a given image configuration.
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    ///
    /// # Panics
    ///
    /// - Panics if `format_info.format` is `None`.
    #[inline]
    pub fn try_sparse_image_format_properties(
        &self,
        format_info: &SparseImageFormatInfo<'_>,
    ) -> Result<Vec<SparseImageFormatProperties>, Box<ValidationError>> {
        self.validate_sparse_image_format_properties(format_info)?;

        Ok(unsafe { self.sparse_image_format_properties_unchecked(format_info) })
    }

    fn validate_sparse_image_format_properties(
        &self,
        format_info: &SparseImageFormatInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        format_info
            .validate(self)
            .map_err(|err| err.add_context("format_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn sparse_image_format_properties_unchecked(
        &self,
        format_info: &SparseImageFormatInfo<'_>,
    ) -> Vec<SparseImageFormatProperties> {
        self.sparse_image_format_properties
            .get_or_insert(format_info.wrap(), || {
                let format_info2_vk = format_info.to_vk();

                let fns = self.instance.fns();

                if self.api_version() >= Version::V1_1
                    || self
                        .instance
                        .enabled_extensions()
                        .khr_get_physical_device_properties2
                {
                    let mut count = 0;

                    if self.api_version() >= Version::V1_1 {
                        unsafe {
                            (fns.v1_1.get_physical_device_sparse_image_format_properties2)(
                                self.handle,
                                &format_info2_vk,
                                &mut count,
                                ptr::null_mut(),
                            )
                        };
                    } else {
                        unsafe {
                            (fns.khr_get_physical_device_properties2
                                .get_physical_device_sparse_image_format_properties2_khr)(
                                self.handle,
                                &format_info2_vk,
                                &mut count,
                                ptr::null_mut(),
                            )
                        };
                    }

                    let mut sparse_image_format_properties2 =
                        vec![SparseImageFormatProperties::to_mut_vk2(); count as usize];

                    if self.api_version() >= Version::V1_1 {
                        unsafe {
                            (fns.v1_1.get_physical_device_sparse_image_format_properties2)(
                                self.handle,
                                &format_info2_vk,
                                &mut count,
                                sparse_image_format_properties2.as_mut_ptr(),
                            )
                        };
                    } else {
                        unsafe {
                            (fns.khr_get_physical_device_properties2
                                .get_physical_device_sparse_image_format_properties2_khr)(
                                self.handle,
                                &format_info2_vk,
                                &mut count,
                                sparse_image_format_properties2.as_mut_ptr(),
                            )
                        };
                    }

                    unsafe { sparse_image_format_properties2.set_len(count as usize) };

                    (
                        format_info.to_owned(),
                        sparse_image_format_properties2
                            .into_iter()
                            .map(|properties2_vk| {
                                SparseImageFormatProperties::from_vk(&properties2_vk.properties)
                            })
                            .collect(),
                    )
                } else {
                    let mut count = 0;

                    unsafe {
                        (fns.v1_0.get_physical_device_sparse_image_format_properties)(
                            self.handle,
                            format_info2_vk.format,
                            format_info2_vk.ty,
                            format_info2_vk.samples,
                            format_info2_vk.usage,
                            format_info2_vk.tiling,
                            &mut count,
                            ptr::null_mut(),
                        )
                    };

                    let mut sparse_image_format_properties =
                        vec![SparseImageFormatProperties::to_mut_vk(); count as usize];

                    unsafe {
                        (fns.v1_0.get_physical_device_sparse_image_format_properties)(
                            self.handle,
                            format_info2_vk.format,
                            format_info2_vk.ty,
                            format_info2_vk.samples,
                            format_info2_vk.usage,
                            format_info2_vk.tiling,
                            &mut count,
                            sparse_image_format_properties.as_mut_ptr(),
                        )
                    };

                    unsafe { sparse_image_format_properties.set_len(count as usize) };

                    (
                        format_info.to_owned(),
                        sparse_image_format_properties
                            .into_iter()
                            .map(|properties_vk| {
                                SparseImageFormatProperties::from_vk(&properties_vk)
                            })
                            .collect(),
                    )
                }
            })
    }

    /// Returns the capabilities that are supported by the physical device for the given surface,
    /// panicking on a validation error.
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    ///
    /// This is a shortcut for `try_surface_capabilities().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_surface_capabilities`] returns a [`ValidationError`].
    /// - Panics if the physical device and the surface don't belong to the same instance.
    ///
    /// [`try_surface_capabilities`]: Self::try_surface_capabilities
    #[track_caller]
    pub fn surface_capabilities(
        &self,
        surface: &Surface,
        surface_info: &SurfaceInfo<'_>,
    ) -> Result<SurfaceCapabilities, VulkanError> {
        match self.try_surface_capabilities(surface, surface_info) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Returns the capabilities that are supported by the physical device for the given surface.
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    ///
    /// # Panics
    ///
    /// - Panics if the physical device and the surface don't belong to the same instance.
    pub fn try_surface_capabilities(
        &self,
        surface: &Surface,
        surface_info: &SurfaceInfo<'_>,
    ) -> Result<SurfaceCapabilities, Validated<VulkanError>> {
        self.validate_surface_capabilities(surface, surface_info)?;

        Ok(unsafe { self.surface_capabilities_unchecked(surface, surface_info) }?)
    }

    fn validate_surface_capabilities(
        &self,
        surface: &Surface,
        surface_info: &SurfaceInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        if !(self
            .instance
            .enabled_extensions()
            .khr_get_surface_capabilities2
            || self.instance.enabled_extensions().khr_surface)
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::InstanceExtension("khr_get_surface_capabilities2")]),
                    RequiresAllOf(&[Requires::InstanceExtension("khr_surface")]),
                ]),
                ..Default::default()
            }));
        }

        // VUID-vkGetPhysicalDeviceSurfaceCapabilities2KHR-commonparent
        assert_eq!(self.instance(), surface.instance());

        if !(0..self.queue_family_properties.len() as u32).any(|index| {
            unsafe { self.surface_support_unchecked(index, surface) }.unwrap_or_default()
        }) {
            return Err(Box::new(ValidationError {
                context: "surface".into(),
                problem: "is not supported by the physical device".into(),
                vuids: &["VUID-vkGetPhysicalDeviceSurfaceCapabilities2KHR-pSurfaceInfo-06210"],
                ..Default::default()
            }));
        }

        surface_info
            .validate(self)
            .map_err(|err| err.add_context("surface_info"))?;

        let &SurfaceInfo {
            present_mode,
            full_screen_exclusive,
            win32_monitor,
            _ne: _,
        } = surface_info;

        if let Some(present_mode) = present_mode {
            let present_modes = unsafe {
                self.surface_present_modes_unchecked(
                    surface,
                    &SurfaceInfo {
                        present_mode: None,
                        ..*surface_info
                    },
                )
            }
            .map_err(|_err| {
                Box::new(ValidationError {
                    problem: "`PhysicalDevice::surface_present_modes` \
                        returned an error"
                        .into(),
                    ..Default::default()
                })
            })?;

            if !present_modes.into_iter().any(|mode| mode == present_mode) {
                return Err(Box::new(ValidationError {
                    problem: "`surface_info.present_mode` is not supported for `surface`".into(),
                    vuids: &["VUID-VkSurfacePresentModeEXT-presentMode-07780"],
                    ..Default::default()
                }));
            }
        }

        if surface.api() == SurfaceApi::Win32 {
            if full_screen_exclusive == FullScreenExclusive::ApplicationControlled
                && win32_monitor.is_none()
            {
                return Err(Box::new(ValidationError {
                    problem: "`surface` is a Win32 surface, and \
                        `surface_info.full_screen_exclusive` is \
                        `FullScreenExclusive::ApplicationControlled`, but \
                        `surface_info.win32_monitor` is `None`"
                        .into(),
                    vuids: &["VUID-VkPhysicalDeviceSurfaceInfo2KHR-pNext-02672"],
                    ..Default::default()
                }));
            }
        } else if win32_monitor.is_some() {
            return Err(Box::new(ValidationError {
                problem: "`surface` is not a Win32 surface, but `surface_info.win32_monitor` is \
                    `Some`"
                    .into(),
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn surface_capabilities_unchecked(
        &self,
        surface: &Surface,
        surface_info: &SurfaceInfo<'_>,
    ) -> Result<SurfaceCapabilities, VulkanError> {
        /* Input */

        let mut info2_extensions_vk = surface_info.to_vk2_extensions();
        let info2_vk = surface_info.to_vk2(surface.handle(), &mut info2_extensions_vk);

        /* Output */

        let mut capabilities_fields1_vk = SurfaceCapabilities::to_mut_vk2_fields();
        let mut capabilities_extensions_vk = SurfaceCapabilities::to_mut_vk2_extensions(
            &mut capabilities_fields1_vk,
            self,
            surface_info,
        );
        let mut capabilities_vk = SurfaceCapabilities::to_mut_vk2(&mut capabilities_extensions_vk);

        let fns = self.instance.fns();

        if self
            .instance
            .enabled_extensions()
            .khr_get_surface_capabilities2
        {
            unsafe {
                (fns.khr_get_surface_capabilities2
                    .get_physical_device_surface_capabilities2_khr)(
                    self.handle(),
                    &info2_vk,
                    &mut capabilities_vk,
                )
            }
            .result()
            .map_err(VulkanError::from)?;
        } else {
            unsafe {
                (fns.khr_surface.get_physical_device_surface_capabilities_khr)(
                    self.handle(),
                    info2_vk.surface,
                    &mut capabilities_vk.surface_capabilities,
                )
            }
            .result()
            .map_err(VulkanError::from)?;
        };

        // Unborrow
        let capabilities_vk = vk::SurfaceCapabilities2KHR {
            _marker: PhantomData,
            ..capabilities_vk
        };
        let capabilities_extensions_vk = capabilities_extensions_vk.unborrow();

        Ok(SurfaceCapabilities::from_vk2(
            &capabilities_vk,
            &capabilities_fields1_vk,
            &capabilities_extensions_vk,
        ))
    }

    /// Returns the combinations of format and color space that are supported by the physical
    /// device for the given surface, panicking on a validation error.
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    ///
    /// This is a shortcut for `try_surface_formats().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_surface_formats`] returns a [`ValidationError`].
    /// - Panics if the physical device and the surface don't belong to the same instance.
    ///
    /// [`try_surface_formats`]: Self::try_surface_formats
    #[track_caller]
    pub fn surface_formats(
        &self,
        surface: &Surface,
        surface_info: &SurfaceInfo<'_>,
    ) -> Result<Vec<(Format, ColorSpace)>, VulkanError> {
        match self.try_surface_formats(surface, surface_info) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Returns the combinations of format and color space that are supported by the physical
    /// device for the given surface.
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    ///
    /// # Panics
    ///
    /// - Panics if the physical device and the surface don't belong to the same instance.
    pub fn try_surface_formats(
        &self,
        surface: &Surface,
        surface_info: &SurfaceInfo<'_>,
    ) -> Result<Vec<(Format, ColorSpace)>, Validated<VulkanError>> {
        self.validate_surface_formats(surface, surface_info)?;

        Ok(unsafe { self.surface_formats_unchecked(surface, surface_info) }?)
    }

    fn validate_surface_formats(
        &self,
        surface: &Surface,
        surface_info: &SurfaceInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        if !(self
            .instance
            .enabled_extensions()
            .khr_get_surface_capabilities2
            || self.instance.enabled_extensions().khr_surface)
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::InstanceExtension("khr_get_surface_capabilities2")]),
                    RequiresAllOf(&[Requires::InstanceExtension("khr_surface")]),
                ]),
                ..Default::default()
            }));
        }

        // VUID-vkGetPhysicalDeviceSurfaceFormats2KHR-commonparent
        assert_eq!(self.instance(), surface.instance());

        if !(0..self.queue_family_properties.len() as u32).any(|index| {
            unsafe { self.surface_support_unchecked(index, surface) }.unwrap_or_default()
        }) {
            return Err(Box::new(ValidationError {
                context: "surface".into(),
                problem: "is not supported by the physical device".into(),
                vuids: &["VUID-vkGetPhysicalDeviceSurfaceFormats2KHR-pSurfaceInfo-06522"],
                ..Default::default()
            }));
        }

        surface_info
            .validate(self)
            .map_err(|err| err.add_context("surface_info"))?;

        let &SurfaceInfo {
            present_mode,
            full_screen_exclusive,
            win32_monitor,
            _ne: _,
        } = surface_info;

        if let Some(present_mode) = present_mode {
            let present_modes = unsafe {
                self.surface_present_modes_unchecked(
                    surface,
                    &SurfaceInfo {
                        present_mode: None,
                        ..*surface_info
                    },
                )
            }
            .map_err(|_err| {
                Box::new(ValidationError {
                    problem: "`PhysicalDevice::surface_present_modes` \
                        returned an error"
                        .into(),
                    ..Default::default()
                })
            })?;

            if !present_modes.into_iter().any(|mode| mode == present_mode) {
                return Err(Box::new(ValidationError {
                    problem: "`surface_info.present_mode` is not supported for `surface`".into(),
                    vuids: &["VUID-VkSurfacePresentModeEXT-presentMode-07780"],
                    ..Default::default()
                }));
            }
        }

        if win32_monitor.is_some() {
            if surface.api() != SurfaceApi::Win32 {
                return Err(Box::new(ValidationError {
                    problem: "`surface_info.win32_monitor` is `Some`, but \
                        `surface` is not a Win32 surface"
                        .into(),
                    ..Default::default()
                }));
            }
        } else if surface.api() == SurfaceApi::Win32
            && full_screen_exclusive == FullScreenExclusive::ApplicationControlled
        {
            return Err(Box::new(ValidationError {
                problem: "`surface` is a Win32 surface, and \
                    `surface_info.full_screen_exclusive` is \
                    `FullScreenExclusive::ApplicationControlled`, but \
                    `surface_info.win32_monitor` is `None`"
                    .into(),
                vuids: &["VUID-VkPhysicalDeviceSurfaceInfo2KHR-pNext-02672"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn surface_formats_unchecked(
        &self,
        surface: &Surface,
        surface_info: &SurfaceInfo<'_>,
    ) -> Result<Vec<(Format, ColorSpace)>, VulkanError> {
        surface
            .surface_formats
            .get_or_try_insert(&(self.handle, surface_info.to_owned()), || {
                let mut info2_extensions_vk = surface_info.to_vk2_extensions();
                let info2_vk = surface_info.to_vk2(surface.handle(), &mut info2_extensions_vk);

                let fns = self.instance.fns();

                if self
                    .instance
                    .enabled_extensions()
                    .khr_get_surface_capabilities2
                {
                    let surface_format2s_vk = loop {
                        let mut count = 0;
                        unsafe {
                            (fns.khr_get_surface_capabilities2
                                .get_physical_device_surface_formats2_khr)(
                                self.handle(),
                                &info2_vk,
                                &mut count,
                                ptr::null_mut(),
                            )
                        }
                        .result()
                        .map_err(VulkanError::from)?;

                        let mut surface_format2s_vk =
                            vec![vk::SurfaceFormat2KHR::default(); count as usize];
                        let result = unsafe {
                            (fns.khr_get_surface_capabilities2
                                .get_physical_device_surface_formats2_khr)(
                                self.handle(),
                                &info2_vk,
                                &mut count,
                                surface_format2s_vk.as_mut_ptr(),
                            )
                        };

                        match result {
                            vk::Result::SUCCESS => {
                                unsafe { surface_format2s_vk.set_len(count as usize) };
                                break surface_format2s_vk;
                            }
                            vk::Result::INCOMPLETE => (),
                            err => return Err(VulkanError::from(err)),
                        }
                    };

                    Ok((
                        (self.handle, surface_info.to_owned()),
                        surface_format2s_vk
                            .iter()
                            .filter_map(|surface_format2_vk| {
                                let &vk::SurfaceFormat2KHR {
                                    surface_format:
                                        vk::SurfaceFormatKHR {
                                            format,
                                            color_space,
                                        },
                                    ..
                                } = surface_format2_vk;

                                format.try_into().ok().zip(color_space.try_into().ok())
                            })
                            .collect(),
                    ))
                } else {
                    let surface_formats = loop {
                        let mut count = 0;
                        unsafe {
                            (fns.khr_surface.get_physical_device_surface_formats_khr)(
                                self.handle(),
                                surface.handle(),
                                &mut count,
                                ptr::null_mut(),
                            )
                        }
                        .result()
                        .map_err(VulkanError::from)?;

                        let mut surface_formats = Vec::with_capacity(count as usize);
                        let result = unsafe {
                            (fns.khr_surface.get_physical_device_surface_formats_khr)(
                                self.handle(),
                                surface.handle(),
                                &mut count,
                                surface_formats.as_mut_ptr(),
                            )
                        };

                        match result {
                            vk::Result::SUCCESS => {
                                unsafe { surface_formats.set_len(count as usize) };
                                break surface_formats;
                            }
                            vk::Result::INCOMPLETE => (),
                            err => return Err(VulkanError::from(err)),
                        }
                    };

                    Ok((
                        (self.handle, surface_info.to_owned()),
                        surface_formats
                            .iter()
                            .filter_map(|surface_format_vk| {
                                let &vk::SurfaceFormatKHR {
                                    format,
                                    color_space,
                                } = surface_format_vk;

                                format.try_into().ok().zip(color_space.try_into().ok())
                            })
                            .collect(),
                    ))
                }
            })
    }

    /// Returns the present modes that are supported by the physical device for the given surface,
    /// panicking on a validation error.
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    ///
    /// This is a shortcut for `try_surface_present_modes().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_surface_present_modes`] returns a [`ValidationError`].
    /// - Panics if the physical device and the surface don't belong to the same instance.
    ///
    /// [`try_surface_present_modes`]: Self::try_surface_present_modes
    #[track_caller]
    pub fn surface_present_modes(
        &self,
        surface: &Surface,
        surface_info: &SurfaceInfo<'_>,
    ) -> Result<Vec<PresentMode>, VulkanError> {
        match self.try_surface_present_modes(surface, surface_info) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Returns the present modes that are supported by the physical device for the given surface.
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    ///
    /// # Panics
    ///
    /// - Panics if the physical device and the surface don't belong to the same instance.
    pub fn try_surface_present_modes(
        &self,
        surface: &Surface,
        surface_info: &SurfaceInfo<'_>,
    ) -> Result<Vec<PresentMode>, Validated<VulkanError>> {
        self.validate_surface_present_modes(surface, surface_info)?;

        Ok(unsafe { self.surface_present_modes_unchecked(surface, surface_info) }?)
    }

    fn validate_surface_present_modes(
        &self,
        surface: &Surface,
        surface_info: &SurfaceInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        if !self.instance.enabled_extensions().khr_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "khr_surface",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-vkGetPhysicalDeviceSurfacePresentModesKHR-commonparent
        assert_eq!(self.instance(), surface.instance());

        if !(0..self.queue_family_properties.len() as u32).any(|index| {
            unsafe { self.surface_support_unchecked(index, surface) }.unwrap_or_default()
        }) {
            return Err(Box::new(ValidationError {
                context: "surface".into(),
                problem: "is not supported by the physical device".into(),
                vuids: &["VUID-vkGetPhysicalDeviceSurfacePresentModes2EXT-pSurfaceInfo-06522"],
                ..Default::default()
            }));
        }

        surface_info
            .validate(self)
            .map_err(|err| err.add_context("surface_info"))?;

        let &SurfaceInfo {
            present_mode,
            full_screen_exclusive,
            win32_monitor,
            _ne: _,
        } = surface_info;

        // We can't validate supported present modes while querying for supported present modes,
        // so just demand that it's `None` here.
        if present_mode.is_some() {
            return Err(Box::new(ValidationError {
                context: "surface_info.present_mode".into(),
                problem: "is `Some`".into(),
                ..Default::default()
            }));
        }

        if surface.api() == SurfaceApi::Win32 {
            if full_screen_exclusive == FullScreenExclusive::ApplicationControlled
                && win32_monitor.is_none()
            {
                return Err(Box::new(ValidationError {
                    problem: "`surface` is a Win32 surface, and \
                        `surface_info.full_screen_exclusive` is \
                        `FullScreenExclusive::ApplicationControlled`, but \
                        `surface_info.win32_monitor` is `None`"
                        .into(),
                    vuids: &["VUID-VkPhysicalDeviceSurfaceInfo2KHR-pNext-02672"],
                    ..Default::default()
                }));
            }
        } else if win32_monitor.is_some() {
            return Err(Box::new(ValidationError {
                problem: "`surface` is not a Win32 surface, but `surface_info.win32_monitor` is \
                    `Some`"
                    .into(),
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn surface_present_modes_unchecked(
        &self,
        surface: &Surface,
        surface_info: &SurfaceInfo<'_>,
    ) -> Result<Vec<PresentMode>, VulkanError> {
        surface.surface_present_modes.get_or_try_insert(
            &(self.handle, surface_info.to_owned()),
            || {
                let mut info2_extensions_vk = SurfaceInfo2ExtensionsVk {
                    present_mode_vk: None,
                    ..surface_info.to_vk2_extensions()
                };
                let info2_vk = surface_info.to_vk2(surface.handle(), &mut info2_extensions_vk);

                let fns = self.instance.fns();

                if self.supported_extensions().ext_full_screen_exclusive {
                    let modes = loop {
                        let mut count = 0;
                        unsafe {
                            (fns.ext_full_screen_exclusive
                                .get_physical_device_surface_present_modes2_ext)(
                                self.handle(),
                                &info2_vk,
                                &mut count,
                                ptr::null_mut(),
                            )
                        }
                        .result()
                        .map_err(VulkanError::from)?;

                        let mut modes = Vec::with_capacity(count as usize);
                        let result = unsafe {
                            (fns.ext_full_screen_exclusive
                                .get_physical_device_surface_present_modes2_ext)(
                                self.handle(),
                                &info2_vk,
                                &mut count,
                                modes.as_mut_ptr(),
                            )
                        };

                        match result {
                            vk::Result::SUCCESS => {
                                unsafe { modes.set_len(count as usize) };
                                break modes;
                            }
                            vk::Result::INCOMPLETE => (),
                            err => return Err(VulkanError::from(err)),
                        }
                    };

                    Ok((
                        (self.handle, surface_info.to_owned()),
                        modes
                            .into_iter()
                            .filter_map(|mode_vk| mode_vk.try_into().ok())
                            .collect(),
                    ))
                } else {
                    let modes = loop {
                        let mut count = 0;
                        unsafe {
                            (fns.khr_surface
                                .get_physical_device_surface_present_modes_khr)(
                                self.handle(),
                                surface.handle(),
                                &mut count,
                                ptr::null_mut(),
                            )
                        }
                        .result()
                        .map_err(VulkanError::from)?;

                        let mut modes = Vec::with_capacity(count as usize);
                        let result = unsafe {
                            (fns.khr_surface
                                .get_physical_device_surface_present_modes_khr)(
                                self.handle(),
                                surface.handle(),
                                &mut count,
                                modes.as_mut_ptr(),
                            )
                        };

                        match result {
                            vk::Result::SUCCESS => {
                                unsafe { modes.set_len(count as usize) };
                                break modes;
                            }
                            vk::Result::INCOMPLETE => (),
                            err => return Err(VulkanError::from(err)),
                        }
                    };

                    Ok((
                        (self.handle, surface_info.to_owned()),
                        modes
                            .into_iter()
                            .filter_map(|mode_vk| mode_vk.try_into().ok())
                            .collect(),
                    ))
                }
            },
        )
    }

    /// Returns whether queues of the given queue family support presentation to the given surface,
    /// panicking on a validation error.
    ///
    /// The results of this function are cached, so that future calls with the same arguments do
    /// not need to make a call to the Vulkan API again.
    ///
    /// See also [`presentation_support`] for determining if a queue family supports presentation
    /// to the surface of any window of a given event loop, for instance in cases where you have no
    /// window and hence no surface at hand to test with or when you could have multiple windows.
    ///
    /// This is a shortcut for `try_surface_support().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_surface_support`] returns a [`ValidationError`].
    ///
    /// [`presentation_support`]: Self::presentation_support
    /// [`try_surface_support`]: Self::try_surface_support
    #[inline]
    #[track_caller]
    pub fn surface_support(
        &self,
        queue_family_index: u32,
        surface: &Surface,
    ) -> Result<bool, VulkanError> {
        match self.try_surface_support(queue_family_index, surface) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Returns whether queues of the given queue family support presentation to the given surface.
    ///
    /// The results of this function are cached, so that future calls with the same arguments do
    /// not need to make a call to the Vulkan API again.
    ///
    /// See also [`presentation_support`] for determining if a queue family supports presentation
    /// to the surface of any window of a given event loop, for instance in cases where you have no
    /// window and hence no surface at hand to test with or when you could have multiple windows.
    ///
    /// [`presentation_support`]: Self::presentation_support
    #[inline]
    pub fn try_surface_support(
        &self,
        queue_family_index: u32,
        surface: &Surface,
    ) -> Result<bool, Validated<VulkanError>> {
        self.validate_surface_support(queue_family_index, surface)?;

        Ok(unsafe { self.surface_support_unchecked(queue_family_index, surface) }?)
    }

    fn validate_surface_support(
        &self,
        queue_family_index: u32,
        _surface: &Surface,
    ) -> Result<(), Box<ValidationError>> {
        if !self.instance.enabled_extensions().khr_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "khr_surface",
                )])]),
                ..Default::default()
            }));
        }

        if queue_family_index >= self.queue_family_properties.len() as u32 {
            return Err(Box::new(ValidationError {
                context: "queue_family_index".into(),
                problem: "is not less than the number of queue families in the physical device"
                    .into(),
                vuids: &["VUID-vkGetPhysicalDeviceSurfaceSupportKHR-queueFamilyIndex-01269"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn surface_support_unchecked(
        &self,
        queue_family_index: u32,
        surface: &Surface,
    ) -> Result<bool, VulkanError> {
        surface
            .surface_support
            .get_or_try_insert(&(self.handle, queue_family_index), || {
                let support = {
                    let fns = self.instance.fns();
                    let mut output = MaybeUninit::uninit();
                    unsafe {
                        (fns.khr_surface.get_physical_device_surface_support_khr)(
                            self.handle,
                            queue_family_index,
                            surface.handle(),
                            output.as_mut_ptr(),
                        )
                    }
                    .result()
                    .map_err(VulkanError::from)?;

                    unsafe { output.assume_init() }
                };

                Ok(((self.handle, queue_family_index), support != 0))
            })
    }

    /// Retrieves the properties of tools that are currently active on the physical device,
    /// panicking on a validation error.
    ///
    /// These properties may change during runtime, so the result only reflects the current
    /// situation and is not cached.
    ///
    /// The physical device API version must be at least 1.3, or the [`ext_tooling_info`]
    /// extension must be supported by the physical device.
    ///
    /// This is a shortcut for `try_tool_properties().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_tool_properties`] returns a [`ValidationError`].
    ///
    /// [`ext_tooling_info`]: crate::device::DeviceExtensions::ext_tooling_info
    /// [`try_tool_properties`]: Self::try_tool_properties
    #[inline]
    #[track_caller]
    pub fn tool_properties(&self) -> Result<Vec<ToolProperties>, VulkanError> {
        match self.try_tool_properties() {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Retrieves the properties of tools that are currently active on the physical device.
    ///
    /// These properties may change during runtime, so the result only reflects the current
    /// situation and is not cached.
    ///
    /// The physical device API version must be at least 1.3, or the [`ext_tooling_info`]
    /// extension must be supported by the physical device.
    ///
    /// [`ext_tooling_info`]: crate::device::DeviceExtensions::ext_tooling_info
    #[inline]
    pub fn try_tool_properties(&self) -> Result<Vec<ToolProperties>, Validated<VulkanError>> {
        self.validate_tool_properties()?;

        Ok(unsafe { self.tool_properties_unchecked() }?)
    }

    fn validate_tool_properties(&self) -> Result<(), Box<ValidationError>> {
        if !(self.api_version() >= Version::V1_3 || self.supported_extensions().ext_tooling_info) {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::DeviceExtension("ext_tooling_info")]),
                ]),
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn tool_properties_unchecked(&self) -> Result<Vec<ToolProperties>, VulkanError> {
        let fns = self.instance.fns();

        loop {
            let mut count = 0;

            if self.api_version() >= Version::V1_3 {
                unsafe {
                    (fns.v1_3.get_physical_device_tool_properties)(
                        self.handle(),
                        &mut count,
                        ptr::null_mut(),
                    )
                }
            } else {
                unsafe {
                    (fns.ext_tooling_info.get_physical_device_tool_properties_ext)(
                        self.handle(),
                        &mut count,
                        ptr::null_mut(),
                    )
                }
            }
            .result()
            .map_err(VulkanError::from)?;

            let mut tool_properties = Vec::with_capacity(count as usize);
            let result = if self.api_version() >= Version::V1_3 {
                unsafe {
                    (fns.v1_3.get_physical_device_tool_properties)(
                        self.handle(),
                        &mut count,
                        tool_properties.as_mut_ptr(),
                    )
                }
            } else {
                unsafe {
                    (fns.ext_tooling_info.get_physical_device_tool_properties_ext)(
                        self.handle(),
                        &mut count,
                        tool_properties.as_mut_ptr(),
                    )
                }
            };

            match result {
                vk::Result::INCOMPLETE => (),
                vk::Result::SUCCESS => {
                    unsafe { tool_properties.set_len(count as usize) };

                    return Ok(tool_properties
                        .iter()
                        .map(ToolProperties::from_vk)
                        .collect());
                }
                err => return Err(VulkanError::from(err)),
            }
        }
    }

    /// Returns whether queues of the given queue family support presentation to surfaces of
    /// windows of the given event loop, panicking on a validation error.
    ///
    /// On the X11 platform, this checks if the given queue family supports presentation to
    /// surfaces of windows created with the root visual. This means that if you create your
    /// window(s) with a different visual, the result of this function doesn't guarantee support
    /// for that window's surface, and you should use [`xcb_presentation_support`] or
    /// [`xlib_presentation_support`] directly to determine support for presentation to such
    /// surfaces.
    ///
    /// See also [`surface_support`] for determining if a queue family supports presentation to a
    /// specific surface.
    ///
    /// This is a shortcut for `try_presentation_support().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_presentation_support`] returns a [`ValidationError`].
    ///
    /// [`xcb_presentation_support`]: Self::xcb_presentation_support
    /// [`xlib_presentation_support`]: Self::xlib_presentation_support
    /// [`surface_support`]: Self::surface_support
    /// [`try_presentation_support`]: Self::try_presentation_support
    #[cfg(feature = "raw_window_handle")]
    #[track_caller]
    pub fn presentation_support(
        &self,
        queue_family_index: u32,
        event_loop: &impl HasDisplayHandle,
    ) -> bool {
        self.try_presentation_support(queue_family_index, event_loop)
            .unwrap()
    }

    /// Returns whether queues of the given queue family support presentation to surfaces of
    /// windows of the given event loop.
    ///
    /// On the X11 platform, this checks if the given queue family supports presentation to
    /// surfaces of windows created with the root visual. This means that if you create your
    /// window(s) with a different visual, the result of this function doesn't guarantee support
    /// for that window's surface, and you should use [`xcb_presentation_support`] or
    /// [`xlib_presentation_support`] directly to determine support for presentation to such
    /// surfaces.
    ///
    /// See also [`surface_support`] for determining if a queue family supports presentation to a
    /// specific surface.
    ///
    /// [`xcb_presentation_support`]: Self::xcb_presentation_support
    /// [`xlib_presentation_support`]: Self::xlib_presentation_support
    /// [`surface_support`]: Self::surface_support
    #[cfg(feature = "raw_window_handle")]
    pub fn try_presentation_support(
        &self,
        queue_family_index: u32,
        event_loop: &impl HasDisplayHandle,
    ) -> Result<bool, Validated<HandleError>> {
        let support = match event_loop
            .display_handle()
            .map_err(Validated::Error)?
            .as_raw()
        {
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap34.html#platformQuerySupport_android
            RawDisplayHandle::Android(_) => true,
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap34.html#platformQuerySupport_macos
            RawDisplayHandle::AppKit(_) => true,
            RawDisplayHandle::Wayland(display) => {
                let display = display.display.as_ptr();

                unsafe {
                    self.try_wayland_presentation_support(queue_family_index, display.cast())
                }?
            }
            RawDisplayHandle::Windows(_display) => {
                self.try_win32_presentation_support(queue_family_index)?
            }
            #[cfg(all(
                any(
                    target_os = "dragonfly",
                    target_os = "freebsd",
                    target_os = "hurd",
                    target_os = "illumos",
                    target_os = "linux",
                    target_os = "netbsd",
                    target_os = "openbsd",
                    target_os = "solaris"
                ),
                feature = "x11"
            ))]
            RawDisplayHandle::Xcb(display) => {
                let screen = display.screen;
                let connection = display.connection.unwrap().as_ptr();
                let visual_id = unsafe { get_xcb_root_visual_id(connection, screen) };

                unsafe {
                    self.try_xcb_presentation_support(
                        queue_family_index,
                        connection.cast(),
                        visual_id,
                    )
                }?
            }
            #[cfg(all(
                any(
                    target_os = "dragonfly",
                    target_os = "freebsd",
                    target_os = "hurd",
                    target_os = "illumos",
                    target_os = "linux",
                    target_os = "netbsd",
                    target_os = "openbsd",
                    target_os = "solaris"
                ),
                feature = "x11"
            ))]
            RawDisplayHandle::Xlib(display) => {
                let screen = display.screen;
                let display = display.display.unwrap().as_ptr();
                let visual_id = unsafe { get_xlib_root_visual_id(display, screen) };

                unsafe {
                    self.try_xlib_presentation_support(
                        queue_family_index,
                        display.cast(),
                        visual_id,
                    )
                }?
            }
            #[cfg(all(
                any(
                    target_os = "dragonfly",
                    target_os = "freebsd",
                    target_os = "hurd",
                    target_os = "illumos",
                    target_os = "linux",
                    target_os = "netbsd",
                    target_os = "openbsd",
                    target_os = "solaris"
                ),
                not(feature = "x11")
            ))]
            RawDisplayHandle::Xcb(_) | RawDisplayHandle::Xlib(_) => panic!("unsupported platform"),
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap34.html#platformQuerySupport_ios
            RawDisplayHandle::UiKit(_) => true,
            _ => unimplemented!(
                "the event loop was created with a windowing API that is not supported by \
                Vulkan/Vulkano",
            ),
        };

        Ok(support)
    }

    /// Queries whether the physical device supports presenting to Wayland surfaces from queues of
    /// the given queue family, panicking on a validation error.
    ///
    /// This is a shortcut for `try_wayland_presentation_support().unwrap()`.
    ///
    /// # Safety
    ///
    /// - `display` must be a valid Wayland `wl_display` handle.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_wayland_presentation_support`] returns a [`ValidationError`].
    ///
    /// [`try_wayland_presentation_support`]: Self::try_wayland_presentation_support
    #[track_caller]
    pub unsafe fn wayland_presentation_support(
        &self,
        queue_family_index: u32,
        display: *mut vk::wl_display,
    ) -> bool {
        unsafe { self.try_wayland_presentation_support(queue_family_index, display) }.unwrap()
    }

    /// Queries whether the physical device supports presenting to Wayland surfaces from queues of
    /// the given queue family.
    ///
    /// # Safety
    ///
    /// - `display` must be a valid Wayland `wl_display` handle.
    pub unsafe fn try_wayland_presentation_support(
        &self,
        queue_family_index: u32,
        display: *mut vk::wl_display,
    ) -> Result<bool, Box<ValidationError>> {
        self.validate_wayland_presentation_support(queue_family_index, display)?;

        Ok(unsafe { self.wayland_presentation_support_unchecked(queue_family_index, display) })
    }

    fn validate_wayland_presentation_support(
        &self,
        queue_family_index: u32,
        _display: *mut vk::wl_display,
    ) -> Result<(), Box<ValidationError>> {
        if !self.instance.enabled_extensions().khr_wayland_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "khr_wayland_surface",
                )])]),
                ..Default::default()
            }));
        }

        if queue_family_index >= self.queue_family_properties.len() as u32 {
            return Err(Box::new(ValidationError {
                context: "queue_family_index".into(),
                problem: "is not less than the number of queue families in the physical device"
                    .into(),
                vuids: &[
                    "VUID-vkGetPhysicalDeviceWaylandPresentationSupportKHR-queueFamilyIndex-01306",
                ],
                ..Default::default()
            }));
        }

        // VUID-vkGetPhysicalDeviceWaylandPresentationSupportKHR-display-parameter
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn wayland_presentation_support_unchecked(
        &self,
        queue_family_index: u32,
        display: *mut vk::wl_display,
    ) -> bool {
        let fns = self.instance.fns();
        let support = unsafe {
            (fns.khr_wayland_surface
                .get_physical_device_wayland_presentation_support_khr)(
                self.handle,
                queue_family_index,
                display,
            )
        };

        support != 0
    }

    /// Queries whether the physical device supports presenting to Win32 surfaces from queues of
    /// the given queue family, panicking on a validation error.
    ///
    /// This is a shortcut for `try_win32_presentation_support().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_win32_presentation_support`] returns a [`ValidationError`].
    ///
    /// [`try_win32_presentation_support`]: Self::try_win32_presentation_support
    #[inline]
    #[track_caller]
    pub fn win32_presentation_support(&self, queue_family_index: u32) -> bool {
        self.try_win32_presentation_support(queue_family_index)
            .unwrap()
    }

    /// Queries whether the physical device supports presenting to Win32 surfaces from queues of
    /// the given queue family.
    #[inline]
    pub fn try_win32_presentation_support(
        &self,
        queue_family_index: u32,
    ) -> Result<bool, Box<ValidationError>> {
        self.validate_win32_presentation_support(queue_family_index)?;

        Ok(unsafe { self.win32_presentation_support_unchecked(queue_family_index) })
    }

    fn validate_win32_presentation_support(
        &self,
        queue_family_index: u32,
    ) -> Result<(), Box<ValidationError>> {
        if !self.instance.enabled_extensions().khr_win32_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "khr_win32_surface",
                )])]),
                ..Default::default()
            }));
        }

        if queue_family_index >= self.queue_family_properties.len() as u32 {
            return Err(Box::new(ValidationError {
                context: "queue_family_index".into(),
                problem: "is not less than the number of queue families in the physical device"
                    .into(),
                vuids: &[
                    "VUID-vkGetPhysicalDeviceWin32PresentationSupportKHR-queueFamilyIndex-01309",
                ],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn win32_presentation_support_unchecked(&self, queue_family_index: u32) -> bool {
        let fns = self.instance.fns();
        let support = unsafe {
            (fns.khr_win32_surface
                .get_physical_device_win32_presentation_support_khr)(
                self.handle,
                queue_family_index,
            )
        };

        support != 0
    }

    /// Queries whether the physical device supports presenting to XCB surfaces from queues of the
    /// given queue family, panicking on a validation error.
    ///
    /// This is a shortcut for `try_xcb_presentation_support().unwrap()`.
    ///
    /// # Safety
    ///
    /// - `connection` must be a valid X11 `xcb_connection_t` handle.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_xcb_presentation_support`] returns a [`ValidationError`].
    ///
    /// [`try_xcb_presentation_support`]: Self::try_xcb_presentation_support
    #[track_caller]
    pub unsafe fn xcb_presentation_support(
        &self,
        queue_family_index: u32,
        connection: *mut vk::xcb_connection_t,
        visual_id: vk::xcb_visualid_t,
    ) -> bool {
        unsafe { self.try_xcb_presentation_support(queue_family_index, connection, visual_id) }
            .unwrap()
    }

    /// Queries whether the physical device supports presenting to XCB surfaces from queues of the
    /// given queue family.
    ///
    /// # Safety
    ///
    /// - `connection` must be a valid X11 `xcb_connection_t` handle.
    pub unsafe fn try_xcb_presentation_support(
        &self,
        queue_family_index: u32,
        connection: *mut vk::xcb_connection_t,
        visual_id: vk::xcb_visualid_t,
    ) -> Result<bool, Box<ValidationError>> {
        self.validate_xcb_presentation_support(queue_family_index, connection, visual_id)?;

        Ok(unsafe {
            self.xcb_presentation_support_unchecked(queue_family_index, connection, visual_id)
        })
    }

    fn validate_xcb_presentation_support(
        &self,
        queue_family_index: u32,
        _connection: *mut vk::xcb_connection_t,
        _visual_id: vk::xcb_visualid_t,
    ) -> Result<(), Box<ValidationError>> {
        if !self.instance.enabled_extensions().khr_xcb_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "khr_xcb_surface",
                )])]),
                ..Default::default()
            }));
        }

        if queue_family_index >= self.queue_family_properties.len() as u32 {
            return Err(Box::new(ValidationError {
                context: "queue_family_index".into(),
                problem: "is not less than the number of queue families in the physical device"
                    .into(),
                vuids: &[
                    "VUID-vkGetPhysicalDeviceXcbPresentationSupportKHR-queueFamilyIndex-01312",
                ],
                ..Default::default()
            }));
        }

        // VUID-vkGetPhysicalDeviceXcbPresentationSupportKHR-connection-parameter
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn xcb_presentation_support_unchecked(
        &self,
        queue_family_index: u32,
        connection: *mut vk::xcb_connection_t,
        visual_id: vk::xcb_visualid_t,
    ) -> bool {
        let fns = self.instance.fns();
        let support = unsafe {
            (fns.khr_xcb_surface
                .get_physical_device_xcb_presentation_support_khr)(
                self.handle,
                queue_family_index,
                connection,
                visual_id,
            )
        };

        support != 0
    }

    /// Queries whether the physical device supports presenting to Xlib surfaces from queues of the
    /// given queue family, panicking on a validation error.
    ///
    /// This is a shortcut for `try_xlib_presentation_support().unwrap()`.
    ///
    /// # Safety
    ///
    /// - `display` must be a valid Xlib `Display` handle.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_xlib_presentation_support`] returns a [`ValidationError`].
    ///
    /// [`try_xlib_presentation_support`]: Self::try_xlib_presentation_support
    #[track_caller]
    pub unsafe fn xlib_presentation_support(
        &self,
        queue_family_index: u32,
        display: *mut vk::Display,
        visual_id: vk::VisualID,
    ) -> bool {
        unsafe { self.try_xlib_presentation_support(queue_family_index, display, visual_id) }
            .unwrap()
    }

    /// Queries whether the physical device supports presenting to Xlib surfaces from queues of the
    /// given queue family.
    ///
    /// # Safety
    ///
    /// - `display` must be a valid Xlib `Display` handle.
    pub unsafe fn try_xlib_presentation_support(
        &self,
        queue_family_index: u32,
        display: *mut vk::Display,
        visual_id: vk::VisualID,
    ) -> Result<bool, Box<ValidationError>> {
        self.validate_xlib_presentation_support(queue_family_index, display, visual_id)?;

        Ok(unsafe {
            self.xlib_presentation_support_unchecked(queue_family_index, display, visual_id)
        })
    }

    fn validate_xlib_presentation_support(
        &self,
        queue_family_index: u32,
        _display: *mut vk::Display,
        _visual_id: vk::VisualID,
    ) -> Result<(), Box<ValidationError>> {
        if !self.instance.enabled_extensions().khr_xlib_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "khr_xlib_surface",
                )])]),
                ..Default::default()
            }));
        }

        if queue_family_index >= self.queue_family_properties.len() as u32 {
            return Err(Box::new(ValidationError {
                context: "queue_family_index".into(),
                problem: "is not less than the number of queue families in the physical device"
                    .into(),
                vuids: &[
                    "VUID-vkGetPhysicalDeviceXlibPresentationSupportKHR-queueFamilyIndex-01315",
                ],
                ..Default::default()
            }));
        }

        // VUID-vkGetPhysicalDeviceXlibPresentationSupportKHR-dpy-parameter
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn xlib_presentation_support_unchecked(
        &self,
        queue_family_index: u32,
        display: *mut vk::Display,
        visual_id: vk::VisualID,
    ) -> bool {
        let fns = self.instance.fns();
        let support = unsafe {
            (fns.khr_xlib_surface
                .get_physical_device_xlib_presentation_support_khr)(
                self.handle,
                queue_family_index,
                display,
                visual_id,
            )
        };

        support != 0
    }

    /// Queries whether the physical device supports presenting to DirectFB surfaces from queues of
    /// the given queue family, panicking on a validation error.
    ///
    /// This is a shortcut for `try_directfb_presentation_support().unwrap()`.
    ///
    /// # Safety
    ///
    /// - `dfb` must be a valid DirectFB `IDirectFB` handle.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_directfb_presentation_support`] returns a [`ValidationError`].
    ///
    /// [`try_directfb_presentation_support`]: Self::try_directfb_presentation_support
    #[inline]
    #[track_caller]
    pub unsafe fn directfb_presentation_support(
        &self,
        queue_family_index: u32,
        dfb: *mut vk::IDirectFB,
    ) -> bool {
        unsafe { self.try_directfb_presentation_support(queue_family_index, dfb) }.unwrap()
    }

    /// Queries whether the physical device supports presenting to DirectFB surfaces from queues of
    /// the given queue family.
    ///
    /// # Safety
    ///
    /// - `dfb` must be a valid DirectFB `IDirectFB` handle.
    #[inline]
    pub unsafe fn try_directfb_presentation_support(
        &self,
        queue_family_index: u32,
        dfb: *mut vk::IDirectFB,
    ) -> Result<bool, Box<ValidationError>> {
        self.validate_directfb_presentation_support(queue_family_index, dfb)?;

        Ok(unsafe { self.directfb_presentation_support_unchecked(queue_family_index, dfb) })
    }

    fn validate_directfb_presentation_support(
        &self,
        queue_family_index: u32,
        _dfb: *mut vk::IDirectFB,
    ) -> Result<(), Box<ValidationError>> {
        if !self.instance.enabled_extensions().ext_directfb_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "ext_directfb_surface",
                )])]),
                ..Default::default()
            }));
        }

        if queue_family_index >= self.queue_family_properties.len() as u32 {
            return Err(Box::new(ValidationError {
                context: "queue_family_index".into(),
                problem: "is not less than the number of queue families in the physical device"
                    .into(),
                vuids: &[
                    "VUID-vkGetPhysicalDeviceDirectFBPresentationSupportEXT-queueFamilyIndex-04119",
                ],
                ..Default::default()
            }));
        }

        // VUID-vkGetPhysicalDeviceDirectFBPresentationSupportEXT-dfb-parameter
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn directfb_presentation_support_unchecked(
        &self,
        queue_family_index: u32,
        dfb: *mut vk::IDirectFB,
    ) -> bool {
        let fns = self.instance.fns();
        let support = unsafe {
            (fns.ext_directfb_surface
                .get_physical_device_direct_fb_presentation_support_ext)(
                self.handle,
                queue_family_index,
                dfb,
            )
        };

        support != 0
    }

    /// Queries whether the physical device supports presenting to QNX Screen surfaces from queues
    /// of the given queue family, panicking on a validation error.
    ///
    /// This is a shortcut for `try_qnx_screen_presentation_support().unwrap()`.
    ///
    /// # Safety
    ///
    /// - `window` must be a valid QNX Screen `_screen_window` handle.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_qnx_screen_presentation_support`] returns a [`ValidationError`].
    ///
    /// [`try_qnx_screen_presentation_support`]: Self::try_qnx_screen_presentation_support
    #[track_caller]
    pub unsafe fn qnx_screen_presentation_support(
        &self,
        queue_family_index: u32,
        window: *mut vk::_screen_window,
    ) -> bool {
        unsafe { self.try_qnx_screen_presentation_support(queue_family_index, window) }.unwrap()
    }

    /// Queries whether the physical device supports presenting to QNX Screen surfaces from queues
    /// of the given queue family.
    ///
    /// # Safety
    ///
    /// - `window` must be a valid QNX Screen `_screen_window` handle.
    pub unsafe fn try_qnx_screen_presentation_support(
        &self,
        queue_family_index: u32,
        window: *mut vk::_screen_window,
    ) -> Result<bool, Box<ValidationError>> {
        self.validate_qnx_screen_presentation_support(queue_family_index, window)?;

        Ok(unsafe { self.qnx_screen_presentation_support_unchecked(queue_family_index, window) })
    }

    fn validate_qnx_screen_presentation_support(
        &self,
        queue_family_index: u32,
        _window: *mut vk::_screen_window,
    ) -> Result<(), Box<ValidationError>> {
        if !self.instance.enabled_extensions().qnx_screen_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "qnx_screen_surface",
                )])]),
                ..Default::default()
            }));
        }

        if queue_family_index >= self.queue_family_properties.len() as u32 {
            return Err(Box::new(ValidationError {
                context: "queue_family_index".into(),
                problem: "is not less than the number of queue families in the physical device"
                    .into(),
                vuids: &[
                    "VUID-vkGetPhysicalDeviceScreenPresentationSupportQNX-queueFamilyIndex-04743",
                ],
                ..Default::default()
            }));
        }

        // VUID-vkGetPhysicalDeviceScreenPresentationSupportQNX-window-parameter
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn qnx_screen_presentation_support_unchecked(
        &self,
        queue_family_index: u32,
        window: *mut vk::_screen_window,
    ) -> bool {
        let fns = self.instance.fns();
        let support = unsafe {
            (fns.qnx_screen_surface
                .get_physical_device_screen_presentation_support_qnx)(
                self.handle,
                queue_family_index,
                window,
            )
        };

        support != 0
    }
}

impl Debug for PhysicalDevice {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        let Self {
            handle,
            instance,
            id,

            api_version,
            supported_extensions,
            supported_features,
            properties,
            extension_properties,
            memory_properties,
            queue_family_properties,

            display_properties: _,
            display_plane_properties: _,
            external_buffer_properties: _,
            external_fence_properties: _,
            external_semaphore_properties: _,
            format_properties: _,
            image_format_properties: _,
            sparse_image_format_properties: _,
        } = self;

        f.debug_struct("PhysicalDevice")
            .field("handle", handle)
            .field("instance", instance)
            .field("id", id)
            .field("api_version", api_version)
            .field("supported_extensions", supported_extensions)
            .field("supported_features", supported_features)
            .field("properties", properties)
            .field("extension_properties", extension_properties)
            .field("memory_properties", memory_properties)
            .field("queue_family_properties", queue_family_properties)
            .finish_non_exhaustive()
    }
}

unsafe impl VulkanObject for PhysicalDevice {
    type Handle = vk::PhysicalDevice;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl InstanceOwned for PhysicalDevice {
    #[inline]
    fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }
}

impl_id_counter!(PhysicalDevice);

#[cfg(all(
    feature = "raw_window_handle",
    feature = "x11",
    any(
        target_os = "dragonfly",
        target_os = "freebsd",
        target_os = "hurd",
        target_os = "illumos",
        target_os = "linux",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "solaris"
    )
))]
unsafe fn get_xcb_root_visual_id(
    connection: *mut std::ffi::c_void,
    screen_id: std::ffi::c_int,
) -> u32 {
    use x11rb::connection::Connection;

    let connection =
        unsafe { x11rb::xcb_ffi::XCBConnection::from_raw_xcb_connection(connection, false) }
            .unwrap();
    let screen = &connection.setup().roots[screen_id as usize];

    screen.root_visual
}

#[cfg(all(
    feature = "raw_window_handle",
    feature = "x11",
    any(
        target_os = "dragonfly",
        target_os = "freebsd",
        target_os = "hurd",
        target_os = "illumos",
        target_os = "linux",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "solaris"
    )
))]
unsafe fn get_xlib_root_visual_id(
    display: *mut std::ffi::c_void,
    screen_id: std::ffi::c_int,
) -> u32 {
    let xlib_xcb = x11_dl::xlib_xcb::Xlib_xcb::open().unwrap();
    let connection = unsafe { (xlib_xcb.XGetXCBConnection)(display.cast()) };

    unsafe { get_xcb_root_visual_id(connection, screen_id) }
}

/// Properties of a group of physical devices that can be used to create a single logical device.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct PhysicalDeviceGroupProperties {
    /// The physical devices that make up the group.
    ///
    /// A physical device can belong to at most one group, but a group can consist of a single
    /// physical device.
    pub physical_devices: Vec<Arc<PhysicalDevice>>,

    /// Whether memory can be allocated from a subset of the devices in the group,
    /// rather than only from the group as a whole.
    ///
    /// If the length of `physical_devices` is 1, then this is always `false`.
    pub subset_allocation: bool,
}

#[repr(C)]
pub(crate) struct PhysicalDeviceGroupPropertiesRaw {
    pub(crate) physical_device_count: u32,
    pub(crate) physical_devices: [vk::PhysicalDevice; vk::MAX_DEVICE_GROUP_SIZE],
    pub(crate) subset_allocation: vk::Bool32,
}

vulkan_enum! {
    #[non_exhaustive]

    /// Type of a physical device.
    PhysicalDeviceType = PhysicalDeviceType(i32);

    /// The device is an integrated GPU.
    IntegratedGpu = INTEGRATED_GPU,

    /// The device is a discrete GPU.
    DiscreteGpu = DISCRETE_GPU,

    /// The device is a virtual GPU.
    VirtualGpu = VIRTUAL_GPU,

    /// The device is a CPU.
    Cpu = CPU,

    /// The device is something else.
    Other = OTHER,
}

impl Default for PhysicalDeviceType {
    #[inline]
    fn default() -> Self {
        PhysicalDeviceType::Other
    }
}

/// The version of the Vulkan conformance test that a driver is conformant against.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ConformanceVersion {
    pub major: u8,
    pub minor: u8,
    pub subminor: u8,
    pub patch: u8,
}

impl From<vk::ConformanceVersion> for ConformanceVersion {
    #[inline]
    fn from(val: vk::ConformanceVersion) -> Self {
        ConformanceVersion {
            major: val.major,
            minor: val.minor,
            subminor: val.subminor,
            patch: val.patch,
        }
    }
}

impl Debug for ConformanceVersion {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl std::fmt::Display for ConformanceVersion {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        Debug::fmt(self, f)
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// An identifier for the driver of a physical device.
    DriverId = DriverId(i32);

    // TODO: document
    AMDProprietary = AMD_PROPRIETARY,

    // TODO: document
    AMDOpenSource = AMD_OPEN_SOURCE,

    // TODO: document
    MesaRADV = MESA_RADV,

    // TODO: document
    NvidiaProprietary = NVIDIA_PROPRIETARY,

    // TODO: document
    IntelProprietaryWindows = INTEL_PROPRIETARY_WINDOWS,

    // TODO: document
    IntelOpenSourceMesa = INTEL_OPEN_SOURCE_MESA,

    // TODO: document
    ImaginationProprietary = IMAGINATION_PROPRIETARY,

    // TODO: document
    QualcommProprietary = QUALCOMM_PROPRIETARY,

    // TODO: document
    ARMProprietary = ARM_PROPRIETARY,

    // TODO: document
    GoogleSwiftshader = GOOGLE_SWIFTSHADER,

    // TODO: document
    GGPProprietary = GGP_PROPRIETARY,

    // TODO: document
    BroadcomProprietary = BROADCOM_PROPRIETARY,

    // TODO: document
    MesaLLVMpipe = MESA_LLVMPIPE,

    // TODO: document
    MoltenVK = MOLTENVK,

    // TODO: document
    CoreAVIProprietary = COREAVI_PROPRIETARY,

    // TODO: document
    JuiceProprietary = JUICE_PROPRIETARY,

    // TODO: document
    VeriSiliconProprietary = VERISILICON_PROPRIETARY,

    // TODO: document
    MesaTurnip = MESA_TURNIP,

    // TODO: document
    MesaV3DV = MESA_V3DV,

    // TODO: document
    MesaPanVK = MESA_PANVK,

    // TODO: document
    SamsungProprietary = SAMSUNG_PROPRIETARY,

    // TODO: document
    MesaVenus = MESA_VENUS,

    // TODO: document
    MesaDozen = MESA_DOZEN,
}

/// Information provided about an active tool.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ToolProperties {
    /// The name of the tool.
    pub name: String,

    /// The version of the tool.
    pub version: String,

    /// The purposes supported by the tool.
    pub purposes: ToolPurposes,

    /// A description of the tool.
    pub description: String,

    /// The layer implementing the tool, or empty if it is not implemented by a layer.
    pub layer: String,
}

impl ToolProperties {
    pub(crate) fn from_vk(val_vk: &vk::PhysicalDeviceToolProperties<'_>) -> Self {
        let &vk::PhysicalDeviceToolProperties { purposes, .. } = val_vk;

        Self {
            name: unsafe { crate::c_str_to_string_unchecked(val_vk.name_as_c_str().unwrap()) },
            version: unsafe {
                crate::c_str_to_string_unchecked(val_vk.version_as_c_str().unwrap())
            },
            purposes: purposes.into(),
            description: unsafe {
                crate::c_str_to_string_unchecked(val_vk.description_as_c_str().unwrap())
            },
            layer: unsafe { crate::c_str_to_string_unchecked(val_vk.layer_as_c_str().unwrap()) },
        }
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// The purpose of an active tool.
    ToolPurposes = ToolPurposeFlags(u32);

    /// The tool provides validation of API usage.
    VALIDATION = VALIDATION,

    /// The tool provides profiling of API usage.
    PROFILING = PROFILING,

    /// The tool is capturing data about the application's API usage.
    TRACING = TRACING,

    /// The tool provides additional API features or extensions on top of the underlying
    /// implementation.
    ADDITIONAL_FEATURES = ADDITIONAL_FEATURES,

    /// The tool modifies the API features, limits or extensions presented to the application.
    MODIFYING_FEATURES = MODIFYING_FEATURES,

    /// The tool reports information to the user via a
    /// [`DebugUtilsMessenger`](crate::instance::debug::DebugUtilsMessenger).
    DEBUG_REPORTING = DEBUG_REPORTING_EXT
    RequiresOneOf([
        RequiresAllOf([InstanceExtension(ext_debug_utils)]),
        RequiresAllOf([InstanceExtension(ext_debug_report)]),
    ]),

    /// The tool consumes debug markers or object debug annotation, queue labels or command buffer
    /// labels.
    DEBUG_MARKERS = DEBUG_MARKERS_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_debug_marker)]),
        RequiresAllOf([InstanceExtension(ext_debug_utils)]),
    ]),
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Specifies which subgroup operations are supported.
    SubgroupFeatures = SubgroupFeatureFlags(u32);

    // TODO: document
    BASIC = BASIC,

    // TODO: document
    VOTE = VOTE,

    // TODO: document
    ARITHMETIC = ARITHMETIC,

    // TODO: document
    BALLOT = BALLOT,

    // TODO: document
    SHUFFLE = SHUFFLE,

    // TODO: document
    SHUFFLE_RELATIVE = SHUFFLE_RELATIVE,

    // TODO: document
    CLUSTERED = CLUSTERED,

    // TODO: document
    QUAD = QUAD,

    // TODO: document
    PARTITIONED = PARTITIONED_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_shader_subgroup_partitioned)]),
    ]),
}

vulkan_enum! {
    #[non_exhaustive]

    /// Specifies how the device clips single point primitives.
    PointClippingBehavior = PointClippingBehavior(i32);

    /// Points are clipped if they lie outside any clip plane, both those bounding the view volume
    /// and user-defined clip planes.
    AllClipPlanes = ALL_CLIP_PLANES,

    /// Points are clipped only if they lie outside a user-defined clip plane.
    UserClipPlanesOnly = USER_CLIP_PLANES_ONLY,
}

vulkan_enum! {
    #[non_exhaustive]

    /// Specifies whether, and how, shader float controls can be set independently.
    ShaderFloatControlsIndependence = ShaderFloatControlsIndependence(i32);

    // TODO: document
    Float32Only = TYPE_32_ONLY,

    // TODO: document
    All = ALL,

    // TODO: document
    None = NONE,
}

/// Specifies shader core properties.
#[derive(Clone, Copy, Debug)]
pub struct ShaderCoreProperties {}

impl From<vk::ShaderCorePropertiesFlagsAMD> for ShaderCoreProperties {
    #[inline]
    fn from(_val: vk::ShaderCorePropertiesFlagsAMD) -> Self {
        Self {}
    }
}

vulkan_enum! {
    #[non_exhaustive]

    LayeredDriverUnderlyingApi = LayeredDriverUnderlyingApiMSFT(i32);

    // TODO: document
    None = NONE,

    // TODO: document
    D3D12 = D3D12,
}

vulkan_bitflags! {
    #[non_exhaustive]

    PhysicalDeviceSchedulingControlsFlags = PhysicalDeviceSchedulingControlsFlagsARM(u64);

    // TODO: document
    SHADER_CORE_COUNT = SHADER_CORE_COUNT,
}

vulkan_bitflags! {
    #[non_exhaustive]

    // TODO: document
    MemoryDecompressionMethods = MemoryDecompressionMethodFlagsNV(u64);

    // TODO: document
    GDEFLATE_1_0 = GDEFLATE_1_0,
}

vulkan_bitflags! {
    #[non_exhaustive]

    // TODO: document
    OpticalFlowGridSizes = OpticalFlowGridSizeFlagsNV(u32);

    // TODO: document
    SIZE_1X1 = TYPE_1X1,

    // TODO: document
    SIZE_2X2 = TYPE_2X2,

    // TODO: document
    SIZE_4X4 = TYPE_4X4,

    // TODO: document
    SIZE_8X8 = TYPE_8X8,
}

vulkan_enum! {
    #[non_exhaustive]

    // TODO: document
    PipelineRobustnessBufferBehavior = PipelineRobustnessBufferBehaviorEXT(i32);

    // TODO: document
    DeviceDefault = DEVICE_DEFAULT,

    // TODO: document
    Disabled = DISABLED,

    // TODO: document
    RobustBufferAccess = ROBUST_BUFFER_ACCESS,

    // TODO: document
    RobustBufferAccess2 = ROBUST_BUFFER_ACCESS_2,
}

vulkan_enum! {
    #[non_exhaustive]

    // TODO: document
    PipelineRobustnessImageBehavior = PipelineRobustnessImageBehaviorEXT(i32);

    // TODO: document
    DeviceDefault = DEVICE_DEFAULT,

    // TODO: document
    Disabled = DISABLED,

    // TODO: document
    RobustImageAccess = ROBUST_IMAGE_ACCESS,

    // TODO: document
    RobustImageAccess2 = ROBUST_IMAGE_ACCESS_2,
}

vulkan_enum! {
    #[non_exhaustive]

    // TODO: document
    RayTracingInvocationReorderMode = RayTracingInvocationReorderModeNV(i32);

    // TODO: document
    None = NONE,

    // TODO: document
    Reorder = REORDER,
}
