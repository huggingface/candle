//! Link between Vulkan and a window and/or the screen.
//!
//! Before you can draw on the screen or a window, you have to create two objects:
//!
//! - Create a `Surface` object that represents the location where the image will show up (either a
//!   window or a monitor).
//! - Create a `Swapchain` that uses that `Surface`.
//!
//! Creating a surface can be done with only an `Instance` object. However creating a swapchain
//! requires a `Device` object.
//!
//! Once you have a swapchain, you can retrieve `Image` objects from it and draw to them just like
//! you would draw on any other image.
//!
//! # Surfaces
//!
//! A surface is an object that represents a location where to render. It can be created from an
//! instance and either a window handle (in a platform-specific way) or a monitor.
//!
//! In order to use surfaces, you will have to enable the `VK_KHR_surface` extension on the
//! instance. See the `instance` module for more information about how to enable extensions.
//!
//! ## Creating a surface from a window
//!
//! There are 5 extensions that each allow you to create a surface from a type of window:
//!
//! - `VK_KHR_xlib_surface`
//! - `VK_KHR_xcb_surface`
//! - `VK_KHR_wayland_surface`
//! - `VK_KHR_android_surface`
//! - `VK_KHR_win32_surface`
//!
//! For example if you want to create a surface from an Android surface, you will have to enable
//! the `VK_KHR_android_surface` extension and use `Surface::from_android`.
//! See the documentation of `Surface` for all the possible constructors.
//!
//! Trying to use one of these functions without enabling the proper extension will result in an
//! error.
//!
//! **Note that the `Surface` object is potentially unsafe**. It is your responsibility to
//! keep the window alive for at least as long as the surface exists. In many cases Surface
//! may be able to do this for you, if you pass it ownership of your Window (or a
//! reference-counting container for it).
//!
//! ### Examples
//!
//! ```no_run
//! use std::ptr;
//! use vulkano::{
//!     instance::{Instance, InstanceCreateInfo, InstanceExtensions},
//!     swapchain::Surface,
//!     Version, VulkanLibrary,
//! };
//!
//! let instance = {
//!     let library = unsafe { VulkanLibrary::new() }
//!         .unwrap_or_else(|err| panic!("couldn't load Vulkan library: {:?}", err));
//!
//!     let extensions = InstanceExtensions {
//!         khr_surface: true,
//!         khr_win32_surface: true, // If you don't enable this, `from_win32` will fail.
//!         ..InstanceExtensions::empty()
//!     };
//!
//!     Instance::new(
//!         &library,
//!         &InstanceCreateInfo {
//!             enabled_extensions: &extensions,
//!             ..Default::default()
//!         },
//!     )
//!     .unwrap_or_else(|err| panic!("couldn't create instance: {:?}", err))
//! };
//!
//! # use ash::vk;
//! # use std::sync::Arc;
//! #
//! # struct Window(*const u32);
//! #
//! # impl Window {
//! #     fn hwnd(&self) -> *const u32 {
//! #         self.0
//! #     }
//! # }
//! #
//! # unsafe impl Send for Window {}
//! # unsafe impl Sync for Window {}
//! #
//! # fn build_window() -> Arc<Window> {
//! #     Arc::new(Window(ptr::null()))
//! # }
//! #
//! let window = build_window(); // Third-party function, not provided by vulkano
//! let _surface = {
//!     let hinstance: vk::HINSTANCE = 0; // Windows-specific object
//!     unsafe {
//!         Surface::from_win32(
//!             &instance,
//!             hinstance,
//!             window.hwnd() as vk::HWND,
//!             Some(window),
//!         )
//!     }
//!     .unwrap()
//! };
//! ```
//!
//! # Swapchains
//!
//! A surface represents a location on the screen and can be created from an instance. Once you
//! have a surface, the next step is to create a swapchain. Creating a swapchain requires a device,
//! and allocates the resources that will be used to display images on the screen.
//!
//! A swapchain is composed of one or multiple images. Each image of the swapchain is presented in
//! turn on the screen, one after another. More information below.
//!
//! Swapchains have several properties:
//!
//!  - The number of images that will cycle on the screen.
//!  - The format of the images.
//!  - The 2D dimensions of the images, plus a number of layers, for a total of three dimensions.
//!  - The usage of the images, similar to creating other images.
//!  - The queue families that are going to use the images, similar to creating other images.
//!  - An additional transformation (rotation or mirroring) to perform on the final output.
//!  - How the alpha of the final output will be interpreted.
//!  - How to perform the cycling between images in regard to vsync.
//!
//! You can query the supported values of all these properties from the physical device.
//!
//! ## Creating a swapchain
//!
//! In order to create a swapchain, you will first have to enable the `VK_KHR_swapchain` extension
//! on the device (and not on the instance like `VK_KHR_surface`):
//!
//! ```no_run
//! # use vulkano::device::DeviceExtensions;
//! #
//! let ext = DeviceExtensions {
//!     khr_swapchain: true,
//!     ..DeviceExtensions::empty()
//! };
//! ```
//!
//! Then, query the capabilities of the surface with
//! [`PhysicalDevice::surface_capabilities`](crate::device::physical::PhysicalDevice::surface_capabilities)
//! and
//! [`PhysicalDevice::surface_formats`](crate::device::physical::PhysicalDevice::surface_formats)
//! and choose which values you are going to use.
//!
//! ```no_run
//! # use std::{cmp::{max, min}, error::Error, sync::Arc};
//! # use vulkano::{device::Device, swapchain::Surface};
//! #
//! # fn choose_caps(device: Arc<Device>, surface: Arc<Surface>) -> Result<(), Box<dyn Error>> {
//! let surface_capabilities = device
//!     .physical_device()
//!     .surface_capabilities(&surface, &Default::default())?;
//!
//! // Use the current window size or some fixed resolution.
//! let image_extent = surface_capabilities.current_extent.unwrap_or([640, 480]);
//!
//! // Try to use double-buffering.
//! let min_image_count = match surface_capabilities.max_image_count {
//!     None => max(2, surface_capabilities.min_image_count),
//!     Some(limit) => min(max(2, surface_capabilities.min_image_count), limit)
//! };
//!
//! // Preserve the current surface transform.
//! let pre_transform = surface_capabilities.current_transform;
//!
//! // Use the first available format.
//! let (image_format, color_space) = device
//!     .physical_device()
//!     .surface_formats(&surface, &Default::default())?[0];
//! #
//! # Ok(())
//! # }
//! ```
//!
//! Then, call [`Swapchain::new`].
//!
//! ```no_run
//! # use std::{error::Error, sync::Arc};
//! # use vulkano::{
//! #     device::Device,
//! #     format::Format,
//! #     image::ImageUsage,
//! #     swapchain::{
//! #         CompositeAlpha, ColorSpace, FullScreenExclusive, PresentMode, Surface,
//! #         SurfaceTransform, Swapchain, SwapchainCreateInfo,
//! #     },
//! # };
//! #
//! # fn create_swapchain(
//! #     device: Arc<Device>,
//! #     surface: Arc<Surface>,
//! #     min_image_count: u32,
//! #     image_format: Format,
//! #     image_extent: [u32; 2],
//! #     pre_transform: SurfaceTransform,
//! #     composite_alpha: CompositeAlpha,
//! #     present_mode: PresentMode,
//! #     full_screen_exclusive: FullScreenExclusive,
//! # ) -> Result<(), Box<dyn Error>> {
//! // Create the swapchain and its images.
//! let (swapchain, images) = Swapchain::new(
//!     // Create the swapchain in this `device`'s memory.
//!     &device,
//!     // The surface where the images will be presented.
//!     &surface,
//!     // The creation parameters.
//!     &SwapchainCreateInfo {
//!         // How many images to use in the swapchain.
//!         min_image_count,
//!         // The format of the images.
//!         image_format,
//!         // The size of each image.
//!         image_extent,
//!         // The created swapchain images will be used as a color attachment for rendering.
//!         image_usage: ImageUsage::COLOR_ATTACHMENT,
//!         // What transformation to use with the surface.
//!         pre_transform,
//!         // How to handle the alpha channel.
//!         composite_alpha,
//!         // How to present images.
//!         present_mode,
//!         // How to handle full-screen exclusivity
//!         full_screen_exclusive,
//!         ..Default::default()
//!     },
//! )?;
//! #
//! # Ok(())
//! # }
//! ```
//!
//! Creating a swapchain not only returns the swapchain object, but also all the images that belong
//! to it.
//!
//! ## Acquiring and presenting images
//!
//! Once you created a swapchain and retrieved all the images that belong to it (see previous
//! section), you can draw on it. This is done in three steps:
//!
//!  - Call `swapchain::acquire_next_image`. This function will return the index of the image
//!    (within the list returned by `Swapchain::new`) that is available to draw, plus a future
//!    representing the moment when the GPU will gain access to that image.
//!  - Draw on that image just like you would draw to any other image (see the documentation of the
//!    `pipeline` module). You need to chain the draw after the future that was returned by
//!    `acquire_next_image`.
//!  - Call `Swapchain::present` with the same index and by chaining the futures, in order to tell
//!    the implementation that you are finished drawing to the image and that it can queue a
//!    command to present the image on the screen after the draw operations are finished.
//!
//! ```
//! use vulkano::swapchain::{self, SwapchainPresentInfo};
//! use vulkano::sync::GpuFuture;
//!
//! # let queue: ::std::sync::Arc<::vulkano::device::Queue> = return;
//! # let mut swapchain: ::std::sync::Arc<swapchain::Swapchain> = return;
//! #
//! // let mut (swapchain, images) = Swapchain::new(...);
//! loop {
//!     # let mut command_buffer: ::std::sync::Arc<::vulkano::command_buffer::PrimaryAutoCommandBuffer> = return;
//!     let (image_index, suboptimal, acquire_future)
//!         = swapchain::acquire_next_image(swapchain.clone(), None).unwrap();
//!
//!     // The command_buffer contains the draw commands that modify the framebuffer
//!     // constructed from images[image_index]
//!     acquire_future
//!         .then_execute(queue.clone(), command_buffer)
//!         .unwrap()
//!         .then_swapchain_present(
//!             queue.clone(),
//!             SwapchainPresentInfo::new(swapchain.clone(), image_index),
//!         )
//!         .then_signal_fence_and_flush()
//!         .unwrap();
//! }
//! ```
//!
//! ## Recreating a swapchain
//!
//! In some situations, the swapchain will become invalid by itself. This includes for example when
//! the window is resized (as the images of the swapchain will no longer match the window's) or,
//! on Android, when the application went to the background and goes back to the foreground.
//!
//! In this situation, acquiring a swapchain image or presenting it will return an error. Rendering
//! to an image of that swapchain will not produce any error, but may or may not work. To continue
//! rendering, you will need to *recreate* the swapchain by creating a new swapchain and passing
//! as last parameter the old swapchain.
//!
//! ```
//! use vulkano::{
//!     swapchain::{self, SwapchainCreateInfo, SwapchainPresentInfo},
//!     sync::GpuFuture,
//!     Validated, VulkanError,
//! };
//!
//! // let (swapchain, images) = Swapchain::new(...);
//! # let mut swapchain: ::std::sync::Arc<::vulkano::swapchain::Swapchain> = return;
//! # let mut images: Vec<::std::sync::Arc<::vulkano::image::Image>> = return;
//! # let queue: ::std::sync::Arc<::vulkano::device::Queue> = return;
//! let mut recreate_swapchain = false;
//!
//! loop {
//!     if recreate_swapchain {
//!         let (new_swapchain, new_images) = swapchain
//!             .recreate(&SwapchainCreateInfo {
//!                 image_extent: [1024, 768],
//!                 ..swapchain.create_info()
//!             })
//!             .unwrap();
//!         swapchain = new_swapchain;
//!         images = new_images;
//!         recreate_swapchain = false;
//!     }
//!
//!     let (image_index, suboptimal, acq_future) =
//!         match swapchain::acquire_next_image(swapchain.clone(), None).map_err(Validated::unwrap)
//!         {
//!             Ok(r) => r,
//!             Err(VulkanError::OutOfDate) => {
//!                 recreate_swapchain = true;
//!                 continue;
//!             }
//!             Err(err) => panic!("{:?}", err),
//!         };
//!
//!     // ...
//!
//!     let final_future = acq_future
//!         // .then_execute(...)
//!         .then_swapchain_present(
//!             queue.clone(),
//!             SwapchainPresentInfo::new(swapchain.clone(), image_index),
//!         )
//!         .then_signal_fence_and_flush()
//!         .unwrap(); // TODO: PresentError?
//!
//!     if suboptimal {
//!         recreate_swapchain = true;
//!     }
//! }
//! ```

pub use self::{acquire_present::*, surface::*};
use crate::{
    device::{Device, DeviceOwned},
    format::Format,
    image::{Image, ImageCreateFlags, ImageFormatInfo, ImageTiling, ImageType, ImageUsage},
    instance::InstanceOwnedDebugWrapper,
    macros::{impl_id_counter, vulkan_bitflags, vulkan_bitflags_enum, vulkan_enum},
    sync::{OwnedSharing, Sharing},
    NonExhaustive, Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, Version,
    VulkanError, VulkanObject,
};
use ash::vk;
use ash::vk::TaggedStructure;

use parking_lot::Mutex;
use smallvec::SmallVec;
use std::{
    fmt::Debug,
    mem::MaybeUninit,
    num::NonZero,
    ptr,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

mod acquire_present;
mod surface;

/// Contains the swapping system and the images that can be shown on a surface.
#[derive(Debug)]
pub struct Swapchain {
    handle: vk::SwapchainKHR,
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    surface: InstanceOwnedDebugWrapper<Arc<Surface>>,
    id: NonZero<u64>,

    flags: SwapchainCreateFlags,
    min_image_count: u32,
    image_format: Format,
    image_view_formats: Vec<Format>,
    image_color_space: ColorSpace,
    image_extent: [u32; 2],
    image_array_layers: u32,
    image_usage: ImageUsage,
    image_sharing: OwnedSharing,
    pre_transform: SurfaceTransform,
    composite_alpha: CompositeAlpha,
    present_mode: PresentMode,
    present_modes: SmallVec<[PresentMode; PresentMode::COUNT]>,
    clipped: bool,
    scaling_behavior: Option<PresentScaling>,
    present_gravity: Option<[PresentGravity; 2]>,
    full_screen_exclusive: FullScreenExclusive,
    win32_monitor: Option<Win32Monitor>,

    prev_present_id: AtomicU64,

    // Whether full-screen exclusive is currently held.
    full_screen_exclusive_held: AtomicBool,

    // The images of this swapchain.
    images: Vec<ImageEntry>,

    // If true, that means we have tried to use this swapchain to recreate a new swapchain. The
    // current swapchain can no longer be used for anything except presenting already-acquired
    // images.
    //
    // We use a `Mutex` instead of an `AtomicBool` because we want to keep that locked while
    // we acquire the image.
    is_retired: Mutex<bool>,
}

#[derive(Debug)]
struct ImageEntry {
    handle: vk::Image,
    layout_initialized: AtomicBool,
}

impl Swapchain {
    /// Creates a new `Swapchain`, panicking on a validation error.
    ///
    /// This function returns the swapchain plus a list of the images that belong to the
    /// swapchain. The order in which the images are returned is important for the
    /// `acquire_next_image` and `present` functions.
    ///
    /// This is a shortcut for `try_new().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_new`] returns a [`ValidationError`].
    /// - Panics if the device and the surface don't belong to the same instance.
    /// - Panics if `create_info.usage` is empty.
    ///
    /// [`try_new`]: Self::try_new
    #[inline]
    #[track_caller]
    pub fn new(
        device: &Arc<Device>,
        surface: &Arc<Surface>,
        create_info: &SwapchainCreateInfo<'_>,
    ) -> Result<(Arc<Swapchain>, Vec<Arc<Image>>), VulkanError> {
        match Self::try_new(device, surface, create_info) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a new `Swapchain`.
    ///
    /// This function returns the swapchain plus a list of the images that belong to the
    /// swapchain. The order in which the images are returned is important for the
    /// `acquire_next_image` and `present` functions.
    ///
    /// # Panics
    ///
    /// - Panics if the device and the surface don't belong to the same instance.
    /// - Panics if `create_info.usage` is empty.
    #[inline]
    pub fn try_new(
        device: &Arc<Device>,
        surface: &Arc<Surface>,
        create_info: &SwapchainCreateInfo<'_>,
    ) -> Result<(Arc<Swapchain>, Vec<Arc<Image>>), Validated<VulkanError>> {
        Self::validate_new_inner(device, surface, create_info)?;

        Ok(unsafe { Self::new_unchecked(device, surface, create_info) }?)
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn new_unchecked(
        device: &Arc<Device>,
        surface: &Arc<Surface>,
        create_info: &SwapchainCreateInfo<'_>,
    ) -> Result<(Arc<Swapchain>, Vec<Arc<Image>>), VulkanError> {
        let (handle, image_handles) =
            unsafe { Self::new_inner_unchecked(device, surface, create_info, None) }?;

        unsafe { Self::from_handle(device, handle, image_handles, surface, create_info) }
    }

    /// Creates a new swapchain from this one, panicking on a validation error.
    ///
    /// Use this when a swapchain has become invalidated, such as due to window resizes.
    ///
    /// This is a shortcut for `try_recreate().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_recreate`] returns a [`ValidationError`].
    ///
    /// [`try_recreate`]: Self::try_recreate
    #[track_caller]
    pub fn recreate(
        self: &Arc<Self>,
        create_info: &SwapchainCreateInfo<'_>,
    ) -> Result<(Arc<Swapchain>, Vec<Arc<Image>>), VulkanError> {
        match Self::try_recreate(self, create_info) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a new swapchain from this one.
    ///
    /// Use this when a swapchain has become invalidated, such as due to window resizes.
    pub fn try_recreate(
        self: &Arc<Self>,
        create_info: &SwapchainCreateInfo<'_>,
    ) -> Result<(Arc<Swapchain>, Vec<Arc<Image>>), Validated<VulkanError>> {
        Self::validate_new_inner(&self.device, &self.surface, create_info)?;

        {
            let mut is_retired = self.is_retired.lock();

            // The swapchain has already been used to create a new one.
            if *is_retired {
                return Err(Box::new(ValidationError {
                    context: "self".into(),
                    problem: "has already been used to recreate a swapchain".into(),
                    vuids: &["VUID-VkSwapchainCreateInfoKHR-oldSwapchain-01933"],
                    ..Default::default()
                })
                .into());
            } else {
                // According to the documentation of VkSwapchainCreateInfoKHR:
                //
                // > Upon calling vkCreateSwapchainKHR with a oldSwapchain that is not VK_NULL_HANDLE,
                // > any images not acquired by the application may be freed by the implementation,
                // > which may occur even if creation of the new swapchain fails.
                //
                // Therefore, we set retired to true and keep it to true even if the call to `vkCreateSwapchainKHR` below fails.
                *is_retired = true;
            }
        }

        Ok(unsafe { self.recreate_unchecked(create_info) }?)
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn recreate_unchecked(
        self: &Arc<Self>,
        create_info: &SwapchainCreateInfo<'_>,
    ) -> Result<(Arc<Swapchain>, Vec<Arc<Image>>), VulkanError> {
        // According to the documentation of VkSwapchainCreateInfoKHR:
        //
        // > Upon calling vkCreateSwapchainKHR with a oldSwapchain that is not VK_NULL_HANDLE,
        // > any images not acquired by the application may be freed by the implementation,
        // > which may occur even if creation of the new swapchain fails.
        //
        // Therefore, we set retired to true and keep it to true,
        // even if the call to `vkCreateSwapchainKHR` below fails.
        *self.is_retired.lock() = true;

        let (handle, image_handles) = unsafe {
            Self::new_inner_unchecked(&self.device, &self.surface, create_info, Some(self))
        }?;

        let (swapchain, swapchain_images) = unsafe {
            Self::from_handle(
                &self.device,
                handle,
                image_handles,
                &self.surface,
                create_info,
            )
        }?;

        if self.full_screen_exclusive == FullScreenExclusive::ApplicationControlled {
            swapchain.full_screen_exclusive_held.store(
                self.full_screen_exclusive_held.load(Ordering::Relaxed),
                Ordering::Relaxed,
            );
        };

        Ok((swapchain, swapchain_images))
    }

    fn validate_new_inner(
        device: &Device,
        surface: &Surface,
        create_info: &SwapchainCreateInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        if !device.enabled_extensions().khr_swapchain {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "khr_swapchain",
                )])]),
                ..Default::default()
            }));
        }

        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;

        let &SwapchainCreateInfo {
            flags: _,
            min_image_count,
            image_format,
            image_view_formats: _,
            image_color_space,
            image_extent,
            image_array_layers,
            image_usage,
            image_sharing: _,
            pre_transform,
            composite_alpha,
            present_mode,
            present_modes,
            clipped: _,
            scaling_behavior,
            present_gravity,
            full_screen_exclusive,
            win32_monitor,
            _ne: _,
        } = create_info;

        assert_eq!(device.instance(), surface.instance());

        // VUID-VkSwapchainCreateInfoKHR-surface-01270
        if !device.active_queue_family_indices().iter().any(|&index| {
            unsafe {
                device
                    .physical_device()
                    .surface_support_unchecked(index, surface)
            }
            .unwrap_or_default()
        }) {
            return Err(Box::new(ValidationError {
                context: "surface".into(),
                problem: "is not supported by the physical device".into(),
                vuids: &["VUID-VkSwapchainCreateInfoKHR-surface-01270"],
                ..Default::default()
            }));
        }

        if win32_monitor.is_some() {
            if surface.api() != SurfaceApi::Win32 {
                return Err(Box::new(ValidationError {
                    problem: "`surface` is not a Win32 surface, but \
                        `create_info.win32_monitor` is `Some`"
                        .into(),
                    ..Default::default()
                }));
            }
        } else if surface.api() == SurfaceApi::Win32
            && full_screen_exclusive == FullScreenExclusive::ApplicationControlled
        {
            return Err(Box::new(ValidationError {
                problem: "`surface` is a Win32 surface, and \
                    `create_info.full_screen_exclusive` is \
                    `FullScreenExclusive::ApplicationControlled`, but \
                    `create_info.win32_monitor` is `None`"
                    .into(),
                vuids: &["VUID-VkSwapchainCreateInfoKHR-pNext-02679"],
                ..Default::default()
            }));
        }

        let surface_capabilities = unsafe {
            device.physical_device().surface_capabilities_unchecked(
                surface,
                &SurfaceInfo {
                    present_mode: device
                        .enabled_extensions()
                        .ext_swapchain_maintenance1
                        .then_some(present_mode),
                    full_screen_exclusive,
                    win32_monitor,
                    ..Default::default()
                },
            )
        }
        .map_err(|_err| {
            Box::new(ValidationError {
                problem: "`PhysicalDevice::surface_capabilities` \
                    returned an error"
                    .into(),
                ..Default::default()
            })
        })?;
        let surface_formats = unsafe {
            device.physical_device().surface_formats_unchecked(
                surface,
                &SurfaceInfo {
                    present_mode: device
                        .enabled_extensions()
                        .ext_swapchain_maintenance1
                        .then_some(present_mode),
                    full_screen_exclusive,
                    win32_monitor,
                    ..Default::default()
                },
            )
        }
        .map_err(|_err| {
            Box::new(ValidationError {
                problem: "`PhysicalDevice::surface_formats` \
                    returned an error"
                    .into(),
                ..Default::default()
            })
        })?;
        let surface_present_modes = unsafe {
            device.physical_device().surface_present_modes_unchecked(
                surface,
                &SurfaceInfo {
                    full_screen_exclusive,
                    win32_monitor,
                    ..Default::default()
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

        if surface_capabilities
            .max_image_count
            .is_some_and(|c| min_image_count > c)
        {
            return Err(Box::new(ValidationError {
                problem: "`create_info.min_image_count` is greater than the `max_image_count` \
                    value of the capabilities of `surface`"
                    .into(),
                vuids: &["VUID-VkSwapchainCreateInfoKHR-minImageCount-01272"],
                ..Default::default()
            }));
        }

        if min_image_count < surface_capabilities.min_image_count {
            return Err(Box::new(ValidationError {
                problem: "`create_info.min_image_count` is less than the `min_image_count` \
                    value of the capabilities of `surface`"
                    .into(),
                vuids: &["VUID-VkSwapchainCreateInfoKHR-presentMode-02839"],
                ..Default::default()
            }));
        }

        if !surface_formats.contains(&(image_format, image_color_space)) {
            return Err(Box::new(ValidationError {
                problem: "the combination of `create_info.image_format` and \
                    `create_info.image_color_space` is not supported for `surface`"
                    .into(),
                vuids: &["VUID-VkSwapchainCreateInfoKHR-imageFormat-01273"],
                ..Default::default()
            }));
        }

        if image_array_layers > surface_capabilities.max_image_array_layers {
            return Err(Box::new(ValidationError {
                problem: "`create_info.image_array_layers` is greater than the \
                    `max_image_array_layers` value of the capabilities of `surface`"
                    .into(),
                vuids: &["VUID-VkSwapchainCreateInfoKHR-imageArrayLayers-01275"],
                ..Default::default()
            }));
        }

        if matches!(
            present_mode,
            PresentMode::Immediate
                | PresentMode::Mailbox
                | PresentMode::Fifo
                | PresentMode::FifoRelaxed
        ) && !surface_capabilities
            .supported_usage_flags
            .contains(image_usage)
        {
            return Err(Box::new(ValidationError {
                problem: "`create_info.present_mode` is `PresentMode::Immediate`, \
                    `PresentMode::Mailbox`, `PresentMode::Fifo` or `PresentMode::FifoRelaxed`, \
                    and `create_info.image_usage` contains flags that are not set in \
                    the `supported_usage_flags` value of the capabilities of `surface`"
                    .into(),
                vuids: &["VUID-VkSwapchainCreateInfoKHR-presentMode-01427"],
                ..Default::default()
            }));
        }

        if !surface_capabilities
            .supported_transforms
            .contains_enum(pre_transform)
        {
            return Err(Box::new(ValidationError {
                problem: "`create_info.pre_transform` is not present in the \
                    `supported_transforms` value of the capabilities of `surface`"
                    .into(),
                vuids: &["VUID-VkSwapchainCreateInfoKHR-preTransform-01279"],
                ..Default::default()
            }));
        }

        if !surface_capabilities
            .supported_composite_alpha
            .contains_enum(composite_alpha)
        {
            return Err(Box::new(ValidationError {
                problem: "`create_info.composite_alpha` is not present in the \
                    `supported_composite_alpha` value of the capabilities of `surface`"
                    .into(),
                vuids: &["VUID-VkSwapchainCreateInfoKHR-compositeAlpha-01280"],
                ..Default::default()
            }));
        }

        if !surface_present_modes.contains(&present_mode) {
            return Err(Box::new(ValidationError {
                problem: "`create_info.present_mode` is not supported for `surface`".into(),
                vuids: &["VUID-VkSwapchainCreateInfoKHR-presentMode-01281"],
                ..Default::default()
            }));
        }

        if present_modes.is_empty() {
            if let Some(scaling_behavior) = scaling_behavior {
                if !surface_capabilities
                    .supported_present_scaling
                    .contains_enum(scaling_behavior)
                {
                    return Err(Box::new(ValidationError {
                        problem: "`create_info.scaling_behavior` is not present in the \
                            `supported_present_scaling` value of the \
                            capabilities of `surface`"
                            .into(),
                        vuids: &[
                            "VUID-VkSwapchainPresentScalingCreateInfoEXT-scalingBehavior-07770",
                        ],
                        ..Default::default()
                    }));
                }
            }

            if let Some(present_gravity) = present_gravity {
                for (axis_index, (present_gravity, supported_present_gravity)) in present_gravity
                    .into_iter()
                    .zip(surface_capabilities.supported_present_gravity)
                    .enumerate()
                {
                    if !supported_present_gravity.contains_enum(present_gravity) {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`create_info.present_gravity[{0}]` is not present in the \
                                `supported_present_gravity[{0}]` value of the \
                                capabilities of `surface`",
                                axis_index,
                            )
                            .into(),
                            vuids: &[
                                "VUID-VkSwapchainPresentScalingCreateInfoEXT-presentGravityX-07772",
                                "VUID-VkSwapchainPresentScalingCreateInfoEXT-presentGravityY-07774",
                            ],
                            ..Default::default()
                        }));
                    }
                }
            }
        } else {
            for (index, &present_mode) in present_modes.iter().enumerate() {
                if !surface_present_modes.contains(&present_mode) {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`create_info.present_modes[{}]` is not supported for `surface`",
                            index,
                        )
                        .into(),
                        vuids: &["VUID-VkSwapchainPresentModesCreateInfoEXT-None-07762"],
                        ..Default::default()
                    }));
                }

                if !surface_capabilities
                    .compatible_present_modes
                    .contains(&present_mode)
                {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`create_info.present_modes[{}]` is not present in the \
                            `compatible_present_modes` value of the \
                            capabilities of `surface`",
                            index,
                        )
                        .into(),
                        vuids: &["VUID-VkSwapchainPresentModesCreateInfoEXT-pPresentModes-07763"],
                        ..Default::default()
                    }));
                }

                if scaling_behavior.is_some() || present_gravity.is_some() {
                    let surface_capabilities = unsafe {
                        device.physical_device().surface_capabilities_unchecked(
                            surface,
                            &SurfaceInfo {
                                present_mode: Some(present_mode),
                                full_screen_exclusive,
                                win32_monitor,
                                ..Default::default()
                            },
                        )
                    }
                    .map_err(|_err| {
                        Box::new(ValidationError {
                            problem: "`PhysicalDevice::surface_capabilities` \
                                        returned an error"
                                .into(),
                            ..Default::default()
                        })
                    })?;

                    if let Some(scaling_behavior) = scaling_behavior {
                        if !surface_capabilities
                            .supported_present_scaling
                            .contains_enum(scaling_behavior)
                        {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "`create_info.scaling_behavior` is not present in the \
                                    `supported_present_scaling` value of the \
                                    capabilities of `surface` for \
                                    `create_info.present_modes[{}]`",
                                    index,
                                )
                                .into(),
                                vuids: &[
                                    "VUID-VkSwapchainPresentScalingCreateInfoEXT-scalingBehavior-07771",
                                ],
                                ..Default::default()
                            }));
                        }
                    }

                    if let Some(present_gravity) = present_gravity {
                        for (axis_index, (present_gravity, supported_present_gravity)) in
                            present_gravity
                                .into_iter()
                                .zip(surface_capabilities.supported_present_gravity)
                                .enumerate()
                        {
                            if !supported_present_gravity.contains_enum(present_gravity) {
                                return Err(Box::new(ValidationError {
                                    problem: format!(
                                        "`create_info.present_gravity[{0}]` is not present in the \
                                        `supported_present_gravity[{0}]` value of the \
                                        capabilities of `surface` for \
                                        `create_info.present_modes[{1}]`",
                                        axis_index, index,
                                    )
                                    .into(),
                                    vuids: &[
                                        "VUID-VkSwapchainPresentScalingCreateInfoEXT-presentGravityX-07773",
                                        "VUID-VkSwapchainPresentScalingCreateInfoEXT-presentGravityY-07775",
                                    ],
                                    ..Default::default()
                                }));
                            }
                        }
                    }
                }
            }
        }

        if scaling_behavior.is_some() {
            if let Some(min_scaled_image_extent) = surface_capabilities.min_scaled_image_extent {
                if image_extent[0] < min_scaled_image_extent[0]
                    || image_extent[1] < min_scaled_image_extent[1]
                {
                    return Err(Box::new(ValidationError {
                        problem: "`scaling_behavior` is `Some`, and an element of \
                            `create_info.image_extent` is less than the corresponding element \
                            of the `min_scaled_image_extent` value of the \
                            capabilities of `surface`"
                            .into(),
                        vuids: &["VUID-VkSwapchainCreateInfoKHR-pNext-07782"],
                        ..Default::default()
                    }));
                }
            }

            if let Some(max_scaled_image_extent) = surface_capabilities.max_scaled_image_extent {
                if image_extent[0] > max_scaled_image_extent[0]
                    || image_extent[1] > max_scaled_image_extent[1]
                {
                    return Err(Box::new(ValidationError {
                        problem: "`scaling_behavior` is `Some`, and an element of \
                            `create_info.image_extent` is greater than the corresponding element \
                            of the `max_scaled_image_extent` value of the \
                            capabilities of `surface`"
                            .into(),
                        vuids: &["VUID-VkSwapchainCreateInfoKHR-pNext-07782"],
                        ..Default::default()
                    }));
                }
            }
        } else {
            /*
            This check is in the spec, but in practice leads to race conditions.
            The window can be resized between calling `surface_capabilities` to get the
            min/max extent, and then creating the swapchain.

            See this discussion:
            https://github.com/KhronosGroup/Vulkan-Docs/issues/1144

            if image_extent[0] < surface_capabilities.min_image_extent[0]
                || image_extent[1] < surface_capabilities.min_image_extent[1]
            {
                return Err(Box::new(ValidationError {
                    problem: "`scaling_behavior` is `Some`, and an element of \
                        `create_info.image_extent` is less than the corresponding element \
                        of the `min_image_extent` value of the \
                        capabilities of `surface`"
                        .into(),
                    vuids: &["VUID-VkSwapchainCreateInfoKHR-pNext-07781"],
                    ..Default::default()
                }));
            }

            if image_extent[0] > surface_capabilities.max_image_extent[0]
                || image_extent[1] > surface_capabilities.max_image_extent[1]
            {
                return Err(Box::new(ValidationError {
                    problem: "`scaling_behavior` is `Some`, and an element of \
                        `create_info.image_extent` is greater than the corresponding element \
                        of the `max_image_extent` value of the \
                        capabilities of `surface`"
                        .into(),
                    vuids: &["VUID-VkSwapchainCreateInfoKHR-pNext-07781"],
                    ..Default::default()
                }));
            }
            */
        }

        Ok(())
    }

    unsafe fn new_inner_unchecked(
        device: &Device,
        surface: &Surface,
        create_info: &SwapchainCreateInfo<'_>,
        old_swapchain: Option<&Swapchain>,
    ) -> Result<(vk::SwapchainKHR, Vec<vk::Image>), VulkanError> {
        let create_info_fields1_vk = create_info.to_vk_fields1();
        let mut create_info_extensions_vk = create_info.to_vk_extensions(&create_info_fields1_vk);
        let create_info_vk = create_info.to_vk(
            surface.handle(),
            old_swapchain.map_or(vk::SwapchainKHR::null(), |os| os.handle),
            &mut create_info_extensions_vk,
        );

        let fns = device.fns();

        let handle = {
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.khr_swapchain.create_swapchain_khr)(
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

        let image_handles = loop {
            let mut count = 0;
            unsafe {
                (fns.khr_swapchain.get_swapchain_images_khr)(
                    device.handle(),
                    handle,
                    &mut count,
                    ptr::null_mut(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;

            let mut images = Vec::with_capacity(count as usize);
            let result = unsafe {
                (fns.khr_swapchain.get_swapchain_images_khr)(
                    device.handle(),
                    handle,
                    &mut count,
                    images.as_mut_ptr(),
                )
            };

            match result {
                vk::Result::SUCCESS => {
                    unsafe { images.set_len(count as usize) };
                    break images;
                }
                vk::Result::INCOMPLETE => (),
                err => return Err(VulkanError::from(err)),
            }
        };

        Ok((handle, image_handles))
    }

    /// Creates a new `Swapchain` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` and `image_handles` must be valid Vulkan object handles created from `device`.
    /// - `handle` must not be retired.
    /// - `image_handles` must be swapchain images owned by `handle`, in the same order as they
    ///   were returned by `vkGetSwapchainImagesKHR`.
    /// - `surface` and `create_info` must match the info used to create the object.
    pub unsafe fn from_handle(
        device: &Arc<Device>,
        handle: vk::SwapchainKHR,
        image_handles: impl IntoIterator<Item = vk::Image>,
        surface: &Arc<Surface>,
        create_info: &SwapchainCreateInfo<'_>,
    ) -> Result<(Arc<Swapchain>, Vec<Arc<Image>>), VulkanError> {
        let &SwapchainCreateInfo {
            flags,
            min_image_count,
            image_format,
            image_view_formats,
            image_color_space,
            image_extent,
            image_array_layers,
            image_usage,
            image_sharing,
            pre_transform,
            composite_alpha,
            present_mode,
            present_modes,
            clipped,
            scaling_behavior,
            present_gravity,
            full_screen_exclusive,
            win32_monitor,
            _ne: _,
        } = create_info;

        let swapchain = Arc::new(Swapchain {
            handle,
            device: InstanceOwnedDebugWrapper(device.clone()),
            surface: InstanceOwnedDebugWrapper(surface.clone()),
            id: Self::next_id(),

            flags,
            min_image_count,
            image_format,
            image_view_formats: image_view_formats.to_owned(),
            image_color_space,
            image_extent,
            image_array_layers,
            image_usage,
            image_sharing: image_sharing.to_owned(),
            pre_transform,
            composite_alpha,
            present_mode,
            present_modes: present_modes.iter().copied().collect(),
            clipped,
            scaling_behavior,
            present_gravity,
            full_screen_exclusive,
            win32_monitor,

            prev_present_id: Default::default(),
            full_screen_exclusive_held: AtomicBool::new(false),
            images: image_handles
                .into_iter()
                .map(|handle| ImageEntry {
                    handle,
                    layout_initialized: AtomicBool::new(false),
                })
                .collect(),
            is_retired: Mutex::new(false),
        });

        let swapchain_images = swapchain
            .images
            .iter()
            .enumerate()
            .map(|(image_index, entry)| {
                Ok(Arc::new(unsafe {
                    Image::from_swapchain(entry.handle, &swapchain, image_index as u32)
                }?))
            })
            .collect::<Result<_, VulkanError>>()?;

        Ok((swapchain, swapchain_images))
    }

    /// Returns the creation parameters of the swapchain.
    #[inline]
    pub fn create_info(&self) -> SwapchainCreateInfo<'_> {
        SwapchainCreateInfo {
            flags: self.flags,
            min_image_count: self.min_image_count,
            image_format: self.image_format,
            image_view_formats: &self.image_view_formats,
            image_color_space: self.image_color_space,
            image_extent: self.image_extent,
            image_array_layers: self.image_array_layers,
            image_usage: self.image_usage,
            image_sharing: self.image_sharing.as_ref(),
            pre_transform: self.pre_transform,
            composite_alpha: self.composite_alpha,
            present_mode: self.present_mode,
            present_modes: &self.present_modes,
            clipped: self.clipped,
            scaling_behavior: self.scaling_behavior,
            present_gravity: self.present_gravity,
            full_screen_exclusive: self.full_screen_exclusive,
            win32_monitor: self.win32_monitor,
            _ne: crate::NE,
        }
    }

    /// Returns the surface that the swapchain was created from.
    #[inline]
    pub fn surface(&self) -> &Arc<Surface> {
        &self.surface
    }

    /// Returns the flags that the swapchain was created with.
    #[inline]
    pub fn flags(&self) -> SwapchainCreateFlags {
        self.flags
    }

    /// If `image` is one of the images of this swapchain, returns its index within the swapchain.
    #[inline]
    pub fn index_of_image(&self, image: &Image) -> Option<u32> {
        self.images
            .iter()
            .position(|entry| entry.handle == image.handle())
            .map(|i| i as u32)
    }

    /// Returns the number of images of the swapchain.
    #[inline]
    pub fn image_count(&self) -> u32 {
        self.images.len() as u32
    }

    /// Returns the format of the images of the swapchain.
    #[inline]
    pub fn image_format(&self) -> Format {
        self.image_format
    }

    /// Returns the formats that an image view created from a swapchain image can have.
    #[inline]
    pub fn image_view_formats(&self) -> &[Format] {
        &self.image_view_formats
    }

    /// Returns the color space of the images of the swapchain.
    #[inline]
    pub fn image_color_space(&self) -> ColorSpace {
        self.image_color_space
    }

    /// Returns the extent of the images of the swapchain.
    #[inline]
    pub fn image_extent(&self) -> [u32; 2] {
        self.image_extent
    }

    /// Returns the number of array layers of the images of the swapchain.
    #[inline]
    pub fn image_array_layers(&self) -> u32 {
        self.image_array_layers
    }

    /// Returns the usage of the images of the swapchain.
    #[inline]
    pub fn image_usage(&self) -> ImageUsage {
        self.image_usage
    }

    /// Returns the sharing of the images of the swapchain.
    #[inline]
    pub fn image_sharing(&self) -> Sharing<'_> {
        self.image_sharing.as_ref()
    }

    /// Returns the pre-transform that was passed when creating the swapchain.
    #[inline]
    pub fn pre_transform(&self) -> SurfaceTransform {
        self.pre_transform
    }

    /// Returns the alpha mode that was passed when creating the swapchain.
    #[inline]
    pub fn composite_alpha(&self) -> CompositeAlpha {
        self.composite_alpha
    }

    /// Returns the present mode that was passed when creating the swapchain.
    #[inline]
    pub fn present_mode(&self) -> PresentMode {
        self.present_mode
    }

    /// Returns the alternative present modes that were passed when creating the swapchain.
    #[inline]
    pub fn present_modes(&self) -> &[PresentMode] {
        &self.present_modes
    }

    /// Returns the value of `clipped` that was passed when creating the swapchain.
    #[inline]
    pub fn clipped(&self) -> bool {
        self.clipped
    }

    /// Returns the scaling behavior that was passed when creating the swapchain.
    #[inline]
    pub fn scaling_behavior(&self) -> Option<PresentScaling> {
        self.scaling_behavior
    }

    /// Returns the scaling behavior that was passed when creating the swapchain.
    #[inline]
    pub fn present_gravity(&self) -> Option<[PresentGravity; 2]> {
        self.present_gravity
    }

    /// Returns the value of 'full_screen_exclusive` that was passed when creating the swapchain.
    #[inline]
    pub fn full_screen_exclusive(&self) -> FullScreenExclusive {
        self.full_screen_exclusive
    }

    /// Acquires temporary ownership of a swapchain image, panicking on a validation error.
    ///
    /// The function returns the index of the image in the array of images that was returned
    /// when creating the swapchain. The image will not be available immediately after the function
    /// returns successfully.
    ///
    /// When the image becomes available, a semaphore and/or fence will be signaled. The image can
    /// then be accessed by the host and/or device. After this, the image must be *presented* back
    /// to the swapchain using the [`present`] queue command or *released* using the
    /// [`release_images`] method.
    ///
    /// This is a shortcut for `try_acquire_next_image().map_err(Validated::unwrap)`.
    ///
    /// # Forward progress
    ///
    /// For the purposes of forward progress of acquiring swapchain images, Vulkan defines an
    /// overall minimum number of swapchain images like so:
    /// - If [`present_modes`] is not empty, the maximum of
    ///   [`SurfaceCapabilities::min_image_count`] returned by
    ///   [`PhysicalDevice::surface_capabilities`] called with a [`SurfaceInfo::present_mode`] set
    ///   to each of the present modes in [`present_modes`].
    /// - Otherwise, the [`SurfaceCapabilities::min_image_count`] returned with
    ///   [`SurfaceInfo::present_mode`] unset.
    ///
    /// Vulkan also defines a surplus of swapchain images as [`image_count`] minus the above
    /// overall minimum number of swapchain images.
    ///
    /// The forward progress guarantee, then, is upheld when the number of currently acquired
    /// images is less than or equal to the surplus of images and acquiring another image returns
    /// in finite time. "Currently acquired" only includes images that were merely acquired but not
    /// yet presented or released. The forward progress guarantee exists to prevent deadlocks, and
    /// it is undefined behavior if it isn't upheld. You should not acquire another image if the
    /// number of currently acquired images exceeds the surplus of images.
    ///
    /// The most practical way to guarantee forward progress is to:
    /// - have a *frames in flight system* to ensure only one image is ever currently acquired, and
    /// - have one more swapchain image than the overall minimum defined above.
    ///
    /// Besides guaranteeing forward progress, this also guarantees that acquiring another image
    /// never blocks on the host, and so this is a standard solution that everyone already employs
    /// for this alone.
    ///
    /// # Safety
    ///
    /// - `self` must be kept alive until either `acquire_info.semaphore` or `acquire_info.fence`
    ///   is signaled.
    /// - If [forward progress] cannot be guaranteed then `acquire_info.timeout` must not be
    ///   `None`.
    ///
    /// If `acquire_info.semaphore` is `Some`:
    /// - The semaphore must be kept alive until it is signaled.
    /// - When the signal operation is executed, the semaphore must be in the unsignaled state.
    ///
    /// If `acquire_info.fence` is `Some`:
    /// - The fence must be kept alive until it is signaled.
    /// - The fence must be unsignaled and must not be associated with any other command that is
    ///   still executing.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_acquire_next_image`] returns a [`ValidationError`].
    ///
    /// [`present`]: crate::device::QueueGuard::present
    /// [`release_images`]: Self::release_images
    /// [`present_modes`]: Self::present_modes
    /// [`SurfaceCapabilities::min_image_count`]: crate::swapchain::surface::SurfaceCapabilities::min_image_count
    /// [`PhysicalDevice::surface_capabilities`]: crate::device::physical::PhysicalDevice::surface_capabilities
    /// [`SurfaceInfo::present_mode`]: crate::swapchain::surface::SurfaceInfo::present_mode
    /// [`image_count`]: Self::image_count
    /// [forward progress]: Self#forward-progress
    /// [`try_acquire_next_image`]: Self::try_acquire_next_image
    #[inline]
    #[track_caller]
    pub unsafe fn acquire_next_image(
        &self,
        acquire_info: &AcquireNextImageInfo<'_>,
    ) -> Result<AcquiredImage, VulkanError> {
        match unsafe { self.try_acquire_next_image(acquire_info) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Acquires temporary ownership of a swapchain image.
    ///
    /// The function returns the index of the image in the array of images that was returned
    /// when creating the swapchain. The image will not be available immediately after the function
    /// returns successfully.
    ///
    /// When the image becomes available, a semaphore and/or fence will be signaled. The image can
    /// then be accessed by the host and/or device. After this, the image must be *presented* back
    /// to the swapchain using the [`present`] queue command or *released* using the
    /// [`release_images`] method.
    ///
    /// # Forward progress
    ///
    /// For the purposes of forward progress of acquiring swapchain images, Vulkan defines an
    /// overall minimum number of swapchain images like so:
    /// - If [`present_modes`] is not empty, the maximum of
    ///   [`SurfaceCapabilities::min_image_count`] returned by
    ///   [`PhysicalDevice::surface_capabilities`] called with a [`SurfaceInfo::present_mode`] set
    ///   to each of the present modes in [`present_modes`].
    /// - Otherwise, the [`SurfaceCapabilities::min_image_count`] returned with
    ///   [`SurfaceInfo::present_mode`] unset.
    ///
    /// Vulkan also defines a surplus of swapchain images as [`image_count`] minus the above
    /// overall minimum number of swapchain images.
    ///
    /// The forward progress guarantee, then, is upheld when the number of currently acquired
    /// images is less than or equal to the surplus of images and acquiring another image returns
    /// in finite time. "Currently acquired" only includes images that were merely acquired but not
    /// yet presented or released. The forward progress guarantee exists to prevent deadlocks, and
    /// it is undefined behavior if it isn't upheld. You should not acquire another image if the
    /// number of currently acquired images exceeds the surplus of images.
    ///
    /// The most practical way to guarantee forward progress is to:
    /// - have a *frames in flight system* to ensure only one image is ever currently acquired, and
    /// - have one more swapchain image than the overall minimum defined above.
    ///
    /// Besides guaranteeing forward progress, this also guarantees that acquiring another image
    /// never blocks on the host, and so this is a standard solution that everyone already employs
    /// for this alone.
    ///
    /// # Safety
    ///
    /// - `self` must be kept alive until either `acquire_info.semaphore` or `acquire_info.fence`
    ///   is signaled.
    /// - If [forward progress] cannot be guaranteed then `acquire_info.timeout` must not be
    ///   `None`.
    ///
    /// If `acquire_info.semaphore` is `Some`:
    /// - The semaphore must be kept alive until it is signaled.
    /// - When the signal operation is executed, the semaphore must be in the unsignaled state.
    ///
    /// If `acquire_info.fence` is `Some`:
    /// - The fence must be kept alive until it is signaled.
    /// - The fence must be unsignaled and must not be associated with any other command that is
    ///   still executing.
    ///
    /// [`present`]: crate::device::QueueGuard::present
    /// [`release_images`]: Self::release_images
    /// [`present_modes`]: Self::present_modes
    /// [`SurfaceCapabilities::min_image_count`]: crate::swapchain::surface::SurfaceCapabilities::min_image_count
    /// [`PhysicalDevice::surface_capabilities`]: crate::device::physical::PhysicalDevice::surface_capabilities
    /// [`SurfaceInfo::present_mode`]: crate::swapchain::surface::SurfaceInfo::present_mode
    /// [`image_count`]: Self::image_count
    /// [forward progress]: Self#forward-progress-1
    #[inline]
    pub unsafe fn try_acquire_next_image(
        &self,
        acquire_info: &AcquireNextImageInfo<'_>,
    ) -> Result<AcquiredImage, Validated<VulkanError>> {
        let is_retired_lock = self.is_retired.lock();
        self.validate_acquire_next_image(acquire_info, *is_retired_lock)?;

        Ok(unsafe { self.acquire_next_image_unchecked(acquire_info) }?)
    }

    fn validate_acquire_next_image(
        &self,
        acquire_info: &AcquireNextImageInfo<'_>,
        is_retired: bool,
    ) -> Result<(), Box<ValidationError>> {
        acquire_info
            .validate(&self.device)
            .map_err(|err| err.add_context("acquire_info"))?;

        if is_retired {
            return Err(Box::new(ValidationError {
                problem: "this swapchain is retired".into(),
                vuids: &["VUID-VkAcquireNextImageInfoKHR-swapchain-01675"],
                ..Default::default()
            }));
        }

        // unsafe
        // VUID-vkAcquireNextImage2KHR-surface-07784
        // VUID-VkAcquireNextImageInfoKHR-semaphore-01288
        // VUID-VkAcquireNextImageInfoKHR-semaphore-01781
        // VUID-VkAcquireNextImageInfoKHR-fence-01289

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn acquire_next_image_unchecked(
        &self,
        acquire_info: &AcquireNextImageInfo<'_>,
    ) -> Result<AcquiredImage, VulkanError> {
        let acquire_info_vk = acquire_info.to_vk(self.handle(), self.device.device_mask());

        let (image_index, is_suboptimal) = {
            let fns = self.device.fns();
            let mut output = MaybeUninit::uninit();

            let result = if self.device.api_version() >= Version::V1_1
                || self.device.enabled_extensions().khr_device_group
            {
                unsafe {
                    (fns.khr_swapchain.acquire_next_image2_khr)(
                        self.device.handle(),
                        &acquire_info_vk,
                        output.as_mut_ptr(),
                    )
                }
            } else {
                debug_assert!(acquire_info_vk.p_next.is_null());
                unsafe {
                    (fns.khr_swapchain.acquire_next_image_khr)(
                        self.device.handle(),
                        acquire_info_vk.swapchain,
                        acquire_info_vk.timeout,
                        acquire_info_vk.semaphore,
                        acquire_info_vk.fence,
                        output.as_mut_ptr(),
                    )
                }
            };

            match result {
                vk::Result::SUCCESS => (unsafe { output.assume_init() }, false),
                vk::Result::SUBOPTIMAL_KHR => (unsafe { output.assume_init() }, true),
                vk::Result::NOT_READY => return Err(VulkanError::NotReady),
                vk::Result::TIMEOUT => return Err(VulkanError::Timeout),
                err => {
                    let err = VulkanError::from(err);

                    if matches!(err, VulkanError::FullScreenExclusiveModeLost) {
                        self.full_screen_exclusive_held
                            .store(false, Ordering::SeqCst);
                    }

                    return Err(err);
                }
            }
        };

        Ok(AcquiredImage {
            image_index,
            is_suboptimal,
        })
    }

    /// Releases ownership of a set of swapchain images, panicking on a validation error.
    ///
    /// This method is very useful to avoid having to present an acquired image, which is
    /// especially useful when an image of an old swapchain is currently acquired and the swapchain
    /// has been recreated.
    ///
    /// See the [`acquire_next_image`] method for more information about image acquiring.
    ///
    /// This is a shortcut for `try_release_images().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - `release_info.image_indices` must only contain indices of images that are currently
    ///   acquired. "Currently acquired" only includes images that have merely been acquired but
    ///   not yet presented or released.
    /// - `release_info.image_indices` must only contain indices of images that are not in use.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_release_images`] returns a [`ValidationError`].
    ///
    /// [`acquire_next_image`]: Self::acquire_next_image
    /// [`try_release_images`]: Self::try_release_images
    #[inline]
    #[track_caller]
    pub unsafe fn release_images(
        &self,
        release_info: &ReleaseSwapchainImagesInfo<'_>,
    ) -> Result<(), VulkanError> {
        unsafe { self.try_release_images(release_info) }.map_err(Validated::unwrap)
    }

    /// Releases ownership of a set of swapchain images.
    ///
    /// This method is very useful to avoid having to present an acquired image, which is
    /// especially useful when an image of an old swapchain is currently acquired and the swapchain
    /// has been recreated.
    ///
    /// See the [`acquire_next_image`] method for more information about image acquiring.
    ///
    /// # Safety
    ///
    /// - `release_info.image_indices` must only contain indices of images that are currently
    ///   acquired. "Currently acquired" only includes images that have merely been acquired but
    ///   not yet presented or released.
    /// - `release_info.image_indices` must only contain indices of images that are not in use.
    ///
    /// [`acquire_next_image`]: Self::acquire_next_image
    #[inline]
    pub unsafe fn try_release_images(
        &self,
        release_info: &ReleaseSwapchainImagesInfo<'_>,
    ) -> Result<(), Validated<VulkanError>> {
        self.validate_release_images(release_info)?;

        Ok(unsafe { self.release_images_unchecked(release_info) }?)
    }

    fn validate_release_images(
        &self,
        release_info: &ReleaseSwapchainImagesInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        if !self.device.enabled_features().swapchain_maintenance1 {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "swapchain_maintenance1",
                )])]),
                vuids: &["VUID-vkReleaseSwapchainImagesKHR-swapchainMaintenance1-10159"],
                ..Default::default()
            }));
        }

        for &image_index in release_info.image_indices {
            if image_index >= self.image_count() {
                return Err(Box::new(ValidationError {
                    context: format!("release_info.image_indices[{}]", image_index).into(),
                    problem: "is not less than `self.image_count()`".into(),
                    vuids: &["VUID-VkReleaseSwapchainImagesInfoKHR-pImageIndices-07785"],
                    ..Default::default()
                }));
            }
        }

        // unsafe
        // VUID-VkReleaseSwapchainImagesInfoKHR-pImageIndices-07785
        // VUID-VkReleaseSwapchainImagesInfoKHR-pImageIndices-07786

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn release_images_unchecked(
        &self,
        release_info: &ReleaseSwapchainImagesInfo<'_>,
    ) -> Result<(), VulkanError> {
        let release_info_vk = release_info.to_vk(self.handle());

        let fns = self.device().fns();

        unsafe {
            (fns.ext_swapchain_maintenance1.release_swapchain_images_ext)(
                self.device().handle(),
                &release_info_vk,
            )
        }
        .result()
        .map_err(VulkanError::from)
    }

    /// Waits for a swapchain image with a specific present ID to be presented to the user,
    /// panicking on a validation error.
    ///
    /// For this to work, you must set [`SwapchainPresentInfo::present_id`] to `Some` when
    /// presenting. This function will then wait until the swapchain image with the specified ID is
    /// presented.
    ///
    /// Returns whether the presentation was suboptimal. This has the same meaning as in
    /// [`AcquiredImage::is_suboptimal`].
    ///
    /// This is a shortcut for `try_wait_for_present().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_wait_for_present`] returns a [`ValidationError`].
    ///
    /// [`try_wait_for_present`]: Self::try_wait_for_present
    #[inline]
    #[track_caller]
    pub fn wait_for_present(
        &self,
        present_id: NonZero<u64>,
        timeout: Option<Duration>,
    ) -> Result<bool, VulkanError> {
        match self.try_wait_for_present(present_id, timeout) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Waits for a swapchain image with a specific present ID to be presented to the user.
    ///
    /// For this to work, you must set [`SwapchainPresentInfo::present_id`] to `Some` when
    /// presenting. This function will then wait until the swapchain image with the specified ID is
    /// presented.
    ///
    /// Returns whether the presentation was suboptimal. This has the same meaning as in
    /// [`AcquiredImage::is_suboptimal`].
    #[inline]
    pub fn try_wait_for_present(
        &self,
        present_id: NonZero<u64>,
        timeout: Option<Duration>,
    ) -> Result<bool, Validated<VulkanError>> {
        let is_retired_lock = self.is_retired.lock();
        self.validate_wait_for_present(present_id, timeout, *is_retired_lock)?;

        Ok(unsafe { self.wait_for_present_unchecked(present_id, timeout) }?)
    }

    fn validate_wait_for_present(
        &self,
        _present_id: NonZero<u64>,
        timeout: Option<Duration>,
        is_retired: bool,
    ) -> Result<(), Box<ValidationError>> {
        if !self.device.enabled_features().present_wait {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "present_wait",
                )])]),
                vuids: &["VUID-vkWaitForPresentKHR-presentWait-06234"],
                ..Default::default()
            }));
        }

        if is_retired {
            return Err(Box::new(ValidationError {
                problem: "this swapchain is retired".into(),
                vuids: &["VUID-vkWaitForPresentKHR-swapchain-04997"],
                ..Default::default()
            }));
        }

        if let Some(timeout) = timeout {
            if timeout.as_nanos() >= u64::MAX as u128 {
                return Err(Box::new(ValidationError {
                    context: "timeout".into(),
                    problem: "is not less than `u64::MAX` nanoseconds".into(),
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn wait_for_present_unchecked(
        &self,
        present_id: NonZero<u64>,
        timeout: Option<Duration>,
    ) -> Result<bool, VulkanError> {
        let result = {
            let fns = self.device.fns();
            unsafe {
                (fns.khr_present_wait.wait_for_present_khr)(
                    self.device.handle(),
                    self.handle,
                    present_id.get(),
                    timeout.map_or(u64::MAX, |duration| {
                        u64::try_from(duration.as_nanos()).unwrap()
                    }),
                )
            }
        };

        match result {
            vk::Result::SUCCESS => Ok(false),
            vk::Result::SUBOPTIMAL_KHR => Ok(true),
            vk::Result::TIMEOUT => Err(VulkanError::Timeout),
            err => {
                let err = VulkanError::from(err);

                if matches!(err, VulkanError::FullScreenExclusiveModeLost) {
                    self.full_screen_exclusive_held
                        .store(false, Ordering::SeqCst);
                }

                Err(err)
            }
        }
    }

    /// Acquires full-screen exclusivity, panicking on a validation error.
    ///
    /// The swapchain must have been created with [`FullScreenExclusive::ApplicationControlled`],
    /// and must not already hold full-screen exclusivity. Full-screen exclusivity is held until
    /// either the `release_full_screen_exclusive` is called, or if any of the the other
    /// `Swapchain` functions return `FullScreenExclusiveLost`.
    ///
    /// This is a shortcut for
    /// `try_acquire_full_screen_exclusive_mode().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_acquire_full_screen_exclusive_mode`] returns a [`ValidationError`].
    ///
    /// [`try_acquire_full_screen_exclusive_mode`]: Self::try_acquire_full_screen_exclusive_mode
    #[inline]
    #[track_caller]
    pub fn acquire_full_screen_exclusive_mode(&self) -> Result<(), VulkanError> {
        match self.try_acquire_full_screen_exclusive_mode() {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Acquires full-screen exclusivity.
    ///
    /// The swapchain must have been created with [`FullScreenExclusive::ApplicationControlled`],
    /// and must not already hold full-screen exclusivity. Full-screen exclusivity is held until
    /// either the `release_full_screen_exclusive` is called, or if any of the the other
    /// `Swapchain` functions return `FullScreenExclusiveLost`.
    #[inline]
    pub fn try_acquire_full_screen_exclusive_mode(&self) -> Result<(), Validated<VulkanError>> {
        self.validate_acquire_full_screen_exclusive_mode()?;

        Ok(unsafe { self.acquire_full_screen_exclusive_mode_unchecked() }?)
    }

    fn validate_acquire_full_screen_exclusive_mode(&self) -> Result<(), Box<ValidationError>> {
        if *self.is_retired.lock() {
            return Err(Box::new(ValidationError {
                problem: "this swapchain is retired".into(),
                vuids: &["VUID-vkAcquireFullScreenExclusiveModeEXT-swapchain-02674"],
                ..Default::default()
            }));
        }

        if self.full_screen_exclusive != FullScreenExclusive::ApplicationControlled {
            return Err(Box::new(ValidationError {
                context: "self.full_screen_exclusive()".into(),
                problem: "is not `FullScreenExclusive::ApplicationControlled`".into(),
                vuids: &["VUID-vkAcquireFullScreenExclusiveModeEXT-swapchain-02675"],
                ..Default::default()
            }));
        }

        // This must be the last check in this function.
        if self
            .full_screen_exclusive_held
            .swap(true, Ordering::Acquire)
        {
            return Err(Box::new(ValidationError {
                problem: "this swapchain already holds full-screen exclusive access".into(),
                vuids: &["VUID-vkAcquireFullScreenExclusiveModeEXT-swapchain-02676"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn acquire_full_screen_exclusive_mode_unchecked(&self) -> Result<(), VulkanError> {
        self.full_screen_exclusive_held
            .store(true, Ordering::Relaxed);

        let fns = self.device.fns();
        unsafe {
            (fns.ext_full_screen_exclusive
                .acquire_full_screen_exclusive_mode_ext)(
                self.device.handle(), self.handle
            )
        }
        .result()
        .map_err(VulkanError::from)?;

        Ok(())
    }

    /// Releases full-screen exclusivity, panicking on a validation error.
    ///
    /// The swapchain must have been created with [`FullScreenExclusive::ApplicationControlled`],
    /// and must currently hold full-screen exclusivity.
    ///
    /// This is a shortcut for
    /// `try_release_full_screen_exclusive_mode().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_release_full_screen_exclusive_mode`] returns a [`ValidationError`].
    ///
    /// [`try_release_full_screen_exclusive_mode`]: Self::try_release_full_screen_exclusive_mode
    #[inline]
    #[track_caller]
    pub fn release_full_screen_exclusive_mode(&self) -> Result<(), VulkanError> {
        match self.try_release_full_screen_exclusive_mode() {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Releases full-screen exclusivity.
    ///
    /// The swapchain must have been created with [`FullScreenExclusive::ApplicationControlled`],
    /// and must currently hold full-screen exclusivity.
    #[inline]
    pub fn try_release_full_screen_exclusive_mode(&self) -> Result<(), Validated<VulkanError>> {
        self.validate_release_full_screen_exclusive_mode()?;

        Ok(unsafe { self.release_full_screen_exclusive_mode_unchecked() }?)
    }

    fn validate_release_full_screen_exclusive_mode(&self) -> Result<(), Box<ValidationError>> {
        if *self.is_retired.lock() {
            return Err(Box::new(ValidationError {
                problem: "this swapchain is retired".into(),
                vuids: &["VUID-vkReleaseFullScreenExclusiveModeEXT-swapchain-02677"],
                ..Default::default()
            }));
        }

        if self.full_screen_exclusive != FullScreenExclusive::ApplicationControlled {
            return Err(Box::new(ValidationError {
                context: "self.full_screen_exclusive()".into(),
                problem: "is not `FullScreenExclusive::ApplicationControlled`".into(),
                vuids: &["VUID-vkReleaseFullScreenExclusiveModeEXT-swapchain-02678"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn release_full_screen_exclusive_mode_unchecked(&self) -> Result<(), VulkanError> {
        self.full_screen_exclusive_held
            .store(false, Ordering::Release);

        let fns = self.device.fns();
        unsafe {
            (fns.ext_full_screen_exclusive
                .release_full_screen_exclusive_mode_ext)(
                self.device.handle(), self.handle
            )
        }
        .result()
        .map_err(VulkanError::from)?;

        Ok(())
    }

    /// `FullScreenExclusive::AppControlled` is not the active full-screen exclusivity mode,
    /// then this function will always return false. If true is returned the swapchain
    /// is in `FullScreenExclusive::AppControlled` full-screen exclusivity mode and exclusivity
    /// is currently acquired.
    #[inline]
    pub fn is_full_screen_exclusive(&self) -> bool {
        if self.full_screen_exclusive != FullScreenExclusive::ApplicationControlled {
            false
        } else {
            self.full_screen_exclusive_held.load(Ordering::SeqCst)
        }
    }

    // This method is necessary to allow `SwapchainImage`s to signal when they have been
    // transitioned out of their initial `undefined` image layout.
    //
    // See the `ImageAccess::layout_initialized` method documentation for more details.
    pub(crate) fn image_layout_initialized(&self, image_index: u32) {
        let image_entry = self.images.get(image_index as usize);
        if let Some(image_entry) = image_entry {
            image_entry
                .layout_initialized
                .store(true, Ordering::Relaxed);
        }
    }

    pub(crate) fn is_image_layout_initialized(&self, image_index: u32) -> bool {
        let image_entry = self.images.get(image_index as usize);
        if let Some(image_entry) = image_entry {
            image_entry.layout_initialized.load(Ordering::Relaxed)
        } else {
            false
        }
    }

    #[inline]
    pub(crate) unsafe fn full_screen_exclusive_held(&self) -> &AtomicBool {
        &self.full_screen_exclusive_held
    }

    #[inline]
    pub(crate) unsafe fn try_claim_present_id(&self, present_id: NonZero<u64>) -> bool {
        let present_id = u64::from(present_id);
        self.prev_present_id.fetch_max(present_id, Ordering::SeqCst) < present_id
    }
}

impl Drop for Swapchain {
    #[inline]
    fn drop(&mut self) {
        let fns = self.device.fns();
        unsafe {
            (fns.khr_swapchain.destroy_swapchain_khr)(
                self.device.handle(),
                self.handle,
                ptr::null(),
            )
        };
    }
}

unsafe impl VulkanObject for Swapchain {
    type Handle = vk::SwapchainKHR;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for Swapchain {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl_id_counter!(Swapchain);

/// Parameters to create a new `Swapchain`.
///
/// Many of the values here must be supported by the physical device.
/// [`PhysicalDevice`](crate::device::physical::PhysicalDevice) has several
/// methods to query what is supported.
#[derive(Clone, Debug)]
pub struct SwapchainCreateInfo<'a> {
    /// Additional properties of the swapchain.
    ///
    /// The default value is empty.
    pub flags: SwapchainCreateFlags,

    /// The minimum number of images that will be created.
    ///
    /// The implementation is allowed to create more than this number, but never less.
    ///
    /// The default value is `2`.
    pub min_image_count: u32,

    /// The format of the created images.
    ///
    /// The default value is `Format::UNDEFINED`.
    pub image_format: Format,

    /// The formats that an image view can have when it is created from one of the swapchain
    /// images.
    ///
    /// If the list is not empty, and `flags` does not contain
    /// [`SwapchainCreateFlags::MUTABLE_FORMAT`], then the list must contain at most one element,
    /// otherwise any number of elements are allowed.
    /// The view formats must be compatible with `format`.
    ///
    /// If `flags` contains [`SwapchainCreateFlags::MUTABLE_FORMAT`], then the list must contain
    /// `image_format.
    ///
    /// If this is not empty, then the device API version must be at least 1.2, or the
    /// [`khr_image_format_list`] extension must be enabled on the device.
    ///
    /// The default value is empty.
    ///
    /// [`khr_image_format_list`]: crate::device::DeviceExtensions::khr_image_format_list
    pub image_view_formats: &'a [Format],

    /// The color space of the created images.
    ///
    /// The default value is [`ColorSpace::SrgbNonLinear`].
    pub image_color_space: ColorSpace,

    /// The extent of the created images.
    ///
    /// Both values must be greater than zero. Note that on some platforms,
    /// [`SurfaceCapabilities::current_extent`] will be zero if the surface is minimized.
    /// Care must be taken to check for this, to avoid trying to create a zero-size swapchain.
    ///
    /// The default value is `[0, 0]`, which must be overridden.
    pub image_extent: [u32; 2],

    /// The number of array layers of the created images.
    ///
    /// The default value is `1`.
    pub image_array_layers: u32,

    /// How the created images will be used.
    ///
    /// The default value is [`ImageUsage::empty()`], which must be overridden.
    pub image_usage: ImageUsage,

    /// Whether the created images can be shared across multiple queues, or are limited to a single
    /// queue.
    ///
    /// The default value is [`Sharing::Exclusive`].
    pub image_sharing: Sharing<'a>,

    /// The transform that should be applied to an image before it is presented.
    ///
    /// The default value is [`SurfaceTransform::Identity`].
    pub pre_transform: SurfaceTransform,

    /// How alpha values of the pixels in the image are to be treated.
    ///
    /// The default value is [`CompositeAlpha::Opaque`].
    pub composite_alpha: CompositeAlpha,

    /// How the swapchain should behave when multiple images are waiting in the queue to be
    /// presented.
    ///
    /// The default value is [`PresentMode::Fifo`].
    pub present_mode: PresentMode,

    /// Alternative present modes that can be used with this swapchain. The mode specified in
    /// `present_mode` is the default mode, but can be changed for future present operations by
    /// specifying it when presenting.
    ///
    /// If this is not empty, then the
    /// [`ext_swapchain_maintenance1`](crate::device::DeviceExtensions::ext_swapchain_maintenance1)
    /// extension must be enabled on the device.
    /// It must always contain the mode specified in `present_mode`.
    ///
    /// The default value is empty.
    pub present_modes: &'a [PresentMode],

    /// Whether the implementation is allowed to discard rendering operations that affect regions
    /// of the surface which aren't visible. This is important to take into account if your
    /// fragment shader has side-effects or if you want to read back the content of the image
    /// afterwards.
    ///
    /// The default value is `true`.
    pub clipped: bool,

    /// The scaling method to use when the surface is not the same size as the swapchain image.
    ///
    /// `None` means the behavior is implementation defined.
    ///
    /// If this is `Some`, then the
    /// [`ext_swapchain_maintenance1`](crate::device::DeviceExtensions::ext_swapchain_maintenance1)
    /// extension must be enabled on the device.
    ///
    /// The default value is `None`.
    pub scaling_behavior: Option<PresentScaling>,

    /// The horizontal and vertical alignment to use when the swapchain image, after applying
    /// scaling, does not fill the whole surface.
    ///
    /// `None` means the behavior is implementation defined.
    ///
    /// If this is `Some`, then the
    /// [`ext_swapchain_maintenance1`](crate::device::DeviceExtensions::ext_swapchain_maintenance1)
    /// extension must be enabled on the device.
    ///
    /// The default value is `None`.
    pub present_gravity: Option<[PresentGravity; 2]>,

    /// How full-screen exclusivity is to be handled.
    ///
    /// If this is not [`FullScreenExclusive::Default`], then the
    /// [`ext_full_screen_exclusive`](crate::device::DeviceExtensions::ext_full_screen_exclusive)
    /// extension must be enabled on the device.
    ///
    /// If the surface is a Win32 surface, and this is
    /// [`FullScreenExclusive::ApplicationControlled`], then `win32_monitor` must be `Some`.
    ///
    /// The default value is [`FullScreenExclusive::Default`].
    pub full_screen_exclusive: FullScreenExclusive,

    /// For Win32 surfaces, this specifies the monitor on which full-screen exclusivity should be
    /// used.
    ///
    /// If this is `Some`, then `full_screen_exclusive` must not be
    /// [`FullScreenExclusive::Default`], and the surface must be a Win32 surface.
    ///
    /// The default value is `None`.
    pub win32_monitor: Option<Win32Monitor>,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for SwapchainCreateInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> SwapchainCreateInfo<'a> {
    /// Returns a default `SwapchainCreateInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            flags: SwapchainCreateFlags::empty(),
            min_image_count: 2,
            image_format: Format::UNDEFINED,
            image_view_formats: &[],
            image_color_space: ColorSpace::SrgbNonLinear,
            image_extent: [0, 0],
            image_array_layers: 1,
            image_usage: ImageUsage::empty(),
            image_sharing: Sharing::Exclusive,
            pre_transform: SurfaceTransform::Identity,
            composite_alpha: CompositeAlpha::Opaque,
            present_mode: PresentMode::Fifo,
            present_modes: &[],
            clipped: true,
            scaling_behavior: None,
            present_gravity: None,
            full_screen_exclusive: FullScreenExclusive::Default,
            win32_monitor: None,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            min_image_count: _,
            image_format,
            image_view_formats,
            image_color_space,
            image_extent,
            image_array_layers,
            image_usage,
            image_sharing,
            pre_transform,
            composite_alpha,
            present_mode,
            present_modes,
            clipped: _,
            scaling_behavior,
            present_gravity,
            full_screen_exclusive,
            win32_monitor,
            _ne: _,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkSwapchainCreateInfoKHR-flags-parameter"])
        })?;

        if flags.intersects(SwapchainCreateFlags::DEFERRED_MEMORY_ALLOCATION)
            && !device.enabled_features().swapchain_maintenance1
        {
            return Err(Box::new(ValidationError {
                context: "flags".into(),
                problem: "contains `SwapchainCreateFlags::DEFERRED_MEMORY_ALLOCATION`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "swapchain_maintenance1",
                )])]),
                vuids: &["VUID-VkSwapchainCreateInfoKHR-swapchainMaintenance1-10157"],
            }));
        }

        image_format.validate_device(device).map_err(|err| {
            err.add_context("image_format")
                .set_vuids(&["VUID-VkSwapchainCreateInfoKHR-imageFormat-parameter"])
        })?;

        image_color_space.validate_device(device).map_err(|err| {
            err.add_context("image_color_space")
                .set_vuids(&["VUID-VkSwapchainCreateInfoKHR-imageColorSpace-parameter"])
        })?;

        image_usage.validate_device(device).map_err(|err| {
            err.add_context("image_usage")
                .set_vuids(&["VUID-VkSwapchainCreateInfoKHR-imageUsage-parameter"])
        })?;

        if image_usage.is_empty() {
            return Err(Box::new(ValidationError {
                context: "image_usage".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkSwapchainCreateInfoKHR-imageUsage-requiredbitmask"],
                ..Default::default()
            }));
        }

        pre_transform.validate_device(device).map_err(|err| {
            err.add_context("pre_transform")
                .set_vuids(&["VUID-VkSwapchainCreateInfoKHR-preTransform-parameter"])
        })?;

        composite_alpha.validate_device(device).map_err(|err| {
            err.add_context("composite_alpha")
                .set_vuids(&["VUID-VkSwapchainCreateInfoKHR-compositeAlpha-parameter"])
        })?;

        present_mode.validate_device(device).map_err(|err| {
            err.add_context("present_mode")
                .set_vuids(&["VUID-VkSwapchainCreateInfoKHR-presentMode-parameter"])
        })?;

        if image_extent.contains(&0) {
            return Err(Box::new(ValidationError {
                context: "image_extent".into(),
                problem: "one or more elements are zero".into(),
                vuids: &["VUID-VkSwapchainCreateInfoKHR-imageExtent-01689"],
                ..Default::default()
            }));
        }

        if image_array_layers == 0 {
            return Err(Box::new(ValidationError {
                context: "image_array_layers".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkSwapchainCreateInfoKHR-imageArrayLayers-01275"],
                ..Default::default()
            }));
        }

        match image_sharing {
            Sharing::Exclusive => (),
            Sharing::Concurrent(queue_family_indices) => {
                if queue_family_indices.len() < 2 {
                    return Err(Box::new(ValidationError {
                        context: "image_sharing".into(),
                        problem: "is `Sharing::Concurrent`, and contains less than 2 \
                            queue family indices"
                            .into(),
                        vuids: &["VUID-VkSwapchainCreateInfoKHR-imageSharingMode-01278"],
                        ..Default::default()
                    }));
                }

                let queue_family_count =
                    device.physical_device().queue_family_properties().len() as u32;

                for (index, &queue_family_index) in queue_family_indices.iter().enumerate() {
                    if queue_family_indices[..index].contains(&queue_family_index) {
                        return Err(Box::new(ValidationError {
                            context: "queue_family_indices".into(),
                            problem: format!(
                                "the queue family index in the list at index {} is contained in \
                                the list more than once",
                                index,
                            )
                            .into(),
                            vuids: &["VUID-VkSwapchainCreateInfoKHR-imageSharingMode-01428"],
                            ..Default::default()
                        }));
                    }

                    if queue_family_index >= queue_family_count {
                        return Err(Box::new(ValidationError {
                            context: format!("queue_family_indices[{}]", index).into(),
                            problem: "is not less than the number of queue families in the \
                                physical device"
                                .into(),
                            vuids: &["VUID-VkSwapchainCreateInfoKHR-imageSharingMode-01428"],
                            ..Default::default()
                        }));
                    }
                }
            }
        };

        let image_format_properties = unsafe {
            device
                .physical_device()
                .image_format_properties_unchecked(&ImageFormatInfo {
                    format: image_format,
                    image_type: ImageType::Dim2d,
                    tiling: ImageTiling::Optimal,
                    usage: image_usage,
                    ..Default::default()
                })
        }
        .map_err(|_err| {
            Box::new(ValidationError {
                problem: "`PhysicalDevice::image_format_properties` \
                            returned an error"
                    .into(),
                ..Default::default()
            })
        })?;

        if image_format_properties.is_none() {
            return Err(Box::new(ValidationError {
                problem: "the combination of `image_format` and `image_usage` is not supported \
                    for images by the physical device"
                    .into(),
                vuids: &["VUID-VkSwapchainCreateInfoKHR-imageFormat-01778"],
                ..Default::default()
            }));
        }

        if flags.intersects(SwapchainCreateFlags::MUTABLE_FORMAT)
            && !image_view_formats.contains(&image_format)
        {
            return Err(Box::new(ValidationError {
                problem: "`flags` contains `SwapchainCreateFlags::MUTABLE_FORMAT`, but \
                    `image_view_formats` does not contain `image_format`"
                    .into(),
                vuids: &["VUID-VkSwapchainCreateInfoKHR-flags-03168"],
                ..Default::default()
            }));
        }

        if !image_view_formats.is_empty() {
            if !(device.api_version() >= Version::V1_2
                || device.enabled_extensions().khr_image_format_list)
            {
                return Err(Box::new(ValidationError {
                    context: "image_view_formats".into(),
                    problem: "is not empty".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_2)]),
                        RequiresAllOf(&[Requires::DeviceExtension("khr_image_format_list")]),
                    ]),
                    ..Default::default()
                }));
            }

            if !flags.intersects(SwapchainCreateFlags::MUTABLE_FORMAT)
                && image_view_formats.len() != 1
            {
                return Err(Box::new(ValidationError {
                    problem: "`flags` does not contain `SwapchainCreateFlags::MUTABLE_FORMAT`, \
                        but `image_view_formats` contains more than one element"
                        .into(),
                    vuids: &["VUID-VkSwapchainCreateInfoKHR-flags-04100"],
                    ..Default::default()
                }));
            }

            for (index, &image_view_format) in image_view_formats.iter().enumerate() {
                image_view_format.validate_device(device).map_err(|err| {
                    err.add_context(format!("image_view_formats[{}]", index))
                        .set_vuids(&["VUID-VkImageFormatListCreateInfo-pViewFormats-parameter"])
                })?;

                if image_view_format.compatibility() != image_format.compatibility() {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`image_view_formats[{}]` is not compatible with `image_format`",
                            index
                        )
                        .into(),
                        vuids: &["VUID-VkSwapchainCreateInfoKHR-pNext-04099"],
                        ..Default::default()
                    }));
                }
            }
        }

        if !present_modes.is_empty() {
            if !device.enabled_extensions().ext_swapchain_maintenance1 {
                return Err(Box::new(ValidationError {
                    context: "present_modes".into(),
                    problem: "is not empty".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                        "ext_swapchain_maintenance1",
                    )])]),
                    ..Default::default()
                }));
            }

            for (index, &present_mode) in present_modes.iter().enumerate() {
                present_mode.validate_device(device).map_err(|err| {
                    err.add_context(format!("present_modes[{}]", index))
                        .set_vuids(&[
                            "VUID-VkSwapchainPresentModesCreateInfoEXT-pPresentModes-parameter",
                        ])
                })?;
            }

            if !present_modes.contains(&present_mode) {
                return Err(Box::new(ValidationError {
                    problem: "`present_modes` is not empty, but does not contain `present_mode`"
                        .into(),
                    vuids: &["VUID-VkSwapchainPresentModesCreateInfoEXT-presentMode-07764"],
                    ..Default::default()
                }));
            }
        }

        if let Some(scaling_behavior) = scaling_behavior {
            if !device.enabled_extensions().ext_swapchain_maintenance1 {
                return Err(Box::new(ValidationError {
                    context: "scaling_behavior".into(),
                    problem: "is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                        "ext_swapchain_maintenance1",
                    )])]),
                    ..Default::default()
                }));
            }

            scaling_behavior.validate_device(device).map_err(|err| {
                err.add_context("scaling_behavior").set_vuids(&[
                    "VUID-VkSwapchainPresentScalingCreateInfoEXT-scalingBehavior-parameter",
                ])
            })?;

            // VUID-VkSwapchainPresentScalingCreateInfoEXT-scalingBehavior-07767
            // Ensured by the use of an enum.
        }

        if let Some(present_gravity) = present_gravity {
            if !device.enabled_extensions().ext_swapchain_maintenance1 {
                return Err(Box::new(ValidationError {
                    context: "present_gravity".into(),
                    problem: "is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                        "ext_swapchain_maintenance1",
                    )])]),
                    ..Default::default()
                }));
            }

            for (axis_index, present_gravity) in present_gravity.into_iter().enumerate() {
                present_gravity.validate_device(device).map_err(|err| {
                    err.add_context(format!("present_gravity[{}]", axis_index))
                        .set_vuids(&[
                            "VUID-VkSwapchainPresentScalingCreateInfoEXT-presentGravityX-parameter",
                            "VUID-VkSwapchainPresentScalingCreateInfoEXT-presentGravityY-parameter",
                        ])
                })?;
            }

            // VUID-VkSwapchainPresentScalingCreateInfoEXT-presentGravityX-07765
            // VUID-VkSwapchainPresentScalingCreateInfoEXT-presentGravityX-07766
            // Ensured by the use of an array of enums wrapped in `Option`.

            // VUID-VkSwapchainPresentScalingCreateInfoEXT-presentGravityX-07768
            // VUID-VkSwapchainPresentScalingCreateInfoEXT-presentGravityY-07769
            // Ensured by the use of an enum.
        }

        if full_screen_exclusive != FullScreenExclusive::Default {
            if !device.enabled_extensions().ext_full_screen_exclusive {
                return Err(Box::new(ValidationError {
                    context: "full_screen_exclusive".into(),
                    problem: "is not `FullScreenExclusive::Default`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                        "ext_full_screen_exclusive",
                    )])]),
                    ..Default::default()
                }));
            }

            full_screen_exclusive
                .validate_device(device)
                .map_err(|err| {
                    err.add_context("full_screen_exclusive").set_vuids(&[
                        "VUID-VkSurfaceFullScreenExclusiveInfoEXT-fullScreenExclusive-parameter",
                    ])
                })?;
        } else if win32_monitor.is_some() {
            return Err(Box::new(ValidationError {
                problem: "`full_screen_exclusive` is `FullScreenExclusive::Default`, but \
                    `win32_monitor` is `Some`"
                    .into(),
                ..Default::default()
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk(
        &'a self,
        surface_vk: vk::SurfaceKHR,
        old_swapchain_vk: vk::SwapchainKHR,
        extensions_vk: &'a mut SwapchainCreateInfoExtensionsVk<'_>,
    ) -> vk::SwapchainCreateInfoKHR<'a> {
        let &Self {
            flags,
            min_image_count,
            image_format,
            image_view_formats: _,
            image_color_space,
            image_extent,
            image_array_layers,
            image_usage,
            image_sharing,
            pre_transform,
            composite_alpha,
            present_mode,
            present_modes: _,
            clipped,
            scaling_behavior: _,
            present_gravity: _,
            full_screen_exclusive: _,
            win32_monitor: _,
            _ne: _,
        } = self;

        let (image_sharing_mode_vk, queue_family_indices_vk) = match image_sharing {
            Sharing::Exclusive => (vk::SharingMode::EXCLUSIVE, [].as_slice()),
            Sharing::Concurrent(ids) => (vk::SharingMode::CONCURRENT, ids),
        };

        let mut val_vk = vk::SwapchainCreateInfoKHR::default()
            .flags(flags.into())
            .surface(surface_vk)
            .min_image_count(min_image_count)
            .image_format(image_format.into())
            .image_color_space(image_color_space.into())
            .image_extent(vk::Extent2D {
                width: image_extent[0],
                height: image_extent[1],
            })
            .image_array_layers(image_array_layers)
            .image_usage(image_usage.into())
            .image_sharing_mode(image_sharing_mode_vk)
            .queue_family_indices(queue_family_indices_vk)
            .pre_transform(pre_transform.into())
            .composite_alpha(composite_alpha.into())
            .present_mode(present_mode.into())
            .clipped(clipped)
            .old_swapchain(old_swapchain_vk);

        let SwapchainCreateInfoExtensionsVk {
            full_screen_exclusive_vk,
            full_screen_exclusive_win32_vk,
            image_format_list_vk,
            present_modes_vk,
            present_scaling_vk,
        } = extensions_vk;

        if let Some(next) = full_screen_exclusive_vk {
            val_vk = val_vk.push(next);
        }

        if let Some(next) = full_screen_exclusive_win32_vk {
            val_vk = val_vk.push(next);
        }

        if let Some(next) = image_format_list_vk {
            val_vk = val_vk.push(next);
        }

        if let Some(next) = present_modes_vk {
            val_vk = val_vk.push(next);
        }

        if let Some(next) = present_scaling_vk {
            val_vk = val_vk.push(next);
        }

        val_vk
    }

    pub(crate) fn to_vk_extensions(
        &self,
        fields1_vk: &'a SwapchainCreateInfoFields1Vk,
    ) -> SwapchainCreateInfoExtensionsVk<'a> {
        let &Self {
            flags: _,
            min_image_count: _,
            image_format: _,
            image_view_formats: _,
            image_color_space: _,
            image_extent: _,
            image_array_layers: _,
            image_usage: _,
            image_sharing: _,
            pre_transform: _,
            composite_alpha: _,
            present_mode: _,
            present_modes: _,
            clipped: _,
            scaling_behavior,
            present_gravity,
            full_screen_exclusive,
            win32_monitor,
            _ne: _,
        } = self;
        let SwapchainCreateInfoFields1Vk {
            present_modes_vk,
            view_formats_vk,
        } = fields1_vk;

        let full_screen_exclusive_vk = (full_screen_exclusive != FullScreenExclusive::Default)
            .then(|| {
                vk::SurfaceFullScreenExclusiveInfoEXT::default()
                    .full_screen_exclusive(full_screen_exclusive.into())
            });

        let full_screen_exclusive_win32_vk = win32_monitor.map(|win32_monitor| {
            vk::SurfaceFullScreenExclusiveWin32InfoEXT::default().hmonitor(win32_monitor.0)
        });

        let image_format_list_vk = (!view_formats_vk.is_empty())
            .then(|| vk::ImageFormatListCreateInfo::default().view_formats(view_formats_vk));

        let present_modes_vk = (!present_modes_vk.is_empty()).then(|| {
            vk::SwapchainPresentModesCreateInfoEXT::default().present_modes(present_modes_vk)
        });

        let present_scaling_vk =
            (scaling_behavior.is_some() || present_gravity.is_some()).then(|| {
                let [present_gravity_x, present_gravity_y] =
                    present_gravity.map_or_else(Default::default, |pg| pg.map(Into::into));
                vk::SwapchainPresentScalingCreateInfoEXT::default()
                    .scaling_behavior(scaling_behavior.map_or_else(Default::default, Into::into))
                    .present_gravity_x(present_gravity_x)
                    .present_gravity_y(present_gravity_y)
            });

        SwapchainCreateInfoExtensionsVk {
            full_screen_exclusive_vk,
            full_screen_exclusive_win32_vk,
            image_format_list_vk,
            present_modes_vk,
            present_scaling_vk,
        }
    }

    pub(crate) fn to_vk_fields1(&self) -> SwapchainCreateInfoFields1Vk {
        let present_modes_vk = self.present_modes.iter().copied().map(Into::into).collect();
        let view_formats_vk = self
            .image_view_formats
            .iter()
            .copied()
            .map(vk::Format::from)
            .collect();

        SwapchainCreateInfoFields1Vk {
            present_modes_vk,
            view_formats_vk,
        }
    }
}

pub(crate) struct SwapchainCreateInfoExtensionsVk<'a> {
    pub(crate) full_screen_exclusive_vk: Option<vk::SurfaceFullScreenExclusiveInfoEXT<'static>>,
    pub(crate) full_screen_exclusive_win32_vk:
        Option<vk::SurfaceFullScreenExclusiveWin32InfoEXT<'static>>,
    pub(crate) image_format_list_vk: Option<vk::ImageFormatListCreateInfo<'a>>,
    pub(crate) present_modes_vk: Option<vk::SwapchainPresentModesCreateInfoEXT<'a>>,
    pub(crate) present_scaling_vk: Option<vk::SwapchainPresentScalingCreateInfoEXT<'static>>,
}

pub(crate) struct SwapchainCreateInfoFields1Vk {
    pub(crate) present_modes_vk: SmallVec<[vk::PresentModeKHR; PresentMode::COUNT]>,
    pub(crate) view_formats_vk: Vec<vk::Format>,
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags specifying additional properties of a swapchain.
    SwapchainCreateFlags = SwapchainCreateFlagsKHR(u32);

    /* TODO: enable
    /// Creates swapchain images with the [`ImageCreateFlags::SPLIT_INSTANCE_BIND_REGIONS`] flag.
    SPLIT_INSTANCE_BIND_REGIONS = SPLIT_INSTANCE_BIND_REGIONS
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_device_group)]),
    ]),*/

    /* TODO: enable
    /// Creates swapchain images with the [`ImageCreateFlags::PROTECTED`] flag.
    PROTECTED = PROTECTED
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
    ]),*/

    /// Creates swapchain images with both the [`ImageCreateFlags::MUTABLE_FORMAT`] and
    /// [`ImageCreateFlags::EXTENDED_USAGE`] flags.
    MUTABLE_FORMAT = MUTABLE_FORMAT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_swapchain_mutable_format)]),
    ]),

    /// Allows the implementation to defer the allocation of a swapchain image until its image
    /// index is acquired for the first time.
    DEFERRED_MEMORY_ALLOCATION = DEFERRED_MEMORY_ALLOCATION_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_swapchain_maintenance1)]),
    ]),
}

impl From<SwapchainCreateFlags> for ImageCreateFlags {
    #[inline]
    fn from(flags: SwapchainCreateFlags) -> Self {
        // Per https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCreateSwapchainKHR.html#_description
        let mut result = ImageCreateFlags::empty();

        /* TODO: enable
        if flags.intersects(SwapchainCreateFlags::SPLIT_INSTANCE_BIND_REGIONS) {
            result |= ImageCreateFlags::SPLIT_INSTANCE_BIND_REGIONS;
        } */

        /* TODO: enable
        if flags.intersects(SwapchainCreateFlags::PROTECTED) {
            result |= ImageCreateFlags::PROTECTED;
        } */

        if flags.intersects(SwapchainCreateFlags::MUTABLE_FORMAT) {
            result |= ImageCreateFlags::MUTABLE_FORMAT | ImageCreateFlags::EXTENDED_USAGE;
        }

        result
    }
}

vulkan_bitflags_enum! {
    #[non_exhaustive]

    /// A set of [`PresentScaling`] values.
    PresentScalingFlags,

    /// The way a swapchain image is scaled, if it does not exactly fit the surface.
    PresentScaling,

    = PresentScalingFlagsEXT(u32);

    /// No scaling is performed; one swapchain image pixel maps to one surface pixel.
    ONE_TO_ONE, OneToOne = ONE_TO_ONE,

    /// Both axes of the image are scaled equally, without changing the aspect ratio of the image,
    /// to the largest size in which both axes fit inside the surface.
    ASPECT_RATIO_STRETCH, AspectRatioStretch = ASPECT_RATIO_STRETCH,

    /// Each axis of the image is scaled independently to fit the surface,
    /// which may change the aspect ratio of the image.
    STRETCH, Stretch = STRETCH,
}

vulkan_bitflags_enum! {
    #[non_exhaustive]

    /// A set of [`PresentGravity`] values.
    PresentGravityFlags,

    /// The way a swapchain image is aligned, if it does not exactly fit the surface.
    PresentGravity,

    = PresentGravityFlagsEXT(u32);

    /// Aligned to the top or left side of the surface.
    MIN, Min = MIN,

    /// Aligned to the bottom or right side of the surface.
    MAX, Max = MAX,

    /// Aligned to the middle of the surface.
    CENTERED, Centered = CENTERED,
}

vulkan_enum! {
    #[non_exhaustive]

    /// The way full-screen exclusivity is handled.
    FullScreenExclusive = FullScreenExclusiveEXT(i32);

    /// Indicates that the driver should determine the appropriate full-screen method
    /// by whatever means it deems appropriate.
    Default = DEFAULT,

    /// Indicates that the driver may use full-screen exclusive mechanisms when available.
    /// Such mechanisms may result in better performance and/or the availability of
    /// different presentation capabilities, but may require a more disruptive transition
    // during swapchain initialization, first presentation and/or destruction.
    Allowed = ALLOWED,

    /// Indicates that the driver should avoid using full-screen mechanisms which rely
    /// on disruptive transitions.
    Disallowed = DISALLOWED,

    /// Indicates the application will manage full-screen exclusive mode by using the
    /// [`Swapchain::acquire_full_screen_exclusive_mode`] and
    /// [`Swapchain::release_full_screen_exclusive_mode`] functions.
    ApplicationControlled = APPLICATION_CONTROLLED,
}

/// A wrapper around a Win32 monitor handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Win32Monitor(pub(crate) vk::HMONITOR);

impl Win32Monitor {
    /// Wraps a Win32 monitor handle.
    ///
    /// # Safety
    ///
    /// - `hmonitor` must be a valid handle as returned by the Win32 API.
    pub unsafe fn new(hmonitor: vk::HMONITOR) -> Self {
        Self(hmonitor)
    }
}

// Winit's `MonitorHandle` is Send on Win32, so this seems safe.
unsafe impl Send for Win32Monitor {}
unsafe impl Sync for Win32Monitor {}

/// Parameters to release swapchain images.
#[derive(Clone, Debug)]
pub struct ReleaseSwapchainImagesInfo<'a> {
    /// The indices of currently acquired swapchain images that should be released.
    ///
    /// The default value is empty.
    pub image_indices: &'a [u32],

    pub _ne: NonExhaustive<'a>,
}

impl Default for ReleaseSwapchainImagesInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self {
            image_indices: &[],
            _ne: crate::NE,
        }
    }
}

impl<'a> ReleaseSwapchainImagesInfo<'a> {
    pub(crate) fn to_vk(
        &self,
        swapchain_vk: vk::SwapchainKHR,
    ) -> vk::ReleaseSwapchainImagesInfoEXT<'a> {
        vk::ReleaseSwapchainImagesInfoEXT::default()
            .swapchain(swapchain_vk)
            .image_indices(self.image_indices)
    }
}
