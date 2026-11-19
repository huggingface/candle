use super::{FullScreenExclusive, PresentGravityFlags, PresentScalingFlags, Win32Monitor};
#[cfg(feature = "raw_window_handle")]
use crate::instance::InstanceExtensions;
use crate::{
    cache::OnceCache,
    device::physical::PhysicalDevice,
    display::{DisplayMode, DisplayPlaneAlpha},
    format::Format,
    image::ImageUsage,
    instance::{Instance, InstanceOwned},
    macros::{impl_id_counter, vulkan_bitflags_enum, vulkan_enum},
    DebugWrapper, Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, VulkanError,
    VulkanObject,
};
use ash::vk;
use ash::vk::TaggedStructure;

#[cfg(feature = "raw_window_handle")]
use raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, RawDisplayHandle,
    RawWindowHandle, WindowHandle,
};
use smallvec::SmallVec;
use std::{
    any::Any,
    ffi::c_void,
    fmt::{Debug, Error as FmtError, Formatter},
    marker::PhantomData,
    mem::MaybeUninit,
    num::NonZero,
    ptr,
    sync::Arc,
};
#[cfg(feature = "raw_window_handle")]
use std::{error::Error, fmt::Display};

/// Represents a surface on the screen.
///
/// Creating a `Surface` is platform-specific.
pub struct Surface {
    handle: vk::SurfaceKHR,
    instance: DebugWrapper<Arc<Instance>>,
    id: NonZero<u64>,
    api: SurfaceApi,
    object: Option<SurfaceObject>,
    // Data queried by the user at runtime, cached for faster lookups.
    // This is stored here rather than on `PhysicalDevice` to ensure that it's freed when the
    // `Surface` is destroyed.
    pub(crate) surface_formats:
        OnceCache<(vk::PhysicalDevice, SurfaceInfo<'static>), Vec<(Format, ColorSpace)>>,
    pub(crate) surface_present_modes:
        OnceCache<(vk::PhysicalDevice, SurfaceInfo<'static>), Vec<PresentMode>>,
    pub(crate) surface_support: OnceCache<(vk::PhysicalDevice, u32), bool>,
}

/// The `window` or `object` parameter that is passed when creating a [`Surface`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum SurfaceObject {
    #[cfg(feature = "raw_window_handle")]
    Window(Arc<dyn Window>),
    Other(Arc<dyn Any + Send + Sync>),
}

impl Surface {
    /// Returns the instance extensions required to create a surface from a window of the given
    /// event loop, panicking on a handle error.
    ///
    /// This is a shortcut for `try_required_extensions().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_required_extensions`] returns a [`HandleError`].
    ///
    /// [`try_required_extensions`]: Self::try_required_extensions
    #[cfg(feature = "raw_window_handle")]
    #[track_caller]
    pub fn required_extensions(event_loop: &impl HasDisplayHandle) -> InstanceExtensions {
        Self::required_extensions_inner(event_loop.display_handle())
    }

    #[cfg(feature = "raw_window_handle")]
    #[track_caller]
    fn required_extensions_inner(
        display_handle: Result<DisplayHandle<'_>, HandleError>,
    ) -> InstanceExtensions {
        Self::try_required_extensions_inner(display_handle).unwrap()
    }

    /// Returns the instance extensions required to create a surface from a window of the given
    /// event loop.
    #[cfg(feature = "raw_window_handle")]
    pub fn try_required_extensions(
        event_loop: &impl HasDisplayHandle,
    ) -> Result<InstanceExtensions, HandleError> {
        Self::try_required_extensions_inner(event_loop.display_handle())
    }

    #[cfg(feature = "raw_window_handle")]
    fn try_required_extensions_inner(
        display_handle: Result<DisplayHandle<'_>, HandleError>,
    ) -> Result<InstanceExtensions, HandleError> {
        let mut extensions = InstanceExtensions {
            khr_surface: true,
            ..InstanceExtensions::empty()
        };
        match display_handle?.as_raw() {
            RawDisplayHandle::Android(_) => extensions.khr_android_surface = true,
            RawDisplayHandle::AppKit(_) => extensions.ext_metal_surface = true,
            RawDisplayHandle::UiKit(_) => extensions.ext_metal_surface = true,
            RawDisplayHandle::Windows(_) => extensions.khr_win32_surface = true,
            RawDisplayHandle::Wayland(_) => extensions.khr_wayland_surface = true,
            RawDisplayHandle::Xcb(_) => extensions.khr_xcb_surface = true,
            RawDisplayHandle::Xlib(_) => extensions.khr_xlib_surface = true,
            _ => unimplemented!(),
        }

        Ok(extensions)
    }

    /// Creates a new `Surface` from the given `window`, panicking on a validation error or handle
    /// error.
    ///
    /// This is a shortcut for
    /// `try_from_window().map_err(Validated::unwrap).map_err(FromWindowError::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_from_window`] returns a [`ValidationError`] or [`HandleError`].
    ///
    /// [`try_from_window`]: Self::try_from_window
    #[cfg(feature = "raw_window_handle")]
    #[track_caller]
    pub fn from_window(
        instance: &Arc<Instance>,
        window: &Arc<impl Window + ?Sized>,
    ) -> Result<Arc<Self>, VulkanError> {
        Self::from_window_inner(
            instance,
            window.clone().as_dyn(),
            window.window_handle(),
            window.display_handle(),
        )
    }

    #[cfg(feature = "raw_window_handle")]
    #[track_caller]
    fn from_window_inner(
        instance: &Arc<Instance>,
        window: Arc<dyn Window>,
        window_handle: Result<WindowHandle<'_>, HandleError>,
        display_handle: Result<DisplayHandle<'_>, HandleError>,
    ) -> Result<Arc<Self>, VulkanError> {
        match Self::try_from_window_inner(instance, window, window_handle, display_handle) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap().unwrap()),
        }
    }

    /// Creates a new `Surface` from the given `window`.
    #[cfg(feature = "raw_window_handle")]
    pub fn try_from_window(
        instance: &Arc<Instance>,
        window: &Arc<impl Window + ?Sized>,
    ) -> Result<Arc<Self>, Validated<FromWindowError>> {
        let window_handle = window.window_handle();
        let display_handle = window.display_handle();

        Self::try_from_window_inner(
            instance,
            window.clone().as_dyn(),
            window_handle,
            display_handle,
        )
    }

    #[cfg(feature = "raw_window_handle")]
    fn try_from_window_inner(
        instance: &Arc<Instance>,
        window: Arc<dyn Window>,
        window_handle: Result<WindowHandle<'_>, HandleError>,
        display_handle: Result<DisplayHandle<'_>, HandleError>,
    ) -> Result<Arc<Self>, Validated<FromWindowError>> {
        let mut surface =
            unsafe { Self::try_from_window_ref_inner(instance, window_handle, display_handle) }?;
        Arc::get_mut(&mut surface).unwrap().object = Some(SurfaceObject::Window(window));

        Ok(surface)
    }

    /// Creates a new `Surface` from the given `window` without ensuring that the window outlives
    /// the surface, panicking on a validation error or handle error.
    ///
    /// This is a shortcut for
    /// `try_from_window_ref().map_err(Validated::unwrap).map_err(FromHandleError::unwrap)`.
    ///
    /// # Safety
    ///
    /// - The given `window` must outlive the created surface.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_from_window_ref`] returns a [`ValidationError`] or [`HandleError`].
    ///
    /// [`try_from_window_ref`]: Self::try_from_window_ref
    #[cfg(feature = "raw_window_handle")]
    #[track_caller]
    pub unsafe fn from_window_ref(
        instance: &Arc<Instance>,
        window: &(impl HasWindowHandle + HasDisplayHandle),
    ) -> Result<Arc<Self>, VulkanError> {
        let window_handle = window.window_handle();
        let display_handle = window.display_handle();

        unsafe { Self::from_window_ref_inner(instance, window_handle, display_handle) }
    }

    #[cfg(feature = "raw_window_handle")]
    #[track_caller]
    unsafe fn from_window_ref_inner(
        instance: &Arc<Instance>,
        window_handle: Result<WindowHandle<'_>, HandleError>,
        display_handle: Result<DisplayHandle<'_>, HandleError>,
    ) -> Result<Arc<Self>, VulkanError> {
        match unsafe { Self::try_from_window_ref_inner(instance, window_handle, display_handle) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap().unwrap()),
        }
    }

    /// Creates a new `Surface` from the given `window` without ensuring that the window outlives
    /// the surface.
    ///
    /// # Safety
    ///
    /// - The given `window` must outlive the created surface.
    #[cfg(feature = "raw_window_handle")]
    pub unsafe fn try_from_window_ref(
        instance: &Arc<Instance>,
        window: &(impl HasWindowHandle + HasDisplayHandle),
    ) -> Result<Arc<Self>, Validated<FromWindowError>> {
        let window_handle = window.window_handle();
        let display_handle = window.display_handle();

        unsafe { Self::try_from_window_ref_inner(instance, window_handle, display_handle) }
    }

    #[cfg(feature = "raw_window_handle")]
    unsafe fn try_from_window_ref_inner(
        instance: &Arc<Instance>,
        window_handle: Result<WindowHandle<'_>, HandleError>,
        display_handle: Result<DisplayHandle<'_>, HandleError>,
    ) -> Result<Arc<Self>, Validated<FromWindowError>> {
        let window_handle = window_handle
            .map_err(FromWindowError::RetrieveHandle)
            .map_err(Validated::Error)?;
        let display_handle = display_handle
            .map_err(FromWindowError::RetrieveHandle)
            .map_err(Validated::Error)?;

        match (window_handle.as_raw(), display_handle.as_raw()) {
            (RawWindowHandle::AndroidNdk(window), RawDisplayHandle::Android(_display)) => unsafe {
                Self::try_from_android(instance, window.a_native_window.as_ptr().cast(), None)
            },
            #[cfg(target_vendor = "apple")]
            (RawWindowHandle::AppKit(handle), _) => {
                let layer = unsafe { raw_window_metal::Layer::from_ns_view(handle.ns_view) };

                // Vulkan retains the CAMetalLayer, so no need to retain it past this invocation
                unsafe { Self::try_from_metal(instance, layer.as_ptr().as_ptr(), None) }
            }
            #[cfg(target_vendor = "apple")]
            (RawWindowHandle::UiKit(handle), _) => {
                let layer = unsafe { raw_window_metal::Layer::from_ui_view(handle.ui_view) };

                // Vulkan retains the CAMetalLayer, so no need to retain it past this invocation
                unsafe { Self::try_from_metal(instance, layer.as_ptr().as_ptr(), None) }
            }
            (RawWindowHandle::Wayland(window), RawDisplayHandle::Wayland(display)) => unsafe {
                Self::try_from_wayland(
                    instance,
                    display.display.as_ptr().cast(),
                    window.surface.as_ptr().cast(),
                    None,
                )
            },
            (RawWindowHandle::Win32(window), RawDisplayHandle::Windows(_display)) => unsafe {
                Self::try_from_win32(
                    instance,
                    window.hinstance.unwrap().get() as vk::HINSTANCE,
                    window.hwnd.get() as vk::HWND,
                    None,
                )
            },
            (RawWindowHandle::Xcb(window), RawDisplayHandle::Xcb(display)) => unsafe {
                Self::try_from_xcb(
                    instance,
                    display.connection.unwrap().as_ptr().cast(),
                    window.window.get() as vk::xcb_window_t,
                    None,
                )
            },
            (RawWindowHandle::Xlib(window), RawDisplayHandle::Xlib(display)) => unsafe {
                Self::try_from_xlib(
                    instance,
                    display.display.unwrap().as_ptr().cast(),
                    window.window as vk::Window,
                    None,
                )
            },
            _ => unimplemented!(
                "the window was created with a windowing API that is not supported by \
                Vulkan/vulkano",
            ),
        }
        .map_err(|err| err.map(FromWindowError::CreateSurface))
    }

    /// Creates a `Surface` from a raw handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `instance`.
    /// - `handle` must have been created using the function specified by `api`.
    /// - The window object that `handle` was created from must outlive the created `Surface`. The
    ///   `object` parameter can be used to ensure this.
    pub unsafe fn from_handle(
        instance: &Arc<Instance>,
        handle: vk::SurfaceKHR,
        api: SurfaceApi,
        object: Option<SurfaceObject>,
    ) -> Self {
        Surface {
            handle,
            instance: DebugWrapper(instance.clone()),
            id: Self::next_id(),
            api,
            object,
            surface_formats: OnceCache::new(),
            surface_present_modes: OnceCache::new(),
            surface_support: OnceCache::new(),
        }
    }

    /// Creates a `Surface` with no backing window or display, panicking on a validation error.
    ///
    /// Presenting to a headless surface does nothing, so this is mostly useless in itself.
    /// However, it may be useful for testing, and it is available for future extensions to layer
    /// on top of.
    ///
    /// This is a shortcut for `try_headless().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_headless`] returns a [`ValidationError`].
    ///
    /// [`try_headless`]: Self::try_headless
    #[track_caller]
    pub fn headless(
        instance: &Arc<Instance>,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        match Self::try_headless(instance, object) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a `Surface` with no backing window or display.
    ///
    /// Presenting to a headless surface does nothing, so this is mostly useless in itself.
    /// However, it may be useful for testing, and it is available for future extensions to layer
    /// on top of.
    pub fn try_headless(
        instance: &Arc<Instance>,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_headless(instance)?;

        Ok(unsafe { Self::headless_unchecked(instance, object) }?)
    }

    fn validate_headless(instance: &Instance) -> Result<(), Box<ValidationError>> {
        if !instance.enabled_extensions().ext_headless_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "ext_headless_surface",
                )])]),
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn headless_unchecked(
        instance: &Arc<Instance>,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        let create_info_vk = vk::HeadlessSurfaceCreateInfoEXT::default()
            .flags(vk::HeadlessSurfaceCreateFlagsEXT::empty());

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.ext_headless_surface.create_headless_surface_ext)(
                    instance.handle(),
                    &create_info_vk,
                    ptr::null(),
                    output.as_mut_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;
            unsafe { output.assume_init() }
        };

        Ok(Arc::new(unsafe {
            Self::from_handle(
                instance,
                handle,
                SurfaceApi::Headless,
                object.map(SurfaceObject::Other),
            )
        }))
    }

    /// Creates a `Surface` from a `DisplayMode` and display plane, panicking on a validation
    /// error.
    ///
    /// This is a shortcut for `try_from_display_plane().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_from_display_plane`] returns a [`ValidationError`].
    ///
    /// [`try_from_display_plane`]: Self::try_from_display_plane
    #[inline]
    #[track_caller]
    pub fn from_display_plane(
        display_mode: &Arc<DisplayMode>,
        create_info: &DisplaySurfaceCreateInfo<'_>,
    ) -> Result<Arc<Self>, VulkanError> {
        match Self::try_from_display_plane(display_mode, create_info) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a `Surface` from a `DisplayMode` and display plane.
    #[inline]
    pub fn try_from_display_plane(
        display_mode: &Arc<DisplayMode>,
        create_info: &DisplaySurfaceCreateInfo<'_>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_from_display_plane(display_mode, create_info)?;

        Ok(unsafe { Self::from_display_plane_unchecked(display_mode, create_info) }?)
    }

    fn validate_from_display_plane(
        display_mode: &DisplayMode,
        create_info: &DisplaySurfaceCreateInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        if !display_mode.instance().enabled_extensions().khr_display {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "khr_display",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-vkCreateDisplayPlaneSurfaceKHR-pCreateInfo-parameter
        create_info
            .validate(display_mode.display().physical_device())
            .map_err(|err| err.add_context("create_info"))?;

        let &DisplaySurfaceCreateInfo {
            plane_index,
            plane_stack_index,
            transform,
            alpha_mode,
            global_alpha: _,
            image_extent: _,
            _ne: _,
        } = create_info;

        let display_plane_properties_raw = unsafe {
            display_mode
                .display()
                .physical_device()
                .display_plane_properties_raw()
        }
        .map_err(|_err| {
            Box::new(ValidationError {
                problem: "`PhysicalDevice::display_plane_properties` \
                            returned an error"
                    .into(),
                ..Default::default()
            })
        })?;

        if display_mode.display().plane_reorder_possible() {
            if plane_stack_index as usize >= display_plane_properties_raw.len() {
                return Err(Box::new(ValidationError {
                    problem: "`plane_stack_index` is not less than the number of display planes \
                        on the physical device"
                        .into(),
                    vuids: &["VUID-VkDisplaySurfaceCreateInfoKHR-planeReorderPossible-01253"],
                    ..Default::default()
                }));
            }
        } else {
            if plane_stack_index
                != display_plane_properties_raw[plane_index as usize].current_stack_index
            {
                return Err(Box::new(ValidationError {
                    problem: "`display_mode.display().plane_reorder_possible()` is `false`, and \
                        `plane_stack_index` does not equal the `current_stack_index` value of the \
                        display plane properties"
                        .into(),
                    vuids: &["VUID-VkDisplaySurfaceCreateInfoKHR-planeReorderPossible-01253"],
                    ..Default::default()
                }));
            }
        }

        let display_plane_capabilities =
            unsafe { display_mode.display_plane_capabilities_unchecked(plane_index) }.map_err(
                |_err| {
                    Box::new(ValidationError {
                        problem: "`DisplayMode::display_plane_capabilities` \
                            returned an error"
                            .into(),
                        ..Default::default()
                    })
                },
            )?;

        if !display_plane_capabilities
            .supported_alpha
            .contains_enum(alpha_mode)
        {
            return Err(Box::new(ValidationError {
                problem: "the `supported_alpha` value of the display plane capabilities of \
                    `display_mode` for the provided plane index does not contain \
                    `create_info.alpha_mode`"
                    .into(),
                vuids: &["VUID-VkDisplaySurfaceCreateInfoKHR-alphaMode-01255"],
                ..Default::default()
            }));
        }

        if !display_mode
            .display()
            .supported_transforms()
            .contains_enum(transform)
        {
            return Err(Box::new(ValidationError {
                problem: "`display_mode.display().supported_transforms()` does not contain \
                    `create_info.transform`"
                    .into(),
                vuids: &["VUID-VkDisplaySurfaceCreateInfoKHR-transform-06740"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn from_display_plane_unchecked(
        display_mode: &Arc<DisplayMode>,
        create_info: &DisplaySurfaceCreateInfo<'_>,
    ) -> Result<Arc<Self>, VulkanError> {
        let create_info_vk = create_info.to_vk(display_mode.handle());
        let instance = display_mode.instance();

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.khr_display.create_display_plane_surface_khr)(
                    instance.handle(),
                    &create_info_vk,
                    ptr::null(),
                    output.as_mut_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;
            unsafe { output.assume_init() }
        };

        Ok(Arc::new(unsafe {
            Self::from_handle(instance, handle, SurfaceApi::DisplayPlane, None)
        }))
    }

    /// Creates a `Surface` from an Android window, panicking on a validation error.
    ///
    /// This is a shortcut for `try_from_android().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - `window` must be a valid Android `ANativeWindow` handle.
    /// - The object referred to by `window` must outlive the created `Surface`. The `object`
    ///   parameter can be used to ensure this.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_from_android`] returns a [`ValidationError`].
    ///
    /// [`try_from_android`]: Self::try_from_android
    #[track_caller]
    pub unsafe fn from_android(
        instance: &Arc<Instance>,
        window: *mut vk::ANativeWindow,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        match unsafe { Self::try_from_android(instance, window, object) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a `Surface` from an Android window.
    ///
    /// # Safety
    ///
    /// - `window` must be a valid Android `ANativeWindow` handle.
    /// - The object referred to by `window` must outlive the created `Surface`. The `object`
    ///   parameter can be used to ensure this.
    pub unsafe fn try_from_android(
        instance: &Arc<Instance>,
        window: *mut vk::ANativeWindow,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_from_android(instance, window)?;

        Ok(unsafe { Self::from_android_unchecked(instance, window, object) }?)
    }

    fn validate_from_android(
        instance: &Instance,
        _window: *mut vk::ANativeWindow,
    ) -> Result<(), Box<ValidationError>> {
        if !instance.enabled_extensions().khr_android_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "khr_android_surface",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-VkAndroidSurfaceCreateInfoKHR-window-01248
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn from_android_unchecked(
        instance: &Arc<Instance>,
        window: *mut vk::ANativeWindow,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        let create_info_vk = vk::AndroidSurfaceCreateInfoKHR::default()
            .flags(vk::AndroidSurfaceCreateFlagsKHR::empty())
            .window(window);

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.khr_android_surface.create_android_surface_khr)(
                    instance.handle(),
                    &create_info_vk,
                    ptr::null(),
                    output.as_mut_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;
            unsafe { output.assume_init() }
        };

        Ok(Arc::new(unsafe {
            Self::from_handle(
                instance,
                handle,
                SurfaceApi::Android,
                object.map(SurfaceObject::Other),
            )
        }))
    }

    /// Creates a `Surface` from a DirectFB surface, panicking on a validation error.
    ///
    /// This is a shortcut for `try_from_directfb().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - `dfb` must be a valid DirectFB `IDirectFB` handle.
    /// - `surface` must be a valid DirectFB `IDirectFBSurface` handle.
    /// - The object referred to by `dfb` and `surface` must outlive the created `Surface`. The
    ///   `object` parameter can be used to ensure this.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_from_directfb`] returns a [`ValidationError`].
    ///
    /// [`try_from_directfb`]: Self::try_from_directfb
    #[track_caller]
    pub unsafe fn from_directfb(
        instance: &Arc<Instance>,
        dfb: *mut vk::IDirectFB,
        surface: *mut vk::IDirectFBSurface,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        match unsafe { Self::try_from_directfb(instance, dfb, surface, object) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a `Surface` from a DirectFB surface.
    ///
    /// # Safety
    ///
    /// - `dfb` must be a valid DirectFB `IDirectFB` handle.
    /// - `surface` must be a valid DirectFB `IDirectFBSurface` handle.
    /// - The object referred to by `dfb` and `surface` must outlive the created `Surface`. The
    ///   `object` parameter can be used to ensure this.
    pub unsafe fn try_from_directfb(
        instance: &Arc<Instance>,
        dfb: *mut vk::IDirectFB,
        surface: *mut vk::IDirectFBSurface,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_from_directfb(instance, dfb, surface)?;

        Ok(unsafe { Self::from_directfb_unchecked(instance, dfb, surface, object) }?)
    }

    fn validate_from_directfb(
        instance: &Instance,
        _dfb: *mut vk::IDirectFB,
        _surface: *mut vk::IDirectFBSurface,
    ) -> Result<(), Box<ValidationError>> {
        if !instance.enabled_extensions().ext_directfb_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "ext_directfb_surface",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-VkDirectFBSurfaceCreateInfoEXT-dfb-04117
        // Can't validate, therefore unsafe

        // VUID-VkDirectFBSurfaceCreateInfoEXT-surface-04118
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn from_directfb_unchecked(
        instance: &Arc<Instance>,
        dfb: *mut vk::IDirectFB,
        surface: *mut vk::IDirectFBSurface,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        let create_info_vk = vk::DirectFBSurfaceCreateInfoEXT::default()
            .flags(vk::DirectFBSurfaceCreateFlagsEXT::empty())
            .dfb(dfb)
            .surface(surface);

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.ext_directfb_surface.create_direct_fb_surface_ext)(
                    instance.handle(),
                    &create_info_vk,
                    ptr::null(),
                    output.as_mut_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;
            unsafe { output.assume_init() }
        };

        Ok(Arc::new(unsafe {
            Self::from_handle(
                instance,
                handle,
                SurfaceApi::DirectFB,
                object.map(SurfaceObject::Other),
            )
        }))
    }

    /// Creates a `Surface` from an Fuchsia ImagePipe, panicking on a validation error.
    ///
    /// This is a shortcut for `try_from_fuchsia_image_pipe().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - `image_pipe_handle` must be a valid Fuchsia `zx_handle_t` handle.
    /// - The object referred to by `image_pipe_handle` must outlive the created `Surface`. The
    ///   `object` parameter can be used to ensure this.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_from_fuchsia_image_pipe`] returns a [`ValidationError`].
    ///
    /// [`try_from_fuchsia_image_pipe`]: Self::try_from_fuchsia_image_pipe
    #[track_caller]
    pub unsafe fn from_fuchsia_image_pipe(
        instance: &Arc<Instance>,
        image_pipe_handle: vk::zx_handle_t,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        match unsafe { Self::try_from_fuchsia_image_pipe(instance, image_pipe_handle, object) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a `Surface` from an Fuchsia ImagePipe.
    ///
    /// # Safety
    ///
    /// - `image_pipe_handle` must be a valid Fuchsia `zx_handle_t` handle.
    /// - The object referred to by `image_pipe_handle` must outlive the created `Surface`. The
    ///   `object` parameter can be used to ensure this.
    pub unsafe fn try_from_fuchsia_image_pipe(
        instance: &Arc<Instance>,
        image_pipe_handle: vk::zx_handle_t,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_from_fuchsia_image_pipe(instance, image_pipe_handle)?;

        Ok(
            unsafe {
                Self::from_fuchsia_image_pipe_unchecked(instance, image_pipe_handle, object)
            }?,
        )
    }

    fn validate_from_fuchsia_image_pipe(
        instance: &Instance,
        _image_pipe_handle: vk::zx_handle_t,
    ) -> Result<(), Box<ValidationError>> {
        if !instance.enabled_extensions().fuchsia_imagepipe_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "fuchsia_imagepipe_surface",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-VkImagePipeSurfaceCreateInfoFUCHSIA-imagePipeHandle-04863
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn from_fuchsia_image_pipe_unchecked(
        instance: &Arc<Instance>,
        image_pipe_handle: vk::zx_handle_t,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        let create_info_vk = vk::ImagePipeSurfaceCreateInfoFUCHSIA::default()
            .flags(vk::ImagePipeSurfaceCreateFlagsFUCHSIA::empty())
            .image_pipe_handle(image_pipe_handle);

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.fuchsia_imagepipe_surface
                    .create_image_pipe_surface_fuchsia)(
                    instance.handle(),
                    &create_info_vk,
                    ptr::null(),
                    output.as_mut_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;
            unsafe { output.assume_init() }
        };

        Ok(Arc::new(unsafe {
            Self::from_handle(
                instance,
                handle,
                SurfaceApi::FuchsiaImagePipe,
                object.map(SurfaceObject::Other),
            )
        }))
    }

    /// Creates a `Surface` from a Google Games Platform stream descriptor, panicking on a
    /// validation error.
    ///
    /// This is a shortcut for `try_from_ggp_stream_descriptor().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - `stream_descriptor` must be a valid Google Games Platform `GgpStreamDescriptor` handle.
    /// - The object referred to by `stream_descriptor` must outlive the created `Surface`. The
    ///   `object` parameter can be used to ensure this.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_from_ggp_stream_descriptor`] returns a [`ValidationError`].
    ///
    /// [`try_from_ggp_stream_descriptor`]: Self::try_from_ggp_stream_descriptor
    #[track_caller]
    pub unsafe fn from_ggp_stream_descriptor(
        instance: &Arc<Instance>,
        stream_descriptor: vk::GgpStreamDescriptor,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        match unsafe { Self::try_from_ggp_stream_descriptor(instance, stream_descriptor, object) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a `Surface` from a Google Games Platform stream descriptor.
    ///
    /// # Safety
    ///
    /// - `stream_descriptor` must be a valid Google Games Platform `GgpStreamDescriptor` handle.
    /// - The object referred to by `stream_descriptor` must outlive the created `Surface`. The
    ///   `object` parameter can be used to ensure this.
    pub unsafe fn try_from_ggp_stream_descriptor(
        instance: &Arc<Instance>,
        stream_descriptor: vk::GgpStreamDescriptor,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_from_ggp_stream_descriptor(instance, stream_descriptor)?;

        Ok(unsafe {
            Self::from_ggp_stream_descriptor_unchecked(instance, stream_descriptor, object)
        }?)
    }

    fn validate_from_ggp_stream_descriptor(
        instance: &Instance,
        _stream_descriptor: vk::GgpStreamDescriptor,
    ) -> Result<(), Box<ValidationError>> {
        if !instance.enabled_extensions().ggp_stream_descriptor_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "ggp_stream_descriptor_surface",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-VkStreamDescriptorSurfaceCreateInfoGGP-streamDescriptor-02681
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn from_ggp_stream_descriptor_unchecked(
        instance: &Arc<Instance>,
        stream_descriptor: vk::GgpStreamDescriptor,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        let create_info_vk = vk::StreamDescriptorSurfaceCreateInfoGGP::default()
            .flags(vk::StreamDescriptorSurfaceCreateFlagsGGP::empty())
            .stream_descriptor(stream_descriptor);

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.ggp_stream_descriptor_surface
                    .create_stream_descriptor_surface_ggp)(
                    instance.handle(),
                    &create_info_vk,
                    ptr::null(),
                    output.as_mut_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;
            unsafe { output.assume_init() }
        };

        Ok(Arc::new(unsafe {
            Self::from_handle(
                instance,
                handle,
                SurfaceApi::GgpStreamDescriptor,
                object.map(SurfaceObject::Other),
            )
        }))
    }

    /// Creates a `Surface` from an iOS `UIView`, panicking on a validation error.
    ///
    /// This is a shortcut for `try_from_ios().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - `metal_layer` must be a valid `IOSMetalLayer` handle.
    /// - The object referred to by `metal_layer` must outlive the created `Surface`. The `object`
    ///   parameter can be used to ensure this.
    /// - The `UIView` must be backed by a `CALayer` instance of type `CAMetalLayer`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_from_ios`] returns a [`ValidationError`].
    ///
    /// [`try_from_ios`]: Self::try_from_ios
    #[track_caller]
    pub unsafe fn from_ios(
        instance: &Arc<Instance>,
        view: *const c_void,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        match unsafe { Self::try_from_ios(instance, view, object) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a `Surface` from an iOS `UIView`.
    ///
    /// # Safety
    ///
    /// - `metal_layer` must be a valid `IOSMetalLayer` handle.
    /// - The object referred to by `metal_layer` must outlive the created `Surface`. The `object`
    ///   parameter can be used to ensure this.
    /// - The `UIView` must be backed by a `CALayer` instance of type `CAMetalLayer`.
    pub unsafe fn try_from_ios(
        instance: &Arc<Instance>,
        view: *const c_void,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_from_ios(instance, view)?;

        Ok(unsafe { Self::from_ios_unchecked(instance, view, object) }?)
    }

    fn validate_from_ios(
        instance: &Instance,
        _view: *const c_void,
    ) -> Result<(), Box<ValidationError>> {
        if !instance.enabled_extensions().mvk_ios_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "mvk_ios_surface",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-VkIOSSurfaceCreateInfoMVK-pView-04143
        // Can't validate, therefore unsafe

        // VUID-VkIOSSurfaceCreateInfoMVK-pView-01316
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn from_ios_unchecked(
        instance: &Arc<Instance>,
        view: *const c_void,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        let create_info_vk = vk::IOSSurfaceCreateInfoMVK::default()
            .flags(vk::IOSSurfaceCreateFlagsMVK::empty())
            .view(view);

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.mvk_ios_surface.create_ios_surface_mvk)(
                    instance.handle(),
                    &create_info_vk,
                    ptr::null(),
                    output.as_mut_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;
            unsafe { output.assume_init() }
        };

        Ok(Arc::new(unsafe {
            Self::from_handle(
                instance,
                handle,
                SurfaceApi::Ios,
                object.map(SurfaceObject::Other),
            )
        }))
    }

    /// Creates a `Surface` from a MacOS `NSView`, panicking on a validation error.
    ///
    /// This is a shortcut for `try_from_mac_os().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - `view` must be a valid `CAMetalLayer` or `NSView` handle.
    /// - The object referred to by `view` must outlive the created `Surface`. The `object`
    ///   parameter can be used to ensure this.
    /// - The `NSView` must be backed by a `CALayer` instance of type `CAMetalLayer`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_from_mac_os`] returns a [`ValidationError`].
    ///
    /// [`try_from_mac_os`]: Self::try_from_mac_os
    #[track_caller]
    pub unsafe fn from_mac_os(
        instance: &Arc<Instance>,
        view: *const c_void,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        match unsafe { Self::try_from_mac_os(instance, view, object) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a `Surface` from a MacOS `NSView`.
    ///
    /// # Safety
    ///
    /// - `view` must be a valid `CAMetalLayer` or `NSView` handle.
    /// - The object referred to by `view` must outlive the created `Surface`. The `object`
    ///   parameter can be used to ensure this.
    /// - The `NSView` must be backed by a `CALayer` instance of type `CAMetalLayer`.
    pub unsafe fn try_from_mac_os(
        instance: &Arc<Instance>,
        view: *const c_void,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_from_mac_os(instance, view)?;

        Ok(unsafe { Self::from_mac_os_unchecked(instance, view, object) }?)
    }

    fn validate_from_mac_os(
        instance: &Instance,
        _view: *const c_void,
    ) -> Result<(), Box<ValidationError>> {
        if !instance.enabled_extensions().mvk_macos_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "mvk_macos_surface",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-VkMacOSSurfaceCreateInfoMVK-pView-04144
        // Can't validate, therefore unsafe

        // VUID-VkMacOSSurfaceCreateInfoMVK-pView-01317
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn from_mac_os_unchecked(
        instance: &Arc<Instance>,
        view: *const c_void,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        let create_info_vk = vk::MacOSSurfaceCreateInfoMVK::default()
            .flags(vk::MacOSSurfaceCreateFlagsMVK::empty())
            .view(view);

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.mvk_macos_surface.create_mac_os_surface_mvk)(
                    instance.handle(),
                    &create_info_vk,
                    ptr::null(),
                    output.as_mut_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;
            unsafe { output.assume_init() }
        };

        Ok(Arc::new(unsafe {
            Self::from_handle(
                instance,
                handle,
                SurfaceApi::MacOs,
                object.map(SurfaceObject::Other),
            )
        }))
    }

    /// Create a `Surface` from a [`CAMetalLayer`], panicking on a validation error.
    ///
    /// If you want to create this from a `NSView` or `UIView`, it is recommended that you use the
    /// `raw-window-metal` crate to get access to a layer without overwriting the view's layer.
    ///
    /// This is a shortcut for `try_from_metal().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - `layer` must point to a valid, initialized, non-NULL `CAMetalLayer`.
    /// - The surface will retain the layer internally, so you do not need to keep the source from
    ///   where this came from alive.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_from_metal`] returns a [`ValidationError`].
    ///
    /// [`CAMetalLayer`]: https://developer.apple.com/documentation/quartzcore/cametallayer
    /// [`try_from_metal`]: Self::try_from_metal
    #[track_caller]
    pub unsafe fn from_metal(
        instance: &Arc<Instance>,
        layer: *const vk::CAMetalLayer,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        match unsafe { Self::try_from_metal(instance, layer, object) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Create a `Surface` from a [`CAMetalLayer`].
    ///
    /// If you want to create this from a `NSView` or `UIView`, it is
    /// recommended that you use the `raw-window-metal` crate to get access to
    /// a layer without overwriting the view's layer.
    ///
    /// [`CAMetalLayer`]: https://developer.apple.com/documentation/quartzcore/cametallayer
    ///
    /// # Safety
    ///
    /// `layer` must point to a valid, initialized, non-NULL `CAMetalLayer`.
    ///
    /// The surface will retain the layer internally, so you do not need to
    /// keep the source from where this came from alive.
    pub unsafe fn try_from_metal(
        instance: &Arc<Instance>,
        layer: *const vk::CAMetalLayer,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_from_metal(instance, layer)?;

        Ok(unsafe { Self::from_metal_unchecked(instance, layer, object) }?)
    }

    fn validate_from_metal(
        instance: &Instance,
        layer: *const vk::CAMetalLayer,
    ) -> Result<(), Box<ValidationError>> {
        if !instance.enabled_extensions().ext_metal_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "ext_metal_surface",
                )])]),
                ..Default::default()
            }));
        }

        assert!(!layer.is_null(), "CAMetalLayer must not be NULL");

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn from_metal_unchecked(
        instance: &Arc<Instance>,
        layer: *const vk::CAMetalLayer,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        let create_info_vk = vk::MetalSurfaceCreateInfoEXT::default()
            .flags(vk::MetalSurfaceCreateFlagsEXT::empty())
            .layer(layer);

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.ext_metal_surface.create_metal_surface_ext)(
                    instance.handle(),
                    &create_info_vk,
                    ptr::null(),
                    output.as_mut_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;
            unsafe { output.assume_init() }
        };

        Ok(Arc::new(unsafe {
            Self::from_handle(
                instance,
                handle,
                SurfaceApi::Metal,
                object.map(SurfaceObject::Other),
            )
        }))
    }

    /// Creates a `Surface` from a QNX Screen window, panicking on a validation error.
    ///
    /// This is a shortcut for `try_from_qnx_screen().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - `context` must be a valid QNX Screen `_screen_context` handle.
    /// - `window` must be a valid QNX Screen `_screen_window` handle.
    /// - The object referred to by `window` must outlive the created `Surface`. The `object`
    ///   parameter can be used to ensure this.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_from_qnx_screen`] returns a [`ValidationError`].
    ///
    /// [`try_from_qnx_screen`]: Self::try_from_qnx_screen
    #[track_caller]
    pub unsafe fn from_qnx_screen(
        instance: &Arc<Instance>,
        context: *mut vk::_screen_context,
        window: *mut vk::_screen_window,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        match unsafe { Self::try_from_qnx_screen(instance, context, window, object) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a `Surface` from a QNX Screen window.
    ///
    /// # Safety
    ///
    /// - `context` must be a valid QNX Screen `_screen_context` handle.
    /// - `window` must be a valid QNX Screen `_screen_window` handle.
    /// - The object referred to by `window` must outlive the created `Surface`. The `object`
    ///   parameter can be used to ensure this.
    pub unsafe fn try_from_qnx_screen(
        instance: &Arc<Instance>,
        context: *mut vk::_screen_context,
        window: *mut vk::_screen_window,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_from_qnx_screen(instance, context, window)?;

        Ok(unsafe { Self::from_qnx_screen_unchecked(instance, context, window, object) }?)
    }

    fn validate_from_qnx_screen(
        instance: &Instance,
        _context: *mut vk::_screen_context,
        _window: *mut vk::_screen_window,
    ) -> Result<(), Box<ValidationError>> {
        if !instance.enabled_extensions().qnx_screen_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "qnx_screen_surface",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-VkScreenSurfaceCreateInfoQNX-context-04741
        // Can't validate, therefore unsafe

        // VUID-VkScreenSurfaceCreateInfoQNX-window-04742
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn from_qnx_screen_unchecked(
        instance: &Arc<Instance>,
        context: *mut vk::_screen_context,
        window: *mut vk::_screen_window,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        let create_info_vk = vk::ScreenSurfaceCreateInfoQNX {
            flags: vk::ScreenSurfaceCreateFlagsQNX::empty(),
            context,
            window,
            ..Default::default()
        };

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.qnx_screen_surface.create_screen_surface_qnx)(
                    instance.handle(),
                    &create_info_vk,
                    ptr::null(),
                    output.as_mut_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;
            unsafe { output.assume_init() }
        };

        Ok(Arc::new(unsafe {
            Self::from_handle(
                instance,
                handle,
                SurfaceApi::QnxScreen,
                object.map(SurfaceObject::Other),
            )
        }))
    }

    /// Creates a `Surface` from a `nn::vi::Layer`, panicking on a validation error.
    ///
    /// This is a shortcut for `try_from_vi().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - `window` must be a valid `nn::vi::NativeWindowHandle` handle.
    /// - The object referred to by `window` must outlive the created `Surface`. The `object`
    ///   parameter can be used to ensure this.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_from_vi`] returns a [`ValidationError`].
    ///
    /// [`try_from_vi`]: Self::try_from_vi
    #[track_caller]
    pub unsafe fn from_vi(
        instance: &Arc<Instance>,
        window: *mut c_void,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        match unsafe { Self::try_from_vi(instance, window, object) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a `Surface` from a `nn::vi::Layer`.
    ///
    /// # Safety
    ///
    /// - `window` must be a valid `nn::vi::NativeWindowHandle` handle.
    /// - The object referred to by `window` must outlive the created `Surface`. The `object`
    ///   parameter can be used to ensure this.
    pub unsafe fn try_from_vi(
        instance: &Arc<Instance>,
        window: *mut c_void,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_from_vi(instance, window)?;

        Ok(unsafe { Self::from_vi_unchecked(instance, window, object) }?)
    }

    fn validate_from_vi(
        instance: &Instance,
        _window: *mut c_void,
    ) -> Result<(), Box<ValidationError>> {
        if !instance.enabled_extensions().nn_vi_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "nn_vi_surface",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-VkViSurfaceCreateInfoNN-window-01318
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn from_vi_unchecked(
        instance: &Arc<Instance>,
        window: *mut c_void,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        let create_info_vk = vk::ViSurfaceCreateInfoNN::default()
            .flags(vk::ViSurfaceCreateFlagsNN::empty())
            .window(window);

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.nn_vi_surface.create_vi_surface_nn)(
                    instance.handle(),
                    &create_info_vk,
                    ptr::null(),
                    output.as_mut_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;
            unsafe { output.assume_init() }
        };

        Ok(Arc::new(unsafe {
            Self::from_handle(
                instance,
                handle,
                SurfaceApi::Vi,
                object.map(SurfaceObject::Other),
            )
        }))
    }

    /// Creates a `Surface` from a Wayland window, panicking on a validation error.
    ///
    /// The window's dimensions will be set to the size of the swapchain.
    ///
    /// This is a shortcut for `try_from_wayland().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - `display` must be a valid Wayland `wl_display` handle.
    /// - `surface` must be a valid Wayland `wl_surface` handle.
    /// - The objects referred to by `display` and `surface` must outlive the created `Surface`.
    ///   The `object` parameter can be used to ensure this.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_from_wayland`] returns a [`ValidationError`].
    ///
    /// [`try_from_wayland`]: Self::try_from_wayland
    #[track_caller]
    pub unsafe fn from_wayland(
        instance: &Arc<Instance>,
        display: *mut vk::wl_display,
        surface: *mut vk::wl_surface,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        match unsafe { Self::try_from_wayland(instance, display, surface, object) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a `Surface` from a Wayland window.
    ///
    /// The window's dimensions will be set to the size of the swapchain.
    ///
    /// # Safety
    ///
    /// - `display` must be a valid Wayland `wl_display` handle.
    /// - `surface` must be a valid Wayland `wl_surface` handle.
    /// - The objects referred to by `display` and `surface` must outlive the created `Surface`.
    ///   The `object` parameter can be used to ensure this.
    pub unsafe fn try_from_wayland(
        instance: &Arc<Instance>,
        display: *mut vk::wl_display,
        surface: *mut vk::wl_surface,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_from_wayland(instance, display, surface)?;

        Ok(unsafe { Self::from_wayland_unchecked(instance, display, surface, object) }?)
    }

    fn validate_from_wayland(
        instance: &Instance,
        _display: *mut vk::wl_display,
        _surface: *mut vk::wl_surface,
    ) -> Result<(), Box<ValidationError>> {
        if !instance.enabled_extensions().khr_wayland_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "khr_wayland_surface",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-VkWaylandSurfaceCreateInfoKHR-display-01304
        // Can't validate, therefore unsafe

        // VUID-VkWaylandSurfaceCreateInfoKHR-surface-01305
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn from_wayland_unchecked(
        instance: &Arc<Instance>,
        display: *mut vk::wl_display,
        surface: *mut vk::wl_surface,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        let create_info_vk = vk::WaylandSurfaceCreateInfoKHR::default()
            .flags(vk::WaylandSurfaceCreateFlagsKHR::empty())
            .display(display)
            .surface(surface);

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.khr_wayland_surface.create_wayland_surface_khr)(
                    instance.handle(),
                    &create_info_vk,
                    ptr::null(),
                    output.as_mut_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;
            unsafe { output.assume_init() }
        };

        Ok(Arc::new(unsafe {
            Self::from_handle(
                instance,
                handle,
                SurfaceApi::Wayland,
                object.map(SurfaceObject::Other),
            )
        }))
    }

    /// Creates a `Surface` from a Win32 window, panicking on a validation error.
    ///
    /// The surface's min, max and current extent will always match the window's dimensions.
    ///
    /// This is a shortcut for `try_from_win32().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - `hinstance` must be a valid Win32 `HINSTANCE` handle.
    /// - `hwnd` must be a valid Win32 `HWND` handle.
    /// - The objects referred to by `hwnd` and `hinstance` must outlive the created `Surface`. The
    ///   `object` parameter can be used to ensure this.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_from_win32`] returns a [`ValidationError`].
    ///
    /// [`try_from_win32`]: Self::try_from_win32
    #[track_caller]
    pub unsafe fn from_win32(
        instance: &Arc<Instance>,
        hinstance: vk::HINSTANCE,
        hwnd: vk::HWND,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        match unsafe { Self::try_from_win32(instance, hinstance, hwnd, object) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a `Surface` from a Win32 window.
    ///
    /// The surface's min, max and current extent will always match the window's dimensions.
    ///
    /// # Safety
    ///
    /// - `hinstance` must be a valid Win32 `HINSTANCE` handle.
    /// - `hwnd` must be a valid Win32 `HWND` handle.
    /// - The objects referred to by `hwnd` and `hinstance` must outlive the created `Surface`. The
    ///   `object` parameter can be used to ensure this.
    pub unsafe fn try_from_win32(
        instance: &Arc<Instance>,
        hinstance: vk::HINSTANCE,
        hwnd: vk::HWND,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_from_win32(instance, hinstance, hwnd)?;

        Ok(unsafe { Self::from_win32_unchecked(instance, hinstance, hwnd, object) }?)
    }

    fn validate_from_win32(
        instance: &Instance,
        _hinstance: vk::HINSTANCE,
        _hwnd: vk::HWND,
    ) -> Result<(), Box<ValidationError>> {
        if !instance.enabled_extensions().khr_win32_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "khr_win32_surface",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-VkWin32SurfaceCreateInfoKHR-hinstance-01307
        // Can't validate, therefore unsafe

        // VUID-VkWin32SurfaceCreateInfoKHR-hwnd-01308
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn from_win32_unchecked(
        instance: &Arc<Instance>,
        hinstance: vk::HINSTANCE,
        hwnd: vk::HWND,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        let create_info_vk = vk::Win32SurfaceCreateInfoKHR::default()
            .flags(vk::Win32SurfaceCreateFlagsKHR::empty())
            .hinstance(hinstance)
            .hwnd(hwnd);

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.khr_win32_surface.create_win32_surface_khr)(
                    instance.handle(),
                    &create_info_vk,
                    ptr::null(),
                    output.as_mut_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;
            unsafe { output.assume_init() }
        };

        Ok(Arc::new(unsafe {
            Self::from_handle(
                instance,
                handle,
                SurfaceApi::Win32,
                object.map(SurfaceObject::Other),
            )
        }))
    }

    /// Creates a `Surface` from an XCB window, panicking on a validation error.
    ///
    /// The surface's min, max and current extent will always match the window's dimensions.
    ///
    /// This is a shortcut for `try_from_xcb().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - `connection` must be a valid X11 `xcb_connection_t` handle.
    /// - `window` must be a valid X11 `xcb_window_t` handle.
    /// - The objects referred to by `connection` and `window` must outlive the created `Surface`.
    ///   The `object` parameter can be used to ensure this.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_from_xcb`] returns a [`ValidationError`].
    ///
    /// [`try_from_xcb`]: Self::try_from_xcb
    #[track_caller]
    pub unsafe fn from_xcb(
        instance: &Arc<Instance>,
        connection: *mut vk::xcb_connection_t,
        window: vk::xcb_window_t,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        match unsafe { Self::try_from_xcb(instance, connection, window, object) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a `Surface` from an XCB window.
    ///
    /// The surface's min, max and current extent will always match the window's dimensions.
    ///
    /// # Safety
    ///
    /// - `connection` must be a valid X11 `xcb_connection_t` handle.
    /// - `window` must be a valid X11 `xcb_window_t` handle.
    /// - The objects referred to by `connection` and `window` must outlive the created `Surface`.
    ///   The `object` parameter can be used to ensure this.
    pub unsafe fn try_from_xcb(
        instance: &Arc<Instance>,
        connection: *mut vk::xcb_connection_t,
        window: vk::xcb_window_t,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_from_xcb(instance, connection, window)?;

        Ok(unsafe { Self::from_xcb_unchecked(instance, connection, window, object) }?)
    }

    fn validate_from_xcb(
        instance: &Instance,
        _connection: *mut vk::xcb_connection_t,
        _window: vk::xcb_window_t,
    ) -> Result<(), Box<ValidationError>> {
        if !instance.enabled_extensions().khr_xcb_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "khr_xcb_surface",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-VkXcbSurfaceCreateInfoKHR-connection-01310
        // Can't validate, therefore unsafe

        // VUID-VkXcbSurfaceCreateInfoKHR-window-01311
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn from_xcb_unchecked(
        instance: &Arc<Instance>,
        connection: *mut vk::xcb_connection_t,
        window: vk::xcb_window_t,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        let create_info_vk = vk::XcbSurfaceCreateInfoKHR::default()
            .flags(vk::XcbSurfaceCreateFlagsKHR::empty())
            .connection(connection)
            .window(window);

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.khr_xcb_surface.create_xcb_surface_khr)(
                    instance.handle(),
                    &create_info_vk,
                    ptr::null(),
                    output.as_mut_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;
            unsafe { output.assume_init() }
        };

        Ok(Arc::new(unsafe {
            Self::from_handle(
                instance,
                handle,
                SurfaceApi::Xcb,
                object.map(SurfaceObject::Other),
            )
        }))
    }

    /// Creates a `Surface` from an Xlib window, panicking on a validation error.
    ///
    /// The surface's min, max and current extent will always match the window's dimensions.
    ///
    /// This is a shortcut for `try_from_xlib().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - `display` must be a valid Xlib `Display` handle.
    /// - `window` must be a valid Xlib `Window` handle.
    /// - The objects referred to by `display` and `window` must outlive the created `Surface`. The
    ///   `object` parameter can be used to ensure this.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_from_xlib`] returns a [`ValidationError`].
    ///
    /// [`try_from_xlib`]: Self::try_from_xlib
    #[track_caller]
    pub unsafe fn from_xlib(
        instance: &Arc<Instance>,
        display: *mut vk::Display,
        window: vk::Window,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        match unsafe { Self::try_from_xlib(instance, display, window, object) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a `Surface` from an Xlib window.
    ///
    /// The surface's min, max and current extent will always match the window's dimensions.
    ///
    /// # Safety
    ///
    /// - `display` must be a valid Xlib `Display` handle.
    /// - `window` must be a valid Xlib `Window` handle.
    /// - The objects referred to by `display` and `window` must outlive the created `Surface`. The
    ///   `object` parameter can be used to ensure this.
    pub unsafe fn try_from_xlib(
        instance: &Arc<Instance>,
        display: *mut vk::Display,
        window: vk::Window,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_from_xlib(instance, display, window)?;

        Ok(unsafe { Self::from_xlib_unchecked(instance, display, window, object) }?)
    }

    fn validate_from_xlib(
        instance: &Instance,
        _display: *mut vk::Display,
        _window: vk::Window,
    ) -> Result<(), Box<ValidationError>> {
        if !instance.enabled_extensions().khr_xlib_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "khr_xlib_surface",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-VkXlibSurfaceCreateInfoKHR-dpy-01313
        // Can't validate, therefore unsafe

        // VUID-VkXlibSurfaceCreateInfoKHR-window-01314
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn from_xlib_unchecked(
        instance: &Arc<Instance>,
        display: *mut vk::Display,
        window: vk::Window,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        let create_info_vk = vk::XlibSurfaceCreateInfoKHR::default()
            .flags(vk::XlibSurfaceCreateFlagsKHR::empty())
            .dpy(display)
            .window(window);

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.khr_xlib_surface.create_xlib_surface_khr)(
                    instance.handle(),
                    &create_info_vk,
                    ptr::null(),
                    output.as_mut_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;
            unsafe { output.assume_init() }
        };

        Ok(Arc::new(unsafe {
            Self::from_handle(
                instance,
                handle,
                SurfaceApi::Xlib,
                object.map(SurfaceObject::Other),
            )
        }))
    }

    /// Returns the instance this surface was created with.
    #[inline]
    pub fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }

    /// Returns the windowing API that was used to construct the surface.
    #[inline]
    pub fn api(&self) -> SurfaceApi {
        self.api
    }

    /// Returns a reference to the `window` or `object` parameter that was passed when creating the
    /// surface.
    #[inline]
    pub fn object(&self) -> Option<&SurfaceObject> {
        self.object.as_ref()
    }
}

impl Drop for Surface {
    #[inline]
    fn drop(&mut self) {
        let fns = self.instance.fns();
        unsafe {
            (fns.khr_surface.destroy_surface_khr)(self.instance.handle(), self.handle, ptr::null())
        };
    }
}

unsafe impl VulkanObject for Surface {
    type Handle = vk::SurfaceKHR;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl InstanceOwned for Surface {
    #[inline]
    fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }
}

impl Debug for Surface {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        let Self {
            handle,
            instance,
            id,
            api,
            object,

            surface_formats: _,
            surface_present_modes: _,
            surface_support: _,
        } = self;

        f.debug_struct("Surface")
            .field("handle", handle)
            .field("instance", instance)
            .field("id", id)
            .field("api", api)
            .field("object", object)
            .finish_non_exhaustive()
    }
}

impl_id_counter!(Surface);

/// A trait for windows that can be passed to [`Surface::from_window`].
#[cfg(feature = "raw_window_handle")]
pub trait Window: HasWindowHandle + HasDisplayHandle + Any + Send + Sync {
    fn as_dyn(self: Arc<Self>) -> Arc<dyn Window>;
}

#[cfg(feature = "raw_window_handle")]
impl<T: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static> Window for T {
    #[inline]
    fn as_dyn(self: Arc<Self>) -> Arc<dyn Window> {
        self
    }
}

#[cfg(feature = "raw_window_handle")]
impl Debug for dyn Window {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        f.debug_struct("Window").finish_non_exhaustive()
    }
}

/// Parameters to create a surface from a display mode and plane.
#[derive(Clone, Debug)]
pub struct DisplaySurfaceCreateInfo<'a> {
    /// The index of the display plane in which the surface will appear.
    ///
    /// The default value is 0.
    pub plane_index: u32,

    /// The z-order of the plane.
    ///
    /// The default value is 0.
    pub plane_stack_index: u32,

    /// The transformation to apply to images when presented.
    ///
    /// The default value is [`SurfaceTransform::Identity`].
    pub transform: SurfaceTransform,

    /// How to do alpha blending on the surface when presenting new content.
    ///
    /// The default value is [`DisplayPlaneAlpha::Opaque`].
    pub alpha_mode: DisplayPlaneAlpha,

    /// If `alpha_mode` is `DisplayPlaneAlpha::Global`, specifies the global alpha value to use for
    /// all blending.
    ///
    /// The default value is 1.0.
    pub global_alpha: f32,

    /// The size of the presentable images that will be used.
    ///
    /// The default value is `[0; 2]`, which must be overridden.
    pub image_extent: [u32; 2],

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for DisplaySurfaceCreateInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl DisplaySurfaceCreateInfo<'_> {
    /// Returns a default `DisplaySurfaceCreateInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            plane_index: 0,
            plane_stack_index: 0,
            transform: SurfaceTransform::Identity,
            alpha_mode: DisplayPlaneAlpha::Opaque,
            global_alpha: 1.0,
            image_extent: [0; 2],
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(
        &self,
        physical_device: &PhysicalDevice,
    ) -> Result<(), Box<ValidationError>> {
        let &Self {
            plane_index,
            plane_stack_index: _,
            transform,
            alpha_mode,
            global_alpha,
            image_extent,
            _ne: _,
        } = self;

        let properties = physical_device.properties();

        transform
            .validate_physical_device(physical_device)
            .map_err(|err| {
                err.add_context("transform")
                    .set_vuids(&["VUID-VkDisplaySurfaceCreateInfoKHR-transform-parameter"])
            })?;

        alpha_mode
            .validate_physical_device(physical_device)
            .map_err(|err| {
                err.add_context("alpha_mode")
                    .set_vuids(&["VUID-VkDisplaySurfaceCreateInfoKHR-alphaMode-parameter"])
            })?;

        let display_plane_properties_raw =
            unsafe { physical_device.display_plane_properties_raw() }.map_err(|_err| {
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
                vuids: &["VUID-VkDisplaySurfaceCreateInfoKHR-planeIndex-01252"],
                ..Default::default()
            }));
        }

        if alpha_mode == DisplayPlaneAlpha::Global && !(0.0..=1.0).contains(&global_alpha) {
            return Err(Box::new(ValidationError {
                problem: "`alpha_mode` is `DisplayPlaneAlpha::Global`, but \
                    `global_alpha` is not between 0 and 1 inclusive"
                    .into(),
                vuids: &["VUID-VkDisplaySurfaceCreateInfoKHR-alphaMode-01254"],
                ..Default::default()
            }));
        }

        if image_extent[0] > properties.max_image_dimension2_d {
            return Err(Box::new(ValidationError {
                context: "image_extent[0]".into(),
                problem: "is greater than the `max_image_dimension2_d` device limit".into(),
                vuids: &["VUID-VkDisplaySurfaceCreateInfoKHR-width-01256"],
                ..Default::default()
            }));
        }

        if image_extent[1] > properties.max_image_dimension2_d {
            return Err(Box::new(ValidationError {
                context: "image_extent[1]".into(),
                problem: "is greater than the `max_image_dimension2_d` device limit".into(),
                vuids: &["VUID-VkDisplaySurfaceCreateInfoKHR-width-01256"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk(
        &self,
        display_mode_vk: vk::DisplayModeKHR,
    ) -> vk::DisplaySurfaceCreateInfoKHR<'static> {
        let &Self {
            plane_index,
            plane_stack_index,
            transform,
            alpha_mode,
            global_alpha,
            image_extent,
            _ne: _,
        } = self;

        vk::DisplaySurfaceCreateInfoKHR::default()
            .flags(vk::DisplaySurfaceCreateFlagsKHR::empty())
            .display_mode(display_mode_vk)
            .plane_index(plane_index)
            .plane_stack_index(plane_stack_index)
            .transform(transform.into())
            .alpha_mode(alpha_mode.into())
            .global_alpha(global_alpha)
            .image_extent(vk::Extent2D {
                width: image_extent[0],
                height: image_extent[1],
            })
    }
}

/// The windowing API function that was used to construct a surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SurfaceApi {
    /// The surface was constructed using [`Surface::headless`] or the
    /// `vkCreateHeadlessSurfaceEXT` Vulkan API function.
    Headless,

    /// The surface was constructed using [`Surface::from_display_plane`] or the
    /// `vkCreateDisplayPlaneSurfaceKHR` Vulkan API function.
    DisplayPlane,

    /*
        Alphabetical order
    */
    /// The surface was constructed using [`Surface::from_android`] or the
    /// `vkCreateAndroidSurfaceKHR` Vulkan API function.
    Android,

    /// The surface was constructed using [`Surface::from_directfb`] or the
    /// `vkCreateDirectFBSurfaceEXT` Vulkan API function.
    DirectFB,

    /// The surface was constructed using [`Surface::from_fuchsia_image_pipe`] or the
    /// `vkCreateImagePipeSurfaceFUCHSIA` Vulkan API function.
    FuchsiaImagePipe,

    /// The surface was constructed using [`Surface::from_ggp_stream_descriptor`] or the
    /// `vkCreateStreamDescriptorSurfaceGGP` Vulkan API function.
    GgpStreamDescriptor,

    /// The surface was constructed using [`Surface::from_ios`] or the
    /// `vkCreateIOSSurfaceMVK` Vulkan API function.
    Ios,

    /// The surface was constructed using [`Surface::from_mac_os`] or the
    /// `vkCreateMacOSSurfaceMVK` Vulkan API function.
    MacOs,

    /// The surface was constructed using [`Surface::from_metal`] or the
    /// `vkCreateMetalSurfaceEXT` Vulkan API function.
    Metal,

    /// The surface was constructed using [`Surface::from_qnx_screen`] or the
    /// `vkCreateScreenSurfaceQNX` Vulkan API function.
    QnxScreen,

    /// The surface was constructed using [`Surface::from_vi`] or the
    /// `vkCreateViSurfaceNN` Vulkan API function.
    Vi,

    /// The surface was constructed using [`Surface::from_wayland`] or the
    /// `vkCreateWaylandSurfaceKHR` Vulkan API function.
    Wayland,

    /// The surface was constructed using [`Surface::from_win32`] or the
    /// `vkCreateWin32SurfaceKHR` Vulkan API function.
    Win32,

    /// The surface was constructed using [`Surface::from_xcb`] or the
    /// `vkCreateXcbSurfaceKHR` Vulkan API function.
    Xcb,

    /// The surface was constructed using [`Surface::from_xlib`] or the
    /// `vkCreateXlibSurfaceKHR` Vulkan API function.
    Xlib,
}

#[cfg(feature = "raw_window_handle")]
impl TryFrom<RawWindowHandle> for SurfaceApi {
    type Error = ();

    fn try_from(handle: RawWindowHandle) -> Result<Self, Self::Error> {
        match handle {
            RawWindowHandle::UiKit(_) => Ok(SurfaceApi::Metal),
            RawWindowHandle::AppKit(_) => Ok(SurfaceApi::Metal),
            RawWindowHandle::Orbital(_) => Err(()),
            RawWindowHandle::Xlib(_) => Ok(SurfaceApi::Xlib),
            RawWindowHandle::Xcb(_) => Ok(SurfaceApi::Xcb),
            RawWindowHandle::Wayland(_) => Ok(SurfaceApi::Wayland),
            RawWindowHandle::Drm(_) => Err(()),
            RawWindowHandle::Gbm(_) => Err(()),
            RawWindowHandle::Win32(_) => Ok(SurfaceApi::Win32),
            RawWindowHandle::WinRt(_) => Err(()),
            RawWindowHandle::Web(_) => Err(()),
            RawWindowHandle::AndroidNdk(_) => Ok(SurfaceApi::Android),
            RawWindowHandle::Haiku(_) => Err(()),
            _ => Err(()),
        }
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// The mode of action when a swapchain image is presented.
    ///
    /// Swapchain images can be in one of three possible states:
    /// - Exactly one image is currently displayed on the screen.
    /// - Zero or more are acquired by the application, or available to be acquired.
    /// - Some may be held inside the presentation engine waiting to be displayed. The present mode
    ///   concerns the behavior of this category, and by extension, which images are left over for
    ///   acquiring.
    ///
    /// The present mode affects what is commonly known as "vertical sync" or "vsync" for short.
    /// The `Immediate` mode is equivalent to disabling vertical sync, while the others enable
    /// vertical sync in various forms. An important aspect of the present modes is their potential
    /// *latency*: the time between when an image is presented, and when it actually appears on
    /// the display.
    ///
    /// Only `Fifo` is guaranteed to be supported on every device. For the others, you must call
    /// [`surface_present_modes`] to see if they are supported.
    ///
    /// [`surface_present_modes`]: PhysicalDevice::surface_present_modes
    PresentMode = PresentModeKHR(i32);

    /// The presentation engine holds only the currently displayed image. When presenting an image,
    /// the currently displayed image is immediately replaced with the presented image. The old
    /// image will be available for future acquire operations.
    ///
    /// This mode has the lowest latency of all present modes, but if the display is not in a
    /// vertical blanking period when the image is replaced, a tear will be visible.
    Immediate = IMMEDIATE,

    /// The presentation engine holds the currently displayed image, and optionally another in a
    /// waiting slot. The presentation engine waits until the next vertical blanking period, then
    /// removes any image from the waiting slot and displays it. Tearing will never be visible.
    /// When presenting an image, it is stored in the waiting slot. Any previous entry
    /// in the slot is discarded, and will be available for future acquire operations.
    ///
    /// Latency is relatively low with this mode, and will never be longer than the time between
    /// vertical blanking periods. However, if a previous image in the waiting slot is discarded,
    /// the work that went into producing that image was wasted.
    ///
    /// With two swapchain images, this mode behaves essentially identical to `Fifo`: once both
    /// images are held in the presentation engine, no images can be acquired until one is finished
    /// displaying. But with three or more swapchain images, any images beyond those two are always
    /// available to acquire.
    Mailbox = MAILBOX,

    /// The presentation engine holds the currently displayed image, and a queue of waiting images.
    /// When presenting an image, it is added to the tail of the queue, after previously presented
    /// images. The presentation engine waits until the next vertical blanking period, then removes
    /// an image from the head of the queue and displays it. Tearing will never be visible. Images
    /// become available for future acquire operations only after they have been displayed.
    ///
    /// This mode is guaranteed to be always supported. It is possible for all swapchain images to
    /// end up being held by the presentation engine, either being displayed or in the queue. When
    /// that happens, no images can be acquired until one is finished displaying. This can be used
    /// to limit the presentation rate to the display frame rate. Latency is bounded only by the
    /// number of images in the swapchain.
    ///
    /// This is the equivalent of OpenGL's `SwapInterval` with a value of 1.
    Fifo = FIFO,

    /// Similar to `Fifo`, but with the ability for images to "skip the queue" if presentation is
    /// lagging behind the display frame rate. If the queue is empty and a vertical blanking period
    /// has already passed since the previous image was displayed, then the currently displayed
    /// image is immediately replaced with the presented image, as in `Immediate`.
    ///
    /// This mode has high latency if images are presented faster than the display frame rate,
    /// as they will accumulate in the queue. But the latency is low if images are presented slower
    /// than the display frame rate. However, slower presentation can result in visible tearing.
    ///
    /// This is the equivalent of OpenGL's `SwapInterval` with a value of -1.
    FifoRelaxed = FIFO_RELAXED,

    /* TODO: enable
    // TODO: document
    SharedDemandRefresh = SHARED_DEMAND_REFRESH_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_shared_presentable_image)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    SharedContinuousRefresh = SHARED_CONTINUOUS_REFRESH_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_shared_presentable_image)]),
    ]),*/
}

vulkan_bitflags_enum! {
    #[non_exhaustive]

    /// A set of [`SurfaceTransform`] values.
    SurfaceTransforms,

    /// The presentation transform to apply when presenting a swapchain image to a surface.
    SurfaceTransform,

    = SurfaceTransformFlagsKHR(u32);

    /// Don't transform the image.
    IDENTITY, Identity = IDENTITY,

    /// Rotate 90 degrees.
    ROTATE_90, Rotate90 = ROTATE_90,

    /// Rotate 180 degrees.
    ROTATE_180, Rotate180 = ROTATE_180,

    /// Rotate 270 degrees.
    ROTATE_270, Rotate270 = ROTATE_270,

    /// Mirror the image horizontally.
    HORIZONTAL_MIRROR, HorizontalMirror = HORIZONTAL_MIRROR,

    /// Mirror the image horizontally and rotate 90 degrees.
    HORIZONTAL_MIRROR_ROTATE_90, HorizontalMirrorRotate90 = HORIZONTAL_MIRROR_ROTATE_90,

    /// Mirror the image horizontally and rotate 180 degrees.
    HORIZONTAL_MIRROR_ROTATE_180, HorizontalMirrorRotate180 = HORIZONTAL_MIRROR_ROTATE_180,

    /// Mirror the image horizontally and rotate 270 degrees.
    HORIZONTAL_MIRROR_ROTATE_270, HorizontalMirrorRotate270 = HORIZONTAL_MIRROR_ROTATE_270,

    /// Let the operating system or driver implementation choose.
    INHERIT, Inherit = INHERIT,
}

impl Default for SurfaceTransform {
    #[inline]
    fn default() -> SurfaceTransform {
        SurfaceTransform::Identity
    }
}

vulkan_bitflags_enum! {
    #[non_exhaustive]

    /// A set of [`CompositeAlpha`] values.
    CompositeAlphas,

    /// How the alpha values of the pixels of the window are treated.
    CompositeAlpha,

    = CompositeAlphaFlagsKHR(u32);

    /// The alpha channel of the image is ignored. All the pixels are considered as if they have a
    /// value of 1.0.
    OPAQUE, Opaque = OPAQUE,

    /// The alpha channel of the image is respected. The color channels are expected to have
    /// already been multiplied by the alpha value.
    PRE_MULTIPLIED, PreMultiplied = PRE_MULTIPLIED,

    /// The alpha channel of the image is respected. The color channels will be multiplied by the
    /// alpha value by the compositor before being added to what is behind.
    POST_MULTIPLIED, PostMultiplied = POST_MULTIPLIED,

    /// Let the operating system or driver implementation choose.
    INHERIT, Inherit = INHERIT,
}

vulkan_enum! {
    #[non_exhaustive]

    /// How the presentation engine should interpret the data.
    ///
    /// # A quick lesson about color spaces
    ///
    /// ## What is a color space?
    ///
    /// Each pixel of a monitor is made of three components: one red, one green, and one blue. In
    /// the past, computers would simply send to the monitor the intensity of each of the three
    /// components.
    ///
    /// This proved to be problematic, because depending on the brand of the monitor the colors
    /// would not exactly be the same. For example on some monitors, a value of `[1.0, 0.0, 0.0]`
    /// would be a
    /// bit more orange than on others.
    ///
    /// In order to standardize this, there exist what are called *color spaces*: sRGB, AdobeRGB,
    /// DCI-P3, scRGB, etc. When you manipulate RGB values in a specific color space, these values
    /// have a precise absolute meaning in terms of color, that is the same across all systems and
    /// monitors.
    ///
    /// <div class="vulkano-alert-note">
    ///
    /// > Note
    /// >
    /// > Color spaces are orthogonal to concept of RGB. *RGB* only indicates what is the
    /// > representation of the data, but not how it is interpreted. You can think of this a bit
    /// > like text encoding. An *RGB* value is a like a byte, in other words it is the medium by
    /// > which values are communicated, and a *color space* is like a text encoding (e.g.. UTF-8),
    /// > in other words it is the way the value should be interpreted.
    ///
    /// </div>
    ///
    /// The most commonly used color space today is sRGB. Most monitors today use this color space,
    /// and most images files are encoded in this color space.
    ///
    /// ## Pixel formats and linear vs non-linear
    ///
    /// In Vulkan all images have a specific format in which the data is stored. The data of an
    /// image consists of pixels in RGB but contains no information about the color space (or lack
    /// thereof) of these pixels. You are free to store them in whatever color space you want.
    ///
    /// But one big practical problem with color spaces is that they are sometimes not linear, and
    /// in particular the popular sRGB color space is not linear. In a non-linear color space, a
    /// value of `[0.6, 0.6, 0.6]` for example is **not** twice as bright as a value of `[0.3, 0.3,
    /// 0.3]`. This is problematic, because operations such as taking the average of two colors or
    /// calculating the lighting of a texture with a dot product are mathematically incorrect and
    /// will produce incorrect colors.
    ///
    /// <div class="vulkano-alert-note">
    ///
    /// > Note
    /// >
    /// > If the texture format has an alpha component, it is not affected by the color space and
    /// > always behaves linearly.
    ///
    /// </div>
    ///
    /// In order to solve this Vulkan also provides image formats with the `Srgb` suffix, which are
    /// expected to contain RGB data in the sRGB color space. When you sample an image with such a
    /// format from a shader, the implementation will automatically turn the pixel values into a
    /// linear color space that is suitable for linear operations (such as additions or
    /// multiplications). When you write to a framebuffer attachment with such a format, the
    /// implementation will automatically perform the opposite conversion. These conversions are
    /// most of the time performed by the hardware and incur no additional cost.
    ///
    /// ## Color space of the swapchain
    ///
    /// The color space that you specify when you create a swapchain is how the implementation will
    /// interpret the raw data inside of the image.
    ///
    /// <div class="vulkano-alert-note">
    ///
    /// > Note
    /// >
    /// > The implementation can choose to send the data in the swapchain image directly to the
    /// > monitor, but it can also choose to write it in an intermediary buffer that is then read
    /// > by the operating system or windowing system. Therefore the color space that the
    /// > implementation supports is not necessarily the same as the one supported by the monitor.
    ///
    /// </div>
    ///
    /// It is *your* job to ensure that the data in the swapchain image is in the color space
    /// that is specified here, otherwise colors will be incorrect. The implementation will never
    /// perform any additional automatic conversion after the colors have been written to the
    /// swapchain image.
    ///
    /// # How do I handle this correctly?
    ///
    /// The easiest way to handle color spaces in a cross-platform program is:
    ///
    /// - Always request the `SrgbNonLinear` color space when creating the swapchain.
    /// - Make sure that all your image files use the sRGB color space, and load them in images
    ///   whose format has the `Srgb` suffix. Only use non-sRGB image formats for intermediary
    ///   computations or to store non-color data.
    /// - Swapchain images should have a format with the `Srgb` suffix.
    ///
    /// <div class="vulkano-alert-note">
    ///
    /// > Note
    /// >
    /// > Lots of developers are confused by color spaces. You can sometimes find articles talking
    /// > about gamma correction and suggestion to put your colors to the power 2.2 for example.
    /// > These are all hacks and you should use the sRGB pixel formats instead.
    ///
    /// </div>
    ///
    /// If you follow these three rules, then everything should render the same way on all
    /// platforms.
    ///
    /// Additionally you can try detect whether the implementation supports any additional color
    /// space and perform a manual conversion to that color space from inside your shader.
    ColorSpace = ColorSpaceKHR(i32);

    // TODO: document
    SrgbNonLinear = SRGB_NONLINEAR,

    // TODO: document
    DisplayP3NonLinear = DISPLAY_P3_NONLINEAR_EXT
    RequiresOneOf([
        RequiresAllOf([InstanceExtension(ext_swapchain_colorspace)]),
    ]),

    // TODO: document
    ExtendedSrgbLinear = EXTENDED_SRGB_LINEAR_EXT
    RequiresOneOf([
        RequiresAllOf([InstanceExtension(ext_swapchain_colorspace)]),
    ]),

    // TODO: document
    ExtendedSrgbNonLinear = EXTENDED_SRGB_NONLINEAR_EXT
    RequiresOneOf([
        RequiresAllOf([InstanceExtension(ext_swapchain_colorspace)]),
    ]),

    // TODO: document
    DisplayP3Linear = DISPLAY_P3_LINEAR_EXT
    RequiresOneOf([
        RequiresAllOf([InstanceExtension(ext_swapchain_colorspace)]),
    ]),

    // TODO: document
    DciP3NonLinear = DCI_P3_NONLINEAR_EXT
    RequiresOneOf([
        RequiresAllOf([InstanceExtension(ext_swapchain_colorspace)]),
    ]),

    // TODO: document
    Bt709Linear = BT709_LINEAR_EXT
    RequiresOneOf([
        RequiresAllOf([InstanceExtension(ext_swapchain_colorspace)]),
    ]),

    // TODO: document
    Bt709NonLinear = BT709_NONLINEAR_EXT
    RequiresOneOf([
        RequiresAllOf([InstanceExtension(ext_swapchain_colorspace)]),
    ]),

    // TODO: document
    Bt2020Linear = BT2020_LINEAR_EXT
    RequiresOneOf([
        RequiresAllOf([InstanceExtension(ext_swapchain_colorspace)]),
    ]),

    // TODO: document
    Hdr10St2084 = HDR10_ST2084_EXT
    RequiresOneOf([
        RequiresAllOf([InstanceExtension(ext_swapchain_colorspace)]),
    ]),

    // TODO: document
    DolbyVision = DOLBYVISION_EXT
    RequiresOneOf([
        RequiresAllOf([InstanceExtension(ext_swapchain_colorspace)]),
    ]),

    // TODO: document
    Hdr10Hlg = HDR10_HLG_EXT
    RequiresOneOf([
        RequiresAllOf([InstanceExtension(ext_swapchain_colorspace)]),
    ]),

    // TODO: document
    AdobeRgbLinear = ADOBERGB_LINEAR_EXT
    RequiresOneOf([
        RequiresAllOf([InstanceExtension(ext_swapchain_colorspace)]),
    ]),

    // TODO: document
    AdobeRgbNonLinear = ADOBERGB_NONLINEAR_EXT
    RequiresOneOf([
        RequiresAllOf([InstanceExtension(ext_swapchain_colorspace)]),
    ]),

    // TODO: document
    PassThrough = PASS_THROUGH_EXT
    RequiresOneOf([
        RequiresAllOf([InstanceExtension(ext_swapchain_colorspace)]),
    ]),

    // TODO: document
    DisplayNative = DISPLAY_NATIVE_AMD
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(amd_display_native_hdr)]),
    ]),
}

/// Parameters for [`PhysicalDevice::surface_capabilities`] and
/// [`PhysicalDevice::surface_formats`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SurfaceInfo<'a> {
    /// If this is `Some`, the
    /// [`ext_surface_maintenance1`](crate::instance::InstanceExtensions::ext_surface_maintenance1)
    /// extension must be enabled on the instance.
    ///
    /// The default value is `None`.
    pub present_mode: Option<PresentMode>,

    /// If this is not [`FullScreenExclusive::Default`], then the
    /// [`ext_full_screen_exclusive`](crate::device::DeviceExtensions::ext_full_screen_exclusive)
    /// extension must be supported by the physical device.
    ///
    /// If the surface is a Win32 surface, and this is
    /// [`FullScreenExclusive::ApplicationControlled`], then `win32_monitor` must be `Some`.
    ///
    /// The default value is [`FullScreenExclusive::Default`].
    pub full_screen_exclusive: FullScreenExclusive,

    /// If this is `Some`, then then `full_screen_exclusive` must not be
    /// [`FullScreenExclusive::Default`], and the surface must be a Win32 surface.
    ///
    /// The default value is `None`.
    pub win32_monitor: Option<Win32Monitor>,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for SurfaceInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> SurfaceInfo<'a> {
    /// Returns a default `SurfaceInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            present_mode: None,
            full_screen_exclusive: FullScreenExclusive::Default,
            win32_monitor: None,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(
        &self,
        physical_device: &PhysicalDevice,
    ) -> Result<(), Box<ValidationError>> {
        let &Self {
            present_mode,
            full_screen_exclusive,
            win32_monitor,
            _ne: _,
        } = self;

        if let Some(present_mode) = present_mode {
            if !physical_device
                .instance()
                .enabled_extensions()
                .ext_surface_maintenance1
            {
                return Err(Box::new(ValidationError {
                    context: "present_mode".into(),
                    problem: "is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                        Requires::InstanceExtension("ext_surface_maintenance1"),
                    ])]),
                    ..Default::default()
                }));
            }

            present_mode
                .validate_physical_device(physical_device)
                .map_err(|err| {
                    err.add_context("present_mode")
                        .set_vuids(&["VUID-VkSurfacePresentModeEXT-presentMode-parameter"])
                })?;
        }

        if full_screen_exclusive != FullScreenExclusive::Default {
            if !physical_device
                .supported_extensions()
                .ext_full_screen_exclusive
            {
                return Err(Box::new(ValidationError {
                    context: "full_screen_exclusive".into(),
                    problem: "is not `FullScreenExclusive::Default`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                        "ext_full_screen_exclusive",
                    )])]),
                    ..Default::default()
                }));
            }
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

    pub(crate) fn to_vk2(
        &self,
        surface_vk: vk::SurfaceKHR,
        extensions_vk: &'a mut SurfaceInfo2ExtensionsVk,
    ) -> vk::PhysicalDeviceSurfaceInfo2KHR<'a> {
        let mut val_vk = vk::PhysicalDeviceSurfaceInfo2KHR::default().surface(surface_vk);

        let SurfaceInfo2ExtensionsVk {
            full_screen_exclusive_vk,
            full_screen_exclusive_win32_vk,
            present_mode_vk,
        } = extensions_vk;

        if let Some(next) = full_screen_exclusive_vk {
            val_vk = val_vk.push(next);
        }

        if let Some(next) = full_screen_exclusive_win32_vk {
            val_vk = val_vk.push(next);
        }

        if let Some(next) = present_mode_vk {
            val_vk = val_vk.push(next);
        }

        val_vk
    }

    pub(crate) fn to_vk2_extensions(&self) -> SurfaceInfo2ExtensionsVk {
        let &Self {
            present_mode,
            full_screen_exclusive,
            win32_monitor,
            _ne,
        } = self;

        let full_screen_exclusive_vk = (full_screen_exclusive != FullScreenExclusive::Default)
            .then(|| {
                vk::SurfaceFullScreenExclusiveInfoEXT::default()
                    .full_screen_exclusive(full_screen_exclusive.into())
            });

        let full_screen_exclusive_win32_vk = win32_monitor.map(|win32_monitor| {
            vk::SurfaceFullScreenExclusiveWin32InfoEXT::default().hmonitor(win32_monitor.0)
        });

        let present_mode_vk = present_mode.map(|present_mode| {
            vk::SurfacePresentModeEXT::default().present_mode(present_mode.into())
        });

        SurfaceInfo2ExtensionsVk {
            full_screen_exclusive_vk,
            full_screen_exclusive_win32_vk,
            present_mode_vk,
        }
    }

    pub(crate) fn to_owned(&self) -> SurfaceInfo<'static> {
        SurfaceInfo {
            _ne: crate::NE,
            ..*self
        }
    }
}

pub(crate) struct SurfaceInfo2ExtensionsVk {
    pub(crate) full_screen_exclusive_vk: Option<vk::SurfaceFullScreenExclusiveInfoEXT<'static>>,
    pub(crate) full_screen_exclusive_win32_vk:
        Option<vk::SurfaceFullScreenExclusiveWin32InfoEXT<'static>>,
    pub(crate) present_mode_vk: Option<vk::SurfacePresentModeEXT<'static>>,
}

/// The capabilities of a surface when used by a physical device.
///
/// You have to match these capabilities when you create a swapchain.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SurfaceCapabilities {
    /// Minimum number of images that must be present in the swapchain.
    pub min_image_count: u32,

    /// Maximum number of images that must be present in the swapchain, or `None` if there is no
    /// maximum value. Note that "no maximum" doesn't mean that you can set a very high value, as
    /// you may still get out of memory errors.
    pub max_image_count: Option<u32>,

    /// The current dimensions of the surface.
    ///
    /// `None` means that the surface's dimensions will depend on the dimensions of the swapchain.
    pub current_extent: Option<[u32; 2]>,

    /// Minimum width and height of a swapchain that uses this surface.
    pub min_image_extent: [u32; 2],

    /// Maximum width and height of a swapchain that uses this surface.
    pub max_image_extent: [u32; 2],

    /// Maximum number of image layers if you create an image array. The minimum is 1.
    pub max_image_array_layers: u32,

    /// List of transforms supported for the swapchain.
    pub supported_transforms: SurfaceTransforms,

    /// Current transform used by the surface.
    pub current_transform: SurfaceTransform,

    /// List of composite alpha modes supports for the swapchain.
    pub supported_composite_alpha: CompositeAlphas,

    /// List of image usages that are supported for images of the swapchain. Only
    /// the `color_attachment` usage is guaranteed to be supported.
    pub supported_usage_flags: ImageUsage,

    /// When [`SurfaceInfo::present_mode`] is provided,
    /// lists that present mode and any modes that are compatible with that present mode.
    ///
    /// If [`SurfaceInfo::present_mode`] was not provided, the value will be empty.
    pub compatible_present_modes: SmallVec<[PresentMode; PresentMode::COUNT]>,

    /// When [`SurfaceInfo::present_mode`] is provided,
    /// the supported present scaling modes for the queried present mode.
    ///
    /// If [`SurfaceInfo::present_mode`] was not provided, the value will be empty.
    pub supported_present_scaling: PresentScalingFlags,

    /// When [`SurfaceInfo::present_mode`] is provided,
    /// the supported present gravity modes, horizontally and vertically,
    /// for the queried present mode.
    ///
    /// If [`SurfaceInfo::present_mode`] was not provided, both values will be empty.
    pub supported_present_gravity: [PresentGravityFlags; 2],

    /// When [`SurfaceInfo::present_mode`] is provided,
    /// the smallest allowed extent for a swapchain, if it uses the queried present mode, and
    /// one of the scaling modes in `supported_present_scaling`.
    ///
    /// This is never greater than [`SurfaceCapabilities::min_image_extent`].
    ///
    /// `None` means that the surface's dimensions will depend on the dimensions of the swapchain.
    ///
    /// If [`SurfaceInfo::present_mode`] was not provided, this is will be equal to
    /// `min_image_extent`.
    pub min_scaled_image_extent: Option<[u32; 2]>,

    /// When [`SurfaceInfo::present_mode`] is provided,
    /// the largest allowed extent for a swapchain, if it uses the queried present mode, and
    /// one of the scaling modes in `supported_present_scaling`.
    ///
    /// This is never less than [`SurfaceCapabilities::max_image_extent`].
    ///
    /// `None` means that the surface's dimensions will depend on the dimensions of the swapchain.
    ///
    /// If [`SurfaceInfo::present_mode`] was not provided, this is will be equal to
    /// `max_image_extent`.
    pub max_scaled_image_extent: Option<[u32; 2]>,

    /// Whether creating a protected swapchain is supported.
    pub supports_protected: bool,

    /// Whether full-screen exclusivity is supported.
    ///
    /// This is currently only reported if [`SurfaceInfo::win32_monitor`] is `Some`.
    pub full_screen_exclusive_supported: bool,
}

impl SurfaceCapabilities {
    pub(crate) fn to_mut_vk2<'a>(
        extensions_vk: &'a mut SurfaceCapabilities2ExtensionsVk<'_>,
    ) -> vk::SurfaceCapabilities2KHR<'a> {
        let mut val_vk = vk::SurfaceCapabilities2KHR::default();

        let SurfaceCapabilities2ExtensionsVk {
            full_screen_exclusive_vk,
            present_mode_compatibility_vk: present_modes_vk,
            present_scaling_vk,
            protected_vk,
        } = extensions_vk;

        if let Some(next) = full_screen_exclusive_vk {
            val_vk = val_vk.push(next);
        }

        if let Some(next) = present_modes_vk {
            val_vk = val_vk.push(next);
        }

        if let Some(next) = present_scaling_vk {
            val_vk = val_vk.push(next);
        }

        if let Some(next) = protected_vk {
            val_vk = val_vk.push(next);
        }

        val_vk
    }

    pub(crate) fn to_mut_vk2_extensions<'a>(
        fields1_vk: &'a mut SurfaceCapabilities2Fields1Vk,
        physical_device: &PhysicalDevice,
        surface_info: &SurfaceInfo<'_>,
    ) -> SurfaceCapabilities2ExtensionsVk<'a> {
        let SurfaceCapabilities2Fields1Vk { present_modes_vk } = fields1_vk;

        // VUID-vkGetPhysicalDeviceSurfaceCapabilities2KHR-pNext-02671
        let full_screen_exclusive_vk = (surface_info.win32_monitor.is_some())
            .then(vk::SurfaceCapabilitiesFullScreenExclusiveEXT::default);

        let present_mode_compatibility_vk = (surface_info.present_mode.is_some()).then(|| {
            vk::SurfacePresentModeCompatibilityEXT::default().present_modes(present_modes_vk)
        });

        let present_scaling_vk = (surface_info.present_mode.is_some())
            .then(vk::SurfacePresentScalingCapabilitiesEXT::default);

        let protected_vk = (physical_device
            .instance()
            .enabled_extensions()
            .khr_surface_protected_capabilities)
            .then(vk::SurfaceProtectedCapabilitiesKHR::default);

        SurfaceCapabilities2ExtensionsVk {
            full_screen_exclusive_vk,
            present_mode_compatibility_vk,
            present_scaling_vk,
            protected_vk,
        }
    }

    pub(crate) fn to_mut_vk2_fields() -> SurfaceCapabilities2Fields1Vk {
        let present_modes_vk = Default::default();

        SurfaceCapabilities2Fields1Vk { present_modes_vk }
    }

    pub(crate) fn from_vk2(
        val_vk: &vk::SurfaceCapabilities2KHR<'_>,
        fields1_vk: &SurfaceCapabilities2Fields1Vk,
        extensions_vk: &SurfaceCapabilities2ExtensionsVk<'_>,
    ) -> Self {
        let &vk::SurfaceCapabilities2KHR {
            surface_capabilities:
                vk::SurfaceCapabilitiesKHR {
                    min_image_count,
                    max_image_count,
                    current_extent,
                    min_image_extent,
                    max_image_extent,
                    max_image_array_layers,
                    supported_transforms,
                    current_transform,
                    supported_composite_alpha,
                    supported_usage_flags,
                },
            ..
        } = val_vk;

        let mut val = Self {
            min_image_count,
            max_image_count: (max_image_count != 0).then_some(max_image_count),
            current_extent: filter_max(current_extent),
            min_image_extent: [min_image_extent.width, min_image_extent.height],
            max_image_extent: [max_image_extent.width, max_image_extent.height],
            max_image_array_layers,
            supported_transforms: supported_transforms.into(),
            current_transform: SurfaceTransforms::from(current_transform)
                .into_iter()
                .next()
                .unwrap(), // TODO:
            supported_composite_alpha: supported_composite_alpha.into(),
            supported_usage_flags: ImageUsage::from(supported_usage_flags),

            compatible_present_modes: Default::default(),
            supported_present_scaling: Default::default(),
            supported_present_gravity: Default::default(),
            min_scaled_image_extent: Some([min_image_extent.width, min_image_extent.height]),
            max_scaled_image_extent: Some([max_image_extent.width, max_image_extent.height]),
            supports_protected: false,
            full_screen_exclusive_supported: false,
        };

        let SurfaceCapabilities2ExtensionsVk {
            full_screen_exclusive_vk,
            present_mode_compatibility_vk,
            present_scaling_vk,
            protected_vk,
        } = extensions_vk;
        let SurfaceCapabilities2Fields1Vk { present_modes_vk } = fields1_vk;

        if let Some(val_vk) = full_screen_exclusive_vk {
            let &vk::SurfaceCapabilitiesFullScreenExclusiveEXT {
                full_screen_exclusive_supported,
                ..
            } = val_vk;

            val = Self {
                full_screen_exclusive_supported: full_screen_exclusive_supported != 0,
                ..val
            };
        }

        if let Some(val_vk) = present_mode_compatibility_vk {
            let &vk::SurfacePresentModeCompatibilityEXT {
                present_mode_count, ..
            } = val_vk;

            val = Self {
                compatible_present_modes: present_modes_vk[..present_mode_count as usize]
                    .iter()
                    .copied()
                    .map(PresentMode::try_from)
                    .filter_map(Result::ok)
                    .collect(),
                ..val
            };
        }

        if let Some(val_vk) = present_scaling_vk {
            let &vk::SurfacePresentScalingCapabilitiesEXT {
                supported_present_scaling,
                supported_present_gravity_x,
                supported_present_gravity_y,
                min_scaled_image_extent,
                max_scaled_image_extent,
                ..
            } = val_vk;

            val = Self {
                supported_present_scaling: supported_present_scaling.into(),
                supported_present_gravity: [
                    supported_present_gravity_x.into(),
                    supported_present_gravity_y.into(),
                ],
                min_scaled_image_extent: filter_max(min_scaled_image_extent),
                max_scaled_image_extent: filter_max(max_scaled_image_extent),
                ..val
            };
        }

        if let Some(val_vk) = protected_vk {
            let &vk::SurfaceProtectedCapabilitiesKHR {
                supports_protected, ..
            } = val_vk;

            val = Self {
                supports_protected: supports_protected != 0,
                ..val
            };
        }

        val
    }
}

pub(crate) struct SurfaceCapabilities2ExtensionsVk<'a> {
    pub(crate) full_screen_exclusive_vk:
        Option<vk::SurfaceCapabilitiesFullScreenExclusiveEXT<'static>>,
    pub(crate) present_mode_compatibility_vk: Option<vk::SurfacePresentModeCompatibilityEXT<'a>>,
    pub(crate) present_scaling_vk: Option<vk::SurfacePresentScalingCapabilitiesEXT<'static>>,
    pub(crate) protected_vk: Option<vk::SurfaceProtectedCapabilitiesKHR<'static>>,
}

impl SurfaceCapabilities2ExtensionsVk<'_> {
    pub(crate) fn unborrow(self) -> SurfaceCapabilities2ExtensionsVk<'static> {
        let Self {
            full_screen_exclusive_vk,
            present_mode_compatibility_vk,
            present_scaling_vk,
            protected_vk,
        } = self;

        let present_mode_compatibility_vk =
            present_mode_compatibility_vk.map(|val_vk| vk::SurfacePresentModeCompatibilityEXT {
                _marker: PhantomData,
                ..val_vk
            });

        SurfaceCapabilities2ExtensionsVk {
            full_screen_exclusive_vk,
            present_mode_compatibility_vk,
            present_scaling_vk,
            protected_vk,
        }
    }
}

pub(crate) struct SurfaceCapabilities2Fields1Vk {
    pub(crate) present_modes_vk: [vk::PresentModeKHR; PresentMode::COUNT],
}

fn filter_max(extent: vk::Extent2D) -> Option<[u32; 2]> {
    if extent.width == u32::MAX && extent.height == u32::MAX {
        None
    } else {
        Some([extent.width, extent.height])
    }
}

/// Error that can happen when creating a [`Surface`] from a window.
#[cfg(feature = "raw_window_handle")]
#[derive(Clone, Debug)]
pub enum FromWindowError {
    /// Retrieving the window or display handle failed.
    RetrieveHandle(HandleError),
    /// Creating the surface failed.
    CreateSurface(VulkanError),
}

#[cfg(feature = "raw_window_handle")]
impl FromWindowError {
    /// Returns the inner `CreateSurface` value, or panics if it contains `RetrieveHandle`.
    #[inline(always)]
    #[track_caller]
    pub fn unwrap(self) -> VulkanError {
        match self {
            Self::RetrieveHandle(err) => {
                panic!("called `FromWindowError::unwrap` on a `RetrieveHandle` value: {err:?}");
            }
            Self::CreateSurface(err) => err,
        }
    }
}

#[cfg(feature = "raw_window_handle")]
impl Error for FromWindowError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::RetrieveHandle(err) => Some(err),
            Self::CreateSurface(err) => Some(err),
        }
    }
}

#[cfg(feature = "raw_window_handle")]
impl Display for FromWindowError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::RetrieveHandle(_) => {
                write!(f, "retrieving the window or display handle has failed")
            }
            Self::CreateSurface(_) => write!(f, "creating the surface has failed"),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        swapchain::Surface, Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError,
    };
    use std::ptr;

    #[test]
    fn khr_win32_surface_ext_missing() {
        let instance = instance!();
        match unsafe { Surface::try_from_win32(&instance, 0, 0, None) } {
            Err(Validated::ValidationError(err))
                if matches!(
                    *err,
                    ValidationError {
                        requires_one_of: RequiresOneOf([RequiresAllOf([
                            Requires::InstanceExtension("khr_win32_surface")
                        ])]),
                        ..
                    }
                ) => {}
            _ => panic!(),
        }
    }

    #[test]
    fn khr_xcb_surface_ext_missing() {
        let instance = instance!();
        match unsafe { Surface::try_from_xcb(&instance, ptr::null_mut(), 0, None) } {
            Err(Validated::ValidationError(err))
                if matches!(
                    *err,
                    ValidationError {
                        requires_one_of: RequiresOneOf([RequiresAllOf([
                            Requires::InstanceExtension("khr_xcb_surface")
                        ])]),
                        ..
                    }
                ) => {}
            _ => panic!(),
        }
    }

    #[test]
    fn khr_xlib_surface_ext_missing() {
        let instance = instance!();
        match unsafe { Surface::try_from_xlib(&instance, ptr::null_mut(), 0, None) } {
            Err(Validated::ValidationError(err))
                if matches!(
                    *err,
                    ValidationError {
                        requires_one_of: RequiresOneOf([RequiresAllOf([
                            Requires::InstanceExtension("khr_xlib_surface")
                        ])]),
                        ..
                    }
                ) => {}
            _ => panic!(),
        }
    }

    #[test]
    fn khr_wayland_surface_ext_missing() {
        let instance = instance!();
        match unsafe {
            Surface::try_from_wayland(&instance, ptr::null_mut(), ptr::null_mut(), None)
        } {
            Err(Validated::ValidationError(err))
                if matches!(
                    *err,
                    ValidationError {
                        requires_one_of: RequiresOneOf([RequiresAllOf([
                            Requires::InstanceExtension("khr_wayland_surface")
                        ])]),
                        ..
                    }
                ) => {}
            _ => panic!(),
        }
    }

    #[test]
    fn khr_android_surface_ext_missing() {
        let instance = instance!();
        match unsafe { Surface::try_from_android(&instance, ptr::null_mut(), None) } {
            Err(Validated::ValidationError(err))
                if matches!(
                    *err,
                    ValidationError {
                        requires_one_of: RequiresOneOf([RequiresAllOf([
                            Requires::InstanceExtension("khr_android_surface")
                        ])]),
                        ..
                    }
                ) => {}
            _ => panic!(),
        }
    }
}
