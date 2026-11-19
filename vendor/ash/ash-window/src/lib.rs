#![warn(trivial_casts, trivial_numeric_casts, unused_qualifications)]

use std::os::raw::c_char;

use ash::{
    ext::metal_surface,
    khr::{android_surface, surface, wayland_surface, win32_surface, xcb_surface, xlib_surface},
    ohos::surface as ohos_surface,
    vk, Entry, Instance, VkResult,
};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

#[derive(Clone)]
enum SurfaceExtension {
    Windows(win32_surface::Instance),
    Wayland(
        raw_window_handle::WaylandDisplayHandle,
        wayland_surface::Instance,
    ),
    Xlib(raw_window_handle::XlibDisplayHandle, xlib_surface::Instance),
    Xcb(raw_window_handle::XcbDisplayHandle, xcb_surface::Instance),
    Android(android_surface::Instance),
    Ohos(ohos_surface::Instance),
    #[cfg(target_os = "macos")]
    AppKit(metal_surface::Instance),
    #[cfg(target_os = "ios")]
    UiKit(metal_surface::Instance),
}

/// Holder for a loaded platform-specific Vulkan `Surface` extension, used to create surfaces.
///
/// Also stores the platform-specific [`raw_window_handle::RawDisplayHandle`] variant if necessary
/// to create surfaces, identifying the selected display server handle that was used to load the
/// relevant extension when creating a [`SurfaceFactory`].
#[derive(Clone)]
pub struct SurfaceFactory(SurfaceExtension);

impl SurfaceFactory {
    /// Load the relevant surface extension for a given [`RawDisplayHandle`].
    ///
    /// `instance` must have been created with platform specific surface extensions enabled, acquired
    /// through [`enumerate_required_extensions()`].
    pub fn new(
        entry: &Entry,
        instance: &Instance,
        display_handle: RawDisplayHandle,
    ) -> VkResult<Self> {
        Ok(Self(match display_handle {
            RawDisplayHandle::Windows(_) => {
                SurfaceExtension::Windows(win32_surface::Instance::load(entry, instance))
            }

            RawDisplayHandle::Wayland(display) => {
                SurfaceExtension::Wayland(display, wayland_surface::Instance::load(entry, instance))
            }

            RawDisplayHandle::Xlib(display) => {
                SurfaceExtension::Xlib(display, xlib_surface::Instance::load(entry, instance))
            }

            RawDisplayHandle::Xcb(display) => {
                SurfaceExtension::Xcb(display, xcb_surface::Instance::load(entry, instance))
            }

            RawDisplayHandle::Android(_) => {
                SurfaceExtension::Android(android_surface::Instance::load(entry, instance))
            }

            RawDisplayHandle::Ohos(_) => {
                SurfaceExtension::Ohos(ohos_surface::Instance::load(entry, instance))
            }

            #[cfg(target_os = "macos")]
            RawDisplayHandle::AppKit(_) => {
                SurfaceExtension::AppKit(metal_surface::Instance::load(entry, instance))
            }

            #[cfg(target_os = "ios")]
            RawDisplayHandle::UiKit(_) => {
                SurfaceExtension::UiKit(metal_surface::Instance::load(entry, instance))
            }

            _ => return Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT),
        }))
    }

    /// Create a surface from a raw window handle that is compatible with the `display_handle` this
    /// [`SurfaceFactory`] object was created with.
    ///
    /// # Safety
    ///
    /// There is a [parent/child relation] between [`Instance`] and [`Entry`], and the resulting
    /// [`vk::SurfaceKHR`].  The application must not [destroy][Instance::destroy_instance()] these
    /// parent objects before first [destroying][surface::Instance::destroy_surface()] the returned
    /// [`vk::SurfaceKHR`] child object.  [`vk::SurfaceKHR`] does _not_ implement [drop][drop()]
    /// semantics and can only be destroyed via [`destroy_surface()`][surface::Instance::destroy_surface()].
    ///
    /// See the [`Entry::create_instance()`] documentation for more destruction ordering rules on
    /// [`Instance`].
    ///
    /// The window represented by `window_handle` must be associated with the display connection
    /// that was passed to `display_handle` in [`Self::new()`].
    ///
    //FUTURE: This display_handle lifetime can be implicitly represented by storing DisplayHandle<'a>
    /// `window_handle` and `display_handle` must be associated with a valid window and display
    /// connection, which must not be destroyed for the lifetime of the returned [`vk::SurfaceKHR`].
    ///
    /// [parent/child relation]: https://docs.vulkan.org/spec/latest/chapters/fundamentals.html#fundamentals-objectmodel-lifetime
    pub unsafe fn create_surface(
        &self,
        window_handle: RawWindowHandle,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) -> VkResult<vk::SurfaceKHR> {
        match (&self.0, window_handle) {
            (SurfaceExtension::Windows(surface_fn), RawWindowHandle::Win32(window)) => {
                let surface_desc = vk::Win32SurfaceCreateInfoKHR::default()
                    .hwnd(window.hwnd.get())
                    .hinstance(
                        window
                            .hinstance
                            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?
                            .get(),
                    );
                surface_fn.create_win32_surface(&surface_desc, allocation_callbacks)
            }

            (SurfaceExtension::Wayland(display, surface_fn), RawWindowHandle::Wayland(window)) => {
                let surface_desc = vk::WaylandSurfaceCreateInfoKHR::default()
                    .display(display.display.as_ptr())
                    .surface(window.surface.as_ptr());
                surface_fn.create_wayland_surface(&surface_desc, allocation_callbacks)
            }

            (SurfaceExtension::Xlib(display, surface_fn), RawWindowHandle::Xlib(window)) => {
                let surface_desc = vk::XlibSurfaceCreateInfoKHR::default()
                    .dpy(
                        display
                            .display
                            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?
                            .as_ptr(),
                    )
                    .window(window.window);
                surface_fn.create_xlib_surface(&surface_desc, allocation_callbacks)
            }

            (SurfaceExtension::Xcb(display, surface_fn), RawWindowHandle::Xcb(window)) => {
                let surface_desc = vk::XcbSurfaceCreateInfoKHR::default()
                    .connection(
                        display
                            .connection
                            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?
                            .as_ptr(),
                    )
                    .window(window.window.get());
                surface_fn.create_xcb_surface(&surface_desc, allocation_callbacks)
            }

            (SurfaceExtension::Android(surface_fn), RawWindowHandle::AndroidNdk(window)) => {
                let surface_desc = vk::AndroidSurfaceCreateInfoKHR::default()
                    .window(window.a_native_window.as_ptr());
                surface_fn.create_android_surface(&surface_desc, allocation_callbacks)
            }

            (SurfaceExtension::Ohos(surface_fn), RawWindowHandle::OhosNdk(window)) => {
                let surface_desc =
                    vk::SurfaceCreateInfoOHOS::default().window(window.native_window.as_ptr());
                surface_fn.create_surface(&surface_desc, allocation_callbacks)
            }

            #[cfg(target_os = "macos")]
            (SurfaceExtension::AppKit(surface_fn), RawWindowHandle::AppKit(window)) => {
                use raw_window_metal::{appkit, Layer};

                let layer = match appkit::metal_layer_from_handle(window) {
                    Layer::Existing(layer) | Layer::Allocated(layer) => layer.cast(),
                };

                let surface_desc = vk::MetalSurfaceCreateInfoEXT::default().layer(&*layer);
                surface_fn.create_metal_surface(&surface_desc, allocation_callbacks)
            }

            #[cfg(target_os = "ios")]
            (SurfaceExtension::UiKit(surface_fn), RawWindowHandle::UiKit(window)) => {
                use raw_window_metal::{uikit, Layer};

                let layer = match uikit::metal_layer_from_handle(window) {
                    Layer::Existing(layer) | Layer::Allocated(layer) => layer.cast(),
                };

                let surface_desc = vk::MetalSurfaceCreateInfoEXT::default().layer(&*layer);
                surface_fn.create_metal_surface(&surface_desc, allocation_callbacks)
            }

            _ => Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT),
        }
    }
}

/// Query the required instance extensions for creating a surface from a raw display handle.
///
/// This [`RawDisplayHandle`] can typically be acquired from a window, but is usually also
/// accessible earlier through an "event loop" concept to allow querying required instance
/// extensions and creation of a compatible Vulkan instance prior to creating a window.
///
/// The returned extensions will include all extension dependencies.
pub fn enumerate_required_extensions(
    display_handle: RawDisplayHandle,
) -> VkResult<&'static [*const c_char]> {
    let extensions = match display_handle {
        RawDisplayHandle::Windows(_) => {
            const WINDOWS_EXTS: [*const c_char; 2] =
                [surface::NAME.as_ptr(), win32_surface::NAME.as_ptr()];
            &WINDOWS_EXTS
        }

        RawDisplayHandle::Wayland(_) => {
            const WAYLAND_EXTS: [*const c_char; 2] =
                [surface::NAME.as_ptr(), wayland_surface::NAME.as_ptr()];
            &WAYLAND_EXTS
        }

        RawDisplayHandle::Xlib(_) => {
            const XLIB_EXTS: [*const c_char; 2] =
                [surface::NAME.as_ptr(), xlib_surface::NAME.as_ptr()];
            &XLIB_EXTS
        }

        RawDisplayHandle::Xcb(_) => {
            const XCB_EXTS: [*const c_char; 2] =
                [surface::NAME.as_ptr(), xcb_surface::NAME.as_ptr()];
            &XCB_EXTS
        }

        RawDisplayHandle::Android(_) => {
            const ANDROID_EXTS: [*const c_char; 2] =
                [surface::NAME.as_ptr(), android_surface::NAME.as_ptr()];
            &ANDROID_EXTS
        }

        RawDisplayHandle::Ohos(_) => {
            const OHOS_EXTS: [*const c_char; 2] =
                [surface::NAME.as_ptr(), ohos_surface::NAME.as_ptr()];
            &OHOS_EXTS
        }

        RawDisplayHandle::AppKit(_) | RawDisplayHandle::UiKit(_) => {
            const METAL_EXTS: [*const c_char; 2] =
                [surface::NAME.as_ptr(), metal_surface::NAME.as_ptr()];
            &METAL_EXTS
        }

        _ => return Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT),
    };

    Ok(extensions)
}
