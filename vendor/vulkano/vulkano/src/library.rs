//! Vulkan library loading system.
//!
//! Before Vulkano can do anything, it first needs to find a library containing an implementation
//! of Vulkan. A Vulkan implementation is defined as a single `vkGetInstanceProcAddr` function,
//! which can be accessed through [`VulkanLibrary`].
//!
//! You can create a `VulkanLibrary` by [loading the default Vulkan library for the platform], by
//! [loading from a specific path], or by [using a statically-linked Vulkan library]. Once you have
//! created a `VulkanLibrary`, you can use it to create an [`Instance`].
//!
//! [loading the default Vulkan library for the platform]: VulkanLibrary::new
//! [loading from a specific path]: VulkanLibrary::from_path
//! [using a statically-linked Vulkan library]: statically_linked_vulkan_library
//! [`Instance`]: crate::instance::Instance

pub use crate::fns::EntryFunctions;
use crate::{
    instance::{InstanceExtensions, LayerProperties},
    ExtensionProperties, Version, VulkanError,
};
use ash::vk;
use libloading::{Error as LibloadingError, Library};
use std::{
    error::Error,
    ffi::CString,
    fmt::{Debug, Display, Error as FmtError, Formatter},
    mem,
    os::raw::c_char,
    path::Path,
    ptr,
    sync::Arc,
};

/// A loaded library containing a valid Vulkan implementation.
#[derive(Debug)]
pub struct VulkanLibrary {
    get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    _library: Option<Library>,
    fns: EntryFunctions,

    api_version: Version,
    extension_properties: Vec<ExtensionProperties>,
    supported_extensions: InstanceExtensions,
}

impl VulkanLibrary {
    /// Creates a new `VulkanLibrary` by loading the default Vulkan library for this platform.
    ///
    /// # Safety
    ///
    /// - If there is a library at the default path for this platform, it must be a valid Vulkan
    ///   implementation.
    /// - Library loading is inherently unsafe.
    pub unsafe fn new() -> Result<Arc<Self>, LoadingError> {
        #[cfg(any(target_os = "ios", target_os = "tvos"))]
        {
            unsafe { crate::statically_linked_vulkan_library!() }
        }
        #[cfg(not(any(target_os = "ios", target_os = "tvos")))]
        {
            #[cfg(windows)]
            const PATHS: [&str; 1] = ["vulkan-1.dll"];
            #[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
            const PATHS: [&str; 1] = ["libvulkan.so.1"];
            #[cfg(target_os = "macos")]
            const PATHS: [&str; 6] = [
                "libvulkan.dylib",
                "libvulkan.1.dylib",
                "libMoltenVK.dylib",
                "vulkan.framework/vulkan",
                "MoltenVK.framework/MoltenVK",
                // Stock macOS no longer has `/usr/local/lib` in `LD_LIBRARY_PATH` like it used to,
                // but libraries (including MoltenVK installed through the Vulkan SDK) are still
                // installed here. Try the absolute path as a last resort.
                "/usr/local/lib/libvulkan.dylib",
            ];
            #[cfg(target_os = "android")]
            const PATHS: [&str; 2] = ["libvulkan.so.1", "libvulkan.so"];

            let mut err: Option<LoadingError> = None;

            for path in PATHS {
                match unsafe { Self::from_path(path) } {
                    Ok(library) => return Ok(library),
                    Err(e) => err = Some(e),
                }
            }

            Err(err.unwrap())
        }
    }

    /// Creates a new `VulkanLibrary` loaded from the given path.
    ///
    /// # Safety
    ///
    /// - If there is a library at the given `path`, it must be a valid Vulkan implementation.
    /// - Library loading is inherently unsafe.
    pub unsafe fn from_path(path: impl AsRef<Path>) -> Result<Arc<Self>, LoadingError> {
        unsafe { Self::from_path_inner(path.as_ref()) }
    }

    unsafe fn from_path_inner(path: &Path) -> Result<Arc<Self>, LoadingError> {
        let library = unsafe { Library::new(path) }.map_err(LoadingError::LibraryLoadFailure)?;

        let get_instance_proc_addr = *unsafe { library.get(b"vkGetInstanceProcAddr") }
            .map_err(LoadingError::LibraryLoadFailure)?;

        unsafe { Self::from_loader_inner(get_instance_proc_addr, Some(library)) }
    }

    /// Creates a Vulkan library from an existing loader.
    ///
    /// # Safety
    ///
    /// - `get_instance_proc_addr` must be the loader of a valid Vulkan implementation.
    /// - `get_instance_proc_addr` and any function pointers loaded through it must be valid for as
    ///   long as the returned `VulkanLibrary` exists.
    pub unsafe fn from_loader(
        get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    ) -> Result<Arc<Self>, LoadingError> {
        unsafe { Self::from_loader_inner(get_instance_proc_addr, None) }
    }

    unsafe fn from_loader_inner(
        get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
        library: Option<Library>,
    ) -> Result<Arc<Self>, LoadingError> {
        let fns = EntryFunctions::load(|name| {
            unsafe { get_instance_proc_addr(vk::Instance::null(), name.as_ptr()) }
                .map_or(ptr::null(), |func| func as _)
        });

        let api_version = unsafe { Self::get_api_version(get_instance_proc_addr) }?;
        let extension_properties = unsafe { Self::get_extension_properties(&fns, None) }?;
        let supported_extensions = InstanceExtensions::from_vk(
            extension_properties
                .iter()
                .map(|property| property.extension_name.as_str()),
        );

        Ok(Arc::new(VulkanLibrary {
            get_instance_proc_addr,
            _library: library,
            fns,
            api_version,
            extension_properties,
            supported_extensions,
        }))
    }

    unsafe fn get_api_version(
        get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    ) -> Result<Version, VulkanError> {
        // Per the Vulkan spec:
        // If the vkGetInstanceProcAddr returns NULL for vkEnumerateInstanceVersion, it is a
        // Vulkan 1.0 implementation. Otherwise, the application can call
        // vkEnumerateInstanceVersion to determine the version of Vulkan.

        let func = unsafe {
            get_instance_proc_addr(vk::Instance::null(), c"vkEnumerateInstanceVersion".as_ptr())
        };

        let version = if func.is_some() {
            let func = unsafe {
                mem::transmute::<vk::PFN_vkVoidFunction, vk::PFN_vkEnumerateInstanceVersion>(func)
            };
            let mut api_version = 0;
            unsafe { func(&mut api_version) }
                .result()
                .map_err(VulkanError::from)?;

            Version::from(api_version)
        } else {
            Version {
                major: 1,
                minor: 0,
                patch: 0,
            }
        };

        Ok(version)
    }

    unsafe fn get_extension_properties(
        fns: &EntryFunctions,
        layer: Option<&str>,
    ) -> Result<Vec<ExtensionProperties>, VulkanError> {
        let layer_vk = layer.map(|layer| CString::new(layer).unwrap());

        loop {
            let mut count = 0;
            unsafe {
                (fns.v1_0.enumerate_instance_extension_properties)(
                    layer_vk
                        .as_ref()
                        .map_or(ptr::null(), |layer| layer.as_ptr()),
                    &mut count,
                    ptr::null_mut(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;

            let mut output = Vec::with_capacity(count as usize);
            let result = unsafe {
                (fns.v1_0.enumerate_instance_extension_properties)(
                    layer_vk
                        .as_ref()
                        .map_or(ptr::null(), |layer| layer.as_ptr()),
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

    /// Returns the `get_instance_proc_addr` function pointer.
    ///
    /// The function pointer must not be used after `self` has been dropped.
    #[inline]
    pub fn loader(&self) -> vk::PFN_vkGetInstanceProcAddr {
        self.get_instance_proc_addr
    }

    /// Returns pointers to the raw global Vulkan functions of the library.
    #[inline]
    pub fn fns(&self) -> &EntryFunctions {
        &self.fns
    }

    /// Returns the highest Vulkan version that is supported for instances.
    #[inline]
    pub fn api_version(&self) -> Version {
        self.api_version
    }

    /// Returns the extension properties reported by the core library.
    #[inline]
    pub fn extension_properties(&self) -> &[ExtensionProperties] {
        &self.extension_properties
    }

    /// Returns the extensions that are supported by the core library.
    #[inline]
    pub fn supported_extensions(&self) -> &InstanceExtensions {
        &self.supported_extensions
    }

    /// Returns the list of layers that are available when creating an instance.
    ///
    /// In order to enable a layer, you need to pass its name (returned by
    /// `LayerProperties::name()`) when creating the [`Instance`].
    ///
    /// <div class="vulkano-alert-note">
    ///
    /// > Note
    /// >
    /// > The available layers may change between successive calls to this function, so each call
    /// > may return different results. It is possible that one of the layers enumerated here is no
    /// > longer available when you create the `Instance`. This will lead to an error when calling
    /// > `Instance::new`.
    ///
    /// </div>
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use vulkano::VulkanLibrary;
    ///
    /// let library = unsafe { VulkanLibrary::new() }.unwrap();
    ///
    /// for layer in library.layer_properties().unwrap() {
    ///     println!("Available layer: {}", layer.name());
    /// }
    /// ```
    ///
    /// [`Instance`]: crate::instance::Instance
    pub fn layer_properties(&self) -> Result<Vec<LayerProperties>, VulkanError> {
        let fns = self.fns();

        let layer_properties_vk = loop {
            let mut count = 0;
            unsafe { (fns.v1_0.enumerate_instance_layer_properties)(&mut count, ptr::null_mut()) }
                .result()
                .map_err(VulkanError::from)?;

            let mut properties_vk = Vec::with_capacity(count as usize);
            let result = unsafe {
                (fns.v1_0.enumerate_instance_layer_properties)(
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

        let layer_properties = layer_properties_vk
            .into_iter()
            .map(|p| LayerProperties { props: p })
            .collect();

        Ok(layer_properties)
    }

    /// Returns the extension properties that are reported by the given layer.
    #[inline]
    pub fn layer_extension_properties(
        &self,
        layer: &str,
    ) -> Result<Vec<ExtensionProperties>, VulkanError> {
        unsafe { Self::get_extension_properties(&self.fns, Some(layer)) }
    }

    /// Returns the extensions that are supported by the given layer.
    #[inline]
    pub fn supported_layer_extensions(
        &self,
        layer: &str,
    ) -> Result<InstanceExtensions, VulkanError> {
        let supported_extensions = InstanceExtensions::from_vk(
            self.layer_extension_properties(layer)?
                .iter()
                .map(|property| property.extension_name.as_str()),
        );
        Ok(supported_extensions)
    }

    /// Returns the union of the extensions that are supported by the core library and all
    /// the given layers.
    #[inline]
    pub fn supported_extensions_with_layers(
        &self,
        layers: &[&str],
    ) -> Result<InstanceExtensions, VulkanError> {
        layers
            .iter()
            .try_fold(self.supported_extensions, |extensions, layer| {
                self.supported_layer_extensions(layer)
                    .map(|layer_extensions| extensions.union(&layer_extensions))
            })
    }

    /// Calls the underlying `get_instance_proc_addr` function.
    #[inline]
    pub unsafe fn get_instance_proc_addr(
        &self,
        instance: vk::Instance,
        name: *const c_char,
    ) -> vk::PFN_vkVoidFunction {
        unsafe { (self.get_instance_proc_addr)(instance, name) }
    }
}

/// Creates a new `VulkanLibrary`, assuming that the Vulkan library is linked statically.
///
/// If you use this macro, you must link to a library that provides the `vkGetInstanceProcAddr`
/// symbol. Because of this, this is provided as a macro, as the macro expands to an `extern` block
/// that expects this symbol to exist.
///
/// # Safety
///
/// - The statically linked library must be a valid Vulkan implementation.
#[macro_export]
macro_rules! statically_linked_vulkan_library {
    () => {{
        unsafe extern "system" {
            fn vkGetInstanceProcAddr(
                instance: vk::Instance,
                pName: *const c_char,
            ) -> vk::PFN_vkVoidFunction;
        }

        $crate::VulkanLibrary::from_loader(vkGetInstanceProcAddr)
    }};
}

/// Error that can happen when loading a Vulkan library.
#[derive(Debug)]
pub enum LoadingError {
    /// Failed to load the Vulkan shared library.
    LibraryLoadFailure(LibloadingError),

    /// The Vulkan driver returned an error and was unable to complete the operation.
    VulkanError(VulkanError),
}

impl Error for LoadingError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            //Self::LibraryLoadFailure(err) => Some(err),
            Self::VulkanError(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for LoadingError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::LibraryLoadFailure(_) => write!(f, "failed to load the Vulkan shared library"),
            Self::VulkanError(err) => write!(f, "a runtime error occurred: {err}"),
        }
    }
}

impl From<VulkanError> for LoadingError {
    fn from(err: VulkanError) -> Self {
        Self::VulkanError(err)
    }
}
