#![doc(html_logo_url = "https://raw.githubusercontent.com/vulkano-rs/vulkano/master/logo.png")]
//! Safe and rich Rust wrapper around the Vulkan API.
//!
//! # Starting off with Vulkano
//!
//! The steps for using Vulkan through Vulkano are in principle not any different from using
//! the raw Vulkan API, but the details may be different for the sake of idiomaticity, safety
//! and convenience.
//!
//! 1. Create a [`VulkanLibrary`]. This represents a Vulkan library on the system, which must be
//!    loaded before you can do anything with Vulkan.
//!
//! 2. Create an [`Instance`]. This is the API entry point, and represents an initialized Vulkan
//!    library.
//!
//! 3. If you intend to show graphics to the user on a window or a screen, create a [`Surface`].
//!    A `Surface` is created from a window identifier or handle, that is specific to the display or
//!    windowing system being used. Vulkano uses `raw-window-handle` to abstract over the different
//!    windowing systems. Note that you have to make sure that the `raw-window-handle` that your
//!    windowing library uses is compatible with the `raw-window-handle` that vulkano uses. For
//!    example, if you use a `winit` version that uses a different version from the one vulkano
//!    uses, you can add one of the [features](https://docs.rs/crate/winit/latest/features) that
//!    starts with `rwh` to `winit`. Currently, vulkano is compatible with `rwh_06`.
//!
//! 4. [Enumerate the physical devices] that are available on the `Instance`, and choose one that
//!    is suitable for your program. A [`PhysicalDevice`] represents a Vulkan-capable device that
//!    is available on the system, such as a graphics card, a software implementation, etc.
//!
//! 5. Create a [`Device`] and accompanying [`Queue`]s from the selected `PhysicalDevice`. The
//!    `Device` is the most important object of Vulkan, and you need one to create almost every
//!    other object. `Queue`s are created together with the `Device`, and are used to submit work
//!    to the device to make it do something.
//!
//! 6. If you created a `Surface` earlier, create a [`Swapchain`]. This object contains special
//!    images that correspond to the contents of the surface. Whenever you want to change the
//!    contents (show something new to the user), you must first *acquire* one of these images from
//!    the swapchain, fill it with the new contents (by rendering, copying or any other means), and
//!    then *present* it back to the swapchain. A swapchain can become outdated if the properties
//!    of the surface change, such as when the size of the window changes. It then becomes
//!    necessary to create a new swapchain.
//!
//! 7. Record a [*command buffer*](command_buffer), containing commands that the device must
//!    execute. Then build the command buffer and submit it to a `Queue`.
//!
//! Many different operations can be recorded to a command buffer, such as *draw*, *compute* and
//! *transfer* operations. To do any of these things, you will need to create several other
//! objects, depending on your specific needs. This includes:
//!
//! - [*Buffers*] store general-purpose data on memory accessible by the device. This can include
//!   mesh data (vertices, texture coordinates etc.), lighting information, matrices, and anything
//!   else you can think of.
//!
//! - [*Images*] store texel data, arranged in a grid of one or more dimensions. They can be used
//!   as textures, depth/stencil buffers, framebuffers and as part of a swapchain.
//!
//! - [*Pipelines*] describe operations on the device. They include one or more [*shader*]s, small
//!   programs that the device will execute as part of a pipeline. Pipelines come in several types:
//!   - A [`ComputePipeline`] describes how *dispatch* commands are to be performed.
//!   - A [`GraphicsPipeline`] describes how *draw* commands are to be performed.
//!
//! - [*Descriptor sets*] make buffers, images and other objects available to shaders. The
//!   arrangement of these resources in shaders is described by a [`DescriptorSetLayout`]. One or
//!   more of these layouts in turn forms a [`PipelineLayout`], which is used when creating a
//!   pipeline object.
//!
//! - For more complex, multi-stage draw operations, you can create a [`RenderPass`] object. This
//!   object describes the stages, known as subpasses, that draw operations consist of, how they
//!   interact with one another, and which types of images are available in each subpass. You must
//!   also create a [`Framebuffer`], which contains the image objects that are to be used in a
//!   render pass.
//!
//! # `_unchecked` functions
//!
//! Many functions in Vulkano have two versions: the normal function, which is usually safe to
//! call, and another function with `_unchecked` added onto the end of the name, which is unsafe
//! to call. The `_unchecked` functions skip all validation checks, so they are usually more
//! efficient, but you must ensure that you meet the validity/safety requirements of the function.
//!
//! For all `_unchecked` functions, a call to the function is valid, if a call to the
//! corresponding normal function with the same arguments would return without any error.
//! This includes following all the valid usage requirements of the Vulkan specification, but may
//! also include additional requirements specific to Vulkano.
//! **All other usage of `_unchecked` functions may be undefined behavior.**
//!
//! Because there are potentially many `_unchecked` functions, and because their name and operation
//! can be straightforwardly understood based on the corresponding normal function, they are hidden
//! from the Vulkano documentation by default. You can unhide them by enabling the
//! `document_unchecked` cargo feature, and then generating the documentation with the command
//! `cargo doc --open`.
//!
//! # Cargo features
//!
//! | Feature              | Description                                                                               |
//! |----------------------|-------------------------------------------------------------------------------------------|
//! | `macros`             | Include reexports from [`vulkano-macros`]. Enabled by default.                            |
//! | `raw_window_handle`  | Enables interop with windowing libraries using [`raw-window-handle`]. Enabled by default. |
//! | `x11`                | When `raw_window_handle` is enabled, support for X11 platforms. Enabled by default.       |
//! | `document_unchecked` | Include `_unchecked` functions in the generated documentation.                            |
//! | `serde`              | Enables (de)serialization of certain types using [`serde`].                               |
//!
//! [`Instance`]: instance::Instance
//! [`Surface`]: swapchain::Surface
//! [Enumerate the physical devices]: instance::Instance::enumerate_physical_devices
//! [`PhysicalDevice`]: device::physical::PhysicalDevice
//! [`Device`]: device::Device
//! [`Queue`]: device::Queue
//! [`Swapchain`]: swapchain::Swapchain
//! [*command buffer*]: command_buffer
//! [*Buffers*]: buffer
//! [*Images*]: image
//! [*Pipelines*]: pipeline
//! [*shader*]: shader
//! [`ComputePipeline`]: pipeline::ComputePipeline
//! [`GraphicsPipeline`]: pipeline::GraphicsPipeline
//! [*Descriptor sets*]: descriptor_set
//! [`DescriptorSetLayout`]: descriptor_set::layout
//! [`PipelineLayout`]: pipeline::layout
//! [`RenderPass`]: render_pass::RenderPass
//! [`Framebuffer`]: render_pass::Framebuffer
//! [`vulkano-macros`]: vulkano_macros
//! [`raw-window-handle`]: raw_window_handle
//! [`serde`]: https://crates.io/crates/serde

pub use ash;
use ash::vk;
use bytemuck::{Pod, Zeroable};
pub use extensions::ExtensionProperties;
pub use half;
pub use library::{LoadingError, VulkanLibrary};
use std::{
    borrow::Cow,
    error::Error,
    ffi::CStr,
    fmt::{Debug, Display, Error as FmtError, Formatter},
    marker::PhantomData,
    num::NonZero,
    ops::Deref,
    sync::Arc,
};
pub use version::Version;
pub use vk::Handle;

#[macro_use]
mod tests;

pub mod acceleration_structure;
pub mod buffer;
mod cache;
pub mod command_buffer;
pub mod deferred;
pub mod descriptor_set;
pub mod device;
pub mod display;
mod extensions;
mod fns;
pub mod format;
pub mod image;
pub mod instance;
pub mod library;
mod macros;
pub mod memory;
pub mod padded;
pub mod pipeline;
pub mod query;
mod range_map;
pub mod range_set;
pub mod render_pass;
mod self_referential;
pub mod shader;
pub mod swapchain;
pub mod sync;
mod version;

/// Represents memory size and offset values on a Vulkan device.
/// Analogous to the Rust `usize` type on the host.
pub use vk::DeviceSize;

/// A [`DeviceSize`] that is known not to equal zero.
pub type NonZeroDeviceSize = NonZero<DeviceSize>;

/// Represents an address (pointer) on a Vulkan device.
pub use vk::DeviceAddress;

/// A [`DeviceAddress`] that is known not to equal zero.
pub type NonNullDeviceAddress = NonZero<DeviceAddress>;

/// Raw file descriptors.
///
/// This is the same type alias as [`std::os::fd::RawFd`] except that it isn't gated with
/// conditional compilation.
pub type RawFd = std::ffi::c_int;

/// Represents a region of device addresses with a stride.
#[derive(Debug, Copy, Clone, Default)]
pub struct StridedDeviceAddressRegion {
    pub device_address: DeviceAddress,
    pub stride: DeviceSize,
    pub size: DeviceSize,
}

impl StridedDeviceAddressRegion {
    #[doc(hidden)]
    pub fn to_vk(&self) -> vk::StridedDeviceAddressRegionKHR {
        vk::StridedDeviceAddressRegionKHR {
            device_address: self.device_address,
            stride: self.stride,
            size: self.size,
        }
    }
}

/// Holds 24 bits in the least significant bits of memory, and 8 bits in the most significant bits
/// of that memory, occupying a single [`u32`] in total.
//
// NOTE: This is copied from Ash, but duplicated here so that we can implement traits on it.
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(transparent)]
pub struct Packed24_8(u32);

unsafe impl Pod for Packed24_8 {}
unsafe impl Zeroable for Packed24_8 {}

impl Packed24_8 {
    /// Creates a new `Packed24_8` from the low 24 and high 8 bits.
    ///
    /// # Panics
    ///
    /// - Panics if `low_24` has any of its high 8 bits set.
    #[inline]
    #[track_caller]
    pub const fn new(low_24: u32, high_8: u8) -> Self {
        assert!(low_24 & 0xff00_0000 == 0);

        Self(low_24 | ((high_8 as u32) << 24))
    }

    /// Creates a new `Packed24_8` from the underlying `u32`.
    #[inline]
    pub const fn from_u32(bits: u32) -> Self {
        Self(bits)
    }

    /// Returns the underlying `u32`.
    #[inline]
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    /// Returns the least-significant 24 bits (3 bytes) of this integer.
    #[inline]
    pub const fn low_24(self) -> u32 {
        self.0 & 0x00ff_ffff
    }

    /// Returns the most significant 8 bits (single byte) of this integer.
    #[inline]
    pub const fn high_8(self) -> u8 {
        (self.0 >> 24) as u8
    }
}

// Allow referring to crate by its name to work around limitations of proc-macros
// in doctests.
// See https://github.com/rust-lang/cargo/issues/9886
// and https://github.com/bkchr/proc-macro-crate/issues/10
#[allow(unused_extern_crates)]
extern crate self as vulkano;

/// Alternative to the `Deref` trait. Contrary to `Deref`, must always return the same object.
pub unsafe trait SafeDeref: Deref {}
unsafe impl<T: ?Sized> SafeDeref for &T {}
unsafe impl<T: ?Sized> SafeDeref for Arc<T> {}
unsafe impl<T: ?Sized> SafeDeref for Box<T> {}

/// Gives access to the internal identifier of an object.
pub unsafe trait VulkanObject {
    /// The type of the object.
    type Handle: Handle;

    /// Returns the raw Vulkan handle of the object.
    fn handle(&self) -> Self::Handle;
}

unsafe impl<T, U> VulkanObject for T
where
    T: SafeDeref<Target = U>,
    U: VulkanObject + ?Sized,
{
    type Handle = U::Handle;

    #[inline]
    fn handle(&self) -> Self::Handle {
        (**self).handle()
    }
}

/// A wrapper that displays only the contained object's type and handle when debug-formatted. This
/// is useful because we have a lot of dependency chains, and the same dependencies would otherwise
/// be debug-formatted along with the dependents over and over leading to royal levels of spam.
#[repr(transparent)]
struct DebugWrapper<T>(T);

impl<T> Debug for DebugWrapper<T>
where
    T: Debug + VulkanObject,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(f, "0x{:x}", self.0.handle().as_raw())
    }
}

impl<T> Deref for DebugWrapper<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[inline]
unsafe fn c_str_to_string_unchecked(s: &CStr) -> String {
    debug_assert!(s.to_str().is_ok());

    unsafe { String::from_utf8_unchecked(s.to_bytes().to_owned()) }
}

include!(crate::autogen_output!("errors.rs"));

impl Error for VulkanError {}

impl Display for VulkanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        let msg = match self {
            VulkanError::NotReady => "a resource is not yet ready",
            VulkanError::Timeout => "an operation has not completed in the specified time",
            VulkanError::OutOfHostMemory => "a host memory allocation has failed",
            VulkanError::OutOfDeviceMemory => "a device memory allocation has failed",
            VulkanError::InitializationFailed => {
                "initialization of an object could not be completed for implementation-specific \
                reasons"
            }
            VulkanError::DeviceLost => "the logical or physical device has been lost",
            VulkanError::MemoryMapFailed => "mapping of a memory object has failed",
            VulkanError::LayerNotPresent => {
                "a requested layer is not present or could not be loaded"
            }
            VulkanError::ExtensionNotPresent => "a requested extension is not supported",
            VulkanError::FeatureNotPresent => "a requested feature is not supported",
            VulkanError::IncompatibleDriver => {
                "the requested version of Vulkan is not supported by the driver or is otherwise \
                incompatible for implementation-specific reasons"
            }
            VulkanError::TooManyObjects => "too many objects of the type have already been created",
            VulkanError::FormatNotSupported => "a requested format is not supported on this device",
            VulkanError::FragmentedPool => {
                "a pool allocation has failed due to fragmentation of the pool's memory"
            }
            VulkanError::Unknown => {
                "an unknown error has occurred; either the application has provided invalid input, \
                or an implementation failure has occurred"
            }
            VulkanError::OutOfPoolMemory => "a pool memory allocation has failed",
            VulkanError::InvalidExternalHandle => {
                "an external handle is not a valid handle of the specified type"
            }
            VulkanError::Fragmentation => {
                "a descriptor pool creation has failed due to fragmentation"
            }
            VulkanError::InvalidOpaqueCaptureAddress => {
                "a buffer creation or memory allocation failed because the requested address is \
                not available. A shader group handle assignment failed because the requested \
                shader group handle information is no longer valid"
            }
            VulkanError::IncompatibleDisplay => {
                "the display used by a swapchain does not use the same presentable image layout, \
                or is incompatible in a way that prevents sharing an image"
            }
            VulkanError::NotPermitted => "a requested operation was not permitted",
            VulkanError::SurfaceLost => "a surface is no longer available",
            VulkanError::NativeWindowInUse => {
                "the requested window is already in use by Vulkan or another API in a manner which \
                prevents it from being used again"
            }
            VulkanError::OutOfDate => {
                "a surface has changed in such a way that it is no longer compatible with the \
                swapchain, and further presentation requests using the swapchain will fail"
            }
            VulkanError::InvalidVideoStdParameters => {
                "the provided Video Std parameters do not adhere to the requirements of the used \
                video compression standard"
            }
            VulkanError::ValidationFailed => "validation failed",
            VulkanError::FullScreenExclusiveModeLost => {
                "an operation on a swapchain created with application controlled full-screen \
                access failed as it did not have exclusive full-screen access"
            }
            VulkanError::InvalidDrmFormatModifierPlaneLayout => {
                "the requested DRM format modifier plane layout is invalid"
            }
            VulkanError::InvalidShader => "one or more shaders failed to compile or link",
            VulkanError::ImageUsageNotSupported => "the requested `ImageUsage` are not supported",
            VulkanError::VideoPictureLayoutNotSupported => {
                "the requested video picture layout is not supported"
            }
            VulkanError::VideoProfileOperationNotSupported => {
                "a video profile operation specified via `VideoProfileInfo::video_codec_operation` \
                is not supported"
            }
            VulkanError::VideoProfileFormatNotSupported => {
                "format parameters in a requested `VideoProfileInfo` chain are not supported"
            }
            VulkanError::VideoProfileCodecNotSupported => {
                "codec-specific parameters in a requested `VideoProfileInfo` chain are not \
                supported"
            }
            VulkanError::VideoStdVersionNotSupported => {
                "the specified video Std header version is not supported"
            }
            VulkanError::CompressionExhausted => {
                "an image creation failed because internal resources required for compression are \
                exhausted"
            }
            VulkanError::NotEnoughSpace => {
                "a descriptor set update failed because there was not enough space in the \
                descriptor pool"
            }
            VulkanError::PresentTimingQueueFull => "the presentation timing queue is full",
            VulkanError::Unnamed(result) => {
                return write!(f, "unnamed error, VkResult value {}", result.as_raw());
            }
        };

        write!(f, "{msg}")
    }
}

impl From<VulkanError> for Validated<VulkanError> {
    fn from(err: VulkanError) -> Self {
        Self::Error(err)
    }
}

/// A wrapper for error types of functions that can return validation errors.
#[derive(Clone)]
pub enum Validated<E> {
    /// A non-validation error occurred.
    Error(E),

    /// A validation error occurred.
    ValidationError(Box<ValidationError>),
}

impl<E> Validated<E> {
    /// Maps the inner `Error` value using the provided function, or does nothing if the value is
    /// `ValidationError`.
    #[inline]
    pub fn map<F>(self, f: impl FnOnce(E) -> F) -> Validated<F> {
        match self {
            Self::Error(err) => Validated::Error(f(err)),
            Self::ValidationError(err) => Validated::ValidationError(err),
        }
    }

    #[inline]
    fn map_validation(self, f: impl FnOnce(Box<ValidationError>) -> Box<ValidationError>) -> Self {
        match self {
            Self::Error(err) => Self::Error(err),
            Self::ValidationError(err) => Self::ValidationError(f(err)),
        }
    }

    /// Returns the inner `Error` value, or panics if it contains `ValidationError`.
    #[inline(always)]
    #[track_caller]
    pub fn unwrap(self) -> E {
        match self {
            Self::Error(err) => err,
            Self::ValidationError(err) => {
                panic!(
                    "called `Validated::unwrap` on a `ValidationError` value: {:?}",
                    err
                )
            }
        }
    }
}

impl<E> Error for Validated<E>
where
    E: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Error(err) => Some(err),
            Self::ValidationError(err) => Some(err),
        }
    }
}

impl<E> Display for Validated<E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::Error(_) => write!(f, "a non-validation error occurred"),
            Self::ValidationError(_) => write!(f, "a validation error occurred"),
        }
    }
}

impl<E> Debug for Validated<E>
where
    E: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::Error(err) => write!(f, "a non-validation error occurred: {err}"),
            Self::ValidationError(err) => {
                write!(f, "a validation error occurred\n\nCaused by:\n    {err:?}")
            }
        }
    }
}

impl<E> From<Box<ValidationError>> for Validated<E> {
    fn from(err: Box<ValidationError>) -> Self {
        Self::ValidationError(err)
    }
}

/// The arguments or other context of a call to a Vulkan function were not valid.
#[derive(Clone, Default)]
pub struct ValidationError {
    /// The context in which the problem exists (e.g. a specific parameter).
    pub context: Cow<'static, str>,

    /// A description of the problem.
    pub problem: Cow<'static, str>,

    /// If applicable, settings that the user could enable to avoid the problem in the future.
    pub requires_one_of: RequiresOneOf,

    /// *Valid Usage IDs* (VUIDs) in the Vulkan specification that relate to the problem.
    pub vuids: &'static [&'static str],
}

impl ValidationError {
    fn from_error<E: Error>(err: E) -> Self {
        Self {
            context: "".into(),
            problem: err.to_string().into(),
            requires_one_of: RequiresOneOf::default(),
            vuids: &[],
        }
    }

    fn add_context(mut self: Box<Self>, context: impl Into<Cow<'static, str>>) -> Box<Self> {
        if self.context.is_empty() {
            self.context = context.into();
        } else {
            self.context = format!("{}.{}", context.into(), self.context).into();
        }

        self
    }

    fn set_vuids(mut self: Box<Self>, vuids: &'static [&'static str]) -> Box<Self> {
        self.vuids = vuids;
        self
    }
}

impl Debug for ValidationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        if self.context.is_empty() {
            write!(f, "{}", self.problem)?;
        } else {
            write!(f, "{}: {}", self.context, self.problem)?;
        }

        if !self.requires_one_of.is_empty() {
            if self.context.is_empty() && self.problem.is_empty() {
                write!(f, "{:?}", self.requires_one_of)?;
            } else {
                write!(f, "\n\n{:?}", self.requires_one_of)?;
            }
        }

        if !self.vuids.is_empty() {
            write!(f, "\n\nVulkan VUIDs:")?;

            for vuid in self.vuids {
                write!(f, "\n    {}", vuid)?;
            }
        }

        Ok(())
    }
}

impl Display for ValidationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        if self.context.is_empty() {
            write!(f, "{}", self.problem)?;
        } else {
            write!(f, "{}: {}", self.context, self.problem)?;
        }

        if !self.requires_one_of.is_empty() {
            if self.problem.is_empty() {
                write!(f, "{}", self.requires_one_of)?;
            } else {
                write!(f, " -- {}", self.requires_one_of)?;
            }
        }

        if let Some((first, rest)) = self.vuids.split_first() {
            write!(f, " (Vulkan VUIDs: {}", first)?;

            for vuid in rest {
                write!(f, ", {}", vuid)?;
            }

            write!(f, ")")?;
        }

        Ok(())
    }
}

impl Error for ValidationError {}

/// Used in errors to indicate a set of alternatives that needs to be available/enabled to allow
/// a given operation.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct RequiresOneOf(pub &'static [RequiresAllOf]);

impl RequiresOneOf {
    /// Returns the number of alternatives.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns whether there are any alternatives.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Debug for RequiresOneOf {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(f, "Requires one of:")?;

        for requires_all_of in self.0 {
            write!(f, "\n    {}", requires_all_of)?;
        }

        Ok(())
    }
}

impl Display for RequiresOneOf {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(f, "requires one of: ")?;

        if let Some((first, rest)) = self.0.split_first() {
            if first.0.len() > 1 {
                write!(f, "({})", first)?;
            } else {
                write!(f, "{}", first)?;
            }

            for rest in rest {
                if first.0.len() > 1 {
                    write!(f, " or ({})", rest)?;
                } else {
                    write!(f, " or {}", rest)?;
                }
            }
        }

        Ok(())
    }
}

/// Used in errors to indicate a set of requirements that all need to be available/enabled to allow
/// a given operation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RequiresAllOf(pub &'static [Requires]);

impl Display for RequiresAllOf {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        if let Some((first, rest)) = self.0.split_first() {
            write!(f, "{}", first)?;

            for rest in rest {
                write!(f, " + {}", rest)?;
            }
        }

        Ok(())
    }
}

/// Something that needs to be supported or enabled to allow a particular operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Requires {
    APIVersion(Version),
    DeviceFeature(&'static str),
    DeviceExtension(&'static str),
    InstanceExtension(&'static str),
}

impl Display for Requires {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Requires::APIVersion(Version { major, minor, .. }) => {
                write!(f, "Vulkan API version {}.{}", major, minor)
            }
            Requires::DeviceFeature(device_feature) => {
                write!(f, "device feature `{}`", device_feature)
            }
            Requires::DeviceExtension(device_extension) => {
                write!(f, "device extension `{}`", device_extension)
            }
            Requires::InstanceExtension(instance_extension) => {
                write!(f, "instance extension `{}`", instance_extension)
            }
        }
    }
}

/// A helper type for non-exhaustive structs.
///
/// This type cannot be constructed outside Vulkano. Structures with a field of this type can only
/// be constructed by calling a constructor function or `Default::default()`. The effect is similar
/// to the standard Rust `#[non_exhaustive]` attribute, except that it does not prevent update
/// syntax from being used.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)] // add traits as needed
pub struct NonExhaustive<'a>(PhantomData<&'a ()>);

const NE: NonExhaustive<'_> = NonExhaustive(PhantomData);

macro_rules! autogen_output {
    ($file:literal) => {
        concat!(std::env!("CARGO_MANIFEST_DIR"), "/autogen-out/", $file)
    };
}
use autogen_output;

// TODO: Replace with `std::hint::cold_path` once we bump the MSRV.
#[cold]
#[inline(always)]
const fn cold_path() {}

// The following functions are explicitly written to NOT have an early return. An early return
// inhibits the loop unroller and auto-vectorizer. These functions are used with arrays small
// enough where an early return would never be a gain (quite the opposite). The loops as written
// get unrolled entirely and replaced with a sequence of vector instructions. That's not to say
// that runtime performance is important here -- it absolutely isn't -- but this optimization
// couldn't be simpler to make.

#[inline]
const fn array_count<const N: usize>(array: &[bool; N]) -> usize {
    let mut out = 0;
    let mut i = 0;

    while i < N {
        out += array[i] as usize;
        i += 1;
    }

    out
}

#[inline]
const fn array_is_empty<const N: usize>(array: &[bool; N]) -> bool {
    let mut out = false;
    let mut i = 0;

    while i < N {
        out |= array[i];
        i += 1;
    }

    !out
}

#[inline]
const fn array_intersects<const N: usize>(lhs: &[bool; N], rhs: &[bool; N]) -> bool {
    let mut out = false;
    let mut i = 0;

    while i < N {
        out |= lhs[i] && rhs[i];
        i += 1;
    }

    out
}

#[inline]
const fn array_contains<const N: usize>(lhs: &[bool; N], rhs: &[bool; N]) -> bool {
    let mut out = true;
    let mut i = 0;

    while i < N {
        out &= lhs[i] || !rhs[i];
        i += 1;
    }

    out
}

#[inline]
const fn array_union<const N: usize>(lhs: &[bool; N], rhs: &[bool; N]) -> [bool; N] {
    let mut out = [false; N];
    let mut i = 0;

    while i < N {
        out[i] = lhs[i] || rhs[i];
        i += 1;
    }

    out
}

#[inline]
const fn array_intersection<const N: usize>(lhs: &[bool; N], rhs: &[bool; N]) -> [bool; N] {
    let mut out = [false; N];
    let mut i = 0;

    while i < N {
        out[i] = lhs[i] && rhs[i];
        i += 1;
    }

    out
}

#[inline]
const fn array_difference<const N: usize>(lhs: &[bool; N], rhs: &[bool; N]) -> [bool; N] {
    let mut out = [false; N];
    let mut i = 0;

    while i < N {
        out[i] = lhs[i] && !rhs[i];
        i += 1;
    }

    out
}

#[inline]
const fn array_symmetric_difference<const N: usize>(lhs: &[bool; N], rhs: &[bool; N]) -> [bool; N] {
    let mut out = [false; N];
    let mut i = 0;

    while i < N {
        out[i] = lhs[i] ^ rhs[i];
        i += 1;
    }

    out
}
