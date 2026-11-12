//! Control and use of display devices (e.g. monitors).
//!
//! A `Display` represents a display device, which is usually a monitor but can also be something
//! else that can display graphical content. You do not create `Display` objects yourself, but
//! you get them from the physical device instead. To get a list of all available displays on the
//! system, you can call [`PhysicalDevice::display_properties`].
//!
//! A *display plane* is a single layer within a display device or graphics stack that a surface
//! can be created from. Depending on the setup used by the system, there may be one fixed
//! display plane for each display, multiple display planes for each display, or even
//! a pool of display planes that multiple displays can make use of.
//!
//! # Creating surfaces that render directly to a display
//!
//! - Choose the `Display` that you want to render to.
//! - Get display plane properties with [`PhysicalDevice::display_plane_properties`], and choose a
//!   display plane index that is supported with the chosen display.
//! - Choose a `DisplayMode`, which is the combination of a display, a resolution and a refresh
//!   rate. You can enumerate the modes available on a display with
//!   [`Display::display_mode_properties`], or create your own mode. A display can show multiple
//!   planes in a stacking fashion.
//! - Create a `Surface` object with `Surface::from_display_plane`, and pass the chosen
//!   `DisplayMode` and display plane index.

use crate::{
    cache::{OnceCache, WeakArcOnceCache},
    device::physical::PhysicalDevice,
    instance::{Instance, InstanceOwned, InstanceOwnedDebugWrapper},
    macros::vulkan_bitflags_enum,
    swapchain::SurfaceTransforms,
    Validated, ValidationError, VulkanError, VulkanObject,
};
use ash::vk;
use std::{
    hash::{Hash, Hasher},
    mem::MaybeUninit,
    ptr,
    sync::Arc,
};

/// A display device connected to a physical device.
#[derive(Debug)]
pub struct Display {
    physical_device: InstanceOwnedDebugWrapper<Arc<PhysicalDevice>>,
    handle: vk::DisplayKHR,

    name: Option<String>,
    physical_dimensions: [u32; 2],
    physical_resolution: [u32; 2],
    supported_transforms: SurfaceTransforms,
    plane_reorder_possible: bool,
    persistent_content: bool,

    display_modes: WeakArcOnceCache<vk::DisplayModeKHR, DisplayMode>,
}

impl Display {
    /// Creates a new `Display` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `properties` must match the properties retrieved from `physical_device`.
    #[inline]
    pub fn from_handle(
        physical_device: &Arc<PhysicalDevice>,
        handle: vk::DisplayKHR,
        properties: DisplayProperties,
    ) -> Arc<Self> {
        let DisplayProperties {
            name,
            physical_dimensions,
            physical_resolution,
            supported_transforms,
            plane_reorder_possible,
            persistent_content,
        } = properties;

        Arc::new(Self {
            physical_device: InstanceOwnedDebugWrapper(physical_device.clone()),
            handle,

            name,
            physical_dimensions,
            physical_resolution,
            supported_transforms,
            plane_reorder_possible,
            persistent_content,

            display_modes: WeakArcOnceCache::new(),
        })
    }

    /// Returns the physical device that this display belongs to.
    #[inline]
    pub fn physical_device(&self) -> &Arc<PhysicalDevice> {
        &self.physical_device
    }

    /// Returns the name of the display.
    #[inline]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Returns the physical dimensions of the display, in millimeters.
    #[inline]
    pub fn physical_dimensions(&self) -> [u32; 2] {
        self.physical_dimensions
    }

    /// Returns the physical, or preferred, resolution of the display.
    #[inline]
    pub fn physical_resolution(&self) -> [u32; 2] {
        self.physical_resolution
    }

    /// Returns the transforms that are supported by the display.
    #[inline]
    pub fn supported_transforms(&self) -> SurfaceTransforms {
        self.supported_transforms
    }

    /// Returns whether planes on this display can have their z-order changed.
    #[inline]
    pub fn plane_reorder_possible(&self) -> bool {
        self.plane_reorder_possible
    }

    /// Returns whether the content of the display is buffered internally, and therefore
    /// persistent.
    #[inline]
    pub fn persistent_content(&self) -> bool {
        self.persistent_content
    }

    /// Returns the display modes that this display supports by default.
    pub fn display_mode_properties(self: &Arc<Self>) -> Result<Vec<Arc<DisplayMode>>, VulkanError> {
        let fns = self.physical_device.instance().fns();

        if self
            .instance()
            .enabled_extensions()
            .khr_get_display_properties2
        {
            let properties_vk = loop {
                let mut count = 0;
                unsafe {
                    (fns.khr_get_display_properties2
                        .get_display_mode_properties2_khr)(
                        self.physical_device.handle(),
                        self.handle,
                        &mut count,
                        ptr::null_mut(),
                    )
                }
                .result()
                .map_err(VulkanError::from)?;

                let mut properties_vk =
                    vec![vk::DisplayModeProperties2KHR::default(); count as usize];
                let result = unsafe {
                    (fns.khr_get_display_properties2
                        .get_display_mode_properties2_khr)(
                        self.physical_device.handle(),
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
                .into_iter()
                .map(|properties_vk| {
                    let properties_vk = &properties_vk.display_mode_properties;
                    self.display_modes
                        .get_or_insert(properties_vk.display_mode, |&handle| {
                            DisplayMode::from_handle(
                                self,
                                handle,
                                &DisplayModeCreateInfo::from_vk_parameters(
                                    &properties_vk.parameters,
                                ),
                            )
                        })
                })
                .collect())
        } else {
            let properties_vk = loop {
                let mut count = 0;
                unsafe {
                    (fns.khr_display.get_display_mode_properties_khr)(
                        self.physical_device.handle(),
                        self.handle,
                        &mut count,
                        ptr::null_mut(),
                    )
                }
                .result()
                .map_err(VulkanError::from)?;

                let mut properties = Vec::with_capacity(count as usize);
                let result = unsafe {
                    (fns.khr_display.get_display_mode_properties_khr)(
                        self.physical_device.handle(),
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

            Ok(properties_vk
                .into_iter()
                .map(|properties_vk| {
                    self.display_modes
                        .get_or_insert(properties_vk.display_mode, |&handle| {
                            DisplayMode::from_handle(
                                self,
                                handle,
                                &DisplayModeCreateInfo::from_vk_parameters(
                                    &properties_vk.parameters,
                                ),
                            )
                        })
                })
                .collect())
        }
    }
}

unsafe impl VulkanObject for Display {
    type Handle = vk::DisplayKHR;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl InstanceOwned for Display {
    #[inline]
    fn instance(&self) -> &Arc<Instance> {
        self.physical_device.instance()
    }
}

impl PartialEq for Display {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.physical_device == other.physical_device && self.handle == other.handle
    }
}

impl Eq for Display {}

impl Hash for Display {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.physical_device.hash(state);
        self.handle.hash(state);
    }
}

/// The properties of a display.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct DisplayProperties {
    /// The name of the display.
    pub name: Option<String>,

    /// The physical dimensions of the display, in millimeters.
    pub physical_dimensions: [u32; 2],

    /// The physical, or preferred, resolution of the display.
    pub physical_resolution: [u32; 2],

    /// The transforms that are supported by the display.
    pub supported_transforms: SurfaceTransforms,

    /// Whether planes on this display can have their z-order changed.
    pub plane_reorder_possible: bool,

    /// Whether the content of the display is buffered internally, and therefore persistent.
    pub persistent_content: bool,
}

impl DisplayProperties {
    pub(crate) fn to_mut_vk2() -> vk::DisplayProperties2KHR<'static> {
        vk::DisplayProperties2KHR::default()
    }

    pub(crate) fn from_vk(val_vk: &vk::DisplayPropertiesKHR<'_>) -> Self {
        let display_name_vk = unsafe { val_vk.display_name_as_c_str() };
        let &vk::DisplayPropertiesKHR {
            display: _,
            display_name: _,
            physical_dimensions,
            physical_resolution,
            supported_transforms,
            plane_reorder_possible,
            persistent_content,
            ..
        } = val_vk;

        DisplayProperties {
            name: display_name_vk.map(|name| {
                name.to_str()
                    .expect("non UTF-8 characters in display name")
                    .to_owned()
            }),
            physical_dimensions: [physical_dimensions.width, physical_dimensions.height],
            physical_resolution: [physical_resolution.width, physical_resolution.height],
            supported_transforms: supported_transforms.into(),
            plane_reorder_possible: plane_reorder_possible != vk::FALSE,
            persistent_content: persistent_content != vk::FALSE,
        }
    }
}

/// Represents a mode on a specific display.
///
/// A display mode describes a supported display resolution and refresh rate.
#[derive(Debug)]
pub struct DisplayMode {
    display: InstanceOwnedDebugWrapper<Arc<Display>>,
    handle: vk::DisplayModeKHR,

    visible_region: [u32; 2],
    refresh_rate: u32,

    display_plane_capabilities: OnceCache<u32, DisplayPlaneCapabilities>,
}

impl DisplayMode {
    /// Creates a custom display mode, panicking on a validation error.
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
        display: &Arc<Display>,
        create_info: &DisplayModeCreateInfo<'_>,
    ) -> Result<Arc<Self>, VulkanError> {
        match Self::try_new(display, create_info) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a custom display mode.
    #[inline]
    pub fn try_new(
        display: &Arc<Display>,
        create_info: &DisplayModeCreateInfo<'_>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_new(display, create_info)?;

        Ok(unsafe { Self::new_unchecked(display, create_info) }?)
    }

    fn validate_new(
        display: &Display,
        create_info: &DisplayModeCreateInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        // VUID-vkCreateDisplayModeKHR-pCreateInfo-parameter
        create_info
            .validate(&display.physical_device)
            .map_err(|err| err.add_context("create_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        display: &Arc<Display>,
        create_info: &DisplayModeCreateInfo<'_>,
    ) -> Result<Arc<Self>, VulkanError> {
        let physical_device = &display.physical_device;
        let create_info_vk = create_info.to_vk();

        let handle = {
            let fns = physical_device.instance().fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.khr_display.create_display_mode_khr)(
                    physical_device.handle(),
                    display.handle,
                    &create_info_vk,
                    ptr::null(),
                    output.as_mut_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;
            unsafe { output.assume_init() }
        };

        Ok(display.display_modes.get_or_insert(handle, |&handle| {
            Self::from_handle(display, handle, create_info)
        }))
    }

    /// Creates a new `DisplayMode` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `display`.
    /// - `create_info` must match the info used to create the object, or retrieved from `display`.
    #[inline]
    pub fn from_handle(
        display: &Arc<Display>,
        handle: vk::DisplayModeKHR,
        create_info: &DisplayModeCreateInfo<'_>,
    ) -> Arc<Self> {
        let &DisplayModeCreateInfo {
            visible_region,
            refresh_rate,
            _ne: _,
        } = create_info;

        Arc::new(Self {
            display: InstanceOwnedDebugWrapper(display.clone()),
            handle,

            visible_region,
            refresh_rate,

            display_plane_capabilities: OnceCache::new(),
        })
    }

    /// Returns the display that this display mode belongs to.
    #[inline]
    pub fn display(&self) -> &Arc<Display> {
        &self.display
    }

    /// Returns the extent of the visible region.
    #[inline]
    pub fn visible_region(&self) -> [u32; 2] {
        self.visible_region
    }

    /// Returns the refresh rate in millihertz (i.e. `60_000` is 60 times per second).
    #[inline]
    pub fn refresh_rate(&self) -> u32 {
        self.refresh_rate
    }

    /// Returns the capabilities of a display plane, when used with this display mode, panicking on
    /// a validation error.
    ///
    /// This is a shortcut for `try_display_plane_capabilities().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_display_plane_capabilities`] returns a [`ValidationError`].
    ///
    /// [`try_display_plane_capabilities`]: Self::try_display_plane_capabilities
    #[inline]
    #[track_caller]
    pub fn display_plane_capabilities(
        &self,
        plane_index: u32,
    ) -> Result<DisplayPlaneCapabilities, VulkanError> {
        match self.try_display_plane_capabilities(plane_index) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Returns the capabilities of a display plane, when used with this display mode.
    #[inline]
    pub fn try_display_plane_capabilities(
        &self,
        plane_index: u32,
    ) -> Result<DisplayPlaneCapabilities, Validated<VulkanError>> {
        self.validate_display_plane_capabilities(plane_index)?;

        Ok(unsafe { self.display_plane_capabilities_unchecked(plane_index) }?)
    }

    fn validate_display_plane_capabilities(
        &self,
        plane_index: u32,
    ) -> Result<(), Box<ValidationError>> {
        let display_plane_properties_raw =
            unsafe { self.display.physical_device.display_plane_properties_raw() }.map_err(
                |_err| {
                    Box::new(ValidationError {
                        problem: "`PhysicalDevice::display_plane_properties` returned an error"
                            .into(),
                        ..Default::default()
                    })
                },
            )?;

        if plane_index as usize >= display_plane_properties_raw.len() {
            return Err(Box::new(ValidationError {
                problem: "`plane_index` is not less than the number of display planes on the \
                    physical device"
                    .into(),
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn display_plane_capabilities_unchecked(
        &self,
        plane_index: u32,
    ) -> Result<DisplayPlaneCapabilities, VulkanError> {
        self.display_plane_capabilities
            .get_or_try_insert(&plane_index, || {
                let fns = self.display.physical_device.instance().fns();

                let mut capabilities_vk = DisplayPlaneCapabilities::to_mut_vk2();

                if self
                    .instance()
                    .enabled_extensions()
                    .khr_get_display_properties2
                {
                    let info_vk = vk::DisplayPlaneInfo2KHR::default()
                        .mode(self.handle)
                        .plane_index(plane_index);

                    unsafe {
                        (fns.khr_get_display_properties2
                            .get_display_plane_capabilities2_khr)(
                            self.display.physical_device.handle(),
                            &info_vk,
                            &mut capabilities_vk,
                        )
                    }
                    .result()
                    .map_err(VulkanError::from)?;
                } else {
                    unsafe {
                        (fns.khr_display.get_display_plane_capabilities_khr)(
                            self.display.physical_device.handle(),
                            self.handle,
                            plane_index,
                            &mut capabilities_vk.capabilities,
                        )
                    }
                    .result()
                    .map_err(VulkanError::from)?;
                }

                Ok((
                    plane_index,
                    DisplayPlaneCapabilities::from_vk2(&capabilities_vk),
                ))
            })
    }
}

unsafe impl VulkanObject for DisplayMode {
    type Handle = vk::DisplayModeKHR;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl InstanceOwned for DisplayMode {
    #[inline]
    fn instance(&self) -> &Arc<Instance> {
        self.display.instance()
    }
}

impl PartialEq for DisplayMode {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.display == other.display && self.handle == other.handle
    }
}

impl Eq for DisplayMode {}

impl Hash for DisplayMode {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.display.hash(state);
        self.handle.hash(state);
    }
}

/// Parameters to create a new `DisplayMode`.
#[derive(Clone, Debug)]
pub struct DisplayModeCreateInfo<'a> {
    /// The extent of the visible region. Neither coordinate may be zero.
    ///
    /// The default value is `[0; 2]`, which must be overridden.
    pub visible_region: [u32; 2],

    /// The refresh rate in millihertz (i.e. `60_000` is 60 times per second).
    /// This must not be zero.
    ///
    /// The default value is 0, which must be overridden.
    pub refresh_rate: u32,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for DisplayModeCreateInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl DisplayModeCreateInfo<'_> {
    /// Returns a default `DisplayModeCreateInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            visible_region: [0; 2],
            refresh_rate: 0,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(
        &self,
        _physical_device: &PhysicalDevice,
    ) -> Result<(), Box<ValidationError>> {
        let &Self {
            visible_region,
            refresh_rate,
            _ne: _,
        } = self;

        if visible_region[0] == 0 {
            return Err(Box::new(ValidationError {
                context: "visible_region[0]".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkDisplayModeParametersKHR-width-01990"],
                ..Default::default()
            }));
        }

        if visible_region[1] == 0 {
            return Err(Box::new(ValidationError {
                context: "visible_region[1]".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkDisplayModeParametersKHR-height-01991"],
                ..Default::default()
            }));
        }

        if refresh_rate == 0 {
            return Err(Box::new(ValidationError {
                context: "refresh_rate".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkDisplayModeParametersKHR-refreshRate-01992"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::DisplayModeCreateInfoKHR<'static> {
        let &Self {
            visible_region,
            refresh_rate,
            _ne: _,
        } = self;

        vk::DisplayModeCreateInfoKHR::default()
            .flags(vk::DisplayModeCreateFlagsKHR::empty())
            .parameters(vk::DisplayModeParametersKHR {
                visible_region: vk::Extent2D {
                    width: visible_region[0],
                    height: visible_region[1],
                },
                refresh_rate,
            })
    }

    pub(crate) fn from_vk_parameters(val_vk: &vk::DisplayModeParametersKHR) -> Self {
        let &vk::DisplayModeParametersKHR {
            visible_region,
            refresh_rate,
        } = val_vk;

        DisplayModeCreateInfo {
            visible_region: [visible_region.width, visible_region.height],
            refresh_rate,
            _ne: crate::NE,
        }
    }
}

/// The properties of a display plane.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct DisplayPlaneProperties {
    /// The display that the display plane is currently associated with.
    pub current_display: Option<Arc<Display>>,

    /// The current z-order of the plane. This will always be less than the total number of planes.
    pub current_stack_index: u32,
}

#[derive(Clone, Debug)]
pub(crate) struct DisplayPlanePropertiesRaw {
    pub(crate) current_display: Option<vk::DisplayKHR>,
    pub(crate) current_stack_index: u32,
}

impl DisplayPlanePropertiesRaw {
    pub(crate) fn to_mut_vk2() -> vk::DisplayPlaneProperties2KHR<'static> {
        vk::DisplayPlaneProperties2KHR::default()
    }

    pub(crate) fn from_vk(val_vk: &vk::DisplayPlanePropertiesKHR) -> Self {
        let &vk::DisplayPlanePropertiesKHR {
            current_display,
            current_stack_index,
        } = val_vk;

        DisplayPlanePropertiesRaw {
            current_display: Some(current_display).filter(|&x| x != vk::DisplayKHR::null()),
            current_stack_index,
        }
    }
}

/// The capabilities of a display plane.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct DisplayPlaneCapabilities {
    /// The supported alpha blending modes.
    pub supported_alpha: DisplayPlaneAlphaFlags,

    /// The minimum supported source rectangle offset.
    pub min_src_position: [u32; 2],

    /// The maximum supported source rectangle offset.
    pub max_src_position: [u32; 2],

    /// The minimum supported source rectangle size.
    pub min_src_extent: [u32; 2],

    /// The maximum supported source rectangle size.
    pub max_src_extent: [u32; 2],

    /// The minimum supported destination rectangle offset.
    pub min_dst_position: [u32; 2],

    /// The maximum supported destination rectangle offset.
    pub max_dst_position: [i32; 2],

    /// The minimum supported destination rectangle size.
    pub min_dst_extent: [u32; 2],

    /// The maximum supported destination rectangle size.
    pub max_dst_extent: [u32; 2],
}

impl DisplayPlaneCapabilities {
    pub(crate) fn to_mut_vk2() -> vk::DisplayPlaneCapabilities2KHR<'static> {
        vk::DisplayPlaneCapabilities2KHR::default()
    }

    pub(crate) fn from_vk2(val_vk: &vk::DisplayPlaneCapabilities2KHR<'_>) -> Self {
        let &vk::DisplayPlaneCapabilities2KHR {
            capabilities:
                vk::DisplayPlaneCapabilitiesKHR {
                    supported_alpha,
                    min_src_position,
                    max_src_position,
                    min_src_extent,
                    max_src_extent,
                    min_dst_position,
                    max_dst_position,
                    min_dst_extent,
                    max_dst_extent,
                },
            ..
        } = val_vk;

        DisplayPlaneCapabilities {
            supported_alpha: supported_alpha.into(),
            min_src_position: [min_src_position.x as u32, min_src_position.y as u32],
            max_src_position: [max_src_position.x as u32, max_src_position.y as u32],
            min_src_extent: [min_src_extent.width, min_src_extent.height],
            max_src_extent: [max_src_extent.width, max_src_extent.height],
            min_dst_position: [min_dst_position.x as u32, min_dst_position.y as u32],
            max_dst_position: [max_dst_position.x, max_dst_position.y],
            min_dst_extent: [min_dst_extent.width, min_dst_extent.height],
            max_dst_extent: [max_dst_extent.width, max_dst_extent.height],
        }
    }
}

vulkan_bitflags_enum! {
    #[non_exhaustive]

    /// A set of [`DisplayPlaneAlpha`] values.
    DisplayPlaneAlphaFlags,

    /// The alpha blending mode to use for a display mode.
    DisplayPlaneAlpha,

    = DisplayPlaneAlphaFlagsKHR(u32);

    /// The source image is treated as opaque.
    OPAQUE, Opaque = OPAQUE,

    /// Use a single global alpha value that will be used for all pixels in the source image.
    GLOBAL, Global = GLOBAL,

    /// Use the alpha component of each pixel in the source image.
    PER_PIXEL, PerPixel = PER_PIXEL,

    /// Use the alpha component of each pixel in the source image,
    /// but treat the other components as having already been multiplied by the alpha component.
    PER_PIXEL_PREMULTIPLIED, PerPixelPremultiplied = PER_PIXEL_PREMULTIPLIED,
}
