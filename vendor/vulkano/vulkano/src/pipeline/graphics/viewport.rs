//! Configures the area of the framebuffer that pixels will be written to.
//!
//! There are two different concepts to determine where things will be drawn:
//!
//! - The viewport is the region of the image which corresponds to the vertex coordinates `-1.0` to
//!   `1.0`.
//! - Any pixel outside of the scissor box will be discarded.
//!
//! In other words, modifying the viewport will stretch the image, while modifying the scissor
//! box acts like a filter.
//!
//! It is legal and sensible to use a viewport that is larger than the target image or that
//! only partially overlaps the target image.
//!
//! # Multiple viewports
//!
//! In most situations, you only need a single viewport and a single scissor box.
//!
//! If, however, you use a geometry shader, you can specify multiple viewports and scissor boxes.
//! Then in your geometry shader you can specify in which viewport and scissor box the primitive
//! should be written to. In GLSL this is done by writing to the special variable
//! `gl_ViewportIndex`.
//!
//! If you don't use a geometry shader or use a geometry shader where don't set which viewport to
//! use, then the first viewport and scissor box will be used.
//!
//! # Dynamic and fixed
//!
//! Vulkan allows four different setups:
//!
//! - The state of both the viewports and scissor boxes is known at pipeline creation.
//! - The state of viewports is known at pipeline creation, but the state of scissor boxes is only
//!   known when submitting the draw command.
//! - The state of scissor boxes is known at pipeline creation, but the state of viewports is only
//!   known when submitting the draw command.
//! - The state of both the viewports and scissor boxes is only known when submitting the draw
//!   command.
//!
//! In all cases the number of viewports and scissor boxes must be the same.

use crate::{
    device::Device, self_referential::self_referential, Requires, RequiresAllOf, RequiresOneOf,
    ValidationError, Version,
};
use ash::vk;
use smallvec::SmallVec;

/// List of viewports and scissors that are used when creating a graphics pipeline object.
///
/// Note that the number of viewports and scissors must be the same.
#[derive(Clone, Debug)]
pub struct ViewportState<'a> {
    /// Specifies the viewport transforms.
    ///
    /// When [`DynamicState::Viewport`] is used, the values of each viewport are ignored
    /// and must be set dynamically, but the number of viewports is fixed and
    /// must be matched when setting the dynamic value.
    /// When [`DynamicState::ViewportWithCount`] is used, the number of viewports is also dynamic,
    /// and `viewports` must be empty.
    ///
    /// If neither the number of viewports nor the number of scissors is dynamic, then the
    /// number of both must be identical.
    ///
    /// The default value is a single element of `Viewport::default()`.
    ///
    /// [`DynamicState::Viewport`]: crate::pipeline::DynamicState::Viewport
    /// [`DynamicState::ViewportWithCount`]: crate::pipeline::DynamicState::ViewportWithCount
    pub viewports: &'a [Viewport],

    /// Specifies the scissor rectangles.
    ///
    /// When [`DynamicState::Scissor`] is used, the values of each scissor are ignored
    /// and must be set dynamically, but the number of scissors is fixed and
    /// must be matched when setting the dynamic value.
    /// When [`DynamicState::ScissorWithCount`] is used, the number of scissors is also dynamic,
    /// and `scissors` must be empty.
    ///
    /// If neither the number of viewports nor the number of scissors is dynamic, then the
    /// number of both must be identical.
    ///
    /// The default value is a single element of `Scissor::default()`.
    ///
    /// [`DynamicState::Scissor`]: crate::pipeline::DynamicState::Scissor
    /// [`DynamicState::ScissorWithCount`]: crate::pipeline::DynamicState::ScissorWithCount
    pub scissors: &'a [Scissor],

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for ViewportState<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> ViewportState<'a> {
    /// Returns a default `ViewportState`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            viewports: &[const { Viewport::new() }],
            scissors: &[const { Scissor::new() }],
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            viewports,
            scissors,
            _ne: _,
        } = self;

        let properties = device.physical_device().properties();

        for (index, viewport) in viewports.iter().enumerate() {
            viewport
                .validate(device)
                .map_err(|err| err.add_context(format!("viewports[{}].0", index)))?;
        }

        for (index, scissor) in scissors.iter().enumerate() {
            let &Scissor { offset, extent } = scissor;

            // VUID-VkPipelineViewportStateCreateInfo-x-02821
            // Ensured by the use of an unsigned integer.

            if i32::try_from(offset[0])
                .ok()
                .zip(i32::try_from(extent[0]).ok())
                .and_then(|(o, e)| o.checked_add(e))
                .is_none()
            {
                return Err(Box::new(ValidationError {
                    context: format!("scissors[{}]", index).into(),
                    problem: "`offset[0] + extent[0]` is greater than `i32::MAX`".into(),
                    vuids: &["VUID-VkPipelineViewportStateCreateInfo-offset-02822"],
                    ..Default::default()
                }));
            }

            if i32::try_from(offset[1])
                .ok()
                .zip(i32::try_from(extent[1]).ok())
                .and_then(|(o, e)| o.checked_add(e))
                .is_none()
            {
                return Err(Box::new(ValidationError {
                    context: format!("scissors[{}]", index).into(),
                    problem: "`offset[1] + extent[1]` is greater than `i32::MAX`".into(),
                    vuids: &["VUID-VkPipelineViewportStateCreateInfo-offset-02823"],
                    ..Default::default()
                }));
            }
        }

        if viewports.len() > 1 && !device.enabled_features().multi_viewport {
            return Err(Box::new(ValidationError {
                context: "viewports".into(),
                problem: "the length is greater than 1".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "multi_viewport",
                )])]),
                vuids: &["VUID-VkPipelineViewportStateCreateInfo-viewportCount-01216"],
            }));
        }

        if scissors.len() > 1 && !device.enabled_features().multi_viewport {
            return Err(Box::new(ValidationError {
                context: "scissors".into(),
                problem: "the length is greater than 1".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "multi_viewport",
                )])]),
                vuids: &["VUID-VkPipelineViewportStateCreateInfo-scissorCount-01217"],
            }));
        }

        if viewports.len() > properties.max_viewports as usize {
            return Err(Box::new(ValidationError {
                context: "viewports".into(),
                problem: "the length exceeds the `max_viewports` limit".into(),
                vuids: &["VUID-VkPipelineViewportStateCreateInfo-viewportCount-01218"],
                ..Default::default()
            }));
        }

        if scissors.len() > properties.max_viewports as usize {
            return Err(Box::new(ValidationError {
                context: "scissors".into(),
                problem: "the length exceeds the `max_viewports` limit".into(),
                vuids: &["VUID-VkPipelineViewportStateCreateInfo-scissorCount-01219"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk(
        &self,
        fields1_vk: &'a ViewportStateFields1Vk,
    ) -> vk::PipelineViewportStateCreateInfo<'a> {
        let ViewportStateFields1Vk {
            viewports_vk,
            scissors_vk,
        } = fields1_vk;

        let mut val_vk = vk::PipelineViewportStateCreateInfo::default()
            .flags(vk::PipelineViewportStateCreateFlags::empty());

        if !viewports_vk.is_empty() {
            val_vk = val_vk.viewports(viewports_vk);
        }

        if !scissors_vk.is_empty() {
            val_vk = val_vk.scissors(scissors_vk);
        }

        val_vk
    }

    pub(crate) fn to_vk_fields1(&self) -> ViewportStateFields1Vk {
        let &Self {
            viewports,
            scissors,
            _ne: _,
        } = self;

        let viewports_vk = viewports.iter().map(Viewport::to_vk).collect();
        let scissors_vk = scissors.iter().map(Scissor::to_vk).collect();

        ViewportStateFields1Vk {
            viewports_vk,
            scissors_vk,
        }
    }

    pub(crate) fn to_owned(&self) -> OwnedViewportState {
        let viewports = self.viewports.iter().cloned().collect();
        let scissors = self.scissors.iter().cloned().collect();

        OwnedViewportState::new(viewports, scissors, |viewports, scissors| ViewportState {
            viewports,
            scissors,
            _ne: crate::NE,
        })
    }
}

pub(crate) struct ViewportStateFields1Vk {
    pub(crate) viewports_vk: SmallVec<[vk::Viewport; 2]>,
    pub(crate) scissors_vk: SmallVec<[vk::Rect2D; 2]>,
}

self_referential! {
    mod owned_viewport_state {
        pub(crate) struct OwnedViewportState {
            inner: ViewportState<'_>,
            viewports: SmallVec<[Viewport; 1]>,
            scissors: SmallVec<[Scissor; 1]>,
        }
    }
}

/// State of a single viewport.
#[derive(Debug, Clone, PartialEq)]
pub struct Viewport {
    /// Coordinates in pixels of the top-left hand corner of the viewport.
    ///
    /// The default value is `[0.0; 2]`.
    pub offset: [f32; 2],

    /// Dimensions in pixels of the viewport.
    ///
    /// The default value is `[1.0; 2]`, which you probably want to override if you are not
    /// using dynamic state.
    pub extent: [f32; 2],

    /// Minimum depth value.
    ///
    /// The values `0.0` to `1.0` of each vertex's Z coordinate will be mapped to the range
    /// \[`min_depth`,&nbsp;`max_depth`\] before being compared to the existing depth value.
    ///
    /// Together with `max_depth`, this is equivalent to `glDepthRange` in OpenGL, except that
    /// OpenGL uses the Z coordinate range from `-1.0` to `1.0` instead.
    ///
    /// The default value is `0.0`.
    pub min_depth: f32,

    /// Maximum depth value.
    ///
    /// The values `0.0` to `1.0` of each vertex's Z coordinate will be mapped to the range
    /// \[`min_depth`,&nbsp;`max_depth`\] before being compared to the existing depth value.
    ///
    /// Together with `min_depth`, this is equivalent to `glDepthRange` in OpenGL, except that
    /// OpenGL uses the Z coordinate range from `-1.0` to `1.0` instead.
    ///
    /// The default value is `1.0`.
    pub max_depth: f32,
}

impl Default for Viewport {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Viewport {
    /// Returns a default `Viewport`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            offset: [0.0; 2],
            extent: [1.0; 2],
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            offset,
            extent,
            min_depth,
            max_depth,
        } = self;

        let properties = device.physical_device().properties();

        if extent[0] <= 0.0 {
            return Err(Box::new(ValidationError {
                context: "extent[0]".into(),
                problem: "is not greater than zero".into(),
                vuids: &["VUID-VkViewport-width-01770"],
                ..Default::default()
            }));
        }

        if extent[0] > properties.max_viewport_dimensions[0] as f32 {
            return Err(Box::new(ValidationError {
                context: "extent[0]".into(),
                problem: "exceeds the `max_viewport_dimensions[0]` limit".into(),
                vuids: &["VUID-VkViewport-width-01771"],
                ..Default::default()
            }));
        }

        if extent[1] <= 0.0
            && !(device.api_version() >= Version::V1_1
                || device.enabled_extensions().khr_maintenance1)
        {
            return Err(Box::new(ValidationError {
                context: "extent[1]".into(),
                problem: "is not greater than zero".into(),
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                    RequiresAllOf(&[Requires::DeviceExtension("khr_maintenance1")]),
                ]),
                vuids: &["VUID-VkViewport-apiVersion-07917"],
            }));
        }

        if extent[1].abs() > properties.max_viewport_dimensions[1] as f32 {
            return Err(Box::new(ValidationError {
                context: "extent[1]".into(),
                problem: "exceeds the `max_viewport_dimensions[1]` limit".into(),
                vuids: &["VUID-VkViewport-height-01773"],
                ..Default::default()
            }));
        }

        if offset[0] < properties.viewport_bounds_range[0] {
            return Err(Box::new(ValidationError {
                problem: "`offset[0]` is less than the `viewport_bounds_range[0]` property".into(),
                vuids: &["VUID-VkViewport-x-01774"],
                ..Default::default()
            }));
        }

        if offset[0] + extent[0] > properties.viewport_bounds_range[1] {
            return Err(Box::new(ValidationError {
                problem: "`offset[0] + extent[0]` is greater than the \
                    `viewport_bounds_range[1]` property"
                    .into(),
                vuids: &["VUID-VkViewport-x-01232"],
                ..Default::default()
            }));
        }

        if offset[1] < properties.viewport_bounds_range[0] {
            return Err(Box::new(ValidationError {
                problem: "`offset[1]` is less than the `viewport_bounds_range[0]` property".into(),
                vuids: &["VUID-VkViewport-y-01775"],
                ..Default::default()
            }));
        }

        if offset[1] > properties.viewport_bounds_range[1] {
            return Err(Box::new(ValidationError {
                problem: "`offset[1]` is greater than the `viewport_bounds_range[1]` property"
                    .into(),
                vuids: &["VUID-VkViewport-y-01776"],
                ..Default::default()
            }));
        }

        if offset[1] + extent[1] < properties.viewport_bounds_range[0] {
            return Err(Box::new(ValidationError {
                problem: "`offset[1] + extent[1]` is less than the \
                    `viewport_bounds_range[0]` property"
                    .into(),
                vuids: &["VUID-VkViewport-y-01777"],
                ..Default::default()
            }));
        }

        if offset[1] + extent[1] > properties.viewport_bounds_range[1] {
            return Err(Box::new(ValidationError {
                problem: "`offset[1] + extent[1]` is greater than the \
                    `viewport_bounds_range[1]` property"
                    .into(),
                vuids: &["VUID-VkViewport-y-01233"],
                ..Default::default()
            }));
        }

        if !device.enabled_extensions().ext_depth_range_unrestricted {
            if !(0.0..=1.0).contains(&min_depth) {
                return Err(Box::new(ValidationError {
                    problem: "`min_depth` is not between 0.0 and 1.0 inclusive".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                        "ext_depth_range_unrestricted",
                    )])]),
                    vuids: &["VUID-VkViewport-minDepth-01234"],
                    ..Default::default()
                }));
            }

            if !(0.0..=1.0).contains(&max_depth) {
                return Err(Box::new(ValidationError {
                    problem: "`max_depth` is not between 0.0 and 1.0 inclusive".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                        "ext_depth_range_unrestricted",
                    )])]),
                    vuids: &["VUID-VkViewport-maxDepth-01235"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[doc(hidden)]
    pub fn to_vk(&self) -> vk::Viewport {
        let &Self {
            offset,
            extent,
            min_depth,
            max_depth,
        } = self;

        vk::Viewport {
            x: offset[0],
            y: offset[1],
            width: extent[0],
            height: extent[1],
            min_depth,
            max_depth,
        }
    }
}

/// A two-dimensional subregion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scissor {
    /// Coordinates of the top-left hand corner of the box.
    ///
    /// The default value is `[0; 2]`.
    pub offset: [u32; 2],

    /// Dimensions of the box.
    ///
    /// The default value is `[i32::MAX; 2]`.
    pub extent: [u32; 2],
}

impl Default for Scissor {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Scissor {
    /// Returns a default `Scissor`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            offset: [0; 2],
            extent: [i32::MAX as u32; 2],
        }
    }

    /// Returns a scissor that, when used, will instruct the pipeline to draw to the entire
    /// framebuffer no matter its size.
    #[deprecated(since = "0.34.0", note = "use `Scissor::new` instead")]
    #[inline]
    pub fn irrelevant() -> Scissor {
        Self::new()
    }

    #[allow(clippy::wrong_self_convention)]
    #[doc(hidden)]
    pub fn to_vk(&self) -> vk::Rect2D {
        let &Self { offset, extent } = self;

        vk::Rect2D {
            offset: vk::Offset2D {
                x: offset[0] as i32,
                y: offset[1] as i32,
            },
            extent: vk::Extent2D {
                width: extent[0],
                height: extent[1],
            },
        }
    }
}
