//! Configures how primitives should be converted into collections of fragments.

use crate::{
    device::Device, macros::vulkan_enum, Requires, RequiresAllOf, RequiresOneOf, ValidationError,
};
use ash::vk;
use ash::vk::TaggedStructure;

/// The state in a graphics pipeline describing how the rasterization stage should behave.
#[derive(Clone, Debug)]
pub struct RasterizationState<'a> {
    /// If true, then the depth value of the vertices will be clamped to the range [0.0, 1.0]. If
    /// false, fragments whose depth is outside of this range will be discarded.
    ///
    /// If enabled, the [`depth_clamp`](crate::device::DeviceFeatures::depth_clamp) feature must be
    /// enabled on the device.
    ///
    /// The default value is `false`.
    pub depth_clamp_enable: bool,

    /// If true, all the fragments will be discarded, and the fragment shader will not be run. This
    /// is usually used when your vertex shader has some side effects and you don't need to run the
    /// fragment shader.
    ///
    /// The default value is `false`.
    pub rasterizer_discard_enable: bool,

    /// This setting can ask the rasterizer to downgrade triangles into lines or points, or lines
    /// into points.
    ///
    /// If set to a value other than `Fill`, the
    /// [`fill_mode_non_solid`](crate::device::DeviceFeatures::fill_mode_non_solid) feature must be
    /// enabled on the device.
    ///
    /// The default value is [`PolygonMode::Fill`].
    pub polygon_mode: PolygonMode,

    /// Specifies whether front faces or back faces should be discarded, or none, or both.
    ///
    /// The default value is [`CullMode::None`].
    pub cull_mode: CullMode,

    /// Specifies which triangle orientation is considered to be the front of the triangle.
    ///
    /// The default value is [`FrontFace::CounterClockwise`].
    pub front_face: FrontFace,

    /// Sets how to modify depth values in the rasterization stage.
    ///
    /// If set to `None`, depth biasing is disabled, the depth values will pass to the fragment
    /// shader unmodified.
    ///
    /// The default value is `None`.
    pub depth_bias: Option<DepthBiasState>,

    /// Width, in pixels, of lines when drawing lines.
    ///
    /// Setting this to a value other than 1.0 requires the
    /// [`wide_lines`](crate::device::DeviceFeatures::wide_lines) feature to be enabled on
    /// the device.
    ///
    /// The default value is `1.0`.
    pub line_width: f32,

    /// The rasterization mode for lines.
    ///
    /// If this is not set to `Default`, the
    /// [`ext_line_rasterization`](crate::device::DeviceExtensions::ext_line_rasterization)
    /// extension and an additional feature must be enabled on the device.
    ///
    /// The default value is [`LineRasterizationMode::Default`].
    pub line_rasterization_mode: LineRasterizationMode,

    /// Enables and sets the parameters for line stippling.
    ///
    /// If this is set to `Some`, the
    /// [`ext_line_rasterization`](crate::device::DeviceExtensions::ext_line_rasterization)
    /// extension and an additional feature must be enabled on the device.
    ///
    /// The default value is `None`.
    pub line_stipple: Option<LineStipple>,

    /// Enables a mode of rasterization where the edges of primitives are modified so that
    /// fragments are generated if the edge of a primitive touches any part of a pixel, or if a
    /// pixel is fully covered by a primitive.
    ///
    /// If this is set to `Some`, the
    /// [`ext_conservative_rasterization`](crate::device::DeviceExtensions::ext_conservative_rasterization)
    /// extension must be enabled on the device.
    ///
    /// The default value is `None`.
    pub conservative: Option<RasterizationConservativeState>,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for RasterizationState<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> RasterizationState<'a> {
    /// Returns a default `RasterizationState`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            depth_clamp_enable: false,
            rasterizer_discard_enable: false,
            polygon_mode: PolygonMode::Fill,
            cull_mode: CullMode::None,
            front_face: FrontFace::CounterClockwise,
            depth_bias: None,
            line_width: 1.0,
            line_rasterization_mode: LineRasterizationMode::Default,
            line_stipple: None,
            conservative: None,
            _ne: crate::NE,
        }
    }

    /// Sets the polygon mode.
    #[inline]
    #[deprecated(since = "0.34.0")]
    pub fn polygon_mode(mut self, polygon_mode: PolygonMode) -> Self {
        self.polygon_mode = polygon_mode;
        self
    }

    /// Sets the cull mode.
    #[inline]
    #[deprecated(since = "0.34.0")]
    pub fn cull_mode(mut self, cull_mode: CullMode) -> Self {
        self.cull_mode = cull_mode;
        self
    }

    /// Sets the front face.
    #[inline]
    #[deprecated(since = "0.34.0")]
    pub fn front_face(mut self, front_face: FrontFace) -> Self {
        self.front_face = front_face;
        self
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            depth_clamp_enable,
            rasterizer_discard_enable,
            polygon_mode,
            cull_mode,
            front_face,
            depth_bias: _,
            line_width: _,
            line_rasterization_mode,
            ref line_stipple,
            ref conservative,
            _ne: _,
        } = self;

        let properties = device.physical_device().properties();

        polygon_mode.validate_device(device).map_err(|err| {
            err.add_context("polygon_mode")
                .set_vuids(&["VUID-VkPipelineRasterizationStateCreateInfo-polygonMode-parameter"])
        })?;

        line_rasterization_mode
            .validate_device(device)
            .map_err(|err| err.add_context("line_rasterization_mode").set_vuids(&["VUID-VkPipelineRasterizationLineStateCreateInfoEXT-lineRasterizationMode-parameter"]))?;

        if depth_clamp_enable && !device.enabled_features().depth_clamp {
            return Err(Box::new(ValidationError {
                context: "depth_clamp_enable".into(),
                problem: "is `true`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "depth_clamp",
                )])]),
                vuids: &["VUID-VkPipelineRasterizationStateCreateInfo-depthClampEnable-00782"],
            }));
        }

        if polygon_mode != PolygonMode::Fill && !device.enabled_features().fill_mode_non_solid {
            return Err(Box::new(ValidationError {
                context: "polygon_mode".into(),
                problem: "is not `PolygonMode::Fill`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "fill_mode_non_solid",
                )])]),
                vuids: &["VUID-VkPipelineRasterizationStateCreateInfo-polygonMode-01507"],
            }));
        }

        if device.enabled_extensions().khr_portability_subset
            && !device.enabled_features().point_polygons
            && !rasterizer_discard_enable
            && polygon_mode == PolygonMode::Point
        {
            return Err(Box::new(ValidationError {
                problem: "this device is a portability subset device, \
                    `rasterizer_discard_enable` is `false`, and \
                    `polygon_mode` is `PolygonMode::Point`"
                    .into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "point_polygons",
                )])]),
                vuids: &["VUID-VkPipelineRasterizationStateCreateInfo-pointPolygons-04458"],
                ..Default::default()
            }));
        }

        cull_mode.validate_device(device).map_err(|err| {
            err.add_context("cull_mode")
                .set_vuids(&["VUID-VkPipelineRasterizationStateCreateInfo-cullMode-parameter"])
        })?;

        front_face.validate_device(device).map_err(|err| {
            err.add_context("front_face")
                .set_vuids(&["VUID-VkPipelineRasterizationStateCreateInfo-frontFace-parameter"])
        })?;

        if line_rasterization_mode != LineRasterizationMode::Default {
            if !device.enabled_extensions().ext_line_rasterization {
                return Err(Box::new(ValidationError {
                    context: "line_rasterization_mode".into(),
                    problem: "`is not `LineRasterizationMode::Default`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                        "ext_line_rasterization",
                    )])]),
                    ..Default::default()
                }));
            }

            match line_rasterization_mode {
                LineRasterizationMode::Default => (),
                LineRasterizationMode::Rectangular => {
                    if !device.enabled_features().rectangular_lines {
                        return Err(Box::new(ValidationError {
                            context: "line_rasterization_mode".into(),
                            problem: "is `LineRasterizationMode::Rectangular`".into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                                "rectangular_lines",
                            )])]),
                            vuids: &["VUID-VkPipelineRasterizationLineStateCreateInfoEXT-lineRasterizationMode-02768"],
                        }));
                    }
                }
                LineRasterizationMode::Bresenham => {
                    if !device.enabled_features().bresenham_lines {
                        return Err(Box::new(ValidationError {
                            context: "line_rasterization_mode".into(),
                            problem: "is `LineRasterizationMode::Bresenham`".into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                                "bresenham_lines",
                            )])]),
                            vuids: &["VUID-VkPipelineRasterizationLineStateCreateInfoEXT-lineRasterizationMode-02769"],
                        }));
                    }
                }
                LineRasterizationMode::RectangularSmooth => {
                    if !device.enabled_features().smooth_lines {
                        return Err(Box::new(ValidationError {
                            context: "line_rasterization_mode".into(),
                            problem: "is `LineRasterizationMode::RectangularSmooth`".into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                                "smooth_lines",
                            )])]),
                            vuids: &["VUID-VkPipelineRasterizationLineStateCreateInfoEXT-lineRasterizationMode-02770"],
                        }));
                    }
                }
            }
        }

        if line_stipple.is_some() {
            if !device.enabled_extensions().ext_line_rasterization {
                return Err(Box::new(ValidationError {
                    context: "line_stipple".into(),
                    problem: "is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                        "ext_line_rasterization",
                    )])]),
                    ..Default::default()
                }));
            }

            match line_rasterization_mode {
                LineRasterizationMode::Default => {
                    if !device.enabled_features().stippled_rectangular_lines {
                        return Err(Box::new(ValidationError {
                            problem: "`line_stipple` is `Some`, and \
                                `line_rasterization_mode` is \
                                `LineRasterizationMode::Default`".into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                                "stippled_rectangular_lines",
                            )])]),
                            vuids: &["VUID-VkPipelineRasterizationLineStateCreateInfoEXT-stippledLineEnable-02774"],
                            ..Default::default()
                        }));
                    }

                    if !properties.strict_lines {
                        return Err(Box::new(ValidationError {
                            problem: "`line_stipple` is `Some`, and \
                                `line_rasterization_mode` is \
                                `LineRasterizationMode::Default`, \
                                but the `strict_lines` property is `false`".into(),
                            vuids: &["VUID-VkPipelineRasterizationLineStateCreateInfoEXT-stippledLineEnable-02774"],
                            ..Default::default()
                        }));
                    }
                }
                LineRasterizationMode::Rectangular => {
                    if !device.enabled_features().stippled_rectangular_lines {
                        return Err(Box::new(ValidationError {
                            problem: "`line_stipple` is `Some`, and \
                                `line_rasterization_mode` is \
                                `LineRasterizationMode::Rectangular`".into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                                "stippled_rectangular_lines",
                            )])]),
                            vuids: &["VUID-VkPipelineRasterizationLineStateCreateInfoEXT-stippledLineEnable-02771"],
                            ..Default::default()
                        }));
                    }
                }
                LineRasterizationMode::Bresenham => {
                    if !device.enabled_features().stippled_bresenham_lines {
                        return Err(Box::new(ValidationError {
                            problem: "`line_stipple` is `Some`, and \
                                `line_rasterization_mode` is \
                                `LineRasterizationMode::Bresenham`".into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                                "stippled_bresenham_lines",
                            )])]),
                            vuids: &["VUID-VkPipelineRasterizationLineStateCreateInfoEXT-stippledLineEnable-02772"],
                            ..Default::default()
                        }));
                    }
                }
                LineRasterizationMode::RectangularSmooth => {
                    if !device.enabled_features().stippled_smooth_lines {
                        return Err(Box::new(ValidationError {
                            problem: "`line_stipple` is `Some`, and \
                                `line_rasterization_mode` is \
                                `LineRasterizationMode::RectangularSmooth`".into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                                "stippled_smooth_lines",
                            )])]),
                            vuids: &["VUID-VkPipelineRasterizationLineStateCreateInfoEXT-stippledLineEnable-02773"],
                            ..Default::default()
                        }));
                    }
                }
            }
        }

        if let Some(conservative) = conservative {
            if !device.enabled_extensions().ext_conservative_rasterization {
                return Err(Box::new(ValidationError {
                    context: "conservative".into(),
                    problem: "is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                        "ext_conservative_rasterization",
                    )])]),
                    ..Default::default()
                }));
            }

            conservative
                .validate(device)
                .map_err(|err| err.add_context("conservative"))?;
        }

        Ok(())
    }

    pub(crate) fn to_vk(
        &self,
        extensions_vk: &'a mut RasterizationStateExtensionsVk,
    ) -> vk::PipelineRasterizationStateCreateInfo<'a> {
        let &Self {
            depth_clamp_enable,
            rasterizer_discard_enable,
            polygon_mode,
            cull_mode,
            front_face,
            ref depth_bias,
            line_width,
            line_rasterization_mode: _,
            line_stipple: _,
            conservative: _,
            _ne: _,
        } = self;

        let (
            depth_bias_enable_vk,
            depth_bias_constant_factor_vk,
            depth_bias_clamp_vk,
            depth_bias_slope_factor_vk,
        ) = if let Some(depth_bias_state) = depth_bias {
            let &DepthBiasState {
                constant_factor,
                clamp,
                slope_factor,
            } = depth_bias_state;

            (true, constant_factor, clamp, slope_factor)
        } else {
            (false, 0.0, 0.0, 0.0)
        };

        let mut val_vk = vk::PipelineRasterizationStateCreateInfo::default()
            .flags(vk::PipelineRasterizationStateCreateFlags::empty())
            .depth_clamp_enable(depth_clamp_enable)
            .rasterizer_discard_enable(rasterizer_discard_enable)
            .polygon_mode(polygon_mode.into())
            .cull_mode(cull_mode.into())
            .front_face(front_face.into())
            .depth_bias_enable(depth_bias_enable_vk)
            .depth_bias_constant_factor(depth_bias_constant_factor_vk)
            .depth_bias_clamp(depth_bias_clamp_vk)
            .depth_bias_slope_factor(depth_bias_slope_factor_vk)
            .line_width(line_width);

        let RasterizationStateExtensionsVk {
            line_vk,
            conservative_vk,
        } = extensions_vk;

        if let Some(next) = line_vk {
            val_vk = val_vk.push(next);
        }

        if let Some(next) = conservative_vk {
            val_vk = val_vk.push(next);
        }

        val_vk
    }

    pub(crate) fn to_vk_extensions(&self) -> RasterizationStateExtensionsVk {
        let &Self {
            line_rasterization_mode,
            ref line_stipple,
            ref conservative,
            ..
        } = self;

        let line_vk = (line_rasterization_mode != LineRasterizationMode::Default).then(|| {
            let (stippled_line_enable, line_stipple_factor, line_stipple_pattern) =
                if let Some(line_stipple) = line_stipple {
                    (true, line_stipple.factor, line_stipple.pattern)
                } else {
                    (false, 1, 0)
                };

            vk::PipelineRasterizationLineStateCreateInfoKHR::default()
                .line_rasterization_mode(line_rasterization_mode.into())
                .stippled_line_enable(stippled_line_enable)
                .line_stipple_factor(line_stipple_factor)
                .line_stipple_pattern(line_stipple_pattern)
        });
        let conservative_vk = conservative
            .as_ref()
            .map(RasterizationConservativeState::to_vk);

        RasterizationStateExtensionsVk {
            line_vk,
            conservative_vk,
        }
    }

    pub(crate) fn to_owned(&self) -> RasterizationState<'static> {
        RasterizationState {
            _ne: crate::NE,
            ..*self
        }
    }
}

pub(crate) struct RasterizationStateExtensionsVk {
    pub(crate) line_vk: Option<vk::PipelineRasterizationLineStateCreateInfoKHR<'static>>,
    pub(crate) conservative_vk:
        Option<vk::PipelineRasterizationConservativeStateCreateInfoEXT<'static>>,
}

/// The values to use for depth biasing.
#[derive(Clone, Copy, Debug)]
pub struct DepthBiasState {
    /// Specifies a constant factor to be multiplied to every depth value.
    ///
    /// The default value is `1.0`.
    pub constant_factor: f32,

    /// The maximum (or minimum) depth bias of a fragment.
    ///
    /// Setting this to a value other than 0.0 requires the
    /// [`depth_bias_clamp`](crate::device::DeviceFeatures::depth_bias_clamp) feature to be enabled
    /// on the device.
    ///
    /// The default value is `0.0`.
    pub clamp: f32,

    /// A scalar factor to multiply with a fragment's slope in depth bias calculations.
    ///
    /// The default value is `1.0`.
    pub slope_factor: f32,
}

impl Default for DepthBiasState {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl DepthBiasState {
    /// Returns a default `DepthBiasState`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            constant_factor: 1.0,
            clamp: 0.0,
            slope_factor: 1.0,
        }
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// Specifies the culling mode.
    ///
    /// This setting works in pair with `front_face`. The `front_face` setting tells the GPU whether
    /// clockwise or counter-clockwise correspond to the front and the back of each triangle. Then
    /// `cull_mode` lets you specify whether front faces should be discarded, back faces should be
    /// discarded, or none, or both.
    CullMode = CullModeFlags(u32);

    /// No culling.
    None = NONE,

    /// The faces facing the front of the screen (i.e. facing the user) will be removed.
    Front = FRONT,

    /// The faces facing the back of the screen will be removed.
    Back = BACK,

    /// All faces will be removed.
    FrontAndBack = FRONT_AND_BACK,
}

impl Default for CullMode {
    #[inline]
    fn default() -> CullMode {
        CullMode::None
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// Specifies which triangle orientation corresponds to the front or the triangle.
    FrontFace = FrontFace(i32);

    /// Triangles whose vertices are oriented counter-clockwise on the screen will be considered
    /// as facing their front. Otherwise they will be considered as facing their back.
    CounterClockwise = COUNTER_CLOCKWISE,

    /// Triangles whose vertices are oriented clockwise on the screen will be considered
    /// as facing their front. Otherwise they will be considered as facing their back.
    Clockwise = CLOCKWISE,
}

impl Default for FrontFace {
    #[inline]
    fn default() -> FrontFace {
        FrontFace::CounterClockwise
    }
}

vulkan_enum! {
    #[non_exhaustive]

    // TODO: document
    PolygonMode = PolygonMode(i32);

    // TODO: document
    Fill = FILL,

    // TODO: document
    Line = LINE,

    // TODO: document further
    /// On [portability subset](crate::instance#portability-subset-devices-and-the-enumerate_portability-flag)
    /// devices, unless `rasterizer_discard_enable` is active, the
    /// [`point_polygons`](crate::device::DeviceFeatures::point_polygons)
    /// feature must be enabled on the device.
    Point = POINT,

    /* TODO: enable
    // TODO: document
    FillRectangle = FILL_RECTANGLE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_fill_rectangle)]),
    ]),*/
}

impl Default for PolygonMode {
    #[inline]
    fn default() -> PolygonMode {
        PolygonMode::Fill
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// The rasterization mode to use for lines.
    LineRasterizationMode = LineRasterizationModeEXT(i32);

    /// If the [`strict_lines`](crate::device::DeviceProperties::strict_lines) device property is `true`,
    /// then this is the same as `Rectangular`. Otherwise, lines are drawn as parallelograms.
    ///
    /// If [`RasterizationState::line_stipple`] is `Some`, then the
    /// [`strict_lines`](crate::device::DeviceProperties::strict_lines) property must be `true` and the
    /// [`stippled_rectangular_lines`](crate::device::DeviceFeatures::stippled_rectangular_lines) feature
    /// must be enabled on the device.
    Default = DEFAULT,

    /// Lines are drawn as if they were rectangles extruded from the line.
    ///
    /// The [`rectangular_lines`](crate::device::DeviceFeatures::rectangular_lines) feature must be
    /// enabled on the device. If [`RasterizationState::line_stipple`] is `Some`, then the
    /// [`stippled_rectangular_lines`](crate::device::DeviceFeatures::stippled_rectangular_lines) must
    /// also be enabled.
    Rectangular = RECTANGULAR,

    /// Lines are drawn by determining which pixel diamonds the line intersects and exits.
    ///
    /// The [`bresenham_lines`](crate::device::DeviceFeatures::bresenham_lines) feature must be
    /// enabled on the device. If [`RasterizationState::line_stipple`] is `Some`, then the
    /// [`stippled_bresenham_lines`](crate::device::DeviceFeatures::stippled_bresenham_lines) must
    /// also be enabled.
    Bresenham = BRESENHAM,

    /// As `Rectangular`, but with alpha falloff.
    ///
    /// The [`smooth_lines`](crate::device::DeviceFeatures::smooth_lines) feature must be
    /// enabled on the device. If [`RasterizationState::line_stipple`] is `Some`, then the
    /// [`stippled_smooth_lines`](crate::device::DeviceFeatures::stippled_smooth_lines) must
    /// also be enabled.
    RectangularSmooth = RECTANGULAR_SMOOTH,
}

impl Default for LineRasterizationMode {
    #[inline]
    fn default() -> Self {
        Self::Default
    }
}

/// The parameters of a stippled line.
#[derive(Clone, Copy, Debug)]
pub struct LineStipple {
    /// The repeat factor used in stippled line rasterization. Must be between 1 and 256 inclusive.
    pub factor: u32,

    /// The bit pattern used in stippled line rasterization.
    pub pattern: u16,
}

/// The state in a graphics pipeline describing how the conservative rasterization mode should
/// behave.
#[derive(Clone, Copy, Debug)]
pub struct RasterizationConservativeState {
    /// Sets the conservative rasterization mode.
    ///
    /// The default value is [`ConservativeRasterizationMode::Disabled`].
    pub mode: ConservativeRasterizationMode,

    /// The extra size in pixels to increase the generating primitive during conservative
    /// rasterization. If the mode is set to anything other than
    /// [`ConservativeRasterizationMode::Overestimate`] this value is ignored.
    ///
    ///  The default value is 0.0.
    pub overestimation_size: f32,
}

impl Default for RasterizationConservativeState {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl RasterizationConservativeState {
    /// Returns a default `RasterizationConservativeState`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            mode: ConservativeRasterizationMode::Disabled,
            overestimation_size: 0.0,
        }
    }

    pub(crate) fn validate(self, device: &Device) -> Result<(), Box<ValidationError>> {
        let Self {
            mode,
            overestimation_size,
        } = self;

        let properties = device.physical_device().properties();

        mode.validate_device(device).map_err(|err| {
            err.add_context("mode").set_vuids(&[
                "VUID-VkPipelineRasterizationConservativeStateCreateInfoEXT-conservativeRasterizationMode-parameter",
            ])
        })?;

        if overestimation_size < 0.0
            || overestimation_size > properties.max_extra_primitive_overestimation_size.unwrap()
        {
            return Err(Box::new(ValidationError {
                context: "overestimation size".into(),
                problem: "the overestimation size is not in the range of 0.0 to `max_extra_primitive_overestimation_size` inclusive".into(),
                vuids: &[
                    "VUID-VkPipelineRasterizationConservativeStateCreateInfoEXT-extraPrimitiveOverestimationSize-01769",
                ],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[allow(clippy::wrong_self_convention, clippy::trivially_copy_pass_by_ref)]
    pub(crate) fn to_vk(&self) -> vk::PipelineRasterizationConservativeStateCreateInfoEXT<'static> {
        let &Self {
            mode,
            overestimation_size,
        } = self;

        vk::PipelineRasterizationConservativeStateCreateInfoEXT::default()
            .flags(vk::PipelineRasterizationConservativeStateCreateFlagsEXT::empty())
            .conservative_rasterization_mode(mode.into())
            .extra_primitive_overestimation_size(overestimation_size)
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// Describes how fragments will be generated based on how much is covered by a primitive.
    ConservativeRasterizationMode = ConservativeRasterizationModeEXT(i32);

    /// Conservative rasterization is disabled and rasterization proceeds as normal.
    Disabled = DISABLED,

    /// Fragments will be generated if any part of a primitive touches a pixel.
    Overestimate = OVERESTIMATE,

    /// Fragments will be generated only if a primitive completely covers a pixel.
    Underestimate = UNDERESTIMATE,
}

impl Default for ConservativeRasterizationMode {
    #[inline]
    fn default() -> ConservativeRasterizationMode {
        ConservativeRasterizationMode::Disabled
    }
}
