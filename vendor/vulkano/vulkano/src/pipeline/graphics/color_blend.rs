//! Configures how the color output of the fragment shader is written to the attachment.
//!
//! # Blending in details
//!
//! There are three kinds of color attachments for the purpose of blending:
//!
//! - Attachments with a floating-point or fixed point format.
//! - Attachments with a (non-normalized) integer format.
//! - Attachments with a normalized integer format.
//!
//! For floating-point and fixed-point formats, the blending operation is applied. For integer
//! formats, the logic operation is applied. For normalized integer formats, the logic operation
//! will take precedence if it is activated, otherwise the blending operation is applied.

use super::subpass::PipelineSubpassType;
use crate::{
    device::Device,
    format::Format,
    macros::{vulkan_bitflags, vulkan_enum},
    pipeline::inout_interface::{ShaderInterfaceLocationInfo, ShaderInterfaceLocationWidth},
    self_referential::self_referential,
    Requires, RequiresAllOf, RequiresOneOf, ValidationError,
};
use ash::vk;
use ash::vk::TaggedStructure;

use foldhash::HashMap;
use smallvec::SmallVec;

/// Describes how the color output of the fragment shader is written to the attachment. See the
/// documentation of the `blend` module for more info.
#[derive(Clone, Debug)]
pub struct ColorBlendState<'a> {
    /// Additional properties of the color blend state.
    ///
    /// The default value is empty.
    pub flags: ColorBlendStateFlags,

    /// Sets the logical operation to perform between the incoming fragment color and the existing
    /// fragment in the framebuffer attachment.
    ///
    /// If set to `Some`, the [`logic_op`](crate::device::DeviceFeatures::logic_op) feature must be
    /// enabled on the device.
    ///
    /// The default value is `None`.
    pub logic_op: Option<LogicOp>,

    /// Sets the blend and output state for each color attachment. The number of elements must
    /// match the number of color attachments in the subpass.
    ///
    /// If there are multiple elements, and the `blend` and `color_write_mask` members of each
    /// element differ, then the
    /// [`independent_blend`](crate::device::DeviceFeatures::independent_blend) feature must be
    /// enabled on the device.
    ///
    /// The default value is empty,
    /// which must be overridden if the subpass has color attachments.
    pub attachments: &'a [ColorBlendAttachmentState],

    /// The constant color to use for some of the `BlendFactor` variants.
    ///
    /// The default value is `[0.0; 4]`.
    pub blend_constants: [f32; 4],

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for ColorBlendState<'_> {
    /// Returns [`ColorBlendState::new(1)`].
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> ColorBlendState<'a> {
    /// Returns a default `ColorBlendState`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            flags: ColorBlendStateFlags::empty(),
            logic_op: None,
            attachments: &[],
            blend_constants: [0.0; 4],
            _ne: crate::NE,
        }
    }

    /// Enables logical operations with the given logical operation.
    #[inline]
    #[deprecated(since = "0.34.0")]
    pub fn logic_op(mut self, logic_op: LogicOp) -> Self {
        self.logic_op = Some(logic_op);
        self
    }

    /// Sets the blend constants.
    #[inline]
    #[deprecated(since = "0.34.0")]
    pub fn blend_constants(mut self, constants: [f32; 4]) -> Self {
        self.blend_constants = constants;
        self
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            logic_op,
            attachments,
            blend_constants: _,
            _ne: _,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkPipelineColorBlendStateCreateInfo-flags-parameter"])
        })?;

        if let Some(logic_op) = logic_op {
            if !device.enabled_features().logic_op {
                return Err(Box::new(ValidationError {
                    context: "logic_op".into(),
                    problem: "is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "logic_op",
                    )])]),
                    vuids: &["VUID-VkPipelineColorBlendStateCreateInfo-logicOpEnable-00606"],
                }));
            }

            logic_op.validate_device(device).map_err(|err| {
                err.add_context("logic_op")
                    .set_vuids(&["VUID-VkPipelineColorBlendStateCreateInfo-logicOpEnable-00607"])
            })?;
        }

        if device.enabled_features().independent_blend {
            for (index, state) in attachments.iter().enumerate() {
                state
                    .validate(device)
                    .map_err(|err| err.add_context(format!("attachments[{}]", index)))?;
            }
        } else if let Some(first) = attachments.first() {
            first
                .validate(device)
                .map_err(|err| err.add_context("attachments[0]"))?;

            for (index, state) in attachments.iter().enumerate().skip(1) {
                if state != first {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`attachments[{}]` does not equal `attachments[0]`",
                            index
                        )
                        .into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                            Requires::DeviceFeature("independent_blend"),
                        ])]),
                        vuids: &["VUID-VkPipelineColorBlendStateCreateInfo-pAttachments-00605"],
                        ..Default::default()
                    }));
                }
            }
        }

        Ok(())
    }

    pub(crate) fn validate_required_fragment_outputs(
        &self,
        subpass: PipelineSubpassType<'_>,
        fragment_shader_outputs: &HashMap<u32, ShaderInterfaceLocationInfo>,
    ) -> Result<(), Box<ValidationError>> {
        let validate_location =
            |attachment_index: usize, format: Format| -> Result<(), Box<ValidationError>> {
                let &ColorBlendAttachmentState {
                    ref blend,
                    color_write_mask,
                    color_write_enable,
                } = &self.attachments[attachment_index];

                if !color_write_enable || color_write_mask.is_empty() {
                    return Ok(());
                }

                // An output component is written if it exists in the format,
                // and is included in the write mask.
                let format_components = format.components();
                let written_output_components = [
                    format_components[0] != 0 && color_write_mask.intersects(ColorComponents::R),
                    format_components[1] != 0 && color_write_mask.intersects(ColorComponents::G),
                    format_components[2] != 0 && color_write_mask.intersects(ColorComponents::B),
                    format_components[3] != 0 && color_write_mask.intersects(ColorComponents::A),
                ];

                // Gather the input components (for source0 and source1) that are needed to
                // produce the components in `written_components`.
                let mut source_components_used = [ColorComponents::empty(); 2];

                fn add_components(dst: &mut [ColorComponents; 2], src: [ColorComponents; 2]) {
                    for (dst, src) in dst.iter_mut().zip(src) {
                        *dst |= src;
                    }
                }

                if let Some(blend) = blend {
                    let &AttachmentBlend {
                        src_color_blend_factor,
                        dst_color_blend_factor,
                        color_blend_op,
                        src_alpha_blend_factor,
                        dst_alpha_blend_factor,
                        alpha_blend_op,
                    } = blend;

                    let mut add_blend =
                        |output_component: usize,
                         blend_op: BlendOp,
                         blend_factors: [BlendFactor; 2]| {
                            if written_output_components[output_component] {
                                add_components(
                                    &mut source_components_used,
                                    blend_op.source_components_used(output_component),
                                );

                                if blend_op.uses_blend_factors() {
                                    for blend_factor in blend_factors {
                                        add_components(
                                            &mut source_components_used,
                                            blend_factor.source_components_used(output_component),
                                        );
                                    }
                                }
                            }
                        };

                    add_blend(
                        0,
                        color_blend_op,
                        [src_color_blend_factor, dst_color_blend_factor],
                    );
                    add_blend(
                        1,
                        color_blend_op,
                        [src_color_blend_factor, dst_color_blend_factor],
                    );
                    add_blend(
                        2,
                        color_blend_op,
                        [src_color_blend_factor, dst_color_blend_factor],
                    );
                    add_blend(
                        3,
                        alpha_blend_op,
                        [src_alpha_blend_factor, dst_alpha_blend_factor],
                    );
                } else {
                    let mut add_passthrough = |output_component: usize| {
                        if written_output_components[output_component] {
                            add_components(
                                &mut source_components_used,
                                [
                                    ColorComponents::from_index(output_component),
                                    ColorComponents::empty(),
                                ],
                            )
                        }
                    };

                    add_passthrough(0);
                    add_passthrough(1);
                    add_passthrough(2);
                    add_passthrough(3);
                }

                // If no components from either input source are used,
                // then there is nothing more to check.
                if source_components_used == [ColorComponents::empty(); 2] {
                    return Ok(());
                }

                let location = attachment_index as u32;
                let location_info = fragment_shader_outputs.get(&location).ok_or_else(|| {
                    Box::new(ValidationError {
                        problem: format!(
                            "the subpass and color blend state of color attachment {0} use the \
                            fragment shader output, but \
                            the fragment shader does not have an output variable with location {0}",
                            location,
                        )
                        .into(),
                        ..Default::default()
                    })
                })?;
                let attachment_numeric_type = format.numeric_format_color().unwrap().numeric_type();

                if attachment_numeric_type != location_info.numeric_type {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "the subpass and color blend state of color attachment {0} use the \
                            fragment shader output, but \
                            the numeric type of the color attachment format ({1:?}) \
                            does not equal the numeric type of the fragment shader output \
                            variable with location {0} ({2:?})",
                            location, attachment_numeric_type, location_info.numeric_type,
                        )
                        .into(),
                        ..Default::default()
                    }));
                }

                if location_info.width != ShaderInterfaceLocationWidth::Bits32 {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "the subpass and color blend state of color attachment {0} use the \
                            fragment shader output, and \
                            the color attachment format is not 64 bit, but \
                            the format of the fragment output variable with location {0} is 64-bit",
                            location,
                        )
                        .into(),
                        ..Default::default()
                    }));
                }

                for (index, (source_components_provided, source_components_used)) in location_info
                    .components
                    .into_iter()
                    .zip(source_components_used)
                    .enumerate()
                {
                    let missing_components = source_components_used - source_components_provided;

                    if !missing_components.is_empty() {
                        let component = if missing_components.intersects(ColorComponents::R) {
                            0
                        } else if missing_components.intersects(ColorComponents::G) {
                            1
                        } else if missing_components.intersects(ColorComponents::B) {
                            2
                        } else if missing_components.intersects(ColorComponents::A) {
                            3
                        } else {
                            unreachable!()
                        };

                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the subpass and color blend state of color attachment {0} use \
                                location {0}, index {1}, component {2} from the fragment shader \
                                output, but the fragment shader does not have an output variable \
                                with that location, index and component",
                                location, index, component,
                            )
                            .into(),
                            ..Default::default()
                        }));
                    }
                }

                Ok(())
            };

        match subpass {
            PipelineSubpassType::BeginRenderPass(subpass) => {
                debug_assert_eq!(
                    self.attachments.len(),
                    subpass.subpass_desc().color_attachments.len()
                );
                let render_pass_attachments = subpass.render_pass().attachments();

                for (attachment_index, atch_ref) in
                    subpass.subpass_desc().color_attachments.iter().enumerate()
                {
                    match atch_ref {
                        Some(atch_ref) => validate_location(
                            attachment_index,
                            render_pass_attachments[atch_ref.attachment as usize].format,
                        )?,
                        None => continue,
                    }
                }
            }
            PipelineSubpassType::BeginRendering(rendering_create_info) => {
                debug_assert_eq!(
                    self.attachments.len(),
                    rendering_create_info.color_attachment_formats.len()
                );

                for (attachment_index, &format) in rendering_create_info
                    .color_attachment_formats
                    .iter()
                    .enumerate()
                {
                    match format {
                        Some(format) => validate_location(attachment_index, format)?,
                        None => continue,
                    }
                }
            }
        }

        Ok(())
    }

    pub(crate) fn to_vk(
        &self,
        fields1_vk: &'a ColorBlendStateFields1Vk,
        extensions_vk: &'a mut ColorBlendStateExtensionsVk<'_>,
    ) -> vk::PipelineColorBlendStateCreateInfo<'a> {
        let &Self {
            flags,
            logic_op,
            attachments: _,
            blend_constants,
            _ne: _,
        } = self;
        let ColorBlendStateFields1Vk {
            color_blend_attachments_vk,
            color_write_enables_vk: _,
        } = fields1_vk;

        let (logic_op_enable_vk, logic_op_vk) = if let Some(logic_op) = logic_op {
            (true, logic_op.into())
        } else {
            (false, Default::default())
        };

        let mut val_vk = vk::PipelineColorBlendStateCreateInfo::default()
            .flags(flags.into())
            .logic_op_enable(logic_op_enable_vk)
            .logic_op(logic_op_vk)
            .attachments(color_blend_attachments_vk)
            .blend_constants(blend_constants);

        let ColorBlendStateExtensionsVk { color_write_vk } = extensions_vk;

        if let Some(next) = color_write_vk {
            val_vk = val_vk.push(next);
        }

        val_vk
    }

    pub(crate) fn to_vk_extensions(
        &self,
        fields1_vk: &'a ColorBlendStateFields1Vk,
    ) -> ColorBlendStateExtensionsVk<'a> {
        let ColorBlendStateFields1Vk {
            color_blend_attachments_vk: _,
            color_write_enables_vk,
        } = fields1_vk;

        let color_write_vk = (!color_write_enables_vk.is_empty()).then(|| {
            vk::PipelineColorWriteCreateInfoEXT::default()
                .color_write_enables(color_write_enables_vk)
        });

        ColorBlendStateExtensionsVk { color_write_vk }
    }

    pub(crate) fn to_vk_fields1(&self) -> ColorBlendStateFields1Vk {
        let mut color_blend_attachments_vk = SmallVec::with_capacity(self.attachments.len());
        let mut color_write_enables_vk = SmallVec::with_capacity(self.attachments.len());
        let mut has_color_write_enables = false;

        for color_blend_attachment_state in self.attachments {
            color_blend_attachments_vk.push(color_blend_attachment_state.to_vk());
            color_write_enables_vk.push(color_blend_attachment_state.color_write_enable as _);
            has_color_write_enables |= !color_blend_attachment_state.color_write_enable;
        }

        ColorBlendStateFields1Vk {
            color_blend_attachments_vk,
            color_write_enables_vk: if has_color_write_enables {
                color_write_enables_vk
            } else {
                Default::default()
            },
        }
    }

    pub(crate) fn to_owned(&self) -> OwnedColorBlendState {
        let attachments = self.attachments.to_owned();

        OwnedColorBlendState::new(attachments, |attachments| ColorBlendState {
            attachments,
            _ne: crate::NE,
            ..*self
        })
    }
}

pub(crate) struct ColorBlendStateExtensionsVk<'a> {
    pub(crate) color_write_vk: Option<vk::PipelineColorWriteCreateInfoEXT<'a>>,
}

pub(crate) struct ColorBlendStateFields1Vk {
    color_blend_attachments_vk: SmallVec<[vk::PipelineColorBlendAttachmentState; 4]>,
    color_write_enables_vk: SmallVec<[vk::Bool32; 4]>,
}

self_referential! {
    mod owned_color_blend_state {
        pub(crate) struct OwnedColorBlendState {
            inner: ColorBlendState<'_>,
            attachments: Vec<ColorBlendAttachmentState>,
        }
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags specifying additional properties of the color blend state.
    ColorBlendStateFlags = PipelineColorBlendStateCreateFlags(u32);

    /* TODO: enable
    // TODO: document
    RASTERIZATION_ORDER_ATTACHMENT_ACCESS = RASTERIZATION_ORDER_ATTACHMENT_ACCESS_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_rasterization_order_attachment_access)]),
        RequiresAllOf([DeviceExtension(arm_rasterization_order_attachment_access)]),
    ]), */
}

vulkan_enum! {
    #[non_exhaustive]
    /// Which logical operation to apply to the output values.
    ///
    /// The operation is applied individually for each channel (red, green, blue and alpha).
    ///
    /// Only relevant for integer or unsigned attachments.
    ///
    /// Also note that some implementations don't support logic operations.
    LogicOp = LogicOp(i32);

    /// Returns `0`.
    Clear = CLEAR,

    /// Returns `source & destination`.
    And = AND,

    /// Returns `source & !destination`.
    AndReverse = AND_REVERSE,

    /// Returns `source`.
    Copy = COPY,

    /// Returns `!source & destination`.
    AndInverted = AND_INVERTED,

    /// Returns `destination`.
    Noop = NO_OP,

    /// Returns `source ^ destination`.
    Xor = XOR,

    /// Returns `source | destination`.
    Or = OR,

    /// Returns `!(source | destination)`.
    Nor = NOR,

    /// Returns `!(source ^ destination)`.
    Equivalent = EQUIVALENT,

    /// Returns `!destination`.
    Invert = INVERT,

    /// Returns `source | !destination.
    OrReverse = OR_REVERSE,

    /// Returns `!source`.
    CopyInverted = COPY_INVERTED,

    /// Returns `!source | destination`.
    OrInverted = OR_INVERTED,

    /// Returns `!(source & destination)`.
    Nand = NAND,

    /// Returns `!0` (all bits set to 1).
    Set = SET,

}

impl Default for LogicOp {
    #[inline]
    fn default() -> LogicOp {
        LogicOp::Noop
    }
}

/// Describes how a framebuffer color attachment is handled in the pipeline during the color
/// blend stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColorBlendAttachmentState {
    /// The blend parameters for the attachment.
    ///
    /// If set to `None`, blending is disabled, and all incoming pixels will be used directly.
    ///
    /// The default value is `None`.
    pub blend: Option<AttachmentBlend>,

    /// Sets which components of the final pixel value are written to the attachment.
    ///
    /// The default value is `ColorComponents::all()`.
    pub color_write_mask: ColorComponents,

    /// Sets whether anything at all is written to the attachment. If enabled, the pixel data
    /// that is written is determined by the `color_write_mask`. If disabled, the mask is ignored
    /// and nothing is written.
    ///
    /// If set to `false`, the
    /// [`color_write_enable`](crate::device::DeviceFeatures::color_write_enable) feature must be
    /// enabled on the device.
    ///
    /// The default value is `true`.
    pub color_write_enable: bool,
}

impl Default for ColorBlendAttachmentState {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl ColorBlendAttachmentState {
    /// Returns a default `ColorBlendAttachmentState`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            blend: None,
            color_write_mask: ColorComponents::all(),
            color_write_enable: true,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            ref blend,
            color_write_mask: _,
            color_write_enable,
        } = self;

        if let Some(blend) = blend {
            blend
                .validate(device)
                .map_err(|err| err.add_context("blend"))?;
        }

        if !color_write_enable && !device.enabled_features().color_write_enable {
            return Err(Box::new(ValidationError {
                context: "color_write_enable".into(),
                problem: "is `false`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "color_write_enable",
                )])]),
                vuids: &["VUID-VkPipelineColorWriteCreateInfoEXT-pAttachments-04801"],
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::PipelineColorBlendAttachmentState {
        let &Self {
            ref blend,
            color_write_mask,
            color_write_enable: _,
        } = self;

        vk::PipelineColorBlendAttachmentState {
            color_write_mask: color_write_mask.into(),
            ..blend
                .as_ref()
                .map(AttachmentBlend::to_vk)
                .unwrap_or_default()
        }
    }
}

/// Describes how the blending system should behave for an attachment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttachmentBlend {
    /// The operation to apply to the source color component before applying `color_op`.
    ///
    /// The default value is [`BlendFactor::SrcColor`].
    pub src_color_blend_factor: BlendFactor,

    /// The operation to apply to the destination color component before applying `color_op`.
    ///
    /// The default value is [`BlendFactor::Zero`].
    pub dst_color_blend_factor: BlendFactor,

    /// The operation to apply between the color components of the source and destination pixels,
    /// to produce the final pixel value.
    ///
    /// The default value is [`BlendOp::Add`].
    pub color_blend_op: BlendOp,

    /// The operation to apply to the source alpha component before applying `alpha_op`.
    ///
    /// The default value is [`BlendFactor::SrcColor`].
    pub src_alpha_blend_factor: BlendFactor,

    /// The operation to apply to the destination alpha component before applying `alpha_op`.
    ///
    /// The default value is [`BlendFactor::Zero`].
    pub dst_alpha_blend_factor: BlendFactor,

    /// The operation to apply between the alpha component of the source and destination pixels,
    /// to produce the final pixel value.
    ///
    /// The default value is [`BlendOp::Add`].
    pub alpha_blend_op: BlendOp,
}

impl Default for AttachmentBlend {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl AttachmentBlend {
    /// Returns a default `AttachmentBlend`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            src_color_blend_factor: BlendFactor::SrcColor,
            dst_color_blend_factor: BlendFactor::Zero,
            color_blend_op: BlendOp::Add,
            src_alpha_blend_factor: BlendFactor::SrcColor,
            dst_alpha_blend_factor: BlendFactor::Zero,
            alpha_blend_op: BlendOp::Add,
        }
    }

    /// Builds an `AttachmentBlend` where the output of the fragment shader is ignored and the
    /// destination is untouched.
    #[inline]
    pub fn ignore_source() -> Self {
        Self {
            src_color_blend_factor: BlendFactor::Zero,
            dst_color_blend_factor: BlendFactor::DstColor,
            color_blend_op: BlendOp::Add,
            src_alpha_blend_factor: BlendFactor::Zero,
            dst_alpha_blend_factor: BlendFactor::DstColor,
            alpha_blend_op: BlendOp::Add,
        }
    }

    /// Builds an `AttachmentBlend` where the output will be merged with the existing value
    /// based on the alpha of the source.
    #[inline]
    pub fn alpha() -> Self {
        Self {
            src_color_blend_factor: BlendFactor::SrcAlpha,
            dst_color_blend_factor: BlendFactor::OneMinusSrcAlpha,
            color_blend_op: BlendOp::Add,
            src_alpha_blend_factor: BlendFactor::SrcAlpha,
            dst_alpha_blend_factor: BlendFactor::OneMinusSrcAlpha,
            alpha_blend_op: BlendOp::Add,
        }
    }

    /// Builds an `AttachmentBlend` where the colors are added, and alpha is set to the maximum of
    /// the two.
    #[inline]
    pub fn additive() -> Self {
        Self {
            src_color_blend_factor: BlendFactor::One,
            dst_color_blend_factor: BlendFactor::One,
            color_blend_op: BlendOp::Add,
            src_alpha_blend_factor: BlendFactor::One,
            dst_alpha_blend_factor: BlendFactor::One,
            alpha_blend_op: BlendOp::Max,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            src_color_blend_factor,
            dst_color_blend_factor,
            color_blend_op,
            src_alpha_blend_factor,
            dst_alpha_blend_factor,
            alpha_blend_op,
        } = self;

        src_color_blend_factor
            .validate_device(device)
            .map_err(|err| {
                err.add_context("src_color_blend_factor").set_vuids(&[
                    "VUID-VkPipelineColorBlendAttachmentState-srcColorBlendFactor-parameter",
                ])
            })?;

        dst_color_blend_factor
            .validate_device(device)
            .map_err(|err| {
                err.add_context("dst_color_blend_factor").set_vuids(&[
                    "VUID-VkPipelineColorBlendAttachmentState-dstColorBlendFactor-parameter",
                ])
            })?;

        color_blend_op.validate_device(device).map_err(|err| {
            err.add_context("color_blend_op")
                .set_vuids(&["VUID-VkPipelineColorBlendAttachmentState-colorBlendOp-parameter"])
        })?;

        src_alpha_blend_factor
            .validate_device(device)
            .map_err(|err| {
                err.add_context("src_alpha_blend_factor").set_vuids(&[
                    "VUID-VkPipelineColorBlendAttachmentState-srcAlphaBlendFactor-parameter",
                ])
            })?;

        dst_alpha_blend_factor
            .validate_device(device)
            .map_err(|err| {
                err.add_context("dst_alpha_blend_factor").set_vuids(&[
                    "VUID-VkPipelineColorBlendAttachmentState-dstAlphaBlendFactor-parameter",
                ])
            })?;

        alpha_blend_op.validate_device(device).map_err(|err| {
            err.add_context("alpha_blend_op")
                .set_vuids(&["VUID-VkPipelineColorBlendAttachmentState-alphaBlendOp-parameter"])
        })?;

        if !device.enabled_features().dual_src_blend {
            if matches!(
                src_color_blend_factor,
                BlendFactor::Src1Color
                    | BlendFactor::OneMinusSrc1Color
                    | BlendFactor::Src1Alpha
                    | BlendFactor::OneMinusSrc1Alpha
            ) {
                return Err(Box::new(ValidationError {
                    context: "src_color_blend_factor".into(),
                    problem: "is `BlendFactor::Src1*`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "dual_src_blend",
                    )])]),
                    vuids: &["VUID-VkPipelineColorBlendAttachmentState-srcColorBlendFactor-00608"],
                }));
            }

            if matches!(
                dst_color_blend_factor,
                BlendFactor::Src1Color
                    | BlendFactor::OneMinusSrc1Color
                    | BlendFactor::Src1Alpha
                    | BlendFactor::OneMinusSrc1Alpha
            ) {
                return Err(Box::new(ValidationError {
                    context: "dst_color_blend_factor".into(),
                    problem: "is `BlendFactor::Src1*`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "dual_src_blend",
                    )])]),
                    vuids: &["VUID-VkPipelineColorBlendAttachmentState-dstColorBlendFactor-00609"],
                }));
            }

            if matches!(
                src_alpha_blend_factor,
                BlendFactor::Src1Color
                    | BlendFactor::OneMinusSrc1Color
                    | BlendFactor::Src1Alpha
                    | BlendFactor::OneMinusSrc1Alpha
            ) {
                return Err(Box::new(ValidationError {
                    context: "src_alpha_blend_factor".into(),
                    problem: "is `BlendFactor::Src1*`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "dual_src_blend",
                    )])]),
                    vuids: &["VUID-VkPipelineColorBlendAttachmentState-srcAlphaBlendFactor-00610"],
                }));
            }

            if matches!(
                dst_alpha_blend_factor,
                BlendFactor::Src1Color
                    | BlendFactor::OneMinusSrc1Color
                    | BlendFactor::Src1Alpha
                    | BlendFactor::OneMinusSrc1Alpha
            ) {
                return Err(Box::new(ValidationError {
                    context: "dst_alpha_blend_factor".into(),
                    problem: "is `BlendFactor::Src1*`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "dual_src_blend",
                    )])]),
                    vuids: &["VUID-VkPipelineColorBlendAttachmentState-dstAlphaBlendFactor-00611"],
                }));
            }
        }

        if device.enabled_extensions().khr_portability_subset
            && !device.enabled_features().constant_alpha_color_blend_factors
        {
            if matches!(
                src_color_blend_factor,
                BlendFactor::ConstantAlpha | BlendFactor::OneMinusConstantAlpha
            ) {
                return Err(Box::new(ValidationError {
                    problem: "this device is a portability subset device, and \
                        `src_color_blend_factor` is `BlendFactor::ConstantAlpha` or \
                        `BlendFactor::OneMinusConstantAlpha`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "constant_alpha_color_blend_factors",
                    )])]),
                    vuids: &["VUID-VkPipelineColorBlendAttachmentState-constantAlphaColorBlendFactors-04454"],
                    ..Default::default()
                }));
            }

            if matches!(
                dst_color_blend_factor,
                BlendFactor::ConstantAlpha | BlendFactor::OneMinusConstantAlpha
            ) {
                return Err(Box::new(ValidationError {
                    problem: "this device is a portability subset device, and \
                        `dst_color_blend_factor` is `BlendFactor::ConstantAlpha` or \
                        `BlendFactor::OneMinusConstantAlpha`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "constant_alpha_color_blend_factors",
                    )])]),
                    vuids: &["VUID-VkPipelineColorBlendAttachmentState-constantAlphaColorBlendFactors-04455"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::PipelineColorBlendAttachmentState {
        let &Self {
            src_color_blend_factor,
            dst_color_blend_factor,
            color_blend_op,
            src_alpha_blend_factor,
            dst_alpha_blend_factor,
            alpha_blend_op,
        } = self;

        vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::TRUE,
            src_color_blend_factor: src_color_blend_factor.into(),
            dst_color_blend_factor: dst_color_blend_factor.into(),
            color_blend_op: color_blend_op.into(),
            src_alpha_blend_factor: src_alpha_blend_factor.into(),
            dst_alpha_blend_factor: dst_alpha_blend_factor.into(),
            alpha_blend_op: alpha_blend_op.into(),
            color_write_mask: vk::ColorComponentFlags::empty(),
        }
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// The operation that takes `source` (output from the fragment shader), `destination` (value
    /// currently in the framebuffer attachment) and `blend_constant` input values,
    /// and produces new inputs to be fed to `BlendOp`.
    ///
    /// Some operations take `source1` as an input, representing the second source value. The
    /// [`dual_src_blend`](crate::device::DeviceFeatures::dual_src_blend) feature must be enabled on the
    /// device when these are used.
    BlendFactor = BlendFactor(i32);

    /// Always `0`.
    Zero = ZERO,

    /// Always `1`.
    One = ONE,

    /// `source` component-wise.
    SrcColor = SRC_COLOR,

    /// `1 - source` component-wise.
    OneMinusSrcColor = ONE_MINUS_SRC_COLOR,

    /// `destination` component-wise.
    DstColor = DST_COLOR,

    /// `1 - destination` component-wise.
    OneMinusDstColor = ONE_MINUS_DST_COLOR,

    /// `source.a` for all components.
    SrcAlpha = SRC_ALPHA,

    /// `1 - source.a` for all components.
    OneMinusSrcAlpha = ONE_MINUS_SRC_ALPHA,

    /// `destination.a` for all components.
    DstAlpha = DST_ALPHA,

    /// `1 - destination.a` for all components.
    OneMinusDstAlpha = ONE_MINUS_DST_ALPHA,

    /// `blend_constants` component-wise.
    ConstantColor = CONSTANT_COLOR,

    /// `1 - blend_constants` component-wise.
    OneMinusConstantColor = ONE_MINUS_CONSTANT_COLOR,

    /// `blend_constants.a` for all components.
    ///
    /// On [portability subset](crate::instance#portability-subset-devices-and-the-enumerate_portability-flag)
    /// devices, if this value is used for the `color_source` or `color_destination` blend factors,
    /// then the
    /// [`constant_alpha_color_blend_factors`](crate::device::DeviceFeatures::constant_alpha_color_blend_factors)
    /// feature must be enabled on the device.
    ConstantAlpha = CONSTANT_ALPHA,

    /// `1 - blend_constants.a` for all components.
    ///
    /// On [portability subset](crate::instance#portability-subset-devices-and-the-enumerate_portability-flag)
    /// devices, if this value is used for the `color_source` or `color_destination` blend factors,
    /// then the
    /// [`constant_alpha_color_blend_factors`](crate::device::DeviceFeatures::constant_alpha_color_blend_factors)
    /// feature must be enabled on the device.
    OneMinusConstantAlpha = ONE_MINUS_CONSTANT_ALPHA,

    /// For the alpha component, always `1`. For the color components,
    /// `min(source.a, 1 - destination.a)` for all components.
    SrcAlphaSaturate = SRC_ALPHA_SATURATE,

    /// `source1` component-wise.
    Src1Color = SRC1_COLOR,

    /// `1 - source1` component-wise.
    OneMinusSrc1Color = ONE_MINUS_SRC1_COLOR,

    /// `source1.a` for all components.
    Src1Alpha = SRC1_ALPHA,

    /// `1 - source1.a` for all components.
    OneMinusSrc1Alpha = ONE_MINUS_SRC1_ALPHA,
}

impl BlendFactor {
    const fn source_components_used(self, output_component: usize) -> [ColorComponents; 2] {
        match self {
            BlendFactor::Zero
            | BlendFactor::One
            | BlendFactor::DstColor
            | BlendFactor::OneMinusDstColor
            | BlendFactor::DstAlpha
            | BlendFactor::OneMinusDstAlpha
            | BlendFactor::ConstantColor
            | BlendFactor::OneMinusConstantColor
            | BlendFactor::ConstantAlpha
            | BlendFactor::OneMinusConstantAlpha => [ColorComponents::empty(); 2],
            BlendFactor::SrcColor | BlendFactor::OneMinusSrcColor => [
                ColorComponents::from_index(output_component),
                ColorComponents::empty(),
            ],
            BlendFactor::Src1Color | BlendFactor::OneMinusSrc1Color => [
                ColorComponents::empty(),
                ColorComponents::from_index(output_component),
            ],
            BlendFactor::SrcAlpha
            | BlendFactor::OneMinusSrcAlpha
            | BlendFactor::SrcAlphaSaturate => [ColorComponents::A, ColorComponents::empty()],
            BlendFactor::Src1Alpha | BlendFactor::OneMinusSrc1Alpha => {
                [ColorComponents::empty(), ColorComponents::A]
            }
        }
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// The arithmetic operation that is applied between the `source` and `destination` component
    /// values, after the appropriate `BlendFactor` is applied to both.
    BlendOp = BlendOp(i32);

    /// `source + destination`.
    Add = ADD,

    /// `source - destination`.
    Subtract = SUBTRACT,

    /// `destination - source`.
    ReverseSubtract = REVERSE_SUBTRACT,

    /// `min(source, destination)`.
    Min = MIN,

    /// `max(source, destination)`.
    Max = MAX,

    /* TODO: enable
    // TODO: document
    Zero = ZERO_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Src = SRC_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Dst = DST_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    SrcOver = SRC_OVER_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    DstOver = DST_OVER_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    SrcIn = SRC_IN_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    DstIn = DST_IN_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    SrcOut = SRC_OUT_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    DstOut = DST_OUT_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    SrcAtop = SRC_ATOP_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    DstAtop = DST_ATOP_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Xor = XOR_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Multiply = MULTIPLY_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Screen = SCREEN_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Overlay = OVERLAY_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Darken = DARKEN_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Lighten = LIGHTEN_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Colordodge = COLORDODGE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Colorburn = COLORBURN_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Hardlight = HARDLIGHT_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Softlight = SOFTLIGHT_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Difference = DIFFERENCE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Exclusion = EXCLUSION_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Invert = INVERT_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    InvertRgb = INVERT_RGB_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Lineardodge = LINEARDODGE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Linearburn = LINEARBURN_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Vividlight = VIVIDLIGHT_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Linearlight = LINEARLIGHT_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Pinlight = PINLIGHT_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Hardmix = HARDMIX_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    HslHue = HSL_HUE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    HslSaturation = HSL_SATURATION_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    HslColor = HSL_COLOR_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    HslLuminosity = HSL_LUMINOSITY_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Plus = PLUS_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    PlusClamped = PLUS_CLAMPED_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    PlusClampedAlpha = PLUS_CLAMPED_ALPHA_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    PlusDarker = PLUS_DARKER_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Minus = MINUS_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    MinusClamped = MINUS_CLAMPED_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Contrast = CONTRAST_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    InvertOvg = INVERT_OVG_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Red = RED_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Green = GREEN_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Blue = BLUE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/
}

impl BlendOp {
    /// Returns whether the blend op will use the specified blend factors, or ignore them.
    #[inline]
    pub const fn uses_blend_factors(self) -> bool {
        match self {
            BlendOp::Add | BlendOp::Subtract | BlendOp::ReverseSubtract => true,
            BlendOp::Min | BlendOp::Max => false,
        }
    }

    const fn source_components_used(self, output_component: usize) -> [ColorComponents; 2] {
        match self {
            BlendOp::Add
            | BlendOp::Subtract
            | BlendOp::ReverseSubtract
            | BlendOp::Min
            | BlendOp::Max => [
                ColorComponents::from_index(output_component),
                ColorComponents::empty(),
            ],
        }
    }
}

vulkan_bitflags! {
    /// A mask specifying color components that can be written to a framebuffer attachment.
    ColorComponents = ColorComponentFlags(u32);

    /// The red component.
    R = R,

    /// The green component.
    G = G,

    /// The blue component.
    B = B,

    /// The alpha component.
    A = A,
}

impl ColorComponents {
    #[inline]
    pub(crate) const fn from_index(index: usize) -> Self {
        match index {
            0 => ColorComponents::R,
            1 => ColorComponents::G,
            2 => ColorComponents::B,
            3 => ColorComponents::A,
            _ => unreachable!(),
        }
    }
}
