//! Fragment shading rate introduces the ability to change the rate at which fragments are shaded.
//!
//! This feature is part of the [`khr_fragment_shading_rate`] device extension.
//!
//! The fragment shading rate can be controlled per-draw, per-primitive, and per-region:
//! - Per-draw shading rate requires the [`pipeline_fragment_shading_rate`] feature to be enabled
//!   on the device, and the rate can be set in a graphics pipeline or dynamically via
//!   [`set_fragment_shading_rate`].
//! - Per-primitive shading rate requires the [`primitive_fragment_shading_rate`] feature to be
//!   enabled on the device, and is set in the last active pre-rasterization shader stage.
//! - Per-region shading rate requires the [`attachment_fragment_shading_rate`] feature to be
//!   enabled on the device and an additional specialized fragment shading rate image attachment.
//!
//! `vulkano` does not currently support per-region shading state.
//!
//! [`khr_fragment_shading_rate`]: crate::device::DeviceExtensions::khr_fragment_shading_rate
//! [`pipeline_fragment_shading_rate`]: crate::device::DeviceFeatures::pipeline_fragment_shading_rate
//! [`set_fragment_shading_rate`]: crate::command_buffer::AutoCommandBufferBuilder::set_fragment_shading_rate
//! [`primitive_fragment_shading_rate`]: crate::device::DeviceFeatures::primitive_fragment_shading_rate
//! [`attachment_fragment_shading_rate`]: crate::device::DeviceFeatures::attachment_fragment_shading_rate

use crate::{device::Device, macros::vulkan_enum, ValidationError};
use ash::vk;

/// The state in a graphics pipeline describing the fragment shading rate.
#[derive(Clone, Debug)]
pub struct FragmentShadingRateState {
    /// The pipeline fragment shading rate.
    ///
    /// The default value is `[1, 1]`.
    pub fragment_size: [u32; 2],

    /// Determines how the pipeline, primitive, and attachment shading rates are combined for
    /// fragments generated.
    ///
    /// The default value is `[FragmentShadingRateCombinerOp::Keep; 2]`.
    pub combiner_ops: [FragmentShadingRateCombinerOp; 2],

    pub _ne: crate::NonExhaustive<'static>,
}

impl Default for FragmentShadingRateState {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl FragmentShadingRateState {
    /// Returns a default `FragmentShadingRateState`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            fragment_size: [1, 1],
            combiner_ops: [FragmentShadingRateCombinerOp::Keep; 2],
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            fragment_size,
            combiner_ops,
            _ne: _,
        } = self;

        let properties = device.physical_device().properties();
        let features = device.enabled_features();

        if !matches!(fragment_size[0], 1 | 2 | 4) {
            return Err(Box::new(ValidationError {
                context: "fragment_size[0]".into(),
                problem: "`fragment_size[0]` must be 1, 2, or 4".into(),
                vuids: &[
                    "VUID-VkGraphicsPipelineCreateInfo-pDynamicState-04494",
                    "VUID-VkGraphicsPipelineCreateInfo-pDynamicState-04496",
                    "VUID-VkGraphicsPipelineCreateInfo-pDynamicState-04498",
                ],
                ..Default::default()
            }));
        }

        if !matches!(fragment_size[1], 1 | 2 | 4) {
            return Err(Box::new(ValidationError {
                context: "fragment_size[1]".into(),
                problem: "`fragment_size[1]` must be 1, 2, or 4".into(),
                vuids: &[
                    "VUID-VkGraphicsPipelineCreateInfo-pDynamicState-04495",
                    "VUID-VkGraphicsPipelineCreateInfo-pDynamicState-04497",
                    "VUID-VkGraphicsPipelineCreateInfo-pDynamicState-04499",
                ],
                ..Default::default()
            }));
        }

        if !features.pipeline_fragment_shading_rate {
            return Err(Box::new(ValidationError {
                context: "features.pipeline_fragment_shading_rate".into(),
                problem: "the `pipeline_fragment_shading_rate` feature must be enabled".into(),
                vuids: &["VUID-VkGraphicsPipelineCreateInfo-pDynamicState-04500"],
                ..Default::default()
            }));
        }

        combiner_ops[0].validate_device(device).map_err(|err| {
            err.add_context("combiner_ops[0]")
                .set_vuids(&["VUID-VkGraphicsPipelineCreateInfo-pDynamicState-06567"])
        })?;

        combiner_ops[1].validate_device(device).map_err(|err| {
            err.add_context("combiner_ops[1]")
                .set_vuids(&["VUID-VkGraphicsPipelineCreateInfo-pDynamicState-06568"])
        })?;

        if !features.primitive_fragment_shading_rate
            && combiner_ops[0] != FragmentShadingRateCombinerOp::Keep
        {
            return Err(Box::new(ValidationError {
                context: "combiner_ops[0]".into(),
                problem: "the `primitive_fragment_shading_rate` feature must be enabled if \
                    `combiner_ops[0]` is not `FragmentShadingRateCombinerOp::Keep`"
                    .into(),
                vuids: &["VUID-VkGraphicsPipelineCreateInfo-pDynamicState-04501"],
                ..Default::default()
            }));
        }

        if !features.attachment_fragment_shading_rate
            && combiner_ops[1] != FragmentShadingRateCombinerOp::Keep
        {
            return Err(Box::new(ValidationError {
                context: "combiner_ops[1]".into(),
                problem: "the `attachment_fragment_shading_rate` feature must be enabled if \
                    `combiner_ops[1]` is not `FragmentShadingRateCombinerOp::Keep`"
                    .into(),
                vuids: &["VUID-VkGraphicsPipelineCreateInfo-pDynamicState-04502"],
                ..Default::default()
            }));
        }

        if let Some(fragment_shading_rate_non_trivial_combiner_ops) =
            properties.fragment_shading_rate_non_trivial_combiner_ops
        {
            if !fragment_shading_rate_non_trivial_combiner_ops
                && (!matches!(
                    combiner_ops[0],
                    FragmentShadingRateCombinerOp::Keep | FragmentShadingRateCombinerOp::Replace
                ) || !matches!(
                    combiner_ops[1],
                    FragmentShadingRateCombinerOp::Keep | FragmentShadingRateCombinerOp::Replace
                ))
            {
                return Err(Box::new(ValidationError {
                context: "combiner_ops[0]".into(),
                problem: "the `fragment_shading_rate_non_trivial_combiner_ops` feature must be \
                    enabled if `combiner_ops[0]` or `combiner_ops[1]` is not \
                    `FragmentShadingRateCombinerOp::Keep` or \
                    `FragmentShadingRateCombinerOp::Replace`"
                    .into(),
                vuids: &[
                    "VUID-VkGraphicsPipelineCreateInfo-fragmentShadingRateNonTrivialCombinerOps-04506",
                ],
                ..Default::default()
            }));
            }
        }

        Ok(())
    }

    pub(crate) fn to_vk<'a>(&self) -> vk::PipelineFragmentShadingRateStateCreateInfoKHR<'a> {
        let fragment_size = vk::Extent2D {
            width: self.fragment_size[0],
            height: self.fragment_size[1],
        };
        let combiner_ops: [vk::FragmentShadingRateCombinerOpKHR; 2] =
            [self.combiner_ops[0].into(), self.combiner_ops[1].into()];

        vk::PipelineFragmentShadingRateStateCreateInfoKHR::default()
            .fragment_size(fragment_size)
            .combiner_ops(combiner_ops)
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// Control how fragment shading rates are combined.
    FragmentShadingRateCombinerOp = FragmentShadingRateCombinerOpKHR(i32);

    /// Specifies a combiner operation of combine(Axy,Bxy) = Axy.
    Keep = KEEP,

    /// Specifies a combiner operation of combine(Axy,Bxy) = Bxy.
    Replace = REPLACE,

    /// Specifies a combiner operation of combine(Axy,Bxy) = min(Axy,Bxy).
    Min = MIN,

    /// Specifies a combiner operation of combine(Axy,Bxy) = max(Axy,Bxy).
    Max = MAX,

    /// Specifies a combiner operation of combine(Axy,Bxy) = Axy * Bxy.
    ///
    /// See the Vulkan specification for more information on how this operation is performed if `fragmentShadingRateStrictMultiplyCombiner` is `false`.
    Mul = MUL,
}

impl Default for FragmentShadingRateCombinerOp {
    #[inline]
    fn default() -> Self {
        Self::Keep
    }
}
