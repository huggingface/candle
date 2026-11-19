//! Generates multiple fragments per framebuffer pixel when rasterizing. This can be used for
//! anti-aliasing.

use crate::{
    device::Device, image::SampleCount, Requires, RequiresAllOf, RequiresOneOf, ValidationError,
};
use ash::vk;

// TODO: handle some weird behaviors with non-floating-point targets

/// State of the multisampling.
#[derive(Clone, Debug)]
pub struct MultisampleState<'a> {
    /// The number of rasterization samples to take per pixel. The GPU will pick this many
    /// different locations within each pixel and assign to each of these locations a different
    /// depth value. The depth and stencil test will then be run for each sample.
    ///
    /// The default value is [`SampleCount::Sample1`].
    pub rasterization_samples: SampleCount,

    /// Controls the proportion (between 0.0 and 1.0) of the samples that will be run through the
    /// fragment shader.
    ///
    /// If the value is 1.0, then all sub-pixel samples will run
    /// through the shader and get a different value. If the value is 0.5, about half of the
    /// samples will run through the shader and the other half will get their values from the
    /// ones which went through the shader.
    ///
    /// If set to `Some`, the
    /// [`sample_rate_shading`](crate::device::DeviceFeatures::sample_rate_shading)
    /// feature must be enabled on the device.
    ///
    /// The default value is `None`.
    pub sample_shading: Option<f32>,

    /// A mask of bits that is ANDed with the coverage mask of each set of `rasterization_samples`
    /// samples. Only the first `rasterization_samples / 32` bits are used, the rest is ignored.
    ///
    /// The default value is `[u32::MAX; 2]`.
    pub sample_mask: [u32; 2], // 64 bits for needed for 64 SampleCount

    /// Controls whether the alpha value of the fragment will be used in an implementation-defined
    /// way to determine which samples get disabled or not. For example if the alpha value is 0.5,
    /// then about half of the samples will be discarded. If you render to a multisample image,
    /// this means that the color will end up being mixed with whatever color was underneath,
    /// which gives the same effect as alpha blending.
    ///
    /// The default value is `false`.
    pub alpha_to_coverage_enable: bool,

    /// Controls whether the alpha value of all the samples will be forced to 1.0 (or the
    /// maximum possible value) after the effects of `alpha_to_coverage` have been applied.
    ///
    /// If set to `true`, the [`alpha_to_one`](crate::device::DeviceFeatures::alpha_to_one)
    /// feature must be enabled on the device.
    ///
    /// The default value is `false`.
    pub alpha_to_one_enable: bool,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for MultisampleState<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl MultisampleState<'_> {
    /// Returns a default `MultisampleState`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            rasterization_samples: SampleCount::Sample1,
            sample_shading: None,
            sample_mask: [u32::MAX; 2],
            alpha_to_coverage_enable: false,
            alpha_to_one_enable: false,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            rasterization_samples,
            sample_shading,
            sample_mask: _,
            alpha_to_coverage_enable: _,
            alpha_to_one_enable,
            _ne: _,
        } = self;

        rasterization_samples
            .validate_device(device)
            .map_err(|err| {
                err.add_context("rasterization_samples").set_vuids(&[
                    "VUID-VkPipelineMultisampleStateCreateInfo-rasterizationSamples-parameter",
                ])
            })?;

        if let Some(min_sample_shading) = sample_shading {
            if !device.enabled_features().sample_rate_shading {
                return Err(Box::new(ValidationError {
                    context: "min_sample_shading".into(),
                    problem: "is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "sample_rate_shading",
                    )])]),
                    vuids: &["VUID-VkPipelineMultisampleStateCreateInfo-sampleShadingEnable-00784"],
                }));
            }

            if !(0.0..=1.0).contains(&min_sample_shading) {
                return Err(Box::new(ValidationError {
                    context: "min_sample_shading".into(),
                    problem: "is not between 0.0 and 1.0 inclusive".into(),
                    vuids: &["VUID-VkPipelineMultisampleStateCreateInfo-minSampleShading-00786"],
                    ..Default::default()
                }));
            }
        }

        if alpha_to_one_enable && !device.enabled_features().alpha_to_one {
            return Err(Box::new(ValidationError {
                context: "alpha_to_one_enable".into(),
                problem: "is `true`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "alpha_to_one",
                )])]),
                vuids: &["VUID-VkPipelineMultisampleStateCreateInfo-alphaToOneEnable-00785"],
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::PipelineMultisampleStateCreateInfo<'_> {
        let &Self {
            rasterization_samples,
            sample_shading,
            ref sample_mask,
            alpha_to_coverage_enable,
            alpha_to_one_enable,
            _ne: _,
        } = self;

        let (sample_shading_enable_vk, min_sample_shading_vk) =
            if let Some(min_sample_shading) = sample_shading {
                (true, min_sample_shading)
            } else {
                (false, 0.0)
            };

        vk::PipelineMultisampleStateCreateInfo::default()
            .flags(vk::PipelineMultisampleStateCreateFlags::empty())
            .rasterization_samples(rasterization_samples.into())
            .sample_shading_enable(sample_shading_enable_vk)
            .min_sample_shading(min_sample_shading_vk)
            .sample_mask(sample_mask)
            .alpha_to_coverage_enable(alpha_to_coverage_enable)
            .alpha_to_one_enable(alpha_to_one_enable)
    }

    pub(crate) fn to_owned(&self) -> MultisampleState<'static> {
        MultisampleState {
            _ne: crate::NE,
            ..*self
        }
    }
}
