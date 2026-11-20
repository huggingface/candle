//! Subdivides primitives into smaller primitives.

use crate::{
    device::Device, macros::vulkan_enum, Requires, RequiresAllOf, RequiresOneOf, ValidationError,
    Version,
};
use ash::vk;
use ash::vk::TaggedStructure;

/// The state in a graphics pipeline describing the tessellation shader execution of a graphics
/// pipeline.
#[derive(Clone, Debug)]
pub struct TessellationState<'a> {
    /// The number of patch control points to use.
    ///
    /// The default value is 3.
    pub patch_control_points: u32,

    /// The origin to use for the tessellation domain.
    ///
    /// If this is not [`TessellationDomainOrigin::UpperLeft`], the device API version must be at
    /// least 1.1, or the [`khr_maintenance2`](crate::device::DeviceExtensions::khr_maintenance2)
    /// extension must be enabled on the device.
    ///
    /// The default value is [`TessellationDomainOrigin::UpperLeft`].
    pub domain_origin: TessellationDomainOrigin,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for TessellationState<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> TessellationState<'a> {
    /// Returns a default `TessellationState`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            patch_control_points: 3,
            domain_origin: TessellationDomainOrigin::UpperLeft,
            _ne: crate::NE,
        }
    }

    /// Sets the number of patch control points.
    #[inline]
    #[deprecated(since = "0.34.0")]
    pub fn patch_control_points(mut self, num: u32) -> Self {
        self.patch_control_points = num;
        self
    }

    #[allow(clippy::trivially_copy_pass_by_ref)]
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            patch_control_points,
            domain_origin,
            _ne: _,
        } = self;

        let properties = device.physical_device().properties();

        if patch_control_points == 0 {
            return Err(Box::new(ValidationError {
                context: "patch_control_points".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkPipelineTessellationStateCreateInfo-patchControlPoints-01214"],
                ..Default::default()
            }));
        }

        if patch_control_points > properties.max_tessellation_patch_size {
            return Err(Box::new(ValidationError {
                context: "patch_control_points".into(),
                problem: "exceeds the `max_tessellation_patch_size` limit".into(),
                vuids: &["VUID-VkPipelineTessellationStateCreateInfo-patchControlPoints-01214"],
                ..Default::default()
            }));
        }

        domain_origin.validate_device(device).map_err(|err| {
            err.add_context("domain_origin").set_vuids(&[
                "VUID-VkPipelineTessellationDomainOriginStateCreateInfo-domainOrigin-parameter",
            ])
        })?;

        if domain_origin != TessellationDomainOrigin::UpperLeft
            && !(device.api_version() >= Version::V1_1
                || device.enabled_extensions().khr_maintenance2)
        {
            return Err(Box::new(ValidationError {
                context: "domain_origin".into(),
                problem: "is not `TessellationDomainOrigin::UpperLeft`".into(),
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_2)]),
                    RequiresAllOf(&[Requires::DeviceExtension("khr_maintenance2")]),
                ]),
                ..Default::default()
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk(
        &self,
        extensions_vk: &'a mut TessellationStateExtensionsVk,
    ) -> vk::PipelineTessellationStateCreateInfo<'a> {
        let &Self {
            patch_control_points,
            domain_origin: _,
            _ne: _,
        } = self;

        let mut val_vk = vk::PipelineTessellationStateCreateInfo::default()
            .flags(vk::PipelineTessellationStateCreateFlags::empty())
            .patch_control_points(patch_control_points);

        let TessellationStateExtensionsVk { domain_origin_vk } = extensions_vk;

        if let Some(next) = domain_origin_vk {
            val_vk = val_vk.push(next);
        }

        val_vk
    }

    pub(crate) fn to_vk_extensions(&self) -> TessellationStateExtensionsVk {
        let &Self {
            patch_control_points: _,
            domain_origin,
            _ne: _,
        } = self;

        let domain_origin_vk = (domain_origin != TessellationDomainOrigin::default()).then(|| {
            vk::PipelineTessellationDomainOriginStateCreateInfo::default()
                .domain_origin(domain_origin.into())
        });

        TessellationStateExtensionsVk { domain_origin_vk }
    }

    pub(crate) fn to_owned(&self) -> TessellationState<'static> {
        TessellationState {
            _ne: crate::NE,
            ..*self
        }
    }
}

pub(crate) struct TessellationStateExtensionsVk {
    pub(crate) domain_origin_vk:
        Option<vk::PipelineTessellationDomainOriginStateCreateInfo<'static>>,
}

vulkan_enum! {
    #[non_exhaustive]

    /// The origin of the tessellation domain.
    TessellationDomainOrigin = TessellationDomainOrigin(i32);

    /// The origin is in the upper left corner.
    ///
    /// This is the default.
    UpperLeft = UPPER_LEFT,

    /// The origin is in the lower left corner.
    LowerLeft = LOWER_LEFT,
}

impl Default for TessellationDomainOrigin {
    #[inline]
    fn default() -> Self {
        Self::UpperLeft
    }
}
