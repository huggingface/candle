//! A test to discard pixels that would be written to certain areas of a framebuffer.
//!
//! The discard rectangle test is similar to, but separate from the scissor test.

use crate::{
    device::Device, macros::vulkan_enum, pipeline::graphics::viewport::Scissor,
    self_referential::self_referential, ValidationError,
};
use ash::vk;
use smallvec::SmallVec;

/// The state in a graphics pipeline describing how the discard rectangle test should behave.
#[derive(Clone, Debug)]
pub struct DiscardRectangleState<'a> {
    /// Sets whether the discard rectangle test operates inclusively or exclusively.
    ///
    /// The default value is [`DiscardRectangleMode::Exclusive`].
    pub mode: DiscardRectangleMode,

    /// Specifies the discard rectangles.
    ///
    /// When [`DynamicState::DiscardRectangle`] is used, the values of each rectangle are ignored
    /// and must be set dynamically, but the number of discard rectangles is fixed and
    /// must be matched when setting the dynamic value.
    ///
    /// If this not not empty, then the
    /// [`ext_discard_rectangles`](crate::device::DeviceExtensions::ext_discard_rectangles)
    /// extension must be enabled on the device.
    ///
    /// The default value is empty.
    ///
    /// [`DynamicState::DiscardRectangle`]: crate::pipeline::DynamicState::DiscardRectangle
    pub rectangles: &'a [Scissor],

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for DiscardRectangleState<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> DiscardRectangleState<'a> {
    /// Returns a default `DiscardRectangleState`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            mode: DiscardRectangleMode::Exclusive,
            rectangles: &[],
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            mode,
            rectangles,
            _ne: _,
        } = self;

        let properties = device.physical_device().properties();

        mode.validate_device(device).map_err(|err| {
            err.add_context("mode").set_vuids(&[
                "VUID-VkPipelineDiscardRectangleStateCreateInfoEXT-discardRectangleMode-parameter",
            ])
        })?;

        if rectangles.len() as u32 > properties.max_discard_rectangles.unwrap() {
            return Err(Box::new(ValidationError {
                context: "rectangles".into(),
                problem: "the length exceeds the `max_discard_rectangles` limit".into(),
                vuids: &[
                    "VUID-VkPipelineDiscardRectangleStateCreateInfoEXT-discardRectangleCount-00582",
                ],
                ..Default::default()
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk(
        &self,
        fields1_vk: &'a DiscardRectangleStateFields1Vk,
    ) -> vk::PipelineDiscardRectangleStateCreateInfoEXT<'a> {
        let &Self {
            mode,
            rectangles: _,
            _ne: _,
        } = self;
        let DiscardRectangleStateFields1Vk {
            discard_rectangles_vk,
        } = fields1_vk;

        vk::PipelineDiscardRectangleStateCreateInfoEXT::default()
            .flags(vk::PipelineDiscardRectangleStateCreateFlagsEXT::empty())
            .discard_rectangle_mode(mode.into())
            .discard_rectangles(discard_rectangles_vk)
    }

    pub(crate) fn to_vk_fields1(&self) -> DiscardRectangleStateFields1Vk {
        let &Self {
            mode: _,
            rectangles,
            _ne: _,
        } = self;

        let discard_rectangles_vk = rectangles.iter().map(|rect| rect.to_vk()).collect();

        DiscardRectangleStateFields1Vk {
            discard_rectangles_vk,
        }
    }

    pub(crate) fn to_owned(&self) -> OwnedDiscardRectangleState {
        let rectangles = self.rectangles.to_owned();

        OwnedDiscardRectangleState::new(rectangles, |rectangles| DiscardRectangleState {
            rectangles,
            _ne: crate::NE,
            ..*self
        })
    }
}

pub(crate) struct DiscardRectangleStateFields1Vk {
    pub(crate) discard_rectangles_vk: SmallVec<[vk::Rect2D; 2]>,
}

self_referential! {
    mod owned_discard_rectangle_state {
        pub(crate) struct OwnedDiscardRectangleState {
            inner: DiscardRectangleState<'_>,
            rectangles: Vec<Scissor>,
        }
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// The mode in which the discard rectangle test operates.
    DiscardRectangleMode = DiscardRectangleModeEXT(i32);

    /// Samples that are inside a rectangle are kept, samples that are outside all rectangles
    /// are discarded.
    Inclusive = INCLUSIVE,

    /// Samples that are inside a rectangle are discarded, samples that are outside all rectangles
    /// are kept.
    Exclusive = EXCLUSIVE,
}
