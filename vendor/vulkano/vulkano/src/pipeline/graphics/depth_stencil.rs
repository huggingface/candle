//! Configures the operation of the depth, stencil and depth bounds tests.
//!
//! The depth test passes of fails depending on how the depth value of each fragment compares
//! to the existing depth value in the depth buffer at that fragment's location. Depth values
//! are always between 0.0 and 1.0.
//!
//! The depth bounds test allows you to ask the GPU to exclude fragments that are outside of a
//! certain range. This is done in addition to the regular depth test.
//!
//! The stencil test passes or fails depending on how a reference value compares to the existing
//! value in the stencil buffer at each fragment's location. Depending on the outcome of the
//! depth and stencil tests, the value of the stencil buffer at that location can be updated.

use crate::{
    device::Device,
    macros::{vulkan_bitflags, vulkan_enum},
    Requires, RequiresAllOf, RequiresOneOf, ValidationError,
};
use ash::vk;

/// The state in a graphics pipeline describing how the depth, depth bounds and stencil tests
/// should behave.
#[derive(Clone, Debug)]
pub struct DepthStencilState<'a> {
    /// Additional properties of the depth/stencil state.
    ///
    /// The default value is empty.
    pub flags: DepthStencilStateFlags,

    /// The state of the depth test.
    ///
    /// If set to `None`, the depth test is disabled, all fragments will pass and no depth writes
    /// are performed.
    ///
    /// The default value is `None`.
    pub depth: Option<DepthState>,

    /// The minimum and maximum depth values to use for the depth bounds test. Fragments with
    /// values outside this range are discarded.
    ///
    /// If set to `None`, the depth bounds test is disabled, all fragments will pass.
    ///
    /// The default value is `None`.
    pub depth_bounds: Option<(f32, f32)>,

    /// The state of the stencil test.
    ///
    /// If set to `None`, the stencil test is disabled, all fragments will pass and no stencil
    /// writes are performed.
    ///
    /// The default value is `None`.
    pub stencil: Option<StencilState>,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for DepthStencilState<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl DepthStencilState<'_> {
    /// Returns a default `DepthStencilState`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            flags: DepthStencilStateFlags::empty(),
            depth: None,
            depth_bounds: None,
            stencil: None,
            _ne: crate::NE,
        }
    }

    /// Creates a `DepthStencilState` where all tests are disabled and have no effect.
    #[inline]
    #[deprecated(since = "0.34.0", note = "use `DepthStencilState::new` instead")]
    pub fn disabled() -> Self {
        Self::default()
    }

    /// Creates a `DepthStencilState` with a `Less` depth test, `depth_write` set to true, and
    /// other tests disabled.
    #[inline]
    #[deprecated(since = "0.34.0", note = "use `DepthState::simple` instead")]
    pub fn simple_depth_test() -> Self {
        Self {
            flags: DepthStencilStateFlags::empty(),
            depth: Some(DepthState::simple()),
            depth_bounds: Default::default(),
            stencil: Default::default(),
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            ref depth,
            ref depth_bounds,
            ref stencil,
            _ne: _,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkPipelineDepthStencilStateCreateInfo-flags-parameter"])
        })?;

        if let Some(depth_state) = depth {
            depth_state
                .validate(device)
                .map_err(|err| err.add_context("depth"))?;
        }

        if let Some(depth_bounds) = depth_bounds {
            if !device.enabled_features().depth_bounds {
                return Err(Box::new(ValidationError {
                    context: "depth_bounds".into(),
                    problem: "is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "depth_bounds",
                    )])]),
                    vuids: &[
                        "VUID-VkPipelineDepthStencilStateCreateInfo-depthBoundsTestEnable-00598",
                    ],
                }));
            }

            if !device.enabled_extensions().ext_depth_range_unrestricted {
                if !(0.0..=1.0).contains(&depth_bounds.0) {
                    return Err(Box::new(ValidationError {
                        context: "depth_bounds.start".into(),
                        problem: "is not between 0.0 and 1.0 inclusive".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                            Requires::DeviceExtension("ext_depth_range_unrestricted"),
                        ])]),
                        vuids: &["VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-02510"],
                    }));
                }

                if !(0.0..=1.0).contains(&depth_bounds.1) {
                    return Err(Box::new(ValidationError {
                        context: "depth_bounds.end".into(),
                        problem: "is not between 0.0 and 1.0 inclusive".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                            Requires::DeviceExtension("ext_depth_range_unrestricted"),
                        ])]),
                        vuids: &["VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-02510"],
                    }));
                }
            }
        }

        if let Some(stencil_state) = stencil {
            stencil_state
                .validate(device)
                .map_err(|err| err.add_context("stencil"))?;
        }

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::PipelineDepthStencilStateCreateInfo<'static> {
        let &Self {
            flags,
            ref depth,
            ref depth_bounds,
            ref stencil,
            _ne: _,
        } = self;

        let (depth_test_enable_vk, depth_write_enable_vk, depth_compare_op_vk) =
            if let Some(depth_state) = depth {
                let &DepthState {
                    write_enable,
                    compare_op,
                } = depth_state;

                (true, write_enable, compare_op.into())
            } else {
                (false, false, vk::CompareOp::ALWAYS)
            };

        let (depth_bounds_test_enable_vk, min_depth_bounds_vk, max_depth_bounds_vk) =
            if let Some(depth_bounds) = depth_bounds {
                (true, depth_bounds.0, depth_bounds.1)
            } else {
                (false, 0.0, 1.0)
            };

        let (stencil_test_enable_vk, front_vk, back_vk) = if let Some(stencil_state) = stencil {
            let StencilState { front, back } = stencil_state;

            let [front, back] = [front, back].map(|stencil_op_state| stencil_op_state.to_vk());

            (true, front, back)
        } else {
            (false, Default::default(), Default::default())
        };

        vk::PipelineDepthStencilStateCreateInfo::default()
            .flags(flags.into())
            .depth_test_enable(depth_test_enable_vk)
            .depth_write_enable(depth_write_enable_vk)
            .depth_compare_op(depth_compare_op_vk)
            .depth_bounds_test_enable(depth_bounds_test_enable_vk)
            .stencil_test_enable(stencil_test_enable_vk)
            .front(front_vk)
            .back(back_vk)
            .min_depth_bounds(min_depth_bounds_vk)
            .max_depth_bounds(max_depth_bounds_vk)
    }

    pub(crate) fn to_owned(&self) -> DepthStencilState<'static> {
        DepthStencilState {
            _ne: crate::NE,
            ..*self
        }
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags specifying additional properties of the depth/stencil state.
    DepthStencilStateFlags = PipelineDepthStencilStateCreateFlags(u32);

    /* TODO: enable
    // TODO: document
    RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS = RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_rasterization_order_attachment_access)]),
        RequiresAllOf([DeviceExtension(arm_rasterization_order_attachment_access)]),
    ]), */

    /* TODO: enable
    // TODO: document
    RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS = RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_rasterization_order_attachment_access)]),
        RequiresAllOf([DeviceExtension(arm_rasterization_order_attachment_access)]),
    ]), */
}

/// The state in a graphics pipeline describing how the depth test should behave when enabled.
#[derive(Clone, Copy, Debug)]
pub struct DepthState {
    /// Sets whether the value in the depth buffer will be updated when the depth test succeeds.
    ///
    /// The default value is `false`.
    pub write_enable: bool,

    /// Comparison operation to use between the depth value of each incoming fragment and the depth
    /// value currently in the depth buffer.
    ///
    /// The default value is [`CompareOp::Always`].
    pub compare_op: CompareOp,
}

impl Default for DepthState {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl DepthState {
    /// Returns a default `DepthState`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            write_enable: false,
            compare_op: CompareOp::Always,
        }
    }

    /// Returns a `DepthState` with a `Less` depth test and depth writes enabled.
    #[inline]
    pub fn simple() -> Self {
        Self {
            compare_op: CompareOp::Less,
            write_enable: true,
        }
    }

    /// Returns a `DepthState` with a `Greater` depth test and depth writes enabled.
    #[inline]
    pub fn reverse() -> Self {
        Self {
            compare_op: CompareOp::Greater,
            write_enable: true,
        }
    }

    pub(crate) fn validate(self, device: &Device) -> Result<(), Box<ValidationError>> {
        let Self {
            write_enable: _,
            compare_op,
        } = self;

        compare_op.validate_device(device).map_err(|err| {
            err.add_context("compare_op")
                .set_vuids(&["VUID-VkPipelineDepthStencilStateCreateInfo-depthCompareOp-parameter"])
        })?;

        Ok(())
    }
}

/// The state in a graphics pipeline describing how the stencil test should behave when enabled.
#[derive(Clone, Copy, Debug)]
pub struct StencilState {
    /// The stencil operation state to use for points and lines, and for triangles whose front is
    /// facing the user.
    ///
    /// The default value is `StencilOpState::default()`.
    pub front: StencilOpState,

    /// The stencil operation state to use for triangles whose back is facing the user.
    ///
    /// The default value is `StencilOpState::default()`.
    pub back: StencilOpState,
}

impl Default for StencilState {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl StencilState {
    /// Returns a default `StencilState`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            front: StencilOpState::new(),
            back: StencilOpState::new(),
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let Self { front, back } = self;

        front
            .ops
            .validate(device)
            .map_err(|err| err.add_context("front.ops"))?;

        back.ops
            .validate(device)
            .map_err(|err| err.add_context("back.ops"))?;

        Ok(())
    }
}

/// Stencil test operations for a single face.
#[derive(Clone, Copy, Debug)]
pub struct StencilOpState {
    /// The stencil operations to perform.
    ///
    /// The default value is `StencilOps::default()`.
    pub ops: StencilOps,

    /// A bitmask that selects the bits of the unsigned integer stencil values participating in the
    /// stencil test. Ignored if `compare_op` is `Never` or `Always`.
    ///
    /// The default value is [`u32::MAX`].
    pub compare_mask: u32,

    /// A bitmask that selects the bits of the unsigned integer stencil values updated by the
    /// stencil test in the stencil framebuffer attachment. Ignored if the relevant operation is
    /// `Keep`.
    ///
    /// The default value is [`u32::MAX`].
    pub write_mask: u32,

    /// Reference value that is used in the unsigned stencil comparison. The stencil test is
    /// considered to pass if the `compare_op` between the stencil buffer value and this reference
    /// value yields true.
    ///
    /// On [portability
    /// subset](crate::instance#portability-subset-devices-and-the-enumerate_portability-flag)
    /// devices, if culling is disabled, and the `reference` values of the front and back face
    /// are not equal, then the
    /// [`separate_stencil_mask_ref`](crate::device::DeviceFeatures::separate_stencil_mask_ref)
    /// feature must be enabled on the device.
    ///
    /// The default value is [`u32::MAX`].
    pub reference: u32,
}

impl Default for StencilOpState {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl StencilOpState {
    /// Returns a default `StencilOpState`.
    #[inline]
    pub const fn new() -> Self {
        StencilOpState {
            ops: StencilOps::new(),
            compare_mask: u32::MAX,
            write_mask: u32::MAX,
            reference: u32::MAX,
        }
    }

    #[allow(clippy::wrong_self_convention)]
    pub(crate) fn to_vk(&self) -> vk::StencilOpState {
        let &Self {
            ops,
            compare_mask,
            write_mask,
            reference,
        } = self;

        vk::StencilOpState {
            fail_op: ops.fail_op.into(),
            pass_op: ops.pass_op.into(),
            depth_fail_op: ops.depth_fail_op.into(),
            compare_op: ops.compare_op.into(),
            compare_mask,
            write_mask,
            reference,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StencilOps {
    /// The operation to perform when the stencil test failed.
    ///
    /// The default value is [`StencilOp::Keep`].
    pub fail_op: StencilOp,

    /// The operation to perform when both the depth test and the stencil test passed.
    ///
    /// The default value is [`StencilOp::Keep`].
    pub pass_op: StencilOp,

    /// The operation to perform when the stencil test passed but the depth test failed.
    ///
    /// The default value is [`StencilOp::Keep`].
    pub depth_fail_op: StencilOp,

    /// The comparison to perform between the existing stencil value in the stencil buffer, and
    /// the reference value (given by `reference`).
    ///
    /// The default value is [`CompareOp::Never`].
    pub compare_op: CompareOp,
}

impl Default for StencilOps {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl StencilOps {
    /// Returns a default `StencilOps`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            pass_op: StencilOp::Keep,
            fail_op: StencilOp::Keep,
            depth_fail_op: StencilOp::Keep,
            compare_op: CompareOp::Never,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            fail_op,
            pass_op,
            depth_fail_op,
            compare_op,
        } = self;

        fail_op.validate_device(device).map_err(|err| {
            err.add_context("fail_op")
                .set_vuids(&["VUID-VkStencilOpState-failOp-parameter"])
        })?;

        pass_op.validate_device(device).map_err(|err| {
            err.add_context("pass_op")
                .set_vuids(&["VUID-VkStencilOpState-passOp-parameter"])
        })?;

        depth_fail_op.validate_device(device).map_err(|err| {
            err.add_context("depth_fail_op")
                .set_vuids(&["VUID-VkStencilOpState-depthFailOp-parameter"])
        })?;

        compare_op.validate_device(device).map_err(|err| {
            err.add_context("compare_op")
                .set_vuids(&["VUID-VkStencilOpState-compareOp-parameter"])
        })?;

        Ok(())
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// Operation to perform after the depth and stencil tests.
    StencilOp = StencilOp(i32);

    // TODO: document
    Keep = KEEP,

    // TODO: document
    Zero = ZERO,

    // TODO: document
    Replace = REPLACE,

    // TODO: document
    IncrementAndClamp = INCREMENT_AND_CLAMP,

    // TODO: document
    DecrementAndClamp = DECREMENT_AND_CLAMP,

    // TODO: document
    Invert = INVERT,

    // TODO: document
    IncrementAndWrap = INCREMENT_AND_WRAP,

    // TODO: document
    DecrementAndWrap = DECREMENT_AND_WRAP,
}

vulkan_enum! {
    #[non_exhaustive]

    /// Specifies a face for stencil operations.
    StencilFaces = StencilFaceFlags(u32);

    // TODO: document
    Front = FRONT,

    // TODO: document
    Back = BACK,

    // TODO: document
    FrontAndBack = FRONT_AND_BACK,
}

/// Specifies a dynamic state value for the front and back faces.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DynamicStencilValue {
    pub front: u32,
    pub back: u32,
}

vulkan_enum! {
    #[non_exhaustive]

    /// Specifies how two values should be compared to decide whether a test passes or fails.
    ///
    /// Used for both depth testing and stencil testing.
    CompareOp = CompareOp(i32);

    /// The test never passes.
    Never = NEVER,

    /// The test passes if `value < reference_value`.
    Less = LESS,

    /// The test passes if `value == reference_value`.
    Equal = EQUAL,

    /// The test passes if `value <= reference_value`.
    LessOrEqual = LESS_OR_EQUAL,

    /// The test passes if `value > reference_value`.
    Greater = GREATER,

    /// The test passes if `value != reference_value`.
    NotEqual = NOT_EQUAL,

    /// The test passes if `value >= reference_value`.
    GreaterOrEqual = GREATER_OR_EQUAL,

    /// The test always passes.
    Always = ALWAYS,
}
