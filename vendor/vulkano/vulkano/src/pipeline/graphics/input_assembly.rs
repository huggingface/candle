//! Configures how input vertices are assembled into primitives.

use crate::{
    device::Device, macros::vulkan_enum, Requires, RequiresAllOf, RequiresOneOf, ValidationError,
};
use ash::vk;

/// The state in a graphics pipeline describing how the input assembly stage should behave.
#[derive(Clone, Debug)]
pub struct InputAssemblyState<'a> {
    /// The type of primitives.
    ///
    /// When [`DynamicState::PrimitiveTopology`] is used, if the
    /// [`dynamic_primitive_topology_unrestricted`] device property is `false`, then
    /// the dynamically set primitive topology must belong to the same *topology class* as
    /// `topology`.
    /// In practice, this is simply the first word in the name of the topology.
    ///
    /// The default value is [`PrimitiveTopology::TriangleList`].
    ///
    /// [`DynamicState::PrimitiveTopology`]: crate::pipeline::DynamicState::PrimitiveTopology
    /// [`dynamic_primitive_topology_unrestricted`]: crate::device::DeviceProperties::dynamic_primitive_topology_unrestricted
    pub topology: PrimitiveTopology,

    /// If true, then when drawing with an index buffer, the special index value consisting of the
    /// maximum unsigned value (`0xff`, `0xffff` or `0xffffffff`) will tell the GPU that it is the
    /// end of the current primitive. A new primitive will restart at the next index.
    ///
    /// Primitive restart is mostly useful in combination with "strip" and "fan" topologies. "List"
    /// topologies require a feature to be enabled on the device when combined with primitive
    /// restart.
    ///
    /// The default value is `false`.
    pub primitive_restart_enable: bool,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for InputAssemblyState<'_> {
    /// Returns [`InputAssemblyState::new()`].
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl InputAssemblyState<'_> {
    /// Returns a default `InputAssemblyState`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            topology: PrimitiveTopology::TriangleList,
            primitive_restart_enable: false,
            _ne: crate::NE,
        }
    }

    /// Sets the primitive topology.
    #[inline]
    #[deprecated(since = "0.34.0")]
    pub fn topology(mut self, topology: PrimitiveTopology) -> Self {
        self.topology = topology;
        self
    }

    /// Enables primitive restart.
    #[inline]
    #[deprecated(since = "0.34.0")]
    pub fn primitive_restart_enable(mut self) -> Self {
        self.primitive_restart_enable = true;
        self
    }

    #[allow(clippy::trivially_copy_pass_by_ref)]
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            topology,
            primitive_restart_enable,
            _ne: _,
        } = self;

        topology.validate_device(device).map_err(|err| {
            err.add_context("topology")
                .set_vuids(&["VUID-VkPipelineInputAssemblyStateCreateInfo-topology-parameter"])
        })?;

        match topology {
            PrimitiveTopology::TriangleFan => {
                if device.enabled_extensions().khr_portability_subset
                    && !device.enabled_features().triangle_fans
                {
                    return Err(Box::new(ValidationError {
                        problem: "this device is a portability subset device, and \
                            `topology` is `PrimitiveTopology::TriangleFan`"
                            .into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                            Requires::DeviceFeature("triangle_fans"),
                        ])]),
                        vuids: &["VUID-VkPipelineInputAssemblyStateCreateInfo-triangleFans-04452"],
                        ..Default::default()
                    }));
                }
            }
            PrimitiveTopology::LineListWithAdjacency
            | PrimitiveTopology::LineStripWithAdjacency
            | PrimitiveTopology::TriangleListWithAdjacency
            | PrimitiveTopology::TriangleStripWithAdjacency => {
                if !device.enabled_features().geometry_shader {
                    return Err(Box::new(ValidationError {
                        context: "topology".into(),
                        problem: "is `PrimitiveTopology::*WithAdjacency`".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                            Requires::DeviceFeature("geometry_shader"),
                        ])]),
                        vuids: &["VUID-VkPipelineInputAssemblyStateCreateInfo-topology-00429"],
                    }));
                }
            }
            PrimitiveTopology::PatchList => {
                if !device.enabled_features().tessellation_shader {
                    return Err(Box::new(ValidationError {
                        context: "topology".into(),
                        problem: "is `PrimitiveTopology::PatchList`".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                            Requires::DeviceFeature("tessellation_shader"),
                        ])]),
                        vuids: &["VUID-VkPipelineInputAssemblyStateCreateInfo-topology-00430"],
                    }));
                }
            }
            _ => (),
        }

        if primitive_restart_enable {
            match topology {
                PrimitiveTopology::PointList
                | PrimitiveTopology::LineList
                | PrimitiveTopology::TriangleList
                | PrimitiveTopology::LineListWithAdjacency
                | PrimitiveTopology::TriangleListWithAdjacency => {
                    if !device.enabled_features().primitive_topology_list_restart {
                        return Err(Box::new(ValidationError {
                            problem: "`topology` is `PrimitiveTopology::*List`, and \
                                `primitive_restart_enable` is `true`"
                                .into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                Requires::DeviceFeature("primitive_topology_list_restart"),
                            ])]),
                            vuids: &["VUID-VkPipelineInputAssemblyStateCreateInfo-topology-06252"],
                            ..Default::default()
                        }));
                    }
                }
                PrimitiveTopology::PatchList => {
                    if !device
                        .enabled_features()
                        .primitive_topology_patch_list_restart
                    {
                        return Err(Box::new(ValidationError {
                            problem: "`topology` is `PrimitiveTopology::PatchList`, and \
                                `primitive_restart_enable` is `true`"
                                .into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                Requires::DeviceFeature("primitive_topology_patch_list_restart"),
                            ])]),
                            vuids: &["VUID-VkPipelineInputAssemblyStateCreateInfo-topology-06253"],
                            ..Default::default()
                        }));
                    }
                }
                _ => (),
            }
        }

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::PipelineInputAssemblyStateCreateInfo<'static> {
        let &Self {
            topology,
            primitive_restart_enable,
            _ne: _,
        } = self;

        vk::PipelineInputAssemblyStateCreateInfo::default()
            .flags(vk::PipelineInputAssemblyStateCreateFlags::empty())
            .topology(topology.into())
            .primitive_restart_enable(primitive_restart_enable)
    }

    pub(crate) fn to_owned(&self) -> InputAssemblyState<'static> {
        InputAssemblyState {
            _ne: crate::NE,
            ..*self
        }
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// Describes how vertices must be grouped together to form primitives.
    ///
    /// When [`DynamicState::PrimitiveTopology`] is used, if the
    /// [`dynamic_primitive_topology_unrestricted`] device property is `false`, then
    /// the dynamically set primitive topology must belong to the same *topology class* as what
    /// was set during pipeline creation.
    /// In practice, this is simply the first word in the name of the topology.
    ///
    /// When enabling primitive restart, "list" topologies require a feature to be enabled on the
    /// device:
    /// - The `PatchList` topology requires the
    ///   [`primitive_topology_patch_list_restart`](crate::device::DeviceFeatures::primitive_topology_patch_list_restart)
    ///   feature.
    /// - All other "list" topologies require the
    ///   [`primitive_topology_list_restart`](crate::device::DeviceFeatures::primitive_topology_list_restart)
    ///   feature.
    ///
    /// [`DynamicState::PrimitiveTopology`]: crate::pipeline::DynamicState::PrimitiveTopology
    /// [`dynamic_primitive_topology_unrestricted`]: crate::device::DeviceProperties::dynamic_primitive_topology_unrestricted
    PrimitiveTopology = PrimitiveTopology(i32);

    /// A series of separate point primitives.
    ///
    /// Topology class: Point
    PointList = POINT_LIST,

    /// A series of separate line primitives.
    ///
    /// Topology class: Line
    LineList = LINE_LIST,

    /// A series of consecutive line primitives, with consecutive lines sharing a vertex.
    ///
    /// Topology class: Line
    LineStrip = LINE_STRIP,

    /// A series of separate triangle primitives.
    ///
    /// Topology class: Triangle
    TriangleList = TRIANGLE_LIST,

    /// A series of consecutive triangle primitives, with consecutive triangles sharing an edge (two vertices).
    ///
    /// Topology class: Triangle
    TriangleStrip = TRIANGLE_STRIP,

    /// A series of consecutive triangle primitives, with all triangles sharing a common vertex (the first).
    ///
    /// On [portability subset](crate::instance#portability-subset-devices-and-the-enumerate_portability-flag)
    /// devices, the [`triangle_fans`](crate::device::DeviceFeatures::triangle_fans)
    /// feature must be enabled on the device.
    ///
    /// Topology class: Triangle
    TriangleFan = TRIANGLE_FAN,

    /// As `LineList, but with adjacency, used in combination with geometry shaders. Requires the
    /// [`geometry_shader`](crate::device::DeviceFeatures::geometry_shader) feature.
    ///
    /// Topology class: Line
    LineListWithAdjacency = LINE_LIST_WITH_ADJACENCY,

    /// As `LineStrip`, but with adjacency, used in combination with geometry shaders. Requires the
    /// [`geometry_shader`](crate::device::DeviceFeatures::geometry_shader) feature.
    ///
    /// Topology class: Line
    LineStripWithAdjacency = LINE_STRIP_WITH_ADJACENCY,

    /// As `TriangleList`, but with adjacency, used in combination with geometry shaders. Requires
    /// the [`geometry_shader`](crate::device::DeviceFeatures::geometry_shader) feature.
    ///
    /// Topology class: Triangle
    TriangleListWithAdjacency = TRIANGLE_LIST_WITH_ADJACENCY,

    /// As `TriangleStrip`, but with adjacency, used in combination with geometry shaders. Requires
    /// the [`geometry_shader`](crate::device::DeviceFeatures::geometry_shader) feature.
    ///
    /// Topology class: Triangle
    TriangleStripWithAdjacency = TRIANGLE_STRIP_WITH_ADJACENCY,

    /// Separate patch primitives, used in combination with tessellation shaders. Requires the
    /// [`tessellation_shader`](crate::device::DeviceFeatures::tessellation_shader) feature.
    ///
    /// Topology class: Patch
    PatchList = PATCH_LIST,

}

// TODO: use the #[default] attribute once it's stable.
// See: https://github.com/rust-lang/rust/issues/87517
impl Default for PrimitiveTopology {
    #[inline]
    fn default() -> Self {
        PrimitiveTopology::TriangleList
    }
}
