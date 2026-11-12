use super::{definition::VertexDefinition, VertexBufferDescription};
use crate::{
    pipeline::graphics::vertex_input::{Vertex, VertexInputState},
    shader::EntryPoint,
    ValidationError,
};

/// A vertex definition for any number of vertex and instance buffers.
#[deprecated(
    since = "0.33.0",
    note = "use `VertexBufferDescription` directly instead as returned by `Vertex::per_vertex` or `Vertex::per_instance`"
)]
#[derive(Clone, Debug, Default)]
pub struct BuffersDefinition(Vec<VertexBufferDescription>);

#[allow(deprecated)]
impl BuffersDefinition {
    /// Constructs a new definition.
    #[inline]
    pub fn new() -> Self {
        BuffersDefinition(Vec::new())
    }

    /// Adds a new vertex buffer containing elements of type `V` to the definition.
    pub fn vertex<V: Vertex>(mut self) -> Self {
        self.0.push(V::per_vertex());
        self
    }

    /// Adds a new instance buffer containing elements of type `V` to the definition.
    pub fn instance<V: Vertex>(mut self) -> Self {
        self.0.push(V::per_instance());
        self
    }

    /// Adds a new instance buffer containing elements of type `V` to the definition, with the
    /// specified input rate divisor.
    ///
    /// This requires the [`vertex_attribute_instance_rate_divisor`] feature has been enabled on
    /// the device, unless `divisor` is 1.
    ///
    /// `divisor` can be 0 if the [`vertex_attribute_instance_rate_zero_divisor`] feature is also
    /// enabled. This means that every vertex will use the same vertex and instance data.
    ///
    /// [`vertex_attribute_instance_rate_divisor`]: crate::device::DeviceFeatures::vertex_attribute_instance_rate_divisor
    /// [`vertex_attribute_instance_rate_zero_divisor`]: crate::device::DeviceFeatures::vertex_attribute_instance_rate_zero_divisor
    pub fn instance_with_divisor<V: Vertex>(mut self, divisor: u32) -> Self {
        self.0.push(V::per_instance_with_divisor(divisor));
        self
    }
}

#[allow(deprecated)]
unsafe impl VertexDefinition for BuffersDefinition {
    #[inline]
    fn definition(
        &self,
        entry_point: &EntryPoint,
    ) -> Result<VertexInputState, Box<ValidationError>> {
        self.0.definition(entry_point)
    }
}
