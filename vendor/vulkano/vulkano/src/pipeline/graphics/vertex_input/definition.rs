use super::{
    VertexBufferDescription, VertexInputAttributeDescription, VertexInputBindingDescription,
    VertexMemberInfo,
};
use crate::{
    pipeline::{
        graphics::vertex_input::VertexInputState,
        inout_interface::{
            input_output_map, InputOutputData, InputOutputKey, InputOutputUserKey,
            InputOutputVariableBlock,
        },
    },
    shader::{
        spirv::{ExecutionModel, Instruction, StorageClass},
        EntryPoint,
    },
    ValidationError,
};
use foldhash::HashMap;
use std::{borrow::Cow, collections::hash_map::Entry};

/// Trait for types that can create a [`VertexInputState`] from an [`EntryPoint`].
pub unsafe trait VertexDefinition {
    /// Builds the `VertexInputState` for the provided `entry_point`.
    fn definition(
        &self,
        entry_point: &EntryPoint,
    ) -> Result<VertexInputState, Box<ValidationError>>;
}

unsafe impl VertexDefinition for &[VertexBufferDescription] {
    #[inline]
    fn definition(
        &self,
        entry_point: &EntryPoint,
    ) -> Result<VertexInputState, Box<ValidationError>> {
        let spirv = entry_point.module().spirv();
        let Some(&Instruction::EntryPoint {
            execution_model,
            ref interface,
            ..
        }) = spirv.function(entry_point.id()).entry_point()
        else {
            unreachable!()
        };

        if execution_model != ExecutionModel::Vertex {
            return Err(Box::new(ValidationError {
                context: "entry_point".into(),
                problem: "is not a vertex shader".into(),
                ..Default::default()
            }));
        }

        let bindings = self
            .iter()
            .enumerate()
            .map(|(binding, buffer_description)| {
                let &VertexBufferDescription {
                    members: _,
                    stride,
                    input_rate,
                } = buffer_description;

                (
                    binding.try_into().unwrap(),
                    VertexInputBindingDescription {
                        stride,
                        input_rate,
                        ..Default::default()
                    },
                )
            })
            .collect();
        let mut attributes: HashMap<u32, VertexInputAttributeDescription> = HashMap::default();

        for variable_id in interface.iter().copied() {
            input_output_map(
                spirv,
                execution_model,
                variable_id,
                StorageClass::Input,
                |key, data| -> Result<(), Box<ValidationError>> {
                    let InputOutputKey::User(key) = key else {
                        return Ok(());
                    };
                    let InputOutputUserKey {
                        mut location,
                        component,
                        index: _,
                    } = key;

                    // TODO: can we make this work somehow?
                    if component != 0 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the shader interface contains an input variable (location {}) \
                                with a non-zero component decoration ({}), which is not yet \
                                supported by `VertexDefinition` in Vulkano",
                                location, component,
                            )
                            .into(),
                            ..Default::default()
                        }));
                    }

                    let InputOutputData {
                        variable_id,
                        pointer_type_id: _,
                        block,
                        type_id: _,
                    } = data;

                    // Find the name of the variable defined in the shader,
                    // or use a default placeholder.
                    let names = if let Some(block) = block {
                        let InputOutputVariableBlock {
                            type_id,
                            member_index,
                        } = block;

                        spirv.id(type_id).members()[member_index].names()
                    } else {
                        spirv.id(variable_id).names()
                    };
                    let name = names
                        .iter()
                        .find_map(|instruction| match *instruction {
                            Instruction::Name { ref name, .. }
                            | Instruction::MemberName { ref name, .. } => {
                                Some(Cow::Borrowed(name.as_str()))
                            }
                            _ => None,
                        })
                        .unwrap_or_else(|| Cow::Owned(format!("vertex_input_{}", location)));

                    // Find a vertex member whose name matches the one in the shader.
                    let (vertex_member_info, binding) = self
                        .iter()
                        .enumerate()
                        .find_map(|(binding, buffer)| {
                            buffer
                                .members
                                .get(name.as_ref())
                                .map(|info| (info, binding.try_into().unwrap()))
                        })
                        .ok_or_else(|| {
                            Box::new(ValidationError {
                                problem: format!(
                                    "the shader interface contains an input variable named \"{}\" \
                                    (location {}, component {}), but no such attribute exists in \
                                    the vertex definition",
                                    name, location, component,
                                )
                                .into(),
                                ..Default::default()
                            })
                        })?;

                    let &VertexMemberInfo {
                        mut offset,
                        format,
                        num_elements,
                        mut stride,
                    } = vertex_member_info;

                    let locations_per_element;

                    if num_elements > 1 {
                        locations_per_element = format.locations();

                        if u64::from(stride) < format.block_size() {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "in the vertex member named \"{}\" in buffer {}, the `stride` is \
                                    less than the block size of `format`",
                                    name, binding,
                                )
                                .into(),
                                ..Default::default()
                            }));
                        }
                    } else {
                        stride = 0;
                        locations_per_element = 0;
                    }

                    // Add an attribute description for every element in the member.
                    for _ in 0..num_elements {
                        match attributes.entry(location) {
                            Entry::Occupied(_) => {
                                return Err(Box::new(ValidationError {
                                    problem: format!(
                                        "the vertex definition specifies a variable at \
                                        location {}, but that location is already occupied by \
                                        another variable",
                                        location,
                                    )
                                    .into(),
                                    ..Default::default()
                                }));
                            }
                            Entry::Vacant(entry) => {
                                entry.insert(VertexInputAttributeDescription {
                                    binding,
                                    format,
                                    offset,
                                    ..Default::default()
                                });
                            }
                        }

                        location = location.checked_add(locations_per_element).unwrap();
                        offset = offset.checked_add(stride).unwrap();
                    }

                    Ok(())
                },
            )?;
        }

        Ok(VertexInputState {
            bindings,
            attributes,
            ..Default::default()
        })
    }
}

unsafe impl<const N: usize> VertexDefinition for [VertexBufferDescription; N] {
    #[inline]
    fn definition(
        &self,
        entry_point: &EntryPoint,
    ) -> Result<VertexInputState, Box<ValidationError>> {
        self.as_slice().definition(entry_point)
    }
}

unsafe impl VertexDefinition for Vec<VertexBufferDescription> {
    #[inline]
    fn definition(
        &self,
        entry_point: &EntryPoint,
    ) -> Result<VertexInputState, Box<ValidationError>> {
        self.as_slice().definition(entry_point)
    }
}

unsafe impl VertexDefinition for VertexBufferDescription {
    #[inline]
    fn definition(
        &self,
        entry_point: &EntryPoint,
    ) -> Result<VertexInputState, Box<ValidationError>> {
        std::slice::from_ref(self).definition(entry_point)
    }
}
