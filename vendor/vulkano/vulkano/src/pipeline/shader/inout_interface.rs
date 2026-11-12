use crate::{
    format::NumericType,
    pipeline::graphics::color_blend::ColorComponents,
    shader::{
        reflect::get_constant,
        spirv::{BuiltIn, Decoration, ExecutionModel, Id, Instruction, Spirv, StorageClass},
    },
    ValidationError,
};
use foldhash::HashMap;
use std::{collections::hash_map::Entry, convert::Infallible};

pub(crate) fn validate_interfaces_compatible(
    out_spirv: &Spirv,
    out_execution_model: ExecutionModel,
    out_interface: &[Id],
    in_spirv: &Spirv,
    in_execution_model: ExecutionModel,
    in_interface: &[Id],
    allow_larger_output_vector: bool,
) -> Result<(), Box<ValidationError>> {
    let out_variables_by_key = get_variables_by_key(
        out_spirv,
        out_execution_model,
        out_interface,
        StorageClass::Output,
    );
    let in_variables_by_key = get_variables_by_key(
        in_spirv,
        in_execution_model,
        in_interface,
        StorageClass::Input,
    );

    for ((location, component), in_variable_info) in in_variables_by_key {
        let out_variable_info = out_variables_by_key
            .get(&(location, component))
            .ok_or_else(|| {
                Box::new(ValidationError {
                    problem: format!(
                        "the input interface includes a variable at location {}, component {}, \
                        but the output interface does not contain a variable with the same \
                        location and component",
                        location, component,
                    )
                    .into(),
                    vuids: &["VUID-RuntimeSpirv-OpEntryPoint-08743"],
                    ..Default::default()
                })
            })?;

        if !are_interface_decoration_sets_compatible(
            out_spirv,
            out_variable_info.variable_decorations,
            in_spirv,
            in_variable_info.variable_decorations,
            decoration_filter_variable,
        ) {
            return Err(Box::new(ValidationError {
                problem: format!(
                    "for interface location {}, component {}, \
                    the input variable doesn't have the same decorations as the output variable",
                    location, component,
                )
                .into(),
                vuids: &["VUID-RuntimeSpirv-OpVariable-08746"],
                ..Default::default()
            }));
        }

        if !are_interface_decoration_sets_compatible(
            out_spirv,
            out_variable_info.pointer_type_decorations,
            in_spirv,
            in_variable_info.pointer_type_decorations,
            decoration_filter,
        ) {
            return Err(Box::new(ValidationError {
                problem: format!(
                    "for interface location {}, component {}, \
                    the input variable's pointer type doesn't have the same decorations as \
                    the output variable's pointer type",
                    location, component,
                )
                .into(),
                // vuids?
                ..Default::default()
            }));
        }

        match (
            &out_variable_info.block_type_info,
            &in_variable_info.block_type_info,
        ) {
            (Some(out_block_type_info), Some(in_block_type_info)) => {
                if !are_interface_decoration_sets_compatible(
                    out_spirv,
                    out_block_type_info.block_type_decorations,
                    in_spirv,
                    in_block_type_info.block_type_decorations,
                    decoration_filter,
                ) {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "for interface location {}, component {}, \
                            the input block structure type doesn't have the same decorations \
                            as the output block structure type",
                            location, component,
                        )
                        .into(),
                        vuids: &["VUID-RuntimeSpirv-OpVariable-08746"],
                        ..Default::default()
                    }));
                }

                if !are_interface_decoration_sets_compatible(
                    out_spirv,
                    out_block_type_info.block_member_decorations,
                    in_spirv,
                    in_block_type_info.block_member_decorations,
                    decoration_filter,
                ) {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "for interface location {}, component {}, \
                            the input block structure member doesn't have the same decorations as \
                            the output block structure member",
                            location, component,
                        )
                        .into(),
                        vuids: &["VUID-RuntimeSpirv-OpVariable-08746"],
                        ..Default::default()
                    }));
                }
            }
            (None, None) => (),
            (Some(_), None) | (None, Some(_)) => {
                // TODO: this may be allowed, depending on the outcome of this discussion:
                // https://github.com/KhronosGroup/Vulkan-Docs/issues/2242
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "for interface location {}, component {}, \
                        the input variable doesn't have the same or a compatible type as \
                        the output variable",
                        location, component,
                    )
                    .into(),
                    vuids: &[
                        "VUID-RuntimeSpirv-OpEntryPoint-07754",
                        "VUID-RuntimeSpirv-maintenance4-06817",
                    ],
                    ..Default::default()
                }));
            }
        }

        if !are_interface_types_compatible(
            out_spirv,
            out_variable_info.type_id,
            in_spirv,
            in_variable_info.type_id,
            allow_larger_output_vector,
        ) {
            return Err(Box::new(ValidationError {
                problem: format!(
                    "for interface location {}, component {}, \
                    the input variable doesn't have the same or a compatible type as \
                    the output variable",
                    location, component,
                )
                .into(),
                vuids: &[
                    "VUID-RuntimeSpirv-OpEntryPoint-07754",
                    "VUID-RuntimeSpirv-maintenance4-06817",
                ],
                ..Default::default()
            }));
        }
    }

    Ok(())
}

struct InterfaceVariableInfo<'a> {
    variable_decorations: &'a [Instruction],
    pointer_type_decorations: &'a [Instruction],
    block_type_info: Option<InterfaceVariableBlockInfo<'a>>,
    type_id: Id,
}

struct InterfaceVariableBlockInfo<'a> {
    block_type_decorations: &'a [Instruction],
    block_member_decorations: &'a [Instruction],
}

fn get_variables_by_key<'a>(
    spirv: &'a Spirv,
    execution_model: ExecutionModel,
    interface: &[Id],
    filter_storage_class: StorageClass,
) -> HashMap<(u32, u32), InterfaceVariableInfo<'a>> {
    // Collect all variables into a hashmap indexed by location and component.
    let mut variables_by_key: HashMap<_, _> = HashMap::default();

    for variable_id in interface.iter().copied() {
        input_output_map(
            spirv,
            execution_model,
            variable_id,
            filter_storage_class,
            |key, data| -> Result<(), Infallible> {
                if let InputOutputKey::User(InputOutputUserKey {
                    location,
                    component,
                    ..
                }) = key
                {
                    let InputOutputData {
                        variable_id,
                        pointer_type_id,
                        block,
                        type_id,
                        ..
                    } = data;

                    variables_by_key.insert(
                        (location, component),
                        InterfaceVariableInfo {
                            variable_decorations: spirv.id(variable_id).decorations(),
                            pointer_type_decorations: spirv.id(pointer_type_id).decorations(),
                            block_type_info: block.map(|block| {
                                let block_type_id_info = spirv.id(block.type_id);
                                InterfaceVariableBlockInfo {
                                    block_type_decorations: block_type_id_info.decorations(),
                                    block_member_decorations: block_type_id_info.members()
                                        [block.member_index]
                                        .decorations(),
                                }
                            }),
                            type_id,
                        },
                    );
                }

                Ok(())
            },
        )
        .unwrap();
    }

    variables_by_key
}

fn are_interface_types_compatible(
    out_spirv: &Spirv,
    out_type_id: Id,
    in_spirv: &Spirv,
    in_type_id: Id,
    allow_larger_output_vector: bool,
) -> bool {
    let out_id_info = out_spirv.id(out_type_id);
    let in_id_info = in_spirv.id(in_type_id);

    // Decorations must be compatible.
    if !are_interface_decoration_sets_compatible(
        out_spirv,
        out_id_info.decorations(),
        in_spirv,
        in_id_info.decorations(),
        decoration_filter,
    ) {
        return false;
    }

    // Type definitions must be compatible, potentially recursively.
    // TODO: Add more types. What else can appear in a shader interface?
    match (out_id_info.instruction(), in_id_info.instruction()) {
        (
            &Instruction::TypeInt {
                width: out_width,
                signedness: out_signedness,
                ..
            },
            &Instruction::TypeInt {
                width: in_width,
                signedness: in_signedness,
                ..
            },
        ) => out_width == in_width && out_signedness == in_signedness,
        (
            &Instruction::TypeFloat {
                width: out_width, ..
            },
            &Instruction::TypeFloat {
                width: in_width, ..
            },
        ) => out_width == in_width,
        (
            &Instruction::TypeVector {
                component_type: out_component_type,
                component_count: out_component_count,
                ..
            },
            &Instruction::TypeVector {
                component_type: in_component_type,
                component_count: in_component_count,
                ..
            },
        ) => {
            let is_component_count_compatible = if allow_larger_output_vector {
                out_component_count >= in_component_count
            } else {
                out_component_count == in_component_count
            };

            is_component_count_compatible
                && are_interface_types_compatible(
                    out_spirv,
                    out_component_type,
                    in_spirv,
                    in_component_type,
                    allow_larger_output_vector,
                )
        }
        (
            &Instruction::TypeMatrix {
                column_type: out_column_type,
                column_count: out_column_count,
                ..
            },
            &Instruction::TypeMatrix {
                column_type: in_column_type,
                column_count: in_column_count,
                ..
            },
        ) => {
            out_column_count == in_column_count
                && are_interface_types_compatible(
                    out_spirv,
                    out_column_type,
                    in_spirv,
                    in_column_type,
                    allow_larger_output_vector,
                )
        }
        (
            &Instruction::TypeArray {
                element_type: out_element_type,
                length: out_length,
                ..
            },
            &Instruction::TypeArray {
                element_type: in_element_type,
                length: in_length,
                ..
            },
        ) => {
            let out_length = match *out_spirv.id(out_length).instruction() {
                Instruction::Constant { ref value, .. } => {
                    value.iter().rev().fold(0, |a, &b| (a << 32) | b as u64)
                }
                _ => unreachable!(),
            };
            let in_length = match *in_spirv.id(in_length).instruction() {
                Instruction::Constant { ref value, .. } => {
                    value.iter().rev().fold(0, |a, &b| (a << 32) | b as u64)
                }
                _ => unreachable!(),
            };

            out_length == in_length
                && are_interface_types_compatible(
                    out_spirv,
                    out_element_type,
                    in_spirv,
                    in_element_type,
                    allow_larger_output_vector,
                )
        }
        (
            Instruction::TypeStruct {
                member_types: out_member_types,
                ..
            },
            Instruction::TypeStruct {
                member_types: in_member_types,
                ..
            },
        ) => {
            out_member_types.len() == in_member_types.len()
                && out_member_types
                    .iter()
                    .zip(out_id_info.members())
                    .zip(in_member_types.iter().zip(in_id_info.members()))
                    .all(
                        |(
                            (&out_member_type, out_member_info),
                            (&in_member_type, in_member_info),
                        )| {
                            are_interface_decoration_sets_compatible(
                                out_spirv,
                                out_member_info.decorations(),
                                in_spirv,
                                in_member_info.decorations(),
                                decoration_filter,
                            ) && are_interface_types_compatible(
                                out_spirv,
                                out_member_type,
                                in_spirv,
                                in_member_type,
                                allow_larger_output_vector,
                            )
                        },
                    )
        }
        _ => false,
    }
}

fn are_interface_decoration_sets_compatible(
    out_spirv: &Spirv,
    out_instructions: &[Instruction],
    in_spirv: &Spirv,
    in_instructions: &[Instruction],
    filter: fn(&Instruction) -> bool,
) -> bool {
    if out_instructions.is_empty() && in_instructions.is_empty() {
        return true;
    }

    // This is O(n²), but instructions are not expected to have very many decorations.
    out_instructions
        .iter()
        .filter(|i| filter(i))
        .all(|out_instruction| {
            in_instructions.iter().any(|in_instruction| {
                are_interface_decorations_compatible(
                    out_spirv,
                    out_instruction,
                    in_spirv,
                    in_instruction,
                )
            })
        })
        && in_instructions
            .iter()
            .filter(|i| filter(i))
            .all(|in_instruction| {
                out_instructions.iter().any(|out_instruction| {
                    are_interface_decorations_compatible(
                        out_spirv,
                        out_instruction,
                        in_spirv,
                        in_instruction,
                    )
                })
            })
}

fn are_interface_decorations_compatible(
    out_spirv: &Spirv,
    out_instruction: &Instruction,
    in_spirv: &Spirv,
    in_instruction: &Instruction,
) -> bool {
    match (out_instruction, in_instruction) {
        // Regular decorations are equal if the decorations match.
        (
            Instruction::Decorate {
                decoration: out_decoration,
                ..
            },
            Instruction::Decorate {
                decoration: in_decoration,
                ..
            },
        )
        | (
            Instruction::MemberDecorate {
                decoration: out_decoration,
                ..
            },
            Instruction::MemberDecorate {
                decoration: in_decoration,
                ..
            },
        )
        | (
            Instruction::DecorateString {
                decoration: out_decoration,
                ..
            },
            Instruction::DecorateString {
                decoration: in_decoration,
                ..
            },
        )
        | (
            Instruction::MemberDecorateString {
                decoration: out_decoration,
                ..
            },
            Instruction::MemberDecorateString {
                decoration: in_decoration,
                ..
            },
        ) => out_decoration == in_decoration,

        // DecorateId needs more care, because the Ids must first be resolved before comparing.
        (
            Instruction::DecorateId {
                decoration: out_decoration,
                ..
            },
            Instruction::DecorateId {
                decoration: in_decoration,
                ..
            },
        ) => match (out_decoration, in_decoration) {
            (
                &Decoration::UniformId {
                    execution: out_execution,
                },
                &Decoration::UniformId {
                    execution: in_execution,
                },
            ) => match (
                out_spirv.id(out_execution).instruction(),
                in_spirv.id(in_execution).instruction(),
            ) {
                (
                    Instruction::Constant {
                        value: out_value, ..
                    },
                    Instruction::Constant {
                        value: in_value, ..
                    },
                ) => out_value == in_value,
                _ => unimplemented!("the Ids of `Decoration::UniformId` are not both constants"),
            },
            (&Decoration::AlignmentId { .. }, &Decoration::AlignmentId { .. }) => {
                unreachable!("requires the `Kernel` capability, which Vulkan does not support")
            }
            (
                &Decoration::MaxByteOffsetId {
                    max_byte_offset: out_max_byte_offset,
                },
                &Decoration::MaxByteOffsetId {
                    max_byte_offset: in_max_byte_offset,
                },
            ) => match (
                out_spirv.id(out_max_byte_offset).instruction(),
                in_spirv.id(in_max_byte_offset).instruction(),
            ) {
                (
                    Instruction::Constant {
                        value: out_value, ..
                    },
                    Instruction::Constant {
                        value: in_value, ..
                    },
                ) => out_value == in_value,
                _ => unimplemented!(
                    "the Ids of `Decoration::MaxByteOffsetId` are not both constants"
                ),
            },
            (&Decoration::CounterBuffer { .. }, &Decoration::CounterBuffer { .. }) => {
                unreachable!("can decorate only the `Uniform` storage class")
            }
            (
                &Decoration::AliasScopeINTEL {
                    aliasing_scopes_list: _out_aliasing_scopes_list,
                },
                &Decoration::AliasScopeINTEL {
                    aliasing_scopes_list: _in_aliasing_scopes_list,
                },
            ) => unimplemented!(),
            (
                &Decoration::NoAliasINTEL {
                    aliasing_scopes_list: _out_aliasing_scopes_list,
                },
                &Decoration::NoAliasINTEL {
                    aliasing_scopes_list: _in_aliasing_scopes_list,
                },
            ) => unimplemented!(),
            _ => false,
        },
        _ => false,
    }
}

fn decoration_filter(instruction: &Instruction) -> bool {
    match instruction {
        Instruction::Decorate { decoration, .. }
        | Instruction::MemberDecorate { decoration, .. }
        | Instruction::DecorateString { decoration, .. }
        | Instruction::MemberDecorateString { decoration, .. }
        | Instruction::DecorateId { decoration, .. } => !matches!(
            decoration,
            Decoration::Location { .. }
                | Decoration::XfbBuffer { .. }
                | Decoration::XfbStride { .. }
                | Decoration::Offset { .. }
                | Decoration::Stream { .. }
                | Decoration::Component { .. }
                | Decoration::NoPerspective
                | Decoration::Flat
                | Decoration::Centroid
                | Decoration::Sample
                | Decoration::PerVertexKHR
        ),
        _ => false,
    }
}

fn decoration_filter_variable(instruction: &Instruction) -> bool {
    match instruction {
        Instruction::Decorate { decoration, .. }
        | Instruction::MemberDecorate { decoration, .. }
        | Instruction::DecorateString { decoration, .. }
        | Instruction::MemberDecorateString { decoration, .. }
        | Instruction::DecorateId { decoration, .. } => !matches!(
            decoration,
            Decoration::Location { .. }
                | Decoration::XfbBuffer { .. }
                | Decoration::XfbStride { .. }
                | Decoration::Offset { .. }
                | Decoration::Stream { .. }
                | Decoration::Component { .. }
                | Decoration::NoPerspective
                | Decoration::Flat
                | Decoration::Centroid
                | Decoration::Sample
                | Decoration::PerVertexKHR
                | Decoration::RelaxedPrecision
        ),
        _ => false,
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ShaderInterfaceLocationInfo {
    pub(crate) numeric_type: NumericType,
    pub(crate) width: ShaderInterfaceLocationWidth,
    pub(crate) components: [ColorComponents; 2], // Index 0 and 1
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ShaderInterfaceLocationWidth {
    Bits32,
    Bits64,
}

impl ShaderInterfaceLocationWidth {
    pub(crate) fn component_count(self) -> u32 {
        match self {
            ShaderInterfaceLocationWidth::Bits32 => 1,
            ShaderInterfaceLocationWidth::Bits64 => 2,
        }
    }
}

impl From<u32> for ShaderInterfaceLocationWidth {
    #[inline]
    fn from(value: u32) -> Self {
        if value > 32 {
            Self::Bits64
        } else {
            Self::Bits32
        }
    }
}

pub(crate) fn shader_interface_location_info(
    spirv: &Spirv,
    entry_point_id: Id,
    filter_storage_class: StorageClass,
) -> HashMap<u32, ShaderInterfaceLocationInfo> {
    let (execution_model, interface) = match spirv.function(entry_point_id).entry_point() {
        Some(&Instruction::EntryPoint {
            execution_model,
            ref interface,
            ..
        }) => (execution_model, interface),
        _ => unreachable!(),
    };

    let mut locations: HashMap<u32, ShaderInterfaceLocationInfo> = HashMap::default();
    let mut scalar_func = |key: InputOutputUserKey,
                           width: ShaderInterfaceLocationWidth,
                           numeric_type: NumericType| {
        let InputOutputUserKey {
            location,
            component,
            index,
        } = key;

        let location_info = match locations.entry(location) {
            Entry::Occupied(entry) => {
                let location_info = entry.into_mut();
                debug_assert_eq!(location_info.numeric_type, numeric_type);
                debug_assert_eq!(location_info.width, width);
                location_info
            }
            Entry::Vacant(entry) => entry.insert(ShaderInterfaceLocationInfo {
                numeric_type,
                width,
                components: [ColorComponents::empty(); 2],
            }),
        };
        let components = &mut location_info.components[index as usize];

        let components_to_add = match width {
            ShaderInterfaceLocationWidth::Bits32 => ColorComponents::from_index(component as usize),
            ShaderInterfaceLocationWidth::Bits64 => {
                debug_assert!(component & 1 == 0);
                ColorComponents::from_index(component as usize)
                    | ColorComponents::from_index(component as usize + 1)
            }
        };
        debug_assert!(!components.intersects(components_to_add));
        *components |= components_to_add;
    };

    for &variable_id in interface {
        input_output_map(
            spirv,
            execution_model,
            variable_id,
            filter_storage_class,
            |key, data| -> Result<(), Infallible> {
                if let InputOutputKey::User(key) = key {
                    let InputOutputData { type_id, .. } = data;
                    shader_interface_analyze_type(spirv, type_id, key, &mut scalar_func);
                }

                Ok(())
            },
        )
        .unwrap();
    }

    locations
}

/// Recursively analyzes the type `type_id` with the given `key`. Calls `scalar_func` on every
/// scalar type that is encountered, and returns the number of locations and components to advance.
pub(crate) fn shader_interface_analyze_type(
    spirv: &Spirv,
    type_id: Id,
    mut key: InputOutputUserKey,
    scalar_func: &mut impl FnMut(InputOutputUserKey, ShaderInterfaceLocationWidth, NumericType),
) -> (u32, u32) {
    debug_assert!(key.component < 4);

    match *spirv.id(type_id).instruction() {
        Instruction::TypeInt {
            width, signedness, ..
        } => {
            let numeric_type = if signedness == 1 {
                NumericType::Int
            } else {
                NumericType::Uint
            };

            let width = ShaderInterfaceLocationWidth::from(width);
            scalar_func(key, width, numeric_type);
            (1, width.component_count())
        }
        Instruction::TypeFloat { width, .. } => {
            let width = ShaderInterfaceLocationWidth::from(width);
            scalar_func(key, width, NumericType::Float);
            (1, width.component_count())
        }
        Instruction::TypeVector {
            component_type,
            component_count,
            ..
        } => {
            let mut total_locations_added = 1;

            for _ in 0..component_count {
                // Overflow into next location
                if key.component == 4 {
                    key.component = 0;
                    key.location += 1;
                    total_locations_added += 1;
                } else {
                    debug_assert!(key.component < 4);
                }

                let (_, components_added) =
                    shader_interface_analyze_type(spirv, component_type, key, scalar_func);
                key.component += components_added;
            }

            (total_locations_added, 0)
        }
        Instruction::TypeMatrix {
            column_type,
            column_count,
            ..
        } => {
            let mut total_locations_added = 0;

            for _ in 0..column_count {
                let (locations_added, _) =
                    shader_interface_analyze_type(spirv, column_type, key, scalar_func);
                key.location += locations_added;
                total_locations_added += locations_added;
            }

            (total_locations_added, 0)
        }
        Instruction::TypeArray {
            element_type,
            length,
            ..
        } => {
            let length = get_constant(spirv, length).unwrap();
            let mut total_locations_added = 0;

            for _ in 0..length {
                let (locations_added, _) =
                    shader_interface_analyze_type(spirv, element_type, key, scalar_func);
                key.location += locations_added;
                total_locations_added += locations_added;
            }

            (total_locations_added, 0)
        }
        Instruction::TypeStruct {
            ref member_types, ..
        } => {
            let mut total_locations_added = 0;

            for &member_type in member_types {
                let (locations_added, _) =
                    shader_interface_analyze_type(spirv, member_type, key, scalar_func);
                key.location += locations_added;
                total_locations_added += locations_added;
            }

            (total_locations_added, 0)
        }
        _ => unimplemented!(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum InputOutputKey {
    User(InputOutputUserKey),
    BuiltIn(BuiltIn),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct InputOutputUserKey {
    pub(crate) location: u32,
    pub(crate) component: u32,
    pub(crate) index: u32,
}

pub(crate) struct InputOutputData {
    pub(crate) variable_id: Id,
    pub(crate) pointer_type_id: Id,
    pub(crate) block: Option<InputOutputVariableBlock>,
    pub(crate) type_id: Id,
}

pub(crate) struct InputOutputVariableBlock {
    pub(crate) type_id: Id,
    pub(crate) member_index: usize,
}

pub(crate) fn input_output_map<E>(
    spirv: &Spirv,
    execution_model: ExecutionModel,
    variable_id: Id,
    filter_storage_class: StorageClass,
    mut func: impl FnMut(InputOutputKey, InputOutputData) -> Result<(), E>,
) -> Result<(), E> {
    let variable_id_info = spirv.id(variable_id);
    let (pointer_type_id, storage_class) = match *variable_id_info.instruction() {
        Instruction::Variable {
            result_type_id,
            storage_class,
            ..
        } if storage_class == filter_storage_class => (result_type_id, storage_class),
        _ => return Ok(()),
    };
    let pointer_type_id_info = spirv.id(pointer_type_id);
    let type_id = match *pointer_type_id_info.instruction() {
        Instruction::TypePointer { ty, .. } => {
            strip_array(spirv, storage_class, execution_model, variable_id, ty)
        }
        _ => unreachable!(),
    };

    let mut location = None;
    let mut component = 0;
    let mut index = 0;
    let mut built_in = None;

    for instruction in variable_id_info.decorations() {
        if let Instruction::Decorate { ref decoration, .. } = *instruction {
            match *decoration {
                Decoration::Location { location: l } => location = Some(l),
                Decoration::Component { component: c } => component = c,
                Decoration::Index { index: i } => index = i,
                Decoration::BuiltIn { built_in: b } => built_in = Some(b),
                _ => (),
            }
        }
    }

    if let Some(location) = location {
        func(
            InputOutputKey::User(InputOutputUserKey {
                location,
                component,
                index,
            }),
            InputOutputData {
                variable_id,
                pointer_type_id,
                block: None,
                type_id,
            },
        )
    } else if let Some(built_in) = built_in {
        func(
            InputOutputKey::BuiltIn(built_in),
            InputOutputData {
                variable_id,
                pointer_type_id,
                block: None,
                type_id,
            },
        )
    } else {
        let block_type_id = type_id;
        let block_type_id_info = spirv.id(block_type_id);
        let member_types = match block_type_id_info.instruction() {
            Instruction::TypeStruct { member_types, .. } => member_types,
            _ => return Ok(()),
        };

        for (member_index, (&type_id, member_info)) in member_types
            .iter()
            .zip(block_type_id_info.members())
            .enumerate()
        {
            location = None;
            component = 0;
            index = 0;
            built_in = None;

            for instruction in member_info.decorations() {
                if let Instruction::MemberDecorate { ref decoration, .. } = *instruction {
                    match *decoration {
                        Decoration::Location { location: l } => location = Some(l),
                        Decoration::Component { component: c } => component = c,
                        Decoration::Index { index: i } => index = i,
                        Decoration::BuiltIn { built_in: b } => built_in = Some(b),
                        _ => (),
                    }
                }
            }

            if let Some(location) = location {
                func(
                    InputOutputKey::User(InputOutputUserKey {
                        location,
                        component,
                        index,
                    }),
                    InputOutputData {
                        variable_id,
                        pointer_type_id,
                        block: Some(InputOutputVariableBlock {
                            type_id: block_type_id,
                            member_index,
                        }),
                        type_id,
                    },
                )?;
            } else if let Some(built_in) = built_in {
                func(
                    InputOutputKey::BuiltIn(built_in),
                    InputOutputData {
                        variable_id,
                        pointer_type_id,
                        block: Some(InputOutputVariableBlock {
                            type_id: block_type_id,
                            member_index,
                        }),
                        type_id,
                    },
                )?;
            }
        }

        Ok(())
    }
}

fn strip_array(
    spirv: &Spirv,
    storage_class: StorageClass,
    execution_model: ExecutionModel,
    variable_id: Id,
    pointed_type_id: Id,
) -> Id {
    let variable_decorations = spirv.id(variable_id).decorations();
    let variable_has_decoration = |has_decoration: Decoration| -> bool {
        variable_decorations.iter().any(|instruction| {
            matches!(
                instruction,
                Instruction::Decorate {
                    decoration,
                    ..
                } if *decoration == has_decoration
            )
        })
    };

    let must_strip_array = match storage_class {
        StorageClass::Output => match execution_model {
            ExecutionModel::TaskEXT | ExecutionModel::MeshEXT | ExecutionModel::MeshNV => true,
            ExecutionModel::TessellationControl => !variable_has_decoration(Decoration::Patch),
            ExecutionModel::TaskNV => !variable_has_decoration(Decoration::PerTaskNV),
            _ => false,
        },
        StorageClass::Input => match execution_model {
            ExecutionModel::Geometry | ExecutionModel::MeshEXT => true,
            ExecutionModel::TessellationControl | ExecutionModel::TessellationEvaluation => {
                !variable_has_decoration(Decoration::Patch)
            }
            ExecutionModel::Fragment => variable_has_decoration(Decoration::PerVertexKHR),
            ExecutionModel::MeshNV => !variable_has_decoration(Decoration::PerTaskNV),
            _ => false,
        },
        _ => unreachable!(),
    };

    if must_strip_array {
        match spirv.id(pointed_type_id).instruction() {
            &Instruction::TypeArray { element_type, .. } => element_type,
            _ => pointed_type_id,
        }
    } else {
        pointed_type_id
    }
}
