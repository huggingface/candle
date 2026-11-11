use crate::shader::{
    spirv::{Decoration, Id, IdInfo, Instruction, SpecConstantInstruction},
    SpecializationConstant,
};
use foldhash::HashMap;
use half::f16;
use smallvec::{smallvec, SmallVec};

/// Go through all the specialization constant instructions,
/// and updates their values and replaces them with regular constants.
pub(super) fn replace_specialization_instructions(
    specialization_info: &[(u32, SpecializationConstant)],
    instructions_global: impl IntoIterator<Item = Instruction>,
    ids: &HashMap<Id, IdInfo>,
    mut next_new_id: u32,
) -> Vec<Instruction> {
    let get_specialization = |id: Id| -> Option<SpecializationConstant> {
        ids[&id]
            .decorations
            .iter()
            .find_map(|instruction| match instruction {
                &Instruction::Decorate {
                    decoration:
                        Decoration::SpecId {
                            specialization_constant_id,
                        },
                    ..
                } => specialization_info
                    .iter()
                    .find_map(|&(id, value)| (id == specialization_constant_id).then_some(value)),
                _ => None,
            })
    };

    // This stores the constants we've seen so far. Since composite and op constants must
    // use constants that were defined earlier, this works.
    let mut constants: HashMap<Id, Constant> = HashMap::default();

    instructions_global
        .into_iter()
        .flat_map(|instruction| -> SmallVec<[Instruction; 1]> {
            let new_instructions: SmallVec<[Instruction; 1]> = match instruction {
                Instruction::SpecConstantFalse {
                    result_type_id,
                    result_id,
                }
                | Instruction::SpecConstantTrue {
                    result_type_id,
                    result_id,
                } => {
                    let value = get_specialization(result_id).map_or_else(
                        || matches!(instruction, Instruction::SpecConstantTrue { .. }),
                        |sc| matches!(sc, SpecializationConstant::Bool(true)),
                    );
                    let new_instruction = if value {
                        Instruction::ConstantTrue {
                            result_type_id,
                            result_id,
                        }
                    } else {
                        Instruction::ConstantFalse {
                            result_type_id,
                            result_id,
                        }
                    };

                    smallvec![new_instruction]
                }
                Instruction::SpecConstant {
                    result_type_id,
                    result_id,
                    ref value,
                } => {
                    if let Some(specialization) = get_specialization(result_id) {
                        smallvec![Instruction::Constant {
                            result_type_id,
                            result_id,
                            value: match specialization {
                                SpecializationConstant::Bool(_) => unreachable!(),
                                SpecializationConstant::U8(num) => vec![num as u32],
                                SpecializationConstant::U16(num) => vec![num as u32],
                                SpecializationConstant::U32(num) => vec![num],
                                SpecializationConstant::U64(num) =>
                                    vec![num as u32, (num >> 32) as u32],
                                SpecializationConstant::I8(num) => vec![num as u32],
                                SpecializationConstant::I16(num) => vec![num as u32],
                                SpecializationConstant::I32(num) => vec![num as u32],
                                SpecializationConstant::I64(num) =>
                                    vec![num as u32, (num >> 32) as u32],
                                SpecializationConstant::F16(num) => vec![num.to_bits() as u32],
                                SpecializationConstant::F32(num) => vec![num.to_bits()],
                                SpecializationConstant::F64(num) => {
                                    let num = num.to_bits();
                                    vec![num as u32, (num >> 32) as u32]
                                }
                            },
                        }]
                    } else {
                        smallvec![Instruction::Constant {
                            result_type_id,
                            result_id,
                            value: value.clone(),
                        }]
                    }
                }
                Instruction::SpecConstantComposite {
                    result_type_id,
                    result_id,
                    ref constituents,
                } => {
                    smallvec![Instruction::ConstantComposite {
                        result_type_id,
                        result_id,
                        constituents: constituents.clone(),
                    }]
                }
                Instruction::SpecConstantOp {
                    result_type_id,
                    result_id,
                    ref opcode,
                } => evaluate_spec_constant_op(
                    &mut next_new_id,
                    ids,
                    &constants,
                    result_type_id,
                    result_id,
                    opcode,
                ),
                _ => smallvec![instruction],
            };

            for instruction in &new_instructions {
                match *instruction {
                    Instruction::ConstantFalse {
                        result_type_id,
                        result_id,
                        ..
                    } => {
                        constants.insert(
                            result_id,
                            Constant {
                                type_id: result_type_id,
                                value: ConstantValue::Scalar(0),
                            },
                        );
                    }
                    Instruction::ConstantTrue {
                        result_type_id,
                        result_id,
                        ..
                    } => {
                        constants.insert(
                            result_id,
                            Constant {
                                type_id: result_type_id,
                                value: ConstantValue::Scalar(1),
                            },
                        );
                    }
                    Instruction::Constant {
                        result_type_id,
                        result_id,
                        ref value,
                    } => {
                        let constant_value = match *ids[&result_type_id].instruction() {
                            Instruction::TypeInt {
                                width, signedness, ..
                            } => {
                                if width == 64 {
                                    assert_eq!(value.len(), 2);
                                } else {
                                    assert_eq!(value.len(), 1);
                                }

                                match (signedness, width) {
                                    (0, 8) => value[0] as u64,
                                    (0, 16) => value[0] as u64,
                                    (0, 32) => value[0] as u64,
                                    (0, 64) => (value[0] as u64) | ((value[1] as u64) << 32),
                                    (1, 8) => value[0] as i32 as u64,
                                    (1, 16) => value[0] as i32 as u64,
                                    (1, 32) => value[0] as i32 as u64,
                                    (1, 64) => {
                                        ((value[0] as i64) | ((value[1] as i64) << 32)) as u64
                                    }
                                    _ => unimplemented!(),
                                }
                            }
                            Instruction::TypeFloat { width, .. } => {
                                if width == 64 {
                                    assert_eq!(value.len(), 2);
                                } else {
                                    assert_eq!(value.len(), 1);
                                }

                                match width {
                                    16 => f16::from_bits(value[0] as u16).to_f64() as u64,
                                    32 => f32::from_bits(value[0]) as f64 as u64,
                                    64 => f64::from_bits(
                                        (value[0] as u64) | ((value[1] as u64) << 32),
                                    ) as u64,
                                    _ => unimplemented!(),
                                }
                            }
                            _ => unreachable!(),
                        };

                        constants.insert(
                            result_id,
                            Constant {
                                type_id: result_type_id,
                                value: ConstantValue::Scalar(constant_value),
                            },
                        );
                    }
                    Instruction::ConstantComposite {
                        result_type_id,
                        result_id,
                        ref constituents,
                    } => {
                        constants.insert(
                            result_id,
                            Constant {
                                type_id: result_type_id,
                                value: ConstantValue::Composite(constituents.as_slice().into()),
                            },
                        );
                    }
                    _ => (),
                }
            }

            new_instructions
        })
        .collect()
}

struct Constant {
    type_id: Id,
    value: ConstantValue,
}

#[derive(Clone)]
enum ConstantValue {
    // All scalar constants are stored as u64, regardless of their original type.
    // They are converted from and to their actual representation
    // when they are first read or when they are written back.
    // Signed integers are sign extended to i64 first, floats are cast to f64 and then
    // bit-converted.
    Scalar(u64),

    Composite(SmallVec<[Id; 4]>),
}

impl ConstantValue {
    fn as_scalar(&self) -> u64 {
        match self {
            Self::Scalar(val) => *val,
            Self::Composite(_) => panic!("called `as_scalar` on a composite value"),
        }
    }

    fn as_composite(&self) -> &[Id] {
        match self {
            Self::Scalar(_) => panic!("called `as_composite` on a scalar value"),
            Self::Composite(val) => val,
        }
    }
}

fn numeric_constant_to_words(
    constant_type: &Instruction,
    constant_value: u64,
) -> SmallVec<[u32; 2]> {
    match *constant_type {
        Instruction::TypeInt {
            width, signedness, ..
        } => match (signedness, width) {
            (0, 8) => smallvec![constant_value as u8 as u32],
            (0, 16) => smallvec![constant_value as u16 as u32],
            (0, 32) => smallvec![constant_value as u32],
            (0, 64) => smallvec![constant_value as u32, (constant_value >> 32) as u32],
            (1, 8) => smallvec![constant_value as i8 as u32],
            (1, 16) => smallvec![constant_value as i16 as u32],
            (1, 32) => smallvec![constant_value as u32],
            (1, 64) => smallvec![constant_value as u32, (constant_value >> 32) as u32],
            _ => unimplemented!(),
        },
        Instruction::TypeFloat { width, .. } => match width {
            16 => smallvec![f16::from_f64(f64::from_bits(constant_value)).to_bits() as u32],
            32 => smallvec![(f64::from_bits(constant_value) as f32).to_bits()],
            64 => smallvec![constant_value as u32, (constant_value >> 32) as u32],
            _ => unimplemented!(),
        },
        _ => unreachable!(),
    }
}

// Evaluate a SpecConstantInstruction.
fn evaluate_spec_constant_op(
    next_new_id: &mut u32,
    ids: &HashMap<Id, IdInfo>,
    constants: &HashMap<Id, Constant>,
    result_type_id: Id,
    result_id: Id,
    opcode: &SpecConstantInstruction,
) -> SmallVec<[Instruction; 1]> {
    let scalar_constant_to_instruction =
        |constant_type_id: Id, constant_id: Id, constant_value: u64| -> Instruction {
            match *ids[&constant_type_id].instruction() {
                Instruction::TypeBool { .. } => {
                    if constant_value != 0 {
                        Instruction::ConstantTrue {
                            result_type_id: constant_type_id,
                            result_id: constant_id,
                        }
                    } else {
                        Instruction::ConstantFalse {
                            result_type_id: constant_type_id,
                            result_id: constant_id,
                        }
                    }
                }
                ref result_type @ (Instruction::TypeInt { .. } | Instruction::TypeFloat { .. }) => {
                    Instruction::Constant {
                        result_type_id: constant_type_id,
                        result_id: constant_id,
                        value: numeric_constant_to_words(result_type, constant_value).to_vec(),
                    }
                }
                _ => unreachable!(),
            }
        };

    let constant_to_instruction = |constant_id: Id| -> SmallVec<[Instruction; 1]> {
        let constant = &constants[&constant_id];
        if !matches!(
            opcode,
            SpecConstantInstruction::UConvert { .. }
                | SpecConstantInstruction::SConvert { .. }
                | SpecConstantInstruction::FConvert { .. }
        ) {
            debug_assert_eq!(constant.type_id, result_type_id);
        }
        match constant.value {
            ConstantValue::Scalar(value) => smallvec![scalar_constant_to_instruction(
                result_type_id,
                result_id,
                value
            )],
            ConstantValue::Composite(ref constituents) => {
                smallvec![Instruction::ConstantComposite {
                    result_type_id,
                    result_id,
                    constituents: constituents.to_vec(),
                }]
            }
        }
    };

    match *opcode {
        SpecConstantInstruction::VectorShuffle {
            vector_1,
            vector_2,
            ref components,
        } => {
            let vector_1 = constants[&vector_1].value.as_composite();
            let vector_2 = constants[&vector_2].value.as_composite();
            let concatenated: SmallVec<[Id; 8]> =
                vector_1.iter().chain(vector_2.iter()).copied().collect();
            let constituents: SmallVec<[Id; 4]> = components
                .iter()
                .map(|&component| {
                    concatenated[if component == 0xFFFFFFFF {
                        0 // Spec says the value is undefined, so we can pick anything.
                    } else {
                        component as usize
                    }]
                })
                .collect();

            smallvec![Instruction::ConstantComposite {
                result_type_id,
                result_id,
                constituents: constituents.to_vec(),
            }]
        }
        SpecConstantInstruction::CompositeExtract {
            composite,
            ref indexes,
        } => {
            // Go through the index chain to find the Id to extract.
            let id = indexes.iter().fold(composite, |current_id, &index| {
                constants[&current_id].value.as_composite()[index as usize]
            });

            constant_to_instruction(id)
        }
        SpecConstantInstruction::CompositeInsert {
            object,
            composite,
            ref indexes,
        } => {
            let new_id_count = indexes.len() as u32 - 1;
            let new_ids = (0..new_id_count).map(|i| Id(*next_new_id + i));

            // Go down the type tree, starting from the top-level type `composite`.
            let mut old_constituent_id = composite;

            let new_result_ids = std::iter::once(result_id).chain(new_ids.clone());
            let new_constituent_ids = new_ids.chain(std::iter::once(object));

            let mut new_instructions: SmallVec<_> = indexes
                .iter()
                .copied()
                .zip(new_result_ids.zip(new_constituent_ids))
                .map(|(index, (new_result_id, new_constituent_id))| {
                    let constant = &constants[&old_constituent_id];

                    // Get the Id of the original constituent value to iterate further,
                    // then replace it with the new Id.
                    let mut constituents = constant.value.as_composite().to_vec();
                    old_constituent_id = constituents[index as usize];
                    constituents[index as usize] = new_constituent_id;

                    Instruction::ConstantComposite {
                        result_type_id: constant.type_id,
                        result_id: new_result_id,
                        constituents,
                    }
                })
                .collect();

            *next_new_id += new_id_count;
            new_instructions.reverse(); // so that new constants are defined before use
            new_instructions
        }
        SpecConstantInstruction::Select {
            condition,
            object_1,
            object_2,
        } => match constants[&condition].value {
            ConstantValue::Scalar(condition) => {
                let result = if condition != 0 { object_1 } else { object_2 };

                constant_to_instruction(result)
            }
            ConstantValue::Composite(ref conditions) => {
                let object_1 = constants[&object_1].value.as_composite();
                let object_2 = constants[&object_2].value.as_composite();

                assert_eq!(conditions.len(), object_1.len());
                assert_eq!(conditions.len(), object_2.len());

                let constituents: SmallVec<[Id; 4]> = conditions
                    .iter()
                    .map(|c| constants[c].value.as_scalar())
                    .zip(object_1.iter().zip(object_2.iter()))
                    .map(
                        |(condition, (&object_1, &object_2))| {
                            if condition != 0 {
                                object_1
                            } else {
                                object_2
                            }
                        },
                    )
                    .collect();

                smallvec![Instruction::ConstantComposite {
                    result_type_id,
                    result_id,
                    constituents: constituents.to_vec(),
                }]
            }
        },
        SpecConstantInstruction::UConvert {
            unsigned_value: value,
        }
        | SpecConstantInstruction::SConvert {
            signed_value: value,
        }
        | SpecConstantInstruction::FConvert { float_value: value } => {
            constant_to_instruction(value)
        }
        _ => {
            let result = evaluate_spec_constant_calculation_op(opcode, constants);

            if let &[result] = result.as_slice() {
                smallvec![scalar_constant_to_instruction(
                    result_type_id,
                    result_id,
                    result,
                )]
            } else {
                let component_type_id = match *ids[&result_type_id].instruction() {
                    Instruction::TypeVector { component_type, .. } => component_type,
                    _ => unreachable!(),
                };

                // We have to create new constants with new ids,
                // to hold each component of the result.
                // In theory, we could go digging among the existing constants to see if any
                // of them already fit...
                let new_id_count = result.len() as u32;
                let new_instructions = result
                    .into_iter()
                    .enumerate()
                    .map(|(i, result)| {
                        scalar_constant_to_instruction(
                            component_type_id,
                            Id(*next_new_id + i as u32),
                            result,
                        )
                    })
                    .chain(std::iter::once(Instruction::ConstantComposite {
                        result_type_id,
                        result_id,
                        constituents: (0..new_id_count).map(|i| Id(*next_new_id + i)).collect(),
                    }))
                    .collect();
                *next_new_id += new_id_count;
                new_instructions
            }
        }
    }
}

// Evaluate a SpecConstantInstruction that does calculations on scalars or paired vector
// components.
fn evaluate_spec_constant_calculation_op(
    instruction: &SpecConstantInstruction,
    constants: &HashMap<Id, Constant>,
) -> SmallVec<[u64; 4]> {
    let unary_op = |operand: Id, op: fn(u64) -> u64| -> SmallVec<[u64; 4]> {
        match constants[&operand].value {
            ConstantValue::Scalar(operand) => smallvec![op(operand)],
            ConstantValue::Composite(ref constituents) => constituents
                .iter()
                .map(|constituent| {
                    let operand = constants[constituent].value.as_scalar();
                    op(operand)
                })
                .collect(),
        }
    };

    let binary_op = |operand1: Id, operand2: Id, op: fn(u64, u64) -> u64| -> SmallVec<[u64; 4]> {
        match (&constants[&operand1].value, &constants[&operand2].value) {
            (&ConstantValue::Scalar(operand1), &ConstantValue::Scalar(operand2)) => {
                smallvec![op(operand1, operand2)]
            }
            (ConstantValue::Composite(constituents1), ConstantValue::Composite(constituents2)) => {
                assert_eq!(constituents1.len(), constituents2.len());
                constituents1
                    .iter()
                    .zip(constituents2.iter())
                    .map(|(constituent1, constituent2)| {
                        let operand1 = constants[constituent1].value.as_scalar();
                        let operand2 = constants[constituent2].value.as_scalar();
                        op(operand1, operand2)
                    })
                    .collect()
            }
            _ => unreachable!(),
        }
    };

    match *instruction {
        SpecConstantInstruction::VectorShuffle { .. }
        | SpecConstantInstruction::CompositeExtract { .. }
        | SpecConstantInstruction::CompositeInsert { .. }
        | SpecConstantInstruction::Select { .. }
        | SpecConstantInstruction::UConvert { .. }
        | SpecConstantInstruction::SConvert { .. }
        | SpecConstantInstruction::FConvert { .. } => unreachable!(),
        SpecConstantInstruction::SNegate { operand } => {
            unary_op(operand, |operand| operand.wrapping_neg())
        }
        SpecConstantInstruction::IAdd { operand1, operand2 } => {
            binary_op(operand1, operand2, |operand1, operand2| {
                operand1.wrapping_add(operand2)
            })
        }
        SpecConstantInstruction::ISub { operand1, operand2 } => {
            binary_op(operand1, operand2, |operand1, operand2| {
                operand1.wrapping_sub(operand2)
            })
        }
        SpecConstantInstruction::IMul { operand1, operand2 } => {
            binary_op(operand1, operand2, |operand1, operand2| {
                operand1.wrapping_mul(operand2)
            })
        }
        SpecConstantInstruction::UDiv { operand1, operand2 } => {
            binary_op(operand1, operand2, |operand1, operand2| {
                operand1.wrapping_div(operand2)
            })
        }
        SpecConstantInstruction::UMod { operand1, operand2 } => {
            binary_op(operand1, operand2, |operand1, operand2| {
                operand1.wrapping_rem(operand2)
            })
        }
        SpecConstantInstruction::SDiv { operand1, operand2 } => {
            binary_op(operand1, operand2, |operand1, operand2| {
                let operand1 = operand1 as i64;
                let operand2 = operand2 as i64;
                operand1.wrapping_div(operand2) as u64
            })
        }
        SpecConstantInstruction::SRem { operand1, operand2 } => {
            binary_op(operand1, operand2, |operand1, operand2| {
                let operand1 = operand1 as i64;
                let operand2 = operand2 as i64;
                operand1.wrapping_rem(operand2) as u64
            })
        }
        SpecConstantInstruction::SMod { operand1, operand2 } => {
            binary_op(operand1, operand2, |operand1, operand2| {
                let operand1 = operand1 as i64;
                let operand2 = operand2 as i64;
                ((operand1.wrapping_rem(operand2) + operand2) % operand2) as u64
            })
        }
        SpecConstantInstruction::LogicalEqual { operand1, operand2 } => {
            binary_op(operand1, operand2, |operand1, operand2| {
                ((operand1 != 0) == (operand2 != 0)) as u64
            })
        }
        SpecConstantInstruction::LogicalNotEqual { operand1, operand2 } => {
            binary_op(operand1, operand2, |operand1, operand2| {
                ((operand1 != 0) != (operand2 != 0)) as u64
            })
        }
        SpecConstantInstruction::LogicalOr { operand1, operand2 } => {
            binary_op(operand1, operand2, |operand1, operand2| {
                (operand1 != 0 || operand2 != 0) as u64
            })
        }
        SpecConstantInstruction::LogicalAnd { operand1, operand2 } => {
            binary_op(operand1, operand2, |operand1, operand2| {
                (operand1 != 0 && operand2 != 0) as u64
            })
        }
        SpecConstantInstruction::LogicalNot { operand } => {
            unary_op(operand, |operand| (operand == 0) as u64)
        }
        SpecConstantInstruction::IEqual { operand1, operand2 } => {
            binary_op(operand1, operand2, |operand1, operand2| {
                (operand1 == operand2) as u64
            })
        }
        SpecConstantInstruction::INotEqual { operand1, operand2 } => {
            binary_op(operand1, operand2, |operand1, operand2| {
                (operand1 != operand2) as u64
            })
        }
        SpecConstantInstruction::UGreaterThan { operand1, operand2 } => {
            binary_op(operand1, operand2, |operand1, operand2| {
                (operand1 > operand2) as u64
            })
        }
        SpecConstantInstruction::SGreaterThan { operand1, operand2 } => {
            binary_op(operand1, operand2, |operand1, operand2| {
                let operand1 = operand1 as i64;
                let operand2 = operand2 as i64;
                (operand1 > operand2) as u64
            })
        }
        SpecConstantInstruction::UGreaterThanEqual { operand1, operand2 } => {
            binary_op(operand1, operand2, |operand1, operand2| {
                (operand1 >= operand2) as u64
            })
        }
        SpecConstantInstruction::SGreaterThanEqual { operand1, operand2 } => {
            binary_op(operand1, operand2, |operand1, operand2| {
                let operand1 = operand1 as i64;
                let operand2 = operand2 as i64;
                (operand1 >= operand2) as u64
            })
        }
        SpecConstantInstruction::ULessThan { operand1, operand2 } => {
            binary_op(operand1, operand2, |operand1, operand2| {
                (operand1 < operand2) as u64
            })
        }
        SpecConstantInstruction::SLessThan { operand1, operand2 } => {
            binary_op(operand1, operand2, |operand1, operand2| {
                let operand1 = operand1 as i64;
                let operand2 = operand2 as i64;
                (operand1 < operand2) as u64
            })
        }
        SpecConstantInstruction::ULessThanEqual { operand1, operand2 } => {
            binary_op(operand1, operand2, |operand1, operand2| {
                (operand1 <= operand2) as u64
            })
        }
        SpecConstantInstruction::SLessThanEqual { operand1, operand2 } => {
            binary_op(operand1, operand2, |operand1, operand2| {
                let operand1 = operand1 as i64;
                let operand2 = operand2 as i64;
                (operand1 <= operand2) as u64
            })
        }
        SpecConstantInstruction::ShiftRightLogical { base, shift } => {
            binary_op(base, shift, |base, shift| base >> shift)
        }
        SpecConstantInstruction::ShiftRightArithmetic { base, shift } => {
            binary_op(base, shift, |base, shift| {
                let base = base as i64;
                (base >> shift) as u64
            })
        }
        SpecConstantInstruction::ShiftLeftLogical { base, shift } => {
            binary_op(base, shift, |base, shift| base << shift)
        }
        SpecConstantInstruction::BitwiseOr { operand1, operand2 } => {
            binary_op(operand1, operand2, |operand1, operand2| operand1 | operand2)
        }
        SpecConstantInstruction::BitwiseXor { operand1, operand2 } => {
            binary_op(operand1, operand2, |operand1, operand2| operand1 ^ operand2)
        }
        SpecConstantInstruction::BitwiseAnd { operand1, operand2 } => {
            binary_op(operand1, operand2, |operand1, operand2| operand1 & operand2)
        }
        SpecConstantInstruction::Not { operand } => unary_op(operand, |operand| !operand),
        SpecConstantInstruction::QuantizeToF16 { value } => unary_op(value, |value| {
            let value = f64::from_bits(value);
            f16::from_f64(value).to_f64().to_bits()
        }),
    }
}
