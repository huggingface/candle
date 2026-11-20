use super::{write_file, SpirvGrammar};
use foldhash::{HashMap, HashSet};
use heck::ToSnakeCase;
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};
use std::{borrow::Cow, sync::LazyLock};

// From the documentation of the OpSpecConstantOp instruction.
// The instructions requiring the Kernel capability are not listed,
// as this capability is not supported by Vulkan.
static SPEC_CONSTANT_OP: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    HashSet::from_iter([
        "SConvert",
        "UConvert",
        "FConvert",
        "SNegate",
        "Not",
        "IAdd",
        "ISub",
        "IMul",
        "UDiv",
        "SDiv",
        "UMod",
        "SRem",
        "SMod",
        "ShiftRightLogical",
        "ShiftRightArithmetic",
        "ShiftLeftLogical",
        "BitwiseOr",
        "BitwiseXor",
        "BitwiseAnd",
        "VectorShuffle",
        "CompositeExtract",
        "CompositeInsert",
        "LogicalOr",
        "LogicalAnd",
        "LogicalNot",
        "LogicalEqual",
        "LogicalNotEqual",
        "Select",
        "IEqual",
        "INotEqual",
        "ULessThan",
        "SLessThan",
        "UGreaterThan",
        "SGreaterThan",
        "ULessThanEqual",
        "SLessThanEqual",
        "UGreaterThanEqual",
        "SGreaterThanEqual",
        "QuantizeToF16",
    ])
});

pub fn write(grammar: &SpirvGrammar) {
    let mut instr_members = instruction_members(grammar);
    let instr_output = instruction_output(&instr_members, false);

    instr_members.retain(|member| SPEC_CONSTANT_OP.contains(member.name.to_string().as_str()));
    instr_members.iter_mut().for_each(|member| {
        if member.has_result_type_id {
            member.operands.remove(0);
        }
        if member.has_result_id {
            member.operands.remove(0);
        }
    });
    let spec_constant_instr_output = instruction_output(&instr_members, true);

    let bit_enum_output = bit_enum_output(&bit_enum_members(grammar));
    let value_enum_output = value_enum_output(&value_enum_members(grammar));

    write_file(
        "spirv_parse.rs",
        format!(
            "SPIR-V grammar version {}.{}.{}",
            grammar.major_version, grammar.minor_version, grammar.revision
        ),
        quote! {
            #instr_output
            #spec_constant_instr_output
            #bit_enum_output
            #value_enum_output
        },
    );
}

#[derive(Clone, Debug)]
struct InstructionMember {
    name: Ident,
    is_atomic_operation: bool,
    is_cooperative_matrix: bool,
    is_cooperative_matrix_nv: bool,
    is_group_operation: bool,
    is_quad_group_operation: bool,
    is_image_gather: bool,
    is_image_fetch: bool,
    is_image_sample: bool,
    has_result_id: bool,
    has_result_type_id: bool,
    has_execution_scope_id: bool,
    has_memory_scope_id: bool,
    has_image_operands: Option<bool>,
    opcode: u16,
    operands: Vec<OperandMember>,
}

#[derive(Clone, Debug)]
struct OperandMember {
    name: Ident,
    ty: TokenStream,
    parse: TokenStream,
}

fn instruction_output(members: &[InstructionMember], spec_constant: bool) -> TokenStream {
    let struct_items = members
        .iter()
        .map(|InstructionMember { name, operands, .. }| {
            if operands.is_empty() {
                quote! { #name, }
            } else {
                let operands = operands.iter().map(|OperandMember { name, ty, .. }| {
                    quote! { #name: #ty, }
                });
                quote! {
                    #name {
                        #(#operands)*
                    },
                }
            }
        });
    let parse_items = members.iter().map(
        |InstructionMember {
             name,
             opcode,
             operands,
             ..
         }| {
            if operands.is_empty() {
                quote! {
                    #opcode => Ok(Self::#name),
                }
            } else {
                let operands_items =
                    operands.iter().map(|OperandMember { name, parse, .. }| {
                        quote! {
                            #name: #parse,
                        }
                    });

                // We wrap the parsing code for each match arm in its own function such that we
                // don't have to rely on LLVM to prove that the different stack allocations can
                // safely be overlapped. These optimizations are not enabled in debug mode, and
                // that previously lead to gigantic stack consumption in debug mode. The closure
                // syntax is a nice and easy way to define a function; it is not a closure in the
                // sense that it doesn't capture anything and lowers to a regular function.
                quote! {
                    #opcode => (|reader: &mut InstructionReader<'_>| Ok(Self::#name {
                        #(#operands_items)*
                    }))(reader),
                }
            }
        },
    );

    let doc = if spec_constant {
        "An instruction that is used as the operand of the `SpecConstantOp` instruction."
    } else {
        "A parsed SPIR-V instruction."
    };

    let enum_name = if spec_constant {
        format_ident!("SpecConstantInstruction")
    } else {
        format_ident!("Instruction")
    };

    let result_fns = if spec_constant {
        quote! {}
    } else {
        let result_id_items = members.iter().filter_map(
            |InstructionMember {
                 name,
                 has_result_id,
                 ..
             }| {
                if *has_result_id {
                    Some(quote! { Self::#name { result_id, .. } })
                } else {
                    None
                }
            },
        );
        let result_type_id_items = members.iter().filter_map(
            |InstructionMember {
                 name,
                 has_result_type_id,
                 ..
             }| {
                if *has_result_type_id {
                    Some(quote! { Self::#name { result_type_id, .. } })
                } else {
                    None
                }
            },
        );
        let is_cooperative_matrix_items = members.iter().filter_map(
            |InstructionMember {
                 name,
                 is_cooperative_matrix,
                 ..
             }| {
                if *is_cooperative_matrix {
                    Some(quote! { Self::#name { .. } })
                } else {
                    None
                }
            },
        );
        let is_cooperative_matrix_nv_items = members.iter().filter_map(
            |InstructionMember {
                 name,
                 is_cooperative_matrix_nv,
                 ..
             }| {
                if *is_cooperative_matrix_nv {
                    Some(quote! { Self::#name { .. } })
                } else {
                    None
                }
            },
        );
        let is_group_operation_items = members.iter().filter_map(
            |InstructionMember {
                 name,
                 is_group_operation,
                 ..
             }| {
                if *is_group_operation {
                    Some(quote! { Self::#name { .. } })
                } else {
                    None
                }
            },
        );
        let is_quad_group_operation_items = members.iter().filter_map(
            |InstructionMember {
                 name,
                 is_quad_group_operation,
                 ..
             }| {
                if *is_quad_group_operation {
                    Some(quote! { Self::#name { .. } })
                } else {
                    None
                }
            },
        );
        let is_image_fetch_items = members.iter().filter_map(
            |InstructionMember {
                 name,
                 is_image_fetch,
                 ..
             }| {
                if *is_image_fetch {
                    Some(quote! { Self::#name { .. } })
                } else {
                    None
                }
            },
        );
        let is_image_gather_items = members.iter().filter_map(
            |InstructionMember {
                 name,
                 is_image_gather,
                 ..
             }| {
                if *is_image_gather {
                    Some(quote! { Self::#name { .. } })
                } else {
                    None
                }
            },
        );
        let is_image_sample_items = members.iter().filter_map(
            |InstructionMember {
                 name,
                 is_image_sample,
                 ..
             }| {
                if *is_image_sample {
                    Some(quote! { Self::#name { .. } })
                } else {
                    None
                }
            },
        );
        let atomic_pointer_id_items = members.iter().filter_map(
            |InstructionMember {
                 name,
                 is_atomic_operation,
                 ..
             }| {
                if *is_atomic_operation {
                    Some(quote! { Self::#name { pointer, .. } })
                } else {
                    None
                }
            },
        );
        let execution_scope_id_items = members.iter().filter_map(
            |InstructionMember {
                 name,
                 has_execution_scope_id,
                 ..
             }| {
                if *has_execution_scope_id {
                    Some(quote! { Self::#name { execution, .. } })
                } else {
                    None
                }
            },
        );
        let memory_scope_id_items = members.iter().filter_map(
            |InstructionMember {
                 name,
                 has_memory_scope_id,
                 ..
             }| {
                if *has_memory_scope_id {
                    Some(quote! { Self::#name { memory, .. } })
                } else {
                    None
                }
            },
        );
        let image_operands_items = members.iter().filter_map(
            |InstructionMember {
                 name,
                 has_image_operands,
                 ..
             }| {
                if let Some(has_image_operands) = *has_image_operands {
                    if has_image_operands {
                        Some(quote! { Self::#name { image_operands: Some(image_operands), .. } })
                    } else {
                        Some(quote! { Self::#name { image_operands, .. } })
                    }
                } else {
                    None
                }
            },
        );

        quote! {
            /// Returns the `Id` that is assigned by this instruction, if any.
            pub fn result_id(&self) -> Option<Id> {
                match self {
                    #(#result_id_items)|* => Some(*result_id),
                    _ => None
                }
            }

            /// Returns the `Id` of the type of `result_id`, if any.
            pub fn result_type_id(&self) -> Option<Id> {
                match self {
                    #(#result_type_id_items)|* => Some(*result_type_id),
                    _ => None
                }
            }

            /// Returns the `Id` of the pointer in an atomic operation, if any.
            pub fn atomic_pointer_id(&self) -> Option<Id> {
                match self {
                    #(#atomic_pointer_id_items)|* => Some(*pointer),
                    _ => None
                }
            }

            /// Returns whether the instruction is a cooperative matrix instruction.
            pub fn is_cooperative_matrix(&self) -> bool {
                matches!(
                    self,
                    #(#is_cooperative_matrix_items)|*
                )
            }

            /// Returns whether the instruction is an NV cooperative matrix instruction.
            pub fn is_cooperative_matrix_nv(&self) -> bool {
                matches!(
                    self,
                    #(#is_cooperative_matrix_nv_items)|*
                )
            }

            /// Returns whether the instruction is a group operation instruction.
            pub fn is_group_operation(&self) -> bool {
                matches!(
                    self,
                    #(#is_group_operation_items)|*
                )
            }

            /// Returns whether the instruction is a quad group operation instruction.
            pub fn is_quad_group_operation(&self) -> bool {
                matches!(
                    self,
                    #(#is_quad_group_operation_items)|*
                )
            }

            /// Returns whether the instruction is an `ImageFetch*` instruction.
            pub fn is_image_fetch(&self) -> bool {
                matches!(
                    self,
                    #(#is_image_fetch_items)|*
                )
            }

            /// Returns whether the instruction is an `Image*Gather` instruction.
            pub fn is_image_gather(&self) -> bool {
                matches!(
                    self,
                    #(#is_image_gather_items)|*
                )
            }

            /// Returns whether the instruction is an `ImageSample*` instruction.
            pub fn is_image_sample(&self) -> bool {
                matches!(
                    self,
                    #(#is_image_sample_items)|*
                )
            }

            /// Returns the `Id` of the execution scope ID operand, if any.
            pub fn execution_scope_id(&self) -> Option<Id> {
                match self {
                    #(#execution_scope_id_items)|* => Some(*execution),
                    _ => None
                }
            }

            /// Returns the `Id` of the memory scope ID operand, if any.
            pub fn memory_scope_id(&self) -> Option<Id> {
                match self {
                    #(#memory_scope_id_items)|* => Some(*memory),
                    _ => None
                }
            }

            /// Returns the image operands, if any.
            pub fn image_operands(&self) -> Option<&ImageOperands> {
                match self {
                    #(#image_operands_items)|* => Some(image_operands),
                    _ => None
                }
            }
        }
    };

    let opcode_error = if spec_constant {
        format_ident!("UnknownSpecConstantOpcode")
    } else {
        format_ident!("UnknownOpcode")
    };

    quote! {
        #[derive(Clone, Debug, PartialEq)]
        #[doc=#doc]
        pub enum #enum_name {
            #(#struct_items)*
        }

        impl #enum_name {
            #[allow(dead_code)]
            fn parse(reader: &mut InstructionReader<'_>) -> Result<Self, ParseError> {
                let opcode = (reader.next_word()? & 0xffff) as u16;

                match opcode {
                    #(#parse_items)*
                    opcode => Err(reader.map_err(ParseErrors::#opcode_error(opcode))),
                }
            }

            #result_fns
        }
    }
}

fn instruction_members(grammar: &SpirvGrammar) -> Vec<InstructionMember> {
    let operand_kinds = kinds_to_types(grammar);
    grammar
        .instructions
        .iter()
        .map(|instruction| {
            let name = instruction.opname.strip_prefix("Op").unwrap();
            let is_atomic_operation = instruction.class == "Atomic";
            let is_cooperative_matrix =
                name.starts_with("CooperativeMatrix") && !name.ends_with("NV");
            let is_cooperative_matrix_nv =
                name.starts_with("CooperativeMatrix") && name.ends_with("NV");
            let is_group_operation =
                instruction.class == "Group" || instruction.class == "Non-Uniform";
            let is_quad_group_operation = is_group_operation && instruction.opname.contains("Quad");
            let is_image_fetch = name.starts_with("ImageFetch");
            let is_image_gather = name.starts_with("Image") && name.ends_with("Gather");
            let is_image_sample = name.starts_with("ImageSample");
            let mut has_result_id = false;
            let mut has_result_type_id = false;
            let mut has_execution_scope_id = false;
            let mut has_memory_scope_id = false;
            let mut has_image_operands = None;
            let mut operand_names = HashMap::default();

            let mut operands = instruction
                .operands
                .iter()
                .map(|operand| {
                    let name = if operand.kind == "IdResult" {
                        has_result_id = true;
                        format_ident!("result_id")
                    } else if operand.kind == "IdResultType" {
                        has_result_type_id = true;
                        format_ident!("result_type_id")
                    } else {
                        let member_name = to_member_name(&operand.kind, operand.name.as_deref());

                        if operand.kind == "IdScope" {
                            if member_name == "execution" {
                                has_execution_scope_id = true;
                            } else if member_name == "memory" {
                                has_memory_scope_id = true;
                            }
                        } else if operand.kind == "ImageOperands" {
                            if operand.quantifier == Some('?') {
                                has_image_operands = Some(true);
                            } else {
                                has_image_operands = Some(false);
                            }
                        }

                        format_ident!("{}", member_name)
                    };

                    *operand_names.entry(name.clone()).or_insert(0) += 1;

                    let (ty, parse) = &operand_kinds[operand.kind.as_str()];
                    let ty = match operand.quantifier {
                        Some('?') => quote! { Option<#ty> },
                        Some('*') => quote! { Vec<#ty> },
                        _ => ty.clone(),
                    };
                    let parse = match operand.quantifier {
                        Some('?') => quote! {
                            if !reader.is_empty() {
                                Some(#parse)
                            } else {
                                None
                            }
                        },
                        Some('*') => quote! {{
                            let mut vec = Vec::new();
                            while !reader.is_empty() {
                                vec.push(#parse);
                            }
                            vec
                        }},
                        _ => parse.clone(),
                    };

                    OperandMember { name, ty, parse }
                })
                .collect::<Vec<_>>();

            // Add number to operands with identical names
            for name in operand_names
                .into_iter()
                .filter_map(|(n, c)| if c > 1 { Some(n) } else { None })
            {
                for (num, operand) in (1..).zip(operands.iter_mut().filter(|o| o.name == name)) {
                    operand.name = format_ident!("{}{}", name, format!("{}", num));
                }
            }

            InstructionMember {
                name: format_ident!("{}", name),
                is_atomic_operation,
                is_cooperative_matrix,
                is_cooperative_matrix_nv,
                is_group_operation,
                is_quad_group_operation,
                is_image_fetch,
                is_image_gather,
                is_image_sample,
                has_result_id,
                has_result_type_id,
                has_execution_scope_id,
                has_memory_scope_id,
                has_image_operands,
                opcode: instruction.opcode,
                operands,
            }
        })
        .collect()
}

#[derive(Clone, Debug)]
struct KindEnumMember {
    name: Ident,
    value: u32,
    parameters: Vec<OperandMember>,
}

fn bit_enum_output(enums: &[(Ident, Vec<KindEnumMember>)]) -> TokenStream {
    let enum_items = enums.iter().map(|(name, members)| {
        let members_items = members.iter().map(
            |KindEnumMember {
                 name, parameters, ..
             }| {
                if parameters.is_empty() {
                    quote! {
                        pub #name: bool,
                    }
                } else if let [OperandMember { ty, .. }] = parameters.as_slice() {
                    quote! {
                        pub #name: Option<#ty>,
                    }
                } else {
                    let params = parameters.iter().map(|OperandMember { ty, .. }| {
                        quote! { #ty }
                    });
                    quote! {
                        pub #name: Option<(#(#params),*)>,
                    }
                }
            },
        );
        let from_items = members.iter().map(
            |KindEnumMember {
                 name,
                 value,
                 parameters,
             }| {
                if parameters.is_empty() {
                    quote! { #name: value & #value != 0, }
                } else {
                    quote! { #name: None, }
                }
            },
        );
        let parse_items = members.iter().map(
            |KindEnumMember {
                 name,
                 value,
                 parameters,
                 ..
             }| {
                if parameters.is_empty() {
                    quote! {
                        #name: value & #value != 0,
                    }
                } else {
                    let some = if let [OperandMember { parse, .. }] = parameters.as_slice() {
                        quote! { #parse }
                    } else {
                        let parse = parameters.iter().map(|OperandMember { parse, .. }| parse);
                        quote! { (#(#parse),*) }
                    };

                    quote! {
                        #name: if value & #value != 0 {
                            Some(#some)
                        } else {
                            None
                        },
                    }
                }
            },
        );

        quote! {
            #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
            #[allow(non_camel_case_types)]
            pub struct #name {
                #(#members_items)*
            }

            impl #name {
                #[allow(dead_code)]
                fn parse(reader: &mut InstructionReader<'_>) -> Result<#name, ParseError> {
                    let value = reader.next_word()?;

                    Ok(Self {
                        #(#parse_items)*
                    })
                }
            }

            impl From<u32> for #name {
                #[allow(unused_variables)]
                fn from(value: u32) -> Self {
                    Self {
                        #(#from_items)*
                    }
                }
            }
        }
    });

    quote! {
        #(#enum_items)*
    }
}

fn bit_enum_members(grammar: &SpirvGrammar) -> Vec<(Ident, Vec<KindEnumMember>)> {
    let parameter_kinds = kinds_to_types(grammar);

    grammar
        .operand_kinds
        .iter()
        .filter(|operand_kind| operand_kind.category == "BitEnum")
        .map(|operand_kind| {
            let mut previous_value = None;

            let members = operand_kind
                .enumerants
                .iter()
                .filter_map(|enumerant| {
                    // Skip enumerants with the same value as the previous.
                    if previous_value == Some(&enumerant.value) {
                        return None;
                    }

                    previous_value = Some(&enumerant.value);

                    let value = enumerant
                        .value
                        .as_str()
                        .unwrap()
                        .strip_prefix("0x")
                        .unwrap();
                    let value = u32::from_str_radix(value, 16).unwrap();

                    if value == 0 {
                        return None;
                    }

                    let name = match enumerant.enumerant.as_str() {
                        "2x2" => format_ident!("two_x2"),
                        _ => match enumerant.enumerant.to_snake_case().as_str() {
                            "const" => format_ident!("constant"),
                            "not_na_n" => format_ident!("not_nan"),
                            name => format_ident!("{}", name),
                        },
                    };

                    let parameters = enumerant
                        .parameters
                        .iter()
                        .map(|param| {
                            let name = format_ident!(
                                "{}",
                                to_member_name(&param.kind, param.name.as_deref())
                            );
                            let (ty, parse) = parameter_kinds[param.kind.as_str()].clone();

                            OperandMember { name, ty, parse }
                        })
                        .collect();

                    Some(KindEnumMember {
                        name,
                        value,
                        parameters,
                    })
                })
                .collect();

            (format_ident!("{}", operand_kind.kind), members)
        })
        .collect()
}

fn value_enum_output(enums: &[(Ident, Vec<KindEnumMember>)]) -> TokenStream {
    let enum_items = enums.iter().map(|(name, members)| {
        let members_items = members.iter().map(
            |KindEnumMember {
                 name, value, parameters, ..
             }| {
                if parameters.is_empty() {
                    quote! {
                        #name = #value,
                    }
                } else {
                    let params = parameters.iter().map(|OperandMember { name, ty, .. }| {
                        quote! { #name: #ty, }
                    });
                    quote! {
                        #name {
                            #(#params)*
                        } = #value,
                    }
                }
            },
        );
        let try_from_items = members
            .iter()
            .filter(|member| member.parameters.is_empty())
            .map(|KindEnumMember { name, value, .. }| {
                quote! { #value => Ok(Self::#name), }
            });
        let parse_items = members.iter().map(
            |KindEnumMember {
                 name,
                 value,
                 parameters,
                 ..
             }| {
                if parameters.is_empty() {
                    quote! {
                        #value => Ok(Self::#name),
                    }
                } else {
                    let params_items =
                        parameters.iter().map(|OperandMember { name, parse, .. }| {
                            quote! {
                                #name: #parse,
                            }
                        });

                    // See the documentation in `instruction_output` for why we wrap each match arm
                    // in a function.
                    quote! {
                        #value => (|reader: &mut InstructionReader<'_>| Ok(Self::#name {
                            #(#params_items)*
                        }))(reader),
                    }
                }
            },
        );
        let name_string = name.to_string();

        let derives = match name_string.as_str() {
            "Decoration" => quote! { #[derive(Clone, Debug, PartialEq)] },
            _ => quote! { #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)] },
        };

        quote! {
            #derives
            #[allow(non_camel_case_types)]
            #[repr(u32)]
            pub enum #name {
                #(#members_items)*
            }

            impl #name {
                #[allow(dead_code)]
                fn parse(reader: &mut InstructionReader<'_>) -> Result<#name, ParseError> {
                    match reader.next_word()? {
                        #(#parse_items)*
                        value => Err(reader.map_err(ParseErrors::UnknownEnumerant(#name_string, value))),
                    }
                }
            }

            impl TryFrom<u32> for #name {
                type Error = ();

                fn try_from(val: u32) -> Result<Self, Self::Error> {
                    match val {
                        #(#try_from_items)*
                        _ => Err(()),
                    }
                }
            }
        }
    });

    quote! {
        #(#enum_items)*
    }
}

fn value_enum_members(grammar: &SpirvGrammar) -> Vec<(Ident, Vec<KindEnumMember>)> {
    let parameter_kinds = kinds_to_types(grammar);

    grammar
        .operand_kinds
        .iter()
        .filter(|operand_kind| operand_kind.category == "ValueEnum")
        .map(|operand_kind| {
            let mut previous_value = None;

            let members = operand_kind
                .enumerants
                .iter()
                .filter_map(|enumerant| {
                    // Skip enumerants with the same value as the previous.
                    if previous_value == Some(&enumerant.value) {
                        return None;
                    }

                    previous_value = Some(&enumerant.value);

                    let name = match enumerant.enumerant.as_str() {
                        "1D" => format_ident!("Dim1D"),
                        "2D" => format_ident!("Dim2D"),
                        "3D" => format_ident!("Dim3D"),
                        name => format_ident!("{}", name),
                    };
                    let parameters = enumerant
                        .parameters
                        .iter()
                        .map(|param| {
                            let name = format_ident!(
                                "{}",
                                to_member_name(&param.kind, param.name.as_deref())
                            );
                            let (ty, parse) = parameter_kinds[param.kind.as_str()].clone();

                            OperandMember { name, ty, parse }
                        })
                        .collect();

                    Some(KindEnumMember {
                        name,
                        value: enumerant.value.as_u64().unwrap() as u32,
                        parameters,
                    })
                })
                .collect();

            (format_ident!("{}", operand_kind.kind), members)
        })
        .collect()
}

fn to_member_name(kind: &str, name: Option<&str>) -> Cow<'static, str> {
    if let Some(name) = name {
        let name = name.to_snake_case();

        // Fix some weird names
        match name.as_str() {
            "argument_0_argument_1" => "arguments".into(),
            "member_0_type_member_1_type" => "member_types".into(),
            "operand_1_operand_2" => "operands".into(),
            "parameter_0_type_parameter_1_type" => "parameter_types".into(),
            "the_name_of_the_opaque_type" => "name".into(),
            "d_ref" => "dref".into(),
            "type" => "ty".into(),   // type is a keyword
            "use" => "usage".into(), // use is a keyword
            _ => name.replace("operand_", "operand").into(),
        }
    } else {
        kind.to_snake_case().into()
    }
}

fn kinds_to_types(grammar: &SpirvGrammar) -> HashMap<&str, (TokenStream, TokenStream)> {
    grammar
        .operand_kinds
        .iter()
        .map(|k| {
            let (ty, parse) = match k.kind.as_str() {
                "LiteralContextDependentNumber" => {
                    (quote! { Vec<u32> }, quote! { reader.remainder() })
                }
                "LiteralInteger" | "LiteralExtInstInteger" => {
                    (quote! { u32 }, quote! { reader.next_word()? })
                }
                "LiteralSpecConstantOpInteger" => (
                    quote! { SpecConstantInstruction },
                    quote! { SpecConstantInstruction::parse(reader)? },
                ),
                "LiteralString" => (quote! { String }, quote! { reader.next_string()? }),
                "PairIdRefIdRef" => (
                    quote! { (Id, Id) },
                    quote! {
                        (
                            Id(reader.next_word()?),
                            Id(reader.next_word()?),
                        )
                    },
                ),
                "PairIdRefLiteralInteger" => (
                    quote! { (Id, u32) },
                    quote! {
                        (
                            Id(reader.next_word()?),
                            reader.next_word()?
                        )
                    },
                ),
                "PairLiteralIntegerIdRef" => (
                    quote! { (u32, Id) },
                    quote! {
                    (
                        reader.next_word()?,
                        Id(reader.next_word()?)),
                    },
                ),
                _ if k.kind.starts_with("Id") => {
                    (quote! { Id }, quote! { Id(reader.next_word()?) })
                }
                ident => {
                    let ident = format_ident!("{}", ident);
                    (quote! { #ident }, quote! { #ident::parse(reader)? })
                }
            };

            (k.kind.as_str(), (ty, parse))
        })
        .chain([(
            "LiteralFloat",
            (
                quote! { f32 },
                quote! { f32::from_bits(reader.next_word()?) },
            ),
        )])
        .collect()
}
