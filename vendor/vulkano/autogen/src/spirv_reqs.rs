use super::{
    spirv_grammar::{SpirvGrammar, SpirvKindEnumerant},
    write_file, IndexMap, RequiresOneOf, VkRegistryData,
};
use heck::ToSnakeCase;
use indexmap::map::Entry;
use nom::{
    bytes::complete::tag,
    character::complete,
    combinator::all_consuming,
    sequence::{preceded, separated_pair},
    IResult, Parser,
};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use vk_parse::SpirvExtOrCap;

pub fn write(vk_data: &VkRegistryData<'_>, grammar: &SpirvGrammar) {
    let grammar_enumerants = grammar
        .operand_kinds
        .iter()
        .find(|operand_kind| operand_kind.kind == "Capability")
        .unwrap()
        .enumerants
        .as_slice();
    let spirv_capabilities_output = spirv_reqs_output(
        &spirv_capabilities_members(&vk_data.spirv_capabilities, grammar_enumerants),
        false,
    );
    let spirv_extensions_output =
        spirv_reqs_output(&spirv_extensions_members(&vk_data.spirv_extensions), true);
    write_file(
        "spirv_reqs.rs",
        format!(
            "vk.xml header version {}.{}.{}",
            vk_data.header_version.0, vk_data.header_version.1, vk_data.header_version.2
        ),
        quote! {
            #spirv_capabilities_output
            #spirv_extensions_output
        },
    );
}

struct SpirvReqsMember {
    name: String,
    requires_one_of: RequiresOneOf,
    requires_properties: Vec<RequiresProperty>,
}

struct RequiresProperty {
    name: String,
    value: PropertyValue,
}

enum PropertyValue {
    Bool,
    FlagsIntersects {
        path: TokenStream,
        ty: String,
        flag: String,
    },
}

fn spirv_reqs_output(members: &[SpirvReqsMember], is_extension: bool) -> TokenStream {
    let (item_type, fn_def, not_supported_vuid, item_vuid) = if is_extension {
        (
            "extension",
            quote! { validate_spirv_extension(device: &Device, item: &str) },
            "VUID-VkShaderModuleCreateInfo-pCode-08739",
            "VUID-VkShaderModuleCreateInfo-pCode-08740",
        )
    } else {
        (
            "capability",
            quote! { validate_spirv_capability(device: &Device, item: Capability) },
            "VUID-VkShaderModuleCreateInfo-pCode-08741",
            "VUID-VkShaderModuleCreateInfo-pCode-08742",
        )
    };

    let items = members.iter().map(
        |SpirvReqsMember {
             name,
             requires_one_of,
             requires_properties,
         }| {
            let arm = if is_extension {
                quote! { #name }
            } else {
                let name = format_ident!("{}", name);
                quote! { Capability::#name }
            };

            let (requires_one_of_condition_items, requires_one_of_items) =
                if !requires_one_of.is_empty() {
                    let &RequiresOneOf {
                        api_version,
                        ref device_extensions,
                        instance_extensions: _,
                        ref device_features,
                    } = requires_one_of;
                    let condition_items = api_version
                        .iter()
                        .map(|version| {
                            let version = format_ident!("V{}_{}", version.0, version.1);
                            quote! { api_version >= crate::Version::#version }
                        })
                        .chain(device_extensions.iter().map(|name| {
                            let ident = format_ident!("{}", name);
                            quote! { device_extensions.#ident }
                        }))
                        .chain(device_features.iter().map(|name| {
                            let ident = format_ident!("{}", name);
                            quote! { device_features.#ident }
                        }))
                        .collect();
                    let items = api_version
                        .iter()
                        .map(|(major, minor)| {
                            let version = format_ident!("V{}_{}", major, minor);
                            quote! {
                                crate::RequiresAllOf(&[
                                    crate::Requires::APIVersion(crate::Version::#version),
                                ]),
                            }
                        })
                        .chain(device_extensions.iter().map(|name| {
                            quote! {
                                crate::RequiresAllOf(&[
                                    crate::Requires::DeviceExtension(#name),
                                ]),
                            }
                        }))
                        .chain(device_features.iter().map(|name| {
                            quote! {
                                crate::RequiresAllOf(&[
                                    crate::Requires::DeviceFeature(#name),
                                ]),
                            }
                        }))
                        .collect();
                    (condition_items, items)
                } else {
                    (Vec::new(), Vec::new())
                };
            let requires_properties_condition_items: Vec<TokenStream> = requires_properties
                .iter()
                .map(|RequiresProperty { name, value }| {
                    let name = format_ident!("{}", name);
                    let access = match value {
                        PropertyValue::Bool => quote! {},
                        PropertyValue::FlagsIntersects { path, ty, flag } => {
                            let ty = format_ident!("{}", ty);
                            let flag = format_ident!("{}", flag);
                            quote! {
                                .map(|x| x.intersects(#path :: #ty :: #flag))
                            }
                        }
                    };
                    quote! {
                        device.physical_device().properties().#name #access .unwrap_or(false)
                    }
                })
                .collect();
            let condition_items: Vec<TokenStream> = requires_one_of_condition_items
                .into_iter()
                .chain(requires_properties_condition_items)
                .collect();
            if condition_items.is_empty() {
                return quote! { #arm => (), };
            }
            let problem = {
                let mut parts: Vec<String> = Vec::new();
                if !requires_one_of.is_empty() {
                    parts.push(format!("uses the SPIR-V {} `{}`", item_type, name));
                }
                if !requires_properties.is_empty() {
                    let requirements_items: Vec<_> = requires_properties
                        .iter()
                        .map(|RequiresProperty { name, value }| match value {
                            PropertyValue::Bool => format!("`{}` must be `true`", name),
                            PropertyValue::FlagsIntersects { path: _, ty, flag } => {
                                format!("`{}` must contain `{}::{}`", name, ty, flag)
                            }
                        })
                        .collect();
                    parts.push(format!(
                        "the device properties do not meet at least one of the requirements ({})",
                        requirements_items.join(" or ")
                    ));
                }
                parts.join(", ")
            };
            let requires_one_of_field = if !requires_one_of.is_empty() {
                quote! {
                    requires_one_of: crate::RequiresOneOf(&[
                        #(#requires_one_of_items)*
                    ]),
                }
            } else {
                quote! {}
            };
            quote! {
                #arm => {
                    if !(#(#condition_items)||*) {
                        return Err(Box::new(crate::ValidationError {
                            problem: #problem.into(),
                            #requires_one_of_field
                            vuids: &[#item_vuid],
                            ..Default::default()
                        }));
                    }
                },
            }
        },
    );

    let problem = format!(
        "uses the SPIR-V {} `{{item:?}}`, which is not supported by Vulkan",
        item_type,
    );
    quote! {
        fn #fn_def -> Result<(), Box<ValidationError>> {
            #[allow(unused_variables)]
            let api_version = device.api_version();
            #[allow(unused_variables)]
            let device_extensions = device.enabled_extensions();
            #[allow(unused_variables)]
            let device_features = device.enabled_features();
            #[allow(unused_variables)]
            let properties = device.physical_device().properties();

            match item {
                #(#items)*
                _ => {
                    return Err(Box::new(crate::ValidationError {
                        problem: format!(#problem).into(),
                        vuids: &[#not_supported_vuid],
                        ..Default::default()
                    }));
                }
            }
            Ok(())
        }
    }
}

fn spirv_capabilities_members(
    capabilities: &[&SpirvExtOrCap],
    grammar_enumerants: &[SpirvKindEnumerant],
) -> Vec<SpirvReqsMember> {
    let mut members: IndexMap<String, SpirvReqsMember> = IndexMap::default();

    for ext_or_cap in capabilities {
        let (requires_one_of, requires_properties) = make_requires(&ext_or_cap.enables);

        // Find the capability in the list of enumerants, then go backwards through the list to
        // find the first enumerant with the same value.
        let enumerant_pos = match grammar_enumerants
            .iter()
            .position(|enumerant| enumerant.enumerant == ext_or_cap.name)
        {
            Some(pos) => pos,
            // This can happen if the grammar file is behind on the vk.xml file.
            None => continue,
        };
        let enumerant_value = &grammar_enumerants[enumerant_pos].value;

        let name = if let Some(enumerant) = grammar_enumerants[..enumerant_pos]
            .iter()
            .rev()
            .take_while(|enumerant| &enumerant.value == enumerant_value)
            .last()
        {
            // Another enumerant was found with the same value, so this one is an alias.
            &enumerant.enumerant
        } else {
            // No other enumerant was found, so this is its canonical name.
            &ext_or_cap.name
        };

        match members.entry(name.clone()) {
            Entry::Occupied(entry) => {
                let member = entry.into_mut();
                member.requires_one_of |= &requires_one_of;
                member.requires_properties.extend(requires_properties);
            }
            Entry::Vacant(entry) => {
                entry.insert(SpirvReqsMember {
                    name: name.clone(),
                    requires_one_of,
                    requires_properties,
                });
            }
        }
    }

    members.into_iter().map(|(_, v)| v).collect()
}

fn spirv_extensions_members(extensions: &[&SpirvExtOrCap]) -> Vec<SpirvReqsMember> {
    extensions
        .iter()
        .map(|ext_or_cap| {
            let (requires_one_of, requires_properties) = make_requires(&ext_or_cap.enables);

            SpirvReqsMember {
                name: ext_or_cap.name.clone(),
                requires_one_of,
                requires_properties,
            }
        })
        .collect()
}

fn make_requires(enables: &[vk_parse::Enable]) -> (RequiresOneOf, Vec<RequiresProperty>) {
    fn vk_api_version(input: &str) -> IResult<&str, (u32, u32)> {
        all_consuming(preceded(
            tag("VK_API_VERSION_").or(tag("VK_VERSION_")),
            separated_pair(complete::u32, complete::char('_'), complete::u32),
        ))(input)
    }

    let mut requires_one_of = RequiresOneOf::default();
    let mut requires_properties = vec![];

    for enable in enables {
        match enable {
            vk_parse::Enable::Version(version) => {
                if version != "VK_VERSION_1_0" {
                    requires_one_of.api_version = Some(vk_api_version(version).unwrap().1);
                }
            }
            vk_parse::Enable::Extension(extension) => {
                requires_one_of
                    .device_extensions
                    .push(extension.strip_prefix("VK_").unwrap().to_snake_case());
            }
            vk_parse::Enable::Feature(feature) => {
                requires_one_of
                    .device_features
                    .push(feature.feature.to_snake_case());
            }
            vk_parse::Enable::Property(property) => {
                let name = property.member.to_snake_case();

                let value = if property.value == "VK_TRUE" {
                    PropertyValue::Bool
                } else if let Some(member) = property.value.strip_prefix("VK_SUBGROUP_FEATURE_") {
                    PropertyValue::FlagsIntersects {
                        path: quote! { crate::device::physical },
                        ty: "SubgroupFeatures".to_string(),
                        flag: member
                            .trim_end_matches("_BIT_NV")
                            .trim_end_matches("_BIT")
                            .to_string(),
                    }
                } else {
                    unimplemented!()
                };

                requires_properties.push(RequiresProperty { name, value });
            }
            _ => unimplemented!(),
        }
    }

    // A single SPIR-V requirement may be satisfied by any of several
    // extensions/features OR device properties, so both may be present.

    requires_one_of.device_extensions.sort_unstable();
    requires_one_of.device_extensions.dedup();

    requires_one_of.device_features.sort_unstable();
    requires_one_of.device_features.dedup();

    (requires_one_of, requires_properties)
}
