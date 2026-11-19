use super::{write_file, IndexMap, VkRegistryData};
use foldhash::HashMap;
use heck::ToSnakeCase;
use nom::{
    bytes::complete::{tag, take_until, take_while1},
    character::complete::{self, digit1},
    combinator::{all_consuming, eof},
    sequence::{delimited, tuple},
    IResult,
};
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote, ToTokens};
use vk_parse::{Extension, Type, TypeMember, TypeMemberMarkup, TypeSpec};

pub fn write(vk_data: &VkRegistryData<'_>) {
    let device_properties = DeviceProperties::new(&vk_data.types, &vk_data.extensions);
    let device_properties_output = device_properties.to_output();

    write_file(
        "properties.rs",
        format!(
            "vk.xml header version {}.{}.{}",
            vk_data.header_version.0, vk_data.header_version.1, vk_data.header_version.2
        ),
        device_properties_output,
    );
}

struct DeviceProperties {
    properties: Vec<Property>,
    structs: Vec<PropertiesStruct>,
    extension_structs: Vec<PropertiesStruct>,
}

impl DeviceProperties {
    fn new(
        types: &HashMap<&str, (&Type, Vec<&str>)>,
        extensions: &IndexMap<&str, &Extension>,
    ) -> Self {
        let mut structs: Vec<_> = [
            "VkPhysicalDeviceProperties",
            "VkPhysicalDeviceLimits",
            "VkPhysicalDeviceSparseProperties",
        ]
        .into_iter()
        .filter_map(|struct_name_c| {
            let (struct_type_info, provided_by) = &types[struct_name_c];
            PropertiesStruct::new(struct_type_info, provided_by, extensions)
        })
        .collect();

        let mut extension_structs: Vec<_> = extension_structs_sorted(types)
            .into_iter()
            .filter_map(|(struct_type_info, provided_by)| {
                PropertiesStruct::new(struct_type_info, provided_by, extensions)
            })
            .collect();
        // Sort properties
        let mut properties = HashMap::default();
        let mut property_names = Vec::new();

        for properties_struct in structs.iter_mut().chain(extension_structs.iter_mut()) {
            for property in &properties_struct.members {
                properties
                    .entry(property.property_name.to_string())
                    .or_insert_with_key(|key| {
                        property_names.push(key.clone());
                        property
                    });
            }
        }

        property_names.sort_unstable();

        Self {
            properties: property_names
                .into_iter()
                .map(|name| properties.remove(&name).unwrap().clone())
                .collect(),
            structs,
            extension_structs,
        }
    }

    fn to_output(&self) -> TokenStream {
        let Self {
            properties,
            structs,
            extension_structs,
        } = self;

        let struct_definition = {
            let iter = properties.iter().map(Property::to_properties_struct_member);

            quote! {
                /// Represents all the properties of a physical device.
                ///
                /// Depending on the highest version of Vulkan supported by the physical device, and
                /// the available extensions, not every property may be available. For that reason,
                /// some properties are wrapped in an `Option`.
                #[derive(Clone, Debug)]
                pub struct DeviceProperties {
                    #(#iter)*
                    pub _ne: crate::NonExhaustive<'static>,
                }
            }
        };

        let to_mut_vk = {
            quote! {
                pub(crate) fn to_mut_vk(
                ) -> ash::vk::PhysicalDeviceProperties {
                    ash::vk::PhysicalDeviceProperties::default()
                }
            }
        };

        let to_mut_vk2 = {
            let extensions_vk_push_next_iter = extension_structs
                .iter()
                .map(PropertiesStruct::to_extensions_vk_push_next);

            quote! {
                pub(crate) fn to_mut_vk2<'a>(
                    extensions_vk: &'a mut PhysicalDeviceProperties2ExtensionsVk<'_>,
                ) -> ash::vk::PhysicalDeviceProperties2<'a> {
                    let mut val_vk = ash::vk::PhysicalDeviceProperties2::default();

                    #(#extensions_vk_push_next_iter)*

                    val_vk
                }
            }
        };

        let to_mut_vk2_extensions = {
            let fields1_vk_destructure_iter = extension_structs
                .iter()
                .filter_map(PropertiesStruct::to_fields1_vk_destructure);
            let extensions_vk_constructor_iter = extension_structs
                .iter()
                .map(PropertiesStruct::to_extensions_vk_constructor);

            quote! {
                pub(crate) fn to_mut_vk2_extensions<'a>(
                    fields1_vk: &'a mut PhysicalDeviceProperties2Fields1Vk,
                    api_version: Version,
                    device_extensions: &DeviceExtensions,
                    instance_extensions: &InstanceExtensions,
                ) -> PhysicalDeviceProperties2ExtensionsVk<'a> {
                    let PhysicalDeviceProperties2Fields1Vk {
                        #(#fields1_vk_destructure_iter)*
                    } = fields1_vk;

                    PhysicalDeviceProperties2ExtensionsVk {
                        #(#extensions_vk_constructor_iter)*
                    }
                }
            }
        };

        let to_mut_vk2_extensions_query_count = {
            let extensions_vk_constructor_iter = extension_structs
                .iter()
                .map(PropertiesStruct::to_extensions_vk_constructor_query_count);

            quote! {
                pub(crate) fn to_mut_vk2_extensions_query_count(
                    #[allow(unused)]
                    api_version: Version,
                    device_extensions: &DeviceExtensions,
                    #[allow(unused)]
                    instance_extensions: &InstanceExtensions,
                ) -> PhysicalDeviceProperties2ExtensionsVk<'static> {
                    PhysicalDeviceProperties2ExtensionsVk {
                        #(#extensions_vk_constructor_iter)*
                    }
                }
            }
        };

        let to_mut_vk2_fields1 = {
            let extensions_vk_destructure_iter = extension_structs
                .iter()
                .filter(PropertiesStruct::has_fields_struct)
                .map(PropertiesStruct::to_extensions_vk_destructure);
            let fields1_vk_constructor_iter = extension_structs
                .iter()
                .filter_map(PropertiesStruct::to_fields1_vk_constructor);

            quote! {
                pub(crate) fn to_mut_vk2_fields1(
                    extensions_vk: PhysicalDeviceProperties2ExtensionsVk<'_>,
                ) -> PhysicalDeviceProperties2Fields1Vk {
                    let PhysicalDeviceProperties2ExtensionsVk {
                        #(#extensions_vk_destructure_iter)*
                        ..
                    } = extensions_vk;

                    PhysicalDeviceProperties2Fields1Vk {
                        #(#fields1_vk_constructor_iter)*
                    }
                }
            }
        };

        let from_vk = {
            let device_properties_members_iter =
                properties.iter().map(Property::to_properties_constructor);
            let structs_iter = structs.iter().map(PropertiesStruct::to_destructure);

            quote! {
                pub(crate) fn from_vk(
                    val_vk: &ash::vk::PhysicalDeviceProperties,
                ) -> Self {
                    let ash::vk::PhysicalDeviceProperties {
                        limits,
                        sparse_properties,
                        ..
                    } = val_vk;

                    #(#structs_iter)*

                    Self {
                        #(#device_properties_members_iter)*
                        _ne: crate::NE,
                    }
                }
            }
        };

        let from_vk2 = {
            let device_properties_members_iter =
                properties.iter().map(Property::to_properties_constructor);
            let destructure_extensions_vk_iter = extension_structs
                .iter()
                .map(PropertiesStruct::to_extensions_vk_destructure);
            let destructure_fields1_vk_iter = extension_structs
                .iter()
                .filter_map(PropertiesStruct::to_fields1_vk_destructure);
            let structs_iter = structs.iter().map(PropertiesStruct::to_destructure);
            let extension_structs_iter = extension_structs
                .iter()
                .map(PropertiesStruct::to_from_vk2_extension);

            quote! {
                pub(crate) fn from_vk2(
                    val_vk: &ash::vk::PhysicalDeviceProperties2<'_>,
                    extensions_vk: &PhysicalDeviceProperties2ExtensionsVk<'_>,
                    fields1_vk: &PhysicalDeviceProperties2Fields1Vk,
                ) -> Self {
                    let ash::vk::PhysicalDeviceProperties2 {
                        properties: val_vk @ ash::vk::PhysicalDeviceProperties {
                            limits,
                            sparse_properties,
                            ..
                        },
                        ..
                    } = val_vk;
                    let PhysicalDeviceProperties2ExtensionsVk {
                        #(#destructure_extensions_vk_iter)*
                    } = extensions_vk;
                    let PhysicalDeviceProperties2Fields1Vk {
                        #(#destructure_fields1_vk_iter)*
                    } = fields1_vk;

                    #(#structs_iter)*

                    let mut val = Self {
                        #(#device_properties_members_iter)*
                        _ne: crate::NE,
                    };

                    #(#extension_structs_iter)*

                    val
                }
            }
        };

        let default = {
            let iter = properties.iter().map(Property::to_default_constructor);

            quote! {
                impl Default for DeviceProperties {
                    fn default() -> Self {
                        DeviceProperties {
                            #(#iter)*
                            _ne: crate::NE,
                        }
                    }
                }
            }
        };

        let structs_vk = {
            let extensions_vk_members_iter = extension_structs
                .iter()
                .map(PropertiesStruct::to_extensions_vk_struct_member);
            let extensions_vk_destructure_iter = extension_structs
                .iter()
                .map(PropertiesStruct::to_extensions_vk_destructure);
            let extensions_vk_unborrow_fields1_iter = extension_structs
                .iter()
                .filter(PropertiesStruct::has_fields_struct)
                .map(PropertiesStruct::to_extension_vk_unborrow);
            let extensions_vk_constructor_iter = extension_structs
                .iter()
                .map(PropertiesStruct::to_extensions_vk_destructure);

            let fields1_vk_members_iter = extension_structs
                .iter()
                .filter_map(PropertiesStruct::to_fields1_vk_struct_member);
            let fields_structs_iter = extension_structs
                .iter()
                .map(PropertiesStruct::to_fields_struct);

            quote! {
                pub(crate) struct PhysicalDeviceProperties2ExtensionsVk<'a> {
                    #(#extensions_vk_members_iter)*
                }

                impl PhysicalDeviceProperties2ExtensionsVk<'_> {
                    pub(crate) fn unborrow(self) -> PhysicalDeviceProperties2ExtensionsVk<'static> {
                        let Self {
                            #(#extensions_vk_destructure_iter)*
                        } = self;

                        #(#extensions_vk_unborrow_fields1_iter)*

                        PhysicalDeviceProperties2ExtensionsVk {
                            #(#extensions_vk_constructor_iter)*
                        }
                    }
                }

                pub(crate) struct PhysicalDeviceProperties2Fields1Vk {
                    #(#fields1_vk_members_iter)*
                }

                #(#fields_structs_iter)*
            }
        };

        quote! {
            #struct_definition

            impl DeviceProperties {
                #to_mut_vk
                #to_mut_vk2
                #to_mut_vk2_extensions
                #to_mut_vk2_extensions_query_count
                #to_mut_vk2_fields1

                #from_vk
                #from_vk2
            }

            #default
            #structs_vk
        }
    }
}

#[derive(Debug)]
struct PropertiesStruct {
    struct_name: Ident,
    var_name: Ident,
    var_ty: TokenStream,
    is_extension: bool,
    provided_by: Vec<TokenStream>,
    members: Vec<Property>,
    fields_struct: Option<FieldsStruct>,
}

#[derive(Debug)]
struct FieldsStruct {
    fields_struct_name: Ident,
    fields1_vk_member: Ident,
}

impl PropertiesStruct {
    fn new(
        struct_type_info: &Type,
        provided_by: &[&str],
        extensions: &IndexMap<&str, &Extension>,
    ) -> Option<Self> {
        let TypeSpec::Members(members) = &struct_type_info.spec else {
            return None;
        };

        let is_extension = is_extension_struct(struct_type_info);

        let struct_name_c = struct_type_info.name.as_ref().unwrap();
        let struct_name = struct_name_c.strip_prefix("Vk").unwrap();
        // Skip structs containing a member type we cannot map to a Rust type
        // (e.g. nested structs or new Vulkan types we do not support).
        if members.iter().any(member_is_unmappable) {
            return None;
        }
        let mut has_pointer_property = false;

        let members = members
            .iter()
            .filter_map(|member| {
                let property = Property::new(member, members, struct_name_c, is_extension)?;

                if property.pointer.is_some() {
                    has_pointer_property = true;
                }

                Some(property)
            })
            .collect();

        Some(PropertiesStruct {
            struct_name: format_ident!("{}", struct_name),
            var_name: match struct_name {
                "PhysicalDeviceProperties" => format_ident!("val_vk"),
                "PhysicalDeviceLimits" => format_ident!("limits"),
                "PhysicalDeviceSparseProperties" => format_ident!("sparse_properties"),
                _ => format_ident!(
                    "{}_vk",
                    struct_name
                        .strip_prefix("PhysicalDevice")
                        .unwrap()
                        .to_snake_case()
                ),
            },
            var_ty: format_ident!("{}", struct_name).to_token_stream(),
            is_extension,
            provided_by: provided_by
                .iter()
                .map(|provided_by| {
                    let version = provided_by
                        .strip_prefix("VK_VERSION_")
                        .or_else(|| provided_by.strip_prefix("VK_BASE_VERSION_"))
                        .or_else(|| provided_by.strip_prefix("VK_COMPUTE_VERSION_"))
                        .or_else(|| provided_by.strip_prefix("VK_GRAPHICS_VERSION_"));
                    if let Some(version) = version {
                        let version = format_ident!("V{}", version);
                        quote! { api_version >= Version::#version }
                    } else {
                        let member = format_ident!(
                            "{}_extensions",
                            extensions[provided_by].ext_type.as_ref().unwrap().as_str()
                        );
                        let name = format_ident!(
                            "{}",
                            provided_by
                                .strip_prefix("VK_")
                                .unwrap()
                                .to_ascii_lowercase(),
                        );

                        quote! { #member.#name }
                    }
                })
                .collect(),
            members,
            fields_struct: has_pointer_property.then(|| FieldsStruct {
                fields_struct_name: format_ident!("{}Fields1Vk", struct_name),
                fields1_vk_member: format_ident!("{}_fields1_vk", struct_name.to_snake_case()),
            }),
        })
    }

    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn has_fields_struct(self: &&Self) -> bool {
        self.fields_struct.is_some()
    }

    fn to_fields_struct(&self) -> Option<TokenStream> {
        let Self {
            members,
            fields_struct,
            ..
        } = self;

        fields_struct.as_ref().map(|fields_struct| {
            let FieldsStruct {
                fields_struct_name, ..
            } = fields_struct;
            let members_iter = members.iter().map(Property::to_fields_struct_member);

            quote! {
                pub(crate) struct #fields_struct_name {
                    #(#members_iter)*
                }
            }
        })
    }

    fn to_extensions_vk_struct_member(&self) -> TokenStream {
        let Self {
            var_name,
            var_ty,
            fields_struct,
            ..
        } = self;

        let lifetime = if fields_struct.is_some() {
            quote! { 'a }
        } else {
            quote! { 'static }
        };

        quote! {
            pub(crate) #var_name: Option<ash::vk::#var_ty<#lifetime>>,
        }
    }

    fn to_extensions_vk_constructor(&self) -> TokenStream {
        let Self {
            var_name,
            var_ty,
            provided_by,
            fields_struct,
            members,
            ..
        } = self;

        if let Some(fields_struct) = fields_struct {
            let FieldsStruct {
                fields_struct_name,
                fields1_vk_member,
                ..
            } = fields_struct;

            let destructure_iter = members
                .iter()
                .filter(Property::is_pointer)
                .map(Property::to_destructure);
            let builder_iter = members
                .iter()
                .filter(Property::is_pointer)
                .map(Property::to_builder);

            quote! {
                #var_name: #fields1_vk_member
                    .as_mut()
                    .filter(|_| #(#provided_by)||*)
                    .map(|fields_struct| {
                        let #fields_struct_name {
                            #(#destructure_iter)*
                        } = fields_struct;

                        <ash::vk::#var_ty<'_>>::default()
                            #(#builder_iter)*
                    }),
            }
        } else {
            quote! {
                #var_name: (#(#provided_by)||*)
                    .then(<ash::vk::#var_ty<'_>>::default),
            }
        }
    }

    fn to_extensions_vk_constructor_query_count(&self) -> TokenStream {
        let Self {
            var_name,
            var_ty,
            provided_by,
            fields_struct,
            ..
        } = self;

        let value = if fields_struct.is_some() {
            quote! {
                (#(#provided_by)||*).then(<ash::vk::#var_ty<'_>>::default)
            }
        } else {
            quote! {
                None
            }
        };

        quote! {
            #var_name: #value,
        }
    }

    fn to_extensions_vk_destructure(&self) -> TokenStream {
        let Self { var_name, .. } = self;

        quote! { #var_name, }
    }

    fn to_extensions_vk_push_next(&self) -> TokenStream {
        let Self { var_name, .. } = self;

        quote! {
            if let Some(next) = &mut extensions_vk.#var_name {
                val_vk = val_vk.push(next);
            }
        }
    }

    fn to_extension_vk_unborrow(&self) -> TokenStream {
        let Self {
            var_name, var_ty, ..
        } = self;

        quote! {
            let #var_name = #var_name.map(|val_vk| {
                ash::vk::#var_ty {
                    _marker: std::marker::PhantomData,
                    ..val_vk
                }
            });
        }
    }

    fn to_destructure(&self) -> TokenStream {
        let &Self {
            ref struct_name,
            ref var_name,
            is_extension,
            ref members,
            ..
        } = self;

        let destructure_iter = members.iter().map(Property::to_destructure_len_or_name);

        if is_extension {
            quote! {
                let &ash::vk::#struct_name {
                    #(#destructure_iter)*
                    ..
                } = val_vk;
            }
        } else {
            quote! {
                let &ash::vk::#struct_name {
                    #(#destructure_iter)*
                    ..
                } = #var_name;
            }
        }
    }

    fn to_from_vk2_extension(&self) -> TokenStream {
        let Self {
            var_name,
            fields_struct,
            members,
            ..
        } = self;

        let destructure = self.to_destructure();
        let constructor_iter = members
            .iter()
            .map(Property::to_extension_properties_constructor);

        if let Some(fields_struct) = fields_struct {
            let FieldsStruct {
                fields_struct_name,
                fields1_vk_member,
                ..
            } = fields_struct;

            let destructure_fields_iter = members
                .iter()
                .filter(Property::is_pointer)
                .map(Property::to_destructure);

            quote! {
                if let Some((val_vk, fields_vk)) = #var_name.as_ref().zip(#fields1_vk_member.as_ref()) {
                    #destructure
                    let #fields_struct_name {
                        #(#destructure_fields_iter)*
                    } = fields_vk;

                    #(#constructor_iter)*
                }
            }
        } else {
            quote! {
                if let Some(val_vk) = #var_name {
                    #destructure

                    #(#constructor_iter)*
                }
            }
        }
    }

    fn to_fields1_vk_struct_member(&self) -> Option<TokenStream> {
        let Self { fields_struct, .. } = self;

        fields_struct.as_ref().map(|fields_struct| {
            let FieldsStruct {
                fields_struct_name,
                fields1_vk_member,
                ..
            } = fields_struct;

            quote! {
                pub(crate) #fields1_vk_member: Option<#fields_struct_name>,
            }
        })
    }

    fn to_fields1_vk_constructor(&self) -> Option<TokenStream> {
        let Self {
            struct_name,
            var_name,
            fields_struct,
            members,
            ..
        } = self;

        fields_struct.as_ref().map(|fields_struct| {
            let FieldsStruct {
                fields_struct_name,
                fields1_vk_member,
            } = fields_struct;

            let fields_struct_constructor_iter = members
                .iter()
                .filter_map(Property::to_fields1_vk_constructor);
            let len_field_destructure_iter =
                members.iter().filter_map(Property::to_destructure_len);

            quote! {
                #fields1_vk_member: #var_name.map(|val_vk| {
                    let ash::vk::#struct_name {
                        #(#len_field_destructure_iter)*
                        ..
                    } = val_vk;

                    #fields_struct_name {
                        #(#fields_struct_constructor_iter)*
                    }
                }),
            }
        })
    }

    fn to_fields1_vk_destructure(&self) -> Option<TokenStream> {
        let Self { fields_struct, .. } = self;

        fields_struct.as_ref().map(|fields_struct| {
            let FieldsStruct {
                fields1_vk_member, ..
            } = fields_struct;

            quote! {
                #fields1_vk_member,
            }
        })
    }
}

#[derive(Clone, Debug)]
struct Property {
    property_name: Ident,
    property_name_c: String,
    property_ty: TokenStream,
    pointer: Option<PointerProperty>,
    struct_name_c: String,
    is_extension: bool,
}

#[derive(Clone, Debug)]
struct PointerProperty {
    len_name: Ident,
    pointed_ty_vk: TokenStream,
}

impl Property {
    fn new(
        member: &TypeMember,
        members: &[TypeMember],
        struct_name_c: &str,
        is_extension: bool,
    ) -> Option<Self> {
        let TypeMember::Definition(definition) = member else {
            return None;
        };

        let property_name_c = definition.markup.iter().find_map(|markup| match markup {
            TypeMemberMarkup::Name(name) => Some(name.as_str()),
            _ => None,
        })?;

        if matches!(property_name_c, "sType" | "pNext") {
            return None;
        }

        let ty_c = definition.markup.iter().find_map(|markup| match markup {
            TypeMemberMarkup::Type(ty) => Some(ty.as_str()),
            _ => None,
        })?;

        if matches!(
            ty_c,
            "VkPhysicalDeviceLimits" | "VkPhysicalDeviceSparseProperties"
        ) {
            return None;
        }

        let len_name = definition
            .len
            .as_deref()
            .filter(|&len_member| {
                // Only use the len= field if it refers to another member of this
                // struct, since the len= value can also be
                // other special values such as
                // "null-terminated".
                members.iter().any(|member| match member {
                    TypeMember::Definition(definition) => definition
                        .markup
                        .iter()
                        .find_map(|markup| match markup {
                            TypeMemberMarkup::Name(name) => Some(name.as_str()),
                            _ => None,
                        })
                        .is_some_and(|member_name| member_name == len_member),
                    _ => false,
                })
            })
            .map(|len_member| format_ident!("{}", len_member.to_snake_case()));

        if len_name.is_none() {
            // If this member is the len= field of another member, skip it.
            let is_len = members.iter().any(|member2| match member2 {
                TypeMember::Definition(definition2) => {
                    definition2.len.as_deref() == Some(property_name_c)
                }
                _ => false,
            });

            if is_len {
                return None;
            }
        }

        let array_len = {
            fn array_len(input: &str) -> IResult<&str, &str> {
                let (input, _) = take_until("[")(input)?;
                all_consuming(delimited(
                    complete::char('['),
                    take_while1(|c: char| c.is_ascii_alphanumeric() || c == '_'),
                    complete::char(']'),
                ))(input)
            }

            definition
                .markup
                .iter()
                .find_map(|markup| match markup {
                    TypeMemberMarkup::Enum(len) => Some(len.as_str()),
                    _ => None,
                })
                .or_else(|| array_len(&definition.code).map(|(_, len)| len).ok())
        };
        let property_ty =
            c_type_to_vulkano_type(property_name_c, ty_c, array_len, len_name.is_some())?;

        let property_name = if len_name.is_some() {
            property_name_c.strip_prefix("p").unwrap()
        } else {
            property_name_c
        }
        .to_snake_case();
        let property_name_ident = format_ident!("{}", property_name);

        let pointer = len_name.map(|len_name| {
            let pointed_ty_vk = c_type_to_vk_type(ty_c);

            PointerProperty {
                len_name,
                pointed_ty_vk,
            }
        });

        Some(Property {
            property_name: property_name_ident.clone(),
            property_name_c: property_name_c.to_owned(),
            property_ty: property_ty.clone(),
            pointer,
            struct_name_c: struct_name_c.to_owned(),
            is_extension,
        })
    }

    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn is_pointer(self: &&Self) -> bool {
        self.pointer.is_some()
    }

    fn to_properties_struct_member(&self) -> TokenStream {
        let &Self {
            ref property_name,
            ref property_name_c,
            ref property_ty,
            ref struct_name_c,
            is_extension,
            ..
        } = self;

        let doc = format!(
            "- [Vulkan documentation](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/{}.html#limits-{})",
            struct_name_c,
            property_name_c,
        );

        if is_extension {
            quote! {
                #[doc = #doc]
                pub #property_name: Option<#property_ty>,
            }
        } else {
            quote! {
                #[doc = #doc]
                pub #property_name: #property_ty,
            }
        }
    }

    fn to_fields_struct_member(&self) -> Option<TokenStream> {
        let Self {
            property_name,
            pointer,
            ..
        } = self;

        pointer.as_ref().map(|pointer| {
            let pointed_ty_vk = &pointer.pointed_ty_vk;

            quote! {
                pub(crate) #property_name: Vec<#pointed_ty_vk>,
            }
        })
    }

    fn to_default_constructor(&self) -> TokenStream {
        let Self { property_name, .. } = self;

        quote! {
            #property_name: Default::default(),
        }
    }

    fn to_properties_constructor(&self) -> TokenStream {
        let &Self {
            ref property_name,
            ref property_ty,
            is_extension,
            ..
        } = self;

        if is_extension {
            quote! {
                #property_name: None,
            }
        } else {
            quote! {
                #property_name: <#property_ty>::from_vulkan(#property_name).unwrap(),
            }
        }
    }

    fn to_extension_properties_constructor(&self) -> TokenStream {
        let Self {
            property_name,
            property_ty,
            pointer,
            ..
        } = self;

        let expr = if let Some(pointer) = pointer {
            let PointerProperty { len_name, .. } = pointer;

            quote! {
                &#property_name[..#len_name as usize]
            }
        } else {
            property_name.to_token_stream()
        };

        quote! {
            if val.#property_name.is_none() {
                val.#property_name = <#property_ty>::from_vulkan(#expr);
            }
        }
    }

    fn to_fields1_vk_constructor(&self) -> Option<TokenStream> {
        let Self {
            property_name,
            pointer,
            ..
        } = self;

        pointer.as_ref().map(|pointer| {
            let len_name = &pointer.len_name;

            quote! {
                #property_name: vec![
                    Default::default();
                    #len_name as usize
                ],
            }
        })
    }

    fn to_destructure(&self) -> TokenStream {
        let Self { property_name, .. } = self;

        quote! {
            #property_name,
        }
    }

    fn to_destructure_len(&self) -> Option<TokenStream> {
        let Self { pointer, .. } = self;

        pointer.as_ref().map(|pointer| {
            let len_name = &pointer.len_name;

            quote! {
                #len_name,
            }
        })
    }

    fn to_destructure_len_or_name(&self) -> TokenStream {
        let Self {
            property_name,
            pointer,
            ..
        } = self;

        let member = pointer
            .as_ref()
            .map_or(property_name, |pointer| &pointer.len_name);

        quote! {
            #member,
        }
    }

    fn to_builder(&self) -> TokenStream {
        let Self { property_name, .. } = self;

        quote! {
            .#property_name(#property_name)
        }
    }
}

fn is_extension_struct(ty: &Type) -> bool {
    ty.structextends.as_deref() == Some("VkPhysicalDeviceProperties2")
}

fn extension_structs_sorted<'a>(
    types: &'a HashMap<&str, (&'a Type, Vec<&'a str>)>,
) -> Vec<&'a (&'a Type, Vec<&'a str>)> {
    let mut extension_structs: Vec<_> = types
        .values()
        .filter(|(ty, _)| is_extension_struct(ty))
        .collect();

    fn is_vulkan_n_struct(name: &str) -> bool {
        tuple((
            tag::<_, &str, ()>("VkPhysicalDeviceVulkan"),
            digit1,
            tag("Properties"),
            eof,
        ))(name)
        .is_ok()
    }

    extension_structs.sort_unstable_by_key(|&(ty, provided_by)| {
        let name = ty.name.as_ref().unwrap();

        // Sort by groups:
        // - PhysicalDeviceVulkanNProperties, sorted ascending by Vulkan version
        // - Other core Vulkan properties structs, sorted ascending by Vulkan version
        // - _KHR extension structs
        // - _EXT extension structs
        // - Other extension structs
        let group_key = if is_vulkan_n_struct(name) {
            0
        } else if let Some(version) = provided_by
            .iter()
            .find_map(|s| s.strip_prefix("VK_VERSION_"))
        {
            let (major, minor) = version.split_once('_').unwrap();
            (major.parse::<i32>().unwrap() << 22) | (minor.parse::<i32>().unwrap() << 12)
        } else if provided_by.iter().any(|s| s.starts_with("VK_KHR_")) {
            i32::MAX - 2
        } else if provided_by.iter().any(|s| s.starts_with("VK_EXT_")) {
            i32::MAX - 1
        } else {
            i32::MAX
        };

        (group_key, name)
    });

    extension_structs
}

fn c_type_to_vk_type(ty: &str) -> TokenStream {
    match ty {
        "float" => quote! { f32 },
        "int32_t" => quote! { i32 },
        "int64_t" => quote! { i64 },
        "size_t" => quote! { usize },
        "uint8_t" => quote! { u8 },
        "uint32_t" => quote! { u32 },
        "uint64_t" => quote! { u64 },
        _ => {
            let ident = format_ident!(
                "{}",
                ty.strip_prefix("Vk")
                    .unwrap_or_else(|| unimplemented!("{}", ty))
            );
            quote! { ash::vk::#ident }
        }
    }
}

fn vulkano_type_basic(ty: &str) -> Option<TokenStream> {
    // TODO: make this more automatic?
    Some(match ty {
        "float" => quote! { f32 },
        "int32_t" => quote! { i32 },
        "int64_t" => quote! { i64 },
        "size_t" => quote! { usize },
        "uint8_t" => quote! { u8 },
        "uint32_t" => quote! { u32 },
        "uint64_t" => quote! { u64 },
        "VkBool32" => quote! { bool },
        "VkConformanceVersion" => quote! { ConformanceVersion },
        "VkDeviceSize" => quote! { DeviceSize },
        "VkDriverId" => quote! { DriverId },
        "VkExtent2D" => quote! { [u32; 2] },
        "VkMemoryDecompressionMethodFlagsNV" => quote! { MemoryDecompressionMethods },
        "VkOpticalFlowGridSizeFlagsNV" => quote! { OpticalFlowGridSizes },
        "VkPhysicalDeviceType" => quote! { PhysicalDeviceType },
        "VkPipelineRobustnessBufferBehavior" => quote! { PipelineRobustnessBufferBehavior },
        "VkPipelineRobustnessBufferBehaviorEXT" => quote! { PipelineRobustnessBufferBehavior },
        "VkPipelineRobustnessImageBehavior" => quote! { PipelineRobustnessImageBehavior },
        "VkPipelineRobustnessImageBehaviorEXT" => quote! { PipelineRobustnessImageBehavior },
        "VkPointClippingBehavior" => quote! { PointClippingBehavior },
        "VkQueueFlags" => quote! { QueueFlags },
        "VkRayTracingInvocationReorderModeNV" => quote! { RayTracingInvocationReorderMode },
        "VkResolveModeFlags" => quote! { ResolveModes },
        "VkSampleCountFlags" => quote! { SampleCounts },
        "VkSampleCountFlagBits" => quote! { SampleCount },
        "VkShaderCorePropertiesFlagsAMD" => quote! { ShaderCoreProperties },
        "VkShaderFloatControlsIndependence" => quote! { ShaderFloatControlsIndependence },
        "VkShaderStageFlags" => quote! { ShaderStages },
        "VkSubgroupFeatureFlags" => quote! { SubgroupFeatures },
        "VkImageLayout" => quote! { ImageLayout },
        "VkImageUsageFlags" => quote! { ImageUsage },
        "VkBufferUsageFlags" => quote! { BufferUsage },
        "VkChromaLocation" => quote! { ChromaLocation },
        "VkLayeredDriverUnderlyingApiMSFT" => quote! { LayeredDriverUnderlyingApi },
        "VkPhysicalDeviceSchedulingControlsFlagsARM" => {
            quote! { PhysicalDeviceSchedulingControlsFlags }
        }
        _ => return None,
    })
}
fn is_mappable_property_type(ty_c: &str) -> bool {
    // Types handled/skipped specially rather than mapped to a value type.
    if matches!(
        ty_c,
        "char"
            | "void"
            | "VkStructureType"
            | "VkPhysicalDeviceLimits"
            | "VkPhysicalDeviceSparseProperties"
    ) {
        return true;
    }
    vulkano_type_basic(ty_c).is_some()
}
fn member_is_unmappable(member: &TypeMember) -> bool {
    let TypeMember::Definition(definition) = member else {
        return false;
    };
    let Some(ty_c) = definition.markup.iter().find_map(|markup| match markup {
        TypeMemberMarkup::Type(ty) => Some(ty.as_str()),
        _ => None,
    }) else {
        return false;
    };
    !is_mappable_property_type(ty_c)
}
fn c_type_to_vulkano_type(
    name_c: &str,
    ty_c: &str,
    array_len: Option<&str>,
    has_len_field: bool,
) -> Option<TokenStream> {
    // Override the type for these specific properties.
    match name_c {
        "apiVersion" => Some(quote! { Version }),
        // The same property name appears in both DescriptorBuffer (size_t) and
        // DescriptorHeap (VkDeviceSize) extension structs; pin to one type so the
        // merged `DeviceProperties` field and every `from_vulkan` call agree.
        "samplerDescriptorSize" | "tensorDescriptorSize" => Some(quote! { usize }),
        "bufferImageGranularity"
        | "minStorageBufferOffsetAlignment"
        | "minTexelBufferOffsetAlignment"
        | "minUniformBufferOffsetAlignment"
        | "nonCoherentAtomSize"
        | "optimalBufferCopyOffsetAlignment"
        | "optimalBufferCopyRowPitchAlignment"
        | "robustStorageBufferAccessSizeAlignment"
        | "robustUniformBufferOffsetAlignment"
        | "storageTexelBufferOffsetAlignmentBytes"
        | "uniformTexelBufferOffsetAlignmentBytes"
        | "minPlacedMemoryMapAlignment" => Some(quote! { DeviceAlignment }),
        _ => {
            let inner = if ty_c == "char" && array_len.is_some() {
                quote! { String }
            } else {
                let element_ty = vulkano_type_basic(ty_c)?;
                if let Some(array_len) = array_len {
                    let array_len: usize = match array_len {
                        "VK_LUID_SIZE" => 8,
                        "VK_UUID_SIZE" => 16,
                        _ => array_len
                            .parse()
                            .unwrap_or_else(|_| unimplemented!("{}[{}]", ty_c, array_len)),
                    };
                    quote! { [#element_ty; #array_len] }
                } else {
                    element_ty
                }
            };
            if has_len_field {
                Some(quote! { Vec<#inner> })
            } else {
                Some(inner)
            }
        }
    }
}
