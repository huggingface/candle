use super::{write_file, IndexMap, VkRegistryData};
use foldhash::HashMap;
use heck::ToSnakeCase;
use nom::{bytes::complete::tag, character::complete::digit1, combinator::eof, sequence::tuple};
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote, ToTokens as _};
use std::{cmp::Ordering, collections::BTreeSet};
use vk_parse::{Extension, Type, TypeMember, TypeMemberMarkup, TypeSpec};

pub fn write(vk_data: &VkRegistryData<'_>) {
    let device_features = DeviceFeatures::new(&vk_data.types, &vk_data.extensions);
    let device_features_output = device_features.to_output();

    write_file(
        "features.rs",
        format!(
            "vk.xml header version {}.{}.{}",
            vk_data.header_version.0, vk_data.header_version.1, vk_data.header_version.2
        ),
        device_features_output,
    );
}

struct DeviceFeatures {
    features: Vec<Feature>,
    structs: Vec<FeaturesStruct>,
    extension_structs: Vec<FeaturesStruct>,
    extension_requires: Vec<ExtensionRequires>,
}

impl DeviceFeatures {
    fn new(
        types: &HashMap<&str, (&Type, Vec<&str>)>,
        extensions: &IndexMap<&str, &Extension>,
    ) -> Self {
        let extension_requires = get_extension_requires();
        let requires = get_requires();

        let mut structs: Vec<_> = ["VkPhysicalDeviceFeatures"]
            .into_iter()
            .filter_map(|struct_name_c| {
                let (struct_type_info, provided_by) = &types[struct_name_c];
                FeaturesStruct::new(
                    struct_type_info,
                    provided_by,
                    extensions,
                    &extension_requires,
                    &requires,
                )
            })
            .collect();

        let mut extension_structs: Vec<_> = extension_structs_sorted(types)
            .into_iter()
            .filter_map(|(struct_type_info, provided_by)| {
                FeaturesStruct::new(
                    struct_type_info,
                    provided_by,
                    extensions,
                    &extension_requires,
                    &requires,
                )
            })
            .collect();

        // Sort
        let mut features: HashMap<Ident, Feature> = HashMap::default();
        let mut feature_names = Vec::new();

        for features_struct in structs.iter_mut().chain(extension_structs.iter_mut()) {
            for feature in &features_struct.members {
                let feature = features
                    .entry(feature.feature_name.clone())
                    .or_insert_with_key(|key| {
                        feature_names.push(key.clone());
                        feature.clone()
                    });

                for requires in features_struct.provided_by.0.iter() {
                    // Avoid having two APIVersions
                    if let (
                        Requires::APIVersion(new_version),
                        Some(Requires::APIVersion(existing_version)),
                    ) = (requires, feature.provided_by.0.first())
                    {
                        if new_version < existing_version {
                            feature.provided_by.0.pop_first();
                            feature.provided_by.0.insert(requires.clone());
                        }
                    } else {
                        feature.provided_by.0.insert(requires.clone());
                    }
                }
            }
        }

        feature_names.sort_unstable();

        Self {
            features: feature_names
                .into_iter()
                .map(|name| features.remove(&name).unwrap())
                .collect(),
            structs,
            extension_structs,
            extension_requires,
        }
    }

    fn to_output(&self) -> TokenStream {
        let struct_definition = self.to_struct_definition();
        let helpers = self.to_helpers();
        let validate = self.to_validate();
        let enable_dependencies = self.to_enable_dependencies();
        let to_vk = self.to_vk();
        let to_vk2 = self.to_vk2();
        let traits = self.to_traits();
        let structs_vk = self.to_structs_vk();

        quote! {
            #struct_definition

            impl DeviceFeatures {
                #helpers
                #validate
                #enable_dependencies
                #to_vk
                #to_vk2
            }

            #traits
            #structs_vk
        }
    }

    fn to_struct_definition(&self) -> TokenStream {
        let Self { features, .. } = self;

        let iter = features.iter().map(Feature::to_features_struct_member);

        quote! {
            /// Represents all the features that are available on a physical device or enabled
            /// on a logical device.
            ///
            /// Depending on the highest version of Vulkan supported by the physical device, and
            /// the available extensions, not every feature may be available.
            #[derive(Copy, Clone, PartialEq, Eq, Hash)]
            #[repr(C)]
            pub struct DeviceFeatures {
                #(#iter)*
                pub _ne: crate::NonExhaustive<'static>,
            }
        }
    }

    fn to_helpers(&self) -> TokenStream {
        let Self { features, .. } = self;

        let len = features.len();
        let feature_name_c_iter = features.iter().map(|feature| &feature.feature_name_c);

        quote! {
            const COUNT: usize = #len;

            const NAMES_C: [&str; Self::COUNT] = [#(#feature_name_c_iter),*];

            /// Returns a `DeviceFeatures` with none of the members set.
            #[inline]
            pub const fn empty() -> Self {
                Self::from_array([false; Self::COUNT])
            }

            /// Returns a `DeviceFeatures` with all of the members set.
            #[cfg(test)]
            pub(crate) const fn all() -> DeviceFeatures {
                Self::from_array([true; Self::COUNT])
            }

            /// Returns whether any members are set in both `self` and `other`.
            #[inline]
            pub const fn intersects(&self, other: &Self) -> bool {
                crate::array_intersects(self.as_array(), other.as_array())
            }

            /// Returns whether all members in `other` are set in `self`.
            #[inline]
            pub const fn contains(&self, other: &Self) -> bool {
                crate::array_contains(self.as_array(), other.as_array())
            }

            /// Returns the union of `self` and `other`.
            #[inline]
            pub const fn union(&self, other: &Self) -> Self {
                Self::from_array(crate::array_union(self.as_array(), other.as_array()))
            }

            /// Returns the intersection of `self` and `other`.
            #[inline]
            pub const fn intersection(&self, other: &Self) -> Self {
                Self::from_array(crate::array_intersection(self.as_array(), other.as_array()))
            }

            /// Returns `self` without the members set in `other`.
            #[inline]
            pub const fn difference(&self, other: &Self) -> Self {
                Self::from_array(crate::array_difference(self.as_array(), other.as_array()))
            }

            /// Returns the members set in `self` or `other`, but not both.
            #[inline]
            pub const fn symmetric_difference(&self, other: &Self) -> Self {
                Self::from_array(
                    crate::array_symmetric_difference(self.as_array(), other.as_array()),
                )
            }

            #[inline]
            const fn from_array(array: [bool; Self::COUNT]) -> Self {
                // SAFETY: `DeviceFeatures` is nothing more than an array of `bool`s.
                unsafe { ::std::mem::transmute::<[bool; Self::COUNT], DeviceFeatures>(array) }
            }

            #[inline]
            const fn as_array(&self) -> &[bool; Self::COUNT] {
                // SAFETY: `DeviceFeatures` is nothing more than an array of `bool`s.
                unsafe { ::std::mem::transmute::<&DeviceFeatures, &[bool; Self::COUNT]>(self) }
            }

            #[inline]
            fn iter(&self) -> features::Iter<'_> {
                features::Iter {
                    inner: Self::NAMES_C.iter().copied().zip(self.as_array().iter().copied()),
                }
            }
        }
    }

    fn to_validate(&self) -> TokenStream {
        let Self { features, .. } = self;

        let iter = features.iter().map(Feature::to_validate);

        quote! {
            pub(crate) fn validate(
                &self,
                supported: &Self,
                api_version: Version,
                device_extensions: &DeviceExtensions,
            ) -> Result<(), Box<ValidationError>> {
                #(#iter)*
                Ok(())
            }
        }
    }

    fn to_enable_dependencies(&self) -> TokenStream {
        let Self {
            features,
            extension_requires,
            ..
        } = self;

        let extension_iter = extension_requires
            .iter()
            .map(ExtensionRequires::to_enable_dependencies);
        let feature_iter = features.iter().map(Feature::to_enable_dependencies);

        quote! {
            pub(crate) fn enable_dependencies(
                &self,
                api_version: Version,
                device_extensions: &DeviceExtensions,
            ) -> Self {
                let mut enabled = *self;
                #(#extension_iter)*
                #(#feature_iter)*
                enabled
            }
        }
    }

    fn to_vk(&self) -> TokenStream {
        let Self {
            features, structs, ..
        } = self;

        let from_vk_vulkano_members_iter = features.iter().map(Feature::to_vulkano_constructor);
        let from_vk_structs_iter = structs.iter().map(FeaturesStruct::to_destructure_members);
        let to_vk_iter = features
            .iter()
            .filter(Feature::is_not_extension)
            .map(Feature::to_builder);

        quote! {
            pub(crate) fn to_mut_vk() -> vk::PhysicalDeviceFeatures {
                vk::PhysicalDeviceFeatures::default()
            }

            pub(crate) fn from_vk(
                val_vk: &vk::PhysicalDeviceFeatures,
            ) -> Self {
                #(#from_vk_structs_iter)*

                Self {
                    #(#from_vk_vulkano_members_iter)*
                    _ne: crate::NE,
                }
            }

            #[allow(clippy::wrong_self_convention)]
            pub(crate) fn to_vk(&self) -> vk::PhysicalDeviceFeatures {
                vk::PhysicalDeviceFeatures::default()
                    #(#to_vk_iter)*
            }
        }
    }

    fn to_vk2(&self) -> TokenStream {
        let Self {
            features,
            structs,
            extension_structs,
            ..
        } = self;

        let to_mut_vk2_iter = extension_structs
            .iter()
            .map(FeaturesStruct::to_extensions_vk_push_next);
        let to_mut_vk2_extensions_iter = extension_structs
            .iter()
            .map(FeaturesStruct::to_extensions_vk_constructor_default);

        let from_vk2_destructure_extensions_iter =
            extension_structs.iter().map(FeaturesStruct::to_destructure);
        let from_vk2_structs_iter = structs.iter().map(FeaturesStruct::to_destructure_members);
        let from_vk2_vulkano_members_iter = features.iter().map(Feature::to_vulkano_constructor);
        let from_vk2_extension_structs_iter = extension_structs
            .iter()
            .map(FeaturesStruct::to_vulkano_from_extension);

        let to_vk2_builder_iter = features
            .iter()
            .filter(Feature::is_not_extension)
            .map(Feature::to_builder);
        let to_vk2_extensions_iter = extension_structs
            .iter()
            .map(FeaturesStruct::to_extensions_vk_push_next);

        let to_vk2_extensions_constructor_iter = extension_structs
            .iter()
            .map(FeaturesStruct::to_extensions_vk_constructor);

        quote! {
            pub(crate) fn to_mut_vk2(
                extensions_vk: &mut PhysicalDeviceFeatures2ExtensionsVk,
            ) -> vk::PhysicalDeviceFeatures2<'_> {
                let mut val_vk = vk::PhysicalDeviceFeatures2::default();

                #(#to_mut_vk2_iter)*

                val_vk
            }

            pub(crate) fn to_mut_vk2_extensions(
                api_version: Version,
                device_extensions: &DeviceExtensions,
                #[allow(unused)]
                instance_extensions: &InstanceExtensions,
            ) -> PhysicalDeviceFeatures2ExtensionsVk {
                PhysicalDeviceFeatures2ExtensionsVk {
                    #(#to_mut_vk2_extensions_iter)*
                }
            }

            pub(crate) fn from_vk2(
                val_vk: &vk::PhysicalDeviceFeatures2<'_>,
                extensions_vk: &PhysicalDeviceFeatures2ExtensionsVk,
            ) -> Self {
                let vk::PhysicalDeviceFeatures2 {
                    features: val_vk,
                    ..
                } = val_vk;
                let PhysicalDeviceFeatures2ExtensionsVk {
                    #(#from_vk2_destructure_extensions_iter)*
                } = extensions_vk;

                #(#from_vk2_structs_iter)*

                let mut val = Self {
                    #(#from_vk2_vulkano_members_iter)*
                    _ne: crate::NE,
                };

                #(#from_vk2_extension_structs_iter)*

                val
            }

            #[allow(clippy::wrong_self_convention)]
            pub(crate) fn to_vk2<'a>(
                &self,
                extensions_vk: &'a mut PhysicalDeviceFeatures2ExtensionsVk,
            ) -> vk::PhysicalDeviceFeatures2<'a> {
                let mut val_vk = vk::PhysicalDeviceFeatures2::default().features(
                    vk::PhysicalDeviceFeatures::default()
                        #(#to_vk2_builder_iter)*
                );

                #(#to_vk2_extensions_iter)*

                val_vk
            }

            #[allow(clippy::wrong_self_convention)]
            pub(crate) fn to_vk2_extensions(
                &self,
                api_version: Version,
                device_extensions: &DeviceExtensions,
                #[allow(unused)]
                instance_extensions: &InstanceExtensions,
            ) -> PhysicalDeviceFeatures2ExtensionsVk {
                PhysicalDeviceFeatures2ExtensionsVk {
                    #(#to_vk2_extensions_constructor_iter)*
                }
            }
        }
    }

    fn to_traits(&self) -> TokenStream {
        quote! {
            impl std::fmt::Debug for DeviceFeatures {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
                    f.debug_list()
                        .entries(self.iter().flat_map(|(feature_name_c, enabled)| {
                            enabled.then_some(feature_name_c)
                        }))
                        .finish()
                }
            }

            impl Default for DeviceFeatures {
                #[inline]
                fn default() -> Self {
                    Self::empty()
                }
            }

            impl<'a> IntoIterator for &'a DeviceFeatures {
                type Item = (&'static str, bool);
                type IntoIter = features::Iter<'a>;

                #[inline]
                fn into_iter(self) -> Self::IntoIter {
                    self.iter()
                }
            }

            mod features {
                pub struct Iter<'a> {
                    pub(crate) inner: std::iter::Zip<
                        std::iter::Copied<std::slice::Iter<'a, &'static str>>,
                        std::iter::Copied<std::slice::Iter<'a, bool>>,
                    >,
                }

                impl<'a> Iterator for Iter<'a> {
                    type Item = (&'static str, bool);

                    #[inline]
                    fn next(&mut self) -> Option<Self::Item> {
                        self.inner.next()
                    }

                    #[inline]
                    fn size_hint(&self) -> (usize, Option<usize>) {
                        self.inner.size_hint()
                    }

                    #[inline]
                    fn nth(&mut self, n: usize) -> Option<Self::Item> {
                        self.inner.nth(n)
                    }
                }

                impl<'a> DoubleEndedIterator for Iter<'a> {
                    #[inline]
                    fn next_back(&mut self) -> Option<Self::Item> {
                        self.inner.next_back()
                    }
                }

                impl ExactSizeIterator for Iter<'_> {
                    #[inline]
                    fn len(&self) -> usize {
                        self.inner.len()
                    }
                }

                impl std::iter::FusedIterator for Iter<'_> {}
            }

            impl std::ops::BitAnd for DeviceFeatures {
                type Output = DeviceFeatures;

                #[inline]
                fn bitand(self, rhs: Self) -> Self::Output {
                    self.intersection(&rhs)
                }
            }

            impl std::ops::BitAndAssign for DeviceFeatures {
                #[inline]
                fn bitand_assign(&mut self, rhs: Self) {
                    *self = self.intersection(&rhs);
                }
            }

            impl std::ops::BitOr for DeviceFeatures {
                type Output = DeviceFeatures;

                #[inline]
                fn bitor(self, rhs: Self) -> Self::Output {
                    self.union(&rhs)
                }
            }

            impl std::ops::BitOrAssign for DeviceFeatures {
                #[inline]
                fn bitor_assign(&mut self, rhs: Self) {
                    *self = self.union(&rhs);
                }
            }

            impl std::ops::BitXor for DeviceFeatures {
                type Output = DeviceFeatures;

                #[inline]
                fn bitxor(self, rhs: Self) -> Self::Output {
                    self.symmetric_difference(&rhs)
                }
            }

            impl std::ops::BitXorAssign for DeviceFeatures {
                #[inline]
                fn bitxor_assign(&mut self, rhs: Self) {
                    *self = self.symmetric_difference(&rhs);
                }
            }

            impl std::ops::Sub for DeviceFeatures {
                type Output = DeviceFeatures;

                #[inline]
                fn sub(self, rhs: Self) -> Self::Output {
                    self.difference(&rhs)
                }
            }

            impl std::ops::SubAssign for DeviceFeatures {
                #[inline]
                fn sub_assign(&mut self, rhs: Self) {
                    *self = self.difference(&rhs);
                }
            }
        }
    }

    fn to_structs_vk(&self) -> TokenStream {
        let Self {
            extension_structs, ..
        } = self;

        let iter = extension_structs
            .iter()
            .map(FeaturesStruct::to_extensions_vk_struct_member);

        quote! {
            pub(crate) struct PhysicalDeviceFeatures2ExtensionsVk {
                #(#iter)*
            }
        }
    }
}

struct FeaturesStruct {
    struct_name: Ident,
    var_name: Ident,
    var_ty: TokenStream,
    is_extension: bool,
    provided_by: RequiresOneOf,
    members: Vec<Feature>,
}

impl FeaturesStruct {
    fn new(
        struct_type_info: &Type,
        provided_by: &[&str],
        extensions: &IndexMap<&str, &Extension>,
        extension_requires: &[ExtensionRequires],
        requires: &HashMap<&'static str, Vec<Ident>>,
    ) -> Option<Self> {
        let TypeSpec::Members(members) = &struct_type_info.spec else {
            return None;
        };

        let is_extension = is_extension(struct_type_info);
        let struct_name_c = struct_type_info.name.as_ref().unwrap();
        let struct_name = struct_name_c.strip_prefix("Vk").unwrap();

        let members = members
            .iter()
            .filter_map(|member| {
                Feature::new(
                    member,
                    struct_name_c,
                    is_extension,
                    extension_requires,
                    requires,
                )
            })
            .collect();

        Some(FeaturesStruct {
            struct_name: format_ident!("{}", struct_name),
            var_name: match struct_name {
                "PhysicalDeviceProperties" | "PhysicalDeviceFeatures" => format_ident!("val_vk"),
                "PhysicalDeviceLimits" => format_ident!("limits"),
                "PhysicalDeviceSparseProperties" => format_ident!("sparse_properties"),
                _ => {
                    let snake_case = struct_name
                        .strip_prefix("PhysicalDevice")
                        .unwrap()
                        .to_snake_case();

                    let (base, suffix) = snake_case.rsplit_once("_features").unwrap();
                    format_ident!("features_{}{}_vk", base, suffix)
                }
            },
            var_ty: format_ident!("{}", struct_name).to_token_stream(),
            is_extension,
            provided_by: RequiresOneOf(
                provided_by
                    .iter()
                    .map(|&provided_by| Requires::from_provided_by(provided_by, extensions))
                    .filter(|requires| !requires.is_vulkan_1_0())
                    .collect(),
            ),
            members,
        })
    }

    fn to_destructure(&self) -> TokenStream {
        let Self { var_name, .. } = self;

        quote! { #var_name, }
    }

    fn to_destructure_members(&self) -> TokenStream {
        let &Self {
            ref struct_name,
            ref var_name,
            is_extension,
            ref members,
            ..
        } = self;

        let destructure_iter = members.iter().map(Feature::to_destructure);

        if is_extension {
            quote! {
                let &vk::#struct_name {
                    #(#destructure_iter)*
                    ..
                } = val_vk;
            }
        } else {
            quote! {
                let &vk::#struct_name {
                    #(#destructure_iter)*
                    ..
                } = #var_name;
            }
        }
    }

    fn to_vulkano_from_extension(&self) -> TokenStream {
        let Self {
            var_name, members, ..
        } = self;

        let destructure = self.to_destructure_members();
        let constructor_iter = members.iter().map(Feature::to_vulkano_from_extension);

        quote! {
            if let Some(val_vk) = #var_name {
                #destructure

                #(#constructor_iter)*
            }
        }
    }

    fn to_extensions_vk_struct_member(&self) -> TokenStream {
        let Self {
            var_name, var_ty, ..
        } = self;

        quote! {
            pub(crate) #var_name: Option<vk::#var_ty<'static>>,
        }
    }

    fn to_extensions_vk_constructor(&self) -> TokenStream {
        let Self {
            var_name,
            var_ty,
            provided_by,
            members,
            ..
        } = self;

        let builder_iter = members.iter().map(Feature::to_builder);
        let validate_expr = provided_by.to_validate_expr();

        quote! {
            #var_name: (#validate_expr).then(|| {
                <vk::#var_ty<'_>>::default()
                    #(#builder_iter)*
            }),
        }
    }

    fn to_extensions_vk_constructor_default(&self) -> TokenStream {
        let Self {
            var_name,
            var_ty,
            provided_by,
            ..
        } = self;

        let validate_expr = provided_by.to_validate_expr();

        quote! {
            #var_name: (#validate_expr).then(<vk::#var_ty<'_>>::default),
        }
    }

    fn to_extensions_vk_push_next(&self) -> TokenStream {
        let Self { var_name, .. } = self;

        quote! {
            if let Some(next) = &mut extensions_vk.#var_name {
                val_vk = val_vk.push(next);
            }
        }
    }
}

#[derive(Clone)]
struct Feature {
    feature_name: Ident,
    feature_name_c: String,
    struct_name_c: String,
    is_extension: bool,
    provided_by: RequiresOneOf,
    required_by_extensions: Vec<Ident>,
    requires: Vec<Ident>,
}

impl Feature {
    fn new(
        member: &TypeMember,
        struct_name_c: &str,
        is_extension: bool,
        extension_requires: &[ExtensionRequires],
        requires: &HashMap<&'static str, Vec<Ident>>,
    ) -> Option<Self> {
        let TypeMember::Definition(definition) = member else {
            return None;
        };

        let feature_name_c = definition.markup.iter().find_map(|markup| match markup {
            TypeMemberMarkup::Name(name) => Some(name.as_str()),
            _ => None,
        })?;

        if matches!(feature_name_c, "sType" | "pNext") {
            return None;
        }

        let feature_name = feature_name_c.to_snake_case();
        let required_by_extensions = extension_requires
            .iter()
            .filter(|&extension_requires| {
                extension_requires
                    .features
                    .iter()
                    .any(|feature| feature == feature_name.as_str())
            })
            .map(|extension_requires| extension_requires.extension.clone())
            .collect();
        let requires = requires
            .get(feature_name.as_str())
            .cloned()
            .unwrap_or_default();

        Some(Feature {
            feature_name: format_ident!("{}", feature_name),
            feature_name_c: feature_name_c.to_owned(),
            struct_name_c: struct_name_c.to_owned(),
            is_extension,
            provided_by: Default::default(),
            required_by_extensions,
            requires,
        })
    }

    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn is_not_extension(self: &&Self) -> bool {
        !self.is_extension
    }

    fn to_features_struct_member(&self) -> TokenStream {
        let Self {
            feature_name,
            feature_name_c,
            struct_name_c,
            provided_by,
            required_by_extensions,
            requires,
            ..
        } = self;

        let doc = {
            let doc = format!(
                "- [Vulkan documentation](https://registry.khronos.org/vulkan/specs/1.3-extensions\
                /man/html/{}.html#features-{})",
                struct_name_c, feature_name_c,
            );
            quote! {
                #[doc = #doc]
            }
        };

        let provided_by = (!provided_by.0.is_empty()).then(|| {
            let iter = provided_by.0.iter().map(|requires| {
                let doc = match requires {
                    Requires::APIVersion(version) => {
                        format!("  - Vulkan API version {}.{}", version.major, version.minor)
                    }
                    Requires::DeviceExtension(extension_name) => {
                        format!(
                            "  - Device extension \
                            [`{}`](crate::device::DeviceExtensions::{0})",
                            extension_name.vulkano
                        )
                    }
                    Requires::InstanceExtension(extension_name) => {
                        format!(
                            "  - Instance extension \
                            [`{}`](crate::instance::InstanceExtensions::{0})",
                            extension_name.vulkano
                        )
                    }
                };
                quote! { #[doc = #doc] }
            });

            quote! {
                #[doc = "- Requires one of:"]
                #(#iter)*
            }
        });

        let required_by_extensions = (!required_by_extensions.is_empty()).then(|| {
            let links: Vec<_> = required_by_extensions
                .iter()
                .map(|extension| format!("[`{}`](crate::device::DeviceExtensions::{0})", extension))
                .collect();
            let doc = format!(
                "- Automatically enabled by device extension{}: {}",
                if required_by_extensions.len() > 1 {
                    "s"
                } else {
                    ""
                },
                links.join(", ")
            );
            quote! {
                #[doc = #doc]
            }
        });

        let requires = (!requires.is_empty()).then(|| {
            let links: Vec<_> = requires
                .iter()
                .map(|feature| format!("[`{}`](Self::{0})", feature))
                .collect();
            let doc = format!(
                "- Requires feature{}: {}",
                if requires.len() > 1 { "s" } else { "" },
                links.join(", ")
            );
            quote! {
                #[doc = #doc]
            }
        });

        quote! {
            #doc
            #provided_by
            #required_by_extensions
            #requires
            pub #feature_name: bool,
        }
    }

    fn to_destructure(&self) -> TokenStream {
        let Self { feature_name, .. } = self;

        quote! {
            #feature_name,
        }
    }

    fn to_builder(&self) -> TokenStream {
        let Self { feature_name, .. } = self;

        quote! {
            .#feature_name(self.#feature_name)
        }
    }

    fn to_validate(&self) -> TokenStream {
        let Self {
            feature_name,
            provided_by,
            ..
        } = self;

        let validate_supported = {
            let problem = format!(
                "contains `{}`, but this feature is not supported \
                by the physical device",
                feature_name,
            );
            quote! {
                if !supported.#feature_name {
                    return Err(Box::new(crate::ValidationError {
                        problem: #problem.into(),
                        ..Default::default()
                    }));
                }
            }
        };

        let validate_provided_by = (!provided_by.is_empty()).then(|| {
            let validate_expr = provided_by.to_validate_expr();
            let problem = format!("contains `{}`", feature_name);
            let requires_one_of = provided_by.to_constructor();

            quote! {
                if !(#validate_expr) {
                    return Err(Box::new(crate::ValidationError {
                        problem: #problem.into(),
                        requires_one_of: #requires_one_of,
                        ..Default::default()
                    }));
                }
            }
        });

        quote! {
            if self.#feature_name {
                #validate_supported
                #validate_provided_by
            }
        }
    }

    fn to_enable_dependencies(&self) -> Option<TokenStream> {
        let Self {
            feature_name,
            requires,
            ..
        } = self;

        (!requires.is_empty()).then(|| {
            let iter = requires.iter().map(|required_feature| {
                quote! {
                    enabled.#required_feature = true;
                }
            });

            quote! {
                if self.#feature_name {
                    #(#iter)*
                }
            }
        })
    }

    fn to_vulkano_constructor(&self) -> TokenStream {
        let &Self {
            ref feature_name,
            is_extension,
            ..
        } = self;

        if is_extension {
            quote! {
                #feature_name: false,
            }
        } else {
            quote! {
                #feature_name: #feature_name != 0,
            }
        }
    }

    fn to_vulkano_from_extension(&self) -> TokenStream {
        let Self {
            feature_name: item_name,
            ..
        } = self;

        quote! {
            val.#item_name |= #item_name != 0;
        }
    }
}

fn is_extension(ty: &Type) -> bool {
    matches!(
        ty.structextends.as_deref(),
        Some("VkPhysicalDeviceFeatures2,VkDeviceCreateInfo")
    )
}

fn extension_structs_sorted<'a>(
    types: &'a HashMap<&str, (&'a Type, Vec<&'a str>)>,
) -> Vec<&'a (&'a Type, Vec<&'a str>)> {
    let mut extension_structs: Vec<_> = types.values().filter(|(ty, _)| is_extension(ty)).collect();

    let is_vulkan_n_struct = |name: &str| -> bool {
        tuple((
            tag::<_, &str, ()>("VkPhysicalDeviceVulkan"),
            digit1,
            tag("Features"),
            eof,
        ))(name)
        .is_ok()
    };

    extension_structs.sort_unstable_by_key(|&(ty, provided_by)| {
        let name = ty.name.as_ref().unwrap();

        // Sort by groups:
        // - PhysicalDeviceVulkanN*, sorted ascending by Vulkan version
        // - Other core Vulkan structs, sorted ascending by Vulkan version
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

#[derive(Clone, Default)]
struct RequiresOneOf(BTreeSet<Requires>);

impl RequiresOneOf {
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn to_constructor(&self) -> TokenStream {
        let requires_one_of = &self.0;
        let iter = requires_one_of
            .iter()
            .map(Requires::to_constructor)
            .map(|requires| {
                quote! {
                    RequiresAllOf(&[#requires])
                }
            });

        quote! {
            RequiresOneOf(&[
                #(#iter),*
            ])
        }
    }

    fn to_validate_expr(&self) -> TokenStream {
        let requires_one_of = &self.0;
        let iter = requires_one_of.iter().map(Requires::to_validate_expr);

        quote! {
            #(#iter)||*
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum Requires {
    APIVersion(Version),
    DeviceExtension(ExtensionName),
    InstanceExtension(ExtensionName),
}

impl Requires {
    fn from_provided_by(provided_by: &str, extensions: &IndexMap<&str, &Extension>) -> Self {
        let version = provided_by
            .strip_prefix("VK_VERSION_")
            .or_else(|| provided_by.strip_prefix("VK_BASE_VERSION_"))
            .or_else(|| provided_by.strip_prefix("VK_COMPUTE_VERSION_"))
            .or_else(|| provided_by.strip_prefix("VK_GRAPHICS_VERSION_"));
        if let Some(version) = version {
            let (major, minor) = version.split_once('_').unwrap();
            Self::APIVersion(Version {
                major: major.parse().unwrap(),
                minor: minor.parse().unwrap(),
            })
        } else {
            match extensions[provided_by].ext_type.as_deref() {
                Some("device") => Self::DeviceExtension(ExtensionName::new(provided_by)),
                Some("instance") => Self::InstanceExtension(ExtensionName::new(provided_by)),
                _ => unimplemented!(),
            }
        }
    }

    fn is_vulkan_1_0(&self) -> bool {
        matches!(self, Requires::APIVersion(Version { major: 1, minor: 0 }))
    }

    fn to_constructor(&self) -> TokenStream {
        match self {
            Self::APIVersion(version) => {
                let version = version.to_ident();
                quote! { Requires::APIVersion(Version::#version) }
            }
            Self::DeviceExtension(extension) => {
                let name = extension.vulkano.to_string();
                quote! { Requires::DeviceExtension(#name) }
            }
            Self::InstanceExtension(extension) => {
                let name = extension.vulkano.to_string();
                quote! { Requires::InstanceExtension(#name) }
            }
        }
    }

    fn to_validate_expr(&self) -> TokenStream {
        match self {
            Self::APIVersion(version) => {
                let version = version.to_ident();
                quote! { api_version >= Version::#version }
            }
            Self::DeviceExtension(extension) => {
                let ident = &extension.vulkano;
                quote! { device_extensions.#ident }
            }
            Self::InstanceExtension(extension) => {
                let ident = &extension.vulkano;
                quote! { instance_extensions.#ident }
            }
        }
    }
}

impl Ord for Requires {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::APIVersion(self_version), Self::APIVersion(other_version)) => {
                self_version.cmp(other_version)
            }
            (Self::APIVersion(_), _) => Ordering::Less,
            (_, Self::APIVersion(_)) => Ordering::Greater,
            (
                Self::DeviceExtension(self_extension) | Self::InstanceExtension(self_extension),
                Self::DeviceExtension(other_extension) | Self::InstanceExtension(other_extension),
            ) => self_extension.cmp(other_extension),
        }
    }
}

impl PartialOrd for Requires {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Version {
    major: u16,
    minor: u16,
}

impl Version {
    fn to_ident(self) -> Ident {
        let Self { major, minor, .. } = self;
        format_ident!("V{}_{}", major, minor)
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct ExtensionName {
    vulkano: Ident,
    vk: String,
    vendor_index: usize,
}

impl ExtensionName {
    fn new(vk: &str) -> Self {
        let vk = if vk.starts_with("VK_") {
            vk.to_owned()
        } else {
            format!("VK_{}", vk)
        };

        let (vendor, _rest) = vk["VK_".len()..].split_once('_').unwrap();
        let vendor_index = match vendor {
            "KHR" => 0,
            "EXT" => 1,
            _ => usize::MAX,
        };

        Self {
            vulkano: format_ident!("{}", vk["VK_".len()..].to_ascii_lowercase()),
            vk,
            vendor_index,
        }
    }
}

impl Ord for ExtensionName {
    fn cmp(&self, other: &Self) -> Ordering {
        self.vendor_index
            .cmp(&other.vendor_index)
            .then_with(|| self.vk["VK_".len()..].cmp(&other.vk["VK_".len()..]))
    }
}

impl PartialOrd for ExtensionName {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct ExtensionRequires {
    api_version: Version,
    extension: Ident,
    features: Vec<Ident>,
}

impl ExtensionRequires {
    fn to_enable_dependencies(&self) -> TokenStream {
        let Self {
            api_version,
            extension,
            features,
        } = self;

        let api_version = api_version.to_ident();
        let iter = features.iter().map(|required_feature| {
            quote! {
                enabled.#required_feature = true;
            }
        });

        quote! {
            if api_version >= Version::#api_version && device_extensions.#extension {
                #(#iter)*
            }
        }
    }
}

fn get_extension_requires() -> Vec<ExtensionRequires> {
    vec![
        // VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-04476
        ExtensionRequires {
            api_version: Version { major: 1, minor: 1 },
            extension: format_ident!("khr_shader_draw_parameters"),
            features: vec![format_ident!("shader_draw_parameters")],
        },
        // VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-02831
        ExtensionRequires {
            api_version: Version { major: 1, minor: 2 },
            extension: format_ident!("khr_draw_indirect_count"),
            features: vec![format_ident!("draw_indirect_count")],
        },
        // VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-02832
        ExtensionRequires {
            api_version: Version { major: 1, minor: 2 },
            extension: format_ident!("khr_sampler_mirror_clamp_to_edge"),
            features: vec![format_ident!("sampler_mirror_clamp_to_edge")],
        },
        // VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-02833
        ExtensionRequires {
            api_version: Version { major: 1, minor: 2 },
            extension: format_ident!("ext_descriptor_indexing"),
            features: vec![format_ident!("descriptor_indexing")],
        },
        // VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-02834
        ExtensionRequires {
            api_version: Version { major: 1, minor: 2 },
            extension: format_ident!("ext_sampler_filter_minmax"),
            features: vec![format_ident!("sampler_filter_minmax")],
        },
        // VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-02835
        ExtensionRequires {
            api_version: Version { major: 1, minor: 2 },
            extension: format_ident!("ext_shader_viewport_index_layer"),
            features: vec![
                format_ident!("shader_output_viewport_index"),
                format_ident!("shader_output_layer"),
            ],
        },
    ]
}

fn get_requires() -> HashMap<&'static str, Vec<Ident>> {
    HashMap::from_iter([
        // VUID-VkPhysicalDeviceVariablePointersFeatures-variablePointers-01431
        (
            "variable_pointers",
            vec![format_ident!("variable_pointers_storage_buffer")],
        ),
        // VUID-VkPhysicalDeviceMultiviewFeatures-multiviewGeometryShader-00580
        (
            "multiview_geometry_shader",
            vec![format_ident!("multiview")],
        ),
        // VUID-VkPhysicalDeviceMultiviewFeatures-multiviewTessellationShader-00581
        (
            "multiview_tessellation_shader",
            vec![format_ident!("multiview")],
        ),
        // VUID-VkPhysicalDeviceMeshShaderFeaturesEXT-multiviewMeshShader-07032
        ("multiview_mesh_shader", vec![format_ident!("multiview")]),
        // VUID-VkPhysicalDeviceMeshShaderFeaturesEXT-primitiveFragmentShadingRateMeshShader-07033
        (
            "primitive_fragment_shading_rate_mesh_shader",
            vec![format_ident!("primitive_fragment_shading_rate")],
        ),
        // VUID-VkPhysicalDeviceRayTracingPipelineFeaturesKHR-rayTracingPipelineShaderGroupHandleCaptureReplayMixed-03575
        (
            "ray_tracing_pipeline_shader_group_handle_capture_replay_mixed",
            vec![format_ident!(
                "ray_tracing_pipeline_shader_group_handle_capture_replay"
            )],
        ),
        // VUID-VkPhysicalDeviceRobustness2FeaturesEXT-robustBufferAccess2-04000
        (
            "robust_buffer_access2",
            vec![format_ident!("robust_buffer_access")],
        ),
        // VUID-VkDeviceCreateInfo-None-04896
        (
            "sparse_image_int64_atomics",
            vec![format_ident!("shader_image_int64_atomics")],
        ),
        // VUID-VkDeviceCreateInfo-None-04897
        (
            "sparse_image_float32_atomics",
            vec![format_ident!("shader_image_float32_atomics")],
        ),
        // VUID-VkDeviceCreateInfo-None-04898
        (
            "sparse_image_float32_atomic_add",
            vec![format_ident!("shader_image_float32_atomic_add")],
        ),
        // VUID-VkDeviceCreateInfo-sparseImageFloat32AtomicMinMax-04975
        (
            "sparse_image_float32_atomic_min_max",
            vec![format_ident!("shader_image_float32_atomic_min_max")],
        ),
    ])
}
