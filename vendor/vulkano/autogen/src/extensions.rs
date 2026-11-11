use super::{write_file, IndexMap, RequiresOneOf, VkRegistryData};
use crate::conjunctive_normal_form::ConjunctiveNormalForm;
use heck::ToSnakeCase;
use nom::{
    branch::alt, bytes::complete::take_while1, character::complete, combinator::all_consuming,
    multi::separated_list0, sequence::delimited, IResult, Parser,
};
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};
use std::ffi::CString;
use vk_parse::Extension as VkExtension;

pub fn write(vk_data: &VkRegistryData<'_>) {
    let instance_extensions = Extensions::new(ExtensionType::Instance, &vk_data.extensions);
    let device_extensions = Extensions::new(ExtensionType::Device, &vk_data.extensions);

    write_file(
        "instance_extensions.rs",
        format!(
            "vk.xml header version {}.{}.{}",
            vk_data.header_version.0, vk_data.header_version.1, vk_data.header_version.2
        ),
        instance_extensions.to_items(),
    );

    write_file(
        "device_extensions.rs",
        format!(
            "vk.xml header version {}.{}.{}",
            vk_data.header_version.0, vk_data.header_version.1, vk_data.header_version.2
        ),
        device_extensions.to_items(),
    );
}

struct Extensions {
    extension_ty: ExtensionType,
    extensions: Vec<Extension>,
    required_if_supported: Vec<Ident>,
}

impl Extensions {
    fn new(extension_ty: ExtensionType, vk_extensions: &IndexMap<&str, &VkExtension>) -> Self {
        let required_if_supported = get_required_if_supported(extension_ty);

        let extensions = vk_extensions
            .values()
            .filter(|vk_extension| {
                vk_extension.ext_type.as_deref() == Some(extension_ty.as_str_lowercase())
            })
            .map(|vk_extension| Extension::new(vk_extension, vk_extensions, &required_if_supported))
            .collect();

        Self {
            extension_ty,
            extensions,
            required_if_supported,
        }
    }

    fn to_items(&self) -> TokenStream {
        let &Self { extension_ty, .. } = self;

        let struct_name = extension_ty.struct_name();
        let struct_definition = self.to_struct_definition();
        let helpers = self.to_helpers();
        let validate = self.to_validate();
        let enable_dependencies = self.to_enable_dependencies();
        let to_vk = self.to_vk();
        let traits = self.to_traits();

        quote! {
            #struct_definition

            impl #struct_name {
                #helpers
                #validate
                #enable_dependencies
                #to_vk
            }

            #traits
        }
    }

    fn to_struct_definition(&self) -> TokenStream {
        let &Self {
            extension_ty,
            ref extensions,
            ..
        } = self;

        let doc = format!(
            "List of {} extensions that are enabled or available.",
            extension_ty.as_str_lowercase()
        );
        let struct_name = extension_ty.struct_name();
        let iter = extensions.iter().map(Extension::to_struct_member);

        quote! {
            #[doc = #doc]
            #[derive(Copy, Clone, PartialEq, Eq)]
            #[repr(C)]
            pub struct #struct_name {
                #(#iter)*
                pub _ne: crate::NonExhaustive<'static>,
            }
        }
    }

    fn to_helpers(&self) -> TokenStream {
        let &Self {
            extension_ty,
            ref extensions,
            ..
        } = self;

        let struct_name = extension_ty.struct_name();
        let empty_doc = format!(
            "Returns a `{}` with none of the members set.",
            extension_ty.struct_name()
        );
        let len = extensions.len();
        let extension_name_c_iter = extensions
            .iter()
            .map(|extension| CString::new(extension.extension_name_c.as_str()).unwrap());

        quote! {
            const COUNT: usize = #len;

            const NAMES_C: [&std::ffi::CStr; Self::COUNT] = [#(#extension_name_c_iter),*];

            #[doc = #empty_doc]
            #[inline]
            pub const fn empty() -> Self {
                Self::from_array([false; Self::COUNT])
            }

            /// Returns the number of members set in self.
            #[inline]
            pub const fn count(self) -> u64 {
                crate::array_count(self.as_array()) as u64
            }

            /// Returns whether no members are set in `self`.
            #[inline]
            pub const fn is_empty(self) -> bool {
                crate::array_is_empty(self.as_array())
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
                // SAFETY: `DeviceExtensions` / `InstanceExtensions` are nothing more than an array
                // of `bool`s.
                unsafe { ::std::mem::transmute::<[bool; Self::COUNT], #struct_name>(array) }
            }

            #[inline]
            const fn as_array(&self) -> &[bool; Self::COUNT] {
                // SAFETY: `DeviceExtensions` / `InstanceExtensions` are nothing more than an array
                // of `bool`s.
                unsafe { ::std::mem::transmute::<&#struct_name, &[bool; Self::COUNT]>(self) }
            }

            #[inline]
            const fn as_mut_array(&mut self) -> &mut [bool; Self::COUNT] {
                // SAFETY: `DeviceExtensions` / `InstanceExtensions` are nothing more than an array
                // of `bool`s.
                unsafe {
                    ::std::mem::transmute::<&mut #struct_name, &mut [bool; Self::COUNT]>(self)
                }
            }

            #[inline]
            fn iter(&self) -> extensions::Iter<'_> {
                extensions::Iter {
                    inner: Self::NAMES_C.iter().copied().zip(self.as_array().iter().copied()),
                }
            }
        }
    }

    fn to_validate(&self) -> TokenStream {
        let &Self {
            extension_ty,
            ref extensions,
            ..
        } = self;

        let params = match extension_ty {
            ExtensionType::Instance => {
                quote! {}
            }
            ExtensionType::Device => {
                quote! { instance_extensions: &InstanceExtensions, }
            }
        };
        let iter = extensions.iter().map(|m| m.to_validate(extension_ty));

        quote! {
            pub(crate) fn validate(
                &self,
                supported: &Self,
                api_version: Version,
                #params
            ) -> Result<(), Box<ValidationError>> {
                #(#iter)*
                Ok(())
            }
        }
    }

    fn to_enable_dependencies(&self) -> TokenStream {
        let &Self {
            extension_ty,
            ref extensions,
            ref required_if_supported,
            ..
        } = self;

        let required_if_supported_iter = required_if_supported.iter().map(|extension| {
            quote! {
                if supported.#extension {
                    enabled.#extension = true;
                }
            }
        });
        let required_iter = extensions
            .iter()
            .filter_map(|m| m.to_enable_dependencies(extension_ty));

        quote! {
            pub(crate) fn enable_dependencies(
                &self,
                #[allow(unused_variables)] api_version: Version,
                #[allow(unused_variables)] supported: &Self,
            ) -> Self {
                let mut enabled = *self;
                #(#required_if_supported_iter)*
                #(#required_iter)*
                enabled
            }
        }
    }

    fn to_vk(&self) -> TokenStream {
        quote! {
            pub(crate) fn from_vk<'a>(iter: impl IntoIterator<Item = &'a str>) -> Self {
                // SAFETY: `NAMES_C` only contains UTF-8 strings.
                let names = Self::NAMES_C.iter().map(|name| unsafe {
                    std::str::from_utf8_unchecked(name.to_bytes())
                });

                let mut val = Self::empty();
                for name in iter {
                    if let Some(index) = names.clone().position(|n| n == name) {
                        val.as_mut_array()[index] = true;
                    }
                }
                val
            }

            #[allow(clippy::wrong_self_convention)]
            pub(crate) fn to_vk(&self) -> Vec<&'static CStr> {
                let mut val_vk = Vec::new();
                for index in 0..Self::COUNT {
                    if self.as_array()[index] {
                        val_vk.push(Self::NAMES_C[index]);
                    }
                }
                val_vk
            }
        }
    }

    fn to_traits(&self) -> TokenStream {
        let &Self { extension_ty, .. } = self;

        let struct_name = extension_ty.struct_name();

        quote! {
            impl std::fmt::Debug for #struct_name {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
                    f.debug_list()
                        .entries(self.iter().flat_map(|(extension_name_c, enabled)| {
                            enabled.then_some(extension_name_c)
                        }))
                        .finish()
                }
            }

            impl Default for #struct_name {
                #[inline]
                fn default() -> Self {
                    Self::empty()
                }
            }

            impl std::ops::BitAnd for #struct_name {
                type Output = #struct_name;

                #[inline]
                fn bitand(self, rhs: Self) -> Self::Output {
                    self.intersection(&rhs)
                }
            }

            impl std::ops::BitAndAssign for #struct_name {
                #[inline]
                fn bitand_assign(&mut self, rhs: Self) {
                    *self = self.intersection(&rhs);
                }
            }

            impl std::ops::BitOr for #struct_name {
                type Output = #struct_name;

                #[inline]
                fn bitor(self, rhs: Self) -> Self::Output {
                    self.union(&rhs)
                }
            }

            impl std::ops::BitOrAssign for #struct_name {
                #[inline]
                fn bitor_assign(&mut self, rhs: Self) {
                    *self = self.union(&rhs);
                }
            }

            impl std::ops::BitXor for #struct_name {
                type Output = #struct_name;

                #[inline]
                fn bitxor(self, rhs: Self) -> Self::Output {
                    self.symmetric_difference(&rhs)
                }
            }

            impl std::ops::BitXorAssign for #struct_name {
                #[inline]
                fn bitxor_assign(&mut self, rhs: Self) {
                    *self = self.symmetric_difference(&rhs);
                }
            }

            impl std::ops::Sub for #struct_name {
                type Output = #struct_name;

                #[inline]
                fn sub(self, rhs: Self) -> Self::Output {
                    self.difference(&rhs)
                }
            }

            impl std::ops::SubAssign for #struct_name {
                #[inline]
                fn sub_assign(&mut self, rhs: Self) {
                    *self = self.difference(&rhs);
                }
            }

            impl<'a> IntoIterator for &'a #struct_name {
                type Item = (&'static str, bool);
                type IntoIter = extensions::Iter<'a>;

                #[inline]
                fn into_iter(self) -> Self::IntoIter {
                    self.iter()
                }
            }

            mod extensions {
                pub struct Iter<'a> {
                    pub(crate) inner: std::iter::Zip<
                        std::iter::Copied<std::slice::Iter<'a, &'static std::ffi::CStr>>,
                        std::iter::Copied<std::slice::Iter<'a, bool>>,
                    >,
                }

                impl<'a> Iterator for Iter<'a> {
                    type Item = (&'static str, bool);

                    #[inline]
                    fn next(&mut self) -> Option<Self::Item> {
                        self.inner.next().map(|(name, enabled)| {
                            (
                                // SAFETY: `NAMES_C` only contains UTF-8 strings.
                                unsafe { std::str::from_utf8_unchecked(name.to_bytes()) },
                                enabled,
                            )
                        })
                    }

                    #[inline]
                    fn size_hint(&self) -> (usize, Option<usize>) {
                        self.inner.size_hint()
                    }

                    #[inline]
                    fn nth(&mut self, n: usize) -> Option<Self::Item> {
                        self.inner.nth(n).map(|(name, enabled)| {
                            (
                                // SAFETY: `NAMES_C` only contains UTF-8 strings.
                                unsafe { std::str::from_utf8_unchecked(name.to_bytes()) },
                                enabled,
                            )
                        })
                    }
                }

                impl<'a> DoubleEndedIterator for Iter<'a> {
                    #[inline]
                    fn next_back(&mut self) -> Option<Self::Item> {
                        self.inner.next_back().map(|(name, enabled)| {
                            (
                                // SAFETY: `NAMES_C` only contains UTF-8 strings.
                                unsafe { std::str::from_utf8_unchecked(name.to_bytes()) },
                                enabled,
                            )
                        })
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
        }
    }
}

#[derive(Clone, Debug)]
struct Extension {
    extension_name: Ident,
    extension_name_c: String,
    required_if_supported: bool,
    requires_all_of: Vec<RequiresOneOf>,
    status: Option<ExtensionStatus>,
}

impl Extension {
    fn new(
        vk_extension: &VkExtension,
        vk_extensions: &IndexMap<&str, &VkExtension>,
        required_if_supported: &[Ident],
    ) -> Self {
        let extension_name_c = vk_extension.name.to_owned();
        let extension_name = format_ident!(
            "{}",
            extension_name_c
                .strip_prefix("VK_")
                .unwrap()
                .to_snake_case()
        );

        let required_if_supported = required_if_supported.contains(&extension_name);
        let requires_all_of = vk_extension
            .depends
            .as_ref()
            .map_or_else(Vec::new, |depends| {
                let depends_expression = DependsExpression::parse(depends).unwrap_or_else(|err| {
                    panic!(
                        "couldn't parse `depends={:?}` attribute for extension \
                                        `{}`: {}",
                        vk_extension.name, depends, err
                    )
                });

                convert_depends_expression(depends_expression.clone(), vk_extensions).take()
            });

        Extension {
            extension_name,
            extension_name_c,
            required_if_supported,
            requires_all_of,
            status: ExtensionStatus::new(vk_extension, vk_extensions),
        }
    }

    fn to_struct_member(&self) -> TokenStream {
        let &Self {
            ref extension_name,
            ref extension_name_c,
            required_if_supported,
            ref requires_all_of,
            ref status,
            ..
        } = self;

        let doc = {
            let doc = format!(
                "- [Vulkan documentation]\
                (https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/{}.html)",
                extension_name_c
            );

            quote! {
                #[doc = #doc]
            }
        };

        let required_if_supported = required_if_supported.then(|| {
            quote! {
                #[doc = "- Must be enabled if it is supported by the physical device"]
            }
        });

        let status = status.as_ref().map(|status| {
            let doc = match status {
                ExtensionStatus::PromotedTo(replacement) => match replacement {
                    Requires::APIVersion(major, minor) => {
                        format!("- Promoted to Vulkan {}.{}", major, minor)
                    }
                    Requires::DeviceExtension(ext_name) => {
                        format!("- Promoted to [`{}`](DeviceExtensions::{0})", ext_name)
                    }
                    Requires::InstanceExtension(ext_name) => {
                        format!("- Promoted to [`{}`](InstanceExtensions::{0})", ext_name)
                    }
                },
                ExtensionStatus::DeprecatedBy(replacement) => match replacement {
                    Some(Requires::APIVersion(major, minor)) => {
                        format!("- Deprecated by Vulkan {}.{}", major, minor)
                    }
                    Some(Requires::DeviceExtension(ext_name)) => {
                        format!("- Deprecated by [`{}`](DeviceExtensions::{0})", ext_name)
                    }
                    Some(Requires::InstanceExtension(ext_name)) => {
                        format!("- Deprecated by [`{}`](InstanceExtensions::{0})", ext_name)
                    }
                    None => "- Deprecated without a replacement".to_string(),
                },
            };

            quote! {
                #[doc = #doc]
            }
        });

        let requires = (!requires_all_of.is_empty()).then(|| {
            let iter = requires_all_of.iter().map(|require| {
                let RequiresOneOf {
                    api_version,
                    device_extensions,
                    instance_extensions,
                    ..
                } = require;

                let api_version = api_version
                    .map(|(major, minor)| format!("Vulkan API version {}.{}", major, minor));
                let device_extensions = device_extensions
                    .iter()
                    .map(|ext| format!("device extension [`{}`](DeviceExtensions::{0})", ext));
                let instance_extensions = instance_extensions
                    .iter()
                    .map(|ext| format!("instance extension [`{}`](InstanceExtensions::{0})", ext,));
                let mut line: Vec<_> = api_version
                    .into_iter()
                    .chain(device_extensions)
                    .chain(instance_extensions)
                    .collect();

                if let Some(first) = line.first_mut() {
                    first[0..1].make_ascii_uppercase();
                }

                let doc = format!("  - {}", line.join(" or "));

                quote! {
                    #[doc = #doc]
                }
            });

            if requires_all_of.len() > 1 {
                quote! {
                    #[doc = "- Requires all of:"]
                    #(#iter)*
                }
            } else {
                quote! {
                    #[doc = "- Requires:"]
                    #(#iter)*
                }
            }
        });

        quote! {
            #doc
            #required_if_supported
            #status
            #requires
            pub #extension_name: bool,
        }
    }

    fn to_validate(&self, extensions_level: ExtensionType) -> TokenStream {
        let Self {
            extension_name,
            requires_all_of,
            ..
        } = self;

        let iter = requires_all_of
            .iter()
            .filter_map(|r| r.to_extension_dependency_check(extensions_level, extension_name));

        let problem = format!(
            "contains `{}`, but this extension is not supported by the {}",
            extension_name,
            extensions_level.as_supporter_str(),
        );

        quote! {
            if self.#extension_name {
                if !supported.#extension_name {
                    return Err(Box::new(ValidationError {
                        problem: #problem.into(),
                        ..Default::default()
                    }));
                }

                #(#iter)*
            }
        }
    }

    fn to_enable_dependencies(&self, extensions_level: ExtensionType) -> Option<TokenStream> {
        let Self {
            extension_name,
            requires_all_of,
            ..
        } = self;

        if requires_all_of.is_empty() {
            return None;
        }

        let requires_all_of_items = requires_all_of
            .iter()
            .filter_map(|r| r.to_extension_dependencies_enable(extensions_level))
            .collect::<Vec<_>>();

        if requires_all_of_items.is_empty() {
            return None;
        }

        Some(quote! {
            if self.#extension_name {
                #(#requires_all_of_items)*
            }
        })
    }
}

#[derive(Clone, Copy)]
enum ExtensionType {
    Instance,
    Device,
}

impl ExtensionType {
    fn struct_name(self) -> Ident {
        match self {
            ExtensionType::Instance => format_ident!("InstanceExtensions"),
            ExtensionType::Device => format_ident!("DeviceExtensions"),
        }
    }

    fn as_str_lowercase(self) -> &'static str {
        match self {
            Self::Instance => "instance",
            Self::Device => "device",
        }
    }

    fn as_supporter_str(self) -> &'static str {
        match self {
            ExtensionType::Instance => "library",
            ExtensionType::Device => "physical device",
        }
    }
}

#[derive(Clone, Debug)]
enum ExtensionStatus {
    PromotedTo(Requires),
    DeprecatedBy(Option<Requires>),
}

impl ExtensionStatus {
    fn new(
        vk_extension: &VkExtension,
        vk_extensions: &IndexMap<&str, &VkExtension>,
    ) -> Option<Self> {
        vk_extension
            .promotedto
            .as_deref()
            .and_then(|pr| {
                if let Some(version) = pr.strip_prefix("VK_VERSION_") {
                    let (major, minor) = version.split_once('_').unwrap();
                    Some(Self::PromotedTo(Requires::APIVersion(
                        major.parse().unwrap(),
                        minor.parse().unwrap(),
                    )))
                } else {
                    let ext_name = pr.strip_prefix("VK_").unwrap().to_snake_case();
                    match vk_extensions[pr].ext_type.as_ref().unwrap().as_str() {
                        "device" => Some(Self::PromotedTo(Requires::DeviceExtension(ext_name))),
                        "instance" => Some(Self::PromotedTo(Requires::InstanceExtension(ext_name))),
                        _ => unreachable!(),
                    }
                }
            })
            .or_else(|| {
                vk_extension.deprecatedby.as_deref().and_then(|depr| {
                    if depr.is_empty() {
                        Some(Self::DeprecatedBy(None))
                    } else if let Some(version) = depr.strip_prefix("VK_VERSION_") {
                        let (major, minor) = version.split_once('_').unwrap();
                        Some(Self::DeprecatedBy(Some(Requires::APIVersion(
                            major.parse().unwrap(),
                            minor.parse().unwrap(),
                        ))))
                    } else {
                        let ext_name = depr.strip_prefix("VK_").unwrap().to_snake_case();
                        match vk_extensions[depr].ext_type.as_ref().unwrap().as_str() {
                            "device" => Some(Self::DeprecatedBy(Some(Requires::DeviceExtension(
                                ext_name,
                            )))),
                            "instance" => Some(Self::DeprecatedBy(Some(
                                Requires::InstanceExtension(ext_name),
                            ))),
                            _ => unreachable!(),
                        }
                    }
                })
            })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Requires {
    APIVersion(u32, u32),
    DeviceExtension(String),
    InstanceExtension(String),
}

fn convert_depends_expression(
    depends_expression: DependsExpression<'_>,
    extensions: &IndexMap<&str, &VkExtension>,
) -> ConjunctiveNormalForm<RequiresOneOf> {
    match depends_expression {
        DependsExpression::Name(vk_name) => {
            let new = dependency_chain(vk_name, extensions);

            let mut result = ConjunctiveNormalForm::empty();
            result.add_conjunction(new);

            result
        }
        DependsExpression::AllOf(all_of) => {
            let mut result = ConjunctiveNormalForm::empty();

            for expr in all_of {
                let cnf = convert_depends_expression(expr, extensions);
                for it in cnf.take() {
                    result.add_conjunction(it);
                }
            }

            result
        }
        DependsExpression::OneOf(one_of) => {
            let mut result = ConjunctiveNormalForm::empty();

            for expr in one_of {
                let cnf = convert_depends_expression(expr, extensions);
                result.add_bijection(cnf);
            }

            result
        }
    }
}

fn dependency_chain<'a>(
    mut vk_name: &'a str,
    extensions: &IndexMap<&str, &'a VkExtension>,
) -> RequiresOneOf {
    let mut requires_one_of = RequiresOneOf::default();

    loop {
        if let Some(version) = vk_name.strip_prefix("VK_VERSION_") {
            let (major, minor) = version.split_once('_').unwrap();
            requires_one_of.api_version = Some((major.parse().unwrap(), minor.parse().unwrap()));
            break;
        } else {
            let ext_name = vk_name.strip_prefix("VK_").unwrap().to_snake_case();
            let extension = extensions[vk_name];

            match extension.ext_type.as_deref() {
                Some("device") => &mut requires_one_of.device_extensions,
                Some("instance") => &mut requires_one_of.instance_extensions,
                _ => unreachable!(),
            }
            .push(ext_name);

            if let Some(promotedto) = extension.promotedto.as_ref() {
                vk_name = promotedto.as_str();
            } else {
                break;
            }
        }
    }

    requires_one_of.device_extensions.reverse();
    requires_one_of.instance_extensions.reverse();
    requires_one_of
}

#[derive(Debug, Clone)]
enum DependsExpression<'a> {
    Name(&'a str),
    OneOf(Vec<Self>),
    AllOf(Vec<Self>),
}

impl<'a> DependsExpression<'a> {
    fn parse(depends: &'a str) -> Result<Self, String> {
        fn name(input: &str) -> IResult<&str, &str> {
            take_while1(|c: char| c.is_ascii_alphanumeric() || c == '_')(input)
        }

        fn term(input: &str) -> IResult<&str, DependsExpression<'_>> {
            alt((
                name.map(DependsExpression::Name),
                delimited(complete::char('('), one_of_expression, complete::char(')')),
            ))(input)
        }

        fn all_of_expression(input: &str) -> IResult<&str, DependsExpression<'_>> {
            let (input, mut all_of) = separated_list0(complete::char('+'), term)(input)?;

            Ok((input, {
                if all_of.len() == 1 {
                    all_of.remove(0)
                } else {
                    DependsExpression::AllOf(all_of)
                }
            }))
        }

        fn one_of_expression(input: &str) -> IResult<&str, DependsExpression<'_>> {
            let (input, mut one_of) =
                separated_list0(complete::char(','), all_of_expression)(input)?;

            Ok((input, {
                if one_of.len() == 1 {
                    one_of.remove(0)
                } else {
                    DependsExpression::OneOf(one_of)
                }
            }))
        }

        match all_consuming(one_of_expression)(depends) {
            Ok((_, expr)) => Ok(expr),
            Err(err) => Err(format!("{:?}", err)),
        }
    }
}

fn get_required_if_supported(extension_ty: ExtensionType) -> Vec<Ident> {
    match extension_ty {
        ExtensionType::Instance => Vec::new(),
        ExtensionType::Device => vec![
            // VUID-VkDeviceCreateInfo-pProperties-04451
            format_ident!("khr_portability_subset"),
        ],
    }
}

impl RequiresOneOf {
    fn to_extension_dependency_check(
        &self,
        extension_ty: ExtensionType,
        item_name: &Ident,
    ) -> Option<TokenStream> {
        let Self { api_version, .. } = self;
        let (same_type_extensions, other_type_extensions) = self.select_extensions(extension_ty);

        if !same_type_extensions.is_empty()
            || api_version.is_none() && other_type_extensions.is_empty()
        {
            return None;
        }

        let condition_iter = api_version
            .iter()
            .map(|version| {
                let version = format_ident!("V{}_{}", version.0, version.1);
                quote! { api_version >= crate::Version::#version }
            })
            .chain(other_type_extensions.iter().map(|ext_name| {
                let ident = format_ident!("{}", ext_name);
                quote! { instance_extensions.#ident }
            }));
        let requires_one_of_iter = api_version
            .iter()
            .map(|(major, minor)| {
                let version = format_ident!("V{}_{}", major, minor);
                quote! {
                    crate::RequiresAllOf(&[
                        crate::Requires::APIVersion(crate::Version::#version),
                    ]),
                }
            })
            .chain(other_type_extensions.iter().map(|ext_name| {
                quote! {
                    crate::RequiresAllOf(&[
                        crate::Requires::InstanceExtension(#ext_name),
                    ]),
                }
            }));
        let problem = format!("contains `{}`", item_name);

        Some(quote! {
            if !(#(#condition_iter)||*) {
                return Err(Box::new(crate::ValidationError {
                    problem: #problem.into(),
                    requires_one_of: crate::RequiresOneOf(&[
                        #(#requires_one_of_iter)*
                    ]),
                    ..Default::default()
                }));
            }
        })
    }

    fn to_extension_dependencies_enable(&self, extension_ty: ExtensionType) -> Option<TokenStream> {
        let Self { api_version, .. } = self;
        let (same_type_extensions, _) = self.select_extensions(extension_ty);

        if same_type_extensions.is_empty() {
            return None;
        }

        let condition_iter = api_version
            .iter()
            .map(|(major, minor)| {
                let version = format_ident!("V{}_{}", major, minor);
                quote! { api_version >= crate::Version::#version }
            })
            .chain(same_type_extensions.iter().map(|ext_name| {
                let ident = format_ident!("{}", ext_name);
                quote! { self.#ident }
            }));

        let (base_requirement, promoted_requirements) = same_type_extensions.split_last().unwrap();

        Some(if promoted_requirements.is_empty() {
            let ident = format_ident!("{}", base_requirement);
            quote! {
                if !(#(#condition_iter)||*) {
                    enabled.#ident = true;
                }
            }
        } else {
            let ident = format_ident!("{}", base_requirement);
            let promoted_requirement_iter = promoted_requirements.iter().map(|name| {
                let ident = format_ident!("{}", name);
                quote! {
                    if supported.#ident {
                        enabled.#ident = true;
                    }
                }
            });

            quote! {
                if !(#(#condition_iter)||*) {
                    #(#promoted_requirement_iter)else*
                    else {
                        enabled.#ident = true;
                    }
                }
            }
        })
    }

    fn select_extensions(&self, extension_ty: ExtensionType) -> (&[String], &[String]) {
        let Self {
            api_version: _,
            device_extensions,
            instance_extensions,
            device_features: _,
        } = self;

        match extension_ty {
            ExtensionType::Instance => (instance_extensions.as_slice(), [].as_slice()),
            ExtensionType::Device => (device_extensions.as_slice(), instance_extensions.as_slice()),
        }
    }
}
