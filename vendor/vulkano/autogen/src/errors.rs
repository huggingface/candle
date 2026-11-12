use super::{write_file, VkRegistryData};
use heck::ToUpperCamelCase;
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};

pub fn write(vk_data: &VkRegistryData<'_>) {
    write_file(
        "errors.rs",
        format!(
            "vk.xml header version {}.{}.{}",
            vk_data.header_version.0, vk_data.header_version.1, vk_data.header_version.2
        ),
        errors_output(&errors_members(&vk_data.errors)),
    );
}

#[derive(Clone, Debug)]
struct ErrorsMember {
    name: Ident,
    ffi_name: Ident,
}

fn errors_output(members: &[ErrorsMember]) -> TokenStream {
    let enum_items = members.iter().map(|ErrorsMember { name, .. }| {
        quote! { #name, }
    });
    let try_from_items = members.iter().map(|ErrorsMember { name, ffi_name }| {
        quote! { ash::vk::Result::#ffi_name => Self::#name, }
    });

    quote! {
        /// An enumeration of runtime errors that can be returned by Vulkan.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[repr(i32)]
        #[non_exhaustive]
        pub enum VulkanError {
            #(#enum_items)*
            Unnamed(ash::vk::Result),
        }

        impl From<ash::vk::Result> for VulkanError {
            fn from(val: ash::vk::Result) -> VulkanError {
                match val {
                    #(#try_from_items)*
                    x => Self::Unnamed(x),
                }
            }
        }
    }
}

fn errors_members(errors: &[&str]) -> Vec<ErrorsMember> {
    errors
        .iter()
        .map(|error| {
            let ffi_name = error.strip_prefix("VK_").unwrap();

            let mut parts = ffi_name.split('_').collect::<Vec<_>>();

            if parts[0] == "ERROR" {
                parts.remove(0);
            }

            if ["EXT", "KHR", "NV"].contains(parts.last().unwrap()) {
                parts.pop();
            }

            let name = parts.join("_").to_upper_camel_case();

            ErrorsMember {
                name: format_ident!("{}", name),
                ffi_name: format_ident!("{}", ffi_name),
            }
        })
        .collect()
}
