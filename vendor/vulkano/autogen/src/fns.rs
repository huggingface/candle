use super::{write_file, IndexMap, VkRegistryData};
use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};
use vk_parse::{Command, Extension, ExtensionChild, InterfaceItem};

pub fn write(vk_data: &VkRegistryData<'_>) {
    // TODO: Long strings break rustfmt

    let entry_fns_output = fns_output(
        std::iter::empty(),
        "Entry",
        &["1_0", "1_1" /*, "1_2", "1_3"*/],
        "Raw Vulkan global entry point-level functions.\n\nTo use these, you need to include the Ash crate, using the same version Vulkano uses.",
    );
    let instance_fns_output = fns_output(
        extension_fns_members(&vk_data.commands, &vk_data.extensions, false),
        "Instance",
        &["1_0", "1_1" /*, "1_2"*/, "1_3"],
        "Raw Vulkan instance-level functions.\n\nTo use these, you need to include the Ash crate, using the same version Vulkano uses.",
    );
    let device_fns_output = fns_output(
        extension_fns_members(&vk_data.commands, &vk_data.extensions, true),
        "Device",
        &["1_0", "1_1", "1_2", "1_3"],
        "Raw Vulkan device-level functions.\n\nTo use these, you need to include the Ash crate, using the same version Vulkano uses.",
    );
    write_file(
        "fns.rs",
        format!(
            "vk.xml header version {}.{}.{}",
            vk_data.header_version.0, vk_data.header_version.1, vk_data.header_version.2
        ),
        quote! {
            #entry_fns_output
            #instance_fns_output
            #device_fns_output
        },
    );
}

#[derive(Clone, Debug)]
struct FnsMember {
    name: Ident,
    /// Path to the struct
    fn_struct: TokenStream,
}

fn fns_output(
    extension_members: impl Iterator<Item = FnsMember>,
    fns_level: &str,
    fns_versions: &[&str], // TODO: Parse from vk.xml?
    doc: &str,
) -> TokenStream {
    let struct_name = format_ident!("{}Functions", fns_level);
    let members = fns_versions
        .iter()
        .map(|version| {
            let fn_struct = format_ident!("{}FnV{}", fns_level, version);
            FnsMember {
                name: format_ident!("v{}", version),
                fn_struct: quote!(#fn_struct),
            }
        })
        .chain(extension_members)
        .collect::<Vec<_>>();

    let struct_items = members.iter().map(|FnsMember { name, fn_struct }| {
        quote! { pub #name: ash::#fn_struct, }
    });

    let load_items = members.iter().map(|FnsMember { name, fn_struct }| {
        quote! { #name: ash::#fn_struct::load(&mut load_fn), }
    });

    quote! {
        #[doc = #doc]
        #[allow(missing_docs)]
        pub struct #struct_name {
            #(#struct_items)*
            pub _ne: crate::NonExhaustive<'static>,
        }

        impl #struct_name {
            pub(crate) fn load<F>(mut load_fn: F) -> #struct_name
                where F: FnMut(&CStr) -> *const c_void
            {
                #struct_name {
                    #(#load_items)*
                    _ne: crate::NE,
                }
            }
        }

        impl std::fmt::Debug for #struct_name {
            #[inline]
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
                f.debug_struct(stringify!(#struct_name)).finish_non_exhaustive()
            }
        }
    }
}

/// Returns [`false`] when this is an `instance` or `entry` command
fn is_device_command(commands: &IndexMap<&str, &Command>, name: &str) -> bool {
    // Based on
    // https://github.com/ash-rs/ash/blob/b724b78dac8d83879ed7a1aad2b91bb9f2beb5cf/generator/src/lib.rs#L568-L586

    let mut name = name;
    loop {
        let command = commands[name];
        match command {
            Command::Alias { alias, .. } => name = alias.as_str(),
            Command::Definition(command) => {
                break command.params.first().is_some_and(|field| {
                    matches!(
                        field.definition.type_name.as_deref(),
                        Some("VkDevice" | "VkCommandBuffer" | "VkQueue")
                    )
                })
            }
            _ => todo!(),
        }
    }
}

fn extension_fns_members<'a>(
    commands: &'a IndexMap<&str, &Command>,
    extensions: &'a IndexMap<&str, &Extension>,
    device_functions: bool,
    // extension_filter: impl Fn(&str) -> bool,
) -> impl Iterator<Item = FnsMember> + use<'a> {
    let fn_struct_name = if device_functions {
        format_ident!("DeviceFn")
    } else {
        format_ident!("InstanceFn")
    };

    extensions
        .values()
        .filter(move |ext| {
            ext.children.iter().any(move |ch| {
                if let ExtensionChild::Require { items, .. } = ch {
                    items
                        .iter()
                        .any(move |i| matches!(i, InterfaceItem::Command { name, .. } if device_functions == is_device_command(commands, name)))
                } else {
                    false
                }
            })
        })
        .map(move |ext| {
            let base = ext.name.strip_prefix("VK_").unwrap().to_lowercase();
            let (vendor, extension) = base.split_once('_').unwrap();
            let vendor = Ident::new(vendor, Span::call_site());
            let extension = Ident::new(extension, Span::call_site());
            let name = format_ident!("{}", base);
            let fn_struct = quote!(#vendor::#extension::#fn_struct_name);
            FnsMember { name, fn_struct }
        })
}
