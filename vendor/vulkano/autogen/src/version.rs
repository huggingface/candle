use super::{write_file, VkRegistryData};
use proc_macro2::{Literal, TokenStream};
use quote::quote;

pub fn write(vk_data: &VkRegistryData<'_>) {
    let version_output = version_output(vk_data.header_version);
    write_file(
        "version.rs",
        format!(
            "vk.xml header version {}.{}.{}",
            vk_data.header_version.0, vk_data.header_version.1, vk_data.header_version.2
        ),
        quote! {
            #version_output
        },
    );
}

fn version_output((major, minor, patch): (u16, u16, u16)) -> TokenStream {
    let major = Literal::u16_unsuffixed(major);
    let minor = Literal::u16_unsuffixed(minor);
    let patch = Literal::u16_unsuffixed(patch);

    quote! {
        impl Version {
            /// The highest Vulkan API version currently supported by Vulkano.
            ///
            /// It is allowed for applications that use Vulkano to make use of features from higher
            /// versions than this. However, Vulkano itself will not make use of those features and
            /// will not expose their APIs, so they must be accessed by other means.
            ///
            /// The `max_api_version` of an [`Instance`](crate::instance::Instance) equals
            /// `HEADER_VERSION` by default, which locks out features from newer versions. In order
            /// to enable the use of higher versions, the `max_api_version` must be overridden when
            /// creating an instance.
            pub const HEADER_VERSION: Version = Version {
                major: #major,
                minor: #minor,
                patch: #patch,
            };
        }
    }
}
