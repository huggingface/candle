use ash::vk;

/// Properties of an extension in the loader or a physical device.
#[derive(Clone, Debug)]
pub struct ExtensionProperties {
    /// The name of the extension.
    pub extension_name: String,

    /// The version of the extension.
    pub spec_version: u32,
}

impl From<vk::ExtensionProperties> for ExtensionProperties {
    #[inline]
    fn from(val: vk::ExtensionProperties) -> Self {
        Self {
            extension_name: unsafe {
                crate::c_str_to_string_unchecked(val.extension_name_as_c_str().unwrap())
            },
            spec_version: val.spec_version,
        }
    }
}
