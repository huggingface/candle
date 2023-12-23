extern crate proc_macro;
use proc_macro::TokenStream;
use syn::parse_macro_input;

/// Attribute macro applied to a function to turn it into a unit test with Cpu, Cuda, Metal device.
/// # Examples
///
/// ```rust
/// use candle_test_macro::test_device;
/// use candle::{Device, Result};
///
/// #[test_device]
/// fn test_example(dev: &Device) -> Result<()>{
///     println!("Current device is {:?}", dev);
///     Ok(())
/// }
/// ```
#[proc_macro_attribute]
pub fn test_device(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let fn_body = item.to_string();
    let fn_name = &parse_macro_input!(item as syn::ItemFn).sig.ident.to_string();

    format!(r#"

    #[test]
    fn {fn_name}_cpu() -> Result<()> {{
        {fn_name}(&Device::Cpu)
    }}

    #[cfg(feature = "cuda")]
    #[test]
    fn {fn_name}_cuda() -> Result<()> {{
        {fn_name}(&Device::new_cuda(0)?)
    }}

    #[cfg(feature = "metal")]
    #[test]
    fn {fn_name}_metal() -> Result<()> {{
        {fn_name}(&Device::new_metal(0)?)
    }}

    {fn_body}

    "#).parse().unwrap()
}