//! Macros for [`vulkano`].
//!
//! [`vulkano`]: https://crates.io/crates/vulkano

use proc_macro::TokenStream;
use proc_macro_crate::{crate_name, FoundCrate};
use syn::{parse_macro_input, DeriveInput, Error};

mod derive_buffer_contents;
mod derive_vertex;

/// Derives the [`Vertex`] trait.
///
/// [`Vertex`]: https://docs.rs/vulkano/latest/vulkano/pipeline/graphics/vertex_input/trait.Vertex.html
#[proc_macro_derive(Vertex, attributes(name, format))]
pub fn derive_vertex(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let crate_ident = crate_ident();

    derive_vertex::derive_vertex(&crate_ident, ast)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

/// Derives the [`BufferContents`] trait.
///
/// [`BufferContents`]: https://docs.rs/vulkano/latest/vulkano/buffer/trait.BufferContents.html
#[proc_macro_derive(BufferContents)]
pub fn derive_buffer_contents(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let crate_ident = crate_ident();

    derive_buffer_contents::derive_buffer_contents(&crate_ident, ast)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

fn crate_ident() -> syn::Ident {
    let found_crate = crate_name("vulkano").unwrap();
    let name = match &found_crate {
        // We use `vulkano` by default as we are exporting crate as vulkano in vulkano/lib.rs.
        FoundCrate::Itself => "vulkano",
        FoundCrate::Name(name) => name,
    };

    syn::Ident::new(name, proc_macro2::Span::call_site())
}

macro_rules! bail {
    ($msg:expr $(,)?) => {
        return Err(syn::Error::new(proc_macro2::Span::call_site(), $msg))
    };
    ($span:expr, $msg:expr $(,)?) => {
        return Err(syn::Error::new_spanned($span, $msg))
    };
}
use bail;
