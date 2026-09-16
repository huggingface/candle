//! Proc-macros for candle-wax kernels.
//!
//! `#[module]` captures a kernel module's source text so the frontend can re-parse it, and
//! `#[entry]` marks the functions inside it that become entry points.

extern crate proc_macro;

mod expand;

use proc_macro::TokenStream;
use quote::quote;
use sha2::{Digest, Sha256};
use syn::parse_macro_input;

/// Mark a module as a wax kernel module.
///
/// Captures the module's source text at compile time and emits
/// `__module_ast_self() -> wax_frontend::ast::Module` in its place, so that
/// `KernelCompiler::new(my_mod::__module_ast_self, ...)` can recover the AST.
///
/// Module kernel bodies are typechecked by rustc.
#[proc_macro_attribute]
pub fn module(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let raw_source = item.to_string();
    let module_item = parse_macro_input!(item as syn::ItemMod);
    emit_module(module_item, raw_source)
}

fn emit_module(module_item: syn::ItemMod, raw_source: String) -> TokenStream {
    let mod_name = &module_item.ident;
    let mod_name_str = mod_name.to_string();
    let vis = &module_item.vis;
    let attrs = &module_item.attrs;

    let item_start_span = match &module_item.vis {
        syn::Visibility::Public(vis_pub) => vis_pub.span,
        syn::Visibility::Restricted(vis_r) => vis_r.pub_token.span,
        syn::Visibility::Inherited => module_item.mod_token.span,
    };

    let source_file = item_start_span.file().to_string();
    let base_line = item_start_span.start().line;
    let base_col = item_start_span.start().column;

    let source_text: String = {
        let full_span = module_item
            .content
            .as_ref()
            .and_then(|(brace, _)| item_start_span.join(brace.span.close()));
        full_span
            .and_then(|sp| sp.source_text())
            .unwrap_or(raw_source)
    };

    // Emit concrete kernel items (one per rank), plus dispatch scaffolding, so rustc can typecheck.
    let concrete_items: Vec<::proc_macro2::TokenStream> = match emitted_items(&module_item) {
        Ok(items) => items,
        Err(err) => return err.into(),
    };

    // Create SHA-256 hash of source.
    let source_hash = format!("{:x}", Sha256::digest(source_text.as_bytes()));

    quote! {
        #(#attrs)*
        #vis mod #mod_name {
            // The DSL is a Rust subset. Certain lints are not applicable.
            #![allow(nonstandard_style)]
            #![allow(dead_code)]
            #![allow(unused_variables)]
            #![allow(clippy::all)]

            #(#concrete_items)*
            /// Parses this module's source text into a `wax_frontend::ast::Module` for use with
            /// `KernelCompiler::new(...)`.
            #[allow(non_snake_case)]
            pub fn __module_ast_self() -> ::wax_frontend::ast::Module {
                let source_text: &str = #source_text;
                let parsed_mod: ::syn::ItemMod = ::syn::parse_str(source_text)
                    .expect("__module_ast_self: failed to re-parse captured source text");
                let span_base = ::wax_frontend::ast::SpanBase::new(
                    #source_file.to_string(),
                    #base_line,
                    #base_col,
                );
                let mut this_ast = ::wax_frontend::ast::Module::with_span_base(
                    #mod_name_str,
                    parsed_mod,
                    span_base,
                );
                this_ast.set_absolute_path(::std::module_path!().to_string());
                this_ast
            }

            /// SHA-256 of module source, computed at compile time.
            ///
            /// Simple and backend neutral. Not guaranteed to be suited as a backend cache
            /// key.
            /// Does not take into account changes outside this module (helper fn's etc).
            pub const SOURCE_HASH: &str = #source_hash;
        }
    }
    .into()
}

/// Module items in rustc token form, or a `compile_error!`.
fn emitted_items(module_item: &syn::ItemMod) -> Result<Vec<proc_macro2::TokenStream>, TokenStream> {
    let Some((_, items)) = &module_item.content else {
        return Ok(Vec::new());
    };
    expand::process_items(items).map_err(|e| TokenStream::from(e.to_compile_error()))
}

/// Mark a function inside `#[module]` as an entry point.
///
/// Emits the function unchanged. The frontend reads this attribute from the captured source
/// text rather than from the expansion.
#[proc_macro_attribute]
pub fn entry(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}
