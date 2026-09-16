//! Emits a kernel module items for rustc to typecheck.

use cutile_expand::error::{Error, SpannedError, syn_err};
use cutile_expand::rank_instantiation::{
    instantiate_function_for_rank, instantiate_impl_for_rank, instantiate_static_for_rank,
    instantiate_struct_for_rank, instantiate_type_alias_for_rank, variadic_impl, variadic_struct,
};
use cutile_expand::shadow_dispatch::{
    desugar_variadic_trait_decl, desugar_variadic_trait_impl, emit_shadow_dispatch,
};
use cutile_expand::validate_dsl_syntax::{validate_entry_attribute, validate_entry_point_parameters};
use cutile_syn_utils::syn_utils::{clear_attributes, get_meta_list, get_meta_list_by_last_segment};
use cutile_syn_utils::type_aliases::{collect_type_aliases, normalize_item_fn_param_type_aliases};
use proc_macro2::{Ident, Span, TokenStream as TokenStream2};
use quote::{ToTokens, quote};
use std::collections::{HashMap, HashSet};
use syn::{ItemFn, ItemImpl, ItemStruct, ItemTrait, ItemType};

/// Walk a module's items, returning what rustc should see in their place.
///
/// Recurses into inline submodules. A file-loaded submodule (`mod foo;`) is rejected: the body
/// is not available at expansion time, so there is nothing to transform.
pub fn process_items(items: &[syn::Item]) -> Result<Vec<TokenStream2>, Error> {
    let type_aliases = collect_type_aliases(items);
    let mut out: Vec<TokenStream2> = Vec::new();

    for item in items {
        if is_cuda_only(item_attrs(item)) {
            continue;
        }
        match item {
            syn::Item::Fn(f) => out.push(function(f.clone(), &type_aliases)?),
            syn::Item::Struct(s) => out.push(structure(s.clone())?),
            syn::Item::Trait(t) => out.push(trait_(t.clone())?),
            syn::Item::Impl(i) => out.push(implementation(i.clone())?),
            syn::Item::Type(t) => out.push(instantiate_type_alias_for_rank(t)?.to_token_stream()),
            syn::Item::Static(s) => out.push(instantiate_static_for_rank(s)?.to_token_stream()),
            syn::Item::Use(u) => out.push(u.to_token_stream()),
            syn::Item::Const(c) => out.push(c.to_token_stream()),
            syn::Item::Macro(m) => out.push(m.to_token_stream()),
            syn::Item::Mod(submod) => {
                let Some((_, sub_items)) = &submod.content else {
                    return submod.err(
                        "a submodule inside `#[module]` must have an inline body \
                         (`mod foo { ... }`); `mod foo;` is not supported because the macro \
                         needs the body at expansion time",
                    );
                };
                let sub = process_items(sub_items)?;
                let (name, attrs, vis) = (&submod.ident, &submod.attrs, &submod.vis);
                out.push(quote! {
                    #(#attrs)*
                    #vis mod #name { #(#sub)* }
                });
            }
            other => return other.err("unsupported item type in a kernel module"),
        }
    }
    Ok(out)
}

/// A function: entry-point validation, then rank instantiation or shadow dispatch.
///
/// The two are exclusive. A rank-polymorphic fn (`#[cuda_tile::variadic_op]`) is served entirely
/// by the synthesized trait, so emitting per-rank free functions as well would only create
/// ambiguity at the call site.
fn function(
    mut item: ItemFn,
    type_aliases: &HashMap<String, ItemType>,
) -> Result<TokenStream2, Error> {
    if get_meta_list_by_last_segment("entry", &item.attrs).is_some() {
        validate_entry_attribute(&item)?;
        // Validation runs on an alias-resolved copy, never the original: `validate_dsl_syntax`
        // matches on the spelled type, so `type Buf = Tensor<f32, S>;` would otherwise read as
        // an unsupported parameter.
        let resolved = normalize_item_fn_param_type_aliases(&item, type_aliases)
            .map_err(|msg| syn_err(item.sig.ident.span(), &msg))?;
        validate_entry_point_parameters(&resolved)?;
    }

    let attributes = get_meta_list("cuda_tile :: variadic_op", &item.attrs);
    let emit_trait = attributes.is_some();
    let method_override: Option<Ident> = attributes
        .as_ref()
        .and_then(|a| a.parse_string("method"))
        .map(|s| Ident::new(&s, Span::call_site()));
    let trait_name_override: Option<Ident> = attributes
        .as_ref()
        .and_then(|a| a.parse_string("trait_name"))
        .map(|s| Ident::new(&s, Span::call_site()));

    // Snapshot before mutation: shadow dispatch needs the signature in its original CGA form.
    let original = emit_trait.then(|| item.clone());
    clear_attributes(
        HashSet::from([
            "cuda_tile :: variadic_op",
            "cuda_tile :: op",
            "cuda_tile :: compiler_op",
        ]),
        &mut item.attrs,
    );
    clear_entry_attribute(&mut item.attrs);

    let concrete = if emit_trait {
        Vec::new()
    } else {
        vec![instantiate_function_for_rank(&item)?]
    };
    let shadow = match original {
        Some(orig) => emit_shadow_dispatch(&orig, method_override, trait_name_override)?,
        None => TokenStream2::new(),
    };
    Ok(quote! { #(#concrete)* #shadow })
}

/// A struct: one concrete type per rank when variadic, otherwise CGA desugaring only.
fn structure(mut item: ItemStruct) -> Result<TokenStream2, Error> {
    let attributes = get_meta_list("cuda_tile :: variadic_struct", &item.attrs);
    clear_attributes(
        HashSet::from(["cuda_tile :: variadic_struct", "cuda_tile :: ty"]),
        &mut item.attrs,
    );
    let Some(attributes) = attributes else {
        let item = instantiate_struct_for_rank(&item)?;
        return Ok(quote! { #item });
    };
    let emitted = variadic_struct(&attributes, item)?;
    let structs = emitted.iter().map(|(s, _)| s);
    let impls = emitted.iter().filter_map(|(_, i)| i.as_ref());
    Ok(quote! { #(#structs)* #(#impls)* })
}

/// A trait: desugared when variadic, passed through otherwise.
///
/// `#[cuda_tile::unchecked]` drops the trait entirely. It marks a declaration that exists for the
/// frontend to read out of the source text, with no rustc-facing form to emit.
fn trait_(mut item: ItemTrait) -> Result<TokenStream2, Error> {
    if get_meta_list("cuda_tile :: unchecked", &item.attrs).is_some() {
        return Ok(TokenStream2::new());
    }
    let attributes = get_meta_list("cuda_tile :: variadic_trait", &item.attrs);
    clear_attributes(
        HashSet::from(["cuda_tile :: variadic_trait", "cuda_tile :: ty"]),
        &mut item.attrs,
    );
    match attributes {
        Some(a) if a.name_as_str().as_deref() == Some("cuda_tile :: variadic_trait") => {
            desugar_variadic_trait_decl(&item)
        }
        _ => Ok(quote! { #item }),
    }
}

/// An impl block: one per rank when variadic, covering both inherent and operator impls.
///
/// Routing is by attribute, not by shape. `#[cuda_tile::variadic_trait_impl]` owns its own rank
/// enumeration through the trait's CGAs, so it takes precedence and `#[variadic_impl(N=..)]` is
/// merely tolerated alongside it. Deciding on `item.trait_.is_some()` instead looks equivalent
/// and is not: a marker-trait impl such as `unsafe impl Sync for Global<E, D>` has no method for
/// case-3c substitution to work from, and fails.
fn implementation(mut item: ItemImpl) -> Result<TokenStream2, Error> {
    if get_meta_list("cuda_tile :: unchecked", &item.attrs).is_some() {
        return Ok(TokenStream2::new());
    }
    // Read before clearing, which wipes it.
    let is_variadic_trait_impl =
        get_meta_list("cuda_tile :: variadic_trait_impl", &item.attrs).is_some();
    let attributes = get_meta_list("cuda_tile :: variadic_impl", &item.attrs);
    clear_attributes(
        HashSet::from([
            "cuda_tile :: variadic_trait_impl",
            "cuda_tile :: variadic_impl",
            "cuda_tile :: ty",
        ]),
        &mut item.attrs,
    );
    if is_variadic_trait_impl {
        return desugar_variadic_trait_impl(&item);
    }
    match attributes {
        Some(attributes) => {
            let impls = variadic_impl(&attributes, item)?;
            Ok(quote! { #(#impls)* })
        }
        None => {
            let item = instantiate_impl_for_rank(&item)?;
            Ok(quote! { #item })
        }
    }
}

/// Whether an item is gated to cuTile's `cuda` feature, and so is not ours to emit.
///
/// A proc-macro runs before `#[cfg]` is evaluated, and the transformations do not carry the
/// attribute onto what they synthesise: shadow dispatch turns one gated fn into a trait and a
/// set of impls that name CUDA-only element types unconditionally, and those fail to resolve.
/// The gate has to be honoured here or not at all.
///
/// Reading a `cfg` predicate inside a macro is normally wrong, because the macro cannot know the
/// consuming crate's features. This one case is decidable: the predicate appears only in our
/// vendored DSL surface, where it marks types re-exported from `cuda-core`, and a backend that
/// had them would be using cuTile's own expander rather than this one. If a second predicate
/// ever appears in that file, this stops being safe and the gate belongs upstream instead, as
/// attribute propagation through `emit_shadow_dispatch`.
fn is_cuda_only(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|a| {
        a.path().is_ident("cfg")
            && a.parse_args::<syn::Meta>().is_ok_and(|m| {
                m.to_token_stream().to_string().replace(' ', "") == "feature=\"cuda\""
            })
    })
}

/// An item's attributes, for the kinds a kernel module may contain.
fn item_attrs(item: &syn::Item) -> &[syn::Attribute] {
    match item {
        syn::Item::Fn(i) => &i.attrs,
        syn::Item::Struct(i) => &i.attrs,
        syn::Item::Trait(i) => &i.attrs,
        syn::Item::Impl(i) => &i.attrs,
        syn::Item::Type(i) => &i.attrs,
        syn::Item::Static(i) => &i.attrs,
        syn::Item::Const(i) => &i.attrs,
        syn::Item::Use(i) => &i.attrs,
        syn::Item::Macro(i) => &i.attrs,
        syn::Item::Mod(i) => &i.attrs,
        _ => &[],
    }
}

/// Strip `entry` by last path segment, so `wax::entry` and `metal_tile::entry` both match.
fn clear_entry_attribute(attrs: &mut Vec<syn::Attribute>) {
    attrs.retain(|a| {
        a.path()
            .segments
            .last()
            .is_none_or(|seg| seg.ident != "entry")
    });
}

