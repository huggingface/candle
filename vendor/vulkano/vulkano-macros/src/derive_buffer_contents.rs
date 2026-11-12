use crate::bail;
use proc_macro2::TokenStream;
use quote::{quote, quote_spanned};
use syn::{
    parse_quote, spanned::Spanned, Data, DeriveInput, Fields, FieldsNamed, FieldsUnnamed, Ident,
    Result, Type, TypeArray, TypeSlice, WherePredicate,
};

pub fn derive_buffer_contents(crate_ident: &Ident, mut ast: DeriveInput) -> Result<TokenStream> {
    let struct_ident = &ast.ident;

    let is_repr_rust = ast
        .attrs
        .iter()
        .filter(|&attr| attr.path().is_ident("repr"))
        .all(|attr| {
            let mut is_repr_rust = true;

            let _ = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("C") || meta.path.is_ident("transparent") {
                    is_repr_rust = false;
                }

                Ok(())
            });

            is_repr_rust
        });

    if is_repr_rust {
        bail!(
            "deriving `BufferContents` is only supported for types that are marked `#[repr(C)]` \
            or `#[repr(transparent)]`",
        );
    }

    let data = match &ast.data {
        Data::Struct(data) => data,
        Data::Enum(_) => bail!("deriving `BufferContents` for enums is not supported"),
        Data::Union(_) => bail!("deriving `BufferContents` for unions is not supported"),
    };

    let fields = match &data.fields {
        Fields::Named(FieldsNamed { named, .. }) => named,
        Fields::Unnamed(FieldsUnnamed { unnamed, .. }) => unnamed,
        Fields::Unit => bail!("zero-sized types are not valid buffer contents"),
    };

    let (impl_generics, type_generics, where_clause) = {
        let predicates = ast
            .generics
            .type_params()
            .map(|ty| {
                parse_quote! { #ty: ::#crate_ident::buffer::BufferContents }
            })
            .collect::<Vec<WherePredicate>>();

        ast.generics
            .make_where_clause()
            .predicates
            .extend(predicates);

        ast.generics.split_for_impl()
    };

    let mut field_types = fields.iter().map(|field| &field.ty);
    let Some(last_field_type) = field_types.next_back() else {
        bail!("zero-sized types are not valid buffer contents");
    };
    let mut field_layouts = Vec::new();

    let mut bound_types = Vec::new();

    // Accumulate the field layouts and types that have to implement `BufferContents` in order for
    // the struct to implement the trait as well.
    for field_type in field_types {
        bound_types.push(find_innermost_element_type(field_type));
        field_layouts.push(quote! { ::std::alloc::Layout::new::<#field_type>() });
    }

    let layout;
    let ptr_from_slice;

    // The last field needs special treatment.
    match last_field_type {
        // An array might not implement `BufferContents` depending on the element. However, we know
        // the type must be sized, so we can generate the layout and function easily.
        Type::Array(TypeArray { elem, .. }) => {
            bound_types.push(find_innermost_element_type(elem));

            layout = quote! {
                ::#crate_ident::buffer::BufferContentsLayout::from_sized(
                    ::std::alloc::Layout::new::<Self>()
                )
            };

            ptr_from_slice = quote! {
                <*mut [u8]>::cast::<Self>(slice.as_ptr())
            };
        }
        // A slice might contain an array which might not implement `BufferContents`. However, we
        // know the type must be unsized, so can generate the layout and function easily.
        Type::Slice(TypeSlice { elem, .. }) => {
            bound_types.push(find_innermost_element_type(elem));

            layout = quote! {
                ::#crate_ident::buffer::BufferContentsLayout::from_field_layouts(
                    &[ #( #field_layouts ),* ],
                    ::#crate_ident::buffer::BufferContentsLayout::from_slice(
                        ::std::alloc::Layout::new::<#elem>(),
                    ),
                )
            };

            ptr_from_slice = quote! {
                let data = <*mut [u8]>::cast::<#elem>(slice.as_ptr());

                let head_size = <Self as ::#crate_ident::buffer::BufferContents>::LAYOUT
                    .head_size() as usize;
                let element_size = ::std::mem::size_of::<#elem>();

                ::std::debug_assert!(slice.len() >= head_size);
                let tail_size = slice.len() - head_size;
                ::std::debug_assert!(tail_size % element_size == 0);
                let len = tail_size / element_size;

                ::std::ptr::slice_from_raw_parts_mut(data, len) as *mut Self
            };
        }
        // Any other type may be either sized or unsized and the macro has no way of knowing. But
        // it surely implements `BufferContents`, so we can use the existing layout and function.
        ty => {
            bound_types.push(ty);

            layout = quote! {
                ::#crate_ident::buffer::BufferContentsLayout::from_field_layouts(
                    &[ #( #field_layouts ),* ],
                    <#last_field_type as ::#crate_ident::buffer::BufferContents>::LAYOUT,
                )
            };

            ptr_from_slice = quote! {
                let data = <*mut [u8]>::cast::<u8>(slice.as_ptr());

                let head_size = <Self as ::#crate_ident::buffer::BufferContents>::LAYOUT
                    .head_size() as usize;
                let element_size = <Self as ::#crate_ident::buffer::BufferContents>::LAYOUT
                    .element_size()
                    .unwrap_or(1) as usize;

                ::std::debug_assert!(slice.len() >= head_size);
                let tail_size = slice.len() - head_size;
                ::std::debug_assert!(tail_size % element_size == 0);

                <#last_field_type as ::#crate_ident::buffer::BufferContents>::ptr_from_slice(
                    ::std::ptr::NonNull::new_unchecked(::std::ptr::slice_from_raw_parts_mut(
                        data.add(head_size),
                        tail_size,
                    )),
                )
                .byte_sub(head_size) as *mut Self
            };
        }
    };

    let bounds = bound_types.into_iter().map(|ty| {
        quote_spanned! { ty.span() =>
            // HACK: This works around Rust issue #48214, which makes it impossible to put these
            // bounds in the where clause of the trait implementation where they actually belong
            // until that is resolved.
            let _: ::#crate_ident::buffer::AssertParamIsBufferContents<#ty>;
        }
    });

    Ok(quote! {
        #[allow(unsafe_code)]
        unsafe impl #impl_generics ::#crate_ident::buffer::BufferContents
            for #struct_ident #type_generics #where_clause
        {
            const LAYOUT: ::#crate_ident::buffer::BufferContentsLayout = #layout;

            #[inline(always)]
            unsafe fn ptr_from_slice(slice: ::std::ptr::NonNull<[u8]>) -> *mut Self {
                #( #bounds )*

                #ptr_from_slice
            }
        }
    })
}

// HACK: This works around an inherent limitation of bytemuck, namely that an array where the
// element is `AnyBitPattern` is itself not `AnyBitPattern`, by only requiring that the innermost
// type in the array implements `BufferContents`.
fn find_innermost_element_type(mut field_type: &Type) -> &Type {
    while let Type::Array(TypeArray { elem, .. }) = field_type {
        field_type = elem;
    }

    field_type
}

#[cfg(test)]
mod tests {
    use super::*;
    use proc_macro2::Span;

    #[test]
    fn repr() {
        let crate_ident = Ident::new("vulkano", Span::call_site());

        let default_repr = parse_quote! {
            struct Test(u8, [u8]);
        };
        assert!(derive_buffer_contents(&crate_ident, default_repr).is_err());

        let irellevant_reprs = parse_quote! {
            #[repr(packed(2), align(16))]
            struct Test(u8, [u8]);
        };
        assert!(derive_buffer_contents(&crate_ident, irellevant_reprs).is_err());

        let transparent_repr = parse_quote! {
            #[repr(transparent)]
            struct Test([u8]);
        };
        assert!(derive_buffer_contents(&crate_ident, transparent_repr).is_ok());

        let multiple_reprs = parse_quote! {
            #[repr(align(16))]
            #[repr(C)]
            #[repr(packed)]
            struct Test(u8, [u8]);
        };
        assert!(derive_buffer_contents(&crate_ident, multiple_reprs).is_ok());
    }

    #[test]
    fn zero_sized() {
        let crate_ident = Ident::new("vulkano", Span::call_site());

        let unit = parse_quote! {
            struct Test;
        };
        assert!(derive_buffer_contents(&crate_ident, unit).is_err());
    }

    #[test]
    fn unsupported_datatype() {
        let crate_ident = Ident::new("vulkano", Span::call_site());

        let enum_ = parse_quote! {
            #[repr(C)]
            enum Test { A, B, C }
        };
        assert!(derive_buffer_contents(&crate_ident, enum_).is_err());

        let union = parse_quote! {
            #[repr(C)]
            union Test {
                a: u32,
                b: f32,
            }
        };
        assert!(derive_buffer_contents(&crate_ident, union).is_err());
    }
}
